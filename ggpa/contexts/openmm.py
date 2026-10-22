"""OpenMM conditional dynamics in nm, ps, kJ/mol and K.

context() builds an engine with coupling forces; wrap() adopts a prepared Simulation. The fixed physical factor includes the force field, temperature, masses and constraints. Finite-step Langevin dynamics approximate its conditional equilibrium law.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import weakref
from collections.abc import Callable

import openmm as mm
import openmm.unit as u
import torch
from torch import Tensor

from ..conventions import Coupling, check_tensor
from ..priors.kernels import Gaussian, Level, WrappedGaussian
from ..projection import Cartesian, Torsions

R = 0.00831446261815324  # kJ/mol/K; the molar gas constant used by OpenMM

_OWNERS = weakref.WeakKeyDictionary()


def context(
    system,
    *,
    initial,
    temperature,
    dt,
    steps,
    friction,
    restraints=None,
    velocities=None,
    platform="Reference",
    properties=None,
):
    """Construct an NVT context with its own OpenMM engine.

    initial has shape (B, N, 3) in nm. velocities has the same shape in nm/ps and defaults to zero. Temperature, timestep and friction use K, ps and inverse ps.

    restraints contains one builder per prior attachment, each returning (force, setter) from (coupling, kT). Default builders support Gaussian Cartesian and wrapped-Gaussian torsion couplings; their energies are checked against the kernels.

    The System is copied and the engine is built on the first update. Each context belongs to one sampler.
    """
    for name, value in (("temperature", temperature), ("dt", dt), ("friction", friction)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")

    xml = mm.XmlSerializer.serialize(system)
    obj = _OpenMM()
    obj.system = mm.XmlSerializer.deserialize(xml)
    obj.temperature, obj.steps = float(temperature), steps
    obj.settings = dt, friction, platform, dict(properties or {})
    obj.builders = None if restraints is None else tuple(restraints)
    obj.initial = torch.as_tensor(initial, dtype=torch.float64).detach().clone()
    obj.velocities = (
        torch.zeros_like(obj.initial) if velocities is None else torch.as_tensor(velocities).to(obj.initial).clone()
    )
    obj.exchange_key = ("openmm-nvt", hashlib.sha256(xml.encode()).hexdigest(), float(temperature))
    return obj


def wrap(simulation, *, couplings, setters, restraints, temperature, steps):
    """Adopt an initialized Simulation with coupling forces already installed.

    couplings contains the prior/projection pairs passed to compose. Each setter updates an installed force from (engine, clean_sample, alpha, sigma). Matching restraint builders support independent energy checks; callers must ensure they represent the installed forces.

    Read initial positions and velocities from the engine. Only one configuration is supported. The integrator must use the specified temperature and all force groups.

    The engine belongs exclusively to one sampler. Equality of base factors with another wrapped engine is not inferred for exchange.
    """
    from ..conventions import attachments

    if not math.isfinite(temperature) or temperature <= 0 or not isinstance(steps, int) or steps < 1:
        raise ValueError("temperature must be positive and finite; steps must be a positive integer")

    obj = _OpenMM()
    obj.engine, obj.integrator, obj.system = simulation.context, simulation.integrator, simulation.system
    obj.temperature, obj.steps = float(temperature), steps
    obj.builders, obj.setters = tuple(restraints), tuple(setters)
    obj.bound = attachments(couplings)
    state = obj.engine.getState(getPositions=True, getVelocities=True)
    obj.initial = torch.tensor(state.getPositions(asNumpy=True).value_in_unit(u.nanometer)).unsqueeze(0)
    obj.velocities = torch.tensor(
        state.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond)
    ).unsqueeze(0)
    obj.wrapped = True
    _check_integrator(obj.integrator, temperature)
    return obj


def _check_integrator(integrator, temperature):
    """Check that a wrapped engine samples the declared temperature and includes every force group."""
    # The base factor's thermodynamic conditions must match the scale used for the added forces.
    get_temperature = getattr(integrator, "getTemperature", None)
    if not callable(get_temperature) or not math.isclose(
        get_temperature().value_in_unit(u.kelvin), temperature, rel_tol=1e-12
    ):
        raise ValueError("wrapped integrator must expose the same thermostat temperature")
    if integrator.getIntegrationForceGroups() != -1:
        raise ValueError("wrapped integrator must integrate all force groups, including every GGPA restraint")


class _OpenMM:
    """Manage one sampler's engine, coupling forces and native checkpoints."""

    def __init__(self):
        self.engine = self.integrator = self.bound = self._owner = None
        self.wrapped = False
        self.exchange_key = None
        self._checked = None

    def _claim(self, sampler):
        if self._owner is not None and self._owner() is not None:
            raise ValueError(
                "a live OpenMM wrapper belongs to one sampler; construct another context for another sampler"
            )
        self._owner = weakref.ref(sampler)
        if self.engine is not None:
            owner = _OWNERS.get(self.engine)
            if owner is not None and owner() is not None:
                self._owner = None
                raise ValueError("this OpenMM engine is already owned by another live sampler")
            _OWNERS[self.engine] = weakref.ref(sampler)

    def initial_state(self):
        return self.initial.clone(), self.velocities.clone()

    def validate_context_state(self, context_state, s):
        check_tensor(context_state, "OpenMM velocities", like=s, shape=s.shape)

    def validate(self, couplings, t, s):
        if tuple(s.shape[1:]) != (self.system.getNumParticles(), 3):
            raise ValueError("OpenMM state must have shape (B, system particle count, 3) in nm")
        if self.wrapped and len(s) != 1:
            raise ValueError("a wrapped live Simulation supports B=1")
        if any("Barostat" in type(f).__name__ for f in self.system.getForces()):
            raise ValueError("this OpenMM adapter supports NVT; box/pressure dynamics need a custom context")
        if self.bound is not None and (
            len(couplings) != len(self.bound)
            or any(
                a.prior is not b.prior or a.projection is not b.projection
                for a, b in zip(couplings, self.bound, strict=True)
            )
        ):
            raise ValueError("OpenMM context is already bound to different priors/projections")
        builders = (_default_restraint,) * len(couplings) if self.builders is None else self.builders
        if len(builders) != len(couplings) or (self.wrapped and len(self.setters) != len(couplings)):
            raise ValueError("provide one restraint builder/setter per prior attachment")
        # Check restraint energies against each kernel at initialization and when times change. Probes use no chain randomness.
        key = (tuple(couplings), t)
        if key != self._checked:
            i = torch.arange(s[0].numel(), device=s.device, dtype=s.dtype)
            shifts = 0.01 * torch.sin((torch.arange(1, 5, device=s.device, dtype=s.dtype)[:, None]) * (i + 0.7))
            probes = torch.cat((s[:4], s[0] + shifts.reshape(4, *s.shape[1:])))
            for c, time, builder in zip(couplings, t, builders, strict=True):
                x = c.project(s[:1])[0]
                error = check_restraint(builder, c, self.system, probes, x, time, self.temperature)
                if not math.isfinite(error) or error > 1e-6:
                    raise ValueError(
                        f"engine restraint differs from its kernel by {error:g} kT; check units/PBC/anchors"
                    )
            self._checked = key

    def _build(self, couplings, rng):
        if self.engine is not None:
            return
        system = mm.XmlSerializer.clone(self.system)
        self.setters = []
        builders = (_default_restraint,) * len(couplings) if self.builders is None else self.builders
        for c, builder in zip(couplings, builders, strict=True):
            force, setter = builder(c, R * self.temperature)
            system.addForce(force)
            self.setters.append(setter)
        dt, friction, platform, properties = self.settings
        integrator = mm.LangevinMiddleIntegrator(self.temperature, friction, dt)
        integrator.setRandomNumberSeed(int(torch.randint(1, 2**31 - 1, (), generator=rng, device=rng.device)))
        self.engine = mm.Context(system, integrator, mm.Platform.getPlatformByName(platform), properties)
        self.integrator, self.bound = integrator, couplings
        if self._owner is not None:
            _OWNERS[self.engine] = self._owner

    def update(self, s, prior_samples, couplings, t, context_state, rng):
        """Advance each batch configuration with current coordinates, velocities and clean anchors.

        Set every coupling to its live diffusion level before integrating. Return coordinates and velocities in the sampler's dtype and device; velocities use nm/ps.
        """
        self._build(couplings, rng)
        positions, velocities = [], []
        levels = [c.level(time, s) for c, time in zip(couplings, t, strict=True)]
        for row in range(len(s)):
            self.engine.setPositions(s[row].detach().cpu().double().numpy())
            self.engine.setVelocities(context_state[row].detach().cpu().double().numpy())
            for setter, x, level in zip(self.setters, prior_samples, levels, strict=True):
                setter(self.engine, x[row], float(level.alpha[row]), float(level.sigma[row]))
            self.integrator.step(self.steps)
            state = self.engine.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=False)
            positions.append(torch.tensor(state.getPositions(asNumpy=True).value_in_unit(u.nanometer)))
            velocities.append(torch.tensor(state.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond)))
        return torch.stack(positions).to(s), torch.stack(velocities).to(s)

    def checkpoint(self):
        return {"native": None if self.engine is None else self.engine.createCheckpoint()}

    def restore(self, record, couplings, t, s):
        if record["native"] is None:
            if self.wrapped:
                raise ValueError("a wrapped-engine checkpoint must contain native state")
            self.engine = self.integrator = self.bound = None
            return
        # Use a separate construction stream; loadCheckpoint restores the engine RNG.
        self._build(couplings, torch.Generator(device="cpu"))
        self.engine.loadCheckpoint(record["native"])


Restraint = Callable[[Coupling, float], tuple[mm.Force, Callable]]


def _default_restraint(coupling, kT):
    builder = cartesian_restraint if type(coupling.projection) is Cartesian else torsion_restraint
    return builder(coupling, kT)


def cartesian_restraint(coupling: Coupling, kT: float) -> tuple[mm.CustomCompoundBondForce, Callable]:
    """Construct the Cartesian coupling energy kT*||Phi(s)-alpha*x||^2/(2*sigma^2).

    Atom order, centering and unit conversion come from the projection. Differentiating the force expression includes motion of the geometric center. Position-independent Gaussian normalization is omitted.

    Supports Gaussian VP and VE kernels. No rotation alignment or periodic imaging is applied.
    """
    projection, kernel = coupling.projection, coupling.kernel
    if type(projection) is not Cartesian or type(kernel) is not Gaussian:
        raise ValueError("cartesian_restraint needs Cartesian and an isotropic Gaussian kernel")
    copies, atoms = projection.atoms.shape
    if tuple(kernel.shape) != (atoms, 3):
        raise ValueError("Cartesian kernel shape must equal (atoms_per_copy, 3)")

    amplitude, variance = f"alpha_{id(coupling):x}", f"s2_{id(coupling):x}"
    terms = [
        f"(({axis}{i}-c{axis})/{projection.scale!r}-{amplitude}*a{axis}{i})^2"
        for i in range(1, atoms + 1)
        for axis in "xyz"
    ]
    centers = [
        f"c{axis}=(" + "+".join(f"{axis}{i}" for i in range(1, atoms + 1)) + f")/{atoms}"
        if projection.center
        else f"c{axis}=0"
        for axis in "xyz"
    ]
    expression = f"{kT!r}/(2*{variance})*(" + "+".join(terms) + ");" + ";".join(centers)
    force = mm.CustomCompoundBondForce(atoms, expression)
    force.addGlobalParameter(amplitude, 1.0)
    force.addGlobalParameter(variance, 1.0)
    for i in range(1, atoms + 1):
        for axis in "xyz":
            force.addPerBondParameter(f"a{axis}{i}")
    indices = projection.atoms.tolist()
    for row in indices:
        force.addBond(row, [0.0] * (3 * atoms))
    force.setForceGroup(31)

    def set_anchors(context, x, alpha, sigma):
        for i, (row, anchors) in enumerate(zip(indices, x.detach().cpu().reshape(copies, -1).tolist(), strict=True)):
            force.setBondParameters(i, row, anchors)
        force.updateParametersInContext(context)
        context.setParameter(amplitude, alpha)
        context.setParameter(variance, sigma**2)

    return force, set_anchors


def torsion_restraint(coupling: Coupling, kT: float) -> tuple[mm.CustomTorsionForce, Callable]:
    """Construct a wrapped-Gaussian torsion coupling in force group 31.

    Each angle contributes -kT*log(q), omitting the position-independent Gaussian normalization. The force uses the kernel's winding range, one clean anchor per angle and one variance per attachment.

    Requires Torsions and WrappedGaussian. Periodic bonds are enabled when the projection supplies a box, matching its minimum-image geometry.
    """
    proj, kernel = coupling.projection, coupling.prior.kernel
    if type(proj) is not Torsions or type(kernel) is not WrappedGaussian:
        raise ValueError(
            f"torsion_restraint: the coupling pairs {type(proj).__name__} with a {type(kernel).__name__} kernel; "
            "it needs exactly Torsions and WrappedGaussian. Write a restraint for this coupling "
            "and check it with check_restraint"
        )

    s2 = f"s2_{id(coupling):x}"  # Separate parameter names keep attachment noise levels independent.
    images = " + ".join(
        f"exp(-(2*pi*({m})*w + 2*pi^2*{m * m})/{s2})"  # Image ratios cannot exceed one when |w| <= pi.
        for m in itertools.chain(range(-kernel.windings, 0), range(1, kernel.windings + 1))
    )
    # Factor the nearest Gaussian image out of q, then add the other image ratios. Symmetric winding indices give the same sum for either sign of the cross term.
    force = mm.CustomTorsionForce(
        f"kT*(w^2/(2*{s2}) - log(1 + {images})); w = d - 2*pi*floor((d + pi)/(2*pi)); d = theta - theta0; "
        f"kT = {kT!r}; pi = {math.pi!r}"
    )
    force.addGlobalParameter(s2, 1.0)
    force.addPerTorsionParameter("theta0")
    quartets = proj.quartets.reshape(-1, 4).tolist()  # Match the flattened clean-angle order.
    for q in quartets:
        force.addTorsion(*q, [0.0])
    force.setForceGroup(31)
    force.setUsesPeriodicBoundaryConditions(proj.box is not None)

    def set_anchors(context, x: Tensor, alpha: float, sigma: float) -> None:
        """Set clean angles in radians and the noise variance; VE kernels have alpha=1."""
        for i, (q, a) in enumerate(zip(quartets, x.reshape(-1).tolist(), strict=True)):
            force.setTorsionParameters(i, *q, [a])
        force.updateParametersInContext(context)
        context.setParameter(s2, sigma**2)

    return force, set_anchors


def check_restraint(
    restraint: Restraint, coupling: Coupling, system, positions: Tensor, x: Tensor, t: float, temperature: float
) -> float:
    """Return the variation of E/(R*T)+log(q) across probe configurations, in kT units.

    positions has shape (P, N, 3) in nm. The clean anchor x has shape (M, *prior_shape) and stays fixed. A temporary Reference engine uses the System's particle count and periodic box.

    A matching restraint gives a coordinate-independent offset, so this variation is zero up to numerical error.
    """
    lv = coupling.prior.kernel.level(torch.tensor([float(t)], dtype=torch.float64))
    return _spread(restraint, coupling, system, positions, x, lv, R * temperature)


def _spread(restraint, coupling, system, positions, x, lv, kT) -> float:
    """Return the largest probe variation of E/kT+log(q) over the supplied noise levels."""
    probe = mm.System()
    for _ in range(system.getNumParticles()):
        probe.addParticle(1.0)
    probe.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    force, set_anchors = restraint(coupling, kT)
    probe.addForce(force)
    context = mm.Context(probe, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    positions, x = positions.detach().cpu().double(), x.detach().cpu().double()
    y, P, worst = coupling.projection(positions), len(positions), 0.0
    for i in range(len(lv.sigma)):
        # Apply one level to all probes.
        rows = Level(*(None if f is None else f[i : i + 1].cpu().expand(P) for f in lv))
        set_anchors(context, x, rows.alpha[0].item(), rows.sigma[0].item())
        energy = []
        for p in positions.numpy():
            context.setPositions(p)
            energy.append(context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole))
        log_q = coupling.prior.kernel.log_q(y, x.expand(P, *x.shape), rows).sum(1)
        # Constant across probes when force and kernel agree.
        v = torch.tensor(energy, dtype=torch.float64) / kT + log_q
        worst = max(worst, (v.max() - v.min()).item())
    return worst


def zero_nonbonded_within(system, atoms) -> mm.System:
    """Copy the System and remove nonbonded interactions between the selected atoms.

    Set charge products and epsilon to zero for every selected pair, adding or replacing NonbondedForce exceptions. Other interactions remain intact. Unsupported nonbonded force types raise ValueError.
    """
    system, atoms = mm.XmlSerializer.clone(system), sorted({int(a) for a in atoms})
    for i, force in enumerate(system.getForces()):
        if type(force) is mm.NonbondedForce:
            for a, b in itertools.combinations(atoms, 2):
                # Zero charge product and epsilon; sigma is then irrelevant.
                force.addException(a, b, 0.0, 1.0, 0.0, replace=True)
        elif "Nonbonded" in type(force).__name__:
            raise ValueError(
                f"zero_nonbonded_within: {type(force).__name__} at index {i} cannot be edited soundly per pair; "
                "remove these terms from the System before composing"
            )
    return system


def zero_torsions_within(system, atoms) -> mm.System:
    """Copy the System and zero periodic torsions entirely within the selected atoms.

    All other force-field terms remain intact.
    """
    system, atoms = mm.XmlSerializer.clone(system), {int(a) for a in atoms}
    for force in system.getForces():
        if type(force) is mm.PeriodicTorsionForce:
            for i in range(force.getNumTorsions()):
                *q, n, phase, _ = force.getTorsionParameters(i)
                if set(q) <= atoms:
                    force.setTorsionParameters(i, *q, n, phase, 0.0)
    return system
