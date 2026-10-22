"""Alanine dipeptide with a sodium ion or a second peptide in vacuum.

Each peptide uses a prior on its (phi, psi) angles. Cartesian coordinates and velocities evolve under restrained MD. Units are nm, ps, kJ/mol and K; angles are in radians.

Optional minimization uses the edited physical system before coupling forces are added. Velocities start at zero and remain part of the sampler state.

System builders accept a context-edit callback for alternative prior decompositions.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mdtraj
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u
import torch
from torch import Tensor

from ggpa import Prior, compose
from ggpa.contexts import openmm as md
from ggpa.priors import torus_ancestral
from ggpa.priors.networks import EAGER_FP32, Precision, load_net
from ggpa.projection import Torsions
from ggpa.systems import runner
from ggpa.utils import stack

FORCEFIELD = ("amber99sbildn.xml", "tip3p.xml")  # tip3p.xml supplies sodium parameters.
TEMPERATURE, FRICTION = 300.0, 1.0  # K and inverse ps.
MD_STEPS = 100  # Restrained MD steps per Gibbs sweep.
T_PROD = 0.1  # Sodium coupling time.
LEASH_R0, LEASH_K = 0.4, 50.0  # Ion/CA restraint: nm and kJ/mol/nm^2.
COM_K, COM_D0 = 50.0, 0.9  # Dimer COM restraint: kJ/mol/nm^2 and nm.
ION_OFFSET = 0.5  # Initial ion offset along x from the peptide centroid, in nm.
MIN_TOL = 10.0  # Minimization force tolerance in kJ/mol/nm.
GROUP = 30  # Physical restraints; coupling forces use group 31.


def backbone_quartets(topology: app.Topology) -> Tensor:
    """Return paired phi/psi atom quartets in topology order as an int64 (M, 2, 4) tensor.

    Pair phi=(C_prev, N, CA, C) and psi=(N, CA, C, N_next) by their shared CA atom. Include only residues with both angles.
    """
    top = mdtraj.Topology.from_openmm(topology)
    traj = mdtraj.Trajectory(np.zeros((1, top.n_atoms, 3)), top)  # Only topology is needed to identify quartets.
    phi, psi = mdtraj.compute_phi(traj)[0], mdtraj.compute_psi(traj)[0]
    by_ca = {int(q[1]): q for q in psi}
    return torch.tensor(np.array([[q, by_ca[int(q[2])]] for q in phi if int(q[2]) in by_ca]), dtype=torch.long)


def segments(quartets: Tensor) -> list[Tensor]:
    """Return the unique atoms in each copy's modeled backbone torsions."""
    return [q.unique() for q in quartets]


def vacuum_system(topology: app.Topology) -> mm.System:
    """Build the vacuum force field with no nonbonded cutoff and constrained hydrogen bonds."""
    return app.ForceField(*FORCEFIELD).createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)


def _edited(topology: app.Topology, positions, restraint: mm.Force, edit_context) -> tuple[mm.System, Tensor, Tensor]:
    """Add the physical restraint and apply the selected force-field decomposition.

    The default removes internal backbone nonbonded and torsion terms. A callback replaces these edits. Return (system, coordinates, quartets), with coordinates shaped (N, 3) in nm and quartets shaped (M, 2, 4).
    """
    system, quartets = vacuum_system(topology), backbone_quartets(topology)
    restraint.setForceGroup(GROUP)
    system.addForce(restraint)
    if edit_context is None:
        for seg in segments(quartets):
            system = md.zero_torsions_within(md.zero_nonbonded_within(system, seg), seg)
    else:
        system = edit_context(system, quartets)
    return system, torch.tensor(np.array(positions.value_in_unit(u.nanometer)), dtype=torch.float64), quartets


def sodium_system(
    pdb_path,
    *,
    offset_nm: float = ION_OFFSET,
    leash_r0: float = LEASH_R0,
    leash_k: float = LEASH_K,
    edit_context=None,
):
    """Add a sodium ion and a harmonic ion/CA restraint to the input peptide.

    Place the ion offset_nm along x from the peptide centroid. The restraint is leash_k*(r-leash_r0)^2/2, with distances in nm and stiffness in kJ/mol/nm^2. Return the System, float64 coordinates in nm and backbone quartets.

    edit_context(system, quartets) replaces the default backbone decomposition. It receives the full force field plus restraint and must preserve atom order, masses and constraints.
    """
    pdb = app.PDBFile(str(pdb_path))
    centre = pdb.getPositions(asNumpy=True).value_in_unit(u.nanometer).mean(0)
    ion = app.Topology()
    ion.addAtom("Na+", app.element.sodium, ion.addResidue("NA", ion.addChain()))
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.add(ion, [mm.Vec3(centre[0] + offset_nm, centre[1], centre[2])] * u.nanometer)
    leash = mm.CustomBondForce("0.5*k*(r - r0)^2")
    for name in ("k", "r0"):
        leash.addPerBondParameter(name)
    ca = next(a.index for a in modeller.topology.atoms() if a.residue.name == "ALA" and a.name == "CA")
    leash.addBond(ca, modeller.topology.getNumAtoms() - 1, [leash_k, leash_r0])
    return _edited(modeller.topology, modeller.getPositions(), leash, edit_context)


def dimer_system(pdb_path, *, com_k: float = COM_K, com_d0: float = COM_D0, edit_context=None):
    """Construct a two-peptide system with a flat-bottom center-of-mass restraint.

    The restraint is com_k*max(0,d-com_d0)^2/2 using mass-weighted centers, distances in nm and stiffness in kJ/mol/nm^2. Return the System, float64 coordinates in nm and backbone quartets.

    edit_context replaces the default backbone decomposition, as in sodium_system. The center-of-mass restraint is installed before the callback.
    """
    pdb = app.PDBFile(str(pdb_path))
    com = mm.CustomCentroidBondForce(2, "0.5*k*max(0, distance(g1, g2) - d0)^2")
    for name in ("k", "d0"):
        com.addPerBondParameter(name)
    groups = [com.addGroup([a.index for a in chain.atoms()]) for chain in pdb.topology.chains()]
    com.addBond(groups, [com_k, com_d0])
    return _edited(pdb.topology, pdb.positions, com, edit_context)


def minimize(system: mm.System, positions: Tensor, platform: str = "Reference") -> Tensor:
    """Minimize the supplied System from positions shaped (N, 3) in nm.

    Apply its constraints and MIN_TOL force tolerance. Return a new float64 CPU position tensor.
    """
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName(platform))
    context.setPositions(positions.detach().cpu().double().numpy())
    mm.LocalEnergyMinimizer.minimize(context, MIN_TOL)
    out = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
    return torch.from_numpy(np.array(out, dtype=np.float64))


@dataclass(frozen=True, kw_only=True)
class Sodium:
    """Settings for sodium trajectories and direct torsion sampling.

    For each checkpoint, run trajectories at T_PROD and one direct-sampling point.
    """

    seed: int
    device: str  # Network and state device; OpenMM uses platform.
    precision: dict  # Fields accepted by Precision.
    checkpoints: dict[int, str]
    pdb: str
    trajectories: int
    sweeps: int
    dt_ps: float  # Integration timestep in ps.
    torus_steps: int  # Number of denoising grid intervals at t=1.
    platform: str
    properties: dict  # OpenMM platform properties.
    minimize: bool
    direct_samples: int


@dataclass(frozen=True, kw_only=True)
class Dimer:
    """Settings for dimer trajectories with optional replica exchange.

    Run trajectories on ladder when nonempty, and fixed-time trajectories for each entry of t.
    """

    seed: int
    device: str
    precision: dict
    checkpoint: str
    pdb: str
    trajectories: int
    sweeps: int
    ladder: list[float]
    t: list[float]
    dt_ps: float  # Integration timestep in ps.
    torus_steps: int  # Number of denoising grid intervals at t=1.
    platform: str
    properties: dict
    minimize: bool


def points(cfg: Sodium | Dimer) -> list[runner.Point]:
    """Enumerate trajectory and direct-sampling points with stable IDs for output paths and seeds."""
    k = range(cfg.trajectories)
    if isinstance(cfg, Sodium):
        pts = [(f"ion{p}pct_traj{i:02d}", cfg.sweeps, {"pct": p, "traj": i}) for p in cfg.checkpoints for i in k]
        pts += [(f"direct_{p}pct", 0, {"pct": p, "traj": None}) for p in cfg.checkpoints]
    else:
        pts = [(f"re_traj{i:02d}", cfg.sweeps, {"t": list(cfg.ladder), "traj": i}) for i in k if cfg.ladder]
        pts += [(f"t{t:.4f}_traj{i:02d}", cfg.sweeps, {"t": [t], "traj": i}) for t in cfg.t for i in k]
    return [runner.Point(i, *p) for i, p in enumerate(pts)]


def load_prior(path, *, device="cpu", steps=200, precision: Precision = EAGER_FP32) -> tuple[Prior, dict]:
    """Load a two-angle torsion checkpoint as (prior, metadata).

    The prior uses the wrapped-VE ancestral sampler and its coupling above the sigma_min floor.
    """
    from ggpa.priors.kernels import WrappedGaussian

    net, kernel, meta = load_net(path, torch.device(device), precision)
    if type(kernel) is not WrappedGaussian or tuple(kernel.shape) != (2,):
        raise ValueError("load_prior requires a wrapped-Gaussian (phi, psi) checkpoint")
    return torus_ancestral(net, kernel, steps), meta


def _gen(cfg: Sodium | Dimer, point: runner.Point) -> torch.Generator:
    return torch.Generator(torch.device(cfg.device)).manual_seed(runner.point_seed(cfg.seed, point.id))


def _sample(cfg: Sodium | Dimer, point: runner.Point, out: Path, built: tuple, path: str, times) -> None:
    """Run one engine per replica and record coordinates, clean angles and swap decisions."""
    system, s0, quartets = built
    prior, meta = load_prior(path, device=cfg.device, steps=cfg.torus_steps, precision=Precision(**cfg.precision))
    s0 = (minimize(system, s0) if cfg.minimize else s0).to(cfg.device)[None]
    projection = Torsions(quartets)
    samplers = []
    for r, t in enumerate(times):
        ctx = md.context(
            system,
            initial=s0,
            temperature=TEMPERATURE,
            dt=cfg.dt_ps,
            steps=MD_STEPS,
            friction=FRICTION,
            platform=cfg.platform,
            properties=cfg.properties,
        )
        samplers.append(
            compose([(prior, projection)], ctx, t=t, seed=runner.point_seed(cfg.seed, f"{point.id}/replica{r}"))
        )
    runner.write_resolved(out, cfg, point, [prior], times, (None,) * len(times), [meta], samplers=samplers)
    records, t0, accepted = [], time.perf_counter(), None
    for _ in range(point.sweeps):
        accepted = runner.step(samplers, exchange=len(samplers) > 1)
        rec = {
            "s": torch.stack([s.state.s for s in samplers]).float(),
            "x": torch.stack([s.state.prior_samples[0] for s in samplers]).float(),
        }
        records.append(rec | {"accepted": accepted} if len(samplers) > 1 else rec)
    engine = None
    if samplers[0].context.engine is not None:
        contexts = [s.context for s in samplers]
        seeds = [c.integrator.getRandomNumberSeed() for c in contexts]
        platform = contexts[0].engine.getPlatform()
        properties = {k: platform.getPropertyValue(contexts[0].engine, k) for k in platform.getPropertyNames()}
        engine = {"seed": seeds[0], "properties": properties}
        if len(seeds) > 1:
            engine["seeds"] = seeds
    runner.finish(out, stack(records), runner.snapshot(samplers, accepted), time.perf_counter() - t0, engine=engine)


def _direct(cfg: Sodium, point: runner.Point, out: Path, path: str) -> None:
    """Denoise uniform torus samples from t=1 without a physical context.

    Save float32 angles of shape (direct_samples, 1, 2) in radians. No sampler checkpoint is written.
    """
    prior, meta = load_prior(path, device=cfg.device, steps=cfg.torus_steps, precision=Precision(**cfg.precision))
    runner.write_resolved(out, cfg, point, [prior], (1.0,), (None,), [meta])
    gen, n, dev = _gen(cfg, point), cfg.direct_samples, torch.device(cfg.device)
    t0 = time.perf_counter()
    y = 2 * math.pi * torch.rand((n, 2), generator=gen, dtype=torch.float64, device=dev) - math.pi
    x = prior.denoise(y, torch.ones(n, dtype=torch.float64, device=dev), gen).view(n, 1, 2)
    runner.finish(out, {"x": x.cpu().numpy()}, None, time.perf_counter() - t0)


def sodium_point(cfg: Sodium, point: runner.Point, out: Path) -> None:
    """Run one fixed-time sodium trajectory or the checkpoint's direct-sampling point."""
    path = cfg.checkpoints[point.values["pct"]]
    if point.values["traj"] is None:
        return _direct(cfg, point, out, path)
    return _sample(cfg, point, out, sodium_system(cfg.pdb), path, (T_PROD,))


def dimer_point(cfg: Dimer, point: runner.Point, out: Path) -> None:
    """Run one dimer trajectory with a fixed context and the selected coupling times."""
    t = tuple(point.values["t"])
    _sample(cfg, point, out, dimer_system(cfg.pdb), cfg.checkpoint, t)


if __name__ == "__main__":
    verbs = {"sodium": (Sodium, points, sodium_point), "dimer": (Dimer, points, dimer_point)}
    raise SystemExit(runner.main(sys.argv[1:], verbs))
