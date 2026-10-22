"""Periodic two-dimensional phi4 lattice and reference Monte Carlo.

The dimensionless energy is H=sum_i(phi_i^2-1)^2+J*sum_<ij>(phi_i-phi_j)^2-h*sum_i(phi_i), counting each bond once.

GGPA couples one on-site prior to each lattice site and updates the noisy field through an exact Gaussian Fourier conditional. FFTs use orthonormal normalization and an (L, L//2+1) half-spectrum. Reference sampling uses checkerboard Metropolis sweeps.
"""

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ggpa import compose
from ggpa.contexts import deconvolve
from ggpa.priors import ddpm, v_to_eps
from ggpa.priors.kernels import Gaussian
from ggpa.priors.networks import Precision, load_net
from ggpa.projection import Select
from ggpa.systems import runner
from ggpa.utils import stack

L = 32  # Lattice side length.


def z2(net):
    """Enforce an odd velocity prediction v(x,t)=[net(x,t)-net(-x,t)]/2.

    Both signs are evaluated in one network call.
    """

    def v(x: Tensor, t: Tensor) -> Tensor:
        w = net(torch.cat([x, -x]), t.repeat(2))
        return 0.5 * (w[: len(x)] - w[len(x) :])

    return v


def laplacian(L: int) -> Tensor:
    """Return periodic lattice Laplacian eigenvalues on the rfft2 grid in float64.

    lambda_k=4*sin(pi*kx/L)^2+4*sin(pi*ky/L)^2, so phi.T*Lap*phi equals the sum of squared differences over bonds counted once.
    """
    kx = torch.arange(L, dtype=torch.float64)[:, None]
    ky = torch.arange(L // 2 + 1, dtype=torch.float64)
    return 4 * torch.sin(math.pi * kx / L) ** 2 + 4 * torch.sin(math.pi * ky / L) ** 2


@dataclass(frozen=True, eq=False)
class LatticeContext:
    """Fixed lattice context with an exact Fourier conditional at the current coupling time."""

    J: float
    h: float
    L: int
    kernel: Gaussian
    context_noise: float | None = None

    def spectrum(self) -> Tensor:
        # H_int = J*phi.T*Lap*phi, so its quadratic precision is 2J*Lap.
        q = 2 * self.J * laplacian(self.L)
        return q if self.context_noise is None else deconvolve(q, self.kernel, self.context_noise)

    def field(self) -> float:
        alpha = (
            1.0
            if self.context_noise is None
            else self.kernel.level(torch.tensor([self.context_noise], dtype=torch.float64)).alpha.item()
        )
        # The orthonormal zero-mode field is h*L; deconvolution scales it by 1/alpha.
        return self.h * self.L / alpha

    def log_factor(self, s):
        # Double columns with omitted conjugates; DC and Nyquist columns retain weight one.
        w = torch.full((self.L // 2 + 1,), 2.0, dtype=s.dtype, device=s.device)
        w[0] = 1.0
        if self.L % 2 == 0:
            w[-1] = 1.0
        p = torch.fft.rfft2(s, norm="ortho")
        return -0.5 * (w * self.spectrum().to(s) * p.abs().square()).sum((-2, -1)) + self.field() * p[..., 0, 0].real

    def validate(self, couplings, t, s):
        if tuple(s.shape[1:]) != (self.L, self.L):
            raise ValueError("phi4 state must have shape (B, L, L)")
        if len(couplings) != 1 or couplings[0].kernel is not self.kernel:
            raise ValueError("phi4 Fourier context needs its one original Gaussian kernel")
        c = couplings[0]
        want = torch.arange(self.L**2, device=s.device).view(-1, 1)
        if (
            type(c.kernel) is not Gaussian
            or c.kernel.shape != (1,)
            or type(c.projection) is not Select
            or not torch.equal(c.projection.index.to(s.device), want)
        ):
            raise ValueError(
                "Fourier update needs one Gaussian prior and Select covering every site once in site order"
            )
        lv = c.level(t[0], s)
        A = self.spectrum().to(s) + lv.sigma[0].square().reciprocal()
        if not torch.isfinite(A).all() or (A <= 0).any():
            raise ValueError("Fourier conditional precision must be finite and positive")

    def update(self, s, prior_samples, couplings, t, context_state, rng):
        lv = couplings[0].level(t[0], s)
        alpha = lv.alpha.to(s)[:, None, None]
        sigma2 = lv.sigma.to(s).square()[:, None, None]
        # Add coupling precision to each context mode; the field shifts only the zero mode.
        A = sigma2.reciprocal() + self.spectrum().to(s)
        numerator = alpha / sigma2 * torch.fft.rfft2(prior_samples[0].reshape(s.shape), norm="ortho")
        numerator[..., 0, 0] += self.field()
        z = torch.randn(s.shape, generator=rng, dtype=s.dtype, device=s.device)
        # Transform real white noise to preserve Hermitian symmetry, then scale by A^-1/2.
        draw = torch.fft.irfft2(
            numerator / A + torch.fft.rfft2(z, norm="ortho") / A.sqrt(), s=(self.L, self.L), norm="ortho"
        )
        return draw, context_state


def context(kernel, J, h, L=L, *, context_noise=None):
    """Construct a bare or deconvolved lattice factor with a fixed reference noise."""
    return LatticeContext(J, h, L, kernel, context_noise)


def checkerboard_mc(phi: Tensor, J: float, h: float, delta: float, sweeps: int, gen: torch.Generator):
    """Apply checkerboard Metropolis sweeps to an even periodic lattice.

    phi has trailing shape (L, L). Each sweep updates both colors in order, proposing uniform changes in [-delta, delta] and accepting by the full local energy difference. Each color consumes one proposal array and one acceptance array.

    Return the updated field and its spatial mean after each sweep, preserving leading chain axes.
    """
    i = torch.arange(phi.shape[-1], device=phi.device)
    black, m = (i[:, None] + i) % 2 == 0, []
    for _ in range(sweeps):
        for mask in (black, ~black):
            nb = phi.roll(1, -1) + phi.roll(-1, -1) + phi.roll(1, -2) + phi.roll(-1, -2)
            new = phi + delta * (2 * torch.rand(phi.shape, generator=gen, dtype=phi.dtype, device=phi.device) - 1)
            # Changing one site affects its on-site term, four bonds and external field.
            dH = (new**2 - 1) ** 2 - (phi**2 - 1) ** 2 + J * (4 * (new**2 - phi**2) - 2 * (new - phi) * nb)
            dH = dH - h * (new - phi)
            u = torch.rand(phi.shape, generator=gen, dtype=phi.dtype, device=phi.device)
            phi = torch.where(mask & (torch.log(u) < -dH), new, phi)
        m.append(phi.mean((-2, -1)))
    return phi, torch.stack(m)


def observe(samplers, accepted) -> dict[str, Tensor]:
    """Record clean and noisy magnetizations and swap decisions, retaining replica and chain axes."""
    return {
        "m_x": torch.stack([s.state.prior_samples[0].double().mean((1, 2)) for s in samplers]),
        "m_s": torch.stack([s.state.s.mean((1, 2)) for s in samplers]),
        "accepted": accepted,
    }


@dataclass(frozen=True, kw_only=True)
class Scan:
    """Settings for lattice sampling across couplings J and fields h.

    J accepts explicit values, linspace segments or a scaling-window specification. rungs=1 samples at fixed t_prod. extend maps point IDs to individual sweep budgets.
    """

    seed: int
    device: str
    precision: dict
    checkpoint: str
    J: list | dict
    h: list[float]
    t_prod: float
    t_top: float
    rungs: int
    sweeps: int
    extend: dict[str, int]
    L: int = L
    chains: int = 1


@dataclass(frozen=True, kw_only=True)
class MC:
    """Settings for checkerboard reference sampling, using the same J and h grids as Scan."""

    seed: int
    J: list | dict
    h: list[float]
    equilibration: int
    measurement: int
    delta: float
    L: int = L
    chains: int = 1
    device: str = "cpu"


def grid(spec: list | dict, h: float) -> list[float]:
    """Return coupling values at field h from explicit values or a scaling window.

    Lists may contain values and [lo, hi, n] linspace segments. A scaling window samples J=J_c+x*h^(8/15) uniformly between clipped endpoints.
    """
    if isinstance(spec, dict):
        w, (x0, x1), (c0, c1) = h ** (8 / 15), spec["x"], spec["clip"]
        return np.linspace(max(spec["J_c"] + x0 * w, c0), min(spec["J_c"] + x1 * w, c1), spec["n"]).tolist()
    return [J for e in spec for J in (np.linspace(*e).tolist() if isinstance(e, list) else [float(e)])]


def points(cfg: Scan | MC) -> list[runner.Point]:
    """Enumerate (J, h) points in ascending J order within each field.

    IDs use six decimal places; duplicate J values at that precision share a point. Sampling budgets come from extend or sweeps; reference budgets include equilibration and measurement.
    """
    pts, extend = [], getattr(cfg, "extend", {})
    for h in cfg.h:
        ids = {}
        for J in sorted(grid(cfg.J, h)):
            ids.setdefault(f"J{J:.6f}_h{h:.6f}", J)
        pts += [(k, {"J": J, "h": float(h)}) for k, J in ids.items()]
    if bad := sorted(set(extend) - {k for k, _ in pts}):
        raise ValueError(
            f"extend: {bad[0]!r} names no point of this config ({len(pts)} points); ids are J{{J:.6f}}_h{{h:.6f}}"
        )
    base = cfg.sweeps if isinstance(cfg, Scan) else cfg.equilibration + cfg.measurement
    return [runner.Point(i, k, extend.get(k, base), v) for i, (k, v) in enumerate(pts)]


def sample_point(cfg: Scan, point: runner.Point, out: Path) -> None:
    """Run one lattice point with a shared context fixed at t_prod.

    Replica times are geometric from t_prod to t_top. Initial sites are uniform on [-2, 2]. Attempt exchange after every sweep and record the resulting states.
    """
    dev = torch.device(cfg.device)
    net, kernel, meta = load_net(cfg.checkpoint, dev, Precision(**cfg.precision))
    prior = ddpm(v_to_eps(z2(net), kernel), kernel)
    projection = Select(torch.arange(cfg.L * cfg.L).view(cfg.L * cfg.L, 1))
    ctx = context(kernel, J=point.values["J"], h=point.values["h"], L=cfg.L, context_noise=cfg.t_prod)
    times = tuple(np.geomspace(cfg.t_prod, cfg.t_top, cfg.rungs).tolist())
    gen = torch.Generator(dev).manual_seed(runner.point_seed(cfg.seed, point.id))
    initial = 4 * torch.rand(cfg.rungs, cfg.chains, cfg.L, cfg.L, generator=gen, device=dev, dtype=torch.float64) - 2
    samplers = [
        compose(
            [(prior, projection)],
            ctx,
            initial=initial[r],
            t=t,
            seed=runner.point_seed(cfg.seed, f"{point.id}/replica{r}"),
        )
        for r, t in enumerate(times)
    ]
    runner.write_resolved(out, cfg, point, [prior], times, (cfg.t_prod,) * cfg.rungs, [meta], samplers=samplers)
    records, t0, accepted = [], time.perf_counter(), None
    for _ in range(point.sweeps):
        accepted = runner.step(samplers, exchange=True)
        records.append(observe(samplers, accepted))
    runner.finish(out, stack(records), runner.snapshot(samplers, accepted), wall_s=time.perf_counter() - t0)


def reference_point(cfg: MC, point: runner.Point, out: Path) -> None:
    """Run checkerboard reference dynamics from independent uniform fields on [-2, 2].

    Discard equilibration sweeps, then save magnetizations and final fields. Multiple chains have a trailing chain axis in m and a leading chain axis in field; a single chain has no chain axis.
    """
    if cfg.L < 2 or cfg.L % 2:
        raise ValueError("checkerboard sampling requires an even lattice size of at least 2")
    dev = torch.device(cfg.device)
    gen = torch.Generator(dev).manual_seed(runner.point_seed(cfg.seed, point.id))
    J, h = point.values["J"], point.values["h"]
    shape = (cfg.L, cfg.L) if cfg.chains == 1 else (cfg.chains, cfg.L, cfg.L)
    phi = 4 * torch.rand(shape, generator=gen, device=dev, dtype=torch.float64) - 2
    runner.write_resolved(out, cfg, point, [], (), (), [])
    t0 = time.perf_counter()
    phi, _ = checkerboard_mc(phi, J, h, cfg.delta, cfg.equilibration, gen)
    phi, m = checkerboard_mc(phi, J, h, cfg.delta, cfg.measurement, gen)
    arrays = {"m": m.cpu().numpy(), "field": phi.cpu().numpy()}
    runner.finish(out, arrays, None, wall_s=time.perf_counter() - t0)


if __name__ == "__main__":
    verbs = {"sample": (Scan, points, sample_point), "reference": (MC, points, reference_point)}
    raise SystemExit(runner.main(sys.argv[1:], verbs))
