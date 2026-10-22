"""Coupled double-well example, analytic marginals and sampling.

The explicit state contains a noisy system coordinate and an environmental coordinate. Energies use the dimensionless convention beta=1.
"""

from __future__ import annotations

import dataclasses
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ggpa import Prior, compose
from ggpa.contexts import admissible, deconvolve, quadratic
from ggpa.priors import ddpm, grid_posterior, v_to_eps
from ggpa.priors.kernels import Gaussian
from ggpa.priors.networks import Precision, load_net
from ggpa.projection import Select
from ggpa.systems import runner
from ggpa.utils import stack

F64 = torch.float64
A, K_C, K_B, U_EQ = 8.0, 4.0, 1.0, 1.0  # Dimensionless energy parameters, beta=1.
K_EFF = K_B * K_C / (K_B + K_C)  # Harmonic stiffness after integrating out the environment.
MD_DT, MD_STRIDE, MD_RECORDS, MD_TRAJECTORIES = 5e-3, 200, 25_000, 5
_X = torch.linspace(-4.0, 4.0, 8001, dtype=F64)  # Uniform quadrature grid for marginal normalization.


def energy(x_sys, x_env):
    """Evaluate the double-well energy plus system/environment and environmental harmonic terms."""
    return A * (x_sys**2 - 1) ** 2 + K_C / 2 * (x_sys - x_env) ** 2 + K_B / 2 * (x_env - U_EQ) ** 2


def context(kernel: Gaussian, context_noise=None):
    """Construct a fixed quadratic context on (y, x_env).

    The factor has energy kappa*(y-alpha*x_env)^2/2 + K_B*(x_env-U_EQ)^2/2. Without deconvolution, kappa=K_C and alpha=1. Otherwise both coefficients use the fixed reference context_noise, independently of sampling time.
    """
    kap, alpha = K_C, 1.0
    if context_noise is not None:
        kap = deconvolve(torch.tensor([K_C], dtype=F64), kernel, context_noise).item()
        alpha = kernel.level(torch.tensor([context_noise], dtype=F64)).alpha.item()
    P = torch.tensor([[kap, -kap * alpha], [-kap * alpha, K_B + kap * alpha**2]], dtype=F64)
    return quadratic(P, torch.tensor([0.0, K_B * U_EQ], dtype=F64))


def grid_prior(kernel: Gaussian, points: int = 6001) -> Prior:
    """Construct a grid approximation to the double-well conditional on [-3, 3].

    The prior is proportional to exp(-A*(x^2-1)^2); node selection and cell jitter follow grid_posterior.
    """
    return grid_posterior(lambda x: -A * (x[:, 0] ** 2 - 1) ** 2, kernel, (torch.linspace(-3, 3, points, dtype=F64),))


def _density(log_f, x) -> Tensor:
    """Normalize exp(log_f(x)) using a uniform Riemann sum on the fixed quadrature grid."""
    log_z = torch.logsumexp(log_f(_X), 0) + math.log(_X[1] - _X[0])
    return torch.exp(log_f(torch.as_tensor(x, dtype=F64)) - log_z)


def marginal(x) -> Tensor:
    """Evaluate the physical system marginal in float64.

    Integrating out the environment gives a density proportional to exp(-A*(x^2-1)^2-K_EFF*(x-U_EQ)^2/2).
    """
    return _density(lambda v: -A * (v**2 - 1) ** 2 - K_EFF / 2 * (v - U_EQ) ** 2, x)


def replica_density(x, kernel: Gaussian, t_r: float, t_star: float) -> Tensor:
    """Evaluate the clean marginal at t_r with the context fixed at t_star.

    The density is proportional to p(x)*exp(-(alpha_r*x-alpha_star*U_EQ)^2/(2*v)), where v=sigma_r^2-sigma_star^2+alpha_star^2*(1/K_C+1/K_B). At t_r=t_star this reduces to marginal(x).
    """
    (a, a_), (s, s_) = kernel.level(torch.tensor([t_r, t_star], dtype=F64))[1:]
    v = s**2 - s_**2 + a_**2 * (1 / K_C + 1 / K_B)
    return _density(lambda u: -A * (u**2 - 1) ** 2 - (a * u - a_ * U_EQ) ** 2 / (2 * v), x)


def md_reference(gen, trajectories=MD_TRAJECTORIES, records=MD_RECORDS, dt=MD_DT, stride=MD_STRIDE) -> Tensor:
    """Simulate overdamped Langevin dynamics using Euler-Maruyama steps.

    Start each trajectory at (1, 1) and apply x <- x-dt*grad(U)+sqrt(2*dt)*noise, using float64 noise from gen.

    Return a tensor of shape (trajectories, records+1, 2), including the starting configuration. Stored records are separated by stride integration steps.
    """
    x_system, x_environment = np.ones(trajectories), np.ones(trajectories)
    out = [np.stack([x_system, x_environment], 1)]
    for _ in range(records):
        eta = torch.randn((stride, 2, trajectories), generator=gen, dtype=F64, device=gen.device).cpu().numpy()
        for e_s, e_e in eta * math.sqrt(2 * dt):
            d = x_system - x_environment
            f_s, f_e = (
                -4 * A * x_system * (x_system * x_system - 1) - K_C * d,
                K_C * d - K_B * (x_environment - U_EQ),
            )  # -grad U_tot
            x_system, x_environment = x_system + dt * f_s + e_s, x_environment + dt * f_e + e_e
        out.append(np.stack([x_system, x_environment], 1))
    return torch.from_numpy(np.stack(out, 1))


def direct(prior: Prior, n: int, gen: torch.Generator) -> Tensor:
    """Draw (n, 1) prior samples by denoising standard-normal observations from t=1."""
    y = torch.randn((n, 1), generator=gen, dtype=F64, device=gen.device)
    return prior.denoise(y, torch.ones(n, dtype=F64, device=gen.device), gen)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Scan:
    """Settings for fixed-time scans, replica exchange, direct sampling and reference dynamics.

    start supplies (y, x_env); ladder uses a shared context constructed at t_prod. reference=None disables reference dynamics.
    """

    seed: int
    device: str
    precision: dict  # Fields accepted by Precision.
    checkpoint: str
    chains: int
    start: list[float]  # Initial (y, x_env) for each chain.
    scan_t: list[float]  # Fixed times; use deconvolution where admissible, otherwise the bare context.
    scan_sweeps: int
    long: dict[float, int]  # Map fixed times to sweep budgets.
    ladder: list[float]  # Replica times sharing the context constructed at t_prod.
    t_prod: float
    re_sweeps: int
    direct_samples: int
    reference: dict | None  # Keyword arguments for md_reference; None disables it.


def points(cfg: Scan) -> list[runner.Point]:
    """Enumerate the fixed-time scan, long chains, exchange ladder, direct draws and optional reference."""
    specs = [("scan", cfg.scan_sweeps, {"t": cfg.scan_t})]
    specs += [(f"long_t{t:.2f}", n, {"t": [t]}) for t, n in cfg.long.items()]
    specs += [(f"re_tp{cfg.t_prod:.2f}", cfg.re_sweeps, {"t": cfg.ladder, "t_prod": cfg.t_prod})]
    specs += [("direct", 0, {"t": [1.0], "n": cfg.direct_samples})]
    specs += [] if cfg.reference is None else [("reference", 0, dict(cfg.reference))]
    return [runner.Point(i, *spec) for i, spec in enumerate(specs)]


def sample_point(cfg: Scan, point: runner.Point, out: Path) -> None:
    """Run one point and save its configuration and samples.

    Gibbs records contain float32 clean samples with axes (sweeps, replicas, chains, copies, coordinates), plus acceptance records for exchange. Direct and reference samples have no sampler checkpoint.
    """
    dev = torch.device(cfg.device)
    gen = torch.Generator(dev).manual_seed(runner.point_seed(cfg.seed, point.id))
    if point.id == "reference":
        runner.write_resolved(out, cfg, point, [], (0.0,), (None,), [])
        t0 = time.perf_counter()
        x = md_reference(gen, **point.values)
        return runner.finish(out, {"x": x.numpy()}, None, time.perf_counter() - t0)
    net, kernel, meta = load_net(cfg.checkpoint, dev, Precision(**cfg.precision))
    prior = ddpm(v_to_eps(net, kernel), kernel)
    projection = Select([[0]])
    t, fixed_q = tuple(point.values["t"]), point.values.get("t_prod")
    own = tuple(c if admissible(kernel, c, K_C) else None for c in t)
    context_times = own if fixed_q is None else (fixed_q,) * len(t)
    contexts = {c: context(kernel, c) for c in dict.fromkeys(context_times)}
    initial = torch.tensor(cfg.start, dtype=F64, device=dev).expand(cfg.chains, 2)
    samplers = (
        []
        if point.id == "direct"
        else [
            compose(
                [(prior, projection)],
                contexts[c],
                initial=initial,
                t=tr,
                seed=runner.point_seed(cfg.seed, f"{point.id}/replica{r}"),
            )
            for r, (tr, c) in enumerate(zip(t, context_times, strict=True))
        ]
    )
    runner.write_resolved(out, cfg, point, [prior], t, context_times, [meta], samplers=samplers)
    t0, records, accepted = time.perf_counter(), [], None
    if point.id == "direct":
        arrays, saved = {"x": direct(prior, point.values["n"], gen).float().cpu().numpy()}, None
    else:
        for _ in range(point.sweeps):
            accepted = runner.step(samplers, exchange=fixed_q is not None)
            rec = {"x": torch.stack([s.state.prior_samples[0] for s in samplers]).float()}
            records.append(rec | ({} if fixed_q is None else {"accepted": accepted}))
        arrays, saved = stack(records), runner.snapshot(samplers, accepted)
    runner.finish(out, arrays, saved, time.perf_counter() - t0)


if __name__ == "__main__":
    raise SystemExit(runner.main(sys.argv[1:], {"sample": (Scan, points, sample_point)}))
