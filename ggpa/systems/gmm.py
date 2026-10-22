"""Gaussian-mixture inverse problem and split Gibbs comparison.

Use a 25-component prior, an identity projection and a constant Gaussian split coupling N(z; x, rho^2 I). Matched and unmatched contexts differ in their observation covariance.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal

from ggpa import Prior, compose
from ggpa.contexts import deconvolve, quadratic
from ggpa.priors import mixture_posterior
from ggpa.priors.kernels import Gaussian, GeometricVE
from ggpa.projection import Select
from ggpa.systems import runner
from ggpa.utils import stack

F64 = torch.float64
_AXIS = torch.tensor([-5.0, -2.5, 0.0, 2.5, 5.0], dtype=F64)
MEANS, SIGMA_X = torch.cartesian_prod(_AXIS, _AXIS), 0.8  # Equally weighted isotropic components.
# Linear observation Y = H*x + noise.
H, Y, SIGMA_ETA = torch.tensor([[1.0, 0.0]], dtype=F64), torch.zeros(1, dtype=F64), 2.5
RHO_MAX = SIGMA_ETA / torch.linalg.matrix_norm(H, ord=2).item()  # Strict noise bound for a positive matched covariance.


def grid25_prior(kernel: Gaussian) -> Prior:
    """Construct the exact conditional for 25 equally weighted isotropic Gaussian components."""
    covs = SIGMA_X**2 * torch.eye(2, dtype=F64).expand(25, 2, 2)
    return mixture_posterior(torch.full((25,), 1 / 25, dtype=F64), MEANS, covs, kernel)


def likelihood(kernel: Gaussian, matched: bool):
    """Construct the fixed quadratic observation factor on z.

    The energy is z.T*P*z/2-b.T*z, with P=H.T*Sigma_split^-1*H and b=H.T*Sigma_split^-1*Y. The unmatched factor uses Sigma_split=SIGMA_ETA^2 I.

    The matched factor subtracts rho^2*H*H.T from this covariance and requires rho<RHO_MAX. Its precision is computed by spectral deconvolution at the constant VE noise level.
    """
    eye = torch.eye(len(H), dtype=F64)
    if matched:
        lam, V = torch.linalg.eigh(H.T @ H / SIGMA_ETA**2)
        P = V @ torch.diag(deconvolve(lam.clamp(min=0), kernel, 0.0)) @ V.T  # = H^T Sigma_split^-1 H
        split = SIGMA_ETA**2 * eye - kernel.level(torch.zeros(1, dtype=F64)).sigma.item() ** 2 * H @ H.T
    else:
        split = SIGMA_ETA**2 * eye
        P = H.T @ torch.linalg.solve(split, H)
    b = H.T @ torch.linalg.solve(split, Y)  # Observation natural parameter H.T*Sigma_split^-1*Y.
    return quadratic(P, b)


def posterior(x: Tensor, var: float = SIGMA_ETA**2) -> Tensor:
    """Evaluate the normalized Gaussian-mixture posterior at x shaped (B, 2).

    var=SIGMA_ETA^2 gives the physical posterior; adding rho^2 gives the unmatched sampler's clean marginal for this observation matrix. Return float64 densities of shape (B,).

    Component weights follow the observation likelihood. Conditional covariance is (C^-1+H.T*H/var)^-1 and means follow from precision-weighted prior means and observations.
    """
    c_inv = torch.eye(2, dtype=F64) / SIGMA_X**2
    cov = torch.linalg.inv(c_inv + H.T @ H / var)
    means = (MEANS @ c_inv + H.T @ Y / var) @ cov  # (25, 2); cov is symmetric
    log_w = MultivariateNormal(MEANS @ H.T, var * torch.eye(1, dtype=F64) + SIGMA_X**2 * H @ H.T).log_prob(Y)
    log_p = MultivariateNormal(means, cov).log_prob(torch.as_tensor(x, dtype=F64)[:, None])  # (n, 25)
    return torch.logsumexp(torch.log_softmax(log_w, 0) + log_p, 1).exp()


@dataclasses.dataclass(frozen=True, kw_only=True)
class Split:
    """Settings for Gaussian-mixture split Gibbs runs."""

    seed: int
    device: str
    runs: list[dict]  # Each run specifies ratios r=rho/RHO_MAX, chains and sweeps, including burn-in.


def points(cfg: Split) -> list[runner.Point]:
    """Enumerate matched and unmatched runs for every noise ratio and chain count."""
    runs = [(r, s, u["chains"], u["sweeps"]) for u in cfg.runs for r in u["r"] for s in ("matched", "unmatched")]
    ids = [f"r{r:.2f}_{s}_c{c}" for r, s, c, _ in runs]
    return [runner.Point(i, ids[i], n, {"r": r, "split": s, "chains": c}) for i, (r, s, c, n) in enumerate(runs)]


def sample_point(cfg: Split, point: runner.Point, out: Path) -> None:
    """Run one split Gibbs point from z=0.

    Save float32 system records of shape (sweeps, 1, chains, 2) and clean records of shape (sweeps, 1, chains, 1, 2).
    """
    dev, rho, matched = torch.device(cfg.device), point.values["r"] * RHO_MAX, point.values["split"] == "matched"
    kernel = Gaussian(GeometricVE(rho, rho), (2,))  # Constant split coupling N(z; x, rho^2 I).
    prior = grid25_prior(kernel)
    sampler = compose(
        [(prior, Select([[0, 1]]))],
        likelihood(kernel, matched),
        t=0.0,
        initial=torch.zeros(point.values["chains"], 2, dtype=F64, device=dev),
        seed=runner.point_seed(cfg.seed, point.id),
    )
    runner.write_resolved(out, cfg, point, [prior], (0.0,), (0.0 if matched else None,), [], samplers=[sampler])
    records, t0 = [], time.perf_counter()
    for _ in range(point.sweeps):
        state = sampler.step()
        records.append({"s": state.s[None].float(), "x": state.prior_samples[0][None].float()})
    runner.finish(out, stack(records), runner.snapshot([sampler]), time.perf_counter() - t0)


if __name__ == "__main__":
    raise SystemExit(runner.main(sys.argv[1:], {"sample": (Split, points, sample_point)}))
