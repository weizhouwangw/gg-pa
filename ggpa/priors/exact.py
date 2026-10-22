"""Gaussian-mixture conditionals and grid approximations for reference calculations."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal

from ..conventions import Prior
from .kernels import Gaussian, WrappedGaussian, wrap


def _categorical(logits: Tensor, gen: torch.Generator) -> Tensor:
    """Sample one category per row of logits using one float64 uniform draw and the inverse CDF."""
    cdf = torch.softmax(logits, -1).cumsum(-1)
    u = torch.rand((len(cdf), 1), generator=gen, dtype=torch.float64, device=cdf.device)
    return torch.searchsorted(cdf, u * cdf[:, -1:]).squeeze(1).clamp(max=cdf.shape[1] - 1)


def grid_posterior(
    log_p: Callable[[Tensor], Tensor], kernel: Gaussian | WrappedGaussian, axes: tuple[Tensor, ...]
) -> Prior:
    """Approximate a conditional prior using weights on a uniform Cartesian grid.

    axes contains one uniformly spaced vector per event dimension. log_p maps grid points of shape (G, *prior_shape) to G log prior weights. Posterior node weights are proportional to exp(log_p(x))*q_t(y|x).

    Select a node by inverse CDF, then add uniform jitter within its cell. Periodic samples wrap to [-pi, pi). Node probabilities match the tabulated weights; the continuous output is a grid approximation. Samples and random draws use float64.
    """
    grid = torch.cartesian_prod(*axes).reshape(-1, *kernel.shape).double()  # (G, *shape)
    width = torch.stack([a[1] - a[0] for a in axes]).reshape(kernel.shape).double()  # cell width per dimension
    lp = log_p(grid).double()

    def denoise(y: Tensor, t: Tensor, gen: torch.Generator) -> Tensor:
        g, n = grid.to(y.device), y.shape[0]
        lq = kernel.log_q(y[:, None].expand(n, *g.shape), g.expand(n, *g.shape), kernel.level(t))  # (n, G)
        j = _categorical(lp.to(y.device) + lq, gen)
        u = torch.rand(y.shape, generator=gen, dtype=torch.float64, device=y.device)
        x = g[j] + (u - 0.5) * width.to(y.device)
        return x if kernel.period is None else wrap(x)

    return Prior(denoise, kernel)


def mixture_posterior(weights: Tensor, means: Tensor, covs: Tensor, kernel: Gaussian) -> Prior:
    """Construct an exact Gaussian-mixture conditional under an isotropic Gaussian kernel.

    weights, means and covs have shapes (J,), (J, d) and (J, d, d). Select component j with weight proportional to w_j*N(y; alpha*m_j, alpha^2*C_j+sigma^2 I).

    Within the component, precision is C_j^-1+alpha^2/sigma^2 I and the natural parameter is C_j^-1*m_j+alpha*y/sigma^2. Each observation uses one uniform component draw and one normal vector. Calculations and outputs use float64.
    """
    if type(kernel) is not Gaussian:
        raise ValueError(
            f"mixture_posterior: a {type(kernel).__name__} kernel; the closed-form mixture posterior needs "
            "type(kernel) is Gaussian (N(y; alpha x, sigma^2 I) on R^d). Use grid_posterior, which reads kernel.log_q"
        )
    prec = torch.linalg.inv(covs.double())  # (J, d, d)

    def denoise(y: Tensor, t: Tensor, gen: torch.Generator) -> Tensor:
        lv, d, n = kernel.level(t), means.shape[1], y.shape[0]
        z = y.reshape(n, 1, d).double()  # (n, 1, d)
        m, C, P = means.to(z), covs.to(z), prec.to(z)
        a, s2 = lv.alpha.view(n, 1, 1), lv.sigma.square().view(n, 1, 1)
        eye = torch.eye(d, dtype=z.dtype, device=z.device)
        marginal = MultivariateNormal(a * m, a[..., None] ** 2 * C + s2[..., None] * eye, validate_args=False)  # (n, J)
        j = _categorical(torch.log(weights.to(z)) + marginal.log_prob(z), gen)
        # Condition the selected component by adding prior and observation precisions.
        V = torch.linalg.inv(P[j] + a**2 / s2 * eye)  # (n, d, d)
        mu = V @ (P[j] @ m[j, :, None] + a / s2 * z.mT)  # (n, d, 1)
        noise = torch.linalg.cholesky(V) @ torch.randn((n, d, 1), generator=gen, dtype=z.dtype, device=z.device)
        return (mu + noise).reshape(y.shape)

    return Prior(denoise, kernel)
