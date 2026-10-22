"""DDPM and wrapped-VE conditional samplers.

All draws use the supplied generator. Noise schedules and network precision follow the trained model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace

import torch
from torch import Tensor, nn

from ..conventions import Prior
from .kernels import Gaussian, WrappedGaussian, wrap


def _ragged(bound: Tensor, descending: bool = False) -> Iterator[tuple[int, Tensor]]:
    """Yield each step and the sorted rows whose integer limits satisfy bound > step.

    Reverse step order when descending=True. Precompute nested row sets with one CPU transfer to avoid synchronization at every reverse step. Preserving row order preserves the assignment of random draws.
    """
    c = torch.bincount(bound.cpu())  # rows per bound value
    v = torch.nonzero(c).squeeze(1)  # (D,) the distinct bounds, ascending
    sizes = c.flip(0).cumsum(0).flip(0)[v]  # rows with bound >= v_j
    vd = v.pin_memory().to(bound.device, non_blocking=True) if bound.is_cuda else v
    flat = torch.nonzero_static((bound >= vd[:, None]).flatten(), size=int(sizes.sum())).squeeze(1)  # (j, row) order
    spans = list(zip([0, *v[:-1].tolist()], v.tolist(), (flat % len(bound)).split(sizes.tolist()), strict=True))
    for lo, hi, rows in reversed(spans) if descending else spans:
        for i in range(hi - 1, lo - 1, -1) if descending else range(lo, hi):
            yield i, rows


def ddpm(eps: Callable[[Tensor, Tensor], Tensor], kernel: Gaussian) -> Prior:
    """Construct an ancestral DDPM approximation to the prior conditional.

    The callable eps(x, k) predicts noise from float32 observations and integer training indices. Denoising starts at the observation's index k(t), uses the kernel's trained alpha_bar table, and returns float32 clean samples.

    Each step evaluates only active rows. Nonterminal steps draw one normal vector per active row; coefficients retain the schedule's stored precision.
    """
    ab = kernel.schedule.alpha_bar
    alpha = torch.cat([ab[:1], ab[1:] / ab[:-1]])  # alpha_k = abar_k / abar_{k-1}, alpha_0 = abar_0
    beta = 1.0 - alpha

    def denoise(y: Tensor, t: Tensor, gen: torch.Generator) -> Tensor:
        k0 = kernel.schedule.index(t)
        x = y.to(torch.float32, copy=True)
        for k, rows in _ragged(k0 + 1, descending=True):  # k = max(k0) .. 0 over the rows with k0 >= k
            xa, sigma = x[rows], torch.sqrt(1.0 - ab[k])
            e = eps(xa, torch.full_like(rows, k))
            # mean = (x - beta_k / sigma_k eps) / sqrt(alpha_k), sigma_k = sqrt(1 - abar_k)
            x_new = (xa - (beta[k] / sigma) * e) / torch.sqrt(alpha[k])
            if k > 0:
                # Conditional variance of the forward Gaussian bridge.
                btilde = (1.0 - ab[k - 1]) / (1.0 - ab[k]) * beta[k]
                z = torch.randn(xa.shape, generator=gen, dtype=xa.dtype, device=xa.device)
                x_new = x_new + torch.sqrt(btilde) * z
            x[rows] = x_new
        return x

    return Prior(denoise, kernel)


def v_to_eps(net: nn.Module, kernel: Gaussian) -> Callable[[Tensor, Tensor], Tensor]:
    """Convert velocity prediction to noise prediction using eps = alpha*v + sigma*x.

    The network receives float32 time k/(T-1). Use the same kernel for this adapter and ddpm so both read the same trained alpha_bar table. Schedule values are cached per device.
    """
    ab = kernel.schedule.alpha_bar
    T, on = ab.numel(), {}  # Cache the trained table per device.

    def eps(x: Tensor, k: Tensor) -> Tensor:
        v = net(x, (k.double() / (T - 1)).float())
        if x.device not in on:
            on[x.device] = ab.to(x.device)
        a = on[x.device][k].view(-1, *(1,) * (x.dim() - 1))  # (n, 1, ...) abar_k per row
        return torch.sqrt(a) * v + torch.sqrt(1.0 - a) * x

    return eps


def torus_ancestral(score: nn.Module, kernel: WrappedGaussian, steps: int) -> Prior:
    """Construct the wrapped-VE ancestral (Gaussian-bridge) conditional sampler on a finite time grid.

    score(x, t) returns sigma_t times the score from float32 angles and times. Each row uses n=max(int(t*steps), 2) intervals on t_i=t*(1-i/n), computed in float64. With r=sigma_next^2/sigma^2 and the clean estimate x0=wrap(x+sigma(t_i)*score(x,t_i)), each step draws wrap(x0+r*wrap(x-x0)+sigma_next*sqrt(1-r)*noise).

    The final draw keeps the sigma_min noise floor at t=0: the prior is the data at that floor, and the returned kernel couples it to level t with variance sigma(t)^2 - sigma_min^2. Outputs and random draws use float32; schedule calculations use float64.
    """

    def denoise(y: Tensor, t: Tensor, gen: torch.Generator) -> Tensor:
        t = t.to(torch.float64)
        n = torch.clamp((t * steps).long(), min=2)
        x = y.to(torch.float32, copy=True)
        for i, rows in _ragged(n):
            xa, tr, nr = x[rows], t[rows], n[rows]
            b = (-1, *(1,) * (xa.dim() - 1))  # broadcast (m,) over the event dims
            t_i, t_next = tr * (1 - i / nr.double()), tr * (1 - (i + 1) / nr.double())
            s, s1 = kernel.level(t_i).sigma, kernel.level(t_next).sigma
            r = (s1 / s).square()  # Gaussian bridge variance ratio.
            # Wrapped clean estimate: x + sigma^2*grad(log p_t).
            x0 = wrap(xa + s.float().view(b) * score(xa, t_i.float()))
            z = torch.randn(xa.shape, generator=gen, dtype=xa.dtype, device=xa.device)
            x[rows] = wrap(x0 + r.float().view(b) * wrap(xa - x0) + (s1 * torch.sqrt(1 - r)).float().view(b) * z)
        return x

    return Prior(denoise, replace(kernel, schedule=replace(kernel.schedule, above_floor=True)))
