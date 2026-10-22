"""Context interface, custom updates and built-in context factories."""

from __future__ import annotations

import math
from typing import Protocol

import torch

from ..conventions import check_tensor, coupling_energy
from .analytic import admissible, deconvolve, deconvolved_quadratic, gaussian, quadratic

__all__ = [
    "Context",
    "admissible",
    "coupling_energy",
    "custom",
    "deconvolve",
    "deconvolved_quadratic",
    "from_energy",
    "gaussian",
    "quadratic",
]


class Context(Protocol):
    """Interface for a fixed context's conditional update.

    update(s, prior_samples, couplings, t, context_state, rng) returns new system and persistent context state. Inputs are read-only; couplings provide the matching prior kernels and projections.

    Only update is required. Validation, initialization and log-factor evaluation are optional hooks.
    """

    def update(self, s, prior_samples, couplings, t, context_state, rng): ...


def custom(update, *, validate=None, initial_state=None, log_factor=None):
    """Wrap an update function and optional context hooks without subclassing."""
    from types import SimpleNamespace

    if not callable(update):
        raise TypeError("update must be callable")
    hooks = {"update": update, "validate": validate, "initial_state": initial_state, "log_factor": log_factor}
    if any(fn is not None and not callable(fn) for fn in hooks.values()):
        raise TypeError("context hooks must be callable or None")
    return SimpleNamespace(**hooks)


def from_energy(energy, *, step_size, steps=1):
    """Construct a context from a dimensionless energy using MALA transitions.

    energy(s) returns one differentiable value per chain for the physical context alone. The update adds the coupling energies with clean prior samples held fixed.

    step_size is h in s-h*grad(U)+sqrt(2h)*noise. Each inner step draws a normal proposal and uniform acceptance variable. Acceptance includes the reverse/forward proposal correction and requires finite proposed energy and gradient.
    """
    if not math.isfinite(step_size) or step_size <= 0 or not isinstance(steps, int) or steps < 1:
        raise ValueError("step_size must be positive and finite; steps must be a positive integer")

    def log_factor(s):
        return -energy(s)

    def update(s, prior_samples, couplings, t, context_state, rng):
        h = step_size

        def value_grad(z):
            with torch.enable_grad():
                z = z.detach().requires_grad_(True)
                base = energy(z)
                check_tensor(base, "context energy", shape=(len(z),), finite=False)
                if base.device != z.device:
                    raise ValueError("context energy must remain on the state device")
                u = base + coupling_energy(z, prior_samples, couplings, t)
                grad = torch.autograd.grad(u.sum(), z)[0]
                return u.detach(), grad.detach()

        u, grad = value_grad(s)
        if not torch.isfinite(u).all() or not torch.isfinite(grad).all():
            raise ValueError("MALA needs finite energy and gradient at its current state")
        for _ in range(steps):
            y = s - h * grad + math.sqrt(2 * h) * torch.randn(s.shape, generator=rng, device=s.device, dtype=s.dtype)
            uy, gy = value_grad(y)
            fwd = (y - s + h * grad).square().flatten(1).sum(1)
            bwd = (s - y + h * gy).square().flatten(1).sum(1)
            # log pi(y)/pi(s) + log q(s|y)/q(y|s); proposal covariance is 2h I.
            log_a = u - uy + (fwd - bwd) / (4 * h)
            ok = torch.log(torch.rand((len(s),), generator=rng, device=s.device, dtype=s.dtype)) < log_a
            ok &= torch.isfinite(uy) & torch.isfinite(gy).flatten(1).all(1)
            mask = ok.reshape(len(s), *((1,) * (s.ndim - 1)))
            s, u, grad = torch.where(mask, y, s), torch.where(ok, uy, u), torch.where(mask, gy, grad)
        return s, context_state

    return custom(update, log_factor=log_factor)
