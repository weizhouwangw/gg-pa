"""Tensor layouts and component contracts for composition.

The system state s has shape (B, *state_shape). Each projection and clean prior sample has shape (B, M, *prior_shape), with B independent chains and M copies of one prior. A sampler fixes its state shape, device and float32 or float64 dtype.

Components treat inputs as read-only, preserve their output layouts, and draw randomness from the supplied generator. Priors may return either supported floating dtype; their outputs are converted to the state dtype. Units are defined by the components and projections.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real

import torch
from torch import Tensor


@dataclass(frozen=True, eq=False)
class Prior:
    """Pair a conditional sampler with its forward kernel.

    The callable denoise(y, t, rng) receives y shaped (B*M, *prior_shape) and one time per row. It returns clean samples with the same shape and device, using the supplied generator.

    The target conditional is p(x | y, t) proportional to p(x)*q_t(y | x). The kernel supplies shape, optional period, level(t), and log_q(y, x, level); denoise must use this same forward kernel.
    """

    denoise: Callable[[Tensor, Tensor, torch.Generator], Tensor]
    kernel: object


def prepare_initial(context, initial, context_state, *, dtype, device):
    """Resolve initial coordinates and convert them to the sampler's dtype and device.

    A context may supply initial_state() when initial is omitted. Direct tensor context state is converted here; structured state retains its component-defined layout and types. The sampler takes its own copy.
    """
    if not callable(getattr(context, "update", None)):
        raise TypeError(
            "context must provide update(s, prior_samples, couplings, t, context_state, rng); "
            "use contexts.custom for a function"
        )
    if initial is None:
        get_initial = getattr(context, "initial_state", None)
        if not callable(get_initial):
            raise ValueError("provide initial=(B, *state_shape), or a context with initial_state()")
        initial, supplied_context_state = get_initial()
        if context_state is None:
            context_state = supplied_context_state
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("state dtype must be torch.float32 or torch.float64")
    s = torch.as_tensor(initial, dtype=dtype, device=device).detach().clone()
    if isinstance(context_state, torch.Tensor):
        context_state_dtype = s.dtype if context_state.is_floating_point() else context_state.dtype
        context_state = context_state.detach().to(device=s.device, dtype=context_state_dtype)
    return s, context_state


def times(value, count: int) -> tuple[float, ...]:
    """Return one finite time per attachment, broadcasting a scalar when needed."""
    if isinstance(value, Real) and not isinstance(value, bool):
        result = (float(value),) * count
    elif isinstance(value, tuple) and len(value) == count:
        if any(not isinstance(v, Real) or isinstance(v, bool) for v in value):
            raise TypeError("t must contain real numbers")
        result = tuple(float(v) for v in value)
    else:
        raise ValueError(f"t must be a scalar or a tuple of {count} times (one per prior attachment)")
    if not all(math.isfinite(v) for v in result):
        raise ValueError("diffusion times must be finite")
    return result


def check_tensor(value, name: str, *, like: Tensor | None = None, shape=None, finite=True) -> Tensor:
    """Check a float32 or float64 tensor without copying or converting it.

    When supplied, like fixes the dtype and device, and shape fixes the dimensions. Values must be finite unless finite=False.
    """
    if not isinstance(value, Tensor) or value.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"{name} must be a float32 or float64 tensor")
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} has shape {tuple(value.shape)}; expected {tuple(shape)}")
    if like is not None and (value.device != like.device or value.dtype != like.dtype):
        raise ValueError(f"{name} must use device {like.device} and dtype {like.dtype}")
    if finite and not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def check_log_density(value, name, *, shape, device):
    """Check log-density shape and device, allowing -inf for zero density.

    NaN and positive infinity are invalid.
    """
    check_tensor(value, name, shape=shape, finite=False)
    if value.device != device or torch.isnan(value).any() or torch.isposinf(value).any():
        raise ValueError(f"{name} must be a log density on {device} (finite or -inf)")
    return value


def _check_projection(y, s, shape):
    """Check batch, copy and event axes before a projected state reaches a prior."""
    check_tensor(y, "projection output", like=s)
    if y.ndim != 2 + len(shape) or y.shape[0] != len(s) or tuple(y.shape[2:]) != shape or y.shape[1] < 1:
        raise ValueError(f"projection must return (B, M, {shape}) with B={len(s)} and M >= 1; got {tuple(y.shape)}")


@dataclass(frozen=True, eq=False)
class Coupling:
    """Bind a prior to its projection and evaluate the matching coupling density."""

    prior: object
    projection: object

    @property
    def kernel(self):
        return self.prior.kernel

    def project(self, s):
        return self.projection(s)

    def sample_prior(self, s, t, rng):
        """Sample flattened batch/copy rows and restore the projection's layout.

        Copy the result into the state dtype and check for conversion overflow.
        """
        observations = self.project(s)
        flat = observations.flatten(0, 1)
        batch_times = torch.full((len(flat),), t, dtype=torch.float64, device=s.device)
        sample = self.prior.denoise(flat, batch_times, rng)
        sample = sample.detach().to(dtype=s.dtype, copy=True)
        check_tensor(sample, "converted prior output", like=s, shape=flat.shape)
        return sample.reshape(observations.shape)

    def level(self, t, s):
        return self.kernel.level(torch.full((len(s),), t, device=s.device, dtype=torch.float64))

    def log_q(self, s, x, t):
        y = self.project(s)
        return self.kernel.log_q(y, x, self.level(t, s)).sum(1)


def attachments(priors) -> tuple[Coupling, ...]:
    """Validate prior/projection pairs and bind their callable interfaces and periodicity."""
    result = []
    for i, pair in enumerate(priors):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError(f"prior {i}: expected a (prior, projection) pair")
        c = Coupling(*pair)
        for name, fn in (
            ("prior.denoise", getattr(c.prior, "denoise", None)),
            ("projection", c.projection),
            ("kernel.level", getattr(c.kernel, "level", None)),
            ("kernel.log_q", getattr(c.kernel, "log_q", None)),
        ):
            if not callable(fn):
                raise TypeError(f"prior {i}: {name} must be callable")
        if getattr(c.projection, "period", None) != getattr(c.kernel, "period", None):
            raise ValueError(f"prior {i}: projection and kernel periodicity differ")
        result.append(c)
    if not result:
        raise ValueError("compose needs at least one prior/projection pair")
    return tuple(result)


def validate(couplings, t, s, context):
    """Check state layout, kernel time domains and context-specific assumptions."""
    check_tensor(s, "state")
    if s.ndim < 2 or len(s) < 1 or s[0].numel() < 1:
        raise ValueError("state must have shape (B, *event_shape) with a nonempty batch and event")
    for c, time in zip(couplings, t, strict=True):
        _check_projection(c.project(s), s, tuple(c.kernel.shape))
        c.level(time, s)  # Validate the kernel-specific time domain.
    check = getattr(context, "validate", None)
    if callable(check):
        check(couplings, t, s)


def coupling_energy(s, prior_samples, couplings, t):
    """Return -sum_i log q_i(Phi_i(s) | x_i, t_i) for each chain.

    Each attachment sums over its prior copies. The fixed physical context energy is excluded.
    """
    energy = -sum(c.log_q(s, x, time) for c, x, time in zip(couplings, prior_samples, t, strict=True))
    check_log_density(-energy, "coupling log density", shape=(len(s),), device=s.device)
    return energy
