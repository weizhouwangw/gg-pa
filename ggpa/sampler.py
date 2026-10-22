"""Gibbs updates, state snapshots, restart and replica exchange.

Each full step updates the clean prior variables, then the explicit system state. State or time edits provide warm starts, not equilibrium samples for the changed target.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from .conventions import attachments, check_log_density, check_tensor, coupling_energy, prepare_initial, times, validate


def compose(priors, context, *, initial=None, context_state=None, t=0.1, seed=0, dtype=torch.float64, device=None):
    """Connect prior/projection pairs to a fixed context and initialize a Gibbs sampler.

    initial has shape (B, *state_shape). If omitted, context.initial_state() supplies coordinates and persistent context state, such as MD velocities. Coordinates default to float64; a scalar t applies to all attachments, while a tuple assigns one time to each.

    Initialization draws no randomness. Clean prior samples start as copies of the projections, so the initial snapshot is a warm start.
    """
    s, context_state = prepare_initial(context, initial, context_state, dtype=dtype, device=device)
    return GibbsSampler(attachments(priors), context, s, context_state=context_state, t=t, seed=seed)


@dataclass(frozen=True)
class State:
    """Snapshot of system coordinates, clean prior samples and persistent context state.

    s has shape (B, *state_shape). Each entry of prior_samples has shape (B, M, *prior_shape) and represents the current clean variables. context_state holds auxiliary update state, such as velocities; step counts completed context sweeps.

    Treat tensors as read-only and use sampler.set_state() to replace values.
    """

    s: torch.Tensor
    prior_samples: tuple[torch.Tensor, ...]
    context_state: object = None
    step: int = 0


_KEEP = object()


class GibbsSampler:
    """Alternate prior and context updates with an independent random generator.

    Construct through compose(). The context factor stays fixed when coupling times change. Components preserve their declared layouts after initialization. A failed update leaves the snapshot unchanged but may consume randomness or advance an external engine.
    """

    def __init__(self, couplings, context, s, *, context_state, t, seed):
        self.couplings, self.context = couplings, context
        self._t = times(t, len(couplings))
        validate(couplings, self._t, s, context)
        self.rng = torch.Generator(device=s.device).manual_seed(seed)
        self._state = State(s, tuple(c.project(s).detach().clone() for c in couplings), copy.deepcopy(context_state))
        self._validate_context_state(context_state, s)
        claim = getattr(context, "_claim", None)
        if callable(claim):
            claim(self)

    @property
    def state(self):
        return self._state

    @property
    def t(self):
        return self._t

    def _validate_context_state(self, context_state, s):
        check = getattr(self.context, "validate_context_state", None)
        if callable(check):
            check(context_state, s)

    def set_time(self, t):
        """Validate and replace coupling times while retaining the fixed context factor."""
        proposed = times(t, len(self.couplings))
        validate(self.couplings, proposed, self.state.s, self.context)
        self._t = proposed

    def set_state(self, *, s=None, prior_samples=None, context_state=_KEEP):
        """Replace selected snapshot fields while keeping the others.

        Shapes, dtype and device must match the existing state. Changing coordinates alone neither resamples priors nor resets persistent context state.
        """
        s = self.state.s if s is None else s
        self._state = self._replacement(s=s, prior_samples=prior_samples, context_state=context_state, t=self.t)

    def _replacement(self, *, s, prior_samples, context_state, t):
        old = self.state
        check_tensor(s, "state", like=old.s, shape=old.s.shape, finite=False)
        validate(self.couplings, t, s, self.context)
        prior_samples = old.prior_samples if prior_samples is None else tuple(prior_samples)
        for x, ref in zip(prior_samples, old.prior_samples, strict=True):
            check_tensor(x, "prior state", like=ref, shape=ref.shape)
        context_state = old.context_state if context_state is _KEEP else context_state
        self._validate_context_state(context_state, s)
        return State(
            s.detach().clone(), tuple(x.detach().clone() for x in prior_samples), copy.deepcopy(context_state), old.step
        )

    def _prior_sweep(self, state):
        """Draw clean variables conditional on the projected system state."""
        prior_samples = tuple(
            coupling.sample_prior(state.s, time, self.rng)
            for coupling, time in zip(self.couplings, self.t, strict=True)
        )
        return State(state.s, prior_samples, state.context_state, state.step)

    def _context_sweep(self, state):
        """Update the system and persistent context state with clean variables held fixed."""
        s, context_state = self.context.update(
            state.s, state.prior_samples, self.couplings, self.t, state.context_state, self.rng
        )
        check_tensor(s, "context output", like=state.s, shape=state.s.shape)
        self._validate_context_state(context_state, s)
        return State(s.detach().clone(), state.prior_samples, copy.deepcopy(context_state), state.step + 1)

    @torch.no_grad()
    def _advance(self, *, priors, context):
        state = self._prior_sweep(self.state) if priors else self.state
        self._state = self._context_sweep(state) if context else state
        return self.state

    def update_priors(self):
        """Update clean prior samples conditional on the current system coordinates.

        Coordinates, persistent context state and the context-sweep count stay unchanged.
        """
        return self._advance(priors=True, context=False)

    def update_context(self):
        """Update the system and persistent context state conditional on the clean samples.

        Increment the context-sweep count. Repeated calls apply further context transitions to the same clean samples.
        """
        return self._advance(priors=False, context=True)

    def step(self):
        """Run both Gibbs sweeps and commit the snapshot after both succeed."""
        return self._advance(priors=True, context=True)

    def state_dict(self):
        """Copy the state, coupling times and RNG state for restart.

        The ggpa-sampler/1 record stores prior_samples as xs and context_state as aux. Engine contexts may include a native checkpoint. Model weights and fixed context parameters are excluded; restore into an equivalent composition.
        """
        checkpoint = getattr(self.context, "checkpoint", None)
        return copy.deepcopy(
            {
                "format": "ggpa-sampler/1",
                "s": self.state.s,
                "xs": self.state.prior_samples,
                "aux": self.state.context_state,
                "step": self.state.step,
                "t": self.t,
                "rng": self.rng.get_state(),
                "context": checkpoint() if callable(checkpoint) else None,
            }
        )

    def load_state_dict(self, record):
        """Restore a record from state_dict() into an equivalent composition.

        Check state and time compatibility before restoring. Native engine checkpoints require a compatible restore hook; recovery from an engine failure is adapter-specific.
        """
        if record["format"] != "ggpa-sampler/1":
            raise ValueError("expected a ggpa-sampler/1 checkpoint")
        t = times(tuple(record["t"]), len(self.couplings))
        proposed = self._replacement(s=record["s"], prior_samples=record["xs"], context_state=record["aux"], t=t)
        if record["context"] is not None:
            self.context.restore(record["context"], self.couplings, t, proposed.s)

        self.rng.set_state(record["rng"])
        self._t = t
        self._state = State(proposed.s, proposed.prior_samples, proposed.context_state, record["step"])

    def _check_exchange(self, other, shared_context):
        """Check shared priors, projections and compatible context laws before scoring a swap."""
        # Both slots must describe the same implicit priors and compatible augmented states.
        state, other_state = self.state, other.state
        if any(
            p.prior is not q.prior or p.projection is not q.projection
            for p, q in zip(self.couplings, other.couplings, strict=True)
        ):
            raise ValueError("exchange requires shared prior and projection objects in the same order")
        context_key, other_context_key = (
            getattr(self.context, "exchange_key", None),
            getattr(other.context, "exchange_key", None),
        )
        shared_factor = (
            shared_context
            or self.context is other.context
            or (context_key is not None and context_key == other_context_key)
        )
        if not shared_factor and (state.context_state is not None or other_state.context_state is not None):
            raise ValueError("exchange with auxiliary state needs a declared shared context/auxiliary law")
        return state, other_state, shared_factor

    def _context_exchange_scores(self, other, state, other_state):
        """Evaluate and validate the four fixed-context log factors used by exchange."""
        scores = [
            self.context.log_factor(other_state.s),
            self.context.log_factor(state.s),
            other.context.log_factor(state.s),
            other.context.log_factor(other_state.s),
        ]
        for score in scores:
            check_log_density(score, "context log_factor", shape=(len(state.s),), device=state.s.device)
        check_tensor(scores[1], "current context log_factor")
        check_tensor(scores[3], "other current context log_factor")
        return scores

    @torch.no_grad()
    def exchange(self, other, *, shared_context=False):
        """Attempt a full augmented-state swap for each pair of batch rows.

        Coordinates, clean samples and supported persistent state move together. Times, RNG streams and step counts stay with their sampler slots. Priors and projections must be the same objects in the same order.

        Both samplers must have matching state layouts, dtypes and devices. Persistent state must be None, batched tensors on the state device, or matching dictionaries of these; its validity is the context's responsibility.

        A shared context cancels from the acceptance ratio. Otherwise both contexts must provide log_factor. shared_context=True explicitly asserts equal base factors and auxiliary equilibrium laws.
        """
        state, other_state, shared_factor = self._check_exchange(other, shared_context)

        # Swapping the clean variables with s cancels identical prior factors. The coupling log ratio is U_a(z_a) + U_b(z_b) - U_a(z_b) - U_b(z_a).
        couplings, other_couplings = self.couplings, other.couplings
        own_a = coupling_energy(state.s, state.prior_samples, couplings, self.t)
        own_b = coupling_energy(other_state.s, other_state.prior_samples, other_couplings, other.t)
        check_tensor(own_a, "current coupling energy")
        check_tensor(own_b, "other current coupling energy")
        log_accept_ratio = (
            own_a
            + own_b
            - coupling_energy(other_state.s, other_state.prior_samples, couplings, self.t)
            - coupling_energy(state.s, state.prior_samples, other_couplings, other.t)
        )

        # A different fixed context contributes its own cross-state log-density ratio.
        if not shared_factor:
            scores = self._context_exchange_scores(other, state, other_state)
            log_accept_ratio = log_accept_ratio + scores[0] - scores[1] + scores[2] - scores[3]
        if torch.isnan(log_accept_ratio).any():
            raise ValueError("exchange log ratio is undefined")

        accept = (
            torch.log(torch.rand((len(state.s),), generator=self.rng, device=state.s.device, dtype=torch.float64))
            < log_accept_ratio
        )

        # Commit accepted rows together: coordinates, clean samples and persistent state.
        def swap(x, y):
            return torch.where(accept.reshape(len(accept), *((1,) * (x.ndim - 1))), y, x)

        context_state, other_context_state = _swap_context_state(state.context_state, other_state.context_state, accept)
        self._state = State(
            swap(state.s, other_state.s),
            tuple(swap(x, y) for x, y in zip(state.prior_samples, other_state.prior_samples, strict=True)),
            context_state,
            state.step,
        )
        other._state = State(
            swap(other_state.s, state.s),
            tuple(swap(y, x) for x, y in zip(state.prior_samples, other_state.prior_samples, strict=True)),
            other_context_state,
            other_state.step,
        )
        return accept


def _swap_context_state(a, b, mask):
    """Swap matching auxiliary structures using a per-chain acceptance mask."""
    if a is None:
        return None, None
    if isinstance(a, torch.Tensor):
        m = mask.reshape(len(mask), *((1,) * (a.ndim - 1)))
        return torch.where(m, b, a), torch.where(m, a, b)
    pairs = {k: _swap_context_state(a[k], b[k], mask) for k in a}
    return {k: v[0] for k, v in pairs.items()}, {k: v[1] for k, v in pairs.items()}
