"""Compose conditional priors with fixed contexts into a stateful Gibbs sampler."""

from .conventions import Prior
from .sampler import GibbsSampler, compose

__all__ = ["GibbsSampler", "Prior", "compose"]
__version__ = "2.0.1"
