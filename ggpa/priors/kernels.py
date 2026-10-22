"""Forward coupling kernels and diffusion schedules.

Schedules map times in [0, 1] to signal amplitudes and noise scales. Kernels evaluate log densities for one prior's event shape. Stored training tables retain their precision; evaluated levels and densities use float64 on the input device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import torch
from torch import Tensor


class Level(NamedTuple):
    """Diffusion parameters for each observation row.

    alpha and sigma are float64 vectors. k is an integer grid-index vector, or None for a continuous schedule.
    """

    k: Tensor | None
    alpha: Tensor
    sigma: Tensor


def wrap(a: Tensor) -> Tensor:
    """Wrap angles in radians to [-pi, pi), preserving dtype and device."""
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def _unit(t: Tensor, where: str) -> Tensor:
    """Convert a time vector to float64 and reject values outside [0, 1], including NaN."""
    t = t.to(torch.float64)
    bad = torch.nonzero(~((t >= 0) & (t <= 1)))
    if len(bad):
        i = int(bad[0, 0])
        raise ValueError(f"t = {t[i].item():g} at row {i} is outside [0, 1] for {where}")
    return t


@dataclass(frozen=True, eq=False)
class DiscreteVP:
    """Evaluate a variance-preserving schedule from its trained alpha_bar table.

    At index k, alpha=sqrt(alpha_bar[k]) and sigma=sqrt(1-alpha_bar[k]). The table retains its original precision; evaluated levels use float64.
    """

    alpha_bar: Tensor

    def index(self, t: Tensor) -> Tensor:
        """Map t to round(t*(T-1)) using ties-to-even rounding; require t in [0, 1]."""
        T = self.alpha_bar.numel()
        t = _unit(t, f"the DiscreteVP grid of T = {T}; diffusion times are fractions of the trained grid")
        return torch.round(t * (T - 1)).long()

    def level(self, t: Tensor) -> Level:
        """Return grid indices and float64 signal/noise scales on the time vector's device."""
        k = self.index(t)
        ab = self.alpha_bar.to(k.device, torch.float64)[k]
        return Level(k, ab.sqrt(), (1 - ab).sqrt())


@dataclass(frozen=True)
class GeometricVE:
    """Evaluate alpha=1 and sigma=sigma_min*(sigma_max/sigma_min)^t for t in [0, 1].

    Equal endpoints give a constant Gaussian split coupling. With above_floor, sigma is sqrt(sigma(t)^2 - sigma_min^2), the noise between a draw at the sigma_min floor and level t (the coupling of torus_ancestral); t must then be positive.
    """

    sigma_min: float
    sigma_max: float
    above_floor: bool = False

    def level(self, t: Tensor) -> Level:
        """Return float64 signal/noise scales for the supplied time vector."""
        t = _unit(
            t,
            f"the GeometricVE schedule (sigma {self.sigma_min:g} to {self.sigma_max:g}); a VE network is trained "
            "on t in [0, 1] only",
        )
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        if self.above_floor:
            if bool((t == 0).any()):
                raise ValueError("t = 0 leaves no noise above the sigma_min floor; the coupling needs t > 0")
            sigma = torch.sqrt(sigma.square() - self.sigma_min**2)
        return Level(None, torch.ones_like(t), sigma)


@dataclass(frozen=True, eq=False)
class Gaussian:
    """Evaluate log N(y; alpha*x, sigma^2 I) for a prior with the given event shape.

    The schedule supplies level(t). log_q sums over event dimensions and retains batch and copy axes.
    """

    schedule: DiscreteVP | GeometricVE  # or any object with level(t) -> Level
    shape: tuple[int, ...]
    period: ClassVar[None] = None

    def level(self, t: Tensor) -> Level:
        """Evaluate the underlying schedule."""
        return self.schedule.level(t)

    def log_q(self, y: Tensor, x: Tensor, lv: Level) -> Tensor:
        """Return float64 log densities of shape (B, M) for inputs shaped (B, M, *shape)."""
        r = y.flatten(2).double() - lv.alpha[:, None, None] * x.flatten(2).double()  # (n, M, d)
        s2 = lv.sigma.square()[:, None]  # (n, 1)
        # log N(y; alpha x, sigma^2 I) = -(d/2) log(2 pi sigma^2) - |y - alpha x|^2 / (2 sigma^2)
        return -0.5 * r.shape[-1] * torch.log(2 * math.pi * s2) - r.square().sum(-1) / (2 * s2)


@dataclass(frozen=True, eq=False)
class WrappedGaussian:
    """Evaluate an angular Gaussian coupling using a finite sum of winding images.

    For each angle, sum images from -windings to windings around the nearest-image displacement. The default is two images on either side; at least one is required.

    A GeometricVE schedule keeps alpha=1. log_q sums over event angles and retains batch and copy axes.
    """

    schedule: GeometricVE
    shape: tuple[int, ...]
    windings: int = 2
    period: ClassVar[float] = 2 * math.pi

    def __post_init__(self):
        if type(self.schedule) is not GeometricVE:
            raise ValueError(
                f"WrappedGaussian needs a GeometricVE schedule (alpha = 1); got "
                f"{type(self.schedule).__name__}. Use Gaussian for a VP prior"
            )
        if self.windings < 1:
            raise ValueError(
                f"windings = {self.windings}: at least the images m = -1..1 are needed (m = 0 alone is off by up to "
                "ln 2 at |w| = pi)"
            )

    def level(self, t: Tensor) -> Level:
        """Evaluate the underlying schedule."""
        return self.schedule.level(t)

    def log_q(self, y: Tensor, x: Tensor, lv: Level) -> Tensor:
        """Return float64 log densities of shape (B, M) for angular inputs in radians."""
        w = wrap(y.flatten(2).double() - x.flatten(2).double())  # Nearest-image angular displacement.
        s2 = lv.sigma.square()[:, None, None]  # (n, 1, 1)
        m = torch.arange(1, self.windings + 1, dtype=torch.float64, device=w.device)
        m = torch.cat([-m, m])  # Nonzero winding images.
        # Log image ratios, all nonpositive.
        e = -(2 * math.pi * m * w[..., None] + 2 * math.pi**2 * m**2) / s2[..., None]
        # Factor out the nearest image. The remaining image ratios are <= 1, so log1p evaluates their correction without overflow.
        return (-0.5 * torch.log(2 * math.pi * s2) - w.square() / (2 * s2) + torch.log1p(e.exp().sum(-1))).sum(-1)
