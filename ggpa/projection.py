"""Project system coordinates into the observations consumed by priors.

A projection maps (B, *state_shape) to (B, M, *prior_shape), preserving dtype and device. M counts copies of a prior. Outputs use the prior's units; periodic projections declare a period matching the kernel.

Constructors copy tensor parameters. Treat the stored tensors as read-only during sampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor

from .priors.kernels import wrap


@dataclass(frozen=True, eq=False)
class Cartesian:
    """Select atom coordinates with optional geometric centering and unit conversion.

    atoms has shape (M, N), listing each copy's atoms in model order. Inputs have shape (B, system_atoms, 3) in nm; outputs have shape (B, M, N, 3) in prior units. scale is nm per prior unit: 1 for nm or 0.1 for angstrom.

    center subtracts each copy's arithmetic mean. No rotation alignment or periodic unwrapping is applied; molecules must be whole.
    """

    atoms: Tensor
    center: bool
    scale: float = 1.0
    period: ClassVar[None] = None

    def __post_init__(self):
        atoms = torch.as_tensor(self.atoms)
        if atoms.ndim != 2 or min(atoms.shape) < 1 or atoms.dtype not in (torch.int32, torch.int64):
            raise ValueError("Cartesian atoms must be a nonempty integer array (copies, atoms_per_copy)")
        if (atoms < 0).any() or any(len(row.unique()) != len(row) for row in atoms):
            raise ValueError("Cartesian atom indices must be nonnegative and unique within each copy")
        if not isinstance(self.center, bool) or not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("center must be bool and scale must be positive and finite")

        object.__setattr__(self, "atoms", atoms.detach().long().clone())

    def __call__(self, s: Tensor) -> Tensor:
        if s.ndim != 3 or s.shape[-1] != 3 or int(self.atoms.max()) >= s.shape[1]:
            raise ValueError("Cartesian requires (B, system_atoms, 3) with every selected atom present")

        coordinates = s[:, self.atoms.to(s.device)]
        if self.center:
            coordinates = coordinates - coordinates.mean(-2, keepdim=True)
        return coordinates / self.scale


def minimum_image(d: Tensor, box: Tensor | None) -> Tensor:
    """Return d - box*round(d/box) for an orthorhombic box, or d when box is None.

    d has trailing dimension 3; box contains three lengths in the same units. Rounding ties go to the nearest even integer.
    """
    if box is None:
        return d
    box = box.to(d)
    return d - box * torch.round(d / box)


@dataclass(frozen=True, eq=False)
class Linear:
    """Apply affine projections to each flattened system state.

    matrix has shape (M, d, n), where n is the flattened state size. offset has shape (M, d) and defaults to zero. Outputs have shape (B, M, d); parameters use the input dtype and device.
    """

    matrix: Tensor
    offset: Tensor | None = None

    def __post_init__(self):
        H = torch.as_tensor(self.matrix, dtype=torch.float64).detach().clone()
        if H.ndim != 3 or not torch.isfinite(H).all():
            raise ValueError("Linear matrix must be finite with shape (M, d, n)")
        b = (
            torch.zeros(H.shape[:2], dtype=H.dtype, device=H.device)
            if self.offset is None
            else torch.as_tensor(self.offset, dtype=H.dtype, device=H.device)
        )
        if b.shape != H.shape[:2] or not torch.isfinite(b).all():
            raise ValueError("Linear offset must have shape (M, d) and be finite")

        object.__setattr__(self, "matrix", H)
        object.__setattr__(self, "offset", b.detach().clone())

    def __call__(self, s):
        if s[0].numel() != self.matrix.shape[-1]:
            raise ValueError("Linear input dimension differs from the flattened state")

        return torch.einsum("bn,mdn->bmd", s.flatten(1), self.matrix.to(s)) + self.offset.to(s)


@dataclass(frozen=True, eq=False)
class Select:
    """Select flattened coordinates for each copy of a prior.

    index has shape (M, *prior_shape), including the copy axis even for one copy. Overlapping selections allow several prior factors to act on the same coordinate.
    """

    index: Tensor

    def __post_init__(self):
        object.__setattr__(self, "index", torch.as_tensor(self.index, dtype=torch.long).clone())

    def __call__(self, s: Tensor) -> Tensor:
        """Map (B, *state_shape) coordinates to (B, M, *prior_shape) observations."""
        return s.flatten(1)[:, self.index.to(s.device)]


@dataclass(frozen=True, eq=False)
class Torsions:
    """Compute signed dihedral angles using OpenMM's angle convention.

    quartets has shape (M, d, 4). Positions have shape (B, atoms, 3) in nm; angles have shape (B, M, d) in radians within [-pi, pi).

    An optional orthorhombic box has shape (3,) in nm. Each bond displacement uses the minimum image before the angle is evaluated.
    """

    quartets: Tensor
    box: Tensor | None = None
    period: ClassVar[float] = 2 * math.pi

    def __post_init__(self):
        object.__setattr__(self, "quartets", torch.as_tensor(self.quartets, dtype=torch.long).clone())
        if self.box is not None:
            object.__setattr__(self, "box", self.box.clone())

    def __call__(self, s: Tensor) -> Tensor:
        """Return the signed angles for each batch member and prior copy."""
        p = s[:, self.quartets.to(s.device)]  # (n, M, d, 4, 3)
        b0 = minimum_image(p[..., 0, :] - p[..., 1, :], self.box)
        b1 = minimum_image(p[..., 2, :] - p[..., 1, :], self.box)
        b2 = minimum_image(p[..., 3, :] - p[..., 2, :], self.box)
        b1 = b1 / b1.norm(dim=-1, keepdim=True)
        v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1  # b0 and b2 with their components along the central bond removed
        w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
        return wrap(torch.atan2((torch.linalg.cross(b1, v) * w).sum(-1), (v * w).sum(-1)))
