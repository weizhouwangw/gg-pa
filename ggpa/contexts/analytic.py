"""Exact quadratic context updates and Gaussian deconvolution.

Context parameters stay fixed while coupling times may change during sampling.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..conventions import check_tensor
from ..priors.kernels import DiscreteVP, Gaussian, GeometricVE, Level
from ..projection import Select


def _symmetric_matrix(value, name):
    v = torch.as_tensor(value, dtype=torch.float64).detach().clone()
    check_tensor(v, name)
    if v.ndim != 2 or v.shape[0] != v.shape[1] or len(v) == 0:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not torch.allclose(v, v.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    return v


def _affine_projection(c, s):
    """Return the dense matrix and offset for a Select or Linear projection."""
    from ..projection import Linear

    n = s[0].numel()
    if isinstance(c.projection, Select):
        index = c.projection.index.flatten().to(s.device)
        if ((index < 0) | (index >= n)).any():
            raise ValueError("selection index is outside the state")
        H = torch.eye(n, device=s.device, dtype=s.dtype)[index]
        offset = torch.zeros(len(index), device=s.device, dtype=s.dtype)
    elif isinstance(c.projection, Linear):
        H = c.projection.matrix.to(s).reshape(-1, n)
        offset = c.projection.offset.to(s).flatten()
    else:
        raise TypeError("quadratic exact updates require Select or Linear; use from_energy or custom otherwise")
    return H, offset


def quadratic(precision, linear=None):
    """Construct the fixed factor exp(-s.T*P*s/2 + b.T*s) with an exact conditional update.

    precision is P on the flattened state; linear is b and defaults to zero. Flat context modes are allowed when the combined precision from context and couplings is positive definite. No regularization is added.

    Requires isotropic Gaussian kernels and Select or Linear projections. Dense Cholesky solves provide the conditional mean and noise. This factory does not deconvolve a clean-variable target.
    """
    from . import custom

    P = _symmetric_matrix(precision, "precision")
    b = (
        torch.zeros(len(P), dtype=P.dtype, device=P.device)
        if linear is None
        else torch.as_tensor(linear, dtype=P.dtype, device=P.device).clone()
    )
    check_tensor(b, "linear", shape=(len(P),))
    cache = {}

    def coefficients(couplings, t, s):
        levels = []
        for c, time in zip(couplings, t, strict=True):
            lv = c.level(time, s)
            levels.append((float(lv.alpha[0]), float(lv.sigma[0])))

        # Recheck fixed structure only when the composition or realized kernel parameters change.
        key = (tuple(id(c) for c in couplings), tuple(levels), s[0].numel(), s.device, s.dtype)
        if cache.get("key") != key:
            if s[0].numel() != len(P):
                raise ValueError("quadratic precision dimension differs from the flattened state")
            if any(type(c.kernel) is not Gaussian for c in couplings):
                raise TypeError("quadratic exact updates require isotropic Gaussian kernels")
            if any(not math.isfinite(a) or not math.isfinite(sigma) or sigma <= 0 for a, sigma in levels):
                raise ValueError("quadratic coupling needs finite alpha and positive sigma")
            matrices = [_affine_projection(c, s) for c in couplings]
            A = P.to(s).clone()
            # Each Gaussian observation adds H.T H / sigma^2 to the precision.
            for (H, _), (_, sigma) in zip(matrices, levels, strict=True):
                A += H.T @ H / sigma**2
            L, info = torch.linalg.cholesky_ex(A)
            if info.any():
                raise ValueError("the assembled quadratic conditional precision must be positive definite")
            cache.update(key=key, L=L, matrices=matrices)
        return cache["L"], levels, cache["matrices"]

    def validate(couplings, t, s):
        coefficients(couplings, t, s)

    def log_factor(s):
        z = s.flatten(1)
        return -0.5 * ((z @ P.to(s)) * z).sum(1) + z @ b.to(s)

    def update(s, prior_samples, couplings, t, context_state, rng):
        L, levels, matrices = coefficients(couplings, t, s)

        # Natural parameter: b + sum H.T (alpha*x - offset) / sigma^2.
        rhs = b.to(s).expand(len(s), -1).clone()
        for x, (H, offset), (a, sigma) in zip(prior_samples, matrices, levels, strict=True):
            rhs += (a * x.flatten(1) - offset) @ H / sigma**2

        # If A = L L.T, the conditional has mean A^-1 rhs and covariance A^-1.
        mean = torch.cholesky_solve(rhs.T, L).T
        z = torch.randn(mean.shape, generator=rng, device=s.device, dtype=s.dtype)
        draw = mean + torch.linalg.solve_triangular(L.T, z.T, upper=True).T
        return draw.reshape(s.shape), context_state

    return custom(update, validate=validate, log_factor=log_factor)


def gaussian(mean, covariance):
    """Construct a quadratic context from a mean and positive-definite covariance."""
    C = _symmetric_matrix(covariance, "covariance")
    L, info = torch.linalg.cholesky_ex(C)
    if info.any():
        raise ValueError("Gaussian covariance must be positive definite")
    m = torch.as_tensor(mean, dtype=C.dtype, device=C.device)
    check_tensor(m, "mean", shape=(len(C),))
    P = torch.cholesky_inverse(L)
    return quadratic(P, P @ m)


def deconvolved_quadratic(precision, linear, *, kernel, context_noise):
    """Deconvolve a clean quadratic factor for one full-state Gaussian coupling.

    context_noise is the reference diffusion time passed to kernel.level, not the standard deviation sigma. Transform each precision eigenvalue lambda to lambda/(alpha^2-sigma^2*lambda), and the linear term to alpha*(alpha^2 I-sigma^2 P)^-1*b. P must be positive semidefinite within tolerance, with positive denominators.

    The context requires the original Gaussian kernel and an identity projection covering the full state once. Changing sampling time leaves this factor fixed.
    """
    if type(kernel) is not Gaussian:
        raise TypeError("deconvolved_quadratic requires an isotropic Gaussian kernel")
    P = _symmetric_matrix(precision, "precision")
    b = torch.as_tensor(linear, dtype=P.dtype, device=P.device)
    check_tensor(b, "linear", shape=(len(P),))
    lam, V = torch.linalg.eigh(P)
    if (lam < -1e-12).any():
        raise ValueError("deconvolution requires positive-semidefinite precision")
    lv = kernel.level(torch.tensor([context_noise], dtype=torch.float64, device=P.device))
    a, sigma = lv.alpha[0], lv.sigma[0]
    # Positive denominators are required for the inverse Gaussian convolution.
    denom = a.square() - sigma.square() * lam
    if (denom <= 0).any() or not torch.isfinite(denom).all():
        raise ValueError("context noise exceeds the strict Gaussian deconvolution bound")
    ctx = quadratic((V * (lam / denom)) @ V.T, a * V @ ((V.T @ b) / denom))
    check = ctx.validate

    def validate(couplings, t, s):
        if len(couplings) != 1 or couplings[0].kernel is not kernel:
            raise ValueError("deconvolved context must use its one original full-state kernel")
        H, offset = _affine_projection(couplings[0], s)
        if H.shape != P.shape or not torch.equal(H, torch.eye(len(P), device=s.device, dtype=s.dtype)) or offset.any():
            raise ValueError("deconvolved_quadratic requires identity coverage of the full state")
        check(couplings, t, s)

    ctx.validate = validate
    return ctx


def _ok(lv: Level, lam_max: float) -> Tensor:
    """Check the strict deconvolution condition sigma^2*lambda_max < alpha^2 at each level."""
    return lv.sigma**2 * lam_max < lv.alpha**2


def _bound(kernel, lam_max: float) -> str:
    """Describe the admissible noise range for a failed deconvolution check."""
    sch = getattr(kernel, "schedule", None)
    if type(sch) is DiscreteVP:
        T = sch.alpha_bar.numel()
        ok = torch.nonzero(_ok(sch.level(torch.arange(T, dtype=torch.float64) / (T - 1)), lam_max))
        if not len(ok):
            return "no grid index is admissible"
        k = int(ok.max())
        return f"admissible up to index {k} (t = {k / (T - 1):.4f})"
    if type(sch) is GeometricVE and sch.sigma_min == sch.sigma_max:
        return f"sigma = {sch.sigma_min:g} >= 1/sqrt(lambda_max) = {1 / math.sqrt(lam_max):g}: no t is admissible"
    if type(sch) is GeometricVE:
        tb = math.log(1 / (math.sqrt(lam_max) * sch.sigma_min)) / math.log(sch.sigma_max / sch.sigma_min)
        return f"admissible for t < {tb:.4f}"
    return "no time bound is available for this schedule"


def _check_deconvolution_bound(lam_max, kernel, t_ctx, lv):
    """Reject inadmissible deconvolution with the realized level and an available time bound."""
    if not _ok(lv, lam_max)[0]:
        at = f"t_ctx = {t_ctx:g}" + (
            "" if lv.k is None else f" (index {int(lv.k[0])} of {len(kernel.schedule.alpha_bar) - 1})"
        )
        raise ValueError(
            f"strict deconvolution bound violated at {at}: sigma^2 lambda_max = "
            f"{lv.sigma[0].item() ** 2 * lam_max:.6g} >= alpha^2 = {lv.alpha[0].item() ** 2:.6g}; "
            f"{_bound(kernel, lam_max)}"
        )


def deconvolve(lam: Tensor, kernel: Gaussian, t_ctx: float) -> Tensor:
    """Transform nonnegative interaction eigenvalues at a fixed reference noise.

    Return lambda/(alpha^2-sigma^2*lambda) in float64, preserving zero modes. Require sigma^2*max(lambda) < alpha^2 at the realized kernel level.

    """
    lam = torch.as_tensor(lam, dtype=torch.float64)
    lv = kernel.level(torch.tensor([t_ctx], dtype=torch.float64, device=lam.device))
    _check_deconvolution_bound(lam.max().item(), kernel, t_ctx, lv)

    a2, s2 = lv.alpha[0] ** 2, lv.sigma[0] ** 2
    return lam / (a2 - s2 * lam)


def admissible(kernel: Gaussian, t: float, lam_max: float) -> bool:
    """Check sigma^2*lambda_max < alpha^2 at the kernel level selected by t."""
    return bool(_ok(kernel.level(torch.tensor([t], dtype=torch.float64)), lam_max)[0])
