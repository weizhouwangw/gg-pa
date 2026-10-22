"""Sample recording and coupling diagnostics."""

import math

import numpy as np
import torch
from torch import Tensor


def swap_estimate(sigma: np.ndarray, dims) -> np.ndarray:
    """Approximate neighboring exchange acceptance from Gaussian coupling widths.

    sigma has shape (replicas, attachments), or (replicas,) for one attachment. dims counts scalar observations per attachment, including copies. The fixed context must be shared across replicas.

    For rho=sigma_high^2/sigma_low^2, return erfc(sqrt(m)/2), where m=sum(dims*(rho-1)^2/(2*rho)). This assumes coupling noise determines the spread; confinement by the context can make the estimate pessimistic.
    """
    s2 = np.asarray(sigma, dtype=np.float64).reshape(len(sigma), -1) ** 2  # (R, K)
    rho = s2[1:] / s2[:-1]  # (R - 1, K)
    m = ((rho - 1) ** 2 / (2 * rho) * np.asarray(dims, dtype=np.float64)).sum(1)  # (R - 1,)
    return np.array([math.erfc(math.sqrt(v) / 2) for v in m])


def stack(records: list[dict[str, Tensor]]) -> dict[str, np.ndarray]:
    """Stack tensor records into NumPy arrays with a leading record axis.

    Each key collects only records containing it. Tensors move to CPU; key order follows first appearance.
    """
    keys = dict.fromkeys(k for r in records for k in r)
    return {k: torch.stack([torch.as_tensor(r[k]) for r in records if k in r]).cpu().numpy() for k in keys}


@torch.no_grad()
def logq_reference(kernel, t: float, gen: torch.Generator, n: int = 100_000) -> tuple[float, float]:
    """Estimate the mean and standard deviation of a Gaussian-type coupling log density.

    Draw y=sigma*noise with clean anchor x=0 using float64 and gen. Translation invariance makes this reference independent of the anchor for the bundled Gaussian and wrapped-Gaussian kernels.

    The result describes log density per prior copy. The Monte Carlo standard error of the mean is the returned standard deviation divided by sqrt(n).
    """
    k = kernel
    lv = k.level(torch.full((n,), float(t), dtype=torch.float64, device=gen.device))
    z = torch.randn((n, 1, *k.shape), generator=gen, dtype=torch.float64, device=gen.device)
    lq = k.log_q(lv.sigma.view(n, *(1,) * (z.dim() - 1)) * z, torch.zeros_like(z), lv)[:, 0]  # (n,)
    return lq.mean().item(), lq.std().item()
