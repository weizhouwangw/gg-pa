"""Network architectures and checkpoint loading for the bundled priors.

Weights use float32 in evaluation mode. Compilation and CUDA autocast are explicit options.
"""

from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor, nn

from .kernels import DiscreteVP, Gaussian, GeometricVE, WrappedGaussian


class ResidualMLP(nn.Module):
    """Residual velocity-prediction network with a sinusoidal time embedding.

    Inputs x and t have shapes (B, input_dim) and (B,) in float32, with t in [0, 1]. Outputs have shape (B, output_dim). Each block adds fc2(SiLU(fc1(h)+time_proj(embedding))) to h.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, time_embed_dim: int, n_blocks: int):
        super().__init__()
        half = time_embed_dim // 2
        freqs = torch.exp(torch.arange(half, dtype=torch.float32) * -(math.log(10000) / (half - 1)))
        self.register_buffer("freqs", freqs, persistent=False)
        E = time_embed_dim
        self.time_mlp = nn.Sequential(nn.Linear(E, 2 * E), nn.SiLU(), nn.Linear(2 * E, E))
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            nn.ModuleDict(
                {
                    "fc1": nn.Linear(hidden_dim, hidden_dim),
                    "fc2": nn.Linear(hidden_dim, hidden_dim),
                    "time_proj": nn.Linear(E, hidden_dim),
                }
            )
            for _ in range(n_blocks)
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        arg = t.unsqueeze(1) * self.freqs.unsqueeze(0)  # (n, E/2)
        e = self.time_mlp(torch.cat([torch.sin(arg), torch.cos(arg)], dim=1))  # (n, E)
        h = self.in_proj(x)
        for block in self.blocks:
            h = h + block["fc2"](nn.functional.silu(block["fc1"](h) + block["time_proj"](e)))
        return self.out_proj(h)


class TorsionMLP(nn.Module):
    """Predict sigma_t times the angular score from periodic sine/cosine features.

    x has shape (B, ..., n_angles) in radians and t has shape (B,). Inputs and outputs use float32; output shape matches x. Residual blocks use separate pre-normalization layers and a shared post-normalization layer.
    """

    def __init__(self, n_angles: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        H = hidden_dim
        self.half = H // 2
        self.angle_embed = nn.Sequential(nn.Linear(2 * n_angles, H), nn.SiLU(), nn.Linear(H, H))
        self.merge_layer = nn.Linear(2 * H, H)
        self.layers = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(H), nn.Linear(H, 4 * H), nn.SiLU(), nn.Dropout(dropout), nn.Linear(4 * H, H))
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(H)
        self.out = nn.Sequential(nn.Linear(H, H), nn.SiLU(), nn.Linear(H, n_angles))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        flat = x.reshape(-1, x.shape[-1])  # (n L, K), L the angle sets per row
        tt = t.unsqueeze(1).expand(x.shape[0], flat.shape[0] // x.shape[0]).reshape(-1)  # (n L,)
        h = self.angle_embed(torch.cat([torch.sin(flat), torch.cos(flat)], dim=-1))
        freqs = torch.exp(torch.arange(self.half, device=t.device) * -(math.log(10000) / (self.half - 1)))
        arg = tt.unsqueeze(1) * freqs.unsqueeze(0)  # (n L, H/2)
        h = self.merge_layer(torch.cat([h, torch.sin(arg), torch.cos(arg)], dim=-1))
        for layer in self.layers:
            h = self.norm(h + layer(h))
        return self.out(h).reshape(x.shape)


@dataclass(frozen=True)
class Precision:
    """Execution settings for a loaded network.

    CUDA autocast accepts None, "fp16" or "bf16". tf32 is applied with set_tf32; compile enables torch.compile with dynamic shapes.
    """

    autocast: str | None = None
    tf32: bool = False
    compile: bool = False

    def for_device(self, device) -> Precision:
        """Disable CUDA-only precision options on other devices, retaining the compile setting."""
        return self if torch.device(device).type == "cuda" else replace(self, autocast=None, tf32=False)


EAGER_FP32 = Precision()
_NETS = {"ResidualMLP": ResidualMLP, "TorsionMLP": TorsionMLP}
_AUTOCAST = {"fp16": torch.float16, "bf16": torch.bfloat16}


def set_tf32(enabled: bool) -> None:
    """Set process-wide TF32 flags for CUDA matrix multiplication and cuDNN."""
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def _kernel(spec: dict) -> Gaussian | WrappedGaussian:
    """Reconstruct a checkpoint's coupling kernel, preserving its stored schedule table."""
    sch = spec["schedule"]
    if sch["kind"] == "discrete_vp":
        schedule = DiscreteVP(sch["alpha_bar"])
    else:
        schedule = GeometricVE(float(sch["sigma_min"]), float(sch["sigma_max"]))
    shape = tuple(spec["shape"])
    if spec["family"] == "gaussian":
        return Gaussian(schedule, shape)
    return WrappedGaussian(schedule, shape, int(spec["windings"]))


def load_net(path, device, precision: Precision = EAGER_FP32) -> tuple[Callable, Gaussian | WrappedGaussian, dict]:
    """Load a ggpa-prior/1 checkpoint and return (forward, kernel, metadata).

    Load weights strictly into a float32 evaluation network with gradients disabled. Apply the effective device precision, optionally compiling the network or using CUDA autocast with float32 outputs.

    Reconstruct the kernel from the same checkpoint. Metadata includes the supplied path, file hash and stored provenance.
    """
    raw = Path(path).read_bytes()
    ckpt = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    spec = ckpt["net"]
    net = _NETS[spec["class"]](**spec["config"])
    net.load_state_dict(spec["state_dict"], strict=True)
    net = net.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    eff = precision.for_device(device)
    run = torch.compile(net, dynamic=True) if eff.compile else net
    if eff.autocast is not None:
        dtype, inner = _AUTOCAST[eff.autocast], run

        def run(x: Tensor, t: Tensor) -> Tensor:
            with torch.autocast("cuda", dtype=dtype):
                return inner(x, t).float()

    meta = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "provenance": ckpt.get("provenance")}
    return run, _kernel(ckpt["kernel"]), meta
