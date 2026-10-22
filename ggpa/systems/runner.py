"""Run configured points and record their settings.

Each point has a directory containing resolved.yaml, samples.npz and, when applicable, state.pt. Completed outputs are skipped only if saved settings match. Incomplete points restart from sweep zero.

Point IDs determine random seeds independently of enumeration order.
"""

import argparse
import dataclasses
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import yaml

from ..priors.networks import Precision, set_tf32


class Point(NamedTuple):
    """Scan point with an index, path-safe ID, sweep budget and physical parameter values."""

    index: int
    id: str
    sweeps: int
    values: dict


def point_seed(seed: int, point_id: str) -> int:
    """Derive a stable seed in [1, 2**63) from the base seed and point ID."""
    return int(np.random.SeedSequence([seed, *point_id.encode()]).generate_state(1, np.uint64)[0]) % (2**63 - 1) + 1


def _levels(prior, times, sfx: str) -> dict:
    """Record each replica's realized grid index, signal amplitude and noise scale; use None when absent."""
    level = prior.kernel.level
    lvs = [None if v is None else level(torch.tensor([v], dtype=torch.double))._asdict() for v in times]
    return {f + sfx: [None if x is None or x[f] is None else x[f].item() for x in lvs] for f in ("k", "alpha", "sigma")}


def _plain(v):
    """Convert arrays, tensors and metadata into YAML-safe values, preserving mapping keys."""
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_plain(x) for x in v]
    if hasattr(v, "tolist"):
        return v.tolist()
    return v if v is None or type(v) in (bool, int, float, str) else str(v)


def write_resolved(out, cfg, point: Point, priors, t, t_ctx, metas: list[dict], *, samplers=()) -> None:
    """Write settings, point values, realized noise levels and execution provenance.

    Record prior paths and hashes, requested/effective precision, seeds, package identity and environment versions. Optional seed, device and precision fields default to None, cpu and None. This records an existing composition without initializing it.
    """
    req, device = getattr(cfg, "precision", None), torch.device(getattr(cfg, "device", "cpu"))
    seed, pkg = getattr(cfg, "seed", None), sys.modules[__package__.partition(".")[0]]
    file, cwd = Path(pkg.__file__).resolve(), Path.cwd().resolve()
    env = {m: getattr(sys.modules.get(m), "__version__", None) for m in ("torch", "numpy", "openmm", "mdtraj")}
    name = f"{device} {torch.cuda.get_device_name(device)}" if device.type == "cuda" else str(device)
    env |= {"device": name, "tf32": torch.backends.cuda.matmul.allow_tf32, "threads": torch.get_num_threads()}
    doc = {
        "point": point._asdict(),
        "settings": dataclasses.asdict(cfg),
        "ladder": {"t": list(t), "t_ctx": list(t_ctx)},
        "levels": [_levels(prior, t, "") | _levels(prior, t_ctx, "_ctx") for prior in priors],
        "priors": [{"path": m["path"], "sha256": m["sha256"]} for m in metas],
        "precision_requested": req,
        "precision_effective": None if req is None else dataclasses.asdict(Precision(**req).for_device(device)),
        "seeds": {"seed": seed, "point_seed": None if seed is None else point_seed(seed, point.id)},
        "code": {"file": str(file.relative_to(cwd) if file.is_relative_to(cwd) else file), "version": pkg.__version__},
        "environment": env,
    }
    if samplers:
        doc["seeds"]["sampler_seeds"] = [sampler.rng.initial_seed() for sampler in samplers]
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "resolved.yaml").write_text(yaml.safe_dump(_plain(doc), sort_keys=False))


def finish(out, arrays: dict, state: dict | None, wall_s: float, engine: dict | None = None) -> None:
    """Save sample arrays, timing and an optional sampler checkpoint.

    Publish samples.npz atomically after the checkpoint succeeds; its presence marks the point complete. Include engine seeds and properties when supplied.
    """
    (out := Path(out)).mkdir(parents=True, exist_ok=True)
    extra = {"engine_seed": engine["seed"], "engine_properties": json.dumps(engine["properties"])} if engine else {}
    if engine and "seeds" in engine:
        extra["engine_seeds"] = np.asarray(engine["seeds"], dtype=np.int64)
    with open(out / "samples.npz.tmp", "wb") as f:
        np.savez(f, **arrays, wall_s=wall_s, **extra)
    if state is not None:
        save_state(out / "state.pt", state)
    (out / "samples.npz.tmp").replace(out / "samples.npz")


def step(samplers, *, exchange=False):
    """Advance all replicas and optionally attempt alternating neighboring swaps.

    Return int8 records of shape (replicas-1, chains), using -1 for unattempted pairs.
    """
    if not samplers:
        raise ValueError("step needs at least one sampler")
    if len({s.state.step for s in samplers}) != 1:
        raise ValueError("replicas must have the same completed step count")
    parity = samplers[0].state.step % 2
    for sampler in samplers:
        sampler.step()
    ref = samplers[0].state.s
    accepted = torch.full((len(samplers) - 1, len(ref)), -1, dtype=torch.int8, device=ref.device)
    if exchange:
        for i in range(parity, len(samplers) - 1, 2):
            accepted[i] = samplers[i].exchange(samplers[i + 1]).to(torch.int8)
    return accepted


def snapshot(samplers, accepted=None):
    """Collect each sampler's state and RNG, plus optional swap records, into one checkpoint."""
    return {
        "format": "ggpa-campaign/1",
        "samplers": [s.state_dict() for s in samplers],
        "accepted": None if accepted is None else accepted.detach().clone(),
    }


def save_state(path, state: dict) -> None:
    """Write a checkpoint atomically."""
    torch.save(state, f"{path}.tmp")
    Path(f"{path}.tmp").replace(path)


def load_state(path, device="cpu") -> dict:
    """Load a checkpoint, moving state tensors to device while keeping generator bytes on CPU."""
    rec = torch.load(path, map_location="cpu", weights_only=True)
    if rec.get("format") != "ggpa-campaign/1":
        raise ValueError("expected a ggpa-campaign/1 checkpoint")

    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(v) for key, v in value.items()}
        if isinstance(value, tuple):
            return tuple(move(v) for v in value)
        return value

    for record in rec["samplers"]:
        for key in ("s", "xs", "aux"):
            record[key] = move(record[key])
    rec["accepted"] = move(rec["accepted"])
    return rec


def _canon(v):
    """Normalize mapping keys to strings for comparisons with saved YAML records."""
    if isinstance(v, dict):
        return {str(k): _canon(x) for k, x in v.items()}
    return [_canon(x) for x in v] if isinstance(v, list) else v


def _unchanged(cfg, point: Point, out: Path) -> None:
    """Reject a completed output whose saved settings or point values differ from the requested run."""
    old = yaml.safe_load((out / "resolved.yaml").read_text())
    now = yaml.safe_load(yaml.safe_dump(_plain({"settings": dataclasses.asdict(cfg), "values": point.values})))
    for part, was in (("settings", old.get("settings")), ("values", (old.get("point") or {}).get("values"))):
        a, b = _canon(now[part] or {}), _canon(was or {})
        key = next((k for k in dict.fromkeys([*a, *b]) if a.get(k, "<absent>") != b.get(k, "<absent>")), None)
        if key is not None:
            raise ValueError(
                f"point {point.id} in {out.parent} is complete with {part}.{key} = {b.get(key, '<absent>')!r}, but the "
                f"config gives {a.get(key, '<absent>')!r}; a complete point is never rerun: use a new --out for a "
                "changed configuration"
            )


def _check_points(points, index):
    """Check point IDs and the requested index, then return the index-to-point mapping."""
    first = {}
    for p in points:
        if (prev := first.setdefault(p.id, p.index)) != p.index:
            where = f"id '{p.id}' occurs at indices {prev} and {p.index}"
            raise ValueError(f"points: {where}; point ids name output directories and seeds and must be unique")
    by_index = {q.index: q for q in points}
    if index is not None and index not in by_index:
        raise ValueError(f"--point {index} is not in 0..{len(points) - 1}, this config's points; --list shows them")
    return by_index


def main(argv: list[str], verbs: dict[str, tuple[type, Callable, Callable]]) -> int:
    """Run or list the points of a configuration file.

    Accept <verb> --config C --out D [--point I] [--list]. Each verb supplies a settings type, point enumerator and run function. Point IDs must be unique because they determine directories and seeds.

    --list runs no points. Otherwise run the selected point or all points, skipping completed outputs only when saved settings match.
    """
    parser = argparse.ArgumentParser(description="Run configured points, one directory per point id under --out.")
    parser.add_argument("verb", choices=verbs)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", type=Path, required="--list" not in argv)
    parser.add_argument("--point", type=int)
    parser.add_argument("--list", action="store_true")
    a = parser.parse_args(argv)
    cls, enumerate_points, run_point = verbs[a.verb]
    cfg = cls(**yaml.safe_load(a.config.read_text()))
    set_tf32(getattr(cfg, "precision", None) is not None and Precision(**cfg.precision).for_device(cfg.device).tf32)
    points = list(enumerate_points(cfg))
    by_index = _check_points(points, a.point)

    for p in points if a.list or a.point is None else [by_index[a.point]]:
        done = a.out is not None and (a.out / p.id / "samples.npz").is_file()
        if done and not a.list:
            _unchanged(cfg, p, a.out / p.id)
        status = ("complete" if done else "absent") if a.list else ("skipped: complete" if done else "running")
        print(p.index, p.id, p.sweeps, *([] if a.out is None else [status]), flush=True)
        if not (a.list or done):
            run_point(cfg, p, a.out / p.id)
    return 0
