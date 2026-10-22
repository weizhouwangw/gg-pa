"""Run both CPU demonstrations from the checkout and record their settings and outputs."""

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import doublewell
import inverse_problem
import matplotlib
import numpy as np
import scipy
import torch

import ggpa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "out", nargs="?", type=Path, default=Path("runs") / datetime.now().strftime("quickstart_%Y%m%d_%H%M%S")
    )
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=False)
    for directory in ("results", "figures"):
        (args.out / directory).mkdir()

    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "checkpoints/doublewell_prior.pt"
    settings = {
        "inverse_problem": dict(sweeps=6000, chains=48, ratio=0.95, seed=7),
        "doublewell": dict(sweeps=1000, re_sweeps=500, chains=250, direct_samples=10000, seed=7),
    }
    source = (
        sorted(Path(ggpa.__file__).parent.rglob("*.py"))
        + sorted(Path(__file__).parent.glob("*.py"))
        + [root / "quickstart.sh"]
    )
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "device": "cpu",
        "threads": args.threads,
        "burn_in_fraction": 0.2,
        "versions": dict(
            python=platform.python_version(),
            ggpa=ggpa.__version__,
            torch=torch.__version__,
            numpy=np.__version__,
            scipy=scipy.__version__,
            matplotlib=matplotlib.__version__,
        ),
        "source_sha256": hashlib.sha256(b"".join(path.read_bytes() for path in source)).hexdigest(),
        "checkpoint": {
            "file": "checkpoints/doublewell_prior.pt",
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
        "examples": {
            name: {
                "settings": values,
                "results": f"results/{name}.npz",
                "figures": [f"figures/{name}.{suffix}" for suffix in ("png", "pdf")],
            }
            for name, values in settings.items()
        },
    }
    manifest["examples"]["inverse_problem"]["observation"] = {"H": [[1.0, 0.0]], "y": 0.0, "sigma_eta": 2.5}
    manifest["examples"]["doublewell"]["coupling"] = {"production_t": 0.2, "context_t": 0.2, "ladder": [0.2, 0.4, 0.8]}
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.perf_counter()
    try:
        for module in (inverse_problem, doublewell):
            name = module.__name__
            print(f"Running {name} on CPU ...", flush=True)
            start = time.perf_counter()
            parameters = settings[name] | ({"checkpoint": checkpoint} if name == "doublewell" else {})
            data = module.run(**parameters)
            np.savez_compressed(args.out / f"results/{name}.npz", **data)
            module.plot(data, args.out / "figures", name=name)
            manifest["examples"][name]["elapsed_seconds"] = round(time.perf_counter() - start, 2)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Saved {args.out / 'figures' / (name + '.png')}", flush=True)
        manifest["status"] = "complete"
    finally:
        if manifest["status"] != "complete":
            manifest["status"] = "incomplete"
        manifest["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Run: {args.out} ({manifest['elapsed_seconds']:.0f} s)")


if __name__ == "__main__":
    main()
