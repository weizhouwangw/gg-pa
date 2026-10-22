"""Compare GG-PA and Metropolis sampling using the package's lattice implementations."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from ggpa.priors.networks import Precision, set_tf32
from ggpa.systems import phi4, runner


def run(settings, out):
    """Run both samplers at each configured point and collect the production samples for plotting."""
    out.mkdir(parents=True, exist_ok=False)
    (out / "settings.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    common = {key: settings[key] for key in ("seed", "device", "L", "chains", "J", "h")}
    sample = phi4.Scan(**(common | settings["sample"]), extend={})
    reference = phi4.MC(**(common | settings["reference"]))
    set_tf32(Precision(**sample.precision).for_device(sample.device).tf32)
    data, timings = [], {}
    for point, mc_point in zip(phi4.points(sample), phi4.points(reference), strict=True):
        timings[point.id] = {}
        for name, config, item, function in (
            ("ggpa", sample, point, phi4.sample_point),
            ("metropolis", reference, mc_point, phi4.reference_point),
        ):
            print(f"{name}: J={item.values['J']:g}, h={item.values['h']:g}", flush=True)
            start = time.perf_counter()
            function(config, item, out / "results" / name / item.id)
            timings[point.id][name] = round(time.perf_counter() - start, 2)
        ggpa_dir = out / "results/ggpa" / point.id
        with np.load(ggpa_dir / "samples.npz") as record:
            m = record["m_x"][int(sample.sweeps * settings["burn_in"]) :, 0]
        with np.load(out / "results/metropolis" / point.id / "samples.npz") as record:
            reference_m = record["m"]
        state = runner.load_state(ggpa_dir / "state.pt")["samplers"][0]
        field = state["xs"][0][0].reshape(sample.L, sample.L).numpy()
        data.append((point.values, m, reference_m, field))
    return data, timings


def plot(data, out, method):
    """Show final clean fields and magnetization distributions at the same physical parameters."""
    with plt.rc_context({"font.size": 10, "axes.linewidth": 0.9, "lines.linewidth": 1.7}):
        fig, axes = plt.subplots(2, len(data), figsize=(3.4 * len(data), 5.7), squeeze=False, layout="constrained")
        for i, (point, m, reference, field) in enumerate(data):
            image = axes[0, i].imshow(field, vmin=-1.5, vmax=1.5, cmap="RdBu_r", interpolation="nearest")
            title = f"J = {point['J']:g}" + (f", h = {point['h']:g}" if point["h"] else "")
            axes[0, i].set(title=title, xticks=[], yticks=[])
            for values, label, color, style in (
                (m, method, "#247b97", "-"),
                (reference, "Metropolis", "#222222", "--"),
            ):
                axes[1, i].hist(
                    np.abs(values).ravel(),
                    np.linspace(0, 1.3, 40),
                    density=True,
                    histtype="step",
                    color=color,
                    linestyle=style,
                    label=label,
                )
            axes[1, i].set(xlabel=r"$|m|$", xlim=(0, 1.3))
            axes[1, i].spines[["top", "right"]].set_visible(False)
        axes[1, 0].set_ylabel("Probability density")
        axes[1, 0].legend(frameon=False, fontsize=9)
        for letter, ax in zip("abcdefghijklmnopqrstuvwxyz", axes.flat, strict=False):
            ax.text(-0.12, 1.04, f"({letter})", transform=ax.transAxes, fontweight="bold", fontsize=13)
        fig.colorbar(image, ax=list(axes[0]), label=r"$\phi$", shrink=0.8)
        (out / "figures").mkdir()
        fig.savefig(out / "figures/comparison.png", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("settings.yaml"))
    parser.add_argument("--out", type=Path, default=Path("runs") / datetime.now().strftime("phi4_%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    torch.set_num_threads(settings["threads"])
    started = time.perf_counter()
    data, timings = run(settings, args.out)
    plot(data, args.out, "GG-PA-RE" if settings["sample"]["rungs"] > 1 else "GG-PA")
    manifest = {
        "device": settings["device"],
        "torch": torch.__version__,
        "timings": timings,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {args.out / 'figures/comparison.png'} ({manifest['elapsed_seconds']:.0f} s)")


if __name__ == "__main__":
    main()
