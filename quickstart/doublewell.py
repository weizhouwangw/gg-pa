"""Compose a trained double-well prior with a harmonic environment and plot the clean marginal."""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from ggpa import compose
from ggpa.priors import ddpm, v_to_eps
from ggpa.priors.networks import load_net
from ggpa.projection import Select
from ggpa.systems import doublewell as dw
from ggpa.systems import runner


def run(checkpoint, *, sweeps=1000, re_sweeps=500, chains=250, direct_samples=10000, seed=7):
    """Compare GG-PA and the production slot of GG-PA-RE at t=0.2, after 20% burn-in."""
    net, kernel, _ = load_net(checkpoint, "cpu")
    prior = ddpm(v_to_eps(net, kernel), kernel)
    pairs = [(prior, Select([[0]]))]
    context = dw.context(kernel, context_noise=0.2)
    initial = torch.ones(chains, 2)
    sampler = compose(pairs, context, initial=initial, t=0.2, seed=seed)
    samples = []
    for sweep in range(sweeps):
        state = sampler.step()
        if sweep >= sweeps // 5:
            samples.append(state.prior_samples[0][:, 0, 0])

    # Only the t=0.2 slot has the physical clean marginal for this fixed context.
    ladder = (0.2, 0.4, 0.8)
    replicas = [compose(pairs, context, initial=initial, t=t, seed=seed + 10 + i) for i, t in enumerate(ladder)]
    exchanged, accepted = [], []
    for sweep in range(re_sweeps):
        accepted.append(runner.step(replicas, exchange=True))
        if sweep >= re_sweeps // 5:
            exchanged.append(replicas[0].state.prior_samples[0][:, 0, 0])
    direct = dw.direct(prior, direct_samples, torch.Generator().manual_seed(seed + 1))[:, 0]
    return dict(
        clean=torch.stack(samples).numpy(),
        clean_re=torch.stack(exchanged).numpy(),
        direct=direct.numpy(),
        ladder=np.array(ladder),
        accepted=torch.stack(accepted).numpy(),
    )


def plot(data, out, *, name="comparison"):
    """Show the training potential, coupled PES, and clean marginal in the paper's plotting style."""
    with plt.rc_context({"font.size": 10, "axes.linewidth": 0.9, "lines.linewidth": 1.7}):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), layout="constrained")
        x = np.linspace(-2, 2, 501)
        axes[0].plot(x, dw.A * (x**2 - 1) ** 2, color="#222222")
        axes[0].set(xlim=(-2, 2), ylim=(0, 20), xlabel=r"$x_{\mathrm{sys}}$", ylabel=r"$U_{\mathrm{sys}}$ ($k_BT$)")
        axes[0].set_yticks((0, 5, 10, 15, 20))

        env = np.linspace(-4, 4, 401)
        xx, ee = np.meshgrid(x, env)
        image = axes[1].pcolormesh(x, env, dw.energy(xx, ee), cmap="viridis", vmin=0, vmax=20, rasterized=True)
        fig.colorbar(image, ax=axes[1], label=r"Energy ($k_BT$)", ticks=(0, 5, 10, 15, 20))
        axes[1].set(xlabel=r"$x_{\mathrm{sys}}$", ylabel=r"$x_{\mathrm{env}}$")

        axes[2].plot(x, dw.marginal(x), "--", color="#fa6b83", label="Reference", zorder=4)
        edges = np.linspace(-2, 2, 241)
        centers = (edges[1:] + edges[:-1]) / 2
        for key, label, color, style in (
            ("direct", "Direct Diffusion", "#228b22", ":"),
            ("clean", "GG-PA", "#487bb1", "-"),
            ("clean_re", "GG-PA-RE", "#b02b88", "-."),
        ):
            density = np.histogram(data[key].ravel(), edges, density=True)[0]
            # Smooth only the display, with a fixed width of 0.025 in x; saved samples are unchanged.
            density = gaussian_filter1d(density, 0.025 / (edges[1] - edges[0]), mode="constant")
            axes[2].plot(centers, density, ls=style, color=color, label=label)
        axes[2].set(xlim=(-2, 2), ylim=(0, None), xlabel=r"$x_{\mathrm{sys}}$", ylabel="Density")
        axes[2].legend(frameon=False, fontsize=8, loc="upper left")
        for letter, ax in zip("abc", axes, strict=True):
            ax.text(-0.15, 1.04, f"({letter})", transform=ax.transAxes, fontsize=14, fontweight="bold")
            ax.tick_params(direction="out")
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"{name}.{suffix}", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path(__file__).resolve().parents[1] / "checkpoints/doublewell_prior.pt"
    )
    parser.add_argument("--out", type=Path, default=Path("runs/doublewell"))
    parser.add_argument("--sweeps", type=int, default=1000)
    parser.add_argument("--re-sweeps", type=int, default=500)
    parser.add_argument("--chains", type=int, default=250)
    parser.add_argument("--direct-samples", type=int, default=10000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    data = run(
        args.checkpoint,
        sweeps=args.sweeps,
        re_sweeps=args.re_sweeps,
        chains=args.chains,
        direct_samples=args.direct_samples,
    )
    np.savez_compressed(args.out / "samples.npz", **data)
    plot(data, args.out)
    print(f"Saved {args.out / 'comparison.png'} in {time.perf_counter() - start:.1f} s on CPU")


if __name__ == "__main__":
    main()
