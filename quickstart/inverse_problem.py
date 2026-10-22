"""Compare covariance-corrected and uncorrected split Gibbs with the exact Gaussian-mixture posterior."""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from ggpa import compose
from ggpa.priors.kernels import Gaussian, GeometricVE
from ggpa.projection import Select
from ggpa.systems import gmm


def run(*, sweeps=6000, chains=48, ratio=0.95, seed=7):
    """Collect clean samples at rho=ratio*rho_max, discarding the first 20% of each chain."""
    rho = ratio * gmm.RHO_MAX
    kernel = Gaussian(GeometricVE(rho, rho), (2,))
    prior = gmm.grid25_prior(kernel)
    samples = {}
    for matched in (True, False):
        sampler = compose(
            [(prior, Select([[0, 1]]))],
            gmm.likelihood(kernel, matched),
            initial=torch.zeros(chains, 2),
            t=0.0,
            seed=seed,
        )
        records = []
        for sweep in range(sweeps):
            state = sampler.step()
            if sweep >= sweeps // 5:
                records.append(state.prior_samples[0][:, 0])
        samples["matched" if matched else "unmatched"] = torch.stack(records).numpy()
    return samples | {"rho": rho}


def plot(data, out, *, name="comparison"):
    """Show the prior, observation and posterior, then compare the two split Gibbs samplers."""
    edges = np.linspace(-8, 8, 161)
    x = (edges[1:] + edges[:-1]) / 2
    grid = torch.cartesian_prod(torch.from_numpy(x), torch.from_numpy(x))
    # Equal-weight isotropic mixture from the supplied analytic prior.
    delta = grid[:, None] - gmm.MEANS[None]
    prior = (
        ((-delta.square().sum(-1) / (2 * gmm.SIGMA_X**2)).exp().mean(1) / (2 * np.pi * gmm.SIGMA_X**2))
        .numpy()
        .reshape(len(x), -1)
    )
    posterior = gmm.posterior(grid).numpy().reshape(len(x), -1)
    # Marginalizing the uncorrected split inflates the observation variance by rho^2.
    uncorrected = gmm.posterior(grid, var=gmm.SIGMA_ETA**2 + float(data["rho"]) ** 2).numpy().reshape(len(x), -1)
    likelihood = np.broadcast_to(np.exp(-0.5 * ((x[:, None] - float(gmm.Y[0])) / gmm.SIGMA_ETA) ** 2), prior.shape)
    sampled = [
        np.histogram2d(*data[key].reshape(-1, 2).T, bins=(edges, edges), density=True)[0]
        for key in ("matched", "unmatched")
    ]
    # Apply the same 0.12-coordinate display smoothing to both sampled densities.
    sampled = [gaussian_filter(density, 0.12 / (edges[1] - edges[0]), mode="constant") for density in sampled]
    maximum = max(posterior.max(), *(density.max() for density in sampled))
    with plt.rc_context({"font.size": 10, "axes.linewidth": 0.9, "lines.linewidth": 1.7}):
        fig, axes = plt.subplots(2, 3, figsize=(10, 6.1), layout="constrained")
        panels = (
            (axes[0, 0], prior, "Greys", "Prior", maximum),
            (axes[0, 1], likelihood, "Greens", "Observation: $y=0$", 1.0),
            (axes[0, 2], posterior, "Greys", "Exact posterior", maximum),
            (axes[1, 0], sampled[0], "Blues", "GG-PA", maximum),
            (axes[1, 1], sampled[1], "Reds", "Uncorrected split Gibbs", maximum),
        )
        for ax, density, cmap, title, upper in panels:
            ax.contourf(x, x, density.T, levels=np.linspace(0, upper, 40), cmap=cmap)
            ax.set(xlim=(-7.5, 7.5), ylim=(-7.5, 7.5), xlabel="$x_1$", ylabel="$x_2$", title=title, aspect="equal")
            ax.set_xticks((-5, -2.5, 0, 2.5, 5))
            ax.set_yticks((-5, -2.5, 0, 2.5, 5))
            ax.xaxis.set_major_formatter("{x:g}")
            ax.yaxis.set_major_formatter("{x:g}")
            ax.grid(color="0.5", alpha=0.18, linewidth=0.6)
        axes[0, 1].axvline(0, color="0.3", ls="--", lw=1)
        axes[1, 0].title.set_color("#1f77b4")
        axes[1, 1].title.set_color("#df2020")
        marginal = axes[1, 2]
        target = posterior.sum(1) * (edges[1] - edges[0])
        biased = uncorrected.sum(1) * (edges[1] - edges[0])
        marginal.plot(x, target, "k--", label="Exact", zorder=4)
        for key, label, color in (("matched", "GG-PA", "#1f77b4"), ("unmatched", "Uncorrected", "#df2020")):
            density = np.histogram(data[key][..., 0].ravel(), edges, density=True)[0]
            marginal.plot(x, gaussian_filter1d(density, 1.2, mode="constant"), color=color, label=label)
        marginal.fill_between(x, target, biased, color="#df2020", alpha=0.13, label="Split bias", zorder=0)
        marginal.set(xlim=(-8, 8), ylim=(0, None), xlabel="$Hx=x_1$", ylabel="Density")
        marginal.legend(frameon=False, fontsize=8, loc="upper left")
        for letter, ax in zip("abcdef", axes.flat, strict=True):
            ax.text(-0.15, 1.08, f"({letter})", transform=ax.transAxes, fontsize=14, fontweight="bold")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(direction="out")
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"{name}.{suffix}", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/inverse_problem"))
    parser.add_argument("--sweeps", type=int, default=6000)
    parser.add_argument("--chains", type=int, default=48)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    data = run(sweeps=args.sweeps, chains=args.chains)
    np.savez_compressed(args.out / "samples.npz", **data)
    plot(data, args.out)
    print(f"Saved {args.out / 'comparison.png'} in {time.perf_counter() - start:.1f} s on CPU")


if __name__ == "__main__":
    main()
