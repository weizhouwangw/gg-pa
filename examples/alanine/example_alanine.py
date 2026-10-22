"""Run AD-Na+ and the AD dimer with the package's alanine system and compare them with MD, as in Fig. 3."""

import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import observables as ao
import openmm
import openmm.app as app
import openmm.unit as u
import torch
import yaml
from observables import fields, shipped_fields
from scipy.stats import gaussian_kde

from ggpa.systems import alanine as ad

REFERENCE, GGPA, DIRECT = "#f2637d", "#ad2e7e", "#258b36"  # the paper's Fig. 3 colours
INK, BOND = "#0b0b0b", "#3a9e4f"  # bonds; ion coordination and hydrogen bonds
ELEMENTS = {
    "C": ("#6b6b6b", 60),
    "N": ("#2a5bd7", 60),
    "O": ("#d93b3b", 60),
    "H": ("#ececec", 22),
    "Na": ("#f2c14e", 240),
}


def _point(run_point, cfg, point, out):
    torch.set_num_threads(1)
    start = time.perf_counter()
    run_point(cfg, point, out)
    return f"{out.parent.name}/{point.id}", round(time.perf_counter() - start, 1)


def run(settings, out):
    """Run every sodium and dimer point, one process per point, longest first; return the wall time per point."""
    jobs = []
    for name, cls, run_point in (("dimer", ad.Dimer, ad.dimer_point), ("sodium", ad.Sodium, ad.sodium_point)):
        cfg = cls(**settings["common"], **settings[name])
        jobs += [(run_point, cfg, point, out / "results" / name / point.id) for point in ad.points(cfg)]
    timings = {}
    with ProcessPoolExecutor(settings["workers"], mp_context=mp.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(_point, *job) for job in jobs]):
            name, seconds = future.result()
            timings[name] = seconds
            print(f"{name}: {seconds:.0f} s", flush=True)
    return timings


def direct_oo(pdb, angles):
    """O-O distances (N,) nm of Direct Diffusion draws (N, 2) rad, each (phi, psi) set rigidly on the minimized peptide."""
    structure = app.PDBFile(str(pdb))
    positions = torch.tensor(structure.getPositions(asNumpy=True).value_in_unit(u.nanometer))
    template = ad.minimize(ad.vacuum_system(structure.topology), positions).numpy()
    topology = md.Topology.from_openmm(structure.topology)
    bonded = {atom.index: set() for atom in structure.topology.atoms()}
    for a, b in structure.topology.bonds():
        bonded[a.index].add(b.index)
        bonded[b.index].add(a.index)

    def side(start, behind):  # atoms reachable from start without crossing its bond to behind
        seen, stack = {start, behind}, [start]
        while stack:
            for j in bonded[stack.pop()] - seen:
                seen.add(j)
                stack.append(j)
        return sorted(seen - {behind})

    xyz = np.repeat(template[None], len(angles), axis=0)
    phi, psi = ad.backbone_quartets(structure.topology)[0].tolist()
    for quartet, target in ((psi, angles[:, 1]), (phi, angles[:, 0])):  # psi first; the phi rotation carries it
        b, c = quartet[1], quartet[2]
        now = md.compute_dihedrals(md.Trajectory(template[None], topology), [quartet])[0, 0]
        axis = (template[c] - template[b]) / np.linalg.norm(template[c] - template[b])
        atoms, angle = side(c, b), (target - now)[:, None, None]
        v = xyz[:, atoms] - template[b]  # Rodrigues rotation about b -> c
        v = v * np.cos(angle) + np.cross(axis, v) * np.sin(angle) + axis * (v @ axis)[..., None] * (1 - np.cos(angle))
        xyz[:, atoms] = template[b] + v
    return ao.oo_distance(md.Trajectory(xyz.astype(np.float32), topology))


def reduce(settings, out):
    """O-O distances of AD-Na+, per-frame dimer fields at the analysed rung, and one representative frame of each."""
    burn, results = settings["burn_in"], out / "results"
    peptide = md.load_topology(settings["sodium"]["pdb"])
    frames = []
    for path in sorted((results / "sodium").glob("ion*_traj*/samples.npz")):
        s = np.load(path)["s"][:, 0, 0]  # (sweeps, atoms, 3) nm; the ion is the last atom
        frames.append(s[int(burn * len(s)) :])
    frames = np.concatenate(frames)
    ggpa = ao.oo_distance(md.Trajectory(frames[:, : peptide.n_atoms], peptide))
    direct = np.load(next((results / "sodium").glob("direct_*/samples.npz")))["x"][:, 0]
    oo = {"Reference": ao.load_sodium_ref()["oo_distance"]}
    oo |= {"Direct Diffusion": direct_oo(settings["sodium"]["pdb"], direct), "GG-PA": ggpa}
    # the ion-bound frame: both carbonyl O within 0.3 nm of Na+, O-O closest to the median; Na+ drawn on top
    oxygens = [peptide.select(f"resname {r} and name O")[0] for r in ("ACE", "ALA")]
    bound = np.linalg.norm(frames[:, oxygens] - frames[:, -1:], axis=-1).max(1) < 0.3
    f = frames[np.argmin(np.where(bound, np.abs(ggpa - np.median(ggpa)), np.inf))]
    view = (f[oxygens[1]] - f[oxygens[0]], f[-1] - f[:-1].mean(0))
    structures = {"AD-Na$^+$": (f, peptide, [(o, peptide.n_atoms) for o in oxygens], view)}

    pair, ladder = md.load_topology(settings["dimer"]["pdb"]), settings["dimer"]["ladder"]
    xyz, runs = [], []
    for path in sorted((results / "dimer").glob("re_traj*/samples.npz" if ladder else "t*_traj*/samples.npz")):
        s = np.load(path)["s"][:, 0, 0]  # the first rung of the ladder
        xyz.append(s[int(burn * len(s)) :])
        runs.append(fields(md.Trajectory(xyz[-1], pair)))
    xyz = np.concatenate(xyz)
    sampled = {key: np.concatenate([r[key] for r in runs]) for key in runs[0]}
    # Choose A_LL and the more populated of P_LR/P_RL, then the frame nearest each state's circular-mean torsions.
    states = ao.nine_states(sampled["cos"], sampled["n_hb"], sampled["reciprocal"], sampled["dih1"], sampled["dih2"])
    torsions = np.concatenate([sampled["dih1"], sampled["dih2"]], axis=1)
    chain = np.array([a.residue.chain.index for a in pair.atoms])
    element = [a.element.symbol for a in pair.atoms]
    polar = [i for b in pair.bonds for i, j in ((b.atom1.index, b.atom2.index), (b.atom2.index, b.atom1.index))
             if element[i] == "H" and element[j] == "N"]  # fmt: skip
    first = [a for a in pair.atoms if chain[a.index] == 0]
    ends = (next(a.index for a in first if a.name == "C"), [a.index for a in first if a.name == "N"][-1])
    for name, wanted in (("anti-parallel", [0]), ("parallel", [5, 6])):
        rows = np.flatnonzero(states == max(wanted, key=lambda state: (states == state).sum()))
        if len(rows):
            mean = np.angle(np.exp(1j * torsions[rows]).mean(0))
            f = xyz[rows[np.argmin(np.abs(np.angle(np.exp(1j * (torsions[rows] - mean)))).sum(1))]]
            hbonds = [(h, o) for h in polar for o in pair.select("element O")
                      if chain[h] != chain[o] and np.linalg.norm(f[h] - f[o]) < 0.25]  # fmt: skip
            view = (f[chain == 1].mean(0) - f[chain == 0].mean(0), f[ends[1]] - f[ends[0]])
            structures[name.capitalize()] = (f, pair, hbonds, view)
    return oo, {"Reference": shipped_fields(), "GG-PA-RE" if ladder else "GG-PA": sampled}, structures


def draw(ax, xyz, topology, dotted, view):
    """Ball and stick projection of xyz (n, 3) nm, back to front, dotted atom pairs; view = (right, up) vectors.

    Hydrogens are shown on nitrogen only; atoms beyond the topology are Na+."""
    symbols = [a.element.symbol for a in topology.atoms] + ["Na"] * (len(xyz) - topology.n_atoms)
    pairs = [(b.atom1.index, b.atom2.index) for b in topology.bonds]
    hidden = {i for a, b in pairs for i, j in ((a, b), (b, a)) if symbols[i] == "H" and symbols[j] == "C"}
    right, up = (v / np.linalg.norm(v) for v in view)
    up = (up - (up @ right) * right) / np.linalg.norm(up - (up @ right) * right)
    p = (xyz - xyz.mean(0)) @ np.stack([right, up, np.cross(right, up)]).T  # screen x, y and depth to the viewer
    for i, j in pairs:
        if not hidden & {i, j}:
            ax.plot(*p[[i, j], :2].T, color=INK, lw=1.6, solid_capstyle="round", zorder=p[[i, j], 2].mean())
    for i, j in dotted:
        ax.plot(*p[[i, j], :2].T, ":", color=BOND, lw=1.6, zorder=p[:, 2].max() + 1)
    for i, symbol in enumerate(symbols):
        if i not in hidden:
            colour, size = ELEMENTS[symbol]
            ax.scatter(*p[i, :2], s=size, color=colour, edgecolors=INK, linewidths=0.6, zorder=p[i, 2] + 0.01)
    ax.set_aspect("equal")
    ax.margins(0.15)
    ax.axis("off")


def plot(oo, dimer, structures, out):
    """(a) AD-Na+ O-O distance and (b) dimer |psi1 - psi2| by topology, with representative GG-PA structures inset."""
    colours = {"Reference": REFERENCE, "Direct Diffusion": DIRECT}
    insets = {"AD-Na$^+$": (0, [0.33, 0.42, 0.34, 0.5]), "Anti-parallel": (1, [-0.02, 0.5, 0.36, 0.5])}
    insets["Parallel"] = (1, [0.55, 0.38, 0.36, 0.5])
    with plt.rc_context({"font.size": 10, "axes.linewidth": 0.9, "lines.linewidth": 1.8}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
        x, top = np.linspace(1.5, 6.5, 500), 0.0
        for (name, values), style in zip(oo.items(), ("--", ":", "-"), strict=True):
            density = gaussian_kde(10 * values)(x)
            axes[0].plot(x, density, style, color=colours.get(name, GGPA), label=name)
            top = max(top, density.max())
        axes[0].set(
            xlabel=r"$d_\mathrm{OO}$ ($\mathrm{\AA}$)", ylabel="Density", title="AD-Na$^+$", ylim=(0, 1.3 * top)
        )
        axes[0].legend(frameon=False, fontsize=9, loc="upper right")
        x, top = np.linspace(-np.pi / 2, np.pi, 500), 0.0  # the axis continues below 0, as in the paper
        for name, f in dimer.items():
            states = ao.nine_states(f["cos"], f["n_hb"], f["reciprocal"], f["dih1"], f["dih2"])
            dpsi = np.abs((f["dih1"][:, 1] - f["dih2"][:, 1] + np.pi) % (2 * np.pi) - np.pi)
            for topology, style in ((states < 4, "-"), ((states >= 4) & (states < 8), "--")):
                if topology.sum() > 1:
                    density = gaussian_kde(dpsi[topology])(x)
                    axes[1].plot(x, density, style, color=colours.get(name, GGPA))
                    top = max(top, density.max())
        for label, colour, style in (("Reference", REFERENCE, "-"), (list(dimer)[1], GGPA, "-")):
            axes[1].plot([], [], style, color=colour, label=label)
        axes[1].plot([], [], "-", color="0.3", label="Anti-parallel")
        axes[1].plot([], [], "--", color="0.3", label="Parallel")
        axes[1].set(xlabel=r"$|\psi_1-\psi_2|$", ylabel="Density", title="AD dimer", xlim=(x[0], np.pi))
        axes[1].set(ylim=(0, 1.35 * top))
        axes[1].set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], ["0", "π/4", "π/2", "3π/4", "π"])
        axes[1].legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 0.05))
        for name, structure in structures.items():
            panel, box = insets[name]
            inset = axes[panel].inset_axes(box)
            draw(inset, *structure)
            if panel == 1:
                inset.set_title(name, fontsize=9.5, fontweight="bold", pad=0)
        for letter, ax in zip("ab", axes, strict=True):
            ax.spines[["top", "right"]].set_visible(False)
            ax.text(-0.1, 1.03, f"({letter})", transform=ax.transAxes, fontweight="bold")
        (out / "figures").mkdir()
        fig.savefig(out / "figures/comparison.png", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("settings.yaml"))
    parser.add_argument("--out", type=Path, default=Path("runs") / datetime.now().strftime("alanine_%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "settings.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    torch.set_num_threads(1)
    started = time.perf_counter()
    timings = run(settings, args.out)
    plot(*reduce(settings, args.out), args.out)
    manifest = {
        "workers": settings["workers"],
        "torch": torch.__version__,
        "openmm": openmm.__version__,
        "timings": timings,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {args.out / 'figures/comparison.png'} ({manifest['elapsed_seconds']:.0f} s)")


if __name__ == "__main__":
    main()
