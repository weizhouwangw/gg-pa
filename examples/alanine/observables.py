"""Observables for the alanine example: carbonyl O-O distance, dimer topology-torsion states and the MD references.

Positions are in nm and angles in radians, with frames on axis 0. The dimer's monomers are the topology's two chains.
"""

from pathlib import Path

import mdtraj as md
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" if (ROOT / "data").is_dir() else ROOT.parent / "data"
C_L, R_L = np.array([-2.4234, 2.6856]), 0.5938  # monomer basins L and R: centre and radius in (phi, psi)
C_R, R_R = np.array([-1.4261, 1.1006]), 0.7878
COS_MIN, HB_MIN = 0.8, 2  # anti-parallel cos <= -0.8, parallel cos >= 0.8; at least two hydrogen bonds
R_DA, ANGLE_DHA = 0.35, 120.0  # hydrogen bond: donor-acceptor distance < 0.35 nm, D-H-A angle > 120 degrees
STATES = ("A_LL", "A_LR", "A_RL", "A_RR", "P_LL", "P_LR", "P_RL", "P_RR", "U")


def oo_distance(traj):
    """ACE-to-ALA carbonyl O-O distance (T,) in nm."""
    o = [traj.topology.select(f"resname {r} and name O")[0] for r in ("ACE", "ALA")]
    return np.linalg.norm(traj.xyz[:, o[0]].astype(np.float64) - traj.xyz[:, o[1]], axis=-1)


def _monomers(top):
    return [np.array([a.index for a in c.atoms]) for c in top.chains]


def _basins(dihedrals):
    """L, R or off for (phi, psi) rows (T, 2)."""
    d = np.asarray(dihedrals, dtype=np.float64)
    dl, dr = (np.linalg.norm((d - c + np.pi) % (2 * np.pi) - np.pi, axis=-1) for c in (C_L, C_R))
    return np.where((dl <= R_L) & (dl < dr), "L", np.where((dr <= R_R) & (dr < dl), "R", "off"))


def fields(traj):
    """Per-frame orientation cosine, intermolecular H-bond count, reciprocity and both monomers' (phi, psi)."""
    x, v, chain = traj.xyz.astype(np.float64), [], np.empty(traj.n_atoms, dtype=np.int64)
    for m, atoms in enumerate(_monomers(traj.topology)):
        chain[atoms] = m
        names = [traj.topology.atom(i).name for i in atoms]  # end-to-end vector: last N minus first C
        v.append(x[:, atoms[len(names) - 1 - names[::-1].index("N")]] - x[:, atoms[names.index("C")]])
    cos = np.sum(v[0] * v[1], axis=-1) / (np.linalg.norm(v[0], axis=-1) * np.linalg.norm(v[1], axis=-1) + 1e-12)
    t = md.baker_hubbard(traj, freq=0.0, periodic=False, distance_cutoff=np.inf, angle_cutoff=0.0)
    t = t[chain[t[:, 0]] != chain[t[:, 2]]]  # donor and acceptor in different monomers
    on = (md.compute_distances(traj, t[:, [0, 2]], periodic=False) < R_DA) & (
        md.compute_angles(traj, t, periodic=False) > np.deg2rad(ANGLE_DHA)
    )
    n, donor = on.sum(axis=1), chain[t[:, 0]]
    reciprocal = (n >= 2) & (on & (donor == 0)).any(axis=1) & (on & (donor == 1)).any(axis=1)
    phi, psi = md.compute_phi(traj)[1], md.compute_psi(traj)[1]
    dih = [np.stack([phi[:, m], psi[:, m]], axis=1).astype(np.float64) for m in (0, 1)]
    return {"cos": cos, "n_hb": n.astype(np.int64), "reciprocal": reciprocal, "dih1": dih[0], "dih2": dih[1]}


def nine_states(cos, n_hb, reciprocal, dih1, dih2):
    """Index (T,) into STATES; frames without a clear topology, two H-bonds or both monomers in L or R are U."""
    b1, b2, cos = _basins(dih1), _basins(dih2), np.asarray(cos, dtype=np.float64)
    ok = (np.abs(cos) >= COS_MIN) & (np.asarray(n_hb) >= HB_MIN) & np.asarray(reciprocal, dtype=bool)
    return np.where(ok & (b1 != "off") & (b2 != "off"), 4 * (cos >= COS_MIN) + 2 * (b1 == "R") + (b2 == "R"), 8)


def load_sodium_ref():
    """AD-Na+ MD reference: the O-O distance in nm."""
    return {"oo_distance": np.load(DATA / "ad_sodium_ref/ad_sodium_md_300k_ref.npz")["oo_distance_nm"]}


def shipped_fields():
    """Dimer MD reference in the layout of fields()."""
    z = np.load(DATA / "ad_dimer_ref/ad_dimer_md_100ns_ref.npz")
    return {"cos": z["cosine_sim"], "n_hb": z["hbond_counts"].astype(np.int64), "reciprocal": z["reciprocal_counts"] == 1,
            "dih1": z["dihedrals_seg1_rad"], "dih2": z["dihedrals_seg2_rad"]}  # fmt: skip
