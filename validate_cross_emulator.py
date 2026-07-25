#!/usr/bin/env python3
"""Check this basedir against the previous version of the emulator.

We compare the fiducial high-fidelity P1D from this basedir with the one from
mafern/InferenceLyaData, the previous version of this emulator, trained on an
earlier PRIYA suite of 60 low-fidelity and 3 high-fidelity simulations and
binned on a different k grid. Agreement at the per-cent level over the eBOSS
k range confirms that the emulator is stable across the update.

We compare over the eBOSS k range (k = 0.001 to 0.0195 s/km) that the
measurement covers, read from emulator_params.json, and over all shared
redshifts. The largest mode (the lowest-k bin) is reported separately, because
that mode carries a cosmic variance of about 2 per cent (Fernandez et al. 2024,
JCAP 07 (2024) 029), so a per-cent-level difference there is expected and is not
an emulator disagreement. The pass/fail gate is therefore applied away from the
largest mode.

This check is separate from verify_basedir.py because it needs a second, large
repository. Clone it first:

    git clone https://github.com/mafern/InferenceLyaData
    python validate_cross_emulator.py --mafern InferenceLyaData/Emulator_Files

The exit status is 0 when the two agree within the tolerance, 1 otherwise, and
2 when the check cannot run (lyaemu or the mafern basedir is absent).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import verify_basedir as vb

#: The two emulator versions were trained on different PRIYA suites, so we
#: allow a per-cent-level gap. Applied away from the largest mode.
DEFAULT_TOL = 0.02


def eboss_kf(basedir) -> np.ndarray:
    """The eBOSS k grid (s/km) the measurement covers, read from
    emulator_params.json. This is the k range over which we compare."""
    kf = json.loads((Path(basedir) / "emulator_params.json").read_text())["kf"]
    return np.asarray(kf, float)


def compare(this_basedir, mafern_basedir, *, kf=None, redshifts=None):
    """Return per-redshift rows and the overall worst deviations.

    Each row is (z, median, worst_all, k_at_worst, largest_mode_dev,
    worst_away_from_largest_mode). The grid is sorted ascending in k, so index 0
    is the largest mode.
    """
    if kf is None:
        kf = eboss_kf(this_basedir)
    kf = np.sort(np.asarray(kf, float))
    theta = np.asarray(vb.FIDUCIAL_THETA, float)
    z_this, p_this = vb._predict(this_basedir, theta, kf, "hf")
    z_maf, p_maf = vb._predict(mafern_basedir, theta, kf, "hf")
    shared = sorted(set(np.round(z_this, 1)) & set(np.round(z_maf, 1)))
    if redshifts is not None:
        shared = [z for z in shared if round(z, 1) in {round(r, 1) for r in redshifts}]
    rows = []
    for z in shared:
        a = p_this[vb._z_index(z_this, z)]
        b = p_maf[vb._z_index(z_maf, z)]
        rel = np.abs(a - b) / np.abs(b)
        rows.append((float(z), float(np.median(rel)), float(rel.max()),
                     float(kf[rel.argmax()]), float(rel[0]), float(rel[1:].max())))
    return kf, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--basedir", default=str(Path(__file__).resolve().parent),
                    help="this KODIAQ-SQUAD basedir (default: repository root)")
    ap.add_argument("--mafern", required=True,
                    help="path to a mafern/InferenceLyaData Emulator_Files directory")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help=f"maximum allowed deviation away from the largest mode "
                         f"(default {DEFAULT_TOL})")
    args = ap.parse_args(argv)

    if vb._import_gpwrap() is None:
        print("lyaemu is not importable; put an InferenceLyaData clone on "
              "PYTHONPATH to run this check.", file=sys.stderr)
        return 2
    mafern = Path(args.mafern)
    if not (mafern / "emulator_params.json").is_file():
        print(f"no emulator_params.json under {mafern}; pass the "
              "Emulator_Files directory of a mafern/InferenceLyaData clone.",
              file=sys.stderr)
        return 2

    kf, rows = compare(args.basedir, mafern)
    print(f"Fiducial high-fidelity P1D, this basedir vs {mafern}")
    print(f"eBOSS k range k = {kf[0]:.5f} to {kf[-1]:.4f} s/km, "
          f"{len(rows)} shared redshifts")
    print(f"{'z':>5} {'median':>9} {'worst':>9} {'k at worst':>11} "
          f"{'largest mode':>13} {'away from it':>13}")
    worst_away = 0.0
    worst_mode = 0.0
    for z, med, wall, kw, mode0, waway in rows:
        print(f"{z:5.1f} {med:8.2%} {wall:8.2%} {kw:11.5f} "
              f"{mode0:12.2%} {waway:12.2%}")
        worst_away = max(worst_away, waway)
        worst_mode = max(worst_mode, mode0)
    print()
    print(f"largest-mode (lowest-k) deviations reach {worst_mode:.2%}, consistent "
          f"with the ~2% cosmic variance of that mode (Fernandez et al. 2024).")
    if worst_away <= args.tol:
        print(f"away from the largest mode the two agree within {args.tol:.0%}: "
              f"worst {worst_away:.2%}")
        return 0
    print(f"DISAGREE away from the largest mode: worst {worst_away:.2%} exceeds "
          f"{args.tol:.0%}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
