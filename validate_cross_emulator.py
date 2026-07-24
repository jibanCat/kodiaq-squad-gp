#!/usr/bin/env python3
"""Check this basedir against the previous version of the emulator.

We compare the fiducial high-fidelity P1D from this basedir with the one from
mafern/InferenceLyaData, the previous version of this emulator, trained on an
earlier PRIYA suite of 48 low-fidelity and 3 high-fidelity simulations and
binned on a different k grid. Agreement at the percent level over their common
range confirms that the emulator is stable across the update.

This check is separate from verify_basedir.py because it needs a second, large
repository. Clone it first:

    git clone https://github.com/mafern/InferenceLyaData
    python validate_cross_emulator.py --mafern InferenceLyaData/Emulator_Files

The exit status is 0 when the two agree within the tolerance, 1 otherwise, and
2 when the check cannot run (lyaemu or the mafern basedir is absent).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import verify_basedir as vb

#: mafern's query grid reaches this k, in s/km. We compare just inside it.
MAFERN_KMAX = 0.0195
#: Per-redshift comparison points.
REDSHIFTS = [2.6, 3.6, 4.2]
#: The two emulator versions were trained on different PRIYA suites, so we
#: allow a percent-level gap, with a little headroom at the extreme edge of the
#: previous version's range.
DEFAULT_TOL = 0.02


def _common_grid(kmax: float, n: int = 40) -> np.ndarray:
    return np.logspace(np.log10(0.0011), np.log10(kmax), n)


def compare(this_basedir, mafern_basedir, *, kmax=MAFERN_KMAX, redshifts=REDSHIFTS):
    """Return a list of (z, median_rel, max_rel, k_at_max) rows."""
    kf = _common_grid(kmax)
    theta = np.asarray(vb.FIDUCIAL_THETA, float)
    z_this, p_this = vb._predict(this_basedir, theta, kf, "hf")
    z_maf, p_maf = vb._predict(mafern_basedir, theta, kf, "hf")
    rows = []
    for z in redshifts:
        a = p_this[vb._z_index(z_this, z)]
        b = p_maf[vb._z_index(z_maf, z)]
        rel = np.abs(a - b) / np.abs(b)
        rows.append((z, float(np.median(rel)), float(rel.max()), float(kf[rel.argmax()])))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--basedir", default=str(Path(__file__).resolve().parent),
                    help="this KODIAQ-SQUAD basedir (default: repository root)")
    ap.add_argument("--mafern", required=True,
                    help="path to a mafern/InferenceLyaData Emulator_Files directory")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help=f"maximum allowed relative deviation (default {DEFAULT_TOL})")
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

    rows = compare(args.basedir, mafern)
    print(f"Fiducial high-fidelity P1D, this basedir vs {mafern}")
    print(f"common grid k = 0.0011 to {MAFERN_KMAX} s/km")
    print(f"{'z':>5} {'median':>10} {'worst':>10} {'k at worst (s/km)':>18}")
    worst = 0.0
    for z, med, mx, kmx in rows:
        print(f"{z:5.1f} {med:9.2%} {mx:9.2%} {kmx:18.4f}")
        worst = max(worst, mx)
    if worst <= args.tol:
        print(f"\nagree within {args.tol:.0%}: worst deviation {worst:.2%}")
        return 0
    print(f"\nDISAGREE: worst deviation {worst:.2%} exceeds {args.tol:.0%}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
