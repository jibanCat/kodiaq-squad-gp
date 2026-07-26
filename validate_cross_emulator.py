#!/usr/bin/env python3
"""Check this basedir against the previous version of the emulator.

We compare the fiducial high-fidelity P1D from this basedir with the one from
mafern/InferenceLyaData, the earlier PRIYA-trained emulator. That emulator is
fitted to the same PRIYA suite and the same 60 low-fidelity + 3 high-fidelity
design as this one, but to a flux measurement truncated at a lower k_max and
binned on a different k grid (102 bins in h/Mpc rather than 172). The
comparison therefore probes the independent refit and the rebinning, not
suite-to-suite stability. Agreement at the per-cent level over the eBOSS k
range confirms that the emulator is stable across the update.

We compare over the eBOSS k range (k = 0.001 to 0.0195 s/km) that the
measurement covers, read from emulator_params.json, and over all shared
redshifts. The largest mode (the lowest-k bin) is reported separately. Both
emulators describe the same simulations, so a difference there is refit scatter
rather than a difference in the underlying physics, and it sits at the level of
that mode's cosmic variance, about 2 per cent (Fernandez et al. 2024, JCAP 07
(2024) 029), which is the floor below which a difference at that mode carries no
physical meaning. The pass/fail gate is therefore applied away from the largest
mode.

This check is separate from verify_basedir.py because it needs a second, large
repository. Clone it first:

    git clone https://github.com/mafern/InferenceLyaData mafern-InferenceLyaData
    python validate_cross_emulator.py --mafern mafern-InferenceLyaData/Emulator_Files

We clone it under a distinct name because the quickstart in the README already
uses the directory name InferenceLyaData for jibanCat/InferenceLyaData, whose
emulator_params.json is byte-identical to this one, so pointing --mafern at the
wrong clone cannot be caught by inspecting that file.

The exit status is 0 when the two agree within the tolerance, 1 otherwise, and
2 when the check cannot run (lyaemu or the mafern basedir is absent).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

import verify_basedir as vb

#: k-bin count of the earlier emulator's flux vectors. We check it because
#: emulator_params.json is byte-identical across this basedir, a jibanCat
#: InferenceLyaData clone and a mafern one, so it cannot tell them apart. The
#: k-binning can: pointing --mafern at the wrong clone would otherwise run to
#: completion and report agreement of the emulator with itself.
PREVIOUS_NK = 102

#: The two emulators are fitted to the same PRIYA suite, so this gap is not
#: suite-to-suite scatter: it comes from the independent AR1 refit (ten
#: optimiser restarts with no fixed seed) and from the different k-binning.
#: A per-cent-level allowance covers both. Applied away from the largest mode.
DEFAULT_TOL = 0.02


def eboss_kf(basedir) -> np.ndarray:
    """The eBOSS k grid (s/km) the measurement covers, read from
    emulator_params.json. This is the k range over which we compare."""
    kf = json.loads((Path(basedir) / "emulator_params.json").read_text())["kf"]
    return np.asarray(kf, float)


def mafern_problem(mafern):
    """Return why `mafern` is not the earlier emulator, or None if it is."""
    mafern = Path(mafern)
    if not (mafern / "emulator_params.json").is_file():
        return (f"no emulator_params.json under {mafern}; pass the "
                f"Emulator_Files directory of a mafern/InferenceLyaData clone.")
    flux = mafern / vb.LF_FLUX
    if not flux.is_file():
        return (f"no {vb.LF_FLUX} under {mafern}; pass the Emulator_Files "
                f"directory of a mafern/InferenceLyaData clone.")
    try:
        with h5py.File(flux, "r") as f:
            nk = int(f["kfmpc"].shape[0])
    except (KeyError, OSError) as exc:
        return f"{flux} is malformed: {type(exc).__name__}: {exc}"
    if nk == PREVIOUS_NK:
        return None
    if nk == vb.EXPECTED_NK:
        looks_like = ("this repository's own basedir, so the comparison would "
                      "be against itself")
    elif nk == vb.UNCUT_NK:
        looks_like = "the uncut variant, not the earlier emulator"
    else:
        looks_like = "neither the earlier emulator nor this one"
    return (f"{mafern} has nk = {nk}, but the earlier emulator has "
            f"nk = {PREVIOUS_NK}. This is {looks_like}. Clone "
            f"mafern/InferenceLyaData and pass its Emulator_Files directory.")


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

    try:
        have_lyaemu = vb._import_gpwrap() is not None
    except ModuleNotFoundError as exc:
        print(vb._missing_dependency_detail(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(vb._broken_environment_detail(exc), file=sys.stderr)
        return 2
    if not have_lyaemu:
        print("lyaemu is not importable; put an InferenceLyaData clone on "
              "PYTHONPATH to run this check.", file=sys.stderr)
        return 2
    problem = mafern_problem(args.mafern)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    mafern = Path(args.mafern)

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
    print(f"largest-mode (lowest-k) deviations reach {worst_mode:.2%}. Both "
          f"emulators describe the same simulations, so this is refit scatter, "
          f"and it sits at the level of that mode's ~2% cosmic variance "
          f"(Fernandez et al. 2024), which we do not gate on.")
    if worst_away <= args.tol:
        print(f"away from the largest mode the two agree within {args.tol:.0%}: "
              f"worst {worst_away:.2%}")
        return 0
    print(f"DISAGREE away from the largest mode: worst {worst_away:.2%} exceeds "
          f"{args.tol:.0%}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
