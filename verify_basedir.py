#!/usr/bin/env python3
"""Verify that this directory is the KODIAQ-SQUAD GP emulator basedir.

We run several groups of checks: the file manifest, the k-binning, the trained
GPs, the resolution-correction table, and a fiducial P1D. The grid check is the
reason this script exists: an emulator assembled from the uncut flux vectors
loads without complaint and predicts a P1D that differs at the percent level,
so we test the binning directly rather than trusting the file name. The
prediction check reproduces a stored P1D from the trained GPs, and we skip it
when lyaemu is not importable.

The file, grid, and shape checks compare against constants hard-coded below
rather than read from reference.json, because reference.json is itself part of
what we are checking: a reader who regenerates it on a damaged basedir would
otherwise obtain a file that certifies the damage. The fiducial-P1D check does
read its expected values from reference.json; that is safe because the trained
GPs that produce the P1D are themselves pinned by the hard-coded manifest
digest, so a tampered basedir fails the file check first.

Usage:

    python verify_basedir.py                  # all checks
    python verify_basedir.py --no-predictions # the subset that does not need lyaemu
    python verify_basedir.py --basedir DIR    # verify a directory elsewhere

The exit status is 0 when nothing failed and 1 otherwise (a failing check, a
missing or malformed reference.json, or an --emit-reference refused because the
basedir does not pass its own checks). Skipped checks do not fail the run unless
--strict is given. Usage errors (contradictory flags, or --emit-reference with
predictions requested but lyaemu unavailable) exit 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import namedtuple
from pathlib import Path

import h5py
import numpy as np

Check = namedtuple("Check", "name status detail")

REFERENCE_FILE = "reference.json"

#: k-bin count of the cut flux vectors this basedir ships.
EXPECTED_NK = 172
#: k-bin count of the uncut variant, which we must never use here.
UNCUT_NK = 329
N_ZBINS = 13

LF_FLUX = "mf_emulator_flux_vectors_tau1000000.hdf5"
HF_FLUX = "hires/mf_emulator_flux_vectors_tau1000000.hdf5"
RES_CORR = "res_corr/resolution_correction.h5"

#: The only paths that carry emulator data. Anything else in the directory,
#: such as this script or the README, is documentation and is not verified.
DATA_ROOTS = ("emulator_params.json", LF_FLUX, "hires", "trained_mf", "res_corr")

# --- Ground truth. These pin the basedir independently of reference.json. ---

#: Number of data files the basedir ships.
EXPECTED_N_FILES = 32
#: MD5 over the canonical "relative/path:md5" listing of those files.
MANIFEST_DIGEST = "db5f35716431bb210a650eaebf15a661"
#: Shape of the resolution-correction table: 15 redshifts x 59 k-bins.
RES_CORR_SHAPE = (15, 59)
#: MD5 of the k grid itself, so that a file with the right number of bins but
#: the wrong bin values cannot pass.
KFMPC_DIGEST_LF = "2ab10a4daa04fa386fea178d80f4d7b2"
KFMPC_DIGEST_HF = "99fab60562016b76ca02e6e52ef544a3"

#: Name we give the leading design column. It holds the mean optical-depth
#: scaling and is absent from the param_names map in emulator_params.json.
MEAN_FLUX_PARAM = "tau0"

#: Redshifts at which we store the reference P1D. These are the three the
#: paper reports.
REFERENCE_REDSHIFTS = [2.6, 3.6, 4.2]

#: Relative tolerance for the prediction check. Prediction from a loaded GP is
#: deterministic, so the only expected spread is the platform linear algebra.
PREDICTION_RTOL = 1e-6


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def file_md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _data_files(basedir) -> list[Path]:
    """Every shipped data file, relative to basedir, in a stable order.

    We walk an explicit whitelist and follow symlinks, since a reader may well
    keep the 30 MB of trained GPs on shared storage and link to it.
    """
    basedir = Path(basedir)
    out = []
    for root in DATA_ROOTS:
        p = basedir / root
        if p.is_file():
            out.append(Path(root))
        elif p.is_dir():
            for dirpath, dirnames, names in os.walk(p, followlinks=True):
                # Skip editor cruft such as .ipynb_checkpoints, which a Jupyter
                # session can drop inside the basedir but is not part of it.
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                for name in names:
                    if name.startswith("."):
                        continue
                    out.append((Path(dirpath) / name).relative_to(basedir))
    return sorted(out, key=str)


def manifest_of(basedir) -> dict:
    basedir = Path(basedir)
    return {str(rel): {"bytes": (basedir / rel).stat().st_size,
                       "md5": file_md5(basedir / rel)}
            for rel in _data_files(basedir)}


def manifest_digest(manifest: dict) -> str:
    listing = "".join(f"{k}:{v['md5']}\n" for k, v in sorted(manifest.items()))
    return hashlib.md5(listing.encode()).hexdigest()


def _ar1_files(basedir) -> list[Path]:
    """The extension-less trained_mf/zbin<z> files. These carry the AR1
    multi-fidelity hyperparameters that lyaemu's load_GP reads, and are the
    files that actually differ between the cut and uncut emulators."""
    d = Path(basedir) / "trained_mf"
    if not d.is_dir():
        return []
    return sorted((p for p in d.glob("zbin*") if not p.name.endswith(".json")),
                  key=lambda p: p.name)


def _lf_gp_files(basedir) -> list[Path]:
    d = Path(basedir) / "trained_mf"
    return sorted(d.glob("zbin*.json")) if d.is_dir() else []


def _import_gpwrap():
    """Return lyaemu's GPWrap, or None when lyaemu itself is not importable.

    We only treat a missing `lyaemu` as "not available". A different missing
    dependency (for example matplotlib, which lyaemu imports at module load)
    is a broken environment, not a missing clone, so we re-raise it rather than
    silently reporting it as "lyaemu not on PYTHONPATH"."""
    try:
        from lyaemu.gp_wrap import GPWrap
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] == "lyaemu":
            return None
        raise
    return GPWrap


def _missing_dependency_detail(exc: ModuleNotFoundError) -> str:
    """Name the module that is actually missing, rather than guessing.

    lyaemu imports matplotlib and pandas at module load, and a reader who is
    told the wrong name redoes the step that just failed."""
    missing = exc.name or "a dependency"
    return (f"lyaemu could not be imported because {missing} is missing. "
            f"Install the packages in requirements.txt, which lists it.")


def _broken_environment_detail(exc: Exception) -> str:
    """A non-import failure while loading lyaemu means the installed packages
    do not match each other. The usual case is GPy built against numpy 1.x
    running under numpy 2.x, which raises ValueError, not ImportError."""
    return (f"lyaemu could not be imported: {type(exc).__name__}: {exc}. "
            f"The installed packages probably do not match requirements.txt; "
            f"GPy 1.13.2 needs numpy==1.26.4 and fails under numpy 2.x.")


# ---------------------------------------------------------------------------
# Observing the basedir
# ---------------------------------------------------------------------------


def observe(basedir) -> dict:
    """Read the grid, the design, the trained-GP shapes, and the
    resolution-correction shape from disk."""
    basedir = Path(basedir)
    with h5py.File(basedir / LF_FLUX, "r") as f:
        nk = int(f["kfmpc"].shape[0])
        zout = [float(z) for z in f["zout"][()]]
        lf_params = f["params"][()]
        kfmpc = f["kfmpc"][()]
        kfkms_lf = f["kfkms"][()]
    with h5py.File(basedir / HF_FLUX, "r") as f:
        hf_params = f["params"][()]
        kfkms_hf = f["kfkms"][()]

    names = json.loads((basedir / "emulator_params.json").read_text()).get("param_names", {})
    ordered = [n for n, _ in sorted(names.items(), key=lambda kv: kv[1])]
    # emulator_params.json names nine physics parameters, while the design
    # carries ten columns. The leading column is the mean optical-depth
    # scaling, which upstream does not list, so we name it here to keep the
    # reported ranges aligned with their columns.
    if len(ordered) == int(lf_params.shape[1]) - 1:
        ordered = [MEAN_FLUX_PARAM] + ordered

    zbins = _lf_gp_files(basedir)
    x_shape = y_shape = None
    if zbins:
        payload = json.loads(zbins[0].read_text())
        x_shape = list(np.asarray(payload["X"]).shape)
        y_shape = list(np.asarray(payload["Y"]).shape)

    res_corr_shape = None
    rc = basedir / RES_CORR
    if rc.is_file():
        with h5py.File(rc, "r") as f:
            res_corr_shape = list(f["res_corr"].shape)

    return {
        "grid": {
            "nk": nk,
            "n_zbins": len(zout),
            "zout": zout,
            "kfmpc_range": [float(kfmpc.min()), float(kfmpc.max())],
            "kfkms_range_lf": [float(np.nanmin(kfkms_lf)), float(np.nanmax(kfkms_lf))],
            "kfkms_range_hf": [float(np.nanmin(kfkms_hf)), float(np.nanmax(kfkms_hf))],
        },
        "design": {
            "lf_samples": int(lf_params.shape[0]),
            "hf_samples": int(hf_params.shape[0]),
            "n_params": int(lf_params.shape[1]),
            "param_names": ordered,
            "param_min": [float(v) for v in lf_params.min(axis=0)],
            "param_max": [float(v) for v in lf_params.max(axis=0)],
        },
        "trained_mf": {
            "n_zbins": len(zbins),
            "n_ar1_files": len(_ar1_files(basedir)),
            "x_shape": x_shape,
            "y_shape": y_shape,
        },
        "res_corr": {"shape": res_corr_shape},
    }


def build_reference(basedir, *, predictions: bool = True) -> dict:
    """Derive reference.json from a basedir. Used by --emit-reference."""
    ref = observe(basedir)
    ref["files"] = manifest_of(basedir)
    if predictions:
        ref["predictions"] = build_predictions(basedir)
        ref["provenance"] = _provenance()
    return ref


def _provenance() -> dict:
    """Record what produced the prediction values, so a reader can tell whether
    their environment matches. This is informational; the checks do not gate on
    it. lyaemu is identified by a digest of its .py files rather than a git SHA,
    so the value is the same across identical forks."""
    import platform
    versions = {}
    for pkg in ("numpy", "scipy", "h5py", "GPy", "emukit"):
        try:
            versions[pkg] = __import__(pkg).__version__
        except Exception:
            versions[pkg] = None
    lyaemu_code_md5 = None
    try:
        import lyaemu
        pkgdir = Path(lyaemu.__file__).resolve().parent
        h = hashlib.md5()
        for f in sorted(pkgdir.glob("*.py")):
            h.update(f.read_bytes())
        lyaemu_code_md5 = h.hexdigest()
    except Exception:
        pass
    return {"python": platform.python_version(),
            "lyaemu_code_md5": lyaemu_code_md5, **versions}


def build_predictions(basedir) -> dict:
    """Predict the fiducial P1D from the trained GPs at both fidelities."""
    kf = _reference_kf()
    out = {"theta": list(FIDUCIAL_THETA), "kf": [float(v) for v in kf],
           "rtol": PREDICTION_RTOL, "redshifts": list(REFERENCE_REDSHIFTS)}
    for fidelity in ("lf", "hf"):
        zout, p1d = _predict(basedir, np.asarray(FIDUCIAL_THETA, float), kf, fidelity)
        out[fidelity] = {f"{z:.1f}": [float(v) for v in p1d[_z_index(zout, z)]]
                         for z in REFERENCE_REDSHIFTS}
    return out


#: Fiducial parameter vector, in the physical units the emulator expects.
#: Order: dtau0, tau0, ns, Ap, herei, heref, alphaq, hub, omegamh2, hireionz,
#: bhfeedback.
FIDUCIAL_THETA = [-0.009, 1.09, 0.983, 1.46e-09, 4.0, 2.765, 1.74, 0.688,
                  0.1439, 7.24, 0.05]


def _reference_kf() -> np.ndarray:
    """The 48-point logarithmic grid, k in s/km, at which we evaluate the
    reference P1D. This is a query grid, not a grid the GPs were trained on."""
    return np.logspace(np.log10(0.001), np.log10(0.04), 48)


def _z_index(zout, z) -> int:
    return int(np.argmin(np.abs(np.asarray(zout, float) - z)))


def _predict(basedir, theta, kf, fidelity):
    """Load the emulator at one fidelity and predict P1D on kf."""
    GPWrap = _import_gpwrap()
    if GPWrap is None:
        raise ImportError("lyaemu is not importable.")
    basedir = Path(basedir)
    gp = GPWrap(basedir=str(basedir), emulator_json_file="emulator_params.json",
                kf=kf, tau_thresh=1e6, use_res_corr=False)
    gp.set_emulator(HRbasedir=str(basedir / "hires") if fidelity == "hf" else None,
                    max_z=4.6, min_z=2.2, traindir=str(basedir / "trained_mf"))
    gp.set_mf_param_limits(basedir=str(basedir))
    _, p1d, _ = gp.get_predicted(np.asarray(theta, float))
    return np.asarray(gp.zout, float), np.asarray(p1d, float)


def load_reference(basedir) -> dict:
    return json.loads((Path(basedir) / REFERENCE_FILE).read_text())


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_files(basedir, ref=None) -> list[Check]:
    """Every data file is present, unmodified, and nothing extra has appeared.

    We compare against the hard-coded manifest digest, so this check holds even
    when reference.json has been regenerated or removed.
    """
    basedir = Path(basedir)
    observed = manifest_of(basedir)
    checks = []

    if len(observed) != EXPECTED_N_FILES:
        expected_names = set((ref or {}).get("files", {}))
        found = set(observed)
        extra = sorted(found - expected_names) if expected_names else []
        missing = sorted(expected_names - found) if expected_names else []
        detail = f"found {len(observed)} data files, expected {EXPECTED_N_FILES}"
        if missing:
            detail += f"; missing {', '.join(missing[:4])}"
        if extra:
            detail += f"; unexpected {', '.join(extra[:4])}"
        checks.append(Check("file inventory", "FAIL", detail))
    else:
        checks.append(Check("file inventory", "PASS",
                            f"{EXPECTED_N_FILES} data files"))

    digest = manifest_digest(observed)
    if digest == MANIFEST_DIGEST:
        checks.append(Check("file contents", "PASS",
                            f"manifest digest {digest[:12]} matches"))
    else:
        altered = []
        for rel, entry in sorted((ref or {}).get("files", {}).items()):
            got = observed.get(rel)
            if got and got["md5"] != entry["md5"]:
                altered.append(rel)
        detail = f"manifest digest {digest[:12]} does not match {MANIFEST_DIGEST[:12]}"
        if altered:
            detail += f"; altered: {', '.join(altered[:4])}"
        checks.append(Check("file contents", "FAIL", detail))
    return checks


def check_grid(basedir) -> list[Check]:
    """Confirm the k-binning. We keep this check independent of reference.json,
    so that a regenerated reference cannot silently validate the wrong flux
    vectors."""
    basedir = Path(basedir)
    checks = []
    for label, rel, digest in (("low-fidelity", LF_FLUX, KFMPC_DIGEST_LF),
                               ("high-fidelity", HF_FLUX, KFMPC_DIGEST_HF)):
        path = basedir / rel
        if not path.exists():
            checks.append(Check(f"{label} k-binning", "FAIL", f"{rel} is missing"))
            continue
        try:
            with h5py.File(path, "r") as f:
                kfmpc = f["kfmpc"][()]
        except (KeyError, OSError) as exc:
            checks.append(Check(f"{label} k-binning", "FAIL",
                                f"{rel} is malformed: {type(exc).__name__}: {exc}"))
            continue
        nk = int(kfmpc.shape[0])
        if nk == UNCUT_NK:
            checks.append(Check(
                f"{label} k-binning", "FAIL",
                f"nk = {UNCUT_NK}, expected {EXPECTED_NK}. This is the uncut "
                f"variant. The paper uses the k-cut flux vectors, published in "
                f"InferenceLyaData as mf_emulator_flux_vectors_tau1000000_cut.hdf5. "
                f"Copy the *_cut.hdf5 file over {rel}, or clone this repository "
                f"instead of assembling a basedir by hand."))
            continue
        if nk != EXPECTED_NK:
            checks.append(Check(f"{label} k-binning", "FAIL",
                                f"nk = {nk}, expected {EXPECTED_NK}"))
            continue
        got = hashlib.md5(kfmpc.tobytes()).hexdigest()
        if got == digest:
            checks.append(Check(f"{label} k-binning", "PASS",
                                f"nk = {nk}, k grid matches"))
        else:
            checks.append(Check(f"{label} k-binning", "FAIL",
                                f"nk = {nk} as expected, but the k values differ "
                                f"(digest {got[:12]}, expected {digest[:12]}); "
                                f"these flux vectors were resampled or rebinned"))

    lf = basedir / LF_FLUX
    if lf.exists():
        try:
            with h5py.File(lf, "r") as f:
                n_z = int(f["zout"].shape[0])
                n_s, n_flat = f["flux_vectors"].shape
                nk = int(f["kfmpc"].shape[0])
        except (KeyError, ValueError, OSError) as exc:
            checks.append(Check("flux vector layout", "FAIL",
                                f"{LF_FLUX} is malformed: {type(exc).__name__}: {exc}"))
            return checks
        if n_z == N_ZBINS:
            checks.append(Check("redshift bins", "PASS", f"{n_z} bins"))
        else:
            checks.append(Check("redshift bins", "FAIL",
                                f"{n_z} bins, expected {N_ZBINS}"))
        if n_flat == n_z * nk:
            checks.append(Check("flux vector layout", "PASS",
                                f"{n_s} x {n_flat} = {n_s} x {n_z} x {nk}"))
        else:
            checks.append(Check("flux vector layout", "FAIL",
                                f"{n_flat} columns, expected {n_z} x {nk} = {n_z * nk}"))
    return checks


def check_trained_mf(basedir, ref=None) -> list[Check]:
    """The trained GPs must be present and match the k-binning.

    Two kinds of file live in trained_mf. The extension-less zbin<z> files hold
    the AR1 multi-fidelity hyperparameters, which is what the high-fidelity
    path loads and what differs between the cut and uncut emulators. The
    zbin<z>.json files hold the single-fidelity GP the low-fidelity path loads.
    When either is absent, lyaemu retrains silently, and the result does not
    reproduce the published values.
    """
    basedir = Path(basedir)
    ar1 = _ar1_files(basedir)
    lf_gp = _lf_gp_files(basedir)
    checks = []

    if len(ar1) == N_ZBINS:
        checks.append(Check("AR1 hyperparameters", "PASS",
                            f"{len(ar1)} redshift bins"))
    else:
        checks.append(Check("AR1 hyperparameters", "FAIL",
                            f"{len(ar1)} extension-less trained_mf/zbin<z> files, "
                            f"expected {N_ZBINS}; without them lyaemu retrains the "
                            f"multi-fidelity GP on load and writes the result back "
                            f"into trained_mf/, which does not reproduce the "
                            f"published values"))

    if not lf_gp:
        checks.append(Check("low-fidelity GPs", "FAIL",
                            "trained_mf/ holds no zbin<z>.json files"))
        return checks
    if len(lf_gp) == N_ZBINS:
        checks.append(Check("low-fidelity GPs", "PASS", f"{len(lf_gp)} redshift bins"))
    else:
        checks.append(Check("low-fidelity GPs", "FAIL",
                            f"{len(lf_gp)} zbin<z>.json files, expected {N_ZBINS}"))

    try:
        widths = {int(np.asarray(json.loads(p.read_text())["Y"]).shape[1]) for p in lf_gp}
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
        checks.append(Check("trained GP width", "FAIL",
                            f"a trained_mf/zbin<z>.json is malformed: "
                            f"{type(exc).__name__}: {exc}"))
        return checks
    if widths == {EXPECTED_NK}:
        checks.append(Check("trained GP width", "PASS",
                            f"Y has {EXPECTED_NK} columns in every bin"))
    else:
        checks.append(Check("trained GP width", "FAIL",
                            f"Y column counts {sorted(widths)}, expected "
                            f"{EXPECTED_NK}; these GPs were trained on "
                            f"different flux vectors"))
    return checks


def check_res_corr(basedir) -> list[Check]:
    """Confirm the resolution-correction table is present with the right shape.
    The table is a record of the L15n512 over L15n384 flux ratio; lyaemu loads
    its own copy from the package, so this is a completeness check rather than a
    load path."""
    rc = Path(basedir) / RES_CORR
    if not rc.is_file():
        return [Check("resolution correction", "FAIL", f"{RES_CORR} is missing")]
    with h5py.File(rc, "r") as f:
        if "res_corr" not in f:
            return [Check("resolution correction", "FAIL",
                          f"{RES_CORR} has no res_corr dataset")]
        shape = tuple(int(v) for v in f["res_corr"].shape)
    if shape == RES_CORR_SHAPE:
        return [Check("resolution correction", "PASS",
                      f"{shape[0]} redshifts x {shape[1]} k-bins")]
    return [Check("resolution correction", "FAIL",
                  f"res_corr shape {shape}, expected {RES_CORR_SHAPE}")]


def check_predictions(basedir, ref) -> list[Check]:
    """Reproduce the stored fiducial P1D. We skip this without lyaemu."""
    stored = (ref or {}).get("predictions")
    if not stored:
        return [Check("fiducial P1D", "SKIP", "no predictions in reference.json")]
    try:
        available = _import_gpwrap() is not None
    except ModuleNotFoundError as exc:
        return [Check("fiducial P1D", "FAIL", _missing_dependency_detail(exc))]
    except Exception as exc:
        return [Check("fiducial P1D", "FAIL", _broken_environment_detail(exc))]
    if not available:
        return [Check("fiducial P1D", "SKIP",
                      "lyaemu is not importable; put an InferenceLyaData clone "
                      "on PYTHONPATH to run this check")]

    kf = np.asarray(stored["kf"], float)
    theta = np.asarray(stored["theta"], float)
    rtol = float(stored.get("rtol", PREDICTION_RTOL))
    checks = []
    for fidelity in ("lf", "hf"):
        want = stored.get(fidelity)
        if not want:
            checks.append(Check(f"fiducial P1D ({fidelity})", "SKIP", "not stored"))
            continue
        try:
            zout, p1d = _predict(basedir, theta, kf, fidelity)
        except Exception as exc:
            checks.append(Check(f"fiducial P1D ({fidelity})", "FAIL",
                                f"{type(exc).__name__}: {exc}"))
            continue
        worst, worst_z = 0.0, None
        for z_label, values in want.items():
            got = p1d[_z_index(zout, float(z_label))]
            expect = np.asarray(values, float)
            dev = float(np.max(np.abs(got - expect) / np.abs(expect)))
            if dev > worst:
                worst, worst_z = dev, z_label
        if worst <= rtol:
            checks.append(Check(f"fiducial P1D ({fidelity})", "PASS",
                                f"max relative deviation {worst:.2e} at rtol {rtol:g}"))
        else:
            checks.append(Check(f"fiducial P1D ({fidelity})", "FAIL",
                                f"max relative deviation {worst:.2e} at z = {worst_z}, "
                                f"above rtol {rtol:g}"))
    return checks


def run_all(basedir, *, ref=None, predictions: bool = True) -> list[Check]:
    basedir = Path(basedir)
    if ref is None and (basedir / REFERENCE_FILE).exists():
        ref = load_reference(basedir)
    checks = check_files(basedir, ref)
    checks += check_grid(basedir)
    checks += check_trained_mf(basedir, ref)
    checks += check_res_corr(basedir)
    if predictions:
        checks += check_predictions(basedir, ref)
    return checks


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_report(basedir, checks, ref=None) -> str:
    """Summarise the basedir, then list the checks.

    The summary needs to read every file, so on a damaged basedir it is the
    part that cannot be produced. The checks have already run by then, and
    they are the reason the reader ran the script, so a failure to summarise
    must not discard them: we note why the summary is missing and print the
    checks regardless."""
    lines = [f"KODIAQ-SQUAD GP basedir: {Path(basedir).resolve()}", ""]
    try:
        lines += _summary_lines(basedir, ref)
    except Exception as exc:
        lines += [f"Summary unavailable: {type(exc).__name__}: {exc}",
                  "The checks below still ran; see them for what is wrong."]

    lines += ["", "Checks"]
    for c in checks:
        lines.append(f"  [{c.status:4s}] {c.name:22s} {c.detail}")

    n_fail = sum(1 for c in checks if c.status == "FAIL")
    n_skip = sum(1 for c in checks if c.status == "SKIP")
    summary = f"{len(checks) - n_fail - n_skip} passed, {n_fail} failed, {n_skip} skipped"
    lines += ["", summary]
    return "\n".join(lines)


def _summary_lines(basedir, ref=None) -> list:
    """The grid/design/trained-GP summary. Every number here is read from the
    files themselves, never from reference.json, so the summary cannot
    contradict the checks. The provenance line, if shown, is the exception: it
    echoes the environment recorded in reference.json when the predictions were
    emitted."""
    obs = observe(basedir)
    grid, design, trained = obs["grid"], obs["design"], obs["trained_mf"]
    lines = []

    lines.append("Grid")
    lines.append(f"  k bins                 {grid['nk']}")
    lines.append(f"  redshift bins          {grid['n_zbins']}")
    lines.append(f"  redshifts              {min(grid['zout'])} to {max(grid['zout'])}")
    lo, hi = grid["kfmpc_range"]
    lines.append(f"  k range (h/Mpc)        {lo:.5f} to {hi:.4f}")
    for key, label in (("kfkms_range_lf", "k range LF (s/km)"),
                       ("kfkms_range_hf", "k range HF (s/km)")):
        lo, hi = grid[key]
        lines.append(f"  {label:22s} {lo:.6f} to {hi:.5f}")

    lines += ["", "Design"]
    lines.append(f"  low-fidelity samples   {design['lf_samples']}")
    lines.append(f"  high-fidelity samples  {design['hf_samples']}")
    lines.append(f"  parameters             {design['n_params']}")
    names, lo_all, hi_all = design["param_names"], design["param_min"], design["param_max"]
    for i in range(min(len(names), len(lo_all), len(hi_all))):
        note = "  (mean flux scaling)" if names[i] == MEAN_FLUX_PARAM else ""
        lines.append(f"    {names[i]:12s} {lo_all[i]:12.6g} to {hi_all[i]:12.6g}{note}")

    lines += ["", "Trained GPs"]
    lines.append(f"  AR1 hyperparameters    {trained['n_ar1_files']} bins")
    lines.append(f"  low-fidelity GPs       {trained['n_zbins']} bins")
    lines.append(f"  X shape                {trained['x_shape']}")
    lines.append(f"  Y shape                {trained['y_shape']}")
    lines.append(f"  resolution correction  {obs.get('res_corr', {}).get('shape')}")

    prov = (ref or {}).get("provenance")
    if prov:
        lines += ["", "Reference emitted with"]
        lines.append(f"  lyaemu code md5        {prov.get('lyaemu_code_md5')}")
        lines.append(f"  python                 {prov.get('python')}")
        lines.append("  packages               " + ", ".join(
            f"{k} {prov[k]}" for k in ("numpy", "scipy", "h5py", "GPy", "emukit")
            if prov.get(k)))

    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify the KODIAQ-SQUAD GP basedir.")
    ap.add_argument("--basedir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--no-predictions", action="store_true",
                    help="skip the checks that need lyaemu")
    ap.add_argument("--strict", action="store_true",
                    help="treat skipped checks as failures")
    ap.add_argument("--emit-reference", action="store_true",
                    help="regenerate reference.json (maintainer use)")
    ap.add_argument("--force", action="store_true",
                    help="with --emit-reference, write even when checks fail")
    args = ap.parse_args(argv)
    basedir = Path(args.basedir)

    if args.strict and args.no_predictions:
        print("--strict and --no-predictions contradict each other: the "
              "prediction check is the one --strict exists to enforce.",
              file=sys.stderr)
        return 2

    if args.emit_reference:
        if args.no_predictions and not args.force:
            print("refusing to write a reference without predictions, which "
                  "would disable the strongest check for every later run. "
                  "Pass --force if that is really what you want.", file=sys.stderr)
            return 2
        if not args.no_predictions:
            try:
                have_lyaemu = _import_gpwrap() is not None
            except ModuleNotFoundError as exc:
                print(f"cannot emit predictions: "
                      f"{_missing_dependency_detail(exc)} Or pass "
                      f"--no-predictions --force.", file=sys.stderr)
                return 2
            except Exception as exc:
                print(f"cannot emit predictions: "
                      f"{_broken_environment_detail(exc)} Or pass "
                      f"--no-predictions --force.", file=sys.stderr)
                return 2
            if not have_lyaemu:
                print("cannot emit predictions: lyaemu is not importable. Put an "
                      "InferenceLyaData clone on PYTHONPATH, or pass "
                      "--no-predictions --force to emit without them.",
                      file=sys.stderr)
                return 2
        blocking = [c for c in run_all(basedir, predictions=False) if c.status == "FAIL"]
        if blocking and not args.force:
            print("refusing to regenerate reference.json: the basedir does not "
                  "pass its own checks, so the new reference would certify a "
                  "damaged copy.", file=sys.stderr)
            for c in blocking:
                print(f"  [FAIL] {c.name}: {c.detail}", file=sys.stderr)
            print("Pass --force to override.", file=sys.stderr)
            return 1
        ref = build_reference(basedir, predictions=not args.no_predictions)
        (basedir / REFERENCE_FILE).write_text(json.dumps(ref, indent=1) + "\n")
        print(f"wrote {basedir / REFERENCE_FILE}")
        return 0

    try:
        ref = load_reference(basedir)
    except FileNotFoundError:
        print(f"{REFERENCE_FILE} not found in {basedir}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"{REFERENCE_FILE} in {basedir} is not valid JSON: {exc}", file=sys.stderr)
        return 1
    if not isinstance(ref, dict):
        print(f"{REFERENCE_FILE} in {basedir} is not a JSON object.", file=sys.stderr)
        return 1

    checks = run_all(basedir, ref=ref, predictions=not args.no_predictions)
    print(format_report(basedir, checks, ref))
    n_fail = sum(1 for c in checks if c.status == "FAIL")
    n_skip = sum(1 for c in checks if c.status == "SKIP")
    if args.strict and n_skip:
        print(f"--strict: treating {n_skip} skipped check(s) as failures.")
    return 1 if (n_fail or (args.strict and n_skip)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
