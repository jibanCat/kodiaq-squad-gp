"""Tests for verify_basedir.py.

The positive cases run against the shipped basedir (this repository). The
negative cases build small synthetic basedirs in a temporary directory, so a
regression in the failure paths is caught without a 43 MB copy. Where a
negative case needs genuinely different emulator data, we borrow the uncut
files from an InferenceLyaData Emulator_Files_KS directory named by the
KODIAQ_UNCUT_BASEDIR environment variable, and skip when it is not set.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

import verify_basedir as vb

REPO = Path(__file__).resolve().parent
# An InferenceLyaData Emulator_Files_KS directory, used only by the two tests
# that need real uncut data. Point KODIAQ_UNCUT_BASEDIR at a local clone to run
# them; they skip otherwise. No path is hard-coded, so this file leaks nothing.
_uncut_env = os.environ.get("KODIAQ_UNCUT_BASEDIR")
UNCUT = Path(_uncut_env) if _uncut_env else None
needs_uncut = pytest.mark.skipif(
    UNCUT is None or not UNCUT.is_dir(),
    reason="set KODIAQ_UNCUT_BASEDIR to an InferenceLyaData Emulator_Files_KS dir")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_flux_file(path, *, n_samples, nk, n_z=13):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f["flux_vectors"] = rng.random((n_samples, n_z * nk))
        f["kfkms"] = rng.random((n_samples, n_z, nk))
        f["kfmpc"] = np.linspace(0.05236, 9.0059, nk)
        f["params"] = rng.random((n_samples, 10))
        f["zout"] = np.linspace(4.6, 2.2, n_z)


def _write_trained_mf(basedir, *, nk, n_zbins=13, n_rows=6, ar1=True):
    d = basedir / "trained_mf"
    d.mkdir(parents=True, exist_ok=True)
    for z in np.round(np.linspace(2.2, 4.6, n_zbins), 1):
        (d / f"zbin{z}.json").write_text(json.dumps(
            {"name": "gp", "X": np.zeros((n_rows, 10)).tolist(),
             "Y": np.zeros((n_rows, nk)).tolist()}))
        if ar1:
            with h5py.File(d / f"zbin{z}", "w") as f:
                f["param_array"] = np.zeros(27)


def _write_res_corr(basedir, *, shape=vb.RES_CORR_SHAPE):
    p = basedir / vb.RES_CORR
    p.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(p, "w") as f:
        f["res_corr"] = np.ones(shape)
        f["kfkms"] = np.ones(shape)
        f["zout"] = np.linspace(4.6, 2.0, shape[0])
    (basedir / "res_corr" / "resolution_correction.txt").write_text("1.0\n")


def synthetic_basedir(tmp_path, *, nk=vb.EXPECTED_NK, ar1=True):
    base = tmp_path / "basedir"
    base.mkdir()
    _write_flux_file(base / vb.LF_FLUX, n_samples=600, nk=nk)
    _write_flux_file(base / vb.HF_FLUX, n_samples=30, nk=nk)
    for p in (base / "emulator_params.json", base / "hires" / "emulator_params.json"):
        p.write_text(json.dumps({"param_names": {"ns": 0}, "max_z": 4.6, "min_z": 2.2}))
    _write_trained_mf(base, nk=nk, ar1=ar1)
    _write_res_corr(base)
    return base


def pin_to(monkeypatch, basedir):
    """Point the hard-coded ground truth at a synthetic basedir."""
    manifest = vb.manifest_of(basedir)
    monkeypatch.setattr(vb, "EXPECTED_N_FILES", len(manifest))
    monkeypatch.setattr(vb, "MANIFEST_DIGEST", vb.manifest_digest(manifest))
    for attr, rel in (("KFMPC_DIGEST_LF", vb.LF_FLUX), ("KFMPC_DIGEST_HF", vb.HF_FLUX)):
        with h5py.File(Path(basedir) / rel, "r") as f:
            import hashlib
            monkeypatch.setattr(vb, attr, hashlib.md5(f["kfmpc"][()].tobytes()).hexdigest())


def failed(checks):
    return [c for c in checks if c.status == "FAIL"]


def details(checks):
    return " ".join(c.detail for c in checks).lower()


# ---------------------------------------------------------------------------
# reference.json
# ---------------------------------------------------------------------------


def test_build_reference_has_the_documented_sections(tmp_path):
    ref = vb.build_reference(synthetic_basedir(tmp_path), predictions=False)
    for key in ("grid", "design", "trained_mf", "files"):
        assert key in ref
    assert ref["grid"]["nk"] == vb.EXPECTED_NK
    assert ref["files"]


def test_reference_json_is_shipped_and_loadable():
    ref = vb.load_reference(REPO)
    assert ref["grid"]["nk"] == vb.EXPECTED_NK
    assert ref["trained_mf"]["y_shape"] == [600, vb.EXPECTED_NK]


def test_shipped_reference_manifest_matches_the_hard_coded_digest():
    """reference.json and the constants in the script must agree, otherwise one
    of the two was regenerated without the other."""
    ref = vb.load_reference(REPO)
    assert len(ref["files"]) == vb.EXPECTED_N_FILES
    assert vb.manifest_digest(ref["files"]) == vb.MANIFEST_DIGEST


# ---------------------------------------------------------------------------
# Positive path
# ---------------------------------------------------------------------------


def test_shipped_basedir_passes_every_offline_check():
    assert failed(vb.run_all(REPO, predictions=False)) == []


def test_design_ranges_sit_inside_the_declared_parameter_limits():
    """Guards the column alignment: the design has ten columns while
    emulator_params.json names nine, so an off-by-one silently reports each
    parameter's range against its neighbour."""
    design = vb.observe(REPO)["design"]
    upstream = json.loads((REPO / "emulator_params.json").read_text())
    for name, idx in upstream["param_names"].items():
        col = design["param_names"].index(name)
        lo, hi = upstream["param_limits"][idx]
        assert lo <= design["param_min"][col] <= design["param_max"][col] <= hi, name


def test_mean_flux_column_is_named():
    design = vb.observe(REPO)["design"]
    assert len(design["param_names"]) == design["n_params"]
    assert design["param_names"][0] == vb.MEAN_FLUX_PARAM


def test_shipped_basedir_has_both_kinds_of_trained_file():
    trained = vb.observe(REPO)["trained_mf"]
    assert trained["n_ar1_files"] == 13
    assert trained["n_zbins"] == 13
    assert trained["y_shape"] == [600, 172]


def test_shipped_basedir_has_the_resolution_correction_table():
    assert vb.observe(REPO)["res_corr"]["shape"] == list(vb.RES_CORR_SHAPE)
    assert failed(vb.check_res_corr(REPO)) == []


def test_missing_resolution_correction_is_reported(tmp_path):
    base = synthetic_basedir(tmp_path)
    (base / vb.RES_CORR).unlink()
    assert failed(vb.check_res_corr(base))


def test_wrong_shape_resolution_correction_is_reported(tmp_path):
    base = synthetic_basedir(tmp_path)
    _write_res_corr(base, shape=(13, 172))
    assert failed(vb.check_res_corr(base))


# ---------------------------------------------------------------------------
# The tripwire
# ---------------------------------------------------------------------------


def test_uncut_variant_fails_and_is_named(tmp_path):
    base = synthetic_basedir(tmp_path, nk=vb.UNCUT_NK)
    bad = failed(vb.check_grid(base))
    assert bad
    text = details(bad)
    assert "329" in text and "172" in text and "uncut" in text and "_cut" in text


def test_grid_check_needs_no_reference_file(tmp_path):
    base = synthetic_basedir(tmp_path, nk=vb.UNCUT_NK)
    assert not (base / vb.REFERENCE_FILE).exists()
    assert failed(vb.check_grid(base))


def test_right_bin_count_with_wrong_bin_values_fails(tmp_path, monkeypatch):
    """A file resampled from the uncut grid to 172 bins keeps the bin count but
    not the bin values."""
    base = synthetic_basedir(tmp_path)
    pin_to(monkeypatch, base)
    with h5py.File(base / vb.LF_FLUX, "r+") as f:
        del f["kfmpc"]
        f["kfmpc"] = np.linspace(0.05236, 17.2264, vb.EXPECTED_NK)
    bad = failed(vb.check_grid(base))
    assert bad and "k values differ" in details(bad)


# ---------------------------------------------------------------------------
# The AR1 hyperparameter files: the ones that actually carry the difference
# ---------------------------------------------------------------------------


def test_missing_ar1_files_fail(tmp_path, monkeypatch):
    base = synthetic_basedir(tmp_path, ar1=False)
    pin_to(monkeypatch, base)
    bad = failed(vb.check_trained_mf(base))
    assert bad and "retrains" in details(bad)


@needs_uncut
def test_uncut_ar1_hyperparameters_are_rejected(tmp_path):
    """The extension-less zbin<z> files carry the AR1 hyperparameters. Swapping
    in the uncut ones changes the high-fidelity P1D by about 2.5 per cent while
    every shape stays correct, so only the content check can catch it."""
    base = tmp_path / "swapped"
    shutil.copytree(REPO, base, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    for src in sorted(UNCUT.glob("trained_mf/zbin*")):
        if not src.name.endswith(".json"):
            shutil.copy2(src, base / "trained_mf" / src.name)
    bad = failed(vb.run_all(base, predictions=False))
    assert bad, "swapped AR1 hyperparameters must be rejected"
    assert any(c.name == "file contents" for c in bad)


@needs_uncut
def test_emit_reference_refuses_to_bless_a_damaged_basedir(tmp_path):
    base = tmp_path / "swapped"
    shutil.copytree(REPO, base, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    for src in sorted(UNCUT.glob("trained_mf/zbin*")):
        if not src.name.endswith(".json"):
            shutil.copy2(src, base / "trained_mf" / src.name)
    before = (base / vb.REFERENCE_FILE).read_text()
    rc = vb.main(["--basedir", str(base), "--emit-reference", "--no-predictions"])
    assert rc != 0
    assert (base / vb.REFERENCE_FILE).read_text() == before, "reference was rewritten"


# ---------------------------------------------------------------------------
# reference.json must not be able to certify itself
# ---------------------------------------------------------------------------


def test_emptied_reference_neither_passes_nor_fails_a_good_basedir(tmp_path):
    """An emptied reference.json used to make check_files vacuously pass. The
    verdict must now come from the basedir itself, so a correct copy still
    passes and a damaged one still fails, whatever reference.json says."""
    assert failed(vb.check_files(REPO, {"files": {}})) == []
    base = synthetic_basedir(tmp_path)          # not the shipped basedir
    assert failed(vb.check_files(base, {"files": {}})), "damage must still be caught"


def test_hard_coded_digest_rejects_a_regenerated_reference(tmp_path, monkeypatch):
    """Regenerating reference.json on a modified basedir must not make it pass,
    because the digest lives in the script."""
    base = synthetic_basedir(tmp_path)
    ref = vb.build_reference(base, predictions=False)
    assert failed(vb.check_files(base, ref)), "synthetic dir must not match the shipped digest"


def test_extra_file_is_reported(tmp_path, monkeypatch):
    base = synthetic_basedir(tmp_path)
    pin_to(monkeypatch, base)
    assert failed(vb.check_files(base, None)) == []
    (base / "trained_mf" / "zbin9.9.json").write_text("{}")
    bad = failed(vb.check_files(base, None))
    assert bad and "expected" in details(bad)


def test_missing_file_is_reported(tmp_path, monkeypatch):
    base = synthetic_basedir(tmp_path)
    pin_to(monkeypatch, base)
    (base / "emulator_params.json").unlink()
    assert failed(vb.check_files(base, None))


def test_corrupted_file_is_reported(tmp_path, monkeypatch):
    base = synthetic_basedir(tmp_path)
    pin_to(monkeypatch, base)
    (base / "emulator_params.json").write_text('{"param_names": {"ns": 0}, "x": 1}')
    assert failed(vb.check_files(base, None))


def test_editor_checkpoint_dirs_are_ignored(tmp_path, monkeypatch):
    """A Jupyter session can drop .ipynb_checkpoints/ inside the basedir; the
    walk must ignore it, or a correct copy fails the file inventory."""
    base = synthetic_basedir(tmp_path)
    pin_to(monkeypatch, base)
    assert failed(vb.check_files(base, None)) == []
    ckpt = base / "trained_mf" / ".ipynb_checkpoints"
    ckpt.mkdir()
    (ckpt / "zbin2.2-checkpoint.json").write_text("{}")
    assert base / ".ipynb_checkpoints" not in vb._data_files(base)
    assert failed(vb.check_files(base, None)) == [], "checkpoint must not fail a good copy"


def test_symlinked_trained_mf_is_still_enumerated(tmp_path, monkeypatch):
    """Keeping the 30 MB of trained GPs on shared storage is a normal move, and
    Path.rglob does not descend a symlinked directory."""
    base = synthetic_basedir(tmp_path)
    moved = tmp_path / "elsewhere"
    n_before = len(vb._data_files(base))
    shutil.move(str(base / "trained_mf"), str(moved))
    (base / "trained_mf").symlink_to(moved)
    assert len(vb._data_files(base)) == n_before


# ---------------------------------------------------------------------------
# Predictions and the CLI
# ---------------------------------------------------------------------------


def test_predictions_skip_when_lyaemu_absent(tmp_path, monkeypatch):
    base = synthetic_basedir(tmp_path)
    ref = vb.build_reference(base, predictions=False)
    ref["predictions"] = {"theta": [], "kf": [], "lf": {}, "hf": {}}
    monkeypatch.setattr(vb, "_import_gpwrap", lambda: None)
    assert all(c.status == "SKIP" for c in vb.check_predictions(base, ref))


def test_predictions_skip_when_reference_has_none(tmp_path, monkeypatch):
    base = synthetic_basedir(tmp_path)
    ref = vb.build_reference(base, predictions=False)
    monkeypatch.setattr(vb, "_import_gpwrap", lambda: object())
    assert all(c.status == "SKIP" for c in vb.check_predictions(base, ref))


def test_strict_and_no_predictions_is_rejected():
    assert vb.main(["--basedir", str(REPO), "--strict", "--no-predictions"]) == 2


def test_emit_reference_without_predictions_is_rejected(tmp_path):
    base = synthetic_basedir(tmp_path)
    assert vb.main(["--basedir", str(base), "--emit-reference", "--no-predictions"]) == 2


def test_corrupt_reference_json_is_reported_cleanly(tmp_path):
    base = synthetic_basedir(tmp_path)
    (base / vb.REFERENCE_FILE).write_text("not json")
    assert vb.main(["--basedir", str(base), "--no-predictions"]) == 1


def test_non_dict_reference_json_is_reported_cleanly(tmp_path):
    base = synthetic_basedir(tmp_path)
    (base / vb.REFERENCE_FILE).write_text('"a string, not an object"')
    assert vb.main(["--basedir", str(base), "--no-predictions"]) == 1


def _patch_lyaemu_import(monkeypatch, missing_name):
    """Make `from lyaemu.gp_wrap import ...` raise ModuleNotFoundError naming
    `missing_name`, without needing a real lyaemu on the path."""
    import builtins
    import sys
    real = builtins.__import__
    for m in [m for m in list(sys.modules) if m.startswith("lyaemu")]:
        monkeypatch.delitem(sys.modules, m, raising=False)

    def fake(name, *a, **k):
        if name.split(".")[0] == "lyaemu":
            raise ModuleNotFoundError(f"No module named {missing_name!r}", name=missing_name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)


def test_import_gpwrap_returns_none_when_lyaemu_itself_missing(monkeypatch):
    _patch_lyaemu_import(monkeypatch, "lyaemu")
    assert vb._import_gpwrap() is None


def test_import_gpwrap_reraises_a_broken_dependency(monkeypatch):
    """A missing dependency of lyaemu (e.g. matplotlib) is a broken environment
    and must not be misreported as a missing clone."""
    _patch_lyaemu_import(monkeypatch, "matplotlib")
    with pytest.raises(ModuleNotFoundError):
        vb._import_gpwrap()


def test_predictions_fail_cleanly_on_broken_dependency(tmp_path, monkeypatch):
    base = synthetic_basedir(tmp_path)
    ref = vb.build_reference(base, predictions=False)
    ref["predictions"] = {"theta": [], "kf": [], "lf": {}, "hf": {}}

    def boom():
        raise ModuleNotFoundError("No module named 'matplotlib'", name="matplotlib")
    monkeypatch.setattr(vb, "_import_gpwrap", boom)
    checks = vb.check_predictions(base, ref)
    assert failed(checks) and "matplotlib" in details(checks)


def test_malformed_flux_file_fails_without_crashing(tmp_path, monkeypatch):
    base = synthetic_basedir(tmp_path)
    pin_to(monkeypatch, base)
    with h5py.File(base / vb.LF_FLUX, "r+") as f:
        del f["flux_vectors"]
        f["flux_vectors"] = np.zeros(5)          # 1-D, not (n_s, n_flat)
    bad = failed(vb.check_grid(base))
    assert any("malformed" in c.detail for c in bad)


def test_malformed_trained_gp_json_fails_without_crashing(tmp_path, monkeypatch):
    base = synthetic_basedir(tmp_path)
    pin_to(monkeypatch, base)
    (base / "trained_mf" / "zbin3.6.json").write_text('{"no_Y_key": true}')
    bad = failed(vb.check_trained_mf(base))
    assert any("malformed" in c.detail for c in bad)


def test_prediction_pass_and_fail_paths(tmp_path, monkeypatch):
    """Exercise the numeric body of check_predictions (previously untested)."""
    base = synthetic_basedir(tmp_path)
    kf = [0.001, 0.01, 0.04]
    ref = {"predictions": {"theta": [0.0], "kf": kf, "rtol": 1e-6,
                           "lf": {"3.6": [1.0, 2.0, 3.0]},
                           "hf": {"3.6": [1.0, 2.0, 3.0]}}}
    monkeypatch.setattr(vb, "_import_gpwrap", lambda: object())

    def exact(basedir, theta, kf, fidelity):
        return np.array([3.6]), np.array([[1.0, 2.0, 3.0]])
    monkeypatch.setattr(vb, "_predict", exact)
    assert failed(vb.check_predictions(base, ref)) == []          # PASS path

    def off(basedir, theta, kf, fidelity):
        return np.array([3.6]), np.array([[1.0, 2.0, 3.3]])       # 10% off
    monkeypatch.setattr(vb, "_predict", off)
    assert failed(vb.check_predictions(base, ref))                # FAIL path


def test_format_report_shows_the_key_numerics():
    text = vb.format_report(REPO, vb.run_all(REPO, predictions=False))
    for token in ("172", "600", "13"):
        assert token in text


def test_cli_exits_zero_on_the_shipped_basedir():
    r = subprocess.run([sys.executable, "verify_basedir.py", "--no-predictions"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_cli_exits_nonzero_on_a_broken_basedir(tmp_path):
    base = synthetic_basedir(tmp_path, nk=vb.UNCUT_NK)
    (base / vb.REFERENCE_FILE).write_text(
        json.dumps(vb.build_reference(base, predictions=False)))
    r = subprocess.run([sys.executable, str(REPO / "verify_basedir.py"),
                        "--basedir", str(base), "--no-predictions"],
                       capture_output=True, text=True)
    assert r.returncode != 0


# ---------------------------------------------------------------------------
# The cross-emulator validation, exercised without touching lyaemu
# ---------------------------------------------------------------------------


def test_cross_validation_skips_cleanly_without_mafern():
    import validate_cross_emulator as cx
    rc = cx.main(["--basedir", str(REPO), "--mafern", "/nonexistent"])
    assert rc == 2


def test_cross_validation_reports_disagreement(monkeypatch):
    import validate_cross_emulator as cx
    monkeypatch.setattr(cx.vb, "_import_gpwrap", lambda: object())
    monkeypatch.setattr(cx.Path, "is_file", lambda self: True)
    monkeypatch.setattr(cx, "eboss_kf", lambda base: np.linspace(0.001, 0.019, 12))

    def fake_predict(basedir, theta, kf, fidelity):
        zout = np.array([4.2, 3.6, 2.6])
        base = np.ones((3, len(kf)))
        if "mafern" in str(basedir):
            base = base * 1.10          # a flat ten-percent offset, all modes
        return zout, base
    monkeypatch.setattr(cx.vb, "_predict", fake_predict)
    # the gap is flat, so even away from the largest mode it exceeds tol -> fail
    assert cx.main(["--basedir", "this", "--mafern", "mafern"]) == 1
