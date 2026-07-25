# KODIAQ-SQUAD GP emulator basedir

This repository holds the trained multi-fidelity Gaussian-process emulator from
the PRIYA KODIAQ-SQUAD analysis (Ho et al. 2026, JCAP 07 (2026) 094,
arXiv:2509.18271).

The repository *is* a basedir, so we clone it and use the clone root directly,
with no assembly step. A nearly identical emulator is published elsewhere, and
we must not substitute it: see [Why the cut k range](#why-the-cut-k-range).

## Quickstart

```bash
git clone https://github.com/jibanCat/kodiaq-squad-gp
git clone https://github.com/jibanCat/InferenceLyaData
git -C InferenceLyaData checkout 253f9ed   # a known-good lyaemu commit (its code md5 is recorded in reference.json)

export PYTHONPATH=$PWD/InferenceLyaData     # the directory that contains lyaemu/, not lyaemu/ itself
pip install -r kodiaq-squad-gp/requirements.txt

cd kodiaq-squad-gp
python verify_basedir.py --strict
```

Run the `export` from the directory that holds both clones, before `cd`, and
point it at the `InferenceLyaData` root (the directory containing `lyaemu/`),
not at `lyaemu/` itself. `InferenceLyaData` supplies the `lyaemu` package, which
is not installable with pip, hence the `PYTHONPATH` entry. We recommend
`--strict`, which turns a skipped check into a failure: without `lyaemu` on the
path the strongest check quietly does not run.

The analysis code that consumes this emulator, released with Ho et al. (2026),
points its own `GP_BASEDIR` at the clone root. Nothing in this repository reads
that variable.

## Verifying a copy

```bash
python verify_basedir.py                  # file checksums, k-binning, trained GPs, fiducial P1D
python verify_basedir.py --no-predictions # the subset that does not need lyaemu
```

The script exits non-zero when a check fails, and prints the grid, the design
and the trained-GP shapes, so it doubles as the summary of what this basedir
contains. It reads those numbers from the files themselves rather than from
`reference.json`, and the values it compares against are hard-coded in the
script, because `reference.json` is part of what we are checking.

## Contents

| Path | Files | Size | Contents |
|---|---|---|---|
| `emulator_params.json` | 1 | 8.3 kB | Low-fidelity design and parameter limits |
| `mf_emulator_flux_vectors_tau1000000.hdf5` | 1 | 21.52 MB | LF flux vectors, 600 samples x 13 redshifts x 172 k-bins |
| `hires/emulator_params.json` | 1 | 1.3 kB | High-fidelity design |
| `hires/mf_emulator_flux_vectors_tau1000000.hdf5` | 1 | 1.09 MB | HF flux vectors, 30 samples x 13 x 172 |
| `trained_mf/zbin<z>.json` | 13 | 29.85 MB | Single-fidelity GPs on the LF design, `X = (600, 10)`, `Y = (600, 172)` |
| `trained_mf/zbin<z>` | 13 | 90.7 kB | AR1 multi-fidelity hyperparameters, one per redshift |
| `res_corr/resolution_correction.h5` | 1 | 16.3 kB | Resolution-correction table, 15 redshifts x 59 k-bins |
| `res_corr/resolution_correction.txt` | 1 | 22.1 kB | The same table as plain text |

32 data files, 52.59 MB in total. The largest single file is 21.5 MB, so Git LFS
is not required, and a clone transfers about 24 MiB once packed.

The two kinds of file in `trained_mf/` are both needed, and a reader debugging a
load failure should know which is which. The extension-less `zbin<z>` files hold
the AR1 hyperparameters that the high-fidelity path loads, and are the files
that differ between the cut and the uncut emulator. The `zbin<z>.json` files
hold the single-fidelity GP that the low-fidelity path loads. When either is
missing, `lyaemu` retrains on load and writes the result back into
`trained_mf/`, which does not reproduce the published values.

The emulator spans 13 redshift bins from z = 2.2 to z = 4.6 in steps of 0.2. We
ship all 13, so that a reader can change the redshift without rebuilding the
basedir. The bins are loaded as a contiguous range, so a partial set is not an
option.

The design carries ten columns. Nine of them are the physics parameters named in
`emulator_params.json` (`ns`, `Ap`, `herei`, `heref`, `alphaq`, `hub`,
`omegamh2`, `hireionz`, `bhfeedback`), while the leading column is the mean
optical-depth scaling, which upstream does not name.

`res_corr/` holds the resolution-convergence correction, the ratio of the flux
power from an L15n512 box to an L15n384 box, tabulated over 15 redshifts and 59
k-bins. The KODIAQ-SQUAD analysis multiplies the emulated P1D by this factor to
correct the residual resolution convergence of the training boxes. We ship it so
that the basedir is a complete record, although `lyaemu` loads its own copy from
the package rather than from here, so the table is not on the emulator load path.

## Provenance

The flux vectors and the parameter files are byte-identical to files that are
already public in
[`jibanCat/InferenceLyaData`](https://github.com/jibanCat/InferenceLyaData),
although two of them are published there under a different name:

| File here | Byte-identical source in `InferenceLyaData/Emulator_Files_KS/` |
|---|---|
| `emulator_params.json` | `emulator_params.json` |
| `hires/emulator_params.json` | `hires/emulator_params.json` |
| `mf_emulator_flux_vectors_tau1000000.hdf5` | `mf_emulator_flux_vectors_tau1000000_cut.hdf5` |
| `hires/mf_emulator_flux_vectors_tau1000000.hdf5` | `hires/mf_emulator_flux_vectors_tau1000000_cut.hdf5` |
| `res_corr/resolution_correction.h5` | `resolution_correction.h5` |
| `res_corr/resolution_correction.txt` | `res_corr/resolution_correction.txt` |
| `trained_mf/` | published, but fitted to the uncut vectors, with `Y = (600, 329)` |

The trained GPs are the reason this repository exists. We fit ours to the cut
flux vectors, so they have `Y = (600, 172)`, while the GPs published in
`InferenceLyaData` are fitted to the uncut vectors and have `Y = (600, 329)`.
Shipping ours means that a reader can reproduce the published numbers without
retraining, which matters because the multi-fidelity AR1 fit uses ten optimiser
restarts with no fixed seed and is not reproducible.

We stripped the basedir from the internal run directory
`kodiaq_2_2_4_6-48-48`, dropping the leave-one-out diagnostics, the
temperature-emulator files, the seed-convergence file, the auxiliary `kims_`
subdirectories, and the alternative HDF5 variants that the forecast never reads.
The two `emulator_params.json` files are carried over verbatim from the previous
emulator version, so their `kf`, `maxk`, and `basedir` fields describe that
version rather than the shipped flux vectors; only `param_names`,
`param_limits`, and `sample_params` bear on this basedir, and the k-binning
comes from the HDF5 files. We preserve them unedited so that the files stay
byte-identical to upstream, including the stale TACC scratch paths in the
`basedir` fields.

## Why the cut k range

This basedir carries the k-cut flux vectors. The emulator covers k = 0.052 to
9.006 h/Mpc, in 172 bins spaced by the box fundamental
k_F = 2 pi / (120 Mpc/h) = 0.052 h/Mpc; the uncut vectors instead run to
17.23 h/Mpc in 329 bins. The simulation grid is in h/Mpc while the P1D is
measured in s/km, related by k [h/Mpc] = velfac(z) x k [s/km], where the
conversion factor velfac runs from roughly 100 to 130 km/s per Mpc/h across
z = 2.2 to 4.6, and is about 120 at z = 3.6. The 9.006 h/Mpc cut is therefore
k = 0.075 s/km at z = 3.6.

In Ho et al. (2026) we use the multi-fidelity emulator only up to k = 0.065 s/km,
which is about 7.8 h/Mpc at z = 3.6. Beyond that scale the residual
resolution-convergence error of the simulations exceeds the KODIAQ-SQUAD
statistical uncertainty (the diagonal of its data covariance), so we do not use
the emulator there. The resolution-convergence correction is tabulated in
`res_corr/resolution_correction.h5`.

An emulator built from the uncut vectors loads without any warning but is a
different emulator: at z = 3.6, over k = 0.001 to 0.04 s/km, its high-fidelity
P1D differs from this one by 2.5 per cent in the median. `verify_basedir.py`
therefore tests the k-binning directly, and names the uncut variant when it
finds 329 bins. A basedir pointed at `InferenceLyaData/Emulator_Files_KS` is
exactly this mistake, so we point `GP_BASEDIR` at a clone of this repository and
never at that directory.

## Consistency with the previous emulator version

We compare against [`mafern/InferenceLyaData`](https://github.com/mafern/InferenceLyaData),
the previous version of this emulator, trained on an earlier PRIYA suite of 60
low-fidelity and 3 high-fidelity simulations and binned differently (102 k-bins
in h/Mpc, queried on the 35-bin eBOSS grid in s/km). This checks that the
emulator is stable across the update, not that two independent emulators agree.

We predict the fiducial high-fidelity P1D from both over the eBOSS k range
(k = 0.001 to 0.0195 s/km, the range the measurement covers) at all 13
redshifts. Away from the largest mode the two agree to better than 1.5 per cent
at every redshift (worst 1.5 per cent at z = 4.4), and to about one per cent or
better in the median. At the largest mode, the lowest-k bin at k = 0.001 s/km,
the difference reaches 2 per cent at z = 2.8; that mode carries a cosmic
variance of order 2 per cent (Fernandez et al. 2024, JCAP 07 (2024) 029,
[arXiv:2309.03943](https://arxiv.org/abs/2309.03943)), so a difference of that
size there is expected and is not an emulator disagreement.

For the three redshifts the paper reports, away from the largest mode:

| z | median | worst |
|---|---|---|
| 2.6 | 0.05 % | 0.2 % |
| 3.6 | 0.60 % | 0.7 % |
| 4.2 | 0.85 % | 1.3 % |

`validate_cross_emulator.py` reproduces the full-redshift comparison given a
clone of that repository, and gates on the agreement away from the largest mode.

## Reference values

`reference.json` records the file manifest with sizes and MD5 sums, the grid,
the design ranges, the trained-GP shapes, and the fiducial P1D at z = 2.6, 3.6
and 4.2 for both fidelities on a 48-point logarithmic grid spanning k = 0.001 to
0.04 s/km. The prediction check compares against those values at a relative
tolerance of 10^-6, since prediction from a loaded GP is deterministic and the
only expected spread comes from the platform linear algebra.

`test_verify_basedir.py` covers the checker itself, including a basedir
assembled from the uncut files and one whose AR1 hyperparameters have been
swapped.

## Citation

When you use this emulator, please cite the paper it comes from:

> Ho, Qezlou, Bird, Yang, Avestruz, Fernandez & Iršič,
> *Small-scale Lyman alpha forest cosmology with PRIYA: Constraints from XQ100
> and KODIAQ-SQUAD one-dimensional flux power spectra*,
> JCAP 07 (2026) 094,
> [doi:10.1088/1475-7516/2026/07/094](https://iopscience.iop.org/article/10.1088/1475-7516/2026/07/094),
> [arXiv:2509.18271](https://arxiv.org/abs/2509.18271),

together with the PRIYA simulation suite on which the emulator is trained:

> Bird, Fernandez, Ho, Qezlou, Monadi, Ni, Chen, Croft & Di Matteo,
> *PRIYA: a new suite of Lyman-alpha forest simulations for cosmology*,
> JCAP 10 (2023) 037, [arXiv:2306.05471](https://arxiv.org/abs/2306.05471).
