# Release re-imputation of base-owned inputs (microcosm#1191 follow-up)

The #1191 PUF-channel wage audit found that the US release build imputes the
PUF-half retirement contributions a second time and does not reproduce the base
frame. This folder holds the scripts and receipts of the follow-up: which
release-time re-runs of base stages change stored values, why, and by how much.
Nothing here changes a build. The tests are
`packages/microcosm-build/tests/engine_free/us/test_us_release_base_input_invariant.py`
and `test_us_release_base_input_properties.py`.

## Files measured

| File | sha256 | Built |
|---|---|---|
| `base_populace_us_2024_puf_support.h5` | `0fb05b674899114e66dc01c2df0ff5349de88bbc8fed440755f4b19f287cedd2` | `tools/build_us_puf_support_base.py` at `4b57d15a287c`, on Modal |
| `populace_us_2024.h5`, release `populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z` | `417d23aed4d044de2e657a7b5d856136975d25739dbdb4ddd72866f841de1559` | `tools/build_us_fiscal_refresh_release.py --base-h5` from that base at the same commit, locally |
| that release's `calibration_diagnostics.json` | `64b55a02aaf573964cffee44d4a86bd63f33dc30c7f5012e4345348eecc06f80` | records the release's population and Social Security repair factors |

The builds' environments come from their run records
(`build_environment.base_0fb05b67.release_417d23ae.json`, each field tied to
the sha256 of the record it came from). The base: the Modal receipt's runner
is CPython 3.14.2 on `Linux-4.19.0-gvisor-x86_64-with-glibc2.36` with uv.lock
`52e3d128…`, and the stage run context records the builder as CPython 3.14.2
(GCC 12.2.0) with numpy 2.4.6, pandas 3.0.3, scikit-learn 1.8.0 and
quantile-forest 1.4.2. The release: its build manifest records Python 3.14.7,
numpy 2.4.6 and pandas 3.0.3 at the same commit, unmodified; the release
command's interpreter is a venv on `cpython-3.14+freethreaded-macos-aarch64`
3.14.7 that holds scikit-learn 1.8.0 and quantile-forest 1.4.2 (read on
2026-10-13); and the run log opens with the free-threaded interpreter's
warning that it enabled the GIL to load quantile-forest. The release's `pool_frame is None` stage calls are the
same at `4b57d15a287c` and at main `7639ef8b0`, and the four stage modules,
`microcosm-fit`, `source_runtime.py` and `us/source_stages.json` have no diff
between the two.

## Result

**What the release re-runs.** With `--base-h5` and no validated pool frame,
`_main` makes 17 stage calls under `if pool_frame is None`. Fourteen re-run a
stage the base builder also calls: weeks unemployed, QBI reconciliation,
childcare, energy subsidy, retirement contributions, immigration,
relationship, SPM independence role, Medicare take-up, retirement
distributions, eligibility, education, pregnancy and WIC. The other three are
the population mass repair and two release-only stages (take-up, hours
worked). `_main` calls no other base-builder stage. The Social Security and
non-Schedule-D capital-gain value repairs run between the first re-run (weeks
unemployed) and the rest.

**Why they re-run.** Childcare and energy subsidy skip only when the raw ASEC
source column is absent (`allow_existing_without_source`), retirement
contributions only when the frame has no support roles, and weeks unemployed
never. The base H5 carries support roles and the raw columns
(`SPM_CHILDCAREXPNS`, `SPM_ENGVAL`, `RETCB_VAL`, `LKWEEKS`), so each stage
recomputes its PUF-half draw.

**What changed** (`column_diff…json`: every column of every entity, rows
matched by id; both files hold the same ids, unique, with none unmatched; the
PUF half is 475,340 persons and 191,267 SPM units):

| Stage | Column | PUF rows changed | ASEC rows changed |
|---|---|---:|---:|
| retirement contributions | `traditional_401k_contributions_desired` | 30,442 persons | 0 |
| | `roth_401k_contributions_desired` | 28,695 | 0 |
| | `traditional_ira_contributions_desired` | 28,913 | 0 |
| | `roth_ira_contributions_desired` | 29,223 | 0 |
| | `self_employed_pension_contributions_desired` | 1,219 | 0 |
| energy subsidy | `spm_unit_energy_subsidy` | 3,283 SPM units | 0 |
| childcare | `spm_unit_pre_subsidy_childcare_expenses` | 1,849 SPM units | 0 |
| weeks unemployed | `weeks_unemployed` | 9 persons | 0 |

The other ten re-runs leave every stored column unchanged. Beyond the re-runs,
the release's value repairs rescale the four Social Security component leaves
and `non_sch_d_capital_gains` by constant factors, calibration changes the
household weights, and the release-only other-health-insurance stage replaces
`health_insurance_premiums_without_medicare_part_b` on 225,373 PUF persons
(its module docstring: it "replaces only the PUF support channel").

**Why** (`replay.*.json`). The replay imports the release builder from the
commit that built the release, loads the base with the builder's
`_load_frame`, and runs the stages itself on macOS arm64:

1. Replaying the release's order (weeks unemployed, mass repair, Social
   Security repair fed the four recorded SSA targets, QBI reconciliation,
   childcare, energy subsidy, retirement contributions) reproduces the
   release's stored values exactly for all four stages: 0 rows differ.
2. Applying only the recorded Social Security factors (1.1475 retirement,
   1.1104 disability, 2.4926 survivors, 0.9056 dependents) to the base and
   re-running childcare, energy subsidy and retirement contributions also
   reproduces the release exactly. `social_security`, the sum of the four
   leaves, is one of the eight predictors of all three QRFs. A tree splits on
   a predictor's order; one common factor would keep that order, and these
   factors change it. Ordering base persons with Social Security by the sum as
   the stages compute it, 1,272 of the 8,777 adjacent ASEC pairs with distinct
   sums swap order after the repair, and 12,745 of the 63,965 tied adjacent
   ASEC pairs stop being tied; among PUF persons, 6,484 of 38,079 and 17,809
   of 36,686 (`social_security_reorder…json`).
3. The population mass repair alone (factor 1.0253 on every household weight)
   changes nothing, and neither does limiting BLAS/OpenMP pools, forest fit
   jobs and draw workers to four threads: each reproduces the re-run on the
   unchanged base bit for bit on all five contribution columns
   (`platform_replay.macos_arm64.{mass_factor,threads4}.json`).
4. Re-running on the unchanged base on macOS does not reproduce the Modal-built
   base: retirement contributions differ on 21,765 PUF persons for the
   traditional 401(k) leaf (21,484 by more than $1, up to $18,523.20, no zero
   or nonzero flips), childcare on 54 and energy subsidy on 17 SPM units
   (below 5.1e-11 and 1.2e-12), weeks unemployed on 9 persons (1 to 10 weeks).
   Weeks unemployed is the release's first re-run, before any repair. As
   aggregate consistency evidence, the base H5's five contribution leaves,
   weighted by its stored household weights, reproduce exactly the totals the
   base stage recorded when it finished (`base_stage_record_check…json`);
   five sums cannot show that every row is the stage's own.
5. Given identical inputs, the stages give different outputs on the two
   platforms (`platform_replay.*.json`). Each stage's input table was exported
   from the base and replayed with the production stage code on linux/amd64
   (CPython 3.14.2 and the base build's uv.lock wheels, under QEMU) and on
   macOS arm64. Each exported table keeps only the columns its stage reads and
   gives the full table's output bit for bit
   (`platform_export_check.retirement.macos_arm64.json`,
   `platform_other_stages.export.macos_arm64.json`). Rows that differ from the
   Modal-built base:

   | Stage | linux/amd64 | macOS arm64 |
   |---|---|---|
   | weeks unemployed | 0 of 907,863 persons | 9 persons, 1 to 10 weeks |
   | childcare | 15 of 367,306 SPM units, below 5.1e-11 | 54, below 5.1e-11 |
   | energy subsidy | 7 SPM units, below 1.2e-12 | 17, below 1.2e-12 |
   | retirement contributions (traditional 401(k)) | 235 persons, below 7e-11 | 21,765 persons, 21,484 by more than $1 |

   The other four contribution leaves differ on 15, 5, 11 and 1 persons on
   Linux, below 1.6e-12. The two platforms' retirement outputs differ from
   each other on 21,764 persons
   (`platform_outputs.linux_amd64_vs_macos_arm64.json`). quantile-forest 1.4.2
   has no cp314 Linux aarch64 wheel, so architecture and operating system were
   not separated.
6. **Which cause moved which row** (`release_change_decomposition…json`). Each
   changed row is classed by whether the macOS replay of the unchanged inputs
   already equals the release (platform only), equals the base (repair only),
   or neither (both):

   | Stage | Changed | Platform only | Repair only | Both |
   |---|---:|---:|---:|---:|
   | traditional 401(k) | 30,442 | 5,314 | 13,013 | 12,115 |
   | Roth 401(k) | 28,695 | 6,311 | 13,927 | 8,457 |
   | traditional IRA | 28,913 | 6,416 | 14,187 | 8,310 |
   | Roth IRA | 29,223 | 6,517 | 14,290 | 8,416 |
   | self-employed pension | 1,219 | 233 | 810 | 176 |
   | childcare | 1,849 | 48 | 1,795 | 6 |
   | energy subsidy | 3,283 | 0 | 3,266 | 17 |
   | weeks unemployed | 9 | 9 | 0 | 0 |

   The 48 platform-only childcare rows move by less than 5.1e-11. The
   repaired replay ran on macOS only, so an interaction between the platform
   and the repair is not separated.
7. **Where the platforms part ways** (`platform_forest_probe.*.json`). The
   production QRF for retirement contributions was fingerprinted step by step
   on both platforms. The five sign gates' training probabilities agree to
   1.2e-16; at prediction their probabilities differ by up to 7.1e-12 and the
   sampled signs are identical in all five draws. The weighted bootstrap sample
   each forest is grown on is bit-identical, and the first target's quantile
   uniforms and features are identical. The forests differ: 427 of the 500
   trees differ in structure (all 100 in each of the first four forests, 27 in
   the fifth). In each diverging tree, the first node the two platforms split
   differently is reached by identical splits and holds the same samples on
   both. In 426 of the 427 both platforms split it, on different features;
   in 425 of those the two splits' weighted impurity decreases agree to within
   a relative 1e-9 (383 within 1e-12), so these are near-ties; in one the gains
   differ by half. In the remaining tree one platform splits the node and the
   other leaves it a leaf. The chain makes near-ties common: each later leaf is
   conditioned on the earlier ones, and
   `derive_us_retirement_contributions_from_manifest` sets the four 401(k)
   and IRA leaves of a person with wages as fixed shares of one remaining
   contribution total, so on those ASEC training rows the predictors are
   proportional (mathematically; the floating-point products need not be
   exact multiples). In the
   traditional-IRA forest, 87 trees first diverge on the traditional and Roth
   401(k) predictors, 61 of them at depth 0 or 1 (median depth 1, median 142
   samples). In the Roth-IRA forest, 86 diverge on a pair of the three earlier
   leaves (median depth 2). In the first forest, which has no chained
   predictor, the first divergence is deeper (median depth 4, median 4
   samples). Which platform-specific operation rounds differently was not
   identified.

**Why about 6% of PUF rows.** 30,442 of 475,340 PUF persons is 6.4%, but most
PUF persons draw no deferral in either pass: 77,713 hold a nonzero traditional
401(k) in the base, and 26,478 of those changed (34%); 3,964 more went from
zero to positive. The stage zeroes 401(k) draws for PUF persons without wages,
and a gate classifier draws the zero-or-positive sign. Relative to the base,
the release flips that sign for 8,006 persons; the platform alone flips none.
Each row's quantile uniforms come from a seeded stream whose consumption
depends only on the number of PUF rows and each target's sign regime, which
the training rows fix, so both passes give a row the same uniforms. On one
platform, a row's value then changes only if the training values in its
forest leaves, its sign, or an earlier leaf in the chain changed; across
platforms the quantile computation can also round differently.

**Direction and size** (`second_pass_effect…json`, PUF half, same rows, the
base's and the release's stored values under one fixed weight vector):

| Column | Under base weights | Under release weights |
|---|---:|---:|
| traditional 401(k), desired | +$1,593M (+1.21%) | +$3,567M (+2.11%) |
| Roth 401(k) | +$285M (+1.23%) | +$638M (+2.14%) |
| traditional IRA | +$75M (+1.18%) | +$151M (+1.87%) |
| Roth IRA | +$100M (+1.01%) | +$224M (+1.79%) |
| self-employed pension | +$11M (+3.48%) | +$33M (+4.18%) |
| childcare expenses | −$182M (−0.67%) | +$347M (+0.93%) |
| energy subsidy | +$2M (+0.18%) | +$127M (+8.13%) |
| weeks unemployed (person-weeks) | −0.06% | −0.22% |

The release weights were solved on the release's values. Neither column
recalibrates; what a release built from the base's values would weigh was not
measured.

**In the engine** (`engine_second_pass_effect…json`). The leaves above are
desired amounts. policyengine-us multiplies the traditional 401(k) amount by
`elective_deferral_contribution_scale`, the elective-deferral limit over the
person's total desired traditional and Roth 401(k) and 403(b) deferrals
(capped at 1), and subtracts the result from wages. Running policyengine-us 2.2.1 on the release twice at its own
weights, once as stored and once with the five leaves set back to the base's
values, the second draw changes the capped `traditional_401k_contributions`,
and so `pre_tax_contributions`, for 30,457 PUF persons (30,222 by more than
$1; +$3,559.5M weighted), and `irs_employment_income` for 30,435 (−$3,577.1M).
No ASEC person changes.

**Which pass ships.** The release's. Its stored values are the second pass's
draws, conditioned on the rescaled Social Security leaves and computed on the
release's platform.

## Scripts and receipts

| Script | What it measures | Receipts |
|---|---|---|
| `scripts/base_release_column_diff.py` | Every common column of every entity, base against release, by channel and clone, with direction; ids must be unique, unmatched rows are counted, integers are compared without conversion | `receipts/column_diff.base_0fb05b67.release_417d23ae.json` |
| `scripts/replay_release_reimputation.py`, `scripts/run_replays.sh` | The four stages replayed under the variants above, each output compared with both files | `receipts/replay.{release_order,isolated,attribution}.json` |
| `scripts/social_security_reorder.py` | How the release's Social Security factors reorder the summed predictor | `receipts/social_security_reorder.base_0fb05b67.json` |
| `scripts/base_stage_record_check.py` | The five leaves' weighted totals recomputed from the base H5 against the totals the base stage recorded | `receipts/base_stage_record_check.base_0fb05b67.json` |
| `scripts/second_pass_effect.py` | Weighted base and release totals on the same rows, by channel and clone | `receipts/second_pass_effect.base_0fb05b67.release_417d23ae.json` |
| `scripts/engine_second_pass_effect.py` | policyengine-us run on the release with and without the base's five leaves | `receipts/engine_second_pass_effect.base_0fb05b67.release_417d23ae.json` |
| `scripts/platform_stage_input.py` | Exports the retirement stage's exact input table (restricted microdata: kept local, not committed), checks it against the full table, replays it on any platform, optionally with a weight factor or a thread limit, and compares two platforms | `receipts/platform_replay*.json`, `receipts/platform_export_check.retirement.macos_arm64.json`, `receipts/platform_outputs.linux_amd64_vs_macos_arm64.json` |
| `scripts/platform_other_stages.py` | The same export and replay for weeks unemployed, childcare and energy subsidy | `receipts/platform_other_stages.export.macos_arm64.json`, `receipts/platform_replay.{weeks_unemployed,childcare,energy_subsidy}.linux_amd64.json` |
| `scripts/release_change_decomposition.py` | Each changed row classed by cause, from the stored values and both platforms' replays | `receipts/release_change_decomposition.base_0fb05b67.release_417d23ae.json` |
| `scripts/platform_forest_probe.py` | Fingerprints each QRF step on one platform (bootstrap sample, every tree, gate probabilities, drawn values) and compares two platforms, down to each tree's first divergent node | `receipts/platform_forest_probe.*.json` |
| `scripts/platform/` | The linux/amd64 harness: a `python:3.14.2-slim-bookworm` image, the stages' third-party imports at the base build's locked versions and hashes (`req.min.txt`), and a loader that imports a stage module without the package `__init__` chains | `receipts/platform_replay*.linux_amd64.json` |

Receipts bind the files they read by name and sha256. The replays of weeks
unemployed also read the Census ASEC 2023 person file, which the stage's own
loader checks against its pinned sha256 and size.

## Reproduce

```bash
RA=<checkout of 4b57d15a287c with its uv venv>
S=docs/evidence/us-release-reimputation-1191/scripts
$RA/.venv/bin/python $S/base_release_column_diff.py BASE.h5 RELEASE.h5 out.json
$S/run_replays.sh $RA BASE.h5 RELEASE.h5 calibration_diagnostics.json ~/.cache/microcosm/cps/asec_2023/pppub23.csv
$RA/.venv/bin/python $S/second_pass_effect.py BASE.h5 RELEASE.h5 out.json
$RA/.venv/bin/python $S/engine_second_pass_effect.py BASE.h5 RELEASE.h5 out.json
$RA/.venv/bin/python $S/platform_stage_input.py export --builder-root $RA --base-h5 BASE.h5 --release-h5 RELEASE.h5 --out input.pkl
$RA/.venv/bin/python $S/platform_stage_input.py replay --input input.pkl --out receipt.json --outputs macos.npz
$RA/.venv/bin/python $S/platform_stage_input.py verify-export --builder-root $RA --base-h5 BASE.h5 --outputs macos.npz --out out.json
```

Each replay group and the engine run hold 12 to 20 GB; run them one at a time.

The linux/amd64 replay runs in a work directory `W` holding `input.pkl`, the
scripts above, `scripts/platform/*`, and `src/` (`git archive 4b57d15a287c`):

```bash
docker build --platform linux/amd64 -t microcosm-replay:amd64 $S/platform
uv pip install --python <a non-free-threaded CPython 3.14> --target W/site-amd64 \
  --python-platform x86_64-manylinux_2_36 --python-version 3.14 --no-deps \
  --require-hashes --only-binary :all: -r $S/platform/req.min.txt
docker run --rm --platform linux/amd64 -v W:/work -e PYTHONPATH=/work/site-amd64 \
  -e PYTHONHASHSEED=0 -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  microcosm-replay:amd64 python3 -B /work/container_replay.py \
  /work/platform_stage_input.py replay --input /work/input.pkl \
  --out /work/receipt.json --outputs /work/linux.npz
$RA/.venv/bin/python $S/platform_stage_input.py compare W/linux.npz macos.npz --out out.json
```

`req.min.txt` is the subset of the builder commit's `uv.lock` export that the
stages import. `uv sync` itself hangs under QEMU, and the full lock pulls in
torch through `microcosm.calibrate`, which the loader's `GateResult` stand-in
avoids. `platform_other_stages.py` and `platform_forest_probe.py` run the same
way.
