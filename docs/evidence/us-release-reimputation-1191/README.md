# Release re-imputation of base-owned inputs (microcosm#1191 follow-up)

The #1191 PUF-channel wage audit found that the US release build imputes the
PUF-half retirement contributions a second time and does not reproduce the base
frame. This folder holds the scripts and receipts of the follow-up: which
release-time re-runs of base stages change stored values, why, and how much.
Nothing here changes a build. The invariant tests are
`packages/microcosm-build/tests/engine_free/us/test_us_release_base_input_invariant.py`
and `test_us_release_base_input_properties.py`.

## Files measured

| File | sha256 | Built |
|---|---|---|
| `base_populace_us_2024_puf_support.h5` | `0fb05b674899114e66dc01c2df0ff5349de88bbc8fed440755f4b19f287cedd2` | `tools/build_us_puf_support_base.py` at `4b57d15a287c`, on Modal (Linux x86_64, CPython 3.14.2, uv.lock `52e3d128…`) |
| `populace_us_2024.h5`, release `populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z` | `417d23aed4d044de2e657a7b5d856136975d25739dbdb4ddd72866f841de1559` | `tools/build_us_fiscal_refresh_release.py --base-h5` from that base at the same commit, on macOS arm64 (free-threaded CPython 3.14.7) |
| that release's `calibration_diagnostics.json` | `64b55a02aaf573964cffee44d4a86bd63f33dc30c7f5012e4345348eecc06f80` | records the release's population and Social Security repair factors |

Both builds used numpy 2.4.6, pandas 3.0.3, scikit-learn 1.8.0 and
quantile-forest 1.4.2.

## Result

**What the release re-runs.** With `--base-h5` and no validated pool frame,
`_main` runs 17 calls under `if pool_frame is None`. Fourteen re-run a stage
the base builder already ran: weeks unemployed, QBI reconciliation, childcare,
energy subsidy, retirement contributions, immigration, relationship, SPM
independence role, Medicare take-up, retirement distributions, eligibility,
education, pregnancy and WIC. The other three are the population mass repair
and two release-only stages (take-up, hours worked). The Social Security and
non-Schedule-D capital-gain value repairs run between the first re-run and the
rest.

**What changed** (`column_diff…json`, every column of every entity, rows matched
by id; the PUF half is 475,340 persons and 191,267 SPM units):

| Column | PUF rows changed | ASEC rows changed |
|---|---:|---:|
| `traditional_401k_contributions_desired` | 30,442 persons | 0 |
| `roth_401k_contributions_desired` | 28,695 | 0 |
| `traditional_ira_contributions_desired` | 28,913 | 0 |
| `roth_ira_contributions_desired` | 29,223 | 0 |
| `self_employed_pension_contributions_desired` | 1,219 | 0 |
| `spm_unit_energy_subsidy` | 3,283 SPM units | 0 |
| `spm_unit_pre_subsidy_childcare_expenses` | 1,849 SPM units | 0 |
| `weeks_unemployed` | 9 persons | 0 |

The other ten re-runs leave every stored column unchanged. Beyond the re-runs,
the release's declared repairs rescale the four Social Security component
leaves and `non_sch_d_capital_gains` by constant factors, calibration changes
the household weights, and the release-only other-health-insurance stage
replaces `health_insurance_premiums_without_medicare_part_b` on 225,373 PUF
persons (its module docstring: it "replaces only the PUF support channel").

**Why** (`replay.*.json`). The replay imports the release builder from the
commit that built the release, loads the base with the builder's
`_load_frame`, and runs the stages itself:

1. Replaying the release's order (weeks unemployed, mass repair, Social
   Security repair fed the four recorded SSA targets, QBI reconciliation,
   childcare, energy subsidy, retirement contributions) reproduces the
   release's stored values exactly for all four stages: 0 rows differ.
2. Applying only the recorded Social Security factors (1.1475 retirement,
   1.1104 disability, 2.4926 survivors, 0.9056 dependents) to the base and
   re-running childcare, energy subsidy and retirement contributions also
   reproduces the release exactly. `social_security`, the sum of the four
   leaves, is one of the eight predictors of all three QRFs, and unequal
   factors reorder it.
3. The population mass repair alone (factor 1.0253 on every household weight)
   changes nothing, and neither does limiting BLAS/OpenMP pools, forest fit
   jobs and draw workers to four threads: each reproduces the re-run on the
   unchanged base bit for bit on all five contribution columns
   (`platform_replay.macos_arm64.{mass_factor,threads4}.json`).
4. Re-running on the unchanged base on macOS does not reproduce the Modal-built
   base: retirement contributions differ on 21,765 PUF persons for the
   traditional 401(k) leaf (21,484 by more than $1, up to $18,523, no zero or
   nonzero flips), childcare on 54 and energy subsidy on 17 SPM units (at most
   5.1e-11), weeks unemployed on 9 persons (1 to 10 weeks). Weeks unemployed is
   the release's first re-run, before any repair. The base H5 reproduces the
   base stage's recorded weighted contribution totals exactly
   (`base_stage_record_check.base_0fb05b67.json`), so the stored outputs and
   weights are the base stage's own.
5. The platform is the cause (`platform_replay.*.json`). The retirement
   stage's exact input table, exported from the base, was replayed with the
   production stage code on two platforms. On linux/amd64 (CPython 3.14.2,
   the base build's uv.lock wheels, under QEMU) the traditional 401(k) leaf
   matches the Modal-built base on all but 235 of 907,863 persons, each by at
   most 6.9e-11; the other four leaves differ on 15, 5, 11 and 1 persons, at
   most 1.6e-12. On macOS arm64 the same input differs from the base on 21,765
   persons, and the two platforms' outputs differ from each other on 21,764
   (21,484 by more than $1;
   `platform_outputs.linux_amd64_vs_macos_arm64.json`). Because Linux
   reproduces the base from the inputs the H5 stores, those inputs are the ones
   the base stage read; the macOS difference is the platform's. quantile-forest 1.4.2
   has no cp314 Linux aarch64 wheel, so architecture and operating system were
   not separated. Only the retirement stage was replayed on Linux; the
   weeks-unemployed, childcare and energy-subsidy differences in point 4 were
   not.

6. The divergence is in growing the trees, at tied splits
   (`platform_forest_probe.*.json`). The production QRF was fingerprinted step
   by step on both platforms. The five sign gates' training probabilities agree
   to 1.1e-16, the weighted bootstrap sample each forest is grown on is
   bit-identical, and the sign draws, quantile uniforms and features of the
   first target are identical. The forests differ: all 100 trees differ in
   structure in each of the first four forests (27 of 100 in the fifth). In
   every diverging tree, the first node split differently is reached by
   identical splits, holds the same number of samples on both platforms and is
   split on a different feature, never the same feature at another threshold,
   and the two splits' weighted impurity decreases agree to rounding (median
   relative difference 6.6e-15 in the first forest, 93 of 100 within 1e-12). The chain makes such ties structural: each
   later leaf is conditioned on the earlier ones, and
   `derive_us_retirement_contributions_from_manifest` sets traditional and
   Roth 401(k) to fixed shares of one pool and the two IRA leaves to fixed
   shares of another, so on the ASEC training rows those predictors are exact
   multiples of each other. In the traditional-IRA forest 87 of 100 trees
   first diverge on a tie between the traditional and Roth 401(k) predictors,
   and in the Roth-IRA forest 86 of 100 on a tie between two of the three
   earlier leaves, at the root or just below it (median depth 1 and 2, median
   node 139 and 155 samples). In the first forest, which has no chained
   predictor, the ties are in small nodes (median 4 samples). Which
   platform-specific operation rounds differently was not identified.

**Why about 6% of PUF rows.** 30,442 of 475,340 PUF persons is 6.4%, but most
PUF persons draw no deferral in either pass: 77,713 hold a nonzero traditional
401(k) in the base, and 26,478 of those changed (34%); 3,964 more went from
zero to positive. The stage zeroes 401(k) draws for PUF persons without wages
and draws the zero-or-positive sign from a gate classifier. Relative to the
base, the release flips that sign for 8,006 persons; the platform alone flips
none. Each row's quantile uniforms come from a seeded stream whose consumption
depends only on the number of PUF rows and each target's sign regime, which
the training rows fix, so a row keeps its exact value unless its forest leaves,
its sign or an earlier leaf in the chain changed.

**Direction and size** (`second_pass_effect…json`, PUF half, same rows):

| Column | Under base weights | Under release weights |
|---|---:|---:|
| traditional 401(k) | +$1,593M (+1.21%) | +$3,567M (+2.11%) |
| Roth 401(k) | +$285M (+1.23%) | +$638M (+2.14%) |
| traditional IRA | +$75M (+1.18%) | +$151M (+1.87%) |
| Roth IRA | +$100M (+1.01%) | +$224M (+1.79%) |
| self-employed pension | +$11M (+3.48%) | +$33M (+4.18%) |
| childcare expenses | −$182M (−0.67%) | +$347M (+0.93%) |
| energy subsidy | +$2M (+0.18%) | +$127M (+8.13%) |
| weeks unemployed (person-weeks) | −0.06% | −0.22% |

The release weights were calibrated on the release values, so the second
column includes calibration's response to the second pass; the first column
isolates the redraw.

**Which pass ships.** The release's. Its stored values are the second pass's
draws, conditioned on the rescaled Social Security leaves and computed on the
release's platform.

## Scripts and receipts

| Script | What it measures | Receipts |
|---|---|---|
| `scripts/base_release_column_diff.py` | Every common column of every entity, base against release, by channel and clone, with direction | `receipts/column_diff.base_0fb05b67.release_417d23ae.json` |
| `scripts/replay_release_reimputation.py`, `scripts/run_replays.sh` | The four stages replayed under the variants above, each output compared with both files | `receipts/replay.{release_order,isolated,attribution}.json` |
| `scripts/base_stage_record_check.py` | The five leaves' weighted totals recomputed from the base H5 against the totals the base stage recorded when it finished | `receipts/base_stage_record_check.base_0fb05b67.json` |
| `scripts/second_pass_effect.py` | Weighted base and release totals on the same rows, by channel and clone | `receipts/second_pass_effect.base_0fb05b67.release_417d23ae.json` |
| `scripts/platform_stage_input.py` | Exports the retirement stage's exact input table (restricted microdata: kept local, not committed) and replays it on any platform, optionally with a weight factor or a thread limit | `receipts/platform_replay.*.json` |
| `scripts/platform_forest_probe.py` | Fingerprints each QRF step on one platform (bootstrap sample, every tree, gate probabilities, drawn values) and compares two platforms, down to each tree's first divergent split | `receipts/platform_forest_probe.*.json` |
| `scripts/platform/` | The linux/amd64 harness: a `python:3.14.2-slim-bookworm` image, the stage's third-party imports at the base build's locked versions and hashes (`req.min.txt`, cross-installed on the host with `uv pip install --python-platform x86_64-manylinux_2_36 --target`), and a loader that imports the stage module without the package `__init__` chains | `receipts/platform_replay.linux_amd64.json` |

## Reproduce

```bash
RA=<checkout of 4b57d15a287c with its uv venv>
S=docs/evidence/us-release-reimputation-1191/scripts
$RA/.venv/bin/python $S/base_release_column_diff.py BASE.h5 RELEASE.h5 out.json
$S/run_replays.sh $RA BASE.h5 RELEASE.h5 calibration_diagnostics.json ~/.cache/microcosm/cps/asec_2023/pppub23.csv
$RA/.venv/bin/python $S/second_pass_effect.py BASE.h5 RELEASE.h5 out.json
$RA/.venv/bin/python $S/platform_stage_input.py export --builder-root $RA --base-h5 BASE.h5 --release-h5 RELEASE.h5 --out input.pkl
$RA/.venv/bin/python $S/platform_stage_input.py replay --input input.pkl --out receipt.json --outputs macos.npz
```

Each replay group holds about 12 GB; `run_replays.sh` runs them one at a time.

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
stage imports. `uv sync` itself hangs under QEMU, and the full lock pulls in
torch through `microcosm.calibrate`, which the loader's `GateResult` stand-in
avoids. `platform_forest_probe.py` runs the same way.
