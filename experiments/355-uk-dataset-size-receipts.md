# #355 dataset-size receipts: 55,000 households by informed L0 on the #877 machinery

Plan of record: `repos/uk-355-size-experiment-plan.md` (approved 2026-09-08, two rounds). These
receipts record what was run, on which pinned inputs, and what was measured. They are candidate
evidence only: no size run is releasable (`--release-candidate` is refused with
`--dataset-households`), no gate threshold is loosened, and nothing here promotes an artifact.
Kept separate from the #762 receipts (`experiments/762-uk-rowwise-candidate-receipts.md`).

Rulings carried in (María, 2026-09-08): K=15 cloning stays in front of the selection (the dense
joint solve runs on the whole 792,690-row pool; the size machinery then keeps 55,000 of those
rows); `--epochs 2000` on size runs (drives the dense solve, every L0 probe and the refit); skip
holdout on the first 55k run; no dense baseline on the new inputs unless the rough comparison with
R17 shows large deviations; engine policyengine-uk 2.94.0; Chronicle feed 6fb700e.

## Step 0 — #877 CI green before any run

CI run 34216367053 on 56aa4e25 (the spec-engine re-pin) was red in four jobs with one cause: the
`solve.py` edit in #877 moved two more source-hashed identities.

| lane | test | pin | fix |
|---|---|---|---|
| `fast (rest)` ×2 | `microcosm-graph/tests/test_acceptance_h_parity.py::test_h1_kernel_parity`, `test_graph_serialize.py::test_generated_parity_graphs_bind_real_kernels_and_direct_bytes` | `calibrate.adam@1` implementation hash a7a0330f… → d24fc41d… (`source_hash` over the calibrate modules) | H1 fixture regenerated on its authoring platform (arm64/darwin/py3.14; numpy 2.4.6, pandas 3.0.3, scipy 1.17.1, torch 2.12.0) with `tools/graph_parity_fixtures.py`; only `calibrate/pins.json` taken (direct bytes and graph unchanged). The generator also resets `fit.qrf/pins.json`'s platform map to the authoring platform, dropping its two linux pins; that file was reverted. |
| `engine-us (us-am)` ×2 | `test_us_multispine_pool_tool.py::test_constants_adapter_equals_live_constants_and_stays_out_of_identities` | US `spec_sha256` 9db29b4d… → 9db2d6db… | re-pinned |

Commit 1b0583b7. Local: graph 336 passed, multispine pool tool 187 passed, spec-engine pin files 50
passed. Pushed with C1/C2 below as d4043ec7; CI run 34223080395.

The run on d4043ec7 then failed in the `wheels` lane only (both Python versions): the size CLI test read
the exported H5 through `pd.HDFStore`, and the wheels venv has no pytables (7,963 passed, 587 skipped, that
one failure). The file's convention is `pytest.importorskip("tables")` + `importorskip("h5py")` at the top of
every CLI test; the size test lacked them, and the two new evaluation test files write PyTables-format H5
the same way. All three now skip without pytables (verified by blocking the import locally); the workspace
and engine lanes, which have pytables, run them in full.

The run on fb85c8fe then failed only in the two UK lanes on one new test: the evaluation command timed
each downstream script with `/usr/bin/time -l`, a BSD flag GNU `time` rejects with exit 125, so on the
linux runners a failing stub could not surface its own exit code. Replaced by an in-process rusage shim
(the child script runs in a child interpreter that reports its own peak resident set in bytes on exit;
wall time measured by the caller; exit code the script's own), commit e8f7c6f2, pushed with the
`--selection-pi-hi` knob (a6b1e0ca) and the by-name refusal (b1d206d6); CI run 34239851890.

## Code landed on the branch before the runs

- **C1 (d4043ec7)** — a size run keeps the dense joint solve it was cut from: `UKRowwiseDoctrineSolve.dense_reference` (weights, initial weights, local and national diagnostics, losses, past-cap censuses; the evidence labelling moved into `_doctrine_solve_evidence` and runs for both results), written as `dense_reference_diagnostics.csv` (every target's dense estimate with a `grain` column) and summarised under `solve.dataset_size.dense_reference`; and the selection itself as `dataset_size_selection.csv` (`pool_row_index, household_id, clone_index, design_weight, inclusion_probability, certainty, ht_baseline_weight, refit_weight`). Both are listed under `outputs` with digests; dense runs are unchanged. The dense reference is byte-identical to a standalone dense run on the same inputs, seed and epochs, so the size-only delta needs no second full-pool solve.
- **C2 (d4043ec7)** — `--selection-seed` (requires `--dataset-households`, defaults to `--seed`) seeds only the informed L0 search, the exact-count draw and the refit, threaded through the holdout as well; recorded as `parameters.selection_seed` and in the size receipt's `seed`. `--seed` still governs the ladder clone assignment and the dense solve, so two selections compare on one pool and one dense reference.
- **C3/C4 (251e67d4)** — evaluation library `uk_runtime/size_evaluation.py` (`load_run`, `run_acceptance`, `fit_tables`, `weight_tables`, `area_support_tables`, `gate_table`, `paired_targets`, `dense_reference_deltas`, `frozen_vs_recomputed`, `footprint`, `summarize` with `PRE_REGISTERED_OUTCOMES_V1`) and the one command `tools/evaluate_uk_dataset_size.py` (steps `00-run-acceptance`, `10-dense-reference`, `20-vs-reference/<label>`, `30-incumbent-score`, `40-incumbent-surface`, `50-downstream`, `90-summary` into `<run>/evaluation/`, each with `receipt.json`); `_fit_by_family` lifted into `uk_runtime/diagnostics.py::uk_fit_by_family`. Implemented by Codex from `.codex-work/PLAN.md`; its final report was lost when the companion broker restarted, so the review was done on the diff directly (lift byte-for-byte; grain rule, `time -l` parsing, multiplier inference matching `id_multiplier_for_values` with the manifest self-check, red-row mapping through family/area/metric, the R18 evaluator's `national_rows`/`candidate_estimate` shape, flags exactly as pre-registered) and verified independently: ruff, format, `ci_test_groups --verify`, 88 tests across the four UK rowwise/evaluation files.
- **Feasibility diagnostic (4bbd01f6)** — see S1 below.

## Rebase onto main after #879 (María's instruction before any further push)

Main moved by PR #879 (SPI income coherence, e07a4735; 13 commits, 42 files) after the branch was
cut. Intersection with the branch: one file, one line — the UK country-bundle digest in
`test_spec_engine_country_bundles.py`, which both sides had moved (#879 through `uk/spec/sources.yaml`
and `country_package.json`; this branch through `solve.py`'s seed-protocol attestation). Rebased
(`git rebase origin/main`), resolved that line, then re-cut it once for the combined tree:
d57d248fe947c3215850020bb16c6def308e373c92afa433395c81e8eda91fce (commit 573c4b43). The am/be
digests, the loader golden vector, the seed-protocol/seed-map digests and the committed US coverage
report were unchanged by #879 (49/50 pin tests passed before the cut). Rebased heads: acbc6332 (size
machinery), 08ae0caf (spec re-pin), 366b8263 (H1 + US spec re-pin), cfd57e03 (C1/C2), 12b9a7ce
(feasibility diagnostic), 6ec596c3 (C3/C4), 573c4b43 (UK digest re-cut).

Isolated re-verification instead of the full suite (her call: the conflict surface is minimal): the
four spec-engine pin files; graph H1 parity and serialisation; the multispine pool-tool test (US spec
digest); the branch's own suites (calibrate, local rowwise, rowwise candidate, informed gates, the two
evaluation tests); and the battery neighbours #879 touched (country spec, local gate battery, gate
battery contract pins, terminal gates, battery bindings). Results on the rebased tree: spec-engine pin
files 50/50 after the one re-cut; graph H1 parity + serialisation + multispine pool tool 195 passed;
calibrate + local rowwise + rowwise candidate + informed gates + the two evaluation tests 313 passed;
battery neighbours 231 passed. Pushed 2026-09-08 (force-with-lease from d4043ec7 → 573c4b43); PR #877
MERGEABLE on main with 7 commits; CI run 34230891965 (the earlier run 34223080395 on d4043ec7 had
every engine lane and the fast trade/rest lanes green when superseded). CI runs the full lanes on the push.

#879 also changed the SPI income spine stage (`hmrc_spi_income_spine` gained `donor_income_period`
2022 and `income_uprating_variables` in `source_stages.json`; `spi_income.py`, `spi_spine.py`), so
spine-o is stale on the rebased tree: **spine-p** is rebuilt from the clean tree at 573c4b43 (same
recipe) and S0 re-run on it before any further size run.

## P0 — inputs and pins

| input | path | sha256 | note |
|---|---|---|---|
| Chronicle consumer artifact 6fb700e | `data/ukds/acceptance/chronicle-uk-artifact-6fb700e/consumer_facts.jsonl` | 6ae49d7d7ab297df25a0b9bfe2d6776827c672d284fbb360957fe8337089549f | 128,717 rows; manifest dcda51d6496aea67f768a284e7955c7520e7c8b91e2bed3569f247567b7153f0; the feed main's registers compile against. Upstream chronicle main is 7 commits ahead; the one fact change (#250, UC family-type totals 2023–2025, merged 2026-09-08) is not consumed by any register and is the next re-pin candidate, not part of this experiment. |
| OA ladder | `populace-877/build/uk/uk_oa_ladder_2021.npz` | 9c6d56b90d2e975d750106b175020a54c5ec6acf42ef8909d304a9d7fc3868a7 | copied from `populace-762b/build/uk/` |
| spine-o | `data/ukds/acceptance/spine-o-355/spine-o.h5` | 3ef32dfb3e8a86e1c93a8f74b463492fe8eeb81d10ebc746c7b396aa47c2eeca | built 2026-09-08 from the clean tree at d4043ec7 (`build_twin.sh` recipe, all licensed tabs; 352 s wall, 7.05 GB peak): 28 stages with `age_tail` second and `uc_deduction_attributes` present, policyengine-uk 2.94.0, 15/15 spine gates passed, `tax_free_childcare_spend_routed_share` nonzero share 1.0, 52,846 households / 61,234 benunits / 113,626 persons, mass 29,247,433.0; sidecar `spine-o.build.json`. Rebuilt because spine-m (R17) predates #850, #842 and #874; see below |

| spine-p | `data/ukds/acceptance/spine-p-355/spine-p.h5` | ae83e3075d1fd99adf3dbb1ea8bbf96398fa48b8eb251b802ae682ada03fb19a | built 2026-09-08 from the clean tree at 573c4b43 (rebased onto #879; same recipe; 424 s wall, 3.84 GB peak): 28 stages, `age_tail` second, `uc_deduction_attributes`, policyengine-uk 2.94.0, 15/15 spine gates, TFC routed share 1.0, 52,846 households / 61,213 benunits / 113,590 persons (the #879 SPI donor change moves 21 benunits and 36 persons against spine-o), mass 29,247,433.0. **The spine every size run from here stands on.** |

Why the spine was rebuilt: R17 stood on spine-m (27 stages, engine 2.92.1, feed 1cab809). Main since
then added `uc_deduction_attributes` and the WAS debt columns (#850), moved `age_tail` second with
int64 age (#842), and added `tax_free_childcare_spend_routed_share`, the `hmrc.tfc.*` and DfT bus
targets, the 6fb700e feed pin and the 2.94.0 engine floor (#874). No spine on disk carried the TFC
column.

## Runs

Runner: `data/ukds/acceptance/355-dataset-size/run_size_candidate.sh <name> [--prev <row>] <driver args>`
(spine-o, ladder, 6fb700e feed, seed 42, `/usr/bin/time -l`, log copied into the run dir). Root
`data/ukds/acceptance/355-dataset-size/spine-o/`. Preflight `--env` on the pins: OK.

### S0 — dry run (`--dry-run --dataset-households 55000 --epochs 2000`)

Exit 0, 96.9 s, 6.0 GB. Pins bound; plan reports the joint surface 20,430 targets × 792,690 households
(K=15 clones of 52,846), `parameters.dataset_households 55000`, `epochs 2000`, `engine_blocks 1`, the
resolved vintages (UC 2025-05, HMRC 2023, ONS housing 2021/2022/2025-12, ONS population 2024).
No files written (dry runs never do).

### S1 — rehearsal (`--dataset-households 55000 --epochs 100 --skip-holdout`, chained on R17's row)

**Refused at the exact-count draw** after 1,542 s (25.7 min; 9.48 GB peak): the dense solve and the
informed L0 budget search completed (100 epochs each; `git_dirty` true because the evaluation tool was
being written in the same tree — a rehearsal, so noted and accepted), then
`microcosm.calibrate.exact_k._draw_boundary` raised

> degenerate boundary mass: proportional normalization would require a boundary inclusion
> probability greater than one; adjust pi_hi or k.

Logbook row d15c0224… (disposition `failed`, phases reached through `targets_bound`, error receipt
`logbook-receipts/…/error.json`). Mechanism: `select_exact_k(pi, k, pi_hi=1.0)` treats only exact-one
gates (the protected carriers) as certainties, scales every other gate's open probability to the
remaining draw size `m`, and refuses when the largest scaled value exceeds one — i.e. when
`m · max(pi_boundary) > Σ pi_boundary`. The L0 budget search stops when the **count** of
not-fully-closed gates is within 5 % of 55,000; with gates only partly polarised that count sits above
the open-probability **mass**, so the boundary mass fell short of the draw. The refusal carried no
numbers, so a feasibility diagnostic was added to `refit_uk_dataset_size` (boundary mass, largest
boundary gate, feasibility at `pi_hi=1`, the largest feasible household count at `pi_hi=1`, the
smallest feasible `pi_hi` on a fixed grid, gate quantiles) — attached to the refusal and to the size
receipt on success — and the rehearsal is re-run below. This is a design ruling for #877, not a run
setting: the US exact-k ladder release runs with `pi_hi = 0.95` (`tools/build_us_exact_k_ladder_release.py`),
while the UK size plan deliberately pins `pi_hi = 1` ("no post-hoc promotion of learned boundary scores").

### S1b — rehearsal re-run with the feasibility numbers (same inputs, seed 42, epochs 100, chained on S1's row)

**Refused at the same point** after 1,592 s (26.5 min; 8.49 GB peak; commit 4bbd01f6). The dense solve
and the L0 budget search are deterministic on the same inputs, so the refusal is the same; the
diagnostic now says why:

| quantity | value |
|---|---|
| pool households | 792,690 |
| requested households (k) | 55,000 |
| protected carriers = certainties at `pi_hi = 1` | 9,869 |
| boundary draw m = k − certainties | 45,131 |
| budget search stopped at `n_nonzero` (count of not-fully-closed gates) | 52,490 (inside the ±5 % band round 55,000) |
| total open-probability mass Σπ | 45,743 |
| boundary mass Σπ (π < 1) | 35,874 |
| largest boundary gate | 0.9982 |
| gates with π ≥ 0.5 / 0.9 / 0.99 / 0.999 | 36,527 / 25,267 / 10,761 / 9,869 |
| π quantiles p50 / p90 | 0.0063 / 0.053 |
| learned λ | 4.2e-06 |
| feasible at `pi_hi = 1`? | no: m · max π = 45,048 > 35,874 |
| largest feasible k at `pi_hi = 1` on these gates | 45,809 |
| smallest feasible `pi_hi` on the grid {0.999 … 0.5} | 0.5 (0.7 fails by 3 %: 22,664 × 0.7 = 15,865 > 15,329) |

Reading: the budget search hit its target on the **count** of not-fully-closed gates (52,490), but the
**expected number of open gates** — the mass the exact-count design can draw from — is 45,743, a fifth
short of 55,000. Most of the 782,821 boundary gates are near-closed (median π 0.006) yet not closed,
so they count towards `n_nonzero` while contributing almost no mass; the ~15,000 gates between 0.9 and
0.99 carry most of the boundary mass and cannot be scaled up to fill a 45,131 draw. With `pi_hi = 1`
the draw is feasible only up to ~45.8k households at this polarisation. The US precedent (`pi_hi =
0.95`) would not rescue it either at 100 epochs — only thresholds at or below about 0.65 are feasible
here. Whether 2,000 epochs polarises the gates enough to close the gap is unknown; the count-versus-
mass mismatch in the stopping rule remains regardless of epochs. Ruling requested from María (see
the session report): (A) expose `pi_hi` as a candidate-run knob; (B) make the size selection's budget
search stop on open-probability mass Σπ ≈ k instead of `n_nonzero`, which keeps `pi_hi = 1` and makes
the draw feasible by construction; (C) accept k ≈ Σπ; (D) spend the 2,000-epoch run to see. Nothing
was clamped; Logbook row for S1b is chained on S1's.

### S0 on spine-p — dry run after the rebase (`--dry-run --dataset-households 55000 --epochs 2000`)

Exit 0, 97.7 s, 4.84 GB (2026-09-08 13:15Z, code 573c4b43, engine 2.94.0, preflight `--env` OK on
`pins-spine-p.txt`). Plan: matrix 20,796 rows (20,430 local + 366 national) × 792,690 columns
(K=15 clones of 52,846); `parameters` dataset_households 55,000, epochs 2,000, n_clones 15, seed 42,
selection_seed 42, engine_blocks 1; `releasable` false; engine not run. The 7 extra national rows
against spine-o's dry run are #879's SPI rows. Every size run from here stands on spine-p; the
spine-o runs above (S0, S1, S1b) remain valid evidence about the L0 machinery, which #879 did not touch.

### Ruling (María, 2026-09-08) and the knob it needed

Smoke: accept the measured feasible count (45,800 at `pi_hi = 1` on the 100-epoch gates) so every
mechanical piece downstream of the draw runs once at scale — the Sampford rejection path on 792,690
gates, the frozen-target refit, the compact export with closed links, the six-gate battery on a compact
frame, the dense-reference and selection sidecars, the manifest's size receipt, and then the evaluation
command end to end (run acceptance, dense-reference deltas, the comparison with R17, the incumbent
scorer on a compact candidate, the incumbent-surface evaluator re-resolving the engine on a compact
rowwise frame, the eval-home scripts loading a compact H5 with their footprint timings, the summary).
Its fit numbers are not evidence (100 epochs); its footprint numbers are.

Full run: `--selection-pi-hi 0.95` (the US exact-k ladder's setting) with `--epochs 2000` at 55,000.
Commit a6b1e0ca makes the certainty threshold a recorded candidate-run knob (default 1.0; bounded to
(0, 1]; requires `--dataset-households`; recorded as `parameters.selection_pi_hi`, in the size receipt
and in the feasibility measurement as `requested_pi_hi` / `feasible_at_requested_pi_hi`). Risk stated
up front: at 100 epochs 0.95 was not feasible either (34,944 × 0.95 > 25,954); 2,000 epochs should
polarise the gates, and if the draw still refuses, the receipt names the smallest feasible threshold.

### Smoke — 45,800 on spine-p (`--dataset-households 45800 --epochs 100 --skip-holdout`, chained on S1b's row 399e1e7d…)

**Refused before the selection**, 608 s in (10.1 min; 10.6 GB peak; code a6b1e0ca), inside
`contribution_initialization`: `nonzero target at row 20648 has no support`. Row 20648 is national
row 218 of the 366 the rebased tree binds; reproducing the driver's engine-free registry compile
names it: `dwp/uc_payment_dist/COUPLE_NO_CHILDREN_annual_payment_27_600_to_28_800` (family
`dwp_universal_credit`, benunit grain, target 746.3 units), whose constraint row is all zeros on
the cloned pool. R17 had support for the same band on spine-m under the pre-#879 engine
`family_type` (initial estimate 551 against a then-target of 654) and no national row without
support at all; #879 re-points the payment-distribution rows to the relationship-based
`uc_calibration_family_type` proxy, under which no benefit unit in the pool is a couple without
children with an annual UC payment in that band. The dense solve carries such a row as a capped miss;
a size selection cannot carry it. Logbook row 1c7f25dd… (failed).

Response: `refit_uk_dataset_size` now refuses unsupported nonzero targets **by name**, all at once,
before initialisation (commit b1d206d6, `unsupported_nonzero_targets()`), and the smoke is re-run to
obtain the complete list. Handling is a ruling for María: a signed measure exclusion for the affected
rows (the A16 register), a fix to the proxy or the band edges upstream in #879's register, or a
recorded smoke-only drop of unsupported rows from the selection problem (the dense reference keeps them
as misses). Nothing is selected around silently.

Re-run (`f100-k15-h45800-e100-smoke2`, chained on 1c7f25dd…; 589 s to the check, 10.6 GB; code b1d206d6):
the by-name pre-check refuses exactly two rows, both in `dwp_universal_credit`:

- `dwp/uc_payment_dist/COUPLE_NO_CHILDREN_annual_payment_27_600_to_28_800@2025` (target 746.3)
- `dwp/uc_payment_dist/COUPLE_NO_CHILDREN_annual_payment_28_800_to_30_000@2025`

These are the same two rows #879's corrected-measurement comparison reports at −100 %
(`experiments/879-corrected-calibration-comparison.md`, rows 112–113): under the relationship-based
`uc_calibration_family_type` proxy no benefit unit in the spine is a couple without children with an
annual UC award above £27,600. The national seam carries them as capped misses; the catalogue and
upper-band repair is tracked in microcosm#736; 16 payment bands already sit in the reviewed measure
exclusion register. **Proposed (unsigned) handling**: two entries in
`calibration_measure_exclusions.json` on the zero-support-channel precedent
(`slc.repayments.england_postgraduate`), approved by María if she rules so; until then no size run
can start on the rebased tree. Logbook row for smoke2 chained on smoke's.

**Ruling (María, 2026-09-08): sign the two exclusions; run the next round; rebase #877 later.** Signed as
commit 2959e177 into `calibration_measure_exclusions.json` on the zero-support-channel precedent
(approved_by juaristi22, approved_on 2026-09-08, expires_on 2026-12-08, tracking microcosm#736,
adjudication microcosm#355); the register's census pins move with it (51 entries; 18 payment-band
exclusions; 82 active payment bands). Smoke re-launched as `f100-k15-h45800-e100-smoke3`, chained on
smoke2's row. No push until the later rebase.

### Smoke3 — 45,800 on spine-p with the exclusions signed (`--epochs 100 --skip-holdout`, `pi_hi` 1.0)

Passed the by-name check (364 national rows) and **refused at the draw** after 1,512 s (25.2 min;
11.7 GB): on spine-p's 100-epoch gates the measured numbers are

| quantity | spine-o (S1b, k 55,000) | spine-p (smoke3, k 45,800) |
|---|---|---|
| protected carriers = certainties at `pi_hi = 1` | 9,869 | 10,026 |
| budget search `n_nonzero` | 52,490 | 43,938 |
| total open-probability mass Σπ | 45,743 | 39,577 |
| boundary draw m / boundary mass | 45,131 / 35,874 | 35,774 / 29,551 |
| largest boundary gate | 0.998 | 0.998 |
| largest feasible k at `pi_hi = 1` | 45,809 | 39,638 |
| smallest feasible `pi_hi` on the grid | 0.5 | 0.7 (0.8 fails) |
| `pi_hi = 0.95` | infeasible | infeasible (27,443 × 0.95 = 26,071 > 21,439) |

Reading: the "largest feasible count at `pi_hi = 1`" is not a stable target. The budget search
re-learns the gates for whatever count is requested and stops on the count of not-fully-closed gates;
in both measurements the open-probability mass came out at 87–90 % of that count, so requesting the
measured feasible count would only measure a new, lower one. What the measurement does guarantee is
feasibility on the same gates at a lower certainty threshold. Smoke4 therefore runs 45,800 at
`--selection-pi-hi 0.7` (the smallest grid value feasible on these gates), 100 epochs, chained on
smoke3's row bd9ebb9b…; it is the first run to pass the draw. Risk carried forward to S2: at 100 epochs
`pi_hi = 0.95` is infeasible by 18 %; whether 2,000 epochs polarise the gates enough is unknown, and
the count-versus-mass mismatch in the stopping rule does not depend on epochs.

### Smoke4 — 45,800 at `pi_hi 0.7` on spine-p: the first run through the whole size pipeline

Exit 1 = gates failed with the evidence bundle written (the candidate outcome). 1,344 s (22.4 min),
10.6 GB, code 27eb74a1 with a clean tree, engine 2.94.0 in one block. Draw: 27,858 certainties at
`pi_hi 0.7` over a boundary pool of 764,832, Sampford, `feasible_at_requested_pi_hi` true; refit on
the frozen surface; compact export 45,800 households (H5 187 MB against R17's 2.36 GB); the six-gate
battery on the compact frame; both sidecars (`dense_reference_diagnostics.csv` 20,794 rows,
`dataset_size_selection.csv` 45,800 rows) listed in `outputs` with digests; Logbook row 363c8c4f….
Plumbing numbers, not evidence (100 epochs): dense reference loss 0.409 → 0.138, compact 0.613 →
0.172; realised stretch 10.0 against the Horvitz–Thompson baseline; mass 29,247,433 → 27,974,030
(−4.4 %); Kish ESS 9,034 (fraction 0.197); max/median positive weight 244; area support 960 of 1,011
areas below floor (constituency ESS minimum 2.3); target fit 1,321 rows past 25 %.

**Defect found in the surface, not the smoke: `ons.rent.private_rent` (314 local-authority rows, family
`private_rent`, source ONS PIPR).** Every row sits at a relative error of 10^5 to 10^6 on both the dense
reference and the compact fit: the target is a mean monthly rent (£552 to £3,633 per authority) while
the metric `rent/private_rent` is `household_rent × is_private_renter` summed with weights, an annual
rent total (10^8 to 10^9). R17's surface had no `private_rent` rows (19,105 local rows; this tree binds
19,419). The rows became active with #874's re-pin of the local references to Chronicle 6fb700e; the
census entry for the source is signed-deferred (`private_rent_pipr_partial_coverage_2025`: the feed's
only PIPR period is 2026-06, after the 2025 target period) yet 314 of 673 candidates are active. At the
loss cap these rows distort the dense solve as much as the selection, so they block S2 regardless of
`pi_hi`. Ruling requested: extend the signed deferral to the whole target (the surface returns to R17's
ten local families, making the comparison like for like) and fix the metric's semantics (a weighted
mean among private renters, on the PIPR period policy) as its own change; or fix the metric first.

**Evaluation plumbing pass on smoke4** (`evaluate_size_run.sh`, 1,446 s, 12.1 GB peak): every step ran or
skipped for a stated reason and the tree `evaluation/00…90` came out as designed. 00 run acceptance,
10 dense-reference deltas, 20 comparison with R17 (both manifest shapes read), 40 the incumbent-surface
evaluator on the compact frame (98.9 s; frozen-versus-recomputed maximum divergence 0.125 on the
population-normalised national rows, a signal to watch at S2), 50 downstream: T3 distributions
(112 s, 7.5 GB; 191 variables: 151 pass / 28 attention / 11 fail / 1 expected), T4 admin benchmarks
(65 s, 8.2 GB), T5 eight reforms (1,085 s, 12.1 GB), the #731 scorecard on the smoke (6 s, 3.3 GB) and
on R17 (59 s, 11.1 GB), 90 the summary with the pre-registered table (all fit flags red, as a
100-epoch smoke must be; footprint real: 187 MB H5, 22 min, 10.6 GB). Two findings: step 30 needs the
frozen scoring register as an input (candidate runs never write it; R14 compiled it separately) — the
command now takes `--scoring-registry`; and the #731 scorecard on the incumbent enhanced-FRS snapshot
fails inside policyengine-uk 2.94.0's dataset loader (`load_dataset … UnboundLocalError: data`), so the
incumbent leg of the three-way scorecard needs a snapshot that loader accepts (the R17 and smoke legs ran).

**Ruling (María, 2026-09-08, evening): fix the rent metric (option b) and make the size selection's budget
search stop on open-probability mass; then run S2.** Mass basis: `calibrate(..., budget_basis=
"open_probability_mass")` measures each probe by Σπ (the gates' expected open count) instead of the count
of not-fully-closed weights; the size refit uses it and records `selection_budget_basis`. On a 60-record
toy problem the mass basis lands Σπ 20.2 for a budget of 20 where the count basis gives count 19 with
mass 18.5. `solve.py` is an attested module, so the spec-engine identities are re-cut once more.
Commit 2f7eb945 (mass basis + re-pin: bundle digests am 924bc8c6… / be 08f63f50… / uk 4fd5fc1b…, loader
golden vector e155a712…, seed protocol 260446f1… and seed map f6834db8… with the regenerated US coverage
report, the H1 calibrate parity fixture, the US spec digest b9b7068f… in the multispine pool-tool test; 245
pin and parity tests pass).

Rent fix, commit 0cfe1928 (implemented by Codex from `.codex-work/PLAN-rent.md`, reviewed and verified
here: ruff clean, 137 tests): `uk_private_rent_mean_to_total` in `uk_runtime/ledger_targets.py`, applied
in `uk_local_target_surface` after the A15/A17 uprating, composes each `rent/private_rent` value as
`12 × mean monthly rent × the same authority's bound tenure/private_rent count`, keeps the mean and the
count in the row's metadata, refuses by area when the count is missing or non-positive, and records every
cell under `cross_grain.private_rent_mean_to_total`; the metric is unchanged; the census rationale for the
PIPR source is corrected (the feed carries the 2025 months the reference averages) and its committed
artifact regenerated. Dry run on spine-p with the composed surface: 314 cells composed (E06000001: £551.83
× 12 × 7,435.5 renters = £49.2m), 364 national rows, 20,794 × 792,690.

### S2 — the experiment: 55,000 households, `--selection-pi-hi 0.95 --epochs 2000 --skip-holdout`

Launched 2026-09-08 17:32Z on spine-p from the clean tree at 1079ef87, chained on smoke4's Logbook row
363c8c4f…, run `spine-p/f100-k15-h55000-e2000-p95-s42`, `/usr/bin/time -l`.

**REFUSED at the draw after 4.8 h** (17,247 s wall, 20,855 s user, 10.9 GB peak RSS; exit 1 at 22:20Z;
Logbook row d5a33c11… `disposition failed`, `phases_reached … targets_bound, error`, chained on 363c8c4f…;
the next run chains on d5a33c11…). Same message as S1/S1b: `degenerate boundary mass … adjust pi_hi or k`,
now with the feasibility receipt. Nothing beyond `run.log` and the error receipt was written: the driver
keeps no checkpoint between the dense solve and the draw, so the 2,000-epoch dense reference and the L0
probes are lost with the refusal.

The stub above said the exact count was "feasible by construction" on the mass basis. That was wrong,
and the numbers say why:

- The budget search stops as soon as `|Σπ − k| ≤ tol` with `tol = max(1, round(0.05 × k))` = 2,750 rows
  at 55,000 (`solve.py::_search_l0_lambda_for_budget`). It accepted Σπ = 54,834, i.e. 166 rows short of
  the request, at `selection_l0_lambda` 1.3335e-06. The mass basis changed what each probe measures, not
  the band that ends the search.
- The gates are near-binary at 2,000 epochs: 10,026 protected carriers at π = 1 exactly, 44,537 gates
  in [0.95, 1) averaging 0.9996 (54,446 above 0.99, 51,100 above 0.999), and the other 738,127 gates
  averaging 0.0004 (π p50 0.00016, p90 0.0015). Σπ over that tail is 291 at `pi_hi` 0.95.
- An exact-k draw at threshold h needs `(k − n_{π≥h}) × max_{π<h} ≤ Σ_{π<h} π`. At 0.95: 437 boundary
  draws × 0.947 = 414 > 291. At 1.0: 44,974 × 0.999999 > 44,808 (Σπ < k, so infeasible outright; the
  largest feasible k at `pi_hi` 1 is 54,834). Scan: 0.999 (3,900 draws / mass 3,745) no; 0.99 no; 0.98 no;
  0.9 (428 / 282) no; 0.8 (421 / 276) no; 0.7 (415 × 0.691 = 287 vs 272) no by 15 mass units; **0.5
  (405 × 0.463 = 188 vs 266) yes**, with 54,595 certainties. The feasible ceiling on this run's π vector
  lies in (0.5, 0.7).
- Had the search landed 166 rows above 55,000 instead of below, `pi_hi` 1.0 would have been feasible;
  which side of the request the ±5% band lands on is chance, not design.

What this means for the machinery: with the informed L0 at 2,000 epochs the selection is essentially
deterministic (≈54.6k gates open, ≈400 rows drawn from a 738k tail), so `pi_hi` only decides whether the
few dozen gates between 0.5 and 0.95 are certainties or boundary draws, and the exact-count draw is
feasible only when the requested threshold sits below that shoulder. Honouring an arbitrary `pi_hi`
would need the search to stop on the draw's feasibility condition at that threshold rather than on the
±5% mass band (a further `solve.py` change, pins move again, one to three extra probes at ~1 h each),
and a checkpoint of the dense solve and the gate probabilities before the draw would make a refusal
cost a refit rerun rather than 4.8 h. Both are for María to rule on; neither was started.

Options put to María (2026-09-09): (B) rerun S2 at `--selection-pi-hi 0.5`, no code change; the dense
solve and the search are seeded and deterministic, so this run's scan is the feasibility evidence for
the rerun (expected, not yet verified at scale); (A) the feasibility-aware stopping rule, then rerun at
0.95; (C) the pre-draw checkpoint, independent of A/B.

### Ruling (María, 2026-09-09) and the two fixes

**Ruling:** launch the 55,000 run at `--selection-pi-hi 0.5` now, build both fixes in parallel, and
launch the full 0.95 run on the fixed code as soon as it is ready, without a further go.

**P50 = `spine-p/f100-k15-h55000-e2000-p50-s42`** launched 2026-09-08 23:13Z from the clean tree at
e80112a4 (engine 2.94.0, spine-p ae83e307…, ladder 9c6d56b9…, feed 6fb700e), chained on S2's failed row
d5a33c11…, `--dataset-households 55000 --epochs 2000 --selection-pi-hi 0.5 --skip-holdout`. The draw's
feasibility at 0.5 on this pool is the S2 scan (405 boundary draws × 0.463 ≤ 266 mass; 54,595
certainties), which the seeded search should reproduce. Driven by `355-dataset-size/chain-p50-p95.sh`:
when P50 exits, the chain waits for `355-dataset-size/READY-p95` (a git ref), fast-forwards
`populace-877` onto it, and launches `f100-k15-h55000-e2000-p95b-s42` at 0.95 chained on P50's row. One
K=15 solve at a time (26 GB machine).

**Fix A — the search stops on the draw's own feasibility** (built on branch `uk-355-fixes`, worktree
`populace-877-fix`, while P50 ran). `exact_k_design_feasibility(pi, k, pi_hi)` in
`microcosm.calibrate.exact_k` is the draw's inequality without the draw (verdicts `feasible`,
`certainties_exceed_k`, `boundary_short_of_draw`, `boundary_mass_short`; 120 random polarised designs
agree with `select_exact_k` on every one). `calibrate(..., feasible_draw_pi_hi=h)` (mass basis only)
makes `_search_l0_lambda_for_budget` accept a probe only when the draw at `h` is feasible on its gate
probabilities, steers an infeasible probe like a count miss (short mass → smaller penalty, surplus
certainties → larger), prefers feasible probes when choosing the best run, and records every probe under
`options["budget_search"]` (`probes[]`, `selected_feasible`, `stopped_on`). The size selection passes its
`pi_hi`; the receipt carries `selection_budget_search`, and the feasibility scan carries each threshold's
verdict and `search_pi_hi`. A synthetic S2 (stub optimiser, 1,000 gates, budget 500: first probe 485 open,
inside the band, undrawable at 0.95) now continues to a drawable probe in five evaluations where the plain
mass basis stopped at the first; when nothing is drawable within the budget the closest run is still
returned and the draw refuses with its measurement. The feasibility scan and `_feasible_at` now use the
same helper, so the scan and the draw can never disagree.

**Fix C — checkpoint before the draw.** `refit_uk_dataset_size` is split into `select_uk_dataset_size`
(unsupported-row refusal, protected carriers, the search) and the draw + refit, which accepts an existing
`UKSizeSelection`. `uk_runtime/size_checkpoint.py` writes `size_selection_checkpoint.{npz,json}` (dense
weights and trajectory, initial weights, loss weights and scales, the selection's weights, gate
probabilities and trajectory, the protected mask; identity of the pool by household-id digest, of the
target surface by names + values digest, of the solve settings and input pins by the caller's identity
mapping, the options incl. the search receipt) and `load_uk_size_checkpoint` re-derives both results on the
resumed run's freshly compiled pool through the new `microcosm.calibrate.rebuild_calibration_result`
(compile, place weights under the recorded mass policy, rebuild diagnostics, recompute the closing loss and
refuse if it disagrees), refusing by name on identity, pool or surface drift. The doctrine solve takes
`size_checkpoint_dir` / `resume_size_checkpoint` / `checkpoint_identity`; the driver writes the
checkpoint into `--out` by default on size runs (`--no-size-checkpoint` opts out), `--resume-size-checkpoint
DIR` continues at the draw (`--selection-pi-hi` may differ; `selection_pi_hi` and
`selection_search_pi_hi` both recorded; `selection_reused` true), Logbook phases
`size_selection_checkpointed` / `size_selection_resumed`, manifest `parameters.size_checkpoint`,
`parameters.resume_size_checkpoint`, `solve.dataset_size.checkpoint.{written|resumed_from}`. Tests: a
synthetic run checkpoints, resumes byte-equal (support, weights, dense reference, selection CSV), re-draws
at another threshold, and refuses a changed epoch, a foreign pool, a moved surface, an overwrite.

`solve.py` and `exact_k.py` are attested: bundle digests am 09f6fffe… / be cbbabdbb… / uk d123633c…,
seed protocol 6174167d…, seed map 892435cc…, loader golden vector b556379f…, US spec digest 47038847…
(multispine pool-tool test), the regenerated US coverage report, the H1 calibrate parity fixture
(regenerated on the authoring platform, `fit.qrf` platform map reverted).

### P50 — 55,000 at `pi_hi 0.5`, 2,000 epochs: the first sized dataset at full fidelity

**Completed 2026-09-09 04:06Z, exit 1 = candidate blocked by the gate battery** (17,543 s = 4.87 h wall,
11.6 GB peak RSS; Logbook row 1b2a4d98…, chained on S2's d5a33c11…). Code e80112a4 clean (pre-fix; the
0.95 run carries the fixes). Every stage ran: dense joint solve (20,794 × 792,690), informed L0 search,
draw, refit, compact export, sidecars, battery, manifest.

- **Determinism at scale:** the search landed on S2's λ 1.3335e-06 and Σπ 54,834 exactly, and the scan
  matches S2's (54,595 certainties at 0.5, 405 boundary draws from 738,095 gates, mass 266, max 0.463).
  The seeded dense solve and search reproduce bit-for-bit on the same inputs.
- **Size:** requested = realized = 55,000; protected carriers 10,026; H5 **213 MB** (R17 dense 2,36 GB).
- **Dense reference** (spine-p, feed 6fb700e, 2,000 epochs): loss 0.2934 → **0.01508** (R17 0.01425 at
  1,500 epochs on spine-m and the old feed: within 6%, no D2 trigger); Kish ESS 125,324 (R17 129,236);
  local rows within 10% / 25%: 98.9% / 99.75%, 53 rows past 25% (9 constituency, 41 LA, 3 national).
- **Compact refit** (55k, HT baseline, stretch bound 10, realized max ratio 10.0): loss 1.0245 → **0.0347**
  (2.3× the dense; pre-registered "≤ 2× expected, > 5× red"); median |rel err| 0.0079 (dense 0.0080);
  within 10% / 25%: **93.0% / 97.7%**, **463 rows past 25%** (175 constituency, 288 LA); past-cap census
  403 → 4 (399 escaped). `max_target_scaled_change` 153 on council_tax/band_g E09000002 (target 50, dense
  2,408, compact 10,076): the handful of tiny council-tax band-F/G targets the dense already misses by 25–47×
  dominate every maximum; the substantive losses are support collapses on cells whose dense support was
  spread over many rows (uc_hh_3plus_children E14001096: 5,074 → 479; tenure/social_rent N09000005:
  14,545 → 2,149; hmrc self-employment amounts in E14001310 / E09000020 at −93%).
- **Area support (the headline):** constituencies 650, rows per area min/median 46/84 (1 below 50), Kish
  ESS min/median **11.4 / 59.2, 211 of 650 below the 50 floor (32%)**, distinct sources median 78 (2 below
  50); local authorities 361, rows 2/128, ESS 2.0/89.3, **42 of 361 below 50 (12%)**, sources 4 below 50.
  The pre-registered response ("breaches > 25% of areas → S4 at 110k") is triggered.
- **Six gates on the compact frame:** area_support FAILED (251 entries), target_fit FAILED (20 rows),
  per_family_fit / weight_ess / weight_ratio / ladder PASSED. Unreleasable by construction.
- Evaluation: light steps 00/10/20/90 run beside the 0.95 solve; 30/40/50 (engine, T3/T4/T5) when the
  machine is free.

**P95b = `spine-p/f100-k15-h55000-e2000-p95b-s42`** launched by the chain 04:06:10Z on the fixed code
e2667da9 (clean; fast-forward of `uk-dataset-sizes-355` onto `uk-355-fixes`), chained on P50's row
1b2a4d98…, `--selection-pi-hi 0.95`; writes `size_selection_checkpoint.{npz,json}` before the draw.

**P50 evaluation, light steps** (`evaluate_size_run.sh … --steps 00-run-acceptance,10-dense-reference,
20-vs-reference,90-summary`, 15 s, beside the 0.95 solve; heavy steps 30/40/50 deferred until no solve
runs). Run acceptance: every check passed. Pre-registered table (P50 / its dense reference / R17):

- dense reference loss 0.01508 / — / 0.01425 → watch (expected ≤ 0.014; red > 0.0285). Compact ÷ dense loss
  2.30 → watch (≤ 2 expected; red > 5). Maximum target-scaled change 153 → red (< 0.5), carried by the tiny
  council-tax band rows below.
- national rows within 10%: **323** / 353 / 340 → red (≥ 335 expected, red < 330); within 25%: 352 / 361 /
  356. Constituency share within 10%: 96.4% / 99.8% / 99.8% → watch; local-authority share: 90.7% / 97.7% /
  97.7% → watch (red < 90%). Rows past 25%: **475** / 53 / 45 → red (≤ 60).
- Kish ESS 33,672 / 125,324 / 129,236 → ok (20–40k expected). Maximum ÷ median positive weight **22.0** /
  389 / 400 → the compact frame is far tighter than the dense (HT baseline + stretch 10). Share of rows
  stretched above 100× the pool design 0.07% → ok.
- minimum constituency ESS 11.4 (R17 63.8); area-support breach share **25.0%** (253 of 1,011 areas) → red
  at the pre-registered 25% line; failed release-blocking gates 2 (area_support, target_fit; per-family,
  ratio, ESS, ladder passed). Frozen-vs-recomputed: not measured (step 40).
- wall 17,543 s / — / 11,122 s; peak RSS 11.6 GB / — / 11.1 GB; H5 **213 MB** / — / 2,356 MB.
- Paired with R17 on 20,473 common rows (94 target values moved with the feed; 321 rows only in P50 = the
  composed rent rows and the TFC/bus rows; 2 only in R17 = the two excluded UC bands): wins 9,541, ties
  168, losses 10,764; of R17's 42 red rows 20 are now green; new red rows are the tiny council-tax band F/G
  targets (E09000002 band G target 50 → 10,076; E08000028; E06000021 band F) and support collapses on
  spread cells (see above).

Reading: at 55,000 rows the exact count, the footprint (11× smaller H5) and the weight discipline (max ÷
median 22 vs 400) are real; the price is support — a third of constituencies under the ESS floor, four
families (ladder census_households, private_rent, council_tax, tenure) under 90% within 10%, and 475 rows
past 25% against 53 in the same run's dense reference. The dense reference itself is close to R17 on the
new inputs (loss +6%, national within 10% 353 vs 340), so no D2 is needed to explain the deltas: they are
size effects. The pre-registered response is S4 at 110,000 to locate the support floor; María's call.

**Progress tracker (María, 2026-09-09 ~08:40Z: "add the progress tracker to the branch").** The doctrine
solve takes a `progress` line sink; the driver writes it to stderr, so the run log now shows a timestamped
loss line every 100 epochs (and the last) of the dense solve, of each budget probe and of the refit, one
line per finished probe with its penalty, open mass, certainties, boundary draw and mass and its drawability
verdict, and one line when the search stops. The events (`budget_probe`, `budget_search_done`) ride the
calibrator's existing `progress_callback` seam beside the per-epoch events (`uk_runtime/solve_progress.py`;
tests at the calibrate, solve and driver levels). `solve.py` moved again, so the pins were re-cut a third
time (bundles am 906e0a55… / be 39619d06… / uk 635f5e39…, loader golden 864f9799…, seed protocol 1daa63de…,
seed map 8a8b353c…, US spec 0a5f4af6…, coverage report, H1 calibrate fixture). The running 0.95 solve
predates this and stays silent until its checkpoint; S4 / S3 will show their probes as they finish.

### P95b — 55,000 at `pi_hi 0.95` on the fixed code: the search trace

Checkpoint `size_selection_checkpoint.{npz,json}` written 12:03Z, 7.95 h after launch (dense solve ≈ 1.1 h,
then ten probes ≈ 41 min each). Dense closing loss 0.015083618… — bit-identical to P50's, the third
reproduction of the same dense solve. The feasibility-aware search (`feasible_draw_pi_hi 0.95`, mass basis)
spent its whole ten-probe budget and stopped `acceptable_within_tolerance` on the last one:

- λ 1e-3 (bracket mid-point): mass 10,072, only the 10,026 protected gates open, tail max 0.0006 → drawable
  but far below the request (steer: smaller penalty).
- λ 1e-5: 22,864, certainties 22,753, 32,247 places from a tail mass of 118 (max 0.925) → boundary mass short.
- λ 1e-6: 59,706, certainties 59,338 → certainties exceed k (steer: larger penalty).
- λ 3.16e-6: 40,825 → short. λ 1.78e-6: 50,223 → short. **λ 1.334e-6: 54,834, certainties 54,563, 437 places
  from mass 291 → short — S2's and P50's landing, now a rejected probe.** λ 1.15e-6: 57,276 → exceed.
  λ 1.24e-6: 56,043 → exceed. λ 1.29e-6: 55,407, certainties 55,124 → exceed by 124.
- **λ 1.3097e-6: mass 55,121, certainties 54,844, 156 places from a tail mass of 296 (max 0.9497 → 156 × 0.95
  = 148 ≤ 296) → drawable, within the band → selected.**

Reading: at 0.95 the drawable window on this pool is roughly Σπ ∈ [54,880, 55,130], about 0.5% of k, and the
bisection on log λ needed all ten probes to land in it (S2's ±5% band had stopped after the first in-band
probe). At 0.5 the first in-band probe was already drawable (P50). The threshold is therefore a cost knob
as much as a design knob: 0.95 cost 6.7 h of probes for a selection that differs from P50's by ~250
certainties and the tail draws. With the checkpoint, any further threshold on this pool is a re-draw.

**P95b completed 12:06Z, exit 1 = candidate blocked by the battery** (28,825 s = 8.0 h wall, 33,048 s user,
10.4 GB peak; Logbook row chained on P50's 1b2a4d98…; code e2667da9 clean). 55,000 exact; certainties
54,844 at 0.95 from λ 1.3097e-06 (Σπ 55,121), 156 tail draws; checkpoint arrays sha 65d0a099….

Pre-registered table (P95b / dense reference / R17; P50 in brackets): dense loss 0.01508 (same solve);
compact ÷ dense **2.07** [2.30] → watch; maximum target-scaled change 166 [153] → red (the same tiny
council-tax band rows); national rows within 10% **321** [323] / 353 / 340 → red; constituency share
within 10% **97.0%** [96.4%] / 99.8%; local-authority share **91.7%** [90.7%] / 97.7%; rows past 25% **448**
[475] / 53 / 45 → red; Kish ESS 34,526 [33,672]; maximum ÷ median positive weight **15.8** [22.0] / 389 /
400; stretch share above 100× design 0.03% [0.07%]; minimum constituency ESS 13.7 [11.4]; area-support
breach share **22.5%** [25.0%] (193 of 650 constituencies [211], 34 of 361 authorities [42]); failed
release-blocking gates 2 (area_support 225 entries, target_fit 20; the other four passed); H5 213 MB; median
|rel err| 0.0064 [0.0079]; by family within 10%: census_households 77.1 [76.0], private_rent 79.9 [77.1],
council_tax 85.4 [84.2], tenure 86.4 [84.9], hmrc 95.3 [94.1], age 96.8 [96.4], uc 99.3 [98.9] — the same
four families under 90%. Paired with R17 on 20,473 rows: wins 10,473 / ties 194 / losses 9,806 [9,541 /
168 / 10,764]; 18 of R17's 42 red rows now green [20].

**Selection stability across the threshold (P95b vs P50: one pool, one dense solve, two search landings
λ 1.3097e-06 vs 1.3335e-06):** only **44,860 of 55,000 rows are common (81.6%)**, 10,140 swapped each way;
44,811 rows are certainties in both. On the common rows the refit weights correlate 0.925, median relative
change 3.7%, 90th percentile 84%. Per-area ESS correlates 0.957 across the two selections (207 areas under
the floor in both, 46 only in P50, 20 only in P95b). On the 20,430 paired local rows P95b is better on
11,262 and worse on 9,168; 371 rows are past 25% in both, 92 only in P50, 65 only in P95b. Reading: the
near-binary gates make each selection deterministic in λ but not stable across λ — a 2% move of the
penalty swaps a fifth of the rows — so S3 (a different selection seed) will mostly measure the same thing
as this pair. The two datasets are statistically alike (fit, ESS, breach counts within a few percent) and
P95b is marginally better on every headline, at the cost of 6.7 h of probes.

Post-exit chain (`355-dataset-size/after-p95b.sh`): `populace-877` fast-forwarded onto `uk-355-fixes`
(0ab86823, tracker included); P95b light evaluation 00/10/20/90 done 12:07Z; heavy steps 30/40/50 (+ 90)
for P50 then P95b running with no solve on the machine.

### Heavy evaluation steps on both 55k runs (30/40/50, 12:07–13:00Z, no solve running)

**30 incumbent score: not measured.** `score_uk_local_candidate.py` refuses R17's frozen scoring register
("requires the frozen active reference count 19419, got 19105"): the register was compiled on the pre-#874
surface and the 55k runs bind 19,419 active local references (the composed rent rows and the re-pinned feed).
Scoring against the incumbent needs a register compiled on the 6fb700e surface; candidate runs do not write
one. Open item.

**40 frozen-vs-recomputed (engine re-resolved on the compact frame, 99 s):** 340 of 364 national rows
matched; maximum relative divergence **9.3% (P50) / 11.1% (P95b)** → the pre-registered red flag (> 5%)
fires. Eleven / twelve rows exceed 1%: `isc.private_school_students` (frozen 557k, recomputed 523k, −6%) and
the seven `voa.council_tax_stock.band_*` rows (recomputed +3% to +10% above frozen). The other 328 matched

> **Erratum (2026-09-15, microcosm#929).** The seven VOA band rows were not diverging: `frozen_vs_recomputed` keyed the incumbent-surface rows by `contract_target_id` and summed the Wales regional rollup into the England-pinned target (band A recomputed − frozen = 248,027, the Wales band-A rollup exactly; bands B and C likewise). Excluding the rolled-up rows, frozen and recomputed agree on every VOA row; the comparison now keys on bound rows only and refuses duplicates.
rows agree within 1%. This is the population-dependent-measure physics the plan said to size: the refit froze
the pool's contributions; the engine on 55k rows resolves the council-tax band imputation and the private
school flag differently.

**50 downstream (uk-candidate-eval under 2.94.0; "incumbent" = R17's dense H5 on spine-m and the old feed, so
input differences ride along):**

- T3 variable distributions vs R17: P50 135 pass / 45 attention / 9 fail / 2 expected; P95b 137 / 44 / 9 / 1.
  The same nine fail in both: `bsp_reported` (2.1× R17's total), `child_tax_credit_reported` (1.6×),
  `jsa_income_reported` (1.6×), `education_consumption` (1.6×), `winter_fuel_allowance_reported` (0.65×),
  `domestic_rates` (0.70×), `pension_credit` / `pension_credit_reported` (totals within 3%, shape), and
  `uc_deduction_combination` (no ratio) — small reported-benefit populations the 55k support cannot carry.
- T4 admin anchors: 3 pass (UC spend 73.9bn vs anchor 79.3bn, R17 75.6bn; UC caseload 6.70m vs 6.76m; carer
  element), 5 fail, 9 pending, differentiators 0/4 — every failure is shared with R17 (LCWRA caseload 6.4m vs
  anchor 2.4m, R17 7.6m; income tax 269bn vs 331bn, R17 268bn; population 68.4m vs 69.5m, R17 69.8m;
  multi-family households; the social-rented housing-element probe errors in the tool for both).
- T5 eight reforms (Δ government balance, bn): candidate 7 of 8 pass vs R17 6 of 8; both fail the UC taper
  reform (−35 vs −17 expected); P50 and P95b agree with each other to 0.1bn on every reform and with R17 to
  0.1–0.3bn on six (VAT +2pp 24.7 vs R17 21.1 vs expected 31.3).
- #731 scorecard (engine 2.94.0 on each H5; the incumbent enhanced-FRS leg still fails in the 2.94.0 loader):
  P50 / P95b / R17 — population 68.7 / 68.9 / 70.1m; **households 27.3 / 27.5 / 29.4m**; state pension
  **127.1 / 127.5 / 137.3bn**; income tax 287.7 / 287.6 / 286.3bn; UC 77.8 / 78.1 / 79.5bn; council tax
  46.2 / 46.5 / 48.6bn; poverty BHC 9.49 / 9.45 / 9.87%; top-1% net income share 7.49 / 7.49 / 6.72%;
  **Gini 0.350 / 0.351 / 0.375**. The pre-registered downstream flags (totals ≤ 2%, poverty ≤ 0.5 pp, Gini
  ≤ 0.005) fire on households, state pension, council tax and Gini.
- Footprint per file (wall, peak RSS): T3 114 s / 6.9 GB; T4 64 s / 9.0 GB; T5 1,166 s / 11.3 GB; the #731
  scorecard 6.6 s / 3.5 GB on the 55k H5 against 55 s / 11.7 GB on R17's — the eight-fold load-and-simulate
  gain the issue is about.

**The mechanism behind the downstream shifts is in the refit's own numbers, not in the inputs.** The
Horvitz–Thompson baseline sums to 29.25m households (the pool total); the free-mass refit ends at
**27.00m (P50) / 27.17m (P95b), −7% to −8%**, and the loss sits on specific groups: national
`ons.household_composition.lone_households_over_65` −36.5% (dense 0.7%), `couple_non_dependent_children`
−57.9% (dense 0.4%), `lone_parent_dependent_children` −50% (dense 23%), `ons.population.female_80_84` −31%
(dense 0.9%), `obr.state_pension` −18.5% (dense −13.4%), the state-pension income bands 9–22% over on the
compact. The selection keeps the rows the local targets need and sheds mass on older lone households, the
oldest women and couples with adult children — which is why the compact frames show fewer households, a
lower state-pension bill, lower poverty and a Gini 0.025 below the dense. National rows within 10%: 323 /
321 against the dense reference's 353.

Reading for the rulings: the 55k size reproduces national totals that are carried by many rows (income
tax, UC, reforms) to within a few percent of the dense, and loses the ones carried by few rows or by the
oldest households; the frozen-target refit cannot be told to hold the household total (mass is free) and
the stretch bound 10 on the HT baseline stops it from re-weighting the few survivors. S4 at 110,000 tests
whether doubling the support closes those gaps; a mass-conserving refit, or protecting a carrier per
national composition row the way the local rows are protected, are the two machinery levers if it does not.

### Rebase onto main and push (María, 2026-09-09 ~13:10Z)

`uk-dataset-sizes-355` rebased onto main a19582bd (#881 policy-year rule, #883 UC claimant contracts with
best-iterate retention in `_optimize`); 26 branch commits replayed (the three superseded re-pin commits fell
away), two `solve.py` conflicts resolved by keeping both sides (main's `selection_receipt` kwarg beside the
gate-initialisation splat; main's `iterate_selection_receipt` beside `budget_search`). Pins re-cut on the
combined tree: bundles am 6352283c… / be 03b57d3f… / uk 1f4c6fd8…, loader golden 66d468c0…, seed map
fbc9ca74…, seed protocol 28869ae8…, US spec 86ad8bb9… (the multispine test's live binding; its second
`spec_sha256` is a fixture constant), the coverage report and the H1 calibrate fixture; main's new
pre-best-iterate oracle attests `gates.py` bytes, re-pinned 379e4adb… → 14608a5e… because #355's gates
changes (per-record initial probabilities, protected mask) move it. Main now locks policyengine-uk
**2.97.0**; the run tree's venv was synced (`uv sync --locked --extra us --extra uk`). Green on the rebased
tree at 2.97.0: calibrate 236, UK build suites 100, registers/ledger 84, spec pins + graph parity 58,
multispine 187, engine re-run 120; ruff, format and the partition verification clean.

Consequence for the next runs: S4 and any later size run execute on engine 2.97.0 with main's best-iterate
dense solver, so their dense reference will not be bit-identical to the one P50 and P95b share (2.94.0,
e2667da9). Comparisons across that line carry an input-and-solver difference, as the R17 comparison does.

### Further evaluation legs (María, 2026-09-09 ~14:00Z: no S4 for now; evaluate the 55k datasets as far as possible)

- **Admin scorecard on one engine (2.97.0, the rebased lock), four files:** P50 / P95b / R17 / enhanced FRS
  7b0a06f0 — population 68.4 / 68.7 / 69.8 / 69.6 m; households 27.2 / 27.4 / 29.3 / 31.3 m; state pension
  120.0 / 120.4 / 129.7 / 125.5 bn; income tax 269.0 / 269.0 / 267.9 / 293.2 bn; taxpayers 38.2 / 38.2 /
  37.7 / 41.8 m; UC 73.9 / 74.3 / 75.8 / 75.3 bn; UC families 6.72 / 6.73 / 6.72 / 6.39 m; child benefit
  16.4 / 16.5 / 16.5 / 16.9 bn; pension credit 6.20 / 6.22 / 6.25 / 6.19 bn; council tax 43.7 / 43.9 / 45.9
  / 49.5 bn; poverty BHC 9.89 / 9.93 / 10.30 / 11.49 %; top-1 % share 7.47 / 7.48 / 6.70 / 6.95 %; Gini 0.350
  / 0.351 / 0.376 / 0.376. Load-and-simulate 7 s / 8 s / 95 s / 7 s. (The incumbent leg loads under
  2.97.0; it failed in the 2.94.0 loader.)
- **Incumbent score (step 30):** the register problem is solved — a scoring register with the 19,419 active
  local references is derivable from the run's own bound `local_target_registry.json` (ladder rows
  excluded, `geography_level` mapped from `area_type`; written through `TargetRegistry.to_json`, so the
  content hash holds). The scorer then refuses because "candidate rotated holdout declares no usable basis":
  the rule-1 score needs the rotated holdout, which both runs skipped by ruling. It is measurable only on a
  holdout run (S2h, 20–33 h).
- **T6 dashboard replays** (`uk-candidate-eval/scripts/run_final_gate.sh`, the uk-data venv at
  policyengine-uk 2.89.2, incumbent = R17): running for P50, then P95b; results appended when done.
- **Exclusions:** the A16 register holds 51 entries, none expired on 2026-09-09; five expire 2026-10-03
  (`ons.savings_interest_income` #866, `obr.housing_benefit` #867, the two SLC plan-2 borrower rows #868,
  `dwp.jsa_claimants` #869 — spine gaps at initialization, not retirement candidates), five 2026-11-25, 39
  2026-11-26, two 2026-12-08 (this PR's UC bands). Nothing was retired: the 46 `measure_excluded` national
  rows the surface evaluator could resolve on the compact frame are still 15–95 % off with three exceptions
  (`SINGLE_annual_payment_9_600_to_10_800` 3 %, `hmrc/state_pension_income_band_1_000_000_to_inf` 2 %,
  `LONE_PARENT_annual_payment_18_000_to_19_200` 11 %), and a retirement decision needs the dense pool
  measured on those rows, which no dense H5 on spine-p exists to do.

- **T6 dashboard replays (`run_final_gate.sh`, uk-data venv at policyengine-uk 2.89.2, R17's dense H5 as the
  comparison file): not measurable on this machine.** For both candidates the candidate legs ran (P95b
  childcare baseline: Tax-Free Childcare £0.97bn, UC childcare element £2.11bn net / £9.18bn gross,
  entitlements £5.45bn, total £8.67bn; the autumn-budget generator completed the candidate side) and every
  comparison leg on the 792,690-row dense file died: the vendored `uk-budget-data generate` failed on it and
  the RF-UC and childcare scripts were killed by the OS (out of memory, 26 GB machine). Roll-up "review" on
  all three. A T6 against the default enhanced-FRS snapshot (a9e52499) would fit; not run.

### eFRS legs and T6 replays (2026-09-09 14:48–16:07Z, `spine-p/efrs-legs/`)

María asked whether the eFRS could be added to the downstream section rather than assumed. The three
uk-candidate-eval instruments were re-run with the enhanced FRS (7b0a06f0) as the comparison file, and the
P95b/R17 pair re-run beside them, all on policyengine-uk **2.97.0** (the rebased lock), so one table holds
P50, P95b, R17 and the eFRS:

- **T5 eight reforms** (impact on the government balance, £bn; expected = uk-data's published expectation,
  tolerance 5): P50 / P95b / R17 / eFRS pass **7 / 7 / 6 / 8** of 8. The eFRS lands the UC taper reform at
  −23.2 (expected −17.2, in tolerance) where P50 / P95b / R17 give −35.4 / −35.9 / −34.3 — the one reform the
  microcosm family misses, dense and compact alike; VAT +2pp: eFRS 33.4 vs expected 31.3, P50 24.7, R17 21.1.
  On the six income-tax and benefit reforms all four files agree within 0.5bn.
- **T4 admin anchors**: P50 and P95b pass the same three (UC spend, UC caseload, carer element) and fail the
  same five as R17; against the eFRS the differentiator "multi-family households" reads 6.3m on the
  candidates and 2.9m on the eFRS (anchor 244k — all far off), income tax 269bn vs the eFRS 293bn (anchor
  331bn), population 68.4m vs 69.6m (anchor 69.5m).
- **T3 variable distributions against the eFRS**: P50 105 pass / 52 attention / 33 fail / 1 expected; P95b
  103 / 53 / 34 / 1 (against R17 the same runs fail 9): the extra fails are variables the microcosm family
  models differently from the eFRS (charitable gifts, corporate wealth, education spending, fuel spending,
  AFCS/BSP reported), not size effects.
- **T6 dashboard replays against the default eFRS snapshot (a9e52499), uk-data environment**: all three
  dashboards ran for both candidates, roll-up "review" (the replay's only non-failing verdict when metrics
  move). Autumn budget (vendored model 2.65.3): 1,302 metrics, P50 819 ok / 473 flagged (472 "move", 6
  benchmark regressions, 3 band migrations), candidate closer to or level with the benchmark on 25 of 37
  benchmarked metrics (bands close/moderate/divergent/none 6/10/13/8 vs the eFRS 3/7/19/8); combined
  budgetary impact FY 2026-27 −4.0bn vs the eFRS −7.9bn. RF UC (2.89.2): 27 metrics, 13 ok / 14 flagged,
  closer on 7 of 10; working-age adults in UC 8.69m vs the eFRS 7.93m (RF headline 8.5m). Childcare (2.89.2):
  35 metrics, 18 ok / 15 flagged / 2 excluded, closer on 8 of 17; Tax-Free Childcare 0.97bn vs 1.23bn (HMRC
  0.63bn), UC childcare element 2.13bn vs 1.84bn. P95b within a few percent of P50 on every dashboard.
  Controls pass (income-curve identical-control 20,100 cells).

All four files' numbers and the per-metric T6 tables are under `efrs-legs/` (JSON + markdown per
dashboard and run); the evaluation page carries the four-file tables.

### Rebase over #891 and Vahid's review of #877 (2026-09-09, evening)

Rebased onto main cc9c953c (#891 UC paid targets: the register counts 51 / 18 / 82 held; the UK bundle
digest moved). Vahid's pass at 37f81933: no blocking defects; three should-fix items, three questions,
nits. Landed:

1. **Doctrine drift on resume.** `_size_checkpoint_identity` carries `_doctrine_bounds()`; the doctrine
   solve asserts the restored `max_weight_ratio`, `mass` and `target_loss_cap` against today's doctrine
   after the load and refuses by name; the writing run's `code_pin` and `build_id` ride in the checkpoint's
   `provenance`, reported on resume, not compared.
2. **Stale checkpoint in `--out`** refused before the solve (`_refuse_stale_size_checkpoint`, beside the
   output-path guard), naming the files and the `--resume-size-checkpoint` way out.
3. **`gates.py` and `initialization.py`** join the seed-protocol implementation digest; the pre-best-iterate
   oracle runs with a frozen copy of the gate module (`fixtures/pre_best_iterate/gates_14608a5e.py`,
   loaded as `pre_best_iterate_gates`; the control asserts the oracle's `HardConcrete` is the frozen class).
   Consequences re-cut: seed protocol fd3e4b06…, seed map 87ba5053…, pointer inventory 2c0423a0…, field
   counts 42,156 / 9,772 and the `resolved_seed_protocol` claim 826 rows, mode and generation-0 counts,
   bundles am d983a5e6… / be 2a7d8348… / uk 2269bc28…, loader golden 57026e58…, US spec 35a02b6b…, the
   coverage report; the H1 calibrate fixture did not move.
4. **The draw's determinism stated:** `certainty_share` and `boundary_draws` in the size receipt (P50 99.3 %
   by threshold, 405 drawn; P95b 99.7 %, 156 drawn); manifest keys
   `realized_max_weight_ratio_vs_stretch_reference` (the refit's reference) and
   `realized_max_weight_ratio_vs_design` (the pool design weights).
5. **`zero_target_rows`** in the size receipt (0 on the licensed surface).
6. The two register changes stay, as their own paragraph in the PR body.
Nits: the written-checkpoint receipt carries no timestamp and no absolute path; the four `options` keys are
additive. Green on the rebased tree: calibrate 236 (with the frozen gates control), UK build suites 100,
local rowwise 52, registers 93, spec pins + parity + coverage tool, multispine 187; ruff and the partition
verification clean. Reply to Vahid drafted for María's go.

