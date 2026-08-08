# Round 5 final report: earnings-universe verification blockers

## Outcome

All four Round 4 blockers and auxiliary A1 are fixed on
`smoke-gapfill-ordering-fix`, starting from the required
`b4ecd763b21507f659d1122bfbca6fdb7d238fee`. The implementation is local and
committed. Nothing was pushed; no GitHub operation, network access, root
journal, state shelving, or 1% smoke was used.

The design decision is cross-arm consistency: the certified ASEC produced
frame uses numeric zero below the age-15 earnings universe and retains
all-child tax units with zero earnings sums. The stacked ACS arm now matches
that contract through named, counted universe-rule applications—not through a
generic null fill. Raw ACS WAGP/SEMP blanks remain unchanged, eligible nulls
fail at the original pre-coercion boundary, and allocated earnings can reach
only age-15-plus people.

The verified 131-target authority digest remains
`6907790735d3645db0f272144beab666f536e89c85ec2eb417444bee803a83ba`.
The 48 early targets, 70 late targets, producer schedule, and
CANONICAL/NON-CANONICAL boundaries remain unchanged and pass.

## ASEC-arm consistency finding

The established producer maps `WSAL_VAL` and `SEMP_VAL` at
`packages/populace-build/src/populace/build/us_runtime/cps_carried.py:150-155`.
Its `_source` coercion at `cps_carried.py:355-362` converts a missing source
value to numeric zero before the produced leaves are written.

The locally certified artifact audited was:

`/Users/maxghenis/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/fa2f8104e62a0f1cce400a495c5487d1a10544cb/populace_us_2024.h5`

Its release manifest identifies Build J and the certified package versions at:

`/Users/maxghenis/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/fa2f8104e62a0f1cce400a495c5487d1a10544cb/releases/populace-us-2024-buildj-sparse-rmloss100-75d5add-20260710T094201Z/release_manifest.json:10-20`

Direct artifact audit result:

- ASEC contains 66,004 people, including 15,509 under age 15. Every under-15
  `WSAL_VAL`, `SEMP_VAL`, mapped employment, and mapped self-employment value is
  numeric zero.
- All seven ASEC tax units containing only under-15 people are retained and
  have zero earnings sums.
- The PUF-detail ASEC clone contains 21,737 under-15 zeros and retains 22
  all-child units with zero sums.

The committed evidence and decision are documented at
`docs/us-multispine-operator-ordering.md:145-190`. This supports the selected
ACS semantics: receipted mapped universe zero, retained all-child recipient
units with receipted zero predictors, raw PUMS blanks preserved, and no
out-of-universe allocation.

## Blocker receipts

| Blocker | Resolution | Primary file:line receipts |
|---|---|---|
| C1 | The ACS contract names both age-15-plus rules; only null mapped cells below age 15 become zero, pre-existing values are refused, raw blanks remain, and exact person/unit/digest receipts are produced. All-child units remain recipients. Person allocation uses an age mask and cannot fall back to an under-15 first person. | `acs_income_universe.py:79-113`, `acs_income_universe.py:116-206`, `acs_income_universe.py:250-405`, `stacked_spine.py:5252-5298`, `puf_support.py:1975-2024`, `puf_support.py:2772-2814`, `puf_support.py:3287-3303` |
| C2 | Primary-QRF root/target schema, outer stacked materializer, and canonical stacked authority are all v6. Base identity binds primary-QRF schema 6, the ACS universe contract, and the QBI mutation contract. Every v1–v5 variant is explicitly rejected. | `puf_qrf_chain.py:77-80`, `puf_qrf_chain.py:153-168`, `puf_qrf_chain.py:227-249`, `puf_qrf_chain.py:623-637`, `stacked_spine.py:1670-1673`, `tools/build_us_multispine_pool.py:247-251`, `tools/build_us_multispine_pool.py:997-1058` |
| C3 | QBI receipt v2 has an exact schema, types, nonnegative counts, SHA-256, nested universe and preservation equations. Generation recomputes the deterministic transition and binds an independent immutable authority. Live frame, driver surface, fixed point, universe, inventory, and authority are authenticated at derive, simulated emission/resume, durable write/load, both manifest constructors, and both publication entries. | `qbi_inputs.py:158-215`, `qbi_inputs.py:460-574`, `qbi_inputs.py:577-697`, `qbi_inputs.py:700-851`, `multispine_pool.py:1581-1635`, `multispine_pool.py:2064-2172`, `tools/build_us_multispine_pool.py:1335-1455`, `tools/build_us_multispine_pool.py:1685-1789`, `tools/build_us_multispine_pool.py:2530-2588`, `tools/build_us_multispine_pool.py:3031-3047`, `tools/build_us_multispine_pool.py:3133-3145`, `tools/build_us_multispine_pool.py:3315-3331`, `tools/build_us_multispine_pool.py:3412-3424` |
| C4 | The strict surface resolves named universe authority, then calls `_require_complete_recipient_predictor_sources` before aggregation. Eligible nulls retain the exact `missing values before coercion` diagnostic plus rule ID. Both 14+ and 16+ mutations reject while naming `acs_2024_pums_wagp_age_15_plus`. Nonfinite aggregate features fail by predictor and count. | `puf_support.py:2677-2751`, `puf_support.py:3036-3167`, `puf_support.py:3197-3208`, `test_us_stacked_spine.py:939-999`, `test_us_stacked_spine.py:1090-1186` |
| A1 | The changelog now states the exact mapped-zero/raw-blank semantics, all-child policy, age-masked allocation, eligible-null and NaN/+inf/-inf rejection, complete QBI authentication lifecycle, independent authority, v6 surfaces, and v1–v5 refusal. | `changelog.d/smoke-gapfill-ordering-fix.fixed.md:3-5` |

Key C1 regression receipts are at
`packages/populace-build/tests/test_us_stacked_spine.py:798-901` (mixed and
all-child exact counts), `:904-914` (unreceipted pre-existing zero),
`:1002-1087` (under-15 first person and all-child allocation), and `:2981-3050`
(raw blank plus mapped-zero stacked receipt).

Key C3 tamper receipts are at
`packages/populace-build/tests/test_us_qbi_inputs.py:329-355` (zero SHA,
negative count, extra field, and regenerated mutated output), `:433-553`
(rebound driver, alternate fixed point, and inventory laundering),
`packages/populace-build/tests/test_us_multispine_pool.py:635-755` and
`:2156-2191` (emission/resume/derive), and
`packages/populace-build/tests/test_us_multispine_pool_tool.py:3986-4348`
(durable checkpoint, manifest, publication, and regenerated-receipt attacks).

## Twelve verifier probes

| # | Probe | Final result |
|---:|---|---|
| 1 | Widen 15+ to 14+ | PASS — rejects and names `acs_2024_pums_wagp_age_15_plus`. |
| 2 | Narrow 15+ to 16+ | PASS — rejects and names `acs_2024_pums_wagp_age_15_plus`. |
| 3 | Eligible age-15-plus mapped/raw null | PASS — fails at `_require_complete_recipient_predictor_sources` with `missing values before coercion` and the rule ID. |
| 4 | `+inf` eligible PUF feature | PASS — rejected before fitting with named predictor and count. |
| 5 | Remove raw WAGP/SEMP authority | PASS — both parameterized twins reject. |
| 6 | Cross-grain predictor collision | PASS — ambiguous person/tax-unit source rejects. |
| 7 | Tamper QRF universe digest/live feature binding | PASS — root/finalization binding rejects mutation. |
| 8 | Forge QBI receipt | PASS — zero SHA, negative count, extra field, and derive-stage forgery reject. |
| 9 | Mutate QBI output and regenerate receipt | PASS — unit and derive-stage live-transition bindings reject it. |
| 10 | Present old strict two-control v5 payload | PASS — root, target, materializer, and self-consistent authority reject every v1–v5 payload; 15/15 parameterized cases pass. |
| 11 | NON-CANONICAL, renamed, stripped, grafted, or forged seams | PASS — 15/15 maintained seam cases reject. |
| 12 | Insufficient-support starvation | PASS — completeness/validity and sign-hole failures precede support classification; 2/2 cases pass. |

The exact final replay totals 48 passing maintained cases:

- Probes 1–6: 8 passed.
- Probes 7–9: 8 passed.
- Probe 10: 15 passed.
- Probes 11–12: 17 passed.

## Proof battery

Focused receipts:

- Final C1/C4 universe/allocation selection: 12 passed.
- QBI receipt, multispine runtime, and pool-tool suites: 31 + 49 + 135 =
  215 passed.
- Final verifier-probe selection: 48 passed.
- Standalone populace #583 source-spine-blindness guard: exactly 495 passed.

Full workspace battery, run in foreground and partitioned by file:

| Chunk | Result |
|---|---:|
| `populace-frame`, `populace-fit`, `populace-calibrate`, `populace-data` | 795 passed, 36 skipped |
| Generic `populace-build` tests | 662 passed |
| UK `populace-build` tests | 759 passed, 18 collected-function skips, plus 1 collection-level module skip |
| US files 1–40 | 987 passed, 1 skipped |
| US files 41–80 | 1,270 passed, 2 skipped |
| US files 81–108 | 1,092 passed |
| **Total** | **5,565 passed; 57 collected-function skips; 1 collection-level module skip** |

Exact function total: **5,622 collected test functions**, with zero failures or
errors. Across terminal summaries, pytest displays 5,565 passed and 58 skipped
outcomes because the separate UK module skip is also reported as a skip.

Quality receipts:

- `.venv/bin/ruff check .` — pass.
- Ruff format-check across all 14 Python files changed since `b4ecd763` — pass.
- `git diff --check b4ecd763..HEAD` — pass.
- Worktree clean before final report generation.

The 1% smoke was deliberately not run, as instructed.

## Commit receipts

Fourteen coherent implementation/proof commits precede this report, from
`0c94286e` through `9accba68`. They separately record initial progress, C1,
C2, C3, documentation/A1, independent C3 hardening, source-spine blindness,
test-fixture alignment, each workspace-proof milestone, formatting, and the
final probe replay. `PROGRESS.md` has remained committed throughout.

## Gradeable next-smoke prediction

**Prediction: PASS, not HOLD.** At the next 1% smoke, the former 2,998 recipient
tax units touching 5,294 under-15 people should cross strict preflight without
an unreceipted structural-null coercion: the 2,980 mixed-age units aggregate
only eligible earnings, the 18 all-child units remain recipients with
rule-owned zero predictors, and no under-15 person receives an allocated
amount. The expected next observable boundary is initialization/resume of the
65-target primary-QRF chain under schema v6 with the universe and feature
digests bound. Any genuinely eligible null, nonfinite feature, stale v1–v5
checkpoint, or forged QBI transition remains terminal by design.

VERDICT: READY FOR VERIFICATION
