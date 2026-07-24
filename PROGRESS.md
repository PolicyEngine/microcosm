# Progress

## State

Populace #515 interim donor-concept carve is complete on
`mortgage-donor-e19200-carve-2` (rebased onto current `origin/main` after the
#517 target-side remap and mortgage critical-register entries merged, and
after the JCT ALD reference and HT2 taxable-interest rebase lanes moved
`main`). The US `puf_tax_detail` donor now carves its E19200-lineage columns
to the SOI TY2015 Table 2.1 mortgage-only concept share before the QRF
learns levels and realized support. The root record-level ETL carve stays
open on populace#515. (The prior entry here — the #462 split-PR remediation
handoff — merged as #491 and nothing remains pending from it.)

## Done

- Traced both production HDF loaders (`tools/build_us_puf_support_base.py`)
  through `puf_tax_unit_donor_from_arrays` and into both QRF execution
  paths; the declarative `source_runtime.read_table` path is not the
  production PUF HDF loader.
- Confirmed the #486 `support_value_repairs` surface is a release-time level
  pin, not a donor-column concept-transform seam, so the carve lives at
  donor construction in `puf_support.py`.
- Added the symbolic TY2015 SOI mortgage-only share
  (`US_PUF_E19200_HOME_MORTGAGE_SHARE = 283_004_465 / 304_461_163`) and a
  centralized exactly-once carve for the three live donor columns plus the
  guarded `interest_deduction` compatibility alias.
- Sol round 1: bumped `PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION` to 2 (the
  checkpointed donor frame carries the carve; pre-carve checkpoints must
  not fit/draw under carved code); pinned the lineage tuple by exact
  membership with a nonzero investment-interest sentinel; documented the
  carve-before-aliases ordering and the raw-audit E19200 total-interest
  label.
- Sol round 2: the raw-target checkpoint loader now validates
  `schema_version` (written since v1, never checked); new regression pins a
  v1 root manifest rejected on load and run, and a v1 target checkpoint
  rejected under a valid v2 root.
- Suites green with worktree sources across PUF support, QRF chain, US
  plan, gates, US fiscal targets, and populace-data. Ruff format/check
  clean.

## Next

- PR review cycle, then merge; expect a rebuilt base/release before
  ratcheting the mortgage critical-fit bound from the interim 0.20 to 0.15
  (the condition documented in `us_critical_targets.py`).
- Root ETL carve (per-record, un-zeroes `investment_interest_expense`; JCT
  reform must then neutralize `deductible_mortgage_interest` instead of
  `interest_deduction`) remains open on populace#515.
- populace#516: investigation complete — the ≥$10M donor mortgage rows are a
  structural outlier cohort (3,066 rows; only 1,823 of them synthetic-id;
  `donor_realized` clip dead per #482) — whole-row screen at the donor seam
  queued behind this PR.
