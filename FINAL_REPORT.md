# Final report: health-coverage labels, round 2

## Outcome

Completed the generating repair on `battery-health-coverage-labels` and
committed it as `ad6cb75e`. The repair does not tune a gate, threshold, band,
seed, incidence, or comparator. It makes pooled ASEC health sources fail closed
and restores the declared ordered QRF target chain while preserving native ACS
outputs.

No build or push was performed. The next cold 1% build remains host-owned.

## Checkpoint diagnosis

The baseline and round-1 clone-0 frames contain the same 38,604 positive-weight
people: 4,311 ASEC rows (weight 80,851,529.27715749) and 34,293 ACS rows
(79,926,522.10879111). Their exact positive category masses are:

| Leg | ASEC in both checkpoints | Baseline ACS | Round-1 ACS | Ratio movement |
|---|---:|---:|---:|---:|
| IHS | 8 rows / 197,606.039624 | 12 / 46,033.921631 | 252 / 482,171.216580 | 0.235654 -> 2.468303 |
| Marketplace | 157 / 2,908,019.635846 | 872 / 2,360,680.784026 | 795 / 2,125,795.272305 | 0.821178 -> 0.739471 |
| non-Marketplace direct purchase | 107 / 2,028,850.769034 | 491 / 1,023,422.890043 | 528 / 1,054,016.416743 | 0.510273 -> 0.525526 |
| TRICARE | 45 / 807,558.923656 | 206 / 502,159.944901 | 153 / 385,243.838170 | 0.629021 -> 0.482568 |

IHS overshot because the ACS `HINS7 == 1` mapping was faithful, but the pooled
ASEC inputs were incomplete. Every sampled 2022 and 2023 `NOW_IHSFLG` value was
null, and the generic reader converted null to zero before applying `code == 1`
(`packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:373-379`,
`:555-560`). Corrected receipt-backed sources change sampled ASEC IHS from
8 rows / 197,606.039624 to 22 / 293,047.966998: 7 / 72,245.403261 in 2022,
7 / 23,196.524112 in 2023, and the unchanged 8 / 197,606.039624 in 2024.

The other movements were not a conserved relabeling. Round 1 shortened the
first boolean family from eight declared targets to six by dropping complete
native ESI and IHS from the model/draw sequence. QRF makes every completed raw
draw a predictor for the next target
(`packages/microcosm-fit/src/microcosm/fit/qrf.py:1195-1215`), so stable bank
indexes could not preserve Marketplace or later draws. Direct purchase and
TRICARE also follow the corrected narrow `NOW_CAID` Medicaid donor target. The
full transition and mass audit is in
`experiments/health_labels/ROUND2_DIAGNOSIS.md`.

## Generating repairs

- One exact nine-label ASEC mapping contract now binds each output to its named
  `NOW_*` field, including IHS -> `NOW_IHSFLG`, TRICARE -> `NOW_MIL`,
  Marketplace -> `NOW_MRK`, direct purchase -> `NOW_NONM`, and narrow Medicaid
  -> `NOW_CAID`
  (`packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:57-69`).
- Production pooling requires the complete nine-column, nonblank `{1, 2}`
  surface for every source year; partial, entirely absent, null, nonnumeric,
  and unsupported-code surfaces fail before pandas union
  (`packages/microcosm-build/src/microcosm/build/us_runtime/asec_pool.py:240-247`,
  `packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:441-487`).
- Any incomplete bounded transfer family now runs every declared target's
  latent model-draw step. The target-level null mask still writes only missing
  cells, so native HINS1/HINS7 outputs remain byte-identical
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:992-1043`,
  `:2383-2415`). Execution-contract schema 3 and target-bank materializer 3
  reject shortened-chain checkpoints.
- Regressions pin exact/overlapping mappings, partial/absent/invalid vintages,
  native preservation, peer-draw invariance, durable bank order, cold/warm
  equivalence, and version-2 bank rejection
  (`packages/microcosm-build/tests/test_us_puf_support.py:1168-1203`,
  `packages/microcosm-build/tests/test_us_asec_pool.py:180-233`,
  `packages/microcosm-build/tests/test_us_acs_transfer.py:804-935`,
  `packages/microcosm-build/tests/test_us_acs_transfer.py:1119-1143`).

## Honest comparator disposition

Marketplace, non-Marketplace direct purchase, and TRICARE still lack exact ACS
counterparts; the repair does not invent identities from `HINS2`, `HINS5`, or
`HIMRKS`. For any such detailed label `L`, the owner-only declared-absence
candidate remains:

```text
A_L(r) = B(r) * 1[person_support_channel(r) = "acs"]
B(r) = 1  =>  isna(L(r)) = A_L(r)
T_L(r) = B(r) * (1 - A_L(r)).
```

IHS is nominally matched, but if common-universe review still finds it
non-comparable, the same equation applies with `L = IHS`. These equations are
documented, not installed; owner adjudication is required. See
`experiments/health_labels/PROVENANCE.md` for the exact scope definition and
aggregate alternatives.

## Next host-owned cold 1% expectation

The exact three-year source paths and SHA-256 values are pinned in
`experiments/health_labels/ROUND2_DIAGNOSIS.md` and the read-only
`/Users/maxghenis/PolicyEngine/_buildo-runtime/inputs/asec-720/receipt_720.json`.

| Leg | Expected next ACS/ASEC ratio | Disposition |
|---|---:|---|
| IHS | `1.664408372482` | red; owner common-universe/absence adjudication |
| Marketplace | host measurement required | concept mismatch; corrected-IHS hybrid unavailable |
| non-Marketplace direct purchase | host measurement required | concept mismatch; corrected-IHS/narrow-Medicaid hybrid unavailable |
| TRICARE | host measurement required | concept mismatch; corrected-IHS/narrow-Medicaid hybrid unavailable |

Only IHS is numerically identifiable from existing read-only receipts.
Correcting the ASEC IHS donor target changes the chain before Marketplace, so
assigning the other three ratios without the host run would be calibration by
guess.

## Verification

- Build: 5,979 passed / 37 skipped across 6,016 collected tests; the 60
  subprocess-sensitive logbook and coverage-tool tests passed in an isolated
  invocation.
- Calibrate: 201 passed.
- Data: 275 passed / 1 skipped.
- Fit: 93 passed.
- Frame: 294 passed / 36 skipped.
- Ruff and `git diff --check`: clean.
- Generated US bundle: current at
  `0f9ddf0bb16469459a5c87ec1667d74b9a1425d5b3d1753f80e26dba8f000e10`.
- Spec-engine coverage: 41,380/41,380 configuration fields and 40/40 inventory
  checks.

Round-2 commits through the generating repair are `48ecbd16`, `15924b4e`, and
`ad6cb75e`. Nothing was pushed.
