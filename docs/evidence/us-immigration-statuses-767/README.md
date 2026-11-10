# US immigration statuses: default-pool stage receipts (microcosm#767)

PR #779 draws humanitarian and temporary-protection statuses in the US
immigration source stage (`us_runtime/immigration.py`).
[`docs/us-release-build-rule.md`](../../us-release-build-rule.md) asks a pull
request that changes how the dataset is built to carry a from-scratch build from
its own tree before it merges. These receipts run that one stage alone, on the
inputs the next certified build uses, at this PR's head and at main. Max decides
whether they stand in for the from-scratch run (d2729, on the terms of
d2647). Nothing here changes a build.

## What the script runs

`scripts/stage_receipt_pooled.py` runs the stage on the three Census ASEC person
files of main's default pool: the 2024, 2025 and 2026 ASEC, income years 2023 to
2025 (`asec_sources.ASEC_DEFAULT_POOL_INCOME_YEARS`). Max's ruling d2603 moves
the next certified build to that pool. The script:

- checks each archive against the SHA-256 that
  `us_runtime/education_assistance_source.py` pins;
- weights each person by the household's `HSUP_WGT` times 0.01 and rescales each
  file to one third of income year 2024's weighted population, as
  `asec_pool.pool_asec_sources` does when `build_pooled_asec_unit_frame` pools
  for target year 2024;
- keys the draws by income year, `PH_SEQ` and `PERIDNUM`, the pool's lineage
  columns;
- runs at seed 0 and period 2024, the `--target-year` of
  `tools/route_a/route_a.sh`;
- runs the composition gate on the output.

The receipts hold aggregates only. Each records the tree's commit and the
SHA-256 of its `immigration.py` and of the script.

## Receipts

| File | Tree | Result |
|---|---|---|
| `receipt.head.json` | this PR at `4b3b6a7a6`, its merge of main `764ae3aa7` | the gate passes with no failures |
| `receipt.main.json` | main at `764ae3aa7` | main draws no humanitarian status |
| `receipt.pre_sync_42bb166b.json` | this PR at `42bb166b1`, before it merged main `1acd884fe` | the stage refuses the 2026 ASEC's `PEINUSYR` code 29 |

The script reran main's and the pre-sync tree from `git archive <commit>
packages`, so those two receipts take their commit from `--source-commit`. Their
`immigration_module_sha256` values equal the SHA-256 of `immigration.py` at
those commits.

## Results

Weighted persons, in millions:

| Status | Main | This PR | Cited target |
|---|---:|---:|---:|
| Citizen | 300.244 | 300.244 | |
| Lawful permanent resident | 11.680 | 10.427 | |
| Undocumented | 13.071 | 12.891 | |
| DACA | 0.376 | 0.277 | |
| Cuban/Haitian entrant | 0.581 | 0.539 | |
| Paroled, one year | 0 | 0.398 | 0.442 |
| Refugee | 0 | 0.160 | 0.160 |
| Asylee | 0 | 0.155 | 0.155 |
| TPS | 0 | 0.860 | 0.858 |
| All | 325.952 | 325.952 | |

- Ten of the twelve positive-target draws emit within 3% of their targets.
  Nicaraguan TPS emits 3,193 against 2,910. The U4U Ukrainian parole draw emits
  112,934 against 158,000: it exhausts its candidates, Ukraine-born non-citizens
  in arrival codes 28 and 29, at 0.71. The composition gate's band runs from
  0.35 to 1.5 of each target. Deportation withheld carries a cited zero and
  emits none.
- Pew's broad unauthorized universe totals 14.574 million against its 14.0
  million anchor, and that universe's labor force 9.699 million against the 9.7
  million control.
- In both trees `ssn_card_type` is `NONE` exactly on the undocumented rows.

The totals move little, but the two trees label different rows undocumented.
Main's EAD spill counts a worker by prior-year wage or self-employment earnings
(`WSAL_VAL`, `SEMP_VAL`) and spills until the remaining undocumented rows match
the controls. This PR counts current labor-force status (`A_LFSR`, age 16 and
over) and spills until Pew's broad universe matches them. A spilled row takes an
EAD, and an EAD holder with no other mark is a lawful permanent resident. Row by
row (millions, main's label down the side, this PR's across):

| Main \ this PR | LPR | Undocumented | DACA | Cuban/Haitian | Paroled | Refugee | Asylee | TPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LPR | 8.406 | 2.415 | | | 0.228 | 0.160 | 0.155 | 0.315 |
| Undocumented | 2.021 | 10.059 | 0.172 | 0.104 | 0.170 | | | 0.545 |
| DACA | | 0.271 | 0.105 | | | | | |
| Cuban/Haitian | | 0.146 | | 0.435 | | | | |

Citizens keep their label in both trees (300.244 million). The table comes from
both runs' `--labels-out` files, which stay out of the repository because they
are record-level; `scripts/xtab_labels.py` rebuilds it from them.

## Rerunning

The archives are Census's public-use ASEC CSV releases at the URLs
`education_assistance_source.py` names. Put the three zips in one or more
directories, then run:

```bash
.venv/bin/python -I docs/evidence/us-immigration-statuses-767/scripts/stage_receipt_pooled.py . --archive-dir <dir> --out receipt.json
```

For another commit, extract `git archive <commit> packages` into a directory and
pass that directory and `--source-commit <commit>`. Add `--labels-out <file>.pkl`
to two runs, then rebuild the row-by-row table:

```bash
.venv/bin/python -I docs/evidence/us-immigration-statuses-767/scripts/xtab_labels.py main.pkl head.pkl
```
