# US immigration statuses: default-pool stage receipts (microcosm#767)

PR #779 draws humanitarian and temporary-protection statuses in the US
immigration source stage (`us_runtime/immigration.py`).
[`docs/us-release-build-rule.md`](../../us-release-build-rule.md) asks a pull
request that changes how Microcosm builds the dataset to carry a from-scratch
build from its own tree before it merges. These receipts run that one stage
alone, on the inputs the next certified build uses, at this PR's head and at
main. Max decides whether they stand in for the from-scratch run (d2729, on the
terms of d2647). Nothing here changes a build.

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
- gives each person the pool's lineage columns, `source_year` (the income
  year), `source_household_id` (`PH_SEQ`) and `source_person_id` (`PERIDNUM`),
  and leaves the tree's kernel to choose which of them key its draws;
- records, for each draw call, the key that reproduces its draws;
- runs at seed 0 and period 2024, the `--target-year` of
  `tools/route_a/route_a.sh`;
- runs that tree's own composition gate on the output.

`--draw-keys full` hands all three lineage columns to every draw call that
brings no key of its own, and `--draw-keys legacy` hands it `source_year` and
`source_person_id`. Two trees run on one key share their draws and differ only
in method.

The receipts hold aggregates only. Each records the tree's commit, the SHA-256
of its `immigration.py` and of the script, and every draw call's key.

## Receipts

| File | Tree | Draw keys | Result |
|---|---|---|---|
| `receipt.head.json` | this PR at `4f94e489a`, which last merged main at `764ae3aa7` | its own: income year, `PH_SEQ`, `PERIDNUM` | its gate passes with no failures |
| `receipt.main.json` | main at `764ae3aa7` | its own: income year, `PERIDNUM` | main draws no humanitarian status; its gate passes |
| `receipt.main_full_keys.json` | main at `764ae3aa7` | this PR's, through `--draw-keys full` | main's method on this PR's draws; its gate passes |
| `receipt.pre_sync_42bb166b.json` | this PR at `42bb166b1`, before it merged main `1acd884fe` | none: the stage refuses before its first draw | the stage refuses the 2026 ASEC's `PEINUSYR` code 29 |
| `receipt.head_medicaid_reporters.json` | this PR at `4f94e489a`, from the head run's labels | its own | 0.391 million of the 0.714 million people in the restored Medicaid statuses report Medicaid at interview |
| `receipt.stacked_supply.json` | this PR at `4f94e489a`, from `receipt.head.json` and the ACS 2024 1-year PUMS | none: it draws no rows | four draws run short of ACS candidates, and none ends below half its target in the worst case |

The script reran main's tree and the pre-sync tree from `git archive <commit>
packages`, so those three receipts take their commit from `--source-commit`.
Their `immigration_module_sha256` values equal the SHA-256 of `immigration.py`
at those commits.

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

## Medicaid at interview in the restored statuses

The reform-coverage probe `hr1_medicaid_humanitarian_eligibility_restoration`
restores the pre-H.R.1 Medicaid status list at 2027 law and counts the people
it enrolls. `scripts/medicaid_reporters.py` joins the head run's labels to the
ASEC's `NOW_MCAID` (Medicaid at interview), which `us_runtime/cps_carried.py`
maps to `has_medicaid_health_coverage_at_interview`, the column the Medicaid
take-up stage anchors on. `receipt.head_medicaid_reporters.json` holds the
aggregates, in millions of weighted people:

| Status | People | Report Medicaid at interview |
|---|---:|---:|
| Refugee | 0.160 | 0.121 |
| Asylee | 0.155 | 0.107 |
| Paroled, one year | 0.398 | 0.163 |
| All restored | 0.714 | 0.391 |

The stage draws no deportation-withheld or conditional-entrant rows. The
probe's 10,000-person floor is 2.6% of the 0.391 million reporters.

## The stacked pool's ACS residuals

The `--exact-k` arm of `tools/build_us_fiscal_refresh_release.py`, the only arm
that runs the improvement gate, takes a pool from
`tools/build_us_multispine_pool.py`, whose default pipeline stacks the ASEC pool
and the ACS 2024 1-year. Each arm carries half the ASEC pool's household mass
(`stacked_spine.DEFAULT_STACKED_HOUSEHOLD_MASS_SHARES`), and the immigration
stage labels the ASEC rows at the full pool's person mass, so those rows carry
about half of each draw's target. The ACS transfer's
`reconcile_us_immigration_humanitarian_transfer` then asks the ACS rows for the
rest, each draw's residual.

The reconciliation used to refuse any draw whose ACS candidates held less
weight than its residual. Now such a draw takes every candidate, as the ASEC
stage's draws do, and its receipt records the shortfall. The reconciliation
refuses only a draw that ends below 0.35 of its target, the floor of the
composition gate's band, and the stacked validator replays the same rule.
Operation Allies Welcome (OAW) parole now takes exact-year ACS arrivals from
2021 and 2022 only, the program's arrival years; its window used to stay open so
that the old rule would not refuse it.

`scripts/stacked_supply.py` projects each draw's ACS supply without building a
stacked pool. It takes the ASEC arm's labelled mass from `receipt.head.json`,
scaled to the stacked weights, and each draw's candidates from the ACS 2024
1-year PUMS through the kernel's own candidate rule, weighted as the frame
weights them and scaled by the ACS arm's weight scale, 0.478.
`receipt.stacked_supply.json` holds the aggregates, in weighted persons:

| Draw | Target | ASEC arm | ACS residual | ACS supply | Projected share | Worst case |
|---|---:|---:|---:|---:|---:|---:|
| Parole, Afghanistan (OAW) | 73,566 | 37,128 | 36,438 | 34,292 | 0.971 | 0.971 |
| Parole, Ukraine (U4U) | 158,000 | 56,467 | 101,533 | 57,046 | 0.718 | 0.718 |
| Parole, Nicaragua | 93,070 | 46,971 | 46,099 | 27,188 | 0.797 | 0.797 |
| Parole, Venezuela | 117,330 | 59,215 | 58,115 | 152,918 | 1.000 | 1.000 |
| Refugee | 160,000 | 80,750 | 79,250 | 363,809 | 1.000 | 0.505 |
| Asylee | 155,000 | 78,227 | 76,773 | 2,292,045 | 1.000 | 0.505 |
| TPS, Venezuela | 605,015 | 305,344 | 299,671 | 338,594 | 1.000 | 0.710 |
| TPS, El Salvador | 170,125 | 85,860 | 84,265 | 114,363 | 1.000 | 1.000 |
| TPS, Honduras | 51,225 | 25,853 | 25,372 | 22,197 | 0.938 | 0.938 |
| TPS, Nicaragua | 2,910 | 1,469 | 1,441 | 6,960 | 1.000 | 1.000 |
| TPS, Nepal | 7,160 | 3,614 | 3,546 | 16,986 | 1.000 | 1.000 |
| TPS, other designated | 21,215 | 10,707 | 10,508 | 141,413 | 1.000 | 1.000 |

- Four draws run short of candidates: U4U, Nicaraguan parole, Honduran TPS and
  OAW. Under the old rule, U4U, the second draw in manifest order, would have
  stopped every stacked build on these inputs. Under the new rule the four end
  at 0.718 to 0.971 of their targets.
- The worst case lets every earlier draw take as much of a draw's candidates as
  it can, up to its own residual. It holds for parole and TPS draws, which take
  any non-citizen SSN code. Refugee and asylee draws take only rows the transfer
  imputes as `OTHER_NON_CITIZEN`, which PUMS cannot show, so their supply column
  overstates and their worst case counts the ASEC arm alone.
- The ASEC arm alone carries 0.505 of each target the ASEC stage met, and 0.357
  of U4U's. No draw falls below the 0.35 floor in the worst case: the lowest is
  0.505.
- The projection scales each arm by one factor. A build stands or falls on the
  reconciliation itself and on the pool's terminal gates, which run the
  composition gate on the stacked pool (`tools/build_us_multispine_pool.py`),
  not on this projection.

## Why the two trees label different rows

The totals move little, but the two trees label different rows. Two changes in
this PR move rows, and `receipt.main_full_keys.json` separates them: it runs
main's method on this PR's key. Comparing it with main's own run isolates the
key, and comparing it with this PR's run isolates the method.

**The draw key.** Each draw hashes a string built from the row's lineage. Main
builds it from the income year and `PERIDNUM`. This PR adds `PH_SEQ`, so two
rows that share an income year and a `PERIDNUM` but not a household draw apart
where main's key would tie them. In these
three files the income year and `PERIDNUM` already tell all 421,119 rows apart,
so the added column neither ties nor separates any rows. It still changes every
string the draws hash, and so every row's draw. Main's EAD spill takes
candidates in draw order until it has spilled its excess, so new draws hand the
EADs to other rows. On this PR's key, main's own method moves 2.400 million
people from undocumented to LPR and 2.392 million back, and 0.220 million from
undocumented to DACA and 0.244 million back. No status total moves by more than
0.025 million.

**The method.** This PR draws the humanitarian statuses after the legal-status
indicators and before the EAD spill. Refugee and asylee draws take only rows
that an indicator marks documented; parole and TPS draws may also take rows from
the residual pool, and those rows take an EAD. Main's spill counts a worker by
prior-year wage or self-employment earnings (`WSAL_VAL`, `SEMP_VAL`), spills
workers and then students, and stops when the workers or students left in the
residual pool match their controls. This PR's spill counts a worker by current
labor-force status (`A_LFSR` 1 to 4, age 16 and over), spills students and then
workers, and stops when the workers or students in Pew's broad unauthorized
universe match the controls. That universe still counts a DACA or Cuban/Haitian
cohort row after it takes an EAD, and every parole and TPS row. In both trees a
spilled row takes an EAD, and a documented non-citizen with no other mark is a
lawful permanent resident. With both runs on this PR's key, the method moves
0.319 million people from undocumented to LPR and 0.706 million back, and 0.031
million from undocumented to DACA and 0.105 million back. The humanitarian
draws take 0.874 million of main's LPR rows and 0.700 million of its
undocumented rows.

The two effects do not add, because both reshuffle one pool. Between LPR and
undocumented, the key alone moves 4.793 million people, the method alone 1.025
million, and the two together 4.436 million.

Row by row, in millions, with the first run's labels down the side and the
second run's across. Main against this PR, each on its own key:

| Main \ this PR | LPR | Undocumented | DACA | Cuban/Haitian | Paroled | Refugee | Asylee | TPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LPR | 8.406 | 2.415 | | | 0.228 | 0.160 | 0.155 | 0.315 |
| Undocumented | 2.021 | 10.059 | 0.172 | 0.104 | 0.170 | | | 0.545 |
| DACA | | 0.271 | 0.105 | | | | | |
| Cuban/Haitian | | 0.146 | | 0.435 | | | | |

The key alone, main on its own key against main on this PR's key:

| Main \ main on this PR's key | LPR | Undocumented | DACA | Cuban/Haitian |
|---|---:|---:|---:|---:|
| LPR | 9.288 | 2.392 | | |
| Undocumented | 2.400 | 10.314 | 0.220 | 0.136 |
| DACA | | 0.244 | 0.131 | |
| Cuban/Haitian | | 0.123 | | 0.458 |

The method alone, main on this PR's key against this PR:

| Main on this PR's key \ this PR | LPR | Undocumented | DACA | Cuban/Haitian | Paroled | Refugee | Asylee | TPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LPR | 10.108 | 0.706 | | | 0.231 | 0.160 | 0.155 | 0.327 |
| Undocumented | 0.319 | 12.013 | 0.031 | 0.012 | 0.167 | | | 0.533 |
| DACA | | 0.105 | 0.246 | | | | | |
| Cuban/Haitian | | 0.067 | | 0.527 | | | | |

Citizens keep their label in every run (300.244 million). The tables come from
the three runs' `--labels-out` files, which stay out of the repository because
they hold person records; `scripts/xtab_labels.py` rebuilds the tables from
them.

## Rerunning

The archives are Census's public-use ASEC CSV releases at the URLs
`education_assistance_source.py` names. Put the three zips in one or more
directories, then run:

```bash
.venv/bin/python -I docs/evidence/us-immigration-statuses-767/scripts/stage_receipt_pooled.py . --archive-dir <dir> --out receipt.json
```

For another commit, extract `git archive <commit> packages` into a directory and
pass that directory and `--source-commit <commit>`; add `--draw-keys full` to
run that tree on this PR's key. Add `--labels-out <file>.pkl` to two runs, then
rebuild a row-by-row table:

```bash
.venv/bin/python -I docs/evidence/us-immigration-statuses-767/scripts/xtab_labels.py main.pkl head.pkl
```

Count the head run's restored-status Medicaid reporters from its labels:

```bash
.venv/bin/python -I docs/evidence/us-immigration-statuses-767/scripts/medicaid_reporters.py head.pkl --archive-dir <dir> --out receipt.head_medicaid_reporters.json
```

Project the stacked pool's ACS supply from the head receipt. Put the ACS 2024
1-year PUMS files `csv_hus.zip` and `csv_pus.zip`, at the URLs
`us_runtime/acs_2024_1yr_sources.json` pins, in one directory; the script also
reads the three ASEC zips for the pool's masses:

```bash
.venv/bin/python -I docs/evidence/us-immigration-statuses-767/scripts/stacked_supply.py receipt.head.json --acs-dir <acs-dir> --archive-dir <dir> --out receipt.stacked_supply.json
```
