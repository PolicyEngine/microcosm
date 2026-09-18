# Census households Chronicle and NI lookup receipts

## Step 0 — Chronicle feed re-pin acceptance

The UK consumer feed moved from Chronicle `6fb700e` to `ec7169b`.

| Field | Old pin | New pin |
|---|---:|---:|
| Fact rows | 128,717 | 131,450 |
| Facts SHA-256 | `6ae49d7d7ab297df25a0b9bfe2d6776827c672d284fbb360957fe8337089549f` | `4a50ee9568a01bbb57f73d927084ed6b4b9e52249b51a2338455874ae6e382b5` |
| Manifest SHA-256 | `dcda51d6496aea67f768a284e7955c7520e7c8b91e2bed3569f247567b7153f0` | `a95d0ee9f87f36947eaecdb3de29cf81a91e47ccaa822fed42da677eedca877f` |
| Manifest schema | `policyengine_ledger.consumer_artifact.v2` | `policyengine_ledger.consumer_artifact.v2` |
| Consumer-fact schema | `ledger.consumer_fact.v1` | `ledger.consumer_fact.v1` |

### Membership diff

| Surface | Status | Old | New | Change |
|---|---|---:|---:|---:|
| National | active | 415 | 415 | 0 |
| National | no_fact_at_or_before_period | 7 | 7 | 0 |
| National | signed_excluded | 8 | 8 | 0 |
| Local | active | 19,419 | 19,419 | 0 |
| Local | no_fact_for_area | 1,586 | 1,586 | 0 |
| Local | signed_deferral_compile_error | 1 | 1 | 0 |
| Local | signed_deferred | 513 | 513 | 0 |

No reference became unsupported, changed status, was added or removed, or moved value at either
grain. The local membership rows and compiled reference values are identical; only the recorded
feed name changed from `chronicle-uk-6fb700e/consumer_facts.jsonl` to
`chronicle-uk-ec7169b/consumer_facts.jsonl`.

The four national UC family-type references below remain active and retain exactly the same
December 2025 resolved values. Their candidate counts move from 9 to 27 because Chronicle
#251/#252 add two payment-indicator × child-entitlement cross packages that also match the
existing family-type selectors. Chronicle regenerated the aggregate fact keys for the resolved
rows, but the selected period and value did not move, so no selector pin or policy ruling is
required.

| Reference | Old matches | New matches | Resolved value (both pins) |
|---|---:|---:|---:|
| `dwp.uc.households_single_no_children` | 9 | 27 | 3,446,962 |
| `dwp.uc.households_single_with_children` | 9 | 27 | 2,226,220 |
| `dwp.uc.households_couple_no_children` | 9 | 27 | 284,345.55555555556 |
| `dwp.uc.households_couple_with_children` | 9 | 27 | 899,534.6666666666 |

The new NRS council household facts do not affect the pre-B1 local surface because
`ons.census.households` is not yet a contract target. They enable B1's record-set-spec selector.

## Increment A — NISRA constituency lookup

The rebuilt ladder retains 239,023 output areas, 650 constituencies, 361 local authorities,
and 28,060,832 households. Relative to the saved pre-change NPZ, only
`constituency_code` changed, at 10 Northern Ireland Data Zones; `metadata_json` also changed to
record the new publisher lookup provenance.

| DZ2021 | Inferred PCON24 | Published PCON24 | Households |
|---|---:|---:|---:|
| `N20000305` | `N05000007` | `N05000011` | 218 |
| `N20000483` | `N05000009` | `N05000017` | 231 |
| `N20000486` | `N05000009` | `N05000017` | 208 |
| `N20000585` | `N05000017` | `N05000009` | 161 |
| `N20000955` | `N05000001` | `N05000013` | 146 |
| `N20001038` | `N05000001` | `N05000003` | 268 |
| `N20001063` | `N05000001` | `N05000003` | 282 |
| `N20001937` | `N05000006` | `N05000018` | 150 |
| `N20002376` | `N05000009` | `N05000004` | 172 |
| `N20002971` | `N05000010` | `N05000018` | 239 |

Against the 18 NISRA PCON24 household facts copied verbatim into the test fixture:

| Ladder mapping | Sum absolute delta | Mean absolute delta | Max absolute delta | Net delta |
|---|---:|---:|---:|---:|
| Retired postcode inference | 3,562 | 197.888889 | 694 | +4 |
| Published DZ2021→PARLCON24 lookup | 126 | 7 | 16 | +4 |

The rebuilt ladder's Northern Ireland household total is 768,813.

## Increment B — Chronicle census-household binding

The new contract compiles exactly 1,011 `ons.census.households@…` references: 650 at
constituency grain and 361 at local-authority grain. The complete local compile has 20,430 active
references, 1,586 `no_fact_for_area` rows, one signed compile-error deferral, and 513 other signed
deferrals. The signed parity receipt reports 20,430 compiled rows, 18,099 drift rows, 4,442
fixture-only rows, 1,327 ledger-only rows, and 23,868 signed differences.

### OA-ladder diagnostic dispersion against Chronicle

The 1,011 cells produce the following diagnostic deltas (`ladder − Chronicle`). Scotland's
larger differences reflect NRS disclosure control at each output level; they are not used as
calibration targets.

| Country | Grain | Cells | Mean absolute delta | Max absolute delta | Net delta |
|---|---|---:|---:|---:|---:|
| England | Constituency | 543 | 7.390424 | 29 | +107 |
| England | Local authority | 296 | 9.347973 | 58 | +105 |
| Wales | Constituency | 32 | 6.40625 | 15 | +7 |
| Wales | Local authority | 22 | 7.590909 | 21 | +7 |
| Scotland | Constituency | 57 | 49.596491 | 157 | −557 |
| Scotland | Local authority | 32 | 58.5625 | 156 | −560 |
| Northern Ireland | Constituency | 18 | 7 | 16 | +4 |
| Northern Ireland | Local authority | 11 | 8.636364 | 30 | +3 |

### A15 per-grain uprating

Both Chronicle census grains uprate independently to the 2025 UK Ledger household control of
29,003,000.

| Grain | Census cells | Census household total | Factor to 2025 |
|---|---:|---:|---:|
| Constituency | 650 | 28,061,271 | 1.0335597414671631 |
| Local authority | 361 | 28,061,277 | 1.0335595204737118 |

The former ladder denominator used `29,003,000 / 28,060,832 = 1.0335759110777614`
(reported as `1.0335759`). The new per-grain Chronicle denominator therefore leaves the A17
rule unchanged while moving the factor applied to all **1,436 tenure cells** from
`1.0335759` to the local-authority factor `1.0335595204737118` (reported as `1.0335595`).
The runtime receipt records this explicitly as: `microcosm#887 (per-grain Chronicle denominator
supersedes #762 A15; A17 rule unchanged, factor moves from 1.0335759 to the LA-grain
1.0335595)`.

### Per-cell before/after evidence

[`docs/evidence/uk-887/census-households-cells.json`](../docs/evidence/uk-887/census-households-cells.json)
contains all 1,011 cells. For each area it records the raw ladder and Chronicle counts, the old
ladder-uprated and new Chronicle-uprated values, and both deltas. The source-only evidence file
has SHA-256 `0c219e643ea22d4f211f05cd8e642e7b250ab2de686ce0c9718bf91c91f649e0`.

The raw `ladder − Chronicle` summary is the table above. After applying the old ladder factor to
the ladder cell and the new grain factor to the Chronicle cell, the per-cell deltas are:

| Country | Grain | Cells | Mean absolute uprated delta | Max absolute uprated delta | Net uprated delta |
|---|---|---:|---:|---:|---:|
| England | Constituency | 543 | 7.643980 | 30.678373 | +489.544862 |
| England | Local authority | 296 | 9.752131 | 53.006703 | +492.656962 |
| Wales | Constituency | 32 | 6.628907 | 14.805645 | +29.017324 |
| Wales | Local authority | 22 | 8.042665 | 23.045174 | +29.315027 |
| Scotland | Constituency | 57 | 51.182024 | 161.586182 | −535.127831 |
| Scotland | Local authority | 32 | 59.924531 | 159.406569 | −537.673977 |
| Northern Ireland | Constituency | 18 | 7.065898 | 17.161507 | +16.565646 |
| Northern Ireland | Local authority | 11 | 8.734463 | 28.561668 | +15.701988 |

María ruled the licensed f100 re-measure out of this review round. The f001 dry run remains the
pre-merge evidence; an f100 re-measure is a licensed follow-up and was not attempted here.

## Independent verification (Claude, 2026-09-09, after the Codex pass)

Re-run from the worktree on the final tree (including the added `axiom` binding on
`ons.census.households` and the corrected candidate-tool help text), never from the Codex report.

- Ladder diff (baseline `9c6d56b9…` vs rebuilt `bed3f13d…`): every array identical except
  `constituency_code` at exactly the 10 Northern Ireland Data Zones listed above; NI sums against
  the 18 NISRA PARLCON24 facts: sum |Δ| 126, max 16, mean 7.0, net +4; NI total 768,813; the NI
  constituency layer metadata carries the workbook URL and sha `8e2e1f6d…`.
- Compile smoke on the `ec7169b` artifact (131,450 facts): local compile 20,430 specs, 0
  unsupported; 1,011 `ons.census.households@` specs (650 constituency: 593 held 2021→2025 + 57 held
  2022→2025; 361 local authority: 329 + 32); grain sums 28,061,271 / 28,061,277; `ons.households_total`
  2025 = 29,003,000; factors 1.0335597414671631 / 1.0335595204737118; no `external:` id in any spec.
- Membership diff HEAD → tree: national unchanged (415 / 7 / 8 of 430); local active 19,419 → 20,430,
  contract targets 33 → 34, candidates 21,519 → 22,530, other statuses unchanged.
- Joint dry run, f001 rung (`--sample-fraction 0.01 --n-clones 4 --seed 42 --dry-run`) on spine-m
  (`fb053cd2…`), ladder `bed3f13d…`, feed `ec7169b`: 1 m 49 s; matrix 19,617 rows (19,251 local +
  366 national) × 2,004 households; `census_household_uprating.applied = true` with the two grain
  factors above, 1,011 household cells and 1,436 tenure cells uprated, 0 skipped;
  `unbound_bridges` names `national_household_composition_partition_vs_census_households` (basis
  `reviewed_exclusion`, the three #757 composition cells); 38 inconsistencies in force, none
  mentioning `ons.census.households` (the `uk.household.count` exact groups are the seven VOA band
  controls, as before); `identity.targets.paired_ladder_sha256` and `household_dispersion` present;
  `ladder_assignment_provenance` replaces `ladder_target_provenance`; the plan JSON contains no
  `external:` string. Rung surface at f001 drops 62 zero-support census-household cells
  (`unreachable_check: deferred_to_build`), as every family does below f100.
- Targeted suites after the `axiom` binding edit: 186 passed, 0 failed
  (`test_uk_population_targets`, `test_uk_target_references` hermetic regeneration,
  `test_country_spec`, `test_spec_engine_country_bundles`). Full CI groups: see the PR body.
