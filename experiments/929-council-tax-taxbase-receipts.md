# microcosm#929 — council tax stock family on the councils' taxbase returns: receipts

Branch `uk-council-tax-taxbase-929`, cut from #906's head `fbeb4045` (region-tier activation) on
2026-09-15; re-targeted to `main` when #906 merges. Plan: `repos/uk-council-tax-family-plan.md`
(local). Every measurement below names the artifact it ran on.

## Part A — re-pin to chronicle `df35af7` (consumer_fact.v3; closes #920)

**Artifact.** Built from `PolicyEngine/chronicle` main `df35af7e7ccf689ad2a5b6e47ce33e99b8c9d3fb`
(tree `7973f020…`, identical to the `arch-data-265` worktree it was built in) with
`build-bundle --suite uk` → `build-consumer-artifact` (14 min): 266,390 rows, every row
`chronicle.consumer_fact.v3` (schema sha `bdb51e2a…`), facts `3e7d5a4f…`, manifest `51aab341…`.
Against the `474a0ae` pin (141,400 rows) the artifact adds chronicle #264's council taxbase
packages: MHCLG CTB 2023/2024/2025 (38,610 rows each: 14 lines × 297 rows × bands A-, A–H +
total), StatsWales CT1 FY2023–FY2026 (8,278 rows, bands A–I, `period_type: fiscal_year`,
integer opening year), CTAXBASE 2023/2024/2025 councils (32 S12 rows per measure per year).
All 244,226 sub-national rows carry `geography.name` (chronicle #266 / #267).

**Microcosm side.** `build.chronicle_epoch` declares `chronicle.consumer_fact.v3` beside v2
(the per-row schema id was never gated, but the identity registry is the authority on what
counts as a Chronicle spelling). `uk/chronicle_feed.json` moves commit, schema version, schema
sha, both digests and the row count. The six-step runbook ran with one ordering correction now
written into `docs/uk-chronicle-feed-repin.md`: the national generator refuses to compile against
vendored resources whose feed identity differs from the pin, so `tools/vendor_uk_ledger_facts.py`
runs before it on a fresh pin.

**Local surface (`tools/generate_uk_local_target_references.py`, 32 s).** 20,430 active,
1,586 `no_fact_for_area`, 513 signed deferred, 1 signed compile error — the #906 counts. Every
one of the 20,430 rows is byte-identical to the committed row; only the membership's feed label
and the register header (description text, the hierarchy providers/categories catalog the
current generator emits) move. The #920 refusal ("`E14001063` has no display label") is gone:
the hierarchy labels every constituency and authority from the fact's `geography.name`.

**National surface (`tools/generate_uk_target_references.py`, 84 s).** 595 active, 7
`no_fact_at_or_before_period`, 7 signed excluded — #906's counts; all 595 references
byte-identical, no value or period moves. One defect surfaced and is fixed in the same commit:
the nine `scotgov.council_tax_stock.*` rows first came back `multi_fact` (3 matches each)
because chronicle spells the vintage into the record-set id without a separator
(`scotgov.ctaxbase2025.chargeable_dwellings.scotland`; MHCLG does the same with `ctb2025`),
which the series-invariant key did not strip, so the 2023, 2024 and 2025 workbooks looked like
three series and latest-not-after refused. `_normalized_record_set_part` now strips a vintage
year glued to a word (`ctaxbase2025` → `ctaxbase`), the way it already reads `fy2025` and
`september2025`; the nine rows resolve `2025-09` again (values unchanged) with
`matched_fact_count_overall` 1 → 3. Unit tests pin the normalization and the three-vintage
resolution.

**Census, vendored resources, validation levels.** `uk_local_target_census.json` restates the
pin (44 lines, identity only, 11 sources). The nine vendored resources regenerate with identity
headers only (rows unchanged). `uk/local_validation_levels.json` `source_feed` restated by hand
(three values).

**Compile-parity receipts (6 min, both surfaces).** Incumbent 2025 national and local registers:
unchanged. Production 2023 register: 337 → 346 compiled, +9 `ledger_only` — the nine Scottish
stock rows now compile at 2023 from the September 2023 CTAXBASE workbook chronicle #264 added.

**Not yet on this branch.** The taxbase facts bind nothing until the family commits below
(no contract target selects `mhclg.council_taxbase.*`, `welshgov.council_tax_dwellings.*` or the
Scottish council record set). The baseline measurement on this re-pinned surface is Part B.

## Part B — the measurement design, and why there is no separate baseline run

**Ruling (María, 2026-09-15 evening).** No separate baseline run on this branch. The family is measured once,
on the family commit, with the D2 recipe shortened to 1,500 epochs (the loss had plateaued by epoch 1,100 in
every attempt: 0.0150 there against 0.0152 at 1,800), and compared against the D2 figures in the 10 September
report. Two dense solves plus the test batch had exhausted the 25.8 GB machine, and the baseline's other
purpose, certifying that the re-pin alone moved nothing but labels, is already receipted at the surface level in
Part A. The comparison therefore carries three known differences besides the family: spine-r (spine-q plus the
#903 `frs_relationships` stage; `council_tax_band` and `council_tax` identical), the df35af7 feed (labels only
on the national surface) and #906's region grain (the 81 region controls that D2 did not have).

**Recipe.** `tools/build_uk_rowwise_candidate.py --n-clones 15 --seed 42 --sample-fraction 1.0 --sample-seed 578
--epochs 1500 --learning-rate 0.15 --target-weight-rule grain_equal --expected-constituency-vintage 2024_pcon
--skip-holdout` on spine-r (`921612e88eefcf21511ac6ad5faf60a1514f577490188c6e39aec49cb4224f27`), the pinned OA
ladder (`bed3f13d3a82eea2d1f39248b71c0abf5ba6960a446ddd9415ae1dbcb7ae07fd`) and the df35af7 artifact (facts
`3e7d5a4f…`, manifest `51aab341…`). Output: `data/ukds/acceptance/929-council-tax/after-spine-r-df35af7-dense-e1500-s42`.

**Four refusals of the baseline attempts before the ruling: two stale inputs, one latent defect, one missing local secret; none a family question.**
(1) The ladder from populace-877 (`9c6d…`) disperses Northern Irish data zones over PARLCON24; the
pinned ladder above is the one #905 built. (2) spine-q predates #903 and has no
`household.ons_household_type`, so the provider refuses the composition rows; spine-r carries it.
(3) The local surface refused to compile: "two different control values at grain 'country' for leg
'S92000003'". The repro (`detect_cross_grain_inconsistencies` on the captured local frame) shows the
group: the ten `ons.household_composition.*` rows at K02000001 plus `scotgov.council_tax_stock.total`
at S92000003 as the winning rows, the region `total` rows below them, on the signature
`(concept uk.household.count, entity household, map_to unspecified, filters unspecified)`. The ten
composition targets bound their partition only through the policyengine `household_conditions`;
their measurement was the bare household count, so under #906's nearest-covering-control rule every
one of them was a candidate country control for every household total. Latent on `main` since #903
for the rowwise driver. Fixed in its own commit ahead of the family (`97ee7d54`): the filter
`uk.household.composition_type == <household_type>` sits on each measurement, mirroring the binding;
both registers are byte-identical (no reference carries the measurement); the repro compiles 20,430
surface rows; the contract tests pin the partition and assert the ten cells share no signature with
an unfiltered household total. (4) The first signed attempt solved all 2,000 epochs and then
refused at the terminal gate battery: `MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY` was not exported in the
launching shell (`unsigned full-scale calibration runs refuse to stage`); nothing but the gate report
is written before that step, so the solve was repeated with the key set. Every earlier dense run exported it at launch from
`data/ukds/acceptance/signing_key.txt` (fingerprint `66bf0e29…` in the D2 and R17 gate reports); it now
reaches every shell from `~/.zshenv`, so the reports on this branch carry the same `signing_key_sha256`.

**Result.** The baseline attempts never completed (the last one, at epoch 1,800 of 2,000, died with the app when the
machine ran out of memory); the family measurement is Part D.

## Part C — the family on the taxbase basis (surface receipts; measurement in Part D)

**Contract.** 244 → 271 targets (registry scope 210 → 220, profile scope 34 → 51; the incumbent's 90 `voa/council_tax/<REGION>/<band>` rows now map onto the `mhclg.*`, `welshgov.*` and `scotgov.*` national ids, and `welshgov.council_tax_stock.band_i` is an unmapped declaration, the only band the incumbent never carried): the 17 `voa.council_tax_stock*` rows retire; `mhclg.council_tax_stock.band_a..h/total` (country + region, `region_composition` from local authorities), `mhclg.council_tax_stock.by_area.band_a..h` (296 English authorities, `area_scope` prefix E), `welshgov.council_tax_stock.band_a..i/total` (W92000004), `welshgov.council_tax_stock.by_area.band_a..i` (22, prefix W) and `scotgov.council_tax_stock.by_area.band_a..h` (32, prefix S) join. Operands: England line 7 (+ A- on band A) − line 11 − line 15; Wales a1 − h7 − h8 with `dimension_values` by band (`dimensions: []` for the totals); Scotland identity on the council record set.

**National surface (`tools/generate_uk_target_references.py`, 3.5 min).** 605 active (595 − 81 VOA region cells + 81 composed MHCLG region cells + 10 Welsh country rows), 7 `no_fact_at_or_before_period`, 7 signed excluded. Every composed region cell resolves the 2025-10 partition; the nine band-A cells sum to 5,590,029 = the publisher's England row (line 7 5,826,342 + A- 17,102 − line 11 64,474 − line 15 188,941) to the unit, and the nine totals to 24,246,267 = 25,056,421 − 267,894 − 542,260. Yorkshire composes 15 authorities across 17 code spellings (Barnsley and Sheffield under both codes); `composed_member_count` counts authorities. Wales band A 194,973.29 = a1 204,448.29 − h7 5,648 − h8 3,827; band I 5,305; total 1,377,633.03 (the 2025-26 row, ruled 2026-09-15). Scotland's nine country rows unchanged.

**Local surface (`tools/generate_uk_local_target_references.py`, 58 s).** 20,885 active (20,430 + 176 Welsh A–H + 22 Welsh I + 255 Scottish + 2 PIPR cells), 1,237 `no_fact_for_area` (1,586 before: the Welsh, Scottish and Northern Irish VOA absences leave the roster with the nation scopes), 341 signed deferred (513 before: the 176 Welsh and the City band-A masks retire; Shetland band H joins under the band-H support reason), 1 signed compile error (pre-existing). Council tax: 2,511 active cells = 294 × 7 English A–G + 198 Welsh A–I + 255 Scottish A–H; the 296 English band-H cells stay under `council_tax_band_h_spine_support_absent`, and Shetland's band-H cell joins them (below). Barnsley (E08000016) and Sheffield (E08000019) bind their 2025-10 MHCLG rows through the crosswalk's declared aliases (before the alias they silently bound the 2024-10 rows under the roster code, visible only as 14 uprating holds), and their PIPR private-rent cells bind for the same reason; `ons.rent.private_rent`'s English absence mask drops to E06000053 and E09000001. Support-floor deferrals: 41 rows / 78 area-cells (24 / 43 before), the same two authorities.

**Four defects found by the re-base, all fixed in this branch.** (1) The series-invariant key treated `ctaxbase2025` / `ctb2025` as three series (Part A). (2) MHCLG's 2025 return files Barnsley and Sheffield under the April 2025 LAD codes; without an alias the roster cells bind the previous vintage silently, and the composed Yorkshire cell refuses (member count 17 vs 19 at the latest period). The crosswalk now declares `code_aliases` and the authoring, surface checks and hierarchy read them. (3) The first after-run refused at the surface: the band-H group's East Midlands leg was empty (every English
band-H cell is deferred on spine support) and the empty-leg licence derived from the membership covered
`mhclg.council_tax_stock.by_area.band_h` but not the Welsh and Scottish band-H authority targets that
share the signature; their `area_scope` never puts a cell under an English region, so the derivation now
licenses a leg holding none of a target's scoped areas outright, and a leg holding some once those are all
deferred (E12000001–3 escaped only because their band-H region controls are the three signed measure
exclusions). The surface compiles: 20,886 rows, 191 bound national ids. (4) The rowwise driver then refused one cell with a
nonzero target and zero household support: Shetland band H (`S12000027/council_tax/band_h`); Scotland's six raw
FRS band-H households leave that council with no band-H clone at K=15. It is signed deferred under the band-H
support reason (the other 31 Scottish and the 22 Welsh band-H cells draw at least one clone and bind), so the
surface is 20,885 rows and 2,511 council-tax cells.

**Registers.** `uk_local_target_census.json` (13 sources: the VOA source retires, three taxbase sources join; fence rule re-worded), `uk_data_target_parity.json` (concern ids follow the new target ids; evidence texts updated), `calibration_measure_exclusions.json` (the three band-H region exclusions re-named onto `mhclg.council_tax_stock.band_h@E1200000{1,2,3}`, tracking and expiry unchanged), `local_binding_adjudications.json` (fence re-adjudicated, "country wins" retired from the census note).
