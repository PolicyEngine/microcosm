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

**National surface (`tools/generate_uk_target_references.py`, 84 s).** 603 active, 7
`no_fact_at_or_before_period`, 7 signed excluded — main's counts after #906 and #921; all 603 references
JSON-equal (about 315 `period_basis_note` strings re-escaped, hierarchy providers and categories added; no value or period moves). One defect surfaced and is fixed in the same commit:
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
unchanged. Production 2023 register: 344 → 353 compiled, +9 `ledger_only` — the nine Scottish
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
for the rowwise driver. Fixed by the partition declaration folded into the family commit (it first rode as its own commit, `97ee7d54` on the pre-rebase branch): the filter
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

**Contract.** 252 → 279 targets on main 4ec72078 (registry scope 218 → 228, profile scope 34 → 51; the incumbent's 90 `voa/council_tax/<REGION>/<band>` rows now map onto the `mhclg.*`, `welshgov.*` and `scotgov.*` national ids, and `welshgov.council_tax_stock.band_i` is an unmapped declaration, the only band the incumbent never carried): the 17 `voa.council_tax_stock*` rows retire; `mhclg.council_tax_stock.band_a..h/total` (country + region, `region_composition` from local authorities), `mhclg.council_tax_stock.by_area.band_a..h` (296 English authorities, `area_scope` prefix E), `welshgov.council_tax_stock.band_a..i/total` (W92000004), `welshgov.council_tax_stock.by_area.band_a..i` (22, prefix W) and `scotgov.council_tax_stock.by_area.band_a..h` (32, prefix S) join. Operands: England line 7 (+ A- on band A) − line 11 − line 15; Wales a1 − h7 − h8 with `dimension_values` by band (`dimensions: []` for the totals); Scotland identity on the council record set.

**National surface (`tools/generate_uk_target_references.py`, 3.5 min).** 613 active (603 − 81 VOA region cells + 81 composed MHCLG region cells + 10 Welsh country rows), 7 `no_fact_at_or_before_period`, 7 signed excluded. Every composed region cell resolves the 2025-10 partition; the nine band-A cells sum to 5,590,029 = the publisher's England row (line 7 5,826,342 + A- 17,102 − line 11 64,474 − line 15 188,941) to the unit, and the nine totals to 24,246,267 = 25,056,421 − 267,894 − 542,260. Yorkshire composes 15 authorities across 17 code spellings (Barnsley and Sheffield under both codes); `composed_member_count` counts authorities. Wales band A 194,973.29 = a1 204,448.29 − h7 5,648 − h8 3,827; band I 5,305; total 1,377,633.03 (the 2025-26 row, ruled 2026-09-15). Scotland's nine country rows unchanged.

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

## Part D — the family measured (run `after-spine-r-df35af7-dense-e1500-s42`, commit 96e6204e)

**Run.** Recipe of Part B, 1,500 epochs, launched 18:10 on 2026-09-15, solve loss 0.012423 at the last epoch
(the VOA-basis attempts sat at 0.0150–0.0152 from epoch 1,100 on); household mass 29,247,433 → 29,003,269
(−0.83 %, the rowwise doctrine's declared move); gate battery: `uk_local_area_support`, ladder,
`uk_local_per_family_fit` and `uk_local_weight_ess` pass; `uk_local_target_fit` and `uk_local_weight_ratio`
fail as they did on D2 (below), so the artifact is a blocked candidate, as D2 was. Comparison figures are the
D2 dense run in the 10 September report (spine-q, feed ec7169b, 2,000 epochs, no region grain).

**Council tax at authority grain (2,511 active cells; D2 had 2,058).**
- Within 10 %: 92.2 % → 99.6 %; within 25 %: 98.4 % → 99.8 %; median absolute error 1.05 % → 0.82 %; cells
  past 25 %: 33 → 5.
- England A–G (the 2,058 cells D2 also bound): within 10 % 99.85 %, mean signed error −0.04 % (D2: every band
  negative, A −4.5 %, B −3.2 %, C −5.3 %, D −1.7 %); by band the mean signed error is now A −0.07 %, B −0.06 %,
  C −0.19 %, D −0.09 %, E +0.03 %, F +0.05 %, G +0.08 %, and the 10th percentile sits at −1.4 to −2.4 % (D2: A −14 %,
  B −11 %, C −19 %). One cell past 25 %: Derbyshire Dales band G (2,026 target, 1,254 estimate, −38 %; −42 % on D2 — a
  band-support residue, cause 3b of the plan).
- The four band-G micro-cells that tripped D2's red flag (Barking & Dagenham +4,683 %, Sandwell +3,729 %,
  Stoke-on-Trent +3,598 %, Kingston upon Hull +2,442 %) fit within 1.2 % (49 → 50, 75 → 76, 184 → 185, 60 → 59): the
  taxbase targets are the same size as the VOA ones (49–184 dwellings), so D2's estimates of 1,500–7,800 were the
  basis mismatch pulling weight onto those rows, not missing support.
- The inner-London A–D misses (Tower Hamlets C −57 %, Newham C −36 %, Islington D −34 %, Croydon C −33 %,
  Exeter B −43 %, Westminster G −39 % on D2) are within 2 % (−0.5 %, −1.9 %, −0.4 %, −0.7 %, 0.0 %, −2.0 %).
- Wales (198 cells, A–I): within 10 % 99.5 %; A–G and I mean signed error −0.8 % to +0.3 %, band I 22/22 within
  10 %; one cell past 25 %, Merthyr Tydfil band H (target 2, estimate 63).
- Scotland (255 cells, A–H): within 10 % 98.0 %; A–G mean signed error −1.5 % to −0.1 % (chargeable basis, the
  +1.2 % residual to households recorded in Part C); three cells past 25 %, all thin: Na h-Eileanan Siar band A
  (−31 %) and band H (target 5, estimate 120), Stirling band H (−43 %). Shetland band H is the signed support
  deferral of Part C.
- Band H is where the residue lives: the 53 bound Welsh and Scottish band-H cells are 50 out of 53 within 10 %
  and carry the three largest misses; the 296 English cells stay deferred (A14).

**The other local families did not move.** Census households 99.7 % → 100 % within 10 % at authority grain
(mean signed +0.5 % → −0.04 %) and 99.4 % → 99.7 % at constituency grain; tenure 98.3 % → 98.8 %; age
structure 100 % → 100 %; SPI income by area 99.2 % → 99.2 % (authority) and 99.4 % → 99.3 % (constituency);
private rent 98.4 % → 98.1 % (316 cells now, 314 before: Barnsley and Sheffield bind through the aliases); UC
households 98.0 % → 98.3 %.

**Gates.** `uk_local_target_fit`: 53 failing targets on D2 → 24, council tax 33 → 5; the other 19 are the SPI
self-employment amounts at −32 % to −95 % that D2 also carried (its 20 listed failures were capped by the
council-tax entries). `uk_local_weight_ratio`: max/median positive weight 296 → 283 against the reviewed 100,
ESS 130.5k → 127.9k, top-1 % weight share 16.0 % → 16.4 %; neither gate is this family's, both stood before it.

**National grain (joint solve controls, and step 40 against the incumbent surface).**
- The 97 `council_tax_stock` controls in the joint solve (78 composed English region cells after the three
  signed band-H exclusions, the 10 Welsh country rows, the 9 Scottish country rows) all sit within 10 %; the
  worst is `mhclg.council_tax_stock.band_h@E12000005` at 3.7 %. The composed cells equal the sum of their
  authority cells by construction (cross-grain factor 1.0), so the region distribution is the authority fit.
- Step 40 (`tools/evaluate_uk_dataset_size.py --steps 40-incumbent-surface`, incumbent 1.57.3, engine pass on
  the 2.56 GB candidate): bound national rows within 10 % 80.9 % → 83.1 % (D2 → this run), median absolute
  error 1.8 % → 2.5 %; the Scottish stock rows within 0.7 % (D2 within 1.1 %). The incumbent's 90
  `voa/council_tax/<REGION>/<band>` rows are still evaluated as region rollups of our authority cells against
  the VOA valuation-list targets, and now read −3 to −7 % on every band A–G (D2: −0.5 to −15 %, uneven by band).
  That uniform gap is the basis difference the family was re-based on (occupied chargeable dwellings against the
  valuation list, England +6.3 %), not a fit miss; the parity register records it as a signed difference. With the
  fixture relabelled onto the `mhclg.*` / `welshgov.*` ids, the evaluation joins the nine Welsh incumbent rows to
  our Welsh country rows (bound national rows 324 → 333; the 81 English region rows stay rollups, since our
  composed cells are nine per band).
- `obr.council_tax` (UK, net of CTR, £50.9bn): −11.0 % on D2 → −12.8 % here (£45.3bn → £44.4bn). The direction
  is expected: D2 over-weighted high-band inner-London dwellings to chase the valuation-list counts, which
  inflated modelled council tax; on the household basis the level gap is what remains (cause 3h of the plan:
  England gross ≈ £1,522 per household against a requirement near £1,815, plus receipts on non-household
  dwellings). It is the CT-C item on #929, not this PR's.
- `frozen_vs_recomputed` no longer carries the nine VOA rows the 55k report called its largest divergence:
  with `size_evaluation.frozen_vs_recomputed` keyed on bound rows only, the rows over 1 % are the same two D2
  also had (`isc.private_school_students`, `ons.land.corporate_land_value`); max divergence 12.4 % → 6.2 %.

**Verdict against the plan's 5f checks.** Family within 10 % above 97 %: yes (99.6 %). Bias by band ≈ 0: yes for
England A–G; Scotland A–C −1.0 to −1.5 % (chargeable basis). National and region rows within 2 %: the 97 controls
within 3.7 %, 96 within 2 %. Gate failures down to the micro-cells and any inner-London residue: the inner-London
misses are gone; five council-tax cells remain, three of them band-H micro-cells. Census and tenure unchanged:
yes. `obr.council_tax` a smaller miss: no, −11.0 % → −12.8 % (explained above; open on #929 as CT-C).

**Re-target (2026-09-15, after the measurement; her ruling: no re-solve).** #906 merged at 12:32 UTC and main
(4ec72078) took #921 and two later #906 commits, so the four commits were replayed onto main: the re-pin
regenerates to 603 national references (main's 603 on the v3 feed), the family to 613 / 20,885. Two overlaps
were merged rather than chosen between: 42254801 filters the council-tax stock totals by band (the totals leave
the composition group from their side; the new `mhclg.*` and `welshgov.*` totals carry the same filter shape, A–H
and A–I) alongside this branch's partition filter on the composition cells, and b5f1331b's resolver-derived leg
licences carry the `area_scope` rule on top. The run in this Part measured the pre-rebase family tree
(96e6204e, kept as the local ref `backup-929-pre-retarget`); the rebased head adds #921's UC element controls
to the joint solve and the totals' filters, neither of which touches a council-tax cell, and was not re-solved.

## Part E — review round 1 (Vahid, 2026-09-16) and the sources of the comparison figures

**Sources of the D2 figures quoted in Part D and the PR body.** The D2 run is
`data/ukds/acceptance/355-dataset-size/spine-q/f100-k15-dense-e2000-s42` (the 10 September report's dense run; its
step-40 evaluation sits under `evaluation/40-incumbent-surface/`). The council-tax family figures (92.2 % within
10 %, 33 past 25 %, the mean signed error by band, the four band-G micro-cells, the inner-London cells) are read
from its `solve_diagnostics.csv` (family `council_tax`, area type `la`); the "+6.3 % England / +6.6 % Wales
valuation list over the household rows" and the 0.97–1.20× per-authority range in the plan and the PR body are
the VOA band A–G sums per authority against the `census_households` rows of the same file; the national grain
(80.9 % within 10 %, `obr.council_tax` −11.0 %, the frozen-versus-recomputed rows) is its
`incumbent_surface_evaluation.json`. The same script summarised both runs.

**Review items and what moved** (commit on the branch after 4db5d6e3):
1. Both compile-parity registers now carry taxbase-basis rationales: the local `calibration_drift` rows on the
   `mhclg…by_area` cells sign the valuation-list → occupied-chargeable translation (with the ruling dates and
   the Westminster example), the `fixture_only` rows name the band-H and support-floor deferrals, the Scottish and
   Welsh `ledger_only` rows say why the incumbent has no counterpart (band I included); on the national register
   the 81 `mhclg…@E12` and nine `welshgov` drift rows sign the same translation with the North East and Wales
   band-A examples, and Welsh band I has its own rationale. Band I joins the tool's metric set.
2. The binding notes no longer name an England country control; the nine composed cells sum to the publisher's
   England row under a feed-gated test (`test_composed_english_region_cells_sum_to_the_publisher_england_row`:
   band A 5,590,029, total 24,246,267, bands sum to the total).
3. The partition commit is folded into the family commit (its test could not stand alone on a base where #906's
   42254801 had already filtered the totals); the changelog fragment says "on the pre-42254801 frame".
4. The four retired generator masks are recorded with dates, approver and rationale under the council-tax family
   of `uk_local_target_census.json` (`retired_deferrals`).
5. The three band-H region exclusions read the MHCLG basis and the #929 measurement (78 cells within 3.7 %).
6. The rebased-head batch result is in this Part (below), not a comment.
7. 2,511 / 255 everywhere (Shetland band H deferred); the PR body too.
8. The #355 erratum sits after both VOA divergence paragraphs and states 12.4 % → 6.2 %.
**Batch on the review-round head.** 2,739 tests, 0 failures, 16 skipped (2026-09-16 13:21, the UK build shard: test_uk_*, ledger targets, authoring, country spec, chronicle epoch, cross-grain); the two feed-gated regeneration tests (national and local) and the
England-sum test ran with the pinned feed present (`.codex-work/consumer_facts_uk.jsonl`, digest-checked against the pin).

Questions: the composed controls' rescale over the two deferred members is stated in the census family text;
the vintage strip keeps classification revisions (`sic2007`) and says the MYE fold is intended; the Welsh
selector notes cite the 2026-09-15 ruling; the two feed-gated regeneration tests and the sum test were run with
the pinned feed present (below).
