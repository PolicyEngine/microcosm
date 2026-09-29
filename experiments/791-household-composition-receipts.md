# microcosm#791 receipts: relationship-to-head frame column and the ONS household-composition rebind

Plan of record: María's local `repos/uk-791-relationship-to-head-plan.md` (approved 2026-09-11 in plan mode; plans are kept outside the tree by her convention, and these receipts carry what the tree needs of it). Branch `uk-household-composition-791` off main `6f7571e1`.

## R1 — vintage documentation check (step 0)

The relationship code list the stage declares (`derive_ons_household_composition.frs_household_grid_relationship_codes`) was read from the FRS 2024-25 question instructions for the pinned release, UK Data Service SN 9563 (`9563_frs_2024-25_question_instructions_final.pdf`, P18492, 408 pages, sha256 `8fec3d4e8f66a2a65ccf0b8455d410c39c41181d0c2148949a14840f1f0eb76d`), Household Grid, question "Relationship" (page 17 of the PDF): 1 spouse, 2 cohabitee, 3 son/daughter (incl adopted/legal dependent), 4 step-son/daughter, 5 foster child, 6 son-in-law/daughter-in-law, 7 father/mother/or guardian, 8 step father/mother, 9 foster parent, 10 father/mother-in-law, 11 brother/sister (incl adopted), 12 step-brother/sister, 13 foster brother/sister, 14 brother/sister-in-law, 15 grand-son/daughter, 16 grand-father/mother, 17 other relative, 18 other non-relative, 20 Civil Partner, 97 not used. Identical to the 2023-24 instructions (SN 9367). The instructions add: relatives of cohabitees are coded as if married; half-siblings are coded with step-siblings; a 16-19 in full-time education living with grandparents has the grandparents coded as legal guardian (7/3). The household reference person is the highest-income householder, the elder on ties (FRS background information and methodology).

The ONS definitions come from Families and households in the UK: 2025 (Measuring the data / Glossary) and "Families and households statistics explained" (ONS, 2021): family, dependent child (under 16, or 16-18 in full-time education, excluding 16-18s with a partner or own child in the household), non-dependent child, one-family household ("households where there is one family and one individual are also classified as one-family households"), two or more unrelated adults ("a household with two or more people, none of whom are living as part of a family (that is, they do not contain either a couple or a parent with their child)", https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/families/articles/familiesandhouseholdsstatisticsexplained/2021-03-02), multi-family household, and foster children / children living with someone other than their parents as separate one-person family units.

The ten category identifiers are Chronicle's `ons-families-households-2025` package, Table 7, `ons.household_type` value ids (`lone_households_under_65` ... `multi_family_households`); the package carries no Table 7 note text, so the definitions above are recorded on the contract rows and here.

## R2 — licensed-tape probe (step 0, read-only, `data/ukds/frs_2024_25`, digests as pinned)

Run of `derive_frs_relationships` over the raw adult (27,714), child (7,252) and househol (16,288) tabs with a frame-shaped person table (adult `age80`, child `age`, `hrpid == 1` as head) and `GROSS4` as the household weight.

Invariants: head_invariant_violations 0 (one HEAD per household, equal to `hrpid == 1`; `relhrp` blank exactly 16,288 times); hrpnum_mismatches 0; person_index_gaps 0 (PERSON is contiguous 1..n, n <= 14, in every household); multi_partner_persons 0; relhrp_unmapped_codes none (adult codes 1-8, 10, 11, 14-18, 20; child codes 3-5, 7, 11, 12, 14, 15, 17, 18); family_index_violations 0; domain_violations 0; partition_closes true; full-time-education source `educft` (no `fted` on the 2024-25 tape).

grid_reciprocity_mismatches 3 (ordered pairs), in two households: two children aged 19 and 17 record the HRP as parent (code 3) while the HRP records them as other non-relative (18); an infant records the HRP's spouse as parent (3) while that adult records the infant as sibling (11). The derivation takes either side's declaration as establishing the link (the child's parent claim wins) and the manifest declares the tolerance as this count, so a re-issued tape with a different count refuses for review. parent_tie_breaks 11 (a child-eligible person linked to two uncoupled parents; natural before step, then lowest person number). one_family_plus_individuals 189 households (the couple-plus-lodger / couple-plus-elderly-parent class the #757 person-count restrictions could not place).

Roles: COUPLE_PARTNER 17,732; DEPENDENT_CHILD 7,238; NON_DEPENDENT_CHILD 1,978; LONE_PARENT 1,584; INDIVIDUAL 6,434. Dependency: under 16 6,376; 16-18 in FTE 862; 16-18 not in FTE 143; 19+ 1,835.

Ten cells at design weights (GROSS4) against the published 2025 values (ratio):

- lone_households_under_65: 2,538 records, 4,364,966 vs 4,316,000 (1.011)
- lone_households_over_65: 3,152 records, 4,115,625 vs 4,254,000 (0.967)
- unrelated_adult_households: 216 records, 590,367 vs 813,000 (0.726)
- couple_no_children_households: 5,132 records, 8,780,988 vs 8,119,000 (1.082)
- couple_under_3_children_households: 2,560 records, 4,952,586 vs 5,439,000 (0.911)
- couple_3_plus_children_households: 457 records, 1,026,478 vs 931,000 (1.103)
- couple_non_dependent_children_only_households: 626 records, 1,824,245 vs 1,848,000 (0.987)
- lone_parent_dependent_children_households: 1,004 records, 1,886,352 vs 1,835,000 (1.028)
- lone_parent_non_dependent_children_households: 536 records, 1,218,884 vs 1,204,000 (1.012)
- multi_family_households: 67 records, 211,942 vs 244,000 (0.869)
- total: 16,288 records, 28,972,433 vs 29,003,000 (0.999)

Every cell is inside the 25 % `uk_target_fit` bound before any calibration. For comparison the #757 rebind packet (benefit-unit proxies) read multi_family +1618 %, unrelated_adult +48 %, couple_no_children +34 %.

`househol.hhcomps` was checked and rejected as a shortcut: its 17 codes are a function of `adulth`, `depchldh` and sex/pension age only (cross-tab on the tape) and carry no family structure.

## R3 — sampled rung (1 %) and the executor placeholder defect

The first 1 % sampled build (`build_791.sh ... --sample-fraction 0.01 --sample-seed 42`, code `6f7571e1` + this branch, engine 2.97.0) refused at the new node: `Patched person.ons_family_index has dtype float64; declaration requires 'int64'`. Instrumenting `_patch_columns` showed the incoming column was `int64` and the household table carried a RangeIndex, but the sampled person table's index was non-contiguous (`[243, 244, 1099, ...]`) after the root filter; the executor built its zero-filled placeholder on a fresh RangeIndex and inserted it label-aligned, NaN-filling the gaps and widening the column. `frs_relationships` is the first stage to add a new dense person column right after the sampling filter, which is why no earlier sampled rung met it. Fix: the placeholder binds to the table's own index (`packages/microcosm-graph/src/microcosm/graph/population.py`, `_patch_columns`), with a regression test on a non-contiguous person index (`test_new_dense_column_on_a_filtered_population_keeps_its_dtype`). Unsampled builds carry RangeIndex tables and were never affected.

With the executor fix the 1 % rung (`spine-r-f001`, 181 s) executed `frs_relationships` and twenty further stages and then refused inside `uc_capital_coherence` ("no positive-weight base-FRS reporter donors for dependent_children=0, couple=True"): a sparsity limit of the 1 % rung in a later stage, unrelated to this change, so the acceptance evidence is taken from the full build (R4) rather than a rung.

## R4 — full licensed build `spine-r`

`build_791.sh` recipe (build_twin.sh inputs: FRS 2024-25 tabs, SPI 2022-23, HMRC 2023-24 and CGT 2025 workbooks, WAS R8, LCFS 2023-24, ETB 1977-2024), tree `6f7571e1` + this branch, policyengine-uk 2.97.0, no sampling, checkpoints on: 373 s wall, 7.1 GB peak; `data/ukds/acceptance/791-relationships/spine-r/spine-r.h5` sha256 `921612e88eefcf21511ac6ad5faf60a1514f577490188c6e39aec49cb4224f27` (167,915,393 bytes); 29 stages; 52,846 households / 61,213 benunits / 113,590 persons (the spine-q counts). Spine battery 17/17 passed, phases `assembled` and `transferred`, `blocked_at_phase` null. The two new gates at the assembled boundary: `uk_stage_frs_relationships_composition` passed with the R2 counts exactly (16,288 base households typed, 3 reciprocity mismatches at the reviewed tolerance 3, every other invariant 0, partition closes); `uk_ons_household_type_enum_domain` passed (one column, ten allowed values, zero invalid). Sidecars: `spine-r.spine_gates.json`, `spine-r.build.json`, `spine-r.nonzero_shares.json`, logbook row `2aaaef18…`.

## R5 — twin diff against spine-q (`diff-vs-spine-q/`)

`compare_uk_h5_payload.py spine-q.h5 spine-r.h5` (whole artifact, row-aligned): store keys equal; root attributes equal; every table's row count and index values equal; `benunit` and `time_period` byte-identical; `person` and `household` have no column with differing values, no column only in the reference, and exactly the candidate-only columns `relationship_to_head`, `ons_family_role`, `ons_family_index` (person, positions 65-67 of 128) and `ons_household_type` (household, position 18 of 74). The shared columns keep their order position-wise (checked directly on the two stores). `classify_uk_payload_diff.py` against `spine-r-payload-expectation.json`: the four columns classify `expected_changed` (surface `column_only_right`); the `column_order` structural surface on person and household reads unexpected only because the classifier refuses structural surfaces under `expected_changed`, and it moves solely because the column set grew — recorded here as the adjudication. No value, weight, dtype, index or attribute moves: the stage adds and rewrites nothing else.

## R6 — parity instrument (`parity/`, `parity-baseline-q/`)

`build_uk_efrs_parity_reference.py --candidate-h5 spine-r.h5 --emit-candidate-json` extracts 159 candidate input layers (the enhanced-FRS reference surface is engine inputs); `verify_uk_spine_parity.py --strict` returns `defect` with 20 unsigned differences (benunit/person counts 61,213/113,590 vs the reference's 61,223/113,617 and 18 nonzero-share deltas: bus fares, corporate wealth, diesel, education consumption, employment income, employment sector, ...). The same command on spine-q (main `0afb1235`) returns the identical 20 unsigned differences and the identical register usage (dormant `scottish-water-sewerage-successor-level`; unused: consumer-debt, donor-selection-rng, frs-benunit-capital, lcfs-fuel-incidence, mortgage-debt, scottish-water-nan-zeroing), so the verdict is the pre-existing state of the pinned reference (Max's #749 re-pin), not this change. The four #791 columns are not engine variables, so the instrument does not extract them and no register entry can ever match: the two `column_missing_in_reference` entries first drafted for them were withdrawn rather than left permanently unused. The instrument sees no difference attributable to #791.

## R7 — derivation delta on spine-r (`delta_791.json`, design weights, 52,846 households, total weight 29,247,433)

The pre-#791 ten bindings (`origin/main` contract rows, engine variables `family_type`, `is_child`, `age`, `household_num_benunits` resolved through `UKMeasureResolver` and applied through `UKFrameTargetAdapter`) against the new `ons_household_type` column, same spine, same weights. Ratio to the published 2025 value, old → new:

- lone_households_under_65: 1.047 → 1.047 (unchanged by construction)
- lone_households_over_65: 1.140 → 1.140 (unchanged)
- unrelated_adult_households: 4.031 → 0.620 (3,277,295 → 503,862)
- couple_no_children_households: 1.094 → 1.112 (8,885,648 → 9,025,783; the couple-plus-individual class joins)
- couple_under_3_children_households: 0.879 → 0.893
- couple_3_plus_children_households: 0.951 → 0.962
- couple_non_dependent_children_only_households: 1.000 → 0.817 (1,847,825 → 1,510,386; the old proxy fitted the number by counting households with an extra adult of any relationship)
- lone_parent_dependent_children_households: 1.137 → 1.004 (2,085,571 → 1,841,865)
- lone_parent_non_dependent_children_households: 2.721 → 0.884 (3,276,524 → 1,064,691)
- multi_family_households: 18.849 → 0.738 (4,599,205 → 180,012)

The old bindings were not a partition: 6,563 households (weight 4,654,447) satisfied two or more old cells at once (none satisfied zero), which is how the old multi-family and unrelated-adult proxies absorbed everything the person-count restrictions excluded. The new column types every household exactly once. Every new cell is inside the 25 % fence at design weights before calibration; the widest, multi-family at 0.738 and unrelated adults at 0.620, are the two thinnest cells (226 and 685 spine households). The design-weighted crosstab (old first-matching cell × new cell) is in `delta_791.json`.

## R8 — national calibration on spine-r (`calibration-r/`, `dev-791-spine-r`)

`calibrate_791.sh`: `tools/calibrate_uk_national_dataset.py` on spine-r (sha `921612e8…`) against the pinned Chronicle artifact `ec7169b` (facts `4a50ee95…`, manifest `a95d0ee9…`), campaign settings `--epochs 1500 --target-weight-rule family_equal`, committed measure-exclusion register (48 entries after the three retirements). 367 targets bound, none skipped; initial loss 0.2987 → final loss 0.0114; 96.5 % of targets within 10 %; 0 of 367 outside the 25 % `uk_target_fit` fence; effective sample size 13,230; realized max weight ratio 10.0; all 52,846 records keep positive weight. Terminal gates 6/6 passed. The run refused to *stage* the calibrated H5 because `MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY` is not set in this environment (unsigned full-scale runs refuse to stage by design); the diagnostics and terminal-gate report were written before that refusal and are the evidence here — no staged artifact is claimed.

The ten ONS household-composition cells, target → initial (design weights) → final, relative error:

- lone_households_under_65: 4,316,000 → 4,520,161 → 4,321,179 (+0.12 %)
- lone_households_over_65: 4,254,000 → 4,850,900 → 4,256,780 (+0.07 %)
- unrelated_adult_households: 813,000 → 503,862 → 812,005 (−0.12 %)
- couple_no_children_households: 8,119,000 → 9,025,783 → 8,189,235 (+0.87 %)
- couple_under_3_children_households: 5,439,000 → 4,854,379 → 5,444,365 (+0.10 %)
- couple_3_plus_children_households: 931,000 → 895,393 → 931,475 (+0.05 %)
- couple_non_dependent_children_only_households: 1,848,000 → 1,510,386 → 1,846,693 (−0.07 %)
- lone_parent_dependent_children_households: 1,835,000 → 1,841,865 → 1,838,917 (+0.21 %)
- lone_parent_non_dependent_children_households: 1,204,000 → 1,064,691 → 1,203,046 (−0.08 %)
- multi_family_households: 244,000 → 180,012 → 243,910 (−0.04 %)

For comparison: the #757 rebind packet (proxies, 403 targets, loss 0.1362) read multi_family +1618 %, unrelated_adult +48 %, couple_no_children +34 %; and the Q50f 55k run of 2026-09-10, with the three cells excluded, read lone_parent_dependent_children +21.9 % as collateral — here +0.21 %. The three retired exclusions are not needed: the cells fit as concepts, not as numbers the solver stuffs.

## R9 — rowwise dry run, f001 rung (`rowwise-dry-run.log`, `rowwise_791.sh`)

`tools/build_uk_rowwise_candidate.py --dry-run --sample-fraction 0.01 --n-clones 4 --seed 42` on spine-r, ladder `bed3f13d…` (the #887 rebuild), feed `ec7169b`: `cross_grain.unbound_bridges` is `[]` for the first time — the `national_household_composition_partition_vs_census_households` bridge appears in `cross_grain.groups` for both lower grains (country controls constituency, country controls la), one leg `England+Wales+Scotland+Northern Ireland` each, `declared_factor` 1.0, `relative_shift` 0.0, `new_total` 29,003,000; `census_household_uprating.applied` true with the #887 per-grain factors (constituency 1.0335597, local authority 1.0335595) unchanged. 41 inconsistencies in force (the 38 recorded on #887 plus the bound bridge's groups). The 1,011 census cells therefore keep their A15-uprated values to float precision, as the plan predicted from closure (the ten cells and the uprating control are both `ons.households_total` 2025); the local surface is unchanged numerically and the dense assembler will stop emitting the `unbound_bridge:` limitation.

## R10 — test surface on the branch

`spine-uk` CI group (127 files, 2,403 tests, 35 min): 2,372 passed, 21 skipped, 10 failed on the first pass; all ten were pins or fixtures that had to follow the change, and every one is green on rerun: the data-contract mirrors (the two gate ids in the entry table, the spine part scope, the full-manifest digests `policy 38a8a014…`, `gates_manifest 49e86be9…`, `spec_fingerprint 25049e61…`, the spine part digests `7f9f07e6…`/`7058df03…` and the release-cut part digests `1aaf29c5…`/`4a93792f…`, computed by the producer the sync test `test_gate_battery_contract_pins.py` holds in lockstep), the uk-data parity register regenerated from its source (`data_target_parity.py`), the local-target census regenerated from its source (`local_target_census.py`, the bound-bridge wording), the spine-driver tests' synthetic tabs and synthetic spec (grid columns, `hrpnum`, the declared operation parameters), and the release-assembler tests (which read the contract mirrors). Also green: `packages/microcosm-data/tests` (398), `packages/microcosm-graph/tests` (337, including the H2 parity acceptance on the regenerated fixture and the executor regression), `test_gate_battery_contract_pins.py`, and the targeted pin set (401). `tools/ci_test_groups.py --verify` ok; `tools/build_uk_release_input_coverage_manifest.py` regenerated (only the `source_manifest_sha256` pins moved; 145 required inputs unchanged); `tools/generate_uk_target_references.py`'s pinned-feed regeneration test passes on the ec7169b artifact (no generated surface moved); `ruff check` and `ruff format --check` clean on every changed file.

One failure is pre-existing and untouched: `test_uk_parity_reference.py::test_cached_reference_regeneration_matches_committed_surface` regenerates `uk/efrs_parity_reference.json` from the cached licensed eFRS artifact and compares it to the committed file; the committed file was produced with policyengine-uk 2.89.0 (223 input variables) while `uv.lock` on main pins 2.97.0 (228), which now recognises `sic_industry_division`, `employment_sector` and `bus_fare_spending` as inputs. The branch changes neither file; the test skips in CI (no HF cache) and fails identically on main in this environment. The re-pin belongs with #749.

## R11 — Vahid's review round 1 (comment at 7779747b, 2026-09-11)

Should-fixes: the `census_disclosure_control_noise` adjudication reason now carries a dated re-wording trailer ("re-worded 2026-09-11 (microcosm#791 …)") beside the #887 one, under the unchanged 2026-08-31 approval; the plan of record stays María's local `repos/uk-791-relationship-to-head-plan.md` (her convention: plans live outside the tree; a first attempt to commit it under `docs/` was reverted on her ruling), and the receipts and PR body say so. Questions: the `unrelated_adult_households` and `lone_parent_dependent_children_households` contract notes record the skip-generation age switch (under 16 with no parent present: two unrelated adults, codes 15/16 never form a family; 16-19 in full-time education: the FRS codes the grandparents as legal guardian, 7/3, so a family with a dependent child) and cite the ONS "Families and households statistics explained" definition of two or more unrelated adults by URL, which is the source R1 quotes. Nits: `age` NaN now refuses instead of reading as 0 (`FRSRelationshipsError`, test added); `stage_health.py` is the main text plus the two composition hunks only (the earlier formatter pass had rewritten unrelated gates; CI lints with `ruff check` only); the enum-domain artifact fallback and the head-plus-child-only H2 fixture are left as noted, the couple and multi-family paths being held by the unit tests. The `test_uk_parity_reference.py` cached-reference regeneration test is retired on María's call (R10: it regenerated the reference under the installed engine, was skipped in CI and failed on main since the 2.97.0 lock); the thirteen other integrity tests on the reference, including the cached-artifact sha check, stay.

## R12 — rebase onto main `295130c9` (2026-09-11)

Main took #894, #855 and #902 after the branch base `6f7571e1`. One conflict, `uk/uk_population_targets.json`, where #855 added `hierarchy` at the top level and `category_id`/`label` on every row (236 rows changed, the ten composition rows among them, their `policyengine` bindings untouched on main). Resolved by taking main's file and re-injecting the branch's ten `policyengine` bindings, written in main's serialisation (`indent=2`, `ensure_ascii=False`); the three commits were re-applied by cherry-pick after a first `git rebase` pass had let a conflict-marked file through. After the rebase: no conflict markers; the UK spec digest is unchanged (`c396ee51…`, main touched no spec, gate or manifest file); the local-target census, uk-data parity register, release-input coverage manifest and CI group partition are all current; the pin, contract, overlap and stage suites rerun green (R10's set plus the #855-touched `test_uk_population_targets.py` and `test_country_spec.py`). One local check no longer runs: the pinned-feed regeneration test (`test_uk_target_references.py`) now stops in #855's hierarchy metadata (`obr.income_tax`: dimension `obr.efo_line` requires exactly one non-empty label, got none) against the local `ec7169b` artifact, which predates the label requirement; the feed pin is unchanged on main and the test is skipped in CI, so this is the local artifact's vintage, not this branch.
