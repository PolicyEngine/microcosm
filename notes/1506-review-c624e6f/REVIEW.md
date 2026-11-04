Reviewed PolicyEngine/microcosm#1506 at head `c624e6fe5038ee2db19ee3a19773a92b570ae945` (base `ea4606d6b9c1e54a5e1a4c971498e7adb0046dbe`, as briefed): **APPROVE with follow-ups**

# Review of microcosm#1506 (US Chronicle feed re-pin to `41b2fb2b`, FSA loan portfolio tables)

A cloud review session for the student debt pilot hub wrote these notes. Every
command ran in this session. Code citations are `path:line` at `c624e6f`
unless they name `0ff293b` (#1357's head) or a Chronicle commit, which I read
through raw.githubusercontent.com only. Logs are in `logs/`, scripts in
`scripts/`. No PR or issue was commented on, approved or changed.

I found nothing that blocks the merge. Every number the PR body and the
hand-off give reproduced: the inputs, the pin, a byte-identical regeneration of
both parity resources, the default compile, a byte-identical `compare.json`
and `added_rows.json`, #1357's 132 targets and dispositions, the fixture
claims and the twelve test files. The findings below are follow-ups. The most
useful one is that the chain test trusts the listing's `source_name`
(finding 1).

## Findings, most severe first

### 1. (Low) The chain test takes an added row's source from the listing without checking it against the record set

`packages/microcosm-build/tests/engine_free/us/test_us_esi_premiums.py:1164-1165`
checks that every row in `added_rows.json` has `source_name == "ed_fsa"`.
Nothing checks that field against anything else in the committed files.
`compare.json` doesn't record sources, so the listing is the only evidence.

- **What goes wrong.** Suppose a later re-pin adds one `irs_soi` row next to
  the FSA rows, and a hand-edited listing calls it `ed_fsa`. The helper
  accepts it, and so does the whole test once the pin's row count agrees.
- **Concrete case.** `scripts/chain_mutations.py` takes the committed
  `compare.json` and adds a key with `new_rows` +1. It lists that key in
  `added_rows.json` with `source_name: "ed_fsa"` and
  `record_set_id: "irs_soi.ty2023.table_1_4"`, and gives the pin
  `fact_row_count` 40,643. Result:
  `[add-non-ed_fsa-row (listing claims ed_fsa, pin row count +1)] PASSED (not refused)`
  (`logs/step6_chain_mutations.log`, last line).
- **Without the pin edit** the case fails, but only at the final row-count
  assert (`test_us_esi_premiums.py:1252`), not for its real reason.
- **The real listing is honest.** `list_added_rows.py` writes `source_name`
  from the feed (`experiments/us-feed-ed-fsa/list_added_rows.py:42-47`), and
  my rerun reproduces `added_rows.json` byte for byte (step 5). So this is a
  hole a careless or hostile edit could use, not an error in this PR.
- **Smallest fix.** Have the declaration also name the added record sets'
  namespace, as `("…/added_rows.json", "ed_fsa", "ed_fsa.")`. Then assert
  `all(row["record_set_id"].startswith(prefix) for row in listed.values())`
  and add one refusing case. All 461 listed rows have `ed_fsa.` record sets.
  The namespace has to be declared rather than taken from `source_name`:
  961 rows of the feed have a record-set namespace that differs from their
  source (`bea` → `bea_regional` and `bea_nipa`, `ssa` → `ssa_supplement`,
  and others).

### 2. (Low) The `ed_fsa` fence gives all 11 families one classification, but 7 of them never bind, even with #1357's opt-in

`tools/build_us_target_parity_manifest.py:137-173` declares the whole
`ed_fsa` namespace `off_by_default`, with a verdict basis of "promote the
bound families to compiled when the default compile binds them".

- **What #1357 binds.** At `0ff293b` it binds four tables:
  `FSA_BOUND_TABLES = (_LOCATION, _AGE, _DEBT_SIZE, _LOAN_TYPE)`
  (`fsa_student_loan_targets.py:153`). The other seven tables
  (`portfolio_summary` and the six repayment-plan and IDR tables) are
  diagnostic: `FSA_DIAGNOSTIC_TABLES`, `fsa_student_loan_targets.py:168-199`.
  So does the loan-type table's recipient count (`:29-31`).
- **The fence text is right about this.** It says the repayment-plan and
  summary tables "stay diagnostic there too".
- **The labels are not.** The manifest puts the same `off_by_default`
  classification on those seven families, and the new test's docstring says
  "the ed_fsa families bind only in a build with the federal student loan
  balance columns"
  (`packages/microcosm-build/tests/engine_free/shared/test_release_target_parity.py:346-349`).
  Seven of the 11 never bind in any build. "Off by default" suggests they
  will be promoted one day.
- **Concrete case.** The step 7 compile with `student_loan_targets=True`
  gives 132 FSA targets in only four families:
  `ed_fsa.portfolio_by_age`, `…_by_debt_size`, `…_by_loan_type` and
  `…_by_location`. The unfiltered registry's parity refusal names those four
  and no others (`logs/step7_opt_in_compile.log`).
- **Smallest fix.**
  - Keep the namespace fence for the four bound tables.
  - Add `_FAMILY_EXCLUSIONS` entries for the seven diagnostic families, with
    a classification that says they never bind (for example `not_modeled`,
    which `release_target_parity.py:235-237` lists). Their reasons can be
    #1357's `FSA_DIAGNOSTIC_TABLES` text.
  - Update the new test to expect 4 + 7.

  The counts stay 32 compiled and 64 exclusions, so this can wait for #1357,
  which owns the reasons. If the hub keeps the single fence, fix the test's
  docstring.

### 3. (Nit) The doc and the new test say "binds them" of all 461 facts; only 136 bind

`docs/us-chronicle-feed-repin.md:60-61` says "only microcosm#1357's
`student_loan_targets=True` binds them", where "them" is the `ed_fsa` facts.
#1357's receipt on this feed binds 136 of the 461 facts: 251 are diagnostic
and 58 are other_quarter (step 7). Suggested wording: "only
microcosm#1357's `student_loan_targets=True` binds any of them (136 of 461)".

### 4. (Nit) Two pieces of #1345's text read as stale after this PR

Neither file is in this PR's diff; the hub can decide whether to touch them.

- `experiments/us-feed-agi-age-labels/compare_feeds.py:6` still says "both
  feeds must hold the same keys". The script doesn't enforce that, and this
  PR relies on it reporting `keys_only_in_new`.
- `docs/us-chronicle-feed-repin.md:440` quotes the refusal at "row 228",
  which was the `c5e5bf8` feed. On this feed the first row that fails the v3
  schema is line 236 (`logs/step9_v3_schema_validation.log`). The paragraph
  says it describes `c5e5bf8`, so this is only a note.

### 5. (Follow-up for #1357, not this PR) #1357's v4 wording

These are #1357's to fix after this merges:

- #1357's fixture docstring
  (`packages/microcosm-build/tests/engine_free/us/test_us_fsa_student_loan_targets.py:3-12`
  at `0ff293b`) says the fixture holds "byte for byte" rows that
  `build-bundle` writes.
- Its body says the FSA rows are v4 and the pin would list both versions.

Against this pin, both are false in one field only. No fixture row matches a
feed row byte for byte, and all 244 do once the stamp is changed (step 7).
No test result depends on it.

### Checked and not a finding

- **The pin's schema list is never checked on the US path.**
  `consumer_fact_schema_versions` is never compared with the feed's rows on
  the US release path. The body says so correctly
  (`us_runtime/chronicle_feed.py:136-143`, `ledger_artifact.py:257-291`, and
  the release tool's `load_ledger_consumer_artifact(...,
  expected_facts_sha256=…)` at `tools/build_us_fiscal_refresh_release.py:12794-12803`).
  That predates this PR, and on this feed the list is right (step 1).
- **The pin isn't Chronicle main.** It is a cherry-pick commit that isn't on
  Chronicle main. #1357's body expected "the next Chronicle release". The
  brief says this cut is intended. The tag and the branch resolve to the
  cut's tree (step 2).

## Steps

### 0. Preflight: passed

`gh api repos/PolicyEngine/microcosm/pulls/1506 --jq '.head.sha, .base.sha'`
gave `c624e6fe…` and `ea4606d6…`, as briefed; the PR is open and a draft. All
five notes files fetched from Chronicle `bf3456f9`, `receipt.json` included.

### 1. The inputs: all passed (`logs/step1_inputs.log`)

- **The feed.** Gunzipped, `consumer_facts.jsonl.gz` is sha256 `1d45a89c…f9ea07`
  with 40,642 lines. All 40,642 rows read `chronicle.consumer_fact.v3`.
- **The scope.** `cmp` of the notes' scope with the committed
  `packages/microcosm-build/src/microcosm/build/us/chronicle_feed_scope.json`
  shows them byte-identical. It hashes to `73a39c16…4615633`. It has 706
  pairs and 72 runs.
  - #1345's 640 pairs come first, in order.
  - The 66 new pairs are all from the six `packages/ed_fsa` packages, at
    build year 2026.
  - Only `runs["2026"]` changes, and no other key.
- **#1345's feed.** `scripts/drop_ed_fsa_pairs.py` drops the 66 new pairs'
  rows, which leaves 40,181 lines, sha256 `4d89e4a4…`: #1345's pin. I used
  that file as the old feed in step 5.
- **The added rows.** The 461 added lines, in feed order, are byte-identical
  to the notes' `ed_fsa_rows.jsonl`, and all are `source.source_name ==
  "ed_fsa"`.

### 2. The pin: passed

- **How microcosm handles the schema digest.** Microcosm never computes
  `consumer_fact_schema_sha256` for the US feed.
  - The US loader only checks that it is 64 lowercase hex characters
    (`us_runtime/chronicle_feed.py:119`).
  - The test pins the value (`test_us_chronicle_feed.py:37`, `:62`).
  - `validate_chronicle_feed_pin` reads it from an artifact manifest
    (`ledger_artifact.py:277`). Only the transport kernels and the BE and NZ
    tools call that function (`git grep`); the US release path never does.
- **Chronicle's side.** Chronicle defines the digest as the sha256 of the
  packaged schema file (`policyengine_chronicle/schema.py:59-70` at
  `41b2fb2b`). `consumer_fact.v3.schema.json` hashes to `bdb51e2a…85d3d` at
  both `41b2fb2b` and `a9026cc`. `schema.py`, `consumer_contract.py` and
  `chronicle/epoch.py` are byte-identical at the two commits.
- **The stamp.** The emit version is v3:
  - `CONSUMER_FACT_ROW_CONTRACTS` ends at v3 (`chronicle/epoch.py:171-176`);
  - `consumer_contract.py:35` uses it;
  - `consumer_contract.py:664` writes it into every row.
  - There is no `consumer_fact.v4.schema.json` at `41b2fb2b` (404).
- **Each field of `us/chronicle_feed.json`.**
  - `source_commit`, `fact_row_count`, `facts_sha256` and `scope_sha256`
    match step 1.
  - `consumer_fact_schema_versions` `["chronicle.consumer_fact.v3"]` matches
    the rows.
  - `manifest_sha256` and `artifact_schema_version` are null: a bare feed.
- **The `build` text.** I read it through.
  - The lineage it gives (`c5e5bf8` … `a9026cc` plus `4448a7c96`, giving
    `41b2fb2b`) matches NOTES.md.
  - Tag `microcosm-us-feed-1074-v1` and branch `feed/us-a9026cc-ed-fsa-1103`
    both resolve to a tree whose `tests/test_chronicle_bundle.py` equals
    `41b2fb2b`'s, which differs from `a9026cc` and from main. This is
    evidence that the tag exists and points at the cut. It doesn't prove the
    tag peels to that commit.
  - The six `ed_fsa` packages exist at `41b2fb2b`, `4448a7c96`, `acbedb5`
    and main, with identical `source_package.yaml`, and not at `a9026cc`.
- **The values changed in `test_us_chronicle_feed.py`.** I checked each one
  independently:
  - `_CHRONICLE_COMMIT`, `_FACTS_SHA256` and 40,642 as above;
  - 706 pairs and 72 runs from the scope file;
  - `reviewed_exclusions` `"64"` from the regenerated manifest.
- **The body's citations.** Each one points at the code it describes:
  - in microcosm, `ledger_artifact.py:207-219`, `:257-291`, `:276`, `:287`
    and `:464-471`, and `us_runtime/chronicle_feed.py:136-143`;
  - in Chronicle, `consumer_contract.py:35`, `epoch.py:171-176` and
    `schema.py:59-70`.

### 3. The parity resources: passed (`logs/step3_regenerate.log`, `logs/step4_default_compile.log`)

- **Regeneration.** With the feed at
  `~/PolicyEngine/_buildh-runtime/inputs/consumer_facts_us_41b2fb2.jsonl`, I
  ran `python tools/build_us_target_parity_manifest.py` in a detached scratch
  worktree at `c624e6f`. It printed "32 compiled, 64 reviewed exclusions (93
  feed families)". `git status --porcelain` was then empty, so both files
  regenerate byte for byte.
- **The generator's diff** (`git diff ea4606d c624e6f -- tools/build_us_target_parity_manifest.py`)
  changes `DEFAULT_FEED_NAME` and adds the `ed_fsa` entry to
  `_NAMESPACE_EXCLUSIONS` (`:137-173`), and nothing else.
- **The manifest's diff** changes `feed`, `feed_sha256` and
  `reviewed_exclusions`, and adds 11 `ed_fsa.*` families, and nothing else.
- **`assert_target_parity_manifest_current(registry=…)`** passes on the
  release-style default registry. `us_release_target_parity_gate` passes
  too: compiled 32, reviewed 64, registry families 32 (step 4).
- **The fence's claims against #1357 at `0ff293b`:**
  - **These tables bind under `student_loan_targets=True`:** location, age,
    debt size and loan type. True: `FSA_BOUND_TABLES`,
    `fsa_student_loan_targets.py:153`.
  - **The repayment-plan and summary tables stay diagnostic.** True:
    `FSA_DIAGNOSTIC_TABLES`, `:168-199`, which lists `portfolio_summary` and
    all six plan and IDR tables. The fence leaves out that the loan-type
    recipient counts are diagnostic too (`:29-31`); see finding 2.
  - **`--scf-education-debt` sets it.** True: the flag is `store_true`
    (`tools/build_us_fiscal_refresh_release.py:1651-1663` at `0ff293b`), and
    the compile passes `student_loan_targets=args.scf_education_debt`
    (`:13012`). The default is `False` (`fiscal_targets.py:1040` at
    `0ff293b`).
  - **The opt-in keeps the two builds from failing each other.** True:
    `_without_opt_in_student_loan_targets` (`:12619-12632`) drops
    `spec.family == "ed_fsa"` before both parity checks (`:13054-13056`).
    Without that filter, the opt-in registry fails the manifest as stale
    (step 7).

### 4. The default compile: passed (`logs/step4_default_compile.log`, `scripts/default_compile.py`)

- **How I compiled.** I followed the release tool
  (`tools/build_us_fiscal_refresh_release.py:12793-12858`), imported from the
  tool's module:
  - `load_ledger_consumer_artifact(feed, expected_facts_sha256=pin)`;
  - `_check_committed_us_ledger_feed_pin`;
  - `_check_us_esi_anchor_against_feed`, which returned `[]`;
  - `compile_us_fiscal_target_registry(PERIOD=2024, packaged CD crosswalk,
    age_targets=True)`;
  - `apply_us_medicaid_enrollment_substitutions`.
- **The result:**
  - **33,250 targets in 32 families**, registry version `b8d571e91c22`;
  - no `ed_fsa` family;
  - no spec whose name, source, family, notes, metadata, filter or measure
    mentions `ed_fsa` or "Federal Student Aid".
  - One name holds "student_loan": JCT's
    `jct.tax_expenditures.cy2024.student_loan_interest_deduction.revenue_loss`.
    It isn't an FSA fact.
- **Run size.** 51 s, peak RSS 2.07 GB.

### 5. The compare: passed, byte-identical (`logs/step5_compare.log`)

I ran `compare_feeds.py <#1345 feed (4d89e4a4…)> <new feed> out/compare.json`,
then `list_added_rows.py out/compare.json <new feed> out/added_rows.json`.
`cmp` shows both outputs byte-identical to the committed
`experiments/us-feed-ed-fsa/compare.json` and `added_rows.json`.

- **Rows.** 40,181 old, 40,642 new, `keys_only_in_old` empty, 461
  `keys_only_in_new`, `changed_rows` 0, `changed_fields` empty.
- **Targets.** 33,250 on both feeds, same order, 0 changed; `b8d571e91c22`
  on both; `national_state` 6,102 targets, `3aa2a0830e7d`, on both.
- **The listing.** 461 rows, all `ed_fsa`, 66 record sets.

### 6. The report chain test: passed, with finding 1 (`logs/step6_chain_mutations.log`, `scripts/chain_mutations.py`)

The harness loads the test module unmodified. It points `_REPOSITORY_ROOT` at
scratch copies of the committed reports and runs the real
`test_the_committed_feed_pin_banks_the_anchor`, printing the assert that
fires for each case.

- **The PR's ten refusing cases each fire at the assert their id names:**

  | Case | Assert that fires |
  |---|---|
  | `undeclared-additions` | `:1157` |
  | `other-source` | `:1165` |
  | `listing-short` | `:1163` |
  | `listing-extra` | `:1163` |
  | `anchor-record-set` | `:1166` |
  | `no-source` | `:1159` |
  | `dropped-row` | `:1151` |
  | `row-count` | `:1169` |
  | `value-change` | `:1153` |
  | `wrong-base` | `:1149` |

- **Mutations of the real committed files:**
  - The unmutated chain passes.
  - **Add a non-`ed_fsa` row,** listed honestly: refused at `:1165`
    (sources `{irs_soi, ed_fsa}`).
  - **The same row, left out of the listing:** refused at `:1163`.
  - **Drop a listed row** from `added_rows.json`: refused at `:1163`. Drop it
    from the report's `keys_only_in_new` instead: also `:1163`.
  - **Drop a kept row:** refused at `:1151`.
  - **Change a kept row:**
    - its value: `:1153`;
    - a source field: `:1153`;
    - a label in an anchor record set: `:1154`;
    - a label elsewhere: accepted, by design (`_LABEL_AND_HASH_FIELDS`).
  - **Declare a second source:**
    - as a tuple `("ed_fsa", "irs_soi")`: refused at `:1165`;
    - as a three-element declaration: refused with a `ValueError` at the
      unpack, `:1240`. That refuses too, but not through an assert.
  - **Other declarations:**
    - the wrong source: `:1165`;
    - `None`: `:1157`;
    - dropping the FSA report from the tuple: `:1251`, the pin digest.
  - **Other listings:**
    - one that repeats a key: `:1162`;
    - one made for another feed: `:1160`.
- **Every check can fire.** In the PR's own cases, two checks never fire:
  `:1160` (the listing's `new_feed` digest) and `:1162` (an added key
  twice). In the mutations above both do fire, so neither is dead code; the
  PR just has no case for them. The check that misses a case is the one in
  finding 1: a listing that labels a row with the declared source passes
  whatever its record set.

### 7. #1357: passed (`logs/step7_*.log`, `scripts/opt_in_compile.py`, `scripts/fixture_check.py`, `scripts/receipt_v3_v4.py`)

- **Merge.** `git merge-tree --write-tree --name-only` gives the same seven
  conflicted files whether #1357's head goes into `c624e6f`, into `ea4606d`
  or into main `79a533c7`:
  - `docs/evidence/spec-engine/us-f0-coverage.json`;
  - `spec_engine/field_usage.py` and `spec_engine/inventory_coverage.py`;
  - `test_spec_engine_coverage_tool.py`, `test_spec_engine_field_usage.py`
    and `test_us_multispine_pool_tool.py`;
  - `tools/spec_engine_coverage.py`.

  #1506's and #1357's diffs share no file. In a local scratch branch,
  `scratch/1506-plus-1357` (never pushed), I merged `0ff293b` into `c624e6f`
  and resolved the seven to ours. `uv.lock` was unchanged. The branch got its
  own venv outside the checkout.
- **#1357's FSA test file.** `test_us_fsa_student_loan_targets.py` gave
  **25 passed**. It reads only its fixture, not the feed.
- **The opt-in compile.** The pin check passed on the feed. With
  `student_loan_targets=True`:
  - **33,382 targets, 132 FSA**: 65 `total_balance`, 1
    `parent_plus_balance` and 1 `balance_excluding_parent_plus` (67 balance),
    and 65 `borrowers`;
  - `_without_opt_in_student_loan_targets` gives 33,250, and
    `assert_target_parity_manifest_current` and the gate pass on it;
  - the unfiltered registry is refused as stale for the four bound
    families.

  With the flag off: 33,250 targets, 0 FSA, and parity passes.
- **The receipt.** Status `bound`, reference period 2026-03, 461 facts.
  Dispositions: diagnostic 251, other_quarter 58, excluded 8, allocated 4,
  out_of_scope 4, bound 136, as the hand-off says.
- **The fixture** is sha256 `9faed0e3…`, 244 rows, all v4:
  - 0 rows match a notes row byte for byte;
  - 244 match once the notes rows read v4;
  - 244 share an `aggregate_fact_key` with a notes row and differ only in
    `schema_version`;
  - the fixture lacks 217 of the notes' rows.
- **Restamped v3.** All 244 rows are then byte-identical to notes rows. In
  scratch, the test file again gave **25 passed**, with the same 25 test ids.
  The FSA receipt and the plan (references, values, metadata, labels,
  controls) from the v3 fixture equal those from the v4 fixture. The original
  fixture was restored, and `git status` was clean.
- **Fixture against feed.** The receipt from the fixture equals the one from
  the full feed in its targets and controls. It differs only in diagnostic
  dispositions (34 against 251), because the fixture lacks the 217
  repayment-plan rows.

### 8. The runs (`logs/step8/`, `logs/step8_summary.log`, `logs/step8_lint_plan.log`)

The feed was in place and `MICROCOSM_US_CHRONICLE_FACTS` pointed at it. I ran
each file in its own process with `-n 4 -p no:cacheprovider`, at `c624e6f`
in the main checkout. These are the `engine_free` copies.
`test_us_fiscal_targets`, `test_us_medicare_irmaa_holdout` and
`test_us_spec_bundle` also exist under `engine_contract` and
`engine_scenario`, which need the `us` extra; it isn't installed, so I didn't
run those. No file I ran notes a peak near 15 GB; the largest note is
`test_us_acs_local_release_tool.py:2860`, "about 3 GB peak RSS". So nothing
was skipped for memory.

| File (`packages/microcosm-build/tests/engine_free/…`) | Result | Hand-off |
|---|---|---:|
| `shared/test_release_target_parity.py` | 39 passed | 39 |
| `us/test_us_chronicle_feed.py` | 28 passed | 28 |
| `us/test_us_esi_premiums.py` | 159 passed | 159 |
| `us/test_us_fiscal_targets.py` | 174 passed | 174 |
| `us/test_us_soi_congressional_district_vintage.py` | 51 passed | 51 |
| `us/test_us_soi_state_agi_bands.py` | 29 passed | 29 |
| `us/test_us_medicare_irmaa_holdout.py` | 24 passed | 24 |
| `us/test_us_acs_local_sparse_cd.py` | 28 passed | 28 |
| `us/test_us_release_gate_preflight.py` | 84 passed | 84 |
| `us/test_us_spec_bundle.py` | 4 passed | 4 |
| `us/test_us_acs_local_release_tool.py` | 215 passed | 215 |
| `us/test_us_fiscal_refresh_builder.py` | 346 passed, 6 skipped | 346, 6 skipped |

Totals: 1,181 passed, 6 skipped, 0 failed. All 6 skips are in
`test_us_fiscal_refresh_builder.py`, and each one gives the reason "requires
the policyengine-us [us] extra" (`:55`). In particular, the pinned-feed
parity tests in `test_release_target_parity.py` (they read
`generator.DEFAULT_FEED_PATH`, `:554`) and the feed-gated tests in
`test_us_acs_local_sparse_cd.py` and `test_us_acs_local_release_tool.py`
ran; none of them skipped.

`ruff check .` passes, and `ruff format --check` passes on the five changed
`.py` files. `tools/ci_test_plan.py verify` reports `verification=ok`.

### 9. A fresh read

- **The diff.** I read all of `c624e6f` against `ea4606d`: 13 files, about
  5,868 insertions, and 4,230 of them are `added_rows.json`.
  - The pin, the scope, the tests, the generator, the doc section, the
    changelog fragment and `list_added_rows.py` all agree with each other
    and with what I ran.
  - The citations hold (step 2).
  - Apart from findings 1–4 I found no claim the code doesn't support and no
    check that can't fire.
- **The new doc section** (`docs/us-chronicle-feed-repin.md:16-94`) and the
  pin table (`:263-270`) match steps 1–5. "Dropping the 66 new pairs' rows
  leaves #1343's feed" means the `a9026cc` feed, which I reproduced.
- **The changelog fragment.** Its counts (461 rows, 66 pairs, 11 families)
  match.
- **One more check (`logs/step9_v3_schema_validation.log`).** I validated
  all 40,642 rows against Chronicle's `consumer_fact.v3.schema.json` at
  `41b2fb2b`, with Draft 2020-12 as `schema.py` uses it.
  - Exactly 994 rows fail: 515 `cms_medicaid`, 468 `census_acs` and 11 `jct`,
    all on `concept_alignment` 'authority' required.
  - All 461 `ed_fsa` rows pass.

#### The body's seven judgment calls

1. **Regenerating the parity resources: agree.**
   - `test_us_chronicle_feed.py:120-132` requires both files to restate the
     pin's digest.
   - `_exclusion_for` (`tools/build_us_target_parity_manifest.py:793-805`)
     halts on an unfenced family.
   - The regeneration reproduces the committed files byte for byte.
   - #1357's own body lists this as a re-pin step.
2. **The fence's origin says no us-data archaeology was done: agree, with a
   note.**
   - The generator's rule (`:1-17`) would rebuild an unarchaeologized fence
     for a us-data-targeted family. But a default build can't materialize
     these targets, so "compiled" isn't on offer whatever us-data did.
   - Stating the gap is the honest choice.
   - I couldn't check us-data, which this session can't reach.
3. **`DEFAULT_FEED_NAME` `consumer_facts_us_41b2fb2.jsonl`: agree.** It
   follows the earlier names.
4. **Leaving `compare_feeds.py` as it is and adding `list_added_rows.py`:
   agree.** The new script refuses a feed whose digest isn't the report's new
   feed (`list_added_rows.py:36-37`), and a report key the feed lacks
   (`:50-51`). It reproduces byte for byte. Finding 1 is the one gap in how
   the test uses it. Finding 4 covers the stale docstring.
5. **Keeping "build-consumer-artifact refuses it at this commit": agree, now
   with evidence.** I didn't run the command either. But the v3 schema file
   is identical at both commits, and on this feed exactly the same 994 rows
   fail it while every new row passes.
6. **What the doc takes from the cut notes: agree.**
   - I confirmed the 217 rows the fixture lacks (step 7).
   - `chronicle/epoch.py` at `0ec21e5` and at `acbedb5` lists v4 as the last
     contract.
   - `consumer_fact.v4.schema.json` exists at both and not at `41b2fb2b`
     (`logs/step9_chronicle_v4_refs.log`).
   - I can't confirm from raw files that `0ec21e5` is where v4 first
     entered.
7. **The branch name: no objection.**

## What I did not do

- I didn't run any `engine_contract`, `engine_scenario` or `engine_workflow`
  test; they need the `us` extra.
- I didn't run `chronicle build-consumer-artifact` or any Chronicle code.
- I didn't run #1357's other edited test files, which the brief doesn't ask
  for.
- I made no change to any PR branch.
