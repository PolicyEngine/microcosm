POST-ROUND-2 HOLD — all five round-1 P0 findings survive at the live PR head. The live head is still exactly `74b4d768`, so the three seam commits named as later changes were already included in round 1 and supply no resolving delta.

# Defensive correctness and completeness audit — microcosm PR #743

Reviewed PR head 74b4d768f7f7c83eb0593464ceb0e2a7c81ec154 against its exact merge base 7b90bb1882. The triple-dot diff is 35 files, 4,881 additions, and 437 deletions. I read every changed executable path and every added runbook, register, fixture, and receipt-producing path. No build was run.

## Round 2 — current-head reverification (2026-08-25)

### Exact head and scope

The live GitHub PR record, fresh `FETCH_HEAD`, and
`origin/uk-national-first-calibrated-candidate-623` independently resolved to
`74b4d768f7f7c83eb0593464ceb0e2a7c81ec154`; `git diff --quiet` found the trees
identical. `git merge-base --is-ancestor <sha> 74b4d768` exited 0 for each of
`96e73047`, `971a9699`, and `e54e546b`. Thus those commits precede the audited
head: there is no commit after round 1 to cite as a resolution. The verdicts
below nevertheless use fresh runtime/static probes, rather than history alone.

The requested five are the five P0 findings named in the round-1 summary
(P0/1–P0/5 below). The three supplemental P1 observations remain preserved in
the historical round-1 section; this matrix does not silently promote or close
them without a separate full reprobe.

| Round-1 finding | Round-2 verdict | Fresh evidence at `74b4d768` and current-head citations |
|---|---|---|
| 1. Documented held-run path aborts on the production target surface | **SURVIVES** | The runbook still invokes the national builder (`docs/uk-national-calibration-runbook-623.md:20-36`), whose stage construction still supplies neither resolver nor exclusions (`tools/build_uk_national_dataset.py:922-934`). The stage still returns no resolution without a resolver and refuses an activated unmaterializable measure (`packages/microcosm-build/src/microcosm/build/uk_runtime/national_calibration.py:86-99,203-218`). The exact packaged no-resolver regression passed while asserting that `RuntimeError` (`packages/microcosm-build/tests/test_uk_national_calibration.py:478-494`); no production-shaped successful builder path was found. |
| 2. Rule-1 scorer cannot score/authenticate production exports | **SURVIVES** | The scorer still calls `score_targets` directly on loaded H5 frames (`tools/score_uk_national_candidate.py:36-59`), while calibration still exports rebuilt pristine tables (`packages/microcosm-build/src/microcosm/build/uk_runtime/national_calibration.py:341-357,380-396`). Fresh inventory: all **397/397** target references are slash-named; a production-shaped `dwp/uc/households` target still raised `ValueError: No targets compiled`. Arbitrary H5s with SHA-256 `c052af90…` and `3827878d…` were still accepted under the default candidate/incumbent labels, and neither digest appeared in the receipt (`tools/score_uk_national_candidate.py:85-93,119-177,186-198`; representative target at `packages/microcosm-build/src/microcosm/build/uk/target_references.json:9532-9545`). |
| 3. Boolean person-count targets collapse with `any` | **SURVIVES** | Production routing still chooses boolean `max` for person-to-household reduction (`packages/microcosm-build/src/microcosm/build/uk_runtime/measure_simulation.py:23-76`). A three-child household again resolved to `[1.0, 0.0]`, not `[3.0, 0.0]`, through `bool_any_collapse_person_to_household`; inventory again found eight affected bindings. The added test still covers only legitimate predicate-any semantics (`packages/microcosm-build/tests/test_uk_measure_simulation.py:80-88`), while representative child-count bindings remain in `packages/microcosm-build/src/microcosm/build/uk/uk_national_targets.json:3641-3780`. |
| 4. Calibration seam cannot enter the ratified Logbook chain safely | **SURVIVES** | The pipeline remains `uk-national-calibration` (`packages/microcosm-build/src/microcosm/build/uk_runtime/calibration_run.py:52-54`), which still derives undeclared scope `uk/national`; the declared scopes remain `uk/frs` and `us` (`tools/logbook.py:63-67,192-200`; `logbook/README.md:3-5,22-44`). A fresh AST/order probe found staging write at line 227, build-record write at 259, record attempt at 265, and predecessor resolution only at 274 (`packages/microcosm-build/src/microcosm/build/uk_runtime/calibration_run.py:201-276`); export still enforces archive/pipeline scope agreement (`tools/logbook.py:419-438`). |
| 5. Release-candidate membership is operator-selectable through unsigned exclusions | **SURVIVES** | The parser still accepts `--release-candidate --measure-exclusions /private/tmp/.../operator.json`, and the custom file is still loaded/applied before the run (`tools/calibrate_uk_national_dataset.py:42-61,127-184`). The scoped battery still omits `uk_target_surface` (`packages/microcosm-build/src/microcosm/build/uk_runtime/calibration_run.py:87-116`). All five packaged exclusions still have only `name`, `reason`, and `tracking` (`packages/microcosm-build/src/microcosm/build/uk/calibration_measure_exclusions.json:1-30`), omitting the approval/adjudication/expiry fields enforced by the repository's complete exclusion receipt (`packages/microcosm-build/src/microcosm/build/uk_runtime/weighted_integrity.py:259-344`). |

### Vahid's second-pass findings: overlap and supersession

I read the four findings in [Vahid Ahmadi's second-pass PR comment](https://github.com/PolicyEngine/microcosm/pull/743#issuecomment-5408202783). None supersedes any of the five findings above; three are distinct defects and the fourth is safe in the current implementation.

| Vahid finding | Relationship to this audit | Current-head check |
|---|---|---|
| Duplicate household IDs collapse in `_aggregate_admin_totals` | **Distinct; no overlap or supersession.** This is an admin-anchor weighting bug, not the Logbook, scorer, target-membership, runbook, or boolean-count defect. | The person path still builds `dict(zip(household_id, household_weights))`, so duplicate clone IDs overwrite earlier weights before person lookup (`packages/microcosm-build/src/microcosm/build/uk_runtime/calibration_run.py:343-389`). |
| `_band_lower_edge` chooses the first sorted Ledger filter | **Adjacent to P0/3, but distinct.** Both affect target materialization; P0/3 concerns the aggregation operator, whereas this concerns which band metadata supplies a threshold. Neither fixes nor subsumes the other. | The function still scans all sorted `ledger_filter_*` keys and returns the first numeric/range match without relating the key to the binding's `groupby_variable` (`packages/microcosm-build/src/microcosm/build/target_materialization.py:291-318`). |
| `UKMeasureResolver.knows` makes the entity fence vacuous | **Adjacent to P0/1, but distinct.** P0/1 proves the documented builder omits the resolver altogether; this finding concerns entity validation when the resolver is used. | For every known variable, `knows` still returns `native == entity or entity in _ENTITY_ID`; a valid requested entity therefore passes even when it is not the variable's native entity (`packages/microcosm-build/src/microcosm/build/uk_runtime/measure_simulation.py:121-126`). |
| Resolver injection might mutate source frames | **Confirmed safe / not a current finding; no supersession.** | `UKFrameTargetAdapter.__init__` copies every entity table and link table before `_inject_measure_inputs` writes adapter state (`packages/microcosm-build/src/microcosm/build/uk_runtime/ledger_targets.py:134-140`). No source-frame mutation path was reproduced. |

### Round-2 verification record

- Required `uv sync --all-packages --extra us` was attempted first. With a
  task-local cache, lock resolution completed and package download stopped on
  DNS. No build was run.
- The fallback interpreter came from the sibling worktree whose `uv.lock`
  SHA-256 exactly matches this worktree:
  `0a36a8d2ae77adad1a64dde76fc8ebc57424abd51bf1c6a099d795416ee6b0cb`.
  All five current-worktree package source roots preceded it on `PYTHONPATH`.
- The corrected focused suite passed across
  `test_uk_calibration_run.py`, `test_uk_calibration_seam_driver.py`,
  `test_score_uk_national_candidate.py`, `test_uk_measure_simulation.py`,
  `test_target_materialization.py`, and the exact
  `test_packaged_materialization_skip_aborts_national_stage` regression.
- Fresh probes produced the values recorded in the matrix: 397 slash-named
  targets, production-shaped scorer refusal, arbitrary-label scorer acceptance,
  `[1, 0]` versus `[3, 0]` child counts, undeclared `uk/national` scope and late
  predecessor resolution, and acceptance of a custom release-candidate
  exclusion file.

## Round 1 — ranked findings

### P0 / 1. The documented held-run command takes a path that is known to abort on this production target surface

The runbook tells María to invoke tools/build_uk_national_dataset.py (docs/uk-national-calibration-runbook-623.md:20-36). That driver constructs UKNationalCalibrationStage directly, with neither the new measure resolver nor the new exclusion register (tools/build_uk_national_dataset.py:922-934). With no resolver, the stage returns no measure resolution (packages/microcosm-build/src/microcosm/build/uk_runtime/national_calibration.py:203-218) and fails when any activated measure cannot be materialized (national_calibration.py:89-99).

This is not hypothetical. The five salary-sacrifice targets named by the new exclusion register are input-substitution counterfactuals (packages/microcosm-build/src/microcosm/build/uk/uk_national_targets.json:1638-1816), while the frame adapter can only read an already-present counterfactual delta and otherwise raises (packages/microcosm-build/src/microcosm/build/uk_runtime/ledger_targets.py:167-180). The repository already contains the exact regression fixture: an activated packaged salary-sacrifice target without a precomputed delta must abort (packages/microcosm-build/tests/test_uk_national_calibration.py:478-494).

Evidence run:

- The exact no-resolver regression exited 0 while asserting the expected RuntimeError:
  packages/microcosm-build/tests/test_uk_national_calibration.py::test_packaged_materialization_skip_aborts_national_stage.
- A repository search confirmed only tools/calibrate_uk_national_dataset.py loads/applies the exclusions and constructs UKMeasureResolver (tools/calibrate_uk_national_dataset.py:57-61,88-103); the runbook never invokes that tool.

Required before merge: make the runbook use the actual calibrated seam, or wire the identical committed one-surface exclusions and resolver into the documented builder, then execute a production-shaped run fixture.

### P0 / 2. The decisive rule-1 scorer cannot score production exports and does not authenticate what it labels as the incumbent

The scorer loads the two exported H5s and calls score_targets directly (tools/score_uk_national_candidate.py:36-59). It never resolves or materializes the frozen register's measures. All 397 packaged UK target references use slash-named prepared measures; dwp/uc/households is representative (packages/microcosm-build/src/microcosm/build/uk/target_references.json:9532-9540). Calibration deliberately rebuilds pristine original tables before H5 export because those scratch columns cannot be written (packages/microcosm-build/src/microcosm/build/uk_runtime/national_calibration.py:351-357,380-396).

The new tests instead persist ordinary measure_a and measure_b columns in both H5s (packages/microcosm-build/tests/test_score_uk_national_candidate.py:20-69,110-147), so they are green on a shape that no production register uses.

The receipt is also identity-unbound:

- Both loaded provenance objects are discarded (tools/score_uk_national_candidate.py:48-49).
- The CLI accepts no expected candidate, incumbent, or registry digest (score_uk_national_candidate.py:186-198).
- Default labels are emitted independently of bytes (score_uk_national_candidate.py:41-42,90-93), even though the committed incumbent has a full filename/revision/SHA/size pin (packages/microcosm-build/src/microcosm/build/uk/efrs_parity_reference.json:382-391).
- The custom registry loader bypasses TargetRegistry.from_json's embedded-format/version checks (score_uk_national_candidate.py:164-177 versus packages/microcosm-calibrate/src/microcosm/calibrate/registry.py:301-327).
- Per-target errors are computed but discarded; the output has aggregate losses and win counts, not the claimed auditable drift table (score_uk_national_candidate.py:60-116,119-161).

Evidence run:

- A valid exported toy UK H5 plus a production-shaped dwp/uc/households TargetSpec raised:

      ValueError: No targets compiled into the constraint system (1 skipped):
      dwp.uc.households@2025: measure column 'dwp/uc/households'
      not on the 'benunit' table

- A read-only inventory returned targets=397, slash_named_measures=397, plain_measures=[].
- Two arbitrary synthetic H5s with SHA-256 78630c39... and 05d7a9c7... were accepted and reported as candidate=populace_uk_2023 and incumbent=enhanced_frs_2024_25; the output contained neither digest.
- Both new scorer tests passed, confirming the false-green persisted-column fixture rather than closing this gap.

Required before merge: score both sides through the same target-resolution/materialization route, fail on every skipped target, load the content-addressed registry through its validating loader, verify both H5s against expected full pins, emit the per-target drift table, and cross-pin the score receipt from signed run evidence.

### P0 / 3. Boolean person-count targets are collapsed with any, so published child totals become household indicators

compute_uk_measure_input chooses aggregation solely from dtype. Every person-native boolean mapped to a household is grouped with max/any (packages/microcosm-build/src/microcosm/build/uk_runtime/measure_simulation.py:39-58). That is correct for a predicate such as is_disabled, but not for a count measure.

The changed UK contract uses boolean is_child or uc_is_child_limit_affected as the numeric value for person-count targets mapped to households. Representative rows are children_affected (packages/microcosm-build/src/microcosm/build/uk/uk_national_targets.json:3645-3671), children_in_affected_households (uk_national_targets.json:3674-3704), and children_in_3_children_households (uk_national_targets.json:3749-3779). A read-only inventory found eight such person-count bindings. The added test covers only the legitimate any case and asserts [1.0, 0.0] for is_disabled (packages/microcosm-build/tests/test_uk_measure_simulation.py:80-88); it never tests a boolean used as a count.

Evidence run:

- A four-person fixture with three children in household 100 returned:

      route: bool_any_collapse_person_to_household
      resolved: [1.0, 0.0]
      required_for_person_count: [3.0, 0.0]

- The focused measure-resolution/materialization suite passed, demonstrating that its current fixtures do not exercise count-versus-any semantics.

Impact: the eight affected contract rows constrain counts of qualifying households, not the published counts of children. A green solve can therefore be calibrated to the wrong statistic.

Required before merge: make reduction an explicit binding semantic and use sum for count measures; add direct tests for every changed two-child-limit row class.

### P0 / 4. The new seam is operationally unloggable and writes artifacts before validating its chain predecessor

The seam declares pipeline uk-national-calibration (packages/microcosm-build/src/microcosm/build/uk_runtime/calibration_run.py:52-54). Logbook derives that as scope uk/national (tools/logbook.py:192-200), but the closed vocabulary contains only us and uk/frs (tools/logbook.py:60-67; supabase/migrations/20260818000000_logbook_chain_scopes.sql:54-69). The FRS spine, staging, and imputation stages are required to share uk/frs (logbook/README.md:22-29), and export rejects a spool whose pipeline does not match its archive scope (tools/logbook.py:419-438).

The seam also records only after successful diagnostics, gates, staging H5, and build-record writes (calibration_run.py:201-276). There is no exception/refusal recording path, despite the binding rule that successful, failed, refused, and discarded attempts all produce rows (logbook/README.md:3-5). resolve_predecessor is deferred to that final call (calibration_run.py:265-275), whereas the existing UK driver validates it before any side effect (tools/build_uk_national_dataset.py:595-626). Finally, build_id is deterministic per release ID (calibration_run.py:149-153), but both the local chain and database reject duplicate build IDs (packages/microcosm-build/src/microcosm/build/logbook.py:481-490; supabase/migrations/20260805000000_logbook.sql:189-193).

Evidence run:

- Scope probe: derived_scope=uk/national, declared=False, declared_scopes=[uk/frs, us].
- With conflicting CLI and environment predecessors, a synthetic seam run raised the disagreement only after staged.h5, diagnostics.json, build.json, and gates.json existed; logbook_rows=[].
- With a wrong input pin, the seam raised before creating its spool; logbook_rows=[].
- The added success test asserts only that a spool file exists (packages/microcosm-build/tests/test_uk_calibration_run.py:147-164); it does not export it into the ratified archive or exercise failure/retry behavior.

Required before merge: use an uk-frs-* pipeline, unique attempt IDs, prevalidate predecessor configuration, and wrap the full attempt so every terminal disposition is recorded.

### P0 / 5. Release-candidate target membership is operator-selectable through waivers with no owner receipt

The new CLI accepts an arbitrary --measure-exclusions path (tools/calibrate_uk_national_dataset.py:127-142), applies it directly to prune the compiled Ledger registry (calibrate_uk_national_dataset.py:50-61), and does not forbid that option under --release-candidate (calibrate_uk_national_dataset.py:151-167). The scoped calibration battery explicitly excludes uk_target_surface (packages/microcosm-build/src/microcosm/build/uk_runtime/calibration_run.py:87-116), so it cannot detect that the operator narrowed membership before the solve.

The committed register's five entries contain only name, reason, and tracking (packages/microcosm-build/src/microcosm/build/uk/calibration_measure_exclusions.json:1-30). The loader requires only a nonempty reason and permits empty tracking (measure_simulation.py:143-180); the applied receipt drops tracking and retains only name-to-reason (measure_simulation.py:183-198). This is weaker than the repository's exclusion-generic receipt contract—approver, adjudication, approval date, and expiry, all sealed (packages/microcosm-build/src/microcosm/build/uk_runtime/weighted_integrity.py:259-344).

Evidence run:

- A canonical release ID plus --release-candidate and --measure-exclusions operator.json parsed successfully:

      release_candidate=True
      custom_measure_exclusions=operator.json
      accepted=True

- A register scan found all five entries missing approved_by, adjudication, approved_on, and expires_on.
- The loader/applier tests passed with entries that contain only name and reason, confirming this permissive contract.

Required before merge: remove the release-candidate membership option, keep one committed target surface, and migrate any adjudicated exclusions to the complete approval/expiry schema without discarding receipt fields.

### P1 / 6. The documented path signs a caller-supplied diagnostics digest before the diagnostics exist

The runbook asks the operator to supply --calibration-diagnostics-sha256 while invoking the build (docs/uk-national-calibration-runbook-623.md:20-36). The driver passes that string into GateBatteryRun release evidence before calibration diagnostics are generated (tools/build_uk_national_dataset.py:936-979; packages/microcosm-build/src/microcosm/build/uk_runtime/national_build.py:333-342). Only after the gated build returns does the driver write calibration_diagnostics.json (tools/build_uk_national_dataset.py:980-1004). It never hashes the resulting file or compares it with the supplied value; the build record copies the supplied value back out (tools/build_uk_national_dataset.py:1218-1229).

Evidence run:

- The specific driver test passed while forwarding and recording the arbitrary value c repeated 64 times (packages/microcosm-build/tests/test_uk_national_build_driver.py:240-245,295-299). The test never compares that value with diagnostics bytes.
- Code-path tracing found no post-write comparison or diagnostics entry in the driver's artifact SHA census.

Impact: the held run can produce signed terminal evidence and a build record that falsely claim a diagnostics identity. A later promotion verifier may reject it, but that does not make this run receipt honest.

Required before merge: write diagnostics first, hash their actual bytes, then construct/re-sign the terminal evidence and build record from that measured digest.

### P1 / 7. The “independently sourced” incumbent target-surface gate derives both sides from the candidate stage

The implementation says the reference must be independently sourced (packages/microcosm-build/src/microcosm/build/uk_runtime/national_build.py:707-719), but candidate_targets comes from the stage's diagnostics and reference_targets comes from the same stage's registry (national_build.py:726-758). EfrsParityReference contains source identity and incumbent input-column shares only; it has no incumbent target surface (packages/microcosm-build/src/microcosm/build/uk_runtime/parity_reference.py:49-56). The release-blocking gate nevertheless claims exact candidate/incumbent target-surface agreement (packages/microcosm-build/src/microcosm/build/uk/gates.json:302-307).

Evidence run:

- A direct probe used a fake parity reference with no target data and a one-row candidate registry. It returned candidate_targets=[only.target@2025], reference_targets=[only.target@2025], same_values=True.
- The added parity test passed with a synthetic reference pinned to an all-zero SHA (packages/microcosm-build/tests/test_uk_national_build.py:1237-1300).

Related completeness gap: both calibration paths write score_vs_enhanced_frs as null (calibration_run.py:181-200; tools/build_uk_national_dataset.py:996-1003), while the scorer produces only a loose JSON (tools/score_uk_national_candidate.py:180-199). The runbook lists a “twin-build payload identity receipt” (docs/uk-national-calibration-runbook-623.md:41-48), but no changed producer writes one. Thus neither the target-surface claim nor the decisive score is sealed into the run evidence.

Required before merge: source the reference target surface from a frozen, incumbent-bound instrument and cross-pin the scored/twin-build receipts into signed diagnostics, gates, and build record.

### P1 / 8. Required Ledger and staging posture facts are verified or declared, then omitted from identity

The standalone CLI verifies both the Ledger facts and manifest hashes (tools/calibrate_uk_national_dataset.py:42-48), but its source_pins contain only input_h5 and a facts SHA/size (calibrate_uk_national_dataset.py:94-110,244-248). run_uk_calibration accepts ledger_artifact but never uses it (packages/microcosm-build/src/microcosm/build/uk_runtime/calibration_run.py:119-135). The required manifest identity and profile therefore disappear from diagnostics, build record, and Logbook identity.

The same CLI accepts --release-candidate for any caller-pinned input H5 (tools/calibrate_uk_national_dataset.py:127-167), while the run always records the input tier as staging_candidate and the output as shippable=false (calibration_run.py:181-189,231-246). The existing national builder correctly refuses release-candidate posture with a declared staging input (tools/build_uk_national_dataset.py:352-355).

Evidence run:

- Repository tracing found ledger_artifact only in the calibrator call and the unused run_uk_calibration parameter.
- The release-candidate parser probe above succeeded on a generic input path; there is no certified-input verifier in this CLI.

Required before merge: seal the verified manifest/profile pin into source_pins and all downstream identities; either refuse release-candidate on this explicitly staging-only seam or define and enforce a certified-input contract.

## Verification record

- Required setup was attempted first. uv sync --all-packages --extra us failed because the shared uv cache is not writable; a task-local-cache retry resolved the lock and then failed at package download because the host could not resolve PyPI/GitHub.
- Tests used /Users/maxghenis/PolicyEngine/_worktrees/microcosm-peus-bump/.venv/bin/python, whose uv.lock SHA-256 exactly matches this worktree: 0a36a8d2ae77adad1a64dde76fc8ebc57424abd51bf1c6a099d795416ee6b0cb. PYTHONPATH forced every Microcosm package import to this review worktree.
- Independent focused slice: 36 tests passed across test_uk_calibration_run.py, test_uk_calibration_seam_driver.py, test_score_uk_national_candidate.py, test_uk_measure_simulation.py, and test_target_materialization.py.
- Runtime-focused slice: 49 tests passed across the scorer, measure resolver, target materializer, and national calibration tests. The two receipt/parity tests cited above also passed. These results establish that current green tests do not cover the adversarial production shapes.
- Direct probes covered production-shaped scoring, arbitrary artifact labelling, boolean count aggregation, Logbook scope and predecessor ordering, release-candidate custom membership, target-surface aliasing, exclusion receipt fields, and Ledger provenance threading.
- The UK spec-bundle digest was independently recomputed as 0674aeda983d218250093be8211832f0ea549a3f435076bcf75ef2664e2a49a6 and matches the updated pin in packages/microcosm-build/tests/test_spec_engine_country_bundles.py:44-45.

## Defensive non-findings

- I found no weakened terminal-fit threshold, default seed, or release-gate parameter in the reviewed path.
- I found no publication, upload, promotion, or certification side effect. The new seam explicitly marks its artifact non-shippable.
- The staging H5 input is SHA-checked before the standalone measure resolver consumes it.
- These positives do not mitigate the HOLD findings: the documented run and acceptance instruments cannot currently produce honest, operationally binding evidence on the production surface.
