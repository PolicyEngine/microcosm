# #893 reconciliation to main's amended graph interface — lane report

> **12 September Codex continuation:** this is the historical layer proposal
> at `80fef2217`, not an approved extraction plan. The final battery
> `bt0nx8fcc` was killed during its first pytest group: CI-group verification
> and coverage check exited zero, but no final consumer result was produced.
> Independent review found that G4 exposes mutable admitted populations to
> observers and that the layer order omits hard packaged-resource dependencies.
> Fix and review both before extracting or landing layers. PR912 is also held
> for fixes from Astra's executed review; the branch-head assumption below is
> superseded until its corrected head lands.
>
> Resource repair `a4b6fe88c` restores all three omitted ASEC JSON contracts
> from their exact historical blobs, matching unchanged loader pins. Eight
> targeted tests and wheel-content verification pass. These three resources,
> `acs_2024_housing_universe.json`, and current-money domains/consumers/price-basis
> resources must accompany their first PR-1 loaders or an earlier prerequisite;
> Python import analysis alone cannot establish extraction order. No engine
> defaults were admitted and no source/engine identity was refreshed.

> **Later 12 September correction:** `6f66fa545` closes the demonstrated G4
> mutation defect by handing observers detached snapshots, including nested
> table values, metadata, schema, weights and ledgers. Independent review is
> clean; 524 graph tests, five focused cases and all four existing synthetic
> consumers passed, including financial and calibration teardown checks. An
> earlier source-edit-during-test failure remains preserved and is excluded.
> Entry 7.4 below is historical: the current observer need not be trusted to
> leave its snapshot unmodified. Retaining snapshots still costs memory.
>
> `cb27bdb0b` makes the offline explanation display executor-owned unreached
> and gate-exception states independently from cache hits. Sixteen graph
> explanation/actual-gate cases and two free-form diagnostic controls passed.
> Its new `explain → availability` import is recorded in the implementation
> inventory; availability was already bound as a whole module in all ten
> using scopes. No scope, dependency, resource, or unbound-use exemption was
> added. All eight prepared-resource/full-manifest tests passed afterward.
>
> Corrected PR912 head `5e33d6971` is undergoing exact-head CI and Fable review.
> Graph-only extraction is being prepared against that revision, incorporating
> G1–G4 and numbered charter entries. Neither that extraction nor the remaining
> country layers are declared accepted by this historical report.

> **Subsequent 12 September integration checks:** corrected PR912 head
> `5e33d6971` received Fable approval; its merge remains conditional on CI.
> The local integration merged that exact revision in `9e61d0d04` and
> recomputed the combined source/seed identities. All 734 graph and fitting
> tests passed. The six-file spec/build sweep collected 248 tests: 247 passed
> and one failed because its invented PUMA fixture still used schema 1 and
> omitted the required joint tract/CD population support. The fixture now
> supplies consistent schema-2 support and explicit invented provenance; that
> single test passed when rerun. The original failed sweep and intermediate
> incomplete fixture repair are preserved. This is separate-run evidence,
> not a claim that the complete 248-test sweep was rerun successfully.
>
> Shared graph consolidation is now public as PR913, temporarily based on
> PR912, with independent Fable review in progress. Its runtime matches this
> integration's graph implementation. Native strict replay, PUF55 host
> acceptance and complete national/CD release acceptance remain separate
> work and are not established by these software checks.


Date: 2026-09-12. Worktree `~/PolicyEngine/_worktrees/microcosm-us-launch-verified-lanes-20260910`,
branch `microcosm-us-launch-verified-lanes-20260910` (PR #893's branch is
`microcosm-us-launch-integration-20260909`). Started at `069d5ed9a`. No push, no
new branch, no stash, `uv.lock` untouched. Every commit below is on the
checked-out branch.

## 1. Outcome

- `origin/main` (`0c3f4f651`, amendment 19) and
  `origin/amend-keyed-seed-and-uniform-draws` (`23ba24770`, amendment 20 / PR
  #912) are merged. **#912 was merged from its branch head**, so once it lands
  the dispatcher should re-run `git merge origin/main`; for the graph and fit
  packages, `docs/graph-interface.lock` and `docs/graph-acceptance.md` that is
  expected to be a no-op (this tree already carries `23ba24770`'s bytes for all
  of them, verified with `git diff --quiet 23ba24770 HEAD -- <paths>`).
- Every frozen or amended file equals `23ba24770` byte for byte: `decl.py`,
  `kernel.py`, `keys.py`, `serialize.py`, `view.py`, `artifact_edges.py`,
  `randomness.py`, `population.py`, `fit/qrf.py`, the lock, the charter,
  `test_graph_kernel_contract.py`, and every `test_acceptance_*` file. The
  interface-lock test passes (it is part of the 518-test graph run below).
- Four not-yet-amended graph pieces the US runtime consumes were re-applied
  as separate named commits on top of main's files; four pieces with no
  consumer were dropped (section 3).
- The five graph tests the brief named are green; the whole graph package is
  green (518 passed); the fit package is green (191 passed); the spec-engine
  check, `ci_test_groups --verify` and ruff are clean. Build-side results are
  in section 5.
- The layer map (section 6) partitions the branch's remaining 407-file diff
  against `origin/main` into the eleven staged layers, the later closure and
  post-staging work, and the graph pieces, with per-layer file lists, test
  files, import dependencies, and a proposed landing order. Section 7 has a
  draft charter amendment entry per kept graph piece. Section 8 lists what
  needs Max.

Lane commits (newest first):

```
07566eb17 Re-pin the US implementation inventory for the reconciled graph package
c704a1be6 Pin that unreached propagates through a structural node and its version
2e1857a4b Bring the #893 reconciliation journal up to the piece commits
a16eeacf8 Re-apply the private population observer on run_graph
d1019762b Re-apply executor execution states: gate exceptions leave their consumers unreached
db1b7821a Re-apply complete Frame metadata storage and non-finite JSON refusal in the store
752ab840f Re-apply the raw-byte source codec on main's graph package
cff8fbf32 Re-pin the calibrate and simulate H1 parity cases the branch's kernels move
dc621c14c Re-pin the seed protocol and compiled seed map digests on the merged tree
051357909 Merge origin/amend-keyed-seed-and-uniform-draws (23ba24770, amendment 20)
3010b7788 Merge origin/main (0c3f4f651, amendment 19) into the #893 integration branch
3ad1b1aae Open the #893 reconciliation lane journal
```

## 2. The two merges and the per-file resolution

Baseline before any change (`uv run --no-sync pytest packages/microcosm-graph/tests`):
5 failed, 362 passed, exit 1 — exactly the five the brief names
(`test_acceptance_b_ownership::test_b2_executor_enforces_ownership`,
`test_acceptance_h_parity::test_h1_kernel_parity`,
`test_graph_executor::test_fit_qrf_tolerance_source_hash_pin_is_current`,
`test_graph_kernel_contract::test_context_numerics_default_empty_and_carry_scopes`,
`test_graph_serialize::test_generated_parity_graphs_bind_real_kernels_and_direct_bytes`).

### Merge 1: `git merge origin/main` (commit `3010b7788`)

Git reported 8 content conflicts plus one in `PROGRESS.md`; `store.py`,
`__init__.py` and `PROGRESS.md` auto-merged. Resolution:

| Path | Resolution |
| --- | --- |
| `packages/microcosm-graph/src/microcosm/graph/{artifact_edges,decl,executor,kernel,keys,manifest,serialize,view}.py` (conflicts) | main's bytes (`git checkout origin/main -- packages/microcosm-graph`) |
| `packages/microcosm-graph/src/microcosm/graph/{store,__init__}.py` (auto-merged) | main's bytes, same checkout; the branch's hunks return as pieces A and B |
| `packages/microcosm-graph/src/microcosm/graph/{attachments,availability,schema}.py` (branch-only) | removed in the merge (`git rm`); `availability.py` returns as piece C, the other two are dropped |
| `packages/microcosm-graph/src/microcosm/graph/randomness.py` (branch-only until amendment 20) | kept as the branch had it; replaced by `23ba24770`'s in merge 2 |
| `packages/microcosm-graph/tests/*` | main's bytes; the branch's copies were strict subsets of main's (numstat showed only removals), so this also dropped nothing of the branch's; `test_frame_metadata_store.py` and `test_lazy_snapshot_metadata_integration.py` (branch-only) removed here, the first returns with piece B, the second is dropped |
| `packages/microcosm-graph/tests/fixtures/parity/kernels/fit.qrf/pins.json` | main's (consistent with fit/qrf.py being main's in this merge) |
| `packages/microcosm-fit/src/microcosm/fit/qrf.py` (auto-merged to the branch's) | main's bytes; the branch's `predict_from_uniforms` returns with amendment 20 in merge 2 (the branch's qrf.py was byte-identical to `23ba24770`'s) |
| `packages/microcosm-fit/src/microcosm/fit/{_graph_legacy_apply,_graph_legacy_qrf,graph_legacy_apply,graph_legacy_apply_matrix,graph_legacy_qrf,graph_legacy_train,model_input,qrf_target}.py` (branch-only) | kept: not amended content, consumed by the US runtime (layer 01) |
| `docs/graph-interface.lock`, `docs/graph-acceptance.md` | main's bytes |
| `PROGRESS.md` | both sides kept: this lane's section, then main's Amendment 19 and #907 sections, then the branch's history |
| everything else | ordinary auto-merge |

### Merge 2: `git merge 23ba24770` (commit `051357909`)

Six conflicts:

| Path | Resolution |
| --- | --- |
| `packages/microcosm-graph/src/microcosm/graph/randomness.py` (add/add) | `23ba24770`'s (adds the signed-zero normalisation the branch lacked) |
| `packages/microcosm-fit/tests/test_qrf_stateless.py` (add/add) | `23ba24770`'s (182 lines; the branch's 159-line copy was a strict subset) |
| `docs/evidence/spec-engine/us-f0-coverage.json` | `23ba24770`'s, then recomputed (section 4) |
| `packages/microcosm-build/src/microcosm/build/spec_engine/inventory_coverage.py` (seed digests) | `23ba24770`'s, then recomputed (section 4) |
| `packages/microcosm-build/tests/test_spec_engine_country_bundles.py` (AM/BE/UK digests) | `23ba24770`'s; see section 5 for the merged-tree check |
| `packages/microcosm-build/tests/test_us_multispine_pool_tool.py` (pool tool `spec_sha256`) | `23ba24770`'s; see section 5 |

After merge 2 the graph package, fit sources, lock and charter were
byte-identical to `23ba24770` except the eight branch-only fit modules above;
`docs/graph-interface.lock` matched `shasum -a 256` of the two frozen files.

## 3. The not-yet-amended graph pieces

Consumers were found by an AST scan of every `.py` under
`packages/microcosm-{build,calibrate,fit,frame,data}` and `tools/` for names
imported from `microcosm.graph.*` that `23ba24770`'s graph package does not
define, plus greps for the keyword-level uses (`_population_observer=`,
`population_retention=`, `emit_validation_artifact=`).

### Kept (each re-applied on main's file as its own commit, with tests)

| Piece | Commit | Consumers outside the graph package | Graph-level tests |
| --- | --- | --- | --- |
| A. `codecs.py` raw-byte mode: `SourceBytesCodec`, `register_bytes`/`load_bytes`, `raw-bytes-v1`, `RAW_BYTES_MAX_BYTES`, `load_source_bytes` (+ two `__init__` exports) | `752ab840f` | 13 modules: `graph_atomic_geography`, `graph_geography`, `graph_sources`, `graph_atomic_survey_population`, `graph_puf55_canonical_donor`, `puf_raw_source`, `puf_monetary_source`, `puf_monetary_agi_projection`, `puf_full_source_graph`, `atomic_block_sources`, `atomic_block_api_sources`, `survey_atomic_geography`, `puf55_survey_recipients`; 8 US test files | the branch had none; 4 added to `test_graph_codecs.py` (8 passed) |
| B. `store.py` Frame-metadata storage (`microcosm-graph-frame-v2`, `_encode/_decode_frame_metadata`, `frame_metadata_sha256`, `_put(validate_existing=…)`) and the non-finite JSON decode hooks | `db1b7821a` | `survey_population_replay.same_replayed_frame`, `population_input_coverage`, `graph_full_puf_enrichment` (all call `_encode_frame_metadata`), and every US stage relying on `Frame.metadata` surviving a store round trip | `test_frame_metadata_store.py` (the branch's file, 19 collected tests) |
| C. `availability.py` + executor/manifest execution states: `gate_exception`, `unreached`, `blocked_by`, cache-record schema 3, manifest schema 4, refusal of kernel-authored execution metadata; replaces amendment 19's refusal of a gate declaring a typed output | `d1019762b` | `graph_atomic_survey_clone` (`atomic_geography_nodes(..., emit_validation_artifact=True)` makes the geography gate declare a typed output, which main's executor refuses outright), its consumers `survey_age_calibration`, `graph_current_survey_predictors`, `survey_origin_budget`; `population_input_coverage` (`execution_state` on every producer receipt); build test `test_atomic_geography.py` parametrises the declaration | the branch had none; main's `test_a_gate_kernel_may_not_declare_a_typed_artifact_output` replaced by 6 tests in `test_graph_executor.py` (five in `d1019762b`, the structural-propagation one in `c704a1be6`) |
| D. `run_graph(_population_observer=…)` | `a16eeacf8` | `survey_age_calibration`, `graph_atomic_survey_financial`, `graph_survey_population`, `graph_atomic_survey_population`; 8 US test files | the branch had none; 1 added |

What was deliberately not re-applied inside those files: the branch's
docstring deletions on the amended surface; its `_opaque_artifact_key`
inline hash (main's `graph_keys.opaque_artifact_key` is the same
derivation); its `KernelContext` argument order (frozen file); its eager
artifact payload reads on cache hits (main authenticates edges without
reading, `9018c4420`); its `StoreMiss`-on-hit for a missing declared artifact
(main's `5c4a8efde` decision, `StoreCorrupt`); the removal of main's #907
datetime64/timedelta64 object-leaf refusal in `store.py`; and main's
`_validate_typed_ancestry` gate-ancestry shape check and the
`NodeRejectedError`→`ValueError` conversion (`402d9a631`, `7d45c32e5`), which
the branch's older copy lacked.

### Dropped (no consumer left)

| Piece | Why dropped |
| --- | --- |
| `attachments.py`, `_PopulationRetention`, `_LazyPopulations`, `run_graph(population_retention="lazy")`, `PopulationView.__slots__ = ("__weakref__",)` (staged as layer 08 GRAPH-ATTACHMENT-METADATA) | no call site outside the graph package passes `population_retention`; its only consumer was the branch's own `test_lazy_snapshot_metadata_integration.py`, dropped with it. It is a memory optimisation for long graphs (the native pilot's 11.5 GB peak is in the PR body); if it is wanted it is a self-contained later PR on top of piece B, which it imports. |
| `schema.py` (`graph_schema`, 316 lines) | imported by nothing in the repository (the `graph_schema_version` hits in the spec bundle are unrelated). |
| `keys.py` `_stream_file` chunked source hashing | no consumer and no test; it changes no identity (same digest sequence, it only bounds memory and refuses a file that changes while hashed). Main's `keys.py` is kept byte-identical. Candidate for a small standalone PR if large-source memory matters. |
| the `_write_node` per-coordinate memory refactor and the `del series` | no consumer; behaviour-preserving; main's amended `_write_node` kept. |

## 4. Identities re-pinned on the merged tree, and why

- **Seed protocol and compiled seed map digests** (`inventory_coverage.py`
  `EXPECTED_HASHES`, `docs/evidence/spec-engine/us-f0-coverage.json`;
  commit `dc621c14c`). Both hash the exact source bytes of the attested kernel
  modules in `spec_engine/seeds.py` (no `microcosm.graph` module is on that
  list, so the graph pieces do not move them). The merged tree carries the
  branch's `acs_transfer` and `housing_inputs` changes and, through amendment
  20, the same `fit.qrf` bytes the branch already carried, so the actual
  digests are the values the branch pinned in `1734b9e90`
  (`a775cccd…` / `c15bff65…`), not `23ba24770`'s (computed on a tree without
  the two US kernel changes). Computed directly (`spec.seed_protocol.implementation_sha256`,
  `sha256_json(compiled.seed_stream_map.to_wire())`), then the report was
  regenerated with `tools/spec_engine_coverage.py` and `--check` exits 0.
- **H1 parity pins for `calibrate` and `simulate`** (commit `cff8fbf32`).
  Six graph tests were red on the merged tree, all on these two cases.
  `CalibrateKernel.implementation_hash` binds `microcosm.calibrate`'s solve and
  diagnostics modules, which the branch's grouped-bound solver path changes
  (layer 05); `SimulateRulesKernel.implementation_hash` binds
  `microcosm.frame.bundle`, to which the branch adds `Frame.__reduce__` (layer
  01). Re-recorded with `tools/graph_parity_repin.py calibrate` and
  `… simulate`, which keep every pinned platform (both cases pin only
  `arm64/darwin/py3.14`) and leave `direct.csv` untouched. The `fit.qrf` pin
  came from `23ba24770` and needed no move.
- **The US implementation inventory** (`us_runtime/graph_implementation_inventory.json`;
  commit `07566eb17`). This reviewed dependency fence is checked by every US
  source stage before it runs (`graph_implementation.implementation_manifest`),
  and its refusal surfaces as `PREPARATION_ISSUANCE_REFUSED` (the preparation
  wraps every exception; the true cause was recovered from the exception's
  `__context__`). Three entries no longer held: the `microcosm.graph` roster
  listed the dropped `attachments.py` and `schema.py` (39 survey tests red on
  that alone); the contracts for `executor.py`, `manifest.py`, `keys.py` and
  `population.py` reflected the branch's versions rather than main's plus the
  kept pieces, and `attachments.py` was a listed stage module in ten stages;
  and `congressional_district_vintage.py`'s contract predated main's
  `05ff48f05` (merged into the branch on 2026-09-11), which added the
  `microcosm.calibrate.geography_constants` import — an inherited drift the
  branch already carried at `069d5ed9a`, verified by recomputing the
  pre-merge tree's contracts against its own inventory. Recomputed with the
  module's own `_dependency_contract` / `_covered_imports`; the import
  classification set is the exact union again. Seven of the ten stages now
  build their manifest; three cannot (section 8.5).
- **Country-bundle digests and the pool tool's `spec_sha256`**: resolved to
  `23ba24770`'s values in merge 2; their merged-tree check is in section 5.

## 5. Commands, exit codes, counts

All from the worktree root with the locked engine environment
(`uv sync --all-packages --locked --extra us --extra uk` → exit 0 before any
change). Tests ran with `uv run --no-sync pytest … -p no:cacheprovider`.

| Stage | Command | Result |
| --- | --- | --- |
| baseline (`069d5ed9a`) | `pytest packages/microcosm-graph/tests` | 5 failed, 362 passed, exit 1 |
| baseline | `python tools/ci_test_groups.py --verify` | exit 0 |
| after both merges (`051357909`) | `pytest packages/microcosm-graph/tests` | 6 failed, 485 passed, exit 1 (calibrate/simulate parity pins; section 4) |
| after both merges | `pytest packages/microcosm-fit/tests` | 191 passed, exit 0 |
| after both merges | `python tools/spec_engine_coverage.py --check` | exit 1 (`seed protocol content digest differs`, `compiled seed map digest differs`) |
| after both merges | `python tools/ci_test_groups.py --verify` | `verification=ok`, exit 0 |
| after both merges | `ruff check .` | All checks passed, exit 0 |
| seed re-pin (`dc621c14c`) | `python tools/spec_engine_coverage.py` then `--check` | exit 0 / exit 0 |
| parity re-pin (`cff8fbf32`) | `python tools/graph_parity_repin.py calibrate`; `… simulate` | exit 0 / exit 0 |
| parity re-pin | `pytest test_acceptance_h_parity.py test_graph_parity_pins.py test_graph_parity_repin.py test_graph_serialize.py` | 27 passed, exit 0 |
| piece A (`752ab840f`) | `pytest packages/microcosm-graph/tests/test_graph_codecs.py` | 8 passed, exit 0 |
| piece B (`db1b7821a`) | `pytest test_frame_metadata_store.py test_graph_store.py test_graph_population.py test_graph_executor.py test_graph_manifest.py` | 274 passed, exit 0 |
| piece C (`d1019762b`) | `pytest packages/microcosm-graph/tests` (whole package, includes the interface-lock test) | **518 passed, exit 0** |
| piece C | `ruff check .`; `ruff format --check packages/microcosm-graph` | exit 0 / exit 0 |
| piece D (`a16eeacf8`) | `pytest test_graph_executor.py test_acceptance_b_ownership.py test_acceptance_f_gates.py` | 94 passed, exit 0 |
| structural test (`c704a1be6`) | `pytest packages/microcosm-graph/tests/test_graph_executor.py` | 82 passed, exit 0 |
| inventory re-pin (`07566eb17`) | `pytest test_us_survey_origin_budget.py::test_actual_initial_budget_and_zero test_us_survey_age_calibration_run.py::test_actual_seven_node_calibration_preserves_full_population` (two of the 39 formerly refused) | 2 passed, exit 0 |
| final tree | `python tools/ci_test_groups.py --verify` | exit 0 |
| final tree | `python tools/spec_engine_coverage.py --check` | exit 0 |

Build-side and shared consumers, final tree (`07566eb17`). A first battery at `a16eeacf8` found 39 survey tests refused (`PREPARATION_ISSUANCE_REFUSED`) — all traced to the implementation inventory above, re-pinned in `07566eb17` — and one non-existent file name in my own raw-bytes group list (`test_us_puf_raw_source.py`, removed); the battery was then rerun in full:

_(Historical draft marker resolved: task `bt0nx8fcc` was killed during availability consumers. No final battery counts or exit code exist; later groups did not start.)_

## 6. Layer map: landing #893 as sequential PRs

Basis: `git diff --name-status origin/main HEAD` = 407 files (340 added, 67
modified, 0 deleted), each attributed to the first branch commit that touched
it (`git log --name-only --no-merges 3094bfe84..HEAD`), with the eleven
`Stage integration layer NN` commits mapped to their layer names, the later
staging commits to layers 12–14, the sixty post-staging commits of 10–12
September grouped by subject (P1–P6), and the graph pieces of section 3 as
their own layer. Import dependencies come from an AST scan of every changed
`.py` file resolved against that attribution (`microcosm.*` imports only).
The scripts and their JSON output are in this lane's scratchpad
(`layers.py`, `layers2.py`, `layers.json`, `per-file-deps.json`).

Two corrections to the staged layer names as they sit in git history:

- Layer 08 GRAPH-ATTACHMENT-METADATA is empty in the map: its only file,
  `attachments.py`, is dropped (section 3).
- Layer 09 F-JOINT-GEOGRAPHY-GATE has no file of its own: its 27-line change
  lives in `graph_geography.py`, which layer 01 created, so it lands with
  layer 01.
- Nineteen files differ from `origin/main` only because `origin/main` does not
  yet carry amendment 20 (`kernel.py`, `randomness.py`, `fit/qrf.py`,
  `test_graph_kernel_contract.py`, `test_graph_parity_*`,
  `test_graph_randomness.py`, `tools/graph_parity_repin.py`, the three
  `pins.json`, the lock, the charter, the amendment-20 journal/receipts/
  changelog fragment) or were re-pinned by this lane. They vanish from the diff
  when #912 lands and are not #893 content; they are listed under
  "already on main's way" below and belong to no PR.

### Dependency facts the order rests on

Three kinds of link were measured, strongest first:

1. **Hard**: a file imports a module that does not exist on `origin/main`
   (a file the branch adds). These force order.
2. **Name-hard**: a file imports, from a module main already has, a *name*
   that main's version of that module does not define (checked by parsing
   `git show origin/main:<file>`). Only three survive after discarding
   submodule imports that merely look like names: `GroupedUpperBounds`
   (layer 03's two-line re-export in `calibrate/__init__.py` of layer 01's
   `group_bounds.py`; imported by layer 12's and 13's tests),
   `decode_us_puma_ladder` (layer 06's `puma_ladder.py`; imported by layer
   13's `test_us_geography_integration.py`), and the graph pieces' names
   (`load_source_bytes`, `load_raw_bytes`, `RAW_BYTES_MAX_BYTES`,
   `_encode_frame_metadata`, `execution_state`).
3. **Soft**: a file imports a module main already has that the branch also
   modifies (`frame_checkpoint`, `asec_checkpoint`, `operator_boundary`,
   `us_runtime/__init__`, `calibrate/__init__`, `solve.py`, `bundle.py`,
   `variable_labels.py`, and amendment 20's `qrf.py`/`kernel.py`). These do
   not force order by themselves; each PR's own test run is what proves the
   modified behaviour is or is not needed.

Hard links between layers (source and tests):

| Layer | hard on | of which tests only |
| --- | --- | --- |
| 01 SAFE-ADDITIVE | nothing | — |
| 03 ACCEPTED-SHARED-RESTORE | 01 (`group_bounds`, `reported_coverage_source`, `_person_signal_summary`, `operator_column_contracts`, `table_identity`), P3 (`asec_raw_stage_v4`) | P3 link is `test_us_asec_checkpoint.py` only |
| 04, 06 | nothing hard (soft on 01/03) | — |
| 05 SOLVE-MERGE-PROPOSAL | 01 (`group_bounds`) | — |
| 07 F-CATALOGUE-OPTIMIZATION | 01 (`survey_population_preparation`) | all three files are tests |
| 10 SOURCE-CLOSURE | G1 (soft) | — |
| 11 PLACEMENT-ADDITIONS | 01 (the eight fit modules, `full_puf_enrichment`, `graph_sources`, `survey_population_replay`), G1, G2 | — |
| 12 ORDINARY-CLOSURE | 01 (fit modules), 03 (`GroupedUpperBounds`) | all files are tests/fixtures |
| 13 INTEGRATION-REGRESSIONS | 01 (`graph_geography`), 03 (`GroupedUpperBounds`), 06 (`decode_us_puma_ladder`) | all three are tests |
| P1 POST-CLONE-ATOMIC-GEOGRAPHY | 01 (`randomness` = amendment 20, fit modules, `graph_combined_clone`, `graph_sources`, `graph_survey_population`, `survey_*`), G1, G3 (the gate's declared artifact), P2 | P2 link is two tests (`test_us_current_survey_predictor_demographics.py`, `test_us_survey_origin_budget_predictor_compatibility.py`) |
| P2 PUF55-TWO-ROUTE-AND-CANONICAL-DONOR | 01 (`model_input`, `_graph_legacy_qrf`, `graph_legacy_*`, and eleven test-side modules), 11 (`graph_full_puf_enrichment`, two tests), P1 (`graph_atomic_survey_financial`, one test), G1 | 11 and P1 links are tests only |
| P3 SURVEY-POPULATION-CATALOGUE-AND-BUDGET | 01 (`reported_coverage_source` from `asec_raw_stage_v4`; six test-side modules), G2, G3 (`population_input_coverage`), P2 (`asec_demographic_source`) | P2 link is `test_us_asec_demographic_source.py` only |
| P4 | nothing hard (soft on 03) | — |
| P5 | nothing hard (soft on 01/03/04/06) | — |
| P6 UK-ADAPTER-DOCS-AND-RECORDS | P1 (`atomic_geography`) | — |

Two cycles exist and both are test-only: P1↔P2 (three tests) and 03↔P3
(one test). Every source-level dependency is acyclic.

### Landing order

1. **PR-G1 raw-byte codec** — piece A (`codecs.py`, `__init__.py`,
   `test_graph_codecs.py`). No dependency. Charter entry 7.1.
2. **PR-G2 Frame metadata storage** — piece B (`store.py`,
   `test_frame_metadata_store.py`,
   `changelog.d/20260906-frame-metadata-storage.fixed.md`). No dependency.
   Charter entry 7.2.
3. **PR-G3 execution states** — piece C (`availability.py`, `executor.py`,
   `manifest.py`, `test_graph_executor.py`). No code dependency, but it
   supersedes an amendment-19 ruling: **needs Max's decision (section 8)
   before it is opened.** Charter entry 7.3.
4. **PR-G4 population observer** — piece D (`executor.py`,
   `test_graph_executor.py`). Stacks on G3 only because both edit
   `executor.py`; G3+G4 can be one PR. Charter entry 7.4.
5. **PR-1 SAFE-ADDITIVE** — all of layer 01: 101 non-test files (100 modules
   plus `graph_implementation_inventory.json`; `randomness.py` is excluded, it
   is amendment 20) and its 26 test files, plus layer 03's two-line
   `calibrate/__init__.py` re-export of `GroupedUpperBounds` (it belongs with
   `group_bounds.py`) and layer 09's 27 lines (already inside
   `graph_geography.py`), and `graph_implementation_inventory.json`, which
   pins the `microcosm.graph` roster and module contracts and therefore has
   to be re-pinned against whatever main's graph package is when PR-1 opens
   (`07566eb17` shows the procedure). Hard on G1, G2 and G4 only (`_population_observer`
   in `survey_age_calibration` and `graph_survey_population`). Its 21 soft
   links are listed in the scratchpad `pr1-refined.json`; the PR's gate is
   its own 26 test files on main + G1–G4 + PR-1.
6. **PR-2 US runtime restore** — layers 03 (without the `__init__` re-export
   moved to PR-1, and with `asec_raw_stage_v4.py` moved in from P3 so
   `test_us_asec_checkpoint.py` runs), 04 (`puf_support.py`), 05
   (`calibrate/solve.py`), 06 (`us_runtime/__init__.py`, `puma_ladder*.py`,
   `tools/build_us_puma_ladder_artifact.py`, six tests), 12's calibrate
   tests and fixture, 13's three regressions and `docs/us-launch-review.md`,
   14 (`dfa7f872c`), and the two re-pins that these files move: the
   calibrate H1 pin (`solve.py`) and the seed digests (`acs_transfer.py`,
   `housing_inputs.py`). 03/04/05/06 touch disjoint files and could be four
   PRs (05 first, since only it is hard on 01); they are one here because
   their tests import across all four.
7. **PR-3 catalogue, source closure, placement** — layers 07 (three digest
   tests + fragment), 10 (`graph_acs_housing_universe.py`, eight JSON source
   definitions), 11 (`graph_full_puf_enrichment.py` + test), and 12's fit
   tests (`test_graph_legacy_*`, `test_qrf_target.py`,
   `legacy-slice-apply-v1.json`). Hard on PR-1 and G1/G2.
8. **PR-4 calibration diagnostics** — P4 (`variable_labels.py`, its test,
   fragments, experiments). Soft on PR-2.
9. **PR-5 survey population, catalogue and budget** — P3 minus
   `asec_raw_stage_v4.py`: `common_frame_export_contract`,
   `input_coverage_profile`, `population_input_coverage`,
   `frame/bundle.py`'s `__reduce__` (carries the simulate H1 re-pin), ten
   tests including `test_us_spine_blindness.py`;
   `test_us_asec_demographic_source.py` rides with PR-7 instead. Hard on
   PR-1, G2, G3.
10. **PR-6 post-clone atomic geography** — P1 (`atomic_geography.py`,
    `graph_atomic_geography.py`, `atomic_block_*`, `graph_atomic_survey_*`,
    `survey_atomic_geography.py`, `current_survey_geography.py`,
    `tools/ci_test_groups.py`, fourteen tests); its two P2-importing tests
    ride with PR-7. Hard on PR-1, G1, G3.
11. **PR-7 PUF55 two-route and canonical donor** — P2 (nineteen modules,
    twenty-one tests plus the three deferred from PR-5/PR-6, PUF docs and
    growth evidence, `pyproject.toml`'s `pythonpath` line). Hard on PR-1,
    PR-3 (11), PR-5, PR-6.
12. **PR-8 CI, seed-identity diagnostics and spine blindness** — P5
    (`.github/workflows/test.yml`, `tools/spec_seed_identity_diagnostics.py`,
    the seven `test_spec_seed_identity_*` files, `CLAUDE.md`,
    `test_spec_engine_loader.py`, `test_us_puf_support.py`,
    `test_us_stacked_spine.py`, the AM/BE/UK bundle digests). Soft on PR-2
    and PR-5; last because the diagnostic's roster names modules from every
    earlier PR.
13. **PR-9 UK adapter, docs and records** — P6 (`uk_runtime/atomic_area_support.py`,
    `atomic_household_identity.py`, two tests, `.gitattributes`, records).
    Hard on PR-6.

Every `experiments/` record and `changelog.d/` fragment rides with the PR
whose code it describes (listed per layer below). The "already on main's
way" files (amendment 20, listed at the top of this section) belong to no PR.

### Per-layer file lists

#### 00 JOURNAL  (1 non-test files, 0 test files)

- `./`: `PROGRESS.md`

#### 01 SAFE-ADDITIVE  (101 non-test files, 26 test files)

- `packages/microcosm-build/src/microcosm/build/`: `survey_allocation.py`, `survey_domain_sample.py`, `table_identity.py`
- `packages/microcosm-build/src/microcosm/build/cd_benchmark/`: `__init__.py`, `canonical.py`, `origin.py`, `protocol.py`, `reasons.py`
- `packages/microcosm-build/src/microcosm/build/us_runtime/`: `_asec_current_money_codec.py`, `_person_signal_summary.py`, `acs_housing_universe.py`, `acs_housing_universe_source.py`, `acs_native_coverage_binding.py`, `acs_person_coverage_authentication.py`, `acs_person_coverage_columns.py`, `acs_population_catalogue.py`, `asec_2024_native_population.py`, `asec_coverage_authentication.py`, `asec_current_money.py`, `asec_current_money_graph_resources.py`, `asec_current_money_resources.py`, `asec_current_money_selection.py`, `asec_current_money_source.py`, `asec_current_money_units.py`, `asec_engine_evaluation.py`, `asec_household_coverage_fields.py`, `asec_household_observations.py`, `asec_housing_status.py`, `asec_housing_status_source.py`, `asec_housing_universe.py`, `asec_housing_universe_source.py`, `asec_income_observations.py`, `asec_original_household_weights.py`, `asec_person_coverage_source.py`, `asec_person_income_source.py`, `asec_population_catalogue.py`, `asec_prepared_source.py`, `asec_student_controls.py`, `cd_reference.py`, `cd_reference_sources.py`, `cps_carried_current.py`, `demographic_calibration_graph.py`, `full_puf_enrichment.py`, `graph_asec_income.py`, `graph_asec_prepared.py`, `graph_combined_clone.py`, `graph_composed_asec_binding.py`, `graph_composed_asec_measures.py`, `graph_composed_contracts.py`, `graph_composed_population.py`, `graph_context.py`, `graph_current_survey_puf_transfer.py`, `graph_geography.py`, `graph_housing_universe.py`, `graph_implementation.py`, `graph_implementation_inventory.json`, `graph_national_age_counts.py`, `graph_native_household_origin.py`, `graph_native_origin_implementation.py`, `graph_puf_detail_transfer.py`, `graph_puf_diagnostic_consumer.py`, `graph_sources.py`, `graph_survey_age_artifact.py`, `graph_survey_budget.py`, `graph_survey_calibration.py`, `graph_survey_population.py`, `national_age_activation.py`, `native_household_origin.py`, `operator_column_contracts.py`, `puf_detail_transfer.py`, `puf_diagnostic_consumer.py`, `puf_growth.py`, `puf_growth_graph.py`, `puf_monetary_agi_projection.py`, `puf_monetary_source.py`, `puf_price_baseline.py`, `puf_raw_source.py`, `reported_coverage_source.py`, `source_csv_builtin.py`, `survey_age_activation.py`, `survey_age_calibration.py`, `survey_age_sources.py`, `survey_calibration_diagnostics.py`, `survey_catalogue_selection.py`, `survey_observed_age.py`, `survey_origin_budget.py`, `survey_population_domains.py`, `survey_population_preparation.py`, `survey_population_replay.py`
- `packages/microcosm-calibrate/src/microcosm/calibrate/`: `group_bounds.py`
- `packages/microcosm-fit/src/microcosm/fit/`: `_graph_legacy_apply.py`, `_graph_legacy_qrf.py`, `graph_legacy_apply.py`, `graph_legacy_apply_matrix.py`, `graph_legacy_qrf.py`, `graph_legacy_train.py`, `model_input.py`, `qrf_target.py`
- `packages/microcosm-frame/src/microcosm/frame/adapters/`: `_policyengine_us_source_index.py`, `policyengine_us.py`
- `packages/microcosm-graph/src/microcosm/graph/`: `randomness.py`
- tests: `test_us_acs_housing_source.py`, `test_us_acs_person_coverage_authentication.py`, `test_us_acs_person_coverage_columns.py`, `test_us_acs_population_catalogue.py`, `test_us_asec_2024_native_population.py`, `test_us_asec_coverage_authentication.py`, `test_us_asec_current_money_source.py`, `test_us_asec_person_income_source.py`, `test_us_current_survey_puf_host.py`, `test_us_current_survey_puf_transfer.py`, `test_us_full_puf_enrichment.py`, `test_us_graph_survey_population.py`, `test_us_national_age_counts.py`, `test_us_puf_detail_transfer.py`, `test_us_puf_price_baseline.py`, `test_us_survey_age_activation.py`, `test_us_survey_age_artifact.py`, `test_us_survey_age_calibration_run.py`, `test_us_survey_age_development.py`, `test_us_survey_age_sources.py`, `test_us_survey_calibration.py`, `test_us_survey_observed_age.py`, `test_us_survey_origin_budget.py`, `test_us_survey_population_preparation.py`, `test_us_survey_population_replay.py`, `test_policyengine_us_ownership_index.py`
- source imports from: 02 GRAPH-RESTORE, 03 ACCEPTED-SHARED-RESTORE, 05 SOLVE-MERGE-PROPOSAL, 06 J-GRAPH-COMPATIBILITY, G RAW-BYTES-CODEC (graph piece), G STORE-FRAME-METADATA (graph piece), P3 SURVEY-POPULATION-CATALOGUE-AND-BUDGET, P4 CALIBRATION-SOLVER-AND-DIAGNOSTICS
- tests import from: 02 GRAPH-RESTORE, 03 ACCEPTED-SHARED-RESTORE, 04 PUF-SUPPORT-MERGE, 06 J-GRAPH-COMPATIBILITY, G RAW-BYTES-CODEC (graph piece), G STORE-FRAME-METADATA (graph piece)

#### 02 GRAPH-RESTORE  (1 non-test files, 0 test files)

- `packages/microcosm-graph/src/microcosm/graph/`: `kernel.py`

#### 03 ACCEPTED-SHARED-RESTORE  (20 non-test files, 2 test files)

- `packages/microcosm-build/src/microcosm/build/`: `frame_checkpoint.py`
- `packages/microcosm-build/src/microcosm/build/us_runtime/`: `acs_inputs.py`, `acs_pums.py`, `acs_transfer.py`, `asec_checkpoint.py`, `congressional_district_vintage.py`, `cps_carried.py`, `eligibility_inputs.py`, `hours_worked.py`, `housing_inputs.py`, `multispine_pool.py`, `operator_boundary.py`, `prior_year_income.py`, `puf_capital_gains_tail.py`, `qbi_inputs.py`, `relationship_inputs.py`, `spine_assembly.py`, `stacked_spine.py`
- `packages/microcosm-calibrate/src/microcosm/calibrate/`: `__init__.py`
- `packages/microcosm-fit/src/microcosm/fit/`: `qrf.py`
- tests: `test_us_acs_pums.py`, `test_us_asec_checkpoint.py`
- source imports from: 01 SAFE-ADDITIVE, 04 PUF-SUPPORT-MERGE, 05 SOLVE-MERGE-PROPOSAL, 06 J-GRAPH-COMPATIBILITY, P4 CALIBRATION-SOLVER-AND-DIAGNOSTICS
- tests import from: 01 SAFE-ADDITIVE, 06 J-GRAPH-COMPATIBILITY, P3 SURVEY-POPULATION-CATALOGUE-AND-BUDGET

#### 04 PUF-SUPPORT-MERGE  (1 non-test files, 0 test files)

- `packages/microcosm-build/src/microcosm/build/us_runtime/`: `puf_support.py`
- source imports from: 01 SAFE-ADDITIVE, 03 ACCEPTED-SHARED-RESTORE

#### 05 SOLVE-MERGE-PROPOSAL  (1 non-test files, 0 test files)

- `packages/microcosm-calibrate/src/microcosm/calibrate/`: `solve.py`
- source imports from: 01 SAFE-ADDITIVE

#### 06 J-GRAPH-COMPATIBILITY  (4 non-test files, 6 test files)

- `packages/microcosm-build/src/microcosm/build/us_runtime/`: `__init__.py`, `puma_ladder.py`, `puma_ladder_sources.py`
- `tools/`: `build_us_puma_ladder_artifact.py`
- tests: `test_us_acs_multispine.py`, `test_us_acs_multispine_legacy_builder.py`, `test_us_base_pool.py`, `test_us_multispine_pool_tool.py`, `test_us_puma_ladder.py`, `test_us_puma_ladder_sources.py`
- tests import from: 03 ACCEPTED-SHARED-RESTORE, 04 PUF-SUPPORT-MERGE

#### 07 F-CATALOGUE-OPTIMIZATION  (1 non-test files, 3 test files)

- `changelog.d/`: `us-survey-catalogue-digest.changed.md`
- tests: `test_us_survey_catalogue_digest.py`, `test_us_survey_catalogue_digest_authority.py`, `test_us_survey_catalogue_digest_fast_path.py`
- tests import from: 01 SAFE-ADDITIVE, 06 J-GRAPH-COMPATIBILITY

#### 10 SOURCE-CLOSURE  (9 non-test files, 0 test files)

- `packages/microcosm-build/src/microcosm/build/us_runtime/`: `acs_2024_housing_universe.json`, `asec_current_money_consumers_v1.json`, `asec_current_money_domains_v1.json`, `asec_current_money_price_basis_v1.json`, `graph_acs_housing_universe.py`, `native_origin_graph_inventory.json`, `puf_2015_monetary_agi_source_projection.json`, `puf_2015_monetary_source_projection.json`, `puf_2015_raw_source_definition.json`
- source imports from: G RAW-BYTES-CODEC (graph piece)

#### 11 PLACEMENT-ADDITIONS  (1 non-test files, 1 test files)

- `packages/microcosm-build/src/microcosm/build/us_runtime/`: `graph_full_puf_enrichment.py`
- tests: `test_us_graph_full_puf_enrichment.py`
- source imports from: 01 SAFE-ADDITIVE, 02 GRAPH-RESTORE, 03 ACCEPTED-SHARED-RESTORE, G RAW-BYTES-CODEC (graph piece), G STORE-FRAME-METADATA (graph piece)
- tests import from: 01 SAFE-ADDITIVE, 04 PUF-SUPPORT-MERGE, 06 J-GRAPH-COMPATIBILITY, G RAW-BYTES-CODEC (graph piece)

#### 12 ORDINARY-CLOSURE  (0 non-test files, 8 test files)

- tests: `legacy_prepatch.json`, `test_group_bounds.py`, `test_grouped_fixed_support.py`, `legacy-slice-apply-v1.json`, `test_graph_legacy_matrix.py`, `test_graph_legacy_qrf.py`, `test_qrf_stateless.py`, `test_qrf_target.py`
- tests import from: 01 SAFE-ADDITIVE, 03 ACCEPTED-SHARED-RESTORE, 05 SOLVE-MERGE-PROPOSAL, G RAW-BYTES-CODEC (graph piece)

#### 13 INTEGRATION-REGRESSIONS  (2 non-test files, 3 test files)

- `changelog.d/`: `us-launch-integration.fixed.md`
- `docs/`: `us-launch-review.md`
- tests: `test_us_geography_integration.py`, `test_us_runtime_facade_union.py`, `test_solve_us_launch_integration.py`
- tests import from: 01 SAFE-ADDITIVE, 02 GRAPH-RESTORE, 03 ACCEPTED-SHARED-RESTORE, 05 SOLVE-MERGE-PROPOSAL, 06 J-GRAPH-COMPATIBILITY, G RAW-BYTES-CODEC (graph piece)

#### Already on main's way (amendment 20 / this lane's re-pins; not #893 content)  (10 non-test files, 9 test files)

- `./`: `PROGRESS-amendment-20-keyed-draws.md`
- `changelog.d/`: `amend-keyed-seed-and-uniform-draws.added.md`
- `docs/`: `graph-acceptance.md`, `graph-interface.lock`
- `experiments/`: `amendment-20-keyed-draws-receipts.md`, `us-native-atomic-financial-required-replay-20260910.json`, `us-puf-donor-boundaries-20260910.json`, `us-puf55-public-recipient-controls-20260910.json`, `us-puf55-two-route-numerical-20260910.json`
- `tools/`: `graph_parity_repin.py`
- tests: `test_us_puf55_model_boundaries.py`, `test_kernels.py`, `pins.json`, `pins.json`, `pins.json`, `test_graph_kernel_contract.py`, `test_graph_parity_pins.py`, `test_graph_parity_repin.py`, `test_graph_randomness.py`
- source imports from: G RAW-BYTES-CODEC (graph piece)
- tests import from: 01 SAFE-ADDITIVE, 03 ACCEPTED-SHARED-RESTORE, 06 J-GRAPH-COMPATIBILITY, G RAW-BYTES-CODEC (graph piece)

#### G RAW-BYTES-CODEC (graph piece)  (2 non-test files, 1 test files)

- `packages/microcosm-graph/src/microcosm/graph/`: `__init__.py`, `codecs.py`
- tests: `test_graph_codecs.py`
- tests import from: G STORE-FRAME-METADATA (graph piece)

#### G STORE-FRAME-METADATA (graph piece)  (2 non-test files, 1 test files)

- `changelog.d/`: `20260906-frame-metadata-storage.fixed.md`
- `packages/microcosm-graph/src/microcosm/graph/`: `store.py`
- tests: `test_frame_metadata_store.py`

#### P1 POST-CLONE-ATOMIC-GEOGRAPHY  (38 non-test files, 16 test files)

- `changelog.d/`: `893-population-only-block-source.added.md`, `893-source-demographic-conditioning.added.md`, `893-survey-atomic-prefix.added.md`, `shared-atomic-geography.added.md`, `us-atomic-financial-composition.added.md`, `us-national-atomic-support.added.md`
- `experiments/`: `us-atomic-age-v2-1-controls-20260909.json`, `us-atomic-block-adapter-20260909.json`, `us-atomic-block-api-sources-35-controls-20260909.json`, `us-atomic-block-sources-36-controls-20260909.json`, `us-atomic-budget-semantic-8-controls-20260909.json`, `us-atomic-clone-graph-20260909.json`, `us-atomic-financial-4-controls-20260909.json`, `us-atomic-financial-composition-acceptance-20260909.json`, `us-atomic-financial-corrected-2-controls-20260909.json`, `us-atomic-national-acquisition-20260909.json`, `us-atomic-native-de-1-control-20260909.json`, `us-atomic-native-de-corrected-1-control-20260909.json`, `us-atomic-native-national-1-control-20260909.json`, `us-atomic-survey-population-controls-20260909.json`, `us-budget-predictor-compatibility-1-controls-20260909.json`, `us-financial-demographics-3-controls-20260909.json`, `us-postclone-geography-70-controls-20260910.json`, `us-puf55-native-donor-20260909.json`, `us-survey-geography-graph-controls-20260909.json`, `us-survey-geography-source-controls-20260909.json`
- `packages/microcosm-build/src/microcosm/build/`: `atomic_geography.py`, `graph_atomic_geography.py`
- `packages/microcosm-build/src/microcosm/build/us_runtime/`: `atomic_block_api_sources.py`, `atomic_block_sources.py`, `atomic_block_support.py`, `current_survey_geography.py`, `graph_atomic_survey_clone.py`, `graph_atomic_survey_financial.py`, `graph_atomic_survey_population.py`, `graph_current_survey_geography.py`, `survey_atomic_geography.py`
- `tools/`: `ci_test_groups.py`
- tests: `test_atomic_geography.py`, `test_us_atomic_block_api_sources.py`, `test_us_atomic_block_api_sources_native_de.py`, `test_us_atomic_block_api_sources_native_national.py`, `test_us_atomic_block_sources.py`, `test_us_atomic_block_support.py`, `test_us_atomic_survey_clone_graph.py`, `test_us_current_survey_geography.py`, `test_us_current_survey_predictor_demographics.py`, `test_us_graph_atomic_survey_financial.py`, `test_us_graph_atomic_survey_population.py`, `test_us_graph_current_survey_geography.py`, `test_us_postclone_geography_identity.py`, `test_us_survey_age_calibration_atomic_geography.py`, `test_us_survey_origin_budget_atomic_geography.py`, `test_us_survey_origin_budget_predictor_compatibility.py`
- source imports from: 01 SAFE-ADDITIVE, 02 GRAPH-RESTORE, G RAW-BYTES-CODEC (graph piece)
- tests import from: 01 SAFE-ADDITIVE, 03 ACCEPTED-SHARED-RESTORE, 04 PUF-SUPPORT-MERGE, 06 J-GRAPH-COMPATIBILITY, G RAW-BYTES-CODEC (graph piece), P2 PUF55-TWO-ROUTE-AND-CANONICAL-DONOR

#### P2 PUF55-TWO-ROUTE-AND-CANONICAL-DONOR  (51 non-test files, 21 test files)

- `./`: `pyproject.toml`
- `changelog.d/`: `893-age-count-fixture.internal.md`, `893-diagnostic-refusal-context.internal.md`, `893-historical-zero-provenance.internal.md`, `puf55-canonical-donor.added.md`, `puf55-survey-social-security.added.md`, `survey-social-security-source.added.md`, `us-puf55-canonical-and-output-seals.added.md`, `us-puf55-survey-ss-measurement.added.md`
- `docs/`: `current-survey-puf59-progress.md`, `geography-assignment.md`, `puf2015-canonical59-and-growth.md`, `puf55-survey-ss-measurement-decision.md`, `survey-social-security-source.md`, `us-uk-release-path.md`
- `docs/evidence/puf2015-target2024/`: `GROWTH-RECIPE.json`, `INDEX-VALUES.json`, `NATIONAL-GROWTH-EXTRACT.json`, `PUBLIC-WORKBOOK-SOURCES.json`
- `experiments/`: `us-acs-compilation-cache-adoption-20260910.json`, `us-budget-detached-view-correction-20260910.json`, `us-financial-fixture-profile-20260910.json`, `us-financial-successor-positive-1-limit-stop-20260909.json`, `us-financial-successor-positive-20260910.json`, `us-financial-successor-remaining-controls-20260910.json`, `us-puf55-canonical-create-47-controls-20260910.json`, `us-puf55-numerical-output-seal-46-controls-20260910.json`, `us-puf55-population-controls-20260909.json`, `us-puf55-public-recipient-positives-20260910.json`, `us-puf55-survey-ss-measurement-31-controls-20260909.json`, `us-puf55-two-route-values-controls-20260910.json`, `us-survey-age-development-20260909.json`
- `packages/microcosm-build/src/microcosm/build/us_runtime/`: `asec_demographic_source.py`, `current_asec_demographics.py`, `current_social_security_source.py`, `current_survey_predictors.py`, `graph_current_survey_predictors.py`, `graph_puf55_canonical_donor.py`, `graph_puf55_survey_recipients.py`, `puf55_canonical_donor.py`, `puf55_route_finalization.py`, `puf55_survey_recipients.py`, `puf55_survey_ss_measurement.py`, `puf59_canonical.py`, `puf59_canonical_artifact.py`, `puf_full_source.py`, `puf_full_source_graph.py`, `puf_qbi_model.py`, `puf_target2024_growth.py`, `survey_financial_successor.py`, `survey_social_security.py`
- tests: `test_frame_checkpoint.py`, `test_spec_seed_diagnostic_refusal_context.py`, `test_us_acs_source_compile_cache.py`, `test_us_current_asec_demographics.py`, `test_us_current_survey_predictors.py`, `test_us_full_puf_output_profiles.py`, `test_us_graph_puf55_canonical_donor.py`, `test_us_graph_puf55_survey_recipients.py`, `test_us_graph_puf55_survey_ss.py`, `test_us_plan.py`, `test_us_puf55_route_finalization.py`, `test_us_puf55_route_numerical_finalization.py`, `test_us_puf55_survey_recipients.py`, `test_us_puf55_survey_ss_measurement.py`, `test_us_puf55_survey_ss_profile.py`, `test_us_puf59_canonical.py`, `test_us_puf_target2024_growth.py`, `test_us_runtime_import_order.py`, `test_us_survey_financial_successor.py`, `test_us_survey_origin_budget_detached_views.py`, `test_us_survey_social_security.py`
- source imports from: 01 SAFE-ADDITIVE, 03 ACCEPTED-SHARED-RESTORE, G RAW-BYTES-CODEC (graph piece)
- tests import from: 01 SAFE-ADDITIVE, 02 GRAPH-RESTORE, 03 ACCEPTED-SHARED-RESTORE, 04 PUF-SUPPORT-MERGE, 06 J-GRAPH-COMPATIBILITY, 11 PLACEMENT-ADDITIONS, G RAW-BYTES-CODEC (graph piece), P1 POST-CLONE-ATOMIC-GEOGRAPHY, P3 SURVEY-POPULATION-CATALOGUE-AND-BUDGET

#### P3 SURVEY-POPULATION-CATALOGUE-AND-BUDGET  (16 non-test files, 10 test files)

- `changelog.d/`: `common-frame-export-contract.added.md`, `us-survey-copy-and-identity.fixed.md`
- `docs/`: `us-input-coverage-diagnostic.md`
- `experiments/`: `us-acs-code-memo-comparison-20260910.json`, `us-acs-code-memo-not-adopted-20260910.patch`, `us-acs-loaded-code-controls-20260910.json`, `us-common-frame-export36-20260910.json`, `us-input-coverage25-20260910.json`, `us-postclone-budget-replay-fence4-20260910.json`, `us-survey-catalogue-memo20-20260910.json`, `us-survey-numeric-diagnostics27-20260910.json`
- `packages/microcosm-build/src/microcosm/build/us_runtime/`: `asec_raw_stage_v4.py`, `common_frame_export_contract.py`, `input_coverage_profile.py`, `population_input_coverage.py`
- `packages/microcosm-frame/src/microcosm/frame/`: `bundle.py`
- tests: `test_us_acs_transfer.py`, `test_us_asec_catalogue_records_memo.py`, `test_us_asec_demographic_source.py`, `test_us_asec_household_coverage_fields.py`, `test_us_common_frame_export_contract.py`, `test_us_population_input_coverage.py`, `test_us_spine_blindness.py`, `test_us_survey_catalogue_immutable_memo.py`, `test_us_survey_frame_identity_encoding.py`, `test_us_survey_origin_budget_replay_producer.py`
- source imports from: 01 SAFE-ADDITIVE, 03 ACCEPTED-SHARED-RESTORE, 06 J-GRAPH-COMPATIBILITY, G RAW-BYTES-CODEC (graph piece), G STORE-FRAME-METADATA (graph piece)
- tests import from: 01 SAFE-ADDITIVE, 03 ACCEPTED-SHARED-RESTORE, 04 PUF-SUPPORT-MERGE, 06 J-GRAPH-COMPATIBILITY, G RAW-BYTES-CODEC (graph piece), P2 PUF55-TWO-ROUTE-AND-CANONICAL-DONOR

#### P4 CALIBRATION-SOLVER-AND-DIAGNOSTICS  (6 non-test files, 1 test files)

- `changelog.d/`: `us-launch-store-grouped-validation.fixed.md`, `us-survey-diagnostic-options.fixed.md`
- `experiments/`: `us-calibration-consolidation-149-20260910.json`, `us-puf55-canonical-create-failed-20260910.json`, `us-puf55-canonical-cycle39-failed-20260910.json`
- `packages/microcosm-calibrate/src/microcosm/calibrate/`: `variable_labels.py`
- tests: `test_variable_labels.py`
- tests import from: 03 ACCEPTED-SHARED-RESTORE

#### P5 CI-SEED-IDENTITY-AND-SPINE-BLINDNESS  (10 non-test files, 10 test files)

- `./`: `CLAUDE.md`
- `.github/workflows/`: `test.yml`
- `changelog.d/`: `893-diagnostic-current-lock.fixed.md`, `893-spec-identity-diagnostics.internal.md`, `893-worker-resource-trace.fixed.md`
- `docs/evidence/spec-engine/`: `us-f0-coverage.json`
- `experiments/`: `spec-seed-current-lock-20260910.json`, `us-acs-producer-profile-20260910.json`
- `packages/microcosm-build/src/microcosm/build/spec_engine/`: `inventory_coverage.py`
- `tools/`: `spec_seed_identity_diagnostics.py`
- tests: `test_spec_engine_country_bundles.py`, `test_spec_engine_loader.py`, `test_spec_seed_identity_cpu_bootstrap.py`, `test_spec_seed_identity_engine_parameter_files.py`, `test_spec_seed_identity_outer_failure.py`, `test_spec_seed_identity_owned_temp.py`, `test_spec_seed_identity_per_code_context.py`, `test_spec_seed_identity_system_metadata.py`, `test_us_puf_support.py`, `test_us_stacked_spine.py`
- tests import from: 01 SAFE-ADDITIVE, 03 ACCEPTED-SHARED-RESTORE, 04 PUF-SUPPORT-MERGE, 06 J-GRAPH-COMPATIBILITY

#### P6 UK-ADAPTER-DOCS-AND-RECORDS  (10 non-test files, 2 test files)

- `./`: `.gitattributes`
- `changelog.d/`: `893-uk-atomic-area-and-lazy-exports.added.md`
- `docs/`: `validation-cost-and-borrow-boundaries.md`
- `experiments/`: `uk-atomic-area-lazy52-20260910.json`, `us-financial-default-2-controls-20260909.json`, `us-native-atomic-financial-cold-20260910.json`, `us-native-postclone-financial-incomplete-20260910.json`
- `packages/microcosm-build/src/microcosm/build/uk_runtime/`: `__init__.py`, `atomic_area_support.py`, `atomic_household_identity.py`
- tests: `test_uk_atomic_area_support.py`, `test_uk_runtime_lazy_exports.py`
- source imports from: P1 POST-CLONE-ATOMIC-GEOGRAPHY
- tests import from: P1 POST-CLONE-ATOMIC-GEOGRAPHY

**Layer-01 modules with no cross-layer import at all (53; `randomness.py` among them is amendment 20, not #893 content):** `__init__.py`, `canonical.py`, `origin.py`, `protocol.py`, `reasons.py`, `survey_allocation.py`, `survey_domain_sample.py`, `table_identity.py`, `_asec_current_money_codec.py`, `_person_signal_summary.py`, `acs_housing_universe.py`, `acs_native_coverage_binding.py`, `acs_person_coverage_authentication.py`, `acs_person_coverage_columns.py`, `acs_population_catalogue.py`, `asec_coverage_authentication.py`, `asec_current_money.py`, `asec_current_money_graph_resources.py`, `asec_current_money_resources.py`, `asec_current_money_units.py`, `asec_household_coverage_fields.py`, `asec_housing_status.py`, `asec_housing_status_source.py`, `asec_housing_universe.py`, `asec_housing_universe_source.py`, `asec_income_observations.py`, `asec_original_household_weights.py`, `asec_person_coverage_source.py`, `asec_population_catalogue.py`, `asec_prepared_source.py`, `asec_student_controls.py`, `cd_reference.py`, `cd_reference_sources.py`, `cps_carried_current.py`, `graph_implementation.py`, `graph_implementation_inventory.json`, `graph_native_origin_implementation.py`, `native_household_origin.py`, `operator_column_contracts.py`, `puf_detail_transfer.py`, `puf_diagnostic_consumer.py`, `puf_growth.py`, `puf_price_baseline.py`, `reported_coverage_source.py`, `source_csv_builtin.py`, `survey_age_activation.py`, `survey_age_sources.py`, `survey_catalogue_selection.py`, `survey_observed_age.py`, `survey_population_domains.py`, `group_bounds.py`, `_policyengine_us_source_index.py`, `randomness.py`


**Layer-01 modules with a cross-layer import (48; 27 of them only on the graph pieces G1/G2, the other 21 soft on modified modules of 03/05/06/P3/P4 — see `pr1-refined.json`):** `acs_housing_universe_source.py`, `asec_2024_native_population.py`, `asec_current_money_selection.py`, `asec_current_money_source.py`, `asec_engine_evaluation.py`, `asec_household_observations.py`, `asec_person_income_source.py`, `demographic_calibration_graph.py`, `full_puf_enrichment.py`, `graph_asec_income.py`, `graph_asec_prepared.py`, `graph_combined_clone.py`, `graph_composed_asec_binding.py`, `graph_composed_asec_measures.py`, `graph_composed_contracts.py`, `graph_composed_population.py`, `graph_context.py`, `graph_current_survey_puf_transfer.py`, `graph_geography.py`, `graph_housing_universe.py`, `graph_national_age_counts.py`, `graph_native_household_origin.py`, `graph_puf_detail_transfer.py`, `graph_puf_diagnostic_consumer.py`, `graph_sources.py`, `graph_survey_age_artifact.py`, `graph_survey_budget.py`, `graph_survey_calibration.py`, `graph_survey_population.py`, `national_age_activation.py`, `puf_growth_graph.py`, `puf_monetary_agi_projection.py`, `puf_monetary_source.py`, `puf_raw_source.py`, `survey_age_calibration.py`, `survey_calibration_diagnostics.py`, `survey_origin_budget.py`, `survey_population_preparation.py`, `survey_population_replay.py`, `_graph_legacy_apply.py`, `_graph_legacy_qrf.py`, `graph_legacy_apply.py`, `graph_legacy_apply_matrix.py`, `graph_legacy_qrf.py`, `graph_legacy_train.py`, `model_input.py`, `qrf_target.py`, `policyengine_us.py`

**Layer-01 tests (all ride with PR-1; 26):** `test_us_acs_housing_source.py`, `test_us_acs_person_coverage_authentication.py`, `test_us_acs_person_coverage_columns.py`, `test_us_acs_population_catalogue.py`, `test_us_asec_2024_native_population.py`, `test_us_asec_coverage_authentication.py`, `test_us_asec_current_money_source.py`, `test_us_asec_person_income_source.py`, `test_us_current_survey_puf_host.py`, `test_us_current_survey_puf_transfer.py`, `test_us_full_puf_enrichment.py`, `test_us_graph_survey_population.py`, `test_us_national_age_counts.py`, `test_us_puf_detail_transfer.py`, `test_us_puf_price_baseline.py`, `test_us_survey_age_activation.py`, `test_us_survey_age_artifact.py`, `test_us_survey_age_calibration_run.py`, `test_us_survey_age_development.py`, `test_us_survey_age_sources.py`, `test_us_survey_calibration.py`, `test_us_survey_observed_age.py`, `test_us_survey_origin_budget.py`, `test_us_survey_population_preparation.py`, `test_us_survey_population_replay.py`, `test_policyengine_us_ownership_index.py`

## 7. Draft charter amendment entries (one per kept graph piece)

Written in the style of `docs/graph-acceptance.md` "Interface freeze". None of
the four touches `decl.py` or `kernel.py`, so none re-records the lock; they
are offered as numbered entries because each changes what a receipt, a store
object or an executor outcome means, which the charter records.

**7.1 Raw-byte sources.** A source codec is registered in exactly one of two
modes and a name belongs to at most one: *Frame mode* (`SourceCodec`,
`SourceCodecRegistry.register` / `load`) decodes a source into a population
`Frame`, and *raw-byte mode* (`SourceBytesCodec`, `register_bytes` /
`load_bytes`, `load_source_bytes`) decodes it into immutable `bytes` that the
consuming kernel alone interprets. Neither mode can be loaded through the
other: the mismatch is a `TypeError` naming the mode the codec has, so no
caller receives bytes where it declared a Frame. The shipped `raw-bytes-v1`
codec reads one regular file, bounded at `RAW_BYTES_MAX_BYTES` (64 MiB) and
opened non-blocking so a directory, FIFO, device or socket is refused from
the descriptor's own mode. Identity, the pre-run content key and the post-run
mutation check stay in the executor (E1, E3 unchanged; C3's "declared
predecessors only" is untouched because a source is not a node). Why: a
lookup NPZ or a crosswalk CSV that an import kernel turns into a typed
`ArtifactOutput` (amendment 19) is not a population, and the only way to read
it before this was a Frame codec registered as a pretence. Raised by the US
launch integration branch (thirteen consumers); no key moves, because
`SourceRef.codec` was already normative and `raw-bytes-v1` is a new name.

**7.2 Frames keep their metadata across the store.** A frame object is
written as `microcosm-graph-frame-v2`: `Frame.metadata` is encoded
losslessly (mappings, tuples, frozensets, `float64` bit patterns, scalars)
into the frame manifest, its SHA-256 is recorded in the object header, and
`_read_frame` refuses a mismatch as `StoreCorruptError`. A v1 object is
`StoreUnavailableError` on load rather than a Frame with its metadata silently
dropped, and writing a frame under a key that already holds one with different
metadata is `StoreCorruptError` (E1: a key identifies exactly one content).
The JSON decode boundary refuses `NaN`/`Infinity` and overflowing numerals,
because every store JSON is written with `allow_nan=False`, so a corrupt
manifest is `StoreCorruptError` at the boundary rather than a `TypeError`
from the canonical re-encoder (E2). Why: the US survey stages carry source
provenance in `Frame.metadata` and replay verifiers compare it
(`same_replayed_frame`); until now a population that went through the store
came back without it. No node key moves: the frame key is the node's, and
metadata was never part of a key. Raised by the US launch integration
branch (`8d9f62c40`); acceptance items E1 and E2.

**7.3 Execution states: a gate exception leaves its consumers unreached.**
Amendment 19 refused a gate kernel that declares a typed artifact output
because it had no regime for an output a node was unable to produce. This
entry supplies the regime and lifts the refusal. The executor owns an
execution state (`"microcosm.graph.execution.v1"`) that a kernel may not
author: a gate that declares typed outputs and raises still becomes a `fail`
verdict and the run continues (amendment 7, now literally true for every
node shape), and its receipt records `gate_exception` with the outputs it
could not produce; a node whose predecessor is `unreached`, or whose declared
byte input its producer recorded as unavailable, is itself recorded
`unreached` with its blockers named by node key — it runs no kernel and
invents no column, frame, weight or byte (F1: nothing is fabricated); a
release behind such an edge derives its tier from the same ancestry and stays
`evidence` (F2). Cache records that carry a state use schema 3; a cached
unreached record is a hit only while the same inputs are unavailable for the
same reason and is `StoreCorruptError` once they exist (E3); `require`
preflight derives blockers from the cached parents. A run manifest carrying
any state serializes at schema 4 and authenticates every blocker on load
(F5, E4). No node key moves: states live in receipts and records, never in a
key. Why: the US post-clone geography gate emits a typed validation artifact
that the financial stages consume, so a geography failure must leave those
stages unreached rather than abort the run or launder the gate into a compute
node. Raised by the US launch integration branch; supersedes the interim
ruling recorded in amendment 19.

**7.4 A private population observer.** `run_graph` accepts
`_population_observer`, a callable handed each node's admitted population
(design anchors included) after the result is applied and before the node is
persisted, on cold execution and on cache hits alike; it must not mutate the
population, an exception it raises refuses the run, and an unreached node
has no population to observe. It is an integration seam for verifiers (the
US survey stages assert design-weight provenance through it), not a kernel
capability: it enters no key, receipt or record (A1, B4 unchanged). Raised by
the US launch integration branch.

## 8. Decisions needed from Max

1. **Piece C supersedes an amendment-19 ruling.** The Amendment 19 lane
   recorded "the gate-artifact-output refusal is the one interim ruling this
   lane made on his behalf". This branch's US runtime needs the opposite (the
   post-clone geography gate declares its validation artifact and three
   stages consume it), so piece C re-applies the branch's execution-state
   regime and replaces main's test that pins the refusal with five tests of
   the new regime. Both the frozen files and every `test_acceptance_*` file
   are untouched, but the executor's behaviour on that one node shape now
   differs from what the charter's amendment 19 text says. Landing PR-G3
   requires adopting entry 7.3 (or an edited form) in the charter; until then
   this branch's executor and the charter disagree on that sentence.
2. **Lazy population retention** (`attachments.py`, layer 08) is dropped for
   lack of a consumer. If the native pilot's memory ceiling is a reason to
   keep it, it is a self-contained PR on top of PR-G2.
3. **`_stream_file` and the `_write_node` refactor** are dropped for lack of a
   consumer. Both are memory-only; say if either should come back as its own
   small PR.
4. **The report's home.** `out.md` at the repo root is a tracked file holding
   the F1 Sol gate round-1 report (2026-09-04). This report is written there
   for the dispatcher but is **not committed** there; the committed copy is
   `experiments/893-reconciliation-amendments-19-20-20260912.md`.
5. **Three US stages name a resource that does not exist anywhere.**
   `asec_prepared_v3`, `composed_asec_binding_v1` and `composed_population_v1`
   list `us_runtime/asec_current_money_engine_defaults_v1.json` in their
   resources; the file is on neither this branch (at `069d5ed9a` or now) nor
   `origin/main` (it appears only in two commits of another branch,
   `b0e7f54b2` / `b6cfc84b0`). `implementation_manifest` for those stages has
   therefore been failing since before this lane. Not invented here; the
   owner of layer 10 SOURCE-CLOSURE has to admit the file or drop the entry,
   and the inventory's "scope review" rule applies to `07566eb17` as well.
6. **Two build tests fail locally for reasons outside this lane** if they are
   run: `test_release_target_parity.py`'s two `_feed_or_skip` tests need a 131
   MB feed outside the repository (already recorded by the Amendment 19 lane).
   They were not part of this lane's runs.

## 9. Where things are

- Journal: `PROGRESS.md` (top section, cumulative file).
- This report: `out.md` (uncommitted, the `-o` path) and
  `experiments/893-reconciliation-amendments-19-20-20260912.md` (committed).
- Scratch evidence (session-local, not committed): the AST consumer scan,
  the layer scripts and JSON, and every gate log named in section 5.
