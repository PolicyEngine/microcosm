# PR #847 gate round 4 report

Starting SHA: `cdbf71888f4b5896e124519d643fa2347c483123`  
Starting merge base with `origin/main`: `6022e4f5c8af31d62fddd1c7dbd1ab9eea81b0f6`  
Local lane branch: `fix-847-r5`

All commands below were run with `uv run`. In this managed sandbox, final
verification used
`UV_CACHE_DIR=/private/tmp/microcosm-fix-847-r5-uv-cache` because the first
plain invocation stopped before Ruff with an `Operation not permitted` error
while initializing `/Users/maxghenis/.cache/uv`.

## 1. F5 absent-decision refusal

**Reproduction.** With a genuinely signed certified graph manifest, delete the
top-level `decisions` records without changing its authenticated body or key,
then run:

```text
uv run pytest packages/microcosm-graph/tests/test_graph_executor.py::test_certified_loader_revalidates_authenticated_required_decisions -q -p no:cacheprovider
```

Before the fix, `RunManifest.load_certified` returned the manifest; the
regression failed with `DID NOT RAISE NodeRejectedError`. Five later mutations
that blanked required legacy/current signed-record fields were accepted in the
same way.

**Fix.** Release receipts now authenticate the normative
`requires_decisions` names. Certified load validates that provenance, validates
both supported signed-record shapes and every required non-empty field,
rechecks that all required names are carried, and rederives the release
outcome. Decisions remain outside node keys.

**Tests.** `test_certified_loader_revalidates_authenticated_required_decisions`,
`test_certified_loader_requires_authenticated_decision_requirements`, and
`test_certified_loader_revalidates_signed_decision_fields`.

**Commits.** `b9db5eaece357518abc962310ff55d88bc11981b`,
`ac859a3e897a8fe61d52b55ba97d01d780621477`.

## 2. Canonical body comparison

**Reproduction.** Change an authenticated schema-v2 node seed from integer
`1` to float `1.0`, recompute the serialized content key, and run:

```text
uv run pytest packages/microcosm-graph/tests/test_graph_manifest.py::test_schema_v2_body_rejects_canonically_distinct_numeric_value -q -p no:cacheprovider
```

Before the fix, Python equality accepted the body and the regression failed
with `DID NOT RAISE ValueError`; the returned manifest then derived a different
key from the accepted serialized key.

**Fix.** Schema-v2 validation compares each stored node receipt and tier by
canonical JSON bytes, then independently requires the serialized key to equal
the reconstructed `manifest.key`.

**Test.** `test_schema_v2_body_rejects_canonically_distinct_numeric_value`.

**Commit.** `5764115a2ba252faee5d09a91657a6f38a72cd8b`.

## 3. Ratchet fail-open

**Reproduction.** Run:

```text
uv run pytest packages/microcosm-graph/tests/test_graph_acceptance_burndown.py::test_suppression_forms_the_marker_scan_cannot_model_are_refused -q -p no:cacheprovider
```

Against the original scanner, annotated `pytestmark`, an assigned marker
alias, and `unittest.skip`, `skipIf`, and `skipUnless` all yielded
`markers=()` and `suppressions=()`; each new case failed. Adversarial audit also
reproduced empty suppression results for indirect marker installation through
`sys.modules[__name__].__dict__.update/setdefault` and for called, imported, or
uncalled `unittest.SkipTest` raises.

**Fix.** The scanner now fails closed on unresolved collected-test decorators,
all module-level `pytestmark` binding forms, pytest aliases/rebinding and
indirect marked parameters, dynamic current-module mutation, the three
`unittest.skip*` decorators, and public/direct-import `unittest.SkipTest`
runtime forms. The suite's fixed literal `_toy` module loader remains allowed.

**Tests.** `test_suppression_forms_the_marker_scan_cannot_model_are_refused`,
`test_aliases_and_indirect_parameters_are_refused`, and
`test_the_literal_toy_module_loader_is_not_a_suppression`.

**Commits.** `594f0428dbc2030c4d3a9d80fad9870eb4ad7448`,
`6c17427b0800448bce3f7051b8292b37173909aa`,
`a8878284b7abe57943582728b1d6a19a5b1e75cc`,
`b2655076662bb4fb3150a64244da363f217d327e`,
`accdfcd17685babb10b3235d1dad80a684908fcd`, and
`6ac970dc7c13e0d70e9f6cdc980e4505d25d5e65`.

## 4. Platform-bitwise input provenance

**Reproduction.** Run:

```text
uv run pytest packages/microcosm-graph/tests/test_graph_executor.py::test_input_numeric_scopes_preserve_platform_across_masked_writer_order packages/microcosm-graph/tests/test_graph_executor.py::test_tolerance_gate_refuses_cross_platform_numeric_scope -q -p no:cacheprovider
```

Before the fix, the integrated gate received an empty `context.numerics` map:
a sole platform-bitwise writer projected to `None`, and a mixed bounded and
platform-bitwise coordinate retained only the finite tolerance. The
cross-platform helper returned `pass` instead of `evidence_absent`.

**Fix.** Executor aggregation now computes a `NumericScope` per input using
`bitwise < platform_bitwise < tolerance_bound`, takes the componentwise loosest
bounded tolerance, retains `platform_fingerprint()` whenever any writer is
platform-bitwise, and derives `tolerances` from those scopes. The tolerance
gate helper records numeric class/platform and refuses an explicitly different
comparison platform.

**Tests.** `test_input_numeric_scopes_preserve_platform_across_masked_writer_order`
(both writer orders) and `test_tolerance_gate_refuses_cross_platform_numeric_scope`.
The integrated test also covers the all-bitwise default and verifies
`tolerances[c] == numerics[c].tolerance` for every input.

**Commit.** `4edb4bb1b4970b900ddc65722264f20f4375f07b`.
The non-behavioral export-order cleanup is
`71db27484df1f9c47ffd22072adc82e4248f136f`.

## 5. QRF worker state in pickle bytes

**Reproduction.** Run:

```text
uv run pytest packages/microcosm-graph/tests/test_graph_executor.py::test_fit_qrf_model_artifact_is_canonical_across_runtime_worker_settings -q -p no:cacheprovider
```

Before the fix, fitting through the graph store with
`POPULACE_FIT_N_JOBS=1` and `=2` produced equal node and artifact identities,
but different pickle bytes (first difference at byte 798), so the artifact-byte
equality assertion failed.

**Fix.** `_Forest.__getstate__` serializes a shallow model copy with canonical
`n_jobs=1`, leaving the live fitted model untouched. Trusted unpickle restores
the current runtime `_fit_n_jobs()` setting. `n_jobs` was not added to the node
key.

**Test.** `test_fit_qrf_model_artifact_is_canonical_across_runtime_worker_settings`.

**Commit.** `01795648093a82052b4a3cbaf7ce24ea6005a14e`.

## 6. H1 off-platform identity assertion

**Reproduction.** Force the simulate parity pin down the off-platform branch,
then run:

```text
uv run pytest packages/microcosm-graph/tests/test_acceptance_h_parity.py::test_h1_kernel_parity -q -p no:cacheprovider
```

Before the fix, the case completed with zero byte comparisons after merely
incrementing the comparison counter; it compared neither node identity nor a
value.

**Fix.** Parity generation persists `node_key` for every pin. H1 now requires a
different local node key for an off-platform platform-bitwise kernel, and
requires equal key plus equal bytes on the pinned platform. Fixture source
identity now comes from stable `inputs.csv`, and affected pins were regenerated.

**Test.** Strengthened `test_h1_kernel_parity` (the explicitly permitted
acceptance-test edit), with generation coverage from
`test_generated_parity_graphs_bind_real_kernels_and_direct_bytes`.

**Commit.** `6eff748c2f19bcbfddac2fda402c414f7bc486b0`.

## 7. Dotted `expand_cells`

**Reproduction.** Run:

```text
uv run pytest packages/microcosm-graph/tests/test_graph_executor.py::test_dotted_expand_cell_is_rejected_before_cold_or_warm_execution -q -p no:cacheprovider
```

Before the fix, a dotted column completed the cold run and authored a receipt
coordinate the warm parser could not represent; the cold `pytest.raises`
failed with `DID NOT RAISE NodeRejected`. A dotted entity was likewise not
rejected at declaration preflight.

**Fix.** The shared EXPAND declaration parser requires dot-free entity and
column names. Executor coordinate discovery uses that parser, and an all-node
preflight rejects malformed declarations before source hashing, key creation,
cache I/O, or kernel execution.

**Test.** `test_dotted_expand_cell_is_rejected_before_cold_or_warm_execution`
for both dotted entity and dotted column, exercising `forbid` and `auto` paths.

**Commit.** `e43f1896e6a238685434b83c1b560c34570629b6`.

## 8. Cached EXPAND incumbent restoration

**Reproduction.** Run:

```text
uv run pytest packages/microcosm-graph/tests/test_graph_population.py::test_cached_expand_rejects_changed_incumbent_storage packages/microcosm-graph/tests/test_graph_population.py::test_cached_expand_requires_exact_declared_column_set -q -p no:cacheprovider
```

Before the fix, restoration accepted all three corruptions: an incumbent cell
changed to `99.0`, an undeclared extra column, and a missing declared column;
the corresponding `pytest.raises` checks failed.

**Fix.** Cached restoration now requires the exact base-plus-declared column
set for every entity, validates declared dtypes, and compares every incumbent
column's full prefix with storage-exact semantics before validating copied
additions.

**Tests.** `test_cached_expand_rejects_changed_incumbent_storage` and
`test_cached_expand_requires_exact_declared_column_set[extra/missing]`.

**Commit.** `4b690bc5c10c1e3db601e61d09f8cfa3d69cfdfc`.

## 9. Manifest tolerance overflow boundary

**Reproduction.** Run:

```text
uv run pytest packages/microcosm-graph/tests/test_graph_manifest.py::test_load_wraps_oversized_tolerance_integer_as_store_corrupt -q -p no:cacheprovider
```

Before the fix, a 400-digit valid JSON integer reached an eager `float()` and
escaped `RunManifest.load` as `OverflowError: int too large to convert to
float`.

**Fix.** The schema-v2 parser passes validated raw numeric values to
`Tolerance`; its existing normalization converts the overflow to `ValueError`,
which `RunManifest.load` wraps as `StoreCorruptError`.

**Test.** `test_load_wraps_oversized_tolerance_integer_as_store_corrupt`.

**Commit.** `49f8ed55bc272a4aa2141a349bba721084782f47`.

## 10. Explain fallback partition fields

**Reproduction.** Run:

```text
uv run pytest packages/microcosm-graph/tests/test_graph_explain.py::test_calibration_view_fallback_renders_partitioned_mass_record -q -p no:cacheprovider
```

Before the fix, removing portable receipt mass while leaving an attached
partitioned `MassRecord` rendered total mass but omitted the partition heading
and row; the HTML assertions failed.

**Fix.** The fallback now uses the canonical `mass_record_receipt` projection,
which retains totals, strata/policy, and every partition entity, column,
before, and after field.

**Test.** `test_calibration_view_fallback_renders_partitioned_mass_record`.

**Commit.** `f0089cd7f2cdb19c503ba09fd64c81bd19424517`.

## Final verification

Final commands used the writable cache prefix described above.

```text
uv run ruff check packages/microcosm-graph packages/microcosm-fit tools/graph_acceptance_burndown.py tools/graph_parity_fixtures.py
exit 0
All checks passed!

uv run ruff format --check packages/microcosm-graph packages/microcosm-fit tools/graph_acceptance_burndown.py tools/graph_parity_fixtures.py
exit 0
52 files already formatted

uv run python tools/graph_acceptance_burndown.py --verify
exit 0
green=41 red=0 missing=0; baseline=origin/main; verification=ok

uv run python tools/ci_test_groups.py --verify
exit 0
tracked_test_files=354; verification=ok

uv run pytest packages/microcosm-graph packages/microcosm-fit/tests -q -p no:cacheprovider
exit 1
432 collected: 429 passed, 2 skipped, 1 failed
failed: packages/microcosm-graph/tests/test_acceptance_b_ownership.py::test_b2_executor_enforces_ownership
```

The only failure asserts that `KernelContext.__dataclass_fields__` contains the
eight pre-amendment-17 fields and treats `numerics` as an unexpected extra.
That exact stale assertion and the locked `numerics` field are both already
present at starting SHA `cdbf7188`; none of this lane's commits changed the
test or `kernel.py`. A separate collect-only invocation exited 0 and reported
the per-file total of 432 used for the counts above. The test run also emitted
one joblib warning about physical-core detection and falling back to logical
cores.

## Deliberately not done

- Did not edit `decl.py`, `kernel.py`, or `docs/graph-interface.lock`.
- Did not edit any acceptance test except the explicitly authorized H-parity
  file. In particular, did not update acceptance B to add `numerics`, because
  the hard rule forbids that edit even though amendment 17 makes its existing
  assertion unsatisfiable.
- Did not put decisions or QRF `n_jobs` into node keys.
- Did not rename the harness-provided local `fix-847-r5` branch.
- Did not push, fetch, publish, promote, or use the network.
- Did not edit repository-root `PROGRESS.md` or `out.md`; this lane's committed
  journal and report live under `docs/graph-847-r5/`.
