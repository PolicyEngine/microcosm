# Combined survey and national age evidence

Public profile: `us_combined_asec_acs_national_age_v1`. This new profile preserves
the existing F001 and ASEC income profiles. It uses the unchanged candidate
writer, with no index update. Its population identity is the materialized
calibrated checkpoint file SHA-256, never a weight-vector digest.

## Stable public attachment contract

The population-scoped `graph_receipts` attachment keeps
`schema_version: microcosm.graph_receipts.v1` and adds the profile above. The
run-scoped `build_manifest` attachment keeps
`schema_version: microcosm.graph_build_summary.v1`, with
`graph_receipts_sha256` and verification `checks`. Both carry `country`,
`candidate_release_id`, `run_id`, and flat string-valued provenance fields.
Only the graph attachment carries top-level `population_sha256`. Both carry
complementary `genuine` and `invented_sources` booleans verified against the root
binding and recorded executions; candidate renaming cannot hide fixture origin.

The new profile uses these public fields:

- `period: "2024"`, `entity: "household"`, `period_basis: "target_year"`.
  `period_reason` explicitly says source observation dates remain unharmonized.
- `scope.households`, `scope.persons`, `scope.entity_rows: [{entity, rows}]`.
  `scope.source_arms: [{id, label, households, persons, source_vintages}]`, with
  ids `asec` and `acs` and vintages as recorded strings. Counts are measured,
  reconciled across receipts, and variable by candidate.
- `composition: {checkpoint_sha256, run_receipt_sha256, acceptance_sha256,
  computation_content_key}` retains the original importance-weight population.
  It contains no key named `population_sha256`.
- `quality: {status: "incomplete", basis: "age_only_development_calibration",
  score: null, missing: [...], note}`. Eligibility is always `excluded`, with
  the development-experiment reason; engineering completion is not launch
  validation.
- `age_calibration.target_year: 2024`, `consumes_standard_errors: false`,
  `tolerance: null`, and `tolerance_status: "unavailable"`.
- `age_calibration.cells` contains all 18 S0101 age bands. Each item contains
  `cell`, `measure`, `label`, `minimum_age`, `maximum_age` (null for 85+),
  `published_value`, `published_se`, `compiled_target`, `initial_estimate`,
  `final_estimate`, `initial_deviation`, `final_deviation`,
  `initial_relative_deviation`, `final_relative_deviation`,
  `supported_households`, `unweighted_people_in_band`,
  `maximum_band_count_in_a_household`, `support_status` (`supported` or
  `zero_support`), and `within_tolerance: null`. Unsupported cells remain
  present with their recorded errors. Display labels are “Published population”,
  “Initial importance weights”, and “Calibrated weights”.
- `age_calibration.error_summary` reports `all_cells` and `supported_cells`,
  each with `denominator`, `excluded_cells`, and
  `maximum_absolute_relative_deviation`. No fraction-within-10-percent pass
  rate is published.
- `age_calibration.weights.initial_importance` and `.calibrated` each contain
  `records`, `positive_records`, `zero_records`, `total`, `minimum`, `maximum`,
  `maximum_weight_share`, `effective_sample_size`,
  `effective_sample_size_fraction`, `top_1pct_count`, `top_1pct_weight_share`,
  and `top_1pct_convention`. These are diagnostics, not quality badges.
- `age_calibration.caps.executor` and `.solver` each contain `anchor`,
  `max_weight_ratio`, and `realized`; keep each cap beside its own anchor.
- `age_calibration.solver` contains recorded `method`, `epochs`,
  `learning_rate`, `seed`, `mass`, `initial_loss`, `final_loss`,
  `convergence_status: "unavailable"`, and `diagnostic_warnings`.
- `age_calibration.trajectory` contains `kind` (`complete_recorded` or
  `recorded_summary`), `summary`, and `values` (the verified recorded list, or
  null). Never interpolate epochs or infer convergence. The API's final loss
  is distinct from the last recorded trajectory item.
- `age_calibration.column_provenance` groups column names with `{columns,
  source_role, source_sha256}` attribution to the pinned phase, diagnostics,
  and independently recomputed support evidence.

`executions` contains exactly four public entries: `composition_cold`,
`composition_warm`, `calibration_extension`, `calibration_replay`. Each keeps
the existing graph-viewer execution fields (`label`, `description`,
`execution_manifest_sha256`, `computation_content_key`, `nodes`, cache/computed
counts and failure lists), plus `cache_origin` and
`source_composition_receipt_sha256`. Node/typed-edge shape is unchanged from
the existing graph attachment. The first two executions contain the nine
composition nodes; the latter two contain those nine plus
`national.age_counts` and `national.demographic_calibration`. The extension
has nine inherited hits and two computed nodes; replay has eleven hits. Only
actual manifest producer references create edges. The cache-only
materialization is root-only and absent from this public execution roster.

Flat metadata provenance includes root binding, both acceptance and run receipt
digests, composition/calibration runner and source revisions, calibration
phase/replay receipt digests, materialization receipt digest, aggregate support
and diagnostics digests, calibrated manifest and weight digests, and exporter
and writer identities. Private paths are input locations only; they are never
copied into public attachments.

## Scope and pending integration

The backend verifies pinned aggregate receipts and manifests only. It never
opens raw sources, source stores, checkpoint/population payloads, PUF data or
household count arrays. Root supplies a pinned aggregate materialization
record after its separate cache-only checkpoint verification. No genuine
export is authorized by an invented fixture or by this schema document.

Group-quarters representativeness, sex/head mapping, observation-date
harmonization, housing transfer, PUF enrichment, national/CD launch validation,
and pruning remain unresolved. Prior wages remain outside this cross-sectional
experiment.

## Root binding and API

Run the module by file path with an already installed Python interpreter. The
exporter uses only the standard library and directly loads the existing graph
helpers and candidate writer. It does not import the Microcosm package, a country
model, numerical libraries, source adapters or an execution engine.

```sh
python packages/microcosm-build/src/microcosm/build/graph_evidence_combined_age.py \
  --binding /private/reviewed/root-binding.json \
  --binding-sha256 REVIEWED_EXTERNAL_SHA256 \
  --output /private/new-candidate-directory
```

The Python entry point is
`export_combined_age_evidence(binding_path=Path(...), binding_sha256="...", output_root=Path(...))`.
It returns the candidate manifest reference, graph/build attachment digests,
calibrated checkpoint identity, measured household/person counts, profile and
`index_updated: false`. The output directory must not exist. Errors are
`EvidenceError` with path-free messages. Input JSON is regular-file-only,
symlink-refusing, size-bounded, duplicate-key-refusing and finite-number-only.

Root must review and externally SHA-pin the entire binding, which has exactly:

- `schema: microcosm.us_combined_age_evidence_binding.v1`, `verdict: APPROVE`,
  `profile: us_combined_asec_acs_national_age_v1`, `invented_sources: boolean`.
- `candidate_release_id`, `run_id`, `created_at` (the writer validates IDs/time).
- `files`: each role below maps to `{path, sha256}`. Paths are private input
  locations and are never published. File digests are rechecked before any
  candidate directory is created.
- `identities`: `composition_head`, `calibration_head`, `materializer_head`
  (40-character Git revisions); `composition_runner_sha256`,
  `calibration_runner_sha256`, `materializer_sha256`, `exporter_sha256`,
  `base_exporter_sha256`, `writer_sha256` (64-character byte digests).
  The last three are verified against the exact executing source files.
- `node_keys` and `implementation_hashes`: exact maps for the eleven node IDs
  listed in this document. They must agree with phases and manifests, including
  the nine unchanged inherited nodes.
- `source_vintages`: `{asec: [recorded strings], acs: [recorded strings]}`.
  These labels are an explicit root binding attestation: root must transcribe
  reviewed source provenance, including ASEC survey versus income-year
  distinctions. The exporter does not infer observation years from the target
  year or fabricate source dates. These labels are attributed to the pinned
  root binding, rather than to a source file the exporter did not open.

The seventeen file roles are:

| Role | Required recorded evidence |
| --- | --- |
| `composition_acceptance` | Root's completed composition acceptance |
| `composition_run` | Original composition run receipt |
| `composition_cold_receipt` | Original cold worker receipt |
| `composition_warm_receipt` | Original fresh warm worker receipt |
| `composition_cold_manifest` | Exact cold manifest bytes |
| `composition_warm_manifest` | Exact warm manifest bytes |
| `calibration_acceptance` | Root's completed age-calibration acceptance |
| `calibration_run` | Age-calibration run receipt |
| `calibration_extension_receipt` | Calibrate worker receipt, nine hits and two computations |
| `calibration_replay_receipt` | Fresh warm worker receipt, eleven hits |
| `calibration_extension_manifest` | Exact extension manifest bytes |
| `calibration_replay_manifest` | Exact replay manifest bytes |
| `materialization_receipt` | Root-only cache materialization receipt, externally hashed |
| `aggregate_verification` | Independent support and concentration aggregate |
| `checkpoint_readback` | Exact tables, IDs, strata and weights readback attestation |
| `diagnostics` | Exact stored diagnostics JSON bytes copied by materializer |
| `calibrated_manifest` | Production-saved materialization manifest |

The expected materializer schema is
`microcosm.us_national_age_cache_materializer.v1`; the aggregate appends
`.aggregate`. Its receipt `outputs` must hash the four supplied materializer
JSON outputs and identify the nonempty calibrated checkpoint file. The exporter
verifies these pinned attestations without opening that checkpoint. Root's
materialization review remains responsible for its actual file digest and
checkpoint readback. Composition ancestry uses `checkpoint_sha256` throughout
its separate public block.

## Interpretation and regression boundaries

The graph's `computation_content_key` is the calibrated replay content key.
Each execution retains its own exact manifest byte digest. Composition cold and
warm share one computation key; extension and replay share another. Cache
origins are respectively `empty_composition_store`, `same_composition_store`,
`mirror_of_accepted_composition_store`, and `same_calibration_store`.

A recorded failed geography gate remains failed, including its recorded check
failures and numeric details. A schema-3 manifest has no executor state record;
therefore `execution_record` stays null and `gate_exception` stays false. A
receipt outcome does not invent an executor exception. Gate checks retain the
existing `{name, passed, failures, details}` shape.

Provenance is a nested map whose values are flat strings, rather than a nested
mapping of source-specific objects. It contains all nine identity fields above,
all seventeen input role digests suffixed `_sha256`, `root_binding_sha256` and
`calibrated_weights_sha256`. Source vintage labels are root-attested through
that binding. The build summary has boolean checks for pinned inputs,
composition ancestry, extension/replay cache origin, all targets, independent
support, checkpoint identity attestation, recorded trajectory, no source or
population reads, and excluded eligibility. These are verification checks,
not statistical acceptance gates.

Initial/final weight extrema and positive/zero counts come from the pinned
phase summaries. Independent sums, squared sums, ESS, maximum share and top-one-
percent concentration come from the materializer aggregate. Overlapping fields
are reconciled; solver float32 concentration scalars allow a relative numerical
comparison of two parts per million against independently summed diagnostics.
No household arrays are reconstructed. The full recorded trajectory is emitted
only after recomputing its complete summary; no terminal convergence verdict is
inferred from it.

Fixtures are generated by `fixture()` in
`packages/microcosm-build/tests/test_graph_evidence_combined_age.py`. They use
invented 20- and 200-household evidence, one zero-support band, a recorded failed
geography gate, different API-final and last-recorded losses, and the two cache
origins. A separate test publishes all 500 recorded trajectory values. Existing
F001, ASEC v3, sized-income and candidate writer regression suites remain
unchanged. No new profile is registered in an index by this exporter.
