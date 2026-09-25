# UK full-build graph

The UK has one calibration build. It constructs the canonical FRS spine, samples the pool when requested, expands linked entities into K geographic copies, assigns locations, constructs the selected contribution matrix, calibrates, optionally selects exactly k households and refits, evaluates gates and diagnostics, and packages a checked H5. It is registered on the shared executable graph (`microcosm.graph`) and served by one driver, `microcosm-build-uk` (`tools/build_uk_full.py`), which carries the release roles of microcosm#823: `--release-role dense` runs this graph, and `--release-role national` is validated by the same posture-aware validator and dispatched to the retained calibration seam (see "The national role" below). `tools/build_uk_rowwise_candidate.py` is a stub over the driver, so the runbook commands that name it keep working.

**The default is to calibrate all applicable geographies together.** An omitted selector and `--target-geographies all` have the same target scope. `--target-geographies country` explicitly selects country-level rows in the same graph; regional rows are not country-level rows. No failure, size request or performance setting changes the selector implicitly. The geographic pool copies K (`--n-clones`), the exported household count k (`--dataset-households`) and the target scope are independent settings. This registration preserves the current UK algorithms and does not establish native candidate acceptance.

## Run the dense build

From a canonical spine checkpoint with its `.build.json` and `.spine_gates.json` sidecars:

```bash
uv run --no-sync python tools/build_uk_full.py --release-role dense \
  --input-h5 /data/uk/spine.h5 --input-sha256 <spine-sha256> \
  --ladder /data/uk/ladder.npz --ladder-sha256 <ladder-sha256> \
  --ledger-facts /data/chronicle/uk-artifact \
  --ledger-facts-sha256 <facts-sha256> --ledger-manifest-sha256 <manifest-sha256> \
  --out /data/uk/full-build
```

`--release-role` is required. The dense role supplies every unset solve default from the local doctrine (epochs, learning rate, seed, K, weight rule, constituency vintage, selection) and refuses the national role's flags (`--target-loss-cap`, `--allow-unpinned-feed`, `--incumbent-h5` and `--incumbent-sha256`). `--ladder` with `--ladder-sha256` is required by the dense role, `--input-sha256` by every `--input-h5` build, and the three Ledger arguments (`--ledger-facts`, `--ledger-facts-sha256`, `--ledger-manifest-sha256`) by every build. The supplied hashes must agree with the committed Chronicle pins: a target-scope filter does not authorise a different source, and the dense role has no unpinned-feed override.

The checkpoint must bind the exact frame content, the current spine stage roster and the gate-report bytes. The bound-spine node compares the checkpoint's gate report digests with the branch's own gate declarations and refuses a spine whose gate manifest differs from them, so `--input-h5` needs a spine built by a branch with the same declarations; every acceptance spine on disk when this registration landed predates them and is not admitted. Historical candidate H5 files and reviewed-bypass sidecars are not alternate build sources. Chronicle facts and manifest must match the independently reviewed national and local feed declarations; filtering targets does not relax source validation.

The target registry binds Census household and demographic rows from the reviewed Chronicle feed, including the approved Northern Ireland constituency geography. The OA ladder supplies geographic assignment and lookup support. Its household counts are not a second source of calibration targets. Source receipts retain the Chronicle identity and the paired ladder digest, so target values and the geography used to assign households can be audited separately.

To include raw spine construction in the same execution, pass `--spine-request /data/uk/spine-request.json` instead of `--input-h5`. This file is a JSON array of the raw-source arguments accepted by `uk_runtime.spine_build` (the arguments of `tools/build_uk_frs_spine.py`):

```json
[
  "--frs-raw-dir", "/data/frs/2024-25",
  "--spi-tab", "/data/spi/put2223uk.tab",
  "--hmrc-ods", "/data/hmrc/collated.ods",
  "--was-tab", "/data/was/household.tab",
  "--nts-household-tab", "/data/nts/household.tab",
  "--nts-individual-tab", "/data/nts/individual.tab",
  "--nts-trip-tab", "/data/nts/trip.tab",
  "--nts-stage-tab", "/data/nts/stage.tab",
  "--nts-ticket-tab", "/data/nts/ticket.tab",
  "--lcfs-hh-tab", "/data/lcfs/household.tab",
  "--lcfs-person-tab", "/data/lcfs/person.tab",
  "--etb-tab", "/data/etb/household.tab"
]
```

The request declares existing source adapters and lazy transforms; the spine stages execute in the same graph, and the driver supplies `--spine-h5` when the request omits it. A spine request has no checkpoint H5 to pin, so `--input-sha256` is not required on that path. The spine-only command remains available to materialise an execution checkpoint.

For smaller outputs, add `--dataset-households 100000` to use the common informed L0 search, exact-count draw and refit. `--baseline-pi-floor` floors the inclusion probability the refit's Horvitz-Thompson baseline divides each selected row's dense weight by (0, the default, is the untrimmed baseline), and `--no-size-checkpoint` skips materialising `size_selection_checkpoint.{npz,json}` into `--out`. `--n-clones K` sets the number of geographic copies in the pool; `--candidate-clone-counts 5,10,15` with `--dry-run` prints the compiled operation inventory for each K without solving. K and k are independent, and neither narrows target scope. A country-only run may explicitly request a smaller K, but this is never inferred. `--households-only` binds only the Chronicle census-household constituency targets (a target-selection family filter).

`--sample-fraction` samples before geographic cloning. Raw-spine sampling in the JSON request instead occurs at the original ingest boundary, before enrichment. An already sampled spine cannot be sampled a second time. The effective sample fraction controls development gate and target-admission policy.

## The national role

`--release-role national` builds the certified national line. The driver parses and validates it with the same posture-aware validator (the national role refuses the dense role's arguments, `--release-candidate` among them, and requires a bound `--input-h5` with `--input-sha256`; a `--spine-request` is refused because the seam reads a pinned checkpoint) and then, before any graph is prepared, dispatches to `uk_runtime.national_role.run_national_role`. That module is the rowwise tool's national branch moved unchanged: the calibration seam of microcosm#823 (`uk_runtime.calibration_run.run_uk_calibration`, the national calibration stage, the release-cut battery and the certifier stay exactly as they were). Its outputs (`microcosm_uk_2024_25.h5`, `build_record.json`, `microcosm_uk_2024_25.terminal_gates.json`, `calibration_diagnostics.json`, the registries and `rowwise_candidate_manifest.json`), its Logbook row, staging telemetry, staged bundle and incumbent evaluation are the seam's own; the [national calibration runbook](uk-national-calibration-runbook-623.md) and the [national release assembly runbook](uk-national-release-assembly-runbook-806.md) describe them. Parity with the previous national line is by identity: the same engine runs the same code. A posture-driven national path through this graph is planned as the next change on this line and is not part of this registration.

## Graph owners and shared contracts

The spine graph is the 33-stage source roster declared by the UK spec, built by `uk_runtime.spine_build` (the spine tool moved into the package; `tools/build_uk_frs_spine.py` is a six-line shim over it). The superseded `frs_hmrc_retained_leaves` and `hmrc_spi_income` stages and the `UK_SPINE_EXCLUSIONS` list that hid them are removed; the active HMRC path is `frs_hmrc_spine_leaves`, `spi_support_channel`, `spi_income_band_donors` and `hmrc_spi_income_spine`, and the FRS HMRC leaf columns come from `uk_runtime.frs_hmrc_source`. The spine's assembled and transferred gate batteries are graph nodes (`uk_runtime.graph_evidence`); stage evidence and fit-weight records are read back from the content store through the shared `stage_evidence` artifact rather than from in-memory collectors; the HMRC replay sidecar is rebuilt from the SPI stage's checkpoint metadata; the spine sidecar records the operation inventory and the graph manifest, and the graph manifest is saved as `spine.graph.json` under the checkpoint root.

| Operation | Graph owner |
| --- | --- |
| Raw FRS and donor preparation, enrichment, support channels | The 33 UK spine stage nodes and their declared composite operations |
| Assembled and transferred spine gates | `spine.gates.assembled`, `spine.gates.transferred` |
| Bound checkpoint admission | `uk.full.spine_checkpoint`, when resuming a saved spine |
| Pool sample and mass normalization | `uk.full.sample`, `uk.full.normalize` |
| Linked entity expansion and ancestry | `uk.full.expand`, `uk.full.expand.owned` |
| Location draw, mapping and integrity | `uk.full.locations`, `uk.full.geography_mapping`, `uk.full.geography_gate` |
| Full pinned source/register compilation | `uk.full.target_compilation` |
| Explicit target selection and inclusion/exclusion receipt | `uk.full.target_selection` |
| Engine measures and ordered contribution problem | `uk.full.measures`, `uk.full.problem` |
| Complete original-pool checkpoint | `uk.full.pool` |
| Source/reference preflight and dense reference | `uk.full.gates.preflight`, `uk.full.dense` |
| Informed search, exact draw and refit (with `--dataset-households`) | `uk.full.size_search`, `uk.full.size_draw`, `uk.full.size_refit` |
| Selected population (with `--dataset-households`) and installed calibrated weights | `uk.full.selected`, `uk.full.calibrated` |
| Rotated local holdout and final gates/diagnostics | `uk.full.holdout`, `uk.full.gates.calibrated` |
| Export contract, H5 readback and package inventory | `uk.full.export.prepare`, `uk.full.export.readback`, `uk.full.package` |
| Gate, comparison and export evidence for certification | `uk.full.certification` |

The modules under `uk_runtime` divide the roster: `graph_build` composes the graph over the bound spine, `graph_population` owns sample through the geography gate, `graph_targets` owns target compilation through the problem, `graph_calibration` owns the dense solve, the size nodes and the calibrated population, `graph_terminal` owns the gate nodes, the holdout, the export and the package, `full_certification` owns the terminal certification node, and `full_build_cli` resolves the request, executes graph endpoints and atomically materialises their stored artifacts.

`operations.json` is generated from the compiled graph, including actual dependencies and artifact owners. It is the execution inventory, rather than a second manually maintained pipeline roster. Composite source/model stages preserve their existing numerical boundary; for example, WAS retains its joint donor/recipient encoding dependency. This registration does not change imputation order, RNG consumption, clone IDs or geography methodology.

Shared machinery includes graph execution/storage/replay, typed artifacts, explicit same-kind weight updates, target selection receipts, ordered sparse calibration problem/solution/result codecs, exact-count selection, atomic artifact materialization and bundle publication. UK adapters retain source interpretation, entity relationships, geography mappings, measure bindings and gate prescriptions.

The full target compiler preserves the unreduced band-edge register, reference-period compilations, approved exclusions and frozen-register completeness checks before target selection. A nonempty country-only problem can have zero local rows. Local holdout is then inapplicable, while source, identity, mass and geographic integrity checks remain applicable. Omitted rows are never reported as fitted constraints.

## Replay and outputs

The shared store defaults to `<out>/.graph-store`. `--graph-store` can reuse another store. `--resume require` requires completed numerical nodes and evidence to be available; output materialization and byte readback still verify the recreated files. `--resume-size-checkpoint` imports a legacy size-search checkpoint only after validating invocation identity, ordered target/household axes, initial weights and recomputed losses. It skips the saved dense solve and search. New runs persist their intermediates as graph artifacts before drawing.

Each attempt also keeps its checkpoint manifests and small stage, gate and provenance reports under `<graph-store>/uk-full-attempts/<attempt-id>`. A failed run's `failure.json` links to this evidence, including a persisted spine gate verdict before downstream admission stops execution. A previously completed output bundle remains intact.

The output bundle is named from the role's posture and the FRS release vintage: `microcosm_uk_2024_25_local.h5`, its signed gate report `microcosm_uk_2024_25_local.local_gates.json`, and the `.diagnostics.json`, `.targets.csv`, `.area_support.csv`, `.holdout.json` and `.target_selection.json` siblings on the same stem, beside `graph.json`, `operations.json`, the stored source/stage/gate evidence and the graph manifests. `candidate.json` is the immutable graph package inventory: it binds the dataset, its evidence, the target selector and the independent K/k request. Comparison inputs refer to this candidate identity. Adding comparison evidence does not rewrite the candidate package. `rowwise_candidate_manifest.json` is projected from the stored artifacts in the schema-4 shape the rowwise tool wrote (`graph_terminal.rowwise_candidate_manifest_from_graph`), so the dense release pre-flight and assembler read a graph build as they read a rowwise-tool build; it records the release role, the release verdict, `staging_delivery` and `staged_dataset`.

The terminal graph node writes unsigned `certification.json` from those identified artifacts and any declared native or matched-size comparisons. `build.json` is the completion marker that binds both `candidate.json` and the certification artifact. Physical output bytes are checked against their declared artifacts. Bundle publication writes the completion marker last and rolls back handled failures or interrupts. A process kill or power loss can leave an absent completion marker; a directory without a valid bound marker is not a completed build.

A non-dry dense run is wrapped in the rowwise tool's operational envelope: the Logbook attempt (a `uk-local-candidate` row spooled under `<out>/logbook-spool` on every terminal outcome, chained through `--logbook-prev-row-digest` or `POPULACE_LOGBOOK_PREV_ROW_DIGEST`, with an error receipt on failure), version 2 staging telemetry with stage events around each graph phase and per-epoch `calibration_progress` rows from the dense solve, and the staged-dataset delivery of the published bundle under `staged/<run_id>/` in the private repository. `--staging-local-only`, `--no-staging`, `--staging-read-back` and `--no-staged-dataset` behave as in [UK staging operations](uk-staging-operations.md). Dry runs plan without solving or writing and record no Logbook row.

Structural failures stop export. The maintained local statistical failure policy may still export an unreleasable diagnostic candidate with a nonzero process status. Missing evidence remains explicit. `--release-candidate` applies the maintained strictness and solve settings; it does not publish, sign or authorize a release. Fixture acceptance proves graph behavior. Native certification additionally requires measured incumbent comparison evidence supplied with `--native-scorecard`; exact-count promotion also requires the measured comparison at the requested k through `--matched-size-scorecard`. The certification node verifies their candidate and output identities before assessing readiness.

`tools/build_uk_rowwise_dataset.py` stays the separate driver it was (its tests load it by path, and it still serves `--candidate-clone-counts`). `tools/calibrate_uk_national_dataset.py` was retired by microcosm#823 and does not forward.

## Known behaviour changes

Recorded for review in `experiments/901-uk-main-rebase-receipts.md` (R3):

- A blocked *assembled* spine gate now fails inside `run_graph`, so the gate report lives in the content store on that path (the previous tool wrote `spine_gates.json` before raising); a blocked *transferred* gate still writes the file.
- `numerical_dependencies` pins installed versions into the H2 fixture.
- `_normalise_uk_local_bound_families` accepts an empty declaration (country-only scope).
- Per-epoch `calibration_progress` staging rows come from the dense solve only.
- The HMRC family names `spi_income_band_donors` (microcosm#1006) as a predecessor and admits its two operation kinds; the contract otherwise refused the drifted operation order.
- `tools/build_uk_rowwise_dataset.py` stays as it was on main (its tests load it by path; it still serves `--candidate-clone-counts`).

The dense and national parity builds through this driver are owed once a spine can be built at main's head (receipts R4); the national role is parity by identity regardless.
