# UK full-build graph

The UK has one calibration build. It constructs the canonical FRS spine, samples the pool when requested, expands linked entities into K geographic copies, assigns locations, constructs the selected contribution matrix, calibrates, optionally selects exactly k households and refits, evaluates gates and diagnostics, and packages a checked H5.

**The default is to calibrate all applicable geographies together.** An omitted selector and `--target-geographies all` have the same target scope. `--target-geographies country` explicitly selects country-level rows in the same graph; regional rows are not country-level rows. No failure, size request or performance setting changes the selector implicitly.

## Run the build

From a canonical spine checkpoint with its `.build.json` and `.spine_gates.json` sidecars:

```bash
uv run --no-sync python tools/build_uk_full.py \
  --input-h5 /data/uk/spine.h5 \
  --ladder /data/uk/ladder.npz \
  --atomic-support-ew /data/uk/supports/uk_ew_output_area_2021_support.npz \
  --atomic-support-scotland /data/uk/supports/uk_scotland_output_area_2022_support.npz \
  --atomic-support-ni /data/uk/supports/uk_ni_data_zone_2021_support.npz \
  --ledger-facts /data/chronicle/uk-artifact \
  --out /data/uk/full-build
```

The three atomic-area supports are the artifacts pinned in `uk/uk_atomic_area_supports.provenance.json` and `uk/spec/sources.yaml` (built by `tools/build_uk_atomic_area_supports.py`); `--atomic-support-sha256-{ew,scotland,ni}` pin them like `--ladder-sha256`, and a release candidate requires all four pins.

The checkpoint must bind the exact frame content, current spine stage roster and gate-report bytes. Historical candidate H5 files and reviewed-bypass sidecars are not alternate build sources. Chronicle facts and manifest must match the independently reviewed national and local feed declarations; filtering targets does not relax source validation.

The target registry binds Census household and demographic rows from the reviewed Chronicle feed, including the approved Northern Ireland constituency geography. Geography is assigned after expansion by the shared atomic-geography operators: `uk.full.identity` keys every household with `household_draw_key` from the spine's explicit lineage (source household id, SPI support channel and clone index, CGT clone and donor flags) and the pool clone index; `uk.full.geography.assign` draws one atomic area per household (E&W 2021 Output Area, Scotland 2022 Output Area, NI 2021 Data Zone) by census household count within the household's FRS region, with a keyed `sha256-u53-v1` stream so a household's draw never depends on row order, K or any other household; `uk.full.geography.derive` reads every larger geography off the support's versioned mappings; the shared `uk.full.geography.gate` and the UK distribution gate `uk.full.geography_gate` follow, and `uk.full.pool` refuses unless the shared gate passed. The OA ladder still supplies the constituency and local-authority rosters, household dispersion and lookup support for target compilation; its household counts are not a second source of calibration targets. Source receipts retain the Chronicle identity, the paired ladder digest and the geography binding (assignment mode, definition sha256, support pins, identity column, stream), so target values and the geography used to assign households can be audited separately. `--geography-assignment legacy` keeps the previous sequential two-stage ladder draw (`uk.full.locations`, `uk.full.geography_mapping`) for measurement builds; `--release-candidate` refuses it.

To include raw spine construction in the same execution, pass `--spine-request /data/uk/spine-request.json`. This file is a JSON array of the raw-source arguments accepted by the maintained spine preparation API:

```json
[
  "--frs-raw-dir", "/data/frs/2024-25",
  "--spi-tab", "/data/spi/put2223uk.tab",
  "--hmrc-ods", "/data/hmrc/collated.ods",
  "--cgt-ods", "/data/hmrc/cgt.ods",
  "--was-tab", "/data/was/household.tab",
  "--lcfs-hh-tab", "/data/lcfs/household.tab",
  "--lcfs-person-tab", "/data/lcfs/person.tab",
  "--etb-tab", "/data/etb/household.tab"
]
```

The request declares existing source adapters and lazy transforms. It does not launch the separate spine command first. A spine-only command remains available to materialize an execution checkpoint.

For smaller outputs, add `--dataset-households 100000` to use the common informed L0 search, exact-count draw and refit. `--n-clones K` sets the number of geographic copies in the pool. K and k are independent, and neither narrows target scope. A country-only run may explicitly request a smaller K, but this is never inferred.

`--sample-fraction` samples before geographic cloning. Raw-spine sampling in the JSON request instead occurs at the original ingest boundary, before enrichment. An already sampled spine cannot be sampled a second time. The effective sample fraction controls development gate and target-admission policy.

## Graph owners and shared contracts

The existing source-stage roster supplies the spine graph. The superseded `frs_hmrc_retained_leaves` and `hmrc_spi_income` executable stages are removed. The active replacements are `frs_hmrc_spine_leaves`, `spi_support_channel` and `hmrc_spi_income_spine`; source extraction helpers remain under source-focused owners.

| Operation | Graph owner |
| --- | --- |
| Raw FRS and donor preparation, enrichment, support channels | Existing UK spine stage nodes and their declared composite operations |
| Assembled and transferred spine gates | `spine.gates.assembled`, `spine.gates.transferred` |
| Bound checkpoint admission | `uk.full.spine_checkpoint`, when resuming a saved spine |
| Pool sample and mass normalization | `uk.full.sample`, `uk.full.normalize` |
| Linked entity expansion and ancestry | `uk.full.expand`, `uk.full.expand.owned` |
| Post-clone household identity | `uk.full.identity` |
| Atomic-area support import, identity-keyed assignment, derivation and shared integrity gate | `uk.full.geography.support.{0,1,2}`, `uk.full.geography.assign`, `uk.full.geography.derive`, `uk.full.geography.gate` (shared `geography.*@1` kernels) |
| UK geography distribution gate | `uk.full.geography_gate` |
| Legacy sequential ladder draw (`--geography-assignment legacy`, measurement builds only) | `uk.full.locations`, `uk.full.geography_mapping`, `uk.full.geography_gate` |
| Full pinned source/register compilation | `uk.full.target_compilation` |
| Explicit target selection and inclusion/exclusion receipt | `uk.full.target_selection` |
| Engine measures and ordered contribution problem | `uk.full.measures`, `uk.full.problem` |
| Complete original-pool checkpoint | `uk.full.pool` |
| Source/reference preflight and dense reference | `uk.full.gates.preflight`, `uk.full.dense` |
| Optional informed search, exact draw and refit | `uk.full.size_search`, `uk.full.size_draw`, `uk.full.size_refit` |
| Selected population and installed calibrated weights | `uk.full.selected`, `uk.full.calibrated` |
| Rotated local holdout and final gates/diagnostics | `uk.full.holdout`, `uk.full.gates.calibrated` |
| Export contract, H5 readback and package inventory | `uk.full.export.prepare`, `uk.full.export.readback`, `uk.full.package` |
| Gate, comparison and export evidence for certification | `uk.full.certification` |

`operations.json` is generated from the compiled graph, including actual dependencies and artifact owners. It is the execution inventory, rather than a second manually maintained pipeline roster. Composite source/model stages preserve their existing numerical boundary; for example, WAS retains its joint donor/recipient encoding dependency. This registration does not change imputation order, RNG consumption, clone IDs or geography methodology.

Shared machinery includes graph execution/storage/replay, typed artifacts, explicit same-kind weight updates, target selection receipts, ordered sparse calibration problem/solution/result codecs, exact-count selection, atomic artifact materialization and bundle publication. UK adapters retain source interpretation, entity relationships, geography mappings, measure bindings and gate prescriptions.

The full target compiler preserves the unreduced band-edge register, reference-period compilations, approved exclusions and frozen-register completeness checks before target selection. A nonempty country-only problem can have zero local rows. Local holdout is then inapplicable, while source, identity, mass and geographic integrity checks remain applicable. Omitted rows are never reported as fitted constraints.

## Replay and outputs

The shared store defaults to `<out>/.graph-store`. `--graph-store` can reuse another store. `--resume require` requires completed numerical nodes and evidence to be available; output materialization and byte readback still verify the recreated files. `--resume-size-checkpoint` imports a legacy size-search checkpoint only after validating invocation identity, ordered target/household axes, initial weights and recomputed losses. It skips the saved dense solve and search. New runs persist their intermediates as graph artifacts before drawing.

Each attempt also keeps its checkpoint manifests and small stage, gate and provenance reports under `<graph-store>/uk-full-attempts/<attempt-id>`. A failed run's `failure.json` links to this evidence, including a persisted spine gate verdict before downstream admission stops execution. A previously completed output bundle remains intact.

The output bundle includes `microcosm_uk_<year>.h5`, selected targets, target diagnostics, area support, rotated holdout, stored source/stage/gate evidence and graph manifests. `candidate.json` is the immutable graph package inventory: it binds the dataset, its evidence, the target selector and the independent K/k request. Comparison inputs refer to this candidate identity. Adding comparison evidence does not rewrite the candidate package.

The terminal graph node writes unsigned `certification.json` from those identified artifacts and any declared native or matched-size comparisons. `build.json` is the completion marker that binds both `candidate.json` and the certification artifact. Physical output bytes are checked against their declared artifacts. Bundle publication writes the completion marker last and rolls back handled failures or interrupts. A process kill or power loss can leave an absent completion marker; a directory without a valid bound marker is not a completed build.

Structural failures stop export. The maintained local statistical failure policy may still export an unreleasable diagnostic candidate with a nonzero process status. Missing evidence remains explicit. `--release-candidate` applies the maintained strictness and solve settings; it does not publish, sign or authorize a release. Fixture acceptance proves graph behavior. Native certification additionally requires measured incumbent comparison evidence supplied with `--native-scorecard`; exact-count promotion also requires the measured comparison at the requested k through `--matched-size-scorecard`. The certification node verifies their candidate and output identities before assessing readiness.

The old `build_uk_rowwise_candidate.py` and `calibrate_uk_national_dataset.py` command names temporarily forward to this exact CLI. Both default to all geographies. They contain no separate scientific execution and should be removed after downstream command invocations migrate.
