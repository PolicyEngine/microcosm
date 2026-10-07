# Transport graph kernels: calibration and terminal side

The country-neutral graph kernels a donor-based (transport) country composes
for its calibration and its terminal evidence. They live in
`microcosm.build.transport` and `microcosm.calibrate.ordered_kernels`, read
everything country-specific from node parameters and declared sources, and
import no country runtime. The population-side kernels (CREATE, boundary
FILTERs, transport rewrites) are a separate work package and are not
described here.

## Kernels

| Ref | Module | Role | Reads | Emits |
| --- | --- | --- | --- | --- |
| `targets.compile@1` | `transport/target_kernels.py` | compute | one declared source (a Chronicle consumer artifact); params `country`, `references`, `references_sha256`, optional `facts_sha256` | `surface` (`microcosm.targets.surface` v1) |
| `targets.problem@1` | `transport/target_kernels.py` | compute | population slices; params `entities`, `weight_entity`; artifact `surface` | `problem` (`microcosm.calibrate.ordered-problem` v1) |
| `calibrate.ordered_adam@1` | `calibrate/ordered_kernels.py` | REWEIGHT | one slice on the weight entity; artifact `problem`; params `epochs`, `learning_rate`, `mass`, optional `max_weight_ratio` with `weight_anchor="design"` | weights; `solution`, `result` |
| `takeup.compare@1` | `transport/target_kernels.py` | compute | one declared source (hold-out facts); population slices; params as `targets.compile@1` plus `entities`, `weight_entity` | `comparison` (`microcosm.targets.comparison` v1) |
| `diagnostics.calibration@1` | `transport/terminal_kernels.py` | compute | one slice on the weight entity; artifacts `problem`, `solution`, `result`, `surface`; params `weight_entity`, optional `build` | `diagnostics` (`microcosm.diagnostics.calibration` v8) |
| `gates.battery@1` | `transport/gate_kernels.py` | GATE | params `country`, `gates`, `gates_sha256`, `phase`, `release_candidate`, optional `synthetic_smoke`, `upstream`, and `entities` with `weight_entity`; any artifacts as evidence | `gate_report` (`microcosm.gates.phase-report` v1) |
| `export.prepare@1` | `transport/terminal_kernels.py` | compute | population slices; the gate reports; params `entities`, `weight_entity`, `time_period`, optional `bindings` | `export_descriptor` |
| `export.readback@1` | `transport/terminal_kernels.py` | GATE | one declared source (the written H5); artifact `export_descriptor` | `export_readback` |
| `transport.package@1` | `transport/terminal_kernels.py` | compute | artifact `export_readback`, the gate reports, any others; optional `bindings` | `receipt` |

Each kernel's typed outputs are listed in `artifact_types.KERNEL_OUTPUTS`
(and, for the solve, `ordered_kernels.OUTPUT_TYPES`). Every kernel refuses a
node whose declared `artifact_outputs` differ from its entry, so a composer
should build its `ArtifactOutput` tuples from those tables.

## Inputs

A structured value travels as one canonical-JSON string parameter (sorted
keys, no whitespace); a non-canonical spelling is refused, so identical JSON
values always give identical node keys. The spec documents, `references` and
`gates`, each come with the SHA-256 of the spec resource they were taken
from (`references_sha256`, `gates_sha256`); the free-form `build` and
`bindings` parameters do not. No kernel binds a whole country spec's
fingerprint into its implementation hash.

- `references` is a document carrying only the keys `country`,
  `schema_version`, `hierarchy` and `target_references`: the subset of a
  country's `target_references.json` the node needs. Its rows are parsed by
  the country-spec loader's own validator, so a row parses to the reference
  `load_country_spec` would build. Any other key (`description`,
  `target_profile`, `allowed_value_operations`) is refused, so the composer
  strips them.
- `gates` is a `gates.json` document. The kernel hands a gate its parameters
  only from this document; a binding that carries values of its own binds
  them into the kernel's hash (see "Gates").

A kernel that rebuilds a frame (`targets.problem@1`, `takeup.compare@1`,
`export.prepare@1`, and `gates.battery@1` when it declares `entities`) needs
the person table and every entity named in `entities`, so the node must
slice each of them. It refuses row-masked slices: a frame rebuilt from a
subset of rows would silently aggregate, calibrate or export that subset.

## Targets and the ordered problem

`targets.compile@1` resolves its references against the source facts with
`compile_ledger_target_references`. Compilation is strict: a placeholder
reference (any `activation_status` other than empty or `"active"`), a
reference no fact matches and an ambiguous match all refuse the node; there
is no skip path. The surface records, for every compiled target, the one
reference that produced it, that reference's status and the fact it
resolved to. It records the facts' content hashes, never their path.

`targets.problem@1` rebuilds the registry, takes `to_target_set()`, compiles
`build_constraint_matrix(frame, targets, weight_entity)` on the node's
population and encodes it with `encode_problem`, binding the surface's
SHA-256. A target whose measure or filter column the node does not slice
refuses the node rather than being skipped. Person- and family-level targets
reach the weight entity through the person table's memberships.

`calibrate.ordered_adam@1` decodes the problem, checks that its entity axis
and starting weights are the node's base population's, and calls the same
`microcosm.calibrate.calibrate(...)` as `calibrate.adam@1`. On equal-valued
fully compiled targets the two install byte-identical weights, because both
build their constraint matrix through `microcosm.calibrate.matrix` and the
solver sees the same matrix. Where the shared solver refuses an input, both
refuse it with the same error. One such input is known: with
`mass="conserve"` and a cap, the solver's conserved total can drift past its
1e-9 tolerance
(`test_ordered_kernels.py`, `_MASS_DRIFT`).

There is one intended difference before the solve: `calibrate.adam@1` skips
uncompilable targets when at least one target compiles (for example, a
measure containing NaN), while `calibrate.ordered_adam@1` refuses a problem
with any skipped target. In the composed transport graph,
`targets.problem@1` already refuses skipped targets, so this difference
cannot reach its calibration node.

## The weight cap

With `max_weight_ratio` set, the node must declare `weight_anchor="design"`.
Two bounds then apply. The solver clamps each weight at the ratio times its
starting weight. The executor checks each installed weight against the ratio
times the design weight captured at CREATE, records
`realized_max_weight_ratio` in the node receipt and rejects the node on a
violation (`microcosm.graph.population`, `_design_cap` and
`_assert_design_weight_cap`). While the starting weights are the design
weights the two bounds coincide. If a step between CREATE and calibration
raises weights above their design values, the executor's check is the one
that rejects; if it lowers them, the solver's clamp is the tighter bound.

## Diagnostics

`diagnostics.calibration@1` rebuilds the completed calibration from the
`problem`, `solution` and `result` artifacts without solving, and emits the
schema-8 model. Each row describes its target as compiled (its own entity,
measure and filter), with its hierarchy from the surface's registry. The
kernel refuses a population whose installed weights are not the solution's,
a result from another solve, and a surface other than the one the problem
binds (`targets.problem@1` always binds it). So a diagnostics node belongs
downstream of the one calibration it describes: a build with several
calibrations (a central run and variants) has one diagnostics node per
calibration.

## Gates

`gates.battery@1` evaluates one phase with `evaluate_phase` and the binding
registry passed to its constructor. Every artifact input is offered to the
battery under its alias, decoded by type where a decoder exists
(`EVIDENCE_DECODERS`). When the node declares `entities`, its slices are
offered as the phase frame.

The bindings' behaviour enters the implementation hash
(`transport/binding_identity.py`). Exact description of arbitrary Python is
not possible by reflection, so a binding must come from a closed vocabulary:
a frozen dataclass defined at module top level (as `FunctionBinding` is),
with no state outside its fields, whose field values are plain data (`None`,
`bool`, `int`, finite `float`, `str`, `bytes`, and exact `tuple`, `list`,
`dict`, `MappingProxyType`, `set` or `frozenset` of plain data with string
keys), enum members, classes, such dataclasses, or undecorated top-level
functions without closures, whose defaults follow the same rules. The hash
covers every field (keeping types and mapping order), each function's
defaults, the source of the modules the binding's functions and classes are
defined in (first-party or not), and the source of every first-party module
(`microcosm.*`, `test_support.*`) those modules can import. Import
statements anywhere in a file, function-local ones included, are read from
source and resolved case-exactly with the import system's path finder,
without executing anything. Closures, partials, bound methods, lambdas,
nested functions, decorated wrappers, builtins, numpy scalars and other
objects are refused when the kernel is constructed. The identity is fixed at
construction; if a binding or a module in its source closure changes
afterwards (a `dict` field edited in place, a source file rewritten), the
kernel refuses to run.

The closure is honest rather than small: `microcosm.build.gates` imports
`us_runtime` lazily inside one function, so a binding over a gate in that
module binds about 190 modules, `us_runtime` included, and an edit there
re-keys the gate node and the export nodes after it (not the calibration).

Not bound, and therefore not allowed to carry gate behaviour: data a
binding's modules read from disk (package JSON, exclusion registers), state a
module mutates at run time (a swapped `__code__`, a reassigned class
attribute), and code reached without an import statement
(`importlib.import_module` or `__import__`, even with a literal name). A
binding takes its values through the manifest's parameters or through
evidence artifacts.

Also not bound: versions of third-party dependencies beyond the kernel's
declared `numpy` and `pandas` dependencies, and submodules loaded by a
package's star import through `__all__` when no import statement names the
submodule. Such dependencies must not carry gate behaviour outside the
bound sources and declared versions.

Gate details must be plain data, and common Python repr addresses
(`... at 0x...>` or `... at 0x...,`) inside failure text and detail keys or
values are blanked. Other text and detail ordering are the binding's own:
failure lines, details and exception text must not depend on unordered set
iteration, including lists built from sets, or the report's bytes would vary
between runs of one computation. Failure text containing a lone surrogate
character is refused by canonical JSON encoding rather than written into a
report.

The node outcome is one of the five graph outcomes:

1. `fail`: a release-blocking entry failed;
2. `unreached`: an entry was not reached because an upstream phase blocked;
3. `evidence_absent`: a release-blocking entry lacked its evidence or binding;
4. `not_applicable`: every entry is declared not applicable, or the phase has
   none (also when an upstream phase blocked, but the artifact is then not
   permitted);
5. `pass`: otherwise.

`artifact_permitted` follows the battery's blocking rule for the node's
posture (`release_candidate`, or `synthetic_smoke`, never both): it holds when
no entry blocks, the phase was reached and no upstream phase blocked. A
kernel exception, including a parameter outside a binding's vocabulary,
becomes the outcome `fail` through the executor, and the report's consumers
become `unreached`. Report signing stays in the outer command that
materializes a release.

The report carries the gate manifest document it was evaluated against, and
`decode_gate_report` replays it. Every row must be one the battery could
have written and must re-serialize to itself; a row is `not_applicable`
exactly when its entry declares a reason, with that reason; rows are
`unreached` exactly when an upstream phase blocked; the upstream entries must
be consistent with their outcome and the posture, be listed among the
evidence aliases, and name one earlier report per phase that can block; and
the outcome and the enforcement are recomputed and compared. A malformed or
inconsistent report is refused with `ValueError`. Decoding cannot tell a
verdict the gate computed from one written by hand, so a report cannot prove
which manifest it was meant to use: `export.prepare@1` and
`transport.package@1` require every report they read to carry the same
manifest document, resource hash and posture, and the report of the last
phase that can block (it folds in every earlier blocking phase); every phase
that can block must therefore come before export. The package also requires
its reports to be exactly the ones `export.prepare@1` recorded in the
descriptor, which the readback carries through. The composer must wire
`gate_report` edges from `gates.battery@1` nodes only.

Phase order: a node must name, as `upstream` artifact inputs, the report of
every earlier phase that can block (one with an applicable release-blocking
entry), each evaluated under this node's posture and the identical manifest
document and resource hash. The documents are compared directly, because the
battery's policy hash leaves out `population_fact_check`. If an upstream
report does not permit the artifact, this phase does not evaluate; its
entries are `unreached`, except those declared `not_applicable`, which keep
that status.

Put a gate node in a version that no calibration node's base contains: a
structural node depends on every member of its base version, so a gate that
is a member of the calibration base would re-key calibration on any
`gates.json` edit. A preflight phase belongs on its own FILTER branch.

### Gate bindings

`transport/gate_bindings.py` holds `TRANSPORT_GATE_REGISTRY`, the registry a
transport country passes to `gates.battery@1`: `DEFAULT_REGISTRY` plus a
`FunctionBinding` for each implemented gate below and two frozen pending
bindings. The implemented bindings read evidence the transport graph
produces, under the alias of the producing output, and take their thresholds
and declared surfaces only from the entry's `parameters` in `gates.json`.
No tunable threshold has a Python default: an entry that omits one fails
closed when it runs. A composer can call `validate_required_gate_parameters`
(the converse of the battery's `validate_gate_parameters`, which the kernel
runs) to refuse such a manifest before any gate runs. The shared comparisons
keep their fixed semantics: non-negative means at least zero, and
`aggregate_admin_gate` measures a miss relative to `max(|value|, 1)`.

| Gate | Evidence | Parameters (required in bold) |
| --- | --- | --- |
| `per_family_fit` | `diagnostics` | **`within`**, **`min_family_share`**, **`hard_within`**, **`min_hard_family_share`**, **`min_family_size`**, `families` |
| `aggregate_admin` | `surface`, `diagnostics` | **`default_rtol`**, `families`, `geography_levels` |
| `calibration_reference_coverage` | `surface`, `problem` | none |
| `target_profile_coverage` | `problem` | **`required_families`**, `reviewed_exclusions` |
| `nonnegative_columns` | frame | **`columns`**, `reviewed_exclusions` |
| `exported_nonzero` | frame | `exemptions` |
| `formula_owned_export` | frame | **`formula_owned_columns`** |
| `weight_ess` | frame | **`minimum_ess_fraction`** |
| `weight_ratio` | frame | **`maximum_max_to_median_ratio`** |

- `per_family_fit` groups the diagnostics rows by registry family; `families`
  restricts the entry to those families (so one family can carry a tighter
  entry) and each declared family must have a target. `hard_within` must be
  a number: the shared gate's report-only `null` belongs in a `diagnostic`
  entry.
- `aggregate_admin` checks each compiled target of the surface against the
  diagnostics' final estimate, sign first. Every anchor uses `default_rtol`
  from `gates.json`; a surface's `TargetSpec.tolerance` cannot change the gate
  threshold. The diagnostics must record that surface (`build.surface_sha256`).
- `calibration_reference_coverage` passes iff at least one reference is
  activated, the activated references, the resolved targets and the matrix
  rows are one set without repeats or skipped targets, and the problem was
  compiled from that surface.
- `target_profile_coverage` requires at least one matrix row in each declared
  family.
- The frame gates read the node's rebuilt population. Each weighted entity's
  typed weights stand in for its `{entity}_weight` column, which the export
  writes from them; entity ids and the memberships of the frame's own groups
  are not measured as layers. The gates check only the columns the node
  slices, so a gate node that checks the export must slice what
  `export.prepare@1` slices; `exported_nonzero` and `formula_owned_export`
  list the columns they read in their details.
- `nonnegative_columns` also fails a non-finite value in a declared column
  (the shared gate skips them): `-inf` is negative, and a missing value
  cannot be certified non-negative.
- Reviewed exclusions must carry non-empty string reasons before the
  bindings forward them to the shared comparison.
- `weight_ess` and `weight_ratio` read the frame's one weighted entity. The
  ESS fraction divides the Kish ESS by every record; the ratio divides the
  maximum by the median positive weight. For a nonzero vector, both
  comparisons use weights normalized by their maximum, retaining the
  original record count and positive-weight mask. Invalid computed
  concentration summaries fail closed. The differential tests compare
  concentration with the UK gates on normalized weights, preserving original
  positive support, and compare metadata in original units on the original
  weights. An unrepresentable diagnostic `total_weight` is reported as `null`;
  it does not change either concentration comparison.

Two additional frozen bindings remain pending: `support` requires the
`donor_support_bounds` artifact for realized donor-support evidence, and
`release_input_coverage` requires `donor_artifact_receipt` and
`axiom_input_closure` for donor authentication and per-module Axiom input
closure. Neither accepts gate parameters. Missing named producer artifacts
resolve to registered `evidence_absent` outcomes. Their
evaluators fail even when artifacts are supplied, until the evidence
checks are implemented. New Zealand keeps both entries applicable, with
empty parameters and `evidence_absent_blocks: true`: absent evidence or a
pending evaluator blocks the build, makes the report non-shippable, and
prevents graph artifact production, even if every other gate passes.

`macro_realism` still has no transport binding and remains `not_applicable`
until destination national-accounts metrics and reviewed bands are packaged.
`weights_audit` keeps its `DEFAULT_REGISTRY` binding, but no transport
kernel emits its `fit_weight_records` evidence yet, so it resolves to a named
`evidence_absent` gap that blocks New Zealand builds.

## Export

`export.prepare@1` describes the exact entity tables to write: the declared
slices of each entity plus the weight column, with columns, dtypes, row count
and a content hash per table, and the period. It refuses when a gate report
does not permit the artifact, when the reports are missing or inconsistent
(see "Gates"), and when the caller's `bindings` already carry a `gates` key
(the kernel records the reports itself). It also refuses an empty table and any column dtype outside
the round-trip allowlist (`bool`, `int32`, `int64`, `float32`, `float64`,
`string`). Nullable `Int64` does not write, and nullable `boolean` comes back
as `bool` without missing values and as `object` with them; the allowlist is
conservative and also refuses dtypes, such as `int8`, that would round-trip.

The dtype allowlist does not guarantee that every value or column name can
round-trip: a `string` cell containing the literal text `"nan"` reads back as
missing, and a trailing NUL is stripped. A column named `index` is accepted
by preparation but makes the HDF5 writer raise `IndexError`. Preparation
does not reject these cases up front; materialization or readback refuses
them, so the package cannot accept an export that differs from its
descriptor.

Writing is an outer step, `materialize_export(frame, descriptor, path)`, run
on every build. It refuses a population whose content differs from the
descriptor and writes through `AxiomEntityTableDataset`. The HDF5 writer
records object times at one-second resolution, so two writes of one
descriptor can differ in bytes while their tables are identical; then
`export.readback@1` and `transport.package@1` re-key, while everything up to
`export.prepare@1` is a store hit.

The written file is declared as a source under the codec
`axiom-entity-table-h5-v1`. The executor only checks that the codec is
registered; the codec's own HDF5-signature check runs only for callers of
`load_source_bytes`. `export.readback@1` reads the file itself and compares
each table, the table set and the period with the descriptor. A file that
cannot be bound or read fails the node with a message that names the source
but not its path, and leaves the package `unreached`. A readable file that
differs fails the outcome, and `transport.package@1` refuses it. The package
binds every input artifact's key, producer key and SHA-256, the gate
outcomes, manifest hash and posture (`release_candidate`, `synthetic_smoke`),
and the readback; it refuses a gate report that does not permit the artifact
or that is not one export was prepared under, and always records
`release_authorized: false`.
