# The gate-battery contract

How a country runs the country-agnostic gate battery
(`microcosm.build.gate_battery`): what it declares, what its build
supplies, and what comes back. The audience is anyone adding or
maintaining a country's release gates. After reading this document alone,
you should be able to list every input a country needs — that list is the
[onboarding checklist](#the-onboarding-checklist-every-input-a-country-supplies)
at the end.

The battery exists so that gate *policy* is data and gate *execution* is
shared. Countries differ in which gates they select, at which build
phases, with which thresholds, against which references — never in how
gates are batched, reported, or enforced. Three disciplines carry that
split:

- **Batched within a phase, fail-closed at the phase boundary.** Every
  gate bound to a phase evaluates — a raising evaluator becomes a failed
  result, never a crash that masks the remaining gates — and the build
  decides at the boundary, not per gate.
- **Write-then-block.** The full report, including the block itself, is on
  disk before `enforce` raises. A blocked build always leaves evidence.
- **Every declared gate has exactly one outcome.** Nothing is omitted:
  evidence that does not exist is a named gap (`evidence_absent`), a
  reviewed non-selection is a receipt (`not_applicable`), and entries
  after a blocked phase are `unreached`. Silence is not in the vocabulary.

## Declaring gates: `gates.json`

A country's gate policy is one JSON resource, `<cc>/gates.json`, inside
its spec-only country package. It must be listed in
`<cc>/country_package.json` `resources` — `load_country_spec` refuses
undeclared files on disk and missing declared ones — and it is parsed and
validated by canonical filename into `CountrySpec.gates`.

The header:

| key | meaning |
| --- | --- |
| `country` | Must equal the package directory name. |
| `version` | Integer ≥ 1; the country's own manifest revision counter. |
| `policy` | Prose stating the selection's intent — why these gates, and what is deliberately not selected. |
| `phases` | The country's phase order, first to last, drawn from the shared vocabulary `{preflight, assembled, transferred, simulated, terminal}`. Order is country data, never a global constant; `unreached` is only computable because the report knows this order. |

Each entry in `gates`:

| key | meaning |
| --- | --- |
| `id` | Unique, country-flavored (`uk_weight_ratio`). One gate function may be selected more than once under different ids and parameters. |
| `gate` | Country-neutral, from `country_spec.ALLOWED_GATE_FUNCTIONS` (`weight_ratio`, `tail_concentration`, ...). The executor fails a gate closed if the evaluated result's name disagrees with this — one gate cannot impersonate another. |
| `phase` | A member of the manifest's declared `phases`. |
| `criticality` | `release_blocking` or `diagnostic`. At least one entry must be release-blocking — a country whose every gate is diagnostic has no release contract. |
| `parameters` | Declarative gate inputs: tolerances, surfaces, exclusion registers, reference pins. Pure data, recursively frozen at load (mappings become read-only proxies, lists become tuples). Thresholds live *here*, covered by the policy hash — never in runner code. |
| `not_applicable` | Reviewed reason this gate deliberately does not run for this country. Mutually exclusive with a non-empty `parameters`; the entry appears in every report as `not_applicable` and never evaluates. |
| `notes` | Free-text rationale and provenance. |

Entry validation is closed-world: an unknown key is refused, not ignored,
because a silently dropped key (a typo'd `parameters`) would run the gate
on defaults while the declared intent vanished from the policy hash — an
unattested threshold.

The same rule holds one level down, inside `parameters`. Every binding
declares its parameter vocabulary (`parameter_keys` — the set of keys it
can route into the gate), and `validate_gate_parameters` checks every
declared entry against it when the battery is armed, before any gate
runs. A key outside the vocabulary is refused with the entry named: left
alone it would sit inside `policy_sha256` while governing nothing. A
binding that declares no vocabulary accepts no parameters.

Reviewed registers appear in `parameters` in one of two forms, by one
rule: a register that an existing runtime loader owns — with its own
schema, validation, and rot mechanics — is a separate package resource
named by a `*_resource` parameter; a plain reviewed list or mapping with
no loader of its own lives inline, where editing it moves the policy
hash directly.

Country packages are spec-only. The package tests reject any JSON key
that looks executable (tokens like `function`, `module`, `handler`, at
any nesting depth including inside `parameters`) and any string value
shaped like a dotted entry point. Name parameters accordingly.

Three digests pin the declaration, all carried in every report:

- `policy_sha256` — over each entry's `id`/`gate`/`phase`/`criticality`/
  `parameters`/`not_applicable`, sorted by id. **Excludes `notes`**:
  editing rationale does not move gate policy.
- `gates_manifest_sha256` — over the full manifest including `notes` and
  the phase order.
- `spec_fingerprint` — the composition fingerprint of the whole country
  package. Adding or editing `gates.json` moves `CountrySpec.fingerprint`;
  anything pinning it must be regenerated in the same change.

## Supplying evidence: `EvidenceContext` and bindings

Gates read evidence through an `EvidenceContext`, built by the country's
build tool at each phase boundary:

- `frame` — the phase's `microcosm.frame.Frame` carrier, when the phase
  has one. Preflight-style phases may run frameless.
- `artifacts` — named runtime evidence the build supplies: objects not
  derivable from the frame. A gate whose declared artifact keys are
  absent resolves to `evidence_absent` with the missing keys named —
  before evaluation, never as a crash or a pass.
- `spec` — the loaded country spec; reserved for bindings that derive
  surfaces from declared manifests.

A **binding** connects an allowlisted gate name to evidence extraction
and evaluation: it declares `required_artifacts` and `requires_frame`
(both may depend on the entry's parameters), runs the gate, and
optionally produces an evidence payload whose canonical sha256 lands in
the report (`evidence_sha256` per entry; unattestable evidence downgrades
a pass to a failure). The shared `DEFAULT_REGISTRY` binds gates whose
evidence already travels as plain data; a country passes `registry=` to
add or override bindings. A declared gate with no binding in the active
registry is a named `evidence_absent` gap — an incomplete registry cannot
manufacture a pass.

Canonical artifact keys:

| key | supplied by | consumed by |
| --- | --- | --- |
| `fit_weight_records` | any build with production fits | `weights_audit` (the UK override fails the audit when the record set is empty — a fit stage that emitted nothing is not vacuously passing) |
| `candidate_input_mass_totals`, `reference_input_mass_totals` | data-only builds | `input_mass_parity` (shared binding) |
| `tail_concentration_values`, `tail_concentration_weights` | data-only builds | `tail_concentration` (shared binding) |
| `coverage_engine`, `coverage_manifest` (optional override) | the UK national build | `release_input_coverage`, both phases |
| `build_stage_names` | the UK national build preflight | `source_coverage` |
| `parity_evidence` | the UK national build | `export_surface`, `target_surface`, `target_fit` |
| `input_mass_reference`, `input_mass_policy` | the UK national build | `input_mass_parity` (UK override) |
| `qrf_tail_policy` | the UK national build | `tail_concentration` (UK override) |

The UK registry
(`microcosm.build.uk_runtime.battery_bindings.UK_GATE_REGISTRY`) is the
reference implementation of a country registry. Its bindings adapt the
`Frame` onto the evidence surface the UK gate modules read, construct
reviewed policy objects from the frozen declared parameters, hold
runtime-supplied references to the spec-declared pins, declare the
parameter vocabulary each evaluator can route (so a stray key is refused
at arm time, not discovered mid-battery), and re-mint exactly
two UK-flavored result names onto the shared vocabulary. Every verdict
is computed by the existing UK gate functions — a binding adapts
evidence; it never re-implements a comparison. The differential test
(`tests/test_uk_battery_bindings.py`) holds the battery's verdicts equal
to the incumbent UK battery's, failure line for failure line, over
identical evidence.

## Outcomes and blocking

Five statuses, one per declared entry per report:

| status | meaning |
| --- | --- |
| `passed` | Evaluated, passed. |
| `failed` | Evaluated and failed — including evaluator crashes, wrong-name results, and unattestable evidence, all failed closed. |
| `evidence_absent` | Declared, but a required artifact, frame, or binding was missing; the gap is named. |
| `not_applicable` | Declared with a reviewed reason not to run. Reported even if its phase never ran. |
| `unreached` | Its phase never ran because an earlier phase blocked. |

`passed` and `not_applicable` are the only shippable statuses.

Blocking is a two-axis decision at each phase boundary:

- Only `release_blocking` entries can block; `diagnostic` entries never
  do.
- `failed` always blocks. `evidence_absent` blocks **release candidates
  only**: a dev build without, say, an incumbent parity snapshot gets an
  honest non-shippable report instead of a crash, while a release build
  cannot excuse missing evidence — a missing frozen reference is not a
  passing gate.

The battery run (`GateBatteryRun`) executes phases in declared order:
`run_phase` evaluates the batch and persists the full report atomically
*before* returning; `enforce` then either raises
`GateBatteryBlockedError` (`BLOCKS_ARTIFACT` — publications: nothing is
written downstream of a block) or returns the verdict while the artifact
is written and marked (`MARKS_ARTIFACT` — intermediates: an artifact with
a failed gate can be worth keeping; it just is not ready). Once a phase
blocks, later phases refuse to run and their entries stay `unreached` —
which is the truth.

The report is signed (HMAC-SHA256) with a per-country key; the signature
is valid over failed reports — a blocked build's evidence is still
attested. A missing key is captured as `signing_error` in the attestation
and forces `shippable: false` rather than crashing the build.

## The worked example

`packages/microcosm-build/tests/test_gate_battery_contract_example.py` is
this document's runnable companion: a minimal country (`xx`) declares
three terminal gates and runs them on the shared registry alone. CI keeps
it honest; change it and this section together.

The declaration inlines the `gates.json` shape: three entries —
`input_mass_parity` with a declared 10% tolerance, `tail_concentration`
with declared `top_k`/`max_top_share`/`min_nonzero_records`, and
`weights_audit` with no parameters. The build supplies totals and tail
columns under the canonical artifact keys and deliberately omits
`fit_weight_records`. One `run_phase("terminal", context)` resolves:

- `xx_input_mass_parity` → `passed` (5% drift inside the declared 10%);
- `xx_tail_concentration` → `failed` (one record carries ~98% of weighted
  mass against a declared 50% ceiling);
- `xx_weights_audit` → `evidence_absent` (the missing key is named).

The same phase report answers both blocking questions: as a dev build the
failure alone blocks; as a release candidate the evidence gap would block
too. `enforce(..., BLOCKS_ARTIFACT)` raises only after the report — with
`blocked_at_phase`, `shippable: false`, the policy and manifest hashes,
and the schema-versioned body and attestation — is on disk.

The other two statuses are exercised in
`tests/test_gate_battery.py::TestOutcomeTaxonomy`, which reaches all five
in one report.

## Two postures, one executor

The two country specs in the tree show how far the same executor
stretches on declaration alone:

| | BE | UK |
| --- | --- | --- |
| phases | `terminal` | `preflight`, `terminal` |
| entries | 9 | 13 |
| incumbent posture | none — incumbent-comparison gates deliberately not selected; external oracles replace self-parity | full — export/target parity against the pinned enhanced-FRS incumbent |
| criticality mix | 8 blocking + 1 diagnostic | all blocking |

The differences are entirely in the two country inputs — the spec file
and the registry — which is the point. The UK national build is the
executor's production consumer: it constructs one `GateBatteryRun` per
build and runs both phases under `BLOCKS_ARTIFACT` (preflight before the
frame loads, terminal immediately before the staging writer), while BE
remains spec-only.

## The reference rule

A parity gate is only as honest as its reference: comparing a calibrated
artifact against the wrong baseline flags correct, target-aligned drift
as failure. The contract therefore requires that **a parity gate's
reference, and its reviewed-exclusion registers, are declared per-country
inputs — never implicit module state.** A country picks its reference on
purpose and cannot silently inherit a mis-referenced comparison.

The UK input-mass entry is the worked case. Its `parameters` declare the
frozen reference by identity and canonical digest —

- `reference_sha256` — the reviewed digest of the frozen reference
  totals (the totals themselves stay uncommitted under the data licence;
  the digest binds them without disclosing them);
- `reference_identity` — filename, revision, artifact sha256, and vintage
  of the pinned incumbent artifact the totals were measured from;
- `reviewed_exclusions_resource` — the committed register resource in the
  UK package, named rather than duplicated

— and the UK binding holds the runtime-supplied reference to the declared
pin, failing closed on drift. The export-surface entry declares its
reviewed comparison registers the same way (the allowed-extra column list
and the reviewed reference-side exclusions), with the hard-required,
never-waivable columns remaining a code-level guard noted in the entry.

## The onboarding checklist: every input a country supplies

1. **The spec**: `<cc>/gates.json` (header, phase order, entries per the
   anatomy above), listed in `<cc>/country_package.json` `resources`.
   Loaded via `load_country_spec("<cc>").gates`; nothing else is needed
   for the spec to reach the battery.
2. **A registry**, if `DEFAULT_REGISTRY` does not already bind every
   selected gate: bindings implementing the `GateBinding` protocol,
   passed as `GateBatteryRun(..., registry=...)`. Unbound declared gates
   are named gaps, so a country can land its spec before its bindings.
3. **Runtime artifacts**, under the canonical keys each selected gate's
   binding declares (table above). Missing artifacts are named gaps that
   block release candidates only.
4. **A `Frame`** at every phase whose gates read one; preflight-style
   phases may run frameless.
5. **Run identity**: `release_id`, `release_candidate`, and a
   `report_path` for the atomically-written report.
6. **The signing key**: `gate_signing_key_env("<cc>")` names the
   variable — `MICROCOSM_<CC>_TERMINAL_GATE_SIGNING_KEY`, base64,
   exactly 32 decoded bytes.
7. **Regenerated pins** wherever the country's spec shape is asserted:
   adding or editing `gates.json` moves `CountrySpec.fingerprint`,
   `gates_manifest_sha256`, and (unless only `notes` changed)
   `policy_sha256`. Anything that pins them — golden spec files, resource
   tuples in tests, release contracts — is a reviewed diff in the same
   change.
