# The gate-battery contract (microcosm#611)

What a country must declare, and what its build must supply, to run the
country-agnostic gate battery. This is the consumer contract for
`microcosm.build.gate_battery`; the executor's own invariants live in that
module's docstring and are restated here only where a country author needs
them. Reading this document alone, you should be able to list every input a
new country needs — that list is the [onboarding
checklist](#the-onboarding-checklist-every-input-a-new-country-supplies),
and its completeness is the document's acceptance test.

The battery exists so that gate *policy* is data and gate *execution* is
shared. Countries differ in which gates they select, at which build phases,
with which thresholds, against which references — never in how gates are
batched, reported, or enforced. Three disciplines carry that split, lifted
from the executor's contract prose:

- **Batched within a phase, fail-closed at the phase boundary.** Every gate
  bound to a phase evaluates — a raising evaluator becomes a failed result,
  never a crash that masks the remaining gates — and the build decides at
  the boundary, not per gate.
- **Write-then-block.** The full report, including the block itself, is on
  disk before `enforce` raises. A blocked build always leaves evidence.
- **Every declared gate has exactly one outcome.** Nothing is omitted:
  evidence that does not exist is a named gap (`evidence_absent`), a
  reviewed non-selection is a receipt (`not_applicable`), and entries after
  a blocked phase are `unreached`. Silence is not in the vocabulary.

## Anatomy of `gates.json`

A country's gate policy is one JSON resource, `<cc>/gates.json`, inside its
spec-only country package. It must be listed in
`<cc>/country_package.json` `resources` — `load_country_spec` refuses
undeclared files on disk and missing declared ones — and it is parsed and
validated by canonical filename into `CountrySpec.gates`.

Header, validated by `country_spec._require_header`:

| key | meaning |
| --- | --- |
| `country` | Must equal the package directory name. |
| `version` | Integer ≥ 1; the country's own manifest revision counter. |
| `policy` | Prose stating the selection's intent — why these gates, what is deliberately not selected, what is pending. |
| `phases` | The country's phase order, first to last, drawn from the shared vocabulary `{preflight, assembled, transferred, simulated, terminal}`. Order is country data, never a global constant; `unreached` is only computable because the report knows this order. |

Each entry in `gates`:

| key | meaning |
| --- | --- |
| `id` | Unique, country-flavored (`uk_weight_ratio`). One gate function may be selected more than once under different ids and parameters. |
| `gate` | Country-neutral, from `country_spec.ALLOWED_GATE_FUNCTIONS` (`weight_ratio`, `tail_concentration`, ...). The executor fails a gate closed if the evaluated result's name disagrees with this — one gate cannot impersonate another. |
| `phase` | A member of the manifest's declared `phases`. |
| `criticality` | `release_blocking` or `diagnostic`. At least one entry must be release-blocking — a country whose every gate is diagnostic has no release contract. |
| `parameters` | Declarative gate inputs: tolerances, surfaces, exclusion registers, reference pins. Pure data, recursively frozen at load (mappings become read-only proxies, lists become tuples). Thresholds live *here*, covered by the policy hash — never in runner code. |
| `not_applicable` | Reviewed reason this gate deliberately does not run for this country yet. Mutually exclusive with a non-empty `parameters`; the entry appears in every report as `not_applicable` and never evaluates. |
| `notes` | Free-text rationale and provenance. |

Entry validation is closed-world: an unknown key is refused, not ignored,
because a silently dropped key (a typo'd `parameters`) would run the gate
on defaults while the declared intent vanished from the policy hash — an
unattested threshold.

Country packages are spec-only. The package tests reject any JSON key that
looks executable (tokens like `function`, `module`, `handler`, at any
nesting depth including inside `parameters`) and any string value shaped
like a dotted entry point. Name parameters accordingly.

Three digests pin the declaration, all carried in every report:

- `policy_sha256` — over each entry's `id`/`gate`/`phase`/`criticality`/
  `parameters`/`not_applicable`, sorted by id. **Excludes `notes`**: editing
  rationale does not move gate policy.
- `gates_manifest_sha256` — over the full manifest including `notes` and
  the phase order.
- `spec_fingerprint` — the composition fingerprint of the whole country
  package. Adding or editing `gates.json` moves `CountrySpec.fingerprint`;
  anything pinning it must be regenerated in the same change.

## The evidence contract

Gates read evidence through an `EvidenceContext`, built by the country's
build tool at each phase boundary:

- `frame` — the phase's `microcosm.frame.Frame` carrier, when the phase has
  one. Preflight phases may run frameless.
- `artifacts` — named runtime evidence the build supplies: objects not
  derivable from the frame. A gate whose declared artifact keys are absent
  resolves to `evidence_absent` with the missing keys named — before
  evaluation, never as a crash or a pass.
- `spec` — the loaded country spec; reserved for bindings that derive
  surfaces from declared manifests (no binding reads it yet).

A **binding** connects an allowlisted gate name to evidence extraction and
evaluation: it declares `required_artifacts` and `requires_frame` (both may
depend on the entry's parameters), runs the gate, and optionally produces
an evidence payload whose canonical sha256 lands in the report
(`evidence_sha256` per entry; unattestable evidence downgrades a pass to a
failure). The shared `DEFAULT_REGISTRY` binds gates whose evidence already
travels as plain data; a country passes `registry=` to add or override
bindings. A declared gate with no binding in the active registry is a named
`evidence_absent` gap — an incomplete registry cannot manufacture a pass.

Canonical artifact keys today:

| key | supplied by | consumed by |
| --- | --- | --- |
| `fit_weight_records` | any build with production fits | `weights_audit` (shared binding) |
| `candidate_input_mass_totals`, `reference_input_mass_totals` | data-only builds | `input_mass_parity` (shared binding; the UK overrides it) |
| `tail_concentration_values`, `tail_concentration_weights` | data-only builds | `tail_concentration` (shared binding; the UK overrides it) |
| `coverage_engine`, `coverage_manifest` (optional override) | UK national build | `release_input_coverage`, both phases |
| `build_stage_names` | UK national build preflight | `source_coverage` |
| `parity_evidence` | UK national build | `export_surface`, `target_surface`, `target_fit` |
| `input_mass_reference`, `input_mass_policy` | UK national build (runtime-armed pending microcosm#630) | `input_mass_parity` (UK override) |
| `qrf_tail_policy` | UK national build (runtime-armed pending microcosm#630) | `tail_concentration` (UK override) |

The UK registry (`microcosm.build.uk_runtime.battery_bindings.UK_GATE_REGISTRY`)
is the reference implementation of a country registry: it adapts the Frame
onto the legacy duck-attr evidence surface, constructs reviewed policy
objects from frozen declared parameters, passes undeclared parameters
through so an unknown key fails closed, and re-mints exactly two
legacy result names (`uk_release_input_coverage`, `qrf_tail_concentration`)
onto the shared vocabulary. Its differential test
(`tests/test_uk_battery_bindings.py`) pins verdict-for-verdict equality
with the legacy UK battery — the template for proving any future
migration behaviour-preserving.

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

- Only `release_blocking` entries can block; `diagnostic` entries never do.
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
is written and marked (`MARKS_ARTIFACT` — intermediates like the US pool:
a pool with a failed agreement gate is worth keeping; it just is not
ready). Once a phase blocks, later phases refuse to run and their entries
stay `unreached` — which is the truth.

The report is signed (HMAC-SHA256) with a per-country key; the signature
is valid over failed reports — a blocked build's evidence is still
attested. A missing key is captured as `signing_error` in the attestation
and forces `shippable: false` rather than crashing the build.

## The worked example

`packages/microcosm-build/tests/test_gate_battery_contract_example.py` is
this document's runnable companion: a minimal greenfield country (`xx`)
declares three terminal gates and runs them on the shared registry alone.
CI keeps it honest; change it and this section together.

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
and the schema-4 body with its schema-6 attestation — is on disk.

The other two statuses are exercised in
`tests/test_gate_battery.py::TestOutcomeTaxonomy`, which reaches all five
in one report.

## Three countries, one executor

| | BE | UK | US pool (planned, microcosm#611 inc 2) |
| --- | --- | --- | --- |
| phases | `terminal` | `preflight`, `terminal` | `assembled`, `transferred`, `simulated`, `terminal` |
| entries | 9 | 13 (2 preflight + 11 terminal) | the stacked pipeline's terminal pair + batched `weights_audit` |
| incumbent posture | none — incumbent-comparison gates deliberately not selected; external oracles replace self-parity | full — export/target parity against the pinned enhanced-FRS incumbent | self-consistency across spines |
| thresholds | declared in `parameters` | declared in `parameters` at the exact certified June values; input-mass and QRF-tail tolerances runtime-armed pending the microcosm#630 adjudication, landing as parameter edits | to be declared |
| blocking mode | `BLOCKS_ARTIFACT` | `BLOCKS_ARTIFACT` | `MARKS_ARTIFACT` |
| criticality mix | 8 blocking + 1 diagnostic (commune-grain fit is diagnostic until the spine supports it) | all blocking (legacy behaviour: every evaluated failure raises) | to be declared |

The differences are entirely in the two country inputs — the spec file and
the registry — which is the point.

## The reference rule (microcosm#327, generalized)

The US export input-mass gate was once wired against the wrong reference
(a calibrated export compared to a raw base), flagging correct
calibration-driven drift as failure. The lesson, promoted to a contract
rule: **a parity gate's reference, and its reviewed-exclusion register,
are declared per-country inputs — never implicit module state.** A new
country must pick its reference on purpose and cannot silently inherit a
mis-referenced comparison.

The UK input-mass entry is the worked case. Its `parameters` declare the
frozen reference by identity and canonical digest —

- `reference_sha256` — the reviewed digest of the frozen 131-column
  reference totals (the totals themselves stay uncommitted under the UKDS
  licence; the digest binds them without disclosing them);
- `reference_identity` — filename, revision, artifact sha256, vintage of
  the pinned enhanced-FRS artifact the totals were measured from;
- `reviewed_exclusions_resource` — the committed register resource in the
  UK package, named rather than duplicated

— and the UK binding holds the runtime-supplied reference to the declared
pin, failing closed on drift. The export-surface entry declares its
reviewed comparison registers the same way (the allowed-extra column list
and the reviewed reference-side exclusion), with the two hard-required,
never-waivable columns remaining a code-level guard noted in the entry.

## The onboarding checklist: every input a new country supplies

1. **The spec**: `<cc>/gates.json` (header, phase order, entries per the
   anatomy above), listed in `<cc>/country_package.json` `resources`.
   Loaded via `load_country_spec("<cc>").gates`; nothing else is needed
   for the spec to reach the battery.
2. **A registry**, if `DEFAULT_REGISTRY` does not already bind every
   selected gate: bindings implementing the `GateBinding` protocol, passed
   as `GateBatteryRun(..., registry=...)`. Unbound declared gates are named
   gaps, so a country can land its spec before its bindings.
3. **Runtime artifacts**, under the canonical keys each selected gate's
   binding declares (table above). Missing artifacts are named gaps that
   block release candidates only.
4. **A `Frame`** at every phase whose gates read one; preflight-style
   phases may run frameless.
5. **Run identity**: `release_id`, `release_candidate`, and a
   `report_path` for the atomically-written report.
6. **The signing key**: `gate_signing_key_env("<cc>")` names the variable —
   `MICROCOSM_<CC>_TERMINAL_GATE_SIGNING_KEY`, base64, exactly 32 decoded
   bytes.

   > **Warning — UK transition state.** The live UK release still signs and
   > verifies with `POPULACE_UK_TERMINAL_GATE_SIGNING_KEY` (the legacy
   > battery in `uk_runtime/terminal_gates.py` and the publication-time
   > verifier in `microcosm-data`'s `contract.py`). The battery mints the
   > `MICROCOSM_` name. Until the CI secret is rotated at the contract
   > flip, a production UK battery run would read a variable nobody sets
   > and record `signing_error`. The rotation is deliberate and owner-held;
   > do not paper over it with a fallback.

7. **Regenerated pins** wherever the country's spec shape is asserted:
   adding or editing `gates.json` moves `CountrySpec.fingerprint`,
   `gates_manifest_sha256`, and (unless only `notes` changed)
   `policy_sha256`. Anything that pins them — golden spec files, resource
   tuples in tests, release contracts — is a reviewed diff in the same
   change.

## Out of scope here, chartered elsewhere

- **The UK orchestration swap** (microcosm#611, A2): `national_build.py`
  still runs the legacy `uk_terminal_gate_report`; moving it onto
  `GateBatteryRun` with `UK_GATE_REGISTRY` retires the legacy evidence
  adapter and the duplicated executor, and changes unarmed-gate semantics
  from omission to recorded `evidence_absent` — a delta the differential
  test already states positively.
- **microcosm#630 adjudication**: the reviewed `source_year` exclusion and
  the receipted `weight_ratio` re-baseline land as edits to `uk/gates.json`
  parameters and the exclusion registers — the first live proof that
  thresholds-as-declared-data works.
- **A UK delivered-take-up gate**: pending a take-up name in
  `ALLOWED_GATE_FUNCTIONS` (owner-held vocabulary change); recorded as a
  named gap in the UK manifest's `policy` prose until then.
- **US pool onboarding** (microcosm#611 inc 2): extends this document's
  side-by-side with the multi-phase, marks-artifact worked example.
