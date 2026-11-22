# UK Firm Support Plan

Microcosm should support firms as a first-class population while keeping the
same boundary that now applies to person and household calibration:

- Ledger is the only source of calibration target values and target contracts.
- policyengine-uk is the executable policy/rules surface for firm taxes: VAT,
  and later corporation tax, employer NICs, the apprenticeship levy and
  business rates, through its `Firm` entity (PolicyEngine/policyengine-uk#2939).
  Axiom's UK VAT RuleSpec stays available as an optional cross-check, not as
  the rules home.
- Microcosm owns frame construction, model features, target activation, solver
  inputs, diagnostics, and published population artifacts.

The firm microsim paper remains a reference implementation and validation
baseline. It should not become the long-term source of target values, policy
logic, or release artifacts.

## Status

Tracking issue: microcosm#1705. The first step brings
`uk_runtime/firm_generation.py` up to the paper (sector-conditional employment,
two-universe calibration with per-band registration propensity, stratified
thinning with base weights, `in_frame` and `vat_scope` flags), moves it onto the
public solver (microcosm#636) and fixes the input-ratio imputation
(microcosm#258). It adds the rules seam (`uk_runtime/firm_rules.py`: a
`UKFirmRules` protocol whose outputs are keyed by `firm_id`) and a firm table
export with policyengine-uk's firm column names (`uk_runtime/firm_table.py`).
The existing Axiom VAT evaluator is wrapped behind the seam.
`uk_runtime/firm_policyengine_uk_rules.py` adds `PolicyEngineUKFirmRules`, which
runs policyengine-uk's firm entity on the candidate table and reads
`firm_vat_liability_if_registered`. It is tested engine-free with a fake
simulation; the released policyengine-uk has no firm entity yet, so its only
real-engine check is local, against PolicyEngine/policyengine-uk#2940.
`tools/build_uk_firms.py` (`uk_runtime/firm_build.py`) runs a local UK firm
build: generation, attributes and export to `uk_firms.parquet` with a JSON
receipt of inputs, configuration, accuracy and rules provenance. It has no
publish or staging path. A parity test on the
paper's committed 2024-25 tables (`test_uk_firm_paper_parity.py`) checks the
generator against golden numbers from paper runs, and
`tools/uk_firm_paper_parity.py` prints the full comparison locally.

Region and legal status are drawn after calibration from ONS UK Business 2025
Tables 8, 12 and 13 (`uk_runtime/firm_attributes.py`; legal status is
conditioned on the turnover band through Table 13 and on the employment band
through Table 12), with Table 15 held out, and
feed policyengine-uk's `firm_region`, `firm_in_england`, `firm_legal_form` and
`firm_is_public_body`. `firm_table.UK_FIRM_POLICYENGINE_UK_INPUT_STATUS` records
which of policyengine-uk's 61 firm inputs (at `57a99a67a`) microcosm sets.
`firm_paybill` is an opt-in assumed input (headcount × ONS ASHE section mean
pay, `uk_runtime/firm_paybill.py`); the rest (the pay distribution within it,
profit, capital expenditure, R&D, rateable value and several flags) wait for
sources. An engine-free test checks the export against a committed snapshot of
policyengine-uk's firm variables. Value-free target candidates for the next
Ledger sources are declared in `uk_runtime/firm_target_candidates.py`; they are
not yet in `uk/uk_firms_targets.json` or the solve.

## External Prerequisites

The first integration should depend on these upstream artifacts:

- Ledger source packages:
  - `ons-uk-business-firm-targets-2025`
  - `hmrc-vat-firm-targets-2024-25`
- Ledger target profile:
  - `uk_firms`
- policyengine-uk firm variables (rules home), once the `Firm` entity merges:
  - `firm_vat_liability_if_registered`, `firm_vat`, `firm_vat_registered`
  - later: corporation tax, employer NICs, apprenticeship levy, business rates
- Axiom UK RuleSpec (optional cross-check only):
  - `uk:policies/govuk/vat#firm_must_register_for_vat`
  - `uk:policies/govuk/vat#firm_vat_registered`
  - `uk:policies/govuk/vat#firm_treated_as_vat_registered`
  - `uk:policies/govuk/vat#standard_rate_output_vat`
  - `uk:policies/govuk/vat#recoverable_input_vat`
  - `uk:policies/govuk/vat#net_vat_liability`

Microcosm should reject a UK firm build if those Ledger packages/profile are not
available. A build needs one rules engine (policyengine-uk once its firm entity is
released; Axiom is accepted only as a cross-check or a stand-in until then). It should not fall back to repo-local CSV targets.

## Phase 1: Frame Contract

Add a `firm` entity table contract in `microcosm-frame`:

- Required identifiers: `firm_id`, `source_firm_id`, `source_firm_key`.
- Required weight column: `firm_weight`.
- Core columns: annual turnover, employment, sector code, VAT registration
  status, standard-rated taxable supplies, input VAT, and net VAT liability.
- Optional link columns for future mixed household-firm populations:
  `owner_person_id`, `owner_household_id`, and ownership share.

Initial UK firm builds can be a single-entity frame. Household/person ownership
links should be a later extension unless a downstream use case needs joint
personal and business incidence.

Acceptance gates:

- Stable row ordering by `firm_id`.
- Strict weight schema validation for `firm_weight`.
- No target-value columns embedded in the frame contract.
- Compatibility with existing `Frame` accounting helpers.

## Phase 2: Ledger Target Activation

Add a `microcosm-build` target-profile resolver for `uk_firms`:

- Read Ledger consumer facts and the `uk_firms` target profile.
- Select only facts matching the profile's source names, source measure IDs,
  record-set IDs, geography level, and groupby dimensions.
- Expand each profile row into solver targets by the selected Ledger fact
  constraints, such as turnover and employment bands.
- Preserve Ledger fact keys in diagnostics and release traces.

Microcosm may decide whether a selected Ledger fact is active, warning-only, or
unsupported, but it must not mutate the target value. Period alignment should
follow the profile's declared `latest_not_after_build_base_period` policy.

Acceptance gates:

- A build fails if an active target lacks a Ledger fact key.
- A build fails if an active target value came from outside Ledger.
- Diagnostics include source package ID, source record ID, aggregate fact key,
  period, geography, and target activation status.

## Phase 3: Rules-engine VAT Metrics

Firm metrics come through the `UKFirmRules` seam. The production adapter is
`PolicyEngineUKFirmRules` (M4; VAT only so far, verified locally against the
unreleased policyengine-uk firm entity), which builds a
policyengine-uk simulation from the firm table (policyengine-uk column names)
and reads its firm variables. The Axiom adapter below stays as an optional
cross-check, and builds will record any difference between the two.

For the Axiom cross-check:

- Convert firm entity rows into the Axiom input variables used by
  `uk:policies/govuk/vat`.
- Treat modeled or source-observed active VAT registration as the
  `firm_has_active_vat_registration` Axiom input and preserve a reconciliation
  diagnostic against `firm_must_register_for_vat`.
- Use `firm_vat_registered` for HMRC registered-trader target filters and
  `firm_treated_as_vat_registered` for liability calculations.
- Compute `standard_rate_output_vat`, `recoverable_input_vat`, and
  `net_vat_liability`.
- Return metric arrays keyed to `firm_id`, not positional row order alone.

The initial implementation can accept `standard_rated_taxable_supplies_value` and
`input_vat_on_business_purchases` as modeled or imputed columns. Classification
of reduced-rate, zero-rated, and exempt supplies should be an explicit future
rules/data task, not hidden in a Microcosm adapter.

Acceptance gates:

- Metric outputs are deterministic for a fixed input frame.
- `net_vat_liability` supports negative values for repayment positions.
- The rules engine and its variable or rule IDs appear in metric provenance.
- No handwritten TypeScript/Python VAT calculation is used as a substitute for
  the rules engine.

## Phase 4: UK Firm Base Builder

Add a `microcosm-build` UK firm builder that constructs an initial national firm
frame before calibration:

- Seed firms from ONS turnover and employment structures.
- Add sector and taxable-supply features from available source distributions.
- Impute inputs and input VAT with a documented stochastic model.
- Mark all modeled/imputed columns with provenance.

The first builder should be national only. Local-area or sector-rich firm
allocation can follow after the national target profile and VAT metrics are
stable.

Acceptance gates:

- The uncalibrated base can be regenerated deterministically from a seed.
- Source and imputation provenance appear in build traces.
- The builder emits no hard-coded calibration target values.

## Phase 5: Calibration and Release

Use `microcosm-calibrate` to solve firm weights against activated Ledger targets:

- ONS enterprise counts by annual-turnover band.
- ONS enterprise counts by employment band.
- HMRC VAT-registered trader counts by turnover band.
- HMRC net VAT liability by turnover band.

The first release artifact should be separate from household releases, for
example `policyengine/populace-uk-firms`, until ownership links and joint
person-household-firm calibration are implemented.

Acceptance gates:

- Solver diagnostics report absolute and relative error by Ledger fact key.
- Release trace contains the Ledger target profile ID and the rules engine
  (policyengine-uk version and variables, or Axiom rule IDs for a cross-check)
  used for VAT metrics.
- Static validation compares against the firm microsim paper's published target
  groups, but those paper CSVs are validation inputs only.
- CI covers the target resolver, rules adapters, frame schema, and a small
  synthetic firm calibration fixture.

## Open Design Questions

- Whether `firm` should remain a standalone entity or become an owned entity
  linked to persons/households in the default UK frame.
- How to reconcile source-observed VAT registration with the rules engine's
  mandatory and voluntary registration rules.
- How to represent sector mappings between SIC, HMRC trade sectors, and future
  supply classifications in policyengine-uk.
- Whether future local firm distributions should be calibrated from Ledger
  targets, allocated from national targets, or both with explicit source
  hierarchy rules.
