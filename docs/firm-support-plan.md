# UK Firm Support Plan

Populace should support firms as a first-class population while keeping the
same boundary that now applies to person and household calibration:

- Ledger is the only source of calibration target values and target contracts.
- Axiom is the executable policy/rules surface for VAT liabilities.
- Populace owns frame construction, model features, target activation, solver
  inputs, diagnostics, and published population artifacts.

The firm microsim paper remains a reference implementation and validation
baseline. It should not become the long-term source of target values, policy
logic, or release artifacts.

## External Prerequisites

The first integration should depend on these upstream artifacts:

- Ledger source packages:
  - `ons-uk-business-firm-targets-2025`
  - `hmrc-vat-firm-targets-2024-25`
- Ledger target profile:
  - `uk_firms`
- Axiom UK RuleSpec:
  - `uk:policies/govuk/vat#firm_must_register_for_vat`
  - `uk:policies/govuk/vat#firm_vat_registered`
  - `uk:policies/govuk/vat#firm_treated_as_vat_registered`
  - `uk:policies/govuk/vat#standard_rate_output_vat`
  - `uk:policies/govuk/vat#recoverable_input_vat`
  - `uk:policies/govuk/vat#net_vat_liability`

Populace should reject a UK firm build if those Ledger packages/profile are not
available. It should not fall back to repo-local CSV targets.

## Phase 1: Frame Contract

Add a `firm` entity table contract in `populace-frame`:

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

Add a `populace-build` target-profile resolver for `uk_firms`:

- Read Ledger consumer facts and the `uk_firms` target profile.
- Select only facts matching the profile's source names, source measure IDs,
  record-set IDs, geography level, and groupby dimensions.
- Expand each profile row into solver targets by the selected Ledger fact
  constraints, such as turnover and employment bands.
- Preserve Ledger fact keys in diagnostics and release traces.

Populace may decide whether a selected Ledger fact is active, warning-only, or
unsupported, but it must not mutate the target value. Period alignment should
follow the profile's declared `latest_not_after_build_base_period` policy.

Acceptance gates:

- A build fails if an active target lacks a Ledger fact key.
- A build fails if an active target value came from outside Ledger.
- Diagnostics include source package ID, source record ID, aggregate fact key,
  period, geography, and target activation status.

## Phase 3: Axiom VAT Metrics

Add an Axiom-backed `RulesEngine` adapter path for firm metrics:

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
rules/data task, not hidden in a Populace adapter.

Acceptance gates:

- Metric outputs are deterministic for a fixed input frame.
- `net_vat_liability` supports negative values for repayment positions.
- Axiom rule IDs appear in metric provenance.
- No handwritten TypeScript/Python VAT calculation is used as a substitute for
  Axiom.

## Phase 4: UK Firm Base Builder

Add a `populace-build` UK firm builder that constructs an initial national firm
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

Use `populace-calibrate` to solve firm weights against activated Ledger targets:

- ONS enterprise counts by annual-turnover band.
- ONS enterprise counts by employment band.
- HMRC VAT-registered trader counts by turnover band.
- HMRC net VAT liability by turnover band.

The first release artifact should be separate from household releases, for
example `policyengine/populace-uk-firms`, until ownership links and joint
person-household-firm calibration are implemented.

Acceptance gates:

- Solver diagnostics report absolute and relative error by Ledger fact key.
- Release trace contains the Ledger target profile ID and Axiom RuleSpec rule
  IDs used for VAT metrics.
- Static validation compares against the firm microsim paper's published target
  groups, but those paper CSVs are validation inputs only.
- CI covers the target resolver, Axiom metric adapter, frame schema, and a small
  synthetic firm calibration fixture.

## Open Design Questions

- Whether `firm` should remain a standalone entity or become an owned entity
  linked to persons/households in the default UK frame.
- How to reconcile source-observed VAT registration with Axiom's mandatory and
  voluntary registration rules.
- How to represent sector mappings between SIC, HMRC trade sectors, and future
  Axiom supply classifications.
- Whether future local firm distributions should be calibrated from Ledger
  targets, allocated from national targets, or both with explicit source
  hierarchy rules.
