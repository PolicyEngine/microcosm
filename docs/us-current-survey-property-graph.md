# Current survey property-income graph

The opt-in `graph_current_survey_property.py` fragment adds 16 operations to the existing US financial graph. It puts qualified ASEC components and modeled ACS components on the complete initial clone pair. The country financial host retains the original source owners, verifies every graph observation and artifact, and issues the result. The fragment creates no separate issuance registry.

`PropertyIncomeOptions(scales=..., atol=..., rtol=..., n_estimators=...)` requires all four settings. The options object is frozen and has a canonical `to_bytes()` representation. No numeric modeling default is supplied. The first three components are nonnegative; broad property receipts can be signed. A negative or zero anchor can coexist with positive components and an offsetting negative broad-property component.

## Visible operations

| Operation | Meaning |
| --- | --- |
| `survey_property.source_projection` | Bind the actual qualified property composition, the existing financial predictor projection/matrix, original host artifacts, and geography validation when present. Emit a projection and exact donor/recipient matrices. |
| `survey_property.asec_eligible_donor` | Filter the original CREATE by the exact eligible ASEC person IDs. Preserve original household DESIGN weights, including their person mapping. |
| `survey_property.asec_donor_columns` | Materialize shared predictor columns, reported total, and four qualified components. |
| `survey_property.acs_eligible_recipient` | Filter original CREATE to ACS persons whose source anchor is known and in the adult reporting universe. |
| `survey_property.acs_recipient_columns` | Materialize shared predictors and adjusted INTP as the reported-total feature. |
| `survey_property.fit.000`–`.003` | Fit the existing DESIGN-weighted chained QRF over the four components. |
| `survey_property.apply.000`–`.003` | Draw once per eligible original ACS person, preserving ordered prior-component conditioning. |
| `survey_property.draws` | Retain raw draws and their complete model/application history. |
| `survey_property.reconcile` | Apply the shared signed-total reconciliation rule with the explicit scales and tolerances. |
| `survey_property.attach` | Join original source identities to both initial clones and retain all existing rows, entities, weights, geography, metadata, and legacy financial leaves. |

Both model branches leave CREATE before importance allocation. Their FILTER operations return person keep masks, preserving actual raw source columns. Shared predictor materialization, including the existing analytical treatment of ACS earnings NIU, happens through declared column operations. The FILTER does not substitute the separately prepared feature frame for CREATE.

The projection binds complete descriptive physical seals, including unknown masks and excluded rows, not just the finite fitting matrices. It carries the donor exclusion summary and recipient exclusion counts. No excluded person silently becomes a zero-valued donor or recipient. Empty eligible donor or recipient branches are currently refused before fitting; an empty-branch execution policy is not implemented.

## Attachment and knownness

The attachment owns the four `property_*` components, their four `draw_property_*` columns, `property_reported_total`, reconciliation adjustments, nullable active-bound diagnostics, residual and objective, two source-basis discrepancy columns, and two knownness flags. `owned_columns()` is the authoritative roster.

ASEC persons retain each individually qualified component and reported total, including NaN for an unresolved value. Donor exclusion does not erase otherwise known amounts. The source-basis discrepancies remain available, and the complete source qualification/exclusion descriptions stay bound to the projection. ASEC raw-draw and reconciliation diagnostics are absent because no draw or reconciliation is performed on these observations.

Eligible ACS persons receive one ordered joint draw and its reconciled components. Their original `person_id` and pandas row index are checked separately, then the original support ID, native source ID, source label, and exact clone roles establish the two-row fan-out. Excluded ACS persons receive NaN for model components and the attached anchor, with false `property_anchor_known` and `property_components_known`. Their original source anchor status remains in the qualified source projection; under-15 values are not silently converted into a new analytical zero.

`property_anchor_known` describes the attached anchor's finite value. `property_components_known` describes all four attached components being finite; it is distinct from eligibility for donor fitting. Reconciliation bound flags use nullable booleans so an unperformed reconciliation does not become a false observation.

## Tax and period boundaries

The eight existing financial tax leaves remain unchanged. Attachment reads all eight and explicitly follows the legacy financial attachment, checks them against its retained draw history, then adds property columns. The legacy capital-gains chain still conditions on legacy INT and DIV draws. The receipt records `tax_split_rebased: false` and the capital-gains limitation. `property_legacy_interest_draw_minus_reconciled_interest` measures the legacy ACS interest draw minus the reconciled ordinary plus retirement interest; it is a diagnostic, not a new tax assignment.

Broad property receipts are not identified with rental income or a tax-law leaf. Retirement-account earnings are not retirement withdrawals. Tax-leaf rebasing and a CAP model conditioned on reconciled components require a separate declared successor.

Periods remain those of the qualified sources: ASEC 2025 interview and calendar-2024 income; ACS 2024 rolling prior twelve months, adjusted by ADJINC to the declared price basis. No equivalence of the reporting windows is asserted. The person-level conditional chain is not a certification of household-level dependence or held-out fit quality.

## Host integration and verification

- `current_survey_property_nodes(qualified, clone_frame, host_pins=..., options=...)` declares the fragment.
- `register_property_kernels(...)` registers the country and shared numerical kernels over the retained source owners.
- `reconstruct_property_results(..., artifacts=..., legacy_matrix_producer_key=...)` returns eight expected non-fit results: projection, two filters, two model-column operations, draws, reconciliation, and attachment. The host patches these at their declared population versions and checks actual observations. The separate model-receipt verifier binds the eight fit/apply receipts to actual donor/recipient populations and stored model bytes.
- `verify_materialized_property_income(..., legacy_population=..., population=...)` requalifies actual source owners and compares the full final population against attachment applied to the independently checked legacy population.

Reconstruction validates the complete ordered raw-draw history, training-model references, recipient index, person IDs and summaries. It invokes the existing deterministic reconciliation rule to derive the expected reconciliation columns; it never fits or draws another model. Store artifact type, producer-key and key validation remains a prerequisite owned by the host. The host must retain its existing final source, implementation, cache, full-population and issuance checks after relevant I/O.

Changing only scales affects the reconciliation and attachment declarations, leaving source selection and QRF fitting declarations unchanged. The absence of property options in a host call preserves the existing strict legacy path. Graph replay validates cached output observations as well as labels; a cache hit is not source admission or a release verdict.
