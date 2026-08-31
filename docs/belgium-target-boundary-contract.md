# Belgium target and behavior boundary contract

This note is value-free: target values remain in Chronicle and are resolved by
reference. It records the reviewed dependency coordinates, intended target
dispositions, ownership boundary, and follow-up work. A *calibration candidate*
is not executable until every listed input, support, period, unit, and mapping
gate is satisfied. *Validation-only* rows must never enter a calibration
objective. *Blocked* rows must fail closed at compilation or binding.

## Reviewed dependency coordinates

| Dependency | Reviewed base | Reviewed head | Relationship |
| --- | --- | --- | --- |
| [Microcosm #824](https://github.com/PolicyEngine/microcosm/pull/824) | `a18c87ee3db8220038b894a25a695e9bc79871e2` | `ed6dc3d8d47947fecfffd1194d1c1009221ae7cb` | Existing draft to update; do not open a duplicate. |
| [Microcosm #825](https://github.com/PolicyEngine/microcosm/pull/825) | — | `510e3e6c9c2cd7f61854c56f6edd2acb14bc1cd8` | Reviewed generic monetary-target head, merged into authoritative `main` by `d1e3e397bdc4b7e6b9e05dc73cf6345e7111e6db`. |
| [Chronicle #212](https://github.com/PolicyEngine/chronicle/pull/212) | `10597ae602767b046ca1b294949e5df7bfd3b367` | `0f75a2bb5fae8a197e4e1f6541a5af34708fc206` | Existing draft dependency supplying official GRAPA and regional child-benefit facts and provenance. |

These hashes identify what was reviewed; later commits must not be described as
reviewed under these coordinates. The Belgium work must incorporate #825 from
the authoritative merge commit while preserving #824's history.

## Target matrix

| Surface | Exact publisher period and scope | Disposition | Fail-closed condition |
| --- | --- | --- | --- |
| Demography | Statbel calendar year 2025; people by age band and sex at NUTS1 under `NUTS_2024`. | Calibration candidate, currently blocked. | Requires scalar cell references, exact 2025 basis selection, NUTS-vintage compatibility, and demonstrated frame support. No consumer-authored projection may masquerade as an observation. |
| Fiscal income by commune | Statbel income/tax year 2023; taxpayer persons by commune under `nis_2025`. | Calibration candidate at diagnostic tier, currently blocked. | Requires commune scalar fanout, a supported taxpayer-person model measure, and exact `nis_2025` coverage. Preserve `metadata.nis_vintage`; never reinterpret the entity as a household. |
| Fiscal income distribution | Statbel income/tax year 2023; personal-income-tax return units, national and published NUTS1 (`NUTS_2024`) cells, including income classes and rank groups. | Validation-only; blocked from calibration. | A tax-return unit is not a Microcosm person or household. Activation requires a separately reviewed return-unit bridge and supported distribution measures; published cells must not be reconstructed or silently summed. |
| Personal income tax | SPF Finances/Statbel income/tax year 2023; Belgium total for taxpayer persons, federal and local tax before withholding. | Calibration candidate, currently blocked. | Bind through #825's monetary primitives only after an exact policy-output or prepared-measure bridge exists on the same tax-year basis. |
| Social contributions | ONSS calendar year 2024; Belgium worker-borne personal social-security contributions for worker persons. | Validation-only under the current mapping; blocked from calibration. | Chronicle currently labels the relation to the Article 17 component as approximate. An exact source-to-model concept and compatible output universe are required before calibration. |
| Unemployment | ONEM/RVA calendar year 2024; Belgium complete-unemployment recipient statistic expressed as a monthly average of persons. | Calibration candidate, currently blocked. | Requires a Microcosm-owned receipt input, exact statistic/period support, and a population-universe mapping. Do not infer receipt from a positive payment or entitlement. |
| Legal pension | SFPD January 2025 snapshot; Belgium recipient persons, with a published all-schemes total and overlapping scheme rows. | Blocked pending source-period correction and population inputs. | Chronicle presently encodes this snapshot as calendar year 2025. Correct it to the supported monthly period before targeting; do not sum overlapping scheme rows. |
| GRAPA | SFPD January 2025 regular-payment snapshot; Belgium beneficiary persons by sex plus the matching monthly payment amount. | Validation-only and execution-blocked. | Chronicle #212 already supplies these exact facts. Binding still requires a Microcosm receipt flag, snapshot support, and an exact monthly monetary/statistical bridge; it is not an annual caseload, eligibility denominator, or take-up rate. |
| Opgroeien child benefit | Rights month 2025-12, provisional; persons receiving the basic amount in the `BE-GROEIPAKKET-SCHEME` administrative scope. | Validation-only and execution-blocked. | Chronicle #212 supplies the fact. The scheme scope is not BE2 residence; require an exact Microcosm scheme-membership/receipt mapping and supported child-person population. |
| Iriscare child benefit | Legal period 2025-12, provisional; entitled-child persons and distinct payment-recipient persons in `BE-IRISCARE-CHILD-BENEFIT-SCHEME`. | Validation-only and execution-blocked. | Chronicle #212 supplies both facts. The administrative scope includes records outside BE1; require separate typed inputs and never relabel it as Brussels residence. |
| Ostbelgien child benefit | December 2025; paid-child persons and distinct payment-recipient persons in the `BE-DG` child-benefit statistical scope. | Validation-only and execution-blocked. | Chronicle #212 supplies both facts. `BE-DG` here is a scheme population, not a NUTS geography; require separate exact mappings and inputs. |
| Walloon child benefit | December 2023; French-language Walloon scheme scope, with child persons and households in four publisher-defined household/social-supplement partitions. | Validation-only and execution-blocked. | Chronicle #212 supplies every partition. Preserve the person/household distinction and the four rows; do not construct an all-scope total or equate the scheme to Walloon residence. |
| NBB national accounts | NBB calendar year 2024; Belgium S.14 household-sector gross disposable income in current-price EUR. | Validation-only. | Keep outside the calibration objective. Comparison requires an explicit model aggregation and unit/period receipt; national accounts are not a population-construction target. |
| EUROMOD | JRC Belgium country-report comparators for calendar years 2021-2023, including 2022 distribution statistics; country-level person or government comparator entities as published. | Validation-only. | Preserve external, SILC, and EUROMOD series identities. None may enter solver targets or be treated as an administrative observation. |
| FPB | Belgium publisher observations for 2022-2025 and FPB-authored projections for 2026-2031 from the June 2026 outlook. | Validation-only. | Preserve each publisher cell's observation/projection assertion. Chronicle stores the projection as a publisher fact; Microcosm must not create or relabel a projection. |
| Constructed comparisons | No independent publisher period or scope; each comparison inherits explicitly pinned operands and construction metadata. | Validation-only. | Construct in Microcosm validation receipts, never Chronicle and never the calibration objective. Refuse operands with mismatched period, geography, entity, unit, or universe unless a reviewed bridge is named. |
| HFCS wealth | No reviewed NBB/ECB HFCS fact is present in the pinned Chronicle dependency; period, wave, wealth concept, universe, and support are therefore deliberately unset. | Blocked. | Do not add a target until an official aggregate package pins the survey wave/reference period, Belgium geography, household universe, weight/statistic, unit/price basis, and usable Microcosm support. |

No regional child-benefit rows may be combined into a national total: their
publishers, entities, periods, definitions, and statistical populations differ.

## Input and behavior ownership

| System | Owns | Must not own |
| --- | --- | --- |
| Chronicle | Publisher facts, exact source semantics, source assertions (including publisher-authored projections), and provenance. | Population construction, target selection, calibration, scheme-to-population mapping, consumer projections, latent inputs, or behavior. |
| Microcosm | Population construction; calibration and validation declarations; measured or latent microdata inputs; population of receipt, application, public-document legal-status, and choice flags; typed scheme-population mappings; support/readiness gates and receipts. | Take-up assignment formulas, labor-supply response, or invented legal concepts. |
| PolicyEngine | Consumption of Microcosm-supplied flags; non-legal behavioral mechanics such as take-up assignment and labor-supply response; orchestration around Axiom. | Supplying the underlying microdata flag, storing publisher facts, or placing behavioral mechanics in Axiom. |
| Axiom | Concepts and computations explicitly grounded in public policy documents, including a legal event, status, claim, or application only when the exact public source supports it. | Synthetic concepts such as `takes_up_grapa_if_eligible`, latent draws, propensities, elasticities, or orchestration. |

The operational direction is therefore **Microcosm input -> PolicyEngine
mechanic -> Axiom legal calculation where applicable**. A positive Axiom
entitlement or payment is never evidence of observed receipt or application.

## Follow-up issue drafts

The following are ready-to-file issue scopes. They do not assert that an issue
has already been opened.

### Chronicle: ingest official Belgium HFCS aggregate wealth facts

**Title:** Add value-faithful NBB/ECB HFCS Belgium wealth aggregates

Ingest a single named official HFCS release without using restricted microdata.
Pin the source artifact and hash; record the wave, fieldwork/reference period,
Belgium geography, household universe, published weighting/statistic, wealth
concept, EUR unit and price basis, missing-value convention, and publisher
assertion. Emit only publisher cells with exact source record IDs. Do not
construct quantiles, interpolate a wave, or select Microcosm targets in
Chronicle. Add source-package and bundle tests. Acceptance requires a reviewer
to reproduce every emitted fact from the pinned official table and leaves
Microcosm blocked until matching household fields and weighted support exist.

### Chronicle: correct the SFPD pension snapshot period

**Title:** Represent the January 2025 SFPD legal-pension caseload as a monthly snapshot

The `sfpd-legal-pension-caseload-2025` package describes January 2025 but emits
`calendar_year: 2025`. Verify the source date and, if supported, migrate the
record set and source record IDs to period type `month`, period `2025-01`, with
an explicit snapshot basis. Update manifests, aliases or downstream migration
notes, bundle tests, and exact selector tests. Retain the published all-schemes
row independently and document that scheme rows overlap; never replace the
published total with their sum. If the source cannot substantiate the month,
keep the target blocked and record the exact unresolved evidence instead.

### Microcosm: populate Belgium receipt/status inputs and scheme mappings

**Title:** Wire Belgium population data to typed benefit input and scheme-scope contracts

Populate the typed Microcosm inputs needed for ONEM, pension, GRAPA, Opgroeien,
Iriscare, Ostbelgien, and the Walloon partitions. For every input, pin entity
(person or household), role, period, source column or latent-data method,
missingness semantics, and boolean/domain validation. For every mapping, retain
the exact Chronicle statistical-scope ID and distinguish child, payment
recipient, beneficiary, and household roles. Never substitute NUTS residence
for scheme membership and never derive receipt from a positive amount. Emit
row-count/value receipts and fail closed when a required field, supported row,
or mapping is absent. State that Microcosm supplies the flag and PolicyEngine
only consumes it to apply behavior; add no behavioral formula here.

### Microcosm: add reviewed Belgium period and unit bridges

**Title:** Receipt Belgian target period, statistic, and monetary-unit bridges

Define explicit bridges for the selected Statbel population year, tax-year PIT
and fiscal-income facts, ONSS annual flows, ONEM monthly-average statistics,
monthly GRAPA/pension/child-benefit snapshots, and current-price monetary
amounts. Each bridge must record source and target periods, stock/flow/statistic
basis, unit scale, price basis or deflator authority, operation, and support
receipt. Use #825 monetary primitives where their annual scalar contract fits;
extend the typed contract before attempting monthly flows. Do not silently
annualize a snapshot, relabel an observation, or turn a consumer projection
into a Chronicle fact. Preserve and test `metadata.nis_vintage` during any
schema migration.

### Chronicle/Microcosm: resolve the ONSS model-concept relation

**Title:** Replace the approximate ONSS Article 17 proxy with an exact, reviewed concept bridge

Audit ONSS Table 6's worker universe, contribution components, sectors/statuses,
timing basis, corrections, and worker/employer split against the public-document
Axiom variables and the proposed Microcosm prepared measure. Do not relabel the
broad worker-borne total as an Article 17 component. Chronicle should retain a
neutral exact publisher concept and provenance; Microcosm may bind it only if a
reviewed model-output aggregation has the same universe and semantics. Add an
exact concept-identity test and an aggregation receipt. If equivalence cannot
be proved, retain the fact as an explicitly approximate validation comparator
and keep calibration fail-closed.
