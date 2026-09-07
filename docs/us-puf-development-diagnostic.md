# PUF Schedule C development diagnostic

`us.puf.schedule_c_wage_mars_diagnostic.v1` permits one conditional-support
diagnostic using the source-qualified price transport and complete combined
ACS+ASEC clone population. It does not admit tax inputs, national or congressional
district scoring, release publication, or the conditional model's scientific
adequacy. Source metadata, including
`fit_admission=not_admitted_pending_basis_review`, remains unchanged.

| Choice | Exact behavior and limit |
| --- | --- |
| Target | Delivered `E00900`, Schedule C net profit/loss, in the separately qualified price artifact. Retain every signed and zero raw draw. No clipping, residuals, accounting repair, participation inference or person allocation. |
| Predictors | Price-restated donor `E00200` and four one-hot MARS categories. Host wages are a reported-wage proxy: head plus actual spouse for JOINT, head only otherwise. Dependents do not contribute. SURVIVING_SPOUSE maps to MARS 2 with head wage only. |
| Host roles | Read the actual microunit `person.tax_unit_role_input` HEAD/SPOUSE/DEPENDENT values and `tax_unit.filing_status_input`. Require one head, exactly one spouse for JOINT and none otherwise. Missing or unsupported roles refuse. The older invented-only Boolean-role API is preserved. |
| Model | Existing targetwise QRF train/apply, one target, two trees, seed 578, `zero_atol=0`. S006 supplies donor DESIGN resampling weights only. Host weights and native cells remain unchanged. |
| Prices | The accepted price helper uses the 2015-file CPI-U baseline and ratio `313689/237017` to 2024 development prices. Raw FLPDYR meaning remains unresolved. This is not nominal income growth or harmonized reference-period evidence. |
| Missing wages | Preserve missingness. Selected earners must have finite nonnegative wages; missing values refuse. The ACS under-15 universe is recorded, but v1 does not turn a blank into zero. |

The IRS booklet identifies E00200 as tax wage and E00900 as Schedule C net
profit/loss. Census gross cash wages and tax wages need not agree: deductions,
additional taxable amounts and tax rules can differ. Filing status is constructed
potential-unit support, not evidence that the survey unit filed a return. The
transported conditional distribution therefore cannot be interpreted as
unconditional population business income. These are modeling limitations, not
missing numerical conversions. See the [CPS definitions](https://www.census.gov/programs-surveys/cps/technical-documentation/subject-definitions.html),
[2024 ACS subject definitions](https://www2.census.gov/programs-surveys/acs/tech_docs/subject_definitions/2024_ACSSubjectDefinitions.pdf),
[2015 Form 1040 instructions](https://www.irs.gov/pub/irs-prior/i1040gi--2015.pdf),
and the retained source references in [the raw PUF definition](../packages/microcosm-build/src/microcosm/build/us_runtime/puf_2015_raw_source_definition.json).

## What the consumer authenticates

The price consumer accepts bytes and external expected pins, with no package-owned
local paths. It authenticates the accepted proposal/config/completion/cold/required
chain, original source definitions and payloads, immutable column metadata, price
contract, runtime, exact ordinary RECID axis and all 22 typed arrays. The four
tagged aggregate records remain excluded without imposing an order on them or
sorting ordinary returns. The packaged route consumes the qualified export; it
does not rerun growth or fabricate a grown-table object. The visibly invented
route cannot supply packaged source authority.

Host qualification requires the actual typed source and producer edges. It
reconstructs native origin, clone parentage and full materialized content; paired
source cells, membership and weights must agree across native/detail roles,
including zero-weight rows. ASEC evidence is reconstructed through the existing
prepared, current-money, arm-row and reported-income readers, retaining native IDs,
income cohort, age, status, validity, zero-origin and monetary-basis evidence.
ACS verifies native WAGP, ADJINC, AGEP, age and source year against the carried
`WAGP * (ADJINC / 1_000_000)` bytes. It preserves the different ASEC and ACS
reference periods and adjustments described in [the ASEC contract](us-asec-current-money.md).

The recipient matrix and attachment bind actual entity IDs and their order,
matrix/raw/state producer keys and payloads, exact feature order and completed
target chain. The full materialized verifier must run on the actual Population
before and after the diagnostic boundary, with design anchors and mass ledger.
The earlier `survey_origins.clones` receipt describes its pre-transfer population;
it is not relabeled as the final population hash after adding diagnostic columns.
Final placement, raw/state evidence and materialized comparison bind that change.

This first consumer requires untouched equal-weight parent/detail pairs. It is
not an adapter for later redistributed or calibrated clone weights. Source channel
chooses the declared survey wage evidence; clone role controls placement and
weight decisions are channel independent.

## Execution boundary

The integrating runner must qualify price transport and the complete host feature
population **before permitting fit**. Donor fit intentionally has no host
ancestry so host feature data can change without retraining. Graph dependency
ordering alone does not enforce this preflight. A changed donor input or DESIGN
weight rekeys fit. Shared adapter source changes can conservatively rekey both
sides; the data-dependency separation does not promise source-edit efficiency.

Invented tests execute the actual source/growth codecs, full 17-node survey graph
and six-node diagnostic extension, followed by fresh-process REQUIRED replay.
Receipt parser fixtures are visibly invented; they are not genuine lifecycle
evidence. The full graph fixture has a nonempty original-design anchor roster;
the separate zero-weight omission test uses a materialized literal population
and does not claim that a raw source admits a zero design weight.

A genuine reader/completeness run still needs accepted host and price pins,
explicit resource bounds and independent review of its runner. The consumer
currently qualifies the full clone population. The earlier proposal's optional
16-unit-per-arm diagnostic selection is not implemented; it would require a
separately reviewed identity-first selection after full lineage binding. A
successful completeness run is not automatic permission to fit or score.
