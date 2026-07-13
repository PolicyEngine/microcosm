# UK input-coverage progress

The coverage baseline is the 145 populated effective loader overrides extracted
from the immutable enhanced-FRS 2023-24 artifact (SHA-256 `584ae33d…`). The
certified `populace_uk_2023` candidate (SHA-256 `f17306cc…`) carries raw
non-default signal for all 145, but `gift_aid` and
`charitable_investment_gifts` carry it only on zero-weight SPI rows. The launch
register therefore honestly has **143 required columns and 2 reviewed
exclusions**. Both exclusions use the campaign reason “not yet ported from
enhanced FRS pipeline — pending review” and the required tracking note.

| Milestone | Coverage change | Evidence | Status |
| --- | ---: | --- | --- |
| Launch contract baseline | 143 required; 2 reviewed exclusions | SHA-pinned enhanced-FRS surface plus owning-entity effective-mass evidence from the certified Populace UK H5 | Complete |
| Loader-override correction | +13 formula-owned persisted overrides | UK Simulation passes every engine-known persisted H5 column through `set_input`; exact cached-artifact replay covers all 145 | Complete |
| Contract entity/pin integrity | No status change | All 145 reference and candidate columns carry owning-entity evidence; wrong-table columns and unproven HF revision mappings fail | Complete |
| National orchestration seam | No status change | Ordered stage protocol, stable verified-byte binding, cheap preflight, final manifest gate, and atomic staging-H5 write | Complete |
| Effective-mass coverage | −2 required; +2 reviewed exclusions | Candidate evidence and the final gate both require signal on at least 0.000001 of owning-entity effective population mass; zero-weight Gift Aid support is excluded honestly until restoration | Complete |
| HMRC/SPI source identities and Q1 | No raw status change | Reviewed donor/ODS pins; real ODS 208-fact parse; documented donor-leaf reconciliation; deterministic TEI/TII/TI synthetic contract; one SRP surface for bands and Table 3.6 | Complete |
| HMRC/SPI income restoration | Not promoted | The adjudicated Option 1 raw-FRS audit found complete sources for only part of the ten-leaf crosswalk; five leaves have no source-faithful monetary measure and two are incomplete, as recorded below | Blocked pending per-constituent adjudication |
| Real-donor replay, 2026-07-13 | No status change | The donor and ODS pins remain verified, but the adjudicated source audit triggered the required stop before donor replay; no staging file, coverage sidecar, or aggregate 208-fact report was emitted | Blocked pending per-constituent adjudication |

## Restoration diagnosis

The HMRC adjudication miss is distributional, not an absent-column gap: total
income-tax liability was £334.629bn against the £277bn SPI-anchored benchmark
(+20.8047%), while all HMRC-family loader inputs had raw non-default signal.
The geography-clone tool only clones a compact H5, so a distinct national
orchestration seam was required before any family restoration could run.

## National orchestration seam

`tools/build_uk_national_dataset.py`, backed by
`populace.build.uk_runtime.national_build`, now performs the minimal reviewed
sequence:

1. validate the checked-in input-coverage manifest and required stage plan;
2. load the certified compact UK person, benunit, and household tables, binding
   the in-memory stage input to the stable file identity captured around the
   candidate SHA-256 verification;
3. run ordered named national stages, validating entity IDs and direct person
   references to households and benunits after each;
4. run the final input-coverage/effective-mass gate, which additionally requires
   every benunit to resolve to exactly one weighted household; and
5. atomically write a caller-named staging H5 and evidence sidecars.

This seam does not clone households, assign local geography, publish a release,
or alter `tools/build_uk_rowwise_dataset.py`. The existing geography tool
remains downstream and separate.

The HMRC stage is designed to drop the certified candidate's zero-weight SPI
rows, rebuild one SPI channel, allocate 50% of unchanged national household
mass to it as `IMPORTANCE` weights, and record the factor-one allocation as a
deliberate `MassChangeRecord`. A successful calibration would then transition
`IMPORTANCE` to `CALIBRATED`, conserve national mass, compile all 208 facts,
respect a 5× record-weight ratio cap, and keep every target within 5%.

While Q2 remains blocked, the manifest records `hmrc_spi_income` as
`deferred_until_restored`: the stage plan and family-specific gates are not
executable release requirements yet, and its two effective-mass columns remain
reviewed exclusions. Promotion is fail-closed and coupled: once source-backed
FRS constituents make the family runnable, the source contract must record the
restored state, the family becomes `required_at_build`, and both columns must
be promoted to required from evidence produced by the restored candidate.

## Reviewed real-source evidence

The licensed donor remains local at `inputs/spi/put2223uk.tab` and is excluded
by `.git/info/exclude`; it must never be committed or pushed. Its reviewed
identity is:

- Survey of Personal Incomes Public Use Tape 2022-23;
- 141,323,762 bytes;
- SHA-256 `5ef829461060c91a2a47be59ad541d9b519fc3976d66ca80d4920f711bb96f66`;
- 836,850 donor records; and
- PolicyEngine's licensed copy from the private `spi_2022_23.zip` artifact.

The real donor passes strict parsing. Its published rounded fields satisfy
`abs(TI - (TEI + TII)) <= £5` on every row (observed maximum £5). That £5
tolerance is source validation only; synthetic accounting identities receive
no tolerance. In the reviewed seed-42, FACT-weighted 100,000-row bootstrap,
3,672 rows carry Gift Aid and 9 carry charitable-investment gifts. These counts
are readiness evidence, not effective-weight restoration.

The official SN 9422 Annex A leaf formulas also reconcile against the pinned
donor. Among 834,538 ordinary records, the maximum absolute differences from
published TEI/TII/TI are £15/£10/£20. Among the 2,312 documented composite
records (`AGERANGE == -1`), they are £180/£10/£180. The larger composite
envelope is expected because PUT anonymisation averages records across the two
nonlinear `max(0, ...)` identities before final £5 field rounding. The source
contract pins the ordinary and composite envelopes separately; it does not
broaden the independent £5 published `TI = TEI + TII` check or the exact
post-draw identity.

The official HMRC 2023-24 ODS remains local at
`inputs/hmrc/Collated_Tables_3_1_to_3_11_2324.ods` with reviewed identity:

- official URL recorded in `hmrc_income_source_stages.json`;
- 166,693 bytes;
- OpenDocument Spreadsheet MIME type; and
- SHA-256 `ad063b06b2bdeef8600dbbb09d48153337a4966f8c7eea50df7a2e0304ebd73e`.

The full real parser contract passes: exact Table 3.6/3.7 sheet and header
layouts, all 13 ordered bands, the single trailing “All ranges” sentinel, and
8 components × 13 bands × 2 measures = **208 positive facts**. Savings interest
and Table 3.7 “Other income” are included; no published component is narrowed.

## Q1 deterministic accounting identity

The first QRF surface draws documented source leaves, including `SRP` and
`OTHERINC`; it does not draw HMRC employed income, TEI, TII, or TI. After each
draw the runtime:

- derives the PolicyEngine `employment_income` input as `PAY + EPB + TAXTERM`,
  matching the pinned enhanced-FRS pipeline;
- derives the broader HMRC employed-income auxiliary from its normalized source
  leaves;
- derives TEI and TII from their constituent draws; and
- assigns `hmrc_spi_assessable_income = TEI + TII` exactly.

The Table 3.6 state-pension measure uses the same drawn SRP auxiliary included
in TEI and band assignment, rather than an independent stage-2/model draw.

Tests assert the identity on every synthetic row. The official rounded `TI`,
`TEI`, and `TII` donor fields are validation inputs only and are never stochastic
QRF outputs.

## Q2 fail-closed blocker

The conductor requires the Table 3.6 employment measure to use one documented
constituent crosswalk identically on both FRS and SPI channels, while preserving
the narrower PolicyEngine employment-input semantics. The official ODS note
defines employment income broadly: pay from employment, taxable benefits,
Incapacity Benefit, contribution-based ESA, and JSA. The SPI documentation's
exact formula also requires `EXPS`, `INCPBEN`, `OSSBEN`, `UBISJA`, and
`MOTHINC` in addition to `PAY`, `EPB`, and `TAXTERM`.

The certified candidate persists only aggregate `employment_income`; it does
not retain the required normalized employment leaves or employment expenses,
and `incapacity_benefit_reported` is absent/all-default. Its
`miscellaneous_income` is an FRS odd-jobs/royalties aggregate, not a source-
faithful substitute for the missing HMRC leaves or SPI `OTHERINC`. Consequently
the published broad measure cannot be reconstructed like-for-like from the
certified base.

The source manifest and runtime therefore require all normalized FRS leaves and
fail before ODS parsing, SPI replacement, donor fitting, calibration, or staging
write when any is absent. They explicitly forbid substituting
`employment_income` or `miscellaneous_income`. This is the adjudicated stop
condition, not a reviewed exclusion or a relaxed gate.

The licensed 2022–23 donor staged on 2026-07-13 does not resolve this blocker.
Its 141,323,762-byte size and SHA-256 `5ef829461060c91a2a47be59ad541d9b519fc3976d66ca80d4920f711bb96f66`
and the ODS identity were reverified before use. A production invocation against
the certified candidate then failed at
`assert_frs_hmrc_auxiliary_crosswalk_available` with all ten normalized FRS
constituents missing. The preflight ran before ODS parsing, donor reading, SPI
replacement, calibration, the release gate, and all staging writes, as designed.
The caller-owned `uk_runtime/` scratch directory remained empty and is locally
excluded from Git.

The final source-semantic audit confirms that this is not a naming-only gap.
The official SPI 2022–23 Annex A defines monetary `EPB` and `EXPS`, separately
taxable `TAXTERM`, `INCPBEN`, `OSSBEN`, and `UBISJA`, and distinct `MOTHINC`
and `OTHERINC` fields. The pinned enhanced-FRS pipeline persists FRS `INEARNS`
only as aggregate PolicyEngine `employment_income`; it has no monetary
equivalents for `EPB` or `EXPS`, cannot separate taxable termination pay from
gross redundancy, and its `miscellaneous_income` combines concepts that cannot
be assigned source-faithfully between the two SPI miscellaneous leaves.
`incapacity_benefit_reported` is also all-default on the pinned eFRS surface.
Consequently, `employment_income` plus reported benefit aggregates is a new
shared proxy, not the conductor-required identical documented constituent
crosswalk. It was not substituted into the release path.

A throwaway, non-release diagnostic using those candidate aggregates was
stopped during stage-two QRF prediction as soon as the codebook audit
established that the mappings were proxies. It wrote no row-level or aggregate
artifact and is not replay evidence. The strict production result above is the
only admissible release-path result.

## Q2 Option 1 raw-FRS source audit, 2026-07-13

The conductor selected Option 1 with no proxy: retain all ten normalized leaves
from source-faithful FRS variables, or, if any leaf has no raw source, document
the per-constituent evidence and stop. The audit covered the 2023-24 raw FRS
`ADULT`, `JOB`, `BENEFITS`, and `ODDJOB` tables and the other income-bearing
`PENSION`, `ACCOUNTS`, and `ASSETS` tables. It also checked the current
`policyengine-uk-data` FRS loader and the
[official FRS 2023-24 benefit definitions](https://doc.ukdataservice.ac.uk/doc/9367/mrdoc/pdf/9367_frs_2023_24_benefits_documentation.pdf)
against the
[SPI 2022-23 Annex A definitions](https://doc.ukdataservice.ac.uk/doc/9422/mrdoc/pdf/9422_put_2223_full_documentation.pdf).
The licensed SPI donor was not opened for this audit.

The mass estimates below use the certified candidate's FRS channel: positive
household weights are folded through the candidate's exact raw-household and
person ancestry, yielding 68,441,459.783 effective person-mass units. A nearby
flag or partial amount is reported only as an at-risk or lower-bound diagnostic;
it is not evidence that the normalized leaf is populated.

| SPI leaf sought | Raw FRS table and variable evidence | Source-faithful finding | Effective-mass implication |
| --- | --- | --- | ---: |
| `PAY` | `ADULT.INEARNS`; `JOB.UGRSPAY` checked for the underlying job-level gross-pay composition | Available as the annualized earned-pay measure. | 38.7291454% has positive pay. |
| `EPB` | `JOB.EXPBEN01`-`EXPBEN13`; partial amount fields `CARVAL`, `CARAMT`, `FUELAMT`, `VCHAMT`, and `CHVAMT` | **Missing.** `EXPBEN*` are receipt flags, and the amount fields cover only selected benefits; they cannot produce complete taxable expenses payments and benefits. | 12.9485464% has at least one receipt flag, but this is not monetary support. |
| `EXPS` | `JOB.EXPBEN04`/`EXPBEN05`, `MILEAMT`, `MOTAMT`, `UMILEAMT`, `UMOTAMT`, `DEDUC1`-`DEDUC9`, and `UDEDUC1`-`UDEDUC9` | **Missing.** These fields describe reimbursements or payroll deductions, not the complete tax-deductible employment-expense amount required by SPI. | 5.1302528% has an adjacent reimbursement flag; the true `EXPS` mass is not estimable. |
| `INCPBEN` | `BENEFITS.BENAMT` where `BENEFIT == 17` | Structurally expressible, but the current FRS has no code-17 rows and therefore no observed monetary signal. | 0% observed mass. |
| `OSSBEN` | `BENEFITS.BENAMT`, `BENEFIT`, and `VAR2`: code 13 and contribution-based code 16 are identifiable; codes 6 and 30 were also searched | **Incomplete.** Carer's Allowance and contribution-based ESA form an identifiable subset, but code 6 mixes tax treatments and code 30 is an undifferentiated catch-all, so the complete taxable family cannot be emitted. | 1.8045088% identifiable lower-bound mass; not counted as support. |
| `TAXTERM` | `ADULT.REDAMT`; `ADULT` and `JOB` searched for a taxable termination split | **Missing.** `REDAMT` is gross redundancy pay and has neither the taxable amount nor non-redundancy termination pay. | 0.3746084% has positive gross redundancy pay; taxable mass is unknown. |
| `UBISJA` | `BENEFITS.BENAMT` where `BENEFIT` is 14 (JSA) or 19 (Income Support) | Available as the annualized source measure. | 0.5378644% has positive source signal. |
| `MOTHINC` | `ODDJOB.OJAMT`/`OJNOW`, `ADULT.ALLPAY2`, `ADULT.ROYYR2`-`ROYYR4`, and `JOB.OWNOTHER` | **Missing.** The fields are heterogeneous and belong to distinct income concepts; assigning their union to SPI miscellaneous employment income would be a proxy. | Odd-job-only mass is 0.1724207%; the broader unresolved miscellaneous pool is 1.4650566%. |
| `OTHERINC` | The same `ADULT`, `ODDJOB`, and `JOB` fields, plus `PENSION`, `ACCOUNTS`, `ASSETS`, and `BENEFITS`, were searched for a distinct residual-income field | **Missing.** No person-level raw FRS variable has SPI `OTHERINC` semantics, and the miscellaneous pool cannot be split between `MOTHINC` and `OTHERINC` from source evidence. | No separable mass estimate; the unresolved pool is 1.4650566%. |
| `SRP` | `BENEFITS.BENAMT` where `BENEFIT == 5`; codes 6 and 9 checked for widow-related amounts | **Incomplete.** Code 5 supplies regular State Pension, but the FRS source does not identify the full SPI combination of State Pension lump sums and widow's pension; code 6 mixes benefits and code 9 is tax-free War Widow's Pension. | 18.1567916% has regular code-5 State Pension; not counted as complete `SRP` support. |

This establishes the adjudicated stop condition: `EPB`, `EXPS`, `TAXTERM`,
`MOTHINC`, and `OTHERINC` have no complete raw FRS source, while `OSSBEN` and
`SRP` are only partial. No imputation, proxy, normalized-leaf stage, source-
manifest promotion, national staging build, or real-donor replay was attempted.
The release contract remains **143 required + 2 reviewed exclusions**.

No national staging artifact has passed the 208 real targets, the weighted
charitable-giving floor, or the income-tax backtest. `gift_aid` and
`charitable_investment_gifts` remain distributional/effective-weight gaps and
the HMRC family is **not counted restored**.
