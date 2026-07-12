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
| HMRC/SPI income restoration | Not promoted | Q2 cannot be materialized like-for-like on the certified FRS channel; fail-closed blocker below | Blocked pending conductor review |

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

Conductor review must choose one of these before restoration can resume:

1. approve a new FRS decomposition/imputation stage that materializes every
   normalized HMRC constituent with source evidence; or
2. explicitly review a shared higher-level proxy and its expected bias.

No national staging artifact has passed the 208 real targets, the weighted
charitable-giving floor, or the income-tax backtest. `gift_aid` and
`charitable_investment_gifts` remain distributional/effective-weight gaps and
the HMRC family is **not counted restored**.
