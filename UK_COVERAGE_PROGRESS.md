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
| HMRC/SPI adjudicated replay | Not promoted | Full PAY/UBISJA/INCPBEN and the explicitly named OSSBEN/SRP subsets are retained; five source-absent leaves and both subsets carry canonical fences; complete FRS Total Income bands remain unavailable | Complete as a fenced replay; not a restored family |
| Real-donor replay, 2026-07-13 | No status change | Pinned donor + ODS, 100,000-row reviewed bootstrap, 432,779 SPI predictions, exact post-draw identity, 1ppm Gift Aid checks, and a complete 0 exact / 0 directional / 208 excluded aggregate report | Complete; final 143+2 gate correctly blocks stale exclusions |

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

The HMRC stage drops the certified candidate's zero-weight SPI rows, rebuilds
one SPI channel, allocates 50% of unchanged national household mass to it as
`IMPORTANCE` weights, and records the factor-one allocation as a deliberate
`MassChangeRecord`. The latest constituent adjudication forbids calibration:
all 208 published facts use non-overlapping Total Income bands, while the FRS
instrument cannot materialize complete Total Income. The replay therefore
keeps `IMPORTANCE` weights and emits a fenced report instead of fitting biased
constraints.

The manifest records `hmrc_spi_income` as `deferred_until_restored`. The two
charitable columns remain reviewed exclusions under the conductor-frozen
**143 required + 2 reviewed exclusions** contract even after the real replay
demonstrates positive-mass signal. This deliberately makes the final anti-rot
gate fail on those stale exclusions; status promotion is a separate conductor
decision and was not performed on this branch.

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

## Q2 fail-closed audit and adjudicated resolution

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

The first production preflight therefore stopped before ODS parsing, donor
reading, SPI replacement, QRF fitting, or staging writes. That fail-closed stop
was the correct pre-adjudication result: it proved the compact candidate alone
could not supply the broad crosswalk and wrote no artifact.

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

The conductor subsequently adjudicated the raw-source audit constituent by
constituent. The national seam now retains full `PAY`, `UBISJA`, and structurally
expressible `INCPBEN` leaves directly from the raw FRS, alongside rather than in
place of PolicyEngine inputs. It separately names
`ossben_identifiable_subset` and `srp_regular_code5`; neither is represented as
the full SPI concept. `EPB`, `EXPS`, `TAXTERM`, `MOTHINC`, and `OTHERINC` remain
source-absent and fenced. The runtime writes `NaN`, not zero or a proxy, where a
full source concept cannot be materialized on the FRS channel.

Because those missing and partial legs prevent a complete like-for-like FRS
Total Income measure, every published fact depending on an income band is a
reviewed exclusion. The partial measures do not establish a one-directional
bound on band membership, so none qualifies as directional. The real replay
therefore evaluates the complete 208-fact surface as 208 fenced exclusions and
performs no HMRC calibration.

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

This establishes the adjudicated source contract: `EPB`, `EXPS`, `TAXTERM`,
`MOTHINC`, and `OTHERINC` have no complete raw FRS source, while `OSSBEN` and
`SRP` are only partial. The retained-leaf stage implements exactly the three
full and two explicitly named subset findings. It does not impute, combine
heterogeneous fields, or promote a subset to a full source concept. The release
contract remains **143 required + 2 reviewed exclusions**.

## Real-donor HMRC replay, 2026-07-13

The production replay reverified both opaque source identities before either
file was read: the 141,323,762-byte licensed SPI donor at SHA-256
`5ef829461060c91a2a47be59ad541d9b519fc3976d66ca80d4920f711bb96f66`
and the 166,693-byte official ODS at SHA-256
`ad063b06b2bdeef8600dbbb09d48153337a4966f8c7eea50df7a2e0304ebd73e`.
The retained FRS sources were also bound by stable verified bytes:
`ADULT` has 28,590 rows and SHA-256 `e09f9647…`; `BENEFITS` has 46,636 rows
and SHA-256 `ff30d054…`. Their source signals are 13,412 `PAY` rows, 163
`UBISJA` rows, zero observed but structurally wired `INCPBEN` rows, 627 OSSBEN
subset rows, and 8,494 regular-code-5 SRP rows.

The seam removed all 200,000 dead zero-weight SPI households and rebuilt one
honest SPI channel. It assigned that channel a reviewed 50% share of the
unchanged 28,840,551.182 national household mass, recorded the deliberate
allocation, and retained `IMPORTANCE` output weights. The reviewed seed-42 QRF
fit used 100,000 donor records, trained the FRS-only fill on 72,496 rows, and
produced 432,779 SPI person predictions. `TEI + TII = TI` holds by construction
on all 432,779 predictions.

The aggregate replay report contains the full 8 components × 13 bands × 2
measures surface. Its result is **0 exact pass, 0 exact fail, 0 directional
pass, 0 directional fail, and 208 excluded with canonical fences**. Every
estimate and delta for an excluded fact is null. This is intentional: a partial
FRS employment or pension measure cannot assign complete HMRC Total Income
bands, so computing those facts would introduce known but unbounded bias.

The 1 ppm effective-mass floor rejects dead support and numerical dust while
remaining roughly two orders of magnitude below the rarest populated reference
share. The rebuilt channel exceeds it honestly: `gift_aid` has 12,894
positive-mass rows and a 0.0133031567 mass share;
`charitable_investment_gifts` has 294 rows and a 0.0002805533 share. Neither is
counted restored while its manifest status remains a reviewed exclusion.

The final release gate sees all 143 required columns, no missing or degenerate
requirement, and no insufficient effective-mass result. It fails only on the
two charitable columns as stale reviewed exclusions, exactly as the frozen
143+2 contract requires. No staging H5 was written. The committed aggregate
artifacts are `hmrc_income_replay_report.json` (SHA-256 `32d343ab…`) and
`hmrc_income_release_gate_report.json` (SHA-256 `21a87534…`); they contain no
row-level donor data or local paths. The HMRC family remains **not counted
restored** pending conductor review of the status promotion and the fenced
published targets.
