# UK input-coverage progress

The coverage baseline is the 132 populated effective loader inputs extracted
from the immutable enhanced-FRS 2023-24 artifact (SHA-256 `584ae33d…`). The
certified `populace_uk_2023` candidate (SHA-256 `f17306cc…`) currently carries
non-default signal for all 132, so the honest launch register has **132 required
columns and no reviewed exclusions**. The committed candidate evidence records
that result per column; exclusions will use the campaign reason “not yet ported
from enhanced FRS pipeline — pending review” and a tracking note if a future
baseline exposes a real gap.

| Milestone | Coverage change | Evidence | Status |
| --- | ---: | --- | --- |
| Launch contract baseline | 132 required; 0 exclusions | SHA-pinned eFRS and certified Populace UK H5 surfaces | Complete |
| National orchestration seam | No coverage-status change | Ordered stage protocol, preflight + final gate, atomic staging-H5 tests | Complete |
| Effective-mass coverage | No raw status change; `gift_aid` and `charitable_investment_gifts` remain distributional/effective-weight gaps | Required signal must carry at least 0.000001 of its owning entity's effective population mass | Complete |
| HMRC/SPI income family | No raw-column status change; adds one `required_at_build` distributional family, including effective-mass requirements for both charitable-giving inputs | Certified-base hash gate; 2022–23 SPI QRF code; complete 2023–24 Tables 3.6/3.7 contract (208 facts); positive-prior, calibration, and source-manifest tests | Implementation wired and covered by synthetic contract tests; **not counted restored** pending the licensed donor, an exact HMRC ODS identity, and a real replay |

## Restoration diagnosis

The HMRC adjudication miss is distributional, not an absent-column gap: total
income-tax liability is £334.629bn against the £277bn SPI-anchored benchmark
(+20.8047%), while every HMRC-family loader input has non-default signal. The
repository at diagnosis time had SPI support-row primitives but no UK national
build orchestrator, SPI donor-model execution, or national target-registry
compilation path; `tools/build_uk_rowwise_dataset.py` only clones a supplied
compact H5.
This diagnosis required a national orchestration seam before an HMRC family
could be recorded. The seam below now supplies that boundary without turning
the geography-clone tool into a different build product.

## National orchestration seam design

Status: implemented in the first post-contract milestone.

The national UK build gets its own narrow entrypoint,
`tools/build_uk_national_dataset.py`, backed by a reusable
`populace.build.uk_runtime.national_build` module. Its fixed sequence is:

1. validate the checked-in input-coverage manifest against the pinned eFRS
   surface before any expensive work;
2. load an existing compact UK single-year H5 as explicit person, benunit, and
   household tables;
3. run an ordered list of named national stages, validating entity IDs and
   membership links after every stage;
4. run the hard input-coverage gate on the final staged tables; and
5. atomically write the gated tables to a caller-named staging H5.

The module owns the stage protocol and H5 boundary so restored families can add
one independently tested transform at a time. The driver now requires and runs
the SPI/HMRC stage. After a successful real-source replay, it is wired to emit
input-coverage and HMRC source/calibration evidence beside the staging file; a
failed gate does not create a failed staging H5.

This seam does **not** clone households, assign local geography, compile local
targets, publish a release, or change `tools/build_uk_rowwise_dataset.py`. The
geography-clone tool remains a downstream, separate build product. National
calibration and additional source families are also out of scope until a family
explicitly wires them through this stage boundary.

## HMRC family adjudications

Status: implementation wired on 2026-07-11 and **not counted restored**. The
repository does not contain the licensed UKDS `put2223uk.tab`, and this sandbox
does not contain a pinned copy of the public HMRC ODS. Tests exercise strict
source parsing, synthetic donor identities, fake QRFs, fake simulations and
calibration, and monkeypatched orchestration. They prove code-path contracts;
they do not prove real-source target feasibility, solver convergence, effective
Gift Aid mass, or correction of the +20.8% income-tax discrepancy. No
substitute donor or target surface was used.

The enhanced-FRS reference is pinned to `enhanced_frs_2023_24.h5` at model
revision `655dd07e4bb9c777b00dac044949611f1feb824f` (SHA-256
`584ae33d…`). The certified Populace candidate already contains SPI support
rows, but all 200,000 SPI-synthetic household rows have zero calibrated weight.
Populace calibration cannot lift a zero initial weight because it optimizes its
log and caps the result relative to that zero. Porting the QRF draws alone would
therefore be fiscally inert; a reviewed positive-prior stage is required first.

The national base is the certified Populace UK candidate. At runtime, its
existing zero-weight SPI rows are dropped and replaced once with a new SPI
support channel. A reviewed share of the unchanged national household mass is
allocated to that channel as importance weights and recorded through a
deliberate `MassChangeRecord`; dead or duplicate SPI support fails closed.

The reviewed allocation is **50% of national household mass**. This makes the
base FRS and SPI-tail channels equal-mass priors before administrative
calibration, matching the two-channel mass-allocation structure used by the US
support spine while preserving the national total exactly. The build manifest
pins the share; the driver exposes no unreviewed override. Replacement samples
the exact incumbent SPI row quota within
`clone_index × household_is_capital_gains_clone × region`, refuses to discard
any live incumbent SPI mass, advances `DESIGN → IMPORTANCE`, and writes the
factor-one mass-change reason into staging-H5 metadata.

The go-forward source pair is the 2022–23 private SPI donor and the published
[2023–24 HMRC Personal Incomes Tables 3.6 and 3.7](https://www.gov.uk/government/statistics/personal-incomes-statistics-for-the-tax-year-2023-to-2024),
explicitly mapped to the candidate's build period. Table 3.7 includes taxable
bank and building-society interest, so savings interest is part of the required
emitted family. Its fourth published category is “other income,” described by
HMRC as other investment income; the SPI crosswalk is `OTHERINV`. The
materializer fails closed if a declared sheet, component, ordered band,
positive cell, or the single trailing “All ranges” sentinel is absent.

The implemented first stage uses the enhanced-FRS pipeline's `FACT`-weighted
100,000-row bootstrap and then uniform typed design weights; applying `FACT`
again after that bootstrap would square the survey weights. It extends the
incumbent six-income QRF with the SPI `OTHERINV` crosswalk for Table 3.7's
eighth component, and emits `GIFTAID` and `GIFTINV` separately. `INCBBS` is
taxable interest; after the second-stage QRF, the PolicyEngine gross input is
reconstructed as `INCBBS + tax_free_savings_income`, then tax-free interest is
subtracted again for the HMRC measure. A tax-free amount above the gross input
fails closed. The second-stage fit uses
household-to-person importance weights and requires every materializable
enhanced-FRS output. Two fields absent from the pinned candidate are reviewed
explicitly rather than silently skipped: `incapacity_benefit_reported`
(absent/all-default) and `maternity_allowance_reported` (absent). Disability
category and flag inputs are recomputed from the newly drawn reported amounts.

The HMRC calibration compiles **8 components × 13 bands × 2 measures = 208**
facts. It uses `income_tax > 0` from a PolicyEngine-UK person-mapped simulation
as the published taxpayer universe. Band assignment uses the authoritative
[SPI-documented](https://doc.ukdataservice.ac.uk/doc/9422/mrdoc/pdf/9422_put_2223_full_documentation.pdf)
joint draw `TI = TEI + TII` on rebuilt SPI rows. On base rows it bridges the
live model with `PolicyEngine total_income + other_investment_income -
tax_free_savings_income`. SPI `TI` also includes `OTHERINC`, so it is not the
sum of the eight target measures. A successful release run must give every
fact positive-mass support, compile every fact, conserve household mass, keep
the per-record weight-ratio cap at 5, and miss no target by more than 5%. The
100,000-row donor sample, 50% SPI prior, ratio cap, and error ceiling are bound
to the executable source manifest and cannot be overridden by the release
driver. After success, runtime SHA-256 values for the supplied donor and ODS,
the target-registry version, fit weight kinds, solver options, and
effective-mass shares are written to the HMRC evidence sidecar.

Effective-mass coverage uses `household_weight`, mapped to the owning entity.
For the charitable-giving family, the numerator is mapped household weight on
strictly positive-mass people with non-default gift signal and
`person_support_channel == "spi"`; the denominator is all people's mapped
household effective mass. Base-channel signal cannot satisfy restoration. The
reviewed minimum share is **0.000001 (one part per million)**. This rejects
zero-weight support and numerical dust. The rarest populated owning-entity
record share in the pinned enhanced-FRS extraction is 0.000104; because that
comparator is unweighted, it is only a scale/headroom heuristic, not validation
of the weighted floor. The floor is committed in the manifest and runtime
together. Until a real rebuilt SPI channel passes it, `gift_aid` and
`charitable_investment_gifts` remain distributional/effective-weight gaps, not
restored inputs, even though their raw manifest status is `required`.

The family is now `required_at_build` in the generated release manifest. The
release manifest pins the source-manifest SHA-256, and runtime checks the full
source-stage contract against executable constants before source I/O. Both
charitable-giving columns remain `distributional_required`: the stage refuses
to call them restored unless SPI-channel signal clears the same floor, and the
final release input gate repeats that check immediately before the staging
write.

## Open source/replay blockers

These block an honest “restored” ledger entry and a production artifact:

- Licensed `put2223uk.tab` is unavailable here. No real donor SHA-256/size, QRF
  fit, or joint-draw replay exists.
- The source contract records the official HMRC ODS URL and hashes whatever
  local artifact a caller supplies, but no reviewed expected ODS SHA-256 and
  size are available to pin its identity. The parser now enforces sheets,
  fixed positions, all 13 ordered bands, positive cells, and the “All ranges”
  sentinel; exact published header strings also remain unverified until the
  real ODS is available.
- No national staging artifact has passed all 208 real targets and the weighted
  SPI charitable-giving floor. The claimed fiscal correction therefore remains
  unmeasured.

Per the adjudication, no replacement source or relaxed gate is substituted.
The family remains not restored, and real-source work stops at this boundary
until the conductor supplies/reviews the licensed donor and exact ODS identity.
