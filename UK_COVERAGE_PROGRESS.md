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
| HMRC/SPI income family | No raw-column status change; adds one `required_at_build` distributional family, including effective-mass requirements for both charitable-giving inputs | Certified-base hash gate; 2022–23 SPI QRFs; complete 2023–24 Tables 3.6/3.7 surface (208 facts); positive-prior, calibration, and source-manifest tests | Implemented and gated; production replay awaits caller-supplied licensed `put2223uk.tab` |

## Restoration diagnosis

The HMRC adjudication miss is distributional, not an absent-column gap: total
income-tax liability is £334.629bn against the £277bn SPI-anchored benchmark
(+20.8047%), while every HMRC-family loader input has non-default signal. The
current repository has SPI support-row primitives but no UK national build
orchestrator, SPI donor-model execution, or national target-registry compilation
path; `tools/build_uk_rowwise_dataset.py` only clones a supplied compact H5.
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
the SPI/HMRC stage. It emits input-coverage and HMRC source/calibration evidence
beside the staging file; a failed gate does not create a failed staging H5.

This seam does **not** clone households, assign local geography, compile local
targets, publish a release, or change `tools/build_uk_rowwise_dataset.py`. The
geography-clone tool remains a downstream, separate build product. National
calibration and additional source families are also out of scope until a family
explicitly wires them through this stage boundary.

## HMRC family adjudications

Status: implemented and wired on 2026-07-11. The repository does not contain
the licensed UKDS `put2223uk.tab`, so this milestone proves the complete stage
with fail-closed source and integration tests but does not claim a production
national artifact was rebuilt locally. No substitute donor was used.

The enhanced-FRS reference is pinned to `enhanced_frs_2023_24.h5` at model
revision `655dd07e4bb9c777b00dac044949611f1feb824f` (SHA-256
`584ae33d…`). The certified Populace candidate already contains SPI support
rows, but all 200,000 SPI-synthetic household rows have zero calibrated weight.
Populace calibration cannot lift a zero initial weight because it optimizes its
log and caps the result relative to that zero. Porting the QRF draws alone would
therefore be fiscally inert; a reviewed positive-prior stage is required first.

The national base is the certified Populace UK candidate. Its existing
zero-weight SPI rows will be dropped and replaced once with a new SPI support
channel. A reviewed share of the unchanged national household mass will be
allocated to that channel as importance weights and recorded through a
deliberate `MassChangeRecord`; there will be no dead or duplicate SPI support.

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
2023–24 HMRC Personal Incomes Tables 3.6 and 3.7, explicitly mapped to the
candidate's build period. Table 3.7 includes bank and building-society
interest, so `savings_interest_income` is part of the required emitted family.
The materializer will fail closed if any declared component or band is absent.

The implemented first stage uses the enhanced-FRS pipeline's `FACT`-weighted
100,000-row bootstrap and then uniform typed design weights; applying `FACT`
again after that bootstrap would square the survey weights. It extends the
incumbent six-income QRF with SPI `OTHERINV` because Table 3.7 publishes that
eighth component, and emits `GIFTAID` and `GIFTINV` separately. `INCBBS` is
taxable interest; after the second-stage QRF, the PolicyEngine gross input is
reconstructed as `INCBBS + tax_free_savings_income`. The second-stage fit uses
household-to-person importance weights and requires every materializable
enhanced-FRS output. Two fields absent from the pinned candidate are reviewed
explicitly rather than silently skipped: `incapacity_benefit_reported`
(absent/all-default) and `maternity_allowance_reported` (absent). Disability
category and flag inputs are recomputed from the newly drawn reported amounts.

The HMRC calibration compiles **8 components × 13 bands × 2 measures = 208**
facts. It uses `income_tax > 0` from a PolicyEngine-UK person-mapped simulation
as the published taxpayer universe. Band assignment uses
`hmrc_spi_assessable_income`, the sum of the eight published components, not
PolicyEngine's broader `total_income`. Every fact must have positive-mass
support, every fact must compile, household mass is conserved, the per-record
weight-ratio cap is 5 (the US dense-release guard), and no target may miss by
more than 5%. Runtime SHA-256 values for the private donor and public ODS, the
content-addressed target-registry version, fit weight kinds, solver options,
and effective-mass shares are written to the HMRC evidence sidecar.

Effective-mass coverage uses `household_weight`, mapped to the owning entity.
A required column must place non-default signal on strictly positive mass and
reach a reviewed minimum mass share of **0.000001 (one part per million)**.
This rejects zero-weight support and numerical dust while sitting about 100
times below the rarest populated record share in the pinned enhanced-FRS
reference (0.000104), leaving substantial headroom for genuinely rare inputs.
The floor is committed in the manifest and runtime together. Until the rebuilt
SPI channel passes it, `gift_aid` and `charitable_investment_gifts` are recorded
as distributional/effective-weight gaps, not restored inputs, even though their
raw manifest status remains `required`.

The family is now `required_at_build` in the generated release manifest. Both
charitable-giving columns remain `distributional_required`: the stage refuses
to call them restored unless their non-default signal clears the same
one-part-per-million floor, and the final release input gate repeats that check
immediately before the staging write.
