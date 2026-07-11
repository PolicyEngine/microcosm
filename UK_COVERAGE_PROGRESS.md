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

## Open restoration question

The HMRC adjudication miss is distributional, not an absent-column gap: total
income-tax liability is £334.629bn against the £277bn SPI-anchored benchmark
(+20.8047%), while every HMRC-family loader input has non-default signal. The
current repository has SPI support-row primitives but no UK national build
orchestrator, SPI donor-model execution, or national target-registry compilation
path; `tools/build_uk_rowwise_dataset.py` only clones a supplied compact H5.
Before recording an HMRC family restoration, add or identify that national
orchestration seam so the SPI QRF stages and HMRC income-band targets can be
wired without turning the geography-clone tool into a different build product.
