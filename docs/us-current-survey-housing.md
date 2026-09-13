# Housing participation in the common US survey graph

The common ACS–ASEC graph connects independent survey housing observations to
two country-model inputs: `receives_housing_assistance` and
`takes_up_housing_assistance_if_eligible`. Both inputs attach to the modeled
assisted family's SPM unit. The country model calculates program awards,
allocates those awards across SPM units and applies the SPM resource cap.

`graph_us_survey_enrichment.run_us_survey_enrichment` includes this connection
after the current survey amount and health fragments. It uses the retained
source preparation and original common frame; it does not construct a separate
ASEC–PUF population or infer participation from `SPM_CAPHOUSESUB`.

## Observations and modeled inputs

The qualifier reads the authenticated original ASEC household and person
members and ACS household and person members already owned by source
preparation. The existing ASEC housing classifier distinguishes known receipt,
known nonreceipt, unknown, not-in-universe and conflicting responses. The
attached household evidence preserves literal fields, status, allocation
quality, knownness, source year and survey year.

`housing_observed_receipt` remains nullable. `housing_receipt` is the completed
model input, and `housing_receipt__origin` records whether it came from an
observation, a modeled exclusion or imputation. A modeled value never changes
the source observation's knownness.

The projection records these assumptions:

| ID | Assumption |
| --- | --- |
| A1 | The household reference person's SPM unit represents the assisted family. |
| A2 | Interview-time participation carries through the preceding income year. |
| A3 | Public-housing and reduced-rent reports represent the modeled HUD assistance family. |
| A4 | ACS group quarters are excluded from this modeled household-assistance family. This is a modeled exclusion, not observed nonreceipt. |
| A5 | Independently observed owner households are excluded from this modeled public/lower-rent family. Positive or conflicting source answers cannot be overridden by this assumption. |

ASEC `A_EXPRRP` and ACS `RELSHIPP` identify the reference person. Row order and
ASEC `P_SEQ` do not establish that role. A household with an ambiguous reference
role refuses attachment. Explicitly classified group quarters may lack a
reference person only when their completed modeled receipt is false.

## Household imputation and attachment

Known ASEC households supply one binary label per household to the existing
regime-gated QRF operator. The donor frame retains complete households and
their original household design weights before geographic allocation and
cloning. Predictors are reference-person age, household size and household
employment and self-employment income. When the parent qualifies demographic
conditioning, the corresponding reference-person predictors also participate.

The model fills only unresolved household participation cells. One draw per
original household attaches to both of its clones through explicit source and
clone identities. If every household already has a completed input, the graph
omits fitting and application. The connection preserves the receiving frame's
existing columns, entity memberships, weights and mass ledger.

Only the reference person's SPM unit receives the two true participation
flags. Other SPM units sharing that household receive false. No program-dollar
or SPM-dollar variable is produced by this fragment.

## Verification and release scope

The host binds source projections, donor values and weights, recipient matrix,
fitted-model history and attachment outputs. It reconstructs each population
change during readback and checks live implementation and configuration around
owner callbacks. Mutating a retained observation, source mapping or output
invalidates the affected result.

The focused tests use invented source files and Frames. They exercise source
knownness, household donor grain, reference-person routing, clone attachment,
cap independence and graph replay. These code checks do not establish
imputation quality on national data or a revised 2025 poverty rate. The
projection and attachment therefore carry `release_eligible: false` until the
separate native population, country-model, calibration and release checks are
complete.
