# Current survey amount successor

Implementation plan and source inventory, 12 September 2026. This is a local
successor to the retained survey PUF55 run, not an amendment to the frozen d35
native pilot. The branch starts at reviewed integration 99543c3.

## Immediate scope

| Native ASEC field | Canonical person input | Reporting evidence |
| --- | --- | --- |
| UC_VAL | unemployment_compensation | UC_YN receipt code; amount question universe UC_YN = 1 |
| PHIP_VAL | health_insurance_premiums_without_medicare_part_b | All persons |
| PMED_VAL | other_medical_expenses | All persons |
| POTC_VAL | over_the_counter_health_expenses | All persons |

These four names are absent from both PUF tax-detail ownership rosters. They
already have qualified current-money domains and direct amount mappings in
`cps_carried_current.py`. The complete legacy carry helper is not reusable:
it also writes existing PUF leaves and a legacy Social Security decomposition.
The modern `survey_social_security` source owner remains the authority for
Social Security; no age-62 zero fallback or prior wages is introduced here.

The current ASEC source is the authenticated original 2025 person member,
income year and price basis 2024. ACS observations use the already qualified
2024 source predictors, whose earnings refer to the rolling prior 12 months.
That temporal harmonization is a documented model choice, not equivalent
observation windows. No new donor dataset is needed.

## Amount and reporting status are different

The [2025 CPS ASEC dictionary](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf),
PDF page 50, defines UC_YN codes 0 (NIU), 1 (yes), and 2 (no), with the receipt
question asked of persons aged 15 or older. UC_VAL is asked of recipients;
its zero code conflates none with NIU. The maintained domain metadata pins
this dictionary to SHA256
`5cb80973326ef8b625fbaae70d80b0c641ce5d2b3911abd2fb4427abd5908a6f`.
The three health amounts have all-persons universes (PDF page 83).

A small bounded UC_YN projection reads the same exact authenticated original
member. Native person and household/line coordinates must join the original
current-money owner exactly; carried columns and detached JSON do not confer
source authority. The projection retains the original amount, amount validity,
source classification, zero_origin, literal receipt code and reporting status.
Today's upstream preparation requires complete qualified current money before
issuing its handle. Literal missing-amount controls test the join's unknown
semantics; they do not claim that a missing amount passes this unchanged parent
prerequisite. Reporting unknownness can still occur with numerically valid amounts.

UC attachment and donor policy:

- A valid positive amount with an in-universe yes code is known receipt.
- A zero amount with an in-universe no code is known nonreceipt.
- Yes with zero is an ambiguous amount, not known nonreceipt. It remains in
  the source projection but is excluded from fit and canonical attachment.
- NIU, blank/unrecognized codes, missing amounts, and contradictory amount/
  receipt pairs remain unresolved and are excluded from fit and attachment.
- Out-of-universe age does not infer a zero. ACS persons outside the receipt
  question universe also retain unknown UC rather than receiving an adult draw.

The health fields retain qualified valid all-persons amounts, including genuine
zero-dollar codes. No noanswer cell is converted to zero. Allocation and source
status remain evidence; known source values are not claimed to be unallocated.
PHIP_VAL is deliberately retained as the declared premium source, rather than
silently substituting PHIP_VAL2, whose treatment of inconsistent reported zero
premiums differs. The [Census health insurance user note](https://www.census.gov/programs-surveys/cps/technical-documentation/user-notes/health-insurance-user-notes/User-Notes-Selected-CPS-ASEC.html)
describes that distinction. Person-level conditional fitting preserves modeled
cross-component dependence; it does not certify household or insurance-unit
correlation, which still needs empirical validation.

## One family, explicit graph operations

The family has one parameterized source projection and attachment mechanism.
Each selected field declares its source transformation, donor eligibility,
recipient matrix, QRF fit and draw as graph nodes/artifacts. UC has its own
reporting mask. The health group uses an ordered PHIP → PMED → POTC conditional
chain, retaining observed health-component dependence through prior-target
conditioning instead of fitting the three amounts independently. This avoids
filling unknown UC to make a joint matrix finite. Fits use the existing regime-gated QRF implementation and
current source predictor qualification; donor weights remain original design
weights before allocation or cloning. No PUF value becomes an ASEC donor.

Draws are keyed to the original ACS source person, then deliberately shared by
its original and PUF support clones. This preserves source observations and
paired support interpretation; clone labels are checked during attachment.
Source amount values and reporting status are copied to both ASEC clones.
Every existing parent column, owner, axis, membership, atomic geography,
weight, design anchor and mass ledger must remain identical.

The parent is the exact `SurveyPuf55Run` checked by
`check_survey_puf55_run`. The original handle is retained and checked before
consumption and after source/store/export I/O. A checked JSON payload or copied
dataclass cannot issue authority. The extension copies kernel/source-codec
registries privately, preserves all prefix declarations/keys/artifacts, and
verifies actual observed populations on cold and required replay. Failed checks
revoke the original handle. The checked-output branch supplies this additive
API; the frozen native packet is left untouched.

## Verification

First run cheap invented controls: all UC code/amount/universe combinations,
health zero and unknown behavior, literal bounds and source-coordinate joins,
clone pairing, missing/duplicate source IDs, invalid source/clone channels,
ownership collisions and raw/status/zero-origin mutation.

Then an actual invented authenticated-source fixture runs the family through
real graph cold and required replay, QRF train/apply, coverage, complete Frame
store/export/readback, parent retention and unchanged prefix seals. Rejection
controls mutate source projection, parent handle/population, declared graph,
artifact bytes, fit/draw lineage and output attachment independently. Acceptance
requires closed receipts and independent review. No native run, engine outcome,
held-out quality, calibration or release certification is claimed by these tests.

## Remaining input inventory

The accepted invented PUF baseline has 161 required names, of which 100 are
absent. Its 61 present names are not necessarily complete: some PUF leaves are
populated only on PUF clones. Counts are inventory for that fixture, not a native
coverage measurement. The next stage reduces absent names while reporting
unresolved cells separately.

| Input family | Existing source producer or route | Integration requirement |
| --- | --- | --- |
| UC and three health amounts | authenticated current-money owner; direct `cps_carried_current` maps | This stage |
| Social Security components | `current_social_security_source`, `survey_social_security` | Preserve source reporting basis and unknown components; never legacy age split |
| Health coverage | Authenticated `current_survey_health_source` and explicit `graph_current_survey_health` fragment | This stage; seven narrow ACS concepts remain unknown where the survey cannot distinguish them |
| Disability, student status, children, veteran status | `eligibility_inputs` | Qualify literals and reporting universes before reuse |
| Parent relationships | `eligibility_inputs._own_children_in_household` resolves PEPAR1/PEPAR2 then drops links | Issue 884: retain parent_1_id/parent_2_id with household/line and clone joins; unresolved and absent differ; consumer issue 9404 is separate |
| Marital roles | `relationship_inputs` | Current source observations and clone preservation |
| Hours, work history, occupation, wage/tips | `hours_worked`, `org_wages`, `weeks_unemployed`, `sipp_tips` | Source/QRF qualification and explicit ownership; prior wages excluded |
| Child support, disability income, workers compensation | respective maintained runtime modules | Current source amount/universe qualification, then existing fit contracts |
| Housing, energy, care, education | `housing`, `energy_subsidy`, `adultcare`, education producers | Household/SPM source and clone-keyed joins; reconcile PUF tuition ownership |
| Financial wealth, vehicles, loans | SCF/SIPP producer modules | Actual donor authentication, existing fit dependencies and held-out quality |
| Retirement distributions/contributions | maintained distribution/contribution modules | Reconcile already PUF-owned taxable IRA/pension and self-employed pension leaves |
| Pregnancy, SSI disability, voluntary filing | maintained producer modules | Resolve source authority and distinct behavioral assumptions |
| Take-up | `take_up_contract` | Unsourced rates remain unresolved; no convenience constant fills |
| Mortgage details, marketplace benchmark ratio, remaining retirement details | Not yet mapped in this bounded inventory | Identify actual producer/owner or make an explicit source/model decision |

The inventory used tracked files and `rg --no-ignore` within explicit source
directories because ordinary searches can skip the tracked build package.
Existing function names establish implementation leads; they do not prove that
those producers are integrated into the current graph or current native file.


## Fixed country composition

`run_us_survey_enrichment` retains the original checked `SurveyPuf55Run` and
assembles the amount family and independently source-qualified health fragment.
Both attach to `survey_puf55.receiving`; the health CREATE and final attachment
explicitly consume `survey_amounts.attach/attachment`. Health adds nine coverage
inputs and their literal/status/knownness evidence after all four monetary leaves.
The host preserves the complete parent Frame, owners, design anchors and mass ledger,
compares every unchanged prefix version against the original parent manifest, and
requalifies both original-source projections before issuing its output. There is
one private issued-handle registry and one public checked country output, not a
family-level authority or executor. The health fragment's descriptive values do
not grant access to native sources.

The combined invented fixture includes the exact source ACS HINS/flags and ASEC
NOW/flags before authenticating source owners. It runs the original financial and
PUF stages, both enrichment fragments, a required replay, input-coverage diagnosis
and complete Frame store/readback. No native build, model engine execution,
calibration, or release approval is implied by these checks.


## Invented acceptance, 12 September 2026

The final production source passed the 271-node combined graph: 245 validated
prefix hits followed by 26 newly executed enrichment nodes, then a required
replay with all 271 hits. Complete parent columns, owners, design anchors,
geography and the mass ledger were preserved. The full resulting Frame was
stored and read back with exact physical comparison. The fixture has 18 people;
required-name coverage improved from 61 to 74 of the 161-input profile, leaving
87 missing names. Presence does not imply complete values or statistical quality.

All 18 people have the three health-cost values and an ESI observation. UC has
10 known and eight unresolved values. The narrow Medicaid-at-interview input has
eight known and ten unresolved values, preserving the ACS semantic gap. These
are counts from invented records, not estimates or native-data diagnostics.

Acceptance combines five passing ordered tests from the third full attempt and
one corrected, focused final parent-revocation test. The third attempt retained
its failed result because that last test named a nonexistent generic income
column; it stopped before mutating anything. The final test uses the actual
`employment_income_before_lsr` input and verifies permanent parent and child
revocation after mutation, including refusal after restoring the original value.
It rebuilds a real parent and one child, without repeating the already-passed
required replay or readback. The focused attempt passed with all 857 Python
source pins stable, 624.872 seconds wall time and 885,669,888 bytes peak RSS.

The separate 32-case cheap suite covers UC reporting classifications, literal
missingness, source/clone identity, ownership collisions and source-origin
serialization when `person_id` names both an index and a column. The serialization
binds the original int64 axis and its name separately from ordered column values.
Independent review covers the frozen UC source, health fragment, combined host
and this serialization correction. Preserved failed attempts remain part of the
local acceptance record; no native execution or release certificate is implied.
