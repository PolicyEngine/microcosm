# Current health coverage on the survey multispine

This lane adds source-qualified interview-date coverage to the common ACS+ASEC
frame and its PUF clones. It does not estimate eligibility, annual receipt,
insurance costs, or a common observation date. The current ASEC arm is income
year 2024 / survey year 2025; ACS coverage was observed during 2024. Older ASEC
income-year vintages are outside this first attachment.

## Source definitions and mapping decisions

The [2025 ASEC dictionary](https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar25.pdf)
defines these current recodes for all persons, with 1=yes and 2=no. Corresponding
`I_NOW_*` flags distinguish reported, hot-deck, logical, and whole-unit imputation.
The [2024 ACS questionnaire, item 16](https://www2.census.gov/programs-surveys/acs/methodology/questionnaires/2024/quest24.pdf)
asks all respondents about current coverage. The [ACS dictionary](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.pdf)
defines HINS1–7 (printed pages 37–38) and allocation/coverage-edit flags (126).

| NATIONAL_CD input | ASEC source | ACS exact source in this first stage |
| --- | --- | --- |
| `has_esi` | `NOW_GRP` | `HINS1` |
| `has_marketplace_health_coverage_at_interview` | `NOW_MRK` | Gap: `HINS2` combines direct-purchase types |
| `has_non_marketplace_direct_purchase_health_coverage_at_interview` | `NOW_NONM` | Same gap |
| `has_medicaid_health_coverage_at_interview` | `NOW_CAID` | Gap: `HINS4` includes CHIP/other means-tested coverage |
| `has_other_means_tested_health_coverage_at_interview` | `NOW_OTHMT` | Same gap |
| `has_tricare_health_coverage_at_interview` | `NOW_MIL` | Gap: `HINS5` includes other military coverage |
| `has_champva_health_coverage_at_interview` | `NOW_CHAMPVA` | No separate ACS item |
| `has_va_health_coverage_at_interview` | `NOW_VACARE` | Keep `HINS6` distinct pending CHAMPVA recode reconciliation |
| `has_indian_health_service_coverage_at_interview` | `NOW_IHSFLG` | `HINS7` |

The existing `cps_carried._fill_health_coverage_inputs` supplies the field roster
and most ASEC mappings, but its absent-column-to-zero helper is unsuitable here.
Its Medicaid mapping uses `NOW_MCAID`, the broader Medicaid/CHIP/other-means-tested
aggregate, rather than `NOW_CAID`. Both names and their distinction also appear
in the [2023](https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar23.pdf) and
[2024](https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar24.pdf) dictionaries.
The old pipeline remains untouched; this stage refuses ownership collisions.

This narrower mapping is supported by the pinned consumer source, rather than
inferred from the input name alone. PolicyEngine-US 1.819.0 declares the Medicaid
input as person-level current Medicaid coverage, declares other means-tested
coverage separately, and uses the Medicaid input alongside `receives_medicaid`
for California Medi-Cal continuity. The checked files are
`variables/household/expense/health/has_medicaid_health_coverage_at_interview.py`
(SHA256 `0500c4e86120b540e6e2f09fc108123758d8f9863c0e4e4bf94b178dd50e09b8`)
and `variables/gov/states/ca/chhs/is_ca_medicaid_immigration_status_eligible.py`
(`6902d01792b8d0902c2693edb78f5ba2554a34f95d4073bd4e85e4ad69ce41a5`).
Only source text was inspected; this is not an engine execution or a blanket
reinterpretation of earlier datasets. Both `NOW_CAID` and `NOW_MCAID`, with their
allocation flags, remain in the graph. A later CMS comparison must declare
whether its target covers Medicaid alone or Medicaid plus CHIP.

The [ACS subject definitions, health insurance](https://www2.census.gov/programs-surveys/acs/tech_docs/subject_definitions/2024_ACSSubjectDefinitions.pdf)
describe source editing and the 2019 change to current VA enrollment wording.
The [Census programming guide](https://www.census.gov/topics/health/health-insurance/guidance/programming-code/acs-estimates.html)
still groups CHAMPVA with HINS6. Consequently this stage does not assume that
HINS6 is interchangeable with the narrower ASEC VACARE recode. Raw HINS2–6 stay
inspectable; no missing concept becomes a false flag. `HIMRKS` measures subsidized
direct purchase, not all Marketplace enrollment, and cannot repair that split.

## Graph and authority

Borrow the original checked `SurveyPuf55Run` and its retained survey preparation.
Capture the exact original ASEC PERSON member and ACS person archive under the
existing native owners' pins. Match source household/person coordinates to the
preparation's original person ledger; never join by row position. Record the
source codes, allocation literals, status, knownness, source year and survey year.

The fragment exposes a source projection artifact, a CREATE source population,
literal code columns, and nine named per-field recodes. CREATE requires the
existing `survey_population_source` reference from the original source registry;
it introduces no additional source file or issuer. A post-PUF attachment follows the retained
original-person link to both clones, verifies native channel/ID and clone roster,
and adds nullable booleans plus provenance. This stage has no fitted model.
Future disaggregation models need separate declarations and assessment.

One fixed country enrichment host assembles the amount and health fragments over
the original checked PUF handle. The health fragment receives the explicit
receiving version and the amount attachment's typed artifact edge. Its ordinary
attachment preserves all incoming columns, including amount unknowns. It does
not return a separate whole-population branch or issue a public health handle.

Portable values do not issue authority. The owning consumer must requalify the
retained parent immediately before consumption and after its last relevant I/O,
then seal the complete successor before returning it. The fragment checks its
retained physical values around each kernel callback; callback success itself
grants no source authority. The host owns source and kernel registries, typed
artifact/store bindings, replay checks, and complete successor verification.

## Bounded verification

The 31-case suite passes over invented records. It exercises the real source
catalogues, preparation issuers, original-member qualifier and requalification;
literal yes/no/allocation and unknown handling; exact source-coordinate joins;
clone identity and collision refusal; all prior amount-column preservation;
and a 13-node actual graph cold run plus required replay. Independent complete
population reconstruction matches every observed health-node output. The graph
fixture's qualified value is explicitly descriptive; it does not claim a new
source admission. A separate test obtains the qualified value through the real
issuer path using privately pinned invented source bytes.

The source-v3 guarded run closed with result 0: 20.61 seconds wall, 19.95 seconds
CPU and 490.9 MB peak RSS. Earlier failed fixtures remain preserved, including
the origin-coordinate type mismatch exposed by the real qualification test.
No native microdata, country engine, public upload or model benchmark was used.
This verifies the fragment, not a launchable population. Actual-source replay,
the combined host's final retained-parent fence and release coverage checks
remain its consumer's responsibilities; the seven unresolved ACS concepts stay
explicit gaps.
