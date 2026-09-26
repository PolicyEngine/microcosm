# Native usual-hours proposals

`microcosm.build.us_runtime.current_survey_hours` provides a pure first slice
for a future native `weekly_hours_worked_before_lsr` graph leaf. It reads no
files, changes no Frame, authenticates no source and grants no release authority.
The implementation continues reviewed native source
`fee4aac9d7be2643fe02ec6b855d231abe5980a4`; that continuation contains APIs not
yet available on main. No parser, native owner, graph host, benchmark, dependency
or accepted-source pin changes in this slice.

`recode_acs_usual_hours(row)` accepts literal source fields for one original
2024 ACS person. It preserves positive WKHP, including40 and99; retains FWKHP
allocation status; and distinguishes source-supported nonwork from unavailable
survey-universe values. FTP blank is NIU; numeric0 is not a valid WKHP answer.
Under16 alone does not establish zero work. Current ESR and last-week hours
are not used. Missing required headers and contradictory codes refuse.

`recode_asec_age15_hours(row)` accepts one supplied original March2025/income2024
age15 donor. It checks HRSWK/WKSWORK and final WRK_CK/WTEMP history, allowing
temporary work after initial WORKYN2. Source allocation flags stay separate.
The [Census 2025 FL_665 codebook](https://api.census.gov/data/2025/cps/asec/mar/variables/FL_665.json)
defines 0 as complete supplement nonresponse, alongside codes 1, 2 and 3.
All four codes are retained as valid source response-status evidence. This
recoding does not exclude nonrespondents or relabel their values as observed;
donor-cohort qualification remains the source owner's separate responsibility.
WTEMP0 is NIU, not an explicit no: positive hours/weeks and final WRK_CK1 remain
consistent with WORKYN2 unless WTEMP2 explicitly contradicts work. This retains
the reviewed donor rule, without claiming any supplied record actually occurred.
The recoder requires positive raw MARSUPWT integer units; it does not qualify
that weight or donor identity against actual files. This is deliberately not
the all-age ASEC hours producer.

`propose_acs_usual_hours(rows, donors=..., age15_policy=...,
under15_policy=...)` returns frozen proposals for original source people.
Rows carry explicit raw literals, source/construction labels and optional donor
identity/allocation evidence. Source values are preserved; missing adult hours
refuse. Age15 completion requires explicitly selecting
`acs-age15-empirical-full-asec2025-v1`. Under15 modeled zero separately requires
`us_hours_under15_zero_completion_v1`, refuses positive wages or nonzero
self-employment, and retains unknown earnings as unknown. Neither policy is
enabled by default. No missing-hours40 fallback exists.

The empirical CDF arithmetic and seed20260914 follow the previously reviewed
pure hours helper. Draws now bind original native source identity, not historical
population row positions. ACS keys contain survey/income year, literal SERIALNO
and numeric SPORDER. ASEC donor keys contain survey/income year, PH_SEQ, PERIDNUM
and A_LINENO. Duplicate keys and inconsistent duplicate ASEC coordinates refuse.
Ties in donor hours use canonical native identity; recipient and donor ordering
or batching do not change draws. These new draws need not match dense historical
draws. A later clone adapter must transport each original person's proposal
through checked ancestry instead of independently drawing for clone identities.

The helper labels every supplied donor cohort **unauthenticated**, even when it
has2174 rows. The future source owner must bind the complete reviewed March2025
cohort, actual source bytes and roles, native joins, allocation/history projection,
policies and implementation; counts and caller-provided hashes cannot establish
that authority. The chosen full-source empirical model is not a generic ASEC15+
fallback, and these proposals do not attest model quality for native recipients.

This slice preserves the separation between the ACS native value, its allocation
status, source nonwork completion, modeled under15 zero and empirical age15 draw.
WKL allocation status is outside its supplied fields, so no unallocated-answer
claim follows from source nonwork. Prior wages are not consumed. Last-week hours,
entity structure, weights, geography and other inputs are outside this helper.

Invented tests cover code/universe refusals, source40/99 preservation, explicit
policies, temporary-work history, allocation provenance, empirical boundaries,
duplicate native identity refusal and permutation/batch invariance. Native
source qualification, parser retention, graph materialization/replay, per-origin
recipient signal gates, actual-data validation and release acceptance remain
separate work.
