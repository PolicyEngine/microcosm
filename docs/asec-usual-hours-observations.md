# ASEC usual-hours observations

`current_asec_usual_hours.recode_asec_usual_hours` interprets original March 2025
work-history literals across published ages 0-85. It returns source observations
without source admission, imputation, weights, graph execution or Frame mutation.

The [2025 Census person dictionary](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf)
defines `HRSWK` on printed page 6C-17 (PDF page 38) as usual weekly hours during
weeks worked. Its universe is positive `WKSWORK`. Zero is NIU, and 99 represents
99 hours or more. Printed page 6C-20 (PDF page 41) distinguishes initial `WORKYN`,
the temporary/part-time follow-up `WTEMP`, and final recode `WRK_CK`. The work
questions cover people aged 15 and over; the follow-up applies to initial no
answers. Both pages were visually checked. The downloaded PDF SHA-256 is
`5cb80973326ef8b625fbaae70d80b0c641ce5d2b3911abd2fb4427abd5908a6f`.

These definitions support three distinct outcomes:

| Source evidence | Result |
| --- | --- |
| Positive hours and weeks, coherent final work history | Preserve source hours and raw top code |
| Age 15+, zero hours/weeks and coherent final nonwork | Zero-hours completion labeled as source nonwork |
| Under 15 with NIU history, or adult with entirely NIU history | Unresolved hours with distinct universe/unanswered labels |

The recoder rejects contradictory histories and invalid codes. An initial no
answer alone does not establish final nonwork. A NIU follow-up alone is not an
explicit no. Shared history and allocation-code checks keep the separate age-15
donor recoder aligned without disguising an older person as a teenager.

Original allocation flags and supplement response status remain separate from
the result's provenance. The [Census variable metadata](https://api.census.gov/data/2025/cps/asec/mar/variables/FL_665.json)
includes complete supplement nonresponse as `FL_665=0`; accepting the code does
not mean a respondent supplied the value. A zero person weight does not erase a
source observation. Positive weight remains required by the separate empirical
donor qualification routine.

The all-NIU adult outcome is a conservative missing-information classification,
not a claim that NIU is a complete valid adult answer. This behavior must be
audited against the full source before graph integration. No earnings value,
last-week hours or historical engine default fills a gap here. In particular,
under-15 zero imputation needs a separately explicit assumption and earnings
conflict checks. This routine is not the all-person release input producer.

`propose_asec_usual_hours` adds a separate, explicit completion operation. It
requires the original wage, nonfarm self-employment and farm self-employment
literals. It preserves observed adult hours independently of earnings. For an
under-15 person with NIU work history, it produces zero only when the caller
selects `us_hours_under15_zero_completion_v1` and all three earnings fields are
known zero. Nonzero earnings, missing columns or unknown earnings refuse that
completion. An unanswered adult remains unresolved and cannot produce a complete
proposal. The child's result stays labeled `under15_explicit_modeled_zero`.

An aggregate diagnostic of the complete current ASEC person member found
142,125 records: 72,460 with positive source hours, 41,986 with source nonwork,
and 27,679 under-15 records with NIU history. No work-history contradiction or
zero/missing-hours record with nonzero earnings appeared in those three fields.
The diagnostic source hash matches the maintained 2025 member pin:
`06921fe83fc66c907e6c7b86b82255dc70458ee7d76258fc48297cb34f0c06b5`.
These are unweighted source counts, not population estimates. A source scan does
not issue source authority or qualify a release.

The version-2 retained source owner binds this observation/completion code and
its protocols, preserves raw earnings alongside work history, and returns a
complete selected-original-person hours projection for both survey arms.
Transport to clones, graph integration and all-person engine input qualification
remain separate. Public release flags remain false.
