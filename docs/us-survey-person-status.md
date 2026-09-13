# Survey person-status observations

The additive person-status qualifier retains published difficulty and enrollment
items from the original ACS 2024 and ASEC 2025 person sources. It does not assign
`is_blind`, `is_disabled`, or `is_full_time_college_student`. It is independent of
the existing legacy eligibility operator and is not attached to a graph host or
included in release coverage by this change.

`qualify_current_survey_person_status(preparation)` accepts the actual retained
`AuthenticatedSurveyPopulationPreparation`. It captures its pinned original
members, selects people through exact original household/person/line identities,
and checks each original age against the unchanged `A_AGE` column. It does not use
a normalized or imputed `age` as the item universe. Twenty-two-digit ASEC native
person keys remain strings; stacked person IDs retain their integer index.

The result contains the original Frame reference, an origin table, literal source
columns, descriptive observations, and an immutable receipt. The result owns no
source authority. Its `validate()` method rechecks the actual preparation and
student owners and compares all retained value seals after the last owner I/O.
A consuming host must keep those owners, validate after its own last relevant
I/O, and independently verify any materialized descendants and replay. Serialized
rows or a copied receipt do not authorize a source or a graph execution.

## Meaning and period

| Observation | Definition | What it does not establish |
| --- | --- | --- |
| `survey_vision_difficulty` | The published answer to blindness **or serious seeing difficulty**, with source and period retained. | Statutory blindness, measured visual acuity or visual field. |
| Other `survey_*_difficulty` items | The particular hearing, cognitive, ambulatory, self-care and independent-living questions. | A program-specific disability determination. |
| `survey_any_applicable_difficulty` | Any known yes among applicable items; false only when every applicable item is a valid no. Completeness and universe contradictions are separate diagnostics. | A medical, duration or eligibility finding. |
| `survey_publisher_disability_recode` | `PRDISFLG` or `DIS`, retained separately from the item-derived battery. Agreement is reported only when both are known. | An independent clinical observation. |
| `survey_full_time_college_student_last_week` | A coherent ASEC enrollment, college and full-time combination for the preceding survey week in 2025. | Annual student status in the 2024 income year or five calendar months of attendance. |
| `survey_college_attended_last_3_months` | ACS school attendance with undergraduate/graduate level in the preceding three months. | Full-time workload. ACS does not measure that here. |

ASEC difficulty items use the published `PRPERTYP = 2` universe, rather than a
replacement age threshold. The CPS publication may retain an earlier rotation
response; its survey year does not imply a fresh March interview answer. ACS
hearing/vision apply at all ages, cognitive/ambulatory/self-care at age 5+, and
independent living at age 15+. Out-of-universe items remain unknown, not false.
A positive or other non-NIU code outside that universe prevents an item-derived
battery answer and remains visible as a contradiction.

For ASEC student items, the printed enrollment universe is ages 16–54; the
named zero explicitly includes NIU, children and Armed Forces. The descriptive
recoder therefore also rejects positive enrollment routes for those named
person types and leaves an unresolved type unknown. This is an additional
published-code consistency check, not an alteration of the existing control
issuer's age-only assertion. Missing or unreadable student literals remain
unresolved even outside the age universe.

All values retain literal/code/parse states. Printed but unnamed numeric range
members remain unresolved. ASEC `PXDIS*` transitions distinguish no change,
longitudinal retention, allocation and changes to blank; nonzero is not a common
allocation flag. Valid published yes/no values survive unknown edit provenance
without becoming claims about an unallocated original answer. ACS `FSCHP` retains
its enrollment scope and the separate `FSCHGP` flag records grade-attending
allocation. The initial source-review packet overlooked `FSCHGP` on dictionary
p.129; this implementation corrects that omission using the same pinned PDF.

## Source contracts

- [ASEC 2025 dictionary](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf), SHA256 `5cb80973326ef8b625fbaae70d80b0c641ce5d2b3911abd2fb4427abd5908a6f`: student values pp.23–24, difficulty values pp.26–27, person type and summary p.28, student allocation pp.28–29, edit transitions p.30 (one-based PDF pages).
- [ACS 2024 dictionary](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.pdf), SHA256 `929c2752995b0af1c16d5c64de8cdc43b4aa7d388ee2d45b4b4df90fecce1dff`: difficulty p.36, attendance/level pp.43–44, summary p.56, allocation pp.124–125 and129–130.
- [ACS 2024 household questionnaire](https://www2.census.gov/programs-surveys/acs/methodology/questionnaires/2024/quest24.pdf): attendance items10a/b, p.12; difficulty items18–20 and their age gates, p.14. This does not claim separate verification of the group-quarters instrument or every interview mode.
- [BLS disability FAQ](https://www.bls.gov/cps/cpsdisability_faq.htm): rotation-period retention and the distinction between the statistical battery and SSA eligibility.

The minimal readsets are declared once in `current_survey_person_status.py` and
recorded in the result receipt. `SCHL` is excluded because it measures completed
attainment. ASEC `A_ENRLW`/`A_FTPT` are retained as current-cohort literals as well
as cross-checked against the maintained `asec_student_controls` issuer. Calling
that issuer consumes its existing **three-cohort** contract (income years
2022–2024); only the matched 2024 cohort is selected into this projection. The
extra current-cohort `A_HSCOL` and allocation fields are authenticated from the
pinned original member rather than assumed to be covered by the control issuer.
Where the parent retains known `A_HSCOL` or `PEDIS*` cells, the qualifier also
checks their numeric representation against the captured member. Missing carried
cells impose no observation; matching NIU or unnamed numeric codes only establish
representation consistency. They remain unknown under the named observation
domains. Per-field comparison counts are recorded without changing the parent.

Code columns have nullable integer dtype even when every code is unknown. The
receipt records actual table columns, named observations, difficulty mappings,
allocation vocabularies and the applicable source periods alongside the table
digests. This metadata describes the projection; a stored receipt cannot replace
the retained owner or its validation callback.

The source modules perform no downloads, no fitting, no tax/benefit calculations,
no cloning, and no graph or dataset publication. A later legal or annual bridge
must be a separate declared operation with its assumptions and validation.

## Validation scope

The committed tests use invented literals and privately pinned small source
fixtures. They cover universe boundaries, partial knownness, named allocation
transitions, enrollment routing, exact keys/ages, copied owners and mutation
seals. This patch was prepared under a source-only boundary: static checks passed,
but these new tests have **not yet been executed**. Bounded runtime acceptance and
any integration into the country graph remain separate work.
