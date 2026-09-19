# Native hours source ownership

`qualify_current_survey_hours(preparation, age15_policy=..., under15_policy=...)`
connects the pure usual-hours proposals to the original ACS and ASEC source
records held by an authenticated survey preparation. Version 2 returns complete
selected-original-person hours for both arms. It preserves source observations
and labels explicit completion and empirical imputation separately. It does not
yet transport those values to clones, produce an all-person engine input or
modify a Frame.

The function captures the preparation's pinned ACS person archive and current
ASEC person member, checks their sizes and hashes, exhausts every CSV record
and archive member, and joins selected people by original survey coordinates.
Original ages must match the preparation. It derives the complete income-2024
age-15 donor key set from the issued ASEC coverage before selection and requires
the captured member's donors to match that exact set. The production roster has
2,174 people; matching that count alone does not qualify a different roster.

The age-15 empirical and under-15 zero-completion policies are explicit arguments.
The pure proposal module supplies their versioned names, seed, native-keyed draw
and interpretation of raw codes. Source allocation and response flags remain
distinct from modeled provenance, including valid `FL_665=0` supplement
nonresponse. Both future clones should receive the same original-person draw.
This source owner neither creates clones nor assigns hours to them.

The returned `QualifiedSurveyHoursProposals` holds private raw tables, proposals,
original source owners and an immutable receipt. Call `validate()` before
borrowing the result and after final relevant I/O. It checks implementation,
retained source identities, exact projection objects, table storage and nested
proposal contents, including ASEC earnings and the combined person projection.
A copied object, arbitrary callback, matching detached table
or receipt alone does not replace the retained object. Raw records and donor
assignments are private evidence; public diagnostics should use aggregate counts
and provenance summaries.

The receipt qualifies selected original-person hours and the ASEC own arm. It
deliberately leaves `all_person_engine_input_qualified`, `source_admission_issued`
and `release_eligible` false. The pure batches' scope flags remain false as well;
the enclosing live source owner supplies the checked relationship rather than
changing those flags. `person_hours` preserves the original-person index order
and includes the target, provenance and policy columns. The ACS `proposals`
and separate `asec_proposals` retain native keys and literals for inspection.

ASEC hours use the [separate observation and completion rules](asec-usual-hours-observations.md).
The under-15 zero assumption requires explicit policy and known-zero source
earnings; it is never relabeled observed zero. Missing adult hours refuse complete
projection construction. Historical engine defaults, last-week hours and prior
wages are not substitutes for current source evidence.

## Verification boundary

The tests use the real source issuers, preparation, private capture and owner
validation with tiny invented archives. A selected sample deliberately excludes
the sole age-15 donor household while the full donor cohort remains available.
Tests compare full versus selected proposals and cover explicit policy choices,
malformed/duplicate records, detached owners, replaced callbacks and receipts,
table and nested-proposal mutation, retained coverage mutation, changed seed,
and projection mutation during final source I/O. The fixture's one-donor limit
is a private pre-issuance test seam; production retains the 2,174-key requirement.

This change continues the native branch based on `fee4aac9d`, including pure
hours and response-status correction `4f34495e5`. Fresh main was inspected and
lacks these native APIs, so it is not an independent main-branch replacement.
Actual full-source owner execution, graph integration, clone transport,
calibrated release comparisons and publication remain separate. A complete
all-age ASEC source diagnostic supports the recoder but does not replace the
retained source owner or qualify its production execution.
