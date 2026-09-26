# Native hours source ownership

`qualify_current_survey_hours(preparation, age15_policy=..., under15_policy=...)`
connects the pure usual-hours proposals to the original ACS and ASEC source
records held by an authenticated survey preparation. Version 2 returns complete
selected-original-person hours for both arms. It preserves source observations
and labels explicit completion and empirical imputation separately. The source
qualifier does not modify a Frame; the graph fragment described below owns
attachment to the receiving population.

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

`borrow_cloned_hours_columns(qualified, receiving_people)` reuses the existing
exact two-clone attachment contract. It transports each original proposal and
its provenance to both descendants without redrawing hours. It checks original
IDs, native IDs, survey channels, complete clone pairs and output ownership.
It validates the retained source owner before and after borrowing, then checks
that source I/O changed neither the recipient table nor the borrowed output.
The returned Series are independent copies. This helper does not issue receiving
population authority: the graph host retains and authenticates its own receiving
owner and binds these columns to execution and replay.

## Graph execution and replay

`graph_current_survey_hours` declares four ordinary operations on the receiving
population. `survey_hours.source` retains the authenticated source projection in
a private artifact. `survey_hours.asec_recode` executes the ASEC recoder and
explicit under-15 completion. `survey_hours.acs_recode_impute` executes ACS
recoding, the original-keyed age-15 empirical draw and under-15 completion.
`survey_hours.attach` checks the two column artifacts against the independent
source-owner result and attaches hours, provenance and policy to both clones.

The attachment reads only the four ancestry and survey-channel columns. It owns
`weekly_hours_worked_before_lsr`, `hours_provenance` and `hours_policy`; existing
ownership refuses before execution. Public graph parameters contain source
hashes, survey periods, policy identifiers and the seed. Raw literals, identifiers
and donor assignments remain in private artifacts.

`graph_us_survey_enrichment` appends these operations after housing attachment.
Its final population is the hours attachment output. On cold execution and
required-cache replay it compares every hours artifact and intermediate
population with expectations derived from the independent retained source
owner. The host checks current source and implementation ownership after final
store I/O. Its receipt records the hours source and attachment hashes and both
policies, while release eligibility remains false. Prior wages are not inputs.

## Verification boundary

The tests use the real source issuers, preparation, private capture and owner
validation with tiny invented archives. A selected sample deliberately excludes
the sole age-15 donor household while the full donor cohort remains available.
Tests compare full versus selected proposals and cover explicit policy choices,
malformed/duplicate records, detached owners, replaced callbacks and receipts,
table and nested-proposal mutation, retained coverage mutation, changed seed,
and projection mutation during final source I/O. The small graph tests also run
cold and required-cache execution, check clone equality after reordering and
preserve existing tables, memberships and weights. They refuse changed private
artifacts, copied or revoked owners, output collisions and mutation during the
final host callback, and check that policy changes invalidate downstream keys.
The fixture's one-donor limit
is a private pre-issuance test seam; production retains the 2,174-key requirement.

This change continues the native branch based on `fee4aac9d`, including pure
hours and response-status correction `4f34495e5`. Fresh main was inspected and
lacks these native APIs, so it is not an independent main-branch replacement.
Actual full-source owner and integrated-host execution, calibrated release
comparisons and publication remain separate. The hours fragment has been tested
without rerunning the financial and full PUF fits; that bounded test does not
establish native production qualification or David's hours-distribution gate.
A complete
all-age ASEC source diagnostic supports the recoder but does not replace the
retained source owner or qualify its production execution.
