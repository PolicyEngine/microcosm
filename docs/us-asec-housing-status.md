# ASEC interview-time housing status

This local source adapter preserves the ASEC household answers `HPUBLIC` and
`HLORENT`, their allocation flags `I_HPUBLI` and `I_HLOREN`, and an explicit
publisher-routed interpretation. It does not establish annual participation,
eligibility, a particular benefit program, rent, bedrooms, subsidy dollars or
an income-year bridge. Every artifact and composition receipt has
`release_eligible: false`.

The source meanings are bound to the public
[2023](https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asec2023_ddl_pub_full.pdf),
[2024](https://www2.census.gov/programs-surveys/cps/datasets/2024/march/asec2024_ddl_pub_full.pdf)
and [2025](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf)
ASEC dictionaries. Printed pages 6A-2 through 6A-4 define interview and tenure
context; pages 6A-9 and 6A-11 define the answer and allocation fields. The module
records the full dictionary hashes and zero-based PDF page indices. The income
cohorts 2022, 2023 and 2024 correspond to observed survey years 2023, 2024 and
2025. The loader reads `H_YEAR` and verifies that relationship.

## Interpretation

The dictionary codes are `0` (NIU), `1` (yes) and `2` (no) for both answers;
allocation flags have domain `{0, 1}`. The 2023 and 2024 printed questionnaires
number the public-housing answers 3/4, while the dictionaries encode 1/2. The
adapter follows the microdata dictionary, without claiming to resolve that
questionnaire discrepancy.

For an interview with tenure 2 (rented) or 3 (no cash rent):

| HPUBLIC | HLORENT | Status | Route |
| --- | --- | --- | --- |
| 1 | 0 | Receipt | Public housing |
| 2 | 1 | Receipt | Government-supported lower rent |
| 2 | 2 | Nonreceipt | Both applicable answers negative |
| 0 | 0 | Unknown | Missing parent answer |
| 2 | 0 | Unknown | Missing required lower-rent answer |
| 1 or 0 | 1 or 2 | Conflict | Lower-rent answer outside its parent route |

Owner interviews and consistently coded noninterviews with both answers zero
are outside the publisher universe. Substantive answers there are conflicts.
Tenure 0 or otherwise unavailable scope remains unknown. HRHTYPE 9/10 produces
a separate group-quarters flag; it does not exclude an otherwise valid answer.
This slice declares that it does not derive non-housing-unit status. The existing
authenticated attachment carries `asec_H_LIVQRT`, but it is outside this
adapter's deliberately smaller context roster.

Unsupported answer, allocation or context codes refuse before classification.
In-domain routing and allocation inconsistencies produce explicit conflict
bits. Allocation quality remains separate from status. A flag of 1 for a zero
or routed-out answer is a conflict. `receipt_valid` is 1 only for consistent,
known receipt or nonreceipt; consumers must retain this mask.

All zero codes retain
`zero_origin_policy=frozen_fillna_zero_origin_unresolved`. Int64 storage is
consistent with the inspected default-CSV producer having performed no missing
value fill in those columns. This is conditional on the inspected producer and
runtime actually having created the pinned HDFs; no historical execution proof
or original-CSV comparison is claimed. A zero allocation flag is therefore
not relabeled as a confirmed observed, unallocated response.
`zero_origin_evidence` records the inspected repository, commit, path and file
digest together with both unresolved evidence checks. Those fields bind an
inspection, not an authenticated historical producer execution.

## Authentication and replay

`load_authenticated_housing_status(source, cohort_paths=...)` requires an
`AuthenticatedCurrentMoneySource`. It reuses that source's fixed parent,
attachment and three cohort pins; the public API has no pin override. Each
cohort is streamed into a private snapshot, its digest is verified, and the
same snapshot is opened. The numeric reader projects only `H_SEQ`, `H_YEAR`,
the three context fields and the four answer/allocation fields from one aligned
int64 block. It does not open unrelated object blocks. Native keys must cover
the sealed household scope, and the freshly read context must match the carried
attachment exactly. Input paths are not stored in the housing receipt.

The status artifact has immutable byte buffers, a bounded canonical JSON
header and a trailing transport checksum. Its content identity commits to the
entire header and raw and derived buffers under a separate hash domain.
`decode_housing_status(payload, expected=...)` requires independently issued
complete expected content, compares every header and buffer byte, and only then
issues a replay object. A changed observation plus a recomputed transport
checksum cannot acquire authenticated status. Synthetic construction returns a
different type and cannot be attached through the production API.

The implementation identity covers the classifier, codec, source reader,
composition, money adapter, Frame implementation and their declared runtime
dependencies. Source evidence does not contain its own content commitment.
Receipts contain structural join counts, without status-frequency statistics.

## Composition after tax reconstruction

`attach_housing_status(tax_result, status)` validates the sealed current-money
tax result first. It creates a separate Frame and receipt, retains the identical
parent tax result and money object, and attaches only the distinctly prefixed
`asec_housing_observed_*` uint8 columns. Raw answers, keys and years remain
available in the authenticated status artifact. This adapter does not emit
operator-owned housing-assistance or tenure columns.

Dropping only those new columns must recover the parent's exact Frame
signature. Entity relationships, all other observed columns, metadata, strata,
weights and mass history remain unchanged. Both values and pandas axis buffers
are detached. `validate_attached_frame` also checks a Frame loaded through the
actual checkpoint codec against its trusted tax result and status parents.
Persisting a Frame alone does not reconstruct those parents or grant authority.

The tests use three tiny invented cohort HDFs and explicit private test-pin
substitution. They exercise the real checkpoint readers/writer and pinned
microunit tax construction. They do not authenticate Census rows or demonstrate
calibration, release readiness or full-source runtime/memory performance.
