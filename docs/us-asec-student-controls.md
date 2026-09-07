# Observed ASEC student controls

The pooled source omits `A_ENRLW` and `A_FTPT` for every income-2022 and
income-2023 person. Both columns exist for income 2024. Money readiness does not
establish that these nonmonetary inputs are complete. Tax preparation refuses
missing controls with `MICROUNIT_CONTROL_MISSING` and the affected field name.
Nonnumeric or boolean storage remains `MICROUNIT_CONTROL_DTYPE`; nonfinite or
nonintegral values remain `MICROUNIT_CONTROL_VALUES`.

The separate S attachment restores the two observations from the same three
code-pinned Census person CSV members used by the PTOTVAL restoration. It does
not modify P, H, canonical T1, the monetary source schema, or prepared money.
The original missing columns remain missing. An owned composition adds only
`asec_A_ENRLW` and `asec_A_FTPT`. The microunit adapter accepts that exact
authenticated composition and projects those aliases into its transient input.
An incomplete source without the composition still refuses.

## Source meaning and coverage

The [2023](https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asec2023_ddl_pub_full.pdf),
[2024](https://www2.census.gov/programs-surveys/cps/datasets/2024/march/asec2024_ddl_pub_full.pdf),
and [2025](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf)
public dictionaries, person section 6C-2, define these as preceding-survey-week
school enrollment controls. `A_ENRLW` has an age universe of 16–54 and codes
0 (not in universe, including children and Armed Forces), 1 (yes), and 2 (no).
`A_FTPT` records full-time (1) or part-time (2) enrollment when `A_ENRLW=1`,
with 0 outside that universe. The survey year is the income cohort plus one.

The attachment enforces complete one-to-one income-year/PERIDNUM coverage,
original household ID, person line and age agreement, exact comparison of every
known incumbent, and complete 2024 incumbents. Values must be exact integer
source tokens in 0/1/2. Enrollment outside ages 16–54 must be zero; enrolled
persons require a nonzero full-time/part-time code; other persons require zero.
Age alone does not require a nonzero enrollment code because the publisher also
excludes Armed Forces. No fill, relabeling, carry-forward or factor is applied.

These observations do **not** validate annual attendance or the five-month
tax-student test. Restoring their source values leaves the existing microunit
proxy interpretation unchanged. Receipts state that limitation explicitly.

## Authority and replay

`asec_student_controls.load_authenticated_student_controls` requires the actual
authenticated restored source and complete ready money. It privately captures
exact-size official CSVs and verifies code-owned hashes before parsing the six
selected columns. A single nonblocking descriptor must identify a regular file
before capture, so FIFO or directory paths refuse without waiting. Callers
supply paths, never expected pins or alias overrides.
The resulting immutable object records the T1 identity and digest, ordered
scope/native-key identity, money header/content identities, source joins,
observation vector hashes, source periods, aliases and implementation closure.

S uses a bounded canonical numeric format: magic/version, a bounded canonical
JSON header, seven ordered little-endian int64 column buffers, and a transport
checksum. The columns are person ID, income year, original household ID, person
line, age, enrollment, and full-time/part-time status. The native PERIDNUMs remain
bound through the parent scope and their ordered digest.

An optional candidate path triggers full independent reconstruction. Replay
privately captures only the expected byte length and compares the complete
canonical digest before returning the independently reconstructed object. It
never parses the candidate header, numerical storage or an HDF container. A
valid checksum on changed content is insufficient. `write_student_controls`
creates a new exclusive directory containing `asec_student_controls.bin` and
`receipt.json`; it does not overwrite existing artifacts.

`attach_student_controls` retains the exact source and ready-money parents and
creates an owned Frame. Validation removes only the two aliases and requires
exact recovery of the complete parent Frame signature, including original
missing values, axes, relationships, weights, strata, metadata and history.
Tax results retain and validate this composition, record its receipt, and allow
the existing tax outputs. Subsequent housing-status attachment uses the original
T1 source identity and preserves the same money parent.

The new module and units adapter are outside the frozen T1 verification closure.
Their hashes enter the student/tax execution identities. Housing's existing tax
adapter dependency therefore changes its implementation identity as expected.
These are local source/preparation artifacts; they do not certify a graph build,
calibration, annual student classification or release.

## Validation boundary

The focused tests use only six invented persons across the three source years.
They preserve the genuine float64/NaN omission pattern, perform actual pinned
microunit reconstruction and housing composition, and exercise wrong source
members, malformed tokens, incomplete/duplicate joins, universe and incumbent
conflicts, altered canonical candidates, private snapshots and mutable Frame
storage. Genuine S production requires separate review and authorization.
