# ASEC demographic source contract

This is a prepared source increment, exercised only on invented fixtures. It
does not change the frozen original inputs, the prepared-source producer, or
the composed graph's population columns. `is_female` and `is_household_head`
remain unbound on the ASEC arm. No genuine extraction or release is certified.

`us_runtime/asec_demographic_source.py` uses the existing authenticated
current-money parent, source-member snapshotter, native person-key joins, and
owned Frame attachment. It can produce an additive local artifact whose
receipt declares its source pins, readset, coordinates, observation periods,
column hashes, derived states, and implementation identity. It is deliberately
not registered as a prepared-source input until the next source decision.

## Primary source contract

The Census public-use ASEC dictionaries for survey years
[2023](https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asec2023_ddl_pub_full.pdf),
[2024](https://www2.census.gov/programs-surveys/cps/datasets/2024/march/asec2024_ddl_pub_full.pdf),
and [2025](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf)
declare the following entries for all persons. The module pins the existing
dictionary hashes and records one-based PDF page locators for each year.

| Field | Printed codes / range | PDF pages, 2023 / 2024 / 2025 | Role |
| --- | --- | --- | --- |
| `A_SEX` | 1 male; 2 female | 22 / 23 / 24 | Sex observation |
| `AXSEX` | Range 0–4; only 0 no change and 4 allocated are named | 27 / 28 / 29 | Sex allocation provenance |
| `A_EXPRRP` | Range 1–14; codes 1/2 explicitly name the reference person; code 6 is unnamed | 21 / 22 / 23 | Household relationship observation |
| `P_SEQ` | Range 0–16; no individually named reference-person code | 20 / 21 / 22 | Separate sequence crosscheck |

The exact CSV readset is `PERIDNUM`, `PH_SEQ`, `A_LINENO`, `A_AGE`,
`A_SEX`, `AXSEX`, `A_EXPRRP`, and `P_SEQ`. `PERIDNUM` remains a 22-digit
string until the exact native-key join. The demographic tokens must be ASCII
digits within the printed field width; blanks, missing tokens, signs,
whitespace, decimals, and non-ASCII numerals refuse the member without
coercion. Every accepted receipt records zero missing tokens. A digit token
that fits the width but is outside the named semantic codes is preserved and
classified as unresolved, with domain counts per cohort.

The pinned members are the original `pppub23.csv`, `pppub24.csv`, and
`pppub25.csv` from the existing education-assistance archive registry. Their
income cohorts are 2022/2023/2024, while these demographic observations describe
the interview household in 2023/2024/2025. No age adjustment or period
harmonization occurs.

## Admission and diagnostics

Sex is admitted only for `A_SEX` 1/2 with `AXSEX` 0/4. The derived integer
states distinguish unbound, male, and female, with a separate no-change or
Census-allocated state. Unknown sex never becomes male; unnamed allocation
codes 1/2/3 remain unresolved.

Household reference status is admitted only when every household relationship
code is named and exactly one member has `A_EXPRRP` 1 or 2. Missing references,
duplicate references, and indeterminate relationship codes leave the whole
household unbound with person and household reason counts. `P_SEQ == 1` is
compared with the original `A_EXPRRP` reference-code flag, including households
whose cardinality prevents admission. It never supplies a fallback.

Household and family roles remain separate. Neither `A_FAMREL` nor family-head
indices are read. `A_LINENO` is a join crosscheck only. The incumbent prepared
`A_EXPRRP` can be an old line/spouse/parent recode: source restoration records
its missingness and discrepancies but never trusts it as observed evidence or
replaces it. The new `asec_A_EXPRRP` alias carries the original CSV observation.
Incumbent sex/allocation/sequence conflicts still refuse.

The artifact appends four `asec_` observation aliases and five declared integer
state/reason columns to an owned copy of the parent. It changes no original
person column, entity, weight, stratum, or metadata. No model boolean is bound.
Candidate artifact bytes are admitted only after independent reconstruction
from the pinned members; a modified candidate refuses.

The diagnostic ACS helper preserves `RELSHIPP == 20` as observed housing-unit
reference status, and codes 37/38 as distinct institutional/noninstitutional
group-quarters source codes. All-GQ households have no observed reference
person. Missing/duplicate HU references, mixed HU/GQ households, and invalid
relationship codes stay unbound. It changes no ACS values and is not wired
into the existing ACS producer, whose GQ relationship recode remains 14.

## Identity and next source increment

The demographic artifact has its own content and implementation identity,
including the source reader, reused restoration/snapshot helpers, parent
verification identity, dependency versions, and existing member pins. No
original source pin, money schema, raw-stage schema, prepared-source input
roster, or source-loader implementation is changed. The composed binding
module changes only its unbound metadata contract; kernels that attest this
module receive a new implementation identity and cannot reuse the prior
binding artifact under that identity.

Before genuine use, review this exact source revision and the fixture receipt.
The next bounded extraction should authenticate each existing original CSV,
restore the three demographic fields plus the diagnostic sequence by native
key into a new output directory, and report aggregate domain, missingness,
allocation, reference-cardinality, incumbent-disagreement, and `P_SEQ`
crosscheck counts for each cohort. Preserve all frozen parents and source
pins. That extraction has not been run here.

Only after reviewing those aggregate results should a separate integration
declare the demographic artifact in the prepared-source manifest/readset,
bind the authenticated artifact to selected rows and graph outputs, and make
an explicit cross-arm decision for unresolved cases and observed periods.
Existing microunit relationship construction remains unchanged; it would need
its own reviewed provenance decision before using the restored reference code.
