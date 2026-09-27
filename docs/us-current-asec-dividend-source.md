# Current ASEC dividend and survivor routes

`current_asec_dividend_source.qualify_current_asec_dividend(preparation)`
qualifies current ASEC dividend observations and the recorded survivor-income
source routes that could overlap with a broad property-income concept. It
reads the original 2025 person member, whose annual money amounts refer to
2024. It returns source observations and provenance, without modeling ACS
values, assigning tax treatment or authorizing a dataset release.

The [2025 ASEC public-use dictionary](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf)
is pinned to SHA-256
`5cb80973326ef8b625fbaae70d80b0c641ce5d2b3911abd2fb4427abd5908a6f`.
The dividend domain also comes from the existing pinned
`asec_current_money_domains_v1.json`. Page numbers below are one-based PDF
pages; positions and widths refer to the printed ASCII layout.

| Field | Position / width | Page | Printed universe |
| --- | --- | --- | --- |
| `DIV_VAL` | 478 / 6 | 45 | `DIV_YN = 1` |
| `DIV_YN` | 484 / 1 | 45 | Age 15+ |
| `SUR_SC1`, `SUR_SC2` | 648 / 2, 650 / 2 | 50 | `SUR_YN = 1` |
| `SUR_YN` | 664 / 1 | 50 | Age 15+ |
| `I_DIVVAL` | 819 / 1 | 54 | `DIV_YN = 1` |
| `I_DIVYN` | 820 / 1 | 54 | Age 15+ |
| `TDIV_VAL` | 905 / 1 | 59 | `DIV_VAL > 0` |
| `TRNT_VAL` | 917 / 1 | 60 | `RNT_VAL > 0` |

## Dividend values and source flags

The projection preserves the exact `DIV_VAL_literal`, parsed
`DIV_VAL_published_amount`, literal and reporting statuses, and a separate
`DIV_VAL_amount_known` mask. `DIV_VAL_amount` contains only the values whose
reporting status is resolved. An in-universe receipt-yes and positive amount
is known; receipt-no and zero establish known nonreceipt. Receipt-yes and
zero remain ambiguous because the dictionary combines none and NIU in its
zero code. Missing literals, NIU and contradictory combinations retain
their source evidence and an unknown canonical amount. No analytical zero
is inferred solely from age.

Source flags never change this knownness or select donors. A known published
cell can still have publisher allocation or topcoding metadata. It is not a
claim about an uncensored latent amount or a respondent-only report.
`I_DIVVAL` refers to the allocation meanings under `I_ANNVAL` on page 53.
`TDIV_VAL` and `TRNT_VAL` retain codes 0 (not topcoded) and 1 (topcoded),
including missing or malformed flags. `TRNT_VAL` is retained only for future
diagnostics: this qualifier does not read `RNT_VAL` or evaluate that flag's
positive-rent universe.

The dictionary is internally inconsistent for `I_DIVYN`: its header prints
0–1, while its Values section refers to the `I_ANNVAL` codes 0–9. The result
preserves the literal and code, physical syntax status, separate
`I_DIVYN_header_range_status` and `I_DIVYN_referenced_values_status`, and an
explicit `I_DIVYN_codebook_status`. Codes 2–9 therefore retain the conflict;
they are not silently dropped or admitted under a preferred interpretation.

## Survivor source routes

`SUR_YN`, `SUR_SC1` and `SUR_SC2` each retain raw literals, parsed codes,
literal status and source labels. Code 8 identifies regular payments from
estates or trusts. It signals a possible property-income overlap, without
establishing an amount or tax classification. Code 10 means other or don't
know: its literal is valid, but its property route remains unspecified.

`survivor_property_route_clear` is nullable. It is true only for an
in-universe known receipt-no with both source slots NIU, or receipt-yes with
every slot readable, at least one active slot, and neither code 8 nor code
10. Receipt-yes with an explicit code 8 sets it false. Missing or unreadable
receipts, unresolved slots, contradictory nonreceipt routes, NIU and code 10
leave it unknown. Under-15 rows remain unknown; incomplete under-15 literals
are labeled unresolved rather than contradictory. Independent booleans
retain the presence of codes 8 and 10 even when the receipt is unresolved.

This clearance describes only the two recorded source slots. The dictionary's
`SRVS_VAL` entry (PDF page 49, printed page 6C-28) also includes unedited third
and fourth sources. Their types are not qualified here, so two clear visible
slots cannot establish complete absence of survivor estate/trust income.
The property donor bridge conservatively excludes positive survivor receipts
until that additional scope is qualified. This clearance does not establish
that the person has no property income. No survivor amount is captured, no
donor exclusion is applied, and a missing slot is never an observed zero
amount. A later measurement bridge must explicitly choose and validate how
these routes affect donor eligibility.

## Retained source boundaries

The qualifier borrows the original authenticated survey preparation and its
retained native ASEC/current-money owner. Before capture it checks the live
money domain against the pinned dividend contract. The domain reader
requires a unique field and current vintage, refuses unsupported negative,
missing-code or zero semantics, and caches only immutable values. Public
entry mappings are detached on every call.

The exact member's archive and member digests, row count and byte count
must agree with the retained owner. A single bounded capture is checked
again after reading. All current-year native keys, original household and
person-line coordinates, and source ages are joined before selection.
`DIV_VAL` validity and every valid float64 bit must match the actual money
owner, including people omitted from the selected survey. Original
`DIV_VAL_parent_statuses`, `DIV_VAL_parent_validity` and
`DIV_VAL_parent_zero_origin` remain in the projection.

`CurrentAsecDividendValues(person, asec_literals, evidence)` is descriptive
transport. `person` is indexed by `person_id` and includes `native_person_id`
and `source_age`; `asec_literals` retains the complete current-year literal
table indexed by native person ID. `dividend_values_seal(values)` covers
both tables, their axes, exact float bits, nullable masks and backing
storage, and detached evidence. The qualifier rechecks the original
preparation after source I/O and verifies the complete seal before return.

Consumers must retain and requalify the actual preparation before
consumption and after their last relevant I/O, and compare the complete
values seal. Matching JSON, a copied dataclass or a digest does not grant
source authority. No host, graph attachment, tax leaf, source issuer,
original/PUF clone change or release gate is added here.
