# Current ASEC retirement details

`current_asec_retirement_detail_source` qualifies pension, disability and
survivor source details from the current ASEC member owned by an authenticated
survey preparation. This is source evidence for a later ACS retirement bridge.
It produces no model, tax leaf, payment-frequency classification or new issuer.

## Source and field ownership

The source is the [2025 ASEC public dictionary](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf),
SHA256 `5cb80973326ef8b625fbaae70d80b0c641ce5d2b3911abd2fb4427abd5908a6f`,
for calendar-year 2024 income. Page references below are physical PDF pages;
the corresponding printed pages are 6C-23 through 6C-39.

| Amount fields | Qualification |
| --- | --- |
| `PNSN_VAL`, `ANN_VAL`, `DST_VAL1/2`, `DST_VAL1_YNG/2_YNG` | Reuse immutable printed domains from the existing income-routing owner. Compare exact retained current-money values and validity. They remain reference observations here; their canonical receipt/account interpretation stays with that owner. |
| `DIS_VAL1/2` | Explicit dictionary domains on page 44, also compared against actual retained current-money values and validity. |
| `PEN_VAL1/2`, `SUR_VAL1/2`, `DSAB_VAL`, `DBTN_VAL`, `SRVS_VAL` | Independently qualify literals from the exact authenticated source member using published width, range, universe and zero semantics. These seven fields are absent from the current retained-money roster; the qualifier never fabricates a `ready.field()` for them. |

The live money domains must agree with the eight existing fields. A future
money roster containing one of the seven independent fields fails until the
corresponding retained comparison is reviewed. The amount-entry cache contains
immutable entries; each public call returns a detached mapping.

The complete source join verifies native person keys, household IDs, person
line numbers and source ages. Selected original ASEC persons retain their
actual composed person IDs. No clone or PUF population is modified. The source
literal table covers the complete current source roster; the person table
covers selected ASEC persons.

## Knownness and comparisons

`PEN_YN`, `DIS_YN` and `SUR_YN` are separate from their two source codes and
amounts (pages 44–45, 47 and 49–50). A positive readable amount is canonical
only with a valid positive source code and yes receipt. A no receipt, NIU source
slot and zero amount yields explicitly derived known nonreceipt. A yes receipt
with a declared source and zero amount stays ambiguous. A yes receipt with a
zero source slot and amount stays unreported, not an observed analytical zero.
Missing, malformed, out-of-range and contradictory values stay distinct.

People under 15 never acquire analytical zeros. Unreadable outside-universe
answers stay unresolved; readable nonzero outside-universe evidence is flagged
as contradictory. `DIS_CS` and `DIS_HP` are preserved source answers. The latter
includes limited work as well as prevented work and does not establish the
ACS disability-pension criterion.

Four raw numerical comparisons retain total-minus-slot differences:

- `PNSN_VAL - PEN_VAL1 - PEN_VAL2`;
- `DSAB_VAL - DIS_VAL1 - DIS_VAL2`;
- `SRVS_VAL - SUR_VAL1 - SUR_VAL2`;
- `DBTN_VAL - DST_VAL1 - DST_VAL2`.

These comparisons include readable published none/NIU zeros; they are not
proof of analytical completeness. Missing or malformed components produce an
unknown comparison. No difference is allocated to a pension, taxable income
or missing source. `DBTN_VAL` is retirement distributions, not disability, and
its printed formula uses the main slots rather than young-person slots.

`SRVS_VAL` includes edited sources 1/2 and unedited sources 3/4 (page 49).
Its visible pair is therefore never declared exhaustive. Source labels retain
Railroad Retirement, estate/trust, annuity and unspecified categories without
assigning them to an ACS aggregate. In particular, positive survivor receipt
with two visible non-property routes does not prove the absence of additional
estate/trust income.

Allocation and top-code flags preserve their own published domains. Source
allocation flags with values 0/1/9 do not acquire all intermediate header-range
values. `I_SURVL2` retains its printed `SURV_VAL2` universe spelling. Allocation
does not invalidate an otherwise observed amount; a disclosure flag does not
explain a particular component discrepancy by itself.

## Why the ACS bridge remains separate

The [2024 ACS questionnaire, question 43g](https://www2.census.gov/programs-surveys/acs/methodology/questionnaires/2024/quest24.pdf#page=18)
asks about regular account withdrawals, while [ASEC Q98Ar](https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar25.pdf#page=205)
asks about any withdrawal/distribution. “Regular IRA” is an account label, not
payment frequency. Account type alone does not establish taxability. The
qualifier assigns neither regularity nor tax treatment and makes no ACS
retirement-component allocation.

## API and validation boundary

`project_retirement_detail_literals(DataFrame)` is a descriptive pure
projection. `qualify_current_asec_retirement_detail(preparation)` requires the
actual preparation and original source ownership, captures the pinned member,
checks source identity and retained money, and requalifies the parent after
the last source I/O. Its last checks include actual owner identity and a full
physical result seal.

The retained-money comparison follows the existing owner's annuity encoding:
published `ANN_VAL=-1` matches normalized positive zero with `DECLARED_NIU`,
while the descriptive literal and published amount remain `-1`. A dollar-zero
literal cannot substitute for that NIU/status pair. Other valid dollar amounts
retain exact float64 comparison. This distinction is tested through actual
invented source preparation, as well as direct mismatched-status controls.

`retirement_detail_values_seal(values)` covers table axes, dtypes, exact values,
nullable Float64 backing storage and masks, literals and detached evidence.
The returned frozen dataclass contains mutable descriptive tables and grants no
source admission. A consuming host retains/revalidates the actual parent and
the result at its own boundaries; a copied dataclass is not an issuer.

Tests use only invented records and the real source-preparation path. They
cover source joins, retained versus independent fields, member-byte drift,
late parent mutation and result mutation beneath null masks. No native source
or country engine is needed, and passing tests do not certify a release.
