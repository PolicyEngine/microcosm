# Full original ASEC immigration donor

`borrow_full_asec_immigration_donor(preparation)` returns a checked retained source
view for a future immigration consumer. It assigns no status, admits no national
stock proxy, and does not attach anything to the graph. The preparation must be
an authentic retained native preparation; a selected Frame or copied receipt is
insufficient.

The view reuses the native owner's complete current-cohort roster with
`selected=None` and its descendant constructor. Original `HSUP_WGT / 100`
fractions supply household DESIGN weights. Selection fractions, source shares,
clones and the legacy parent's pooled weights do not rescale this donor.
The accompanying household projection retains the exact literal weight,
numerator/denominator, reduced fraction and original household key. No person
weight, MARSUPWT authority, stock normalization or prior-wage input is created.

The Frame contains only structural IDs, original person keys, the reviewed
23-field immigration evidence, `age`, `is_female`, and household `state_fips`.
Raw literals remain separately available without loss. Sex comes from the
maintained authenticated A_SEX/AXSEX source reader; unresolved sex binding
refuses. State comes from original household GESTFIPS through the maintained
state reader, with a valid state/DC code required. Carried parent predictors do
not fill missing source evidence. The existing observed-age normalization keeps
common age equal to original A_AGE, including top codes. ASEC observation year
is 2025 and income year is 2024. Neither state residence nor legal status is
asserted as an income-year observation.

## Owning-source token admission

Every one of the 23 required fields must contain a nonblank numeric literal.
Raw surrounding whitespace and leading zeroes remain in the raw view; the
numeric projection admits the corresponding integer code. Blanks, malformed
text, infinities, unexplained codes and unreviewed response values refuse before
any future rules coercion. No blank becomes zero. Legitimate printed NIU codes
remain distinct integers, including PEAFEVER = -1 and the appropriate zero codes.
This validates source code meanings, not a legal-status classification.

The [2025 ASEC person dictionary](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf)
provides the following admitted values:

| Fields | Admitted codes |
| --- | --- |
| PRCITSHP | 1–5; unexplained negative header values refuse |
| PEINUSYR | 0–28; the enumerated 27/28 values override the stale 0:26 header |
| A_AGE | 0–80 and 85; 80 and 85 remain the printed top codes |
| A_MARITL / A_SPOUSE / A_HSCOL | 1–7 / 0–16 / 0–2 |
| A_LFSR | 0, 1, 2, 3, 4, 7; no prior-wage inference |
| MCARE, CAID, IHSFLG, CHAMPVA, MIL, SS_YN, SSI_YN | 0–2, preserving each field's own zero/NIU meaning |
| PEN_SC1/2, RESNSS1/2 | 0–8 |
| PEIO1COW / A_MJOCC | Enumerated 0–8 / 0–11; unexplained header-only values refuse |
| PEAFEVER | -1, 1, 2 |
| SPM_CAPHOUSESUB | Exact whole dollars 0–99,999; trailing decimal zeroes accepted, fractional dollars unresolved |

PENATVTY uses the 162 printed country/area codes in Appendix J of the
[2025 ASEC technical documentation](https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar25.pdf).
The person dictionary's Appendix H cross-reference does not identify the actual
2025 country appendix. Header-only missing/response codes are not assigned a
nationality. The monetary field agrees with the maintained
`asec_current_money_domains_v1.json` contract (2025 dictionary physical page 61,
printed 6C-40). If actual source tokens require additional meanings, keep the
literal owner intact and review that domain explicitly; do not default them.

## Retention and validation

The caller must retain the view and call `validate()` after its last relevant
I/O. The view retains the authentic preparation, native source owner, original
literal qualifier and sex source owner. Independent immutable seals live in an
external issued-state tuple, not solely in an editable validation closure.
Validation checks source owners and implementation bytes, then rechecks pure
source/output/issuance seals. Detached copies, mutated projections, changed
source anchors, altered implementation, or changes during final foreign I/O
refuse. A receipt and a Frame alone confer no authority. The projected Frame's
metadata and mass history remain neutral; the retained receipt explicitly keeps
status assignment and national stock alignment false.

The tests use tiny invented files through the genuine source issuers: a selected
preparation omits a household/person retained by the full donor, original row
orders differ, and exact HSUP_WGT fractions survive. Pure token controls cover
all 23 fields. Source omissions/duplicates, unresolved weights, malformed owning
evidence, original-anchor changes, detached copies, output mutation, and late
I/O mutation are negative controls. No actual population, country model, fit,
calibration, full-51 proof or legal-status distribution was run.

Later work still requires reviewed #779 rule adoption with per-row observation
years, explicit admission of its mixed-date stock proxies, source-stable draws,
ACS transfer/reconciliation and original allocation weights, clone attachment,
graph replay, and final exported-artifact/model qualification. This source slice
settles none of those downstream decisions.
