# Current ASEC child-support observations

`qualify_current_asec_child_support(preparation)` in
`microcosm.build.us_runtime.current_asec_child_support_source` qualifies the
original received- and paid-support observations from the retained 2025 ASEC
person source. The dollar amounts refer to the current 2024 income cohort.

The source is the [2025 ASEC public-use dictionary](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf),
SHA-256 `5cb80973326ef8b625fbaae70d80b0c641ce5d2b3911abd2fb4427abd5908a6f`.
The qualifier reuses the exact packaged monetary domains and temporal
authority for `CSP_VAL` and `CHSP_VAL`. It adds source questions and flags from
the same dictionary, with their positions, widths and printed universes.

| Fields | Source meaning | One-based PDF page |
| --- | --- | --- |
| `CSP_VAL`, `CSP_YN` | Amount received and actual receipt answer; receipt universe age 15+ | 52 |
| `CHSP_VAL`, `CHSP_YN` | Annual amount paid and whether payment is required; the latter is an obligation question | 52 |
| `CHELSEW_YN` | Whether a child lives outside the household; age 15+ | 52 |
| `I_CHELSEWYN`, `I_CHSPVAL`, `I_CHSPYN`, `I_CSPVAL`, `I_CSPYN` | Published allocation flags, using the shared I_ANNVAL code meanings | 54 |
| `TCHSP_VAL`, `TCSP_VAL` | Published top-code indicators for positive amounts | 59 |

## Received amounts

The printed zero for `CSP_VAL` means none or NIU. An in-universe receipt-no
answer plus zero establishes known nonreceipt. A receipt-yes answer plus zero
remains ambiguous; the qualifier does not complete it with a zero-dollar
observation. Positive receipt amounts are retained, including valid allocated
observations. Under-15 NIU, missing answers, malformed literals and
contradictory receipt/amount pairs remain distinguishable and unknown.

## Payments and obligations

`CHSP_YN` asks whether the person is required to pay child support. A no
answer does not rule out voluntary payment. `CHSP_VAL` describes actual annual
payments and prints zero as NIU, so every paid zero remains a published code
with an unknown canonical payment amount. No obligation answer turns it into
an observed zero.

The paid amount's universe is `CHSP_YN = 1`. The obligation question itself
prints only the bare field `CHELSEW_YN`, while the allocation flag `I_CHSPYN`
prints `CHELSEW_YN = 1`. That discrepancy remains in the evidence. The
qualifier conservatively accepts positive payments on the observed
intersection age 15+, child-elsewhere yes and required-to-pay yes. Other
routes keep the published positive amount and an explicit unresolved status.
This intersection is a qualification choice, not a correction of the printed
universe or proof that voluntary payments do not occur.

The obligation and child-elsewhere entries do not independently state a
reference year. Their source context is retained; the qualifier does not
issue a current-at-interview or eligibility input from those answers.

## Provenance and consumption

The result has two complete tables: selected ASEC people on the common
preparation's person identities, and the original current-cohort literals on
native identities. Source amount validity, statuses and zero origin are
retained alongside the descriptive projection. Allocation/top-code literal
values and parse statuses remain separate from amount knownness. All-zero
allocation flags describe the published codes; they are not a blanket claim
of unallocated or respondent-reported data.

The qualifier takes the actual authenticated preparation, captures the
pinned original member once through the shared bounded literal reader,
checks its bytes and complete native-key roster, verifies household/line/raw
age coordinates, and compares both source amounts to their retained
current-money values with exact validity and float64 bits. Source row order
does not define person identity.

`CurrentAsecChildSupportValues` is descriptive transport, not an issuer.
`child_support_values_seal` covers both complete tables, axes, nullable masks
and backing storage, float bits and detached evidence. The actual preparation
is requalified after source I/O, followed by final owner and physical-value
checks. A consumer must retain the actual owner, requalify immediately before
consumption and after its own last relevant I/O, and compare complete value
seals before returning or exporting a successor. Copied data or a matching
digest cannot grant source authority.

This module assigns no engine leaf, changes no clone, fits no ACS model and
issues no new source admission or release eligibility.
