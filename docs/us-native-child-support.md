# Optional native child-support amounts

Select `groups=("child_support",)` together with
`full_original_amount_donors=True` in `run_us_survey_enrichment`. The default
amount groups and selected-donor behavior are unchanged. This is a development
transport assumption from ASEC 2025 income-year 2024 observations to ACS 2024;
scientific qualification and release validation remain outstanding.

The sole fitted target is `child_support_received`, from qualified `CSP_VAL`.
The maintained zero-aware QRF uses all original current ASEC persons with known
receipt or known nonreceipt and finite current predictors, retaining their
original household DESIGN weights. It includes known nonreceipt zeros; ambiguous
recipient zeros and missing or contradictory source routes are not donor labels.
A selected receiving frame can omit a donor without removing it from this fit.
Eligible recipients are original ACS persons aged at least 15. One draw is shared
by both existing clones. Observed ASEC amounts remain source values, including
unknown values; under-15 ACS recipients remain unresolved.

`child_support_expense` is an observed-only additional output. It retains only
qualified positive `CHSP_VAL` payments under the documented source route. Paid
zero/NIU is not treated as known nonpayment. There is no expense model, no inferred
absence of voluntary payments, and no expense or native-input-profile completion
claim. `CHSP_YN` describes an obligation, not observed payment incidence.

Both outputs carry separate source-knownness, literal, route, allocation and
topcode diagnostics under `survey_child_*`. The `survey_current_*_origin` labels
distinguish ACS modeled receipts from ASEC source answers and unresolved values.
A modeled value does not alter its source observation mask. The source qualifier
returns the complete literal table separately from its selected observation view.

If either canonical child-support column already exists, this graph node owns an
explicit rewrite from its newly qualified source/model values. The inherited
values are not accepted as evidence. Supported carried storage is NumPy
`float32` or `float64`; the new values must round-trip without loss, including
unknowns. Other storage refuses. Only these two declared columns can be rewritten;
all unrelated columns, structural IDs, weights, source records and clone mappings
remain unchanged. Earlier source projection, fitting and reporting nodes retain
the prior population version and their complete input checks. A final keep-all
`FILTER` opens a derived version after every prior host branch; only the two
canonical child amounts are written there. The version adds an explicit
conserved-mass ledger record without changing represented mass. Reconstruction
refuses any changed entity axis. The attachment evidence lists the replaced
columns and storage.

The implementation reuses existing source qualification, full-donor CREATE and
FILTER nodes, fit/apply artifacts, model verification and final attachment. It
adds no new source issuer or release authority. Changing the maintained amount
host's implementation closure changes its cache identity, including for existing
routes; old runtime/source snapshots remain independently pinned.
