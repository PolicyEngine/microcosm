Separate the four 2015 PUF disclosure-aggregate records from the individual
return amount universe in `microcosm.us.puf_2015_monetary_source_projection`,
now artifact schema 3. A row is inside that universe if and only if its
authenticated `disclosure_aggregate` flag is 0, exposed as a read-only
`amount_known` mask derived from the flag and declared as a rule string in the
header facts. Both the mask and decoded typed columns use immutable byte
backing, which prevents callers from re-enabling writes or changing the flags
after obtaining a mask. Individual returns keep the unchanged integer, whole-dollar,
twelve-digit grammar and still refuse a fractional token. The aggregates' typed
slots hold the minimum-`int64` out-of-universe sentinel rather than a zero, and
their delivered tokens are retained exactly under a separate bounded lexical
grammar — an explicit operational input bound for this pinned file, not a
publisher specification — that refuses a plus sign, an exponent, whitespace, an
embedded NUL, an empty token, negative zero, more than twelve digits or more
than the declared width. All 207,696 rows stay in delivered order; no amount is
coerced to float, rounded, grown or dropped. Amount tokens are collected before
the status alignment and parsed after it. The aggregate row class reports token
classes only — no zero, sign, extremum or sum key — so no header, receipt or
summary can state an aggregate amount, and the summary's negative census now
covers the two per-return classes instead of summing all three. Column metadata
scopes the retained booklet-derived whole-dollar semantics to individual
returns and declares the aggregate treatment, sentinel, grammar, digit and
width limits, and operational authority; the
projection document's own bytes and the source pins are unchanged. Both kernel
refs and the implementation identity move to version 3, and the reader refuses
the superseded magics by name.
