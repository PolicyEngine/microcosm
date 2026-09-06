Close the PUF raw source artifact envelope and bind its CSV acceptance
profile. The decoder trusted its own header: a payload could report a row
count, a match count or a code frequency the arrays did not carry, and an
exact integer `S006` could be restated as `float64` over an unchanged body
with an unchanged digest. The header is now a closed schema — exact key sets,
an exact dtype string per typed column, an exact allocation width and body
size per lexical column, and a total body bound — and every reported fact is
recomputed from the arrays actually read back, through the same derivation the
producer uses. Separately, `csv.field_size_limit` changed which deliveries
parse without changing the node key, so a warm run could reuse an admission
made under an acceptance that no longer held; the effective profile is now
part of the producer's implementation identity and its receipt, and a decode
refuses when the effective limit is not the pinned one.
