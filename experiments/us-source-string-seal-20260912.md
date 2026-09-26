# Repeated source-string hashing

The ASEC source validator now reuses the exact length-prefixed encoding of a
repeated string within one column's digest calculation. It still visits every
value and computes a new digest on every validation. The cache holds at most
1,024 entries and 256 KiB of encoded payload, then stops admitting new entries.
Only exact Python strings use it; other scalar families retain the original
codec. Dtype metadata, null sentinels, row order and all digest bytes retain
their prior meaning.

Thirty invented-source tests pass, including eleven added byte-stream parity
and later-mutation controls. They cover Python and Arrow strings, distinct
Unicode encodings, nulls, more than 1,024 distinct values, strings exceeding the
payload budget, and the unchanged bytes, boolean, integer and floating object
paths. The suite took 1.860 seconds, with no failures, errors or skips.

The [benchmark receipt](us-source-string-seal-20260912.json) compares the exact
pre-change function from `d5cbe60b2` with the new implementation. Each row reports
the median of three process-CPU measurements on invented columns:

| Column | Rows | Before | After | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Repeated Python strings | 500,000 | 0.15536 s | 0.02525 s | 6.15× faster |
| Repeated Arrow strings | 500,000 | 0.15287 s | 0.02783 s | 5.49× faster |
| Unique source keys | 100,000 | 0.03178 s | 0.03552 s | 12% slower |
| Integer control | 500,000 | 0.00135 s | 0.00133 s | Approximately unchanged |

All four complete digests match exactly. These measurements do not establish
native build speed or quality. Source-file identity changes because executable
code changed; an old graph receipt is not acceptance of the new implementation.
The active native PUF pilot retains its original frozen source.
