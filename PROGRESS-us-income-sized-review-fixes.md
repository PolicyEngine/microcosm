# us-income-sized export — review fixes

Successor lane to `PROGRESS-us-income-sized-export.md`. It closes the six
should-address findings (S1–S6) raised against
`packages/microcosm-build/src/microcosm/build/graph_evidence_income_sized.py`
at f231ac143ab26954c5b8fcedec9f5b40123e89cd, in step with the dashboard
consumer lane that closes A1–A5.

## State

Branch `us-income-sized-review-fixes-20260906`, cut fresh from a newly fetched
`origin/main` (c1b8324183819cf91511c002bf724a2676524e87) and fast-forwarded to
the reviewed parent f231ac143ab26954c5b8fcedec9f5b40123e89cd. No genuine export
has been run in this lane and none is run by it.

## Findings this lane closes

- S1 per-stratum requested/realized arithmetic is inverted against the sampler.
- S2 restatements are published as independent recordings.
- S3 the two-recordings gate is unreachable and untested.
- S4 the selection rule is published without its scope.
- S5 `base_exporter_sha256` changes meaning between profiles.
- S6 recorded-but-undeclared entity facts are dropped instead of refused.

## Done

- Nothing yet beyond lane setup.

## Next

- Trace each finding against the code, then correct it with a focused test.
