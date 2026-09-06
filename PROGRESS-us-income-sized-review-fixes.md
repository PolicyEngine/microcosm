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

All six are closed, each with a focused regression.

- S1 `_cohort_selection_receipt` now states the sampler's own equality per
  stratum, refuses `added_beyond_draw` by name as outside this profile, checks
  the global unit totals against the per-cohort arithmetic, and pins the rule
  string to `SAMPLER_EXACT_COUNT_RULE`. A test reads
  `frame_sampling.EXACT_COUNT_RULE` out of the producer's source and compares.
- S2/S3 `Measurement` keys by recorded location; a repeat of a location is
  checked for agreement and counted as `phase_restatements`. `scope.measurement`
  publishes `recorded_count_locations`. The two-locations gate is exercised
  directly against `_reconcile` rather than through a roster error.
- S4 `selection_rule_scope`, `sampling_fraction` and `eligible_households` are
  published beside the rule.
- S5 `base_exporter_sha256` names `graph_evidence.py`; the reviewed profile gets
  `reviewed_exporter_sha256`.
- S6 `_reconcile` requires the recorded fact set to equal the declared entities
  plus the closed cohort roster, in both directions.

Each invented shape also carries its own checkpoint identity, so the manifest's
population binding is exercised at two values instead of one.

`uv run python -m pytest packages/microcosm-build/tests/test_graph_evidence_income_sized.py`
— 129 passed, 0 failed, 1.79 s (96 before). `test_graph_evidence.py` and
`test_graph_evidence_income_v3.py` — 226 passed, 0 failed, 2.41 s, unchanged.
`uv run ruff check` clean.

## Next

- Nothing in this lane. The dashboard consumer lane closes A1-A5 in step, and
  root owns the genuine 505 export and its binding review.
