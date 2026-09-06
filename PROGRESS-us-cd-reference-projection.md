# US CD reference projection — lane journal

Bounded, candidate-independent projection of the frozen `cd_reference` inventory
into the typed `ReferenceCell` / `PrecisionResult` values the approved
`us-cd-published-benchmark-v4` protocol defines. No candidate is read, no score
is produced, no target is registered and no ledger is written.

Branch `microcosm-us-cd-reference-projection-20260906`, cut from `origin/main`
`c1b83241` and fast-forwarded (no rewrite) to the integration head
`ab10f77854b7b9e8057a07dbec40d15c10b96297`.

## State

Module and tests landed and green. Aggregate-only projection command
implemented but **not run against the real inventory** — that is a separate,
reviewed step.

## Done

- Fresh worktree cut from `origin/main` and fast-forwarded to `ab10f778`; the
  integration worktree it derives from is untouched.
- Independently recomputed, at this head, the three protocol-pinned pure
  adapters and the packaged protocol document:
  `acs_housing_universe.py` `cb34d66a…b491` (9,195 B),
  `asec_housing_universe.py` `5cd32182…e1f8` (14,102 B),
  `cd_reference.py` `aa02378d…df1b` (25,758 B),
  `us-cd-published-benchmark-v4.json` `f7574a2a…c568ea` (86,626 B). All four
  match the contract's pins.
- Baseline suite green with foreign editable shard sources purged from
  `sys.path`: `test_cd_benchmark_protocol.py`, `test_us_cd_reference.py`,
  `test_us_cd_reference_sources.py` — 85 passed, including
  `test_pinned_adapters_match_the_contract`.

## Next

1. Root review, then an independent review, before any genuine projection.
2. Only then, the bounded aggregate-only command in `REPORT.md`.

## Done (implementation)

- `microcosm.build.us_runtime.cd_benchmark_reference` (1,418 lines): digest
  before parse, roster, per-block/per-district projection into the shipped
  `ReferenceCell`/`PrecisionResult` types, typed provenance envelopes, an
  aggregate-only completeness reducer and a gated command line.
- `packages/microcosm-build/tests/test_us_cd_benchmark_reference.py` (43 tests,
  all green) on tiny invented public-shaped fixtures only.
- Focused suites green together: 128 passed
  (`test_cd_benchmark_protocol.py`, `test_us_cd_reference.py`,
  `test_us_cd_reference_sources.py`, `test_us_cd_benchmark_reference.py`).
- `ruff check` clean, `ruff format --check` clean,
  `tools/ci_test_groups.py --verify` → `verification=ok`; the new file lands in
  `fast/rest` and `engine/us-am`, never `[defaulted]`.

## Limits held throughout

- No agent dispatch, push, merge or deploy.
- No candidate or source binary read; no held-out result read.
- The real 38.7 MB reference inventory is **not** parsed in this lane. Every
  fixture here is invented and public-shaped; a fixture is never evidence about
  the real inventory.
- No target registration, no ledger or exposure write, no threshold edit.
- No scoring authorization is created, and no national cell is formed by summing
  districts. National published references require a separate acquisition.
