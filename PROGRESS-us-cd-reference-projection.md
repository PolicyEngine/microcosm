# US CD reference projection — lane journal

Bounded, candidate-independent projection of the frozen `cd_reference` inventory
into the typed `ReferenceCell` / `PrecisionResult` values the approved
`us-cd-published-benchmark-v4` protocol defines. No candidate is read, no score
is produced, no target is registered and no ledger is written.

Branch `microcosm-us-cd-reference-projection-20260906`, cut from `origin/main`
`c1b83241` and fast-forwarded (no rewrite) to the integration head
`ab10f77854b7b9e8057a07dbec40d15c10b96297`.

## State

Complete for this slice. Module and tests landed and green. The aggregate-only
projection command is implemented but has **not** been run against the real
inventory — that is a separate step, after review.

## Done

- Fresh worktree cut from `origin/main` and fast-forwarded to `ab10f778`; the
  integration worktree it derives from is untouched and clean.
- Recomputed, at this head, all six protocol-pinned modules and the packaged
  protocol document. All six match, which settles the open question of whether
  the seven K source-ingestion annotation edits disturbed the pure adapters:
  they did not.
  - `acs_housing_universe.py` `cb34d66a…b491` (9,195 B)
  - `asec_housing_universe.py` `5cd32182…e1f8` (14,102 B)
  - `cd_reference.py` `aa02378d…df1b` (25,758 B)
  - `asec_household_observations.py` `87276fd8…7c3c`
  - `acs_pums.py` `bd0d50e4…46e7`
  - `stacked_spine.py` `ab194811…5301`
  - `us-cd-published-benchmark-v4.json` `f7574a2a…c568ea` (86,626 B)
- Added `microcosm.build.us_runtime.cd_benchmark_reference` — outside the pure
  `cd_benchmark` package, so that package keeps its documented "reads no
  candidate, population, HDF5, ACS or Census artifact" property.
- Added `packages/microcosm-build/tests/test_us_cd_benchmark_reference.py`,
  43 tests, tiny invented public-shaped fixtures only.
- Focused suites green together: **128 passed** (59 + 22 + 4 + 43).
- `ruff check` clean, `ruff format --check` clean,
  `tools/ci_test_groups.py --verify` → `verification=ok`; the new file lands in
  `fast/rest` and `engine/us-am`, never `[defaulted]`.
- Every one of the 174 imported `microcosm` module files came from this
  worktree and is byte-identical to its committed blob at this head.

## Defects found and fixed in this lane's own first attempt

The first attempt of this lane was cut off mid-flight by a provider usage limit
before it ran anything. Its two files were reviewed rather than trusted, and
three real defects were found by running them:

1. `cd_reference.cd_reference_sources` — a sibling module addressed as an
   attribute; every fixture-backed test errored at collection.
2. `PointBlock.relative_ceiling` — no such attribute; the field is
   `reference_relative_moe_ceiling`.
3. The subset-share expectation was computed in the default 28-digit decimal
   context while the package computes in its own 120-digit `ARITHMETIC`
   context, so the assertion compared differently-rounded values.

A fourth was found by inspection: the pure-package purity check set `PYTHONPATH`
but left the interpreter's `.pth` files in force, so an editable install
pointing at another worktree could make it assert *that* tree's purity. Verified
directly that a foreign shard does load that way; the child now purges foreign
shard sources and asserts the module it imported came from this checkout.

## Next

1. Root review, then the independent review, before any genuine projection.
2. Only then, the bounded aggregate-only command recorded in
   `R/us-cd-reference-projection-REPORT.md`.

## Limits held throughout

- No agent dispatch, push, merge or deploy.
- No candidate or source binary read; no held-out result read.
- The real 38.7 MB reference inventory is **not** opened, parsed or hashed in
  this lane. Every fixture here is invented and public-shaped; a fixture is
  never evidence about the real inventory, and no completeness number about it
  exists yet.
- No target registration, no ledger or exposure write, no threshold edit.
- No scoring authorization is created, and no national cell is formed by summing
  districts. National published references require a separate acquisition,
  concept mapping, protocol and approval.
