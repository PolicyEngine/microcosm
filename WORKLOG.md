# Worklog: populace#395 increment 1

## State

Complete on branch `multispine-operator-ordering-395`, based on
`origin/main` commit `0d99d8a`.

This increment lands the opt-in seam and contracts for canonical multispine
operator ordering. It does not wire the current sparse or dense release tools
to the new seam, and it does not push or open a pull request.

The initial `git fetch origin` could not resolve `github.com` in the managed
environment. The existing local `origin/main` already resolved to the exact
requested base commit.

## Done

- Mapped the current `build_us_puf_support_base.py` and
  `build_us_acs_multispine_base.py` call graphs, including the state consumed
  before and after their late ACS append.
- Added opt-in `assemble_spines(...)` before population operators. It accepts
  two or more nullable US peer frames, rejects PUF tax detail as a peer spine,
  remaps colliding structural IDs, preserves inputs, and conserves anchor
  household mass.
- Established four provenance fields on every entity:
  immutable `*_support_channel`, raw local `*_spine_source_id`,
  assembly-unique pre-clone `*_source_id`, and
  `*_support_clone_index`.
- Extended the PUF clone entrypoint across the combined pool. Source channels
  and source IDs remain unchanged; clone indices route PUF detail, QRF
  recipients, and the capital-gains tail.
- Migrated the 27 population-operator modules that previously interpreted
  support channels as ASEC/PUF roles. Clone metadata now determines operator
  roles, while the centralized fallback preserves fail-closed behavior for
  the current unassembled lineage.
- Added the #443-style AST guard. Only reviewed assembly/provenance/agreement
  owners may resolve household source-spine identity, and the exact migrated
  operator registry may not read any entity source-channel column.
- Added the fixed pre-calibration spine-agreement registry and gate. Every
  declared transfer/imputation distribution uses the same weighted nonzero
  incidence-ratio band `[0.8, 1.25]` and conditional q10/q25/q50/q75/q90
  symmetric-relative envelope tolerance `0.25`. Every source-spine pair is
  checked and failures are batched.
- Added the design note and changelog fragment.

## Verification

- Migrated operator regression set: 544 tests passed.
- Current-lineage compatibility set:
  145 tests passed and 2 environment-dependent tests skipped across the PUF
  builder, checkpoint equivalence, ACS builder, ACS transfer, and base pool.
- New seam and clone core:
  89 tests passed across assembly, spine agreement, multispine cloning,
  spine-blindness, PUF support, QRF chaining, and the capital-gains tail.
- Ruff format and lint: all 38 changed Python files pass.
- `git diff origin/main --check`: passes.

## Coherent implementation commits

- `3885f40` Start #395 multispine ordering worklog
- `0c8f512` Add pre-operator multispine assembly seam
- `cc30fa8` Specify pre-calibration spine agreement gate
- `746822a` Separate source spines from PUF clone roles
- `50acee9` Make US population operators source-spine blind
- `1c377b6` Document the canonical multispine operator order

## Next

- Wire source harmonization and `assemble_spines(...)` into the US build before
  the shared clone, impute, derive, seed, and simulate pass.
- Evaluate the spine-agreement gate before calibration and prohibit calibration
  when its batched result fails.
- Retire the late ACS transfer/append lineage only after the replacement
  artifacts pass the declared gates.
- Build the #578 one-suite, full-geography, exact-k release shape on the
  verified ordering.

## Uncommitted

None.
