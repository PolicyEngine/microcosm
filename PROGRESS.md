# PROGRESS — us-puf-raw-source-first-slice-20260905

This opening section records the earlier raw-source slice, not the current
packaging state. The monetary projection revision relocated the unchanged
`puf_2015_raw_source_definition.json` from `microcosm/build/us/` to
`microcosm/build/us_runtime/` and revised the raw loader's implementation
identity. Previously accepted raw artifacts and their evidence remain intact.

## State
COMPLETE (local build only). Branch `us-puf-raw-source-first-slice-20260905`,
a fresh dependent child of `9ece4cadddd1933524231a46a959086b0ff74429` — the
reviewed local integration commit named by the root addendum. Worktree
`/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-puf-raw-source-20260905`.
No push, no merge, no PR, no deploy. No genuine source, capture, receipt or
microdata was read by this slice; the only genuine-file facts used are closed
metadata (digests, header lists, documented domains) already retained in the
review directory.

## Controlling documents
- `us-puf-raw-artifact-first-slice-plan.md` — Fable's Plan B.
- `us-puf-raw-artifact-first-slice-root-addendum.md` — controlling corrections.
- `us-sipp-puf-next-slices-fable51-review.md` §"Plan B" — APPROVE; all six
  specification and test additions carried.
- `us-puf-demographic-domain-root-extract.json` — primary domain metadata.

## Done
- `packages/microcosm-build/src/microcosm/build/us/puf_2015_raw_source_definition.json`
  — closed packaged definition: both source pins (bytes, sha256, git blob sha1,
  canonical header sha256, full delivered header lists in order), the v2 audit
  CSV profile verbatim, twelve slice fields with dtype, grammar, lexical width,
  sentinel policy, closed domains and documentary pins, the AGERANGE
  conditional domain, the four aggregate RECIDs, join and period statements.
- `packages/microcosm-build/src/microcosm/build/us_runtime/puf_raw_source.py`
  — private 128 MiB cap; `load_pinned_puf_bytes` (regular-file, exact length,
  before/after identity, streaming sha256 + git blob sha1); `register_bytes`
  codec registration; structural decoder; typed envelope (4-byte big-endian
  header length, little-endian integer bodies, fixed-width NUL-padded ASCII
  lexical bodies, per-body digests); producer kernel with an exact node-shape
  refusal; consumer kernel; node factories; `SourceRef`s; registry helpers.
- `packages/microcosm-build/tests/test_us_puf_raw_source.py` — 66 tests, all
  on invented fixtures under `tmp_path`.
- `packages/microcosm-build/tests/test_us_spine_blindness.py` — one new
  classification entry for the new module only.
- `docs/us-puf-raw-source-first-slice.md`,
  `changelog.d/20260905-us-puf-raw-source-first-slice.added.md`.

## Verification
- New suite: 66 passed. `ruff check .`: all checks passed. CI grouping verified
  (`tools/ci_test_groups.py --verify` = ok; the new file lands in `us-p`).
- Neighbouring precedent suites green: growth graph, ACS rent artifacts, graph
  codecs, graph source bytes.
- `test_us_spine_blindness.py` has three failures. They are the pre-existing
  baseline — reproduced identically at the base commit with the new files
  removed — and the 21 unclassified modules do not include `puf_raw_source.py`.
  A separate worker owns that fix.

## Next (not in this task)
- Root and peer code review, then a separate reviewed private runner for the
  first genuine rung. See the "Genuine run: prepared, not executed" section of
  `docs/us-puf-raw-source-first-slice.md`.
