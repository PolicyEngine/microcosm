# US CD reference reader — corrections lane journal

Corrections to the bounded, candidate-independent CD reference reader added by
`microcosm-us-cd-reference-projection-20260906`, implementing every actionable
finding of the independent Fable 5.1 review and both executed root
counterexamples. **No genuine projection was run**: the real 38.7 MB
`reference-inventory.json` was never opened, parsed, hashed or characterised in
this lane. Every fixture here is a tiny invented public-shaped bundle, and a
fixture is never evidence about the real inventory.

Branch `microcosm-us-cd-reference-corrections-20260906`, cut from the frozen
projection head `9eed2e3106ec4733e50faa70dfd8477d11f90787`, which is itself
129 commits ahead of `origin/main` `c1b8324183819cf91511c002bf724a2676524e87`
(`origin/main` is an ancestor of it; nothing was rewritten).

## State

Complete for this slice, and green. Two files changed, both this lane's own:
the reference reader and its test file. The protocol document, the pure
`cd_benchmark` package and all six pinned modules are byte-for-byte as found.
The aggregate-only projection command is still **not run** against the real
inventory; that waits on root review and the independent review.

## Done

- Recorded the frozen worktree's exact relation before branching, then cut a
  dependent worktree from that head. The frozen worktree is untouched.
- **Identity.** `ReferenceInventory.from_document` no longer accepts `sha256=`
  or `size_bytes=`; both are refused. Only `load_pinned_inventory`, which
  enforces the pinned byte count from `stat` and the SHA-256 over the file's
  bytes before parsing, can populate them. The payload now carries
  `inventory_identity_verified` beside the digest.
- **Immutability.** A parsed document is deep-copied into read-only mappings
  and tuples before indexing, so neither the indexed inventory nor the caller's
  own object can change the other. Root's mutation is now a `TypeError` and the
  reading is unchanged. JSON meaning is preserved; only mutability is removed.
- **Counting.** `projection_completeness` refuses a repeated
  `(block_id, geography_id)`, and an empty projection stays empty instead of
  defaulting the present set to the whole roster.
- **Written bytes.** The command line writes canonical bytes plus their
  terminating newline with exclusive creation, refuses to replace an existing
  record (`FileExistsError` handled explicitly, ahead of the `OSError` arm, and
  now inside the guarded block), and prints the digest of exactly those bytes.
- **Argument order and documentation.** Both district selectors, neither, and a
  repeated district are judged before the inventory is opened. The
  canonical-only authority values are documented as such and a non-canonical
  protocol is refused rather than silently mixed. The module and the payload
  state that indexing validates the reserved table structures even for an
  S0101-only projection, without computing their values.
- Tests: 43 → 63 in this file; the focused group 128 → **148 passed**, exit 0.
- `ruff check` clean on the two changed files and across the workspace;
  `ruff format --check` clean on both changed files;
  `tools/ci_test_groups.py --verify` → `verification=ok`, the reader test file
  in two groups and never `[defaulted]`.
- All six protocol-pinned modules and the packaged protocol document recomputed
  at this head: all match. 174 imported `microcosm` module files all resolved to
  this worktree and are byte-identical to their committed blobs.

## Next

1. Root review, then the independent review.
2. Only then, the fresh-output aggregate-only command recorded in
   `R/us-cd-reference-projection-corrections-REPORT.md`.

## Limits held throughout

- No genuine reference projection; the real inventory is never opened.
- No protocol, adapter or pure-package file edited.
- No candidate, source binary, microdata or held-out result read.
- No push, merge, deploy, agent dispatch, target registration, ledger or
  exposure write, threshold or bound edit.
- The frozen lane's files, reports, scripts and counterexamples are preserved
  byte-for-byte; every artifact this lane wrote is a new file.
