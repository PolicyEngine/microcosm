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

## Retry continuation, 2026-09-06 (independent re-verification)

The lane was re-entered after the first attempt was cut off immediately after it
wrote its report (tests 07:15:59Z, report 07:18Z, this continuation opened
07:19Z). Nothing from that attempt was destroyed or overwritten: its worktree,
branch, commits, report, evidence, logs and counterexample refutation are all
preserved byte-for-byte, and every artifact this continuation writes carries a
`-verify`/`-continuation` name of its own.

### Re-verified in this continuation, from scratch, not read back from the report

- `W` clean at `c1266cf39272da5e501ffc33d74d1ebb54a2fc9c`, descended from the
  frozen head `9eed2e31…`; `origin/main` `c1b83241…` is an ancestor; 133 ahead,
  0 behind. `F` still clean at `9eed2e31…`, 129 ahead, 0 behind.
- `git diff --name-status` from the frozen head is three files and nothing else.
- Working-tree digests equal the committed blobs at this head for all three.
- **All six** protocol-pinned modules recomputed from disk against the compiled
  protocol's own `pinned_code_sha256`: six of six match, and `git diff` shows
  all six unchanged since the frozen head. The shipped `verify_pinned_adapters`
  returns true for its three adapters.
- Packaged protocol document `us-cd-published-benchmark-v4.json`: 86,626 bytes,
  `f7574a2a…c568ea`, equal to `APPROVED_PROTOCOL_SHA256` and to the compiled
  protocol's own `sha256`; `protocol.canonical` is true. The pure
  `cd_benchmark` package has an empty diff against the frozen head.
- Focused four-file group re-run in this continuation: **148 passed, exit 0**,
  20.38 s, on CPython 3.14.4. Per-suite counts from the JUnit XML — 59 + 22 + 4
  + 63 — add to 148, matching the first attempt's XML exactly.
- Module origin re-walked: **174** imported `microcosm` modules, 0 outside `W`,
  0 differing from their committed blobs, 0 foreign shard entries left on
  `sys.path` after the purge.
- ruff 0.15.16: `check` and `format --check` clean on both changed files
  (exit 0); workspace `check .` exit 0; workspace `format --check .` exit 1 on
  98 pre-existing files, none of them this lane's.
- `tools/ci_test_groups.py --verify` → `verification=ok`, exit 0; the reader
  test file appears in two groups and never under `[defaulted]`.
- The origin-note correction checked against the code: `graph.py`'s
  `_ROOT_CELLS`, which supplies the outputs of the `us.post_transfer.create@1`
  legacy post-transfer static CREATE, contains no `SERIALNO` and no
  `asec_H_SEQ`. The narrower gap is real and correctly scoped; this lane did
  not modify it.

### Next

Adversarial review of the corrections across six dimensions, then any confirmed
fixes, then the continuation report. Still **no genuine projection**.
