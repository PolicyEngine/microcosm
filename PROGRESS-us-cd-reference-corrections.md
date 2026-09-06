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

### Fixed in this continuation, from confirmed review findings

A six-dimension adversarial review of the corrections (freeze, identity,
counting, CLI, tests, report accuracy), each finding then handed to an
independent verifier told to refute it, surfaced defects the first attempt's own
report asserted were impossible. Every one below was reproduced by this lane
before it was fixed.

- **A verified identity could still be forged.** `sha256` and `size_bytes` were
  ordinary constructor fields, so `dataclasses.replace(inventory, sha256=…)` and
  the public dataclass constructor both produced `identity_verified True`. A
  three-district invented fixture emitted the real bundle's pinned digest and
  byte count under a verified label — review finding 1 re-entering by a second
  door, while the module said in three places and the report said in one that
  there was no other door. Both fields are now `field(init=False)`, written by
  `_index_inventory` through `object.__setattr__`; the constructor raises
  `TypeError` and `dataclasses.replace` refuses. The remaining route is
  subclassing or `object.__setattr__` itself, which is code substitution rather
  than a caller mistake, and the docstring now says exactly that instead of
  claiming impossibility.
- **The identity label over-described itself.** `identity_verified` claimed the
  pinned byte count had been matched. That leg only runs when the caller states
  the pinned digest — which is not the arm any test takes. Reworded to what both
  arms guarantee.
- **`**unverifiable` accused typos of forgery.** Catching the whole keyword
  space turned `from_document(doc, protocl=p)` into a `ReferenceProjectionError`
  alleging an identity forgery, and `inspect.signature` advertised arbitrary
  keywords. Now only `sha256` and `size_bytes` are named; they raise `TypeError`
  — a defect in the call, not in the bundle, so a caller that reports every
  `ReferenceProjectionError` as a rejected bundle cannot turn its own mistake
  into a false statement about the inventory. Every other keyword gets Python's
  own diagnostic.
- **One reserved-family field was answering two questions.** A reviewer read
  `reserved_families_projected_into_values` as a claim about cells and called it
  an over-claim, because `occupied_household_total` reported B19001 while the
  same payload's evidence recorded `B19001_001` as a duplicate alias **not**
  projected. The adversarial verifier refuted that: the protocol's
  `/exposure_ledger/reference_family_identity` says the occupied-HU comparison
  "marks both B19001 and B25003 aliases" because the two cells are the same
  published number, and `ledger.exposure_status` classifies holdout freshness
  off `block.exposure_families`, not off cells — so the declared derivation was
  right for exposure. Both answers are now reported under names that cannot be
  confused: `reserved_families_exposure_consumed` (declared, unchanged
  semantics) and `reserved_families_with_cells_read` (cell-derived), with the
  payload prose explaining why they differ.
- **The reserved-indexing sentence was unconditional.** A payload built with no
  inventory still stated that indexing had validated every required table's
  structure. It now says no inventory was indexed, and the family list is empty.
- **`projection_completeness` accepted a non-canonical protocol** the indexing
  entry points refuse, while the payload mixes canonically derived fields with
  that protocol's digest. It now calls `_canonical_only` too.
- **"Only mutability is removed" was false.** A `mappingproxy` is not a `dict`,
  so the frozen document is not directly serializable, picklable or
  deep-copyable: `json.dumps`, `canonical_bytes`, `copy.deepcopy` and `pickle`
  all refuse it. Nothing computes a wrong value; it fails loudly. The docstring
  now says so and says how a consumer thaws. The reviewer also called this a
  regression from the frozen head; the verifier refuted that half — the frozen
  head already exposed `MappingProxyType(dict(document))`, so the top level was
  never directly serializable. Only the docstring was wrong, and only the
  nesting is new.
- **`from_document` was weaker than the byte route on values.** `strict_load`
  refuses `NaN`/`Infinity`; `_frozen_json` passed any float through. It now
  refuses nonfinite floats, so the claim that the two routes are equally strict
  is true rather than asserted.

- **`main` could exit through a traceback.** The freeze walk spends a frame per
  nesting level, so a document nested past the recursion limit raised
  `RecursionError` — which is neither `OSError` nor `ValueError` — straight out
  of `main`, where the pre-freeze reader had refused cleanly. `main` is
  documented and tested to return an int. `RecursionError` is now caught, and a
  depth-2000 document refuses with `maximum recursion depth exceeded` on exit 1.
- **A taken `--output` was judged last, after the whole run.** The docstring
  claimed every inventory-independent argument is judged before the read; the
  one that costs the most to learn late was the exception — a verifier measured
  the refusal arriving after essentially a whole successful run's work. It is now checked
  beside the other pre-load checks, with the exclusive create left as the
  authority because it is the arm that cannot be raced.
- **The `FileExistsError` handler wrapped open, write and close**, so its
  comment ("only around the exclusive create") held by accident. The `try` now
  covers the create alone.
- **A test's comment described a step it never ran.** It said the same arguments
  "over a real path" reach the loader, but every call used the absent path. The
  ordering is now proved both ways: the refusal moves from the selector message
  to the loader message on the absent path, and a real path projects
  successfully, so the selector checks are not merely failing everything early.
- **The module-origin claim was overstated.** The first attempt's collector
  exec'd one of the four test files and called the resulting 174 modules "exactly
  what the suite imports". The suite imports **235**. This continuation measures
  the set the real four-file run actually leaves behind, from a
  `pytest_sessionfinish` hook: 235 modules, all resolved inside `W`, all
  byte-identical to their committed blobs, zero foreign, and no foreign shard
  entry left on `sys.path`.

Tests grew 63 → 70 cases in the reader file, covering each of the above. The freeze's cost, which the first attempt recorded as *not
measured*, is now measured on a synthetic tree sized to the pinned byte count
(the real bundle is still never opened): freeze 0.95 s against 0.24 s to parse
and 0.012 s to digest the same 38.8 MB, ~88 MB peak for the freeze alone, and
recursion is not a risk — the producer's shape nests 6 deep against a limit of
3000.

### Two changes this continuation made and then withdrew

Adversarial verification cut both ways. Two findings reproduced exactly and were
still refuted as defects, and the changes made from them were reverted rather
than kept because they looked safer:

- **The reserved-family derivation** — reverted to the declared exposure
  families, for the protocol reason above, with the cell-derived answer added
  beside it instead of replacing it.
- **A refusal for DC's two published identities.** `districts_projected` counts
  projections supplied for a block, not roster districts. The `…1198` alias is
  simply not a key in the inventory's geographies, so it yields a zero-cell,
  incomplete, flagged projection — DC's real values are still read exactly once
  — and nothing about it is DC-specific: an invented `…9999` behaves
  identically. The payload already discloses the surplus through
  `roster_district_count`, `districts_incomplete` and
  `roster_scope_notes.unexpected_districts`. Special-casing DC to refuse a case
  the module already reports correctly would have been a repair, not a refusal,
  and it would not even have covered the general shape. The refusal is gone; a
  parametrized test now pins the disclosure for both the DC alias and an
  invented id.

- **"containers nothing can change afterwards" was literally false.** A
  `MappingProxyType` is a read-only view, not a deep-immutable object, so
  `gc.get_referents(proxy)[0]` reaches the backing dict — executed here, and the
  reading moved `0.05` → `0.025` with the digest and the file unchanged, exactly
  as root's original counterexample did. A reviewer found this and judged it
  below the bar for a finding because no reasonable caller writes it. The escape
  is indeed not worth engineering against, but the sentence claiming it was
  impossible is worth fixing: the docstring now says a proxy is a view, names
  `gc.get_referents` and `object.__setattr__` as code substitution this module
  does not pretend to catch, and claims only what it delivers — that every
  ordinary route is gone from the caller's object and the indexed one alike.

### Final state of this continuation

Head `afa7524e20403324b6dcf506ed8d7a7efc2dbab1`, clean, 143 ahead of and 0 behind
`origin/main`, still three changed files and nothing else. `F` is untouched at
`9eed2e31…`, clean, and the first attempt's report, evidence, logs and
counterexample refutation are preserved byte-for-byte with their original
timestamps.

- Focused four-file group: **156 passed**, exit 0, 19.15 s, CPython 3.14.4;
  59 + 22 + 4 + 71 sums to 156. The reader file went 43 → 71 collected cases
  (39 → 60 test functions; the counts differ because several tests are
  parametrized).
- Module origin measured from the run that produced those results, via a
  `pytest_sessionfinish` hook: **235** microcosm modules, all inside `W`, all
  byte-identical to their committed blobs, zero foreign, no foreign shard entry
  left on `sys.path`.
- All six protocol-pinned modules, the packaged protocol document and the pure
  `cd_benchmark` package: unchanged and matching.
- ruff clean on both changed files and across the workspace `check`;
  `ci_test_groups.py --verify` → `verification=ok`.
- Root's counterexamples replayed at this head against a fixture bundle hashing
  to the same digest root recorded: both refused, file and reading unchanged.

### Next

1. Root review, then the independent review.
2. Only then, the fresh-output aggregate-only command recorded in
   `R/us-cd-reference-projection-corrections-REPORT-continuation.md`. Still **no
   genuine projection**.
