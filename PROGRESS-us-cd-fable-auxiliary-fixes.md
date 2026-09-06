# US CD benchmark contract v4 — Fable auxiliary review fixes

Lane journal for the child worktree `us-cd-fable-auxiliary-fixes-20260905`, branched from
the frozen head `e413dcc26a76339576efa973f1cce3ec3701eccd`. This file is this lane's own
journal; the root journals and `PROGRESS-us-cd-benchmark-contract.md` are not edited here.

## Scope

From `architecture-reviews/microcosm-20260904/us-cd-contract-v4-e413-fable51-review.md`,
the disjoint approval/ledger/parser slice only:

1. Finding 2 — registration must resolve the records that actually authorize the action
   (`approval.py`).
2. Finding 5 — family lists and the closed payload fields read as identifier sequences are
   typed in `parse_record`, so a bare string is never iterated as a family array
   (`ledger.py`).
3. Finding 6 — a new evaluation attempt requires the prior attempt's terminal record
   (`ledger.py`).
4. Finding 8 (smaller) — `compile_protocol` raises `ProtocolError`, never a leaked
   `KeyError`/`TypeError`, on an absent or mistyped required roster (`protocol.py`).
5. Finding 8 (smaller) — a negative or non-integer `vacant_household_rows_dropped` is
   refused rather than read as a known nonpositive count (`universe.py`).

Owned files: `approval.py`, `ledger.py`, `protocol.py`, `universe.py`,
`test_cd_benchmark_protocol.py`, `test_cd_benchmark_ledger.py`, and the new flat
`test_cd_benchmark_review_auxiliary.py`. Findings 1, 3, 4 and 7 belong to the sibling
child that owns `evaluation.py`, `reference.py` and `metrics.py`; docs, CI wiring and the
changelog are the root's.

## State

- [x] Worktree created from the frozen base; reference worktree left untouched at e413dcc2.
- [x] Guarded runner `run_cd_fable_auxiliary.py` written under the review directory.
- [x] Numeric baseline captured: 217 document numeric values, 94 compiled numeric values,
      document digest equal to `APPROVED_PROTOCOL_SHA256`.
- [x] Behavioral regressions written first and shown red: 25 failures across
      `test_cd_benchmark_protocol.py` and `test_cd_benchmark_ledger.py` with 72 passing,
      plus a collection `ImportError` for the new file. No pre-existing test failed.
- [x] Fixes applied and shown green: 212 passed, 0 failed across the six CD benchmark
      test files under the guarded runner, exit 0, no forbidden import, every loaded
      module byte-equal to its committed blob.
- [x] Document bytes, digest, all 217 document numeric values and all 94 compiled
      numeric values re-verified identical to the base.
- [x] `ruff check` and `ruff format --check` clean on the seven owned files.
- [x] Report `us-cd-fable-auxiliary-fixes-REPORT.md` and
      `us-cd-fable-auxiliary-fixes-receipt.json` written under the review directory.

Head `3dfeaa49`, two commits on `us-cd-fable-auxiliary-fixes-20260905`.

## Next

Nothing is pushed, merged, registered or scored, by instruction. After the root
cherry-picks this lane, one CI step remains and belongs to the root, not here:
`tools/ci_test_groups.py --verify` still exits 0 but reports the new
`test_cd_benchmark_review_auxiliary.py` under `[defaulted] -> shared-spec`, and the repo
guide asks that a new test file never sit under `[defaulted]`. That file is the
sibling's and the root's. A changelog fragment is likewise the root's.

## Non-goals

No push, no merge, no approval registration, no source/reference/candidate/ledger read,
no PolicyEngine import, no network call, and no generic approval or security framework.
