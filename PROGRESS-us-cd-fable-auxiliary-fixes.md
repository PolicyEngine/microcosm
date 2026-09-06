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
- [ ] Behavioral regressions written and shown red.
- [ ] Fixes applied and shown green.
- [ ] Scoped ruff and diff checks.
- [ ] Report and receipt written.

## Next

Write the five regressions against the unmodified head, run them red, then implement.

## Non-goals

No push, no merge, no approval registration, no source/reference/candidate/ledger read,
no PolicyEngine import, no network call, and no generic approval or security framework.
