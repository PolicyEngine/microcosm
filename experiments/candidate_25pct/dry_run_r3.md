# Candidate 25% legacy-arm dry-run receipt, round 3

Date: 2026-08-23

Required label: **one-surface + pkg3, legacy release arm, not exact-k
certified**.

Outcome: **NOT RUN — stopped on a missing mandatory stage-2 input before
launcher construction.**

## Ordered preflight result

The six pinned stage-1 raw inputs were found and their SHA-256 values matched
the queue pins. The reviewed bare v9.4 Ledger JSONL and the other explicit July
legacy inputs were also found and hashed. The full inventory and code-cited
loader trace are in `experiments/candidate_25pct/input_audit_r3.md`.

Current main unconditionally resolves and reads the full 2022 SCF extract for
this release flow (`tools/build_us_fiscal_refresh_release.py:1248-1256,
9572-9587`). Its default host path is:

```text
/Users/maxghenis/.cache/microcosm/scf/p22i6.dta
```

The file is absent, as is the adjacent `scf2022s.zip`. The loader carries no
archive or member SHA-256 pin
(`packages/microcosm-build/src/microcosm/build/us_runtime/scf_auto_loans.py:67-77`),
so no download or substitution was authorized.

## Commands deliberately not emitted

The requested external launcher was not created. Consequently there is no
honest `run-candidate.sh --dry-run` output, no `bash -n` result, and no complete
stage-1, stage-2, or scorer command line to print. Recording a partial command
as a successful dry-run would conceal the missing input and violate the
ordered stop rule.

There is also no truthful pair of dense and sparse scorer commands. One
`--dense-default-dataset` release produces only the dense artifact; the sparse
artifact requires a separate non-dense calibration branch
(`tools/build_us_fiscal_refresh_release.py:10300-10367,11060-11066`). No
candidate artifact of either kind exists.

## Side-effect receipt

- No pool or release builder was invoked.
- No external candidate directory or launcher was created.
- No publication, promotion, push, or tuning occurred.
- `POPULACE_LOGBOOK_PREV_ROW_DIGEST` was not consumed.
- No `--logbook-prev-row-digest` was supplied.
- `logbook-pending-chain.txt` was not read or written.

Resume only after the owner supplies and pins the required full-SCF bytes and
clarifies whether one dense candidate or two separately built arms are in
scope.
