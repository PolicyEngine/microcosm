# Lane journal — PUF raw source review fixes

Child of `d836bf9060cec348ae9eede25e3d3ec9dd7e754b` on branch
`us-puf-raw-source-review-fixes-20260906`. The parent's rotating `PROGRESS.md`
is left exactly as the parent lane wrote it; this file is this lane's own
state, as the assignment requires.

**Invented fixtures only.** No genuine PUF delivery, capture, receipt or
artifact is opened by anything in this lane, and no engine package is imported.

## State

Complete. Both assigned defects are closed and proved closed by re-running
root's own probe shapes inverted; the one optional item turned out to be
concretely evidenced, so it was fixed rather than noted. Six commits on
`us-puf-raw-source-review-fixes-20260906`. Nothing pushed, no PR, no merge,
no deploy. The parent worktree is untouched and still clean at `d836bf90`.

## Assigned defects

1. **Decoder trusted self-reported header facts and column dtypes.** Root
   reproduced 4 typed rows carrying `facts.rows = 987654321` and
   `matched_keys = 777`, and `S006` read back as `float64` from an unchanged
   body with an unchanged digest. **Closed.**
2. **`csv.field_size_limit` changed cold acceptance but not node identity.**
   Root reproduced a cold `HEADER_UNREADABLE` at limit 1 while the warm run
   hit every node under an unchanged producer key. **Closed.**

## Done

- Reproduced root's proof myself against the parent tree, byte-identical to
  root's own JSON (`R/us-puf-raw-artifact-review-fixes-parent-replay.json`).
- Closed the envelope header schema, recomputed every fact from the arrays,
  and re-proved the slice's invariants on readback.
- Bound the effective CSV acceptance profile into the producer's
  implementation identity and receipt, and made the pinned
  `csv_profile.field_size_limit` a checked fact rather than a decorative one.
- Deep-froze the definition document after reproducing a normal-API nested
  mutation that changed parse behaviour under an unchanged node key, and made
  the packaged route compare canonical bytes rather than a digest alone.
- Carried Fable 5.1's actionable findings: dtype/width pinning (1), declared
  expectations (2), fresh-process warm proof (3), the vacuous `str()` receipt
  assertion (4), the `bool`/`<u1` dtype drift (5), the envelope side of the
  error-classification finding (6), and the CSV limit (7).
- 110 lane tests, 44 direct neighbours, 290 CSV-precedent neighbours and 72
  executor/store tests pass. The three blindness failures are the untouched
  baseline, identical on parent and child.

## Not done, and why

- Finding 6's **definition** side: a malformed definition document still
  raises `KeyError` out of `_pin` rather than a typed refusal. The envelope
  side is fixed; the definition side reaches only the packaged file and the
  fixture route, and closing it is a separate readable change.
- No genuine PUF delivery, capture, receipt or artifact was opened, and the
  first genuine run still waits on independent peer review of this code.
