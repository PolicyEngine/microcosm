# Lane journal — PUF raw source review fixes

Child of `d836bf9060cec348ae9eede25e3d3ec9dd7e754b` on branch
`us-puf-raw-source-review-fixes-20260906`. The parent's rotating `PROGRESS.md`
is left exactly as the parent lane wrote it; this file is this lane's own
state, as the assignment requires.

**Invented fixtures only.** No genuine PUF delivery, capture, receipt or
artifact is opened by anything in this lane, and no engine package is imported.

## State

Started. Read the parent module, tests, definition, docs, the first-slice
REPORT, the plan, the root addendum, the build prompt, the root
counterexamples and the independent Fable 5.1 review (which arrived at 00:23
and is incorporated).

## Assigned defects

1. **Decoder trusts self-reported header facts and column dtypes.** Root
   reproduced 4 typed rows carrying `facts.rows = 987654321` and
   `matched_keys = 777`, and `S006` read back as `float64` from an unchanged
   body with an unchanged digest.
2. **`csv.field_size_limit` changes cold acceptance but not node identity.**
   Root reproduced a cold `HEADER_UNREADABLE` at limit 1 while the warm run
   hit every node under an unchanged producer key.

## Done

- (nothing yet)

## Next

- Reproduce the root proof against the parent tree.
