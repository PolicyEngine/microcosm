# F1 portable worker identity progress

## State

Implementation started on `f1-portable-worker-identity` from base
`09abf2ad78e9af3c5314a4b303d42a75e30d49c4`.

## Done

- Read `CLAUDE.md`, the reproduced refusal in `_inputs/STOP.md`, and the F1
  specification in `_inputs/FIX-PLAN.md`.
- Confirmed that runner-owned root journals and task inputs will remain
  untouched.

## Next

- Map the worker execution identity, authenticated H5 loaders, deny-list, and
  receipt propagation paths.
- Add fail-before regressions for portable semantic identity and the explicit
  scoring-only legacy attestation.
- Implement the versioned identity and loader boundary, then run focused and
  required repository checks.
