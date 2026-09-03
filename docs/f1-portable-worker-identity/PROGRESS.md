# F1 portable worker identity progress

## State

Fail-before regressions are committed and the portable identity implementation
is in progress on `f1-portable-worker-identity` from base
`09abf2ad78e9af3c5314a4b303d42a75e30d49c4`.

## Done

- Read `CLAUDE.md`, the reproduced refusal in `_inputs/STOP.md`, and the F1
  specification in `_inputs/FIX-PLAN.md`.
- Confirmed that runner-owned root journals and task inputs will remain
  untouched.
- Traced both authenticated late-DAG validations, every public scoring and
  release loader, the deny-list layers, generated-spec mirrors, and downstream
  release/scoring receipt propagation.
- Specified fail-before coverage for portable semantic equality, semantic
  tamper refusal, the sealed STOP alias mismatch, and the explicit legacy
  scoring-only attestation boundary. The focused portable-identity nodes fail
  against the base implementation with two missing semantic identities and one
  missing legacy mismatch helper (3 failed).
- Mapped the required version cascade: worker identity 1, primary config 5,
  primary-QRF sidecar 2, resource semantics 2, registry 17, producer receipt
  4, transition authority 2, stacked authority 12, checkpoint materializer 13,
  and pool manifest 10.

## Next

- Implement the portable semantic identity and the scoring-only legacy loader
  boundary without changing release API signatures.
- Regenerate typed specifications, surface receipts, document the contract,
  and run focused and required repository checks.
