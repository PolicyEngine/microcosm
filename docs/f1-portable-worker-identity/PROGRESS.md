# F1 portable worker identity progress

## State

The portable worker identity, authenticated H5/scoring boundary, and typed
specification mirrors are implemented and focused green; downstream pinned
hashes, documentation, and full verification are in progress on
`f1-portable-worker-identity` from base
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
- Added a closed, versioned worker identity that binds interpreter bytes,
  implementation/version/ABI/cache tag, canonical semantic `pyvenv.cfg`, the
  exact approved lock, transitive source imports, verified installed RECORD
  contents, canonical argv, and fit/predict controls. Absolute executable,
  prefix, and raw argv aliases are retained separately for audit.
- Switched current primary-QRF resource authentication and checkpoint-resume
  comparison to semantic projections while preserving full audit aliases in
  receipts. Bumped the primary config, checkpoint sidecar, late registry,
  producer receipt, transition authority, and stacked authority sources.
- Green evidence: Ruff passed for the three implementation files and the three
  committed portable-identity regressions passed (`3 passed`).
- Added the explicit schema-9, gate-failed, scoring-only compatibility path.
  Its plan-bound attestation seals the manifest and H5 digests, the exact
  plan-published campaign identifier and campaign lock, the installed
  transitive environment/code identity, recorded worker, semantic worker, and
  the exact two legacy alias mismatches. Neither readiness/release loader has
  an attestation parameter.
- Threaded the authenticated legacy context through both late-DAG validation
  passes and reconstructed the frozen schema-4/config, registry-16,
  receipt-3, transition-1, and authority-11 identities without weakening the
  current validators.
- Published current worker-authentication evidence in manifests, diagnostics,
  authenticated H5 capabilities, release receipts, and head-to-head scorer
  identity/loader receipts. Added the candidate-only scorer CLI attestation
  argument and direct propagation coverage.
- Kept deny-list refusal intact and made compatibility provenance impossible
  to release-launder: schema 9, private legacy provenance, scoring-only receipt
  fields, a changed returned manifest payload, and mismatched current receipts
  each fail closed.
- Hardened the semantic identity implementation so source discovery follows
  the worker module actually resolved by the interpreter, every source and
  installed RECORD byte is re-read at authentication time, direct external
  imports must resolve into the hashed RECORD, package initializers are part of
  the transitive closure, and worker-startup package resources are hashed.
- Extended the regression surface with the 18-field semantic matrix, every
  post-`argv[0]` position, exact legacy attestation/mismatch matrices,
  release-laundering cases, scorer propagation/receipts, and the exact
  non-mutating 12-household origin battery.
- Latest green evidence: four source/resource/scorer unit cases passed; four
  end-to-end current/legacy H5 authentication and release cases passed; the
  12-case legacy attestation/laundering subset passed. Ruff check passed on the
  edited Python boundary files.
- Replaced both constants-era worker templates with the closed portable
  resolver algebra, made the typed projector resolve the alias-free semantic
  receipt, and retained semantic worker fields in inventory identity while
  excluding only `audit_aliases`.
- Updated the imputation and spine JSON schemas and checked-in US YAML mirrors
  to primary config 5, registry/schedule 17, producer receipt 4, transition
  authority 2, resource semantics 2, stacked authority 12, and checkpoint
  materializer 13. JSON parsing, Ruff, bundle loading, and all four imputation
  projector tests pass.

## Next

- Refresh every observed spec/authority/inventory hash and coverage count,
  including generated coverage evidence and version assertions.
- Document the operator attestation/receipt contract, add the changelog, then
  run focused, CI-group, and required repository checks.
