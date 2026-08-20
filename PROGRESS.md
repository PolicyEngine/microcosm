# F1 continuation r4 progress

## State

- Work is local on `spec-engine-f1`; nothing has been pushed and no sample
  build has run in this continuation.
- Deliverable 7 is complete at `ef036572`. The dashboard and affected tests now
  consume compiler IR: 173 typed columns partitioned across 132 graph, 28
  family, and 13 take-up authority columns; 227 compiler-expanded producer
  outputs; 241 lossless write-event segments; 170 final graph segments over
  763 exact cells; and 192 exclusive authority variants (150 graph, 28 family,
  and 14 take-up).
- The first required `git fetch origin` was attempted and failed because this
  host could not resolve `github.com`. The final #698 commits are available on
  the cached local `spec-engine-schema` branch and are the next merge source.
- Deliverable 4 remains incomplete: typed authorities are materialized, but
  production physical kernels and all 72 stochastic sites are not yet routed
  through the generic executor and brokers. Deliverables 5 and 6 therefore
  cannot honestly emit PASS receipts.
- The host ceiling is 20 GiB RSS. Prior cold f001 QRF attempts peaked between
  78.91 and 96.95 GiB, so no 1% certification build may be launched here.

## Done

- Completed the required RFC, schema, compiler, review, coverage, pool-tool,
  and lane-journal readings.
- Implemented and tested the generic executor and broker foundations, sealed
  comparison surfaces, source snapshots, provenance routing, and typed bundle
  kernel-authority materialization.
- Completed the deliverable-7 parallel lane without touching closure,
  segments, or dashboard code from the main lane.
- Reverified all four deliverable-7 modules on merged compiler-schema HEAD
  `da45dfcd` in 426.33 seconds and recorded the exact result in
  `FINAL_REPORT_F1_D7.md`.
- Audited the remaining physical executor, RNG, artifact-selector,
  calibration, receipt, and resume seams. The audit found two F0 contract
  corrections and two exact seed-ledger corrections that require regeneration
  and identity re-pinning after the cached #698 merge.
- Kept the historical root journal and all commits local; no logbook chain was
  touched.

## Next

1. Merge the cached final #698 spec-engine tip, resolving conflicts by keeping
   both main's fixes and the 72-site ledger.
2. Correct the two adaptive-cap ledger rows plus the finalizer structural delta
   and late-transfer effects; regenerate bundles, coverage, and every identity
   pin.
3. Run the allowed focused and repository test gates under 20 GiB RSS.
4. Record the exact D4/D5/D6 blockers and owner-run requirements in the lane
   notes, rollout status, and `FINAL_REPORT.md`; stop without fabricating
   certification evidence.
