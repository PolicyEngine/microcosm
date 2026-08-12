# Progress

## State

Round 10 mechanism adjudication is complete on `tail-stratum-support-652`.
Smoke-r7 did expose a genuine second producer, but not in the late transfer:
the retirement source callback independently QRF-drew clone 1 and its clone-2
tail descendant after the primary PUF/tail stage. The late-transfer producer
already masks both positive clone roles. The derived repair is one source-owned
ASEC clone-1 draw mirrored byte-exactly to ASEC clone 2, together with an
18-row final-owner matrix for all three target/origin/clone combinations bound
into both the late schedule and the tail manifest.

## Done

- Confirmed a clean worktree on the requested branch and exact retrigger HEAD.
- Confirmed the Round 9 commits are intact immediately below the retrigger.
- Read `CLAUDE.md` and the GitNexus debugging workflow. GitNexus MCP tools are
  unavailable in this session, so the producer/call-path trace will use local
  source, tests, commit history, and the supplied build checkpoints.
- Honored the no-network constraint; no fetch, push, GitHub call, or build has
  been attempted.
- Reconstructed smoke-r7 from its log and target checkpoints. The failing
  batch-3 target bank contains draws only for 38,604 clone-0 recipients; all
  clone-1 and clone-2 slots are null, proving late transfer did not rewrite the
  failing positive-clone cells.
- Confirmed tail construction copies clone 1 byte-for-byte into clone 2 and
  overwrites only the five declared capital-gains tail leaves. The final guard
  compares live clone 2 with live clone 1 by assembly-unique source ID after
  the complete DAG; it does not use a stale pre-source snapshot.
- Identified the actual second writer: `support_role_series` classifies every
  positive clone as PUF support, so the retirement source callback predicts
  clone 1 and clone 2 as separate stochastic QRF rows and overwrites both.
- Derived the certified two-spine ownership matrix. Tuition is PUF-owned on
  positive clones and transfer-owned on native rows; education is a consume-
  only byte-exact no-op. For both retirement overlaps, the ASEC source owns
  clone 0 and the final clone-1 value, ASEC clone 2 inherits that final value,
  ACS clone 0 is transfer-owned, and ACS positive clones are PUF-owned.
- Completed the exhaustive set audit. Primary outputs intersect persisted
  post-clone source outputs, late transfer, and recipient-owned QRF outputs in
  exactly two targets: traditional IRA and self-employed pension desired
  contributions. Adding source callback pass-through/touch outputs yields the
  third audited target, qualified tuition. The corresponding tail-owned
  intersection is empty; no other implicit dual-write target exists.

## Next

- Add red registry and runtime tests for the complete 18-cell ownership matrix,
  exhaustive intersection, tuition no-op, and retirement clone-2 mirroring.
- Bind the registry receipt into late schedule identity and the tail manifest.
- Implement and receipt the byte-exact retirement parent mirror without
  changing the preservation guard.
- Run focused proof, exact 495-test #583 proof, full workspace chunked proof,
  ruff/format/diff checks, and record the changelog and smoke-r8 prediction.
