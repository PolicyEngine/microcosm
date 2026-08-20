# F1 deliverable 7 progress

## State

- Deliverable 7 is in discovery: retarget every derived closure, segment, and
  lineage-dashboard consumer from held authored-class fixtures to compiled-IR
  outputs (`typed_closure.py` and `compiler_ir` seams).
- Work is local on `spec-engine-f1`; nothing has been pushed and no build has
  been run.
- Main-lane deliverables 5/6/8 have concurrent uncommitted changes in this
  shared worktree. They are outside this split-out scope and will not be staged
  or altered except for the required append-only coordination note.
- Resource ceiling for this split-out is under 15 GiB RSS; builds above the 1%
  sample rung are prohibited.

## Done

- Read `CLAUDE.md`, `_F1-CHARTER.md`, and `_F1-LANE-NOTES.md` in full.
- Read and adopted the GitNexus refactoring workflow for impact discovery and
  post-change verification.
- Reconciled the charter's historical instruction not to touch `PROGRESS.md`
  with the split-out standing order: this journal follows the newer explicit
  user instruction and is isolated from the main-lane journal changes.

## Next

1. Complete the charter-mandated source reading relevant to the compiler and
   typed derived surfaces, then inventory all held-fixture consumers in tests
   and tools.
2. Map each consumer to a direct compiled-IR output and record the intended
   retarget before editing implementation or tests.
3. Retarget in coherent commits, running focused and complete required suites
   after each commit.
4. Append a clearly marked deliverable-7 handoff section to
   `_F1-LANE-NOTES.md`, write the final report to the requested output file,
   and leave the full suite green.
