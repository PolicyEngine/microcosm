# F1 deliverable 7 progress

## State

- Deliverable 7 discovery is complete. The implementation target is the
  compiler-declared closure (173 typed columns), 227 compiled producer-output
  occurrences, and 241 compiler-derived cell segments, plus the dashboard
  projection that presents them.
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
- Audited the current tree and held `lineage-column-closure-697` branch across
  tests and tools. Active consumers are the lineage dashboard, the high-level
  lineage conformance test, production-shaped typed-closure fixtures, take-up
  segment fixtures, and the derived closure block in the US bundle test.
- Confirmed the dashboard currently reads normalized authored graph outputs:
  92 rows, rather than the compiler-expanded 227 outputs, and omits all 241
  compiled `write_scopes[*].cell_segments`.
- Confirmed the allowed held H5 inventory is not a valid current closure
  authority: its 392-column f025 snapshot predates later predictor work and is
  documented as missing 56 columns. It will not be revived or represented as
  current compiler closure.
- Mapped replacements to `CompiledSpecIR.resource()`, `typed_inventory`,
  `producer_graph.nodes[*].outputs`, and
  `producer_graph.nodes[*].write_scopes`.

## Next

1. Retarget the dashboard to compile the packaged bundle, present compiler
   column closure and exact cell segments, and stop reading authored graph
   outputs.
2. Move each production-shaped closure/segment test fixture to compiled IR;
   retain only genuinely synthetic low-level mutation tests.
3. Run focused and complete required suites before the coherent implementation
   commit.
4. Append a clearly marked deliverable-7 handoff section to
   `_F1-LANE-NOTES.md`, write the final report to the requested output file,
   and leave the full suite green.
