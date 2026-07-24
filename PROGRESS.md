# Issue 530 progress

## State

- Active on branch `qbi-port-530` in
  `.claude/worktrees/populace-wt-530`, based on `c09f6db`.
- The supervisor will rebase, push, and open the PR. This run must not make any
  network call.
- The pinned 1.8.0 `puf_2015.h5` and `puf_2024.h5` files are available only
  from `/Users/maxghenis/ops/populace-qbi-port/assets/`; they will be read in
  place and never copied or committed.

## Done

- Read issue #530 and confirmed the processed-PUF audit, Section 199A port, and
  raw-pin investigation scope.
- Read the GitNexus exploration workflow; its MCP tools are unavailable in this
  session, so source tracing will use repository-native searches.
- Inspected the current `puf_tax_detail` manifest, processed-PUF support
  builder, 15-leaf QBI runtime contract, and test conventions.
- Enumerated both pinned HDF files. Each has the same 74 root datasets, no HDF
  attributes, and only one literal QBI-contract dataset:
  `w2_wages_from_qualified_business`.
- Confirmed the archived loader supplied the apparent 15-leaf surface by
  detecting an old artifact, simulating the missing leaves, and mutating the
  file on load (`datasets/puf/puf.py` lines 993-1302). Populace's bare-h5py
  reader does not invoke that retired upgrade path.

## Next

- Finish the aggregate-only artifact lineage audit and commit its tracked
  documentation and ready-to-paste PR summary.
- Vendor the v1 assumptions and implement a pure, version-gated NumPy QBI
  simulation stage that replaces the retired load-time mutation.
- Establish exact equivalence where the artifact exposes a comparison column;
  otherwise add and report distributional-equivalence evidence.
- Identify the true raw asset and either port 2015-to-target-year aging or
  commit a precise design and blockers.
- Add the changelog fragment, final report, and final status/PR text here; run
  formatting and focused tests; commit every coherent step.
