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
- Added the aggregate-only 74-column artifact audit, including all 40 logical
  inputs (34 raw-field lineages and six retired derivations), all 34 unused
  arrays, the exact current 24-output contract gap, the four requested
  direct-mapping classifications, and ready-to-paste PR text.
- Vendored `qbi_assumptions_v1.yaml` and ported the archived Section 199A
  qualification, SSTB, W-2, UBIA, REIT/PTP, and BDC simulations into pure,
  seeded NumPy. The version, PCG64 generator, four seeds, source order, and
  exposure order are all explicit and validated before a stream is consumed.
- Wired the production PUF donor boundary to request
  `qbi_simulation_version=1`, replace the physical artifact's stale W-2 proxy,
  and create the 15-leaf contract before the shared weighted QRF.
- Pinned all 15 output streams with a compact golden fixture. On the full
  restricted artifact, the replay's five headline SHA-256 values match the
  independent archived-code replay, including W-2
  `3fa8f57f...008fb`, UBIA `d818169f...964f`, SSTB
  `4778f172...20d5e`, REIT/PTP `e913f2e0...faf`, and BDC
  `0f97dc7d...2eb`.
- Classified equivalence honestly: the Populace stream is exact to the
  archived algorithm, but literal artifact-column equivalence is impossible
  because 14 leaves are absent and its one physical leaf is an older
  deterministic W-2 proxy. Distributional checks on the replay find weighted
  nonzero shares of 0.0939% for W-2, 4.7821% for UBIA, 3.2754% for SSTB,
  4.9007% for REIT/PTP, and 0.6740% for BDC. Every current plausibility band
  passes except total W-2, which is 0.0061 percentage points below its 0.1%
  lower bound.
- Ran Ruff and the QBI, QBI-reconciliation, PUF-donor, and staged-builder
  focused test files successfully, including the restricted-artifact test.

## Next

- Identify the true raw asset and either port 2015-to-target-year aging or
  commit a precise design and blockers.
- Add the changelog fragment, final report, and final status/PR text here; run
  formatting and focused tests; commit every coherent step.
