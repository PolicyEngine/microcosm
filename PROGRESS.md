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
- Added the aggregate-only 74-column artifact audit, including all 40 physical
  columns selected before fail-closed donor validation (34 raw-field lineages
  and six retired derivations), all 34 unused arrays, the original 24-output
  contract gap, the four requested direct-mapping classifications, and
  ready-to-paste PR text. QBI v1 closes 14 leaves; ten non-QBI leaves remain
  absent from the transitional HDF.
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
- Identified the true restricted inputs: `puf_2015.csv` is the governing
  207,696-row tax-return source, `demographics_2015.csv` is a retired
  personization supplement, and `irs_puf_2015.h5` is only their two-table
  wrapper. The archived prerequisite records no revision or SHA, so the
  observed local digests remain certification candidates rather than
  authoritative remote pins.
- Proved the artifact labeled 2024 contains only 2015-to-2021 aging. The
  intended 2021-to-2024 processed-array loop iterated the wrong DataFrame axis
  and applied no factors. Release arithmetic also double-aged `E00900` and
  `E26270`; their effective positive/negative factors are now explicit.
- Added a pure raw-PUF aging engine with immutable factor/provenance objects,
  exclusive ownership checks, separate signed legs, strict finite/schema
  validation, aggregate provenance, and a fail-closed `uprate` source-runtime
  handler. The raw manifest now orders disaggregation, raw aging, then
  semantic derivation.
- Vendored audit-only `archived_1_8_0` assumptions for 2015-to-2021 parity.
  Full restricted-data tests reproduce processed weights, direct fields,
  signed business fields, dividend/rental/estate identities, release tuition,
  and the old W-2 formula from raw inputs. The future manifest version is
  `ledger_v1`; it is deliberately unavailable without an explicit factor
  bundle.
- Documented the production raw-pin sequence and blockers in
  `PUF_RAW_PIN_AND_AGING.md`: certified licensed-source revision, historical
  and active-year Ledger facts, corrected target-year policy, HDF-only
  builder ingestion, ten missing non-QBI leaves, source-runtime QRF/clipping,
  and the raw-frame donor adapter.
- Ran Ruff and the combined raw-aging, source-runtime, plan, QBI,
  reconciliation, donor, and staged-builder tests successfully. An offline
  wheel build could not start because Hatchling is absent from the local uv
  cache; no network operation was attempted.

## Final status

### Status summary

- All three issue #530 deliverables are implemented and committed: the
  74-column artifact audit, the explicit Section 199A v1 simulation port, and
  the raw-pin/aging implementation plus migration design.
- QBI equivalence is exact to the archived v1 algorithm and random streams,
  but necessarily distributional against the literal release artifact: 14
  leaves are absent and its sole physical QBI leaf is an older deterministic
  W-2 proxy.
- Raw aging arithmetic is exact for the audited release path through effective
  value year 2021. No claim of 2024 equivalence is made because the retired
  2021-to-2024 loop was a no-op.
- The changelog fragment is `changelog.d/qbi-port-530.added.md`; the complete
  handoff is in `FINAL_REPORT.md`.
- Ruff, JSON parsing, offline lock checking, and 187 focused tests pass,
  including both restricted raw-to-processed aging parity and full-artifact
  QBI replay.

### Blockers

- The private raw-source declarations have no immutable revision or hash.
  Observed local candidate digests require licensed-source certification.
- A corrected `ledger_v1` production factor bundle needs 2015 and active
  target-year facts that are not present in the local Ledger surface.
- The bespoke production builder still reads processed HDF root arrays and
  does not execute the raw source stage.
- After QBI supplies 14 leaves, the transitional HDF still lacks ten non-QBI
  donor outputs.
- The generic source runtime still lacks the manifest's weighted-QRF tail and
  compatible aggregate clipping/adapter path.
- The direct v1 replay's weighted W-2 nonzero share is 0.0939%, just below the
  current 0.1% plausibility floor; all other direct-replay bands pass.
- The offline environment lacks cached Hatchling, so a wheel build could not
  start. The supervisor should verify the two packaged YAML resources after
  rebasing.

### Suggested PR title

Port Section 199A simulation and audit processed PUF inputs

### Suggested PR body

## Summary

- Audit all 74 arrays in the pinned 1.8.0 PUF, separating 40 selected physical
  columns (34 raw-field lineages and six retired derivations) from 34 unused
  arrays and documenting the remaining donor-contract gaps.
- Vendor `qbi_assumptions_v1.yaml` and port the archived seeded qualification,
  SSTB, W-2, UBIA, REIT/PTP, and BDC simulation into Populace-owned NumPy
  logic behind explicit `qbi_simulation_version=1`.
- Identify the restricted raw CSVs, add a versioned/fail-closed raw aging
  engine and archived parity profile, and document the certified-raw plus
  Ledger-backed production migration.

## Equivalence

The new QBI engine reproduces the archived v1 streams exactly. Full-artifact
replay hashes are W-2 `3fa8f57f...008fb`, UBIA `d818169f...964f`, SSTB
`4778f172...20d5e`, REIT/PTP `e913f2e0...faf`, and BDC
`0f97dc7d...2eb`. Literal artifact-column equivalence is unavailable because
14 leaves are missing and the physical W-2 proxy is a different algorithm.
Distributional replay gives weighted nonzero shares of 0.0939% (W-2), 4.7821%
(UBIA), 3.2754% (SSTB), 4.9007% (REIT/PTP), and 0.6740% (BDC).

## Raw pin

The actual source is the restricted 2015 tax-return CSV; both staged HDF files
are processed exports. The artifact labeled 2024 has effective value vintage
2021 because its later aging loop was a no-op. The raw switch remains blocked
on licensed-source certification, complete Ledger factors, and production
builder integration; the new runtime fails closed rather than substituting
archived constants for `ledger_v1`.

## Verification

- Ruff check and format: pass.
- 187 focused source, aging, QBI, donor, plan, and builder tests: pass.
- Restricted raw-to-processed arithmetic and full-artifact QBI replay: pass.
- `uv lock --check --offline`: pass.
- `git diff --check` and manifest JSON parse: pass.

Refs #530
