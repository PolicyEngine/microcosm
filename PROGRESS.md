# Round 8 progress: cross-origin structural absence

## State

The smoke-r5 mechanism is adjudicated: `TYPEHUGQ` is an ACS-only structural
input consumed solely by ACS group-quarters validation and its GQ-rent
lineage. The declaration must use `acs_source` on both the input and the
primary structural output; ASEC rows must remain absent and need no synthesized
value or tolerated-absence receipt. Implementation and regressions are next.

## Done

- Read `CLAUDE.md` and the applicable debugging workflow.
- Confirmed the requested branch and exact starting commit.
- Compared the checkout with the locally cached `origin/main`. The branch is
  75 commits ahead and 15 behind that cache; the requested checkout is being
  preserved because this round explicitly targets PR #660 at `27b07c73` and
  forbids network access.
- Confirmed the GitNexus graph tools are unavailable in this session; the
  equivalent call-site and execution-flow trace will be performed directly
  from source, checkpoint receipts, and tests.
- Read the smoke-r5 launcher, error receipt, chained logbook spool row, assembly
  manifest, and checkpoint H5 without modifying them.
- Proved the exact checkpoint partition: 17,004 households; all 1,688 ASEC
  households have absent `TYPEHUGQ`; all 15,316 ACS households have populated
  codes 1/2/3 (13,421 / 904 / 991). The missing mask exactly equals the ASEC
  household-origin mask.
- Traced schema alignment as the source of the legitimate ASEC nulls and the
  blanket `whole_pool` inventory conversion as the source of the false gate.
- Confirmed every semantic `TYPEHUGQ` read is restricted to ACS households.
  The ACS earnings-universe lineage does not consume it; that producer uses
  ACS person channel, age, WAGP, and SEMP.
- Chose origin scoping over an absence receipt. The primary structural output
  must carry the same ACS scope so post-callback completeness and all 19
  downstream transfer dependencies remain consistent.

## Next

- Complete the exhaustive origin-specific raw-input inventory, including the
  already-scoped ACS earnings and ASEC source-operator surfaces.
- Add a per-requirement scope override to the inventory declaration, bump its
  schema identity, and bind `TYPEHUGQ` input/output coverage to `acs_source`.
- Add the exact 1,688-row regression, ACS-side fail-closed regression, and an
  exhaustive cross-origin raw-input audit.
- Run the requested focused, #583, full-workspace, formatting, lint, and diff
  checks; obtain an independent read-only review; report the smoke-r6
  prediction to stdout.
