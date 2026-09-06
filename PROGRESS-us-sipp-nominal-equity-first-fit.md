# SIPP nominal-equity first fit — progress

Branch `us-sipp-nominal-equity-first-fit-20260906`, dependent child of
`9ece4cadddd1933524231a46a959086b0ff74429`. Local implementation and
invented-fixture verification only. No genuine fit, no genuine source or
projection payload open, no push.

## State
- Worktree created from the pinned base; clean at start.
- Controlling documents read in order: root plan, v2 addendum, v3 addendum,
  both Fable 5.1 reviews. v3's column-ownership fix is the adopted design.
- Accepted receipt read as metadata only: 6,183 bytes, SHA-256
  `ce89d7712f61907754896a41d38bff58c3aaa647e886d2a7a91ac70b3bdad7bf`,
  matching the v2 pin.

## Done
- (nothing yet)

## Next
- Packaged consumer contract with the closed pins.
- Public readback sibling module.
- Frame codec, donor admission, support validator, graph builder.
- Invented fixtures and tests; CI registration; report.
