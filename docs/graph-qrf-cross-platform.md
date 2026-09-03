# fit.qrf@1 across platforms: a measurement

Date: 2026-09-03. Method: `tools/graph_qrf_platform_probe.py`, run natively on
arm64 (Python 3.14) and under Rosetta from an x86_64 environment synced to the
same `uv.lock` (torch stubbed; it is not on the kernel's path). Twenty cases:
seeds 0–4 in four regimes (positive-only, mixed-sign, near-ties, zero-inflated),
600 donors and 300 recipients each, unweighted fits, one draw per recipient.

## Result

| statistic | value |
|---|---|
| cells compared | 6000 |
| cells that differ | 45 |
| largest absolute movement | 2.322e+00 |
| largest relative movement | 6.825e-02 |
| largest int64-view distance (not a ulps count once a donor flips) | 450359958233450 |

Cases with any difference (cases not listed were bit-identical):

| case | max abs | max rel | int64-view distance | differing |
|---|---|---|---|---|
| positive/0 | 3.907e-02 | 3.194e-02 | 175940752005003 | 3/300 |
| positive/1 | 7.406e-02 | 6.825e-02 | 333524120092336 | 2/300 |
| positive/2 | 2.220e-16 | 1.777e-16 | 1 | 2/300 |
| positive/4 | 1.110e-16 | 1.659e-16 | 1 | 1/300 |
| mixed_sign/0 | 2.220e-16 | 1.954e-16 | 1 | 1/300 |
| mixed_sign/1 | 1.901e-02 | 7.927e-03 | 42798880769607 | 2/300 |
| mixed_sign/3 | 4.441e-16 | 1.399e-16 | 1 | 1/300 |
| mixed_sign/4 | 2.220e-16 | 1.747e-16 | 1 | 1/300 |
| near_ties/0 | 1.000e-01 | 6.667e-02 | 450359958233450 | 8/300 |
| near_ties/1 | 1.000e-09 | 6.342e-10 | 4503600 | 3/300 |
| near_ties/2 | 1.399e-02 | 8.227e-03 | 62989703563009 | 6/300 |
| near_ties/3 | 1.000e-09 | 1.377e-09 | 9919248 | 6/300 |
| near_ties/4 | 7.095e-03 | 4.225e-03 | 31951854077424 | 5/300 |
| zero_inflated/0 | 1.819e-12 | 1.598e-16 | 1 | 1/300 |
| zero_inflated/2 | 1.819e-12 | 1.533e-16 | 1 | 2/300 |
| zero_inflated/3 | 2.322e+00 | 5.400e-04 | 2553496694405 | 1/300 |

## Reading

Most differing cells move by one ulp: ordinary floating-point reassociation
between the two architectures. A few cells move by up to 7% relative: a
one-ulp difference inside the forest flips which donor a quantile draw lands
on, and the drawn value jumps to a different donor's value. That is not a
rounding error a per-cell `Tolerance` could bound; it is a discrete outcome
that depends on the platform.

## Consequence (amendment 16)

`fit.qrf@1` declares `Numeric.PLATFORM_BITWISE`: identical bytes on one
platform (H1 parity holds on the platform that produced the pins), no bound
across platforms, and no `Tolerance`. Gates that compare its output across
platforms must say so in their evidence. Earlier drafts declared
`Tolerance(ulps=1)` and then `Tolerance(rtol=1e-6)`; both were unmeasured
and both are false, as the table shows.
