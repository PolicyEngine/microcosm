# 828 — UC capital coherence: I1 before-receipts

Committed, disclosure-safe record of the #828 I1 measurements (2026-08-31), the evidence
anchors for the three `spine_swap_signed_differences.json` entries this PR adds. The raw
measurement scripts and JSON receipts live licensed-side in
`data/ukds/acceptance/828-uc-capital/` (not in this repository); every aggregate below is
weighted or count-based with minimum cell count 3, and each receipt pins its inputs by
digest: `benunit.tab` `66b89462…` (matches the `frs_spine` manifest pin), spine-k H5
`b4403ea4…`, policyengine-uk 2.92.1 at year 2024.

## Part A — TOTCAPB4 domain audit

The FRS 2024-25 `benunit.tab` `TOTCAPB4` column is **fully populated**: 18,850/18,850
benefit units carry a value ≥ 0 (0 NaN rows, 0 negative codes, 2,812 exact zeros, 16,038
positive). **Zero rows map to the −1 unavailable sentinel in this build** — stated loudly
per the round-2 adjudication; the `frs_spine` stage reports its mapped-row count in
checkpoint evidence so a future vintage with absences announces itself. General-population
share above the £16,000 UC capital limit: 26.5% weighted (30.9% unweighted) — against
0.36% among weighted UC reporters, the capital screen that makes donor-preserve unsafe for
synthetic reporters.

## Part B — A3 sizing receipt

Weighted SPI-channel post-fill UC reporters whose donor `TOTCAPB4` exceeds £16,000:
**0.454m weighted (606 records)** — 9× the ruled 0.05m negligibility threshold, so the
conditional redraw stays (adjudication A3). By dependent-children band: 0 → 0.276m,
1 → 0.068m, 2 → 0.096m, 3+ → 0.013m. SPI post-fill reporters total 2.001m weighted, of
which 1.640m are receipt-flips against their donor. Join coverage: 23,301/23,301 SPI
benunit source ids matched to the raw tab.

## Part C — engine blocker aggregates on spine-k

Measured with policyengine-uk 2.92.1 on the spine-k artifact — the "before" side of
acceptance criterion 7:

| Blocker (weighted benunits) | This receipt | Issue #828 evidence |
|---|---|---|
| reported UC and `would_claim_uc = false` | 0.893m | 0.893m |
| reported UC and `uc_assessable_capital` > £16k | 0.936m | 0.939m |
| union | 1.492m | 1.495m |
| false-high: proxy > £16k while own FRS capital ≤ £16k | 0.622m | — (issue's 0.475m was raw-reporter-scoped) |

Reported-UC benunits on spine-k support: 4.375m weighted. The full-spine source-ID join
resolves 61,211/61,211 benunits to raw `TOTCAPB4` — the criterion-1 provenance proof that
benefit-unit capital is reachable with no household-grain reassignment.
