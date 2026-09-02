# #685 net-new UK stages receipts

## Part A — WAS debt source audit

The I1 audit used the manifest-pinned Wealth and Assets Survey round-8
household tab (15,128 rows; licensed source bytes remain outside the
repository). It found no missing or negative values in `HMortGR8`: 10,560
rows were zero and 4,568 were positive.
`consumer_debt`, defined as `HFINWR8_SUM - HFINWNTR8_exSLC_Sum`, had no
missing or negative values: 7,999 rows were zero and 7,129 were positive. The
declared zero clip is therefore a no-op on this vintage while remaining part
of the source contract.

`Ten1R8` confirmed the mortgage-tenure predictor. Mortgage-positive shares
were 6.9% for code 1 (8,135 rows), 99.9% for code 2 (3,862), 100% for code 3
(64), 2.4% for code 4 (2,908), 10.8% for code 5 (157), and 0% for code 6
(one row); the single `-8` sentinel maps to false. The donor indicator is
therefore `Ten1R8 in {2, 3}`, paired with recipient
`tenure_type == OWNED_WITH_MORTGAGE`.

Using `R8xshhwgt`, the donor mortgage-positive share was 0.3226 and mean
mortgage debt was £46,915 per household. The donor consumer-debt-positive
share was 0.5296 and mean consumer debt was £3,623 per household. Comparison
with the realized spine is **pending — I5 licensed acceptance**.

## Part B — UC deduction attribute contract

The `uc_deduction_attributes` stage writes four benefit-unit columns after
`uc_capital_coherence`: two identity-keyed draws rounded to float32 and
clamped below one, a region-conditioned pre-cap latent deduction rate, and a
PolicyEngine-UK deduction-combination enum member name. It assigns latent
attributes to every benefit unit; the nonzero latent-rate share is not a UC
caseload statistic. The committed DWP distribution resource mirrors the
PolicyEngine-UK 2.92.1 parameter tree, and hermetic and engine-bearing tests
pin the resource, mapping, enum names, permutation invariance, and held-draw
round trip.

Licensed-spine realization figures, twin identity, claimant-only engine
round-trip statistics, and the non-claimant zero-effect check are
**pending — I5 licensed acceptance**.

## Part C — bus-fare closure on the landed E6 stages

The #685 bus construction requirement is already present in E6; this
increment adds no bus-stage code.

| Output | Stage | Manifest operation | Support clip | Bounds | Export allow-list | Coverage (family status) |
|---|---|---|---|---|---|---|
| `bus_fare_spending` | `lcfs_consumption` | `fit_weighted_qrf_chain` | Declared; stage-health high/low allowances both zero | `[0, 9000]` | `household.bus_fare_spending` | family status: `required_at_build` |
| `bus_subsidy_spending` | `etb_services` | `fit_weighted_qrf_chain` | Declared; stage-health high/low allowances both zero | `[0, 20000]` | `household.bus_subsidy_spending` | family status: `required_at_build` |

## Part D — licensed bus measurement hand-off

Spine design-weight totals by geography, the fare income-quintile gradient,
and comparison with Chronicle BUS05i/NTS0705a facts are
**pending — I5 licensed acceptance**. This part will provide the measurement
hand-off to #789 and #790; no figures are inferred here.
