# Health-coverage labels: cold 1% round-2 diagnosis

Date: 2026-08-25  
Branch input: `7a66c5a9`  
Baseline input: `2c7a7218`

## Read-only receipts and comparison scope

The diagnosis reads, but does not modify, these host-owned artifacts:

```text
/Users/maxghenis/PolicyEngine/_buildo-runtime/out/battery-verify/
  baseline1pct/pool.gates.json
  baseline1pct/pool.checkpoints/stacked/
    2e45c4d60f66b4321bc00ffa22816470bf162c59fd91956514832f97e066ed3c/
      simulated.checkpoint.h5
      acs-transfer/
        091dc2effbe638687de621f3ed4312f738489a58923bf1a7172ac5da8e3c6eb7/
          asec_survey_to_acs/targets/*.h5
  health1pct/pool.gates.json
  health1pct/pool.checkpoints/stacked/
    2e45c4d60f66b4321bc00ffa22816470bf162c59fd91956514832f97e066ed3c/
      simulated.checkpoint.h5
      acs-transfer/
        b7a106bb7c0294c0afa7ba6e51cf94d0c5490e4770d565f973260b2a50fedd0d/
          asec_survey_to_acs/targets/*.h5
```

The two simulated checkpoints have identical `person_id` order, support
channels, clone indexes, and bit-identical person weights. The battery's
implemented scope is clone 0 with positive resolved weight, split on `asec`
versus `acs`
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11523-11559`).
It contains 4,311 ASEC rows with total weight 80,851,529.27715749 and 34,293
ACS rows with total weight 79,926,522.10879111. Weighted incidence is positive
label mass divided by that origin's total scoped mass
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11819-11856`).

## Exact mass movements

| Leg | ASEC source/output mass in both builds | Baseline ACS output | Round-1 ACS output | ACS mass movement |
|---|---:|---:|---:|---:|
| IHS | `NOW_IHSFLG == 1`: 8 rows; 197,606.039624; 0.002444061 | 12 rows; 46,033.921631; 0.000575953 | `HINS7 == 1`: 252 rows; 482,171.216580; 0.006032681 | +436,137.294950; ratio 0.235654 -> 2.468303 |
| Marketplace | `NOW_MRK == 1`: 157 rows; 2,908,019.635846; 0.035967404 | 872 rows; 2,360,680.784026; 0.029535638 | 795 rows; 2,125,795.272305; 0.026596869 | -234,885.511721; ratio 0.821178 -> 0.739471 |
| non-Marketplace direct purchase | `NOW_NONM == 1`: 107 rows; 2,028,850.769034; 0.025093536 | 491 rows; 1,023,422.890043; 0.012804547 | 528 rows; 1,054,016.416743; 0.013187317 | +30,593.526700; ratio 0.510273 -> 0.525526 |
| TRICARE | `NOW_MIL == 1`: 45 rows; 807,558.923656; 0.009988171 | 206 rows; 502,159.944901; 0.006282770 | 153 rows; 385,243.838170; 0.004819975 | -116,916.106731; ratio 0.629021 -> 0.482568 |

The ASEC mapper is the strict, label-specific construction shown in
`packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:57-69`
and applied at
`packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:431-438`.
For all three adjacent labels, the emitted boolean equals the named raw
`NOW_* == 1` indicator on every scoped row in both frames. Their regression is
therefore wholly on the ACS transfer side.

The changes are not a conserved relabeling. Comparing the same ACS persons,
the output transitions are:

| Leg | `false -> true` | `true -> false` | Total changed rows |
|---|---:|---:|---:|
| Marketplace | 751 rows / weight 1,984,318.165265 | 828 / 2,219,203.676986 | 1,579 |
| non-Marketplace direct purchase | 447 / 894,202.344561 | 410 / 863,608.817861 | 857 |
| TRICARE | 112 / 267,202.617390 | 165 / 384,118.724121 | 277 |

## IHS: faithful ACS mapping exposed an incomplete ASEC source surface

Round 1 maps `HINS7` through the generic required yes/no helper to the IHS
leaf (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_inputs.py:115-129`).
The cold frame proves equality, not merely similar incidence: all 252 and only
the 252 scoped ACS rows with `HINS7 == 1` have the output true. The ASEC side
still equals all and only the eight `NOW_IHSFLG == 1` rows. Reverting to the
baseline QRF's 12 positive ACS rows would recreate an underprediction and
would not be a faithful source construction.

The eight ASEC positives were not the complete pooled source concept. The
checkpoint contains 1,368 sampled 2022 rows (weight 25,266,038.707667) and
1,478 sampled 2023 rows (26,972,301.358297) with `NOW_IHSFLG` null on every
row; only the 1,465 sampled 2024 rows (28,613,189.211194) carry the recode.
The generic `_source` helper filled those nulls with zero before `code == 1`,
silently manufacturing `False` for two pooled vintages
(`packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:373-379`,
`:555-560`).

The read-only `asec-720/receipt_720.json` re-extraction receipts restore the
published `NOW_*` recodes without changing pre-existing columns. Joining the
same sampled people to those receipt-backed sources changes IHS as follows:

| ASEC vintage | Sample rows / total weight | Original IHS true mass | Corrected IHS true mass / incidence |
|---|---:|---:|---:|
| 2022 | 1,368 / 25,266,038.707667 | absent -> 0 | 7 / 72,245.403261 / 0.002859388 |
| 2023 | 1,478 / 26,972,301.358297 | absent -> 0 | 7 / 23,196.524112 / 0.000860013 |
| 2024 | 1,465 / 28,613,189.211194 | 8 / 197,606.039624 | 8 / 197,606.039624 / 0.006906117 |
| pooled | 4,311 / 80,851,529.277157 | 8 / 197,606.039624 | 22 / 293,047.966998 / 0.003624520 |

The exact next-host source tuple pinned by that receipt is:

| Year | H5 path | SHA-256 |
|---:|---|---|
| 2022 | `/Users/maxghenis/PolicyEngine/_buildo-runtime/inputs/asec-720/census_cps_2022.h5` | `797fb7e0ea3e94a52dc6c9c7f5a01d70d28a6c5620f3c112bcbd7c6cf6cffe9f` |
| 2023 | `/Users/maxghenis/PolicyEngine/_buildo-runtime/inputs/asec-720/census_cps_2023.h5` | `5fc2feca08aa125ac2ed127711457038e0d493a5ad1d60aff763fb8e42a9d60a` |
| 2024 | `/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/census_cps_2024.h5` | `ec36604cb735a660b51b0b2f90be27d803b5878f3464fb30d0eacead59c1260d` |

The generating repair does not synthesize reporters or tune incidence. It
requires each raw source year to present all nine exact `NOW_*` columns with
only nonblank `{1, 2}` codes before pooling
(`packages/microcosm-build/src/microcosm/build/us_runtime/asec_pool.py:240-247`,
`packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:441-487`).
The next host run must use the corrected receipt-backed inputs; the old inputs
now fail closed instead of building a biased label.

With those corrected sampled inputs, the unchanged native ACS mass implies an
IHS ratio of `0.006032681066 / 0.003624519779 = 1.664408372482`, not 2.468303.
That remains outside the band. On the complete receipt-backed inputs, the
analogous raw weighted-incidence ratio is 1.942759574413. The surveys'
different health-estimate population frames therefore still require the
common-universe adjudication in `experiments/health_labels/PROVENANCE.md`. If
the owner determines that even the common scope cannot make the two reported
IHS measures comparable, the exact owner-only declared-absence candidate is,
for the base scope `B(r)` defined there,

```text
A_IHS(r) = B(r) * 1[channel(r) = "acs"]
B(r) = 1  =>  isna(IHS(r)) = A_IHS(r)
T_IHS(r) = B(r) * (1 - A_IHS(r)).
```

This lane does not install that rule and does not alter the comparator or its
band.

## Adjacent labels: a chained-target hole, not source remapping

The declared boolean order places CHAMPVA, ESI, IHS, Marketplace, Medicaid,
non-Marketplace direct purchase, other means-tested, and TRICARE in the first
eight-target batch
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:356-365`).
Round 1 correctly preserved complete native ESI/IHS output cells, but
`_missing_target_families` removed those two names from the *fit* as well
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:2383-2420`).

The target checkpoint metadata is decisive:

```text
baseline family_targets:
  [CHAMPVA, ESI, IHS, Marketplace, Medicaid,
   non-Marketplace direct purchase, other means-tested, TRICARE]

round-1 family_targets:
  [CHAMPVA, Marketplace, Medicaid,
   non-Marketplace direct purchase, other means-tested, TRICARE]
```

Durable bank indexes remained 3, 5, and 7 because they are allocated from the
requested plan before active targets are selected
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:985-1029`).
That did not preserve the chain. The banked loop fits and advances only the
active tuple, then adds each completed raw draw as the next target's prior
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1561-1671`).
The QRF kernel likewise defines the next predictors as the base predictors plus
the exact completed-target prefix
(`packages/microcosm-fit/src/microcosm/fit/qrf.py:1195-1215`). Removing ESI/IHS
therefore changed the Marketplace state-before hash and every later draw.

The source-faithful repair is to retain every declared family's latent
model-draw step whenever another family peer needs imputation. Its model step
and bank receipt remain part of the declared joint construction, while the
existing target-level null mask continues to prohibit overwriting a complete
native recipient value. This is a derivation invariant, not an incidence
calibration. The implementation activates the full bounded family on any hole
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:2383-2415`),
then continues to write only null recipient cells
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:992-1043`).
Its execution contract and target-bank materializer versions were advanced so
old shortened-chain checkpoints cannot be reused.

Marketplace, non-Marketplace direct purchase, and TRICARE still lack exact ACS
detailed counterparts. Their owner-only absence equation remains the one in
`experiments/health_labels/PROVENANCE.md`; the chain repair must not manufacture
`HINS2` or `HINS5` identities or claim that a red ratio is comparable.

## Next cold 1% expectation

No local or off-chain build was run. One ratio is identifiable exactly from
the read-only receipts:

- IHS: `1.664408372482` when the host uses the corrected ASEC sources; still
  red pending common-universe/absence adjudication.

Marketplace, non-Marketplace direct purchase, and TRICARE are intentionally
not assigned fabricated numbers. Correcting the 2022/2023 ASEC IHS source
changes the IHS donor outcome before Marketplace, so neither existing
checkpoint contains the required restored-prefix plus corrected-IHS
counterfactual. Direct purchase and TRICARE additionally occur after the
narrow `NOW_CAID` Medicaid target. All three therefore require the next host
measurement and remain concept-mismatch legs; the owner-only declared-absence
equation is already recorded in `PROVENANCE.md`.
