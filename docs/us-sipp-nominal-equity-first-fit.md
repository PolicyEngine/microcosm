# SIPP nominal December-2022 equity: first source-named fit

This consumer turns one already accepted SIPP December-2022 source artifact
into a donor Frame, fits the real weighted sign-regime QRF over a single
predictor, and draws for a caller-supplied recipient population.

## Scope, and what this is not

The target is `sipp_all_vehicle_equity_dec2022`: the published all-vehicle
equity recode at residence/household grain, signed, in nominal December 2022
dollars. It is a **known-target subset** defined by an explicit admission
rule, not a US population estimate, not a PolicyEngine input, not a 2024
stock, and not a validated year transfer. Donor weights are SIPP source
design weights; they are not calibrated recipient weights. A request for any
other concept, or for 2024, refuses before any source access — the year
bridge remains outstanding work.

## Admission

Every December residence and its complete member roster is reconstructed
first, and its mean `WPFINWGT` computed over all members, before any target
or weight filtering. Only then does the target-specific rule apply:

1. a known, household-copy-agreed equity measurement;
2. a known complete-member mean weight;
3. a strictly positive mean weight.

Nothing is reweighted. An unknown weight stays unknown, never zero. Flag 9
(`component_derived_allocation_unresolved`) is the genuine source's only
observed `AHEQ_VEH` flag; admission retains those residences and never
relabels the quality as reported. Vehicle count is a diagnostic, not an
admission requirement or a predictor. Excluded residences are removed
*before* donor CREATE, so the FIT receipt's own positive-weight exclusion
count is zero; every one of them remains in the full coverage report with its
mutually exclusive reason.

For the accepted all-SSUID artifact the packaged contract expects 17,588
admitted residences out of 17,646, with 20 known-equity zero-weight and 38
conflicting-equity residences retained outside the fit.

## Modules

- `microcosm.build.sipp_equity_contract` — the packaged pin contract
  (`sipp_equity_contract_v1.json`) and the concept/period refusal.
- `microcosm.build.sipp_december_readback` — a public sibling of the December
  producer that reads the two closed objects without rereading the 3.7 GB
  parent. It reuses the producer's private `_bounded_object`, `_counts`,
  `_membership` and `_measurement_digest` helpers as an explicit, tested
  internal dependency. The producer module itself is byte-for-byte unchanged,
  so its accepted historical output identity still holds. The historical
  producer binding and contract are validated **by comparison**; the receipt
  and projection key formulas are recomputed only as a consistency check.
- `microcosm.build.sipp_equity_graph` — the prepared read-only source store,
  the `sipp-december-equity-v1` Frame codec, the three kernels and the graph
  builder.

## Graph shape

    donors (CREATE)  ──> fit (QRF train) ──┐
        │  donor_support artifact          │ model artifact
        ▼                                  ▼
    recipients (CREATE, owns              draw (QRF apply)
      roster_household_size)                  ▲
        └──> support (owns household_size) ───┘

The support node reads `roster_household_size` and the typed donor-support
artifact and **owns** `household_size`, copying values only after every
requested size passes the support check. DRAW's Slice is exactly the model
predictors, so the compiler derives a real predecessor edge from column
ownership — not from lexical node ordering. A refusal inside the validator
rejects the node and aborts the run before DRAW, in cold, fresh and warm
paths alike.

## Model configuration

Two trees, seed 0, `zero_atol=0.0`, `max_samples_leaf=None`, predictor
`household_size` only, no extrapolation and no default fill. Exact source
`Decimal` weights are preserved in Frame metadata beside the checked float64
conversion performed at CREATE, because typed Frame weights are float64.

## Genuine execution

Not included here. Everything above is verified against invented fixtures
only. A separately reviewed private rung may fit the full admitted subset and
draw a small explicit recipient fixture; that fixture would be an engineering
recipient population, not a US population estimate. Engine mapping, the 2024
bridge and scientific validation remain separate outstanding work.
