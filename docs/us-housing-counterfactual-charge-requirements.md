# Counterfactual full contract charge: what is still required

**A required future measurement, not a completed one.** Nothing in Microcosm
produces a counterfactual full contract charge today.
`microcosm.build.us_runtime.asec_housing_receipt_consumer` reports
`counterfactual_full_contract_charge` as unavailable and
`counterfactual_charge_availability()` points at this document. That function
opens no source, fits nothing, and implements no part of the lane described
below.

This file exists so the consumer's machine-readable availability record names a
stable in-repository reference instead of a path that only resolves inside the
session that wrote it.

## What the charge is, and is not

The estimand is the **unassisted comparable-market charge**: what a comparable
unassisted renter household pays for a similar unit, conditional on declared
predictors. It is the engine's `pre_subsidy_rent` input.

It is **not** any of these, and none of them may be substituted for it:

* an assisted household's reported contract rent, which is already net of the
  housing authority's payment;
* the modeled `hud_hap`, which is an engine output computed from the very input
  being solved for;
* the capped SPM housing-subsidy valuation, which is a censored resource measure;
* a fair-market-rent lookup, for the blocking reasons below.

`compose_full_contract_charge` refuses each of those fusions by name.

## Blocking defects measured on the pinned engine

The lane's probes measured these against policyengine-us 1.819.0 for 2024; they
block an FMR shortcut rather than merely qualifying it.

| Defect | What the probe measured |
| --- | --- |
| FMR fall-forward | The bundled fair-market-rent table carries no 2024 fiscal year, and a 2024 request resolves to FY2025. |
| Utility allowance geography | `hud_utility_allowance` is encoded for selected public housing agencies only and is zero elsewhere. |
| Bedrooms | Bedrooms are not observed for the assisted universe, and every FMR and payment-standard path is bedroom-indexed. |
| Undated LA branch | `pha_payment_standard` selects a Los Angeles County schedule from hard-coded dollar amounts with no year parameter, so it does not move with the requested period. |

## The source work that would have to happen first

The candidate source is the authenticated December SIPP producer
(`microcosm.build.sipp_december_source`, documented in
[`us-sipp-december-source.md`](us-sipp-december-source.md)), whose current
household-copied field registry holds vehicle count, value, equity and debt
only — **no** rent, mortgage, utility, tenure or assistance field. Adding them
is a separate source lane with its own review; it is neither started nor
implied by this consumer.

That lane must, at minimum:

1. Transcribe each new field's allocation-flag name, code domain, bounds,
   missing sentinel, basis and unit literally off the pinned dictionary, rather
   than synthesizing a flag name from the existing `T*`/`A*` convention.
2. Keep multi-category tenure and assistance statuses as categories with their
   allocation flags and quality vocabulary, exactly as the ASEC housing-status
   classifier keeps `HPUBLIC`/`HLORENT` and `I_HPUBLI`/`I_HLOREN`, so that
   not-in-universe never becomes an observed negative.
3. Restrict the rent universe by tenure before treating any amount as rent — the
   SIPP amount is rent **or** mortgage — and restrict the unassisted donor
   universe on explicit negative assistance answers only, preserving every
   excluded household with its exclusion reason and count.
4. Keep tenant-paid utilities a separate measurement. The engine's
   `hud_gross_rent` already adds `hud_utility_allowance` to `pre_subsidy_rent`,
   so folding utilities into the charge would double-count them.
5. Declare, separately and explicitly, the monthly-to-annual assumption, the
   price basis carrying 2022 nominal dollars to the target year, and the
   population basis. None of the three may be implicit, and a general CPI-U
   deflator is not a rent price index.
6. State the cohort it targets. The ASEC housing status this consumer reads is
   an interview-time status over income cohorts 2022–2024; a 2022 reference-year
   charge model must not be silently paired with a 2024 ASEC cohort.
7. Report common support between the assisted recipient universe and the
   unassisted donor universe per predictor, including the public-housing and
   low-quality region where donors are expected to be thin, and hold the ACS
   reported contract rent out as an independent comparison rather than a
   fitting target.

No landlord subsidy amount is observed in that source. Any statement that the
resulting number is an observed landlord bill would be false; the identifying
assumption is comparability of assisted and unassisted housing conditional on
the declared predictors, and it must be stated wherever the number is used.

## Status

Unbound. Until a lane satisfying the above ships and is reviewed on its own
terms, `counterfactual_charge_availability()["available"]` stays `False` and
this consumer produces no `pre_subsidy_rent`.
