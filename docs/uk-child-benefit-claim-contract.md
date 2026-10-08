# Child Benefit claims and payment opt-outs

The `child_benefit_take_up` stage draws registered claims and payment opt-outs
separately. Registered claimants include families that opt out of payment.
`would_claim_child_benefit` must retain that registration for the model to
restore payment when a reform removes or sufficiently reduces the High Income
Child Benefit Charge. See [policyengine-uk#2140](https://github.com/PolicyEngine/policyengine-uk/pull/2140)
and [policyengine-uk#2065](https://github.com/PolicyEngine/policyengine-uk/issues/2065).

The stage supports two installed-engine contracts:

| Installed model | `would_claim_child_benefit` export | Receipt encoding |
| --- | --- | --- |
| Exposes `gov.hmrc.child_benefit.opt_out_charge_share` | Registered claims, including payment opt-outs | `registered_claims` |
| Does not expose that parameter | Claims excluding payment opt-outs, preserving the older model's payment behavior | `legacy_payment` |

The capability and actual `opt_out_charge_share` value are read from the installed
model's parameter tree alongside the charge thresholds at the build year's
January 1. The stage receipt's `claim_export` records the selected encoding,
capability and engine parameter source, including its package version;
`opt_outs.opt_out_charge_share` records the evaluated share, or `null` for legacy
models. A present but nonnumeric or nonfinite share is rejected.
This avoids guessing the version number of the release containing #2140 and
keeps the currently locked older engine compatible. Both ordinary spine builds
and graph replay reconstruct the same stage transform and read that boundary.

For opt-out-aware models, the draw selects only claiming nonreporters whose
charge fraction is positive and at least the installed share. With the default
share of one, only fully charged families can opt out. If that pool cannot carry
HMRC's target, `pool_shortfall_families` and `pool_exhausted` record the shortfall;
the stage does not fill it with families the model would pay. A lower share can
admit taper families; share zero still excludes incomes at or below the charge
start, and a share above one leaves the pool empty. Fully charged candidates
retain priority. Equal or reversed taper endpoints remain unsupported by the
build boundary (`0 < start < end`); the stage does not infer a cliff rule.

Older models retain the original draw: fully charged families first, then taper
families for any remainder. Neither contract opts out reporters or nonclaimants,
and units without eligible children keep their early claim draw. Identity-keyed
random draws and claim-rate solving remain unchanged.

The stage's `claims`, `opt_outs` and `in_payment` statistics describe the draws
against HMRC targets. `in_payment` counts claims excluding drawn opt-outs, whose
selection now matches the installed baseline payment-suppression rule. It does
not rerun the model to measure cash payments. Synthetic exported-dataset tests
check paid-family and paid-child counts and payment amounts against the model;
they do not establish the population fit after this changed draw.

Use a model containing #2140 and regenerate the Child Benefit stage to produce
the registered-claim contract. Updating model code alone does not migrate old
Microcosm exports: their false claim flag may mean an opted-out registered
family. A migration must establish producer provenance and registration first.
Never infer claims by OR-ing the flags in the UK model: EnhancedFRS draws
opt-outs independently, so false/true can instead mean a genuine nonclaimant.
EnhancedFRS remains compatible with the UK model's independent claim gate.

Code CI uses synthetic frames; it does not rebuild or certify population data.
Existing releases retain their original recorded model/data contract until a
separately authorized rebuild and publication.

Rollout belongs with the engine-upgrade work in [Microcosm #1095](https://github.com/PolicyEngine/microcosm/issues/1095):

- Upgrade the model and rebuild the Child Benefit stage and exported data.
- Inspect the explicit opt-out pool shortfall at the build's weights.
- Compare before/after model-paid families, children and payment amounts.
- Measure the actual `obr.child_benefit` fit before approving calibration or release.
