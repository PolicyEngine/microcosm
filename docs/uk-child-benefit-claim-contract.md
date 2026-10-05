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

The capability is read from the installed model's parameter tree alongside the
charge thresholds. The stage receipt's `claim_export` records the selected
encoding, capability and engine parameter source, including its package version.
This avoids guessing the version number of the release containing #2140 and
keeps the currently locked older engine compatible. Both ordinary spine builds
and graph replay reconstruct the same stage transform and read that boundary.

The stage's `claims`, `opt_outs` and `in_payment` statistics describe the draws
against HMRC targets. `in_payment` still counts claims excluding drawn opt-outs;
it does not rerun the model to measure cash payments. In particular, the draw
can select families inside the charge taper when there are too few fully
charged families, while the model's default behavioral threshold may pay them.
This change does not alter that draw or calibration policy.

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
