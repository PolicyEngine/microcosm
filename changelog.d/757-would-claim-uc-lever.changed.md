The pre-registered `would_claim_uc` U8 lever (uk-data#452) was run and
measured per the #757 `uk_target_fit` dispositions (issue comment
5427936411), and reverted on the receipts: a raise to 0.85 leaves every
failing UC caseload cell unchanged (`dwp.uc.households` -44.9% at both
rates) while destroying the legacy housing-benefit surface
(`obr.housing_benefit` -0.0% at 0.55 vs -52.4% at 0.85 - the raise moves
claimants off legacy benefits) and perturbing the QRF predictor surface
through engine-computed `household_net_income`. The contract entry stays
frozen at 0.55 and now records the run; the receipts live in the 757-swap
acceptance evidence. The binding constraint on UC caseload is capital-test
support (microcosm#750), not take-up support.
