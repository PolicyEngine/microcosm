The UK spine now sets policyengine-uk's `is_in_full_time_education` (microcosm#1095). The `frs_education` stage writes it true where the FRS full-time education answer is yes: `EDUCFT` = 1 on the 2024-25 adult and child tables, or `FTED` where an earlier vintage carries it. `current_education` already reads the same answer. Until now the input stayed at a signed default of false, so every claimant met the Universal Credit condition of not receiving education that policyengine-uk 2.173.2 added.

policyengine-uk 2.174.1 reads the flag only in `uc_receiving_education`:

- a full-time student on an advanced course (`current_education` TERTIARY), or a qualifying young person in full-time education, now receives education unless one of the engine's exceptions covers them (`uc_is_excepted_from_education_condition`: PIP, DLA or Attendance Allowance with limited capability for work, responsibility for a child, and the others it lists);
- a benefit unit is eligible only if one of its claimants meets the condition, and a reported award does not keep an ineligible student on Universal Credit;
- a full-time student below advanced level who is not a qualifying young person still meets it, since `has_education_maintenance_funding` stays false.

`is_in_full_time_education` leaves `UK_ENGINE_DEFAULT_INPUTS`, and the `has_education_maintenance_funding` entry is restated: it now matters for full-time students on courses below advanced level, and the FRS still has no clean source for it (for María's signature). The take-up population's comment and its lockstep test name a fourth engine case the population does not mirror: a unit whose working-age claimants all receive education with no exception.

The bound Universal Credit rows can move: the household counts by family type, the payment distributions, the element rows, `obr.universal_credit` and the local household counts, with `uc_take_up`'s award-band fit and `uc_reporter_redraw`'s screen. `person.is_in_full_time_education` joins the export-surface allow-list as a candidate-only column, since the enhanced FRS does not carry it.
