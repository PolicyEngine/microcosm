The UK spine stores `is_severely_disabled_for_benefits` as the tax credit severe disability condition (uk-data#494, policyengine-uk#1946, microcosm#1095). That condition is in CTC Regs 2002 reg 8 and WTC Regs 2002 reg 17: the DLA care component at the highest rate, the PIP daily living component at the enhanced rate, or higher-rate Attendance Allowance. The flag is read from the categories the stage derives, so it agrees with them by construction.

On the locked policyengine-uk the engine's formula is the same definition, and an engine-lane test checks them against each other category by category. The old rule did not match that formula in two ways:

- it counted lower-rate Attendance Allowance;
- it counted any Armed Forces Compensation Scheme payment. FRS code 8 cannot be told apart from Armed Forces Independence Payment, so it no longer counts, a known under-count the uk-data change shares.

The incumbent's coverage manifest marks the flag as required, so the spine keeps storing it rather than leaving the formula to compute it. Dataset runs use the stored value.
