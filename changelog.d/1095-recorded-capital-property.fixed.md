Universal Credit and Pension Credit recorded capital now include property (uk-data#495, microcosm#1095). policyengine-uk replaces every capital source with a recorded figure of 0 or more and takes it as countable. The FRS benefit-unit capital (TOTCAPB4) counts accounts and assets but no property, so the engine assessed no unit on its household's second homes, buy-to-let property, other buildings or land.

`uc_capital_coherence` now records each capital as the financial carrier plus the unit's share of its household's countable property. The shares follow the engine's own proxies:

- Universal Credit counts other residential and non-residential property, shared by the unit's claimants and partners (`is_uc_claimant`);
- Pension Credit counts those and owned land, shared by the unit's members at or over Pension Credit qualifying age.

A dependant owns none, and the carrier's -1 sentinel passes through to both.

A unit that reports Universal Credit keeps its receipt, and records the carrier alone for Universal Credit. DWP assessed its capital to pay it. The property is imputed without that receipt, since the WAS file has no Universal Credit column, and it is shared over every claimant or partner in the household, so a parent's property would land partly on an adult child's claim. The other units of its household keep their shares.

`uc_reporter_redraw` screens SPI reporters on the carrier plus the property share, so a drawn reporter does not land on a unit whose property is over the capital limit. The `uk_uc_capital_coherence` gate re-derives both capitals from the carrier, the declared shares and the reporter rule, within £0.01 plus one float32 step, and now checks Pension Credit capital too.

policyengine-uk 2.122.2 does not uprate `uc_reported_capital`, so its property share stays at survey-year values in later model years. `pension_credit_reported_capital` uprates by GDP per head.
