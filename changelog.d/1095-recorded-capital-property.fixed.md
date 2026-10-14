Universal Credit and Pension Credit recorded capital now include property (uk-data#495, microcosm#1095). policyengine-uk replaces every capital source with a recorded figure of 0 or more and takes it as countable. The FRS benefit-unit capital (TOTCAPB4) counts accounts and assets but no property, so the engine assessed no unit on its household's second homes, buy-to-let property, other buildings or land.

`uc_capital_coherence` now records each capital as the financial carrier plus the household's countable property, for the unit that holds it:

- **Who holds it.** The household reference person's benefit unit holds all of the household's property, and no other unit records any (María's ruling of 12 October). WAS records property for the household, and each means test counts the claimant's and partner's own capital. The reference person is the one in whose name the home is owned or rented, the best single reading of the owner. policyengine-uk's own proxies share household capital over every claimant or partner, which puts a parent's property on an adult child's claim; they apply only to a unit on the -1 sentinel.
- **At what value.** The property counts at 90% of its value: market value less 10% for the costs of sale (UC Regs 2013 reg 49; SPC Regs 2002 reg 19). Debt secured on the property is not on the frame and is not netted (pe-uk#1969).
- **Which property.** Universal Credit counts other residential and non-residential property. Pension Credit counts those and owned land.

The carrier's -1 sentinel passes through to both.

A unit that reports Universal Credit keeps its receipt, and records the carrier alone for Universal Credit. DWP assessed its capital to pay it, and the property is imputed without that receipt, since the WAS file has no Universal Credit column.

`uc_reporter_redraw` screens SPI units on the carrier plus the property they hold, so a drawn reporter does not land on a unit whose property is over the capital limit. It screens an FRS unit that reports Universal Credit on its carrier alone, so that unit stays in the training set. The `uk_uc_capital_coherence` gate re-derives both capitals from the carrier, the declared rule and the reporter rule, within £0.01 plus one float32 step, and now checks Pension Credit capital too.

policyengine-uk 2.123.0 does not uprate `uc_reported_capital`, so its property stays at survey-year values in later model years. `pension_credit_reported_capital` and the household property columns uprate by GDP per head, so after the survey year the same property counts at a lower value for Universal Credit than for Pension Credit. policyengine-uk 2.124.0 adds `uc_reported_capital` to the dataset projection (pe-uk#2201); the next engine bump closes the gap.
