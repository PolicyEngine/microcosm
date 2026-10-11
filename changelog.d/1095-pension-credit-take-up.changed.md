Pension Credit take-up now follows uk-data#510 (microcosm#1095):

- Each component band's residual rate is solved over Great Britain, which DWP's take-up rates cover, and drawn for every entitled non-reporter, Northern Ireland's included. Before, Northern Ireland's units entered the solve, so the rate held UK-wide rather than in Great Britain.
- A benefit unit with no entitlement that does not report Pension Credit claims at DWP's Savings Credit-only caseload take-up, 37% in FYE 2024, on the same draw (new contract key `pension_credit_newly_entitled`). Before, such a unit never claimed, so a reform or a later year that entitled it paid it nothing.
- Entitlement is read at the release's calibration year, as uk-data reads it. The mixed-age saving is still evaluated at the survey year, and the entitlement read takes it as stored.

SPI-channel reporters stay anchored, since `spi_benefit_coherence` makes their reports coherent; uk-data draws them instead. The stage gate measures each band in Great Britain and fails a receipt measured over any other scope. By channel, the receipt also records the stage's Pension Credit assessable capital and deemed income, and its Guarantee and Savings Credit entitlement.
