The UK LCFS stage now sets household electricity at the level ONS household spending implies, not at DESNZ Energy Trends volume. The dense build's local electricity rows move with it, and the national binds ONS 04.5.1 again (#1418).

The stage still rakes kWh to the NEED shape. For a fuel it declares in `spend_level_fuels`, though, its kWh level is the kWh that ONS Consumer Trends household spending buys:
- the class for calendar 2024 (04.5.1 for electricity);
- less QEP's United Kingdom average fixed cost on every UK domestic meter DESNZ counts (Great Britain's sub-national count with Northern Ireland's row, now vendored);
- divided by QEP's United Kingdom average unit cost.

For electricity that is 84.9 TWh against Energy Trends' 91.7 TWh (−7.4%), from ONS's £28.13bn on the 30 September 2026 release. On the last national's design weights that puts the frame's electricity spend about 1% below ONS, where Energy Trends put it about 5% above. The 1% is the standing charges of the meters that aren't frame households' main homes; the next build measures it. Northern Ireland's meters stay in the standing-charge subtraction, and the level block says why: its tariffs carry the charge in the unit rate.

Gas keeps the Energy Trends level, and its ONS 04.5.2 row stays out of the calibration objective (María's ruling of 2026-10-29). The class counts bottled gas (COICOP 04.5.2.2), which the frame holds in `liquid_fuel_consumption`, so binding it on mains gas would count bottled gas twice: ONS's CPI weights put it at 2.2% to 3.2% of the class. #1095 tracks netting it out once Chronicle carries the split. Decision D1's electricity exclusion is deleted; its gas exclusion is replaced by one with that reason.

The local `desnz.subnational.electricity_kwh` rows bind DESNZ's split between authorities and constituencies at the stage's own level, through one factor per fuel. So they move with it: Great Britain's cells now sum to 82.7 TWh, not 89.2 TWh, and the dense build no longer binds ONS nationally against local rows about 8% higher. The gas rows keep the Energy Trends level.

The `uk_stage_lcfs_consumption_energy_rake` gate recomputes the new level from the vendored rows. Because that recomputation shares the stage's inputs, the gate also holds the ONS-spend level to between 0.85 and 1.0 of the Energy Trends volume (0.926 on this release), so a wrong ONS class or vintage, or a unit error, that moves it by more than about 8% fails. Every QEP payment method's level falls inside those bounds, so the gate checks the stage's declared payment method (`all`) directly. The H5's `energy_price_basis` stays calendar 2024.
