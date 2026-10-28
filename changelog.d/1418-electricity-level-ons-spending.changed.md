The UK LCFS stage now sets household electricity at the level ONS household spending implies, not at DESNZ Energy Trends volume, and the dense build's local electricity rows move with it (#1418). The stage still rakes kWh to the NEED shape. For a fuel it declares in `spend_level_fuels`, though, its kWh level is the kWh that ONS Consumer Trends household spending buys:
- the class for calendar 2024 (04.5.1 for electricity);
- less QEP's United Kingdom average fixed cost on every UK domestic meter DESNZ counts (Great Britain's sub-national count with Northern Ireland's row, now vendored);
- divided by QEP's United Kingdom average unit cost.

For electricity that is 84.9 TWh against Energy Trends' 91.7 TWh (−7.4%), from ONS's £28.13bn on the 30 September 2026 release. On the last national's design weights that puts the frame's electricity spend about 1% below ONS, where Energy Trends put it about 5% above. The 1% is the standing charges of the meters that aren't frame households' main homes; the next build measures it. Gas keeps the Energy Trends level.

The local `desnz.subnational.electricity_kwh` rows bind DESNZ's split between authorities and constituencies at the stage's own level, through one factor per fuel. So they move with it: Great Britain's cells now sum to 82.7 TWh, not 89.2 TWh, and the dense build no longer binds ONS nationally against local rows about 8% higher. The `uk_stage_lcfs_consumption_energy_rake` gate recomputes the new level from the vendored rows. The H5's `energy_price_basis` stays calendar 2024.
