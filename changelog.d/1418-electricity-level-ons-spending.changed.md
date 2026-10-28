The UK LCFS stage now sets household electricity and gas at the level ONS household spending implies, not at DESNZ Energy Trends volume. The dense build's local energy rows move with it, and the national binds ONS 04.5.1 and 04.5.2 again (#1418).

The stage still rakes kWh to the NEED shape. For a fuel it declares in `spend_level_fuels` (both are declared), its kWh level is the kWh ONS Consumer Trends household spending buys:
- the class for calendar 2024 (04.5.1 electricity, 04.5.2 gas);
- less QEP's United Kingdom average fixed cost on every UK domestic meter of the fuel DESNZ counts (Great Britain's sub-national count with Northern Ireland's row, now vendored);
- divided by QEP's United Kingdom average unit cost.

On the 30 September 2026 release that is:

| Fuel | ONS spend | ONS-implied kWh | Energy Trends kWh | Change |
|---|---:|---:|---:|---:|
| Electricity | £28.13bn | 84.9 TWh | 91.7 TWh | −7.4% |
| Gas | £18.30bn | 236.5 TWh | 252.3 TWh | −6.3% |

On the last national's design weights, that puts the frame's spend on each fuel about 1% below ONS, where Energy Trends put it about 5% above. The 1% is the standing charges of the meters that aren't frame households' main homes; the next build measures it.

The local `desnz.subnational.electricity_kwh` and `gas_kwh` rows bind DESNZ's split between authorities and constituencies at the stage's own level, through one factor per fuel. So they move with it: Great Britain's cells now sum to 82.7 TWh of electricity (was 89.2) and 234.0 TWh of gas (was 249.7). The `uk_stage_lcfs_consumption_energy_rake` gate recomputes the new level from the vendored rows. The H5's `energy_price_basis` stays calendar 2024.

With the stage at ONS, the two measure exclusions that held the ONS rows out as published-source residuals (decision D1) are deleted, and the national calibration binds both rows again.
