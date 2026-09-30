Bind the UK resident State Pension on the national surface (microcosm#1069 c3; PolicyEngine/chronicle#302 via #305). The calendar-2025 window is the mean of the four quarterly Stat-Xplore points (February, May, August and November 2025). There are 26 contract targets, compiling to 60 references in the new `dwp_state_pension` family.

- **Level:** Great Britain residents' recipients (12.05m) and amount (£138.6bn), plus Northern Ireland from DfC (0.33m and £3.7bn). The amounts are recipients times the mean weekly amount times 52, restated per point at the engine's 2025 rate.
- **Age:** recipients by age band (66–69 to 90 and over) × sex × State Pension type, binding only the cells DWP populates. Type follows the State Pension age date, so 66–69 is all new State Pension and 75 and over all pre-2016.
- **Area:** recipients by type in the nine English regions (a region-tier fan-out pinned to England), with Scotland and Wales as their own rows.
- **Weekly amount:** recipients by weekly amount band × type. These keep only the May–November points, which DWP pays at the 2025-26 rate the engine pays.

The contract gains the `dfc_ni` provider and the two State Pension categories, so the local references restate the hierarchy.
