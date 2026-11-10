DWP's PIP cases with entitlement (Stat-Xplore PIP_Monthly_new, PolicyEngine/chronicle#420 item 6) join the contract as 94 rows, the source of the award-mix check (P3 of the November event-data plan, microcosm#1095). England and Wales are separate rows, since no England-and-Wales fact exists. Each country has:

- `dwp.pip.cases_<country>`, every award and age;
- `dwp.pip.cases_<country>_daily_living_<award>` and `dwp.pip.cases_<country>_mobility_<award>`, for the standard, enhanced and nil award of each component;
- `dwp.pip.cases_<country>_aged_<low>_to_<high>` from 16 to 19 up to 70 to 74, and `dwp.pip.cases_<country>_aged_75_and_over`. DWP's cube carries no case above 79, so its 75 to 79 band meets the model's recipients aged 75 and over.

The nine English regions follow by each award of each component (`dwp.pip.cases_<region>_<component>_<award>`). Every row takes the mean of the four quarter-end points inside calendar 2025 (January, April, July, October) under the new `dwp_pip` monthly family. The model counts people with `pip` above zero by their reported `pip_dl_category` and `pip_m_category`.

The breakdowns are single, María's call of 8 November 2026: the award-by-age cross would rest on fewer than three survey households in many Welsh and some English cells. Each country's six award rows bind, the award split (P2, María's call of 10 November 2026). Eighty-two signed exclusions hold the other rows out of the solve; María signed them on 9 November 2026. The committed references come with the integration's regeneration.
