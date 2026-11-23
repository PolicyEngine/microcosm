Add MHCLG empty property figures for England as a standalone reader and diagnostics only; the build does not read them (microcosm#1705; `uk_firm_empty_property_published_figures`). The figures are hereditaments in receipt of empty premises relief on 31 December 2023 and 2024 by category, empty relief values for 2024-25 and 2025-26 (NNDR1 supplementary tables), and NNDR3 2024-25 outturn. `firm_property_occupied` and `firm_months_empty` stay unset for three reasons:

- the count is a stock on one date, but policyengine-uk's flag covers the whole year;
- empty hereditaments are not trading firms;
- months empty could only be calibrated to a relief total.
