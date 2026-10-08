The UK national calibration binds Great Britain pension-age Housing Benefit spending (`dwp.hb.amount_pension_age`, microcosm#1095). The row reads the DWP benefit expenditure and caseload tables, Spring Forecast 2026 (PolicyEngine/chronicle#280 / #282), on the line "Housing Benefit over Pension Credit qualifying age". It binds the calendar-2025 window: three twelfths of the FY2024-25 outturn (£6,851.0m) plus nine twelfths of the FY2025-26 forecast (£7,114.7m), which gives £7,048.8m. The model column sums `housing_benefit` over benefit units whose eldest adult is 66 or over, in Great Britain households, the same age rule as the pension-age caseload rows. The row has its own `dwp_benefit_expenditure` family, which allows source projections.

DWP gives each fiscal year its own measure (`expenditure_2024`, `expenditure_2025`). The series identity of `calendar_year_window` includes the measure, so it read the two years as two one-year series and refused the window, which is why #1100 dropped this row. A window selector can now declare `source_measure_id_by_opening_year`, a map from each overlapping year's opening year to its measure:

- A fact matches only the measure that its own year declares.
- The declared measures form the window's single series.
- The map must name exactly the window's two years.
- It cannot be combined with `source_measure_id` or with a list-valued series key, and it is refused under any other value operation.
- The weights, the two-year requirement and the refusal of a partial window are unchanged.

The national target references, the membership report and the compile-parity signed differences are regenerated. The new row is the only change.
