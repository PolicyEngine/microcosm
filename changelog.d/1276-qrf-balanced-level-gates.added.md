Two new ways to calibrate a cross-fitted QRF sign gate:

- **`gate_calibration: cross_fitted_balanced`.** Each fold gate is fitted on class-balanced weights: every class's weights are scaled to the same total, so even a class with a tiny share of the weight shapes how the gate ranks rows. The calibration runs on the rows' own weights as before, so the calibrated rates still equal each class's weighted rate out of fold and on fresh rows. The fold gates say which rows; the calibration says how many. Scored on its own training rows, a balanced gate draws a rare class under its rate.
- **`gate_level_columns`.** The cross-fitted calibration also gives each supported level of the named predictor columns its own offsets. The calibrated gate then holds each class's weighted rate within every offset level, not only over the whole frame.
  - An entry is a column, whose levels are its values, or a one-hot categorical's columns as a group, whose levels are its categories.
  - In each, the level holding the most weight is the reference, so the offsets are identified.
  - A level needs 30 rows of every sign class; a thinner one shares the reference's offsets.
  - Where the fit with levels can't meet the balance, the calibration falls back to the one without levels.
  - `FittedRegimeGatedQRF.gate_levels()` names each target's offset levels. `gate_level_reports()` records what each level fit came to (`GateLevelFit`: offsets, no supported level, or balance not met), with each column's reference and the levels too thin for offsets. `gate_level_settings()` gives the fixed settings behind them.
  - The checkpointed chain refuses level columns for now.
