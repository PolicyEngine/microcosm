Two new ways to calibrate a cross-fitted QRF sign gate:

- **`gate_calibration: cross_fitted_balanced`.** Each fold gate is fitted on class-balanced weights: every class's weights are scaled to the same total, so even a class with a tiny share of the weight shapes how the gate ranks rows. The calibration runs on the rows' own weights as before, so the calibrated rates still equal each class's weighted rate. The fold gates say which rows; the calibration says how many.
- **`gate_level_columns`.** The cross-fitted calibration also gives each supported level of the named predictor columns its own offsets. The calibrated gate then holds each class's weighted rate within every offset level, not only over the whole frame.
  - A level needs 30 rows of every sign class.
  - In each column, the level holding the most weight is the reference. A small ridge identifies the levels of a one-hot categorical.
  - Where the fit with levels can't meet the balance, the calibration falls back to the one without levels.
  - `FittedRegimeGatedQRF.gate_levels()` names each target's offset levels. The checkpointed chain refuses level columns for now.
