`RegimeGatedQRF` takes a `gate_calibration`, named by the new `GateCalibration` constants. The default, `in_sample`, is the sign gate as before, and its draws do not move. That gate is one boosted classifier, calibrated only on its own training rows. It memorises a rare class's training rows, so on every other row, every recipient included, it draws that class too rarely.

`cross_fitted` fits a gate on each of five folds of the training rows, stratified by class. One weighted calibration (Platt scaling, for two classes) is fitted on the gates' out-of-fold scores and maps each of them, and a row's probability is the mean of the mapped gates. A rare class is then drawn at its rate on rows the gates were not fitted to.

- **Where the balance holds.** The rate balance is exact on the pooled out-of-fold donor scores, not on any recipient set.
- **What moves.** The forests and the draw uniforms are unchanged. In a chain, though, a sign that changes moves later targets' amounts through their predictors.
- **Grouped folds.** `groups` (on `fit`, `start_chain` and `fit_draw_next`, and the `fit` helper) deals whole groups, such as a household's members, to the same fold. A chain locks its grouping and refuses a step that changes it.
- **Fallbacks.** A target keeps the in-sample gate when a class has fewer rows (or groups) than folds, or when a fold's training rows carry no weight for a class. The calibration falls back to the class rates when its balance cannot be met or the out-of-fold ranking is worse than none.
- **Recorded outcomes.** `FittedRegimeGatedQRF.gate_outcomes()` names what each target got, with the new `GateOutcome` constants, and the `fit.qrf@1` graph kernel takes `gate_calibration` too.
- **Checkpoints.** A chain checkpoint names the calibration only when it is cross-fitted, and the grouping only when it was given, so in-sample checkpoints are unchanged.
