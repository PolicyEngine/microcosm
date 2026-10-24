`RegimeGatedQRF` takes a `gate_calibration`, named by the new `GateCalibration` constants. The default, `in_sample`, is the sign gate as before, and its draws do not move. The gate is one boosted classifier, calibrated only on its own training rows. It memorises a rare class's training rows, so on every other row, every recipient included, it draws that class too rarely.

`cross_fitted` fits a gate on each of five folds of the training rows, stratified by class. One weighted calibration (Platt scaling, for two classes) is fitted on the gates' out-of-fold scores and maps each of them, and a row's probability is the mean of the mapped gates. A rare class is then drawn at its rate on rows the gates were not fitted to.

The forests and the draw uniforms are unchanged. A class with fewer rows than folds keeps the in-sample gate. A chain checkpoint names the calibration only when it is cross-fitted, so in-sample checkpoints are unchanged.
