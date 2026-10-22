# microcosm#1095 U3: binding thin target rows, receipts for #1333

This receipt gives no row's model figure, and no count or weighted figure that rests on fewer than ten source households. Figures from the SPI channel stay out of it. Publisher figures appear as published.

## The question

María's rule of 2026-10-21: a target row is not left out because few source households carry it. Only a measured distortion of the calibration keeps it out. Thirteen per-row entries in `uk/calibration_measure_exclusions.json` rested on thin support, and #1333 retires them:

- the eleven benefit-cap amount bands, held out after #882: on its arms the bands sat far outside their bound, and the solve moved weight onto cloned households;
- the HMRC £500k to £1m dividend and property-income-count cells, held out after #807: fitting them distorted neighbouring cells.

The support floor now drops only the rows that no source household carries, which no weights can fit. For each retired entry the question is whether binding it distorts the calibration now.

## Method

- One spine, built on #1333's head before its second review (policyengine-uk 2.145.0's release commit, whose package files the PyPI wheel matches), was calibrated twice with `UK_REVIEW_DATE=2026-10-21`:
  - as pushed (the treatment);
  - with the thin-row commit reverted (the control), so the two runs differ only in the binding.
- Both runs pass all seven terminal gates.
- `docs/evidence/uk-thin-rows-1095/scripts/ab_binding.py` recomputes fit and weight anatomy from each run's own target-support sidecars (`target_support_matrix.npz`, `target_support_vectors.npz`). So it measures a withheld row too. Its output stays with the licensed spine, outside the repository.
- A newly bound row is read through the treatment's compiled row, at each run's weights.

## Results

- **Rows.** The control binds 1,216 rows and the treatment 1,244. The 28 new rows are 16 UC payment-distribution bands, 10 benefit-cap amount bands and the two HMRC cells. Of these, 22 rest on fewer than ten source households, so their figures are withheld. The floor drops the bands that no household carries.
- **Shared rows.** Over the 1,216 rows bound in both runs, 99.0% are within 10% of their target in the control and 98.9% in the treatment. The mean absolute relative error is 0.6% in both. One shared row moves just past 10%. It is SPI-derived, so this receipt does not name it. No shared row moves the other way.
- **Weights.** The effective sample size is about 6,900 in both runs. The share of design weight stretched 9.5 times or more is 0.8% in both. The weight-ratio gate passes in both.
- **New rows.** Each newly bound row whose figures are published reaches its target. In the two DWP families, binding does not raise the share of the carriers' design weight that is stretched 9.5 times or more. The HMRC cells' carriers are SPI-derived, so their figures stay out of this receipt.

No retired entry goes back.

## Rows given back by difference (Vahid's second review, item 10)

A withheld row's figure can come back as a family total less its published bands, or through any other linear difference of published figures. The build now withholds a published row beside the thin ones while such a difference isolates fewer than ten source households (`uk_withheld_row_sets` in `uk_runtime/diagnostics.py`).

On the treatment run's compiled rows and weights:

- before the fix, the published rows gave back seven of the 38 withheld rows exactly, each on its own;
- with the fix, the build withholds eight more published rows. Each is a row that one of those differences needs, chosen with the fewest carriers. Most are the next band beside a thin one;
- an independent check then finds no combination of withheld rows that the published rows give back and that rests on fewer than ten source households. The combinations that are still given back, such as the withheld bands of a family together, each rest on ten or more.

This receipt names none of these rows. Naming a row withheld beside another would show that its neighbour is thin.

## To repeat on a later spine

Calibrate the spine twice, with and without the thin rows bound, then run the script on the two runs and the spine. Compare the shared rows' share within 10%, the effective sample size and the stretched share with the figures above.
