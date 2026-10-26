# microcosm#1095 and #1278: the next policyengine-uk bump, Phase A receipts

This receipt gives no count of source households and no row's model figure that rests on fewer than ten of them. Nor does it give a model figure of a row of the SPI channel, or of a row that rests in large part on the SPI channel's households: those rest on the SPI Public Use Tape (SN 9422). Publisher figures appear as published. The fit and weight statistics are over the whole national surface, as in `experiments/1095-uk-thin-row-binding-receipts.md`.

## The question

Phase A of the plan for the next policyengine-uk bump changes no stage, so it is measured on #1333's policyengine-uk 2.145.0 spine without a new spine. It settles the national rows held out of the objective until 2026-11-26 and the entries that lapse soon after:

- the zero-support floor's scope extends to the national families, in scopes of one unit each (María's ruling of 2026-10-22);
- the rows held for thin support bind where any household carries them, unless an arm measures a distortion (María's rule of 2026-10-21);
- the OBR split of Universal Credit at the welfare cap and the England postgraduate repayments leave the targets;
- `obr.vat` binds again, since policyengine-uk 2.135.7 replaced the engine factor it waited on;
- the fifteen DWP benefit-cap rows bind the UC reduction the award loses (#1349).

## Method

- **R0, the control:** #1333's 2.145.0 arm, reused, with no new run.
- **Arm 1:** a calibrate-only run on R0's spine at the Phase A head, review date 2026-10-23, with every candidate row bound.
- **Arm 1b:** the contingency run the plan allows when an arm fails a gate: arm 1's head with the HMRC £1m-and-over cells held again and the remaining entries renewed, at the same review date. It leaves out the three gates the plan adds for #1308's stages, which R0's spine never ran.
- Before arm 1, each held row was compiled on R0's spine through the build's own engine route and estimated at R0's design and final weights. The script reproduces R0's recorded estimates exactly. Its output stays with the licensed spine, outside the repository.
- `docs/evidence/uk-thin-rows-1095/scripts/ab_binding.py` compares each arm with R0 over the rows both bind.

## Results

**Before arm 1.** Every HMRC £1m-and-over cell and every band-H region cell has carriers on R0's spine, so none falls to the zero-support floor; their per-row reasons, which said the cells had none, were stale. The England postgraduate repayments row has no carrier: the stage assigns no postgraduate plan. Plan 5's liable stock fits at design weights and drifts only while it is held out.

**Arm 1.** It binds 1,259 rows against R0's 1,244: the fifteen newly bound rows. The floor drops the same five rows as R0, none of them in the new scopes. Six of the seven terminal gates pass, and `uk_target_fit` fails on one HMRC £1m-and-over cell, which stays outside the 25 percent bound when bound. Over the 1,244 rows both runs bind:

- within 10 percent of the target: 98.95 percent in R0, 98.63 percent in arm 1;
- mean absolute relative error: 0.57 percent, 0.64 percent;
- effective sample size: about 6,900, about 7,000;
- share of design weight stretched 9.5 times or more: 0.78 percent, 0.76 percent.

Four shared rows leave the 10 percent band under the binding and none enters it. Three of them, rows of the SPI channel that had fitted closely, move with the HMRC £1m-and-over cells; this receipt does not name them. That is a measured distortion, so the ten HMRC cells are held again as a family, on that measurement (#1095).

The band-H region rows, Plan 5's liable stock and `obr.vat` reach their targets, and binding them does not raise the share of their carriers' design weight stretched 9.5 times or more. The band-H region rows bind with their figures withheld where fewer than ten source households carry them.

**Arm 1b.** With the ten HMRC £1m-and-over cells held again, all seven terminal gates pass. It binds 1,249 rows: R0's 1,244 and the five new ones. Over the 1,244 rows both runs bind:

- within 10 percent of the target: 98.95 percent in R0, 98.79 percent in arm 1b;
- mean absolute relative error: 0.57 percent, 0.59 percent;
- effective sample size: about 6,900 in both;
- share of design weight stretched 9.5 times or more: 0.78 percent, 0.80 percent.

Two shared rows cross the 10 percent line from inside it, and none crosses the other way: the UC households with the LCWRA element and SLC's Parents' Learning Allowance recipients. Neither is a row of the SPI channel. The band-H region rows, Plan 5's liable stock and `obr.vat` reach their targets, and their carriers' stretched share does not move. The benefit-cap rows whose figures are published reach their targets.

More rows are withheld than on R0: 52 against 38. R0 ran before #1333 added the complementary withholding, which now also withholds a published row that would give a thin one back by difference, and the three band-H region rows bind with their figures withheld.

No retired entry goes back, and the ten HMRC £1m-and-over cells stay held.

## The held rows, renewed

At arm 1's calibrated weights, measured unbound, the rows still held read as on R0's arm:

- the twelve UC payment-distribution bands: every one stays outside 10 percent, eleven short and one over;
- Plan 2: both borrower stocks stay far short of SLC's AY2024/25 figures, 8.94M liable and 3.985M above the threshold;
- Plan 5's repayments: too few source households carry them for any weights to reach SLC's figure;
- the any-tenure UC housing-element count: short by the other or unknown tenure the model does not carry, the structural gap itself.

These rows rest in large part on the households of the SPI support channel, or, for Plan 5's repayments, on fewer than ten source households, so this receipt gives no figure of them. Their entries are renewed on that measurement, signed on 2026-10-23 to 2027-01-23.

## To repeat on a later spine

Compile the held rows on the spine and read them at the run's weights, then calibrate once with the candidate rows bound and compare with the control over the rows both bind: the share within 10 percent, the mean error, the effective sample size and the stretched share, and which shared rows cross the 10 percent line.
