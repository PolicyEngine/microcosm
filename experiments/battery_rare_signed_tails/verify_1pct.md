# tails 1% verify build: failure-line diff vs baseline

Cold 1% build (sample seed 578) on this branch at the serial host queue,
2026-08-25: physical battery failure lines **127 = 127** vs baseline1pct —
zero greened, zero introduced.

This is the expected 1% result, per the lane's own adjudication: at the 1%
rung the retirement training donor (~1,080 rows) sits below the 5,000-row
cap, so the carrier-preservation path is structurally inert at this scale;
the Keogh ACS side remains at zero carriers pre-transfer either way. The
fix's behavioral effect binds at the 25% rung (5 sparse-donor checks) and
via the held-out calibration evidence (40 shape checks). What this build
demonstrates is strictly **no regression** from: the carrier-union
preservation under unchanged cap and seed, and the realized-regime
persistence through transfer banks, receipts, H5 checkpoints, and resume
validation.
