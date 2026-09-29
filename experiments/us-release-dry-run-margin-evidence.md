# US release dry run: evidence for the default margins

The release dry run (`tools/build_us_fiscal_refresh_release.py
--dry-run-gates-report`, `tools/dry_run_us_release_gates.py`) grades the staged
frame at its **base** weights. The release grades its registers at
**calibrated** weights. A base-weight verdict is certain only when the solve
cannot move it; everything else is AT-RISK within stated margins. This note
records the measurements behind the defaults in
`microcosm.build.us_runtime.release_gate_dry_run`, and the dry run's own check
on real data.

The receipt with every row is `us-release-dry-run-margin-evidence.json`.
`us-release-dry-run-margin-evidence.py` produces it from the artifacts below;
the receipt names each input by sha256.

## Source run

Route A run `310842b986d7` (release `20260926T165326Z`, build commit
`310842b986d7b8bfd560e4f46a758a2a23f526a3`, dense full-pool calibration). It ran
13,707 s and failed only on its QRF tail-concentration register. The inputs:

- **Base.** `base_populace_us_2024_puf_support.h5`, sha256 `0581707e…9b55`.
- **Calibrated weights.** The final household weights the release wrote
  beside its refusal (`final_household_weights.npy` and
  `final_household_weight_ids.npy`).
- **The release's own verdicts.** `qrf_tail_concentration.json` and
  `input_mass_parity.json`.
- **The release's own timing.** `calibration_diagnostics.json`
  `build.timing`.
- **Base-weight mass drifts.** The run's own preflight report
  (`preflight-base.json`).

From `build.timing`:

| Phase | Seconds |
| --- | --- |
| Base load, input stages and pre-solve gates (what the dry run replays) | at most 2,049 |
| Target compilation (materialization) | 9,951 |
| Calibration | 1,668 |

The first row is the build's elapsed time through calibration (13,668 s) minus
the other two. By that point the release's process tree had used about 11,000
CPU-seconds, according to the supervisor's 30-second series.

## QRF tail shares

The method is to run `microcosm.build.gates.tail_concentration_gate` (top 100,
threshold 0.75, at least 500 carriers) twice on every column the release's
tail gate checked that the raw base carries:

1. at the base household weights;
2. at the release's final calibrated household weights.

Person and tax-unit weights are broadcast from household weights, as
`Frame.resolve_weights` does. The release checked 33 columns with a share (plus
thin ones); the raw base carries 32 of them. `bond_assets` exists only after
the release's own `scf_wealth` stage.

- **Reproduction.** The calibrated side reproduces the release's recorded
  shares exactly for 31 of 32 columns.
  `self_employed_pension_contributions_desired` differs by 0.0013 and by 53
  carriers, because a release-time stage rewrites it after the base. That
  rewrite is why the dry run replays the release's stages rather than reading
  the base.
- **Carrier counts.** The recomputed counts are identical at base and
  calibrated weights for every column. The full-pool solve parametrizes each
  weight as `exp(log w)`, so no positive weight reaches zero.
- **Shifts.** Calibrated minus base shares run from **-0.033 to +0.299**,
  median +0.105. Calibration concentrates these columns: only
  `partnership_income` fell. The largest rises:

| Column | Base | Calibrated | Shift |
| --- | --- | --- | --- |
| `alimony_income` | 0.474 | 0.774 | +0.299 |
| `alimony_expense` | 0.445 | 0.722 | +0.277 |
| `non_sch_d_capital_gains` | 0.226 | 0.501 | +0.275 |
| `farm_rent_income` | 0.501 | 0.760 | +0.260 |
| `farm_operations_income` | 0.344 | 0.573 | +0.228 |

The route A d177 register (run `8f63bf000254`) recorded six initial-weight
versus calibrated pairs in its reasons, parsed into the receipt. Their shifts
run from +0.06 (`long_term_capital_gains_on_collectibles`, 0.82 to 0.88) to
+0.29 (`alimony_expense`, 0.49 to 0.78).

**Defaults.** The rise margin is **0.35** and the fall margin **0.05**. Together
they cover all 38 measured shifts. On the full-pool path a column is certainly
over the threshold only above 0.80 at base weights, and certainly at or under
it only at or below 0.40. `test_default_margins_cover_every_measured_route_a_shift`
pins that coverage.

## Export input mass

The method pairs two drifts for each of the 20 columns with the largest export
drift:

- the drift at base weights, from the run's own preflight report (raw base,
  same `--export-input-mass-reference-h5`);
- the drift at calibrated weights, from the release's `input_mass_parity.json`.

The drift moved by **-0.30 to +0.84**. For example, `partnership_income` went
from -60% (outside the ±50% band) to +24% (inside it), and
`charitable_cash_donations` from -43% to +8%. The moves exceed the band's own
half-width, so no nonzero-mass verdict is certain at base weights. The dry run
certifies only structural refusals there: a checked column that is absent, or
one with no nonzero record. It flags out-of-band columns, and in-band columns
within the mass margin (**0.10**) of an edge, as AT-RISK. The margin sets which
in-band columns get called out; it is not a bound on the solve.

## The dry run on that run's config

The dry run was replayed on the run's own release config (`release-config.json`
`argv`, the d177 register, build commit `21c1f9ba3`):

- **Verdict.** It exits 1 on `farm_income`, a thin column (469 carriers) whose
  unused entry is certain. Every other column the release refused is AT-RISK,
  none PASS, including `bond_assets` (0.520 at base weights; the release
  recorded 0.766).
- **Differential.** At the stop point, the staged frame with the release's
  saved final weights attached was run through the release's own tail gate
  and register mismatch. It reproduces the recorded `qrf_tail_concentration.json`
  exactly: the same checked and dense columns, every share (largest difference
  0.0), carrier count, thin count, failure line and register-mismatch entry.
  The stop point therefore sees the frame the release calibrated and exported.
- **Identity.** The staged-frame digest matches the identity of the release's
  own target-frame checkpoint.
- **Cost.** Grading took 236 s. Reaching the stop point took 17,691 s on a
  saturated host: the process averaged 0.79 CPU-seconds per wall second and
  was switched out involuntarily 115 million times. Its CPU time, 14,046 s, is
  comparable to the roughly 11,000 CPU-seconds the original run had used by
  the same point, which that run reached in at most 2,049 s.

## L0 (sparse) path

A sparse release keeps the households its L0 selection picks and refits their
weights, so on that path nonzero shares, carrier counts and top-k shares all
move. Carriers cannot rise, so a thin column stays thin. None of it is
measured here: no run pairs a sparse export with its base-weight surface.

- **L0 tail margins.** They default to **1.0**, which admits any share, so no
  L0 share verdict is certain.
- **Support margins.** The defaults are **0.02** on the nonzero share and
  **0.25** carrier retention: the smallest kept fraction of the records
  carrying one signal. They are conservative defaults, **not measurements**.
