# US release dry run: evidence for the default margins

The release dry run (`tools/build_us_fiscal_refresh_release.py
--dry-run-gates-report`, `tools/dry_run_us_release_gates.py`) grades the staged
frame at its **base** weights. The release grades its registers at
**calibrated** weights. A base-weight verdict is certain only when the solve
cannot move it; everything else is AT-RISK within stated margins. This note
records the measurements behind the defaults in
`microcosm.build.us_runtime.release_gate_dry_run`. The receipt with every row
is `us-release-dry-run-margin-evidence.json`, and the script that measured the
tail shares is `us-release-dry-run-margin-evidence.py`.

## Source run

Route A run `310842b986d7` (release `20260926T165326Z`, build commit
`310842b986d7b8bfd560e4f46a758a2a23f526a3`, dense full-pool calibration). It ran
13,707 s and failed only on its QRF tail-concentration register. The inputs:

- **Base.** `base_populace_us_2024_puf_support.h5`, sha256 `0581707e…9b55`.
- **Calibrated weights.** The final household weights that the release wrote
  beside its refusal (`final_household_weights.npy` and
  `final_household_weight_ids.npy`).
- **The release's own verdicts.** `qrf_tail_concentration.json` and
  `input_mass_parity.json`.
- **Base-weight mass drifts.** The run's own preflight report
  (`preflight-base.json`).

The receipt records each file's sha256.

The run's supervisor series and materialization-cache timestamps split its
wall time roughly as follows:

| Phase | Time |
| --- | --- |
| Input stages | about 1 h |
| Target materialization | about 2 h 20 min |
| Solve plus terminal gates | about 28 min |

## QRF tail shares

The method is to run `microcosm.build.gates.tail_concentration_gate` (top 100,
threshold 0.75, at least 500 carriers) twice on every column the release's tail
gate checked:

1. at the base household weights;
2. at the release's final calibrated household weights.

The columns come from the raw base, with person and tax-unit weights broadcast
from household weights, as `Frame.resolve_weights` does.

- **Reproduction.** The calibrated side reproduces the release's recorded
  shares exactly for 31 of 32 columns. `self_employed_pension_contributions_desired`
  differs by 0.0013 (and by 53 carriers), because a release-time stage rewrites
  it after the base. That rewrite is why the dry run replays the release's
  stages rather than reading the base.
- **Carrier counts.** They are identical at base and calibrated weights for
  every column. The full-pool solve parametrizes each weight as `exp(log w)`,
  so no positive weight reaches zero.
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
versus calibrated pairs in its reasons. Their shifts run from +0.06
(`long_term_capital_gains_on_collectibles`, 0.82 to 0.88) to +0.29
(`alimony_expense`, 0.49 to 0.78).

**Defaults.** The rise margin is **0.35** and the fall margin **0.05**. Together
they cover every one of the 38 measured shifts. A column is certainly over the
threshold only above 0.80 at base weights, and certainly at or under it only
at or below 0.40. `test_default_margins_cover_every_measured_route_a_shift` pins that
coverage.

`bond_assets`, one of that run's four unwaived columns, is not in the raw base.
The release's `scf_wealth` stage creates it, so only a dry run that replays the
release's stages can grade it.

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

## L0 (sparse) path

A sparse release keeps the households its L0 selection picks, so on that path
nonzero shares and carrier counts move too. Carriers cannot rise, so a thin
column stays thin. The support defaults are **0.02** on the nonzero share and
**0.25** carrier retention. They are conservative defaults, **not
measurements**. No run here pairs a sparse export with its base-weight surface.
