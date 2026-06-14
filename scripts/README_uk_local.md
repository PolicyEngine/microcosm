# populace UK local-area spine (scaffold)

A faithful UK port of the US "replication-spine" local-area driver
(`populace-local/scripts/build_local_candidate.py`) onto the populace stack.
A small stratified household subsample of the populace UK pool is cloned once
per Westminster constituency (650 2024 boundaries) and reweighted with
populace.calibrate against the constituency target surface.

This is the **spine** approach (small donor pool stacked across areas), *not*
pe-uk-data's existing **wide** dense `areas × households` clone-and-assign flow.

Driver: `scripts/build_local_candidate_uk.py` — 3 cached, resumable stages
(`subsample` → `matrix` → `solve`), the same CLI/stage/caching conventions as
the US script.

## Ingredient map (US → UK)

| Concept | US (`build_local_candidate.py`) | UK (`build_local_candidate_uk.py`) |
|---|---|---|
| Base pool | `populace_us_2024.h5` (HDFStore) | `populace_uk_2023.h5` — a `UKSingleYearDataset` table-format H5 (`~/.claude-worktrees/populace-uk-build/artifacts/`) |
| Stratification var | `adjusted_gross_income` | `total_income` (household-mapped; `--income-variable` to override) |
| Strata rule | keep all top-1% AGI, uniform-sample below | identical (top-1% `total_income`) |
| Design init weight | `household_weight × inv_rate` | identical (`household_weight × inv_rate`, top stratum rate 1.0) |
| Sim adapter | `policyengine_us.Microsimulation(dataset=path)` | `policyengine_uk.Microsimulation(dataset=UKSingleYearDataset)` |
| Area set | 436 congressional districts (sorted GEOIDs) | 650 Westminster constituencies, sorted codes = `constituencies_2024.csv` order |
| Column layout | `cd_index*n_hh + hh_index` | `constituency_index*n_hh + hh_index` (identical) |
| **Per-area resim unit** | **one sim per STATE** (CDs share devolved policy + takeup) | **one sim per COUNTRY** (England/Scotland/Wales/NI) — devolved income tax/rates/benefits differ by country; constituencies within a country share it |
| How the area's policy is set | `sim.set_input("state_fips", …)` (+ `in_nyc`, `county_fips`) | `sim.set_input("region", …)` — `country` is *derived* from `region` in pe-uk (`variables/household/demographic/country.py`), so the region enum forces the country block |
| Target machinery | `UnifiedMatrixBuilder` querying `policy_data.db` | `policyengine_uk_data.calibration.matrix_builder` — `_compute_household_metrics(sim,"constituency")` + `_load_area_targets("constituency",…)` |
| Target sources | ACS/IRS rows in the db | HMRC SPI 3.15 income, ONS subnational age bands, DWP Stat-Xplore UC (+UC-by-children) |
| National/state/district rows | explicit national broadcast + state broadcast + district rows | **constituency rows only** — the national consistency adjustment is folded *into* each constituency target inside `_load_area_targets`, so there are no separate national rows (the chief structural divergence) |
| Solver | `populace.calibrate.solve._optimize` (log-weight Adam + optional L0) | **identical call**, same params shape, design init tiled across the stack |
| Pilot flag | `--pilot-states N` | `--pilot-constituencies N` |

The 17 constituency metric/target columns (verified column-aligned):
`hmrc/{self_employment_income,employment_income}/{amount,count}`,
`age/{0_10…70_80}`, `uc_households`, `uc_hh_{0_children,1_child,2_children,3plus_children}`.

## What runs (verified this turn)

- **`subsample` — FULLY IMPLEMENTED + RUNNABLE.** Smoke-tested against the real
  pool. Produces `base_pool.h5` (a reloadable `UKSingleYearDataset`),
  `base_init_weights.npy`, `base_pool_meta.json`.
- **`matrix` — structurally complete + runs end-to-end** (verified on
  `--pilot-constituencies 3`). Country resim → `region` set → real
  `_compute_household_metrics` / `_load_area_targets` binding → block-local CSR.
  Wrote `X_stacked.npz` (51×36000, block-locality asserted), `targets.parquet`,
  `target_names.json`, `stack_meta.json`.
- **`solve` — complete** (verified on the pilot): the identical `_optimize`
  call consumes the stacked matrix, writes `weights.npy`,
  `solve_diagnostics.csv`, `solve_summary.json`.
- **Per-country resim is meaningful, not a no-op**: `region=SCOTLAND` vs
  `region=SOUTH_EAST` on the same pool moves total `income_tax` £789M → £831M
  (Scottish rates/bands diverge).

## What's TODO / stubbed

- **`TODO(uk-local)` (`build_local_candidate_uk.py`, country loop):** finer
  *within-country* geography is not set. England/Scotland/Wales/NI is the full
  devolution axis for the current 17-metric surface (income tax, UC, age — none
  vary below country here). If a sub-country target is later added (e.g. BRMA
  for housing benefit, local council-tax bands), set those per-constituency
  inputs in the country loop the way the US exporter re-sets
  `state_fips`/`in_nyc`/`county_fips` per area —
  see `populace-local/scripts/build_local_candidate.py:360-367`. No binding is
  fabricated; everything the present surface needs is implemented.
- **Scorer not ported this turn.** The US verdict contract
  (`populace-local/scripts/score_local.py`: matched-household symmetric refit,
  train+holdout rotations, per-area win counts) still needs a UK analog
  (`score_local_uk.py`) scoring our constituency column blocks against the
  pe-uk-data incumbent constituency weights.
- **Full 650-constituency run not executed** (compute deliberately deferred;
  inputs are large). Pilot proves the wiring.

## Environment

Venv: `~/.claude-worktrees/populace-uk-build/.venv` (Python 3.14). Already had
`policyengine_uk` + `torch` + `h5py` + `scipy` + `pandas`. This turn added:

```bash
VIRTUAL_ENV=~/.claude-worktrees/populace-uk-build/.venv \
  uv pip install -e packages/populace-calibrate -e packages/populace-frame pyarrow
```

**blosc2 / PyTables fix (one-time, required).** PyTables 3.10.2 on Python 3.14
hunts for a standalone `libblosc2.dylib` that the installed blosc2 wheel no
longer ships, so `pd.HDFStore` (used by `UKSingleYearDataset.save`/`file_path`)
was broken. Repaired by copying the working arm64 dylib into the PyTables dir:

```bash
cp ~/PolicyEngine/policyengine-us-upstream-main/.venv/lib/python3.14/site-packages/tables/libblosc2.dylib \
   ~/.claude-worktrees/populace-uk-build/.venv/lib/python3.14/site-packages/tables/libblosc2.dylib
```

(The `subsample` stage reads the pool's entity tables directly via `h5py`
— the tables are uncompressed compound datasets at `/<entity>/table` — so the
*read* side never needed PyTables; the dylib is required only for the
`UKSingleYearDataset.save` write and the `matrix`-stage `file_path` reload.)

## Next steps (exact commands)

From the worktree root `~/.claude-worktrees/populace-uk-local`, with
`PY=~/.claude-worktrees/populace-uk-build/.venv/bin/python`:

1. **Subsample** (done; reproduce or re-stratify):
   ```bash
   $PY scripts/build_local_candidate_uk.py --stage subsample --out out/uk-local-v0
   ```

2. **Matrix pilot** (recommended next — a single English-only block, fast):
   ```bash
   $PY scripts/build_local_candidate_uk.py --stage matrix \
       --pilot-constituencies 5 --out out/uk-local-v0
   ```
   For a cross-country pilot (exercises all 4 country sims) raise N so the kept
   constituencies span Scotland/Wales/NI (they sort after the English `E…`
   codes, so N must be large, or add an explicit multi-country pilot selector).

3. **Solve pilot:**
   ```bash
   $PY scripts/build_local_candidate_uk.py --stage solve --out out/uk-local-v0
   ```
   Note: a small-N pilot's `within_10pct` will look poor — the design init tiles
   the pool's full UK mass across only N areas, so per-column starts are ~`650/N`
   too large. At the real 650-stack the per-area init mass is correct.

4. **Port the scorer** → `score_local_uk.py` (UK analog of `score_local.py`).

5. **Full 650 run** (compute-heavy; one sim per country, 4 sims total):
   ```bash
   $PY scripts/build_local_candidate_uk.py --stage matrix --out out/uk-local-v0
   $PY scripts/build_local_candidate_uk.py --stage solve  --out out/uk-local-v0
   ```

## Data protection

The populace UK pool is enhanced-FRS-derived microdata under UK Data Service
licence. Artifacts (`base_pool.h5`, `X_stacked.npz`, weights) stay local. Only
aggregates are logged — never individual rows. `out/` is gitignored.
