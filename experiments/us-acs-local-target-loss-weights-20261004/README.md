# ACS local release: target-loss weights before and after

Until this change the ACS local-area release calibrated with every training
target weighted equally. The national fiscal release weights each target's
loss with `sqrt_value_concept_budget_weighted_mape_50_50_amount_count_target_scale_cap_100pct`.
The calibrate stage now uses the same formula, from the one implementation in
`microcosm.build.us_runtime.target_loss_weights`, under an explicit ACS local
row mapping. This folder measures what that does to the share of the loss
each kind of target carries. It changes the objective of future builds only.
Nothing was rebuilt or published.

A target's **loss share** is its share of the total weight, which is the share
of the loss it carries when every target misses by the same scaled amount.

## The formula and the ACS local mapping

The formula (module docstring of `target_loss_weights.py`):

1. Each target is a count or an amount.
2. Value weight `max(|value|, 1) ** 0.5`, over its basis mean.
3. Targets that share a concept-budget key form a group, which is rescaled to
   sum to its largest member's weight.
4. Counts and amounts each carry half the total weight.
5. Mean 1, then optional family multipliers (`--target-family-loss-multiplier`),
   then mean 1 again.

The ACS local mapping classifies every row and refuses a row it cannot
classify:

- **Ledger rows** (SNAP, Medicaid enrollment, SOI) take the national basis
  rule. That rule must agree with the row's `ledger_measure_unit` (`count` or
  `usd`), and the row must carry `measure_mode` and `source_measure_id`.
  - On both surfaces below, the national rule and the unit agree on every row.
- **Ladder population rows** (`pop_state_SS`, `pop_cd_SSDD`) carry only
  `geography_level`, so the national rule alone files them as amounts.
  - The mapping files them as counts, which is how the national release's
    Census population targets (`measure_mode=indicator_sum`) are filed.
  - Each state's `pop_cd_*` rows share one concept budget.
- **District SOI rows** take the national district key without three keys
  that name the row's district rather than its concept:
  - `ledger_fact_label` and `ledger_layout_groupby_value_label`, which the
    ledger compiler stamps on every row, for example "WV congressional
    district 1";
  - `state_cd_cd_file_value`, the district file's own value, which the
    `state_cd` rebase stamps.

  Each `state_cd` reconciliation block must then be exactly one group. On the
  pinned feed that holds for all 1,907 trained blocks.

## Results

`distribution.py` writes `results.json`. "National mapping as is" applies the
national release's mapping to the ACS local specs unchanged, which shows what
reusing the helper without a mapping would have done.

### The 09-23 release's surface (`--soi-mode state`, the default): 4,459 targets, all trained

| Family | Level | Targets | Equal (before) | National mapping as is | ACS local mapping (after) |
|---|---|---:|---:|---:|---:|
| census_population | congressional_district | 436 | 9.8% | 0.1% | 1.7% |
| census_population | state | 51 | 1.1% | 0.0% | 4.4% |
| cms_medicaid | state | 51 | 1.1% | 2.3% | 2.0% |
| irs_soi | state | 3,819 | 85.6% | 95.7% | 90.2% |
| usda_snap | state | 102 | 2.3% | 1.8% | 1.7% |

Concept groups:

- After: 4,074 groups. The 436 district population rows form one group per
  state (51), of which 7 are single districts (AK, DE, DC, ND, SD, VT, WY).
- As is: 4,459 groups. Every row is its own group, because the national key
  reads `ledger_geography_level`, which ladder rows lack.

Weights after the change run from 0.026 to 20.1, with mean 1.

### `--soi-mode state_cd` on the pinned feed: 23,866 training targets

Inputs:

- the admin surface compiled by the tool's own `state_admin_surface` from the
  feed `chronicle_feed.json` pins (sha256 `b8543739…`);
- the 09-23 release's 487 ladder population specs;
- the default 10% district holdout (118 of 989 units, 2,638 targets held out).

| Family | Level | Targets | Equal (before) | National mapping as is | ACS local mapping (after) |
|---|---|---:|---:|---:|---:|
| census_population | congressional_district | 436 | 1.8% | 0.0% | 1.4% |
| census_population | state | 51 | 0.2% | 0.0% | 3.6% |
| cms_medicaid | state | 51 | 0.2% | 0.8% | 1.6% |
| irs_soi | congressional_district | 19,105 | 80.1% | 64.2% | 16.7% |
| irs_soi | state | 4,121 | 17.3% | 34.4% | 75.4% |
| usda_snap | state | 102 | 0.4% | 0.6% | 1.3% |

Concept groups:

- After: 1,958 district groups (1,907 `state_cd` blocks plus 51 state
  population groups) out of 19,541 district rows.
- As is: 19,541 groups, every district row on its own.

Weights after the change run from 0.0025 to 83.4.

### What the formula does and does not do

- **It does not shrink SOI's share.** SOI's share goes from 85.6% to 90.2% on
  the default surface. The formula balances counts against amounts and large
  values against small ones; it does not balance families. Route A, the
  certified national release, shows the same: SOI is 74.0% of its targets and
  74.2% of its loss weight. The lever for family balance is
  `--target-family-loss-multiplier` (for example `usda_snap=4`), which this
  change adds to the ACS local tool. No default multiplier is set.
- **On `state_cd` it moves the weight from district rows back to state rows.**
  District rows go from 80.1% to 16.7% of the loss. All the districts of one
  concept in one state now carry what their largest district would carry
  alone, so district geography no longer multiplies a concept's weight.

## A finding for the national release (not changed here)

`national_cd_groups.py` compiles the national registry from the pinned feed
(`compile_us_fiscal_target_registry(age_targets=True)`) and keeps its 24,340
district rows, all SOI. The national parser's default, `--target-surface
full`, calibrates those rows. Results are in `national_cd_groups.json`.

| District rows | Concept groups | Singletons | Share of loss |
|---|---:|---:|---:|
| National mapping as is | 24,340 | 24,340 | 47.0% |
| With `ledger_fact_label` and `ledger_layout_groupby_value_label` also excluded | 3,125 | 665 | 11.6% |
| Equal weights, for reference | | | 74.1% |

So under the national mapping the concept budget never engages for compiled
district rows: the compiler's per-district labels reach the key. The existing
national test of the budget uses fixture rows without those labels.

- Route A calibrated `--target-surface national_state`, which has no district
  rows, so it is unaffected.
- This branch moves the national mapping unchanged and tests it bit for bit
  against the code before the move, so the fix is left to a follow-up.

## Reproduce

From the repository root, engine-free:

```bash
uv run python experiments/us-acs-local-target-loss-weights-20261004/distribution.py
uv run python experiments/us-acs-local-target-loss-weights-20261004/national_cd_groups.py
uv run python experiments/us-acs-local-target-loss-weights-20261004/route_a_fixture.py
```

Inputs:

- `distribution.py` reads the 09-23 release's rebuilt `target_registry.json`
  (sha256 `fd25c600…`, from
  `experiments/us-acs-local-l2-basis-20260928/registry.py`) and the pinned
  feed.
- `route_a_fixture.py` reads Route A's `calibration_diagnostics.json` and
  writes `packages/microcosm-build/tests/fixtures/us_route_a_target_loss_weights.json`.
  It refuses to write unless the shared module reproduces every recorded
  weight bit for bit, and the recorded loss basis hash (`206ada09…`) matches.

Each script refuses an input whose sha256 differs from the one it expects.
