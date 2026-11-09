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

The ACS local mapping accepts two kinds of row and refuses any other:

- **Ledger rows** of any family (on these surfaces SNAP, Medicaid enrollment
  and SOI) take the national basis rule. The row must carry `measure_mode` and
  `source_measure_id`, and the rule must agree with its `ledger_measure_unit`
  (`count` or `usd`).
  - The one reviewed exception is `tax_filer_individual_count`, a count of
    people. The national rule files it as an amount, and
    `ACS_LOCAL_LEDGER_BASIS_OVERRIDES` maps it to a count. Only district rows
    carry it; on the pinned feed only the `full` surface has them (445).
  - Any other disagreement is refused.
- **Ladder population rows** (`pop_state_SS`, `pop_cd_SSDD`) carry only
  `geography_level`, so the national rule alone would file them as amounts.
  - The mapping files them as counts, which is how the national release's
    Census population targets (`measure_mode=indicator_sum`) are filed.
  - Each state's `pop_cd_*` rows share one concept budget.

District SOI rows take the national district key without three keys that name
the row's district rather than its concept:

- `ledger_fact_label` and `ledger_layout_groupby_value_label`, which the
  ledger compiler stamps on every row, for example "WV congressional
  district 1";
- `state_cd_cd_file_value`, the district file's own value, which the
  `state_cd` rebase stamps.

A guard (`validate_acs_local_concept_groups`) refuses a surface whose
district rows of one concept in one state land in more than one group, in any
SOI mode. It also refuses a `state_cd` reconciliation block that is not
exactly one group.

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
  California's 52 districts carry 1.51 of weight between them; an at-large
  district carries 1.30-1.70.
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
- As is: 19,541 groups, every district row on its own. That is the 19,105
  trained `state_cd` SOI rows, split by the compiler's labels and the rebase
  stamp, plus the 436 ladder rows.

Weights after the change run from 0.0025 to 83.4.

### Every SOI mode classifies

`results.json` `soi_modes` compiles each `--soi-mode` from the pinned feed,
with the same population rows and before any holdout:

- the ACS local mapping classifies every row in `state`, `totals`, `full` and
  `state_cd`;
- `full` is the only surface with `tax_filer_individual_count` rows (445),
  which take the override;
- a per-district metadata key injected into every district row is refused on
  every surface that has district ledger rows (`totals`, `full`, `state_cd`).

### What the formula does and does not do

- **It balances counts against amounts, and large values against small ones,
  but not families.** On the default surface SOI's share rises from 85.6% to
  90.2%. On `state_cd` it falls from 97.3% to 92.1%, because district rows
  stop multiplying their concept's weight. Route A, the certified national
  release, shows no family balancing either: SOI is 74.0% of its targets and
  74.1% of its loss weight. The family lever is
  `--target-family-loss-multiplier`, which this change adds to the ACS local
  tool. No default multiplier is set.
- **On `state_cd` it moves the weight from district rows back to state rows.**
  District rows go from 80.1% to 16.7% of the loss. All the districts of one
  concept in one state now carry what their largest district would carry
  alone, so district geography no longer multiplies a concept's weight.

### Cost: district population fit without a multiplier

The weighted λ frontier (microcosm#1105, `experiments/us-acs-local-l2-basis-20260928/README.md`,
"On the weighted loss") solved the 09-23 checkpoint at full scale on these
weights. Its runs found this cost:

- **Equal weights:** every trained district population is within 0.7% at
  λ 0.
- **These weights, no multiplier:** 11 of 436 districts miss by more than 10%
  at λ 0 (worst TX-14 −33%, CA-29 −32%). The chi-square penalty multiplies
  the misses: 73 districts at λ 0.03, 135 at λ 0.1.
- **With `--target-family-loss-multiplier census_population=8`:** district
  population returns to 9.5% of the loss and state population rises to
  24.7%, which I recomputed from these weights. No district misses by more
  than 10% at λ 0, and 3 do at λ 0.03 (worst 15%).

Which multiplier and λ the next build uses is decision d797. This change sets
no default multiplier.

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

- **`distribution.py`** reads two inputs, and records the commit it ran at
  and whether the tree was clean:
  - the 09-23 release's rebuilt `target_registry.json` (sha256
    `fd25c600…`), built by `experiments/us-acs-local-l2-basis-20260928/registry.py`
    on branch `acs-local-weighted-lambda-frontier` (microcosm#1105, commit
    430491e2c). Its receipt is copied here as `registry_20260923.receipt.json`:
    all 4,459 names and values equal the release checkpoint's;
  - the pinned feed.
- **`route_a_fixture.py`** reads Route A's `calibration_diagnostics.json`
  (sha256 `64b55a02…`) and writes
  `packages/microcosm-build/tests/fixtures/us_route_a_target_loss_weights.json`.
  It refuses to write unless the shared module reproduces every recorded
  weight bit for bit, and the recorded loss basis hash (`206ada09…`) matches.

Each script refuses an input whose sha256 differs from the one it expects.
