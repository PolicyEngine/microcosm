# #1228: re-solving Route A with the business, partnership and rental rows bound five ways

This experiment re-solves the certified Route A release's `national_state` calibration from its own checkpoint, once per way of binding the 312 Historic Table 2 rows of microcosm#1228:

- the release: `populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z`;
- the surface: 5,694 targets and 352,932 households.

The rows are business (N/A00900), partnership/S-corp (N/A26270) and rental/royalty (N/A25870). Everything else stays fixed:

- the other 5,382 targets;
- the initial weights;
- the solver settings.

So the arms differ only in what those rows ask of the weights. It is research evidence: nothing was published, and no default changed.

## Arms

| Arm | The 312 rows | Loss weights |
|---|---|---|
| `release` | As Route A bound them. Business and partnership take the positive part and count units with a positive total; business reads `self_employment_income`, without SSTB. Rental is signed, includes farm rent, and counts a positive total. | Release's, reproduced exactly |
| `concept_exact` | #1228 (this PR). Signed sums, nonzero counts, Schedule C including SSTB, rental without farm rent. | Identical to the release's: on `national_state` every row is its own concept group and no value changes |
| `concept_exact_no_business_count` | As `concept_exact`, without the 52 Schedule C return counts. The frame has almost no Schedule C loss carriers. | Recomputed (`target_loss_weights`) |
| `drop` | The 260 rows whose release concept differs from the IRS column are dropped. The rental amount keeps its release concept. | Recomputed |
| `rebase_positive_part` | The capital-gains approach of #1036. Business and rental keep one-sided concepts (Schedule C including SSTB; rental plus farm rent), and each HT2 state row becomes its share of a national Table 1.4 TY2022 net-income control. Partnership rows stay concept-exact, because Table 1.4 has no combined net-income column after TY2020. | Recomputed (208 values change) |

The Table 1.4 controls are not in the pinned feed (PolicyEngine/chronicle#346). They were read from the feed-pinned `22in14ar.xls`:

| Control | Returns | Amount |
|---|---:|---:|
| "Business or profession: Net income" | 21,969,832 | $543,564,980 thousand |
| "Total rental and royalty: Net income" | 6,380,131 | $148,879,455 thousand |

Amounts are aged by the rows' own factor, 1.12034597479376; counts are not aged, as in the release.

## Method

- **`prepare.py`** streams the checkpoint's 5,694 target columns into a CSR matrix. It also builds the 312 replacement rows, the arms' loss weights and household-level concept totals. Checks (`MANIFEST.json`):
  - `A @ initial` and `A @ published` reproduce every target's recorded initial and published estimate exactly (max relative gap 0).
  - Rebuilding the 312 rows from person inputs on the release's concepts matches the checkpoint's materialized columns to 5.4e-8, its float32 storage precision.
  - The concept-exact rows come from the same code with the #1228 rules.
  - `target_loss_weights` reproduces the release's recorded loss weights exactly.
- **`prepare_rebase.py`** builds the `rebase_positive_part` rows and values from the same concept totals.
- **`resolve.py`** calls `microcosm.calibrate.calibrate` with the release's options from `calibration_diagnostics.json`: Adam, 1,500 epochs, learning rate 0.02, `mass="conserve"`, `max_weight_ratio=5`, `target_loss_cap=1`, no L1/L2, seed 0. The release arm's epoch-1 loss, 0.202372, equals the release's recorded initial loss, 0.2023716.
- **`modal_resolve.py`** runs the five arms on Modal: 16 CPUs each, torch 2.12, the kernel packages from this branch's HEAD.
- **`analyze.py`** scores each arm and writes `results/summary.{json,md}`.

Inputs live in `/Users/maxghenis/PolicyEngine/_build_artifacts/1228-net-income-resolve/inputs` and runs in `.../runs`. The scripts hard-code the Route A run directory (`_recovered/scratch-backup/893/overnight-20260923/route-a/run-4b57d15a287c`), as research harnesses here do. The arms' recorded metrics predate the inputs-digest and `resolve.py`-digest fields that `modal_resolve.py` now checks before reusing a finished arm.

## Results

`results/summary.md` and `results/summary.json`, written by `analyze.py`. Each arm ran on Modal with 16 CPUs in 20–26 minutes, at kernel HEAD 5146a96b0.

**The replay reproduces the release.** The `release` arm ends at loss 0.028420; the release recorded 0.028426. Its weights correlate with the published ones at 0.9999, the per-household ratio stays within 0.79–1.25, and every model total below matches the published weights' to within 0.1%. The other arms' differences are therefore the binding's, not the solver's.

### Fit

| Arm | Own loss | Own rows within 10% | The 312 rows within 10% (own binding) | The 5,382 shared rows within 10% | Shared rows, mean capped error | Kish ESS |
|---|---:|---:|---:|---:|---:|---:|
| release | 0.02842 | 93.41% | 84.0% | 93.96% | 0.03064 | 77,794 |
| concept_exact | 0.02641 | 93.89% | 92.6% | 93.96% | 0.03074 | 77,306 |
| concept_exact_no_business_count | 0.02583 | 94.04% | 93.1% (260 rows) | 94.09% | 0.03042 | 82,769 |
| drop | 0.02414 | 94.33% | 84.6% (52 rows) | 94.43% | 0.02848 | 84,834 |
| rebase_positive_part | 0.02552 | 94.06% | 92.9% | 94.13% | 0.03011 | 81,036 |

No arm worsens any other target family's mean capped error by more than 0.0025 against the release arm. Under `concept_exact`, AGI's error falls from 0.0138 to 0.0109; the largest rise is itemized deductions', from 0.0439 to 0.0465. Two arms improve one family more: dropping the rows cuts real estate taxes' error by 0.018, and the rebase cuts capital gains' by 0.014.

### Model totals: what each binding does to the income it targets

These are 2024 totals under each arm's weights, summed over tax units from the frame's inputs. "IRS" is Table 1.4 TY2022 aged ×1.12035, the rows' own factor; counts are not aged.

**Schedule C (`self_employment_income + sstb_self_employment_income`)**

| | Positive part | Profit units | Negative part | Loss units | Net | Nonzero units |
|---|---:|---:|---:|---:|---:|---:|
| IRS | $609.0B | 21.97M | −$149.2B | 8.39M | $459.8B (HT2 net: $472.3B) | 30.36M (HT2: 30.35M) |
| release | $836.9B | 30.64M | −$15.9B | 0.65M | $821.0B | 31.29M |
| concept_exact | $532.2B | 28.08M | −$25.3B | 1.29M | $506.9B | 29.37M |
| concept_exact_no_business_count | $513.0B | 13.65M | −$19.5B | 0.75M | $493.5B | 14.40M |
| drop | $1,332.9B | 19.03M | −$12.1B | 0.52M | $1,320.7B | 19.55M |
| rebase_positive_part | $627.7B | 21.89M | −$14.4B | 0.60M | $613.3B | 22.49M |

- **SSTB Schedule C**, which the release's business concept leaves out, totals $347.9B under the release, $95.6B under `concept_exact` and $103.3B under the rebase.
- **The frame carries few Schedule C losses** in any arm: 0.5–1.3M units against the IRS's 8.39M. This is the `puf_tax_detail` stage's `nonnegative_outputs`. So no binding gets both the net and the profits right:
  - `concept_exact` hits the net and the nonzero count, with too many, smaller profit units (1.28×) and profits at 0.87×;
  - the rebase hits profits and profit units, with the net at 1.30–1.33×.
- **Dropping the rows lets Schedule C float to $1.3T**, so other targets do not pin it.

**Partnership and S corporation (`tax_unit_partnership_s_corp_income`)**

| | Positive part | Profit units | Negative part | Loss units | Net | Nonzero units |
|---|---:|---:|---:|---:|---:|---:|
| IRS | ≤ $1,456.2B (legs' sum) | ≤ 7.34M (legs' sum) | | | $1,155.3B (HT2: $1,157.6B) | HT2: 9.55M |
| release | $1,129.0B | 9.48M | −$108.0B | 2.01M | $1,021.0B | 11.49M |
| concept_exact | $1,248.7B | 7.26M | −$128.9B | 2.26M | $1,119.8B | 9.52M |
| drop | $986.9B | 5.13M | −$175.8B | 2.32M | $811.1B | 7.45M |

The release's 9.48M positive units exceed the 7.34M upper bound. Under `concept_exact` both sides land inside the bounds, and the net and nonzero count land near HT2's.

**Rental and royalty**

| | Line 26 net (`rental_income`) | Line 26 nonzero units | With farm rent: positive part | With farm rent: profit units | With farm rent: net |
|---|---:|---:|---:|---:|---:|
| IRS | HT2: $95.9B | HT2: 9.40M | $166.8B | 6.38M | $104.4B |
| release | $248.6B | 14.09M | $345.3B | 9.32M | $253.1B |
| concept_exact | $98.4B | 9.32M | $199.5B | 6.15M | $111.3B |
| drop | $95.1B | 14.25M | $216.1B | 8.95M | $96.2B |
| rebase_positive_part | $118.2B | 10.67M | $192.9B | 6.35M | $119.8B |

The release's positive-only rental count, pinned at HT2's profit-or-loss count, held 9.3M positive units against the net target. That left the rental net at 2.6× its target. `concept_exact` brings both the line-26 net and count within 3% of HT2.

### Reading

- **`concept_exact`** is the only arm that is right on every quantity the HT2 columns measure:
  - the business, partnership and rental nets and nonzero counts land within 0.3–7.3% of HT2 nationally;
  - 82–100% of each measure's state rows land within 10%;
  - fit elsewhere and ESS are unchanged.

  Its remaining error, Schedule C profit units at 1.28× and profits at 0.87×, is the frame's missing Schedule C losses.
- **`rebase_positive_part`** gets Schedule C profits right and the net 30% high, for the same reason. It needs the Table 1.4 legs in Chronicle (PolicyEngine/chronicle#346) and a share assumption. It has no partnership counterpart.
- **Dropping the Schedule C count** (`concept_exact_no_business_count`) halves the Schedule C units (14.4M against 30.4M nonzero returns) and buys almost nothing.
- **Dropping all 260 rows** lets Schedule C reach $1.3T.

Once the frame carries Schedule C losses and Chronicle carries the legs, `concept_exact` plus national Table 1.4 profit and loss targets can get both sides.
