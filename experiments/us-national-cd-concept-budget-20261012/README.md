# National concept budget for congressional-district rows

The national fiscal release weights each target's loss with
`sqrt_value_concept_budget_weighted_mape_50_50_amount_count_target_scale_cap_100pct`
(`microcosm.build.us_runtime.target_loss_weights`). Step 3 of that formula is
the concept budget from microcosm#195 (June 2026): the congressional-district
targets of one concept in one state form one group, which is rescaled to sum
to its largest member's weight, so adding district geography cannot multiply
a concept's weight.

On the full surface it pooled nothing. The district key reads every metadata
key outside `US_FISCAL_TARGET_CONCEPT_METADATA_EXCLUSIONS`. Commit efa6052c9
(2026-09-05, "Normalize calibration hierarchy ownership") began stamping three
per-row labels on every compiled row: `ledger_fact_label`,
`ledger_layout_groupby_value_label` and `diagnostic_target_label`. The first
two name the district ("WV congressional district 1"). The third repeats the
fact label when the fact's period is the build's, which a district row on the
pinned feed does not meet (its facts are TY2023, the build 2024), but a 2023
build would. None of the three was excluded, so every compiled district row
became its own group.

The ACS local mapping (microcosm#1104) already excluded the first two for its
own district rows. That PR measured the national gap and left the national
mapping unchanged on purpose
(`experiments/us-acs-local-target-loss-weights-20261004/`, "A finding for the
national release").

## The change

- `LEDGER_PER_ROW_LABEL_METADATA` names the three labels, and the national
  exclusions now include them. The ACS local key inherits them. Its own
  addition is now only `state_cd_cd_file_value`, so its exclusion set is the
  one it had plus `diagnostic_target_label`.
- `validate_district_concept_groups` now guards the national mapping
  (`us_fiscal_ledger_metadata.v2`), as the ACS local mapping's guard already
  did for its rows. It refuses a surface where:
  - the district rows of one concept in one state (`district_concept_identity`)
    land in more than one group, naming the metadata keys that split them;
  - a district group spans states;
  - a district row has no `state_fips`.

  So a new per-row key fails the weights loudly instead of silently
  disabling the budget again.
- The formula id (`US_FISCAL_TARGET_LOSS_WEIGHTING`) is unchanged. It names
  the formula, and Route A's recorded weights (`national_state`, no district
  rows) are reproduced bit for bit. On the full surface the weights move, so
  the loss vector digest in `target_loss_basis` moves. The exact-k
  incumbent gate compares that digest, so it refuses to compare a run
  weighted the old way with one weighted the new way.

## Results on the pinned feed

`national_cd_concept_budget.py` compiles the pinned feed
(`facts_sha256` `965a29ac…`, Chronicle `51aa40f`) at period 2024 with aging,
which is what `--target-surface full` calibrates. It computes the weights with
the pre-change code (the verbatim copy in `test_support`) and with the
current module. Output: `results.json`.

The surface has 33,249 compiled targets. 24,340 of them (73.2%) are district
rows, all `irs_soi`. They form 3,125 concept-in-state identities:

- 55 concepts from the district file in each state with districts;
- 40 Historic Table 2 proxy concepts in each of eight states (the seven
  single-district states and Montana).

| District concept groups | Groups | Singletons | Largest |
|---|---:|---:|---:|
| Before | 24,340 | 24,340 | 1 |
| After | 3,125 | 665 | 52 (California) |

After the change, each group is exactly one identity: no identity is split
and no group pools two. The 665 singletons are the seven single-district
states (AK, DE, DC, ND, SD, VT, WY), 95 concepts each (55 from the district
file and 40 Historic Table 2 proxies). Each of those concepts has one district
row in its state, so it has nothing to pool.

### Share of the loss (each row's weight over the total; equal misses)

| Level | Targets | Equal weights | Before | After |
|---|---:|---:|---:|---:|
| Congressional district | 24,340 | 73.2% | 45.6% | 11.0% |
| State | 8,409 | 25.3% | 42.1% | 68.8% |
| Country | 500 | 1.5% | 12.3% | 20.2% |

By basis, district amounts go from 23.0% to 5.8% and district counts from
22.6% to 5.2%. Counts and amounts each still carry half of the total.

By family, `irs_soi` goes from 90.5% to 84.5%. Every other family gains by
the same factor as its rows, for example `census_population` 4.8% → 7.9% and
`cms_medicaid` 1.2% → 1.9%.

### What happens to each weight

- **Every non-district target keeps its weight up to one rescale per basis:**
  ×1.6373 for amounts and ×1.6332 for counts. Its group, its value weight and
  its rank among the non-district rows of its basis are unchanged. The
  district rows no longer crowd it out.
- **A district row's weight is scaled by its group's budget over the group's
  total, then by the basis rescale.**
  - The median district weight is 0.157 of what it was. The smallest ratio,
    0.033, falls on rows in California's 52-district groups.
  - A single-district state's rows keep their weight up to the basis rescale
    (ratio 1.633 to 1.637).
  - The districts of one concept in one state now carry together what their
    largest district would carry alone.
- **Weights still have mean 1.** Their range moves from [2.0e-5, 263] to
  [3.3e-5, 430]. The ten largest weights are all national rows (the largest
  is the one `federal_reserve` target), which gain the basis rescale.

### What it does not change

- **Route A** and any `--target-surface national_state` run. Those surfaces
  have no district rows, and a non-district row's key never reads metadata.
  `test_route_a_weights_and_loss_basis_hash_are_reproduced_bit_for_bit` still
  passes against the recorded weights and the loss basis hash `206ada09…`.
- **The ACS local weights.** Its key already excluded the two district labels.
  The one addition, `diagnostic_target_label`, cannot change its grouping:
  - its guard already requires each district identity to be one group;
  - two identities differ in metadata the key still reads;
  - a non-district row's key reads no metadata.

  The label is also absent from every district row of the pinned national
  compile, which the ACS local surfaces start from. The ACS local tests pass
  unchanged.
- **Target values, the compiled surface, and any published release.** This
  changes only the objective of future `--target-surface full` calibrations.

## The CI fixture

The script also writes
`packages/microcosm-build/tests/fixtures/us_national_district_concept_rows.json`.
It holds 31 real compiled district rows from the pinned feed, with the
metadata the compiler stamped:

- seven concepts in West Virginia (two districts), Montana (two districts and
  Historic Table 2 proxies) and Wyoming (at large and proxies);
- the concepts are AGI, return count, EITC two-children claims, itemized
  deductions and net capital gains from the district file, plus return count
  and net capital gains from Historic Table 2.

The tests use it on CI, where the feed is not available.

## Reproduce

From the repository root, with the pinned feed at
`~/PolicyEngine/_buildh-runtime/inputs/consumer_facts_us_51aa40f.jsonl`:

```bash
uv run python experiments/us-national-cd-concept-budget-20261012/national_cd_concept_budget.py
```

The script refuses a feed whose sha256 is not the pin. Where the feed is on
disk, `test_pinned_feed_district_rows_form_one_budget_per_concept_per_state`
in `packages/microcosm-build/tests/engine_free/us/test_us_target_loss_weights.py`
asserts these counts and shares on every run. Elsewhere it skips.
