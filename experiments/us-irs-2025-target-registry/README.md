# US annual IRS target registry experiment

This experiment builds fiscal target registries for tax years 2024 and 2025. Microcosm selects source observations available for each target period and applies its standard aging factors. The experiment keeps the tax year on every IRS observation. Registry compilation does not fit household weights.

| Target year | Base targets | Candidate targets | Identical non-IRS controls | Candidate version |
| --- | ---: | ---: | ---: | --- |
| 2024 | 33,249 | 32,411 | 1,482 | `30de5188bba9` |
| 2025 | 33,219 | 32,381 | 1,484 | `d521a3ce9bab` |

The 2025 candidate has 976 fewer congressional-district-source controls, 102 more state-by-AGI controls and 36 more broad state controls. The legacy source mapped IRS columns N19300 and A19300 to a general interest deduction in all 488 geographies. The 2023 package identifies these columns as home mortgage interest paid to financial institutions. It also identifies interest paid to individuals, deductible points, and investment interest. Microcosm has no selected controls for those eight distinct counts and amounts. Both registries cover the same 436 district codes, 51 states/DC and US total.

The comparison uses the release crosswalk for the 2022 district source and the 118th-to-119th Congress crosswalk for the 2023 source. The county adapter separately yields 6,286 county controls for return counts and adjusted gross income. It uses 2023 IRS reporting areas, the 2024 Census PEP county roster and the candidate’s state totals. Kalawao and Maui share one published IRS cell.

## Inputs

Use the US consumer feed from Chronicle `41b2fb2b` and, for the optional county controls, the county feed from `3e9d3a0`. The [feed reconstruction guide](../../docs/us-chronicle-feed-repin.md) describes these files. The refresh uses four additional Chronicle consumer-contract JSONL files. [Source pins](sources.json) record their SHA-256 digests, IRS files, tax years, row counts and publisher URLs.

| Argument | What the file contains |
| --- | --- |
| `--annual` | Seven 2023 candidate packages in this order: US Historic Table 2, state AGI, state broad totals, state EITC, US totals from the state workbook, Roth IRA contributions and traditional IRA contributions. The integration reconciles [Chronicle PR 295](https://github.com/PolicyEngine/chronicle/pull/295) and [PR 296](https://github.com/PolicyEngine/chronicle/pull/296) to the current source contracts. It emits 6,274 rows. |
| `--geographic` | The 2023 bundle for `soi-county-2023`, `soi-congressional-district-2023` and `soi-w2-statistics`, in that order. It emits 36,113 rows. |
| `--w2-2021`, `--w2-2022` | The `soi-w2-statistics` source suite for each tax year. Each emits five rows. |

The [annual source reconstruction](annual-source/rebuild.py) reads published Git objects from a Chronicle clone. It stages the seven packages in a temporary directory, reconciles their schemas with the original feed's source contracts, applies Chronicle's parsers and verifies the resulting feed digest. From the Microcosm repository root, set `CHRONICLE` to a local clone of [PolicyEngine/chronicle](https://github.com/PolicyEngine/chronicle) and `INPUTS` to a directory for the rebuilt files. Fetch the immutable commits listed below. The Chronicle project supplies the Python dependencies.

```sh
git -C "$CHRONICLE" fetch origin \
  c020783da1621b032467f36bf688d597cdd5604c \
  f55ae34915e52601f25a039e7888b9d027f5b5c4 \
  3185d8762eb6dcda8d588b729ecd5aa6d7f84ec1 \
  25e9a69bfd3228a7f7b66cb12eb8965f57b5ff7f
uv run --project "$CHRONICLE" python \
  experiments/us-irs-2025-target-registry/annual-source/rebuild.py \
  --chronicle "$CHRONICLE" --out "$INPUTS/annual-candidate-feed.jsonl"
```

From the Chronicle source checkout at the [geographic and W-2 commit](https://github.com/PolicyEngine/chronicle/commit/25e9a69bfd3228a7f7b66cb12eb8965f57b5ff7f), run:

```sh
uv run chronicle build-bundle --year 2023 --slim \
  --source soi-county-2023 --source soi-congressional-district-2023 \
  --source soi-w2-statistics --out "$INPUTS/geo-w2-2023"
for year in 2021 2022; do
  uv run chronicle build-suite soi-w2-statistics --year "$year" \
    --out "$INPUTS/w2-suite-$year"
done
```

From the Microcosm repository root, set `BASE`, `COUNTIES` and `INPUTS` to the two pinned feeds and the supplement directory. Use a new output directory.

```sh
uv run python experiments/us-irs-2025-target-registry/compile.py \
  --base "$BASE" --annual "$INPUTS/annual-candidate-feed.jsonl" \
  --geographic "$INPUTS/geo-w2-2023/consumer_facts.jsonl" \
  --w2-2021 "$INPUTS/w2-suite-2021/consumer_facts.jsonl" \
  --w2-2022 "$INPUTS/w2-suite-2022/consumer_facts.jsonl" \
  --county-universe "$COUNTIES" --year 2025 --out "$INPUTS/annual-registry-2025"
```

`registry.jsonl.gz` contains the native `TargetSpec.to_dict()` records. `receipt.json` contains the registry version, counts by record set, source digests, retired target mappings and support exclusions. With the county input, the run also writes `county-registry.jsonl.gz` and the state reconciliation blocks.

## Source selection

The candidate removes 26,893 rows whose 2023 labels point to 2020 W-2 or 2022 district, state and IRA files. It checks that the new source reproduces all 1,020 existing 2023 state-by-AGI observations. For every state and DC, the 2023 source publishes separate $500,000–$1 million and $1 million-and-over taxable-interest counts and amounts. The compiler must select both new groups before the experiment will retire the overlapping 2022 $500,000-and-over control.

Six 2023 observations cover interest amounts and counts for the under-$1 AGI group in Iowa, North Dakota and Vermont. Their predecessors carry reviewed support exclusions. The receipt records their values and reasons, and the experiment holds them out. Native selection maps one 2023 W-2 amount, Box 7 Social Security tips, to a target. It does not currently select the four IRA observations or the four other W-2 measures. The two previously admitted 2024 week-47 EITC controls remain in the base and candidate. The new 2024 and 2025 partial-season releases stay outside this annual comparison.
