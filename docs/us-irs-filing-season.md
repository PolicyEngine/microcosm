# Compare IRS filing seasons

The IRS workbook counts current-year individual returns as the IRS first processes them through a stated cutoff. Chronicle records the tax year, processing week, cutoff, measure, units and adjusted gross income band. Microcosm compares the same processing week across tax years. It calculates the itemization rate by dividing itemizing returns by all processed returns in the same income band and year.

## Reproduce the 2024–2025 comparison

Run these commands from Chronicle commit `25e9a69bfd3228a7f7b66cb12eb8965f57b5ff7f`. Each bundle reads the two official workbooks for its tax year. Inspect each bundle's validation report before combining the rows. [Chronicle source commit](https://github.com/PolicyEngine/chronicle/commit/25e9a69bfd3228a7f7b66cb12eb8965f57b5ff7f)

```sh
for year in 2024 2025; do
    uv run chronicle build-bundle --year "$year" \
        --source "soi-filing-season-week21-$year" \
        --source "soi-filing-season-week30-$year" \
        --out "filing-$year" --slim || exit 1
done
cat filing-2024/consumer_facts.jsonl filing-2025/consumer_facts.jsonl > filing.jsonl
shasum -a 256 filing-2024/consumer_facts.jsonl filing-2025/consumer_facts.jsonl filing.jsonl
```

The following SHA-256 digests identify this comparison's input. Each year contributes 3,944 observations.

| File | SHA-256 |
| --- | --- |
| 2024 fact rows | `9c1888f34a0882c61988a4b908f651157200f4100e88e0f8b8008c86938569c9` |
| 2025 fact rows | `a5bccb16eb65d708490dec7be3259336989e91ba279565d79bc024ae19a5acce` |
| Combined rows in tax-year order | `aec22991b0b4ff64a67b980f52f3c1ab77cfe3b8a16db0fd976176b01e165d8c` |

Run the report from Microcosm with the combined file's path.

```sh
uv run python tools/report_us_filing_season.py /path/to/filing.jsonl \
    --first-year 2024 --last-year 2025 --week 30 --out filing-week30.json
```

For week 30, the IRS processed 12,684,510 itemizing returns through July 24, 2025, for tax year 2024 and 16,223,991 through July 30, 2026, for tax year 2025. The report records the source digest and both cutoff dates. A measure that appears in only one year's observations retains a missing comparison value. [IRS filing-season statistics](https://www.irs.gov/statistics/filing-season-statistics)

The annual fiscal compiler excludes processing-week observations from annual controls and records their source identifiers in the exclusion receipt. The four longstanding TY2024 week-47 earned income tax credit series keep their existing treatment.
