# Static-aging integration benchmark

This experiment compares PolicyEngine-US's uniform population-weight extension
with Microcosm's age-and-sex static aging. It uses 2024 as the base and projects
2025, 2030 and 2035. Both paths use PolicyEngine-US 2.6.10 and the certified
`populace-us-2024-spm-20260909` input. This historical model pin makes the comparison
reproducible; it differs from the model pin in `policyengine` 6.0.0.

The benchmark passes custom `USMultiYearDataset` objects directly to the country
engine and computes population statistics through MicroSeries and `map_to`.
It tests integration on one input dataset. It does not certify a new data release
or resolve the adoption decision in [#333](https://github.com/PolicyEngine/microcosm/issues/333).

## Inputs and assumptions

Supply these files locally. The runner checks their SHA256 hashes before computing.

| Input | SHA256 |
| --- | --- |
| `populace_us_2024.h5`, release `populace-us-2024-spm-20260909` | `6496cc4393d4d3c6574f76eca231de5898c803b9067645591fd5c4d3e65aee84` |
| Social Security Administration (SSA) `SSPopJul_TR2024.csv` | `cb6ab96eba35554e92e279e839662106602078ba7ff0dfaa306a8a0ca5a919e5` |

Static aging anchors each demographic cell to the base frame and applies the SSA
cell's projected growth. It uses seed 0, 300 calibration epochs and a maximum
weight ratio of 5. SSA and Census population definitions differ, so the two
projection paths need not produce the same total population.

For mixed-sign inputs that follow a national-total series, static aging assumes
that gross positive amounts and gross losses each follow that series' growth.
It calculates a separate positive factor for each side. This preserves record
signs and the projected net total even when demographic reweighting changes the
net total's sign. A net projection alone does not identify both gross projections;
the equal-growth assumption supplies that additional constraint. Price-index
inputs use one common index ratio. See [the methodology](../../docs/static-aging.md).

## Reproduce

Run these commands from the Microcosm checkout containing this change. Use an
isolated environment to preserve the benchmark's historical engine pin. Avoid
`uv sync` in this environment because the repository lock can select a different
engine version.

```bash
set -e
bench_repo="$(git rev-parse --show-toplevel)"
bench_env="/tmp/microcosm-957-env"
bench_output="/tmp/microcosm-957-output"
bench_h5="/absolute/path/to/populace_us_2024.h5"
bench_ssa="/absolute/path/to/SSPopJul_TR2024.csv"

uv venv --python 3.14 "$bench_env"
uv pip install --python "$bench_env/bin/python" \
  policyengine==6.0.0 policyengine-us==2.6.10 policyengine-core==3.32.5 \
  spm-calculator==1.0.0 microdf-python==1.3.0 \
  numpy==2.4.6 pandas==3.0.3 torch==2.12.0 \
  -e "$bench_repo/packages/microcosm-graph" \
  -e "$bench_repo/packages/microcosm-frame" \
  -e "$bench_repo/packages/microcosm-calibrate"

run_benchmark() {
  uv run --no-project --python "$bench_env/bin/python" \
    "$bench_repo/experiments/957-static-aging/benchmark.py" "$@" \
    --h5 "$bench_h5" --ssa "$bench_ssa" --output "$bench_output"
}

run_benchmark preflight
run_benchmark engine 2024
for year in 2025 2030 2035; do
  run_benchmark engine "$year"
  run_benchmark aging "$year"
done
```

Each invocation releases its simulation memory before the next starts. Run the
baseline and six comparison cells sequentially; full population calculations use
substantial memory. Keep the source files unchanged until the commands finish.
Use `--repo /path/to/checkout` when the script and editable imports live in
different checkouts. The runner checks that its imported source files belong to
that checkout.

## Validation and receipts

`preflight.json` records exact base-export parity, demographic cell fit, weight
bounds, and checks on every mapped input. The runner independently constructs
expected values and checks float64 Frame and engine exports. It verifies gross
positive, gross negative and net aggregate growth for national-total inputs;
it also verifies record signs, unchanged columns and index factors.

Each `engine-YEAR.json` or `aging-YEAR.json` records population, the share aged 65
or older, employment income, retirement benefits and recipients, partnership
income, poverty, and income tax where the engine completes those calculations.
The `complete` field distinguishes a finished cell from an interrupted one.

PolicyEngine-US 2.6.10 can raise a circular-dependency error for poverty from 2027
through Indiana's state supplement, Medicaid and SNAP
([#9534](https://github.com/PolicyEngine/policyengine-us/issues/9534)). The runner
records the full error and leaves affected outputs null. It stops further formula
calculations in that cell after the error and does not alter policy inputs to
produce a result.

Receipts include model and package versions, input hashes, source hashes, the Git
head, the tracked working-tree diff hash, and Git status. `runner-snapshots/` and
`source-snapshots/` preserve the exact runner, source files and tracked diff.
The runner rejects a receipt if source files change during execution. Outputs
contain aggregate statistics and code provenance; the runner does not copy raw
microdata into the output directory.
