"""Ten-year federal revenue estimate with policyengine.

Scores raising the top individual income tax rate from 37% to 39.6%
starting in 2026, year by year over the 2026-2035 budget window. This
is the first option in Table 2 of CRS Report R49052, so the output
lines up against the Tax-Simulator and Tax-Calculator estimates there.

Setup (one-time, Python 3.11-3.14):
    pip install "policyengine[us]"

Run:
    python ten_year_revenue_estimate.py

The first run downloads the certified national dataset into ./data.
Every release of the policyengine package pins the exact model and
data versions it was built against (printed at the top of the output),
so a given package version reproduces the same numbers.

This is a static, calendar-year estimate with no behavioral responses.
"""

import datetime
import resource
import sys
import time

import policyengine as pe
from policyengine.core import Parameter, ParameterValue, Policy, Simulation
from policyengine.outputs import (
    Aggregate,
    AggregateType,
    ChangeAggregate,
    ChangeAggregateType,
)

YEARS = range(2026, 2036)
CANDIDATE = (
    "candidate_budgeted_full_vector_all_targets.h5"  # output of rake_candidate.py
)
CANDIDATE_STEM = "candidate_budgeted_full_vector_all_targets"

# Federal individual income tax liability (net of refundable credits).
REVENUE_VARIABLE = "income_tax"


def permanent_change(path, value, start=datetime.date(2026, 1, 1)):
    """A parameter change in effect from `start` through 2100."""
    return ParameterValue(
        parameter=Parameter(name=path, tax_benefit_model_version=pe.us.model),
        start_date=start,
        end_date=datetime.date(2100, 12, 31),
        value=value,
    )


# Parameter paths mirror the YAML tree in the policyengine-us repository
# under policyengine_us/parameters/ (e.g. gov/irs/income/bracket.yaml).
REFORM = Policy(
    name="Top individual income tax rate to 39.6%",
    parameter_values=[
        permanent_change("gov.irs.income.bracket.rates.7", 0.396),
    ],
)


def peak_memory_gb():
    """Peak resident memory of this process so far, in GB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6


def main():
    started = time.time()
    bundle = dict(pe.us.model.release_bundle)
    print(
        f"policyengine {bundle['policyengine_version']} | "
        f"{bundle['model_package']} {bundle['model_version']} | "
        f"data {bundle['certified_data_build_id']}",
        flush=True,
    )

    # Downloads (first run only) and uprates the certified national
    # dataset to each analysis year.
    # PROTOTYPE CANDIDATE, not the certified dataset: the only change from
    # ten_year_revenue_estimate.py is where the data comes from.
    from policyengine.tax_benefit_models.us.datasets import create_datasets

    datasets = create_datasets(
        datasets=[CANDIDATE],
        years=list(YEARS),
        data_folder="./data_candidate",
        allow_unmanaged=True,
    )

    rows = []
    for year in YEARS:
        dataset = datasets[f"{CANDIDATE_STEM}_{year}"]
        baseline = Simulation(dataset=dataset, tax_benefit_model_version=pe.us.model)
        reformed = Simulation(
            dataset=dataset,
            tax_benefit_model_version=pe.us.model,
            policy=REFORM,
        )
        baseline.ensure()
        reformed.ensure()

        level = Aggregate(
            simulation=baseline,
            variable=REVENUE_VARIABLE,
            aggregate_type=AggregateType.SUM,
        )
        level.run()
        change = ChangeAggregate(
            baseline_simulation=baseline,
            reform_simulation=reformed,
            variable=REVENUE_VARIABLE,
            aggregate_type=ChangeAggregateType.SUM,
        )
        change.run()

        rows.append((year, level.result / 1e9, change.result / 1e9))
        print(
            f"{year}: baseline ${rows[-1][1]:,.0f}B, "
            f"revenue change {rows[-1][2]:+,.1f}B",
            flush=True,
        )

    total = sum(r[2] for r in rows)
    print(f"\n2026-2035 revenue change: ${total:+,.1f}B", flush=True)
    print(
        f"Run time {(time.time() - started) / 60:.0f} min, "
        f"peak memory {peak_memory_gb():.1f} GB",
        flush=True,
    )

    with open("ten_year_revenue_estimate_candidate.csv", "w") as f:
        f.write("year,baseline_income_tax_bn,revenue_change_bn\n")
        for year, level_bn, change_bn in rows:
            f.write(f"{year},{level_bn:.3f},{change_bn:.3f}\n")
    print("Wrote ten_year_revenue_estimate_candidate.csv", flush=True)


if __name__ == "__main__":
    main()
