"""Income-test failures among own SSI reporters, by support channel (2026-01).

Usage (next to measure_receives_ssi.py):
    PYTHONPATH=<pe-us tree> python income_fail_by_channel.py
"""

import measure_receives_ssi as m
import numpy as np
import pandas as pd
from policyengine_us import Microsimulation


def weighted_median(values: pd.Series, weights: pd.Series) -> float | None:
    if not len(values):
        return None
    order = np.argsort(values.to_numpy())
    cumulative = np.cumsum(weights.to_numpy()[order])
    index = np.searchsorted(cumulative, cumulative[-1] / 2)
    return float(values.to_numpy()[order][index])


def main() -> None:
    ds, _reporter_households, _co_households = m.subset_dataset(False)
    sim = Microsimulation(dataset=ds, dataset_end_year=2026)
    month = "2026-01"
    frame = pd.DataFrame(
        {
            "w": sim.calculate("person_weight", 2026).values,
            "eligible": sim.calculate("is_ssi_eligible", month).values,
            "uncapped": sim.calculate("uncapped_ssi", month).values,
            "countable_income": sim.calculate("ssi_countable_income", month).values,
            "earned": sim.calculate("ssi_earned_income", month).values,
            "unearned": sim.calculate("ssi_unearned_income", month).values,
            "social_security": sim.calculate("social_security", 2026).values / 12,
            "employment": sim.calculate("employment_income", 2026).values / 12,
        }
    )
    frame["channel"] = ds.person["person_support_channel"].astype(str).to_numpy()
    frame["reporter"] = m.own_ssi_receipt(ds.person)
    frame["ssi_val_monthly"] = pd.to_numeric(ds.person["SSI_VAL"]).to_numpy() / 12
    reporters = frame[frame.reporter]
    fail = reporters[reporters.eligible.astype(bool) & (reporters.uncapped <= 0)]
    for channel, group in fail.groupby("channel"):
        weight = group.w
        print(
            channel,
            "income-test failures",
            round(weight.sum()),
            "of reporters",
            round(reporters.loc[reporters.channel == channel, "w"].sum()),
            "| weighted median monthly countable income",
            round(weighted_median(group.countable_income, weight)),
            "unearned",
            round(weighted_median(group.unearned, weight)),
            "Social Security",
            round(weighted_median(group.social_security, weight)),
            "reported SSI",
            round(weighted_median(group.ssi_val_monthly, weight)),
            "| share with Social Security",
            round(
                float((weight * (group.social_security > 0)).sum() / weight.sum()), 3
            ),
            "| share with wages",
            round(float((weight * (group.employment > 0)).sum() / weight.sum()), 3),
        )


if __name__ == "__main__":
    main()
