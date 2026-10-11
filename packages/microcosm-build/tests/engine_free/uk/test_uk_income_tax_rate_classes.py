"""Income tax by HMRC rate category (microcosm#1095, uk-data#533)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from microcosm.build.uk_runtime.income_tax_rate_classes import (
    UK_RATE_CLASSES,
    relief_by_rate_class,
    tax_by_rate_class,
)
from test_support.paths import paths_for

#: Rest-of-UK and Scottish schedules shaped like policyengine-uk's 2025 ones
#: (taxable-income thresholds, after the personal allowance).
REST_OF_UK = ([0.0, 37_700.0, 125_140.0], [0.20, 0.40, 0.45])
SCOTLAND = (
    [0.0, 2_827.0, 14_921.0, 31_092.0, 62_430.0, 112_570.0],
    [0.19, 0.20, 0.21, 0.42, 0.45, 0.48],
)


def _schedule_tax(income: np.ndarray, thresholds, rates) -> np.ndarray:
    uppers = list(thresholds[1:]) + [np.inf]
    return sum(
        rate * np.clip(income - lower, 0.0, upper - lower)
        for lower, upper, rate in zip(thresholds, uppers, rates, strict=True)
    )


@pytest.mark.parametrize("schedule", [REST_OF_UK, SCOTLAND], ids=["ruk", "scotland"])
def test_rate_classes_add_up_to_the_schedule_and_rise_with_income(schedule) -> None:
    rng = np.random.default_rng(0)
    income = np.concatenate(
        [rng.uniform(-1_000.0, 200_000.0, 5_000), np.asarray(schedule[0][1:])]
    )

    classes = tax_by_rate_class(income, *schedule)
    more = tax_by_rate_class(
        income + rng.uniform(0.0, 20_000.0, income.size), *schedule
    )

    assert set(classes) == set(UK_RATE_CLASSES)
    np.testing.assert_allclose(
        sum(classes.values()), _schedule_tax(income, *schedule), atol=1e-6
    )
    for name in UK_RATE_CLASSES:
        assert np.all(more[name] >= classes[name] - 1e-9)


def test_scottish_rates_group_into_hmrc_categories() -> None:
    """Starter, basic and intermediate are basic; higher and advanced are
    higher; top is additional."""

    _, _, _, higher, advanced, top = SCOTLAND[0]
    income = np.array(
        [higher - 1.0, (higher + advanced) / 2, (advanced + top) / 2, top + 10_000.0]
    )

    classes = tax_by_rate_class(income, *SCOTLAND)

    assert classes["higher"][0] == 0.0 and classes["additional"][0] == 0.0
    assert classes["higher"][1] == pytest.approx(0.42 * (income[1] - higher))
    assert classes["higher"][2] == pytest.approx(
        0.42 * (advanced - higher) + 0.45 * (income[2] - advanced)
    )
    assert classes["additional"][2] == 0.0
    assert classes["additional"][3] == pytest.approx(0.48 * 10_000.0)


def test_a_bracket_not_in_force_is_skipped() -> None:
    classes = tax_by_rate_class(
        np.array([50_000.0]), [0.0, 2_000.0, 37_700.0], [0.20, None, 0.40]
    )

    # With the starter bracket skipped, 40% is the only rate above the basic
    # class, so it is the higher rate.
    assert classes["basic"][0] == pytest.approx(0.20 * 37_700.0)
    assert classes["higher"][0] == pytest.approx(0.40 * 12_300.0)
    assert classes["additional"][0] == 0.0


def test_relief_takes_each_persons_own_schedule() -> None:
    # The same GBP 4,000 returned on GBP 29,430 of taxable income: the rest of
    # the UK relieves it at 20%, Scotland at 21% to 31,092 and 42% above.
    relief = relief_by_rate_class(
        baseline_income=np.array([29_430.0, 29_430.0]),
        counterfactual_income=np.array([33_430.0, 33_430.0]),
        scottish=np.array([False, True]),
        schedules={"rest_of_uk": REST_OF_UK, "scotland": SCOTLAND},
    )

    assert relief["basic"].tolist() == pytest.approx([800.0, 0.21 * 1_662.0])
    assert relief["higher"].tolist() == pytest.approx([0.0, 0.42 * 2_338.0])
    assert relief["additional"].tolist() == [0.0, 0.0]


def test_a_schedule_without_an_additional_rate_files_its_top_rate_as_higher() -> None:
    # The top rate is the additional rate only where a higher rate sits below
    # it. With one rate above the basic class, that rate is the higher rate.
    income = np.array([20_000.0, 60_000.0])

    classes = tax_by_rate_class(income, [0.0, 37_700.0], [0.20, 0.40])

    assert classes["basic"].tolist() == pytest.approx([4_000.0, 7_540.0])
    assert classes["higher"].tolist() == pytest.approx([0.0, 8_920.0])
    assert classes["additional"].tolist() == [0.0, 0.0]


def test_the_all_rates_total_is_a_diagnostic_beside_the_higher_rate_band() -> None:
    from microcosm.build.uk_runtime.ledger_targets import (
        UK_REQUIRED_TARGET_DIAGNOSTICS,
    )
    from microcosm.build.uk_runtime.local_targets import load_uk_population_contract

    contract = load_uk_population_contract()
    target_ids = {target["target_id"] for target in contract["targets"]}

    assert "hmrc.salary_sacrifice.it_relief_total" not in target_ids
    for rate_class in UK_RATE_CLASSES:
        assert f"hmrc.salary_sacrifice.it_relief_{rate_class}_rate" in target_ids
    declaration = contract["diagnostic_references"][
        "hmrc.salary_sacrifice.it_relief_total"
    ]
    assert declaration["attach_to_target"] == (
        "hmrc.salary_sacrifice.it_relief_higher_rate"
    )
    assert UK_REQUIRED_TARGET_DIAGNOSTICS[
        "hmrc.salary_sacrifice.it_relief_higher_rate"
    ] == ("hmrc.salary_sacrifice.it_relief_total",)


def test_the_all_rates_total_diagnostic_carries_the_2024_25_relief() -> None:
    """Feed-gated: the diagnostic resolves its exact tax-year 2024-25 fact.

    A selector or fact-key mismatch reads as ``unavailable`` and does not fail
    the compile, so this test holds the declaration to the pinned feed.
    """

    from microcosm.build.uk_runtime.ledger_targets import (
        _attached_diagnostic_metadata,
    )

    root = paths_for("microcosm-build").repository
    configured = os.environ.get("CHRONICLE_UK_FACTS")
    feed = (
        Path(configured) if configured else root / ".codex-work/consumer_facts_uk.jsonl"
    )
    if feed.is_dir():
        feed = feed / "consumer_facts.jsonl"
    if not feed.is_file():
        pytest.skip("pinned UK Chronicle consumer feed is not present")
    with feed.open(encoding="utf-8") as handle:
        facts = tuple(
            json.loads(line)
            for line in handle
            if '"hmrc.ss_income_tax_relief_total"' in line
        )

    metadata = _attached_diagnostic_metadata(
        facts, "hmrc.salary_sacrifice.it_relief_higher_rate"
    )

    prefix = "salary_sacrifice_it_relief_total_diagnostic"
    assert metadata[f"{prefix}_role"] == "diagnostic_only_not_in_fit"
    assert metadata[f"{prefix}_status"] == "available"
    assert float(metadata[f"{prefix}_value_gbp"]) == pytest.approx(8.8e9)
    assert metadata[f"{prefix}_period"] == "2024"
