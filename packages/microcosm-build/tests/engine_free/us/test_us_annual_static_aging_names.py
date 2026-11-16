"""The names an annual candidate gives its H5 files (microcosm#1570).

Each projected year's H5 is ``microcosm_us_<year>``, and the base year keeps
the key of the base national build's dataset, which is also the family's
key. Nothing here builds a candidate.
"""

from __future__ import annotations

import pytest

from microcosm.build import us_annual_static_aging as annual


def test_projected_years_take_microcosm_names_and_the_base_year_a_new_build_s():
    assert annual.annual_dataset_names(2024, 2026) == {
        2024: "microcosm_us_2024",
        2025: "microcosm_us_2025",
        2026: "microcosm_us_2026",
    }


def test_the_base_year_keeps_its_base_build_s_dataset_key():
    """A build the switch moved keeps its ``populace_us_2024`` key."""
    assert annual.annual_dataset_names(2024, 2025, base_dataset="populace_us_2024") == {
        2024: "populace_us_2024",
        2025: "microcosm_us_2025",
    }


@pytest.mark.parametrize(
    "base_dataset", ["", "../populace_us_2024", "Populace", "microcosm_us_2025"]
)
def test_the_base_dataset_is_a_dataset_key_no_projected_year_has(base_dataset):
    with pytest.raises(ValueError, match="base_dataset"):
        annual.annual_dataset_names(2024, 2026, base_dataset=base_dataset)


def test_the_command_takes_the_base_dataset():
    required = [
        "--base-h5=base.h5",
        f"--base-sha256={'a' * 64}",
        "--parent-release=populace-us-2024-9f1260b-20260611",
        "--ssa-csv=ssa.csv",
        f"--ssa-sha256={'b' * 64}",
        "--model-source=model",
        f"--model-commit={'c' * 40}",
        "--model-version=1.0.0",
        "--output-dir=candidate",
    ]
    parser = annual.command_parser()

    assert parser.parse_args(required).base_dataset is None
    assert (
        parser.parse_args([*required, "--base-dataset", "populace_us_2024"])
    ).base_dataset == "populace_us_2024"
