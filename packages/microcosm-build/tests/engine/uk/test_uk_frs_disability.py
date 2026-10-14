"""Tests split from packages/microcosm-build/tests/test_uk_frs_disability.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_frs_disability import *


def test_dwp_readers_share_fiscal_converted_tree() -> None:
    # The readers construct the real engine's parameter tree; the wheel gate
    # and the us-extra CI lane run without policyengine-uk, so skip there.
    from microcosm.build.uk_runtime.frs_disability import (
        uk_dwp_disability_flag_rates,
    )

    category_2024 = uk_dwp_disability_category_rates(2024)
    flags_2024 = uk_dwp_disability_flag_rates(2024)
    category_2023 = uk_dwp_disability_category_rates(2023)
    flags_2023 = uk_dwp_disability_flag_rates(2023)

    assert category_2024.instant == flags_2024.instant == "2024-01-01"
    assert np.isfinite(category_2024.aa_lower)
    assert category_2024.aa_higher == flags_2024.aa_higher == pytest.approx(108.55)
    assert category_2023.aa_higher == flags_2023.aa_higher == pytest.approx(101.75)


def test_stored_severe_flag_matches_the_policyengine_uk_formula() -> None:
    """On every category, the stored flag is the engine's own formula (pe-uk#1946)."""

    import policyengine_uk

    from microcosm.build.uk_runtime.frs_disability import (
        uk_dwp_disability_flag_rates,
    )

    year = 2024
    rates = uk_dwp_disability_category_rates(year)
    flags = uk_dwp_disability_flag_rates(year)
    cases = [
        ("attendance_allowance_reported", "aa_category", "NONE", 0.0),
        ("attendance_allowance_reported", "aa_category", "LOWER", rates.aa_lower),
        ("attendance_allowance_reported", "aa_category", "HIGHER", rates.aa_higher),
        ("dla_sc_reported", "dla_sc_category", "LOWER", rates.dla_sc_lower),
        ("dla_sc_reported", "dla_sc_category", "MIDDLE", rates.dla_sc_middle),
        ("dla_sc_reported", "dla_sc_category", "HIGHER", rates.dla_sc_higher),
        ("pip_dl_reported", "pip_dl_category", "STANDARD", rates.pip_dl_standard),
        ("pip_dl_reported", "pip_dl_category", "ENHANCED", rates.pip_dl_enhanced),
    ]
    rows = []
    for column, _, _, weekly in cases:
        row = dict.fromkeys(
            (
                "attendance_allowance_reported",
                "dla_sc_reported",
                "dla_m_reported",
                "pip_m_reported",
                "pip_dl_reported",
            ),
            0.0,
        )
        row[column] = weekly * WEEKS_IN_YEAR
        rows.append(row)
    stored = derive_frs_disability(
        pd.DataFrame(rows), category_rates=rates, flag_rates=flags
    )
    # The engine reads the same categories the spine stores.
    assert [
        stored.loc[index, category_column]
        for index, (_, category_column, _, _) in enumerate(cases)
    ] == [category for _, _, category, _ in cases]

    people = {}
    for index, (_, category_column, category, _) in enumerate(cases):
        person = {"age": {year: 70 if category_column == "aa_category" else 40}}
        person[category_column] = {year: category}
        people[f"p{index}"] = person
    simulation = policyengine_uk.Simulation(
        situation={
            "people": people,
            "benunits": {
                f"b{index}": {"members": [name]} for index, name in enumerate(people)
            },
            "households": {
                f"h{index}": {"members": [name]} for index, name in enumerate(people)
            },
        }
    )
    engine = np.asarray(
        simulation.calculate("is_severely_disabled_for_benefits", year), dtype=bool
    )

    assert stored["is_severely_disabled_for_benefits"].tolist() == engine.tolist()
