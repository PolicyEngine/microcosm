"""Tests split from packages/microcosm-build/tests/test_us_survivor_benefits_exclusion.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_survivor_benefits_exclusion import *


def test_sha_locked_artifact_schemas_match_the_recorded_absence() -> None:
    from policyengine_us.data import USSingleYearDataset

    evidence = _entry()["evidence"]
    build_summary = json.loads(
        (ROOT / "experiments/build_j_recert/base_j.summary.json").read_text()
    )
    paths = {
        Path(item["path"]).name: Path(item["path"])
        for item in build_summary["base_source"]["sources"]
    }
    if not all(path.is_file() for path in paths.values()):
        pytest.skip("SHA-locked ASEC artifacts are not mounted in this environment")

    for item in evidence["hermetic_inputs"]:
        path = paths[item["filename"]]
        assert _sha256(path) == item["sha256"]
        dataset = USSingleYearDataset(file_path=str(path))
        assert set(item["missing_person_columns"]).isdisjoint(dataset.person.columns)
        assert set(item["present_family_columns"]) <= set(dataset.family.columns)
        assert set(item["present_household_columns"]) <= set(dataset.household.columns)
        positive_multi_person_families = int(
            ((dataset.family["FSURVAL"] > 0) & (dataset.family["FPERSONS"] > 1)).sum()
        )
        assert positive_multi_person_families == item["positive_multi_person_families"]
        family_amount_by_household = dataset.family.groupby("FH_SEQ", sort=False)[
            "FSURVAL"
        ].sum()
        household_amount = dataset.household.set_index("H_SEQ")["HSURVAL"]
        np.testing.assert_array_equal(
            family_amount_by_household.reindex(
                household_amount.index,
                fill_value=0,
            ).to_numpy(),
            household_amount.to_numpy(),
        )


def test_policyengine_2_2_1_requires_a_person_year_input() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    assert version("policyengine-us") == "2.2.1"
    variable = CountryTaxBenefitSystem().variables["survivor_benefits"]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert str(variable.definition_period).lower() == "year"
    assert variable.documentation == (
        "Survivor benefits other than Social Security survivor benefits."
    )
