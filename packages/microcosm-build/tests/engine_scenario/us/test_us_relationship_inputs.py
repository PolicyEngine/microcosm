"""Tests split from packages/microcosm-build/tests/test_us_relationship_inputs.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_relationship_inputs import *


class TestCoverageAndExclusion:
    def test_restored_inputs_are_hard_release_requirements(self) -> None:
        manifest = load_release_input_coverage_manifest()
        assert set(_OUTPUTS) <= RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS
        assert set(_OUTPUTS) <= set(US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS)
        assert set(_OUTPUTS) <= manifest.required_columns
        assert set(_OUTPUTS).isdisjoint(manifest.reviewed_exclusions)

    def test_unmarried_partner_exclusion_is_source_unavailability(self) -> None:
        entry = _known_gap(_UNMARRIED_PARTNER)
        evidence = entry["evidence"]

        assert entry["reason"].startswith("SOURCE UNAVAILABILITY WITH EVIDENCE:")
        assert evidence["classification"] == "source_unavailability"
        assert evidence["retired_derivation"] == {
            "repository_owner": "PolicyEngine",
            "repository_name_parts": ["policyengine-", "us-data"],
            "commit": "42ed5d45c56df80d754fbe24cce21cfeb8d05cbe",
            "path_parts": [
                "policyengine_",
                "us_data",
                "datasets",
                "cps",
                "cps.py",
            ],
            "lines": "1214-1221",
        }
        assert evidence["required_person_columns"] == ["PERRP"]
        assert all(
            "PERRP" in item["missing_person_columns"]
            for item in evidence["hermetic_inputs"]
        )
        assert evidence["semantic_non_substitutes"]["rejection"].startswith(
            "Using only the 2024"
        )

        manifest = load_release_input_coverage_manifest()
        assert manifest.reviewed_exclusions[_UNMARRIED_PARTNER] == entry["reason"]

    def test_locked_artifacts_confirm_source_presence_and_absence(self) -> None:
        summary = json.loads(
            (ROOT / "experiments/build_j_recert/base_j.summary.json").read_text()
        )
        paths = {
            Path(item["path"]).name: Path(item["path"])
            for item in summary["base_source"]["sources"]
        }
        if not all(path.is_file() for path in paths.values()):
            pytest.skip("SHA-locked ASEC artifacts are not mounted")

        evidence = _known_gap(_UNMARRIED_PARTNER)["evidence"]
        expected_heads = {2022: 56_839, 2023: 56_251, 2024: 55_762}
        for item in evidence["hermetic_inputs"]:
            path = paths[item["filename"]]
            assert _sha256(path) == item["sha256"]
            with pd.HDFStore(path, mode="r") as store:
                person = store["person"]
            assert set(item["missing_person_columns"]).isdisjoint(person.columns)
            assert {"PH_SEQ", "P_SEQ", "A_MARITL"} <= set(person.columns)
            year = int(item["filename"].split("_")[-1].split(".")[0])
            assert int((person["P_SEQ"] == 1).sum()) == expected_heads[year]
            assert (
                person.groupby("PH_SEQ")["P_SEQ"].apply(lambda x: (x == 1).sum()) == 1
            ).all()
            if year < 2024:
                recoded, source = _with_relationship_recode(person)
                assert source == "derived:line_spouse_parent"
                assert int((recoded["A_EXPRRP"] == 13).sum()) == 0
            else:
                assert (
                    int((person["PECOHAB"] > 0).sum()) == item["positive_pecohab_rows"]
                )
                assert (
                    int((person["A_EXPRRP"] == 13).sum())
                    == item["a_exprrp_partner_or_roommate_rows"]
                )


def test_policyengine_2_2_1_contract_and_live_bindings() -> None:
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    assert version("policyengine-us") == "2.2.1"
    variables = CountryTaxBenefitSystem().variables
    for name in _OUTPUTS:
        variable = variables[name]
        assert variable.is_input_variable()
        assert variable.entity.key == "person"
        assert variable.value_type is bool
        assert variable.default_value is False
    assert str(variables[_HEAD].definition_period).lower() == "eternity"
    assert str(variables[_SEPARATED].definition_period).lower() == "year"
    assert str(variables[_SURVIVING].definition_period).lower() == "year"

    common = {
        "people": {
            "adult": {
                "age": {"2024": 40},
                "employment_income": {"2024": 50_000},
                "is_tax_unit_head": {"2024": True},
                _SURVIVING: {"2024": True},
            },
            "child": {
                "age": {"2024": 5},
                "is_tax_unit_dependent": {"2024": True},
            },
        },
        "tax_units": {"tax_unit": {"members": ["adult", "child"]}},
        "spm_units": {"spm_unit": {"members": ["adult", "child"]}},
        "households": {
            "household": {
                "members": ["adult", "child"],
                "state_code": {"2024": "CA"},
            }
        },
    }
    assert (
        Simulation(situation=common).calculate("filing_status", 2024).decode()[0].value
        == "Surviving spouse"
    )

    separated = json.loads(json.dumps(common))
    separated["people"]["adult"].pop(_SURVIVING)
    separated["people"]["adult"][_SEPARATED] = {"2024": True}
    assert (
        Simulation(situation=separated)
        .calculate("filing_status", 2024)
        .decode()[0]
        .value
        == "Head of household"
    )
