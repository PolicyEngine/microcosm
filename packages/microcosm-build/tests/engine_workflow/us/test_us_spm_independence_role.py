"""Tests split from packages/microcosm-build/tests/test_us_spm_independence_role.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_spm_independence_role import *


class TestCoverageAndExport:
    def test_role_is_a_hard_release_requirement(self) -> None:
        manifest = load_release_input_coverage_manifest()
        assert _ROLE in POST_REFERENCE_ECPS_REQUIRED_INPUTS
        assert _ROLE in manifest.required_columns
        assert _ROLE not in manifest.reviewed_exclusions
        assert _ROLE in US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS

    def test_engine_adapter_persists_the_role_as_an_input_leaf(
        self, population, tmp_path: Path
    ) -> None:
        """Both adapter paths classify the role by the engine's own declaration."""

        import policyengine_us.spm as spm
        from policyengine_us.data import USSingleYearDataset

        from microcosm.frame.adapters.policyengine_us import (
            PolicyEngineUSEngine,
            PolicyEngineUSVariableMetadataIndex,
        )

        assert spm.DATASET_SOURCE_INPUTS == frozenset({_ROLE})
        engine = PolicyEngineUSEngine()
        index = PolicyEngineUSVariableMetadataIndex()
        assert _ROLE in engine.variables()
        assert _ROLE in index.variables()
        assert engine.formula_owned_outputs([_ROLE, "spm_measurement_adults"]) == {
            "spm_measurement_adults"
        }
        assert index.formula_owned_outputs([_ROLE, "spm_measurement_adults"]) == {
            "spm_measurement_adults"
        }
        assert engine.default_values([_ROLE]) == {}

        _source, path, pin, frame = population
        result = _run(frame, path, pin)
        export = result.table("person")[
            [
                "person_id",
                "person_household_id",
                "person_tax_unit_id",
                "person_spm_unit_id",
                "person_family_id",
                "person_marital_unit_id",
                "age",
                _ROLE,
            ]
        ]
        tables = {entity: result.table(entity) for entity in result.entities}
        tables["person"] = export
        exportable = Frame(
            tables,
            US_SCHEMA,
            {"household": result.weights_for("household")},
        )
        destination = tmp_path / "role.h5"
        engine.write_dataset(exportable, destination, period=_INCOME_YEAR)
        written = USSingleYearDataset(file_path=str(destination)).person
        assert written[_ROLE].tolist() == export[_ROLE].tolist()
