"""Tests split from packages/microcosm-build/tests/test_us_dc_ptc_take_up_exclusion.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_dc_ptc_take_up_exclusion import *


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
    puf_path = Path(build_summary["puf_h5"])
    if not all(path.is_file() for path in [*paths.values(), puf_path]):
        pytest.skip("SHA-locked ASEC/PUF artifacts are not mounted")

    entity_names = ("person", "tax_unit", "spm_unit", "family", "household")
    for item in evidence["hermetic_asec_inputs"]:
        path = paths[item["filename"]]
        assert _sha256(path) == item["sha256"]
        dataset = USSingleYearDataset(file_path=str(path))
        all_columns = {
            str(column)
            for entity in entity_names
            for column in getattr(dataset, entity).columns
        }
        assert set(item["missing_columns"]).isdisjoint(all_columns)
        assert set(item["present_generic_tax_amount_columns"]) <= all_columns
        lower_columns = {column.lower() for column in all_columns}
        assert not any(
            pattern in column
            for pattern in evidence["missing_claim_column_patterns"]
            for column in lower_columns
        )

    assert _sha256(puf_path) == evidence["processed_puf"]["sha256"]
    with h5py.File(puf_path, mode="r") as puf:
        assert set(evidence["processed_puf"]["missing_arrays"]).isdisjoint(puf.keys())
        assert "other_credits" in puf
        lower_arrays = {str(name).lower() for name in puf.keys()}
        assert not any(
            pattern in name
            for pattern in evidence["missing_claim_column_patterns"]
            for name in lower_arrays
        )


def test_policyengine_2_2_1_requires_a_tax_unit_year_boolean() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    assert version("policyengine-us") == "2.2.1"
    variable = CountryTaxBenefitSystem().variables["takes_up_dc_ptc"]
    assert variable.is_input_variable()
    assert variable.entity.key == "tax_unit"
    assert str(variable.definition_period).lower() == "year"
    assert variable.value_type is bool
    assert bool(variable.default_value) is True
