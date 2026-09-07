"""Invented source bytes, real tiny microunit, and preserved attachment parents."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build import frame_checkpoint as checkpoint
from microcosm.build.us_runtime import asec_current_money_source as money_source
from microcosm.build.us_runtime import asec_current_money_units as units
from microcosm.build.us_runtime import asec_housing_status_source as housing
from microcosm.build.us_runtime import asec_housing_universe as hu
from microcosm.build.us_runtime import asec_housing_universe_source as source
from microcosm.build.us_runtime import asec_person_income_source as restoration
from microcosm.build.us_runtime import asec_student_controls as student
from microcosm.frame import MassChangeRecord


def _module(filename):
    spec = importlib.util.spec_from_file_location(
        "hu_invented_" + filename[:-3], Path(__file__).with_name(filename)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path, monkeypatch, *, gq=False, unknown_tenure=False):
    import microunit  # noqa: F401 - missing real dependency must error, never skip
    import tables  # noqa: F401

    fixture = _module("test_us_asec_current_money_source.py")
    real_frame = fixture.Frame
    tenures = [0 if unknown_tenure else 1, 2, 3]

    def with_tenure_and_history(entity_tables, *args, **kwargs):
        entity_tables["household"]["H_TENURE"] = np.array(tenures, dtype="int64")
        return real_frame(
            entity_tables,
            *args,
            **kwargs,
            mass_log=(
                MassChangeRecord(
                    "household", 3.0, 6.0, 2.0, "Invented HU fixture history"
                ),
            ),
            metadata={"invented": {"history": ("preserve",)}},
        )

    fixture.Frame = with_tenure_and_history
    writer = pd.DataFrame.to_hdf

    def write_cohort(table, path, *args, **kwargs):
        if Path(path).name.startswith("invented-cohort-"):
            year = int(Path(path).stem.rsplit("-", 1)[1])
            table = table.assign(
                H_TENURE=tenures[year - 2022],
                H_YEAR=year + 1,
                HPUBLIC=2,
                HLORENT=2,
                I_HPUBLI=0,
                I_HLOREN=0,
            )
            if gq and year == 2024:
                table = table.assign(H_LIVQRT=10, HRHTYPE=10)
        return writer(table, path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pd.DataFrame, "to_hdf", write_cohort)
        p, h, *_ = fixture.invented_checkpoints(tmp_path, monkeypatch)
    loaded = money_source.load_authenticated_current_money_source(p, h)
    return loaded, {
        year: tmp_path / f"invented-cohort-{year}.h5" for year in (2022, 2023, 2024)
    }


@pytest.mark.parametrize("after_housing", [False, True])
def test_three_hu_native_tenure_and_both_exact_parent_types(
    tmp_path, monkeypatch, after_housing
):
    loaded, paths = _fixture(tmp_path, monkeypatch)
    before = money_source._frame_signature(loaded.frame)
    universe = source.load_authenticated_housing_universe(loaded)
    assert universe.array("household_id").tolist() == [1, 2, 3]
    assert universe.array("income_year").tolist() == [2022, 2023, 2024]
    assert universe.array("H_SEQ").tolist() == [7, 7, 7]
    assert universe.array("occupied_hu").tolist() == [1, 1, 1]
    assert universe.array("hu_tenure_class").tolist() == [1, 2, 2]
    assert universe.array("tenure_subtype").tolist() == [1, 2, 3]
    assert universe.header_data["source"]["identity"].encode() == loaded.source.identity
    ready = loaded.ready()
    money_before = (ready.header, ready.fields)
    tax = units.reconstruct_current_money_tax_units(loaded, ready)
    tax_before = tax.receipt
    parent = (
        housing.attach_housing_status(
            tax, housing.load_authenticated_housing_status(loaded, cohort_paths=paths)
        )
        if after_housing
        else tax
    )
    parent_before = parent.receipt
    result = source.attach_housing_universe(parent, universe)
    result.validate()
    assert (
        result.parent is parent and result.tax_result is tax and result.money is ready
    )
    assert result.frame.n("person") == 6
    assert result.frame.mass_log == parent.frame.mass_log
    assert result.frame.metadata == parent.frame.metadata
    assert result.receipt["parent_kind"] == type(parent).__name__
    assert result.receipt["parent_receipt_sha256"] == hu._sha(hu._json(parent_before))
    assert parent.receipt == parent_before and tax.receipt == tax_before
    assert (ready.header, ready.fields) == money_before
    assert money_source._frame_signature(loaded.frame) == before
    if after_housing:
        pd.testing.assert_frame_equal(
            result.frame.table("household")[list(housing.ATTACHED_COLUMNS)],
            parent.frame.table("household")[list(housing.ATTACHED_COLUMNS)],
            check_exact=True,
        )


def test_student_tax_housing_hu_retains_exact_money_missingness_and_aliases(
    tmp_path, monkeypatch
):
    fixture = _module("test_us_asec_student_controls.py")
    loaded, ready, members, _ = fixture._fixture(tmp_path, monkeypatch)
    observation = fixture._load(loaded, ready, members)
    combined = student.attach_student_controls(loaded, ready, observation)
    tax = units.reconstruct_current_money_tax_units(combined, ready)
    status = housing.load_authenticated_housing_status(
        loaded,
        cohort_paths={
            year: tmp_path / f"invented-cohort-{year}.h5" for year in (2022, 2023, 2024)
        },
    )
    parent = housing.attach_housing_status(tax, status)
    before = (parent.receipt, tax.receipt, ready.header, ready.fields)
    universe = source.load_authenticated_housing_universe(loaded)
    result = source.attach_housing_universe(parent, universe)
    result.validate()
    assert result.tax_result._student_source is combined
    assert result.money is ready
    assert result.frame.person.A_ENRLW.isna().tolist() == [True] * 4 + [False] * 2
    assert result.frame.person.A_FTPT.isna().tolist() == [True] * 4 + [False] * 2
    for column in parent.frame.person:
        pd.testing.assert_series_equal(
            result.frame.person[column], parent.frame.person[column], check_exact=True
        )
    assert (parent.receipt, tax.receipt, ready.header, ready.fields) == before


def test_gq_persons_and_unknown_tenure_survive_selection_and_checkpoint(
    tmp_path, monkeypatch
):
    loaded, _ = _fixture(tmp_path, monkeypatch, gq=True, unknown_tenure=True)
    universe = source.load_authenticated_housing_universe(loaded)
    tax = units.reconstruct_current_money_tax_units(loaded, loaded.ready())
    result = source.attach_housing_universe(tax, universe)
    assert universe.array("occupied_hu").tolist() == [1, 1, 2]
    assert universe.array("hu_tenure_class").tolist() == [3, 2, 0]
    assert result.frame.n("person") == loaded.frame.n("person") == 6
    subset = result.frame.select(np.array([True, True, False, False, True, True]))
    assert subset.person.person_id.tolist() == [101, 102, 105, 106]
    source.verify_housing_universe_rows(
        subset.table("household"), universe, lineage_ids=np.array([1, 3], dtype="int64")
    )
    with pytest.raises(hu.HousingUniverseRefusalError):
        source.validate_attached_frame(subset, parent=tax, universe=universe)
    full_path, subset_path = tmp_path / "full.h5", tmp_path / "subset.h5"
    checkpoint.write_frame_checkpoint(full_path, result.frame)
    checkpoint.write_frame_checkpoint(subset_path, subset)
    full_read = checkpoint.load_frame_checkpoint(full_path).frame
    selected_read = checkpoint.load_frame_checkpoint(subset_path).frame
    source.validate_attached_frame(full_read, parent=tax, universe=universe)
    source.verify_housing_universe_rows(
        selected_read.table("household"),
        universe,
        lineage_ids=np.array([1, 3], dtype="int64"),
    )
    assert selected_read.table(
        "household"
    ).asec_housing_universe_unresolved_reasons.tolist() == [64, 0]


@pytest.mark.parametrize(
    "mutation",
    ["derived", "raw", "missing_lineage", "nan", "float", "false_lineage_column"],
)
def test_row_evidence_clone_verification_refuses_loss_and_false_lineage(
    tmp_path, monkeypatch, mutation
):
    loaded, _ = _fixture(tmp_path, monkeypatch)
    universe = source.load_authenticated_housing_universe(loaded)
    result = source.attach_housing_universe(
        units.reconstruct_current_money_tax_units(loaded, loaded.ready()), universe
    )
    clone = result.frame.table("household").iloc[[2, 0, 2]].copy(deep=True)
    clone["household_id"] = [10, 11, 12]
    lineage = np.array([3, 1, 3], dtype="int64")
    clone["household_source_id"] = lineage
    source.verify_housing_universe_rows(clone, universe, lineage_ids=lineage)
    col = "asec_housing_universe_hu_tenure_class"
    if mutation == "derived":
        clone.iloc[0, clone.columns.get_loc(col)] = 1
    elif mutation == "raw":
        clone.iloc[0, clone.columns.get_loc("asec_H_TENURE")] = 1
    elif mutation == "missing_lineage":
        lineage[0] = 999
    elif mutation in ("nan", "float"):
        clone[col] = clone[col].astype(float)
        if mutation == "nan":
            clone.iloc[0, clone.columns.get_loc(col)] = np.nan
    else:
        clone.iloc[0, clone.columns.get_loc("household_source_id")] = 2
    with pytest.raises(hu.HousingUniverseRefusalError):
        source.verify_housing_universe_rows(clone, universe, lineage_ids=lineage)
    result.validate()


@pytest.mark.parametrize(
    "mutation",
    [
        "derived",
        "person",
        "weights",
        "person_index",
        "range_index",
        "household_columns",
    ],
)
def test_owned_composed_result_mutation_never_changes_tax_housing_parents(
    tmp_path, monkeypatch, mutation
):
    loaded, paths = _fixture(tmp_path, monkeypatch)
    universe = source.load_authenticated_housing_universe(loaded)
    tax = units.reconstruct_current_money_tax_units(loaded, loaded.ready())
    parent = housing.attach_housing_status(
        tax, housing.load_authenticated_housing_status(loaded, cohort_paths=paths)
    )
    before = (
        money_source._frame_signature(parent.frame),
        money_source._frame_signature(tax.frame),
        money_source._frame_signature(loaded.frame),
    )
    result = source.attach_housing_universe(parent, universe)
    if mutation == "derived":
        result.frame.table("household").iloc[
            0,
            result.frame.table("household").columns.get_loc(source.ATTACHED_COLUMNS[0]),
        ] = 0
    elif mutation == "person":
        result.frame.person.iloc[
            0, result.frame.person.columns.get_loc("unrelated_raw_observation")
        ] += 1
    elif mutation == "weights":
        values = result.frame.weights_for("household").values
        values.setflags(write=True)
        values[0] += 1
    elif mutation == "person_index":
        result.frame.person.index.array[0] = 999
    elif mutation == "range_index":
        assert type(result.frame.table("household").index) is pd.RangeIndex
        result.frame.table("household").index.array[0] = 999
    else:
        result.frame.table("household").columns.array[0] = "renamed_household_id"
    assert before == (
        money_source._frame_signature(parent.frame),
        money_source._frame_signature(tax.frame),
        money_source._frame_signature(loaded.frame),
    )
    parent.validate()
    with pytest.raises(hu.HousingUniverseRefusalError):
        result.validate()


def test_source_and_composition_authority_cannot_be_borrowed(tmp_path, monkeypatch):
    loaded, _ = _fixture(tmp_path, monkeypatch)
    universe = source.load_authenticated_housing_universe(loaded)
    synthetic = hu.classify_synthetic_housing_universe(
        pd.DataFrame({name: universe.array(name) for name in hu.RAW_COLUMNS})
    )
    tax = units.reconstruct_current_money_tax_units(loaded, loaded.ready())
    with pytest.raises(hu.HousingUniverseRefusalError, match="AUTHENTICATED_UNIVERSE"):
        source.attach_housing_universe(tax, synthetic)
    with pytest.raises(hu.HousingUniverseRefusalError, match="PARENT_REQUIRED"):
        source.attach_housing_universe(loaded, universe)
    with pytest.raises(hu.HousingUniverseRefusalError, match="AUTHENTICATED_SOURCE"):
        source.load_authenticated_housing_universe(tax)
    loaded.frame.table("household").loc[0, "asec_H_LIVQRT"] = 8
    with pytest.raises(hu.HousingUniverseRefusalError):
        source.load_authenticated_housing_universe(loaded)


def test_cross_source_kind_and_expected_replay_are_bound(tmp_path, monkeypatch):
    fixture = _module("test_us_asec_person_income_source.py")
    p, h, members = fixture.invented_sources(
        tmp_path, monkeypatch, missing=False, housing=True
    )
    output = tmp_path / "T1"
    restoration.restore_asec_person_income_source(
        p, h, member_paths=members, output_dir=output
    )
    restored = restoration.load_authenticated_restored_current_money_source(
        p, h, output / restoration.CHECKPOINT_FILENAME, member_paths=members
    )
    # Both source kinds have the same live P/H pins and implementation closure.
    # A wrong attachment must fail source binding, not stale implementation.
    legacy = money_source.load_authenticated_current_money_source(p, h)
    universe = source.load_authenticated_housing_universe(legacy)
    other = source.load_authenticated_housing_universe(restored)
    tax = units.reconstruct_current_money_tax_units(legacy, legacy.ready())
    other_tax = units.reconstruct_current_money_tax_units(restored, restored.ready())
    legacy.validate()
    restored.validate()
    universe.validate()
    other.validate()
    source.attach_housing_universe(tax, universe).validate()
    source.attach_housing_universe(other_tax, other).validate()
    for wrong_tax, wrong_universe in ((tax, other), (other_tax, universe)):
        with pytest.raises(
            hu.HousingUniverseRefusalError, match="UNIVERSE_SOURCE_BINDING"
        ):
            source.attach_housing_universe(wrong_tax, wrong_universe)
    with pytest.raises(hu.HousingUniverseRefusalError, match="EXPECTED_CONTENT"):
        hu.decode_housing_universe(hu.encode_housing_universe(universe), expected=other)
    path = tmp_path / "observed.hu"
    hu.write_housing_universe(universe, path)
    assert (
        hu.read_housing_universe(path, expected=universe).content_sha256
        == universe.content_sha256
    )


@pytest.mark.parametrize("after_housing", [False, True])
def test_invalid_exact_parent_refuses_before_attachment(
    tmp_path, monkeypatch, after_housing
):
    loaded, paths = _fixture(tmp_path, monkeypatch)
    universe = source.load_authenticated_housing_universe(loaded)
    tax = units.reconstruct_current_money_tax_units(loaded, loaded.ready())
    parent = (
        housing.attach_housing_status(
            tax, housing.load_authenticated_housing_status(loaded, cohort_paths=paths)
        )
        if after_housing
        else tax
    )
    parent.frame.table("family")["unreviewed_change"] = 1
    with pytest.raises(hu.HousingUniverseRefusalError):
        source.attach_housing_universe(parent, universe)
    assert not set(source.ATTACHED_COLUMNS) & set(parent.frame.table("household"))


def test_changed_classifier_identity_invalidates_issued_source_evidence(
    tmp_path, monkeypatch
):
    loaded, _ = _fixture(tmp_path, monkeypatch)
    universe = source.load_authenticated_housing_universe(loaded)
    monkeypatch.setattr(
        source, "_implementation", lambda: {"different": "implementation"}
    )
    with pytest.raises(hu.HousingUniverseRefusalError, match="IMPLEMENTATION_CHANGED"):
        universe.validate()
