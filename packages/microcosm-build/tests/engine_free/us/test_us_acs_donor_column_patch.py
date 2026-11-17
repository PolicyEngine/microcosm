"""The proof-of-concept donor patch for the ACS staging (microcosm#922).

``tools/patch_us_acs_donor_columns.py`` writes a patched copy of a US
national donor H5 with reviewed all-zero columns added. These tests use a
small fixture H5 in the nullable layout R1's ``_load_base_frame`` reads and a
stand-in engine, so no country engine is imported.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.acs_local_income import (
    ACS_LOCAL_ALIGNED_RETIREMENT_DONOR_COMPONENTS,
    ACS_LOCAL_INCOME_TRANSFER_COLUMNS,
    ACS_LOCAL_OTHER_INCOME_DONOR_COMPONENTS,
)
from microcosm.build.us_runtime.h5_io import (
    read_nullable_us_h5_metadata,
    write_nullable_us_h5,
)
from microcosm.build.us_runtime.support_provenance import (
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.frame import (
    US_SCHEMA,
    Frame,
    VariableMetadata,
    WeightKind,
    Weights,
    read_frame_table,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
_ROOT_ATTRIBUTES = {
    "populace_release_id": "fixture-release",
    "populace_location_rule": "legacy",
}
_ZERO_COLUMNS = ("strike_benefits", "passive_partnership_s_corp_income")


def _load_tool(relative: str, name: str):
    path = _TEST_PATHS.repository / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Dataclasses resolve their annotations through sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return _load_tool("tools/patch_us_acs_donor_columns.py", "patch_us_acs_donor")


@pytest.fixture(scope="module")
def builder():
    return _load_tool(
        "tools/_legacy/build_us_acs_multispine_base.py",
        "legacy_build_us_acs_multispine_base_for_patch",
    )


class _Engine:
    """Stand-in for the US engine's variable metadata and defaults."""

    def __init__(
        self,
        *,
        entities: dict[str, str] | None = None,
        defaults: dict[str, object] | None = None,
        dtype: str = "float",
    ) -> None:
        self.entities = entities or {}
        self.defaults = {name: 0.0 for name in _ZERO_COLUMNS} | (defaults or {})
        self.dtype = dtype

    def variable_metadata(self, name: str) -> VariableMetadata:
        return VariableMetadata(
            name=name,
            entity=self.entities.get(name, "person"),
            dtype=self.dtype,
            period="year",
        )

    def default_values(self, names):
        return {name: self.defaults[name] for name in names if name in self.defaults}


class _NoDefaults(_Engine):
    def default_values(self, names):
        return {}


def _donor_frame(person_extra: dict[str, object] | None = None) -> Frame:
    """Two ASEC persons and their two PUF clones, one household each.

    The person table carries every column R1's income-donor gate reads,
    except ``strike_benefits``, plus a nullable boolean with a null and a
    text column, so the tests see dtypes kept through the copy.
    """

    n = 4
    ids = np.arange(1, n + 1)
    person = pd.DataFrame(
        {
            "person_id": ids,
            **{f"person_{entity}_id": ids for entity in US_SCHEMA.group_entities},
            "age": [34.0, 70.0, 34.0, 70.0],
            "is_female": [False, True, False, True],
        }
    )
    amounts = (
        *ACS_LOCAL_INCOME_TRANSFER_COLUMNS,
        *ACS_LOCAL_OTHER_INCOME_DONOR_COMPONENTS,
        *ACS_LOCAL_ALIGNED_RETIREMENT_DONOR_COMPONENTS,
    )
    for offset, column in enumerate(dict.fromkeys(amounts)):
        if column != "strike_benefits":
            person[column] = [0.0, 100.0 + offset, 0.0, 200.0 + offset]
    person["partnership_income"] = [0.0, 5_000.0, 0.0, 7_000.0]
    person["s_corp_income"] = np.zeros(n)
    person["has_marketplace_health_coverage_at_interview"] = pd.array(
        [True, None, False, True], dtype="boolean"
    )
    person["fixture_label"] = pd.Series(["a", "b", "c", "d"], dtype=object)
    for column, values in (person_extra or {}).items():
        person[column] = values
    tables = {"person": person}
    for entity in US_SCHEMA.group_entities:
        tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    tables["household"]["state_fips"] = [6, 36, 6, 36]
    roles = ["asec", "asec", "puf_tax_detail", "puf_tax_detail"]
    for entity, table in tables.items():
        table[support_channel_column(entity)] = pd.Series(roles, dtype=object)
        table[support_source_id_column(entity)] = [1, 2, 1, 2]
        table[support_clone_index_column(entity)] = [0, 0, 1, 1]
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([40.0, 60.0, 50.0, 30.0]), WeightKind.CALIBRATED
            )
        },
    )


def _write_donor(path: Path, person_extra: dict[str, object] | None = None) -> Path:
    write_nullable_us_h5(
        _donor_frame(person_extra),
        path,
        period=2024,
        artifact_kind="fixture_national_release_h5",
        root_attributes=_ROOT_ATTRIBUTES,
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _patch(tool, tmp_path: Path, donor: Path, *, engine: _Engine | None = None):
    output = tmp_path / "out" / "donor_patched.h5"
    receipt_path = tmp_path / "out" / "donor_patched.receipt.json"
    receipt = tool.patch_donor_h5(
        donor,
        output,
        steps=tool.build_patch_steps(
            zero_columns=list(_ZERO_COLUMNS), engine=engine or _Engine()
        ),
        receipt_path=receipt_path,
        tool_identity={"commit": "f" * 40, "tool_dirty": False},
        engine_identity={"package": "policyengine-us", "version": "fixture"},
    )
    return output, receipt_path, receipt


def _tables(path: Path) -> tuple[list[str], dict[str, pd.DataFrame], dict, dict]:
    with pd.HDFStore(path, mode="r") as store:
        keys = sorted(store.keys())
        tables = {key: read_frame_table(store, key) for key in US_SCHEMA.entities}
        others = {
            key: store[key] for key in keys if key.lstrip("/") not in US_SCHEMA.entities
        }
        attributes = store.get_node("/")._v_attrs
        root = {name: attributes[name] for name in attributes._v_attrnames}
    return keys, tables, others, root


def test_added_columns_are_present_and_zero_on_their_engine_entity(
    tool, tmp_path: Path
) -> None:
    donor = _write_donor(tmp_path / "donor.h5")
    output, _receipt_path, receipt = _patch(tool, tmp_path, donor)

    _keys, before, _others, _root = _tables(donor)
    _keys, after, _others, _root = _tables(output)
    person = after["person"]
    assert list(person.columns) == [*before["person"].columns, *_ZERO_COLUMNS]
    for column in _ZERO_COLUMNS:
        assert person[column].dtype == np.float64
        assert person[column].tolist() == [0.0] * len(person)
    assert [(r["entity"], r["column"]) for r in receipt["added_columns"]] == [
        ("person", column) for column in _ZERO_COLUMNS
    ]
    for record in receipt["added_columns"]:
        assert record["reason"] == tool.REVIEWED_ZERO_COLUMNS[record["column"]]
        assert record["step"] == tool.ZERO_STEP
        assert record["action"] == "added"
        assert record["engine_default"] == 0.0
        assert record["value"] == 0
        assert record["dtype"] == "float64"
    assert receipt["kept_columns"] == []
    assert receipt["rewritten_tables"] == ["person"]


def test_every_other_table_column_dtype_and_root_attribute_is_unchanged(
    tool, tmp_path: Path
) -> None:
    donor = _write_donor(tmp_path / "donor.h5")
    donor_sha256 = _sha256(donor)
    output, _receipt_path, receipt = _patch(tool, tmp_path, donor)

    keys_before, before, others_before, root_before = _tables(donor)
    keys_after, after, others_after, root_after = _tables(output)
    assert keys_after == keys_before
    assert set(root_after) == set(root_before)
    for name, value in root_before.items():
        assert root_after[name] == value
    for name, value in _ROOT_ATTRIBUTES.items():
        assert root_after[name] == value
    assert others_after.keys() == others_before.keys()
    for key, value in others_before.items():
        assert others_after[key].equals(value)
    assert read_nullable_us_h5_metadata(output) == read_nullable_us_h5_metadata(donor)
    for entity in US_SCHEMA.entities:
        original = before[entity]
        pd.testing.assert_frame_equal(
            after[entity][list(original.columns)], original, check_exact=True
        )
        if entity != "person":
            pd.testing.assert_frame_equal(after[entity], original, check_exact=True)
    person = after["person"]
    nullable = person["has_marketplace_health_coverage_at_interview"]
    assert nullable.isna().tolist() == [False, True, False, False]
    assert person["fixture_label"].tolist() == ["a", "b", "c", "d"]
    assert receipt["preserved"]["tables"] == keys_before
    assert set(receipt["preserved"]["root_attributes"]) == set(root_before)
    # The input is read only.
    assert _sha256(donor) == donor_sha256


def test_receipt_records_correct_hashes_rows_and_commit(tool, tmp_path: Path) -> None:
    donor = _write_donor(tmp_path / "donor.h5")
    output, receipt_path, receipt = _patch(tool, tmp_path, donor)

    assert json.loads(receipt_path.read_text()) == receipt
    assert receipt["input"]["sha256"] == _sha256(donor)
    assert receipt["output"]["sha256"] == _sha256(output)
    assert receipt["input"]["sha256"] != receipt["output"]["sha256"]
    assert receipt["input"]["path"] == str(donor.resolve())
    assert receipt["output"]["path"] == str(output.resolve())
    expected_rows = {entity: 4 for entity in US_SCHEMA.entities}
    assert receipt["input"]["rows"] == expected_rows
    assert receipt["output"]["rows"] == expected_rows
    assert receipt["tool"] == "tools/patch_us_acs_donor_columns.py"
    assert receipt["tool_commit"] == {"commit": "f" * 40, "tool_dirty": False}
    assert receipt["engine"] == {"package": "policyengine-us", "version": "fixture"}
    assert receipt["steps"] == [
        {
            "name": tool.ZERO_STEP,
            "columns": [f"person.{column}" for column in _ZERO_COLUMNS],
        }
    ]
    # No temporary file is left beside the copy.
    assert sorted(path.name for path in output.parent.iterdir()) == [
        "donor_patched.h5",
        "donor_patched.receipt.json",
    ]


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ([0.0, 0.0, 25.0, 0.0], "refusing to overwrite"),
        ([0.0, np.nan, 0.0, 0.0], "refusing to overwrite"),
    ],
)
def test_existing_column_with_a_nonzero_or_null_value_is_refused(
    tool, tmp_path: Path, values: list[float], match: str
) -> None:
    donor = _write_donor(tmp_path / "donor.h5", {"strike_benefits": values})
    donor_sha256 = _sha256(donor)

    with pytest.raises(ValueError, match=match):
        _patch(tool, tmp_path, donor)

    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())
    assert _sha256(donor) == donor_sha256


def test_existing_all_zero_column_is_kept_not_duplicated(tool, tmp_path: Path) -> None:
    donor = _write_donor(tmp_path / "donor.h5", {"strike_benefits": np.zeros(4)})
    output, _receipt_path, receipt = _patch(tool, tmp_path, donor)

    _keys, after, _others, _root = _tables(output)
    assert list(after["person"].columns).count("strike_benefits") == 1
    assert [r["column"] for r in receipt["kept_columns"]] == ["strike_benefits"]
    assert [r["column"] for r in receipt["added_columns"]] == [
        "passive_partnership_s_corp_income"
    ]


def test_zero_columns_follow_the_engine_entity_and_default(
    tool, tmp_path: Path
) -> None:
    donor = _write_donor(tmp_path / "donor.h5")
    engine = _Engine(entities={"passive_partnership_s_corp_income": "tax_unit"})
    output, _receipt_path, receipt = _patch(tool, tmp_path, donor, engine=engine)

    _keys, after, _others, _root = _tables(output)
    assert after["tax_unit"]["passive_partnership_s_corp_income"].tolist() == [0.0] * 4
    assert "passive_partnership_s_corp_income" not in after["person"]
    assert receipt["rewritten_tables"] == ["person", "tax_unit"]

    # A NumPy scalar default is read as its Python value.
    step = tool.reviewed_zero_column_step(
        ["strike_benefits"], engine=_Engine(defaults={"strike_benefits": np.int64(0)})
    )
    assert step.name == tool.ZERO_STEP
    with pytest.raises(ValueError, match="not 0"):
        tool.reviewed_zero_column_step(
            ["strike_benefits"], engine=_Engine(defaults={"strike_benefits": 1.0})
        )
    with pytest.raises(ValueError, match="no default"):
        tool.reviewed_zero_column_step(["strike_benefits"], engine=_NoDefaults())
    with pytest.raises(ValueError, match="numeric input"):
        tool.reviewed_zero_column_step(
            ["strike_benefits"], engine=_Engine(dtype="bool")
        )
    with pytest.raises(ValueError, match="No reviewed reason"):
        tool.reviewed_zero_column_step(["s_corp_income"], engine=_Engine())
    with pytest.raises(SystemExit):
        tool._parse_args(
            ["--input-h5", "a.h5", "--output-h5", "b.h5", "--zero-column", "x"]
        )


def test_r1_income_donor_gate_passes_on_the_patched_fixture(
    tool, builder, tmp_path: Path
) -> None:
    donor = _write_donor(tmp_path / "donor.h5")
    with pytest.raises(SystemExit, match="strike_benefits"):
        builder._require_income_donor(builder._load_base_frame(donor))

    output, _receipt_path, _receipt = _patch(tool, tmp_path, donor)

    builder._require_income_donor(builder._load_base_frame(output))


def test_command_writes_the_default_receipt_beside_the_copy(
    tool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import microcosm.frame.adapters.policyengine_us as adapter

    donor = _write_donor(tmp_path / "donor.h5")
    monkeypatch.setattr(adapter, "PolicyEngineUSEngine", _Engine)
    monkeypatch.setattr(
        tool, "current_tool_identity", lambda: {"commit": "c" * 40, "tool_dirty": False}
    )
    output = tmp_path / "patched.h5"

    assert tool.main(["--input-h5", str(donor), "--output-h5", str(output)]) == 0

    receipt = json.loads((tmp_path / "patched.receipt.json").read_text())
    assert receipt["output"]["sha256"] == _sha256(output)
    assert receipt["tool_commit"]["commit"] == "c" * 40
    assert [r["column"] for r in receipt["added_columns"]] == list(
        tool.REVIEWED_ZERO_COLUMNS
    )

    with pytest.raises(SystemExit, match="Donor patch refused"):
        tool.main(["--input-h5", str(donor), "--output-h5", str(donor)])


def test_table_format_donor_keeps_its_layout(tool, builder, tmp_path: Path) -> None:
    """A donor stored like BuildP: compound tables, every column a data column."""

    frame = _donor_frame()
    donor = tmp_path / "donor-table.h5"
    with pd.HDFStore(donor, mode="w") as store:
        for entity in US_SCHEMA.entities:
            table = frame.table(entity).copy()
            if entity == "household":
                table["household_weight"] = frame.weights_for("household").values
            # Table format stores no null Boolean; the published donor has none.
            table = table.drop(
                columns=["has_marketplace_health_coverage_at_interview"],
                errors="ignore",
            )
            store.put(entity, table, format="table", data_columns=True)
        store.get_node("/")._v_attrs["populace_release_id"] = "fixture-release"

    output, _receipt_path, receipt = _patch(tool, tmp_path, donor)

    keys_before, before, _others, root_before = _tables(donor)
    keys_after, after, _others, root_after = _tables(output)
    assert keys_after == keys_before
    assert set(root_after) == set(root_before)
    assert root_after["populace_release_id"] == "fixture-release"
    with pd.HDFStore(donor, mode="r") as old, pd.HDFStore(output, mode="r") as new:
        for entity in US_SCHEMA.entities:
            assert new.get_storer(entity).is_table
            assert set(new.get_storer(entity).data_columns) == set(
                after[entity].columns
            )
            assert set(old.get_storer(entity).data_columns) == set(
                before[entity].columns
            )
    for entity in US_SCHEMA.entities:
        pd.testing.assert_frame_equal(
            after[entity][list(before[entity].columns)],
            before[entity],
            check_exact=True,
        )
    assert after["person"]["strike_benefits"].tolist() == [0.0] * 4
    assert receipt["rewritten_tables"] == ["person"]
    builder._require_income_donor(builder._load_base_frame(output))
