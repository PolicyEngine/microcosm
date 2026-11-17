"""The proof-of-concept donor patch for the ACS staging (microcosm#922).

``tools/patch_us_acs_donor_columns.py`` writes a patched copy of a US
national donor H5 with the ASEC work columns restored and reviewed all-zero
columns added. These tests use a small fixture H5 in the nullable layout R1's
``_load_base_frame`` reads, a stand-in engine and invented ASEC person
members whose pins are measured here, so no country engine is imported and
no Census file is read or downloaded.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.acs_local_income import (
    ACS_LOCAL_ALIGNED_RETIREMENT_DONOR_COMPONENTS,
    ACS_LOCAL_INCOME_TRANSFER_COLUMNS,
    ACS_LOCAL_OTHER_INCOME_DONOR_COMPONENTS,
)
from microcosm.build.us_runtime.asec_census_person_columns import WEIND_TO_WEMIND
from microcosm.build.us_runtime.h5_io import (
    read_nullable_us_h5_metadata,
    write_nullable_us_h5,
)
from microcosm.build.us_runtime.org_wages import derive_us_org_occupation_inputs
from microcosm.build.us_runtime.spm_role_source import AsecSpmRoleSource
from microcosm.build.us_runtime.support_provenance import (
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.data import contract
from microcosm.data.build_folder_manifests import referenced_build
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


def _patch(
    tool,
    tmp_path: Path,
    donor: Path,
    *,
    engine: _Engine | None = None,
    asec_sources=None,
    **options,
):
    output = tmp_path / "out" / "donor_patched.h5"
    receipt_path = tmp_path / "out" / "donor_patched.receipt.json"
    receipt = tool.patch_donor_h5(
        donor,
        output,
        steps=tool.build_patch_steps(
            zero_columns=list(_ZERO_COLUMNS),
            engine=engine or _Engine(),
            asec_sources=asec_sources,
        ),
        receipt_path=receipt_path,
        tool_identity={"commit": "f" * 40, "tool_dirty": False},
        engine_identity={"package": "policyengine-us", "version": "fixture"},
        **options,
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
    tool, asec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import microcosm.frame.adapters.policyengine_us as adapter

    donor = _write_asec_donor(tmp_path / "donor.h5")
    monkeypatch.setattr(adapter, "PolicyEngineUSEngine", _Engine)
    monkeypatch.setattr(
        tool, "current_tool_identity", lambda: {"commit": "c" * 40, "tool_dirty": False}
    )
    output = tmp_path / "patched.h5"

    assert tool.main(["--input-h5", str(donor), "--output-h5", str(output)]) == 0

    receipt = json.loads((tmp_path / "patched.receipt.json").read_text())
    assert receipt["output"]["sha256"] == _sha256(output)
    assert receipt["tool_commit"]["commit"] == "c" * 40
    assert [r["column"] for r in receipt["added_columns"]] == [
        *tool.ASEC_WORK_COLUMNS,
        *tool.REVIEWED_ZERO_COLUMNS,
    ]

    with pytest.raises(SystemExit, match="Donor patch refused"):
        tool.main(["--input-h5", str(donor), "--output-h5", str(donor)])


def test_an_existing_copy_receipt_or_manifest_is_replaced_only_when_asked(
    tool, asec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review of #1660: a re-run writes another SHA-256, so a staging summary
    that pinned the first copy would name bytes no longer at its path."""

    import microcosm.frame.adapters.policyengine_us as adapter

    donor = _write_asec_donor(tmp_path / "donor.h5")
    monkeypatch.setattr(adapter, "PolicyEngineUSEngine", _Engine)
    monkeypatch.setattr(
        tool, "current_tool_identity", lambda: {"commit": "c" * 40, "tool_dirty": False}
    )
    output = tmp_path / "patched.h5"
    receipt_path = tmp_path / "patched.receipt.json"
    base = ["--input-h5", str(donor), "--output-h5", str(output)]
    assert tool.main(base) == 0
    first = (output.read_bytes(), receipt_path.read_bytes())

    with pytest.raises(SystemExit, match="already exist.*--overwrite"):
        tool.main(base)
    assert (output.read_bytes(), receipt_path.read_bytes()) == first

    # An existing receipt alone is refused too.
    output.unlink()
    with pytest.raises(SystemExit, match="already exist"):
        tool.main(base)
    assert not output.exists()

    assert tool.main([*base, "--overwrite"]) == 0
    assert json.loads(receipt_path.read_text())["output"]["sha256"] == _sha256(output)

    # So is an existing copy release manifest, before any work.
    manifest_path = tmp_path / "release_manifest.json"
    manifest_path.write_text("{}")
    fresh = tmp_path / "fresh.h5"
    with pytest.raises(SystemExit, match="release_manifest.json"):
        tool.main(
            [
                "--input-h5",
                str(donor),
                "--output-h5",
                str(fresh),
                "--release-manifest",
                str(manifest_path),
                "--manifest-release",
                _COPY_RELEASE,
                "--manifest-build",
                _COPY_BUILD,
            ]
        )
    assert manifest_path.read_text() == "{}"
    assert not fresh.exists()


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


_PUBLISHED_RELEASE = "microcosm-us-2024-receipt-donor"
_PUBLISHED_BUILD = "populace-us-2024-spm-receipts-20260923"
_COPY_RELEASE = "county-files-poc-donor"
_COPY_BUILD = "spm-receipts-20260923-patched-20261117"


def _published_manifest(sha256: str) -> dict:
    """The published donor's release manifest, in the build-folder layout."""

    return {
        "schema_version": 1,
        "release": _PUBLISHED_RELEASE,
        "build": _PUBLISHED_BUILD,
        "dataset_role": "national_default",
        "is_default": False,
        "artifacts": {
            "donor": {
                "kind": "microdata",
                "path": "donor.h5",
                "repo_id": "policyengine/populace-us",
                "sha256": sha256,
                "role": "dataset",
            }
        },
    }


def test_release_manifest_names_the_copy_for_r1_and_the_publisher(
    tool, builder, tmp_path: Path
) -> None:
    donor = _write_donor(tmp_path / "donor.h5")
    published = tmp_path / "published_release_manifest.json"
    published.write_text(json.dumps(_published_manifest(_sha256(donor))))
    manifest_path = tmp_path / "out" / "release_manifest.json"
    target = tool.CopyReleaseManifest(
        path=manifest_path, release=_COPY_RELEASE, build=_COPY_BUILD
    )

    output, receipt_path, receipt = _patch(
        tool,
        tmp_path,
        donor,
        input_release_manifest=published,
        release_manifest=target,
    )

    # R1 verifies the copy against the manifest and names the copy's build.
    identity = builder._donor_release_identity(manifest_path, _sha256(output))
    assert {key: identity[key] for key in ("repo_id", "release", "build")} == {
        "repo_id": "policyengine/populace-us-staging",
        "release": _COPY_RELEASE,
        "build": _COPY_BUILD,
    }
    assert identity["sha256"] == _sha256(output)
    assert identity["dataset_role"] == tool.COPY_DATASET_ROLE
    assert identity["is_default"] is False
    copy_folder = f"releases/{_COPY_RELEASE}/{_COPY_BUILD}"
    assert referenced_build(identity).folder == copy_folder
    # The local-area release contract accepts it as the donor reference.
    failures: list[str] = []
    contract._check_local_area_donor_reference(identity, failures)
    assert failures == []
    # The published manifest still names the input, never the copy.
    with pytest.raises(SystemExit, match="does not match its release manifest"):
        builder._donor_release_identity(published, _sha256(output))

    manifest = json.loads(manifest_path.read_text())
    published_folder = f"releases/{_PUBLISHED_RELEASE}/{_PUBLISHED_BUILD}"
    assert manifest["derived_from"]["sha256"] == _sha256(donor)
    assert manifest["derived_from"]["release"]["folder"] == published_folder
    assert manifest["artifacts"]["patch_receipt"] == {
        "kind": "receipt",
        "path": "donor_patched.receipt.json",
        "repo_id": "policyengine/populace-us-staging",
        "sha256": _sha256(receipt_path),
        "role": "patch_receipt",
    }
    assert manifest["patch"]["added_columns"] == [
        f"person.{column}" for column in _ZERO_COLUMNS
    ]
    assert receipt["input"]["release"]["folder"] == published_folder
    assert receipt["input"]["release"]["manifest"] == published.name
    assert receipt["release_manifest"]["folder"] == copy_folder
    # The uploaded manifest names files, not this machine's paths.
    assert str(tmp_path) not in manifest_path.read_text()


def test_input_release_manifest_must_pin_the_input(tool, tmp_path: Path) -> None:
    donor = _write_donor(tmp_path / "donor.h5")
    published = tmp_path / "published_release_manifest.json"
    published.write_text(json.dumps(_published_manifest("0" * 64)))

    with pytest.raises(ValueError, match="pins '0000"):
        _patch(tool, tmp_path, donor, input_release_manifest=published)

    assert not (tmp_path / "out").exists()
    unreadable = _published_manifest(_sha256(donor))
    del unreadable["artifacts"]["donor"]["repo_id"]
    published.write_text(json.dumps(unreadable))
    with pytest.raises(ValueError, match="names no readable build"):
        _patch(tool, tmp_path, donor, input_release_manifest=published)


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"release": "a/b"}, "one folder name"),
        ({"build": ""}, "one folder name"),
        ({"repo_id": "populace-us-staging"}, "OWNER/NAME"),
    ],
)
def test_copy_release_manifest_refuses_a_bad_build_name(
    tool, tmp_path: Path, options: dict, match: str
) -> None:
    fields = {"release": _COPY_RELEASE, "build": _COPY_BUILD} | options
    with pytest.raises(ValueError, match=match):
        tool.CopyReleaseManifest(path=tmp_path / "release_manifest.json", **fields)


def test_command_writes_the_copy_release_manifest_only_when_asked(
    tool, asec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import microcosm.frame.adapters.policyengine_us as adapter

    donor = _write_asec_donor(tmp_path / "donor.h5")
    monkeypatch.setattr(adapter, "PolicyEngineUSEngine", _Engine)
    monkeypatch.setattr(
        tool, "current_tool_identity", lambda: {"commit": "c" * 40, "tool_dirty": False}
    )
    base = ["--input-h5", str(donor), "--output-h5", str(tmp_path / "patched.h5")]
    manifest_path = tmp_path / "release_manifest.json"

    assert (
        tool.main(
            [
                *base,
                "--release-manifest",
                str(manifest_path),
                "--manifest-release",
                _COPY_RELEASE,
                "--manifest-build",
                _COPY_BUILD,
            ]
        )
        == 0
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["artifacts"]["patched"]["sha256"] == _sha256(
        tmp_path / "patched.h5"
    )
    assert manifest["release"] == _COPY_RELEASE
    for partial in (
        ["--release-manifest", str(manifest_path)],
        ["--release-manifest", str(manifest_path), "--manifest-build", _COPY_BUILD],
        ["--manifest-release", _COPY_RELEASE, "--manifest-build", _COPY_BUILD],
    ):
        with pytest.raises(SystemExit):
            tool._parse_args([*base, *partial])


# --------------------------------------------------------------------------
# The ASEC work-column restore step
# --------------------------------------------------------------------------

#: Invented ASEC person members by income year: key -> (PH_SEQ, P_SEQ,
#: A_LINENO, A_AGE, WKSWORK, POCCU2, PRDTRACE, PRDTHSP, WEIND). Key 1 is in
#: both years with different answers (CPS rotation), so a join that ignored
#: source_year would pair the wrong person. Keys 2 and 3 are member-only: the
#: donor is a subset of each member, as the real donor is.
_ASEC_MEMBERS = {
    2022: {
        1: (11, 1, 1, 34, 52, 8, 1, 0, 5),
        2: (12, 1, 1, 9, 0, 0, 1, 0, 0),
    },
    2023: {
        1: (21, 1, 1, 70, 0, 0, 2, 1, 23),
        3: (22, 1, 1, 30, 30, 52, 4, 0, 22),
    },
}
_ASEC_MEMBER_FIELDS = (
    "PH_SEQ",
    "P_SEQ",
    "A_LINENO",
    "A_AGE",
    "WKSWORK",
    "POCCU2",
    "PRDTRACE",
    "PRDTHSP",
    "WEIND",
)
#: The fixture donor's persons as (source_year, key): rows 2 and 3 are the
#: PUF-support clones of rows 0 and 1 (``_donor_frame``'s support ids).
_ASEC_DONOR_KEYS = ((2022, 1), (2023, 1), (2022, 1), (2023, 1))
_EXPECTED_WORK = {
    "detailed_industry_recode": [5, 23, 5, 23],
    "major_industry_recode": [WEIND_TO_WEMIND[5], 15, WEIND_TO_WEMIND[5], 15],
    "worked_last_year": [True, False, True, False],
}
_DONOR_IDENTITY = (
    "source_household_id",
    "P_SEQ",
    "A_LINENO",
    "A_AGE",
    "WKSWORK",
    "POCCU2",
    "PRDTRACE",
    "PRDTHSP",
)


def _peridnum(key: int, *, wide: bool = False) -> str:
    # Leading zeros: the join must compare exact text, never numbers. Wide keys
    # share their first 15 digits, so they collide as float64.
    return f"91{key:020d}" if wide else f"{key * 7:022d}"


def _asec_member_row(year: int, key: int) -> dict[str, int]:
    values = dict(zip(_ASEC_MEMBER_FIELDS, _ASEC_MEMBERS[year][key], strict=True))
    values["WEMIND"] = WEIND_TO_WEMIND[values["WEIND"]]
    return values


def _asec_person_columns(
    *, wide_keys: bool = False, overrides: dict | None = None, drop: tuple = ()
) -> dict[str, pd.Series]:
    """The ASEC person columns a pre-restoration donor carries.

    The identity columns equal each row's member; the carry's other outputs
    are the carry's own; the three work columns and raw WEIND/WEMIND are
    absent, as on 766b7c7's donor.
    """

    members = [_asec_member_row(year, key) for year, key in _ASEC_DONOR_KEYS]
    table = pd.DataFrame(
        {
            "PERIDNUM": [_peridnum(key, wide=wide_keys) for _, key in _ASEC_DONOR_KEYS],
            "source_year": [year for year, _ in _ASEC_DONOR_KEYS],
            "source_household_id": [row["PH_SEQ"] for row in members],
            **{
                column: [row[column] for row in members]
                for column in _DONOR_IDENTITY[1:]
            },
        }
    )
    carried = derive_us_org_occupation_inputs(
        table.assign(
            WEIND=[row["WEIND"] for row in members],
            WEMIND=[row["WEMIND"] for row in members],
        )
    )
    columns = {column: table[column] for column in table.columns}
    for column in carried.columns:
        if column not in _EXPECTED_WORK:
            columns[column] = carried[column]
    for column, values in (overrides or {}).items():
        columns[column] = pd.Series(values)
    return {column: values for column, values in columns.items() if column not in drop}


def _write_asec_donor(path: Path, **options) -> Path:
    return _write_donor(path, _asec_person_columns(**options))


def _write_member(
    path: Path,
    year: int,
    *,
    drop_keys: tuple[int, ...] = (),
    overrides: dict | None = None,
    wide_keys: bool = False,
    duplicate_keys: tuple[int, ...] = (),
) -> Path:
    rows = []
    for key in _ASEC_MEMBERS[year]:
        if key in drop_keys:
            continue
        row = {
            "PERIDNUM": _peridnum(key, wide=wide_keys),
            **_asec_member_row(year, key),
        }
        row.update((overrides or {}).get(key, {}))
        row["A_FNLWGT"] = 1000.0 + key
        rows.append(row)
        if key in duplicate_keys:
            rows.append(dict(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")
    return path


def _measured_pin(path: Path, year: int) -> AsecSpmRoleSource:
    return AsecSpmRoleSource(
        income_year=year,
        survey_year=year + 1,
        csv_sha256=_sha256(path),
        csv_size_bytes=path.stat().st_size,
        persons=int(len(pd.read_csv(path))),
        units=0,
        official_archive_url=f"https://example.invalid/asecpub{str(year + 1)[2:]}csv.zip",
        archive_sha256="0" * 64,
        member=path.name,
    )


def _refuse_download(*args, **kwargs):
    raise AssertionError("a test attempted an ASEC download")


@pytest.fixture
def asec(tool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Invented members with measured pins, as the test's default ASEC cache.

    The command resolves its pins through ``_resolve_pin``; here that falls
    back to the measured pins (the vintage check still runs). No test reads
    the real cache or downloads.
    """

    def build(**member_options) -> SimpleNamespace:
        cache = tmp_path / "asec-cache"
        paths, pins = {}, {}
        for year in _ASEC_MEMBERS:
            options = {
                name: value.get(year, ()) if name != "wide_keys" else value
                for name, value in member_options.items()
            }
            if "overrides" in options and options["overrides"] == ():
                options["overrides"] = None
            path = _write_member(
                cache / f"pppub{str(year + 1)[2:]}.csv", year, **options
            )
            paths[year], pins[year] = path, _measured_pin(path, year)
        resolve = tool._resolve_pin
        monkeypatch.setattr(
            tool,
            "_resolve_pin",
            lambda year, pin: resolve(year, pin if pin is not None else pins[year]),
        )
        monkeypatch.setattr(tool, "DEFAULT_ASEC_SOURCE_CACHE", cache)
        monkeypatch.setattr(
            tool.education_assistance_source,
            "fetch_asec_education_assistance_source",
            _refuse_download,
        )
        return SimpleNamespace(
            cache=cache,
            paths=paths,
            pins=pins,
            sources=tool.AsecPersonSources(paths=paths, pins=pins),
        )

    build()
    return build


def test_work_columns_are_restored_by_the_year_scoped_key(
    tool, asec, tmp_path: Path
) -> None:
    fixture = asec()
    donor = _write_asec_donor(tmp_path / "donor.h5")

    output, _receipt_path, receipt = _patch(
        tool, tmp_path, donor, asec_sources=fixture.sources
    )

    _keys, before, _others, _root = _tables(donor)
    _keys, after, _others, _root = _tables(output)
    person = after["person"]
    assert list(person.columns) == [
        *before["person"].columns,
        *tool.ASEC_WORK_COLUMNS,
        *_ZERO_COLUMNS,
    ]
    for column, dtype in tool.ASEC_WORK_DTYPES.items():
        assert person[column].dtype == np.dtype(dtype)
        assert person[column].tolist() == _EXPECTED_WORK[column]
    # Key 1 takes each year's own answer; each PUF clone its sibling's.
    assert person["detailed_industry_recode"].tolist()[:2] == [5, 23]
    for entity in US_SCHEMA.entities:
        pd.testing.assert_frame_equal(
            after[entity][list(before[entity].columns)],
            before[entity],
            check_exact=True,
        )

    work = [r for r in receipt["added_columns"] if r["step"] == tool.ASEC_WORK_STEP]
    assert [(r["entity"], r["column"], r["dtype"]) for r in work] == [
        ("person", "detailed_industry_recode", "int16"),
        ("person", "major_industry_recode", "int16"),
        ("person", "worked_last_year", "bool"),
    ]
    assert [r["source_item"] for r in work] == ["WEIND", "WEMIND", "WKSWORK > 0"]
    assert receipt["rewritten_tables"] == ["person"]
    step = receipt["steps"][0]
    assert step["name"] == tool.ASEC_WORK_STEP
    assert step["columns"] == [f"person.{c}" for c in tool.ASEC_WORK_COLUMNS]
    assert receipt["steps"][1]["name"] == tool.ZERO_STEP
    for year in (2022, 2023):
        record = step["files"][str(year)]
        assert record["member_sha256"] == _sha256(fixture.paths[year])
        assert record["path"] == str(fixture.paths[year].resolve())
        assert record["form"] == "csv"
        assert (record["income_year"], record["survey_year"]) == (year, year + 1)
        assert record["persons"] == 2
        assert step["join"]["by_income_year"][str(year)] == {
            "donor_rows": 2,
            "donor_rows_matched_once": 2,
            "distinct_keys_matched": 1,
            "member_rows": 2,
            "member_rows_not_in_donor": 1,
            "identity_mismatches": 0,
        }
    assert step["join"]["donor_rows"] == 4
    assert step["join"]["unmatched_donor_rows"] == 0
    assert step["join"]["multiply_matched_donor_rows"] == 0
    assert step["clones"]["keys_with_two_rows"] == 2
    assert step["clones"]["repeated_key_channel_composition"] == {
        "asec+puf_tax_detail": 2
    }
    assert "cps_race" in step["carry"]["reemitted_columns_verified_equal"]
    shares = step["work_experience"]["nonzero_shares"]["worked_last_year"]
    # Persons 1 and 3 (weights 40 and 50 of 180) worked last year.
    assert shares["weighted_nonzero_share"] == pytest.approx(90 / 180)
    json.dumps(receipt, allow_nan=False)


def test_r1_work_experience_gate_passes_on_the_patched_fixture(
    tool, builder, asec, tmp_path: Path
) -> None:
    fixture = asec()
    donor = _write_asec_donor(tmp_path / "donor.h5")
    with pytest.raises(SystemExit, match="detailed_industry_recode"):
        builder._require_donor_work_experience(builder._load_base_frame(donor))

    output, _receipt_path, _receipt = _patch(
        tool, tmp_path, donor, asec_sources=fixture.sources
    )

    frame = builder._load_base_frame(output)
    builder._require_donor_work_experience(frame)
    builder._require_income_donor(frame)


def test_table_format_donor_keeps_its_layout_with_the_work_columns(
    tool, builder, asec, tmp_path: Path
) -> None:
    """The published donor's layout: compound tables, every column indexed."""

    fixture = asec()
    frame = _donor_frame(_asec_person_columns())
    donor = tmp_path / "donor-table.h5"
    with pd.HDFStore(donor, mode="w") as store:
        for entity in US_SCHEMA.entities:
            table = frame.table(entity).copy()
            if entity == "household":
                table["household_weight"] = frame.weights_for("household").values
            table = table.drop(
                columns=["has_marketplace_health_coverage_at_interview"],
                errors="ignore",
            )
            store.put(entity, table, format="table", data_columns=True)

    output, _receipt_path, receipt = _patch(
        tool, tmp_path, donor, asec_sources=fixture.sources
    )

    _keys, after, _others, _root = _tables(output)
    with pd.HDFStore(output, mode="r") as store:
        storer = store.get_storer("person")
        assert storer.is_table
        assert set(tool.ASEC_WORK_COLUMNS) <= set(storer.data_columns)
    for column, dtype in tool.ASEC_WORK_DTYPES.items():
        assert after["person"][column].dtype == np.dtype(dtype)
        assert after["person"][column].tolist() == _EXPECTED_WORK[column]
    assert receipt["rewritten_tables"] == ["person"]
    builder._require_donor_work_experience(builder._load_base_frame(output))


def test_join_compares_exact_text_keys(tool, asec, tmp_path: Path) -> None:
    # Every key shares its first 15 digits, so a numeric join could not tell
    # the people apart.
    fixture = asec(wide_keys=True)
    assert len({float(_peridnum(key, wide=True)) for key in (1, 2, 3)}) == 1
    donor = _write_asec_donor(tmp_path / "donor.h5", wide_keys=True)

    output, _receipt_path, _receipt = _patch(
        tool, tmp_path, donor, asec_sources=fixture.sources
    )

    _keys, after, _others, _root = _tables(output)
    for column, expected in _EXPECTED_WORK.items():
        assert after["person"][column].tolist() == expected


def _refused(tool, tmp_path: Path, donor: Path, sources, match: str) -> None:
    donor_sha256 = _sha256(donor)
    with pytest.raises(ValueError, match=match):
        _patch(tool, tmp_path, donor, asec_sources=sources)
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())
    assert _sha256(donor) == donor_sha256


def test_refuses_a_donor_person_without_a_member_row(
    tool, asec, tmp_path: Path
) -> None:
    fixture = asec(drop_keys={2023: (1,)})
    donor = _write_asec_donor(tmp_path / "donor.h5")

    _refused(
        tool,
        tmp_path,
        donor,
        fixture.sources,
        "2 donor person rows of income year 2023 have no row",
    )


def test_refuses_a_member_that_would_match_a_person_twice(
    tool, asec, tmp_path: Path
) -> None:
    fixture = asec(duplicate_keys={2022: (1,)})
    donor = _write_asec_donor(tmp_path / "donor.h5")

    _refused(tool, tmp_path, donor, fixture.sources, "repeats PERIDNUM on 2 rows")


def test_refuses_a_file_whose_sha256_is_not_pinned(tool, asec, tmp_path: Path) -> None:
    fixture = asec()
    donor = _write_asec_donor(tmp_path / "donor.h5")
    pins = dict(fixture.pins)
    pins[2022] = AsecSpmRoleSource(**{**vars(pins[2022]), "csv_sha256": "0" * 64})

    _refused(
        tool,
        tmp_path,
        donor,
        tool.AsecPersonSources(paths=fixture.paths, pins=pins),
        "SHA-256 mismatch",
    )
    # A file of another year is refused against its year's pin, by name.
    _refused(
        tool,
        tmp_path,
        donor,
        tool.AsecPersonSources(
            paths={2022: fixture.paths[2023], 2023: fixture.paths[2023]},
            pins=fixture.pins,
        ),
        r"pppub23\.csv \(read from pppub24\.csv\)",
    )


def test_refuses_a_pin_for_another_vintage(tool, asec, tmp_path: Path) -> None:
    fixture = asec()
    donor = _write_asec_donor(tmp_path / "donor.h5")
    pins = {2022: fixture.pins[2023], 2023: fixture.pins[2023]}

    _refused(
        tool,
        tmp_path,
        donor,
        tool.AsecPersonSources(paths=fixture.paths, pins=pins),
        "pin is for income year 2023",
    )


@pytest.mark.parametrize("column", _DONOR_IDENTITY)
def test_refuses_one_row_that_disagrees_on_any_identity_column(
    tool, asec, tmp_path: Path, column: str
) -> None:
    fixture = asec()
    # Row 2 is the 2022 PUF clone; only the named column changes.
    values = _asec_person_columns()[column].tolist()
    values[2] += 1
    donor = _write_asec_donor(tmp_path / "donor.h5", overrides={column: values})

    _refused(
        tool,
        tmp_path,
        donor,
        fixture.sources,
        rf"Income year 2022 donor rows disagree .*\{{'{column}': 1\}}",
    )


def test_refuses_a_code_outside_the_reviewed_domain(tool, asec, tmp_path: Path) -> None:
    fixture = asec(overrides={2023: {3: {"WEIND": 99}}})
    donor = _write_asec_donor(tmp_path / "donor.h5")

    _refused(tool, tmp_path, donor, fixture.sources, "outside its reviewed Census")


def test_refuses_a_donor_the_carry_did_not_build(tool, asec, tmp_path: Path) -> None:
    fixture = asec()
    cps_race = _asec_person_columns()["cps_race"].copy()
    cps_race.iloc[1] = 7
    donor = _write_asec_donor(tmp_path / "donor.h5", overrides={"cps_race": cps_race})

    _refused(tool, tmp_path, donor, fixture.sources, "Donor cps_race differs")


def test_refuses_a_donor_without_the_join_columns(tool, asec, tmp_path: Path) -> None:
    fixture = asec()
    donor = _write_asec_donor(tmp_path / "donor.h5", drop=("P_SEQ",))

    _refused(tool, tmp_path, donor, fixture.sources, r"lacks \['P_SEQ'\]")


def test_existing_work_columns_are_kept_only_when_equal(
    tool, asec, tmp_path: Path
) -> None:
    fixture = asec()
    same = pd.Series(_EXPECTED_WORK["worked_last_year"])
    donor = _write_asec_donor(
        tmp_path / "donor.h5", overrides={"worked_last_year": same}
    )

    _output, _receipt_path, receipt = _patch(
        tool, tmp_path, donor, asec_sources=fixture.sources
    )

    assert [r["column"] for r in receipt["kept_columns"]] == ["worked_last_year"]

    other = tmp_path / "other"
    other.mkdir()
    different = pd.Series([True, True, True, False])
    donor = _write_asec_donor(
        other / "donor.h5", overrides={"worked_last_year": different}
    )
    _refused(tool, other, donor, fixture.sources, "refusing to overwrite")

    # A work column on another entity is refused, not duplicated on person.
    step = tool.asec_work_column_step(fixture.sources)
    with pytest.raises(ValueError, match="on tax_unit; refusing to duplicate"):
        step.run(
            {
                "person": pd.DataFrame(_asec_person_columns()),
                "tax_unit": pd.DataFrame({"worked_last_year": [True]}),
            }
        )


def test_reads_the_cache_as_member_csvs_or_official_archives(
    tool, asec, tmp_path: Path
) -> None:
    fixture = asec()
    donor = _write_asec_donor(tmp_path / "donor.h5")
    archives = tmp_path / "archives"
    archives.mkdir()
    pins = {}
    for year, path in fixture.paths.items():
        pin = fixture.pins[year]
        archive = archives / Path(pin.official_archive_url).name
        with zipfile.ZipFile(archive, "w") as handle:
            handle.write(path, arcname=pin.member)
        pins[year] = AsecSpmRoleSource(
            **{**vars(pin), "archive_sha256": _sha256(archive)}
        )

    _output, _receipt_path, receipt = _patch(
        tool,
        tmp_path,
        donor,
        asec_sources=tool.AsecPersonSources(cache=archives, pins=pins),
    )

    files = receipt["steps"][0]["files"]
    assert [files[str(year)]["form"] for year in (2022, 2023)] == ["archive"] * 2
    assert files["2022"]["archive_sha256"] == pins[2022].archive_sha256
    assert files["2022"]["path"].endswith("asecpub23csv.zip")

    # The CSV form, from the default cache.
    shutil.rmtree(tmp_path / "out")
    _output, _receipt_path, receipt = _patch(
        tool, tmp_path, donor, asec_sources=tool.AsecPersonSources(pins=fixture.pins)
    )
    files = receipt["steps"][0]["files"]
    assert files["2023"]["path"] == str((fixture.cache / "pppub24.csv").resolve())
    assert files["2023"]["form"] == "csv"


def test_refuses_absent_members_and_fetches_only_when_asked(
    tool, asec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = asec()
    donor = _write_asec_donor(tmp_path / "donor.h5")
    cache = tmp_path / "partial-cache"
    cache.mkdir()
    shutil.copyfile(fixture.paths[2022], cache / "pppub23.csv")

    _refused(
        tool,
        tmp_path,
        donor,
        tool.AsecPersonSources(cache=cache, pins=fixture.pins),
        r"income year\(s\) \['2023'\] are not in .*--asec-fetch",
    )

    calls = []

    def download(income_year, cache_dir=None):
        calls.append((income_year, Path(cache_dir)))
        target = Path(cache_dir) / fixture.pins[income_year].member
        shutil.copyfile(fixture.paths[income_year], target)
        return target

    monkeypatch.setattr(
        tool.education_assistance_source,
        "fetch_asec_education_assistance_source",
        download,
    )
    _output, _receipt_path, receipt = _patch(
        tool,
        tmp_path,
        donor,
        asec_sources=tool.AsecPersonSources(cache=cache, fetch=True, pins=fixture.pins),
    )
    assert calls == [(2023, cache)]
    assert receipt["steps"][0]["files"]["2023"]["path"] == str(
        (cache / "pppub24.csv").resolve()
    )


def test_command_restores_the_work_columns_and_passes_its_options(
    tool, asec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import microcosm.frame.adapters.policyengine_us as adapter

    fixture = asec()
    donor = _write_asec_donor(tmp_path / "donor.h5")
    monkeypatch.setattr(adapter, "PolicyEngineUSEngine", _Engine)
    monkeypatch.setattr(
        tool, "current_tool_identity", lambda: {"commit": "c" * 40, "tool_dirty": False}
    )
    base = ["--input-h5", str(donor), "--output-h5", str(tmp_path / "patched.h5")]

    # Without the members, the command refuses rather than skip the step.
    monkeypatch.setattr(tool, "DEFAULT_ASEC_SOURCE_CACHE", tmp_path / "absent")
    with pytest.raises(SystemExit, match="Donor patch refused: The pinned ASEC"):
        tool.main(base)

    assert tool.main([*base, "--asec-source-cache", str(fixture.cache)]) == 0
    receipt = json.loads((tmp_path / "patched.receipt.json").read_text())
    assert receipt["steps"][0]["name"] == tool.ASEC_WORK_STEP
    assert receipt["steps"][0]["files"]["2022"]["path"] == str(
        fixture.paths[2022].resolve()
    )

    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        raise ValueError("captured")

    monkeypatch.setattr(tool, "build_patch_steps", capture)
    with pytest.raises(SystemExit, match="captured"):
        tool.main([*base, "--asec-source-cache", str(tmp_path / "c"), "--asec-fetch"])
    assert seen["asec_sources"] == tool.AsecPersonSources(
        cache=tmp_path / "c", fetch=True
    )
    with pytest.raises(SystemExit, match="captured"):
        tool.main(base)
    assert seen["asec_sources"] == tool.AsecPersonSources()
