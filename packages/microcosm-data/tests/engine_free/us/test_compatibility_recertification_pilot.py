"""Pure pilot contracts. Every fixture contains synthetic MOCK inputs only."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.data import compatibility_recertification as recert
from test_support.microcosm_data.compatibility_recertification_pilot import (
    membership_inputs,
    mock_tables,
)
from test_support.paths import paths_for
from tools import build_us_compatibility_recertification as scorer_tool
from tools import probe_us_compatibility_recertification as pilot


def test_direct_cli_outside_checkout_reaches_missing_wheel_refusal(tmp_path):
    script = (
        paths_for("microcosm-data").repository
        / "tools"
        / "probe_us_compatibility_recertification.py"
    )
    wheel = tmp_path / "NONEXISTENT_WHEEL.whl"
    output = tmp_path / "MUST_NOT_BE_CREATED.json"
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    run = subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            "--release-dir",
            str(tmp_path / "NONEXISTENT_RELEASE"),
            "--artifact-root",
            str(tmp_path / "NONEXISTENT_ARTIFACTS"),
            "--output",
            str(output),
            "--compatibility-wheel",
            str(wheel),
            "--max-households",
            "1",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert run.returncode == 2, run.stderr
    assert "Engineering pilot refused:" in run.stderr
    assert str(wheel) in run.stderr
    assert not any(
        message in run.stderr
        for message in ("ModuleNotFoundError", "ImportError", "Traceback")
    )
    assert run.stdout == ""
    assert not output.exists()


@pytest.mark.parametrize("value", [False, True, 0, -1, 33, 1.0, "2", None])
def test_invalid_limit_refuses_before_runtime_or_inputs(value, tmp_path, monkeypatch):
    monkeypatch.setattr(
        scorer_tool, "scoring_runtime", lambda *_: pytest.fail("runtime reached")
    )
    with pytest.raises(pilot.EngineeringPilotError, match="integer from 1 to 32"):
        pilot.probe_candidate(
            release_dir=tmp_path / "absent",
            artifact_root=tmp_path / "absent2",
            output=tmp_path / "result.json",
            compatibility_wheels=(),
            max_households=value,
        )


@settings(max_examples=35, deadline=None)
@given(
    sizes=st.lists(st.integers(1, 4), min_size=1, max_size=40),
    limit=st.integers(1, 32),
    reverse=st.booleans(),
    unsigned=st.booleans(),
)
def test_selection_preserves_complete_source_rows_and_weights(
    sizes, limit, reverse, unsigned
):
    tables = mock_tables(
        tuple(sizes),
        id_dtype="uint64" if unsigned else "int64",
        offset=2**53 + 17 if unsigned else 100,
    )
    if reverse:
        for entity in ("tax_unit", "spm_unit", "family", "marital_unit"):
            tables[entity] = tables[entity].iloc[::-1].reset_index(drop=True)
    before = pilot._frame_inventory(tables)
    memberships, groups = membership_inputs(tables)
    selected, masks = pilot._select_rows(
        groups["household"], memberships, groups, max_households=limit
    )
    count = min(limit, len(sizes))
    np.testing.assert_array_equal(selected, groups["household"][:count])
    assert selected.dtype == groups["household"].dtype
    # Independent positional reference: each fixture household owns its next size rows.
    people = set(tables["person"].person_id.iloc[: sum(sizes[:count])])
    for entity, frame in tables.items():
        if entity == "person":
            expected = frame.person_id.isin(people)
        else:
            members = tables["person"].loc[
                tables["person"].person_id.isin(people), f"person_{entity}_id"
            ]
            expected = frame[f"{entity}_id"].isin(set(members))
        pd.testing.assert_frame_equal(frame.loc[masks[entity]], frame.loc[expected])
        assert len(masks[entity]) == len(frame)
    assert pilot._frame_inventory(tables) == before
    np.testing.assert_array_equal(
        tables["household"].loc[masks["household"], "household_weight"],
        tables["household"].household_weight.iloc[:count],
    )


@pytest.mark.parametrize(
    "damage",
    [
        "shared_unit",
        "dangling_household",
        "missing_group",
        "duplicate_person",
        "duplicate_group",
        "orphan_household",
        "float_id",
    ],
)
def test_membership_damage_refuses_without_selecting_partial_households(damage):
    tables = mock_tables((2, 2, 1))
    if damage == "shared_unit":
        tables["person"].loc[2, "person_tax_unit_id"] = tables[
            "tax_unit"
        ].tax_unit_id.iloc[0]
    elif damage == "dangling_household":
        tables["person"].loc[0, "person_household_id"] = 999999
    elif damage == "missing_group":
        tables["tax_unit"] = tables["tax_unit"].iloc[1:]
    elif damage == "duplicate_person":
        tables["person"].loc[1, "person_id"] = tables["person"].person_id.iloc[0]
    elif damage == "duplicate_group":
        tables["family"].loc[1, "family_id"] = tables["family"].family_id.iloc[0]
    elif damage == "orphan_household":
        tables["person"] = tables["person"].iloc[2:]
    elif damage == "float_id":
        tables["household"]["household_id"] = tables["household"].household_id.astype(
            float
        )
    memberships, groups = membership_inputs(tables)
    with pytest.raises(pilot.EngineeringPilotError):
        pilot._select_rows(groups["household"], memberships, groups, max_households=1)


def test_boundary_members_outside_the_selection_refuse():
    tables = mock_tables((2, 1, 1))
    # The last person's family reaches the first household even though that row lies outside the selection.
    tables["person"].loc[3, "person_family_id"] = tables[
        "person"
    ].person_family_id.iloc[0]
    memberships, groups = membership_inputs(tables)
    with pytest.raises(pilot.EngineeringPilotError, match="another household"):
        pilot._select_rows(groups["household"], memberships, groups, max_households=1)


def test_fingerprints_preserve_dtype_order_categories_and_object_types():
    source = pd.DataFrame(
        {
            "x": np.asarray([0.0, -0.0]),
            "c": pd.Categorical(
                ["a", "b"], categories=["a", "b", "unused"], ordered=True
            ),
        }
    )
    digest = pilot._frame_inventory({"person": source})
    changes = [
        source.iloc[::-1].reset_index(drop=True),
        source[["c", "x"]],
        source.assign(x=source.x.astype("float32")),
        source.assign(c=source.c.cat.reorder_categories(["b", "a", "unused"])),
        source.assign(c=source.c.cat.as_unordered()),
    ]
    assert all(
        pilot._frame_inventory({"person": changed}) != digest for changed in changes
    )
    assert pilot._array_digest(np.asarray([1], dtype=object)) != pilot._array_digest(
        np.asarray([True], dtype=object)
    )
    assert pilot._array_digest(
        np.asarray([b"one"], dtype=object)
    ) != pilot._array_digest(np.asarray(["one"], dtype=object))
    with pytest.raises(pilot.EngineeringPilotError, match="unsupported stored scalar"):
        pilot._array_digest(np.asarray([object()], dtype=object))


def test_known_reviewed_changes_keep_raw_refusal_false_and_untouched():
    changes = {
        recert.CHANGE_IGNORED: [
            {"column": "first_home_mortgage_interest", "tables": ["tax_unit"]}
        ],
        recert.CHANGE_READ: [{"column": "tax_unit_role_input", "tables": ["person"]}],
    }
    raw = {
        "passed": False,
        "refused": ["first_home_mortgage_interest"],
        "failures": ["MOCK raw refusal"],
        "register_sha256": "MOCK",
    }
    before = copy.deepcopy((changes, raw))
    result = pilot._reviewed_columns(changes, raw)
    assert result == {
        "reviewed_changes_accepted": True,
        "changes": changes,
        "raw_verdict": raw,
    }
    assert result["raw_verdict"]["passed"] is False
    assert (changes, raw) == before


@pytest.mark.parametrize(
    "column,direction",
    [
        ("unreviewed_input", recert.CHANGE_IGNORED),
        ("first_home_mortgage_interest", recert.CHANGE_READ),
    ],
)
def test_unreviewed_or_wrong_direction_refuses(column, direction):
    with pytest.raises(pilot.EngineeringPilotError):
        pilot._reviewed_columns(
            {direction: [{"column": column, "tables": ["person"]}]}, {"passed": False}
        )


def test_aggregate_guard_checks_all_branches_and_refuses_existing_cache(monkeypatch):
    seen = []
    simulation = SimpleNamespace()

    def known(sim, variables, **kwargs):
        seen.append((sim, variables, kwargs))
        return {("medicaid_slcsp_state_denominator", "2024")}

    monkeypatch.setattr(recert, "engine_known_periods", known)
    with pytest.raises(
        pilot.EngineeringPilotError, match="population/state aggregates"
    ):
        pilot._aggregate_guard(simulation)
    assert seen == [
        (
            simulation,
            recert.POPULATION_AGGREGATE_VARIABLES,
            {"walk_branches": True, "skip_unknown": True},
        )
    ]


@pytest.mark.parametrize("damage", ["existing", "symlink", "input_root"])
def test_output_refuses_before_runtime(damage, tmp_path, monkeypatch):
    root = tmp_path / "inputs"
    root.mkdir()
    output = tmp_path / "result.json"
    if damage == "existing":
        output.write_text("KEEP")
    elif damage == "symlink":
        output.symlink_to(tmp_path / "absent")
    else:
        output = root / "result.json"
    monkeypatch.setattr(
        scorer_tool, "scoring_runtime", lambda *_: pytest.fail("runtime reached")
    )
    with pytest.raises((ValueError, OSError)):
        pilot.probe_candidate(
            release_dir=root, artifact_root=root, output=output, compatibility_wheels=()
        )
    if damage == "existing":
        assert output.read_text() == "KEEP"


def test_runtime_authentication_precedes_candidate_reader(tmp_path, monkeypatch):
    def refuse(_):
        raise ValueError("MOCK authentic-runtime refusal")

    monkeypatch.setattr(scorer_tool, "scoring_runtime", refuse)
    monkeypatch.setattr(
        recert,
        "validate_recertification_candidate",
        lambda *_a, **_kw: pytest.fail("candidate read before auth"),
    )
    with pytest.raises(ValueError, match="authentic-runtime refusal"):
        pilot.probe_candidate(
            release_dir=tmp_path / "release",
            artifact_root=tmp_path / "artifacts",
            output=tmp_path / "result.json",
            compatibility_wheels=(),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("parent", {"release_type": "source_enrichment"}),
        ("compatibility", {"status": "passed"}),
    ],
)
def test_unsupported_parent_context_refuses_before_payload(
    field, value, tmp_path, monkeypatch
):
    report = {
        "schema_version": 1,
        "parent": {"release_type": "calibration"},
        "compatibility": {"status": "pending"},
    }
    report[field] = value
    monkeypatch.setattr(
        scorer_tool, "scoring_runtime", lambda _: {"MOCK_METADATA_ONLY": True}
    )
    monkeypatch.setattr(recert, "require_trusted_bytecode", lambda: None)
    monkeypatch.setattr(
        recert, "validate_recertification_candidate", lambda *_a, **_kw: report
    )
    monkeypatch.setattr(
        pilot, "_input_files", lambda *_: pytest.fail("payload reached")
    )
    with pytest.raises(
        pilot.EngineeringPilotError, match="pending schema1 calibration"
    ):
        pilot.probe_candidate(
            release_dir=tmp_path / "release",
            artifact_root=tmp_path / "artifacts",
            output=tmp_path / "result.json",
            compatibility_wheels=(),
        )


def test_exclusive_receipt_rejects_overwrite_and_nonfinite(tmp_path):
    output = tmp_path / "result.json"
    pilot._write_receipt(output, {"kind": pilot.PILOT_KIND})
    with pytest.raises(FileExistsError):
        pilot._write_receipt(output, {"kind": "wrong"})
    with pytest.raises(ValueError):
        pilot._write_receipt(tmp_path / "bad.json", {"value": float("nan")})


def test_partial_engineering_receipt_cannot_satisfy_ordinary_fullscore_consumer():
    document = {
        "schema_version": 1,
        "kind": pilot.PILOT_KIND,
        "population_estimate": False,
        "selection": {"actual_households": 2},
        "measurement": {"household_metrics": {}},
    }
    failures = recert._score_document_failures(
        document,
        dataset_sha256="a" * 64,
        definitions={"metrics": []},
        label="MOCK pilot",
        diagnostics_sha256="b" * 64,
        spm={"geography_kind": "county"},
        expected_households=3,
    )
    assert any("both their household and their state passes" in f for f in failures)
    assert any(recert.SCORES_KIND in f for f in failures)


class _MockStore:
    """In-memory HDFStore transport; no H5 library or file access."""

    def __init__(self, tables, *, period=2024, codec=False):
        self.tables = tables
        self.period = period
        self.codec = codec
        self.reads = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def keys(self):
        return ["/" + key for key in (*self.tables, "_time_period")]

    def __getitem__(self, key):
        assert key == "_time_period"
        return pd.Series([self.period])

    def get_storer(self, key):
        attrs = SimpleNamespace(
            _microcosm_frame_table_codec="MOCK_UNSUPPORTED" if self.codec else None
        )
        return SimpleNamespace(
            attrs=attrs,
            nrows=len(self.tables[key]),
            table=SimpleNamespace(rowsize=1024),
        )

    def select(self, key, *, start, stop):
        self.reads.append((key, start, stop))
        return self.tables[key].iloc[start:stop].copy()


def test_chunk_reader_preserves_scattered_group_rows_without_full_entity_reads(
    tmp_path, monkeypatch
):
    from tools import sample_us_export_households as sampler

    tables = mock_tables((2, 3, 1, 2, 1))
    for entity in ("tax_unit", "spm_unit", "family", "marital_unit"):
        tables[entity] = tables[entity].iloc[::-1].reset_index(drop=True)
    transport = _MockStore(tables)
    monkeypatch.setattr(pd, "HDFStore", lambda *_a, **_kw: transport)
    # Small test-local chunk geometry exercises the real shared reader across boundaries.
    monkeypatch.setattr(sampler, "MINIMUM_CHUNK_ROWS", 2)
    monkeypatch.setattr(pilot, "_CHUNK_BYTES", 1)
    selected, year = pilot._read_selected_tables(
        tmp_path / "MOCK.h5", max_households=2, expected_period=2024
    )
    assert year == 2024
    memberships, groups = membership_inputs(tables)
    _, masks = pilot._select_rows(
        groups["household"], memberships, groups, max_households=2
    )
    for entity, frame in selected.items():
        pd.testing.assert_frame_equal(
            frame, tables[entity].loc[masks[entity]].reset_index(drop=True)
        )
    assert transport.reads
    assert all(stop - start <= 2 for _, start, stop in transport.reads)
    assert any(start >= 2 for _, start, _ in transport.reads)


@pytest.mark.parametrize("period", [2023, True, "2024", 2024.0])
def test_period_refuses_before_row_reads(period, tmp_path, monkeypatch):
    transport = _MockStore(mock_tables(), period=period)
    monkeypatch.setattr(pd, "HDFStore", lambda *_a, **_kw: transport)
    with pytest.raises(pilot.EngineeringPilotError, match="period|year"):
        pilot._read_selected_tables(
            tmp_path / "MOCK.h5", max_households=2, expected_period=2024
        )
    assert not transport.reads


def test_unsupported_frame_codec_refuses_before_row_reads(tmp_path, monkeypatch):
    transport = _MockStore(mock_tables(), codec=True)
    monkeypatch.setattr(pd, "HDFStore", lambda *_a, **_kw: transport)
    with pytest.raises(ValueError, match="codec"):
        pilot._read_selected_tables(
            tmp_path / "MOCK.h5", max_households=2, expected_period=2024
        )
    assert not transport.reads
