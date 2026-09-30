"""Bookkeeping in the support-mix bake-off driver (tools/bakeoff_us_support_mix.py).

Invariants:

- SPM flag: a household is flagged exactly when one of its SPM units has no
  member the pinned engine counts as an adult, and the rule refuses a person
  table without household-role columns rather than falling back to age;
- staleness: an arm's receipt depends only on the inputs its arm reads, the
  concept digest moves with part contents (not just sizes), a concept part is
  reusable only if built from the current shards' source h5, and saved
  weights are re-scored only for the same rows and inputs;
- grid: ACS-only arms carry no years, so a grid on newer CPS years keeps them;
- reporting: replicate tables cover only groups with more than one draw, and
  a report with missing receipts is refused unless asked for.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


@pytest.fixture(scope="module")
def tool():
    path = _TEST_PATHS.repository / "tools" / "bakeoff_us_support_mix.py"
    spec = importlib.util.spec_from_file_location("bakeoff_us_support_mix", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tables(people):
    """``people``: (household, spm unit, age, head, spouse) per person."""
    person = pd.DataFrame(people, columns=["person_household_id", "person_spm_unit_id", "age",
                                           "is_household_head", "is_household_spouse"])
    household = pd.DataFrame({"household_id": sorted(set(person["person_household_id"]))})
    return {"household": household, "person": person}


def test_spm_flag_marks_units_without_a_classified_adult(tool) -> None:
    tables = _tables([
        (1, 10, 40, True, False),    # adult
        (2, 20, 16, True, False),    # 15-17 head counts as an adult
        (3, 30, 16, False, False),   # 15-17 non-head, no adult in unit
        (4, 40, 35, True, False),    # household 4: one unit with an adult...
        (4, 41, 10, False, False),   # ...and one with only a child
        (5, 50, 14, True, False),    # under 15 never counts, even as head
    ])
    flags = tool.spm_zero_adult_households(tables)
    assert flags.tolist() == [False, False, True, True, True]


def test_spm_flag_refuses_a_table_without_role_columns(tool) -> None:
    tables = _tables([(1, 10, 16, True, False)])
    tables["person"] = tables["person"].drop(columns=["is_household_head", "is_household_spouse"])
    with pytest.raises(ValueError, match="neither"):
        tool.spm_zero_adult_households(tables)


def test_spm_flags_leaves_an_unchanged_row_index_alone(tool, tmp_path) -> None:
    tables = _tables([(1, 10, 40, True, False), (2, 20, 16, False, False)])
    shard_dir = tmp_path / "shards" / "acs"
    shard_dir.mkdir(parents=True)
    with open(shard_dir / "shard_000.pkl", "wb") as handle:
        pickle.dump({"tables": tables}, handle)
    rows = pd.DataFrame({"household_id": [1, 2], "shard": [0, 0], "group_quarters": [False, True],
                         "spm_zero_adult_unit": [False, True]})
    rows.to_parquet(tmp_path / "rows_acs.parquet", index=False)
    before = (tmp_path / "rows_acs.parquet").read_bytes()
    tool.do_spm_flags(argparse.Namespace(work=tmp_path))
    assert (tmp_path / "rows_acs.parquet").read_bytes() == before
    receipt = json.loads((tmp_path / "spm_flags.json").read_text())
    assert receipt["spm_zero_adult_households"] == 1 == receipt["spm_zero_adult_in_group_quarters"]
    rows.assign(spm_zero_adult_unit=False).to_parquet(tmp_path / "rows_acs.parquet", index=False)
    tool.do_spm_flags(argparse.Namespace(work=tmp_path))
    assert pd.read_parquet(tmp_path / "rows_acs.parquet")["spm_zero_adult_unit"].tolist() == [False, True]


@pytest.mark.parametrize("newest", [2024, 2025])
def test_arm_inputs_cover_exactly_what_each_arm_reads(tool, newest) -> None:
    for arm in tool.arm_grid(True, newest):
        keys = set(tool.arm_input_keys(arm))
        assert {"targets", "truth", "epochs"} <= keys
        assert ({"rows_cps", "concepts_cps"} <= keys) == bool(arm.cps_income_years)
        assert ({"rows_cps", "concepts_cps"} & keys == set()) == (not arm.cps_income_years)
        assert ({"rows_acs", "concepts_acs"} <= keys) == (arm.budget is not None)


def test_acs_only_arms_survive_a_newer_cps_grid(tool) -> None:
    assert tool.year_sets(2025) == ((2025,), (2024, 2025), (2023, 2024, 2025))
    old = {a.label for a in tool.arm_grid(True, 2024)}
    new = {a.label for a in tool.arm_grid(True, 2025)}
    assert len(old) == len(new) == 30
    assert old & new == {label for label in old if ".cps0." in label}


def test_inputs_current_compares_only_the_arms_inputs(tool) -> None:
    current = {"targets": "t", "truth": "u", "epochs": 1500}
    assert tool.inputs_current({**current, "rows_cps": "old"}, current)
    assert not tool.inputs_current({**current, "targets": "x"}, current)
    assert not tool.inputs_current({"targets": "t", "truth": "u"}, current)
    assert not tool.inputs_current(None, current)


def test_rescore_needs_the_same_rows_and_inputs(tool) -> None:
    inputs = {"targets": "t", "epochs": 1500}
    previous = {"rows_digest": "r", "inputs": inputs}
    assert tool.rescore_refusal(previous, inputs, "r") is None
    assert "rows digest" in tool.rescore_refusal({"inputs": inputs}, inputs, "r")
    assert "rows digest" in tool.rescore_refusal(previous, inputs, "other")
    assert "inputs" in tool.rescore_refusal(previous, {**inputs, "targets": "new"}, "r")


def _write_part(directory, name, values, source_sha=None, rows=None):
    directory.mkdir(parents=True, exist_ok=True)
    sp.save_npz(directory / f"{name}.npz", sp.csr_matrix(np.asarray(values, dtype=np.float32)))
    if rows is not None:
        np.save(directory / f"{name}.rows.npy", np.asarray(rows, dtype=np.int64))
    if source_sha is not None:
        (directory / f"{name}.json").write_text(json.dumps({"source_h5_sha256": source_sha}))


def test_concept_digest_moves_with_contents_of_the_same_size(tool, tmp_path) -> None:
    directory = tmp_path / "concepts" / "acs"
    _write_part(directory, "shard_000", [[1.0, 0.0], [0.0, 2.0]], rows=[0, 1])
    first = tool._concept_digest(tmp_path, "acs")
    size = (directory / "shard_000.npz").stat().st_size
    _write_part(directory, "shard_000", [[3.0, 0.0], [0.0, 4.0]], rows=[0, 1])
    assert (directory / "shard_000.npz").stat().st_size == size
    second = tool._concept_digest(tmp_path, "acs")
    assert second != first
    _write_part(directory, "shard_000", [[3.0, 0.0], [0.0, 4.0]], rows=[1, 0])
    assert tool._concept_digest(tmp_path, "acs") != second


def test_stale_parts_are_those_not_built_from_the_current_source(tool, tmp_path) -> None:
    directory = tmp_path / "parts"
    _write_part(directory, "shard_000", [[1.0]], source_sha="new")
    _write_part(directory, "shard_001", [[1.0]], source_sha="old")
    _write_part(directory, "shard_002", [[1.0]])
    assert [p.name for p in tool.stale_concept_parts(directory, "new")] == ["shard_001.npz", "shard_002.npz"]


def _materialize_args(work, replace_stale):
    return argparse.Namespace(work=work, source="cps", replace_stale=replace_stale, rank_range=None,
                              only=None, summary=None, batch=5_000, min_available_gb=0.0)


def test_materialize_refuses_then_replaces_stale_parts(tool, tmp_path) -> None:
    (tmp_path / "concept_specs.pkl").write_bytes(pickle.dumps([]))
    (tmp_path / "shards" / "cps").mkdir(parents=True)
    (tmp_path / "shards" / "cps" / "shard.json").write_text(json.dumps({"h5_sha256": "new"}))
    pd.DataFrame({"shard": pd.Series([], dtype=np.int64)}).to_parquet(tmp_path / "rows_cps.parquet")
    concepts = tmp_path / "concepts" / "cps"
    _write_part(concepts, "shard_000", [[1.0]], source_sha="old", rows=[0])
    with pytest.raises(SystemExit, match="replace-stale"):
        tool.do_materialize(_materialize_args(tmp_path, replace_stale=False))
    assert (concepts / "shard_000.npz").exists()
    tool.do_materialize(_materialize_args(tmp_path, replace_stale=True))
    assert not list(concepts.glob("shard_000*"))


def test_replicate_tables_cover_only_groups_with_several_draws(tool) -> None:
    table = pd.DataFrame({
        "product": ["national"] * 5,
        "arm": ["b300k.cps2024.acs100.s0", "b300k.cps2024.acs100.s1", "b300k.cps2024.acs100.s2",
                "b300k.cps2023-2024.acs100.s0", "b600k.cps2024.acs100.s0"],
        "cps_years": [1, 1, 1, 2, 1], "acs_fill_share": [1.0] * 5, "seed": [0, 1, 2, 0, 0],
        "x.state": [0.1, 0.2, 0.3, 0.9, 0.9],
    })
    spread, mean = tool.replicate_summary(table)
    assert list(mean.index) == [("national", 1)] == list(spread.index)
    assert mean.loc[("national", 1), "x.state"] == pytest.approx(0.2)
    assert spread.loc[("national", 1), "x.state"] == pytest.approx(0.1)


def test_report_refuses_missing_receipts(tool, tmp_path) -> None:
    (tmp_path / "arms").mkdir()
    args = argparse.Namespace(work=tmp_path, out=tmp_path / "report", newest_year=2024, allow_missing=False)
    with pytest.raises(SystemExit, match="60 expected receipts missing"):
        tool.do_report(args)
