"""Bookkeeping in the support-mix bake-off driver (tools/bakeoff_us_support_mix.py).

Invariants:

- SPM flag: a household is flagged exactly when one of its SPM units has no
  member the pinned engine counts as an adult; the flag reads a stored
  ``is_spm_independent_minor_role`` that holds a value on every row, uses the
  role's formula when that column is absent or all null, and refuses a partly
  null column and a person table without household-role columns rather than
  falling back to age;
- staleness: an arm's receipt depends only on the inputs and options its arm
  reads, the concept digest moves with part contents (not just sizes) and
  with the concept specs and measure list, a concept part is reusable only if
  built from the current shards' source h5, concept specs and null-fill
  summary, and saved weights are re-scored only for the same rows and inputs;
- partcheck: a column fails when either side holds a nonfinite value or its
  scaled difference exceeds the tolerance, and a check refuses an unknown or
  empty part selection, a sample under one household and a negative or
  nonfinite tolerance; ``do_partcheck`` applies both rules to the parts it
  re-materializes;
- grid: ACS-only arms carry no years, so a grid on newer CPS years keeps them;
- reporting: a report reads only current receipts of its own grid,
  replicate tables cover only groups with more than one draw, a report
  with missing or stale receipts is refused unless asked for, and
  ``--allow-stale`` alone leaves stale receipts out without counting them as
  missing.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from types import SimpleNamespace

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


def test_spm_flag_follows_a_stored_role_column(tool) -> None:
    tables = _tables([
        (1, 10, 16, True, False),    # a head the stored role leaves a minor
        (2, 20, 16, False, False),   # a non-head the stored role makes an adult
    ])
    person = tables["person"]
    person["is_spm_independent_minor_role"] = [False, True]
    assert tool.spm_zero_adult_households(tables).tolist() == [True, False]
    person["is_spm_independent_minor_role"] = [None, None]
    assert tool.spm_zero_adult_households(tables).tolist() == [False, True]
    person["is_spm_independent_minor_role"] = [True, None]
    with pytest.raises(ValueError, match="partially null"):
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
    def refuse(*_args, **_kwargs):
        raise AssertionError("an unchanged row index was rewritten")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pd.DataFrame, "to_parquet", refuse)
        tool.do_spm_flags(argparse.Namespace(work=tmp_path))
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
        assert tool.arm_inputs(dict.fromkeys(keys, "d"), arm)["exclude_gq"] is False


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
    # Arm options: a receipt that predates recording them was solved with the default.
    assert tool.inputs_current(current, {**current, "exclude_gq": False})
    assert not tool.inputs_current(current, {**current, "exclude_gq": True})
    assert not tool.inputs_current({**current, "exclude_gq": True}, {**current, "exclude_gq": False})


def test_rescore_needs_the_same_rows_and_inputs(tool) -> None:
    inputs = {"targets": "t", "epochs": 1500}
    previous = {"rows_digest": "r", "inputs": inputs}
    assert tool.rescore_refusal(previous, inputs, "r") is None
    assert "rows digest" in tool.rescore_refusal({"inputs": inputs}, inputs, "r")
    assert "rows digest" in tool.rescore_refusal(previous, inputs, "other")
    assert "inputs" in tool.rescore_refusal(previous, {**inputs, "targets": "new"}, "r")


def _write_part(directory, name, values, receipt=None, rows=None):
    directory.mkdir(parents=True, exist_ok=True)
    sp.save_npz(directory / f"{name}.npz", sp.csr_matrix(np.asarray(values, dtype=np.float32)))
    if rows is not None:
        np.save(directory / f"{name}.rows.npy", np.asarray(rows, dtype=np.int64))
    if receipt is not None:
        (directory / f"{name}.json").write_text(json.dumps(receipt))


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


def test_concept_digest_moves_with_what_the_columns_mean(tool, tmp_path) -> None:
    """A recompile that leaves the parts in place (new report specs, say)
    still makes a receipt from before it stale (review r5b, finding 1)."""
    args = _report_work(tmp_path)
    directory = tmp_path / "concepts" / "acs"
    _write_part(directory, "shard_000", [[1.0, 0.0]], rows=[0])
    specs, measures = tmp_path / "concept_specs.pkl", directory / "measures.json"
    specs.write_bytes(pickle.dumps([SimpleNamespace(measure="bakeoff_concept_00000")]))
    measures.write_text(json.dumps(["bakeoff_concept_00000", "state_income_tax"]))
    arm = tool.parse_arm("b300k.cps0.acs100.s0")

    def inputs():
        return tool.arm_inputs(tool.input_digests(tmp_path, args.truth, args.epochs), arm)

    solved = inputs()
    assert tool.inputs_current(solved, inputs())
    specs.write_bytes(pickle.dumps([SimpleNamespace(measure="bakeoff_concept_00000", age_limit=17)]))
    after_specs = inputs()
    assert not tool.inputs_current(solved, after_specs)
    measures.write_text(json.dumps(["state_income_tax", "bakeoff_concept_00000"]))
    assert not tool.inputs_current(after_specs, inputs())


def test_stale_parts_are_those_not_built_from_the_current_provenance(tool, tmp_path) -> None:
    directory = tmp_path / "parts"
    current = {"source_h5_sha256": "new", "concept_specs_sha256": "specs", "summary_sha256": None}
    _write_part(directory, "shard_000", [[1.0]], receipt=current)
    _write_part(directory, "shard_001", [[1.0]], receipt={**current, "source_h5_sha256": "old"})
    _write_part(directory, "shard_002", [[1.0]], receipt={**current, "concept_specs_sha256": "older"})
    _write_part(directory, "shard_003", [[1.0]], receipt={**current, "summary_sha256": "filled"})
    _write_part(directory, "shard_004", [[1.0]], receipt={"source_h5_sha256": "new"})
    _write_part(directory, "shard_005", [[1.0]])
    stale = [p.name for p in tool.stale_concept_parts(directory, current)]
    assert stale == [f"shard_00{i}.npz" for i in range(1, 6)]
    # The concept store checks the source and specs only.
    without_summary = {k: v for k, v in current.items() if k != "summary_sha256"}
    assert "shard_003.npz" not in [p.name for p in tool.stale_concept_parts(directory, without_summary)]


def _materialize_args(work, replace_stale):
    return argparse.Namespace(work=work, source="cps", replace_stale=replace_stale, rank_range=None,
                              only=None, summary=None, batch=5_000, min_available_gb=0.0)


def _materialize_work(tmp_path):
    (tmp_path / "concept_specs.pkl").write_bytes(pickle.dumps([]))
    (tmp_path / "shards" / "cps").mkdir(parents=True)
    (tmp_path / "shards" / "cps" / "shard.json").write_text(json.dumps({"h5_sha256": "new"}))
    pd.DataFrame({"shard": pd.Series([], dtype=np.int64)}).to_parquet(tmp_path / "rows_cps.parquet")
    return tmp_path / "concepts" / "cps"


def test_materialize_refuses_then_replaces_stale_parts(tool, tmp_path) -> None:
    concepts = _materialize_work(tmp_path)
    current = tool.part_provenance(tmp_path, "cps", None)
    _write_part(concepts, "shard_000", [[1.0]], receipt={**current, "source_h5_sha256": "old"}, rows=[0])
    _write_part(concepts, "shard_001", [[1.0]], receipt=current, rows=[1])
    with pytest.raises(SystemExit, match="replace-stale"):
        tool.do_materialize(_materialize_args(tmp_path, replace_stale=False))
    assert (concepts / "shard_000.npz").exists()
    tool.do_materialize(_materialize_args(tmp_path, replace_stale=True))
    assert not list(concepts.glob("shard_000*"))
    assert (concepts / "shard_001.npz").exists()


def test_a_recompile_makes_every_part_stale(tool, tmp_path) -> None:
    concepts = _materialize_work(tmp_path)
    _write_part(concepts, "shard_000", [[1.0]], receipt=tool.part_provenance(tmp_path, "cps", None), rows=[0])
    assert not tool.stale_concept_parts(concepts, tool.part_provenance(tmp_path, "cps", None))
    (tmp_path / "concept_specs.pkl").write_bytes(pickle.dumps([SimpleNamespace(measure="bakeoff_concept_00000")]))
    assert tool.stale_concept_parts(concepts, tool.part_provenance(tmp_path, "cps", None))
    with pytest.raises(SystemExit, match="replace-stale"):
        tool.do_materialize(_materialize_args(tmp_path, replace_stale=False))


def test_partcheck_fails_nonfinite_and_out_of_tolerance_columns(tool) -> None:
    fresh = np.array([[1.0, 2.0, np.nan, np.inf, 1.0, 3.0],
                      [1.0, 2.0, 1.0, np.inf, 1.0, 3.0]])
    stored = np.array([[1.0, 2.0, np.nan, np.inf, 1.5, 3.0],
                       [1.0, 2.0 + 1e-9, 1.0, np.inf, 1.0, np.nan]])
    scaled, bad = tool.partcheck_columns(fresh, stored, 1e-5)
    assert bad.tolist() == [2, 3, 4, 5]
    assert scaled[0] == 0.0
    assert scaled[1] == pytest.approx(5e-10)
    assert scaled[4] == pytest.approx(0.5)


def test_a_nonfinite_value_fails_its_column_at_any_tolerance(tool) -> None:
    """Exhaustive over a 2 x 3 sample: one nonfinite value on either side
    fails exactly its own column, whatever the tolerance."""
    base = np.arange(6.0).reshape(2, 3)
    assert tool.partcheck_columns(base, base.copy(), 0.0)[1].size == 0
    for side in ("fresh", "stored"):
        for row in range(2):
            for col in range(3):
                for value in (np.nan, np.inf, -np.inf):
                    for tolerance in (0.0, 1e-5, 1e300):
                        fresh, stored = base.copy(), base.copy()
                        (fresh if side == "fresh" else stored)[row, col] = value
                        assert tool.partcheck_columns(fresh, stored, tolerance)[1].tolist() == [col]


def test_partcheck_refuses_unknown_and_empty_part_selections(tool, tmp_path) -> None:
    parts = [tmp_path / "shard_000.npz", tmp_path / "shard_001.npz"]
    assert tool.select_parts(parts, None) == parts
    assert tool.select_parts(parts, " shard_001 ,") == parts[1:]
    with pytest.raises(SystemExit, match="shard_007"):
        tool.select_parts(parts, "shard_000,shard_007")
    with pytest.raises(SystemExit, match="no stored parts"):
        tool.select_parts(parts, ",")
    with pytest.raises(SystemExit, match="no stored parts"):
        tool.select_parts([], None)


@pytest.mark.parametrize(("per_part", "tolerance"), [
    (0, 1e-5), (200, -1e-5), (200, float("nan")), (200, float("inf")),
])
def test_partcheck_refuses_a_bad_sample_size_or_tolerance(tool, tmp_path, per_part, tolerance) -> None:
    args = argparse.Namespace(work=tmp_path, source="cps", shards=None, parts=None, per_part=per_part,
                              tolerance=tolerance, summary=None, batch=5_000, min_available_gb=0.0)
    with pytest.raises(SystemExit, match="per-part|tolerance"):
        tool.do_partcheck(args)


class _StubFrame:
    """The slice of a shard frame ``do_partcheck`` touches: person rows and select."""

    def __init__(self, household_ids):
        self.person = pd.DataFrame({"person_household_id": list(household_ids)})

    def table(self, entity):
        assert entity == "person"
        return self.person

    def select(self, mask):
        return _StubFrame(self.person.loc[np.asarray(mask), "person_household_id"])


def _partcheck_work(tmp_path, stored_by_shard):
    """A work directory whose ACS part ``shard_NNN`` stores one row per household.

    ``stored_by_shard`` maps a shard number to its stored rows; households of
    shard ``s`` carry ids ``10 * s + i``. The measures are ``snap``, ``ssi``
    and ``state_income_tax``.
    """
    work = tmp_path / "work"
    with open(_mkdir(work) / "concept_specs.pkl", "wb") as handle:
        pickle.dump([SimpleNamespace(measure="snap"), SimpleNamespace(measure="ssi")], handle)
    (_mkdir(work / "shards" / "acs") / "shard.json").write_text(json.dumps({"h5_sha256": "h5"}))
    parts, rows = _mkdir(work / "concepts" / "acs"), []
    for shard, stored in stored_by_shard.items():
        rows += [(10 * shard + i, shard) for i in range(len(stored))]
        sp.save_npz(parts / f"shard_{shard:03d}.npz", sp.csr_matrix(np.asarray(stored, dtype=float)))
    pd.DataFrame(rows, columns=["household_id", "shard"]).to_parquet(work / "rows_acs.parquet", index=False)
    return work


def _mkdir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_partcheck(tool, work, fresh_by_household, parts=None):
    """``do_partcheck`` with the engine stubbed: re-materializing returns ``fresh_by_household``."""

    def materialize(frame, _specs, *, summary_path, batch):
        ids = frame.table("person")["person_household_id"].drop_duplicates().to_numpy()
        values = np.asarray([fresh_by_household[i] for i in ids], dtype=float)
        household = pd.DataFrame(values, columns=["snap", "ssi", "state_income_tax"])
        household.insert(0, "household_id", ids)
        return household, None, None, None

    def load(path):
        shard = int(path.stem.split("_")[1])
        return _StubFrame([i for i in fresh_by_household if i // 10 == shard])

    args = argparse.Namespace(work=work, source="acs", shards=None, summary=None, parts=parts,
                              per_part=10, tolerance=1e-6, batch=10, min_available_gb=0.0)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tool, "_materialize_frame", materialize)
        patch.setattr(tool, "_load_shard_frame", load)
        patch.setattr(tool, "wait_for_memory", lambda _gb: None)
        tool.do_partcheck(args)


def test_do_partcheck_fails_a_part_whose_fresh_values_are_nonfinite(tool, tmp_path) -> None:
    stored = {0: [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]], 1: [[5.0, 6.0, 0.0]]}
    fresh = {0: [1.0, 2.0, 0.0], 1: [3.0, 4.0, 0.0], 10: [5.0, 6.0, 0.0]}
    work = _partcheck_work(tmp_path, stored)
    _run_partcheck(tool, work, fresh)
    receipt = json.loads((work / "partcheck_acs.json").read_text())
    assert sorted(receipt["parts"]) == ["shard_000.npz", "shard_001.npz"]
    assert receipt["max_scaled_abs_diff"] == 0.0
    fresh[10] = [5.0, float("nan"), 0.0]
    with pytest.raises(SystemExit, match="disagree"):
        _run_partcheck(tool, work, fresh)
    part = json.loads((work / "partcheck_acs.json").read_text())["parts"]["shard_001.npz"]
    assert part["n_nonfinite_columns"] == 1
    assert part["columns_over_tolerance"] == ["ssi"]


def test_do_partcheck_checks_only_the_parts_it_names(tool, tmp_path) -> None:
    stored = {0: [[1.0, 2.0, 0.0]], 1: [[5.0, 6.0, 0.0]]}
    fresh = {0: [1.0, float("nan"), 0.0], 10: [5.0, 6.0, 0.0]}
    work = _partcheck_work(tmp_path, stored)
    _run_partcheck(tool, work, fresh, parts="shard_001")
    assert list(json.loads((work / "partcheck_acs.json").read_text())["parts"]) == ["shard_001.npz"]
    (work / "partcheck_acs.json").unlink()
    with pytest.raises(SystemExit, match="names no stored part"):
        _run_partcheck(tool, work, fresh, parts="shard_009")
    assert not (work / "partcheck_acs.json").exists()


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


def _report_work(tmp_path):
    for name in ("targets.parquet", "rows_cps.parquet", "rows_acs.parquet", "truth.parquet"):
        pd.DataFrame({"x": [1]}).to_parquet(tmp_path / name)
    (tmp_path / "arms" / "national").mkdir(parents=True)
    return argparse.Namespace(work=tmp_path, out=tmp_path / "report", newest_year=2024, truth=tmp_path / "truth.parquet",
                              epochs=1500, allow_missing=False, allow_stale=False)


def test_report_refuses_missing_receipts(tool, tmp_path) -> None:
    args = _report_work(tmp_path)
    with pytest.raises(SystemExit, match="60 expected receipts missing"):
        tool.do_report(args)


def test_report_skips_other_grids_and_refuses_stale_receipts(tool, tmp_path) -> None:
    args = _report_work(tmp_path)
    arms = tmp_path / "arms" / "national"
    (arms / "b300k.cps2025.acs100.s0.json").write_text(json.dumps(
        {"arm": "b300k.cps2025.acs100.s0", "product": "national", "inputs": {}}))
    with pytest.raises(SystemExit, match="60 expected receipts missing"):
        tool.do_report(args)  # the 2025-grid receipt is skipped, not counted or read
    (arms / "b300k.cps0.acs100.s0.json").write_text(json.dumps(
        {"arm": "b300k.cps0.acs100.s0", "product": "national", "inputs": {"targets": "old"}}))
    with pytest.raises(SystemExit, match="1 receipts are stale"):
        tool.do_report(args)


def test_report_with_allow_stale_alone_leaves_stale_receipts_out(tool, tmp_path) -> None:
    """Review r5b, finding 4: ``--allow-stale`` must not also need
    ``--allow-missing`` for receipts that exist but are stale."""
    args = _report_work(tmp_path)
    (tmp_path / "arms" / "local").mkdir()
    for product in ("national", "local"):
        for arm in tool.arm_grid(True, 2024):
            (tmp_path / "arms" / product / f"{arm.label}.json").write_text(json.dumps(
                {"arm": arm.label, "product": product, "inputs": {"targets": "old"}}))
    with pytest.raises(SystemExit, match="60 receipts are stale"):
        tool.do_report(args)
    args.allow_stale = True
    tool.do_report(args)
    assert (args.out / "arms.csv").exists()



def test_the_arm_solve_is_the_optimize_call_it_replaced(tool, monkeypatch) -> None:
    """The arm's own _SparseOperator through the public solve, bit for bit
    against the direct _optimize call the arm made before (microcosm#1585)."""
    import torch

    from microcosm.calibrate.solve import (
        CONSERVE_MASS,
        _optimize,
        default_target_loss_scales,
        solve_constraint_system,
    )

    monkeypatch.setattr(tool, "_SpMV", tool._spmv_class())
    rng = np.random.default_rng(1585)
    matrix = sp.random(8, 300, density=0.3, format="csr", random_state=rng) * 10.0
    w0 = rng.uniform(0.5, 2.0, 300)
    targets = (matrix @ w0) * rng.uniform(0.8, 1.2, 8)
    loss_weights = rng.uniform(0.5, 2.0, 8)
    scales = default_target_loss_scales(targets)
    operator = tool._SparseOperator(matrix)
    weights, trajectory = solve_constraint_system(
        operator,
        targets,
        w0,
        target_loss_weights=loss_weights,
        target_loss_scales=scales,
        target_loss_cap=1.0,
        mass=CONSERVE_MASS,
        max_weight_ratio=5.0,
        epochs=50,
        learning_rate=0.02,
    )
    expected_weights, expected_trajectory = _optimize(
        operator,
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(loss_weights, dtype=torch.float32),
        torch.tensor(scales, dtype=torch.float32),
        1.0,
        w0,
        epochs=50,
        learning_rate=0.02,
        conserve_mass=True,
        max_weight_ratio=5.0,
        l0_lambda=0.0,
        l2_lambda=0.0,
        target_records=None,
        init_mean=0.999,
        temperature=0.25,
    )
    assert weights.view(np.uint64).tolist() == expected_weights.view(np.uint64).tolist()
    assert trajectory.view(np.uint64).tolist() == (
        expected_trajectory.view(np.uint64).tolist()
    )
