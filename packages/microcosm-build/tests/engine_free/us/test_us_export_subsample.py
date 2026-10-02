"""The US export household subsampler (``tools/sample_us_export_households.py``).

Invariants, each checked for every generated input (Hypothesis) and on a
written H5 end to end:

- whole households: a household's persons and every group unit they reference
  enter together, no group unit is split, and nothing else enters;
- per-stratum weight conservation: the sampled weights of each stratum sum to
  its source weights (relative tolerance 1e-9);
- certainty households are always selected and keep their source weights;
- determinism given the seed, and invariance to the source's row order;
- each stratum draws ``max(1, floor(p * N))`` of its ``N`` non-certainty
  households, and ``p = 1`` selects everything at unchanged weights;
- the chunked H5 path realizes the same tables and weights as the in-memory
  ``Frame.select`` reference path (differential).
"""

# ruff: noqa: F403, F405
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from test_support.microcosm_build.us_export_subsample import *
from test_support.paths import paths_for

_SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@st.composite
def household_populations(draw):
    """Unique ids, positive weights, stratum labels and a certainty subset."""
    n = draw(st.integers(min_value=1, max_value=60))
    ids = draw(
        st.lists(
            st.integers(min_value=1, max_value=10_000),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    weights = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=1e5, allow_nan=False),
            min_size=n,
            max_size=n,
        )
    )
    labels = draw(st.lists(st.sampled_from(["a", "b", "c"]), min_size=n, max_size=n))
    certainty = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    fraction = draw(st.floats(min_value=0.01, max_value=1.0))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    return (
        np.asarray(ids, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
        np.asarray(labels, dtype=object),
        [ids[i] for i in range(n) if certainty[i]],
        fraction,
        seed,
    )


@_SETTINGS
@given(household_populations())
def test_draw_conserves_stratum_weights_and_keeps_certainty(sampler, population):
    ids, weights, labels, certain, fraction, seed = population
    draw = sampler.draw_households(
        ids, weights, labels, certain, fraction=fraction, seed=seed
    )
    source = pd.Series(weights, index=ids)
    label_of = pd.Series(labels, index=ids)
    for label in set(labels.tolist()):
        expected = float(weights[labels == label].sum())
        realized = float(draw.adjusted_weights[draw.labels == label].sum())
        assert math.isclose(realized, expected, rel_tol=1e-9), label
        record = draw.strata[label]
        assert math.isclose(record["sampled_weight_total"], expected, rel_tol=1e-9)
    selected = pd.Series(draw.adjusted_weights, index=draw.selected_ids)
    for household_id in certain:
        assert household_id in selected.index
        assert selected[household_id] == source[household_id]
    assert (draw.certainty == np.isin(draw.selected_ids, certain)).all()
    assert (label_of.reindex(draw.selected_ids).to_numpy() == draw.labels).all()
    assert np.all(np.diff(draw.selected_ids) > 0)
    for record in draw.strata.values():
        eligible = record["eligible_noncertainty_households"]
        expected_n = (
            0
            if eligible == 0
            else min(eligible, max(1, math.floor(fraction * eligible)))
        )
        assert record["drawn_noncertainty_households"] == expected_n


@_SETTINGS
@given(household_populations(), st.randoms(use_true_random=False))
def test_draw_is_deterministic_and_row_order_invariant(sampler, population, rng):
    ids, weights, labels, certain, fraction, seed = population
    first = sampler.draw_households(
        ids, weights, labels, certain, fraction=fraction, seed=seed
    )
    order = list(range(len(ids)))
    rng.shuffle(order)
    order = np.asarray(order)
    second = sampler.draw_households(
        ids[order],
        weights[order],
        labels[order],
        list(reversed(certain)),
        fraction=fraction,
        seed=seed,
    )
    assert np.array_equal(first.selected_ids, second.selected_ids)
    assert np.array_equal(first.adjusted_weights, second.adjusted_weights)
    assert first.strata == second.strata


@_SETTINGS
@given(household_populations())
def test_full_fraction_is_a_census(sampler, population):
    ids, weights, labels, certain, _, seed = population
    draw = sampler.draw_households(
        ids, weights, labels, certain, fraction=1.0, seed=seed
    )
    assert np.array_equal(draw.selected_ids, np.sort(ids))
    assert np.array_equal(
        draw.adjusted_weights, pd.Series(weights, index=ids).sort_index().to_numpy()
    )


def test_draw_refuses_bad_inputs(sampler) -> None:
    ids = np.asarray([1, 2, 3])
    weights = np.asarray([1.0, 2.0, 3.0])
    labels = np.asarray(["a", "a", "a"], dtype=object)
    for fraction in (0.0, -0.1, 1.5, math.nan, True):
        with pytest.raises(ValueError, match="fraction"):
            sampler.draw_households(ids, weights, labels, [], fraction=fraction, seed=0)
    with pytest.raises(ValueError, match="seed"):
        sampler.draw_households(ids, weights, labels, [], fraction=0.5, seed=-1)
    with pytest.raises(ValueError, match="unique"):
        sampler.draw_households(
            np.asarray([1, 1, 3]), weights, labels, [], fraction=0.5, seed=0
        )
    with pytest.raises(ValueError, match="non-negative"):
        sampler.draw_households(
            ids, np.asarray([1.0, -2.0, 3.0]), labels, [], fraction=0.5, seed=0
        )
    with pytest.raises(ValueError, match="not in the export"):
        sampler.draw_households(ids, weights, labels, [99], fraction=0.5, seed=0)


def test_seed_changes_the_draw(sampler) -> None:
    ids = np.arange(1, 201, dtype=np.int64)
    weights = np.linspace(1.0, 50.0, 200)
    labels = np.asarray(["a"] * 200, dtype=object)
    draws = {
        tuple(
            sampler.draw_households(
                ids, weights, labels, [], fraction=0.1, seed=seed
            ).selected_ids.tolist()
        )
        for seed in range(5)
    }
    assert len(draws) == 5


@st.composite
def nested_frames(draw):
    n = draw(st.integers(min_value=3, max_value=25))
    seed = draw(st.integers(min_value=0, max_value=10_000))
    return synthetic_export_frame(n, seed=seed, rare_households=(0,))


@_SETTINGS
@given(nested_frames(), st.floats(min_value=0.05, max_value=1.0), st.integers(0, 99))
def test_entity_masks_keep_whole_households(sampler, frame, fraction, seed):
    person = frame.table("person")
    household = frame.table("household")
    draw = sampler.draw_households(
        household["household_id"].to_numpy(),
        frame.weights_for("household").values,
        household["household_support_channel"].to_numpy(dtype=object),
        [],
        fraction=fraction,
        seed=seed,
    )
    group_ids = {
        entity: frame.table(entity)[f"{entity}_id"].to_numpy()
        for entity in sampler.US_GROUP_ENTITIES
    }
    masks = sampler.entity_row_masks(person, group_ids, draw.selected_ids)
    selected = set(draw.selected_ids.tolist())
    kept_people = person[masks["person"]]
    assert set(kept_people["person_household_id"]) == selected
    assert person.loc[~masks["person"], "person_household_id"].isin(selected).sum() == 0
    for entity in sampler.US_GROUP_ENTITIES:
        kept_units = set(group_ids[entity][masks[entity]].tolist())
        referenced = set(kept_people[f"person_{entity}_id"].tolist())
        assert kept_units == referenced, entity
        # No unit is split: every member of a kept unit is kept.
        members = person[person[f"person_{entity}_id"].isin(kept_units)]
        assert members.index.isin(kept_people.index).all(), entity
    sampler.assert_units_nest_in_households(kept_people)


def test_units_spanning_households_are_refused(sampler) -> None:
    person = pd.DataFrame(
        {
            "person_household_id": [1, 2],
            "person_tax_unit_id": [10, 10],
            "person_spm_unit_id": [100, 200],
            "person_family_id": [1000, 2000],
            "person_marital_unit_id": [1, 2],
        }
    )
    with pytest.raises(ValueError, match="tax_unit unit"):
        sampler.assert_units_nest_in_households(person)


def test_person_level_stratum_must_be_constant_within_a_household(sampler) -> None:
    household = pd.DataFrame(
        {"household_id": [1, 2], "household_support_channel": ["asec", "puf"]}
    )
    person = pd.DataFrame(
        {"person_household_id": [1, 1, 2], "source_year": [2022, 2022, 2023]}
    )
    labels, receipt = sampler.household_stratum_labels(
        household, person, ("household_support_channel", "source_year", "absent")
    )
    assert labels.tolist() == [
        "household_support_channel=asec|source_year=2022",
        "household_support_channel=puf|source_year=2023",
    ]
    assert receipt["used"] == ["household_support_channel", "source_year"]
    assert receipt["absent"] == ["absent"]
    person.loc[1, "source_year"] = 2024
    with pytest.raises(ValueError, match="varies within 1 household"):
        sampler.household_stratum_labels(household, person, ("source_year",))


def test_probe_carriers_below_the_threshold_are_certain(sampler) -> None:
    households = np.asarray([1, 1, 2, 3, 4])
    leaves = {
        "rare": ("person", np.asarray([0.0, 5.0, 0.0, 0.0, 0.0]), households),
        "flag": ("person", np.asarray([True, False, True, True, True]), households),
        "text": ("person", np.asarray(["x", "y", "", "z", "w"]), households),
    }
    probes = (
        SamplerProbe("rare", ("rare", "missing_leaf")),
        SamplerProbe("common", ("flag",)),
        SamplerProbe("text", ("text",)),
    )
    records, reasons = sampler.probe_carriers(
        probes, leaves, fraction=0.5, threshold=1.0
    )
    by_id = {record.probe_id: record for record in records}
    assert by_id["rare"].certainty and by_id["rare"].absent_inputs == ("missing_leaf",)
    assert by_id["rare"].carrier_household_ids.tolist() == [1]
    assert not by_id["common"].certainty  # 0.5 x 4 carriers >= 1
    assert by_id["text"].carrier_household_ids.tolist() == []  # text is not numeric
    assert not by_id["text"].certainty  # no carriers: nothing to protect
    assert reasons == {1: ["rare"]}


# ---------------------------------------------------------------------------
# The chunked H5 path, end to end
# ---------------------------------------------------------------------------


def test_sample_export_writes_a_verified_subsample(sampler, tmp_path) -> None:
    frame = synthetic_export_frame(60, seed=3, rare_households=(4, 41))
    path, receipt = sample_synthetic(sampler, tmp_path, frame, fraction=0.25, seed=0)
    written = load_table_h5(path)
    source_weights = pd.Series(
        frame.weights_for("household").values,
        index=frame.table("household")["household_id"].to_numpy(),
    )
    written_weights = pd.Series(
        written.weights_for("household").values,
        index=written.table("household")["household_id"].to_numpy(),
    )
    # The rare-input households (positions 4 and 41) are certain, at source weight.
    certain = receipt["certainty"]["household_ids"]
    assert certain == [5, 42]
    assert receipt["certainty"]["household_reasons"] == {
        "5": ["rare_keogh"],
        "42": ["rare_keogh"],
    }
    for household_id in certain:
        assert written_weights[household_id] == source_weights[household_id]
    # Strata: channel x source year, each conserving its weight total.
    assert receipt["design"]["strata_columns"]["used"] == [
        "household_support_channel",
        "source_year",
    ]
    labels, _ = sampler.household_stratum_labels(
        frame.table("household"),
        frame.table("person"),
        ("household_support_channel", "source_year"),
    )
    source_labels = pd.Series(labels, index=source_weights.index)
    for label, record in receipt["strata"].items():
        members = source_labels.index[source_labels.to_numpy() == label]
        assert math.isclose(
            float(written_weights.reindex(members).sum()),
            float(source_weights.reindex(members).sum()),
            rel_tol=1e-9,
        )
        assert math.isclose(
            record["sampled_weight_total"], record["source_weight_total"], rel_tol=1e-9
        )
    assert math.isclose(
        receipt["selection"]["household_weight_total"],
        receipt["source"]["household_weight_total"],
        rel_tol=1e-9,
    )
    # Whole households, and the verification the tool ran itself.
    sampler.assert_units_nest_in_households(written.table("person"))
    assert set(written.table("person")["person_household_id"]) == set(
        written_weights.index
    )
    assert receipt["verification"]["passed"]
    assert receipt["source"]["sha256"] == sampler.sha256_file(
        tmp_path / "export" / "populace_us_2024.h5"
    )
    assert receipt["output"]["sha256"] == sampler.sha256_file(path)
    assert (tmp_path / "export-sample" / sampler.RECEIPT_FILENAME).exists()


def test_h5_path_matches_the_frame_reference_path(sampler, tmp_path) -> None:
    """Differential: the chunked H5 path and ``Frame.select`` agree."""
    frame = synthetic_export_frame(40, seed=7, rare_households=(2,))
    path, receipt = sample_synthetic(sampler, tmp_path, frame, fraction=0.3, seed=11)
    written = load_table_h5(path)
    household = frame.table("household")
    labels, _ = sampler.household_stratum_labels(
        household, frame.table("person"), ("household_support_channel", "source_year")
    )
    draw = sampler.draw_households(
        household["household_id"].to_numpy(),
        frame.weights_for("household").values,
        labels,
        receipt["certainty"]["household_ids"],
        fraction=0.3,
        seed=11,
    )
    reference = sampler.sample_frame(frame, draw)
    for entity in US_ENTITIES:
        left = written.table(entity).reset_index(drop=True)
        right = reference.table(entity).reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
    assert np.array_equal(
        written.weights_for("household").values,
        reference.weights_for("household").values,
    )


def test_h5_sample_is_invariant_to_source_row_order(sampler, tmp_path) -> None:
    """The person table's stored order is free (group tables are id-sorted in
    any loadable export); a permuted person table selects the same households
    at the same weights and writes the same rows."""
    frame = synthetic_export_frame(40, seed=9, rare_households=(1,))
    path, first = sample_synthetic(
        sampler, tmp_path, frame, fraction=0.3, seed=5, name="ordered"
    )
    tables = {entity: frame.table(entity) for entity in US_ENTITIES}
    order = np.random.default_rng(3).permutation(len(tables["person"]))
    tables["person"] = tables["person"].iloc[order].reset_index(drop=True)
    source = tmp_path / "permuted" / "populace_us_2024.h5"
    source.parent.mkdir()
    write_tables_h5(tables, frame.weights_for("household").values, source)
    second = sampler.sample_export(
        source,
        tmp_path / "permuted-sample",
        fraction=0.3,
        seed=5,
        probes=fixture_sampler_probes(),
        write_dataset=write_table_h5,
        chunk_rows=5,
        refuse_denied=False,
    )
    assert (
        first["selection"]["selected_household_ids_sha256"]
        == second["selection"]["selected_household_ids_sha256"]
    )
    assert first["selection"]["rows"] == second["selection"]["rows"]
    assert first["strata"] == second["strata"]
    assert first["certainty"] == second["certainty"]
    left = load_table_h5(path)
    right = load_table_h5(Path(second["output"]["path"]))
    for entity in US_ENTITIES:
        key = f"{entity}_id"
        pd.testing.assert_frame_equal(
            left.table(entity).sort_values(key).reset_index(drop=True),
            right.table(entity).sort_values(key).reset_index(drop=True),
        )
    assert np.array_equal(
        left.weights_for("household").values, right.weights_for("household").values
    )


def test_sample_export_refuses_split_units_and_self_overwrite(
    sampler, tmp_path
) -> None:
    frame = synthetic_export_frame(10, seed=1)
    source = tmp_path / "src" / "populace_us_2024.h5"
    source.parent.mkdir()
    write_table_h5(frame, source)
    with pytest.raises(ValueError, match="overwrite its source"):
        sampler.sample_export(
            source,
            source.parent,
            fraction=0.5,
            seed=0,
            probes=fixture_sampler_probes(),
            write_dataset=write_table_h5,
            refuse_denied=False,
        )
    person = frame.table("person").copy()
    person.loc[person.index[-1], "person_tax_unit_id"] = person.loc[
        person.index[0], "person_tax_unit_id"
    ]
    with pd.HDFStore(str(source)) as store:
        store.put("person", person, format="table", data_columns=True)
    with pytest.raises(ValueError, match="span more than one household"):
        sampler.sample_export(
            source,
            tmp_path / "out",
            fraction=0.5,
            seed=0,
            probes=fixture_sampler_probes(),
            write_dataset=write_table_h5,
            refuse_denied=False,
        )


def test_verification_catches_a_writer_that_drops_a_column(sampler, tmp_path) -> None:
    frame = synthetic_export_frame(20, seed=2)
    source = tmp_path / "src" / "populace_us_2024.h5"
    source.parent.mkdir()
    write_table_h5(frame, source)

    def lossy_writer(sample, path, period):
        write_table_h5(sample, path, period)
        with pd.HDFStore(str(path)) as store:
            person = store["person"].drop(columns=["keogh_distributions"])
            store.put("person", person, format="table", data_columns=True)

    with pytest.raises(AssertionError, match="stored columns differ"):
        sampler.sample_export(
            source,
            tmp_path / "out",
            fraction=0.5,
            seed=0,
            probes=fixture_sampler_probes(),
            write_dataset=lossy_writer,
            refuse_denied=False,
        )


# ---------------------------------------------------------------------------
# Regression: the lost lane's "hung" dev check (2026-09-28)
# ---------------------------------------------------------------------------

#: The lost lane's dev check, run in a fresh interpreter that refuses every
#: engine import. Its fixture module once imported ``ReformCoverageProbe``,
#: which runs ``microcosm.build.us_runtime``'s package init; with
#: policyengine-us installed that builds the engine's whole tax-benefit system
#: (173 CPU-s, 1.66 GiB measured) and ran for 49+ minutes on a contended host.
_HUNG_DEV_CHECK = r"""
import importlib.abc, json, sys, time
from pathlib import Path

repo, work, prefixes = sys.argv[1], sys.argv[2], tuple(sys.argv[3:])


class RefuseEngineImports(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            raise ImportError(f"the engine-free sampler path imported {name}")
        return None


sys.meta_path.insert(0, RefuseEngineImports())
sys.path.insert(0, repo)
started = time.perf_counter()
from test_support.microcosm_build import us_export_subsample as fixtures

sampler = fixtures._load_tool(
    "sample_us_export_households", "sample_us_export_households.py"
)
frame = fixtures.synthetic_export_frame(60, seed=1, rare_households=(3, 40))
_, receipt = fixtures.sample_synthetic(sampler, Path(work), frame, fraction=0.3, seed=0)
print(
    json.dumps(
        {
            "households": receipt["selection"]["households"],
            "certainty": receipt["certainty"]["household_ids"],
            "verified": receipt["verification"]["passed"],
            "engine_modules": fixtures.loaded_engine_modules(),
            "seconds": time.perf_counter() - started,
        }
    )
)
"""


def test_the_hung_dev_check_is_engine_free_and_finishes(tmp_path) -> None:
    """The exact call that "hung" (60 households, p=0.3, seed 0) completes in
    a fresh interpreter that refuses to import policyengine-us or
    ``microcosm.build.us_runtime``: the engine-free sampler path never builds
    the engine. The timeout turns any future hang into a failure."""
    import json
    import subprocess
    import sys

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _HUNG_DEV_CHECK,
            str(paths_for("microcosm-build").repository),
            str(tmp_path),
            *ENGINE_MODULE_PREFIXES,
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["engine_modules"] == []
    assert result["verified"] is True
    # Positions 3 and 40 carry the rare input: households 4 and 41, kept at
    # certainty. The count is the per-stratum rule, independent of the RNG.
    assert result["certainty"] == [4, 41]
    assert result["households"] == 18


def test_the_cli_default_crosses_the_deny_list_boundary(
    sampler, tmp_path, monkeypatch
) -> None:
    """``refuse_denied=True`` (the CLI default) hashes the source through
    ``h5_io.refuse_denied_pool_h5`` and re-checks it with
    ``assert_h5_unchanged`` after the reads, under the sampler's consumer
    label. A stub module stands in for ``h5_io`` so the engine is not built."""
    import types

    frame = synthetic_export_frame(12, seed=4)
    source = tmp_path / "src" / "populace_us_2024.h5"
    source.parent.mkdir()
    write_table_h5(frame, source)
    calls: list[tuple[str, str]] = []
    digest = sampler.sha256_file(source)

    def refuse_denied_pool_h5(path, *, consumer):
        calls.append(("refuse", consumer))
        assert Path(path) == source.resolve()
        return digest

    def assert_h5_unchanged(path, sha256, *, consumer):
        calls.append(("unchanged", consumer))
        assert sha256 == digest

    stub = types.ModuleType("microcosm.build.us_runtime.h5_io")
    stub.refuse_denied_pool_h5 = refuse_denied_pool_h5
    stub.assert_h5_unchanged = assert_h5_unchanged
    monkeypatch.setitem(sys.modules, "microcosm.build.us_runtime.h5_io", stub)
    receipt = sampler.sample_export(
        source,
        tmp_path / "out",
        fraction=0.5,
        seed=0,
        probes=fixture_sampler_probes(),
        write_dataset=write_table_h5,
    )
    assert [kind for kind, _ in calls] == ["refuse", "unchanged"]
    assert {consumer for _, consumer in calls} == {
        "US export household subsampler (tools/sample_us_export_households.py)"
    }
    assert receipt["source"]["sha256"] == digest


# ---------------------------------------------------------------------------
# The chunked reader
# ---------------------------------------------------------------------------


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.integers(min_value=0, max_value=10_000),
    st.integers(min_value=1, max_value=60),
    st.sampled_from(["none", "some", "all", "empty"]),
    st.booleans(),
)
def test_chunked_reads_equal_a_whole_table_read(
    sampler, tmp_path_factory, seed, chunk_rows, mask_kind, subset
) -> None:
    """For any chunk size, mask and column subset, ``read_table_rows`` equals
    one whole-table ``HDFStore.select`` with the same rows and columns."""
    frame = synthetic_export_frame(25, seed=seed)
    path = tmp_path_factory.mktemp("chunks") / "export.h5"
    write_table_h5(frame, path)
    rng = np.random.default_rng(seed)
    with pd.HDFStore(str(path), mode="r") as store:
        whole = store.select("person")
        n = len(whole)
        mask = {
            "none": None,
            "some": rng.random(n) < 0.3,
            "all": np.ones(n, dtype=bool),
            "empty": np.zeros(n, dtype=bool),
        }[mask_kind]
        columns = (
            ["person_household_id", "age", "keogh_distributions"] if subset else None
        )
        read = sampler.read_table_rows(
            store, "person", columns=columns, row_mask=mask, chunk_rows=chunk_rows
        )
    expected = whole if columns is None else whole[columns]
    if mask is not None:
        expected = expected[mask]
    pd.testing.assert_frame_equal(read, expected.reset_index(drop=True))


def test_column_subset_reads_do_not_pin_whole_chunks(sampler, tmp_path) -> None:
    """A two-column read of a wide table holds about one chunk plus the kept
    columns, not every chunk's full record buffer (each kept part is copied
    out of its chunk)."""
    import tracemalloc

    rows, width = 40_000, 60
    table = pd.DataFrame(
        np.random.default_rng(0).random((rows, width)),
        columns=[f"c{index}" for index in range(width)],
    )
    path = tmp_path / "wide.h5"
    with pd.HDFStore(str(path)) as store:
        store.put("person", table, format="table", data_columns=True)
    table_bytes = rows * width * 8
    with pd.HDFStore(str(path), mode="r") as store:
        tracemalloc.start()
        try:
            read = sampler.read_table_rows(
                store, "person", columns=["c0", "c1"], chunk_rows=2_000
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    assert read.shape == (rows, 2)
    # One 2,000-row chunk is 1/20 of the table; pinning every chunk would
    # trace the whole table (and more).
    assert peak < 0.35 * table_bytes, peak / table_bytes
