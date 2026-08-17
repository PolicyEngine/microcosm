"""Frame carrier foundations for the #612 swap.

Three acceptance facts, each load-bearing for the carrier migration:
construction and the UK residue validation; the new writer/loader pair
round-tripping payload and provenance; and payload **stability across
write -> load -> write generations** — the shadow writer is deleted, so
old-vs-new identity is no longer provable in CI and was settled instead by
the credentialed acceptance run recorded on #612 (loader->writer round trip
of the certified candidate, compared with tools/compare_uk_h5_payload.py).
The SPI replacement test is the CI-side kill-shot for the riskiest unknown:
whether the mid-pipeline tables (cloned households, rebuilt ids) satisfy
Frame's sorted-group-id and bidirectional-membership invariants.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.national_build import (
    load_uk_national_frame,
    write_uk_national_frame,
)
from microcosm.build.uk_runtime.national_frame import (
    UK_NATIONAL_SCHEMA,
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.rowwise_dataset import (
    read_uk_single_year_weight_metadata,
)
from microcosm.build.uk_runtime.spi_support import (
    create_uk_spi_support_tables,
    replace_uk_spi_support_tables,
)
from microcosm.frame import CONSERVE_MASS, Frame, MassChangeRecord, WeightKind, Weights


def person_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3, 4], dtype="int64"),
            "person_benunit_id": np.asarray([101, 201, 201, 301], dtype="int64"),
            "person_household_id": np.asarray([11, 12, 12, 13], dtype="int64"),
            "employment_income": [1000.0, 2000.0, 3000.0, 4000.0],
        }
    )


def benunit_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "benunit_id": np.asarray([101, 201, 301], dtype="int64"),
            "would_claim_uc": [True, False, False],
        }
    )


def household_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "household_id": np.asarray([11, 12, 13], dtype="int64"),
            "household_weight": [10.0, 20.0, 30.0],
            "region": ["LONDON", "WALES", "SCOTLAND"],
        }
    )


def _mass_log(total: float) -> tuple[MassChangeRecord, ...]:
    return (
        MassChangeRecord(
            entity="household",
            old_total=total,
            new_total=total,
            declared_factor=1.0,
            reason="Toy reviewed record.",
        ),
    )


def _frame(**overrides) -> Frame:
    arguments: dict = {
        "person": person_frame(),
        "benunit": benunit_frame(),
        "household": household_frame(),
        "time_period": "2023",
        "weight_kind": WeightKind.DESIGN,
        "mass_log": _mass_log(60.0),
    }
    arguments.update(overrides)
    return uk_national_frame(**arguments)


def test_construction_accessors_and_residue_validation() -> None:
    frame = _frame()

    assert tuple(frame.entities) == UK_NATIONAL_SCHEMA.entities
    assert uk_time_period(frame) == "2023"
    assert uk_household_weight_kind(frame) is WeightKind.DESIGN
    assert frame.mass_log == _mass_log(60.0)
    np.testing.assert_array_equal(
        frame.weights_for("household").values, np.array([10.0, 20.0, 30.0])
    )
    assert "household_weight" not in frame.table("household")
    validate_uk_national_frame(frame)


def test_construction_requires_the_exported_weight_column() -> None:
    with pytest.raises(ValueError, match="household_weight"):
        _frame(household=household_frame().drop(columns=["household_weight"]))


def test_construction_enforces_frame_linkage_invariants() -> None:
    unsorted = household_frame().iloc[[2, 0, 1]].reset_index(drop=True)
    with pytest.raises(ValueError):
        _frame(household=unsorted)

    orphaned = pd.concat(
        [
            benunit_frame(),
            pd.DataFrame({"benunit_id": [901], "would_claim_uc": [False]}),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError):
        _frame(benunit=orphaned)


def test_validate_rejects_exported_weight_column_on_the_carrier() -> None:
    # Only reachable by direct Frame construction; the canonical constructor
    # consumes household_weight into the typed vector and strips the column.
    frame = Frame(
        tables={
            "person": person_frame(),
            "benunit": benunit_frame(),
            "household": household_frame(),
        },
        schema=UK_NATIONAL_SCHEMA,
        weights={
            "household": Weights(
                values=np.array([1.0, 2.0, 3.0]), kind=WeightKind.DESIGN
            )
        },
        metadata={"time_period": "2023"},
    )
    with pytest.raises(ValueError, match="must not persist exported weight"):
        validate_uk_national_frame(frame)


def test_validate_rejects_mass_log_total_disagreement() -> None:
    frame = _frame(
        mass_log=(
            MassChangeRecord(
                entity="household",
                old_total=60.0,
                new_total=999.0,
                declared_factor=None,
                reason="Disagreeing record.",
            ),
        )
    )
    with pytest.raises(ValueError, match="MassChangeRecord"):
        validate_uk_national_frame(frame)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda f: f.table("person").__setitem__("person_household_id", 999),
        lambda f: f.table("person").loc.__setitem__((0, "person_id"), 2),
        lambda f: f.table("benunit").loc.__setitem__((0, "benunit_id"), 201),
        lambda f: f.table("household").sort_values(
            "household_id", ascending=False, inplace=True, ignore_index=True
        ),
        lambda f: f.table("person").__setitem__("region", "LONDON"),
        lambda f: f.table("person").drop(columns=["person_benunit_id"], inplace=True),
    ],
    ids=[
        "dangling_household_membership",
        "duplicate_person_id",
        "duplicate_benunit_id",
        "unsorted_household_ids",
        "cross_entity_column_collision",
        "missing_membership_column",
    ],
)
def test_validate_fails_closed_on_post_construction_corruption(corrupt) -> None:
    """Frame.table returns stored internals; a stage mutating them in place
    must be caught at the next seam, exactly as the retired shadow-carrier
    validator caught it — never validated through and shipped."""

    frame = _frame()
    corrupt(frame)
    with pytest.raises((ValueError, KeyError)):
        validate_uk_national_frame(frame)


def test_validate_rejects_non_household_typed_weights() -> None:
    """The staging artifact exports household_weight alone; a frame carrying
    other typed weights would materialize reserved columns the UK loader
    itself rejects."""

    frame = Frame(
        tables={
            "person": person_frame(),
            "benunit": benunit_frame(),
            "household": household_frame(),
        },
        schema=UK_NATIONAL_SCHEMA,
        weights={
            "household": Weights(
                values=np.array([10.0, 20.0, 30.0]), kind=WeightKind.DESIGN
            ),
            "person": Weights(
                values=np.array([1.0, 1.0, 1.0, 1.0]), kind=WeightKind.DESIGN
            ),
        },
        metadata={"time_period": "2023"},
    )
    with pytest.raises(ValueError, match="household typed weights only"):
        validate_uk_national_frame(frame)


def test_weight_only_update_keeps_the_carrier_columnless() -> None:
    frame = _frame()

    updated = frame.with_weights(
        "household",
        Weights(values=np.array([20.0, 10.0, 30.0]), kind=WeightKind.DESIGN),
        mass=CONSERVE_MASS,
    )
    assert "household_weight" not in updated.table("household")
    validate_uk_national_frame(updated)


def test_write_load_round_trip_with_provenance(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    frame = _frame(weight_kind=WeightKind.IMPORTANCE)
    path = tmp_path / "staging.h5"

    written = write_uk_national_frame(frame, path)
    loaded, provenance = load_uk_national_frame(written)

    for entity in ("person", "benunit", "household"):
        pd.testing.assert_frame_equal(
            loaded.table(entity), frame.table(entity), check_dtype=True
        )
    assert uk_time_period(loaded) == "2023"
    assert uk_household_weight_kind(loaded) is WeightKind.IMPORTANCE
    assert loaded.mass_log == frame.mass_log
    assert provenance.source_h5 == written.resolve()
    assert provenance.fingerprint.size_bytes == written.stat().st_size

    stored_kind, stored_mass_log = read_uk_single_year_weight_metadata(written)
    assert stored_kind is WeightKind.IMPORTANCE
    assert stored_mass_log == frame.mass_log


def test_frame_writer_payload_is_stable_across_generations(
    tmp_path: Path,
) -> None:
    """The payload survives a write -> load -> write round trip unchanged.

    Payload identity, deliberately not byte identity: HDF5 stamps write
    times, so two runs of the *same* writer already differ in bytes. The
    shadow-writer comparison arm retired with the shadow carrier; this pins
    the same keys/column-order/dtype/value/attr surface generation to
    generation, which is what downstream readers actually depend on.
    """

    pytest.importorskip("tables")
    h5py = pytest.importorskip("h5py")
    mass_log = _mass_log(60.0)
    frame = _frame(weight_kind=WeightKind.IMPORTANCE, mass_log=mass_log)
    old_path = write_uk_national_frame(frame, tmp_path / "old.h5")
    reloaded, _provenance = load_uk_national_frame(old_path)
    new_path = write_uk_national_frame(reloaded, tmp_path / "new.h5")

    with (
        pd.HDFStore(old_path, mode="r") as old_store,
        pd.HDFStore(new_path, mode="r") as new_store,
    ):
        assert list(old_store.keys()) == list(new_store.keys())
        for key in old_store.keys():
            old_table = old_store[key]
            new_table = new_store[key]
            if isinstance(old_table, pd.DataFrame):
                assert list(old_table.columns) == list(new_table.columns)
                pd.testing.assert_frame_equal(old_table, new_table, check_dtype=True)
            else:
                pd.testing.assert_series_equal(old_table, new_table)

    with (
        h5py.File(old_path, mode="r") as old_file,
        h5py.File(new_path, mode="r") as new_file,
    ):
        old_attrs = {name: old_file.attrs[name] for name in old_file.attrs}
        new_attrs = {name: new_file.attrs[name] for name in new_file.attrs}
        assert list(old_attrs) == list(new_attrs)
        assert old_attrs == new_attrs


def test_spi_replacement_output_constructs_a_frame() -> None:
    """The mid-pipeline kill-shot: stage 2's output must be frameable.

    The SPI replacement drops the dead synthetic channel, rebuilds cloned
    households/benunits with multiplied ids, and reallocates mass — exactly
    where unsorted group ids or one-directional membership would first
    appear.
    """

    household = pd.DataFrame(
        {
            "household_id": np.arange(1, 9, dtype="int64"),
            "household_weight": np.arange(10.0, 90.0, 10.0),
            "region": ["LONDON", "LONDON", "WALES", "WALES"] * 2,
            "clone_index": [0] * 4 + [1] * 4,
            "household_is_capital_gains_clone": [False, False, True, True] * 2,
        }
    )
    person = pd.DataFrame(
        {
            "person_id": np.arange(101, 109, dtype="int64"),
            "person_household_id": np.arange(1, 9, dtype="int64"),
            "person_benunit_id": np.arange(201, 209, dtype="int64"),
            "employment_income": np.arange(1_000.0, 9_000.0, 1_000.0),
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.arange(201, 209, dtype="int64")})
    dead = create_uk_spi_support_tables(
        person=person,
        benunit=benunit,
        household=household,
        selected_household_ids=(1, 3, 5, 7),
        source_year=2023,
    )

    result = replace_uk_spi_support_tables(
        person=dead.person,
        benunit=dead.benunit,
        household=dead.household,
        seed=7,
        source_year=2023,
    )

    frame = uk_national_frame(
        person=result.person,
        benunit=result.benunit,
        household=result.household,
        time_period="2023",
        weight_kind=result.household_weight_kind,
        mass_log=result.mass_log,
    )
    validate_uk_national_frame(frame)
    assert uk_household_weight_kind(frame) is WeightKind.IMPORTANCE
