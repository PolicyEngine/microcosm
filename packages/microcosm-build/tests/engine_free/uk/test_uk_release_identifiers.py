"""A UK release's identifiers stay inside the range policyengine-uk stores (microcosm#1194).

The clone stages number copies by adding a power of ten per copy, so a built
frame's identifiers run far past the 32-bit integers the engine keeps its five
identifier variables in. Every draw in the build is keyed on those
identifiers, so the build keeps them; the release boundary renumbers and
keeps the build's beside the new ones.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.graph_national import UKNationalReadbackKernel
from microcosm.build.uk_runtime.graph_terminal import (
    describe_uk_export,
    materialize_uk_export,
    validate_uk_export,
)
from microcosm.build.uk_runtime.national_frame import (
    UK_BUILD_IDENTIFIER_COLUMNS,
    UK_ENGINE_IDENTIFIER_MAX,
    load_uk_national_frame,
    uk_build_identifier_frame,
    uk_build_identifier_tables,
    uk_build_identifiers,
    uk_identifier_range_failures,
    uk_national_frame,
    uk_release_export_frame,
    uk_release_identifier_tables,
    write_uk_national_frame,
)
from microcosm.build.uk_runtime.weight_year import uk_dataset_year_frame
from microcosm.frame import WeightKind
from test_support.microcosm_build.uk_graph_terminal import _frame as _ladder_frame

# Two copies of a three-household survey, numbered the way the clone stages
# number them: the copy index times a power of ten past the largest identifier.
COPY = 10**13
HOUSEHOLD = np.asarray([1, 7, COPY + 1, COPY + 7], dtype=np.int64)
BENUNIT = np.asarray([11, 12, 71, COPY + 11, COPY + 12, COPY + 71], dtype=np.int64)
PERSON = np.asarray(
    [101, 102, 103, 701, COPY + 101, COPY + 102, COPY + 103, COPY + 701],
    dtype=np.int64,
)
PERSON_BENUNIT = BENUNIT[[0, 0, 1, 2, 3, 3, 4, 5]]
PERSON_HOUSEHOLD = HOUSEHOLD[[0, 0, 0, 1, 2, 2, 2, 3]]


def _tables() -> dict[str, pd.DataFrame]:
    return {
        "person": pd.DataFrame(
            {
                "person_id": PERSON,
                "person_benunit_id": PERSON_BENUNIT,
                "person_household_id": PERSON_HOUSEHOLD,
                "age": [40, 38, 9, 71, 40, 38, 9, 71],
            }
        ),
        "benunit": pd.DataFrame({"benunit_id": BENUNIT}),
        "household": pd.DataFrame({"household_id": HOUSEHOLD, "region": "LONDON"}),
    }


def _frame(*, kind: WeightKind = WeightKind.CALIBRATED):
    return uk_national_frame(
        **_tables(),
        time_period="2024",
        weight_kind=kind,
        household_weights=np.asarray([10.0, 20.0, 30.0, 40.0]),
    )


def test_release_identifiers_are_one_plus_the_rank_of_the_build_identifier() -> None:
    built = _tables()

    released = uk_release_identifier_tables(built)

    assert released["household"]["household_id"].tolist() == [1, 2, 3, 4]
    assert released["benunit"]["benunit_id"].tolist() == [1, 2, 3, 4, 5, 6]
    assert released["person"]["person_id"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert released["person"]["person_benunit_id"].tolist() == [1, 1, 2, 3, 4, 4, 5, 6]
    assert released["person"]["person_household_id"].tolist() == [
        1,
        1,
        1,
        2,
        3,
        3,
        3,
        4,
    ]
    # The build identifiers stay, right beside the ones they were.
    for entity, carrier in UK_BUILD_IDENTIFIER_COLUMNS.items():
        table = released[entity]
        assert table[carrier].tolist() == built[entity][f"{entity}_id"].tolist()
        assert table.columns.get_loc(carrier) == 1
        assert table[f"{entity}_id"].dtype == built[entity][f"{entity}_id"].dtype
    # Nothing else moves, and the inputs are not touched.
    assert released["person"]["age"].tolist() == built["person"]["age"].tolist()
    assert built["person"]["person_id"].tolist() == PERSON.tolist()
    assert not uk_identifier_range_failures(released)


def test_the_rank_does_not_depend_on_row_order() -> None:
    built = _tables()
    order = [3, 0, 2, 1, 7, 4, 6, 5]
    built["person"] = built["person"].iloc[order].reset_index(drop=True)

    released = uk_release_identifier_tables(built)

    # Each person keeps the number its build identifier ranks at.
    assert released["person"]["person_id"].tolist() == [n + 1 for n in order]
    assert released["person"]["person_build_id"].tolist() == PERSON[order].tolist()


def test_renumbering_is_applied_once_and_undone_exactly() -> None:
    built = _tables()
    released = uk_release_identifier_tables(built)

    again = uk_release_identifier_tables(released)
    for entity in built:
        assert again[entity] is released[entity]

    restored = uk_build_identifier_tables(released)
    for entity in built:
        pd.testing.assert_frame_equal(restored[entity], built[entity])
    # Tables that were never renumbered are their own build tables.
    for entity, table in uk_build_identifier_tables(built).items():
        assert table is built[entity]
    assert uk_build_identifiers(released["household"], "household").tolist() == (
        HOUSEHOLD.tolist()
    )
    assert uk_build_identifiers(built["household"], "household").tolist() == (
        HOUSEHOLD.tolist()
    )


def test_tables_that_cannot_be_renumbered_refuse() -> None:
    repeated = _tables()
    repeated["household"].loc[1, "household_id"] = 1
    with pytest.raises(ValueError, match="household identifiers are not unique"):
        uk_release_identifier_tables(repeated)

    orphan = _tables()
    orphan["person"].loc[0, "person_benunit_id"] = 999
    with pytest.raises(ValueError, match="benunit memberships name an identifier"):
        uk_release_identifier_tables(orphan)

    partial = uk_release_identifier_tables(_tables())
    partial["benunit"] = partial["benunit"].drop(columns=["benunit_build_id"])
    with pytest.raises(ValueError, match="on some entities only"):
        uk_release_identifier_tables(partial)


def test_range_failures_name_the_column_and_a_count_never_an_identifier() -> None:
    built = _tables()

    failures = uk_identifier_range_failures(built)

    assert failures == [
        "Written 'person_id' is above the 32-bit range policyengine-uk stores on "
        "4 row(s).",
        "Written 'benunit_id' is above the 32-bit range policyengine-uk stores on "
        "3 row(s).",
        "Written 'household_id' is above the 32-bit range policyengine-uk stores on "
        "2 row(s).",
        "Written 'person_benunit_id' is above the 32-bit range policyengine-uk "
        "stores on 4 row(s).",
        "Written 'person_household_id' is above the 32-bit range policyengine-uk "
        "stores on 4 row(s).",
    ]
    assert not any(str(COPY) in failure for failure in failures)

    edge = uk_release_identifier_tables(_tables())
    edge["household"]["household_id"] = [0, 2, 2, UK_ENGINE_IDENTIFIER_MAX]
    assert uk_identifier_range_failures(edge) == [
        "Written 'household_id' is below 1 on 1 row(s).",
        "Written 'household_id' repeats a value.",
    ]
    edge["household"]["household_id"] = [1, 2, 3, UK_ENGINE_IDENTIFIER_MAX + 1]
    assert uk_identifier_range_failures(edge) == [
        "Written 'household_id' is above the 32-bit range policyengine-uk stores on "
        "1 row(s)."
    ]


def test_the_release_frame_leaves_renumbered_and_loads_back_to_the_build(
    tmp_path,
) -> None:
    pytest.importorskip("tables")
    built = _frame()

    released = uk_release_export_frame(built)

    assert uk_release_export_frame(released) is released
    assert released.weights_for("household").values.tolist() == [10, 20, 30, 40]
    assert released.mass_log == built.mass_log
    path = write_uk_national_frame(released, tmp_path / "release.h5")
    with pd.HDFStore(path, mode="r") as store:
        stored = {
            entity: store[entity] for entity in ("person", "benunit", "household")
        }
    assert not uk_identifier_range_failures(stored)
    assert stored["household"]["household_build_id"].tolist() == HOUSEHOLD.tolist()

    loaded, _provenance = load_uk_national_frame(path)
    restored = uk_build_identifier_frame(loaded)
    for entity in ("person", "benunit", "household"):
        pd.testing.assert_frame_equal(restored.table(entity), built.table(entity))
    # A frame that was never renumbered is its own build frame.
    assert uk_build_identifier_frame(built) is built


def _readback(frame, path, *, solve_year: int = 2025) -> dict:
    import json

    context = SimpleNamespace(
        tables={entity: frame.table(entity) for entity in frame.entities},
        frame_column_order={},
        weights={"household": frame.weights_for("household")},
        strata=frame.strata,
        frame_mass_log=frame.mass_log,
        frame_metadata=dict(frame.metadata),
        params={"dataset_source": "exported_dataset", "solve_year": solve_year},
        sources={"exported_dataset": str(path)},
    )
    result = UKNationalReadbackKernel().run(context)
    return {
        "outcome": result.receipt["outcome"],
        **json.loads(result.artifacts["readback"]),
    }


def test_the_national_readback_passes_the_release_and_refuses_build_identifiers(
    tmp_path,
) -> None:
    pytest.importorskip("tables")
    built = _frame()
    release = uk_dataset_year_frame(uk_release_export_frame(built), solve_year=2025)
    good = write_uk_national_frame(release, tmp_path / "release.h5")

    report = _readback(built, good)
    assert report["passed"] is True and report["outcome"] == "pass"

    # The same population written on its build identifiers, as releases were.
    unrenumbered = write_uk_national_frame(
        uk_dataset_year_frame(built, solve_year=2025), tmp_path / "build-ids.h5"
    )
    report = _readback(built, unrenumbered)
    assert report["passed"] is False and report["outcome"] == "fail"
    assert (
        "Written 'household_id' is above the 32-bit range policyengine-uk stores on "
        "2 row(s)."
    ) in report["failures"]
    assert not any(str(COPY) in failure for failure in report["failures"])


def test_the_national_readback_refuses_a_repeated_identifier(tmp_path) -> None:
    pytest.importorskip("tables")
    built = _frame()
    release = uk_dataset_year_frame(uk_release_export_frame(built), solve_year=2025)
    path = write_uk_national_frame(release, tmp_path / "release.h5")
    with pd.HDFStore(path) as store:
        benunit = store["benunit"]
        benunit.loc[1, "benunit_id"] = benunit.loc[0, "benunit_id"]
        store.put("benunit", benunit, format="table")

    # The frame itself cannot hold two rows under one identifier.
    with pytest.raises(ValueError):
        _readback(built, path)


def test_the_local_readback_refuses_out_of_range_and_repeated_identifiers(
    tmp_path,
) -> None:
    pytest.importorskip("tables")
    frame = _ladder_frame()
    descriptor = describe_uk_export(frame, bindings={}, solve_year=2025)
    path = tmp_path / "full.h5"
    materialize_uk_export(frame, descriptor, path)
    assert validate_uk_export(path, descriptor)["passed"] is True
    with pd.HDFStore(path, mode="r") as store:
        household = store["household"]
        person = store["person"]
    assert household["household_id"].tolist() == [1, 2]
    assert household["household_build_id"].tolist() == [1, 2]
    assert "household_build_id" in descriptor["tables"]["household"]["columns"]

    with pd.HDFStore(path) as store:
        person["person_id"] = [1, UK_ENGINE_IDENTIFIER_MAX + 1]
        store.put("person", person, format="table")
    report = validate_uk_export(path, descriptor)
    assert report["passed"] is False
    assert (
        "Written 'person_id' is above the 32-bit range policyengine-uk stores on "
        "1 row(s)."
    ) in report["failures"]

    with pd.HDFStore(path) as store:
        person["person_id"] = [1, 1]
        store.put("person", person, format="table")
    report = validate_uk_export(path, descriptor)
    assert report["passed"] is False
    assert "Written 'person_id' repeats a value." in report["failures"]
