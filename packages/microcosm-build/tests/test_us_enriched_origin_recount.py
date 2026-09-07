"""Exact indexed age recount in the enriched-origin admission, invented data only.

Every fixture here is written out in this file. The reference the recount is
checked against is a plain-Python brute force over Python integers, and the
second reference is the frozen row-scan the optimization replaced — neither
calls the age-count producer, so a shared bug cannot cancel out.
"""

import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_us

# The count route reads no bundle. Refuse an ambient one rather than inherit it.
os.environ.pop("MICROCOSM_US_NATIONAL_ACS_BUNDLE", None)

REPOSITORY = Path(__file__).resolve().parents[3]
STAGE_SOURCE = (
    REPOSITORY
    / "packages/microcosm-build/src/microcosm/build/us_runtime"
    / "enriched_origin_calibration.py"
)


@pytest.fixture(scope="module", autouse=True)
def invented_only_guard():
    """Deny genuine trees, network and every policy-engine calculation."""
    from unittest.mock import patch

    denials = []

    def guard(path):
        if isinstance(path, (str, bytes, os.PathLike)):
            text = os.fsdecode(path)
            if any(
                s in text
                for s in (
                    "/_buildo-runtime/",
                    "/.cache/microcosm/",
                    "/storage/",
                    "/native-asec-original-",
                )
            ):
                denials.append("genuine_path")
                raise RuntimeError("GENUINE_PATH_FORBIDDEN")

    def audit(event, args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            denials.append("network")
            raise RuntimeError("NETWORK_FORBIDDEN")
        if event in ("open", "os.listdir", "os.scandir"):
            guard(args[0])

    sys.addaudithook(audit)
    import policyengine_core.simulations as core
    import policyengine_us as us

    def no_calculation(*args, **kwargs):
        denials.append("calculation")
        raise RuntimeError("POLICY_CALCULATION_FORBIDDEN")

    patches = []
    for package in (core, us):
        for name in ("Simulation", "Microsimulation", "IndividualSim"):
            cls = getattr(package, name, None)
            if cls is None:
                continue
            for method in (
                "calculate",
                "calculate_add",
                "calculate_divide",
                "calculate_output",
                "calculate_dataframe",
                "_calculate",
            ):
                if hasattr(cls, method):
                    patches.append(patch.object(cls, method, no_calculation))
    for item in patches:
        item.start()
    yield denials
    for item in reversed(patches):
        item.stop()
    assert denials == []


@pytest.fixture(scope="module")
def stage():
    from microcosm.build.us_runtime import enriched_origin_calibration

    return enriched_origin_calibration


@pytest.fixture(scope="module")
def bands(stage):
    return [
        (band.column, band.low, band.high)
        for band in stage.ages.NATIONAL_AGE_ACTIVATION.bands
    ]


def test_loaded_stage_is_this_repository_source(stage):
    """No borrowed editable install may stand in for the source under test."""
    assert Path(stage.__file__).resolve() == STAGE_SOURCE
    assert STAGE_SOURCE.is_file()


# --- independent references -------------------------------------------------


def oracle_counts(bands, people, households):
    """Brute-force recount in plain Python over exact Python integers.

    Deliberately the naive nested scan: for every band, for every household,
    walk every person. No NumPy, no index, no shared helper with the code
    under test.
    """
    counts = {}
    for column, low, high in bands:
        row = []
        for identifier in households:
            total = 0
            for age, membership in people:
                if (
                    membership == identifier
                    and age >= low
                    and (high is None or age <= high)
                ):
                    total += 1
            row.append(total)
        counts[column] = row
    return counts


def frozen_row_scan(bands, people, households):
    """The recount exactly as it stood at 669edf78, kept as a second reference."""
    import numpy as np

    age = np.asarray([a for a, _ in people], dtype="float64")
    memberships = np.asarray([m for _, m in people], dtype="int64")
    ids = np.asarray(households, dtype="int64")
    counts = {}
    for column, low, high in bands:
        inside = (age >= low) & (True if high is None else age <= high)
        counts[column] = np.asarray(
            [np.count_nonzero(inside & (memberships == i)) for i in ids],
            dtype="int64",
        ).tolist()
    return counts


def oracle_rows(memberships, households):
    """Each membership's household row, by plain-Python identity search."""
    return [list(households).index(int(m)) for m in memberships]


# --- explicit invented shapes ----------------------------------------------

#: (label, people as (age, household id), household ids in table order)
SHAPES = [
    (
        "band_boundaries_and_open_top",
        [(0, 3), (4, 3), (5, 3), (84, 7), (85, 7), (86, 7), (120, 7), (30, 11)],
        [3, 7, 11],
    ),
    (
        "repeated_and_shuffled_membership",
        [(41, 20), (2, 4), (41, 4), (2, 20), (63, 4), (63, 20), (0, 4)],
        [4, 20],
    ),
    (
        "nonconsecutive_ids",
        [(9, 101), (9, 907), (77, 101), (85, 55555)],
        [101, 907, 55555],
    ),
    (
        "ids_above_float64_integer_precision",
        [
            (12, 9007199254740992),
            (12, 9007199254740993),
            (85, 9007199254740994),
            (85, 9007199254740993),
        ],
        [9007199254740992, 9007199254740993, 9007199254740994],
    ),
    ("single_person_single_household", [(0, 1)], [1]),
    (
        "every_person_in_one_household",
        [(a, 42) for a in (0, 1, 4, 5, 9, 10, 19, 20, 64, 65, 84, 85, 119, 120)],
        [42],
    ),
]
SHAPE_IDS = [label for label, _, _ in SHAPES]


def population(stage, people, households, counts=None):
    """One invented Population; row 0 carries a zero design weight deliberately."""
    import numpy as np
    import pandas as pd

    from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
    from microcosm.graph.population import Population

    n = len(people)
    person = pd.DataFrame(
        {
            "person_id": np.arange(n, dtype="int64"),
            "age": np.asarray([a for a, _ in people], dtype="int64"),
            "person_household_id": np.asarray([m for _, m in people], dtype="int64"),
        }
    )
    tables = {"person": person}
    for entity in US_SCHEMA.group_entities:
        if entity == "household":
            continue
        person["person_" + entity + "_id"] = np.arange(n, dtype="int64")
        tables[entity] = pd.DataFrame({entity + "_id": np.arange(n, dtype="int64")})
    # A Frame requires the group id column sorted ascending, so the household
    # table is ordered here and the count columns follow that same order.
    order = sorted(range(len(households)), key=lambda i: households[i])
    household = pd.DataFrame(
        {"household_id": np.asarray([households[i] for i in order], dtype="int64")}
    )
    for column, values in (counts or {}).items():
        household[column] = np.asarray([values[i] for i in order], dtype="int64")
    tables["household"] = household
    # Row 0 deliberately carries a zero design weight; the last row keeps the
    # Frame's "not all zero" contract satisfied even for a single household.
    weights = np.arange(len(households), dtype="float64")
    weights[-1] = float(len(households))
    frame = Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(weights, WeightKind.DESIGN)},
        metadata={"invented": 1},
    )
    owners = {
        (entity, column): (stage.COUNT if column in (counts or {}) else "invented_base")
        for entity in frame.entities
        for column in frame.table(entity).columns
    }
    return Population.from_frame(
        frame,
        version="invented",
        owners=owners,
        design_weights={"household": weights.copy()},
    )


# --- the recount ------------------------------------------------------------


@pytest.mark.parametrize("label,people,households", SHAPES, ids=SHAPE_IDS)
def test_recount_accepts_the_brute_force_oracle(
    stage, bands, label, people, households
):
    expected = oracle_counts(bands, people, households)
    assert expected == frozen_row_scan(bands, people, households)
    before = population(stage, people, households)
    after = population(stage, people, households, expected)
    stage.verify_counts(before, after)


@pytest.mark.parametrize("label,people,households", SHAPES, ids=SHAPE_IDS)
def test_recount_refuses_every_single_count_perturbation(
    stage, bands, label, people, households
):
    expected = oracle_counts(bands, people, households)
    before = population(stage, people, households)
    for column in expected:
        for row in range(len(households)):
            for delta in (1, -1):
                changed = {c: list(v) for c, v in expected.items()}
                changed[column][row] += delta
                after = population(stage, people, households, changed)
                with pytest.raises(stage.budgets.OriginBudgetError, match="COUNT"):
                    stage.verify_counts(before, after)


def test_open_final_band_holds_everyone_at_or_above_its_low(stage, bands):
    people = [(84, 1), (85, 1), (86, 1), (120, 1)]
    counts = oracle_counts(bands, people, [1])
    assert counts["people_age_85_plus"] == [3]
    assert counts["people_age_80_84"] == [1]
    stage.verify_counts(
        population(stage, people, [1]), population(stage, people, [1], counts)
    )


@pytest.mark.parametrize("mutation", ["count", "ledger", "owner"])
def test_recount_refuses_changed_count_ledger_or_owner(stage, bands, mutation):
    from dataclasses import replace

    from microcosm.graph.population import MassRecord

    people, households = SHAPES[0][1], SHAPES[0][2]
    counts = oracle_counts(bands, people, households)
    before = population(stage, people, households)
    after = population(stage, people, households, counts)
    if mutation == "count":
        changed = {c: list(v) for c, v in counts.items()}
        changed["people_age_85_plus"][1] += 1
        after = population(stage, people, households, changed)
    elif mutation == "ledger":
        after = replace(
            after,
            mass_ledger=(
                MassRecord(
                    node_id="invented",
                    operation="invented",
                    policy="invented",
                    before_total=1.0,
                    after_total=1.0,
                    before_by_stratum=(),
                    after_by_stratum=(),
                    entity="household",
                ),
            ),
        )
    else:
        owners = dict(after.owners)
        owners[("household", "people_age_85_plus")] = "invented_wrong_writer"
        after = replace(after, owners=owners)
    with pytest.raises(stage.budgets.OriginBudgetError):
        stage.verify_counts(before, after)


# --- the membership index ---------------------------------------------------


@pytest.mark.parametrize(
    "memberships,households",
    [
        ([7, 3, 11, 7], [11, 3, 7]),
        ([2, 2, 2], [5, 2]),
        (
            [9007199254740993, 9007199254740992],
            [9007199254740994, 9007199254740992, 9007199254740993],
        ),
        ([], [1, 2]),
        ([1, 3], [1, 2, 3]),
    ],
    ids=[
        "shuffled",
        "repeated",
        "above_float_precision",
        "no_people",
        "memberless_row",
    ],
)
def test_membership_rows_match_a_plain_python_identity_search(
    stage, memberships, households
):
    import numpy as np

    rows = stage._membership_rows(
        np.asarray(memberships, dtype="int64"), np.asarray(households, dtype="int64")
    )
    assert rows.tolist() == oracle_rows(memberships, households)


def test_memberless_household_is_addressed_by_no_person(stage):
    """A household row nobody belongs to keeps its row and takes no member."""
    import numpy as np

    households = [10, 20, 30]
    rows = stage._membership_rows(
        np.asarray([10, 30, 30], dtype="int64"), np.asarray(households, dtype="int64")
    )
    assert rows.tolist() == oracle_rows([10, 30, 30], households) == [0, 2, 2]
    assert 1 not in set(rows.tolist())


def test_a_memberless_household_is_refused_by_the_frame_contract(stage):
    """The population contract, not this recount, is what forbids an empty row."""
    with pytest.raises(ValueError, match="referenced by no person"):
        population(stage, [(30, 1)], [1, 2])


@pytest.mark.parametrize(
    "memberships,households,reason",
    [
        ([1], [1, 1], "COUNT_HOUSEHOLD_UNIQUE"),
        ([1, 1], [1, 1, 2], "COUNT_HOUSEHOLD_UNIQUE"),
        ([9], [1, 2], "COUNT_MEMBERSHIP"),
        ([0], [1, 2], "COUNT_MEMBERSHIP"),
        ([3], [1, 2], "COUNT_MEMBERSHIP"),
        ([1], [], "COUNT_MEMBERSHIP"),
    ],
    ids=["duplicate", "duplicate_with_extra", "orphan", "below", "above", "no_rows"],
)
def test_membership_rows_refuse_ambiguous_or_orphan_mappings(
    stage, memberships, households, reason
):
    import numpy as np

    with pytest.raises(stage.budgets.OriginBudgetError, match=reason):
        stage._membership_rows(
            np.asarray(memberships, dtype="int64"),
            np.asarray(households, dtype="int64"),
        )


# --- exact id extraction ----------------------------------------------------


def test_exact_ids_keep_identities_a_float_cast_would_merge(stage):
    import numpy as np
    import pandas as pd

    values = [2**53, 2**53 + 1, 2**53 + 2]
    assert len({float(v) for v in values}) == 2  # float64 merges two of the three
    out = stage._exact_ids(pd.Series(values, dtype="int64"), "COUNT_HOUSEHOLD_ID")
    assert out.dtype == np.dtype("int64")
    assert [int(v) for v in out] == values


@pytest.mark.parametrize(
    "dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "Int64"]
)
def test_exact_ids_accept_every_supported_integer_width(stage, dtype):
    import numpy as np
    import pandas as pd

    out = stage._exact_ids(pd.Series([1, 2, 3], dtype=dtype), "COUNT_HOUSEHOLD_ID")
    assert out.dtype == np.dtype("int64") and out.tolist() == [1, 2, 3]


def test_exact_ids_accept_uint64_inside_int64(stage):
    import pandas as pd

    out = stage._exact_ids(
        pd.Series([2**63 - 1, 2**53 + 1], dtype="uint64"), "COUNT_HOUSEHOLD_ID"
    )
    assert [int(v) for v in out] == [2**63 - 1, 2**53 + 1]


def test_exact_ids_accept_whole_real_ids_at_their_stored_precision(stage):
    import pandas as pd

    out = stage._exact_ids(pd.Series([3.0, -7.0, 0.0], dtype="float64"), "R")
    assert out.tolist() == [3, -7, 0]


@pytest.mark.parametrize(
    "series_args",
    [
        ([True, False], "bool"),
        ([1, None], "Int64"),
        ([1.0, None], "float64"),
        ([1.5, 2.0], "float64"),
        ([float("inf"), 1.0], "float64"),
        ([float("nan"), 1.0], "float64"),
        ([2.0**63, 1.0], "float64"),
        ([2**63, 1], "uint64"),
        (["1", "2"], "object"),
        ([1 + 0j, 2 + 0j], "complex128"),
    ],
    ids=[
        "bool",
        "masked_null",
        "float_null",
        "fractional",
        "infinite",
        "not_a_number",
        "outside_int64",
        "uint64_outside_int64",
        "text",
        "complex",
    ],
)
def test_exact_ids_refuse_anything_that_is_not_an_exact_whole_id(stage, series_args):
    import pandas as pd

    values, dtype = series_args
    with pytest.raises(stage.budgets.OriginBudgetError, match="COUNT_HOUSEHOLD_ID"):
        stage._exact_ids(pd.Series(values, dtype=dtype), "COUNT_HOUSEHOLD_ID")


# --- bounded synthetic shape ------------------------------------------------


def test_bounded_synthetic_shape_matches_the_frozen_row_scan(stage, bands):
    """A matched old/new comparison large enough to exercise the index.

    Deterministic invented ages and memberships; no source, no draw, no model.
    """
    households = [1_000_003 + 7 * i for i in range(400)]
    people = [
        ((i * 37) % 121, households[(i * 17) % len(households)]) for i in range(1200)
    ]
    people += [(0, identifier) for identifier in households]  # every row referenced
    expected = frozen_row_scan(bands, people, households)
    assert expected == oracle_counts(bands, people, households)
    stage.verify_counts(
        population(stage, people, households),
        population(stage, people, households, expected),
    )
