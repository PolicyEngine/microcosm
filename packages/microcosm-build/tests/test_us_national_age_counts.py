"""Invented households through the real age-count operator and calibration.

Every population here is invented: five fictional households and ten fictional
people. No genuine population is opened, nothing is fitted or scored, and no
Census value is established by anything in this file.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import demographic_calibration_graph as demographic
from microcosm.build.us_runtime import graph_national_age_counts
from microcosm.build.us_runtime.demographic_calibration_graph import (
    DemographicCalibrationKernel,
)
from microcosm.build.us_runtime.graph_national_age_counts import (
    AGE_COUNT_SCHEMA_VERSION,
    SUPPORTED_AGE_CONVENTION,
    NationalAgeCountKernel,
    national_age_calibration_nodes,
    national_age_count_node,
)
from microcosm.build.us_runtime.national_age_activation import (
    NATIONAL_AGE_ACTIVATION,
    band_columns,
    demographic_target_hierarchy,
)
from microcosm.calibrate import calibrate, diagnostics_payload
from microcosm.calibrate.hierarchy import HierarchyCategory, HierarchyGeography
from microcosm.calibrate.registry import TargetRegistry, TargetSpec
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
    WeightTransition,
    compile_graph,
    load_source,
    run_graph,
    source_hash,
)

SCHEMA = EntitySchema(group_entities=("household",))
COLUMNS = band_columns()

#: Ascending household ids, as the population contract requires of a group
#: table. Household 15 stands in for a group-quarters address.
HOUSEHOLD_IDS = (7, 15, 22, 40, 90)
GROUP_QUARTERS_HOUSEHOLD = 15

#: (person_id, household, age). Deliberately shuffled, with 0/4/5 and 84/85 on
#: the band boundaries and a topcoded older age above the open top band. Twenty
#: invented people cover all eighteen bands, so every declared target is
#: reachable.
PEOPLE = (
    (109, 15, 85),
    (101, 7, 0),
    (117, 40, 84),
    (110, 22, 5),
    (104, 7, 12),
    (120, 90, 79),
    (102, 15, 4),
    (113, 22, 35),
    (118, 40, 96),
    (105, 15, 19),
    (111, 22, 20),
    (106, 7, 27),
    (119, 90, 60),
    (114, 15, 45),
    (103, 7, 44),
    (115, 22, 52),
    (107, 15, 33),
    (112, 40, 59),
    (108, 22, 70),
    (116, 40, 67),
)

#: Counts per band, in HOUSEHOLD_IDS order. Written out rather than derived, so
#: the expectation is auditable independently of the operator.
EXPECTED = {
    "people_age_0_4": (1, 1, 0, 0, 0),
    "people_age_5_9": (0, 0, 1, 0, 0),
    "people_age_10_14": (1, 0, 0, 0, 0),
    "people_age_15_19": (0, 1, 0, 0, 0),
    "people_age_20_24": (0, 0, 1, 0, 0),
    "people_age_25_29": (1, 0, 0, 0, 0),
    "people_age_30_34": (0, 1, 0, 0, 0),
    "people_age_35_39": (0, 0, 1, 0, 0),
    "people_age_40_44": (1, 0, 0, 0, 0),
    "people_age_45_49": (0, 1, 0, 0, 0),
    "people_age_50_54": (0, 0, 1, 0, 0),
    "people_age_55_59": (0, 0, 0, 1, 0),
    "people_age_60_64": (0, 0, 0, 0, 1),
    "people_age_65_69": (0, 0, 0, 1, 0),
    "people_age_70_74": (0, 0, 1, 0, 0),
    "people_age_75_79": (0, 0, 0, 0, 1),
    "people_age_80_84": (0, 0, 0, 1, 0),
    "people_age_85_plus": (0, 1, 0, 1, 0),
}

#: Importance weights the fixture calibrates from, in HOUSEHOLD_IDS order.
IMPORTANCE_WEIGHTS = (1.0, 0.5, 1.5, 0.5, 1.0)

#: Invented margins over the invented population: exactly twice what these
#: households already show, so the fixture has an exact feasible solution well
#: inside both weight caps. These are not published Census values.
FIXTURE_TARGETS = {
    band.variable: (
        band.column,
        2.0
        * sum(
            weight * count
            for weight, count in zip(
                IMPORTANCE_WEIGHTS, EXPECTED[band.column], strict=True
            )
        ),
    )
    for band in NATIONAL_AGE_ACTIVATION.bands
}

EPOCHS, LEARNING_RATE, CAP = 240, 0.05, 4.0


FIXTURE_BANDS = {band.variable: band for band in NATIONAL_AGE_ACTIVATION.bands}


def fixture_registry() -> TargetRegistry:
    """Fictional values and identities; these are not Census observations."""

    return TargetRegistry(
        [
            TargetSpec(
                name=name,
                entity="household",
                measure=measure,
                value=value,
                period="2024",
                source="Invented fixture; no published Census value",
                family="acs.S0101",
                hierarchy=demographic_target_hierarchy(
                    "S0101", FIXTURE_BANDS[name], geography="0100000US"
                ),
                metadata={
                    "table": "S0101",
                    "reference_sha256": "a" * 64,
                    "geography": "0100000US",
                    "universe": "population",
                    "role": "calibration",
                    "evidence_scope": "invented",
                },
            )
            for name, (measure, value) in FIXTURE_TARGETS.items()
        ],
        country="us",
    )


def fixture_frame(*, counted=False, kind=WeightKind.IMPORTANCE, scale=1.0) -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.array([row[0] for row in PEOPLE], dtype="int64"),
            "person_household_id": np.array([row[1] for row in PEOPLE], dtype="int64"),
            "age": np.array([float(row[2]) for row in PEOPLE]),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": np.array(HOUSEHOLD_IDS, dtype="int64"),
            "is_group_quarters": np.array(
                [identity == GROUP_QUARTERS_HOUSEHOLD for identity in HOUSEHOLD_IDS]
            ),
        }
    )
    if counted:
        for column in COLUMNS:
            household[column] = np.array(EXPECTED[column], dtype="int64")
    return Frame(
        {"person": person, "household": household},
        SCHEMA,
        {"household": Weights(np.array([1.0, 0.5, 1.5, 0.5, 1.0]) * scale, kind)},
    )


def count_context(frame: Frame, node: Node) -> KernelContext:
    return KernelContext(
        node=node,
        tables={
            "person": frame.table("person").loc[
                :, ["person_id", "person_household_id", "age"]
            ],
            "household": frame.table("household").loc[:, ["household_id"]],
        },
        weights={"household": frame.weights_for("household")},
        strata=frame.strata,
        params=node.params,
        rng=np.random.default_rng(0),
    )


def counted(result: KernelResult) -> pd.DataFrame:
    return pd.DataFrame(
        {column: result.columns[("household", column)] for column in COLUMNS}
    )


# --------------------------------------------------------------------------
# The declaration
# --------------------------------------------------------------------------


def test_the_node_declares_exactly_its_inputs_and_owns_eighteen_int64_columns():
    node = national_age_count_node(population="assembled")
    assert node.structural is StructuralDelta.NONE
    assert node.population == "assembled"
    assert node.base is None
    assert node.sources == ()
    assert node.weights is None
    assert node.artifact_inputs == ()
    # ``age`` is an owned cell and travels in a Slice; ids and memberships are
    # structural, so they are declared normatively in the parameters instead.
    assert {(slice_.entity, slice_.columns) for slice_ in node.inputs} == {
        ("person", ("age",))
    }
    assert all(slice_.rows == "all" for slice_ in node.inputs)
    assert node.params["person_columns"] == ("age", "person_household_id")
    assert node.params["household_columns"] == ("household_id",)
    assert tuple(owned.column for owned in node.outputs) == COLUMNS
    assert {owned.dtype for owned in node.outputs} == {"int64"}
    assert {owned.entity for owned in node.outputs} == {"household"}
    assert node.params["age_convention"] == SUPPORTED_AGE_CONVENTION
    assert node.params["schema_version"] == AGE_COUNT_SCHEMA_VERSION
    # No source channel, spine column, weight or model output is declared.
    declared = {
        *(column for slice_ in node.inputs for column in slice_.columns),
        *node.params["person_columns"],
        *node.params["household_columns"],
    }
    assert not any(
        marker in column
        for column in declared
        for marker in ("spine", "support_channel", "weight")
    )


def test_an_altered_band_schema_changes_the_node_key():
    baseline = national_age_count_node(population="assembled")
    moved = national_age_count_node(
        population="assembled",
        bands=(
            replace(NATIONAL_AGE_ACTIVATION.bands[0], column="people_age_0_to_4"),
            *NATIONAL_AGE_ACTIVATION.bands[1:],
        ),
    )
    assert moved.normative() != baseline.normative()
    assert moved.params["bands"] != baseline.params["bands"]
    assert tuple(owned.column for owned in moved.outputs) != COLUMNS


def test_the_kernel_refuses_a_node_that_differs_from_its_declaration():
    frame = fixture_frame()
    node = national_age_count_node(population="assembled")
    for tampered in (
        replace(node, inputs=(Slice("person", ("age", "is_group_quarters")),)),
        replace(node, inputs=(*node.inputs, Slice("household", ("household_id",)))),
        replace(node, outputs=node.outputs[:-1]),
        replace(node, mass="free"),
        replace(node, kernel="us.something_else@1"),
    ):
        with pytest.raises(ValueError, match="complete declaration"):
            NationalAgeCountKernel().run(count_context(frame, tampered))
    convention = replace(
        node, params={**node.params, "age_convention": "aged_to_target_period"}
    )
    with pytest.raises(ValueError, match="implements only"):
        NationalAgeCountKernel().run(count_context(frame, convention))
    version = replace(node, params={**node.params, "schema_version": 2})
    with pytest.raises(ValueError, match="schema version"):
        NationalAgeCountKernel().run(count_context(frame, version))
    extra = replace(node, params={**node.params, "seed": 1})
    with pytest.raises(ValueError, match="exactly its band schema"):
        NationalAgeCountKernel().run(count_context(frame, extra))
    widened = replace(
        node,
        params={
            **node.params,
            "person_columns": ("age", "person_household_id", "is_group_quarters"),
        },
    )
    with pytest.raises(ValueError, match="reads exactly"):
        NationalAgeCountKernel().run(count_context(frame, widened))


def test_the_kernel_refuses_a_household_view_beyond_its_declared_id():
    frame = fixture_frame()
    node = national_age_count_node(population="assembled")
    leaked = replace(
        count_context(frame, node),
        tables={
            "person": frame.table("person").loc[
                :, ["person_id", "person_household_id", "age"]
            ],
            "household": frame.table("household"),
        },
    )
    with pytest.raises(ValueError, match="carry exactly"):
        NationalAgeCountKernel().run(leaked)


@pytest.mark.parametrize(
    "bands",
    [
        # Overlapping bands would count a person twice.
        json.dumps([["a", 0, 4], ["b", 3, 9], ["c", 10, None]]),
        # A gap would drop people silently.
        json.dumps([["a", 0, 4], ["b", 7, None]]),
        # No open top band leaves the oldest uncovered.
        json.dumps([["a", 0, 4], ["b", 5, 9]]),
        # Two open bands cannot be disjoint.
        json.dumps([["a", 0, None], ["b", 5, None]]),
        # Duplicate columns would collide.
        json.dumps([["a", 0, 4], ["a", 5, None]]),
        # Not starting at zero would drop the youngest.
        json.dumps([["a", 1, 4], ["b", 5, None]]),
        "[]",
        "not json",
    ],
)
def test_an_incoherent_band_schema_is_refused(bands):
    frame = fixture_frame()
    node = national_age_count_node(population="assembled")
    with pytest.raises(ValueError):
        NationalAgeCountKernel().run(
            count_context(frame, replace(node, params={**node.params, "bands": bands}))
        )


# --------------------------------------------------------------------------
# What the operator is statically incapable of reading
# --------------------------------------------------------------------------


OPERATOR_MODULE = Path(graph_national_age_counts.__file__)


def _operator_tree() -> ast.Module:
    return ast.parse(OPERATOR_MODULE.read_text())


def test_the_operator_never_touches_weights_sources_artifacts_or_randomness():
    """A static read of the kernel's own source, not a runtime sample."""

    forbidden = {"weights", "sources", "artifacts", "rng", "strata"}
    reads = {
        node.attr
        for node in ast.walk(_operator_tree())
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "context"
        and node.attr in forbidden
    }
    assert not reads, f"the age-count operator reads context.{sorted(reads)}"


def test_the_operator_opens_no_file_and_imports_no_rules_engine():
    tree = _operator_tree()
    imported = {
        name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for name in (alias.name for alias in node.names)
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0
    }
    assert not imported & {"policyengine_us", "policyengine", "microdf", "urllib"}
    called = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert not called & {
        "open",
        "read_bytes",
        "read_text",
        "read_csv",
        "read_hdf",
        "load",
        "calculate",
        "simulate",
    }


# --------------------------------------------------------------------------
# Counting
# --------------------------------------------------------------------------


def test_every_person_is_counted_once_into_the_right_band():
    frame = fixture_frame()
    node = national_age_count_node(population="assembled")
    result = NationalAgeCountKernel().run(count_context(frame, node))
    table = counted(result)
    assert list(table.index) == list(HOUSEHOLD_IDS)
    assert table.index.name == "household_id"
    assert set(table.dtypes) == {np.dtype("int64")}
    for column in COLUMNS:
        assert tuple(table[column]) == EXPECTED[column], column
    assert int(table.to_numpy().sum()) == len(PEOPLE)
    assert result.receipt["people_counted"] == len(PEOPLE)
    assert result.receipt["households"] == len(HOUSEHOLD_IDS)
    assert result.receipt["consumes_weights"] is False
    assert result.receipt["release_eligible"] is False
    assert result.weights is None


@pytest.mark.parametrize(
    ("age", "column"),
    [
        (0, "people_age_0_4"),
        (4, "people_age_0_4"),
        (5, "people_age_5_9"),
        (9, "people_age_5_9"),
        (79, "people_age_75_79"),
        (80, "people_age_80_84"),
        (84, "people_age_80_84"),
        (85, "people_age_85_plus"),
        (99, "people_age_85_plus"),
        (115, "people_age_85_plus"),
        (120, "people_age_85_plus"),
    ],
)
def test_band_boundaries_and_topcoded_ages(age, column):
    """One person, one band: the boundaries land where the publisher puts them."""

    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": np.array([1], dtype="int64"),
                    "person_household_id": np.array([3], dtype="int64"),
                    "age": np.array([float(age)]),
                }
            ),
            "household": pd.DataFrame({"household_id": np.array([3], dtype="int64")}),
        },
        SCHEMA,
        {"household": Weights(np.array([1.0]), WeightKind.IMPORTANCE)},
    )
    node = national_age_count_node(population="assembled")
    table = counted(NationalAgeCountKernel().run(count_context(frame, node)))
    assert int(table[column].iloc[0]) == 1
    assert int(table.to_numpy().sum()) == 1


def test_rows_align_by_id_not_position():
    """Neither axis is assumed sorted, sequential, or shared between tables."""

    household = pd.DataFrame({"household_id": np.array([90, 7, 22], dtype="int64")})
    person = pd.DataFrame(
        {
            "person_id": np.array([5, 2, 9, 4], dtype="int64"),
            "person_household_id": np.array([22, 90, 22, 7], dtype="int64"),
            "age": np.array([0.0, 85.0, 6.0, 40.0]),
        }
    )
    node = national_age_count_node(population="assembled")
    context = KernelContext(
        node=node,
        tables={"person": person, "household": household},
        weights={
            "household": Weights(np.array([1.0, 1.0, 1.0]), WeightKind.IMPORTANCE)
        },
        strata=pd.Series(["x"] * 4),
        params=node.params,
        rng=np.random.default_rng(0),
    )
    table = counted(NationalAgeCountKernel().run(context))
    assert list(table.index) == [90, 7, 22]
    assert list(table["people_age_0_4"]) == [0, 0, 1]
    assert list(table["people_age_5_9"]) == [0, 0, 1]
    assert list(table["people_age_40_44"]) == [0, 1, 0]
    assert list(table["people_age_85_plus"]) == [1, 0, 0]
    # Reordering the person rows cannot change any count.
    shuffled = person.iloc[[3, 1, 0, 2]].reset_index(drop=True)
    reshuffled = counted(
        NationalAgeCountKernel().run(
            replace(context, tables={"person": shuffled, "household": household})
        )
    )
    pd.testing.assert_frame_equal(table, reshuffled)


def test_group_quarters_people_are_counted_with_their_membership_intact():
    frame = fixture_frame()
    node = national_age_count_node(population="assembled")
    # The operator cannot even see the group-quarters flag: it is neither a
    # declared input nor a column of the household view it is handed.
    assert "is_group_quarters" not in {
        *(column for slice_ in node.inputs for column in slice_.columns),
        *node.params["person_columns"],
        *node.params["household_columns"],
    }
    table = counted(NationalAgeCountKernel().run(count_context(frame, node)))
    quarters = HOUSEHOLD_IDS.index(GROUP_QUARTERS_HOUSEHOLD)
    assert int(table.loc[GROUP_QUARTERS_HOUSEHOLD, "people_age_85_plus"]) == 1
    assert int(table.loc[GROUP_QUARTERS_HOUSEHOLD].sum()) == sum(
        EXPECTED[column][quarters] for column in COLUMNS
    )
    assert int(table.to_numpy().sum()) == len(PEOPLE)


@pytest.mark.parametrize("age", [121, 200, 999])
def test_ages_outside_the_declared_envelope_refuse(age):
    node = national_age_count_node(population="assembled")
    context = count_context(fixture_frame(), node)
    person = context.tables["person"].copy()
    person.loc[person.index[0], "age"] = age
    with pytest.raises(ValueError, match="declared maximum"):
        NationalAgeCountKernel().run(
            replace(context, tables={**context.tables, "person": person})
        )


def test_counts_do_not_read_weights():
    node = national_age_count_node(population="assembled")
    baseline = counted(
        NationalAgeCountKernel().run(count_context(fixture_frame(), node))
    )
    for kind in (WeightKind.DESIGN, WeightKind.IMPORTANCE, WeightKind.CALIBRATED):
        for scale in (1.0, 137.0):
            other = counted(
                NationalAgeCountKernel().run(
                    count_context(fixture_frame(kind=kind, scale=scale), node)
                )
            )
            pd.testing.assert_frame_equal(baseline, other)


@pytest.mark.parametrize(
    ("age", "match"),
    [
        (np.nan, "unknown"),
        (-1.0, "negative"),
        (4.5, "whole number"),
        (np.inf, "finite"),
    ],
)
def test_an_unusable_age_refuses_rather_than_dropping_the_person(age, match):
    frame = fixture_frame()
    person = frame.table("person").copy()
    person.loc[0, "age"] = age
    node = national_age_count_node(population="assembled")
    context = replace(
        count_context(frame, node),
        tables={
            "person": person.loc[:, ["person_id", "person_household_id", "age"]],
            "household": frame.table("household").loc[:, ["household_id"]],
        },
    )
    with pytest.raises(ValueError, match=match):
        NationalAgeCountKernel().run(context)


@pytest.mark.parametrize("dtype", ["string", "bool", "complex128"])
def test_a_non_numeric_age_is_not_coerced(dtype):
    frame = fixture_frame()
    person = frame.table("person").copy()
    person["age"] = person["age"].astype(dtype)
    node = national_age_count_node(population="assembled")
    context = replace(
        count_context(frame, node),
        tables={
            "person": person.loc[:, ["person_id", "person_household_id", "age"]],
            "household": frame.table("household").loc[:, ["household_id"]],
        },
    )
    with pytest.raises(ValueError, match="real numeric"):
        NationalAgeCountKernel().run(context)


def test_unresolved_membership_refuses_rather_than_dropping_the_person():
    frame = fixture_frame()
    person = frame.table("person").copy()
    person.loc[0, "person_household_id"] = 999
    node = national_age_count_node(population="assembled")
    context = replace(
        count_context(frame, node),
        tables={
            "person": person.loc[:, ["person_id", "person_household_id", "age"]],
            "household": frame.table("household").loc[:, ["household_id"]],
        },
    )
    with pytest.raises(ValueError, match="must resolve to a household row"):
        NationalAgeCountKernel().run(context)


def test_a_duplicated_household_id_is_refused():
    node = national_age_count_node(population="assembled")
    household = pd.DataFrame({"household_id": np.array([3, 3], dtype="int64")})
    person = pd.DataFrame(
        {
            "person_id": np.array([1, 2], dtype="int64"),
            "person_household_id": np.array([3, 3], dtype="int64"),
            "age": np.array([1.0, 2.0]),
        }
    )
    context = KernelContext(
        node=node,
        tables={"person": person, "household": household},
        weights={"household": Weights(np.array([1.0, 1.0]), WeightKind.IMPORTANCE)},
        strata=pd.Series(["x", "x"]),
        params=node.params,
        rng=np.random.default_rng(0),
    )
    with pytest.raises(ValueError, match="unique"):
        NationalAgeCountKernel().run(context)


def test_a_household_with_no_people_is_refused_explicitly():
    """The population contract requires every group row to have a member."""

    node = national_age_count_node(population="assembled")
    household = pd.DataFrame({"household_id": np.array([3, 8], dtype="int64")})
    person = pd.DataFrame(
        {
            "person_id": np.array([1], dtype="int64"),
            "person_household_id": np.array([3], dtype="int64"),
            "age": np.array([1.0]),
        }
    )
    context = KernelContext(
        node=node,
        tables={"person": person, "household": household},
        weights={"household": Weights(np.array([1.0, 1.0]), WeightKind.IMPORTANCE)},
        strata=pd.Series(["x"]),
        params=node.params,
        rng=np.random.default_rng(0),
    )
    with pytest.raises(ValueError, match="at least one person"):
        NationalAgeCountKernel().run(context)


@pytest.mark.parametrize("bad", [np.nan, 2.5, 3 + 2j, 1e20, np.uint64(2**63 + 3)])
def test_a_broken_membership_column_is_refused(bad):
    node = national_age_count_node(population="assembled")
    household = pd.DataFrame({"household_id": np.array([3], dtype="int64")})
    person = pd.DataFrame(
        {
            "person_id": np.array([1], dtype="int64"),
            "person_household_id": np.array([bad]),
            "age": np.array([1.0]),
        }
    )
    context = KernelContext(
        node=node,
        tables={"person": person, "household": household},
        weights={"household": Weights(np.array([1.0]), WeightKind.IMPORTANCE)},
        strata=pd.Series(["x"]),
        params=node.params,
        rng=np.random.default_rng(0),
    )
    with pytest.raises(ValueError, match="household membership"):
        NationalAgeCountKernel().run(context)


@pytest.mark.parametrize("household_id", [-(2**63), 2**63 - 1])
def test_int64_boundary_ids_keep_their_exact_household_identity(household_id):
    node = national_age_count_node(population="assembled")
    household = pd.DataFrame({"household_id": np.array([household_id], dtype="int64")})
    person = pd.DataFrame(
        {
            "person_id": np.array([1], dtype="int64"),
            "person_household_id": np.array([household_id], dtype="int64"),
            "age": np.array([1.0]),
        }
    )
    context = KernelContext(
        node=node,
        tables={"person": person, "household": household},
        weights={"household": Weights(np.array([1.0]), WeightKind.IMPORTANCE)},
        strata=pd.Series(["x"]),
        params=node.params,
        rng=np.random.default_rng(0),
    )
    result = NationalAgeCountKernel().run(context)
    count = result.columns[("household", "people_age_0_4")]
    assert count.index.tolist() == [household_id]
    assert count.tolist() == [1]


# --------------------------------------------------------------------------
# Composition with the unchanged calibration node
# --------------------------------------------------------------------------


def test_the_composition_wires_counts_into_the_existing_calibration_node():
    count, calibration = national_age_calibration_nodes(
        base="assembled",
        registry=fixture_registry(),
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        max_weight_ratio=CAP,
        max_initial_weight_ratio=CAP,
    )
    assert count.id == "national.age_counts"
    assert calibration.id == "national.demographic_calibration"
    assert calibration.base == "assembled"
    assert count.population == "assembled"
    owned = tuple(item.column for item in count.outputs)
    assert calibration.inputs[0].columns == owned
    assert calibration.structural is StructuralDelta.REWEIGHT
    assert calibration.weights == WeightTransition(
        "household", "calibrated", mass="free"
    )
    # No frame-context artifact is produced or relabelled by this composition:
    # the diagnostics edge is the only artifact either node emits.
    assert count.artifact_outputs == ()
    assert tuple(item.name for item in calibration.artifact_outputs) == ("diagnostics",)


def test_a_registry_that_disagrees_with_the_bands_is_refused():
    """A partial or reordered registry is refused, not half-wired."""

    full = fixture_registry()
    for broken in (
        TargetRegistry(full.specs[:4], country="us"),
        TargetRegistry((full.specs[1], full.specs[0], *full.specs[2:]), country="us"),
    ):
        with pytest.raises(ValueError, match="exactly the declared band columns"):
            national_age_calibration_nodes(
                base="assembled",
                registry=broken,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                max_weight_ratio=CAP,
                max_initial_weight_ratio=CAP,
            )


# --------------------------------------------------------------------------
# The real graph, and a fresh-process warm replay
# --------------------------------------------------------------------------


class FixtureSource(KernelBase):
    ref = "test.age_count_source@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.CREATE,
        dependencies=("numpy", "pandas"),
    )

    def run(self, context):
        return KernelResult(
            frame=load_source("frame-store", context.sources["fixture"])
        )

    def implementation_hash(self):
        # Pytest and the fresh child import this fixture under different module
        # names. Bind its same actual file bytes, not that incidental name.
        return hashlib.sha256(
            Path(__file__).read_bytes() + source_hash(load_source).encode()
        ).hexdigest()


class FixtureImportance(FixtureSource):
    ref = "test.age_count_importance@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.REWEIGHT
    )

    def run(self, context):
        return KernelResult(
            weights=Weights(
                context.weights["household"].values * context.params["factor"],
                WeightKind.IMPORTANCE,
            )
        )


def graph_fixture(output, warm=False):
    """Cold and warm run the same on-disk fixture and real production kernels."""

    output = Path(output)
    store = ContentStore(output / "store")
    if not warm:
        frame = fixture_frame(kind=WeightKind.DESIGN, scale=2.0)
        # This key is local to the fresh invented fixture store. Warm replay
        # consumes the same content-verified object through its persisted path.
        source_key = hashlib.sha256(
            b"test.us-national-age-counts.source.v1"
        ).hexdigest()
        path = store.put_frame(source_key, frame)
        (output / "source-path.txt").write_text(str(path))
    path = Path((output / "source-path.txt").read_text())
    count, calibration = national_age_calibration_nodes(
        base="assembled",
        registry=fixture_registry(),
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        max_weight_ratio=CAP,
        max_initial_weight_ratio=CAP,
    )
    graph = Graph(
        "us",
        (SourceRef("fixture", "frame-store"),),
        (
            Node(
                "source",
                FixtureSource.ref,
                sources=("fixture",),
                structural=StructuralDelta.CREATE,
                outputs=(
                    Owned("person", "age", "float64"),
                    Owned("household", "is_group_quarters", "bool"),
                ),
            ),
            Node(
                "assembled",
                FixtureImportance.ref,
                base="source",
                inputs=(Slice("household", ("is_group_quarters",)),),
                structural=StructuralDelta.REWEIGHT,
                mass="free",
                weights=WeightTransition("household", "importance", mass="free"),
                params={"factor": 0.5},
            ),
            count,
            calibration,
        ),
    )
    registry = KernelRegistry()
    for kernel in (
        FixtureSource(),
        FixtureImportance(),
        NationalAgeCountKernel(),
        DemographicCalibrationKernel(),
    ):
        if warm:

            def forbidden(context):
                raise AssertionError("A warm replay must execute no kernels")

            kernel.run = forbidden
        registry.register(kernel)
    result = run_graph(
        compile_graph(graph), sources={"fixture": path}, store=store, kernels=registry
    )
    population = result.population("national.demographic_calibration")
    node = result.node("national.demographic_calibration")
    payload = store.load_bytes(node.opaque_artifacts["diagnostics"])
    counts = population.table("household").loc[:, list(COLUMNS)]
    return {
        "key": result.key,
        "diagnostics_sha256": hashlib.sha256(payload).hexdigest(),
        "weight_bytes_sha256": hashlib.sha256(
            population.weights_for("household").values.tobytes()
        ).hexdigest(),
        "counts": counts.to_dict(orient="list"),
        "household_ids": population.table("household")["household_id"].tolist(),
        "count_dtypes": sorted({str(dtype) for dtype in counts.dtypes}),
        "weight_kind": population.weights_for("household").kind.value,
        "diagnostics": json.loads(payload),
        "hits": [n.hit for n in result.nodes.values()],
    }


def test_the_real_graph_counts_and_calibrates_the_invented_population(tmp_path):
    cold = graph_fixture(tmp_path)
    diagnostics = cold["diagnostics"]
    assert diagnostics["schema_version"] == 8
    rows = diagnostics["targets"]
    assert [row["target_name"] for row in rows] == list(FIXTURE_TARGETS)
    for row in rows:
        hierarchy = row["hierarchy"]
        assert hierarchy["target"]["id"] == row["target_name"]
        assert hierarchy["provider"]["id"] == "census_acs"
        assert hierarchy["category"]["id"] == "census_acs.population_by_age"
        assert hierarchy["geography"] == {
            "id": "0100000US",
            "label": "United States",
            "level": "country",
        }
        assert [d["value_id"] for d in hierarchy["dimensions"]] == [row["target_name"]]
    assert cold["hits"] == [False, False, False, False]
    assert cold["household_ids"] == list(HOUSEHOLD_IDS)
    assert cold["count_dtypes"] == ["int64"]
    for column, expected in EXPECTED.items():
        assert cold["counts"][column] == list(expected)
    assert sum(sum(values) for values in cold["counts"].values()) == len(PEOPLE)
    assert cold["weight_kind"] == "calibrated"

    # The graph result equals the direct solver over the same counted frame.
    direct = calibrate(
        fixture_frame(counted=True),
        fixture_registry().to_target_set(),
        weight_entity="household",
        method="adam",
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        max_weight_ratio=CAP,
        mass="free",
        seed=0,
    )
    assert (
        cold["weight_bytes_sha256"]
        == hashlib.sha256(direct.weights.tobytes()).hexdigest()
    )
    assert cold["diagnostics"] == diagnostics_payload(
        direct,
        target_registry=fixture_registry(),
        build={
            "solver_weight_anchor": "incoming_importance",
            "solver_max_weight_ratio": CAP,
            "executor_weight_anchor": "original_design",
            "executor_max_weight_ratio": CAP,
        },
    )
    assert cold["diagnostics"]["final_loss"] < cold["diagnostics"]["initial_loss"]
    assert cold["diagnostics"]["skipped"] == []
    # Standard errors are absent from this invented registry, and the loss
    # would ignore them either way.
    assert DemographicCalibrationKernel.capabilities.consumes_se is False


def test_a_fresh_process_replays_warm_with_no_kernel_runs(tmp_path):
    cold = graph_fixture(tmp_path)
    assert cold["hits"] == [False, False, False, False]
    # The child removes foreign editable namespace entries after site startup.
    source_root = Path(__file__).resolve().parents[3]
    code = """import sys, pathlib, importlib.util, json
w = pathlib.Path(sys.argv[1])
sys.path[:] = [p for p in sys.path if not ('/packages/microcosm-' in p and p.endswith('/src'))]
sys.path[:0] = [str(p) for p in w.glob('packages/microcosm-*/src')]
spec = importlib.util.spec_from_file_location('age_count_graph_fixture', sys.argv[2])
m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
pathlib.Path(sys.argv[4]).write_text(json.dumps(m.graph_fixture(sys.argv[3], warm=True)))
"""
    result_path = tmp_path / "fresh-warm.json"
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(source_root),
            __file__,
            str(tmp_path),
            str(result_path),
        ],
        check=True,
        timeout=180,
    )
    warm = json.loads(result_path.read_text())
    assert warm["hits"] == [True, True, True, True]
    assert {k: v for k, v in warm.items() if k != "hits"} == {
        k: v for k, v in cold.items() if k != "hits"
    }


def test_the_registry_json_round_trips_every_calibration_hierarchy():
    registry = fixture_registry()
    frozen = demographic._registry_from_json(demographic._registry_json(registry))
    assert [spec.hierarchy for spec in frozen] == [spec.hierarchy for spec in registry]
    assert demographic._registry_json(frozen) == demographic._registry_json(registry)
    for spec in frozen:
        hierarchy = spec.hierarchy
        assert hierarchy.provider.id == "census_acs"
        assert hierarchy.category == HierarchyCategory(
            "census_acs.population_by_age", "Population by age", "census_acs"
        )
        assert hierarchy.geography == HierarchyGeography(
            "0100000US", "United States", "country"
        )
        (dimension,) = hierarchy.dimensions
        assert dimension.id == "publisher_cell"
        assert dimension.value_id == hierarchy.target.id == spec.name
        assert dimension.value_label == hierarchy.target.label
        assert hierarchy.target.label == FIXTURE_BANDS[spec.name].label


@pytest.mark.parametrize(
    "kind", ["missing", "foreign_cell", "foreign_geography", "foreign_category"]
)
def test_a_spec_without_exactly_its_own_hierarchy_is_refused(kind):
    registry = fixture_registry()
    first, *rest = list(registry)
    hierarchy = first.hierarchy
    if kind == "missing":
        hierarchy = None
    elif kind == "foreign_cell":
        (dimension,) = hierarchy.dimensions
        hierarchy = replace(
            hierarchy, dimensions=(replace(dimension, value_id="S0101_C01_019"),)
        )
    elif kind == "foreign_geography":
        hierarchy = replace(
            hierarchy,
            geography=HierarchyGeography("0400000US06", "California", "state"),
        )
    else:
        hierarchy = replace(
            hierarchy,
            category=HierarchyCategory(
                "census_acs.resident_population", "Resident population", "census_acs"
            ),
        )
    changed = TargetRegistry([replace(first, hierarchy=hierarchy), *rest], country="us")
    with pytest.raises(ValueError, match="calibration hierarchy"):
        demographic._validate_registry(changed)
    demographic._validate_registry(registry)
