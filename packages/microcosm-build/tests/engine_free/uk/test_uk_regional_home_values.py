"""Home values from the region's Land Registry distribution on the spine
(microcosm#1112, microcosm#1095)."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime import property_localisation
from microcosm.build.uk_runtime import regional_home_values as rhv
from microcosm.build.uk_runtime.atomic_area_support import (
    uk_atomic_assignment_definition,
)
from microcosm.build.uk_runtime.graph_kernels import UKClaimKernel
from microcosm.build.uk_runtime.graph_population import (
    append_uk_population_nodes,
    register_uk_population_kernels,
)
from microcosm.build.uk_runtime.household_lineage import HIGH_VALUE_SUPPORT_COPY_COLUMN
from microcosm.build.uk_runtime.national_frame import (
    UK_RELEASE_EXPORT_DROPPED_COLUMNS,
    uk_national_frame,
)
from microcosm.frame import Frame, WeightKind
from microcosm.graph import (
    ContentStore,
    Graph,
    KernelRegistry,
    KernelResult,
    SourceRef,
    compile_graph,
    run_graph,
)
from test_support.microcosm_build.uk_atomic_support_fixtures import (
    toy_support_sources,
    write_toy_property_resources,
    write_toy_supports,
)
from test_support.microcosm_build.uk_full_population_graph import (
    SOURCE_VINTAGE,
    Source,
    source_frame,
    source_node,
)
from test_support.microcosm_build.uk_ladder_rowwise_clone import (
    toy_ladder as toy_ladder,
)

_REGIONS = ("LONDON", "NORTH_EAST", "WALES", "SCOTLAND", "NORTHERN_IRELAND")


def _frame(n: int = 200, seed: int = 5, *, source_ids=None, lineage: bool = True):
    """Households of five regions; ``source_ids`` makes support copies (each
    household's own id by default), ``lineage=False`` drops them."""

    rng = np.random.default_rng(seed)
    region = np.array(_REGIONS)[rng.integers(0, len(_REGIONS), n)]
    tenure = np.where(
        rng.random(n) < 0.7,
        np.where(rng.random(n) < 0.5, "OWNED_OUTRIGHT", "OWNED_WITH_MORTGAGE"),
        "RENT_PRIVATELY",
    )
    value = np.where(tenure != "RENT_PRIVATELY", rng.lognormal(12.5, 0.6, n), 0.0)
    value[:4] = 0.0  # owners without a value keep it
    tenure[:4] = "OWNED_OUTRIGHT"
    other = rng.uniform(0, 50_000, n)
    household = pd.DataFrame(
        {
            "household_id": np.arange(1, n + 1, dtype="int64"),
            "region": pd.array(region, dtype="string"),
            "tenure_type": pd.array(tenure, dtype="string"),
            "main_residence_value": value,
            "property_wealth": value + other,
            HIGH_VALUE_SUPPORT_COPY_COLUMN: rng.random(n) < 0.05,
        }
    )
    if lineage:
        household["household_source_id"] = (
            household["household_id"].to_numpy()
            if source_ids is None
            else np.asarray(source_ids, dtype="int64")
        )
    benunit = pd.DataFrame(
        {
            "benunit_id": np.arange(1, n + 1, dtype="int64"),
            # A context cell every graph node receives.
            "frs_benunit_capital": np.zeros(n),
        }
    )
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, n + 1, dtype="int64"),
            "person_household_id": np.arange(1, n + 1, dtype="int64"),
            "person_benunit_id": np.arange(1, n + 1, dtype="int64"),
            "age": np.full(n, 50, dtype="int64"),
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2024",
        weight_kind=WeightKind.DESIGN,
        household_weights=rng.uniform(500.0, 3_000.0, n),
    )


def test_england_and_wales_owners_take_their_regions_distribution() -> None:
    frame = _frame()
    before = frame.table("household")
    result = rhv.assign_regional_home_values(frame)
    after = result.frame.table("household")
    owner = property_localisation.owner_mask(before["tenure_type"].to_numpy())
    value = before["main_residence_value"].to_numpy()
    mapped = before["region"].isin(list(rhv.FRS_REGION_TO_KNOTS_REGION)).to_numpy()
    target = owner & (value > 0) & mapped
    # Untouched: renters, owners without a value, Scotland and Northern Ireland.
    np.testing.assert_array_equal(
        after["main_residence_value"].to_numpy()[~target], value[~target]
    )
    assert not np.allclose(
        after["main_residence_value"].to_numpy()[target], value[target]
    )
    # The replaced value is kept for the dense node to rank on.
    np.testing.assert_array_equal(after[rhv.RANK_BASIS_COLUMN].to_numpy(), value)
    # Wealth moves by the value's change.
    np.testing.assert_allclose(
        after["property_wealth"].to_numpy(),
        np.maximum(
            before["property_wealth"].to_numpy()
            + after["main_residence_value"].to_numpy()
            - value,
            0.0,
        ),
    )
    assert result.frame.mass_log[-1].declared_factor == 1.0
    receipt = result.receipt
    assert set(receipt["rows_rewritten_by_region"]) <= {
        "E12000007",
        "E12000001",
        "W92000004",
    }


def test_values_keep_the_within_region_order() -> None:
    frame = _frame()
    before = frame.table("household")
    after = rhv.assign_regional_home_values(frame).frame.table("household")
    support = before[HIGH_VALUE_SUPPORT_COPY_COLUMN].to_numpy()
    for region in ("LONDON", "NORTH_EAST", "WALES"):
        rows = (
            (before["region"] == region).to_numpy()
            & property_localisation.owner_mask(before["tenure_type"].to_numpy())
            & (before["main_residence_value"].to_numpy() > 0)
            & ~support
        )
        old = before["main_residence_value"].to_numpy()[rows]
        new = after["main_residence_value"].to_numpy()[rows]
        order = np.argsort(old, kind="stable")
        # Outside the support copies, a higher WAS value keeps a higher value.
        assert np.all(np.diff(new[order]) >= 0)


def test_the_wales_map_uses_the_knots_country_code() -> None:
    assert rhv.FRS_REGION_TO_KNOTS_REGION["WALES"] == "W92000004"
    assert "SCOTLAND" not in rhv.FRS_REGION_TO_KNOTS_REGION
    assert "NORTHERN_IRELAND" not in rhv.FRS_REGION_TO_KNOTS_REGION
    assert len(rhv.FRS_REGION_TO_KNOTS_REGION) == 10


class _FrameSource(Source):
    """The toy source kernel, returning a given frame."""

    def __init__(self, frame) -> None:
        self._frame = frame

    def run(self, context):
        return KernelResult(frame=self._frame)


def _dense_frame(*, rank_basis=None, value=None):
    """The toy dense source with two London owners, so a value can flip a rank."""

    base = source_frame()
    tables = {entity: base.table(entity).copy() for entity in base.entities}
    household = tables["household"]
    household["region"] = pd.array(
        ["LONDON", "LONDON", "SCOTLAND", "NORTHERN_IRELAND"], dtype="string"
    )
    basis = np.array([400_000.0, 250_000.0, 0.0, 0.0])
    household["main_residence_value"] = basis
    if rank_basis is not None:
        household[property_localisation.RANK_BASIS_COLUMN] = rank_basis
        household["main_residence_value"] = value
        household["property_wealth"] = household["property_wealth"] + (value - basis)
    return Frame(
        tables,
        base.schema,
        {"household": base.weights_for("household")},
        base.strata,
        mass_log=base.mass_log,
        metadata=base.metadata,
    )


def _dense_households(frame, directory, ladder_path):
    payloads, paths = write_toy_supports(directory / "supports")
    resource_paths = write_toy_property_resources(directory / "property")
    graph = append_uk_population_nodes(
        Graph("uk", (SourceRef("fixture", "raw-bytes-v1"),), (source_node(frame),)),
        population="source",
        time_period="2023",
        weight_kind="importance",
        n_clones=1,
        seed=7,
        source_year=2023,
        geography_assignment="atomic",
        atomic_geography_definition=uk_atomic_assignment_definition(payloads, seed=7),
        source_vintage=SOURCE_VINTAGE,
        property_localisation_mode="price_paid",
    )
    registry = KernelRegistry()
    registry.register(_FrameSource(frame))
    registry.register(UKClaimKernel())
    register_uk_population_kernels(registry)
    run = run_graph(
        compile_graph(graph),
        sources={
            "uk_ladder": ladder_path,
            "fixture": ladder_path,
            **toy_support_sources(paths),
            **resource_paths,
        },
        store=ContentStore(directory / "store"),
        kernels=registry,
    )
    household = run.population("uk.full.expand").table("household")
    return household[["household_id", "main_residence_value", "property_wealth"]]


def test_the_dense_line_does_not_move_with_the_regional_values(
    toy_ladder, tmp_path
) -> None:
    _, ladder_path = toy_ladder
    basis = np.array([400_000.0, 250_000.0, 0.0, 0.0])
    # The regional stage swaps the two London owners' order.
    regional = np.array([200_000.0, 500_000.0, 0.0, 0.0])
    before = _dense_households(_dense_frame(), tmp_path / "before", ladder_path)
    after = _dense_households(
        _dense_frame(rank_basis=basis, value=regional), tmp_path / "after", ladder_path
    )
    # The value node receives the replaced value through its slice, so the
    # dense line's values and wealth are those it had before the stage.
    pd.testing.assert_frame_equal(
        before.reset_index(drop=True), after.reset_index(drop=True)
    )
    # Ranking on the regional values instead would move them: the test is
    # sensitive to the basis the node reads.
    moved = _dense_households(
        _dense_frame(rank_basis=regional, value=regional),
        tmp_path / "moved",
        ladder_path,
    )
    assert not np.allclose(
        before["main_residence_value"].to_numpy(),
        moved["main_residence_value"].to_numpy(),
    )


def test_the_rank_basis_refuses_a_declaration_mismatch() -> None:
    household = pd.DataFrame({"main_residence_value": [1.0, 2.0]})
    np.testing.assert_array_equal(
        property_localisation.main_residence_rank_basis(household, declared=False),
        [1.0, 2.0],
    )
    with pytest.raises(ValueError, match="declared but missing"):
        property_localisation.main_residence_rank_basis(household, declared=True)
    staged = household.assign(**{property_localisation.RANK_BASIS_COLUMN: [3.0, 4.0]})
    np.testing.assert_array_equal(
        property_localisation.main_residence_rank_basis(staged, declared=True),
        [3.0, 4.0],
    )
    with pytest.raises(ValueError, match="present but undeclared"):
        property_localisation.main_residence_rank_basis(staged, declared=False)


def _graph_evidence(frame) -> dict:
    """The stage's receipt from the production node, run as a graph build runs it.

    The executor hands the kernel only the cells the node declares, so a
    column the stage reads but the node does not declare is missing here.
    """

    from microcosm.build.uk_runtime.graph import uk_spine_graph
    from microcosm.build.uk_runtime.graph_kernels import UKStageKernel
    from microcosm.graph.executor import _project_context
    from microcosm.graph.population import Population

    name = rhv.REGIONAL_HOME_VALUES_STAGE_NAME
    node = uk_spine_graph(source_mode="split").node(name)
    context = _project_context(
        node,
        Population.from_frame(frame, "regional-home-values-input"),
        key="0" * 64,
        sources={},
        tolerances={},
        numerics={},
    )
    stage = load_country_spec("uk").sources.stage_map()[name]
    transform = rhv.UKRegionalHomeValuesStageTransform(stage=stage)
    UKStageKernel(name, transform).run(context)
    return transform.checkpoint_metadata()["evidence"]


def _london_copies_at_the_cap(monkeypatch):
    """Twelve London owners, support copies of three source households, every
    one at a cap the test lowers below all values (on the packaged knots the
    cap lies at or above each region's top knot, which no rank below 1
    reaches)."""

    parsed, knots = rhv.load_region_knots()
    monkeypatch.setattr(
        rhv,
        "load_region_knots",
        lambda payload=None: (dataclasses.replace(parsed, cap=1.0), knots),
    )
    frame = _frame(n=12, seed=9, source_ids=np.arange(12) % 3)
    household = frame.table("household")
    household["region"] = pd.array(["LONDON"] * 12, dtype="string")
    household["tenure_type"] = pd.array(["OWNED_OUTRIGHT"] * 12, dtype="string")
    household["main_residence_value"] = np.linspace(300_000.0, 900_000.0, 12)
    household["property_wealth"] = household["main_residence_value"] + 10_000.0
    # The node's shape anchor, the preceding support stage's copy index.
    household["high_value_support_copy_index"] = np.zeros(12, dtype="int64")
    return frame


def test_the_cap_cell_is_banded_on_its_source_households(monkeypatch) -> None:
    # UKDS disclosure control: twelve rows past ten, but three sources, so the
    # cell reads "<10", directly and through the production node as a graph
    # build runs it (the node declares household_source_id).
    frame = _london_copies_at_the_cap(monkeypatch)
    assert rhv.assign_regional_home_values(frame).receipt["sources_at_cap"] == "<10"
    assert _graph_evidence(frame)["sources_at_cap"] == "<10"


def test_the_stage_refuses_a_frame_without_lineage() -> None:
    with pytest.raises(ValueError, match="household_source_id"):
        rhv.assign_regional_home_values(_frame(lineage=False))


def test_the_release_drops_the_rank_basis() -> None:
    assert rhv.RANK_BASIS_COLUMN in UK_RELEASE_EXPORT_DROPPED_COLUMNS["household"]


def test_the_stage_refuses_a_stored_rank_basis() -> None:
    staged = rhv.assign_regional_home_values(_frame()).frame
    with pytest.raises(ValueError, match="already stored"):
        rhv.assign_regional_home_values(staged)


def test_region_knots_must_be_positive_and_increasing() -> None:
    from importlib.resources import files

    payload = json.loads(
        files("microcosm.build.uk").joinpath(rhv.KNOTS_RESOURCE).read_text()
    )
    payload["levels"]["region"]["E12000007"]["quantiles"][1] = payload["levels"][
        "region"
    ]["E12000007"]["quantiles"][0]
    with pytest.raises(ValueError, match="strictly increasing"):
        rhv.load_region_knots(json.dumps(payload).encode())


def test_the_packaged_stage_runs_after_the_support_copies() -> None:
    stages = {stage.stage: stage for stage in load_country_spec("uk").sources.stages}
    rhv.UKRegionalHomeValuesStageTransform(stage=stages["regional_home_values"])
    order = list(stages)
    position = order.index("regional_home_values")
    assert order.index("high_value_home_support") == position - 1
    assert position < order.index("property_components")
