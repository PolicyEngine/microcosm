"""The spine graph's engine-free prefix at two geographic clones (microcosm#931).

The H2 fixture's invented sources run through the composed spine graph's own
nodes and kernels, in the pull-request tier, at one geographic clone and at
two, through rfdr: the last stage after geographic_expand that reads no
engine (frs_take_up, next, reads the engine's working-age bounds). Three
stages before the clones read the engine or its parameters: frs_disability
(the DWP rates), frs_legacy_proxies and frs_education_grant_split (engine
values and their policies). They run here on invented rates, engine values
and policies, on the same unexpanded frame at both clone counts. Every other
node runs as the fixture binds it. The engine is blocked: an import of
policyengine-uk, or a call of the engine adapter, fails the run.

What the design says nests, and is held here:

- every node before geographic_expand holds the same frame at both counts;
- from geographic_expand on, through geographic_identity and the area stages
  (atomic_area_assign, local_authority, rfdr), clone 0's rows of every table
  are the one-clone frame's, every column and dtype, bit for bit: the
  expansion makes clone 0 the one-clone frame, geographic_identity keys each
  row on its source, the area draw keys each row on its geography key and
  picks its level from the row's own inputs, and the local authority and the
  fuel-relief flag read the drawn area;
- two times each clone-0 household weight is the one-clone weight, bitwise,
  since two is a power of two;
- every table carries clone indices 0 and 1.

The after-merge test_h2_clone_first.py runs the whole spine at four clones
through the engine. Invented data only; messages name nodes, entities and
counts, never a value.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.frs_disability import (
    UKDWPDisabilityCategoryRates,
    UKDWPDisabilityFlagRates,
    UKFRSDisabilityStageTransform,
)
from microcosm.build.uk_runtime.frs_education_grants import (
    UKDSAPolicy,
    UKFRSEducationGrantSplitStageTransform,
)
from microcosm.build.uk_runtime.frs_legacy_proxies import (
    UKFRSLegacyProxiesStageTransform,
    UKLegacyJSAPolicy,
)
from microcosm.build.uk_runtime.geographic_clones import ladder_clone_index_column
from microcosm.build.uk_runtime.geographic_expand import GEOGRAPHIC_EXPAND_STAGE_NAME
from microcosm.build.uk_runtime.graph import uk_registry, uk_spine_graph
from microcosm.build.uk_runtime.graph_kernels import fixture_stage_plan_inputs
from microcosm.build.uk_runtime.spine_geography import RFDR_STAGE_NAME
from microcosm.frame import Frame
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine
from microcosm.graph import ContentStore, compile_graph, run_graph
from test_support.microcosm_graph.acceptance_h_parity import UK_SPINE_PARITY, _require

G1 = GEOGRAPHIC_EXPAND_STAGE_NAME
#: The clone count held to the fixture's one clone.
_CLONES = 2
#: The last stage of the prefix: the last after geographic_expand that reads
#: no engine.
_LAST = RFDR_STAGE_NAME
#: The stages before the clones that read the engine or its parameters, each
#: run on invented values (:func:`_engine_free`).
_STOOD_IN = ("frs_disability", "frs_legacy_proxies", "frs_education_grant_split")


def _roster() -> tuple[str, ...]:
    """The packaged roster's stage names, in order, read from
    source_stages.json: names only, never a stage's notes."""

    roster = json.loads(
        files("microcosm.build.uk")
        .joinpath("source_stages.json")
        .read_text(encoding="utf-8")
    )
    return tuple(str(stage["stage"]) for stage in roster["stages"])


_ROSTER = _roster()
#: The roster's stages the prefix runs, from the root through rfdr.
PREFIX_STAGES = _ROSTER[: _ROSTER.index(_LAST) + 1]


class _InventedEngine:
    """Invented engine values for the stages before the clones: a State
    Pension age of 66 for everyone and zero for every other variable. They
    run on the same unexpanded frame at both clone counts."""

    country = "uk"

    def materialize(self, frame, variables, period, **kwargs):
        people = len(frame.table("person"))
        return {
            variable: np.full(people, 66.0)
            if variable == "state_pension_age"
            else np.zeros(people)
            for variable in variables
        }


def _engine_free(
    implementations: dict[str, object], stages: dict[str, object]
) -> dict[str, object]:
    """The fixture's transforms with the three stages before the clones that
    read the engine or its parameters on invented rates, engine values and
    policies."""

    legacy = implementations["frs_legacy_proxies"]
    return {
        **implementations,
        "frs_disability": UKFRSDisabilityStageTransform(
            stage=stages["frs_disability"],
            category_rates=UKDWPDisabilityCategoryRates(
                aa_lower=70.0,
                aa_higher=105.0,
                dla_sc_lower=28.0,
                dla_sc_middle=70.0,
                dla_sc_higher=105.0,
                dla_m_lower=28.0,
                dla_m_higher=75.0,
                pip_m_standard=30.0,
                pip_m_enhanced=80.0,
                pip_dl_standard=70.0,
                pip_dl_enhanced=105.0,
                instant="2024-04-01",
                source="invented",
            ),
            flag_rates=UKDWPDisabilityFlagRates(
                aa_higher=105.0,
                dla_sc_higher=105.0,
                pip_dl_enhanced=105.0,
                instant="2024-04-01",
                source="invented",
            ),
            carer_min_hours=35.0,
        ),
        "frs_legacy_proxies": UKFRSLegacyProxiesStageTransform(
            legacy.raw_dir,
            stage=stages["frs_legacy_proxies"],
            engine=_InventedEngine(),
            policy=UKLegacyJSAPolicy(
                max_weekly_hours_single=16.0, instant="2024-01-01", source="invented"
            ),
        ),
        "frs_education_grant_split": UKFRSEducationGrantSplitStageTransform(
            stage=stages["frs_education_grant_split"],
            engine=_InventedEngine(),
            policy=UKDSAPolicy(
                maximum=1_000.0, instant="2025-01-01", source="invented"
            ),
        ),
    }


def _sources(directory: Path, clones: int) -> Path:
    """The H2 fixture's sources, at ``clones`` geographic clones."""

    source = _require(UK_SPINE_PARITY / "sources", "tools/graph_uk_spine_fixture.py")
    if clones == 1:
        return source
    target = directory / "sources"
    shutil.copytree(source, target)
    descriptor = json.loads((target / "fixture.json").read_text(encoding="utf-8"))
    assert descriptor["config"]["geographic_clones"] == 1
    descriptor["config"]["geographic_clones"] = clones
    (target / "fixture.json").write_text(
        json.dumps(descriptor, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def _through(graph, endpoint: str):
    """The graph's ancestor-closed part that ends at ``endpoint``."""

    compiled = compile_graph(graph)
    needed, pending = {endpoint}, [endpoint]
    while pending:
        for parent in compiled.predecessors[pending.pop()]:
            if parent not in needed:
                needed.add(parent)
                pending.append(parent)
    return replace(
        graph, nodes=tuple(node for node in graph.nodes if node.id in needed)
    )


def _run(directory: Path, clones: int) -> tuple[tuple[str, ...], dict[str, Frame]]:
    """The prefix at ``clones``: the compiled order and each node's frame."""

    source = _sources(directory, clones)
    stages, implementations = fixture_stage_plan_inputs(source)
    by_name = {stage.stage: stage for stage in stages}
    graph = uk_spine_graph(geographic_clones=clones)
    nodes = {node.id for node in graph.nodes}
    end = f"{_LAST}.owned" if f"{_LAST}.owned" in nodes else _LAST
    compiled = compile_graph(_through(graph, end))
    frames: dict[str, Frame] = {}

    def observe(node_id: str, population) -> None:
        frames[node_id] = population.frame

    run_graph(
        compiled,
        sources={"frs": source},
        store=ContentStore(directory / "store"),
        # Built against the whole spine graph, as the spine build does.
        kernels=uk_registry(_engine_free(dict(implementations), by_name), graph=graph),
        resume="forbid",
        decisions=(),
        _population_observer=observe,
    )
    return tuple(compiled.order), frames


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[int, tuple[tuple[str, ...], dict[str, Frame]]]:
    """The prefix at one clone and at two, with the engine blocked."""

    def no_engine(self, *args, **kwargs):
        raise AssertionError("the spine prefix called the engine")

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "policyengine_uk", None)
        patch.setattr(PolicyEngineUKEngine, "materialize", no_engine)
        return {
            clones: _run(tmp_path_factory.mktemp(f"k{clones}"), clones)
            for clones in (1, _CLONES)
        }


def _stage_nodes(order: tuple[str, ...]) -> tuple[str, ...]:
    """The stage nodes of a compiled order, the root's as its stage."""

    stages = []
    for node in order:
        name = "frs_spine" if node == "create_uk_frs" else node.split(".")[0]
        if name not in stages:
            stages.append(name)
    return tuple(stages)


def test_the_prefix_runs_every_roster_stage_through_rfdr(runs) -> None:
    """The prefix is the roster's own, read from source_stages.json: every
    stage from the root through rfdr, in order, geographic_expand among them,
    and the same nodes at both clone counts."""

    order, _ = runs[1]
    assert runs[_CLONES][0] == order
    assert _stage_nodes(order) == PREFIX_STAGES
    assert G1 in PREFIX_STAGES
    assert set(_STOOD_IN) < set(PREFIX_STAGES[: PREFIX_STAGES.index(G1)])


def test_every_node_before_the_clones_holds_one_frame_at_both_counts(runs) -> None:
    (order, one), (_, two) = runs[1], runs[_CLONES]
    for node in order[: order.index(G1)]:
        for entity in one[node].entities:
            pd.testing.assert_frame_equal(
                two[node].table(entity), one[node].table(entity), check_exact=True
            )
        assert np.array_equal(
            np.asarray(two[node].weights_for("household").values),
            np.asarray(one[node].weights_for("household").values),
        ), f"{node}: the household weights differ before the clones."


def test_clone_zero_is_the_one_clone_frame_through_rfdr(runs) -> None:
    """From geographic_expand through rfdr, clone 0's rows of every table are
    the one-clone frame's, every column and dtype, bit for bit, and two times
    each clone-0 household weight is the one-clone weight, bitwise."""

    (order, one), (_, two) = runs[1], runs[_CLONES]
    checked = order[order.index(G1) :]
    assert checked[-1].split(".")[0] == _LAST
    for node in checked:
        for entity in one[node].entities:
            table = two[node].table(entity)
            index = table[ladder_clone_index_column(entity)]
            assert sorted(set(index.tolist())) == list(range(_CLONES)), (
                f"{node}: the {entity} table does not carry both clone indices."
            )
            pd.testing.assert_frame_equal(
                table.loc[index.eq(0)].reset_index(drop=True),
                one[node].table(entity).reset_index(drop=True),
                check_exact=True,
            )
        household = two[node].table("household")
        zero = household[ladder_clone_index_column("household")].eq(0).to_numpy()
        halves = np.asarray(two[node].weights_for("household").values, dtype=float)
        assert np.array_equal(
            2.0 * halves[zero],
            np.asarray(one[node].weights_for("household").values, dtype=float),
        ), f"{node}: two times clone 0's household weights are not one clone's."
