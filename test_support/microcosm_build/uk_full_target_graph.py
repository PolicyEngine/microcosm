"""Synthetic target inputs, the preflight kernel and the full-graph build
helper shared by the engine-free and engine halves of the target-graph
tests."""

# ruff: noqa: F401

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import (
    full_targets,
    geography_ladder,
    graph_targets,
    ledger_targets,
)
from microcosm.build.uk_runtime.graph_build import (
    UKFullBuildConfig,
    register_uk_full_kernels,
    uk_full_graph,
)
from microcosm.build.uk_runtime.graph_calibration import UKGraphCalibrationConfig
from microcosm.build.uk_runtime.graph_terminal import FULL_GATE_REPORT_TYPE
from microcosm.build.uk_runtime.local_rowwise import UKRowwiseNationalRows
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.calibrate.artifacts import decode_problem
from microcosm.frame import Frame
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    compile_graph,
    run_graph,
)
from test_support.microcosm_build.uk_full_calibration_graph import (
    preflight_payload,
)
from test_support.microcosm_build.uk_full_population_graph import (
    Source,
    graph_and_registry,
)
from test_support.microcosm_build.uk_ladder_rowwise_clone import (
    toy_ladder as toy_ladder,
)


@pytest.fixture
def target_inputs(monkeypatch, toy_ladder):
    national = TargetRegistry(
        [
            TargetSpec(
                name="country_households",
                entity="household",
                measure="test_ones",
                value=34.0,
                period=2026,
                family="fixture",
                source="fixture",
                metadata={
                    "geography_level": "country",
                    "geography_id": "UK",
                    "contract_target_id": "obr.vat",
                },
            ),
            TargetSpec(
                # Repeated names across periods must retain distinct metadata.
                name="country_households",
                entity="household",
                measure="test_ones",
                value=4.0,
                period=2025,
                family="fixture",
                source="fixture",
                filter="test_london",
                metadata={
                    "geography_level": "region",
                    "geography_id": "LONDON",
                    # A region-level contract target on main (#905 SPI region
                    # facts); #934 moved council-tax stock to local-authority
                    # publisher contracts, so the old voa.* id no longer exists.
                    "contract_target_id": (
                        "hmrc.spi_region.income_tax_by_region_50000_70000"
                    ),
                },
            ),
        ],
        country="uk",
    )
    ladder, _ = toy_ladder
    local = TargetRegistry(
        [
            TargetSpec(
                name=f"ons.census.households@{level}:{code}",
                entity="household",
                measure="household_count",
                value=35.0 if level == "local_authority" and i == 0 else 40.0,
                period=2026,
                family="census_households",
                source="chronicle_fixture",
                metadata={
                    "contract_target_id": "ons.census.households",
                    "geography_level": level,
                    "geography_id": str(code),
                    "uprating_from_period": 2022 if str(code).startswith("S") else 2021,
                    "uprating_to_period": 2026,
                },
            )
            for level, codes in (
                ("constituency", ladder.constituency_code),
                ("local_authority", ladder.local_authority_code),
            )
            for i, code in enumerate(codes)
        ],
        country="uk",
    )
    inputs = {
        "national_registry": national,
        "band_edge_registry": national,
        "local_registry": local,
        "artifact": SimpleNamespace(facts=()),
        "calibration_year": 2026,
        "measure_exclusions": {},
        "reviewed_unbound_higher_targets": {},
        "national_source_pin": {"fixture": True},
        "local_source_pin": {"fixture": True},
        "register_completeness": {"fixture": True},
        "ledger_provenance": {"fixture": True},
        "uk_ledger_compiled_registries": {2026: national},
        "uk_ledger_compiled_local_registries": {2026: local},
    }
    monkeypatch.setattr(
        full_targets, "load_uk_full_target_inputs", lambda *args, **kwargs: inputs
    )
    reference = {"value": 33.0, "period": 2026}
    monkeypatch.setattr(
        graph_targets, "uk_ledger_households_total", lambda *args, **kwargs: reference
    )

    def surface():
        return ledger_targets.uk_local_target_surface(
            graph_targets.full_problem._joint_surface_registry(local, national),
            bound_national_target_ids=graph_targets.full_problem._national_contract_target_ids(
                national
            ),
            period=2026,
            # #906: the toy ladder's membership, as the graph and the driver pass.
            area_region_codes=geography_ladder.uk_area_region_codes(ladder),
            census_household_uprating=ledger_targets.uk_census_household_uprating(
                local, reference, period=2026
            ),
        )

    def measures(frame, national_registry, *, local_grains, **kwargs):
        tables = {e: frame.table(e).copy() for e in frame.entities}
        tables["household"]["test_ones"] = 1.0
        tables["household"]["test_london"] = (
            tables["household"]["region"] == "LONDON"
        ).astype(float)
        prepared = Frame(
            tables,
            frame.schema,
            {"household": frame.weights_for("household")},
            frame.strata,
            mass_log=frame.mass_log,
            metadata=frame.metadata,
        )
        return (
            prepared,
            lambda _: frame,
            UKRowwiseNationalRows(
                national_registry.to_target_set(), national_registry, ("fixture",)
            ),
            {
                g: pd.DataFrame(
                    {"households": np.ones(frame.n("household"))},
                    index=frame.table("household")["household_id"],
                )
                for g in local_grains
            },
            {"fixture": True},
        )

    monkeypatch.setattr(graph_targets, "resolve_uk_full_measures", measures)
    return {
        "national": national,
        "local": local,
        "inputs": inputs,
        "surface": surface,
        "measures": measures,
    }


class Preflight(KernelBase):
    """Synthetic source verdict: tests below exercise numerical graph ownership."""

    ref = "uk.test.target-preflight@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC)

    def run(self, context):
        selection = json.loads(context.artifacts["selection"].payload)["receipt"]
        return KernelResult(
            artifacts={"preflight": preflight_payload(selection=selection)}
        )


def build(
    tmp_path,
    ladder_path,
    levels,
    *,
    n_clones=1,
    seed=7,
    dataset_households=None,
    resume="auto",
    forbid_execution=False,
):
    primitive, _ = graph_and_registry(1)
    base = Graph(
        "uk",
        tuple(s for s in primitive.sources if s.name == "fixture"),
        (primitive.node("source"),),
    )
    config = UKFullBuildConfig(
        calibration_year=2026,
        time_period="2023",
        source_year=2023,
        n_clones=n_clones,
        geography_levels=levels,
        seed=seed,
        calibration=UKGraphCalibrationConfig(
            epochs=8, seed=seed, dataset_households=dataset_households
        ),
    )
    full = uk_full_graph(config, spine=base, spine_population="source")
    preflight = Node(
        "fixture.preflight",
        Preflight.ref,
        population="uk.full.pool",
        artifact_inputs=(
            ArtifactInput(
                "selection",
                "uk.full.target_selection",
                "selection",
                graph_targets.TARGET_SELECTION_TYPE,
            ),
        ),
        artifact_outputs=(ArtifactOutput("preflight", FULL_GATE_REPORT_TYPE),),
    )
    full = replace(
        full,
        graph=replace(
            full.graph,
            nodes=(
                *(
                    replace(
                        node,
                        artifact_inputs=(
                            *node.artifact_inputs,
                            ArtifactInput(
                                "preflight",
                                preflight.id,
                                "preflight",
                                FULL_GATE_REPORT_TYPE,
                            ),
                        ),
                    )
                    if node.id == full.calibration.dense_producer
                    else node
                    for node in full.graph.nodes
                ),
                preflight,
            ),
        ),
    )
    registry = KernelRegistry()
    registry.register(Source())
    registry.register(Preflight())
    register_uk_full_kernels(registry)
    if forbid_execution:
        for kernel in registry.as_mapping().values():
            kernel.run = lambda *args, **kwargs: pytest.fail(
                "cached full graph executed"
            )
    store = ContentStore(tmp_path / "store")
    manifest = run_graph(
        compile_graph(full.graph),
        sources={
            "fixture": ladder_path,
            "uk_ladder": ladder_path,
            "uk_ledger_facts": ladder_path,
        },
        store=store,
        kernels=registry,
        resume=resume,
    )
    problem = decode_problem(
        store.load_bytes(manifest.nodes["uk.full.problem"].opaque_artifacts["problem"])
    )
    return full, manifest, problem


__all__ = [name for name in globals() if not name.startswith("__")]
