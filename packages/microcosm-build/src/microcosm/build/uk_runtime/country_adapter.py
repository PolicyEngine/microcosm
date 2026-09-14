"""Bridge the UK country declarations to the canonical executable full graph.

The F0 compiler remains a static schema/identity projection. This adapter
validates its source projection, then uses the existing UK graph composition;
it never loads a historical candidate H5 or creates a second execution graph.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import date
from importlib import metadata
from typing import TYPE_CHECKING

from microcosm.build.country_spec import CountrySpec, load_country_spec
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release

if TYPE_CHECKING:
    from .graph_build import UKFullBuildConfig, UKFullGraph


def validate_uk_country_source_projection(spec: CountrySpec) -> None:
    """Keep raw input declarations and their executable stage pins identical."""
    if spec.country != "uk" or spec.resolved_spec is None:
        raise ValueError(
            "The UK full-build adapter requires a resolved UK country spec."
        )
    resolved = spec.resolved_spec
    sources = resolved.resource("sources").domain.to_wire()
    spine = resolved.resource("spine").domain.to_wire()
    bundle = resolved.resource("bundle").domain.to_wire()
    release = load_uk_frs_release()
    root = next(stage for stage in spec.sources.stages if stage.stage == "frs_spine")
    vintage = f"uk_frs_{release.vintage}"
    expected = [
        {
            "id": f"frs_{artifact['table']}",
            "role": "frs_raw_table",
            "sha256": artifact["sha256"],
            "byte_size": artifact["size_bytes"],
            "loader": "kernel:build_uk_frs_spine",
            "vintages": [f"vintage:{vintage}"],
            **(
                {
                    "vintage_authorities": [
                        {
                            "id": vintage,
                            "kind": "survey_period",
                            "value": release.survey_year,
                        }
                    ]
                }
                if artifact["table"] == "adult"
                else {}
            ),
        }
        for artifact in root.artifacts
        if artifact["role"] == "frs_table"
    ]
    if sources["sources"] != expected:
        raise ValueError(
            "UK country raw-source pins differ from the canonical FRS spine stage."
        )
    if spine["channels"] != [
        {
            "id": "frs",
            "source": [row["id"] for row in expected],
            "observed_geography": "region",
        }
    ]:
        raise ValueError("UK FRS channel must consume the declared raw FRS tables.")
    if bundle["dataset_run"]["target_period"] != release.calibration_year:
        raise ValueError(
            "UK country target period differs from the FRS release calibration year."
        )


def build_uk_country_graph(
    config: UKFullBuildConfig | None = None,
    *,
    spec: CountrySpec | None = None,
    engine_identity: str | None = None,
    review_date: date | None = None,
    release_candidate: bool = False,
    skip_holdout: bool = False,
) -> UKFullGraph:
    """Compile the canonical full graph; default target scope is all geographies."""
    from microcosm.graph import compile_graph
    from microcosm.graph.canonical import canonical_json

    from .graph import uk_spine_graph
    from .graph_build import UKFullBuildConfig, uk_full_graph
    from .graph_evidence import add_uk_spine_gate_nodes
    from .graph_terminal import append_uk_full_gate_nodes

    spec = load_country_spec("uk") if spec is None else spec
    validate_uk_country_source_projection(spec)
    release = load_uk_frs_release()
    if config is None:
        config = UKFullBuildConfig(
            calibration_year=release.calibration_year,
            time_period=release.time_period,
            source_year=release.survey_year,
        )
    if engine_identity is None:
        engine_identity = hashlib.sha256(
            canonical_json(
                {
                    "package": "policyengine-uk",
                    "version": metadata.version("policyengine-uk"),
                }
            )
        ).hexdigest()
    review_date = date.today() if review_date is None else review_date
    spine = add_uk_spine_gate_nodes(
        uk_spine_graph(spec, source_mode="split"),
        spec=spec,
        engine_identity=engine_identity,
        release_candidate=release_candidate,
    )
    full = uk_full_graph(config, spine=spine, review_date=review_date.isoformat())
    graph = append_uk_full_gate_nodes(
        full.graph,
        calibration=full.calibration,
        spine_stage_names=tuple(stage.stage for stage in spec.sources.stages),
        engine_identity=engine_identity,
        review_date=review_date,
        sample_fraction=config.effective_sample_fraction,
        release_candidate=release_candidate,
        skip_holdout=skip_holdout,
    )
    compile_graph(graph)
    return replace(full, graph=graph)
