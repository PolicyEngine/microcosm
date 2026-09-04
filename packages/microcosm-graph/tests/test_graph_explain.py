"""The offline graph explanation page."""

from __future__ import annotations

import html
import importlib.util
import re
import sys
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path

import pytest

import microcosm.graph as graph_api
from microcosm.graph import describe, explain_html, graph_to_json

ROOT = Path(__file__).parents[3]

if "_toy" not in sys.modules:
    _TOY_SPEC = importlib.util.spec_from_file_location(
        "_toy", Path(__file__).with_name("_toy.py")
    )
    assert _TOY_SPEC is not None and _TOY_SPEC.loader is not None
    sys.modules["_toy"] = importlib.util.module_from_spec(_TOY_SPEC)
    _TOY_SPEC.loader.exec_module(sys.modules["_toy"])
toy = sys.modules["_toy"]


def _burndown() -> dict[str, object]:
    path = ROOT / "tools" / "graph_acceptance_burndown.py"
    spec = importlib.util.spec_from_file_location("_graph_burndown", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    data = module.report(module.counts(module.suite_files()))
    data["payoffs"] = {
        "unrelated_edit_misses": 0,
        "unrelated_edit_nodes": 9,
        "describe_target": "target_b",
    }
    return data


def _tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_{name}_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _snapshot(run) -> dict[str, object]:
    depths: dict[str, int] = {}
    nodes = []
    edges = []
    for node_id in run.compiled.order:
        predecessors = run.compiled.predecessors[node_id]
        depths[node_id] = (
            0
            if not predecessors
            else 1 + max(depths[predecessor] for predecessor in predecessors)
        )
        nodes.append(
            {
                "id": node_id,
                "key": run.manifest.nodes[node_id].key,
                "depth": depths[node_id],
                "state": "stable",
            }
        )
        edges.extend((predecessor, node_id) for predecessor in predecessors)
    return {"nodes": nodes, "edges": edges}


def _replays(run) -> list[dict[str, object]]:
    snapshot = _snapshot(run)
    definitions = (
        (
            "wic-dtype-breach",
            "WIC dtype breach",
            "B3",
            ("wic_recode",),
            ("wic_recode",),
            "The executor rejected the dense bool at the owning node.",
        ),
        (
            "0347a009-repack",
            "0347a009 repack",
            "C1 + C2",
            ("leaf_a", "leaf_b", "leaf_c", "leaf_d", "leaf_e"),
            (),
            "Five leaves were removed and zero survivor keys moved.",
        ),
        (
            "engine-less-environment",
            "Engine-less environment",
            "E2",
            (),
            (),
            "The verifier stopped the run before a kernel executed.",
        ),
        (
            "evidence-flip",
            "Evidence flip",
            "F3",
            (),
            (),
            "The loader refused the altered tier.",
        ),
    )
    return [
        {
            "id": identifier,
            "title": title,
            "property": property_id,
            "verdict": "pass",
            "changed_nodes": changed,
            "moved_keys": moved,
            "cell_changes": (),
            "observed": observed,
            "stages": (
                {
                    "label": "Before",
                    "summary": "Original evidence.",
                    "snapshot": snapshot,
                },
                {
                    "label": "Change",
                    "summary": "Incident-shaped edit.",
                    "snapshot": snapshot,
                },
                {"label": "After", "summary": observed, "snapshot": snapshot},
            ),
        }
        for identifier, title, property_id, changed, moved, observed in definitions
    ]


@pytest.fixture
def explanation(tmp_path: Path):
    run = toy.run_toy(toy.full_graph(), tmp_path / "run")
    charter = (ROOT / "docs" / "graph-acceptance.md").read_text(encoding="utf-8")
    rendered = explain_html(
        run.compiled,
        run.manifest,
        charter=charter,
        burndown=_burndown(),
        replays=_replays(run),
    )
    return run, charter, rendered


def test_explain_html_is_the_public_export() -> None:
    assert graph_api.explain_html is explain_html


def test_entrant_person_strata_survive_cache_and_are_explained(tmp_path: Path) -> None:
    expand, claim = toy.entrant_person_node()
    graph = toy.small_graph(nodes=(toy.CREATE, expand, claim))
    cold = toy.run_toy(graph, tmp_path / "cold")
    warm = toy.run_toy(
        graph,
        tmp_path / "warm",
        sources=cold.sources,
        registry=cold.registry,
        store=cold.store,
    )
    entrant_id = int(cold.manifest.population("survey").person["person_id"].max()) + 1

    assert warm.manifest.nodes[expand.id].hit
    assert warm.manifest.population(expand.id).strata.equals(
        cold.manifest.population(expand.id).strata
    )
    assert cold.manifest.nodes[expand.id].receipt["entrant_strata"] == (
        (entrant_id, "urban"),
    )
    assert (
        warm.manifest.nodes[expand.id].receipt["entrant_strata"]
        == cold.manifest.nodes[expand.id].receipt["entrant_strata"]
    )
    detail = describe(cold.compiled, expand.id, cold.manifest)
    rendered = explain_html(cold.compiled, cold.manifest)
    assert "entrant_strata" in detail and "urban" in detail
    assert "entrant_strata" in rendered and "urban" in rendered


def test_page_contains_every_node_and_its_click_detail(explanation) -> None:
    run, _charter, rendered = explanation
    for node_id in run.compiled.order:
        assert f">{html.escape(node_id)}<" in rendered
        assert html.escape(describe(run.compiled, node_id, run.manifest)) in rendered
    assert rendered.count('data-node-detail="') == len(run.compiled.order)
    assert "Predecessor keys" in rendered
    assert "Kernel capabilities" in rendered
    assert "Row mask" in rendered


def test_page_contains_every_charter_property(explanation) -> None:
    _run, charter, rendered = explanation
    identifiers = re.findall(r"^\|\s*([A-Z]\d+)\s*\|", charter, re.MULTILINE)
    assert (
        len(dict.fromkeys(identifiers)) == 45
    )  # 41 + B6, C5, D6, B7 (amendments 11-14)
    for identifier in identifiers:
        assert f"<code>{identifier}</code>" in rendered
    assert "35 green" not in rendered  # V1-V4 are also represented.
    assert "45 green" in rendered
    assert "0 red" in rendered
    assert "<th>Flip PR</th>" in rendered
    assert "Not recorded" in rendered


def test_burndown_renders_available_flip_pr(tmp_path: Path) -> None:
    run = toy.run_toy(toy.full_graph(), tmp_path / "run")
    burndown = _burndown()
    burndown["properties"][0]["pr"] = "#123"

    rendered = explain_html(run.compiled, run.manifest, burndown=burndown)

    assert "#123" in rendered


def test_calibration_view_uses_targets_ratios_and_mass(explanation) -> None:
    _run, _charter, rendered = explanation
    assert "Calibration view" in rendered
    assert "calibrated" in rendered
    assert "household_size" in rendered
    assert "2,500" in rendered
    assert "Weight ratios against design" in rendered
    assert 'class="chart-bar chart-bar-before"' in rendered
    assert 'class="chart-bar chart-bar-after"' in rendered
    assert "Before (n=200)" in rendered
    assert "After (n=200)" in rendered
    assert "Mass ledger" in rendered
    assert "rural" in rendered
    assert "urban" in rendered


def test_calibration_view_renders_partition_mass_with_deltas(tmp_path: Path) -> None:
    run = toy.run_toy(toy.full_graph(), tmp_path / "run")
    original = run.manifest.nodes["calibrated"]
    mass = dict(original.receipt["mass"])
    mass["stratum_before"] = {}
    mass["stratum_after"] = {}
    mass["partition"] = {
        "entity": "household",
        "column": "period",
        "stratum_before": {
            "2024": {"rural": 10.0, "urban": 20.0},
            "2025": {"rural": 5.0},
        },
        "stratum_after": {
            "2024": {"rural": 8.0, "urban": 23.0},
            "2026": {"urban": 10.0},
        },
    }
    changed = replace(
        original,
        receipt={**dict(original.receipt), "mass": mass},
    )
    manifest = replace(
        run.manifest,
        nodes={**dict(run.manifest.nodes), "calibrated": changed},
    )

    rendered = explain_html(run.compiled, manifest)

    assert "Mass by household.period partition" in rendered
    assert (
        "<th>Partition value</th><th>Stratum</th><th>Before</th><th>After</th>"
        "<th>Change</th></tr></thead><tbody>" in rendered
    )
    assert ">2024</td><td>rural</td><td>10</td><td>8</td><td>-2</td>" in rendered
    assert ">2024</td><td>urban</td><td>20</td><td>23</td><td>3</td>" in rendered
    assert ">2025</td><td>rural</td><td>5</td><td>0</td><td>-5</td>" in rendered
    assert ">2026</td><td>urban</td><td>0</td><td>10</td><td>10</td>" in rendered
    assert "http://" not in rendered and "https://" not in rendered


def test_calibration_view_fallback_renders_partitioned_mass_record(
    tmp_path: Path,
) -> None:
    run = toy.run_toy(toy.full_graph(), tmp_path / "run")
    original_receipt = run.manifest.nodes["calibrated"]
    receipt_without_mass = dict(original_receipt.receipt)
    receipt_without_mass.pop("mass")
    changed_receipt = replace(original_receipt, receipt=receipt_without_mass)
    original_record = next(
        record
        for record in run.manifest.mass_ledgers["calibrated"]
        if record.node_id == "calibrated"
    )
    partitioned_record = replace(
        original_record,
        partition_entity="household",
        partition_column="period",
        before_by_partition_stratum=((2024, (("rural", 10.0),)),),
        after_by_partition_stratum=((2024, (("rural", 8.0),)),),
    )
    manifest = replace(
        run.manifest,
        nodes={**dict(run.manifest.nodes), "calibrated": changed_receipt},
        mass_ledgers={"calibrated": (partitioned_record,)},
    )

    rendered = explain_html(run.compiled, manifest)

    assert "Mass by household.period partition" in rendered
    assert ">2024</td><td>rural</td><td>10</td><td>8</td><td>-2</td>" in rendered


def test_calibration_view_reads_adam_diagnostics(tmp_path: Path) -> None:
    run = toy.run_toy(toy.full_graph(), tmp_path / "run")
    targets = (
        ("income", "income", None, 390.0, 7.25),
        ("eligible_income", "income", "eligible", 240.0, 3.0),
    )
    calibrate = replace(
        run.compiled.graph.node("calibrated"),
        kernel="calibrate.adam@1",
        params={
            "targets": targets,
            "epochs": 24,
            "learning_rate": 0.03,
            "mass": "declared",
            "max_weight_ratio": 5.0,
            "weight_anchor": "design",
        },
    )
    compiled = graph_api.compile_graph(toy.replace_node(run.compiled.graph, calibrate))
    old_receipt = run.manifest.nodes["calibrated"]
    adam_receipt = replace(
        old_receipt,
        kernel_ref="calibrate.adam@1",
        receipt={
            "declared_targets": targets,
            "diagnostics": {
                "targets": (
                    {"target_name": "income", "final_estimate": 392.5},
                    {"target_name": "eligible_income", "final_estimate": 239.0},
                )
            },
        },
    )
    manifest = replace(
        run.manifest,
        nodes={**dict(run.manifest.nodes), "calibrated": adam_receipt},
    )

    rendered = explain_html(compiled, manifest)

    assert "calibrate.adam@1" in rendered
    assert "eligible_income" in rendered
    assert "eligible" in rendered
    assert "392.5" in rendered
    assert "239" in rendered
    assert ">2.5<" in rendered
    assert ">-1<" in rendered
    assert ">7.25<" in rendered


def test_four_passing_replay_panels_are_step_through_views(explanation) -> None:
    _run, _charter, rendered = explanation
    for identifier, title in (
        ("wic-dtype-breach", "WIC dtype breach"),
        ("0347a009-repack", "0347a009 repack"),
        ("engine-less-environment", "Engine-less environment"),
        ("evidence-flip", "Evidence flip"),
    ):
        assert f'data-replay="{identifier}"' in rendered
        assert f"<h3>{title}</h3>" in rendered
    assert rendered.count('<span class="badge pass">pass</span>') == 4
    assert rendered.count('data-stage="0"') >= 8
    assert "4 of 4 pass" in rendered


def test_page_is_offline_deterministic_and_parseable(explanation) -> None:
    run, charter, rendered = explanation
    assert "http://" not in rendered
    assert "https://" not in rendered
    assert "<script src" not in rendered.lower()
    parser = HTMLParser()
    parser.feed(rendered)
    parser.close()
    assert rendered == explain_html(
        run.compiled,
        run.manifest,
        charter=charter,
        burndown=_burndown(),
        replays=_replays(run),
    )


def test_gate_outcome_is_text_not_markup(tmp_path: Path) -> None:
    run = toy.run_toy(toy.full_graph(), tmp_path / "run")
    gate = run.manifest.nodes["gate_tax"]
    outcome = 'pass" onmouseover="alert(1)'
    changed_gate = replace(
        gate,
        receipt={**dict(gate.receipt), "outcome": outcome},
    )
    manifest = replace(
        run.manifest,
        nodes={**dict(run.manifest.nodes), "gate_tax": changed_gate},
    )

    rendered = explain_html(run.compiled, manifest)

    assert "gate-unknown" in rendered
    assert 'onmouseover="alert(1)"' not in rendered
    assert "pass&amp;quot;" not in rendered
    assert "pass&quot; onmouseover=&quot;alert(1)" in rendered


def test_manifest_only_page_omits_optional_sections(tmp_path: Path) -> None:
    run = toy.run_toy(toy.full_graph(), tmp_path / "run")
    rendered = explain_html(run.compiled, run.manifest)
    assert "Graph explorer" in rendered
    assert "Calibration view" in rendered
    assert "Acceptance burndown" not in rendered
    assert "Incident replays" not in rendered


def test_saved_run_cli_validates_store_and_reattaches_frames(tmp_path: Path) -> None:
    run = toy.run_toy(toy.full_graph(), tmp_path / "run")
    manifest_path = tmp_path / "run" / "manifest.json"
    graph_path = tmp_path / "run" / "graph.json"
    output_path = tmp_path / "rendered.html"
    run.manifest.save(manifest_path)
    graph_path.write_text(graph_to_json(run.compiled.graph), encoding="utf-8")

    tool = _tool("graph_explain")
    tool.render_saved_run(manifest_path, graph_path, output_path)

    rendered = output_path.read_text(encoding="utf-8")
    assert "Graph explorer" in rendered
    assert "Calibration view" in rendered
    assert 'class="chart-bar chart-bar-before"' in rendered
    assert 'class="chart-bar chart-bar-after"' in rendered
    assert "rural" in rendered
    assert "urban" in rendered


def test_demo_runs_live_replays_and_is_byte_identical(tmp_path: Path) -> None:
    tool = _tool("graph_demo_run")
    output = tool.build_demo(tmp_path / "demo")
    first = output.read_bytes()
    second = tool.build_demo(tmp_path / "demo").read_bytes()

    assert first == second
    assert len(first) < 2_000_000
    rendered = first.decode("utf-8")
    parser = HTMLParser()
    parser.feed(rendered)
    parser.close()
    for heading in (
        "Graph explorer",
        "Acceptance burndown",
        "Calibration view",
        "Incident replays",
    ):
        assert heading in rendered
    assert "4 of 4 pass" in rendered
    assert rendered.count('<span class="badge pass">pass</span>') == 4
    assert "0 of 9 nodes (0.0%)" in rendered
    assert (tmp_path / "demo" / "manifest.json").is_file()
    assert (tmp_path / "demo" / "graph.json").is_file()
    assert (tmp_path / "demo" / "store").is_dir()

    toy_module = tool._load_toy()
    baseline = toy_module.run_toy(
        toy_module.full_graph(), tmp_path / "live-replays" / "baseline"
    )
    replays = tool._run_replays(
        toy_module, tmp_path / "live-replays" / "incidents", baseline
    )
    replay_by_id = {replay["id"]: replay for replay in replays}

    expected_boundaries = {
        "wic-dtype-breach": (("wic_recode",), ("wic_recode",)),
        "0347a009-repack": (
            ("leaf_a", "leaf_b", "leaf_c", "leaf_d", "leaf_e"),
            (),
        ),
        "engine-less-environment": ((), ()),
        "evidence-flip": ((), ()),
    }
    assert set(replay_by_id) == set(expected_boundaries)
    for identifier, (changed_nodes, moved_keys) in expected_boundaries.items():
        replay = replay_by_id[identifier]
        assert replay["verdict"] == "pass"
        assert replay["changed_nodes"] == changed_nodes
        assert replay["moved_keys"] == moved_keys
        assert tuple(stage["label"] for stage in replay["stages"]) == (
            "Before",
            "Change",
            "After",
        )

    wic_after = replay_by_id["wic-dtype-breach"]["stages"][2]["snapshot"]
    wic_states = {node["id"]: node["state"] for node in wic_after["nodes"]}
    assert wic_states["wic_recode"] == "rejected"
    assert wic_states["consumes_the_recode"] == "not-executed"

    removed = set(expected_boundaries["0347a009-repack"][0])
    repack_stages = replay_by_id["0347a009-repack"]["stages"]
    repack_change = {
        node["id"]: node["state"] for node in repack_stages[1]["snapshot"]["nodes"]
    }
    assert {
        node_id for node_id, state in repack_change.items() if state == "removed"
    } == removed
    assert removed.isdisjoint(
        node["id"] for node in repack_stages[2]["snapshot"]["nodes"]
    )

    engine_after = replay_by_id["engine-less-environment"]["stages"][2]["snapshot"]
    assert {node["state"] for node in engine_after["nodes"]} == {"not-executed"}

    evidence_stages = replay_by_id["evidence-flip"]["stages"]
    evidence_change = {
        node["id"]: node["state"] for node in evidence_stages[1]["snapshot"]["nodes"]
    }
    evidence_after = {
        node["id"]: node["state"] for node in evidence_stages[2]["snapshot"]["nodes"]
    }
    assert evidence_change["release"] == "changed"
    assert evidence_after["release"] == "refused"
