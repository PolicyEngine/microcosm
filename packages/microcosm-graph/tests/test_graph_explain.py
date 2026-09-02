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
    assert len(dict.fromkeys(identifiers)) == 41
    for identifier in identifiers:
        assert f"<code>{identifier}</code>" in rendered
    assert "35 green" not in rendered  # V1-V4 are also represented.
    assert "39 green" in rendered
    assert "2 red" in rendered


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
