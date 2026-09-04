"""One-file, offline visual explanation of a graph run."""

from __future__ import annotations

import html
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from .decl import GATE_OUTCOMES, CompiledGraph, StructuralDelta
from .manifest import NodeReceipt, RunManifest
from .population import mass_record_receipt
from .view import describe

if TYPE_CHECKING:
    from .decl import Node

__all__ = ["explain_html"]

_NODE_WIDTH = 194
_NODE_HEIGHT = 96
_X_GAP = 62
_Y_GAP = 24
_GROUP_PAD = 24

_STYLE = r"""
:root {
  color-scheme: light dark;
  --color-blue-600: #2563eb;
  --color-blue-100: #dbeafe;
  --color-green-600: #15803d;
  --color-green-100: #dcfce7;
  --color-red-600: #b91c1c;
  --color-red-100: #fee2e2;
  --color-amber-600: #a16207;
  --color-amber-100: #fef3c7;
  --text: #172033;
  --muted: #5f6b7a;
  --surface: #ffffff;
  --surface-raised: #f7f8fa;
  --surface-muted: #eef1f4;
  --border: #d5dbe3;
  --border-strong: #98a3b3;
  --chart-1: #2563eb;
  --chart-2: #15803d;
  --chart-3: #a16207;
  --chart-4: #7c3aed;
  --shadow: 0 10px 30px rgba(23, 32, 51, 0.08);
  --radius: 10px;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root {
    --color-blue-600: #7db2ff;
    --color-blue-100: #172b4d;
    --color-green-600: #69d38c;
    --color-green-100: #153522;
    --color-red-600: #ff8f8f;
    --color-red-100: #451f25;
    --color-amber-600: #f2c35b;
    --color-amber-100: #443617;
    --text: #edf1f7;
    --muted: #aeb8c7;
    --surface: #111722;
    --surface-raised: #18202d;
    --surface-muted: #202a38;
    --border: #344154;
    --border-strong: #65748a;
    --chart-1: #7db2ff;
    --chart-2: #69d38c;
    --chart-3: #f2c35b;
    --chart-4: #b89cff;
    --shadow: 0 12px 34px rgba(0, 0, 0, 0.3);
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-raised); color: var(--text); }
button, input { font: inherit; }
a { color: var(--color-blue-600); }
.shell { max-width: 1540px; margin: 0 auto; padding: 28px; }
.masthead { display: flex; gap: 24px; justify-content: space-between;
  align-items: flex-start; margin-bottom: 22px; }
.eyebrow { color: var(--color-blue-600); font-size: .76rem; font-weight: 750;
  letter-spacing: .1em; margin: 0 0 7px; text-transform: uppercase; }
h1, h2, h3, h4 { line-height: 1.18; }
h1 { font-size: clamp(1.9rem, 4vw, 3.1rem); letter-spacing: -.035em;
  margin: 0 0 8px; }
h2 { font-size: 1.5rem; letter-spacing: -.02em; margin: 0; }
h3 { font-size: 1.08rem; margin: 0; }
h4 { font-size: .94rem; margin: 22px 0 8px; }
p { line-height: 1.55; }
.lede, .section-intro { color: var(--muted); margin: 0; max-width: 76ch; }
.section-intro { margin-top: 7px; }
.card { background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); box-shadow: var(--shadow); }
.section { margin-top: 22px; padding: 22px; }
.section-head { display: flex; gap: 16px; justify-content: space-between;
  align-items: flex-start; margin-bottom: 18px; }
.metrics { display: flex; flex-wrap: wrap; gap: 8px; }
.metric, .badge { border: 1px solid var(--border); border-radius: 999px;
  background: var(--surface-raised); color: var(--muted); font-size: .76rem;
  font-weight: 700; padding: 5px 9px; white-space: nowrap; }
.badge.pass, .badge.green { background: var(--color-green-100);
  border-color: var(--color-green-600); color: var(--color-green-600); }
.badge.fail, .badge.red { background: var(--color-red-100);
  border-color: var(--color-red-600); color: var(--color-red-600); }
.badge.changed { background: var(--color-amber-100);
  border-color: var(--color-amber-600); color: var(--color-amber-600); }
.explorer { display: grid; grid-template-columns: minmax(0, 1.65fr) minmax(310px, .75fr);
  gap: 16px; align-items: start; }
.canvas, .mini-canvas { background: var(--surface-raised); border: 1px solid var(--border);
  border-radius: 8px; overflow: auto; }
.canvas { min-height: 430px; max-height: 74vh; }
.canvas svg, .mini-canvas svg { display: block; }
.node-panel { padding: 17px; max-height: 74vh; overflow: auto; position: sticky; top: 14px; }
.node-panel-empty { color: var(--muted); padding: 28px 8px; text-align: center; }
.node-detail[hidden], .replay-stage[hidden] { display: none; }
.detail-grid { display: grid; grid-template-columns: auto 1fr; gap: 7px 12px;
  margin: 15px 0; }
.detail-grid dt { color: var(--muted); }
.detail-grid dd { margin: 0; min-width: 0; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
code { overflow-wrap: anywhere; }
pre { background: var(--surface-raised); border: 1px solid var(--border);
  border-radius: 7px; font-size: .76rem; line-height: 1.48; margin: 0;
  max-width: 100%; overflow: auto; padding: 11px; white-space: pre-wrap;
  word-break: break-word; }
.table-wrap { border: 1px solid var(--border); border-radius: 8px; overflow: auto; }
table { border-collapse: collapse; font-size: .84rem; width: 100%; }
th, td { border-bottom: 1px solid var(--border); padding: 9px 11px;
  text-align: left; vertical-align: top; }
th { background: var(--surface-raised); color: var(--muted); font-size: .73rem;
  letter-spacing: .025em; position: sticky; top: 0; text-transform: uppercase; }
tr:last-child td { border-bottom: 0; }
.state-cell { width: 1%; white-space: nowrap; }
.muted { color: var(--muted); }
.empty { background: var(--surface-raised); border: 1px dashed var(--border-strong);
  border-radius: 8px; color: var(--muted); padding: 16px; }
.calibration-list, .replay-list { display: grid; gap: 14px; }
.calibration-card, .replay { border: 1px solid var(--border); border-radius: 9px;
  padding: 17px; }
.calibration-head, .replay-head { display: flex; gap: 12px; justify-content: space-between;
  align-items: flex-start; }
.calibration-grid { display: grid; gap: 14px; grid-template-columns: minmax(0, 1.2fr) minmax(280px, .8fr);
  margin-top: 14px; }
.chart { background: var(--surface-raised); border: 1px solid var(--border);
  border-radius: 8px; padding: 10px; }
.chart svg { display: block; height: auto; max-width: 100%; }
.chart-bar { fill: var(--chart-1); }
.chart-bar-before { fill: var(--chart-3); }
.chart-bar-after { fill: var(--chart-1); }
.chart-axis { stroke: var(--border-strong); stroke-width: 1; }
.chart-label { fill: var(--muted); font-family: ui-sans-serif, system-ui, sans-serif;
  font-size: 11px; }
.replay-controls { display: inline-flex; border: 1px solid var(--border);
  border-radius: 8px; margin: 14px 0 10px; overflow: hidden; }
.replay-controls button { background: var(--surface); border: 0; border-right: 1px solid var(--border);
  color: var(--muted); cursor: pointer; font-size: .8rem; font-weight: 700;
  padding: 8px 13px; }
.replay-controls button:last-child { border-right: 0; }
.replay-controls button[aria-pressed="true"] { background: var(--color-blue-100);
  color: var(--color-blue-600); }
.replay-summary { color: var(--muted); margin: 0 0 10px; }
.diff-grid { display: grid; gap: 12px; grid-template-columns: repeat(2, minmax(0, 1fr));
  margin-top: 12px; }
.diff-box { background: var(--surface-raised); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px; }
.diff-box h4 { margin: 0 0 7px; }
.plain-list { margin: 0; padding-left: 20px; }
.plain-list li { margin: 4px 0; }
details.metadata { min-width: 260px; }
details.metadata summary { cursor: pointer; font-weight: 700; }
details.metadata[open] { background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px; }
.metadata .detail-grid { font-size: .8rem; margin-bottom: 0; }
.graph-edge { fill: none; stroke: var(--border-strong); stroke-width: 1.4; }
.population-group { fill: var(--surface-muted); fill-opacity: .72; stroke: var(--border);
  stroke-dasharray: 5 4; }
.population-label { fill: var(--muted); font-size: 11px; font-weight: 700;
  letter-spacing: .04em; }
.graph-node { cursor: pointer; outline: none; }
.node-box { fill: var(--surface); stroke: var(--border-strong); stroke-width: 1.6; }
.graph-node:hover .node-box, .graph-node:focus .node-box, .graph-node.selected .node-box {
  stroke: var(--color-blue-600); stroke-width: 3; }
.graph-node.status-hit .node-box { fill: var(--color-green-100); stroke: var(--color-green-600); }
.graph-node.status-miss .node-box { fill: var(--color-blue-100); stroke: var(--color-blue-600); }
.graph-node.gate-fail .node-box, .graph-node.gate-evidence_absent .node-box,
.graph-node.gate-unreached .node-box { stroke: var(--color-red-600); stroke-width: 3; }
.graph-node.gate-pass .node-box, .graph-node.gate-not_applicable .node-box {
  stroke: var(--color-green-600); stroke-width: 3; }
.graph-node:focus .node-box, .graph-node.selected .node-box { stroke-width: 4.5; }
.node-title { fill: var(--text); font-size: 13px; font-weight: 760; }
.node-line { fill: var(--muted); font-size: 10.5px; }
.node-status { fill: var(--text); font-size: 10.5px; font-weight: 720; }
.mini-node .node-box { stroke-width: 1.2; }
.mini-node.changed .node-box, .mini-node.removed .node-box,
.mini-node.rejected .node-box, .mini-node.refused .node-box {
  fill: var(--color-amber-100); stroke: var(--color-amber-600); stroke-width: 2.3; }
.mini-node.not-executed .node-box { fill: var(--surface-muted); stroke-dasharray: 4 3; }
.footer { color: var(--muted); font-size: .78rem; margin: 18px 0 2px; text-align: center; }
@media (max-width: 900px) {
  .shell { padding: 16px; }
  .masthead, .section-head { display: block; }
  .masthead details { margin-top: 14px; }
  .explorer, .calibration-grid, .diff-grid { grid-template-columns: 1fr; }
  .node-panel { max-height: none; position: static; }
}
@media print {
  .shell { max-width: none; padding: 0; }
  .card { break-inside: avoid; box-shadow: none; }
  .canvas, .node-panel { max-height: none; }
}
"""

_SCRIPT = r"""
(() => {
  const nodes = Array.from(document.querySelectorAll('[data-node-detail]'));
  const details = Array.from(document.querySelectorAll('.node-detail'));
  const empty = document.querySelector('.node-panel-empty');
  function selectNode(token) {
    nodes.forEach((node) => {
      const selected = node.dataset.nodeDetail === token;
      node.classList.toggle('selected', selected);
      node.setAttribute('aria-pressed', selected ? 'true' : 'false');
    });
    details.forEach((detail) => { detail.hidden = detail.id !== token; });
    if (empty) empty.hidden = true;
  }
  nodes.forEach((node) => {
    node.addEventListener('click', () => selectNode(node.dataset.nodeDetail));
    node.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        selectNode(node.dataset.nodeDetail);
      }
    });
  });
  if (nodes.length) selectNode(nodes[0].dataset.nodeDetail);

  document.querySelectorAll('.replay').forEach((replay) => {
    const buttons = Array.from(replay.querySelectorAll(
      '.replay-controls button[data-stage]'
    ));
    const stages = Array.from(replay.querySelectorAll('.replay-stage[data-stage]'));
    buttons.forEach((button) => button.addEventListener('click', () => {
      const selected = button.dataset.stage;
      buttons.forEach((candidate) => candidate.setAttribute(
        'aria-pressed', candidate.dataset.stage === selected ? 'true' : 'false'
      ));
      stages.forEach((stage) => { stage.hidden = stage.dataset.stage !== selected; });
    }));
  });
})();
"""


@dataclass(frozen=True)
class _Position:
    x: int
    y: int


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _value(value: object) -> object:
    return value.value if isinstance(value, Enum) else value


def _plain(value: object) -> object:
    if isinstance(value, Enum):
        return _plain(value.value)
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_plain(child) for child in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)


def _json(value: object) -> str:
    return json.dumps(_plain(value), ensure_ascii=True, indent=2, sort_keys=True)


def _short(value: object, limit: int = 28) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _role(receipt: NodeReceipt) -> str:
    capabilities = receipt.capabilities
    role = (
        capabilities.get("role", "compute")
        if isinstance(capabilities, Mapping)
        else capabilities.role
    )
    return str(_value(role))


def _node_status(receipt: NodeReceipt) -> tuple[str, str]:
    store = "hit" if receipt.hit else "miss"
    role = _role(receipt)
    if role != "gate":
        return f"status-{store}", store
    outcome = str(receipt.receipt.get("outcome", "unrecorded"))
    outcome_class = outcome if outcome in GATE_OUTCOMES else "unknown"
    return f"status-{store} gate-{outcome_class}", f"{store} · gate {outcome}"


def _depths(compiled: CompiledGraph) -> dict[str, int]:
    depths: dict[str, int] = {}
    for node_id in compiled.order:
        predecessors = compiled.predecessors[node_id]
        depths[node_id] = (
            0
            if not predecessors
            else max(depths[parent] for parent in predecessors) + 1
        )
    return depths


def _graph_layout(
    compiled: CompiledGraph,
) -> tuple[dict[str, _Position], dict[str, tuple[int, int, int, int]], int, int]:
    depths = _depths(compiled)
    versions = tuple(
        dict.fromkeys(compiled.versions[node_id] for node_id in compiled.order)
    )
    by_version_depth: dict[tuple[str, int], list[str]] = defaultdict(list)
    for node_id in compiled.order:
        by_version_depth[(compiled.versions[node_id], depths[node_id])].append(node_id)

    version_heights: dict[str, int] = {}
    for version in versions:
        widest = max(
            (
                len(nodes)
                for (name, _), nodes in by_version_depth.items()
                if name == version
            ),
            default=1,
        )
        version_heights[version] = (
            _GROUP_PAD * 2 + widest * _NODE_HEIGHT + max(0, widest - 1) * _Y_GAP + 22
        )

    version_top: dict[str, int] = {}
    cursor = 24
    for version in versions:
        version_top[version] = cursor
        cursor += version_heights[version] + 18

    positions: dict[str, _Position] = {}
    for (version, depth), node_ids in by_version_depth.items():
        for index, node_id in enumerate(node_ids):
            positions[node_id] = _Position(
                x=34 + depth * (_NODE_WIDTH + _X_GAP),
                y=version_top[version]
                + _GROUP_PAD
                + 18
                + index * (_NODE_HEIGHT + _Y_GAP),
            )

    groups: dict[str, tuple[int, int, int, int]] = {}
    for version in versions:
        members = [
            positions[node_id]
            for node_id in compiled.order
            if compiled.versions[node_id] == version
        ]
        min_x = min(point.x for point in members) - _GROUP_PAD
        max_x = max(point.x for point in members) + _NODE_WIDTH + _GROUP_PAD
        groups[version] = (
            min_x,
            version_top[version],
            max_x - min_x,
            version_heights[version],
        )

    max_depth = max(depths.values(), default=0)
    width = 68 + (max_depth + 1) * _NODE_WIDTH + max_depth * _X_GAP
    height = cursor + 6
    return positions, groups, width, height


def _render_graph(
    compiled: CompiledGraph, manifest: RunManifest
) -> tuple[str, list[str]]:
    positions, groups, width, height = _graph_layout(compiled)
    tokens = {
        node_id: f"node-detail-{index}" for index, node_id in enumerate(compiled.order)
    }
    pieces = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'role="group" aria-label="Topologically layered graph run">'
    ]
    for version, (x, y, group_width, group_height) in groups.items():
        pieces.append(
            f'<rect class="population-group" x="{x}" y="{y}" width="{group_width}" '
            f'height="{group_height}" rx="11"><title>Population version: '
            f"{_escape(version)}</title></rect>"
        )
        pieces.append(
            f'<text class="population-label" x="{x + 12}" y="{y + 16}">'
            f"Population: {_escape(version)}</text>"
        )

    for node_id in compiled.order:
        target = positions[node_id]
        for predecessor in compiled.predecessors[node_id]:
            source = positions[predecessor]
            start_x = source.x + _NODE_WIDTH
            start_y = source.y + _NODE_HEIGHT // 2
            end_x = target.x
            end_y = target.y + _NODE_HEIGHT // 2
            middle = (start_x + end_x) // 2
            pieces.append(
                f'<path class="graph-edge" d="M {start_x} {start_y} C {middle} '
                f'{start_y}, {middle} {end_y}, {end_x} {end_y}" />'
            )

    details: list[str] = []
    for node_id in compiled.order:
        node = compiled.graph.node(node_id)
        receipt = manifest.nodes[node_id]
        point = positions[node_id]
        status_class, status_text = _node_status(receipt)
        role = _role(receipt)
        title = (
            f"{node_id}; {node.kernel}; {role}; {node.structural.value}; {status_text}"
        )
        pieces.extend(
            [
                f'<g class="graph-node {status_class}" data-node-detail="{tokens[node_id]}" '
                f'tabindex="0" role="button" aria-label="{_escape(title)}" '
                'aria-pressed="false">',
                f"<title>{_escape(title)}</title>",
                f'<rect class="node-box" x="{point.x}" y="{point.y}" '
                f'width="{_NODE_WIDTH}" height="{_NODE_HEIGHT}" rx="8" />',
                f'<text class="node-title" x="{point.x + 12}" y="{point.y + 20}">'
                f"{_escape(_short(node_id, 28))}</text>",
                f'<text class="node-line" x="{point.x + 12}" y="{point.y + 39}">'
                f"{_escape(_short(node.kernel, 30))}</text>",
                f'<text class="node-line" x="{point.x + 12}" y="{point.y + 56}">'
                f"{_escape(role)} · {_escape(node.structural.value)}</text>",
                f'<text class="node-status" x="{point.x + 12}" y="{point.y + 74}">'
                f"{_escape(status_text)}</text>",
                f'<text class="node-line" x="{point.x + 12}" y="{point.y + 89}">'
                f"key {_escape(receipt.key[:16])}…</text>",
                "</g>",
            ]
        )
        details.append(
            _render_node_detail(compiled, manifest, node_id, tokens[node_id])
        )
    pieces.append("</svg>")
    return "".join(pieces), details


def _capabilities(receipt: NodeReceipt) -> dict[str, object]:
    capabilities = receipt.capabilities
    if isinstance(capabilities, Mapping):
        return {str(key): _plain(value) for key, value in capabilities.items()}
    tolerance = capabilities.tolerance
    return {
        "determinism": _value(capabilities.determinism),
        "numeric": _value(capabilities.numeric),
        "seed_source": _value(capabilities.seed_source),
        "structural": _value(capabilities.structural),
        "role": _value(capabilities.role),
        "consumes_se": capabilities.consumes_se,
        "dependencies": capabilities.dependencies,
        "tolerance": (
            None
            if tolerance is None
            else {
                "rtol": tolerance.rtol,
                "atol": tolerance.atol,
                "ulps": tolerance.ulps,
            }
        ),
    }


def _receipt_payload(receipt: NodeReceipt) -> dict[str, object]:
    payload = {
        "node_key": receipt.key,
        "store_hit": receipt.hit,
        "seed": receipt.seed,
        "kernel_ref": receipt.kernel_ref,
        "kernel_impl_hash": receipt.kernel_impl_hash,
        "capabilities": _capabilities(receipt),
        "receipt": receipt.receipt,
        "artifacts": [
            {"entity": entity, "column": column, "key": key}
            for (entity, column), key in sorted(receipt.artifacts.items())
        ],
        "frame_key": receipt.frame_key,
        "weight_key": receipt.weight_key,
        "opaque_artifacts": receipt.opaque_artifacts,
        "wall_time": receipt.wall_time,
    }
    if receipt.legacy_capabilities:
        payload["legacy_capabilities"] = True
    return payload


def _render_node_detail(
    compiled: CompiledGraph,
    manifest: RunManifest,
    node_id: str,
    token: str,
) -> str:
    node = compiled.graph.node(node_id)
    receipt = manifest.nodes[node_id]
    predecessor_rows = []
    for predecessor in compiled.predecessors[node_id]:
        predecessor_rows.append(
            "<tr><td>"
            + _escape(predecessor)
            + "</td><td><code>"
            + _escape(manifest.nodes[predecessor].key)
            + "</code></td></tr>"
        )
    predecessors = (
        '<div class="table-wrap"><table><thead><tr><th>Node</th><th>Key</th></tr>'
        "</thead><tbody>" + "".join(predecessor_rows) + "</tbody></table></div>"
        if predecessor_rows
        else '<p class="empty">This node has no graph predecessors.</p>'
    )

    owned_rows = [
        "<tr><td>"
        + _escape(owned.entity)
        + "</td><td>"
        + _escape(owned.column)
        + "</td><td><code>"
        + _escape(owned.dtype)
        + "</code></td><td><code>"
        + _escape(owned.rows)
        + "</code></td><td>"
        + _escape(owned.ownership.value)
        + "</td></tr>"
        for owned in node.outputs
    ]
    owned = (
        '<div class="table-wrap"><table><thead><tr><th>Entity</th><th>Column</th>'
        "<th>Dtype</th><th>Row mask</th><th>Ownership</th></tr></thead><tbody>"
        + "".join(owned_rows)
        + "</tbody></table></div>"
        if owned_rows
        else '<p class="empty">No ordinary cells; this node carries a structural frame.</p>'
        if node.structural is not StructuralDelta.NONE
        else '<p class="empty">This node declares no owned cells.</p>'
    )
    status = "Store hit" if receipt.hit else "Store miss"
    return (
        f'<article class="node-detail" id="{token}" hidden>'
        f"<h3>{_escape(node_id)}</h3>"
        '<div class="metrics">'
        f'<span class="badge">{_escape(_role(receipt))}</span>'
        f'<span class="badge">{_escape(node.structural.value)}</span>'
        f'<span class="badge">{status}</span>'
        "</div>"
        '<dl class="detail-grid">'
        f"<dt>Kernel</dt><dd><code>{_escape(receipt.kernel_ref)}</code></dd>"
        f"<dt>Population</dt><dd><code>{_escape(compiled.versions[node_id])}</code></dd>"
        f"<dt>Seed</dt><dd><code>{receipt.seed}</code></dd>"
        f"<dt>Implementation</dt><dd><code>{_escape(receipt.kernel_impl_hash)}</code></dd>"
        "</dl>"
        "<h4>Node key</h4>"
        f"<pre>{_escape(receipt.key)}</pre>"
        '<p class="muted">The key binds the normative declaration, resolved input '
        "artifacts, population or base frame, kernel implementation, and source "
        "content. Descriptive fields and run provenance do not enter it.</p>"
        "<h4>Predecessor keys</h4>"
        + predecessors
        + "<h4>Owned cells</h4>"
        + owned
        + "<h4>Kernel capabilities</h4>"
        + f"<pre>{_escape(_json(_capabilities(receipt)))}</pre>"
        + "<h4>Receipt</h4>"
        + f"<pre>{_escape(_json(_receipt_payload(receipt)))}</pre>"
        + "<h4>One-screen description</h4>"
        + f"<pre>{_escape(describe(compiled, node_id, manifest))}</pre>"
        + "</article>"
    )


def _run_metadata(manifest: RunManifest) -> str:
    hits = sum(receipt.hit for receipt in manifest.nodes.values())
    try:
        tier = manifest.tier or "Not applicable"
        manifest_key = manifest.key
    except ValueError:
        tier = "Invalid release evidence"
        manifest_key = "Invalid manifest"
    decisions = ", ".join(decision.kind for decision in manifest.decisions) or "None"
    return (
        '<details class="metadata"><summary>Run metadata</summary>'
        '<dl class="detail-grid">'
        f"<dt>Country</dt><dd>{_escape(manifest.country)}</dd>"
        f"<dt>Manifest key</dt><dd><code>{_escape(manifest_key)}</code></dd>"
        f"<dt>Tier</dt><dd>{_escape(tier)}</dd>"
        f"<dt>Nodes</dt><dd>{len(manifest.nodes)}</dd>"
        f"<dt>Store hits</dt><dd>{hits}</dd>"
        f"<dt>Store misses</dt><dd>{len(manifest.nodes) - hits}</dd>"
        f"<dt>Decisions</dt><dd>{_escape(decisions)}</dd>"
        "</dl></details>"
    )


_CHARTER_ROW = re.compile(r"^\|\s*([A-Z]\d+)\s*\|\s*(.*?)\s*\|")


def _statement(cell: str) -> str:
    plain = re.sub(r"[*`]", "", cell).strip()
    title = re.match(r"^\*\*(.+?\.)\*\*\s*(.+?\.)(?:\s|$)", cell.strip())
    if title:
        return re.sub(r"[`*]", "", f"{title.group(1)} {title.group(2)}")
    match = re.match(r"(.+?\.)(?:\s|$)", plain)
    return match.group(1) if match else plain


def _charter_properties(charter: str | None) -> list[tuple[str, str]]:
    if charter is None:
        return []
    found: list[tuple[str, str]] = []
    for line in charter.splitlines():
        match = _CHARTER_ROW.match(line.strip())
        if match and all(identifier != match.group(1) for identifier, _ in found):
            found.append((match.group(1), _statement(match.group(2))))
    return found


def _burndown_properties(
    charter: str | None, burndown: Mapping[str, object]
) -> list[dict[str, object]]:
    raw = burndown.get("properties", ())
    supplied: dict[str, Mapping[str, object]] = {}
    if isinstance(raw, Sequence) and not isinstance(raw, str | bytes | bytearray):
        for item in raw:
            if isinstance(item, Mapping) and isinstance(item.get("id"), str):
                supplied[str(item["id"])] = item

    declared = _charter_properties(charter)
    if not declared:
        declared = [
            (identifier, str(item.get("statement") or item.get("test") or ""))
            for identifier, item in supplied.items()
        ]
    rows = []
    for identifier, statement in declared:
        item = supplied.get(identifier, {})
        raw_state = str(item.get("state", ""))
        if raw_state not in {"green", "red"}:
            raw_state = "green" if identifier.startswith("V") else "red"
        flip_pr = next(
            (
                item[field]
                for field in ("pr", "pull_request", "flipped_by")
                if item.get(field) not in (None, "")
            ),
            "Not recorded",
        )
        rows.append(
            {
                "id": identifier,
                "statement": statement,
                "state": raw_state,
                "flip_pr": flip_pr,
                "reason": "; ".join(str(reason) for reason in item.get("reasons", ()))
                if isinstance(item.get("reasons", ()), Sequence)
                else "",
            }
        )
    return rows


def _payoff_rows(
    compiled: CompiledGraph,
    manifest: RunManifest,
    burndown: Mapping[str, object],
) -> list[tuple[str, str, str]]:
    raw = burndown.get("payoffs", {})
    payoffs = raw if isinstance(raw, Mapping) else {}
    rows: list[tuple[str, str, str]] = []
    misses = payoffs.get("unrelated_edit_misses")
    nodes = payoffs.get("unrelated_edit_nodes")
    if isinstance(misses, int) and isinstance(nodes, int) and nodes:
        rows.append(
            (
                "Refit share on an unrelated edit",
                f"{misses} of {nodes} nodes ({misses / nodes:.1%})",
                "A descriptive-field edit rerun against the populated store.",
            )
        )
    target = str(payoffs.get("describe_target", "target_b"))
    if target in compiled.order:
        line_count = len(describe(compiled, target, manifest).splitlines())
        rows.append(
            (
                "Legibility",
                f"{line_count} describe() lines",
                f"The complete view for {target}.",
            )
        )
    return rows


def _render_burndown(
    compiled: CompiledGraph,
    manifest: RunManifest,
    charter: str | None,
    burndown: Mapping[str, object] | None,
) -> str:
    if burndown is None:
        return ""
    properties = _burndown_properties(charter, burndown)
    property_rows = "".join(
        "<tr><td><code>"
        + _escape(row["id"])
        + "</code></td><td>"
        + _escape(row["statement"])
        + (
            f'<br><span class="muted">{_escape(row["reason"])}</span>'
            if row["reason"]
            else ""
        )
        + f'</td><td class="muted">{_escape(row["flip_pr"])}</td>'
        + f'<td class="state-cell"><span class="badge {row["state"]}">'
        + _escape(row["state"])
        + "</span></td></tr>"
        for row in properties
    )
    payoffs = _payoff_rows(compiled, manifest, burndown)
    payoff_html = ""
    if payoffs:
        payoff_rows = "".join(
            f"<tr><td>{_escape(name)}</td><td><strong>{_escape(value)}</strong></td>"
            f'<td class="muted">{_escape(note)}</td></tr>'
            for name, value, note in payoffs
        )
        payoff_html = (
            "<h3>Measured payoffs</h3>"
            '<div class="table-wrap"><table><thead><tr><th>Measure</th><th>Today</th>'
            f"<th>Method</th></tr></thead><tbody>{payoff_rows}</tbody></table></div>"
        )
    green = sum(row["state"] == "green" for row in properties)
    red = sum(row["state"] == "red" for row in properties)
    return (
        '<section class="section card" id="acceptance-burndown">'
        '<div class="section-head"><div><h2>Acceptance burndown</h2>'
        '<p class="section-intro">Every charter property is paired with the '
        "current executable-suite state.</p></div>"
        f'<div class="metrics"><span class="badge green">{green} green</span>'
        f'<span class="badge red">{red} red</span></div></div>'
        '<div class="table-wrap"><table><thead><tr><th>Id</th><th>Property</th>'
        f"<th>Flip PR</th><th>State</th></tr></thead><tbody>{property_rows}</tbody>"
        "</table></div>" + payoff_html + "</section>"
    )


def _declared_targets(node: Node, receipt: NodeReceipt) -> list[dict[str, object]]:
    raw_targets = node.params.get("targets")
    if not isinstance(raw_targets, Sequence) or isinstance(raw_targets, str | bytes):
        raw_targets = receipt.receipt.get("declared_targets")
    rows: list[dict[str, object]] = []
    if isinstance(raw_targets, Sequence) and not isinstance(raw_targets, str | bytes):
        for index, target in enumerate(raw_targets):
            if isinstance(target, Sequence) and not isinstance(target, str | bytes):
                values = list(target)
                if len(values) == 5:
                    rows.append(
                        {
                            "name": values[0],
                            "measure": values[1],
                            "filter": values[2],
                            "value": values[3],
                            "se": values[4],
                        }
                    )
            elif isinstance(target, Mapping):
                rows.append(
                    {
                        "name": target.get("name", f"target_{index + 1}"),
                        "measure": target.get("measure"),
                        "filter": target.get("filter"),
                        "value": target.get("value", target.get("target")),
                        "se": target.get("se"),
                    }
                )
    if not rows and "target_total" in node.params:
        rows.append(
            {
                "name": node.params.get("target_column", "target"),
                "measure": node.params.get("target_column"),
                "filter": node.params.get("target_filter"),
                "value": node.params.get("target_total"),
                "se": node.params.get("target_se"),
            }
        )
    return rows


def _diagnostic_targets(receipt: NodeReceipt) -> dict[str, Mapping[str, object]]:
    diagnostics = receipt.receipt.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return {}
    targets = diagnostics.get("targets")
    if not isinstance(targets, Sequence) or isinstance(targets, str | bytes):
        return {}
    found: dict[str, Mapping[str, object]] = {}
    for target in targets:
        if isinstance(target, Mapping):
            name = target.get("target_name", target.get("name"))
            if isinstance(name, str):
                found[name] = target
    return found


def _target_results(
    declared: Sequence[Mapping[str, object]], receipt: NodeReceipt
) -> list[dict[str, object]]:
    diagnostics = _diagnostic_targets(receipt)
    rows: list[dict[str, object]] = []
    for index, target in enumerate(declared):
        name = str(target.get("name", f"target_{index + 1}"))
        diagnostic = diagnostics.get(name, {})
        achieved = diagnostic.get("final_estimate")
        if achieved is None and len(declared) == 1:
            achieved = receipt.receipt.get("achieved")
        expected = target.get("value")
        residual: object = None
        if isinstance(achieved, int | float) and isinstance(expected, int | float):
            residual = float(achieved) - float(expected)
        rows.append(
            {**target, "name": name, "achieved": achieved, "residual": residual}
        )
    return rows


def _number(value: object) -> str:
    if value is None:
        return "Not recorded"
    if isinstance(value, float | int) and not isinstance(value, bool):
        return f"{float(value):,.6g}"
    return str(value)


def _frame_ratios(frame, anchor, entity: str) -> list[float]:
    try:
        values = frame.weights_for(entity).values
        design_weights = anchor.weights_for(entity).values
        id_column = frame.schema.entity_id_column(entity)
        entity_ids = list(frame.table(entity)[id_column])
        design_ids = list(anchor.table(entity)[id_column])
    except (KeyError, ValueError):
        return []
    design_by_id = dict(zip(design_ids, design_weights, strict=True))
    ratios = []
    for entity_id, current in zip(entity_ids, values, strict=True):
        design = float(design_by_id.get(entity_id, 0.0))
        current_value = float(current)
        if design > 0:
            ratios.append(current_value / design)
        elif current_value == 0:
            ratios.append(0.0)
        else:
            ratios.append(math.inf)
    return ratios


def _population_ratios(
    compiled: CompiledGraph,
    manifest: RunManifest,
    node: Node,
) -> tuple[list[float], list[float]]:
    if node.weights is None:
        return [], []
    before = None if node.base is None else manifest.populations.get(node.base)
    after = manifest.populations.get(node.id)
    if before is None or after is None:
        return [], []
    entity = node.weights.entity
    anchor_node = node.base
    anchor = None
    while anchor_node is not None:
        candidate = manifest.populations.get(anchor_node)
        if candidate is not None and entity in candidate.weighted_entities:
            weights = candidate.weights_for(entity)
            if str(_value(weights.kind)) == "design":
                anchor = candidate
                break
        declaration = compiled.graph.node(anchor_node)
        anchor_node = declaration.base
    if anchor is None:
        return [], []
    return _frame_ratios(before, anchor, entity), _frame_ratios(after, anchor, entity)


def _numeric_sequence(value: object) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return [float(item) for item in value if isinstance(item, int | float)]


def _receipt_ratios(receipt: NodeReceipt) -> tuple[list[float], list[float]]:
    for name in ("weight_ratios", "ratio_distribution"):
        raw = receipt.receipt.get(name)
        if isinstance(raw, Mapping):
            before = _numeric_sequence(raw.get("before"))
            after = _numeric_sequence(raw.get("after", raw.get("values")))
            if before or after:
                return before, after
        after = _numeric_sequence(raw)
        if after:
            return [], after
    return [], []


def _bin_counts(
    values: Sequence[float], low: float, high: float, bin_count: int
) -> list[int]:
    counts = [0] * bin_count
    if math.isclose(low, high):
        counts[0] = len(values)
        return counts
    width = (high - low) / bin_count
    for value in values:
        index = min(bin_count - 1, int((value - low) / width))
        counts[index] += 1
    return counts


def _histogram_svg(before: Sequence[float], after: Sequence[float]) -> str:
    finite_before = [float(value) for value in before if math.isfinite(float(value))]
    finite_after = [float(value) for value in after if math.isfinite(float(value))]
    finite = [*finite_before, *finite_after]
    nonfinite = len(before) + len(after) - len(finite)
    if not finite:
        return '<p class="empty">Weight-ratio samples are not present in this manifest.</p>'
    low, high = min(finite), max(finite)
    bin_count = min(12, max(4, int(math.sqrt(len(finite)))))
    if math.isclose(low, high):
        bin_count = 1
    before_counts = _bin_counts(finite_before, low, high, bin_count)
    after_counts = _bin_counts(finite_after, low, high, bin_count)
    chart_width, chart_height = 560, 190
    left, top, bottom = 38, 26, 30
    inner_width = chart_width - left - 12
    inner_height = chart_height - top - bottom
    gap = 4
    group_width = (inner_width - gap * max(0, bin_count - 1)) / bin_count
    bar_gap = 1 if finite_before and finite_after else 0
    series_count = 2 if finite_before and finite_after else 1
    bar_width = (group_width - bar_gap * (series_count - 1)) / series_count
    maximum = max([*before_counts, *after_counts]) or 1
    bars = []
    series = []
    if finite_before:
        series.append(("before", before_counts))
    if finite_after:
        series.append(("after", after_counts))
    for series_index, (label, counts) in enumerate(series):
        for index, count in enumerate(counts):
            height = inner_height * count / maximum
            x = (
                left
                + index * (group_width + gap)
                + series_index * (bar_width + bar_gap)
            )
            y = top + inner_height - height
            bars.append(
                f'<rect class="chart-bar chart-bar-{label}" x="{x:.2f}" '
                f'y="{y:.2f}" width="{bar_width:.2f}" height="{height:.2f}" '
                f'rx="2"><title>{label}: {count} records</title></rect>'
            )
    note = f"; {nonfinite} non-finite" if nonfinite else ""
    legend = []
    legend_x = left
    for label, css_class in (("Before", "before"), ("After", "after")):
        values = finite_before if css_class == "before" else finite_after
        if not values:
            continue
        legend.append(
            f'<rect class="chart-bar-{css_class}" x="{legend_x}" y="6" '
            'width="10" height="10" rx="2" />'
            f'<text class="chart-label" x="{legend_x + 15}" y="15">{label} '
            f"(n={len(values)})</text>"
        )
        legend_x += 100
    return (
        f'<svg viewBox="0 0 {chart_width} {chart_height}" role="img" '
        f'aria-label="Before and after histogram of {len(finite)} finite weight '
        f'ratios{note}">'
        + "".join(legend)
        + "".join(bars)
        + f'<path class="chart-axis" d="M {left} {top + inner_height} H '
        f'{chart_width - 12}" />'
        f'<text class="chart-label" x="{left}" y="{chart_height - 8}">'
        f"{_escape(_number(low))}</text>"
        f'<text class="chart-label" text-anchor="end" x="{chart_width - 12}" '
        f'y="{chart_height - 8}">{_escape(_number(high))}</text>'
        "</svg>"
    )


def _mass_payload(
    manifest: RunManifest, node: Node, receipt: NodeReceipt
) -> dict[str, object] | None:
    raw = receipt.receipt.get("mass")
    if isinstance(raw, Mapping):
        payload = {
            "before": raw.get("before"),
            "after": raw.get("after"),
            "stratum_before": raw.get("stratum_before", {}),
            "stratum_after": raw.get("stratum_after", {}),
            "policy": raw.get("policy", node.mass),
        }
        if isinstance(raw.get("partition"), Mapping):
            payload["partition"] = raw["partition"]
        return payload
    for record in reversed(manifest.mass_ledgers.get(node.id, ())):
        if record.node_id == node.id:
            return mass_record_receipt(record)
    if node.base is not None:
        before = manifest.populations.get(node.base)
        after = manifest.populations.get(node.id)
        if before is not None and after is not None:
            before_mass = before.stratum_mass()
            after_mass = after.stratum_mass()
            return {
                "before": float(before_mass.sum()),
                "after": float(after_mass.sum()),
                "stratum_before": {
                    str(key): float(value) for key, value in before_mass.items()
                },
                "stratum_after": {
                    str(key): float(value) for key, value in after_mass.items()
                },
                "policy": node.mass,
            }
    return None


def _partition_mass_table(mass: Mapping[str, object]) -> str:
    raw_partition = mass.get("partition")
    if not isinstance(raw_partition, Mapping):
        return ""
    raw_before = raw_partition.get("stratum_before", {})
    raw_after = raw_partition.get("stratum_after", {})
    by_before = raw_before if isinstance(raw_before, Mapping) else {}
    by_after = raw_after if isinstance(raw_after, Mapping) else {}
    partitions = sorted(set(by_before) | set(by_after), key=str)
    rows: list[str] = []
    for partition in partitions:
        raw_before_strata = by_before.get(partition, {})
        raw_after_strata = by_after.get(partition, {})
        before_strata = (
            raw_before_strata if isinstance(raw_before_strata, Mapping) else {}
        )
        after_strata = raw_after_strata if isinstance(raw_after_strata, Mapping) else {}
        strata = sorted(set(before_strata) | set(after_strata), key=str)
        for stratum in strata:
            before = before_strata.get(stratum, 0.0)
            after = after_strata.get(stratum, 0.0)
            delta = (
                float(after) - float(before)
                if isinstance(before, int | float) and isinstance(after, int | float)
                else None
            )
            rows.append(
                f"<tr><td>{_escape(partition)}</td><td>{_escape(stratum)}</td>"
                f"<td>{_escape(_number(before))}</td>"
                f"<td>{_escape(_number(after))}</td>"
                f"<td>{_escape(_number(delta))}</td></tr>"
            )
    entity = raw_partition.get("entity", "Not recorded")
    column = raw_partition.get("column", "Not recorded")
    heading = f"<h5>Mass by {_escape(f'{entity}.{column}')} partition</h5>"
    if not rows:
        return heading + '<p class="muted">Per-partition mass was not recorded.</p>'
    return (
        heading + '<div class="table-wrap"><table><thead><tr><th>Partition value</th>'
        "<th>Stratum</th><th>Before</th><th>After</th><th>Change</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _mass_tables(mass: Mapping[str, object] | None) -> str:
    if mass is None:
        return '<p class="empty">Mass ledger values are not present in the portable receipt.</p>'
    before = mass.get("before")
    after = mass.get("after")
    delta = (
        float(after) - float(before)
        if isinstance(before, int | float) and isinstance(after, int | float)
        else None
    )
    totals = (
        '<div class="table-wrap"><table><thead><tr><th>Policy</th><th>Before</th>'
        "<th>After</th><th>Change</th></tr></thead><tbody><tr>"
        f"<td>{_escape(mass.get('policy', 'Not recorded'))}</td>"
        f"<td>{_escape(_number(before))}</td><td>{_escape(_number(after))}</td>"
        f"<td>{_escape(_number(delta))}</td></tr></tbody></table></div>"
    )
    raw_before = mass.get("stratum_before", {})
    raw_after = mass.get("stratum_after", {})
    by_before = raw_before if isinstance(raw_before, Mapping) else {}
    by_after = raw_after if isinstance(raw_after, Mapping) else {}
    strata = sorted(set(by_before) | set(by_after), key=str)
    partition_table = _partition_mass_table(mass)
    if not strata:
        return (
            totals
            + '<p class="muted">Per-stratum mass was not recorded.</p>'
            + partition_table
        )
    rows = "".join(
        f"<tr><td>{_escape(stratum)}</td><td>{_escape(_number(by_before.get(stratum)))}</td>"
        f"<td>{_escape(_number(by_after.get(stratum)))}</td></tr>"
        for stratum in strata
    )
    return (
        totals
        + '<div class="table-wrap"><table><thead><tr><th>Stratum</th><th>Before</th>'
        f"<th>After</th></tr></thead><tbody>{rows}</tbody></table></div>"
        + partition_table
    )


def _render_calibration(compiled: CompiledGraph, manifest: RunManifest) -> str:
    transitions = [
        compiled.graph.node(node_id)
        for node_id in compiled.order
        if compiled.graph.node(node_id).weights is not None
    ]
    if not transitions:
        return ""
    cards = []
    for node in transitions:
        receipt = manifest.nodes[node.id]
        targets = _target_results(_declared_targets(node, receipt), receipt)
        if targets:
            target_rows = "".join(
                "<tr>"
                f"<td>{_escape(target.get('name'))}</td>"
                f"<td>{_escape(_number(target.get('measure')))}</td>"
                f"<td>{_escape(_number(target.get('filter')))}</td>"
                f"<td>{_escape(_number(target.get('value')))}</td>"
                f"<td>{_escape(_number(target.get('se')))}</td>"
                f"<td>{_escape(_number(target.get('achieved')))}</td>"
                f"<td>{_escape(_number(target.get('residual')))}</td>"
                "</tr>"
                for target in targets
            )
            target_table = (
                '<div class="table-wrap"><table><thead><tr><th>Name</th><th>Measure</th>'
                "<th>Filter</th><th>Target</th><th>SE</th><th>Achieved</th>"
                f"<th>Residual</th></tr></thead><tbody>{target_rows}</tbody></table></div>"
            )
        else:
            target_table = (
                '<p class="empty">This weight transition declares no target table.</p>'
            )
        before_ratios, after_ratios = _receipt_ratios(receipt)
        if not before_ratios and not after_ratios:
            before_ratios, after_ratios = _population_ratios(compiled, manifest, node)
        mass = _mass_payload(manifest, node, receipt)
        transition = node.weights
        assert transition is not None
        cards.append(
            '<article class="calibration-card">'
            '<div class="calibration-head"><div>'
            f"<h3>{_escape(node.id)}</h3>"
            f'<p class="section-intro"><code>{_escape(node.kernel)}</code> · '
            f"{_escape(transition.entity)} → {_escape(transition.to_kind)}</p></div>"
            f'<span class="badge">{_escape(transition.mass)} mass</span></div>'
            "<h4>Declared targets and results</h4>"
            + target_table
            + '<div class="calibration-grid"><div><h4>Weight ratios against design</h4>'
            '<div class="chart">'
            + _histogram_svg(before_ratios, after_ratios)
            + "</div></div><div><h4>Mass ledger</h4>"
            + _mass_tables(mass)
            + "</div></div>"
            '<p class="muted">Absent achieved values, residuals, standard errors, '
            "ratio samples, or strata are named as not recorded; they are never "
            "inferred from unrelated diagnostics.</p></article>"
        )
    return (
        '<section class="section card" id="calibration-view">'
        '<div class="section-head"><div><h2>Calibration view</h2>'
        '<p class="section-intro">Declared target evidence, achieved fit, design-anchor '
        "weight spread, and mass movement for every typed weight transition.</p></div>"
        f'<span class="metric">{len(transitions)} transitions</span></div>'
        f'<div class="calibration-list">{"".join(cards)}</div></section>'
    )


def _mini_graph(snapshot: Mapping[str, object] | None) -> str:
    if snapshot is None:
        return '<p class="empty">No graph snapshot was recorded for this step.</p>'
    raw_nodes = snapshot.get("nodes", ())
    nodes = (
        [node for node in raw_nodes if isinstance(node, Mapping)]
        if isinstance(raw_nodes, Sequence)
        else []
    )
    if not nodes:
        return '<p class="empty">No nodes reached this step.</p>'
    raw_edges = snapshot.get("edges", ())
    edges = list(raw_edges) if isinstance(raw_edges, Sequence) else []
    levels: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for node in nodes:
        depth = node.get("depth", 0)
        levels[int(depth) if isinstance(depth, int) else 0].append(node)
    mini_width, mini_height = 142, 50
    x_gap, y_gap = 34, 14
    positions: dict[str, _Position] = {}
    for depth in sorted(levels):
        for index, node in enumerate(levels[depth]):
            positions[str(node.get("id", ""))] = _Position(
                16 + depth * (mini_width + x_gap), 18 + index * (mini_height + y_gap)
            )
    width = max(point.x for point in positions.values()) + mini_width + 16
    height = max(point.y for point in positions.values()) + mini_height + 18
    pieces = [
        f'<div class="mini-canvas"><svg viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="Replay graph snapshot">'
    ]
    for edge in edges:
        if (
            not isinstance(edge, Sequence)
            or isinstance(edge, str | bytes)
            or len(edge) != 2
        ):
            continue
        source, target = str(edge[0]), str(edge[1])
        if source not in positions or target not in positions:
            continue
        start, end = positions[source], positions[target]
        pieces.append(
            f'<path class="graph-edge" d="M {start.x + mini_width} '
            f'{start.y + mini_height / 2:g} L {end.x} {end.y + mini_height / 2:g}" />'
        )
    for node in nodes:
        node_id = str(node.get("id", ""))
        point = positions[node_id]
        state = re.sub(r"[^a-z-]", "-", str(node.get("state", "stable")).lower())
        key = str(node.get("key", ""))
        pieces.extend(
            [
                f'<g class="mini-node {state}"><title>{_escape(node_id)}; '
                f"{_escape(state)}; {_escape(key or 'no key')}</title>"
                f'<rect class="node-box" x="{point.x}" y="{point.y}" '
                f'width="{mini_width}" height="{mini_height}" rx="6" />'
                f'<text class="node-title" x="{point.x + 8}" y="{point.y + 19}">'
                f"{_escape(_short(node_id, 20))}</text>"
                f'<text class="node-line" x="{point.x + 8}" y="{point.y + 37}">'
                f"{_escape(state)}{(' · ' + _escape(key[:9]) + '…') if key else ''}</text></g>"
            ]
        )
    pieces.append("</svg></div>")
    return "".join(pieces)


def _list_html(values: object, empty: str) -> str:
    if (
        not isinstance(values, Sequence)
        or isinstance(values, str | bytes)
        or not values
    ):
        return f'<p class="muted">{_escape(empty)}</p>'
    return (
        '<ul class="plain-list">'
        + "".join(f"<li><code>{_escape(value)}</code></li>" for value in values)
        + "</ul>"
    )


def _render_replays(replays: Sequence[Mapping[str, object]] | None) -> str:
    if replays is None:
        return ""
    cards = []
    for index, replay in enumerate(replays):
        identifier = str(replay.get("id", f"replay-{index + 1}"))
        title = str(replay.get("title", identifier))
        verdict_value = replay.get("verdict", "fail")
        passed = verdict_value is True or str(verdict_value).lower() == "pass"
        verdict = "pass" if passed else "fail"
        raw_stages = replay.get("stages", ())
        stages = (
            list(raw_stages)
            if isinstance(raw_stages, Sequence)
            and not isinstance(raw_stages, str | bytes)
            else []
        )
        while len(stages) < 3:
            stages.append({"label": ("Before", "Change", "After")[len(stages)]})
        stages = stages[:3]
        controls = "".join(
            f'<button type="button" data-stage="{stage_index}" aria-pressed="'
            f'{"true" if stage_index == 0 else "false"}">'
            f"{_escape(stage.get('label', ('Before', 'Change', 'After')[stage_index]))}"
            "</button>"
            for stage_index, stage in enumerate(stages)
            if isinstance(stage, Mapping)
        )
        stage_html = []
        for stage_index, stage in enumerate(stages):
            stage = stage if isinstance(stage, Mapping) else {}
            stage_html.append(
                f'<div class="replay-stage" data-stage="{stage_index}"'
                + ("" if stage_index == 0 else " hidden")
                + ">"
                + f'<p class="replay-summary">{_escape(stage.get("summary", "No summary recorded."))}</p>'
                + _mini_graph(
                    stage.get("snapshot")
                    if isinstance(stage.get("snapshot"), Mapping)
                    else None
                )
                + "</div>"
            )
        moved = replay.get("moved_keys", ())
        changed = replay.get("changed_nodes", ())
        cells = replay.get("cell_changes", ())
        cards.append(
            f'<article class="replay" data-replay="{_escape(identifier)}">'
            '<div class="replay-head"><div>'
            f"<h3>{_escape(title)}</h3>"
            f'<p class="section-intro">{_escape(replay.get("property", ""))}</p></div>'
            f'<span class="badge {verdict}">{verdict}</span></div>'
            f'<div class="replay-controls" aria-label="{_escape(title)} steps">{controls}</div>'
            + "".join(stage_html)
            + '<div class="diff-grid"><div class="diff-box"><h4>Changed nodes</h4>'
            + _list_html(changed, "No graph node changed.")
            + '</div><div class="diff-box"><h4>Moved node keys</h4>'
            + _list_html(moved, "No node key moved.")
            + '</div><div class="diff-box"><h4>Cell boundary</h4>'
            + _list_html(cells, "No cell artifact changed.")
            + '</div><div class="diff-box"><h4>Observed verdict</h4>'
            f"<p>{_escape(replay.get('observed', 'Not recorded.'))}</p></div></div></article>"
        )
    passes = sum(
        replay.get("verdict") is True or str(replay.get("verdict")).lower() == "pass"
        for replay in replays
    )
    return (
        '<section class="section card" id="incident-replays">'
        '<div class="section-head"><div><h2>Incident replays</h2>'
        '<p class="section-intro">Step through each synthetic incident from the '
        "starting graph, through the change, to the enforced boundary.</p></div>"
        f'<span class="metric">{passes} of {len(replays)} pass</span></div>'
        f'<div class="replay-list">{"".join(cards)}</div></section>'
    )


def explain_html(
    compiled: CompiledGraph,
    manifest: RunManifest,
    *,
    charter: str | None = None,
    burndown: Mapping[str, object] | None = None,
    replays: Sequence[Mapping[str, object]] | None = None,
    title: str | None = None,
) -> str:
    """Render one deterministic, self-contained explanation page.

    ``charter``, ``burndown``, and ``replays`` are optional enrichment data.
    The graph explorer and any calibration evidence in the run always render.
    ``title`` names the page (the browser tab and any gallery it lands in);
    it defaults to the manifest's country label plus "graph run".
    This function performs no file or network access.
    """

    if not isinstance(compiled, CompiledGraph):
        raise TypeError("explain_html compiled must be a CompiledGraph")
    if not isinstance(manifest, RunManifest):
        raise TypeError("explain_html manifest must be a RunManifest")
    missing = sorted(set(compiled.order) - set(manifest.nodes))
    extra = sorted(set(manifest.nodes) - set(compiled.order))
    if missing or extra:
        raise ValueError(
            "compiled graph and manifest node ids differ "
            f"(missing={missing}, extra={extra})"
        )

    graph, node_details = _render_graph(compiled, manifest)
    hits = sum(manifest.nodes[node_id].hit for node_id in compiled.order)
    misses = len(compiled.order) - hits
    roles = Counter(_role(manifest.nodes[node_id]) for node_id in compiled.order)
    metrics = (
        f'<span class="metric">{len(compiled.order)} nodes</span>'
        f'<span class="metric">{hits} hits</span>'
        f'<span class="metric">{misses} misses</span>'
        + "".join(
            f'<span class="metric">{count} {_escape(role)}</span>'
            for role, count in sorted(roles.items())
            if role != "compute"
        )
    )
    burndown_html = _render_burndown(compiled, manifest, charter, burndown)
    calibration_html = _render_calibration(compiled, manifest)
    replay_html = _render_replays(replays)
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_escape(title or f'{manifest.country} graph run')}</title>"
        f'<style>{_STYLE}</style></head><body><main class="shell">'
        '<header class="masthead"><div><p class="eyebrow">Microcosm graph</p>'
        f"<h1>{_escape(manifest.country)} graph run</h1>"
        '<p class="lede">A portable review surface generated from the compiled '
        "declaration and its run receipts. No server or network is required.</p></div>"
        + _run_metadata(manifest)
        + "</header>"
        '<section class="section card" id="graph-explorer">'
        '<div class="section-head"><div><h2>Graph explorer</h2>'
        '<p class="section-intro">Population versions form the dashed groups. '
        "Select a node for its complete key, ownership, kernel, and receipt evidence."
        f'</p></div><div class="metrics">{metrics}</div></div>'
        '<div class="explorer"><div class="canvas">'
        + graph
        + '</div><aside class="node-panel card" aria-live="polite">'
        '<p class="node-panel-empty">Select a graph node to inspect it.</p>'
        + "".join(node_details)
        + "</aside></div></section>"
        + burndown_html
        + calibration_html
        + replay_html
        + '<p class="footer">Generated entirely from graph declarations and run evidence.</p>'
        f"</main><script>{_SCRIPT}</script></body></html>\n"
    )
