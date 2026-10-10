"""Declare entitlement bridges, scenario branches and their held-out bands.

``entitlement_graph.json`` is an optional legacy-JSON resource. Its ``nodes``
use the skeleton node declaration grammar. ``scenario_nodes`` and
``variant_nodes`` are templates, expanded for the entitlement and calibration
tiers of ``scenarios.json`` respectively. A row's ``nodes`` can replace its
tier's template, for example to select an engine-derived alternative bridge.
Node ids and data strings can contain ``{scenario}`` or ``{variant}``.

Extension parameters additionally accept ``{binding, field}`` selections
(field is engine_ref or period), and ``{scenario, path?, encoding?}`` selections
of the current scenario row. The latter's SHA-256 binds only the selected
subtree, so another scenario can be added without re-keying existing branches.
Binding selections inside selected JSON bridge data are resolved explicitly.
Every policy coefficient, override and solver setting remains spec data.

``bands`` declares id, population and baseline_scenario; ``scenario_gap`` and
``variant_gap`` name each tier's gap producer template. ``package`` names the
skeleton receipt node and the bands artifact alias; its optional ``inputs``
adds other typed extension evidence, such as a held-out validation tripwire.
The only skeleton change is these explicit artifact edges on the final receipt.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from microcosm.graph import ArtifactInput, ArtifactOutput, Graph, Node, SourceRef
from microcosm.graph.canonical import canonical_json

from .compose import (
    SKELETON_KERNELS,
    TransportExtension,
    TransportGraphConfig,
    _artifact_outputs,
    _literal_data,
    _mapping,
    _node,
    _param,
    _resource_param,
    _safe_node_id,
    _sequence,
    transport_resource,
)
from .takeup_kernels import BANDS_TYPE, GAP_TYPE

__all__ = ["ENTITLEMENT_RESOURCE", "entitlement_extension", "transport_extensions"]

ENTITLEMENT_RESOURCE = "entitlement_graph"
_KERNELS = SKELETON_KERNELS | frozenset(
    {
        "simulate.counterfactual@1",
        "simulate.solve_zero@1",
        "takeup.gap@1",
        "takeup.bands@1",
        "takeup.compare@1",
        "transport.scenario_override@1",
    }
)
_FIELDS = frozenset(
    {
        "schema_version",
        "nodes",
        "scenario_nodes",
        "variant_nodes",
        "scenario_gap",
        "variant_gap",
        "sources",
        "checkpoints",
        "bands",
        "package",
    }
)


def _document(spec):
    document = transport_resource(spec, ENTITLEMENT_RESOURCE)
    if document.get("schema_version") != 1 or set(document) - _FIELDS:
        raise ValueError(
            "An entitlement graph requires schema_version 1 and known fields."
        )
    return document


def _format(value, substitutions):
    if isinstance(value, str):
        for key, replacement in substitutions.items():
            value = value.replace("{" + key + "}", replacement)
        return value
    if isinstance(value, Mapping):
        return {key: _format(item, substitutions) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_format(item, substitutions) for item in value]
    return value


def _binding(spec, config, selection):
    if set(selection) != {"binding", "field"}:
        raise ValueError("A binding selection names exactly binding and field.")
    name, field = selection["binding"], selection["field"]
    rows = transport_resource(spec, "axiom_rules_bindings")["bindings"]
    matches = [row for row in rows if row["id"] == name]
    if len(matches) != 1 or name not in config.engine_refs:
        raise ValueError(f"Entitlement binding {name!r} needs one prepared reference.")
    if field == "engine_ref":
        return config.engine_refs[name]
    if field == "period":
        return matches[0]["period"]
    raise ValueError("A binding selection field must be engine_ref or period.")


def _bind_data(spec, config, value):
    if isinstance(value, Mapping):
        if "binding" in value and "field" in value:
            return _binding(spec, config, value)
        return {key: _bind_data(spec, config, item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_bind_data(spec, config, item) for item in value]
    return value


def _parameter(spec, config, value, scenario):
    if isinstance(value, list | tuple):
        if not value or any(isinstance(item, Mapping) for item in value):
            return canonical_json(_bind_data(spec, config, value)).decode("utf-8")
        return tuple(_parameter(spec, config, item, scenario) for item in value)
    if not isinstance(value, Mapping):
        return value
    if "binding" in value:
        return _binding(spec, config, value)
    if "scenario" in value:
        if scenario is None or set(value) - {"scenario", "path", "encoding"}:
            raise ValueError(
                "A scenario selection requires a current row and known fields."
            )
        key = value["scenario"]
        selected = scenario if key == "row" else scenario[key]
        for component in value.get("path", ()):
            selected = _mapping(selected, "scenario selection")[component]
        encoding = value.get("encoding", "value")
        if encoding == "sha256":
            return hashlib.sha256(canonical_json(selected)).hexdigest()
        if encoding == "json":
            return canonical_json(_bind_data(spec, config, selected)).decode("utf-8")
        if encoding == "value" and not isinstance(selected, Mapping):
            return _literal_data(selected)
        raise ValueError(
            "Scenario objects require json encoding; encodings are value, json, sha256."
        )
    if value.get("encoding") == "json" and "resource" in value:
        selected = json.loads(_resource_param(spec, value, allow_json_sequences=True))
        return canonical_json(_bind_data(spec, config, selected)).decode("utf-8")
    if not ({"resource", "prepared"} & set(value)):
        return canonical_json(_bind_data(spec, config, value)).decode("utf-8")
    return _param(spec, value, config.engine_refs)


def _rows(spec):
    """Rows the factory instantiates, shared with activation preflight."""
    document = _document(spec)
    result = [(row, None) for row in document.get("nodes", ())]
    gaps = {}
    rows = transport_resource(spec, "scenarios")["scenarios"]
    for scenario in _sequence(rows, "scenarios.scenarios"):
        scenario = _mapping(scenario, "scenario")
        name = _safe_node_id(scenario.get("id"))
        tier = scenario.get("tier")
        if tier not in {"entitlement", "calibration"} or name in gaps:
            raise ValueError(
                "Scenario ids must be distinct and tiers entitlement or calibration."
            )
        token = "scenario" if tier == "entitlement" else "variant"
        template = "scenario_nodes" if tier == "entitlement" else "variant_nodes"
        declared = scenario.get("nodes", document.get(template, ()))
        if not declared:
            raise ValueError(f"Scenario {name!r} requires a node template.")
        result.extend(
            (_format(_mapping(row, "entitlement node"), {token: name}), scenario)
            for row in _sequence(declared, template)
        )
        gap = document["scenario_gap" if tier == "entitlement" else "variant_gap"]
        gaps[name] = _format(gap, {token: name})
    if not gaps:
        raise ValueError("An entitlement graph needs at least its baseline scenario.")
    return result, gaps


def _types(rows, skeleton=None):
    types = (
        {}
        if skeleton is None
        else {
            (node.id, output.name): output.type
            for node in skeleton.nodes
            for output in node.artifact_outputs
        }
    )
    for row, _ in rows:
        outputs = {"takeup.gap@1": (ArtifactOutput("gap", GAP_TYPE),)}.get(
            row["kernel"], _artifact_outputs(row)
        )
        types.update({(row["id"], output.name): output.type for output in outputs})
    return types


@dataclass(frozen=True)
class _EntitlementFactory:
    def __call__(self, spec: Mapping, config: TransportGraphConfig, skeleton: Graph):
        document = _document(spec)
        rows, gaps = _rows(spec)
        types = _types(rows, skeleton)
        nodes: list[Node] = []
        for row, scenario in rows:
            params = {
                key: _parameter(spec, config, value, scenario)
                for key, value in row.get("params", {}).items()
            }
            if row["kernel"] == "takeup.gap@1" and "artifact_outputs" not in row:
                row = {
                    **row,
                    "artifact_outputs": [
                        {
                            "name": "gap",
                            "type": {
                                "name": GAP_TYPE.name,
                                "schema_version": GAP_TYPE.schema_version,
                            },
                        }
                    ],
                }
            nodes.append(
                _node(
                    spec,
                    row,
                    config,
                    types,
                    allowed_kernels=_KERNELS,
                    resolved_params=params,
                )
            )
        bands = _mapping(document["bands"], "entitlement bands")
        aliases = {f"scenario_{index}": name for index, name in enumerate(sorted(gaps))}
        nodes.append(
            Node(
                id=bands["id"],
                kernel="takeup.bands@1",
                population=bands["population"],
                params={
                    "baseline_scenario": bands["baseline_scenario"],
                    "artifact_scenarios": canonical_json(aliases).decode("utf-8"),
                },
                artifact_inputs=tuple(
                    ArtifactInput(alias, gaps[name], "gap", GAP_TYPE)
                    for alias, name in aliases.items()
                ),
                artifact_outputs=(ArtifactOutput("bands", BANDS_TYPE),),
            )
        )
        return tuple(nodes)


def entitlement_extension(spec: Mapping) -> TransportExtension:
    """The optional entitlement declarations as an ordered transport extension."""
    document = _document(spec)
    rows, _ = _rows(spec)
    bands = _mapping(document["bands"], "entitlement bands")
    package = _mapping(document["package"], "entitlement package")
    types = _types(rows)
    types[bands["id"], "bands"] = BANDS_TYPE
    edges = [
        ArtifactInput(
            package.get("artifact_name", "bands"), bands["id"], "bands", BANDS_TYPE
        )
    ]
    for row in package.get("inputs", ()):
        artifact = row.get("artifact", row["name"])
        edges.append(
            ArtifactInput(
                row["name"], row["producer"], artifact, types[row["producer"], artifact]
            )
        )
    return TransportExtension(
        factory=_EntitlementFactory(),
        sources=tuple(SourceRef(**row) for row in document.get("sources", ())),
        checkpoints=tuple(document.get("checkpoints", (bands["id"],))),
        package_inputs=tuple((package["id"], edge) for edge in edges),
    )


def transport_extensions(spec: Mapping, extensions=()) -> tuple:
    """Add the spec-declared entitlement extension once, preserving caller probes."""
    resources = spec.get("resources", {})
    if not any(
        name in resources
        for name in (ENTITLEMENT_RESOURCE, ENTITLEMENT_RESOURCE + ".json")
    ):
        return extensions
    if any(
        isinstance(getattr(item, "factory", None), _EntitlementFactory)
        for item in extensions
    ):
        return extensions
    return (*extensions, entitlement_extension(spec))
