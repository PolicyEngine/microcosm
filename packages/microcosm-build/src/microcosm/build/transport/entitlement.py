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
The extension adds explicit artifact edges to the final receipt and, when a
cell-support gate is declared, to its matching skeleton gate battery.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace

from microcosm.build.money_normalization import (
    selected_money_factors,
    validate_money_normalization,
)
from microcosm.build.reference_factors import validate_factor_context
from microcosm.graph import ArtifactInput, ArtifactOutput, Graph, Node, SourceRef
from microcosm.graph.canonical import canonical_json

from .compose import (
    GRAPH_RESOURCE,
    SKELETON_KERNELS,
    TransportExtension,
    TransportGraphConfig,
    _artifact_outputs,
    _has_resource,
    _literal_data,
    _mapping,
    _node,
    _param,
    _resource_param,
    _safe_node_id,
    _sequence,
    transport_resource,
)
from .money_kernels import MONEY_NORMALIZE_REF
from .takeup_kernels import (
    BANDS_TYPE,
    CELL_DIAGNOSTICS_TYPE,
    GAP_TYPE,
    reporting_cell_declarations,
)

__all__ = [
    "ENTITLEMENT_RESOURCE",
    "entitlement_extension",
    "money_node_factor_entries",
    "transport_extensions",
]

ENTITLEMENT_RESOURCE = "entitlement_graph"
_KERNELS = SKELETON_KERNELS | frozenset(
    {
        "simulate.counterfactual@1",
        "simulate.solve_zero@1",
        "takeup.gap@1",
        "takeup.bands@1",
        "takeup.cell_diagnostics@1",
        "takeup.compare@1",
        "transport.scenario_override@1",
        "reporting.income_decile@1",
        "reporting.carry_income_decile@1",
        "reporting.receipt_group@1",
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
    if "resource" in value:
        selected = _resource_param(
            spec,
            value,
            allow_json_sequences=True,
            profile_selection=None if scenario is None else scenario.get("knobs", {}),
        )
        if value.get("encoding") != "json":
            return selected
        selected = json.loads(selected)
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


def money_node_factor_entries(spec: Mapping) -> tuple[dict, ...]:
    """Resolve each declared money node's factors through composition's resolvers.

    Skeleton and entitlement rows select their own declaration, profile and
    context before engine preparation or CREATE. Keep contexts separate even
    when nodes select the same factor id. A spec without money nodes selects
    no money factors. Caller extensions contain code rather than spec rows;
    this walk does not cover them.
    """
    rows = []
    if _has_resource(spec, GRAPH_RESOURCE):
        rows.extend(
            (row, None, False)
            for row in transport_resource(spec, GRAPH_RESOURCE).get("nodes", ())
            if isinstance(row, Mapping)
        )
    if _has_resource(spec, ENTITLEMENT_RESOURCE) and _has_resource(spec, "scenarios"):
        rows.extend((row, scenario, True) for row, scenario in _rows(spec)[0])
    config = SimpleNamespace(engine_refs={})
    entries = []
    for row, scenario, entitlement in rows:
        if row.get("kernel") != MONEY_NORMALIZE_REF:
            continue
        documents = {}
        for name in ("declaration", "profile_selection", "factor_context"):
            try:
                value = _mapping(row.get("params", {}), "node params").get(name)
                if isinstance(value, Mapping) and {"binding", "prepared"} & set(value):
                    raise ValueError(
                        "Money parameters need data before engine preparation."
                    )
                value = (
                    _parameter(spec, config, value, scenario)
                    if entitlement
                    else _param(spec, value, None)
                )
                if not isinstance(value, str):
                    raise ValueError("The parameter must contain a JSON object string.")
                document = json.loads(value)
                if not isinstance(document, dict):
                    raise ValueError("The parameter must contain a JSON object string.")
                documents[name] = document
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(
                    f"Money node {row.get('id')!r} {name}: {error}"
                ) from error
        try:
            declaration = validate_money_normalization(documents["declaration"])
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(
                f"Money node {row.get('id')!r} declaration: {error}"
            ) from error
        try:
            _, factors = selected_money_factors(
                declaration, documents["profile_selection"]
            )
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(
                f"Money node {row.get('id')!r} profile_selection: {error}"
            ) from error
        ids = sorted({value for value in factors.values() if value is not None})
        if ids:
            try:
                entries.append(
                    validate_factor_context(documents["factor_context"], ids)
                )
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(
                    f"Money node {row.get('id')!r} factor_context: {error}"
                ) from error
    return tuple(entries)


def _require_matching_cell_targets(spec, contract):
    """Refuse a tracked cell target that disagrees with the support contract.

    The support contract's target thresholds decide the per-cell flags in the
    diagnostics; the ``cell_support_target`` gate reports them. Both are spec
    data, so a spec that changes one copy without the other would flag cells
    against one threshold and report against another.

    Every active target gate is checked, ``"not_applicable": null`` included,
    as the gate parser treats it. Each shared value must be declared on both
    sides. Lists compare as multisets, ordered by their canonical JSON, so a
    malformed list is refused with this error rather than a ``TypeError``.
    """
    pairs = (
        ("minimum_cell_ess", "target_minimum_cell_ess"),
        (
            "maximum_single_donor_dollar_share",
            "target_maximum_single_donor_dollar_share",
        ),
        ("required_cells", "required_cells"),
        ("required_scenarios", "required_scenarios"),
        ("aggregation_column", "aggregation_column"),
    )
    for gate in transport_resource(spec, "gates")["gates"]:
        if gate["gate"] != "cell_support_target" or not _applicable(gate):
            continue
        target = _mapping(gate.get("parameters"), "cell-support target parameters")
        for own, declared in pairs:
            if own not in target or declared not in contract:
                raise ValueError(
                    f"The cell-support target gate {gate['id']!r} and the support "
                    f"contract must both declare {own!r} (contract: {declared!r})."
                )
            if not _same_declaration(target[own], contract[declared]):
                raise ValueError(
                    f"The cell-support target gate {gate['id']!r} declares "
                    f"{own}={target[own]!r}, but the support contract declares "
                    f"{declared}={contract[declared]!r}; the two must agree."
                )


def _applicable(gate):
    """Return whether a gate entry runs: ``not_applicable`` absent or null."""
    return gate.get("not_applicable") is None


def _same_declaration(mine, theirs):
    """Compare two declared values, lists as multisets."""
    if isinstance(mine, list | tuple) and isinstance(theirs, list | tuple):

        def key(item):
            return json.dumps(item, sort_keys=True)

        return sorted(mine, key=key) == sorted(theirs, key=key)
    return mine == theirs


_CELL_CONTRACT_KEYS = (
    "minimum_cell_ess",
    "maximum_contribution_cv",
    "maximum_single_donor_dollar_share",
    "target_minimum_cell_ess",
    "target_maximum_single_donor_dollar_share",
    "required_cells",
    "required_scenarios",
    "aggregation_column",
    "required_weight_kind",
    "required_weight_entity",
)


def _require_cell_contract_shape(gate, parameters):
    """Refuse a malformed support contract with a named error.

    Every key the contract is read for must be present, and the cell and
    scenario lists must hold non-empty strings, before thresholds are read or
    lists are compared as sets.
    """
    missing = [key for key in _CELL_CONTRACT_KEYS if key not in parameters]
    if missing:
        raise ValueError(
            f"The cell-support contract {gate.get('id')!r} omits {missing}."
        )
    for key in ("required_cells", "required_scenarios"):
        values = parameters[key]
        if (
            not isinstance(values, list | tuple)
            or not values
            or not all(isinstance(value, str) and value for value in values)
        ):
            raise ValueError(
                f"The cell-support contract {gate.get('id')!r} declares {key}="
                f"{values!r}; it must be a non-empty list of non-empty strings."
            )


def _cell_contract(spec):
    resources = spec.get("resources", {})
    if not any(name in resources for name in ("gates", "gates.json")):
        return None
    matches = [
        row
        for row in transport_resource(spec, "gates")["gates"]
        if row["gate"] == "cell_support" and _applicable(row)
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(
            "An entitlement graph needs exactly one cell-support contract."
        )
    gate = matches[0]
    parameters = _mapping(gate.get("parameters"), "cell-support parameters")
    _require_cell_contract_shape(gate, parameters)
    thresholds = {
        name: parameters[name]
        for name in (
            "minimum_cell_ess",
            "maximum_contribution_cv",
            "maximum_single_donor_dollar_share",
            "target_minimum_cell_ess",
            "target_maximum_single_donor_dollar_share",
        )
    }
    _require_matching_cell_targets(spec, parameters)
    reporting = transport_resource(spec, "scenarios")["gap_conventions"][
        "reporting_cells"
    ]
    declarations = reporting_cell_declarations(reporting)
    if set(parameters["required_cells"]) != {row["name"] for row in declarations}:
        raise ValueError(
            "The support gate must require exactly the declared reported cells."
        )
    _, gaps = _rows(spec)
    if set(parameters["required_scenarios"]) != set(gaps):
        raise ValueError("The support gate must require every declared scenario.")
    return gate, thresholds, reporting


def _types(rows, skeleton=None, *, cell_support=False):
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
        if row["kernel"] == "takeup.gap@1":
            outputs = (ArtifactOutput("gap", GAP_TYPE),)
            if cell_support:
                outputs += (ArtifactOutput("cell_diagnostics", CELL_DIAGNOSTICS_TYPE),)
        else:
            outputs = _artifact_outputs(row)
        types.update({(row["id"], output.name): output.type for output in outputs})
    return types


@dataclass(frozen=True)
class _EntitlementFactory:
    def __call__(self, spec: Mapping, config: TransportGraphConfig, skeleton: Graph):
        document = _document(spec)
        rows, gaps = _rows(spec)
        cell_contract = _cell_contract(spec)
        types = _types(rows, skeleton, cell_support=cell_contract is not None)
        nodes: list[Node] = []
        for row, scenario in rows:
            params = {
                key: _parameter(spec, config, value, scenario)
                for key, value in row.get("params", {}).items()
            }
            if row["kernel"] == "takeup.gap@1" and cell_contract is not None:
                gate, thresholds, reporting = cell_contract
                contract_params = gate["parameters"]
                params.update(
                    source_household_column=contract_params["aggregation_column"],
                    expected_weight_kind=contract_params["required_weight_kind"],
                    cell_support=canonical_json(thresholds).decode("utf-8"),
                    reporting_cells=canonical_json(reporting).decode("utf-8"),
                )
                if (
                    params.get("weight_entity")
                    != contract_params["required_weight_entity"]
                ):
                    raise ValueError(
                        "Reported cells must use the declared household weight entity."
                    )
                extra_columns = {
                    contract_params["required_weight_entity"]: {
                        contract_params["aggregation_column"]
                    }
                }
                extra_columns.setdefault(reporting["entity"], set()).update(
                    {
                        reporting["measure"],
                        *(
                            axis["column"]
                            for table in reporting["tables"]
                            for axis in table["axes"]
                        ),
                    }
                )
                inputs = [dict(item) for item in row.get("inputs", [])]
                for entity, columns in extra_columns.items():
                    matching = next(
                        (item for item in inputs if item["entity"] == entity), None
                    )
                    if matching is None:
                        inputs.append({"entity": entity, "columns": sorted(columns)})
                    else:
                        existing = list(matching["columns"])
                        matching["columns"] = existing + sorted(columns - set(existing))
                row = {**row, "inputs": inputs}
            if row["kernel"] == "takeup.gap@1":
                outputs = list(
                    row.get(
                        "artifact_outputs",
                        [
                            {
                                "name": "gap",
                                "type": {
                                    "name": GAP_TYPE.name,
                                    "schema_version": GAP_TYPE.schema_version,
                                },
                            }
                        ],
                    )
                )
                if cell_contract is not None and not any(
                    output["name"] == "cell_diagnostics" for output in outputs
                ):
                    outputs.append(
                        {
                            "name": "cell_diagnostics",
                            "type": {
                                "name": CELL_DIAGNOSTICS_TYPE.name,
                                "schema_version": CELL_DIAGNOSTICS_TYPE.schema_version,
                            },
                        }
                    )
                row = {
                    **row,
                    "artifact_outputs": outputs,
                }
            nodes.append(
                _node(
                    spec,
                    row,
                    config,
                    types,
                    allowed_kernels=_KERNELS,
                    resolved_params=params,
                    profile_selection=None
                    if scenario is None
                    else scenario.get("knobs", {}),
                )
            )
        bands = _mapping(document["bands"], "entitlement bands")
        aliases = {f"scenario_{index}": name for index, name in enumerate(sorted(gaps))}
        if cell_contract is not None:
            nodes.append(
                Node(
                    id=bands["id"] + ".cell_diagnostics",
                    kernel="takeup.cell_diagnostics@1",
                    population=bands["population"],
                    params={
                        "artifact_scenarios": canonical_json(aliases).decode("utf-8")
                    },
                    artifact_inputs=tuple(
                        ArtifactInput(
                            alias, gaps[name], "cell_diagnostics", CELL_DIAGNOSTICS_TYPE
                        )
                        for alias, name in aliases.items()
                    ),
                    artifact_outputs=(
                        ArtifactOutput("cell_diagnostics", CELL_DIAGNOSTICS_TYPE),
                    ),
                )
            )
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
    cell_contract = _cell_contract(spec)
    types = _types(rows, cell_support=cell_contract is not None)
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
    checkpoints = tuple(document.get("checkpoints", (bands["id"],)))
    gate_inputs = []
    if cell_contract is not None:
        collector = bands["id"] + ".cell_diagnostics"
        edges.append(
            ArtifactInput(
                "cell_diagnostics", collector, "cell_diagnostics", CELL_DIAGNOSTICS_TYPE
            )
        )
        checkpoints += (collector,)
        screen_gate = cell_contract[0]
        for row in transport_resource(spec, "transport_graph")["nodes"]:
            if (
                row["kernel"] == "gates.battery@1"
                and row.get("params", {}).get("phase") == screen_gate["phase"]
            ):
                gate_inputs.append(
                    (
                        row["id"],
                        ArtifactInput(
                            "cell_diagnostics",
                            collector,
                            "cell_diagnostics",
                            CELL_DIAGNOSTICS_TYPE,
                        ),
                    )
                )
        if not gate_inputs:
            raise ValueError(
                "Cell diagnostics need a matching declared gate-battery node."
            )
    return TransportExtension(
        factory=_EntitlementFactory(),
        sources=tuple(SourceRef(**row) for row in document.get("sources", ())),
        checkpoints=checkpoints,
        package_inputs=tuple((package["id"], edge) for edge in edges),
        gate_inputs=tuple(gate_inputs),
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
