"""Project producer contracts into existing Orrery presentation surfaces."""

from __future__ import annotations

from .canonical import canonical_json
from .evidence import EXECUTION_PROTOCOL, sha256
from .presentation import (
    digest,
    presentation_id,
    require,
    source_reference,
    text,
    validate_presentation,
)
from .schema import _plain_json


def apply_presentation(document: dict, schema: dict) -> None:
    contract = validate_presentation(schema)
    if contract is None:
        return
    nodes = {node["id"]: node for node in document["nodes"]}
    membership = {}
    for group in contract.get("groups", []):
        node = {
            "id": presentation_id("group", group["id"]),
            "label": group["label"],
            "kind": "group",
            "data": {"presentation_group": group["id"]},
        }
        if "description" in group:
            node["description"] = group["description"]
        if "parent" in group:
            node["parentId"] = presentation_id("group", group["parent"])
        nodes[node["id"]] = node
        for plural, kind in (("operations", "operation"), ("sources", "source")):
            for member in group.get(plural, []):
                identity = presentation_id(kind, member)
                membership[identity] = node["id"]
                nodes[identity]["parentId"] = node["id"]
    for node in nodes.values():
        if node["kind"] == "field":
            group = membership.get(
                presentation_id("operation", node["data"]["provider"])
            )
            if group is not None:
                node["parentId"] = group
    for plural, kind in (("operations", "operation"), ("sources", "source")):
        for identity, record in contract.get(plural, {}).items():
            node = nodes[presentation_id(kind, identity)]
            for prop in ("label", "description"):
                if prop in record:
                    node[prop] = record[prop]
            if "composite" in record:
                node["data"]["composite"] = record["composite"]
            if record.get("references"):
                node["sources"] = record["references"]
    for boundary in contract.get("scope", {}).get("boundaries", []):
        node = nodes[presentation_id("operation", boundary["operation"])]
        node["data"]["scope_boundary"] = boundary
        node.setdefault("sources", []).extend(boundary.get("upstream", []))
        node.setdefault("statuses", []).append(
            {
                "label": "Upstream evidence missing"
                if boundary.get("missing")
                else "Upstream references recorded",
                "tone": "warning" if boundary.get("missing") else "neutral",
            }
        )
    document["nodes"] = [nodes[key] for key in sorted(nodes)]
    document["metadata"]["presentation_scope"] = contract.get("scope", {})


def apply_execution(document: dict, schema: dict, execution: object) -> None:
    execution = _plain_json(execution)
    require(type(execution) is dict, "execution overlay must be an object")
    require(
        set(execution)
        == {
            "protocol",
            "graph_sha256",
            "phases",
            "summaries",
            "verification",
            "history",
        }
        and execution["protocol"] == EXECUTION_PROTOCOL,
        "execution overlay protocol/fields",
    )
    require(
        digest(execution["graph_sha256"]) == schema["graph_sha256"],
        "overlay graph binding mismatch",
    )
    require(
        type(execution["phases"]) is list and bool(execution["phases"]),
        "overlay phases",
    )
    text(execution["verification"], "verification scope")
    text(execution["history"], "history scope")
    nodes = {node["id"]: node for node in document["nodes"]}
    operation_ids = {node["id"] for node in schema["graph"]["nodes"]}
    source_ids = {source["name"] for source in schema["graph"]["sources"]}
    activities = []
    artifacts = {}
    seen = set()
    latest = {}
    summary_catalog = {}
    for summary in execution["summaries"]:
        key = digest(summary["key"])
        require(
            key not in summary_catalog and type(summary["data"]) is dict,
            "duplicate or invalid summary",
        )
        summary_catalog[key] = summary
    for phase in execution["phases"]:
        require(
            type(phase) is dict
            and set(phase)
            == {
                "attempt_id",
                "phase",
                "graph_sha256",
                "manifest_sha256",
                "manifest_key",
                "started_at",
                "finished_at",
                "source_identities",
                "operations",
                "artifacts",
                "summaries",
                "references",
            },
            "overlay phase fields",
        )
        attempt = text(phase["attempt_id"], "attempt id")
        name = text(phase["phase"], "phase")
        require((attempt, name) not in seen, "duplicate overlay phase")
        seen.add((attempt, name))
        for key in ("graph_sha256", "manifest_sha256", "manifest_key"):
            digest(phase[key])
        require(
            type(phase["operations"]) is dict
            and set(phase["operations"]) <= operation_ids,
            "overlay operations",
        )
        require(
            type(phase["source_identities"]) is dict
            and set(phase["source_identities"]) <= source_ids,
            "overlay sources",
        )
        for key in phase["source_identities"].values():
            digest(key)
        for source, key in phase["source_identities"].items():
            nodes[presentation_id("source", source)]["data"].setdefault(
                "execution_history", []
            ).append({"attempt_id": attempt, "phase": name, "content_identity": key})
        artifact_ids = {}
        for artifact in phase["artifacts"]:
            key = digest(artifact["key"])
            require(type(artifact["payloads"]) is dict, "artifact payload table")
            ids = []
            for filename, payload in artifact["payloads"].items():
                identity = presentation_id("artifact", key, filename)
                artifacts[identity] = {
                    "id": identity,
                    "label": f"{artifact['producers'][0]['node']}.{artifact['producers'][0]['name']}: {filename}",
                    "sha256": digest(payload["sha256"]),
                }
                ids.append(identity)
            artifact_ids[key] = ids
        require(type(phase["references"]) is dict, "phase references")
        refs = []
        phase_artifacts = []
        for kind, ref in phase["references"].items():
            source_reference(ref)
            identity = presentation_id("run-artifact", attempt, name, kind)
            artifacts[identity] = {
                "id": identity,
                "label": ref["label"],
                "sha256": ref["sha256"],
            }
            if "url" in ref:
                artifacts[identity]["uri"] = ref["url"]
            refs.append(ref)
            phase_artifacts.append(identity)
        summaries = {}
        for summary in phase["summaries"]:
            require(summary["node"] in phase["operations"], "summary operation missing")
            key = digest(summary["key"])
            require(key in artifact_ids, "summary artifact missing")
            require(key in summary_catalog, "summary catalog entry missing")
            summaries.setdefault(summary["node"], []).append(summary)
        for node_id, record in phase["operations"].items():
            require(type(record) is dict, "operation evidence must be an object")
            require(
                record.get("execution")
                in {"completed", "failed", "unreached", "missing"},
                "execution state",
            )
            require(record.get("cache") in {"hit", "miss", "unknown"}, "cache state")
            require(
                record.get("gate")
                in {
                    "pass",
                    "fail",
                    "evidence_absent",
                    "not_applicable",
                    "unreached",
                    "unknown",
                },
                "gate state",
            )
            identity = presentation_id("operation", node_id)
            node = nodes[identity]
            details = {"attempt_id": attempt, "phase": name, **record}
            if node_id in summaries:
                details["artifact_summary_keys"] = [
                    item["key"] for item in summaries[node_id]
                ]
                previews = {}
                for item in summaries[node_id]:
                    data = summary_catalog[item["key"]]["data"]
                    preview = {
                        key: value for key, value in data.items() if key != "tables"
                    }
                    preview["tables"] = {
                        name: {
                            "rows": rows[:50],
                            "total_rows": len(rows),
                            "preview_limit": 50,
                        }
                        for name, rows in data.get("tables", {}).items()
                    }
                    previews[item["artifact"]] = {"key": item["key"], **preview}
                node["data"]["artifact_summaries"] = previews
            node["data"].setdefault("execution_history", []).append(details)
            if refs:
                node.setdefault("sources", []).extend(refs)
            latest[node_id] = record
            own_artifacts = [
                identity
                for artifact in phase["artifacts"]
                if any(
                    producer["node"] == node_id for producer in artifact["producers"]
                )
                for identity in artifact_ids[artifact["key"]]
            ]
            activity = {
                "id": presentation_id("activity", attempt, name, node_id),
                "label": f"{name}: {node['label']}",
                "kind": "execution",
                "outputs": [{"type": "node", "id": identity}],
                "inputs": [
                    {
                        "subject": {
                            "type": "node",
                            "id": presentation_id("operation", parent),
                        },
                        "role": "provided",
                    }
                    for parent in schema["compiled"]["predecessors"][node_id]
                ]
                + [
                    {
                        "subject": {
                            "type": "node",
                            "id": presentation_id("source", source),
                        },
                        "role": "provided",
                    }
                    for operation in schema["graph"]["nodes"]
                    if operation["id"] == node_id
                    for source in operation["sources"]
                ],
                "artifactIds": sorted(set(own_artifacts + phase_artifacts)),
                "data": {
                    **details,
                    "input_scope": "Declared inputs bound into receipt keys; not observed kernel reads.",
                    "timestamp_scope": "Phase timestamps; operation duration is wall_time_s.",
                },
            }
            for timestamp, prop in (
                ("started_at", "startedAt"),
                ("finished_at", "endedAt"),
            ):
                if phase[timestamp]:
                    activity[prop] = phase[timestamp]
            activities.append(activity)
    for node_id in sorted(operation_ids):
        node = nodes[presentation_id("operation", node_id)]
        record = latest.get(
            node_id, {"execution": "missing", "cache": "unknown", "gate": "unknown"}
        )
        statuses = node.setdefault("statuses", [])
        statuses.append(
            {
                "label": f"Execution: {record['execution']}",
                "tone": "negative"
                if record["execution"] == "failed"
                else "warning"
                if record["execution"] in {"missing", "unreached"}
                else "positive",
            }
        )
        statuses.append({"label": f"Cache: {record['cache']}", "tone": "neutral"})
        statuses.append(
            {
                "label": f"Gate: {record['gate']}",
                "tone": "positive"
                if record["gate"] == "pass"
                else "negative"
                if record["gate"] == "fail"
                else "warning"
                if record["gate"] in {"evidence_absent", "unreached", "unknown"}
                else "neutral",
            }
        )
    document["activities"] = activities
    document["artifacts"] = [artifacts[key] for key in sorted(artifacts)]
    document["metadata"]["execution"] = execution
    document["metadata"]["evidence_scope"] = execution["verification"]
    document["metadata"]["revision_scope"] = (
        "Content digests of declarations, presentation and supplied execution records; "
        "not runtime cache keys."
    )
    document["metadata"]["missing_runtime_data"] = [
        "entity identifiers and memberships",
        "field values",
        "source bytes and preparation internals",
        "kernel code, replay and implicit runtime reads",
        "raw runtime receipts and signed release decisions",
    ]
    document["description"] = "Compiler-derived graph with recorded execution evidence."
    document["revision"] = "sha256:" + sha256(canonical_json([schema, execution]))


def apply_contracts(document: dict, schema: dict, execution: object | None) -> None:
    apply_presentation(document, schema)
    if execution is not None:
        apply_execution(document, schema, execution)
