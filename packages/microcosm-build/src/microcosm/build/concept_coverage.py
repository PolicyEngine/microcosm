"""Input concept-coverage diagnostics, never a population/certification gate.

Executable input discovery is adapter-owned. Consumer-authored bindings preserve
publisher scope and evidence; matching names do not establish equivalence. This
v0 inventories missing contracts without filling them with guessed metadata.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from microcosm.build.spec_engine.canonical import CANONICALIZER_ID, sha256_json
from microcosm.frame import Frame
from microcosm.frame.rules import InputInventoryProvider

_TEXT = {"type": "string", "pattern": r"\S"}
_MAYBE_TEXT = {"anyOf": [_TEXT, {"type": "null"}]}
_SHA256 = {"type": "string", "pattern": "^[0-9a-f]{64}$"}


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def _array(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items, "uniqueItems": True}


_KEY = _object({"entity": _TEXT, "name": _TEXT})
_INPUT_KEY = _object(
    {
        "entity": _TEXT,
        "engine_entity": _TEXT,
        "name": _TEXT,
        "canonical_request_name": _TEXT,
    }
)
_FINGERPRINT = _object({"role": _TEXT, "name": _TEXT, "sha256": _SHA256})
_DEFINITION = _object({"uri": _TEXT, "sha256": _SHA256, "locator": _TEXT})
_SCOPE = _object(
    {
        **{
            key: _MAYBE_TEXT
            for key in (
                "statistic",
                "entity",
                "universe",
                "unit",
                "geography",
                "period",
            )
        },
        "measurement_kind": {"enum": ["stock", "flow", "not_applicable", None]},
        "reference_instant": _MAYBE_TEXT,
        "accounting_basis": {
            "enum": [
                "income_year",
                "assessment_year",
                "calendar_year",
                "tax_year",
                "instant",
                "not_applicable",
                None,
            ]
        },
        "entity_definition": {"anyOf": [_DEFINITION, {"type": "null"}]},
        "universe_definition": {"anyOf": [_DEFINITION, {"type": "null"}]},
    }
)
_EVIDENCE = _object({"uri": _TEXT, "sha256": _SHA256, "locator": _TEXT, "claim": _TEXT})
_COLUMN_BINDING = _object(
    {"input": _INPUT_KEY, "artifact_fingerprints": _array(_FINGERPRINT), "column": _KEY}
)
_FACT_BINDING = _object(
    {
        "input": _INPUT_KEY,
        "artifact_fingerprints": _array(_FINGERPRINT),
        "asserted_relationship": {"enum": ["exact", "proxy", "unresolved"]},
        "source": _object(
            {
                "fact_id": _TEXT,
                "concept_id": _TEXT,
                "vintage": _TEXT,
                "artifact_sha256": _SHA256,
                "scope": _SCOPE,
            }
        ),
        "target": _object(
            {
                "concept_id": _MAYBE_TEXT,
                "legal_vintage": _TEXT,
                "scope": _SCOPE,
            }
        ),
        "transformation": _object(
            {
                "kind": {"enum": ["identity", "declared_conversion", "unresolved"]},
                "description": _TEXT,
            }
        ),
        "evidence": _array(_EVIDENCE),
    }
)
_CLASSIFIED_BINDING = _object(
    {
        **_FACT_BINDING["properties"],
        "effective_relationship": {"const": "unresolved"},
        "classification_reason": {
            "enum": [
                "asserted_unresolved",
                "target_semantics_unavailable",
                "semantic_equivalence_unverified",
            ]
        },
    }
)
_METADATA = _object(
    {
        **{
            key: _MAYBE_TEXT
            for key in ("dtype", "unit", "period", "definition", "concept_id")
        },
        "required": {"type": ["boolean", "null"]},
    }
)
MANIFEST_SCHEMA = _object(
    {
        "schema_version": {"const": 1},
        "artifact_kind": {"const": "concept_coverage_diagnostic"},
        "canonicalization": {"const": CANONICALIZER_ID},
        "scope": {"const": "mapped_entity_root_inputs_all_module_versions"},
        "fingerprints": _array(_FINGERPRINT),
        "mapped_entities": _array(_TEXT),
        "entity_discovery": _array(
            _object(
                {
                    "entity": _TEXT,
                    "engine_entity": _TEXT,
                    "status": {"enum": ["complete", "no_derived_program"]},
                    "root_input_count": {"type": ["integer", "null"], "minimum": 0},
                }
            )
        ),
        "root_input_count": {"type": "integer", "minimum": 0},
        "runtime": _object(
            {
                key: _MAYBE_TEXT
                for key in (
                    "engine",
                    "wrapper_distribution_version",
                    "native_distribution_version",
                    "core_version",
                    "python",
                    "platform",
                    "numpy",
                )
            }
        ),
        "dataset": _object(
            {
                "status": {"enum": ["not_supplied", "schema_inspected"]},
                "assessment": {"const": "column_presence_only"},
                "presence_only": {"const": True},
                "schema_sha256": {"anyOf": [_SHA256, {"type": "null"}]},
                "columns": {"anyOf": [_array(_KEY), {"type": "null"}]},
                "extra_columns": {"const": "permitted"},
            }
        ),
        "inputs": _array(
            _object(
                {
                    "name": _TEXT,
                    "entity": _TEXT,
                    "engine_entity": _TEXT,
                    "canonical_request_name": _MAYBE_TEXT,
                    "request_names": _array(_TEXT),
                    "metadata": _METADATA,
                    "metadata_gaps": _array(_TEXT),
                    "column_status": {"enum": ["unassessed", "present", "absent"]},
                    "data_origin": {"const": "unassessed"},
                    "semantic_status": {
                        "enum": ["unassessed", "unverified_assertions"]
                    },
                }
            )
        ),
        "column_bindings": _array(_COLUMN_BINDING),
        "fact_bindings": _array(_CLASSIFIED_BINDING),
        "blocking_gaps": _array(_TEXT),
        "readiness": _object(
            {
                "certified": {"const": False},
                "population_schema_ready": {"const": False},
                "reason": {"const": "diagnostic_only"},
            }
        ),
        "content_sha256": _SHA256,
    }
)


def _validate_schema(schema: dict[str, Any], payload: object) -> None:
    try:
        Draft202012Validator(schema).validate(payload)
    except ValidationError as exc:
        if exc.validator == "uniqueItems":
            raise ValueError(
                "Invalid concept-coverage manifest: duplicate entries"
            ) from exc
        raise ValueError(f"Invalid concept-coverage manifest: {exc.message}") from exc


def _key(item: Mapping[str, Any]) -> tuple[str, str]:
    return item["entity"], item["name"]


def _input_sort_key(item: Mapping[str, Any]) -> tuple[str, str, str]:
    return item["canonical_request_name"] or "", item["entity"], item["name"]


def _classification(binding: dict, item: dict) -> tuple[str, str]:
    # v0 checks assertion consistency, not legal/statistical equivalence. Even
    # full future metadata would still need a semantic adjudication mechanism.
    if binding["asserted_relationship"] == "unresolved":
        return "unresolved", "asserted_unresolved"
    if item["metadata_gaps"]:
        return "unresolved", "target_semantics_unavailable"
    return "unresolved", "semantic_equivalence_unverified"


def _blocking_gaps(report: dict) -> list[str]:
    gaps = {"population_schema_and_certification_not_assessed"}
    if any(item["metadata_gaps"] for item in report["inputs"]):
        gaps.add("input_metadata_unavailable")
    if any(item["canonical_request_name"] is None for item in report["inputs"]):
        gaps.add("canonical_input_addresses_unavailable")
    if report["dataset"]["status"] == "not_supplied":
        gaps.add("dataset_not_supplied")
    if any(item["column_status"] == "unassessed" for item in report["inputs"]):
        gaps.add("dataset_column_coverage_unassessed")
    if any(item["column_status"] == "absent" for item in report["inputs"]):
        gaps.add("explicitly_bound_dataset_columns_absent")
    if report["inputs"]:
        gaps.add("semantic_evidence_unresolved")
    for item in report["entity_discovery"]:
        if item["status"] == "no_derived_program":
            gaps.add(f"no_derived_program:{item['entity']}")
    return sorted(gaps)


def _unique_index(items: Sequence[Mapping[str, Any]], label: str) -> dict:
    result = {}
    for item in items:
        key = _key(item)
        if key in result:
            raise ValueError(f"duplicate {label}: {key}")
        result[key] = item
    return result


def _column_status(key: tuple[str, str], dataset: dict, bindings: dict) -> str:
    if dataset["status"] == "not_supplied" or key not in bindings:
        return "unassessed"
    columns = {_key(item) for item in dataset["columns"]}
    return "present" if _key(bindings[key]["column"]) in columns else "absent"


def build_concept_coverage(
    engine: InputInventoryProvider,
    *,
    dataset: Frame | None = None,
    column_bindings: Sequence[Mapping[str, Any]] = (),
    fact_bindings: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Discover inputs and make missing evidence explicit, without evaluating tax.

    A supplied Frame is inspected only for column presence, using explicit
    input-to-column bindings. Its values, dtypes and weights are not inspected;
    column presence does not identify observed versus imputed origin. No data
    supplied, or no explicit binding, means unassessed, never measured absent.
    Schema digests identify column names only, not the microdata artifact.

    Asserted relationships are preserved separately from machine-effective
    classification. Missing target semantics means unresolved, not proxy.
    v0 never adjudicates semantic equivalence, so it cannot emit effective exact
    or proxy merely because an assertion has well-formed evidence fields.
    Nothing here edits Chronicle or certifies a build.
    """
    if not callable(getattr(engine, "input_inventory", None)):
        raise TypeError("Rules adapter does not support input discovery.")
    if dataset is not None:
        for entity in dataset.entities:
            if any(
                not isinstance(column, str) for column in dataset.table(entity).columns
            ):
                raise ValueError(
                    f"Concept coverage cannot represent non-string column labels on {entity!r}."
                )
    inventory = engine.input_inventory()
    columns = (
        sorted(
            [
                {"entity": entity, "name": column}
                for entity in dataset.entities
                for column in dataset.table(entity).columns
            ],
            key=_key,
        )
        if dataset is not None
        else None
    )
    dataset_report = {
        "status": "schema_inspected" if dataset is not None else "not_supplied",
        "assessment": "column_presence_only",
        "presence_only": True,
        "schema_sha256": sha256_json(columns) if columns is not None else None,
        "columns": columns,
        "extra_columns": "permitted",
    }
    column_records = deepcopy(list(column_bindings))
    fact_records = deepcopy(list(fact_bindings))
    _validate_schema(_array(_COLUMN_BINDING), column_records)
    _validate_schema(_array(_FACT_BINDING), fact_records)
    for binding in column_records + fact_records:
        binding["artifact_fingerprints"].sort(
            key=lambda item: (item["role"], item["name"])
        )
    column_index = _unique_index(
        [{**item["input"], "column": item["column"]} for item in column_records],
        "column binding",
    )
    asserted = {_key(item["input"]) for item in fact_records}
    inputs = []
    for record in inventory.inputs:
        item = asdict(record)
        item["request_names"] = sorted(item["request_names"])
        metadata = {key: item.pop(key) for key in _METADATA["properties"]}
        item.update(
            metadata=metadata,
            metadata_gaps=sorted(
                key for key, value in metadata.items() if value is None
            ),
            column_status=_column_status(_key(item), dataset_report, column_index),
            data_origin="unassessed",
            semantic_status=(
                "unverified_assertions" if _key(item) in asserted else "unassessed"
            ),
        )
        inputs.append(item)
    report = {
        "schema_version": 1,
        "artifact_kind": "concept_coverage_diagnostic",
        "canonicalization": CANONICALIZER_ID,
        "scope": "mapped_entity_root_inputs_all_module_versions",
        "fingerprints": sorted(
            [dict(item) for item in inventory.fingerprints]
            + [
                {
                    "role": "diagnostic_builder",
                    "name": "microcosm.build.concept_coverage",
                    "sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                }
            ],
            key=lambda item: (item["role"], item["name"]),
        ),
        "mapped_entities": sorted(inventory.mapped_entities),
        "entity_discovery": sorted(
            [dict(item) for item in inventory.entity_discovery],
            key=lambda item: item["entity"],
        ),
        "root_input_count": len(inputs),
        "runtime": dict(inventory.runtime),
        "dataset": dataset_report,
        "inputs": sorted(inputs, key=_input_sort_key),
        "column_bindings": sorted(
            column_records, key=lambda item: _input_sort_key(item["input"])
        ),
        "fact_bindings": sorted(
            fact_records,
            key=lambda item: (
                *_input_sort_key(item["input"]),
                item["source"]["fact_id"],
            ),
        ),
        "readiness": {
            "certified": False,
            "population_schema_ready": False,
            "reason": "diagnostic_only",
        },
    }
    input_index = _unique_index(inputs, "input")
    for binding in report["fact_bindings"]:
        key = _key(binding["input"])
        if key not in input_index:
            raise ValueError(f"binding refers to unknown input {key}")
        effective, reason = _classification(binding, input_index[key])
        binding.update(effective_relationship=effective, classification_reason=reason)
    report["blocking_gaps"] = _blocking_gaps(report)
    report["content_sha256"] = sha256_json(report)
    validate_concept_coverage(report)
    return report


def validate_concept_coverage(payload: Mapping[str, Any]) -> None:
    """Validate closed JSON shape, digest, and fail-closed cross-field semantics."""
    report = dict(payload)
    _validate_schema(MANIFEST_SCHEMA, report)
    content = {key: value for key, value in report.items() if key != "content_sha256"}
    if sha256_json(content) != report["content_sha256"]:
        raise ValueError("concept-coverage content digest mismatch")
    inputs = _unique_index(report["inputs"], "input")
    if report["inputs"] != sorted(report["inputs"], key=_input_sort_key):
        raise ValueError("inputs are not sorted by canonical address")
    if report["root_input_count"] != len(inputs):
        raise ValueError("root input count disagrees with inventory")
    _validate_discovery(report, inputs)
    column_index = _unique_index(
        [
            {**item["input"], "column": item["column"]}
            for item in report["column_bindings"]
        ],
        "column binding",
    )
    dataset = report["dataset"]
    if dataset["status"] == "not_supplied":
        if dataset["columns"] is not None or dataset["schema_sha256"] is not None:
            raise ValueError("An unsupplied dataset cannot carry a measured schema.")
    elif (
        dataset["columns"] is None
        or sha256_json(dataset["columns"]) != dataset["schema_sha256"]
    ):
        raise ValueError("dataset schema digest mismatch")
    for binding in report["column_bindings"] + report["fact_bindings"]:
        key = _key(binding["input"])
        if key not in inputs:
            raise ValueError(
                f"binding refers to unknown input {_key(binding['input'])}"
            )
        if binding["input"]["engine_entity"] != inputs[key]["engine_entity"]:
            raise ValueError(
                "binding engine entity disagrees with the compiled input context"
            )
        if (
            binding["input"]["canonical_request_name"]
            != inputs[key]["canonical_request_name"]
        ):
            raise ValueError(
                "binding uses a non-canonical alias or unknown request address"
            )
        expected_pins = [
            item
            for item in report["fingerprints"]
            if item["role"] != "diagnostic_builder"
        ]
        if not expected_pins or binding["artifact_fingerprints"] != expected_pins:
            raise ValueError("binding artifact fingerprint mismatch")
    asserted = {_key(item["input"]) for item in report["fact_bindings"]}
    for key, item in inputs.items():
        if item["entity"] not in report["mapped_entities"]:
            raise ValueError(f"Input {key} is outside mapped entity scope.")
        if (
            item["canonical_request_name"] is not None
            and item["canonical_request_name"] not in item["request_names"]
        ):
            raise ValueError("canonical request name is absent from accepted names")
        gaps = sorted(name for name, value in item["metadata"].items() if value is None)
        if gaps != item["metadata_gaps"]:
            raise ValueError("metadata gap report disagrees with unknown fields")
        if item["column_status"] != _column_status(key, dataset, column_index):
            raise ValueError("column status disagrees with explicit binding and schema")
        expected = "unverified_assertions" if key in asserted else "unassessed"
        if item["semantic_status"] != expected:
            raise ValueError("semantic status disagrees with recorded assertions")
    seen_facts = set()
    target_concepts = {}
    for binding in report["fact_bindings"]:
        key = _key(binding["input"])
        fact_key = (*key, binding["source"]["fact_id"])
        if fact_key in seen_facts:
            raise ValueError(f"duplicate fact binding: {fact_key}")
        seen_facts.add(fact_key)
        concept = binding["target"]["concept_id"]
        if concept is not None:
            if key in target_concepts and target_concepts[key] != concept:
                raise ValueError(
                    "conflicting target concept identities for one runtime slot"
                )
            target_concepts[key] = concept
        relationship = binding["asserted_relationship"]
        if relationship != "unresolved" and not binding["evidence"]:
            raise ValueError(f"{relationship} binding needs pinned evidence")
        effective, reason = _classification(binding, inputs[key])
        if (
            binding["effective_relationship"] != effective
            or binding["classification_reason"] != reason
        ):
            raise ValueError(
                "effective relationship disagrees with available semantics"
            )
        for side in ("source", "target"):
            _validate_scope(binding[side]["scope"], relationship)
        if relationship != "exact":
            continue
        source_scope = binding["source"]["scope"]
        target = binding["target"]
        target_scope = target["scope"]
        if any(
            source_scope[field] != target_scope[field]
            for field in source_scope
            if field not in {"entity_definition", "universe_definition"}
        ):
            raise ValueError("exact binding needs complete equal statistical scope")
        if binding["transformation"]["kind"] != "identity":
            raise ValueError(
                "asserted exact binding requires an identity transformation"
            )
    if report["blocking_gaps"] != _blocking_gaps(report):
        raise ValueError("blocking gaps disagree with observed capability/evidence")


def _validate_discovery(report: dict, inputs: dict) -> None:
    discovery = {item["entity"]: item for item in report["entity_discovery"]}
    if len(discovery) != len(report["entity_discovery"]) or set(discovery) != set(
        report["mapped_entities"]
    ):
        raise ValueError(
            "per-entity discovery must cover each mapped entity exactly once"
        )
    if not any(item["status"] == "complete" for item in discovery.values()):
        raise ValueError("No mapped entity was successfully enumerated.")
    for entity, item in discovery.items():
        records = [value for value in inputs.values() if value["entity"] == entity]
        if item["status"] == "no_derived_program":
            if item["root_input_count"] is not None or records:
                raise ValueError(
                    "An uncompiled entity cannot claim enumerated inputs or zero count."
                )
        elif item["root_input_count"] is None or item["root_input_count"] != len(
            records
        ):
            raise ValueError("runtime root input count disagrees with entity inventory")
        if any(value["engine_entity"] != item["engine_entity"] for value in records):
            raise ValueError("runtime entity metadata disagrees with input entity")
        canonical = [
            value["canonical_request_name"]
            for value in records
            if value["canonical_request_name"] is not None
        ]
        if len(set(canonical)) != len(canonical):
            raise ValueError("multiple runtime slots share one canonical input address")
    names = [(item["role"], item["name"]) for item in report["fingerprints"]]
    if not any(role != "diagnostic_builder" for role, _ in names) or len(names) != len(
        set(names)
    ):
        raise ValueError("missing or conflicting artifact fingerprints")
    if report["fingerprints"] != sorted(
        report["fingerprints"], key=lambda item: (item["role"], item["name"])
    ):
        raise ValueError("artifact fingerprints are not canonically ordered")


def _validate_scope(scope: dict, relationship: str) -> None:
    if scope["measurement_kind"] == "stock" and scope["reference_instant"] is None:
        raise ValueError("stock scope requires a reference instant")
    if scope["measurement_kind"] != "stock" and scope["reference_instant"] is not None:
        raise ValueError("only a stock scope may carry a reference instant")
    if scope["reference_instant"] is not None:
        instant = scope["reference_instant"]
        try:
            if len(instant) != 10:
                raise ValueError
            date.fromisoformat(instant)
        except ValueError as exc:
            raise ValueError(
                "stock reference instant must be an ISO calendar date"
            ) from exc
    if relationship == "unresolved":
        return
    for field in ("entity_definition", "universe_definition"):
        if scope[field] is None:
            raise ValueError(f"{relationship} scope requires a pinned {field}")
    if relationship == "exact" and any(
        value is None for key, value in scope.items() if key != "reference_instant"
    ):
        raise ValueError("asserted exact binding requires complete statistical scope")
