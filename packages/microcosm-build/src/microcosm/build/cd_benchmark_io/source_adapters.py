"""Two bounded CD summaries under an external actual-verifier control.

This invented-only execution contract does not issue scientific or action
approval. Both definition slots and every other required CD binding stay open.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from microcosm.build.cd_benchmark.canonical import (
    canonical_bytes,
    digest,
    is_sha256_hex,
)

from .artifacts import FilePin, ProducerEvidence, document, fields, require

UNIVERSE = "universe_classification_artifact_id_and_sha256"
GEOGRAPHY = "geography_assignment_artifact_id_and_sha256"
SLOTS = (UNIVERSE, GEOGRAPHY)
DEFINITIONS = (
    "universe_definition_id_and_sha256_per_block_per_arm",
    "geography_definition_id_and_sha256",
)
SCOPE = "invented_source_evidence_only"
CONTROL_SCHEMA = "microcosm.cd_source_verifier_control.v1"
SUMMARY_SCHEMA = "microcosm.cd_source_slot_summary.v1"
COMPLETION_SCHEMA = "microcosm.cd_source_verifier_completion.v1"
SEAL_SCHEMA = "microcosm.cd_source_verifier_seal.v1"


def _pin_fields(pin):
    return {"bytes": pin.size, "sha256": pin.sha256}


def _write(path, value):
    raw = canonical_bytes(value)
    pin = FilePin(Path(path), len(raw), digest(raw))
    with pin.path.open("xb") as handle:
        handle.write(raw)
    require(document(pin) == value, "source_writer_readback")
    return pin


def write_verifier_control(
    root, *, result, run_id, proposal_sha256, verifier_implementation_sha256
):
    """Seal caller's actual verifier result; returned pins are NOT approvals.

    Production callers must obtain their expected pins independently of the
    candidate bundle. Tests exercise the real materialized verifier beforehand.
    """
    require(
        type(run_id) is str
        and run_id
        and is_sha256_hex(proposal_sha256)
        and is_sha256_hex(verifier_implementation_sha256),
        "source_execution_identity",
    )
    fields(result, {"identity", "slots"}, "source_verifier_result_fields")
    require(set(result["slots"]) == set(SLOTS), "source_verifier_result_slots")
    for value in result["slots"].values():
        fields(value, {"dependencies", "evidence"}, "source_verifier_slot_fields")
    root = Path(root)
    root.mkdir()
    control = _write(
        root / "verifier-control.json",
        {
            "schema": CONTROL_SCHEMA,
            "scope": SCOPE,
            "status": "PASS",
            "run_id": run_id,
            "proposal_sha256": proposal_sha256,
            "verifier_implementation_sha256": verifier_implementation_sha256,
            "definition_slots_unresolved": list(DEFINITIONS),
            **result,
        },
    )
    completion = _write(
        root / "completion.json",
        {
            "schema": COMPLETION_SCHEMA,
            "control": _pin_fields(control),
            "run_id": run_id,
            "proposal_sha256": proposal_sha256,
            "verifier_implementation_sha256": verifier_implementation_sha256,
        },
    )
    seal = _write(
        root / "completion-seal.json",
        {
            "schema": SEAL_SCHEMA,
            "completion": _pin_fields(completion),
            "run_id": run_id,
            "proposal_sha256": proposal_sha256,
        },
    )
    return control, completion, seal


def write_slot_summaries(root, *, control_pin):
    """Small canonical dicts only; private rows stay in the graph store."""
    control = document(control_pin)
    require(
        control.get("schema") == CONTROL_SCHEMA and control.get("scope") == SCOPE,
        "source_control_schema",
    )
    root = Path(root)
    root.mkdir()
    result = {}
    for slot, item in control["slots"].items():
        require(slot in SLOTS, "source_slot_name")
        result[slot] = _write(
            root / ("universe.json" if slot == UNIVERSE else "geography.json"),
            {
                "schema": SUMMARY_SCHEMA,
                "slot": slot,
                "scope": SCOPE,
                "identity": control["identity"],
                "dependencies": item["dependencies"],
                "evidence": item["evidence"],
                "external_control_sha256": control_pin.sha256,
                "definition_admission": "unresolved",
            },
        )
    return result


def producer_adapters(
    *,
    control_pin=None,
    completion_pin=None,
    seal_pin=None,
    expected_run_id=None,
    expected_proposal_sha256=None,
    expected_verifier_implementation_sha256=None,
    candidate_files: Mapping[str, FilePin],
):
    """Close over externally admitted bytes; bundle documents cannot be controls."""
    require(
        all(type(pin) is FilePin for pin in (control_pin, completion_pin, seal_pin)),
        "source_external_control_required",
    )
    require(
        type(expected_run_id) is str
        and expected_run_id
        and is_sha256_hex(expected_proposal_sha256)
        and is_sha256_hex(expected_verifier_implementation_sha256),
        "source_external_execution_required",
    )
    external = (control_pin, completion_pin, seal_pin)
    require(len({p.path for p in external}) == 3, "source_control_paths")
    for pin in candidate_files.values():
        require(type(pin) is FilePin, "source_candidate_pin")
        require(
            all(
                pin.path != other.path and pin.sha256 != other.sha256
                for other in external
            ),
            "source_control_in_candidate_bundle",
        )
    # document verifies/hash-checks the exact bytes that it decodes, once. No
    # earlier sha(path) followed by a second unpinned json read is permitted.
    control, completion, seal = (document(pin) for pin in external)
    fields(
        control,
        {
            "schema",
            "scope",
            "status",
            "run_id",
            "proposal_sha256",
            "verifier_implementation_sha256",
            "definition_slots_unresolved",
            "identity",
            "slots",
        },
        "source_control_fields",
    )
    fields(
        completion,
        {
            "schema",
            "control",
            "run_id",
            "proposal_sha256",
            "verifier_implementation_sha256",
        },
        "source_completion_fields",
    )
    fields(
        seal,
        {"schema", "completion", "run_id", "proposal_sha256"},
        "source_seal_fields",
    )
    require(
        control["schema"] == CONTROL_SCHEMA
        and completion["schema"] == COMPLETION_SCHEMA
        and seal["schema"] == SEAL_SCHEMA
        and control["scope"] == SCOPE
        and control["status"] == "PASS"
        and control["definition_slots_unresolved"] == list(DEFINITIONS)
        and set(control["slots"]) == set(SLOTS),
        "source_control_schema",
    )
    require(
        all(
            item["run_id"] == expected_run_id
            and item["proposal_sha256"] == expected_proposal_sha256
            for item in (control, completion, seal)
        ),
        "source_control_execution",
    )
    require(
        control["verifier_implementation_sha256"]
        == completion["verifier_implementation_sha256"]
        == expected_verifier_implementation_sha256,
        "source_verifier_implementation",
    )
    require(
        canonical_bytes(completion["control"])
        == canonical_bytes(_pin_fields(control_pin))
        and canonical_bytes(seal["completion"])
        == canonical_bytes(_pin_fields(completion_pin)),
        "source_control_seal",
    )
    frozen = canonical_bytes(control)
    admitted = {}
    for slot in SLOTS:
        slot_value = control["slots"][slot]
        fields(slot_value, {"dependencies", "evidence"}, "source_control_slot_fields")
        expected = canonical_bytes(
            {
                "schema": SUMMARY_SCHEMA,
                "slot": slot,
                "scope": SCOPE,
                "identity": control["identity"],
                "dependencies": slot_value["dependencies"],
                "evidence": slot_value["evidence"],
                "external_control_sha256": control_pin.sha256,
                "definition_admission": "unresolved",
            }
        )
        dependency_bytes = canonical_bytes(slot_value["dependencies"])

        def adapter(
            actual_slot,
            value,
            artifacts,
            *,
            _slot=slot,
            _expected=expected,
            _dependencies=dependency_bytes,
        ):
            require(actual_slot == _slot, "source_adapter_slot")
            # Recheck external controls at the point of use. Any input mutation
            # fails its original external pin; candidate artifacts remain inert.
            require(
                canonical_bytes(document(control_pin)) == frozen,
                "source_control_changed",
            )
            document(completion_pin)
            document(seal_pin)
            require(
                all(
                    digest(canonical_bytes(doc)) not in {p.sha256 for p in external}
                    for doc in artifacts.values()
                ),
                "source_control_echo",
            )
            require(
                canonical_bytes(value) == _expected, "source_actual_verifier_mismatch"
            )
            return ProducerEvidence(
                MappingProxyType(dict(control["identity"])),
                MappingProxyType(dict(slot_value_for(_dependencies))),
            )

        admitted[slot] = adapter
    return MappingProxyType(admitted)


def slot_value_for(raw):
    from microcosm.build.cd_benchmark.canonical import strict_load

    return strict_load(raw)
