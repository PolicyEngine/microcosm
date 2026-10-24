"""Authenticated toy feed and companion references for transport factor checks."""

import json
from copy import deepcopy

from microcosm.build.ledger_artifact import (
    chronicle_feed_identity,
    load_ledger_consumer_artifact,
)
from microcosm.build.reference_factors import resolve_reference_factor
from microcosm.build.reference_profiles import (
    reference_document_sha256,
    resolve_reference_profiles,
)
from test_support.microcosm_build.chronicle_feed import write_pinned_artifact
from test_support.microcosm_build.reference_factors import factor_fixture
from test_support.microcosm_build.transport_graph import reference_document


def factor_transport_fixture(
    tmp_path, *, forged=False, activate=False, manifest_commit=False
):
    """Build evidence from loaded lexical tokens; optionally forge its source claim."""
    _, facts, _ = factor_fixture()
    references = []
    for kind, value in (("count", 3), ("money", 100)):
        key = f"toy-{kind}"
        selector = {
            "aggregate_fact_key": key,
            "source_name": "toy_office",
            "source_measure_id": kind,
            "source_concept": f"toy.{kind}",
            "entity_name": "person",
            "period_type": "year",
            "period_value": 2001,
            "dimensions": ["band"],
            "dimension_values": {"band": "one"},
        }
        facts.append(
            {
                "aggregate_fact_key": key,
                "label": f"Toy published {kind}",
                "value": value,
                "assertion": "observation",
                "period": {"type": "year", "value": 2001},
                "geography": {
                    "level": "country",
                    "id": "XX",
                    "name": "Toy country",
                    "vintage": "toy",
                },
                "entity": {"name": "person"},
                "dimensions": {"band": "one"},
                "dimension_labels": {"band": "Band"},
                "dimension_value_labels": {"band": {"one": "One"}},
                "aggregation": {"method": "sum"},
                "source": {"source_name": "toy_office"},
                "observed_measure": {
                    "source_measure_id": kind,
                    "source_concept": f"toy.{kind}",
                    "unit": kind,
                },
            }
        )
        references.append(
            {
                "name": key,
                "entity": "person",
                "measure": "people" if kind == "count" else "employment_income",
                "family": "toy",
                "category_id": "toy.population",
                "period": 2002,
                "period_match_policy": "latest_not_after",
                "ledger_selector": selector,
                "metadata": {
                    "contract_target_id": key,
                    "reference_value_kind": kind,
                    "activation_status": "requires_prerequisites",
                    "required_reference_profile": "index",
                    "required_measurement_binding": "reviewed_measurement",
                    "reference_prerequisites": json.dumps(
                        ["reference_profile:index", "measurement:reviewed_measurement"]
                    ),
                    "parking_reason": "Reviewed measurement is required.",
                },
            }
        )
    path, pin = write_pinned_artifact(
        tmp_path / "factor-artifact", json.loads(json.dumps(facts, default=float))
    )
    pin["source_commit"] = "c" * 40
    if manifest_commit:
        import hashlib

        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["source_commit"] = pin["source_commit"]
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        pin["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    artifact = load_ledger_consumer_artifact(path, feed_pin=pin, lexical_floats=True)
    identity = chronicle_feed_identity(pin)
    declaration, _, _ = factor_fixture(feed_identity=identity)
    evidence_facts = artifact.facts
    if forged:
        # This evidence is internally consistent but falsely claims the original pin.
        evidence_facts = deepcopy(artifact.facts)
        next(row for row in evidence_facts if row["aggregate_fact_key"] == "target-0")[
            "value"
        ] = 120
        declaration["expected_operational_factor"]["numerator"] = 1200000
    factor, receipt = resolve_reference_factor(declaration, evidence_facts, identity)
    context = {
        "schema_version": 1,
        "feed_identity": identity,
        "factors": [
            {
                "declaration": declaration,
                "operational_factor": factor,
                "receipt": receipt,
            }
        ],
    }
    transformation = {
        "operation": "scale_money_and_intervals",
        "factor_id": declaration["id"],
        "reference_names": ["toy-money"],
        "count_reference_names": ["toy-count"],
        "interval_table_ids": [],
        "required_measurement_binding": "reviewed_measurement",
    }
    profiles = {
        "schema_version": 1,
        "reference_resource": "target_references",
        "factor_context_resource": "factor_context",
        "dimensions": {
            "index": {
                "default": "central",
                "profiles": {
                    name: {"transformations": [deepcopy(transformation)]}
                    for name in ("low", "central", "high")
                },
            }
        },
    }
    base = reference_document(references)
    base["hierarchy"]["target_labels"] = {
        row["name"]: f"Aged toy {row['name']}" for row in references
    }
    document, _, resolution = resolve_reference_profiles(
        base, profiles, {}, factor_context=context
    )
    if activate:
        # Test-only future measurement closure lets the compiler test its own replay
        # and arithmetic independently of this lane's parked country declarations.
        for row in document["target_references"]:
            row["metadata"]["activation_status"] = "active"
        resolution["effective_references_sha256"] = reference_document_sha256(document)
        resolution.pop("receipt_sha256")
        resolution["receipt_sha256"] = reference_document_sha256(resolution)
    spec = {
        "country": "xx",
        "resources": {
            "target_references": base,
            "reference_profiles": profiles,
            "factor_context": context,
            "scenarios": {"scenarios": [{"id": "base", "knobs": {}}]},
        },
    }
    return path, pin, document, resolution, spec
