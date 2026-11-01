"""Toy interval tables and ageing profiles for exercising interval plumbing."""

from __future__ import annotations

import json
from copy import deepcopy

from microcosm.build.reference_factors import resolve_reference_factor
from test_support.microcosm_build.reference_factors import factor_fixture
from test_support.microcosm_build.transport_graph import reference_document

#: Anchor levels against a source level of 100: factors 1.05, 1.1 and 1.15.
PROFILE_ANCHORS = {"low": "105", "central": "110", "high": "115"}
INCOME_COLUMN = "toy_income_cents"
UNIVERSE_COLUMN = "toy_income_universe"


def interval_table_fixture(
    *,
    table_id="toy_income",
    entity="person",
    cuts=(100_000, 1_000_000),
    nil=True,
    closure="right",
    first=0,
    source_period=2001,
    target_period=2002,
    companions=False,
    indicator_prefix="toy_income_band",
):
    """One declared table; companions name ``toy_{count,money}_<band>``."""
    bands = []
    if nil:
        bands.append(
            {"id": "nil", "kind": "nil", "indicator": f"{indicator_prefix}_nil"}
        )
    for index, upper in enumerate((*cuts, None)):
        bands.append(
            {
                "id": f"b{index}",
                "kind": "positive",
                "upper_cent": upper,
                "indicator": f"{indicator_prefix}_b{index}",
            }
        )
    if companions:
        for band in bands:
            band.update(
                count_reference=f"toy_count_{band['id']}",
                amount_reference=f"toy_money_{band['id']}",
            )
    return {
        "id": table_id,
        "entity": entity,
        "currency": "XTS",
        "input_unit": "integer_cent",
        "income_column": INCOME_COLUMN,
        "universe_column": UNIVERSE_COLUMN,
        "source_period": source_period,
        "target_period": target_period,
        "positive_band_closure": closure,
        "nil_band": "present" if nil else "absent",
        "negative_income_policy": "floor_zero",
        "first_lower_cut_cents": first,
        "source_locator": "Toy!A1:A9",
        "diagnostics": {
            "boundary_window_cents": 100,
            "alternative_positive_band_closure": "left"
            if closure == "right"
            else "right",
        },
        "bands": bands,
    }


def interval_bindings(table, *, entity=None):
    """Calibration bindings whose outputs are the table's band indicators."""
    entity = table["entity"] if entity is None else entity
    return {
        "schema_version": 1,
        "required_inputs": {entity: [f"{entity}_id"]},
        "input_domains": {entity: {f"{entity}_id": {"kind": "identity"}}},
        "columns": [
            {
                "entity": entity,
                "name": band["indicator"],
                "interval": {"table": table["id"], "band": band["id"]},
            }
            for band in table["bands"]
        ],
    }


def _factor_entry(profile):
    declaration, facts, identity = factor_fixture(anchor_level=PROFILE_ANCHORS[profile])
    declaration["id"] = f"toy_{profile}"
    factor, receipt = resolve_reference_factor(declaration, facts, identity)
    return identity, {
        "declaration": declaration,
        "operational_factor": factor,
        "receipt": receipt,
    }


def interval_profile_fixture():
    """References, profiles, factor context and intervals for low/central/high.

    Every band has parked count and money companions filtered on its own
    indicator; each profile scales the table with its own factor.
    """
    table = interval_table_fixture(companions=True)
    references = []
    for band in table["bands"]:
        for kind in ("count", "money"):
            name = f"toy_{kind}_{band['id']}"
            references.append(
                {
                    "name": name,
                    "entity": "person",
                    "measure": "people" if kind == "count" else "toy_income",
                    "filter": band["indicator"],
                    "period": 2002,
                    "family": "toy",
                    "category_id": "toy.population",
                    "period_match_policy": "latest_not_after",
                    "ledger_selector": {
                        "aggregate_fact_key": f"toy-{kind}-{band['id']}",
                        "source_name": "toy_office",
                        "source_measure_id": kind,
                        "source_concept": f"toy.{kind}",
                        "entity_name": "person",
                        "period_type": "year",
                        "period_value": 2001,
                        "dimensions": ["band"],
                        "dimension_values": {"band": band["id"]},
                    },
                    "metadata": {
                        "contract_target_id": name,
                        "reference_value_kind": kind,
                        "activation_status": "requires_prerequisites",
                        "required_reference_profile": "index",
                        "required_measurement_binding": "reviewed_measurement",
                        "reference_prerequisites": json.dumps(
                            [
                                "reference_profile:index",
                                "measurement:reviewed_measurement",
                            ]
                        ),
                        "parking_reason": "Reviewed measurement is required.",
                    },
                }
            )
    identity, factors = None, []
    for profile in PROFILE_ANCHORS:
        identity, entry = _factor_entry(profile)
        factors.append(entry)
    context = {"schema_version": 1, "feed_identity": identity, "factors": factors}
    transformation = {
        "operation": "scale_money_and_intervals",
        "reference_names": [
            row["name"] for row in references if row["name"].startswith("toy_money")
        ],
        "count_reference_names": [
            row["name"] for row in references if row["name"].startswith("toy_count")
        ],
        "interval_table_ids": [table["id"]],
        "required_measurement_binding": "reviewed_measurement",
    }
    profiles = {
        "schema_version": 1,
        "reference_resource": "target_references",
        "factor_context_resource": "factor_context",
        "interval_resource": "profile_intervals",
        "dimensions": {
            "index": {
                "default": "central",
                "profiles": {
                    profile: {
                        "transformations": [
                            {**deepcopy(transformation), "factor_id": f"toy_{profile}"}
                        ]
                    }
                    for profile in PROFILE_ANCHORS
                },
            }
        },
    }
    base = reference_document(references)
    intervals = {"schema_version": 1, "tables": [table]}
    return base, profiles, context, intervals
