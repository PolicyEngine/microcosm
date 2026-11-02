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


def _factor_entry(profile, anchor=None):
    declaration, facts, identity = factor_fixture(
        anchor_level=PROFILE_ANCHORS[profile] if anchor is None else anchor
    )
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


# ---------------------------------------------------------------------------
# Summed-band money rows (G8-r2c hunk A)
# ---------------------------------------------------------------------------

SUMMED_MEASUREMENT = "reviewed_measurement"


def band_fact(kind, label, value, *, period=2001, record_set="toy.bands"):
    """One Ledger consumer fact for a native band's count or amount.

    Count and amount facts share one record set and one band dimension, as
    a published table that reports both measures over the same classes does.
    """
    key = f"toy-{kind}-{label}"
    return {
        "aggregate_fact_key": f"ledger.aggregate_fact.v2:{key}",
        "semantic_fact_key": f"ledger.semantic_fact.v2:{key}",
        "legacy_fact_key": f"ledger.fact.v1:{key}",
        "label": f"Toy {kind} in band {label}",
        "lineage": {
            "source_record_id": f"toy.bands.{kind}.{label}",
            "source_cell_keys": [f"ledger.source_cell.v1:{key}"],
            "source_row_keys": [],
        },
        "value": value,
        "period": {"type": "year", "value": period},
        "geography": {"level": "country", "id": "XT", "name": "Toy", "vintage": "v1"},
        "entity": {"name": "person", "role": "person"},
        "observed_measure": {
            "source_name": "toy_office",
            "source_table": "Toy bands",
            "source_measure_id": kind,
            "source_concept": f"toy.{kind}",
            "unit": "people" if kind == "count" else "xts",
        },
        "aggregation": {"method": "sum"},
        "source": {
            "source_name": "toy_office",
            "source_table": "Toy bands",
            "source_file": "toy.csv",
            "url": "https://example.invalid/toy.csv",
            "vintage": "v1",
        },
        "dimensions": {"band": label},
        "dimension_labels": {"band": "Band"},
        "dimension_value_labels": {"band": {label: f"Band {label}"}},
        "universe_constraints": {"domain": "toy"},
        "layout": {
            "record_set_id": record_set,
            "groupby_dimension": "band",
            "groupby_value_id": label,
            "measure_id": kind,
        },
    }


def summed_row(kind, name, members, *, indicator, entity="person"):
    """A parked coarse row over ``members``: ``sum`` counts, ``identity`` money."""
    row = {
        "name": name,
        "entity": entity,
        "measure": "people" if kind == "count" else "toy_income",
        "filter": indicator,
        "period": 2002,
        "family": "toy",
        "category_id": "toy.population",
        "period_match_policy": "latest_not_after",
        "expected_member_count": len(members),
        "ledger_selector": {
            "source_name": "toy_office",
            "source_measure_id": kind,
            "source_concept": f"toy.{kind}",
            "entity_name": "person",
            "period_type": "year",
            "period_value": 2001,
            "dimensions": ["band"],
            "dimension_values": {"band": list(members)},
        },
        "metadata": {
            "contract_target_id": name,
            "reference_value_kind": kind,
            "activation_status": "requires_prerequisites",
            "required_reference_profile": "index",
            "required_measurement_binding": SUMMED_MEASUREMENT,
            "reference_prerequisites": json.dumps(
                ["reference_profile:index", f"measurement:{SUMMED_MEASUREMENT}"]
            ),
            "parking_reason": "Reviewed measurement is required.",
        },
    }
    if kind == "count":
        row["value_operation"] = "sum"
    return row


def summed_profile_fixture(
    groups, *, entity="person", table_id="toy_coarse", anchors=None
):
    """Coarse references, profiles, factor context and a coarse interval table.

    ``groups`` lists the native member labels of each coarse band, in band
    order; band ``b<i>`` sums ``groups[i]``. The table has no nil band, so
    every coarse band is positive and names a summed count/money pair.
    """
    table = interval_table_fixture(
        table_id=table_id,
        entity=entity,
        cuts=tuple(100_000 * (index + 1) for index in range(len(groups) - 1)),
        nil=False,
        companions=True,
    )
    references = []
    for band, members in zip(table["bands"], groups, strict=True):
        for kind in ("count", "money"):
            references.append(
                summed_row(
                    kind,
                    f"toy_{kind}_{band['id']}",
                    members,
                    indicator=band["indicator"],
                    entity=entity,
                )
            )
    identity, factors = None, []
    for profile in PROFILE_ANCHORS:
        identity, entry = _factor_entry(profile, (anchors or {}).get(profile))
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
        "required_measurement_binding": SUMMED_MEASUREMENT,
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
    base = with_target_labels(reference_document(references))
    intervals = {"schema_version": 1, "tables": [table]}
    return base, profiles, context, intervals


def with_target_labels(document):
    """Give every row the Microcosm label a summed or scaled target needs.

    The ledger refuses a multi-fact or transformed target without an explicit
    target label (``ledger_targets.py`` ``_target_label``).
    """
    document["hierarchy"]["target_labels"] = {
        row["metadata"]["contract_target_id"]: f"Toy target {row['name']}"
        for row in document["target_references"]
    }
    return document


def proof_for(producer="toy.measure", calibration="toy.calibrate"):
    """The proof composition returns for one producer upstream of calibration."""
    return {
        SUMMED_MEASUREMENT: {
            "measurement_id": SUMMED_MEASUREMENT,
            "producers": {calibration: producer},
            "columns": [f"person.{INCOME_COLUMN}", f"person.{UNIVERSE_COLUMN}"],
        }
    }
