"""Invented qualified descriptions, routing and support; no model or engine."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_us_current_property_income_sources import (  # noqa: F401
    _detached,
    _values,
    actual,
)

from microcosm.build.us_runtime import current_property_completion_routing as routing
from microcosm.build.us_runtime import current_property_income_sources as sources
from microcosm.frame import Frame, WeightKind, Weights


def test_artifact_declares_diagnostics_without_source_authority():
    output = routing.property_completion_artifact_output()
    assert output.name == "completion_routing"
    assert output.type == routing.PROPERTY_COMPLETION_TYPE
    assert routing.PROTOCOL == "microcosm.us.property-completion-routing.v1"


def _rebuild(qualified, edit):
    values = list(_detached(_values(qualified)))
    edit(values)
    # Keep the existing descriptive projection binding coherent after a pure
    # fixture mutation. This does not issue a preparation or qualify new bytes.
    projection = (
        values[1].anchors.reset_index().to_json(orient="table", index=False).encode()
    )
    values[1] = replace(
        values[1],
        projection=projection,
        evidence={
            **values[1].evidence,
            "projection_sha256": hashlib.sha256(projection).hexdigest(),
        },
    )
    return sources._compose(tuple(values), qualified.origin_document)


def _asec_id(qualified, native=107):
    return qualified.origins.index[
        qualified.origins.source.eq("asec")
        & qualified.origins.native_person_id.eq(native)
    ][0]


def _set_amount(table, original, field, amount, status):
    table.loc[original, field + "_published_amount"] = amount
    table.loc[original, field + "_literal_status"] = (
        "missing" if amount is None else "in_printed_range"
    )
    table.loc[original, field + "_reporting_status"] = status
    known = status in ("known_receipt", "observed_zero_component", "known_nonreceipt")
    table.loc[original, field + "_amount_known"] = known
    table.loc[original, field + "_amount"] = amount if known else pd.NA


def test_actual_source_roster_preserves_independent_knownness_and_all_inputs(actual):  # noqa: F811
    live, qualified, _ = actual
    before = sources.property_income_sources_seal(qualified)
    frame_before = sources._frame_seal(live.clone_population.frame)
    result = routing.build_property_completion_routing(
        qualified, live.clone_population.frame
    )
    assert result.person.index.equals(qualified.origins.index)
    assert len(result.components) == 2 * len(result.person)
    assert len(result.clones) == 2 * len(result.person)
    original = _asec_id(qualified)
    assert result.person.loc[original, "completion_route"] == "carry_known_components"
    assert result.reasons.loc[original, "joint:survivor_possible_property"]
    pair = result.components.loc[result.components.original_person_id.eq(original)]
    assert pair.amount.eq(500).all() and pair.known.all()
    assert pair.origin.eq("observed").all()
    acs = result.person.source.eq("acs")
    eligible = acs & result.person.anchor_known
    assert eligible.any()
    assert (
        result.person.loc[eligible, "completion_route"]
        .eq("existing_acs_anchor_decomposition")
        .all()
    )
    assert result.person.loc[eligible, "anchor_origin"].eq("observed").all()
    assert (
        result.components.loc[
            result.components.original_person_id.isin(result.person.index[acs]),
            "origin",
        ]
        .eq("unresolved")
        .all()
    )
    assert not result.components.origin.eq("modeled").any()
    payload = json.loads(result.payload)
    assert payload["source_authority"] is False
    assert payload["complete_parent_authority"] is False
    assert payload["amounts_assigned"] is False
    assert sources.property_income_sources_seal(qualified) == before
    assert sources._frame_seal(live.clone_population.frame) == frame_before
    assert (
        routing.build_property_completion_routing(
            qualified, live.clone_population.frame
        ).payload
        == result.payload
    )


@pytest.mark.parametrize("field", ["TRDINT_VAL", "DIV_VAL"])
def test_missing_one_component_preserves_known_other(actual, field):  # noqa: F811
    live, qualified, _ = actual
    original = _asec_id(qualified)

    def edit(values):
        table = values[2 if field == "TRDINT_VAL" else 4].person
        _set_amount(table, original, field, None, "missing_amount")

    changed = _rebuild(qualified, edit)
    result = routing.build_property_completion_routing(
        changed, live.clone_population.frame
    )
    assert (
        result.person.loc[original, "completion_route"]
        == "asec_component_completion_review"
    )
    rows = result.components.loc[result.components.original_person_id.eq(original)]
    assert rows.known.sum() == 1
    assert rows.loc[rows.known, "amount"].iloc[0] == 500.0
    assert rows.loc[~rows.known, "amount"].isna().all()
    assert rows.loc[~rows.known, "origin"].eq("unresolved").all()


@pytest.mark.parametrize(
    "status,amount,expected",
    [
        ("missing_source_amount", None, "acs_anchor_completion_review"),
        ("malformed_source_amount", None, "source_review_required"),
        ("outside_published_domain", None, "source_review_required"),
        ("invalid_adjustment", None, "source_review_required"),
        ("observed", -100.0, "existing_acs_anchor_decomposition"),
        ("observed", 0.0, "existing_acs_anchor_decomposition"),
    ],
)
def test_acs_anchor_route_is_not_component_observation(
    actual,  # noqa: F811
    status,
    amount,
    expected,
):
    live, qualified, _ = actual
    original = qualified.recipient_columns.index[0]

    def edit(values):
        anchors = values[1].anchors
        anchors.loc[original, "property_income_status"] = status
        anchors.loc[original, "property_income_amount"] = amount
        anchors.loc[original, "property_income_known"] = status == "observed"

    result = routing.build_property_completion_routing(
        _rebuild(qualified, edit), live.clone_population.frame
    )
    assert result.person.loc[original, "completion_route"] == expected
    rows = result.components.loc[result.components.original_person_id.eq(original)]
    assert not rows.known.any() and rows.amount.isna().all()


@pytest.mark.parametrize(
    "status",
    [
        "outside_universe_blank",
        "outside_universe_observation",
        "malformed_source_amount",
    ],
)
def test_under15_keeps_all_source_reasons_without_zero(actual, status):  # noqa: F811
    live, qualified, _ = actual
    anchors = qualified.acs_anchor_values.anchors
    original = anchors.index[~anchors.property_income_in_income_universe][0]

    def edit(values):
        values[1].anchors.loc[original, "property_income_status"] = status
        if status == "outside_universe_observation":
            values[1].anchors.loc[original, "INTP"] = "100"

    result = routing.build_property_completion_routing(
        _rebuild(qualified, edit), live.clone_population.frame
    )
    assert (
        result.person.loc[original, "completion_route"]
        == "unsupported_under15_measurement"
    )
    assert result.person.loc[original, "anchor_status"] == status
    assert result.reasons.loc[original, "under15"]
    assert result.reasons.loc[original, "source_error"] == (
        status == "malformed_source_amount"
    )
    assert result.reasons.loc[original, "positive_outside_universe"] == (
        status == "outside_universe_observation"
    )
    rows = result.components.loc[result.components.original_person_id.eq(original)]
    assert not rows.known.any() and rows.amount.isna().all()


@pytest.mark.parametrize(
    "status",
    [
        "invalid_amount_literal",
        "contradictory_no_nonzero",
        "unrecognized_receipt_literal",
    ],
)
def test_asec_source_error_precedes_missingness_but_keeps_other_known(actual, status):  # noqa: F811
    live, qualified, _ = actual
    original = _asec_id(qualified)

    def edit(values):
        _set_amount(values[4].person, original, "DIV_VAL", 500.0, status)

    result = routing.build_property_completion_routing(
        _rebuild(qualified, edit), live.clone_population.frame
    )
    assert result.person.loc[original, "completion_route"] == "source_review_required"
    assert result.person.loc[original, "ordinary_interest_known"]
    assert not result.person.loc[original, "dividends_known"]


def test_nonreceipt_zero_and_observed_component_zero_have_distinct_origins(actual):  # noqa: F811
    live, qualified, _ = actual
    original = _asec_id(qualified)

    def edit(values):
        _set_amount(
            values[2].person, original, "TRDINT_VAL", 0.0, "observed_zero_component"
        )
        _set_amount(values[4].person, original, "DIV_VAL", 0.0, "known_nonreceipt")
        values[4].person.loc[original, "DIV_YN_code"] = 2

    result = routing.build_property_completion_routing(
        _rebuild(qualified, edit), live.clone_population.frame
    )
    rows = result.components.loc[
        result.components.original_person_id.eq(original)
    ].set_index("component")
    assert rows.loc["property_ordinary_interest", "origin"] == "observed"
    assert rows.loc["property_dividends", "origin"] == "derived"
    assert rows.loc["property_dividends", "zero_basis"] == "qualified_known_nonreceipt"
    assert rows.amount.eq(0).all()


def test_allocation_flags_do_not_change_completion_route(actual):  # noqa: F811
    live, qualified, _ = actual
    original = _asec_id(qualified)

    def edit(values):
        values[4].person.loc[original, "I_DIVVAL_code"] = 1
        values[4].person.loc[original, "I_DIVYN_code"] = 8
        values[4].person.loc[original, "I_DIVYN_codebook_status"] = (
            "header_reference_conflict"
        )

    result = routing.build_property_completion_routing(
        _rebuild(qualified, edit), live.clone_population.frame
    )
    assert result.person.loc[original, "completion_route"] == "carry_known_components"


def test_summary_counts_originals_and_union_households_not_clones(actual):  # noqa: F811
    live, qualified, _ = actual
    result = routing.build_property_completion_routing(
        qualified, live.clone_population.frame
    )
    row = result.summary.loc["all"]
    frame = qualified.source_frame
    assert row.person_count == len(result.person)
    assert row.household_count == frame.n("household")
    assert row.design_weighted_person_mass == sum(
        frame.resolve_weights("person").values
    )
    assert row.union_household_design_mass == sum(frame.weights_for("household").values)


@pytest.mark.parametrize(
    "fault", ["missing_clone", "duplicate_role", "native_id", "source_label"]
)
def test_clone_pair_identity_refuses(actual, fault):  # noqa: F811
    live, qualified, _ = actual
    frame = _detached(live.clone_population.frame)
    people = frame.person
    p = routing.provenance
    if fault == "missing_clone":
        keep = np.ones(len(people), dtype=bool)
        keep[0] = False
        frame = frame.select(keep)
    elif fault == "duplicate_role":
        people[p.support_clone_index_column("person")] = 0
    elif fault == "native_id":
        people.loc[people.index[0], p.spine_source_id_column("person")] += 1
    else:
        name = p.support_channel_column("person")
        people[name] = people[name].astype(str)
        people.loc[people.index[0], name] = "other"
    with pytest.raises(ValueError, match="PROPERTY_COMPLETION"):
        routing.build_property_completion_routing(qualified, frame)


@pytest.mark.parametrize(
    "fault", ["source_native", "source_axis", "knownness", "unknown_status"]
)
def test_description_identity_and_knownness_refuse(actual, fault):  # noqa: F811
    live, qualified, _ = actual
    value = _detached(qualified)
    original = _asec_id(value)
    if fault == "source_native":
        value.asec_dividend_values.person.loc[original, "native_person_id"] += 1
    elif fault == "source_axis":
        value.asec_dividend_values.person.index = (
            value.asec_dividend_values.person.index[::-1]
        )
    elif fault == "knownness":
        value.asec_dividend_values.person.loc[original, "DIV_VAL_amount_known"] = False
    else:
        value.asec_dividend_values.person.loc[original, "DIV_VAL_reporting_status"] = (
            "new_unreviewed_status"
        )
    with pytest.raises(ValueError, match="PROPERTY_COMPLETION"):
        routing.build_property_completion_routing(value, live.clone_population.frame)


def test_design_kind_required_and_zero_design_mass_is_not_missing(actual):  # noqa: F811
    live, qualified, _ = actual
    for kind, succeeds in ((WeightKind.IMPORTANCE, False), (WeightKind.DESIGN, True)):
        value = _detached(qualified)
        frame = value.source_frame
        weights = np.ones(frame.n("household"))
        weights[0] = 0
        altered = Frame(
            {e: frame.table(e).copy(deep=True) for e in frame.entities},
            frame.schema,
            {"household": Weights(weights, kind)},
            frame.strata.copy(deep=True),
        )
        object.__setattr__(value, "source_frame", altered)
        if succeeds:
            result = routing.build_property_completion_routing(
                value, live.clone_population.frame
            )
            assert result.summary.loc["all", "union_household_design_mass"] == sum(
                weights
            )
            assert result.summary.loc["all", "person_count"] == len(value.origins)
            assert result.person.original_household_design_weight.eq(0).any()
        else:
            with pytest.raises(ValueError, match="PROPERTY_COMPLETION.*DESIGN"):
                routing.build_property_completion_routing(
                    value, live.clone_population.frame
                )


def test_large_native_ids_remain_exact_in_tables_and_json(actual):  # noqa: F811
    live, qualified, _ = actual
    value, clone = _detached(qualified), _detached(live.clone_population.frame)
    original = _asec_id(value)
    native = 2**53 + 17
    value.origins.loc[original, "native_person_id"] = native
    value.shared_predictors.origins.loc[original, "native_person_id"] = native
    document = json.loads(value.origin_document)
    column_names = document["persons"]["columns"]
    for row in document["persons"]["rows"]:
        if row[column_names.index("person_id")] == int(original):
            row[column_names.index("selected_receiving_person_id")] = native
    object.__setattr__(
        value, "origin_document", sources.shared.codec.encode_json(document)
    )
    for table in (
        value.asec_interest_values.person,
        value.asec_routing_values.person,
        value.asec_dividend_values.person,
        value.donor_basis.person,
    ):
        table.loc[original, "native_person_id"] = native
    p = routing.provenance
    for frame in (value.source_frame, value.shared_predictors.source_frame):
        frame.person.loc[
            frame.person.person_id.eq(original), p.spine_source_id_column("person")
        ] = native
    clone.person.loc[
        clone.person[p.support_source_id_column("person")].eq(original),
        p.spine_source_id_column("person"),
    ] = native
    result = routing.build_property_completion_routing(value, clone)
    assert result.person.loc[original, "native_person_id"] == native
    payload = json.loads(result.payload)
    table = payload["tables"]["person"]
    position = table["columns"].index("native_person_id")
    row = table["index"].index(int(original))
    assert type(table["rows"][row][position]) is int
    assert table["rows"][row][position] == native
