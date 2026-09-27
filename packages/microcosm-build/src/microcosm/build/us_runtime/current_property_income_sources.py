"""Qualify original survey property branches without fitting or issuing authority."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from microcosm.fit import model_input
from microcosm.frame import WeightKind
from microcosm.graph import Population

from . import current_acs_income_anchor_source as acs
from . import current_asec_dividend_source as dividend
from . import current_asec_income_routing_source as routing
from . import current_asec_interest_source as interest
from . import current_survey_predictors as shared
from . import graph_full_puf_enrichment as physical
from .current_asec_property_basis import AsecPropertyBasis, build_asec_property_basis
from .property_income_constants import PROPERTY_COMPONENTS, PROPERTY_REPORTED_TOTAL

PROTOCOL = "microcosm.us.current-property-income-sources.v1"


def require(condition, reason):
    if not condition:
        raise ValueError("PROPERTY_INCOME_SOURCES_" + reason)


@dataclass(frozen=True)
class QualifiedPropertyIncomeSources:
    """Descriptive values only; the caller must retain and requalify actual owners."""

    shared_predictors: shared.QualifiedSurveyPredictors
    acs_anchor_values: acs.QualifiedAcsIncomeAnchors
    asec_interest_values: interest.CurrentAsecInterestValues
    asec_routing_values: routing.CurrentAsecIncomeRoutingValues
    asec_dividend_values: dividend.CurrentAsecDividendValues
    donor_basis: AsecPropertyBasis
    source_frame: object
    donor_frame: object | None
    donor_columns: pd.DataFrame
    recipient_frame: object | None
    recipient_columns: pd.DataFrame
    recipient_matrix: bytes | None
    origins: pd.DataFrame
    origin_document: bytes
    recipient_diagnostics: pd.DataFrame
    projection: bytes
    evidence: dict


def _routing_seal(value):
    require(type(value) is routing.CurrentAsecIncomeRoutingValues, "ROUTING_TYPE")
    # The legacy routing qualifier exposes no physical-seal helper. Preserve
    # its complete detached descriptions, including nullable backing storage.
    return (
        physical._table_stamp(value.person),
        physical._table_stamp(value.asec_literals),
        shared.codec.encode_json(value.evidence),
    )


def _source_seals(values):
    predictor, anchors, interest_values, routing_values, dividend_values = values
    return (
        shared._qualified_seal(predictor),
        acs.income_anchor_seal(anchors),
        interest.interest_values_seal(interest_values),
        _routing_seal(routing_values),
        dividend.dividend_values_seal(dividend_values),
    )


def _frame_seal(frame):
    return (
        None
        if frame is None
        else physical._population_stamp(
            Population.from_frame(frame, "property_income_sources.description")
        )
    )


def property_income_sources_seal(value):
    """Complete descriptive physical seal, not source authentication or a token."""
    require(type(value) is QualifiedPropertyIncomeSources, "VALUES_TYPE")
    return (
        _source_seals(
            (
                value.shared_predictors,
                value.acs_anchor_values,
                value.asec_interest_values,
                value.asec_routing_values,
                value.asec_dividend_values,
            )
        ),
        tuple(
            _frame_seal(frame)
            for frame in (value.source_frame, value.donor_frame, value.recipient_frame)
        ),
        tuple(
            physical._table_stamp(table)
            for table in (
                value.donor_basis.person,
                value.donor_basis.provenance,
                value.donor_basis.exclusions,
                value.donor_basis.summary,
                value.donor_columns,
                value.recipient_columns,
                value.origins,
                value.recipient_diagnostics,
            )
        ),
        value.recipient_matrix,
        value.origin_document,
        value.projection,
        shared.codec.encode_json(value.evidence),
    )


def _same_axis(frame, origins, reason):
    require(
        frame.index.equals(origins.index)
        and frame.index.name == origins.index.name
        and frame.native_person_id.equals(origins.native_person_id),
        reason,
    )


def _compose(values, origin_document):
    """Pure assembly of already-qualified values; no source authority from args."""
    predictor, anchors, interest_values, routing_values, dividend_values = values
    frame = predictor.source_frame
    people = frame.person
    origins = predictor.origins.copy(deep=True)
    ids = pd.Index(people.person_id.to_numpy(copy=True), name="person_id")
    require(
        ids.equals(origins.index)
        and np.array_equal(origins.person_id.to_numpy(), ids.to_numpy())
        and set(origins.source) == {"asec", "acs"},
        "ORIGIN_AXIS",
    )
    raw_origins = json.loads(origin_document)["persons"]
    full_origins = pd.DataFrame(raw_origins["rows"], columns=raw_origins["columns"])
    require(
        np.array_equal(full_origins.person_id.to_numpy(), ids.to_numpy())
        and np.array_equal(full_origins.source.to_numpy(), origins.source.to_numpy())
        and np.array_equal(
            full_origins.selected_receiving_person_id.to_numpy(),
            origins.native_person_id.to_numpy(),
        ),
        "FULL_ORIGIN_AXIS",
    )
    donor_origins = origins.loc[origins.source.eq("asec")]
    recipient_origins = origins.loc[origins.source.eq("acs")]
    for selected in (
        interest_values.person,
        routing_values.person,
        dividend_values.person,
    ):
        _same_axis(selected, donor_origins, "ASEC_SOURCE_AXIS")
    _same_axis(anchors.anchors, recipient_origins, "ACS_SOURCE_AXIS")
    require(
        predictor.donor_columns.index.equals(donor_origins.index)
        and np.array_equal(
            predictor.donor_frame.person.person_id.to_numpy(),
            donor_origins.index.to_numpy(),
        ),
        "DONOR_FEATURE_AXIS",
    )
    donor_source = predictor.donor_frame
    require(
        donor_source.weights_for("household").kind is WeightKind.DESIGN
        and donor_source.resolve_weights("person").kind is WeightKind.DESIGN,
        "ORIGINAL_DESIGN_WEIGHTS",
    )
    membership = pd.Series(
        donor_source.person.person_household_id.to_numpy(copy=True),
        index=donor_origins.index,
        name="household_id",
    )
    household_ids = pd.Index(
        donor_source.table("household").household_id.to_numpy(copy=True),
        name="household_id",
    )
    design_weights = pd.Series(
        donor_source.weights_for("household").values.copy(), index=household_ids
    )
    basis = build_asec_property_basis(
        interest=interest_values.person,
        income_routing=routing_values.person,
        dividend=dividend_values.person,
        original_household_membership=membership,
        original_household_design_weights=design_weights,
    )
    require(
        np.array_equal(
            basis.person.original_household_design_weight.to_numpy(),
            donor_source.resolve_weights("person").values,
        ),
        "PERSON_DESIGN_WEIGHT_MAPPING",
    )
    features = shared.feature_columns(predictor.demographic_conditioning)
    donor_columns = predictor.donor_columns.loc[:, list(features)].copy(deep=True)
    for name in (PROPERTY_REPORTED_TOTAL, *PROPERTY_COMPONENTS):
        donor_columns[name] = basis.person[name].to_numpy(copy=True)
    donor_mask = basis.person.joint_component_fit_eligible.to_numpy(dtype=bool)
    donor_columns = donor_columns.loc[donor_mask].copy(deep=True)
    donor_frame = donor_source.select(donor_mask) if donor_mask.any() else None
    decoded = model_input.decode_recipient_matrix(predictor.matrix)
    require(
        decoded.entity == "person"
        and tuple(decoded.features.columns) == features
        and decoded.features.index.equals(recipient_origins.index),
        "RECIPIENT_FEATURE_AXIS",
    )
    original_anchors = anchors.anchors
    diagnostics = original_anchors.copy(deep=True)
    adult = original_anchors.property_income_in_income_universe.to_numpy(dtype=bool)
    known = original_anchors.property_income_known.to_numpy(dtype=bool)
    amounts = original_anchors.property_income_amount.to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    require(np.array_equal(known, np.isfinite(amounts)), "ACS_ANCHOR_KNOWNNESS")
    diagnostics["excluded_under15"] = ~adult
    diagnostics["excluded_unknown_anchor"] = ~known
    diagnostics["eligible_recipient"] = adult & known
    recipient_mask = adult & known
    recipient_columns = decoded.features.copy(deep=True)
    recipient_columns[PROPERTY_REPORTED_TOTAL] = amounts
    recipient_columns = recipient_columns.loc[recipient_mask].copy(deep=True)
    frame_mask = np.zeros(len(frame.person), dtype=bool)
    frame_mask[np.flatnonzero(origins.source.eq("acs"))] = recipient_mask
    recipient_frame = frame.select(frame_mask) if frame_mask.any() else None
    matrix = (
        model_input.encode_recipient_matrix(
            recipient_columns,
            entity="person",
            entity_ids=recipient_columns.index.to_numpy(dtype="<i8", copy=True),
        )
        if len(recipient_columns)
        else None
    )
    for columns in (donor_columns, recipient_columns):
        require(
            all(dtype == np.dtype("float64") for dtype in columns.dtypes)
            and np.isfinite(columns.to_numpy()).all(),
            "BRANCH_FINITE_FLOAT64",
        )
    evidence = {
        "protocol": PROTOCOL,
        "preparation_sha256": predictor.evidence["preparation_sha256"],
        "shared_predictors": list(features),
        "property_predictors": [*features, PROPERTY_REPORTED_TOTAL],
        "components": list(PROPERTY_COMPONENTS),
        "original_persons": len(origins),
        "asec_selected_persons": len(donor_origins),
        "asec_joint_eligible_persons": len(donor_columns),
        "acs_selected_persons": len(recipient_origins),
        "acs_eligible_persons": len(recipient_columns),
        "donor_weight_kind": "original_household_design",
        "empty_donor_branch": donor_frame is None,
        "empty_recipient_branch": recipient_frame is None,
        "shared_feature_age": "qualified_canonical_age; source raw age retained separately",
        "acs_under15_or_missing_anchor": "excluded_with_original_knownness; no_completion",
        "source_admission_issued": False,
        "model_fitted": False,
        "tax_split_applied": False,
        "clone_attachment_performed": False,
        "release_eligible": False,
        "origin_document_sha256": shared.codec.sha(origin_document),
        "recipient_matrix_sha256": None if matrix is None else shared.codec.sha(matrix),
    }
    projection = shared.codec.encode_json(evidence)
    return QualifiedPropertyIncomeSources(
        predictor,
        anchors,
        interest_values,
        routing_values,
        dividend_values,
        basis,
        frame,
        donor_frame,
        donor_columns,
        recipient_frame,
        recipient_columns,
        matrix,
        origins,
        origin_document,
        diagnostics,
        projection,
        evidence,
    )


def qualify_current_property_income_sources(
    preparation,
    allocated_population,
    clone_population,
    *,
    demographic_conditioning=False,
    geography_config=None,
):
    """Borrow original owners, compose, then requalify after all assembly work.

    Support populations satisfy the existing shared-predictor contract. Their
    importance/clone weights never supply the donor design-weight mapping.
    The returned object is descriptive; downstream owners repeat qualification
    before consumption and after their last relevant I/O before return/export.
    """
    require(
        type(preparation) is shared.source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    entry = preparation._checked()
    config_payload = shared.host.survey_budget._config_payload(geography_config)
    origin_document = shared.codec.encode_json(json.loads(entry[1])["origins"])
    support_seals = tuple(
        shared.host.survey_budget._population_identity(p)
        for p in (allocated_population, clone_population)
    )

    def qualify_predictors():
        return shared.qualify_current_survey_predictors(
            preparation,
            allocated_population,
            clone_population,
            demographic_conditioning=demographic_conditioning,
            geography_config=geography_config,
        )

    qualifiers = (
        qualify_predictors,
        lambda: acs.qualify_current_acs_income_anchors(preparation),
        lambda: interest.qualify_current_asec_interest(preparation),
        lambda: routing.qualify_current_asec_income_routing(preparation),
        lambda: dividend.qualify_current_asec_dividend(preparation),
    )
    sealers = (
        shared._qualified_seal,
        acs.income_anchor_seal,
        interest.interest_values_seal,
        _routing_seal,
        dividend.dividend_values_seal,
    )
    values, seals = [], []
    for qualify, seal in zip(qualifiers, sealers, strict=True):
        value = qualify()
        values.append(value)
        seals.append(seal(value))
        require(
            all(
                f(v) == s
                for f, v, s in zip(sealers[: len(values)], values, seals, strict=True)
            ),
            "BORROWED_VALUES_CHANGED",
        )
    require(
        values[0].demographic_conditioning is demographic_conditioning
        and values[0].geography_config_payload == config_payload,
        "SHARED_OPTION_BINDING",
    )
    result = _compose(tuple(values), origin_document)
    result_seal = property_income_sources_seal(result)
    # Fixed country composition, not a per-family issuer or generic executor.
    for qualify, seal, expected in zip(qualifiers, sealers, seals, strict=True):
        require(seal(qualify()) == expected, "SOURCE_REQUALIFICATION_CHANGED")
        require(
            property_income_sources_seal(result) == result_seal,
            "FINAL_DERIVED_VALUES_CHANGED",
        )
    # This actual original-source check is the final I/O borrow. Everything
    # below is a pure check of retained owner identity and complete value seals.
    require(preparation._checked() is entry, "FINAL_PREPARATION_CHANGED")
    shared.source._pure_final(entry[2])
    shared.host.survey_budget._preparation_entry(preparation, entry[1], entry)
    require(
        shared.host.survey_budget._config_payload(geography_config) == config_payload,
        "FINAL_GEOGRAPHY_CONFIG_CHANGED",
    )
    require(
        support_seals
        == tuple(
            shared.host.survey_budget._population_identity(p)
            for p in (allocated_population, clone_population)
        ),
        "FINAL_SUPPORT_CHANGED",
    )
    require(
        property_income_sources_seal(result) == result_seal,
        "FINAL_DERIVED_VALUES_CHANGED",
    )
    return result
