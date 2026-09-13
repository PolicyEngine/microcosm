"""Pure completion diagnostics over original source descriptions; no authority.

The retaining country host must qualify sources and check the complete parent
around relevant I/O. This module diagnoses descriptions, never issues that
authority, fits a model, or assigns an amount to a population.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import fsum

import numpy as np
import pandas as pd

from microcosm.frame import Frame, WeightKind
from microcosm.graph import ArtifactOutput, ArtifactType

from . import current_property_income_sources as sources
from . import support_provenance as provenance
from .property_income_constants import PROPERTY_COMPONENTS

PROTOCOL = "microcosm.us.property-completion-routing.v1"
PROPERTY_COMPLETION_TYPE = ArtifactType("microcosm.us.property_completion_routing", 1)
_COMPONENT_FIELDS = (
    (PROPERTY_COMPONENTS[0], "TRDINT_VAL", "ordinary_interest_known"),
    (PROPERTY_COMPONENTS[2], "DIV_VAL", "dividends_known"),
)
# Routing-policy categories, not alternative source codebooks. Raw source
# statuses remain visible and an unreviewed new status refuses classification.
_ASEC_STATUSES = frozenset(
    {
        "known_receipt",
        "known_nonreceipt",
        "observed_zero_component",
        "outside_reporting_universe",
        "contradictory_outside_reporting_universe",
        "missing_receipt_literal",
        "unrecognized_receipt_literal",
        "missing_amount",
        "invalid_amount_literal",
        "niu",
        "contradictory_niu_nonzero",
        "contradictory_no_nonzero",
        "ambiguous_recipient_zero",
    }
)
_ACS_STATUSES = frozenset(
    {
        "observed",
        "missing_source_amount",
        "malformed_source_amount",
        "outside_published_domain",
        "invalid_adjustment",
        "outside_universe_blank",
        "outside_universe_observation",
    }
)
_LITERAL_STATUSES = frozenset(
    {"missing", "malformed", "outside_printed_range", "in_printed_range"}
)


@dataclass(frozen=True)
class PropertyCompletionRouting:
    """Detached diagnostics. This value is not a source or parent admission."""

    person: pd.DataFrame
    components: pd.DataFrame
    reasons: pd.DataFrame
    clones: pd.DataFrame
    summary: pd.DataFrame
    payload: bytes


def property_completion_artifact_output() -> ArtifactOutput:
    """Declaration for future source-projection wiring; not an attached node."""
    return ArtifactOutput("completion_routing", PROPERTY_COMPLETION_TYPE)


def _require(value, reason):
    if not value:
        raise ValueError("PROPERTY_COMPLETION_" + reason)


def _ids(values, reason):
    _require(values.dtype == np.dtype("int64"), reason)
    return values.to_numpy(copy=True)


def _axis(table, expected, reason):
    _require(
        type(table) is pd.DataFrame
        and table.columns.is_unique
        and table.index.name == "person_id"
        and table.index.dtype == np.dtype("int64")
        and table.index.is_unique
        and table.index.equals(expected.index),
        reason,
    )
    _require(
        table.native_person_id.dtype == np.dtype("int64")
        and table.native_person_id.equals(expected.native_person_id),
        reason + "_NATIVE",
    )


def _source_error(status, literal=None):
    return (
        status.startswith("contradictory_")
        or status
        in {
            "invalid_amount_literal",
            "unrecognized_receipt_literal",
            "malformed_source_amount",
            "outside_published_domain",
            "invalid_adjustment",
        }
        or literal in {"malformed", "outside_printed_range"}
    )


def _origin_agreement(qualified, origins):
    columns = ["person_id", "native_person_id", "source"]
    _require(
        qualified.shared_predictors.origins[columns].equals(origins[columns]),
        "SHARED_ORIGIN_IDENTITY",
    )
    document = json.loads(qualified.origin_document)
    records = document["persons"]
    table = pd.DataFrame(records["rows"], columns=records["columns"])
    _require(
        np.array_equal(_ids(table.person_id, "DOCUMENT_PERSON_ID"), origins.index)
        and np.array_equal(
            _ids(table.selected_receiving_person_id, "DOCUMENT_NATIVE_ID"),
            origins.native_person_id,
        )
        and np.array_equal(table.source, origins.source),
        "DOCUMENT_ORIGIN_IDENTITY",
    )
    for frame in (qualified.source_frame, qualified.shared_predictors.source_frame):
        _require(
            np.array_equal(
                _ids(
                    frame.person[provenance.spine_source_id_column("person")],
                    "SOURCE_NATIVE_ID",
                ),
                origins.native_person_id,
            )
            and np.array_equal(
                frame.person[provenance.support_channel_column("person")].astype(str),
                origins.source,
            ),
            "SOURCE_FRAME_ORIGIN_IDENTITY",
        )


def _asec_components(qualified, origins):
    tables = (
        qualified.asec_interest_values.person,
        qualified.asec_dividend_values.person,
    )
    rows = []
    for table, (component, field, _) in zip(tables, _COMPONENT_FIELDS, strict=True):
        _axis(table, origins, "ASEC_AXIS")
        statuses = table[field + "_reporting_status"]
        literals = table[field + "_literal_status"]
        _require(
            statuses.notna().all() and set(statuses) <= _ASEC_STATUSES, "ASEC_STATUS"
        )
        _require(
            literals.notna().all() and set(literals) <= _LITERAL_STATUSES,
            "LITERAL_STATUS",
        )
        known = table[field + "_amount_known"]
        amount = table[field + "_amount"].to_numpy(dtype=np.float64, na_value=np.nan)
        _require(known.dtype == np.dtype("bool"), "KNOWNNESS_DTYPE")
        expected = statuses.isin(
            ("known_receipt", "known_nonreceipt", "observed_zero_component")
        )
        if field == "DIV_VAL":
            _require(
                not statuses.eq("observed_zero_component").any(), "DIVIDEND_ZERO_STATUS"
            )
        _require(
            known.equals(expected.rename(known.name))
            and np.array_equal(known, np.isfinite(amount))
            and not np.isinf(amount).any()
            and not (amount < 0).any(),
            "COMPONENT_KNOWNNESS",
        )
        basis = qualified.donor_basis.person[component].to_numpy(dtype=np.float64)
        _require(
            np.array_equal(np.isnan(basis), np.isnan(amount))
            and np.array_equal(
                basis[known].view("uint64"), amount[known].view("uint64")
            ),
            "BASIS_COMPONENT_AGREEMENT",
        )
        published = table[field + "_published_amount"].to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        _require(
            np.array_equal(
                published[known].view("uint64"), amount[known].view("uint64")
            ),
            "PUBLISHED_COMPONENT_AGREEMENT",
        )
        for original, value, is_known, status, literal in zip(
            table.index, amount, known, statuses, literals, strict=True
        ):
            derived = status == "known_nonreceipt"
            _require(not derived or value == 0, "NONRECEIPT_NONZERO")
            rows.append(
                (
                    int(original),
                    component,
                    float(value),
                    bool(is_known),
                    "derived"
                    if derived
                    else ("observed" if is_known else "unresolved"),
                    status,
                    literal,
                    "qualified_known_nonreceipt"
                    if derived
                    else (
                        "observed_component_zero" if is_known and value == 0 else "none"
                    ),
                )
            )
    return rows


def _clones(frame, origins):
    _require(type(frame) is Frame, "CLONE_FRAME_TYPE")
    table = frame.person
    ids = _ids(table.person_id, "CLONE_ID_DTYPE")
    original = _ids(
        table[provenance.support_source_id_column("person")], "CLONE_ORIGIN_DTYPE"
    )
    native = _ids(
        table[provenance.spine_source_id_column("person")], "CLONE_NATIVE_DTYPE"
    )
    role = _ids(
        table[provenance.support_clone_index_column("person")], "CLONE_ROLE_DTYPE"
    )
    source = table[provenance.support_channel_column("person")].astype(str).to_numpy()
    _require(
        table.person_id.is_unique and np.isin(original, origins.index).all(),
        "CLONE_COVERAGE",
    )
    lookup = origins.reindex(original)
    _require(
        np.array_equal(native, lookup.native_person_id)
        and np.array_equal(source, lookup.source),
        "CLONE_SOURCE_IDENTITY",
    )
    result = pd.DataFrame(
        {
            "person_id": ids,
            "original_person_id": original,
            "native_person_id": native,
            "source": source,
            "clone_index": role,
        }
    )
    _require(
        np.isin(role, (0, 1)).all()
        and not result.duplicated(["original_person_id", "clone_index"]).any()
        and result.groupby("original_person_id").size().eq(2).all()
        and set(original) == set(origins.index),
        "WHOLE_CLONE_PAIRS",
    )
    return result.set_index("person_id")


def _original_design(frame, origins):
    _require(
        type(frame) is Frame and frame.schema.person_entity == "person", "SOURCE_FRAME"
    )
    _require(
        np.array_equal(_ids(frame.person.person_id, "SOURCE_ID_DTYPE"), origins.index),
        "SOURCE_FRAME_AXIS",
    )
    weights = frame.weights_for("household")
    _require(
        weights.kind is WeightKind.DESIGN
        and frame.resolve_weights("person").kind is WeightKind.DESIGN,
        "ORIGINAL_DESIGN_KIND",
    )
    households = _ids(frame.table("household").household_id, "HOUSEHOLD_ID_DTYPE")
    membership = _ids(frame.person.person_household_id, "MEMBERSHIP_DTYPE")
    _require(
        len(set(households)) == len(households)
        and np.isin(membership, households).all(),
        "HOUSEHOLD_MEMBERSHIP",
    )
    values = np.array(weights.values, dtype=np.float64, copy=True)
    _require(np.isfinite(values).all() and (values >= 0).all(), "DESIGN_VALUES")
    series = pd.Series(values, index=pd.Index(households, name="household_id"))
    return membership, series, series.loc[membership].to_numpy(copy=True)


def _table_document(table):
    def cell(value):
        if value is pd.NA or value is None:
            return None
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            if np.isnan(value):
                return None
            _require(np.isfinite(value), "PAYLOAD_NONFINITE")
        _require(type(value) in (str, int, float, bool), "PAYLOAD_SCALAR")
        return value

    # itertuples preserves integer scalar types; iterrows can coerce IDs >2**53.
    return {
        "columns": list(table.columns),
        "index": [cell(v) for v in table.index],
        "index_name": table.index.name,
        "rows": [
            [cell(v) for v in row] for row in table.itertuples(index=False, name=None)
        ],
    }


def build_property_completion_routing(
    qualified: sources.QualifiedPropertyIncomeSources, clone_frame: Frame
) -> PropertyCompletionRouting:
    """Diagnose original component availability; never consume native I/O or fit.

    The source descriptions and the supplied Frame/weights are not admissions.
    An integrating host retains and requalifies their real issuers around I/O.
    """
    _require(
        type(qualified) is sources.QualifiedPropertyIncomeSources,
        "QUALIFIED_DESCRIPTION_TYPE",
    )
    before = sources.property_income_sources_seal(qualified)
    clone_before = sources._frame_seal(clone_frame)
    origins = qualified.origins
    _require(
        type(origins) is pd.DataFrame
        and origins.index.name == "person_id"
        and origins.index.dtype == np.dtype("int64")
        and origins.index.is_unique
        and origins.columns.is_unique
        and len(origins) > 0,
        "ORIGIN_AXIS",
    )
    _ids(origins.native_person_id, "NATIVE_ID_DTYPE")
    _require(
        np.array_equal(_ids(origins.person_id, "ORIGIN_ID_DTYPE"), origins.index)
        and set(origins.source) <= {"acs", "asec"},
        "ORIGIN_IDENTITY",
    )
    _origin_agreement(qualified, origins)
    asec = origins.loc[origins.source.eq("asec")]
    acs = origins.loc[origins.source.eq("acs")]
    for table in (qualified.donor_basis.person, qualified.asec_routing_values.person):
        _axis(table, asec, "ASEC_BASIS_AXIS")
    anchors = qualified.acs_anchor_values.anchors
    _axis(anchors, acs, "ACS_AXIS")
    ages = pd.Series(index=origins.index, dtype="float64")
    ages.loc[asec.index] = qualified.asec_interest_values.person.source_age
    ages.loc[acs.index] = pd.to_numeric(anchors.AGEP, errors="coerce")
    _require(
        np.isfinite(ages).all()
        and ages.between(0, 99).all()
        and np.equal(ages, np.floor(ages)).all(),
        "SOURCE_AGE",
    )
    for table in (
        qualified.asec_dividend_values.person,
        qualified.asec_routing_values.person,
        qualified.donor_basis.person,
    ):
        _require(
            np.array_equal(table.source_age, ages.loc[asec.index]),
            "SOURCE_AGE_AGREEMENT",
        )
    components = _asec_components(qualified, asec)
    anchor_status = anchors.property_income_status
    _require(
        anchor_status.notna().all() and set(anchor_status) <= _ACS_STATUSES,
        "ACS_STATUS",
    )
    known_anchor = anchors.property_income_known
    anchor_values = anchors.property_income_amount.to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    _require(
        known_anchor.dtype == np.dtype("bool")
        and np.array_equal(known_anchor, anchor_status.eq("observed"))
        and np.array_equal(known_anchor, np.isfinite(anchor_values))
        and np.array_equal(
            anchors.property_income_in_income_universe, ages.loc[acs.index] >= 15
        ),
        "ACS_KNOWNNESS_OR_UNIVERSE",
    )
    for original in acs.index:
        for component, _, _ in _COMPONENT_FIELDS:
            components.append(
                (
                    int(original),
                    component,
                    np.nan,
                    False,
                    "unresolved",
                    "component_not_separately_observed",
                    "not_separately_observed",
                    "none",
                )
            )
    components = pd.DataFrame(
        components,
        columns=[
            "original_person_id",
            "component",
            "amount",
            "known",
            "origin",
            "reporting_status",
            "literal_status",
            "zero_basis",
        ],
    )
    order = {int(original): i for i, original in enumerate(origins.index)}
    components["_order"] = components.original_person_id.map(order)
    components = (
        components.sort_values(["_order", "component"], kind="stable")
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    membership, weights, person_design = _original_design(
        qualified.source_frame, origins
    )
    clones = _clones(clone_frame, origins)
    person = origins[["native_person_id", "source"]].copy(deep=True)
    person["source_age"] = ages.astype("int64")
    person["original_household_id"] = membership
    person["original_household_design_weight"] = person_design
    person["anchor_known"] = False
    person["anchor_amount"] = np.nan
    person["anchor_status"] = "not_an_acs_aggregate_anchor"
    person["anchor_origin"] = "unresolved"
    person.loc[acs.index, "anchor_known"] = known_anchor.to_numpy()
    person.loc[acs.index, "anchor_amount"] = anchor_values
    person.loc[acs.index, "anchor_status"] = anchor_status.to_numpy()
    person.loc[acs.index[known_anchor], "anchor_origin"] = "observed"
    reasons = pd.DataFrame(
        False,
        index=origins.index,
        columns=[
            "under15",
            "source_error",
            "contradictory_evidence",
            "positive_outside_universe",
            "missing_source_evidence",
            "ambiguous_zero",
            "niu",
            "ordinary_interest_unknown",
            "dividends_unknown",
        ],
    )
    reasons["under15"] = ages < 15
    for component, field, known_name in _COMPONENT_FIELDS:
        selected = (
            components.loc[components.component.eq(component)]
            .set_index("original_person_id")
            .reindex(origins.index)
        )
        person[known_name] = selected.known.to_numpy()
        reasons[known_name.replace("_known", "_unknown")] = ~person[known_name]
        status, literal = selected.reporting_status, selected.literal_status
        reasons["source_error"] |= np.array(
            [
                _source_error(status_value, literal_value)
                for status_value, literal_value in zip(status, literal, strict=True)
            ]
        )
        reasons["contradictory_evidence"] |= status.str.startswith("contradictory_")
        reasons["missing_source_evidence"] |= status.str.startswith("missing_")
        reasons["ambiguous_zero"] |= status.eq("ambiguous_recipient_zero")
        reasons["niu"] |= status.eq("niu")
        table = (
            qualified.asec_interest_values.person
            if field == "TRDINT_VAL"
            else qualified.asec_dividend_values.person
        )
        published = table[field + "_published_amount"].to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        reasons.loc[asec.index, "positive_outside_universe"] |= (
            ages.loc[asec.index] < 15
        ) & (published > 0)
    reasons.loc[acs.index, "source_error"] |= np.array(
        [_source_error(s) for s in anchor_status], dtype=bool
    )
    reasons.loc[acs.index, "missing_source_evidence"] |= anchor_status.eq(
        "missing_source_amount"
    )
    reasons.loc[acs.index, "positive_outside_universe"] |= (
        ages.loc[acs.index] < 15
    ) & (pd.to_numeric(anchors.INTP, errors="coerce") > 0)
    reasons.loc[acs.index, "contradictory_evidence"] |= anchor_status.eq(
        "outside_universe_observation"
    )
    for name in qualified.donor_basis.exclusions:
        column = qualified.donor_basis.exclusions[name]
        _require(
            column.index.equals(asec.index) and column.dtype == np.dtype("bool"),
            "JOINT_EXCLUSION_AXIS",
        )
        reasons["joint:" + name] = False
        reasons.loc[asec.index, "joint:" + name] = column.to_numpy()
    routes = []
    for original in origins.index:
        if reasons.loc[original, "under15"]:
            route = "unsupported_under15_measurement"
        elif reasons.loc[original, "source_error"]:
            route = "source_review_required"
        elif (
            person.loc[original, "ordinary_interest_known"]
            and person.loc[original, "dividends_known"]
        ):
            route = "carry_known_components"
        elif person.loc[original, "source"] == "acs":
            route = (
                "existing_acs_anchor_decomposition"
                if person.loc[original, "anchor_known"]
                else "acs_anchor_completion_review"
            )
        else:
            route = "asec_component_completion_review"
        routes.append(route)
    person["completion_route"] = routes
    masks = {"all": np.ones(len(person), dtype=bool)}
    masks.update(
        {
            "route:" + r: person.completion_route.eq(r).to_numpy()
            for r in sorted(set(routes))
        }
    )
    masks.update({"reason:" + c: reasons[c].to_numpy() for c in reasons})
    summary_rows = []
    for name, mask in masks.items():
        households = np.unique(membership[mask])
        mass = fsum(person_design[mask])
        household_mass = fsum(weights.loc[households])
        _require(
            np.isfinite(mass) and np.isfinite(household_mass), "DESIGN_MASS_OVERFLOW"
        )
        summary_rows.append(
            (name, int(mask.sum()), len(households), mass, household_mass)
        )
    summary = pd.DataFrame(
        summary_rows,
        columns=[
            "selection",
            "person_count",
            "household_count",
            "design_weighted_person_mass",
            "union_household_design_mass",
        ],
    ).set_index("selection")
    tables = {
        "person": person,
        "components": components,
        "reasons": reasons,
        "clones": clones,
        "summary": summary,
    }
    payload = json.dumps(
        {
            "protocol": PROTOCOL,
            "source_authority": False,
            "complete_parent_authority": False,
            "amounts_assigned": False,
            "model_executed": False,
            "source_input": "descriptive_qualified_values; host_custody_required",
            "stage": "before_property_models",
            "tables": {k: _table_document(v) for k, v in tables.items()},
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    _require(
        sources.property_income_sources_seal(qualified) == before
        and sources._frame_seal(clone_frame) == clone_before,
        "INPUTS_CHANGED",
    )
    return PropertyCompletionRouting(
        person, components, reasons, clones, summary, payload
    )


# Explicit publication vocabulary. A new source/route reason needs review here
# before it can enter a public receipt; never export arbitrary summary labels.
_PUBLIC_ROUTES = frozenset(
    {
        "unsupported_under15_measurement",
        "source_review_required",
        "carry_known_components",
        "existing_acs_anchor_decomposition",
        "acs_anchor_completion_review",
        "asec_component_completion_review",
    }
)
_PUBLIC_REASONS = frozenset(
    {
        "under15",
        "source_error",
        "contradictory_evidence",
        "positive_outside_universe",
        "missing_source_evidence",
        "ambiguous_zero",
        "niu",
        "ordinary_interest_unknown",
        "dividends_unknown",
        "joint:under15",
        "joint:reported_total_unknown",
        "joint:ordinary_interest_unknown",
        "joint:retirement_interest_unknown",
        "joint:dividends_unknown",
        "joint:property_receipts_unknown",
        "joint:interest_discrepancy_unknown",
        "joint:interest_discrepancy_nonzero",
        "joint:other_income_possible_property",
        "joint:other_income_route_unresolved",
        "joint:survivor_possible_property",
        "joint:survivor_route_unresolved",
        "joint:survivor_additional_sources_unresolved",
    }
)
_PUBLIC_MEASURES = (
    "person_count",
    "household_count",
    "design_weighted_person_mass",
    "union_household_design_mass",
)


def property_completion_public_summary(value: PropertyCompletionRouting) -> dict:
    """Allowlisted aggregate description only; private row payload stays local.

    DESIGN measures describe original support, not calibrated representation or
    independently disclosive proof. This function does not grant source custody.
    """
    _require(type(value) is PropertyCompletionRouting, "PUBLIC_TYPE")
    table = value.summary
    labels = (
        {"all"}
        | {"route:" + r for r in _PUBLIC_ROUTES}
        | {"reason:" + r for r in _PUBLIC_REASONS}
    )
    _require(
        type(table) is pd.DataFrame
        and table.index.is_unique
        and table.index.name == "selection"
        and set(table.index) <= labels
        and "all" in table.index
        and tuple(table.columns) == _PUBLIC_MEASURES,
        "PUBLIC_ROSTER",
    )
    rows = []
    for label, row in zip(table.index, table.itertuples(index=False), strict=True):
        counts, masses = row[:2], row[2:]
        _require(
            all(
                isinstance(n, (int, np.integer))
                and not isinstance(n, (bool, np.bool_))
                and n >= 0
                for n in counts
            )
            and all(
                isinstance(n, (float, np.floating)) and np.isfinite(n) and n >= 0
                for n in masses
            ),
            "PUBLIC_VALUES",
        )
        rows.append(
            {
                "selection": label,
                **dict(
                    zip(
                        _PUBLIC_MEASURES,
                        (
                            int(counts[0]),
                            int(counts[1]),
                            float(masses[0]),
                            float(masses[1]),
                        ),
                        strict=True,
                    )
                ),
            }
        )
    return {
        "protocol": PROTOCOL,
        "stage": "before_property_models",
        "amounts_assigned": False,
        "model_executed": False,
        "source_authority": False,
        "complete_parent_authority": False,
        "support": "original_household_DESIGN; descriptive_only",
        "routes_mutually_exclusive": True,
        "reasons_overlap": True,
        "summary": rows,
    }
