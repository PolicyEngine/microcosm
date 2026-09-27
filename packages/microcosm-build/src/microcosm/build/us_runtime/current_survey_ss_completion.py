"""Qualified inputs and development completion of SS *reports*, not beneficiaries.

Full-original ASEC source support supplies labels. Selected ASEC/ACS originals
supply unresolved reports. The returned data never issues source authority.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from types import FunctionType

import numpy as np
import pandas as pd

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import model_input
from microcosm.frame import Frame, WeightKind

from . import current_social_security_source as observed
from . import current_survey_predictors as predictors
from . import survey_social_security as basis

source = observed.source
PROTOCOL = "microcosm.us.survey-ss-report-completion.v1"
FEATURES = (*predictors.FEATURES, "survey_predictor_ss_report_total")
LABEL = "survey_ss_report_category"
ELIGIBLE = "survey_ss_report_resolved_positive"
MONEY_FIELDS = ("WSAL_VAL", "SEMP_VAL")


def require(condition, reason):
    if not condition:
        raise ValueError("SURVEY_SS_COMPLETION_" + reason)


def _table(table):
    return observed._table_seal(table)


def _bits(values):
    return np.asarray(values, dtype="<f8").tobytes()


def _allowed(table):
    names = ["allowed_" + name for name in basis.COMPONENTS]
    values = table.loc[:, names]
    require(
        all(pd.api.types.is_bool_dtype(d) for d in values.dtypes)
        and not values.isna().any().any(),
        "ALLOWED_MASK",
    )
    return values.to_numpy(dtype=bool, copy=True)


def _axes(table):
    require(
        type(table) is pd.DataFrame
        and type(table.index) is pd.Index
        and table.index.dtype == np.dtype("int64")
        and table.index.name == "person_id"
        and table.index.is_unique,
        "ORIGINAL_AXIS",
    )


def _report_selection(table):
    _axes(table)
    totals = table.social_security_source_total.to_numpy(dtype="float64", copy=True)
    components = table.loc[:, basis.COMPONENTS].to_numpy(dtype="float64", copy=True)
    known = np.isfinite(totals)
    require(not np.isinf(totals).any() and (totals[known] >= 0).all(), "TOTAL_DOMAIN")
    complete = np.isfinite(components).all(axis=1)
    unknown = np.isnan(components).all(axis=1)
    require((complete | unknown).all(), "PARTIAL_BASIS")
    require((known | unknown).all(), "UNKNOWN_TOTAL_COMPONENTS")
    require(not (complete & ~known).any(), "UNKNOWN_TOTAL_COMPONENTS")
    require(
        np.array_equal(components[complete].sum(axis=1), totals[complete]),
        "SOURCE_TOTAL",
    )
    require((components[complete] >= 0).all(), "SOURCE_COMPONENTS")
    allowed = _allowed(table)
    require((components[complete][~allowed[complete]] == 0).all(), "SOURCE_SUPPORT")
    inside = table.source_reporting_universe
    require(
        pd.api.types.is_bool_dtype(inside.dtype) and inside.notna().all(),
        "REPORT_UNIVERSE",
    )
    require(np.array_equal(inside.to_numpy(dtype=bool), known), "REPORT_KNOWNNESS")
    positive = known & (totals > 0)
    resolved = positive & complete
    require(((components[resolved] > 0).sum(axis=1) == 1).all(), "RESOLVED_LABEL")
    require(not (known & (totals == 0) & unknown).any(), "UNRESOLVED_ZERO")
    return resolved, positive & unknown


@dataclass(frozen=True)
class QualifiedSSModelInputs:
    """Detached descriptive values; the graph host retains the real preparation."""

    full_source: observed.FullCurrentSocialSecurityProjection
    source_frame: Frame
    donor_columns: pd.DataFrame
    originals: pd.DataFrame
    recipient_features: pd.DataFrame
    matrix: bytes | None
    donor_projection: bytes
    recipient_projection: bytes
    evidence: dict


def qualified_seal(value):
    require(type(value) is QualifiedSSModelInputs, "QUALIFIED_TYPE")
    return (
        observed.full_social_security_seal(value.full_source),
        source._frame_identity(value.source_frame),
        _table(value.donor_columns),
        _table(value.originals),
        _table(value.recipient_features),
        value.matrix,
        value.donor_projection,
        value.recipient_projection,
        codec.encode_json(value.evidence),
    )


def _source_frame(full):
    """Retain structural axes plus genuine source age for explicit CREATE ownership."""
    tables = {}
    for entity in full.entities:
        keep = [full.schema.entity_id_column(entity)]
        if entity == "person":
            keep.append("age")
            keep.extend(
                full.schema.membership_column(e) for e in full.schema.group_entities
            )
        tables[entity] = full.table(entity).loc[:, keep].copy(deep=True)
    return Frame(
        tables,
        full.schema,
        dict(full._weights),
        full.strata,
        metadata=full.metadata,
        mass_log=full.mass_log,
    )


def qualify_current_survey_ss_model_inputs(preparation):
    """Capture genuine full-source labels and four approved current predictors.

    All publisher-allocation states are retained. Missing required predictors
    refuse; no candidate is silently dropped or filled. No assigned geography,
    prior-year income, or downstream imputed canonical value is consulted.
    """
    require(_live() == _LIVE, "IMPLEMENTATION_CHANGED")
    require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    implementation = observed._file_sha(__file__)
    with source.verification_epoch(join=True):
        entry = preparation._checked()
        native = entry[2].native[1]
        native_entry = native._checked()
        parent = native_entry[2].parent
        full = observed.qualify_full_current_social_security(preparation)
        snapshot = observed.full_social_security_seal(full)
        rows = full.asec_basis
        _axes(rows)
        ids = rows.index
        frame = full.asec_frame
        require(
            np.array_equal(frame.person.person_id.to_numpy(), ids.to_numpy()),
            "FULL_FRAME_AXIS",
        )
        require(
            frame.resolve_weights("person").kind is WeightKind.DESIGN, "DESIGN_WEIGHTS"
        )
        donor_mask, _ = _report_selection(rows)
        ready = parent.ready()
        header = json.loads(ready.header)
        require(
            header["target_year"] == 2024
            and header["semantic"] == "annual_current_money",
            "CURRENT_MONEY",
        )
        scope = parent.scope
        positions = {int(pid): i for i, pid in enumerate(scope.person_ids)}
        require(
            len(positions) == len(scope.person_ids)
            and all(int(pid) in positions for pid in ids),
            "MONEY_AXIS",
        )
        take = np.array([positions[int(pid)] for pid in ids], dtype="int64")
        require(all(scope.person_years[int(i)] == 2024 for i in take), "MONEY_COHORT")
        age = frame.person.age.to_numpy(dtype="float64", copy=True)
        literal_age = pd.to_numeric(rows.A_AGE, errors="raise").to_numpy(
            dtype="float64"
        )
        require(
            np.array_equal(age, literal_age) and np.isfinite(age).all(), "SOURCE_AGE"
        )
        features = pd.DataFrame(index=ids)
        features[FEATURES[0]] = age
        field_evidence = {}
        for name, output in zip(MONEY_FIELDS, FEATURES[1:3], strict=True):
            field = ready.field(name)
            domain = next(d for d in parent.spec.fields if d.name == name)
            require(domain.entity == "person", "MONEY_ENTITY")
            amount = field.amounts[take].copy()
            known = field.validity[take] == 1
            require(
                amount.dtype == np.dtype("float64")
                and np.isfinite(amount[known]).all(),
                "MONEY_DOMAIN",
            )
            amount[~known] = np.nan
            features[output] = amount
            field_evidence[name] = {
                "values_sha256": codec.sha(_bits(amount)),
                "status_sha256": codec.sha(field.statuses[take].tobytes()),
                "validity_sha256": codec.sha(field.validity[take].tobytes()),
                "zero_origin_sha256": codec.sha(field.zero_origin[take].tobytes()),
            }
        features[FEATURES[3]] = rows.social_security_source_total.to_numpy(copy=True)
        originals = full.selected.person.copy(deep=True)
        _, recipient_mask = _report_selection(originals)
        recipients = originals.index[recipient_mask]
        selected_asec = originals.source.eq("asec")
        selected_native = originals.loc[selected_asec, "native_person_id"]
        require(
            selected_native.is_unique and set(selected_native) <= set(ids),
            "SELECTED_NATIVE_JOIN",
        )
        all_features = pd.DataFrame(
            np.nan, index=originals.index, columns=FEATURES, dtype="float64"
        )
        all_features.loc[selected_asec, :] = features.loc[
            selected_native.to_numpy(), FEATURES
        ].to_numpy(copy=True)
        corrected, universe = predictors._acs_earnings(entry[2].frame)
        acs = ~selected_asec
        require(originals.source.isin(("asec", "acs")).all(), "SOURCE_CHANNEL")
        people = corrected.person.set_index("person_id")
        require(people.index.equals(originals.index), "ACS_SELECTED_AXIS")
        all_features.loc[acs, FEATURES[0]] = people.loc[acs, "age"].to_numpy(
            dtype="float64"
        )
        for output, name in zip(FEATURES[1:3], predictors.OUTPUTS[:2], strict=True):
            all_features.loc[acs, output] = people.loc[acs, name].to_numpy(
                dtype="float64"
            )
        all_features.loc[:, FEATURES[3]] = (
            originals.social_security_source_total.to_numpy(copy=True)
        )
        recipient_features = all_features.loc[recipients, FEATURES].copy(deep=True)
        require(
            np.isfinite(recipient_features.to_numpy()).all(),
            "RECIPIENT_PREDICTOR_UNKNOWN",
        )
        labels = pd.Series(
            pd.array([pd.NA] * len(ids), dtype="string"), index=ids, name=LABEL
        )
        resolved = rows.loc[donor_mask, basis.COMPONENTS].to_numpy(dtype="float64")
        labels.loc[donor_mask] = np.asarray(basis.COMPONENTS, dtype=object)[
            resolved.argmax(axis=1)
        ]
        donor_columns = features.copy(deep=True)
        donor_columns[LABEL] = labels
        donor_columns[ELIGIBLE] = donor_mask
        weights = frame.resolve_weights("person").values
        support = {
            name: float(
                weights[
                    donor_mask & labels.eq(name).fillna(False).to_numpy(dtype=bool)
                ].sum()
            )
            for name in basis.COMPONENTS
        }
        if len(recipients):
            require(
                np.isfinite(features.loc[donor_mask, FEATURES].to_numpy()).all(),
                "DONOR_PREDICTOR_UNKNOWN",
            )
            require(
                all(np.isfinite(v) and v > 0 for v in support.values()),
                "FULL_CLASS_SUPPORT",
            )
        matrix = (
            None
            if not len(recipients)
            else model_input.encode_recipient_matrix(
                recipient_features,
                entity="person",
                entity_ids=recipients.to_numpy(copy=True),
            )
        )
        structural = _source_frame(frame)
        evidence = {
            "protocol": PROTOCOL,
            "implementation_sha256": implementation,
            "preparation_sha256": codec.sha(entry[1]),
            "native_sha256": codec.sha(native_entry[1]),
            "full_ss_source_seal": snapshot,
            "money_header_sha256": codec.sha(ready.header),
            "current_predictor_fields": field_evidence,
            "predictors": list(FEATURES),
            "classes": list(basis.COMPONENTS),
            "full_original_asec_rows": len(ids),
            "resolved_positive_donors": int(donor_mask.sum()),
            "original_recipients": len(recipients),
            "class_design_weight_mass": support,
            "weight_kind": "design",
            "weights_sha256": codec.sha(_bits(weights)),
            "asec_interview_year": 2025,
            "asec_income_year": 2024,
            "acs_income_window": "rolling_prior_12_months_at_2024_interview",
            "acs_earnings_universe": dict(universe),
            "publisher_allocation_policy": "retain_all_literal_statuses",
            "estimand": "resolved_report_category_probabilities_as_conditional_mean_report_shares",
            "individual_beneficiary_assignment_claim": False,
            "scientific_qualification": "pending",
            "source_admission_issued": False,
            "release_eligible": False,
        }
        donor_projection = codec.encode_json(
            {
                **evidence,
                "arm": "full_original_asec",
                "source_frame_sha256": source._frame_identity(structural),
                "columns_sha256": _table(donor_columns),
            }
        )
        recipient_projection = codec.encode_json(
            {
                "protocol": PROTOCOL,
                "arm": "selected_original_reports",
                "source_basis_sha256": _table(originals),
                "features_sha256": _table(recipient_features),
                "matrix_sha256": None if matrix is None else codec.sha(matrix),
                "preparation_sha256": codec.sha(entry[1]),
            }
        )
        result = QualifiedSSModelInputs(
            full,
            structural,
            donor_columns,
            originals,
            recipient_features,
            matrix,
            donor_projection,
            recipient_projection,
            evidence,
        )
        seal = qualified_seal(result)
        require(
            preparation._checked() is entry
            and native._checked() is native_entry
            and parent.ready().header == ready.header,
            "FINAL_SOURCE",
        )
        source._pure_final(entry[2])
        require(
            _live() == _LIVE
            and observed.full_social_security_seal(full) == snapshot
            and qualified_seal(result) == seal,
            "FINAL_VALUES",
        )
    # Closing an owner epoch can perform its deferred full validation. Seal the
    # returned values after that final I/O as well, not before context exit.
    require(observed._file_sha(__file__) == implementation, "SOURCE_CODE_CHANGED")
    source._pure_final(entry[2])
    require(
        _live() == _LIVE
        and source._ISSUED.get(id(preparation)) is entry
        and source.asec_native._ISSUED.get(id(native)) is native_entry
        and qualified_seal(result) == seal,
        "FINAL_VALUES",
    )
    return result


def complete_reports(originals, probabilities):
    """Conserve qualified report totals; preserve complete and unknown source bits."""
    _, recipient_mask = _report_selection(originals)
    recipients = originals.index[recipient_mask]
    require(
        type(probabilities) is pd.DataFrame
        and probabilities.index.equals(recipients)
        and tuple(probabilities.columns) == basis.COMPONENTS,
        "SCORE_AXIS",
    )
    require(all(d == np.dtype("float64") for d in probabilities.dtypes), "SCORE_DTYPE")
    raw = probabilities.to_numpy(copy=True)
    require(
        np.isfinite(raw).all()
        and (raw >= 0).all()
        and (raw <= 1).all()
        and np.allclose(raw.sum(axis=1), 1, rtol=0, atol=1e-12),
        "PROBABILITIES",
    )
    amounts = originals.social_security_source_total.to_numpy(
        dtype="float64", copy=True
    )
    source_components = originals.loc[:, basis.COMPONENTS].to_numpy(
        dtype="float64", copy=True
    )
    known = np.isfinite(amounts)
    scores = np.full(source_components.shape, np.nan, dtype="float64")
    scores[recipient_mask] = raw
    completed = source_components.copy()
    completed[known] = basis.complete_positive_basis(
        amounts[known],
        source_components[known],
        _allowed(originals)[known],
        scores[known],
    )
    require(
        np.array_equal(completed[known].sum(axis=1), amounts[known]),
        "FINAL_TOTAL_CONSERVATION",
    )
    require(
        (completed[known][~_allowed(originals)[known]] == 0).all(),
        "FINAL_ALLOWED_SUPPORT",
    )
    require(
        _bits(completed[~recipient_mask]) == _bits(source_components[~recipient_mask]),
        "SOURCE_BITS_CHANGED",
    )
    return pd.DataFrame(
        completed, index=originals.index.copy(), columns=basis.COMPONENTS
    )


def _live():
    return (
        tuple(
            (
                m,
                tuple(
                    (n, source._function_seal(v))
                    for n, v in vars(m).items()
                    if type(v) is FunctionType
                ),
            )
            for m in (
                sys.modules[__name__],
                observed,
                basis,
                predictors,
                predictors.universe,
                model_input,
                codec,
            )
        ),
        source,
        Frame,
        WeightKind,
        WeightKind.DESIGN,
        QualifiedSSModelInputs,
        tuple(
            (name, source._function_seal(method))
            for name, method in vars(QualifiedSSModelInputs).items()
            if type(method) is FunctionType
        ),
        PROTOCOL,
        FEATURES,
        LABEL,
        ELIGIBLE,
        MONEY_FIELDS,
        basis.COMPONENTS,
        type(basis.REASON_COMPONENTS),
        tuple(basis.REASON_COMPONENTS.items()),
        predictors.FEATURES,
        predictors.OUTPUTS,
        predictors.source,
        type(predictors.universe.ACS_PUMS_EARNINGS_SOURCE_COLUMNS),
        tuple(predictors.universe.ACS_PUMS_EARNINGS_SOURCE_COLUMNS.items()),
    )


_LIVE = _live()
