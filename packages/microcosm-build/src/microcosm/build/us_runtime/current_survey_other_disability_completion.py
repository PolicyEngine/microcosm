"""Development-only other-disability dollars, qualified before any model fit.

ASEC observations and ACS model applicability are distinct. Under-15 and
unresolved ASEC observations stay unknown. No descriptive result issues source
authority; the caller retains the genuine preparation and receiving owner.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType

import numpy as np
import pandas as pd

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import model_input
from microcosm.frame import Frame, WeightKind

from . import current_asec_other_disability_source as observed
from . import current_survey_predictors as predictors
from . import support_provenance as provenance

source = predictors.source
PROTOCOL = "microcosm.us.other-disability-completion.v1"
FEATURES = predictors.FEATURES
MONEY_FIELDS = ("WSAL_VAL", "SEMP_VAL")
TARGET = "survey_other_disability_amount_target"
ELIGIBLE = "survey_other_disability_donor_eligible"
MODEL_APPLICABLE = observed.REPORT_PREFIX + "model_applicable"
CANONICAL_KNOWN = observed.REPORT_PREFIX + "canonical_known"
VALUE_ORIGIN = observed.REPORT_PREFIX + "value_origin"
ORIGINS = ("observed", "modeled", "source_unresolved", "outside_model_universe")


def require(condition, reason):
    if not condition:
        raise ValueError("OTHER_DISABILITY_COMPLETION_" + reason)


def _table(table):
    # The source adapter's seal includes nullable physical backing and masks.
    return observed.other_disability_values_seal(
        observed.CurrentAsecOtherDisabilityValues(table, {})
    )


def _bits(values):
    return np.asarray(values, dtype="<f8").tobytes()


@dataclass(frozen=True)
class QualifiedOtherDisabilityCompletion:
    full_source: observed.CurrentAsecOtherDisabilityValues
    selected_source: observed.CurrentAsecOtherDisabilityValues
    source_frame: Frame
    donor_columns: pd.DataFrame
    originals: pd.DataFrame
    recipient_features: pd.DataFrame
    matrix: bytes | None
    donor_projection: bytes
    recipient_projection: bytes
    evidence: dict


def qualified_seal(value):
    require(type(value) is QualifiedOtherDisabilityCompletion, "QUALIFIED_TYPE")
    return (
        observed.other_disability_values_seal(value.full_source),
        observed.other_disability_values_seal(value.selected_source),
        source._frame_identity(value.source_frame),
        _table(value.donor_columns),
        _table(value.originals),
        _table(value.recipient_features),
        value.matrix,
        value.donor_projection,
        value.recipient_projection,
        codec.encode_json(value.evidence),
    )


def _structural_frame(full):
    tables = {}
    for entity in full.entities:
        keep = [full.schema.entity_id_column(entity)]
        if entity == "person":
            keep += ["age"]
            keep += [
                full.schema.membership_column(e) for e in full.schema.group_entities
            ]
        tables[entity] = full.table(entity).loc[:, keep].copy(deep=True)
    return Frame(
        tables,
        full.schema,
        dict(full._weights),
        full.strata,
        metadata=full.metadata,
        mass_log=full.mass_log,
    )


def qualify_current_survey_other_disability_completion(preparation):
    """Borrow full DESIGN donors and selected original ACS age-15+ recipients."""
    require(_live() == _LIVE, "IMPLEMENTATION_CHANGED")
    require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    implementation = codec.sha(Path(__file__).read_bytes())
    with source.verification_epoch(join=True):
        entry = preparation._checked()
        native = entry[2].native[1]
        issued = native._checked()
        retained = issued[2]
        full_source = observed.qualify_current_asec_other_disability(
            preparation, full_original=True
        )
        require(
            full_source.evidence["retirement_detail_evidence"].get("projection_scope")
            == "full_original_current_asec",
            "FULL_SOURCE_SCOPE",
        )
        mask, weights, _, _ = source.asec_native._roster(
            retained.parent, retained.coverage, retained.anchors, retained.fields, None
        )
        full = source._normalized_source_copy(
            source.asec_native._descendant(retained.parent, mask, weights)
        )
        require(
            full.resolve_weights("person").kind is WeightKind.DESIGN, "DESIGN_WEIGHTS"
        )
        ids = pd.Index(full.person.person_id.to_numpy(), name="person_id")
        rows = full_source.person
        require(
            rows.index.equals(ids)
            and np.array_equal(rows.native_person_id.to_numpy(), ids.to_numpy()),
            "FULL_SOURCE_AXIS",
        )
        age = full.person.age.to_numpy(dtype="float64", copy=True)
        require(
            np.isfinite(age).all()
            and np.array_equal(age, rows.source_age.to_numpy(dtype="float64")),
            "SOURCE_AGE",
        )
        ready = retained.parent.ready()
        header = json.loads(ready.header)
        require(
            header["target_year"] == 2024
            and header["semantic"] == "annual_current_money",
            "CURRENT_MONEY",
        )
        scope = retained.parent.scope
        positions = {int(pid): i for i, pid in enumerate(scope.person_ids)}
        require(
            len(positions) == len(scope.person_ids)
            and all(int(pid) in positions for pid in ids),
            "MONEY_AXIS",
        )
        take = np.array([positions[int(pid)] for pid in ids], dtype="int64")
        require(all(scope.person_years[int(i)] == 2024 for i in take), "MONEY_COHORT")
        features = pd.DataFrame({FEATURES[0]: age}, index=ids)
        field_evidence = {}
        for name, output in zip(MONEY_FIELDS, FEATURES[1:], strict=True):
            field = ready.field(name)
            require(
                next(d for d in retained.parent.spec.fields if d.name == name).entity
                == "person",
                "MONEY_ENTITY",
            )
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
                **{
                    axis + "_sha256": codec.sha(getattr(field, axis)[take].tobytes())
                    for axis in ("statuses", "validity", "zero_origin")
                },
            }
        known = rows[observed.KNOWN_COLUMN].to_numpy(dtype=bool)
        amounts = rows[observed.AMOUNT_COLUMN].to_numpy(
            dtype="float64", na_value=np.nan
        )
        require(
            np.array_equal(known, np.isfinite(amounts))
            and not np.isinf(amounts).any()
            and (amounts[known] >= 0).all()
            and (age[known] >= observed.REPORTING_AGE).all(),
            "DONOR_SOURCE_KNOWNNESS",
        )
        donor_columns = features.copy(deep=True)
        donor_columns[TARGET] = amounts
        donor_columns[ELIGIBLE] = known
        selected = entry[2].frame.person
        original_ids = pd.Index(selected.person_id.to_numpy(), name="person_id")
        originals = pd.DataFrame(
            {
                "native_person_id": selected[
                    provenance.spine_source_id_column("person")
                ].to_numpy(),
                "source": selected[
                    provenance.support_channel_column("person")
                ].to_numpy(),
                "age": selected.age.to_numpy(dtype="float64"),
            },
            index=original_ids,
        )
        require(
            original_ids.is_unique
            and original_ids.dtype == np.dtype("int64")
            and originals.source.isin(("asec", "acs")).all(),
            "ORIGINAL_AXIS",
        )
        asec = originals.source.eq("asec")
        native_ids = originals.loc[asec, "native_person_id"]
        require(native_ids.is_unique and set(native_ids) <= set(ids), "SELECTED_JOIN")
        selected_rows = rows.loc[native_ids.to_numpy()].copy(deep=True)
        selected_rows.index = original_ids[asec]
        selected_source = observed.CurrentAsecOtherDisabilityValues(
            selected_rows,
            {
                "protocol": observed.PROTOCOL,
                "projection_scope": "selected_from_qualified_full_original_current_asec",
                "full_source_values_sha256": observed.other_disability_values_seal(
                    full_source
                ),
                "source_admission_issued": False,
            },
        )
        corrected, universe = predictors._acs_earnings(entry[2].frame)
        people = corrected.person.set_index("person_id")
        require(
            people.index.equals(original_ids)
            and np.array_equal(
                originals.age.to_numpy(), people.age.to_numpy(dtype="float64")
            ),
            "RECIPIENT_SOURCE_AXIS",
        )
        requested = (~asec) & originals.age.ge(observed.REPORTING_AGE)
        # Stable within a fixed original roster, independent of clone-row order.
        recipients = original_ids[requested].sort_values()
        recipient_features = pd.DataFrame(index=recipients)
        recipient_features[FEATURES[0]] = people.loc[recipients, "age"].to_numpy(
            dtype="float64"
        )
        for feature, output in zip(FEATURES[1:], predictors.OUTPUTS[:2], strict=True):
            recipient_features[feature] = people.loc[recipients, output].to_numpy(
                dtype="float64"
            )
        design = full.resolve_weights("person").values
        require(np.isfinite(design).all() and (design >= 0).all(), "DESIGN_DOMAIN")
        if len(recipients):
            require(
                np.isfinite(recipient_features.to_numpy()).all(),
                "RECIPIENT_PREDICTOR_UNKNOWN",
            )
            require(
                np.isfinite(features.loc[known].to_numpy()).all(),
                "DONOR_PREDICTOR_UNKNOWN",
            )
            require(known.any() and float(design[known].sum()) > 0, "DONOR_SUPPORT")
        matrix = (
            None
            if not len(recipients)
            else model_input.encode_recipient_matrix(
                recipient_features,
                entity="person",
                entity_ids=recipients.to_numpy(copy=True),
            )
        )
        structural = _structural_frame(full)
        excluded_affirmed = (
            ~known
            & (age >= observed.REPORTING_AGE)
            & rows[observed.RECEIPT_FIELD + "_code"]
            .eq(1)
            .fillna(False)
            .to_numpy(dtype=bool)
        )
        excluded_affirmed_mass = float(design[excluded_affirmed].sum())
        known_positive_mass = float(design[known & (amounts > 0)].sum())
        evidence = {
            "protocol": PROTOCOL,
            "implementation_sha256": implementation,
            "preparation_sha256": codec.sha(entry[1]),
            "native_sha256": codec.sha(issued[1]),
            "money_header_sha256": codec.sha(ready.header),
            "source_values_sha256": observed.other_disability_values_seal(full_source),
            "current_predictor_fields": field_evidence,
            "predictors": list(FEATURES),
            "full_original_asec_rows": len(ids),
            "eligible_donors": int(known.sum()),
            "eligible_design_mass": float(design[known].sum()),
            "donor_all_zero": bool(known.any() and (amounts[known] == 0).all()),
            "excluded_affirmed_receipt_unknown_amount_rows": int(
                excluded_affirmed.sum()
            ),
            "excluded_affirmed_receipt_unknown_amount_design_mass": excluded_affirmed_mass,
            "known_positive_design_mass": known_positive_mass,
            "excluded_affirmed_to_known_positive_mass_ratio": (
                excluded_affirmed_mass / known_positive_mass
                if known_positive_mass > 0
                else None
            ),
            "incidence_bias_caveat": "Development model excludes source-unknown targets, including affirmed receipt with unresolved dollars; source missingness may bias modeled incidence. No model adjustment is inferred from this diagnostic.",
            "source_reason_support": {
                reason: {
                    "rows": int(rows[observed.REASON_COLUMN].eq(reason).sum()),
                    "design_mass": float(
                        design[
                            rows[observed.REASON_COLUMN].eq(reason).to_numpy(dtype=bool)
                        ].sum()
                    ),
                }
                for reason in observed.REASONS
                if reason != observed.UNOBSERVED_REASON
            },
            "weight_kind": "design",
            "weights_sha256": codec.sha(_bits(design)),
            "original_recipients": len(recipients),
            "asec_income_year": 2024,
            "asec_interview_year": 2025,
            "acs_income_window": "rolling_prior_12_months_at_2024_interview",
            "acs_earnings_universe": dict(universe),
            "recipient_order": "ascending_original_person_id; sequential_draws_not_nested_roster_invariant",
            "under15_completed_with_zero": False,
            "asec_unknowns_completed": False,
            "publisher_allocation_policy": "retain_all_literal_statuses",
            "scientific_qualification": "pending",
            "source_admission_issued": False,
            "release_eligible": False,
        }
        donor_projection = codec.encode_json(
            {
                **evidence,
                "source_frame_sha256": source._frame_identity(structural),
                "columns_sha256": _table(donor_columns),
            }
        )
        recipient_projection = codec.encode_json(
            {
                "protocol": PROTOCOL,
                "originals_sha256": _table(originals),
                "features_sha256": _table(recipient_features),
                "matrix_sha256": None if matrix is None else codec.sha(matrix),
            }
        )
        result = QualifiedOtherDisabilityCompletion(
            full_source,
            selected_source,
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
            and native._checked() is issued
            and retained.parent.ready().header == ready.header,
            "FINAL_SOURCE",
        )
        source._pure_final(entry[2])
        require(_live() == _LIVE and qualified_seal(result) == seal, "FINAL_VALUES")
    require(
        codec.sha(Path(__file__).read_bytes()) == implementation, "SOURCE_CODE_CHANGED"
    )
    source._pure_final(entry[2])
    require(
        _live() == _LIVE
        and source._ISSUED.get(id(preparation)) is entry
        and source.asec_native._ISSUED.get(id(native)) is issued
        and qualified_seal(result) == seal,
        "FINAL_VALUES",
    )
    return result


def _receiver(qualified, receiving):
    people = receiving.person
    names = (
        "person_id",
        provenance.support_source_id_column("person"),
        provenance.support_clone_index_column("person"),
        provenance.spine_source_id_column("person"),
    )
    require(
        all(
            name in people and people[name].dtype == np.dtype("int64") for name in names
        )
        and people.person_id.is_unique,
        "RECEIVING_AXES",
    )
    _, original_name, clone_name, native_name = names
    originals, clones = people[original_name].to_numpy(), people[clone_name].to_numpy()
    require(
        len(originals) == 2 * len(qualified.originals)
        and set(originals) == set(qualified.originals.index),
        "CLONE_SOURCE_ROSTER",
    )
    origins = qualified.originals.reindex(originals)
    require(
        np.array_equal(
            people[native_name].to_numpy(), origins.native_person_id.to_numpy()
        )
        and np.array_equal(
            people[provenance.support_channel_column("person")].to_numpy(),
            origins.source.to_numpy(),
        ),
        "CLONE_NATIVE_OR_CHANNEL",
    )
    order = np.lexsort((clones, originals))
    require(
        np.array_equal(
            originals[order], np.repeat(np.sort(qualified.originals.index), 2)
        )
        and np.array_equal(clones[order], np.tile([0, 1], len(qualified.originals))),
        "CLONE_PAIR",
    )
    return originals


def attachment_dtypes(qualified, receiving):
    """Preflight ownership collisions before fitting; precision is checked on draws."""
    _receiver(qualified, receiving)
    result = {
        observed.attached_name(c): str(qualified.selected_source.person[c].dtype)
        for c in qualified.selected_source.person
        if c != "native_person_id"
    }
    result.update(
        {
            observed.OUTPUT: "float64",
            MODEL_APPLICABLE: "boolean",
            CANONICAL_KNOWN: "boolean",
            VALUE_ORIGIN: "string",
        }
    )
    for name in result:
        if name not in receiving.person:
            continue
        require(name == observed.OUTPUT, "REPORT_ALREADY_OWNED:" + name)
        dtype = receiving.person[name].dtype
        require(dtype in (np.dtype("float32"), np.dtype("float64")), "CANONICAL_DTYPE")
        result[name] = str(dtype)
    return result


def attach_other_disability_completion(qualified, receiving, draws):
    """One original draw, exactly transported to both clones; never mark it observed."""
    require(_live() == _LIVE, "IMPLEMENTATION_CHANGED")
    seal = qualified_seal(qualified)
    originals = _receiver(qualified, receiving)
    dtypes = attachment_dtypes(qualified, receiving)
    recipients = qualified.recipient_features.index
    require(
        type(draws) is pd.Series
        and draws.index.equals(recipients)
        and draws.dtype == np.dtype("float64")
        and np.isfinite(draws.to_numpy()).all()
        and (draws.to_numpy() >= 0).all(),
        "DRAW_AXIS_OR_VALUES",
    )
    source_attachment = observed.attach_other_disability_columns(
        qualified.selected_source, receiving
    )
    columns = dict(source_attachment.columns)
    index = pd.Index(receiving.person.person_id.to_numpy(), name="person_id")
    canonical = columns["person", observed.OUTPUT].to_numpy(
        dtype="float64", na_value=np.nan
    )
    applicable = np.isin(originals, recipients.to_numpy())
    canonical[applicable] = draws.reindex(originals[applicable]).to_numpy()
    converted = canonical.astype(dtypes[observed.OUTPUT])
    require(
        np.array_equal(canonical, converted.astype("float64"), equal_nan=True)
        and np.array_equal(np.signbit(canonical), np.signbit(converted)),
        "CANONICAL_PRECISION_LOSS",
    )
    columns["person", observed.OUTPUT] = pd.Series(
        converted, index=index, name=observed.OUTPUT
    )
    source_known = columns[
        "person", observed.attached_name(observed.KNOWN_COLUMN)
    ].to_numpy(dtype=bool)
    outside = (
        qualified.originals.age.reindex(originals).to_numpy() < observed.REPORTING_AGE
    )
    require(not (outside & np.isfinite(canonical)).any(), "UNDER15_COMPLETED")
    origins = np.where(
        applicable,
        "modeled",
        np.where(
            source_known,
            "observed",
            np.where(outside, "outside_model_universe", "source_unresolved"),
        ),
    )
    for name, data, dtype in (
        (MODEL_APPLICABLE, applicable, "boolean"),
        (CANONICAL_KNOWN, np.isfinite(canonical), "boolean"),
        (VALUE_ORIGIN, origins, "string"),
    ):
        columns["person", name] = pd.Series(
            pd.array(data, dtype=dtype), index=index, name=name
        )
    receipt = {
        **source_attachment.receipt,
        "completion_protocol": PROTOCOL,
        "modeled_originals": len(draws),
        "draws_consumed": len(draws),
        "clone_redraw_issued": False,
        "source_knownness_preserved": True,
        "under15_completed_with_zero": False,
        "release_eligible": False,
    }
    stamp = (
        _table(pd.DataFrame({name: column for (_, name), column in columns.items()})),
        codec.encode_json(receipt),
    )
    result = observed.OtherDisabilityAttachment(columns, receipt)
    require(
        result.columns is columns and result.receipt is receipt,
        "ATTACHMENT_PAYLOAD_CHANGED",
    )
    require(qualified_seal(qualified) == seal and _live() == _LIVE, "FINAL_VALUES")
    require(
        (
            _table(
                pd.DataFrame({name: column for (_, name), column in columns.items()})
            ),
            codec.encode_json(receipt),
        )
        == stamp,
        "FINAL_ATTACHMENT_CHANGED",
    )
    return result


def _live():
    return (
        tuple(
            (
                module,
                tuple(
                    (name, source._function_seal(value))
                    for name, value in vars(module).items()
                    if type(value) is FunctionType
                ),
            )
            for module in (
                sys.modules[__name__],
                observed,
                predictors,
                model_input,
                codec,
            )
        ),
        observed._live(),
        source,
        Frame,
        WeightKind,
        WeightKind.DESIGN,
        QualifiedOtherDisabilityCompletion,
        observed._class_methods(QualifiedOtherDisabilityCompletion),
        PROTOCOL,
        FEATURES,
        MONEY_FIELDS,
        TARGET,
        ELIGIBLE,
        MODEL_APPLICABLE,
        CANONICAL_KNOWN,
        VALUE_ORIGIN,
        ORIGINS,
    )


_LIVE = _live()
