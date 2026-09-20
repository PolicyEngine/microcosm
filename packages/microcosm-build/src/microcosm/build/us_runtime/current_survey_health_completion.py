"""Original-ASEC donors and original-ACS matrices for optional health modeling.

Returned values are borrowed projections, never source or Population authority.
The existing enrichment host retains and requalifies their genuine preparation.
The explicit development assumption transports 2025 interview coverage onto
2024 ACS recipients; no temporal equivalence or release qualification is claimed.
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.calibrate.geography_constants import US_STATE_NUMERIC_FIPS_TO_POSTAL
from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import model_input
from microcosm.frame import Frame, WeightKind

from . import current_asec_demographics as demographics
from . import current_survey_health_coverage as health
from . import current_survey_health_source as literals
from . import survey_population_preparation as source

PROTOCOL = "microcosm.us.current-survey-health-completion.v1"
FIELDS = tuple(field for field in health.FIELDS if field.acs is None)
TARGETS = tuple("health_completion_target_" + field.asec for field in FIELDS)
FEATURES = (
    "health_predictor_age",
    "health_predictor_is_female",
    "health_predictor_state_fips",
)
ELIGIBLE = "health_completion_donor_eligible"
SEED = 581


def require(condition, reason):
    if not condition:
        raise ValueError("CURRENT_SURVEY_HEALTH_COMPLETION_" + reason)


def model_columns(people):
    """Encode only exact known source booleans; unknowns exclude training rows."""
    require(people.person_id.is_unique, "DONOR_AXIS")
    result = pd.DataFrame(index=pd.Index(people.person_id.to_numpy(), name="person_id"))
    for field, target in zip(FIELDS, TARGETS, strict=True):
        observed = health.recode(
            people[health.SOURCE_PREFIX + field.asec],
            people[health.SOURCE_PREFIX + "I_" + field.asec],
            survey="asec",
        )
        result[target] = observed.value.to_numpy(dtype="float64", na_value=np.nan)
    features = people.loc[:, list(FEATURES)].to_numpy(dtype="float64")
    require(not np.isinf(features).any(), "DONOR_NONFINITE")
    result[ELIGIBLE] = np.isfinite(features).all(axis=1) & result.notna().all(axis=1)
    return result


def completed_columns(raw, draws):
    """Merge seven ACS semantic-gap draws, retaining all literal source states."""
    acs = raw.source.eq("acs")
    require(
        type(draws) is pd.DataFrame
        and draws.index.equals(raw.index[acs])
        and tuple(draws.columns) == TARGETS,
        "DRAW_AXIS",
    )
    values = draws.to_numpy()
    require(
        all(draws[c].dtype == np.dtype("float64") for c in draws)
        and np.isfinite(values).all()
        and np.isin(values, (0.0, 1.0)).all(),
        "DRAW_DOMAIN",
    )
    result = pd.concat(
        [
            health.source_columns(raw),
            *(health.recode_field(raw, f.output) for f in health.FIELDS),
        ],
        axis=1,
    )
    for field, target in zip(FIELDS, TARGETS, strict=True):
        require(
            result.loc[acs, field.output].isna().all()
            and not result.loc[acs, field.output + "__known"].any()
            and result.loc[acs, field.output + "__source_status"]
            .eq("semantic_gap:" + field.acs_gap)
            .all(),
            "RECIPIENT_SEMANTIC_GAP",
        )
        result.loc[acs, field.output] = draws[target].to_numpy() == 1.0
        result[field.output + "__imputed"] = acs.to_numpy(copy=True)
    return result


def _full_raw(entry, native_entry, roster):
    """Capture the exact retained ASEC member, selecting its full original roster."""
    wanted = {
        (int(row["source_household_id"]), row["PERIDNUM"], int(row["A_LINENO"])): int(
            row["person_id"]
        )
        for row in roster
    }
    require(len(wanted) == len(roster), "FULL_SOURCE_KEYS")
    pins = tuple(pin for pin in literals.asec._MEMBER_PINS if pin[0] == 2024)
    require(len(pins) == 1, "ASEC_PIN_ROSTER")
    year, member, archive, digest, rows, size = pins[0]
    retained = tuple(
        s
        for s in native_entry[2].coverage.receipt["sources"]
        if s["source_year"] == year
    )
    require(
        len(retained) == 1
        and all(
            retained[0][k] == v
            for k, v in (
                ("member", member),
                ("archive_sha256", archive),
                ("member_sha256", digest),
                ("rows", rows),
                ("member_bytes", size),
            )
        ),
        "FULL_MEMBER_BINDING",
    )
    selected = {}
    with tempfile.TemporaryDirectory(prefix="microcosm-health-donor-") as directory:
        path = Path(directory) / member
        identity = literals.asec._capture(
            entry[2].root / "asec" / member,
            path,
            size=size,
            digest=digest,
            budget=[literals.asec._BODY_MAX],
        )
        with path.open("rb") as stream:
            count = literals._scan(
                stream,
                survey="asec",
                wanted=set(wanted),
                selected=selected,
                maximum=rows,
            )
        require(
            count == rows
            and set(selected) == set(wanted)
            and literals.asec._identity(path.stat(follow_symlinks=False)) == identity
            and literals.housing._persisted_sha(path, size) == digest,
            "FULL_CAPTURE_CHANGED",
        )
    raw = pd.DataFrame(
        [selected[key] for key in wanted],
        index=pd.Index(list(wanted.values()), name="person_id"),
    )
    return raw, digest


def _full_source(preparation, entry):
    native = entry[2].native[1]
    native_entry = native._checked()
    retained = native_entry[2]
    document = json.loads(native_entry[1])
    require(
        document["source_year"] == document["income_year"] == 2024
        and document["survey_year"] == 2025,
        "ASEC_PERIOD",
    )
    mask, weights, households, roster = source.asec_native._roster(
        retained.parent, retained.coverage, retained.anchors, retained.fields, None
    )
    full = source._normalized_source_copy(
        source.asec_native._descendant(retained.parent, mask, weights)
    )
    require(
        full.weights_for("household").kind is WeightKind.DESIGN, "DONOR_DESIGN_WEIGHTS"
    )
    raw, raw_sha = _full_raw(entry, native_entry, roster)
    ids = pd.Index(full.person.person_id.to_numpy(), name="person_id")
    require(raw.index.is_unique and set(raw.index) == set(ids), "FULL_PERSON_BIJECTION")
    raw = raw.reindex(ids)
    observed = demographics.demographic.load_authenticated_asec_demographic_source(
        retained.parent,
        member_paths={
            year: entry[2].root / "asec" / f"pppub{year - 1999}.csv"
            for year in (2022, 2023, 2024)
        },
    )
    require(
        observed.receipt["source_identity"] == retained.parent.source.identity.decode(),
        "SEX_PARENT",
    )
    lookup = pd.MultiIndex.from_arrays(
        (observed.array("income_year"), observed.array("person_id"))
    )
    require(lookup.is_unique, "SEX_AXIS")
    positions = lookup.get_indexer(
        pd.MultiIndex.from_arrays((np.full(len(ids), 2024), ids))
    )
    require((positions >= 0).all() and len(set(positions)) == len(ids), "SEX_JOIN")
    sex = observed.array("asec_sex_binding_state")[positions]
    require(np.isin(sex, (0, 1, 2)).all(), "SEX_DOMAIN")
    require(len(retained.fields.document["members"]) == 1, "STATE_MEMBER")
    states, state_pin = demographics._load_current_state(
        entry[2].root / "asec" / "hhpub25.csv", retained.fields.document["members"][0]
    )
    state_by_id = {}
    for row in households:
        year, native_id = row["native_key"]
        require(year == 2024 and native_id in states, "STATE_JOIN")
        state = states[native_id]
        require(state["H_SEQ"] == row["household_fields"]["H_SEQ"], "STATE_LITERAL_KEY")
        code = state["state_code"]
        state_by_id[row["household_id"]] = (
            code if code in US_STATE_NUMERIC_FIPS_TO_POSTAL else np.nan
        )
    require(
        set(state_by_id) == set(full.table("household").household_id),
        "FULL_HOUSEHOLD_BIJECTION",
    )
    schema = full.schema
    structural = [
        schema.person_id_column,
        *(schema.membership_column(e) for e in schema.group_entities),
    ]
    people = full.person.loc[:, structural].copy(deep=True)
    age = full.person.age.to_numpy(dtype="float64")
    require(
        np.array_equal(age, full.person.A_AGE.to_numpy()) and np.isfinite(age).all(),
        "AGE_IDENTITY",
    )
    people[FEATURES[0]] = age
    people[FEATURES[1]] = np.where(sex == 0, np.nan, (sex == 2).astype(float))
    people[FEATURES[2]] = people.person_household_id.map(state_by_id).to_numpy(
        dtype="float64"
    )
    for field in FIELDS:
        for name in (field.asec, "I_" + field.asec):
            people[health.SOURCE_PREFIX + name] = pd.array(
                raw[name], dtype=health.STRING_DTYPE
            )
    tables = {"person": people}
    for entity in schema.group_entities:
        tables[entity] = (
            full.table(entity).loc[:, [schema.id_column(entity)]].copy(deep=True)
        )
    frame = Frame(
        tables,
        schema,
        dict(full._weights),
        full.strata,
        metadata=full.metadata,
        mass_log=full.mass_log,
    )
    require(
        native._checked() is native_entry and preparation._checked() is entry,
        "FULL_OWNER_CHANGED",
    )
    source._pure_final(entry[2])
    return frame, {
        "asec_native_sha256": codec.sha(native_entry[1]),
        "asec_member_sha256": raw_sha,
        "sex_source_sha256": observed.content_sha256,
        "state_source": state_pin,
        "donor_weight_kind": "design",
        "donor_weight_source": "original_HSUP_WGT/100",
        "full_original_donor_persons": len(people),
    }


def _acs_features(entry, observations):
    """Source sex/state/age, never an assigned location or imputed feature."""
    state = entry[2]
    native = state.source_frames[0]
    source.acs_native.verify_acs_native_coverage(state.native[0], native)
    origin = observations.origins.loc[observations.origins.source.eq("acs")]
    require(len(origin) > 0, "ACS_EMPTY")
    people = native.person.set_index("person_id", drop=False)
    require(
        people.index.is_unique and set(people.index) == set(origin.native_person_id),
        "ACS_SOURCE_ROSTER",
    )
    people = people.reindex(origin.native_person_id.to_numpy())
    age = people.age.to_numpy(dtype="float64")
    sex = people.SEX.to_numpy(dtype="float64")
    require(
        np.array_equal(age, people.AGEP.to_numpy())
        and np.isfinite(age).all()
        and np.isin(sex, (1, 2)).all()
        and pd.api.types.is_bool_dtype(people.is_female.dtype)
        and people.is_female.notna().all()
        and np.array_equal(people.is_female.to_numpy(), sex == 2),
        "ACS_PREDICTOR_UNKNOWN_OR_CHANGED",
    )
    households = native.table("household").set_index("household_id")
    require(households.index.is_unique, "ACS_HOUSEHOLD_AXIS")
    codes = []
    for native_id in people.person_household_id:
        require(native_id in households.index, "ACS_HOUSEHOLD_JOIN")
        token = households.loc[native_id, "ST"]
        require(
            type(token) is str
            and re.fullmatch(r"[0-9]{2}", token, re.ASCII) is not None,
            "ACS_STATE_LITERAL",
        )
        code = int(token)
        require(code in US_STATE_NUMERIC_FIPS_TO_POSTAL, "ACS_STATE_UNKNOWN")
        codes.append(code)
    return pd.DataFrame(
        {FEATURES[0]: age, FEATURES[1]: (sex == 2).astype(float), FEATURES[2]: codes},
        index=origin.index,
        dtype="float64",
    )


@dataclass(frozen=True)
class QualifiedHealthCompletion:
    """Private derived values; no issuance registry or receipt admission path."""

    source_frame: Frame
    donor_frame: Frame
    columns: pd.DataFrame
    matrix: bytes
    projection: bytes
    evidence: dict


def assemble(source_frame, recipient_features, *, evidence):
    """Pure checked projection builder, not an authority-bearing entry point."""
    require(
        source_frame.weights_for("household").kind is WeightKind.DESIGN,
        "DONOR_DESIGN_WEIGHTS",
    )
    require(
        tuple(recipient_features.columns) == FEATURES
        and np.isfinite(recipient_features.to_numpy()).all(),
        "RECIPIENT_PREDICTORS",
    )
    columns = model_columns(source_frame.person)
    keep = columns[ELIGIBLE].to_numpy()
    require(keep.any(), "NO_ELIGIBLE_DONORS")
    tables = {
        entity: source_frame.table(entity).copy(deep=True)
        for entity in source_frame.entities
    }
    require(not set(columns) & set(tables["person"]), "MODEL_COLUMN_COLLISION")
    for name in columns:
        tables["person"][name] = columns[name].to_numpy(copy=True)
    modeled = Frame(
        tables,
        source_frame.schema,
        dict(source_frame._weights),
        source_frame.strata,
        metadata=source_frame.metadata,
        mass_log=source_frame.mass_log,
    )
    donor = modeled.select(keep)
    matrix = model_input.encode_recipient_matrix(
        recipient_features,
        entity="person",
        entity_ids=recipient_features.index.to_numpy(dtype="int64"),
    )
    evidence = {
        **evidence,
        "protocol": PROTOCOL,
        "features": list(FEATURES),
        "targets": list(TARGETS),
        "asec_coverage_observation_year": 2025,
        "acs_coverage_observation_year": 2024,
        "temporal_equivalence_claim": False,
        "scientific_qualification": "pending",
        "source_admission_issued": False,
        "release_eligible": False,
        "donor_persons": int(keep.sum()),
        "excluded_donor_persons": int((~keep).sum()),
        "recipient_original_persons": len(recipient_features),
        "unknown_donor_targets": {t: int(columns[t].isna().sum()) for t in TARGETS},
        "unknown_donor_predictors": {
            f: int(source_frame.person[f].isna().sum()) for f in FEATURES
        },
        "source_frame_sha256": source._frame_identity(source_frame),
        "donor_frame_sha256": source._frame_identity(donor),
        "model_columns_sha256": codec.sha(columns.to_json(orient="table").encode()),
        "recipient_matrix_sha256": codec.sha(matrix),
    }
    return QualifiedHealthCompletion(
        source_frame, donor, columns, matrix, codec.encode_json(evidence), evidence
    )


def seal(value):
    require(type(value) is QualifiedHealthCompletion, "QUALIFIED_TYPE")
    return (
        source._frame_identity(value.source_frame),
        source._frame_identity(value.donor_frame),
        value.columns.to_json(orient="table").encode(),
        value.matrix,
        value.projection,
        codec.encode_json(value.evidence),
    )


def qualify_current_survey_health_completion(preparation, observations):
    """Borrow genuine full ASEC sources and selected original ACS predictors."""
    require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    require(
        type(observations) is literals.QualifiedSurveyHealthCoverage, "OBSERVATION_TYPE"
    )
    entry = preparation._checked()
    fresh = literals.qualify_current_survey_health(preparation)
    require(
        observations.source_frame is entry[2].frame
        and fresh.projection == observations.projection
        and fresh.origins.equals(observations.origins)
        and fresh.raw.equals(observations.raw)
        and codec.encode_json(fresh.evidence)
        == codec.encode_json(observations.evidence),
        "OBSERVATION_BINDING",
    )
    frame, evidence = _full_source(preparation, entry)
    features = _acs_features(entry, fresh)
    result = assemble(
        frame,
        features,
        evidence={
            **evidence,
            "preparation_sha256": codec.sha(entry[1]),
            "health_projection_sha256": codec.sha(fresh.projection),
        },
    )
    stamp = seal(result)
    require(preparation._checked() is entry, "FINAL_PREPARATION")
    source._pure_final(entry[2])
    require(seal(result) == stamp, "FINAL_PROJECTION")
    return result
