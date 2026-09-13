"""Source-qualified household housing participation on the common survey graph.

Original interview observations remain nullable and distinct from the completed
participation input. The country model owns every award and dollar allocation.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType

import numpy as np
import pandas as pd

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import model_input
from microcosm.frame import Frame, WeightKind
from microcosm.graph import Owned
from microcosm.graph import population as population_ops

from . import asec_demographic_source as demographics
from . import asec_housing_status as status
from . import asec_original_household_weights as household_source
from . import current_survey_health_source as source_reader
from . import current_survey_predictors as predictors
from . import graph_full_puf_enrichment as physical
from . import graph_survey_puf55 as parent_host
from . import housing_participation as participation
from . import support_provenance as provenance

source = predictors.source
PROTOCOL = "microcosm.us.current-survey-housing.v1"
SEED = 580
TARGET = "survey_housing_target_receipt"
FEATURES = (
    "housing_predictor_head_age",
    "housing_predictor_household_size",
    "housing_predictor_employment_income",
    "housing_predictor_self_employment_income",
)
ASEC_HOUSEHOLD_COLUMNS = (
    "H_SEQ",
    "H_TENURE",
    "H_HHTYPE",
    "HRHTYPE",
    "H_LIVQRT",
    "HPUBLIC",
    "HLORENT",
    "I_HPUBLI",
    "I_HLOREN",
)
ASEC_PERSON_COLUMNS = ("PERIDNUM", "PH_SEQ", "A_LINENO", "A_EXPRRP", "P_SEQ")
ACS_HOUSEHOLD_COLUMNS = ("SERIALNO", "TYPEHUGQ", "NP", "TEN")
ACS_PERSON_COLUMNS = ("SERIALNO", "SPORDER", "RELSHIPP")
SPM_OUTPUTS = ("receives_housing_assistance", "takes_up_housing_assistance_if_eligible")
OWNER_ASSUMPTION = (
    "A5",
    "The modeled public/lower-rent assistance family excludes independently "
    "observed owner households. This is a model-family exclusion, not observed "
    "nonreceipt, and never overrides a positive or conflicting source response.",
)


def require(condition, reason):
    if not condition:
        raise ValueError("CURRENT_SURVEY_HOUSING_" + reason)


def live():
    """Capture callables and closed configuration before any owner callback."""
    result = []
    for module in (sys.modules[__name__], participation, status, source_reader):
        for name, item in vars(module).items():
            if type(item) is FunctionType:
                result.append((module.__name__, name, source._function_seal(item)))
            elif isinstance(item, type) and item.__module__ == module.__name__:
                result.append((module.__name__, name, item))
    result.append(
        (
            PROTOCOL,
            SEED,
            TARGET,
            FEATURES,
            ASEC_HOUSEHOLD_COLUMNS,
            ASEC_PERSON_COLUMNS,
            ACS_HOUSEHOLD_COLUMNS,
            ACS_PERSON_COLUMNS,
            SPM_OUTPUTS,
            OWNER_ASSUMPTION,
            participation.HOUSING_PARTICIPATION_ASSUMPTIONS,
            status.RAW_COLUMNS,
            status.DERIVED_COLUMNS,
            tuple(demographics.A_EXPRRP.named_codes.items()),
            demographics.HOUSEHOLD_REFERENCE_CODES,
            demographics.ACS_OBSERVED_REFERENCE_CODE,
        )
    )
    return tuple(result)


@dataclass(frozen=True)
class QualifiedSurveyHousing:
    projection: bytes
    evidence: dict
    source_frame: Frame
    origins: pd.DataFrame
    native: pd.DataFrame
    donor_frame: Frame | None
    donor_columns: pd.DataFrame
    keep: np.ndarray
    features: tuple[str, ...]
    matrix: bytes | None


def seal(value):
    require(type(value) is QualifiedSurveyHousing, "QUALIFIED_TYPE")
    return (
        value.projection,
        codec.encode_json(value.evidence),
        value.features,
        source._frame_identity(value.source_frame),
        codec.encode_json(
            {
                "table": value.origins.to_dict(orient="tight"),
                "dtypes": [repr(dtype) for dtype in value.origins.dtypes],
                "index_dtype": repr(value.origins.index.dtype),
            }
        ),
        *(physical._table_stamp(t) for t in (value.native, value.donor_columns)),
        None
        if value.donor_frame is None
        else source._frame_identity(value.donor_frame),
        value.keep.dtype.str,
        value.keep.tobytes(),
        value.matrix,
    )


def _scan(stream, columns, *, key, wanted, maximum):
    """Exhaust captured literal bytes; retain only exact selected source keys."""
    require(
        source_reader.source_csv_builtin.capture_csv_reader(csv) is not None,
        "CSV_READER",
    )
    result, count, seen, header = {}, 0, set(), None
    for raw in source_reader.records._records(stream):
        row = source_reader.records._decode_record(raw, first=header is None)
        if header is None:
            header = row
            require(
                bool(header)
                and len(set(header)) == len(header)
                and set(columns) <= set(header),
                "LITERAL_HEADER",
            )
            positions = [header.index(c) for c in columns]
            continue
        count += 1
        require(count <= maximum and len(row) == len(header), "LITERAL_ROWS")
        record = dict(zip(columns, (row[i] for i in positions), strict=True))
        require(
            all(type(v) is str and len(v) <= 64 for v in record.values()),
            "LITERAL_BOUND",
        )
        identity = key(record)
        require(identity not in seen, "LITERAL_DUPLICATE")
        seen.add(identity)
        if identity in wanted:
            result[identity] = record
    require(header is not None, "LITERAL_HEADER")
    return result, count


def _integer(token, *, maximum, minimum=0):
    require(
        type(token) is str and re.fullmatch(r"[0-9]{1,22}", token) is not None,
        "LITERAL_INTEGER",
    )
    number = int(token)
    require(minimum <= number <= maximum, "LITERAL_DOMAIN")
    return number


def _hh_key(row):
    return _integer(row["H_SEQ"], minimum=1, maximum=99999)


def _person_key(row):
    return source_reader._key(row, "asec")


def _acs_person_key(row):
    return source_reader._key(row, "acs")


def _original_columns(preparation):
    """Read the retained original members, never a native H5 or engine output."""
    before = live()
    entry = preparation._checked()
    require(live() == before, "CODE_CHANGED_DURING_OWNER")
    state = entry[2]
    document = json.loads(entry[1])
    person_origins, person_keys = source_reader._origins(state.frame, document)
    origins = pd.DataFrame(document["origins"]["households"]).set_index("household_id")
    require(
        origins.index.is_unique
        and np.array_equal(
            origins.index.to_numpy(),
            state.frame.table("household").household_id.to_numpy(),
        ),
        "HOUSEHOLD_ORIGIN_AXIS",
    )
    native = state.native[1]
    issued = source.asec_native._ISSUED.get(id(native))
    require(
        issued is not None and issued[0]() is native and native.payload == issued[1],
        "ASEC_NATIVE_ISSUANCE",
    )
    asec_keys = {
        int(r.raw_native_id) for r in origins.itertuples() if r.source == "asec"
    }
    expected_member = issued[2].fields.document["members"]
    require(len(expected_member) == 1, "HOUSEHOLD_MEMBER_ROSTER")
    pins = tuple(p for p in household_source._registry() if p.income_year == 2024)
    require(len(pins) == 1, "HOUSEHOLD_PIN")
    hpin = pins[0]
    require(
        hpin.member_sha256 == expected_member[0]["member_sha256"]
        and hpin.canonical_member_id == expected_member[0]["canonical_member_id"]
        and hpin.archive_sha256 == expected_member[0]["archive_sha256"],
        "HOUSEHOLD_MEMBER_BINDING",
    )
    ppins = tuple(p for p in source_reader.asec._MEMBER_PINS if p[0] == 2024)
    require(len(ppins) == 1, "PERSON_PIN")
    year, member, archive_sha, pdigest, prows, psize = ppins[0]
    retained = [
        r for r in issued[2].coverage.receipt["sources"] if r["source_year"] == year
    ]
    require(
        len(retained) == 1
        and all(
            retained[0][k] == v
            for k, v in (
                ("member", member),
                ("archive_sha256", archive_sha),
                ("member_sha256", pdigest),
                ("rows", prows),
                ("member_bytes", psize),
            )
        ),
        "PERSON_MEMBER_BINDING",
    )
    acs_owned = source.acs_catalogue._lookup(state.catalogues[0])
    acs_document = json.loads(acs_owned.receipt)
    selected = {}
    with tempfile.TemporaryDirectory(prefix="microcosm-housing-originals-") as tmp:
        directory = Path(tmp)
        path = directory / "hhpub25.csv"
        require(
            household_source._capture_owner._snapshot(
                state.root / "asec" / "hhpub25.csv", path, size=hpin.size_bytes
            )
            == hpin.member_sha256,
            "HOUSEHOLD_CAPTURE",
        )
        with path.open("rb") as stream:
            selected["asec_household"], count = _scan(
                stream,
                ASEC_HOUSEHOLD_COLUMNS,
                key=_hh_key,
                wanted=asec_keys,
                maximum=hpin.rows,
            )
        require(
            count == hpin.rows
            and set(selected["asec_household"]) == asec_keys
            and source_reader.housing._persisted_sha(path, hpin.size_bytes)
            == hpin.member_sha256,
            "ASEC_HOUSEHOLD_ROSTER",
        )
        path = directory / member
        source_reader.asec._capture(
            state.root / "asec" / member,
            path,
            size=psize,
            digest=pdigest,
            budget=[source_reader.asec._BODY_MAX],
        )
        wanted = {k for s, k in person_keys if s == "asec"}
        with path.open("rb") as stream:
            selected["asec_person"], count = _scan(
                stream,
                ASEC_PERSON_COLUMNS,
                key=_person_key,
                wanted=wanted,
                maximum=prows,
            )
        require(
            count == prows
            and set(selected["asec_person"]) == wanted
            and source_reader.housing._persisted_sha(path, psize) == pdigest,
            "ASEC_PERSON_ROSTER",
        )
        for role, columns, key, wanted in (
            (
                "household",
                ACS_HOUSEHOLD_COLUMNS,
                lambda r: r["SERIALNO"],
                set(origins.loc[origins.source.eq("acs"), "raw_native_id"]),
            ),
            (
                "person",
                ACS_PERSON_COLUMNS,
                _acs_person_key,
                {k for s, k in person_keys if s == "acs"},
            ),
        ):
            apins = tuple(p for p in acs_owned.pins if p[0] == role)
            require(len(apins) == 1, "ACS_PIN")
            _, name, digest, size = apins[0]
            path = directory / name
            require(
                source_reader.housing._copy(
                    state.root / "acs" / name, path, size, exact_size=size
                )
                == digest,
                "ACS_CAPTURE",
            )
            records, count = {}, 0
            maximum = acs_document["counts"][
                "people" if role == "person" else "households"
            ]
            with zipfile.ZipFile(path) as archive:
                members, prefix = source_reader.records._members(archive, role)
                for item in members:
                    with archive.open(item) as stream:
                        if item.filename.casefold().startswith(prefix):
                            part, n = _scan(
                                stream,
                                columns,
                                key=key,
                                wanted=wanted,
                                maximum=maximum - count,
                            )
                            require(not set(records) & set(part), "ACS_DUPLICATE")
                            records.update(part)
                            count += n
                        else:
                            while stream.read(65536):
                                pass
            require(
                count == maximum
                and set(records) == wanted
                and source_reader.housing._persisted_sha(path, size) == digest,
                "ACS_SELECTED_ROSTER",
            )
            selected["acs_" + role] = records
    require(live() == before, "CODE_CHANGED_DURING_CAPTURE")
    require(preparation._checked() is entry, "PREPARATION_CHANGED")
    require(live() == before, "CODE_CHANGED_DURING_OWNER")
    source._pure_final(state)
    require(
        live() == before
        and source._ISSUED.get(id(preparation)) is entry
        and source.asec_native._ISSUED.get(id(native)) is issued,
        "FINAL_OWNER_OR_CODE",
    )
    return (
        state.frame,
        origins,
        person_origins,
        person_keys,
        selected,
        {
            "preparation_sha256": codec.sha(entry[1]),
            "asec_native_sha256": codec.sha(issued[1]),
            "asec_household_member_sha256": hpin.member_sha256,
            "asec_person_member_sha256": pdigest,
            "acs_catalogue_sha256": codec.sha(acs_owned.receipt),
        },
    )


def _source_values(frame, origins, person_origins, person_keys, selected):
    """Qualify housing universe and named reference roles on original support."""
    ids = origins.index
    native = pd.DataFrame(index=ids.copy())
    native["housing_source"] = pd.array(origins.source, dtype="string")
    for name in ("source_year", "survey_year"):
        native["housing_" + name] = origins[name].to_numpy(dtype="int64")
    native["housing_observed_receipt"] = pd.array([pd.NA] * len(ids), dtype="boolean")
    native["housing_observed_receipt__known"] = False
    native["housing_receipt"] = pd.array([pd.NA] * len(ids), dtype="boolean")
    native["housing_receipt__origin"] = pd.array(
        ["unresolved"] * len(ids), dtype="string"
    )
    native["housing_participation_universe"] = pd.array(
        [pd.NA] * len(ids), dtype="string"
    )
    native["housing_source_head_person_id"] = np.full(len(ids), -1, dtype=np.int64)
    for name in (*ASEC_HOUSEHOLD_COLUMNS, *ACS_HOUSEHOLD_COLUMNS):
        native["housing_source_" + name] = pd.array([pd.NA] * len(ids), dtype="string")
    for name in status.DERIVED_COLUMNS:
        native["housing_source_" + name] = pd.array([pd.NA] * len(ids), dtype="Int64")
    heads = pd.Series(False, index=person_origins.index, dtype=bool)
    for (survey, key), pid in person_keys.items():
        row = selected[survey + "_person"][key]
        name = "A_EXPRRP" if survey == "asec" else "RELSHIPP"
        role = _integer(
            row[name],
            minimum=1 if survey == "asec" else 20,
            maximum=14 if survey == "asec" else 38,
        )
        if survey == "asec":
            require(role in demographics.A_EXPRRP.named_codes, "UNNAMED_RELATIONSHIP")
            heads.loc[pid] = role in demographics.HOUSEHOLD_REFERENCE_CODES
        else:
            heads.loc[pid] = role == demographics.ACS_OBSERVED_REFERENCE_CODE
    membership = frame.person.set_index("person_id").person_household_id
    # Build reference counts and exact integer identities once on person support.
    # Keep duplicate household keys until the cardinality check below refuses them.
    reference_members = membership.loc[heads]
    reference_counts = reference_members.value_counts()
    reference_ids = pd.Series(
        reference_members.index.to_numpy(dtype="int64"),
        index=reference_members.to_numpy(dtype="int64"),
        dtype="int64",
    )
    asec_ids = ids[origins.source.eq("asec")]
    if len(asec_ids):
        asec_rows = [
            selected["asec_household"][int(raw_id)]
            for raw_id in origins.loc[asec_ids, "raw_native_id"]
        ]
        for raw in asec_rows:
            # Preserve the original literal universe and tenure refusals.
            require(
                raw["H_HHTYPE"] == "1"
                and 1 <= _integer(raw["HRHTYPE"], maximum=10) <= 8
                and 1 <= _integer(raw["H_LIVQRT"], maximum=12) <= 7,
                "ASEC_HOUSING_UNIVERSE",
            )
            _integer(raw["H_TENURE"], minimum=1, maximum=3)
        numeric = {
            name: np.array(
                [_integer(raw[name] or "0", maximum=99999) for raw in asec_rows],
                dtype="int64",
            )
            for name in status.RAW_COLUMNS
            if name in ASEC_HOUSEHOLD_COLUMNS
        }
        numeric.update(
            household_id=asec_ids.to_numpy(dtype="int64"),
            income_year=np.full(len(asec_ids), 2024, dtype="int64"),
            survey_year=np.full(len(asec_ids), 2025, dtype="int64"),
        )
        derived = status._derive(numeric)
        observations = pd.DataFrame({"household_id": asec_ids, **derived})
        observed = participation.observed_participation(observations).receipt
        native.loc[asec_ids, "housing_observed_receipt"] = observed.array
        for name, values in derived.items():
            native.loc[asec_ids, "housing_source_" + name] = pd.array(
                values, dtype="Int64"
            )
    for hid, row in origins.iterrows():
        survey = row.source
        raw = selected[survey + "_household"][
            int(row.raw_native_id) if survey == "asec" else row.raw_native_id
        ]
        for name, token in raw.items():
            native.at[hid, "housing_source_" + name] = token
        if survey == "asec":
            tenure = _integer(raw["H_TENURE"], minimum=1, maximum=3)
            owner, gq = tenure == 1, False
            if owner:
                require(
                    native.at[hid, "housing_source_conflicts"] == 0
                    and raw["HPUBLIC"] in ("0", "")
                    and raw["HLORENT"] in ("0", ""),
                    "OWNER_SOURCE_CONFLICT",
                )
            observed = native.at[hid, "housing_observed_receipt"]
        else:
            kind = _integer(raw["TYPEHUGQ"], minimum=1, maximum=3)
            count = _integer(raw["NP"], maximum=20)
            gq = kind in (2, 3)
            require(count > 0 and (not gq or count == 1), "ACS_HOUSING_UNIVERSE")
            tenure = None if gq else _integer(raw["TEN"], minimum=1, maximum=4)
            owner = tenure in (1, 2)
            observed = pd.NA
        universe = "group_quarters" if gq else "occupied_housing_unit"
        native.at[hid, "housing_participation_universe"] = universe
        require(
            reference_counts.get(hid, 0) == (0 if gq else 1), "NAMED_REFERENCE_ROLE"
        )
        if not gq:
            native.at[hid, "housing_source_head_person_id"] = int(reference_ids.at[hid])
        if pd.notna(observed):
            native.at[hid, "housing_observed_receipt__known"] = True
            native.at[hid, "housing_receipt"] = bool(observed)
            native.at[hid, "housing_receipt__origin"] = (
                "source_receipt" if observed else "source_nonreceipt"
            )
        elif gq or owner:
            native.at[hid, "housing_receipt"] = False
            native.at[hid, "housing_receipt__origin"] = (
                "modeled_group_quarters_exclusion" if gq else "modeled_owner_exclusion"
            )
    return native


def _qualified_values(frame, origins, native, person_features, evidence):
    """Prepare one weighted donor row and one recipient draw per household."""
    require(
        person_features.index.is_unique
        and np.array_equal(
            person_features.index.to_numpy(), frame.person.person_id.to_numpy()
        ),
        "PERSON_FEATURE_AXIS",
    )
    ids = origins.index
    require(
        type(ids) in (pd.Index, pd.RangeIndex) and ids.dtype == np.dtype("int64"),
        "ORIGIN_HOUSEHOLD_INDEX",
    )
    membership = frame.person.set_index("person_id").person_household_id
    features = pd.DataFrame(index=ids.copy(), columns=FEATURES, dtype="float64")
    features[FEATURES[1]] = (
        membership.value_counts().reindex(ids).to_numpy(dtype="float64")
    )
    head_ids = native.housing_source_head_person_id
    hu = native.housing_participation_universe.eq("occupied_housing_unit")
    features.loc[hu, FEATURES[0]] = person_features.loc[
        head_ids[hu], predictors.FEATURES[0]
    ].to_numpy()
    for out, column in zip(FEATURES[2:], predictors.FEATURES[1:], strict=True):
        features[out] = (
            person_features[column].groupby(membership).sum().reindex(ids).to_numpy()
        )
    names = list(FEATURES)
    for column in predictors.DEMOGRAPHIC_FEATURES[len(predictors.FEATURES) :]:
        if column in person_features:
            name = column.replace("survey_predictor_", "housing_predictor_head_")
            names.append(name)
            features[name] = np.nan
            features.loc[hu, name] = person_features.loc[
                head_ids[hu], column
            ].to_numpy()
    recipient = native.housing_receipt.isna()
    donor = origins.source.eq("asec") & native.housing_observed_receipt__known
    require((~recipient | hu).all(), "RECIPIENT_UNIVERSE")
    require(
        np.isfinite(features.loc[recipient | donor].to_numpy()).all(), "FEATURE_UNKNOWN"
    )
    keep = membership.isin(ids[donor]).to_numpy(dtype=bool)
    donor_frame, matrix = None, None
    donor_columns = features.loc[ids[donor]].copy()
    donor_columns[TARGET] = native.loc[ids[donor], "housing_observed_receipt"].to_numpy(
        dtype="float64"
    )
    if recipient.any():
        require(donor.any(), "NO_QUALIFIED_DONORS")
        donor_frame = frame.select(keep)
        require(
            donor_frame.weights_for("household").kind is WeightKind.DESIGN,
            "DONOR_DESIGN_WEIGHTS",
        )
        require(
            np.array_equal(
                donor_frame.table("household").household_id.to_numpy(),
                donor_columns.index.to_numpy(),
            ),
            "DONOR_HOUSEHOLD_AXIS",
        )
        require(
            donor_frame.weights_for("household").values.sum() > 0, "DONOR_WEIGHT_MASS"
        )
        recipient_features = features.loc[ids[recipient]].copy()
        # Source preparation may compact contiguous IDs into a RangeIndex.
        # Materialize those same int64 IDs for the strict model-input codec.
        recipient_features.index = pd.Index(
            recipient_features.index.to_numpy(copy=True), name=ids.name
        )
        matrix = model_input.encode_recipient_matrix(
            recipient_features,
            entity="household",
            entity_ids=recipient_features.index.to_numpy(copy=True),
        )
    evidence = {
        **evidence,
        "protocol": PROTOCOL,
        "predictors": names,
        "donor_households": int(donor.sum()),
        "donor_persons": int(keep.sum()),
        "recipient_households": int(recipient.sum()),
        "donor_weights": "original_household_design_before_allocation_or_clones",
        "clone_draw_policy": "one_draw_per_original_household_shared_by_both_clones",
        "assumptions": [
            *participation.HOUSING_PARTICIPATION_ASSUMPTIONS,
            OWNER_ASSUMPTION,
        ],
        "source_observation": "ASEC_2025_interview_status; not_2024_annual_receipt",
        "source_observation_knownness_preserved": True,
        "PUF_values_consumed": False,
        "calculated_dollar_outputs": [],
        "source_admission_issued": False,
        "held_out_quality_verified": False,
        "release_eligible": False,
    }
    projection = codec.encode_json(
        {
            **evidence,
            "native_sha256": codec.sha(native.to_json(orient="table").encode()),
            "origins_sha256": codec.sha(origins.to_json(orient="table").encode()),
            "donor_columns_sha256": codec.sha(
                donor_columns.to_json(orient="table").encode()
            ),
            "matrix_sha256": None if matrix is None else codec.sha(matrix),
        }
    )
    return QualifiedSurveyHousing(
        projection,
        evidence,
        frame,
        origins.copy(deep=True),
        native.copy(deep=True),
        donor_frame,
        donor_columns,
        keep,
        tuple(names),
        matrix,
    )


def qualify_current_survey_housing(run):
    """Borrow the retained common PUF run; qualify source households independently."""
    before = live()
    entry = parent_host._run_entry(run)
    parent_host._pure_run(run, entry)
    require(live() == before, "CODE_CHANGED_DURING_OWNER")
    prefix = run.financial_run.prefix
    state = parent_host.financial._run_entry(run.financial_run)[2]
    base = predictors.qualify_current_survey_predictors(
        prefix.preparation,
        prefix.allocated_population,
        prefix.clone_population,
        demographic_conditioning=state.demographic_conditioning,
        geography_config=prefix.geography_config,
    )
    require(live() == before, "CODE_CHANGED_DURING_OWNER")
    require(
        base.projection == run.financial_run.projection
        and base.matrix == run.financial_run.matrix,
        "PARENT_PREDICTORS",
    )
    frame, origins, porigins, keys, selected, evidence = _original_columns(
        prefix.preparation
    )
    require(live() == before, "CODE_CHANGED_DURING_OWNER")
    native = _source_values(frame, origins, porigins, keys, selected)
    names = predictors.feature_columns(base.demographic_conditioning)
    features = pd.DataFrame(index=base.origins.index, columns=names, dtype="float64")
    asec = base.origins.source.eq("asec")
    features.loc[asec] = base.donor_columns.loc[:, names]
    matrix = model_input.decode_recipient_matrix(base.matrix)
    features.loc[~asec] = matrix.features
    result = _qualified_values(
        frame,
        origins,
        native,
        features,
        {
            **evidence,
            "parent_checked_sha256": codec.sha(entry[1]),
            "predictor_projection_sha256": codec.sha(base.projection),
        },
    )
    stamp = seal(result)
    parent_host._pure_run(run, entry)
    require(live() == before and seal(result) == stamp, "FINAL_OWNER_OR_VALUES")
    return result


def output_columns(qualified, receiving):
    """Explicit output schema without issuing a fabricated model draw."""
    require(type(qualified) is QualifiedSurveyHousing, "QUALIFIED_TYPE")
    return tuple(
        Owned(
            "household",
            c,
            "bool"
            if c == "housing_receipt"
            else population_ops.token_for_dtype(qualified.native[c].dtype),
        )
        for c in qualified.native
    ) + tuple(Owned("spm_unit", c, "bool") for c in SPM_OUTPUTS)


def attach_columns(qualified, receiving, draw):
    """Join source observations and household draws, then route to assisted SPMs."""
    before, stamp = live(), seal(qualified)
    completed = qualified.native.copy(deep=True)
    if qualified.matrix is None:
        require(
            draw is None or (type(draw) is pd.DataFrame and draw.empty),
            "UNEXPECTED_DRAW",
        )
    else:
        matrix = model_input.decode_recipient_matrix(qualified.matrix)
        require(
            type(draw) is pd.DataFrame
            and draw.index.equals(matrix.features.index)
            and tuple(draw) == (TARGET,)
            and draw[TARGET].dtype == np.dtype("float64")
            and np.isin(draw[TARGET].to_numpy(), (0.0, 1.0)).all(),
            "DRAW_VALUES_OR_AXIS",
        )
        require(
            completed.loc[draw.index, "housing_receipt"].isna().all(),
            "DRAW_OVERWRITES_SOURCE",
        )
        completed.loc[draw.index, "housing_receipt"] = draw[TARGET].to_numpy(dtype=bool)
        completed.loc[draw.index, "housing_receipt__origin"] = (
            "modeled_from_current_asec"
        )
    require(completed.housing_receipt.notna().all(), "UNRESOLVED_MODEL_INPUT")
    completed["housing_receipt"] = completed.housing_receipt.astype(bool)
    h = receiving.table("household")
    hid = pd.Index(h.household_id.to_numpy(), name="household_id")
    original = h[provenance.support_source_id_column("household")].to_numpy()
    clone = h[provenance.support_clone_index_column("household")].to_numpy()
    native = h[provenance.spine_source_id_column("household")].to_numpy()
    channel = h[provenance.support_channel_column("household")].to_numpy()
    require(
        original.dtype == clone.dtype == native.dtype == np.dtype("int64")
        and len(original) == 2 * len(qualified.origins)
        and set(original) == set(qualified.origins.index),
        "CLONE_HOUSEHOLD_ROSTER",
    )
    expected = qualified.origins.reindex(original)
    require(
        np.array_equal(native, expected.selected_receiving_household_id.to_numpy())
        and np.array_equal(channel, expected.source.to_numpy())
        and pd.MultiIndex.from_arrays((original, clone)).is_unique
        and np.isin(clone, (0, 1)).all(),
        "CLONE_HOUSEHOLD_IDENTITY",
    )
    require(not set(completed) & set(h), "HOUSEHOLD_OWNERSHIP_COLLISION")
    for name in SPM_OUTPUTS:
        require(name not in receiving.table("spm_unit"), "SPM_OWNERSHIP_COLLISION")
    attached = completed.reindex(original).copy()
    attached.index = hid
    person = receiving.person.copy(deep=True)
    psource = person[provenance.support_source_id_column("person")]
    hsource = pd.Series(original, index=hid)
    original_people = qualified.source_frame.person.set_index("person_id")
    pclone = person[provenance.support_clone_index_column("person")]
    require(
        psource.dtype == pclone.dtype == np.dtype("int64")
        and len(person) == 2 * len(original_people)
        and set(psource) == set(original_people.index)
        and pd.MultiIndex.from_arrays((psource, pclone)).is_unique
        and np.isin(pclone, (0, 1)).all(),
        "CLONE_PERSON_ROSTER",
    )
    expected_people = original_people.reindex(psource.to_numpy())
    require(
        np.array_equal(
            person.person_household_id.map(hsource).to_numpy(),
            expected_people.person_household_id.to_numpy(),
        )
        and np.array_equal(
            pclone.to_numpy(),
            person.person_household_id.map(pd.Series(clone, index=hid)).to_numpy(),
        ),
        "CLONE_PERSON_HOUSEHOLD",
    )
    for key in (
        provenance.spine_source_id_column("person"),
        provenance.support_channel_column("person"),
    ):
        require(
            np.array_equal(person[key].to_numpy(), expected_people[key].to_numpy()),
            "CLONE_PERSON_NATIVE_OR_CHANNEL",
        )
    expected_head = person.person_household_id.map(hsource).map(
        completed.housing_source_head_person_id
    )
    actual_head = psource.to_numpy() == expected_head.to_numpy()
    if "is_household_head" in person:
        observed_head = person.is_household_head
        known_head = observed_head.notna().to_numpy()
        require(
            (
                observed_head.dtype == np.dtype("bool")
                or isinstance(observed_head.dtype, pd.BooleanDtype)
            )
            and np.array_equal(
                observed_head.loc[known_head].to_numpy(dtype=bool),
                actual_head[known_head],
            ),
            "EXISTING_HEAD_CONFLICT",
        )
    person["is_household_head"] = actual_head
    tables = {e: receiving.table(e).copy(deep=True) for e in receiving.entities}
    tables["person"] = person
    tables["household"]["housing_participation_universe"] = (
        attached.housing_participation_universe.to_numpy()
    )
    routing = Frame(
        tables,
        receiving.schema,
        dict(receiving._weights),
        receiving.strata,
        metadata=receiving.metadata,
        mass_log=receiving.mass_log,
    )
    flags = participation.route_participation(routing, attached.housing_receipt)
    require(
        live() == before and seal(qualified) == stamp, "ATTACH_CODE_OR_VALUES_CHANGED"
    )
    return {
        **{("household", c): attached[c] for c in attached},
        **{("spm_unit", c): flags[c] for c in SPM_OUTPUTS},
    }
