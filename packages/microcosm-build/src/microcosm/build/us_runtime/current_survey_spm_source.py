"""Borrow annual SPM evidence from a live native survey preparation.

This module does not issue source authority, fetch data, construct a country
engine, reconstruct units, or qualify a release. Detached tables/receipts are
descriptive only; consumers must retain and revalidate the originating owner.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import tempfile
import weakref
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import CodeType, FunctionType

import pandas as pd

from microcosm.build import acs_spm_scope as acs
from microcosm.build import spm_input_contract as contract
from microcosm.build.spm_input_contract import INCLUDED, UNRESOLVED

from . import current_survey_health_source as original
from . import current_survey_hours_source as seals
from . import current_survey_spm_projection as projection
from . import source_csv_builtin
from . import spm_role_source as asec_roles
from . import survey_population_preparation as source

PROTOCOL = "microcosm.us.current-survey-spm-source.v1"
STRING = pd.StringDtype(storage="python", na_value=pd.NA)
_ISSUED = weakref.WeakKeyDictionary()
_ASEC_FIELDS = (
    *original.ASEC_KEYS,
    "A_AGE",
    "SPM_ID",
    "SPM_HEAD",
    "A_FAMTYP",
    "A_FAMREL",
    "SPM_HAGE",
    "SPM_NUMADULTS",
    "SPM_NUMKIDS",
    "SPM_NUMPER",
)


@dataclass(frozen=True)
class AsecSpmScopePolicy:
    """One closed, reviewed source-membership rule; not caller-written evidence."""

    name: str
    survey_year: int
    income_year: int
    reference: str
    reference_sha256: str
    section: str


ASEC_2025_INCOME_2024_SPM_POLICY = AsecSpmScopePolicy(
    "census_asec2025_person_membership_income2024_spm_v1",
    2025,
    2024,
    "https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar25.pdf",
    "7f0cb9f791f2737ad12fd552b8cb201f4d1de266eac160322597344e73499fb4",
    "Appendix D, D-3 (PDF page 164): SPM All People, No restriction",
)


def _require(condition, code):
    if not condition:
        raise ValueError("NATIVE_SPM_SOURCE_" + code)


def _modules():
    return (
        sys.modules[__name__],
        acs,
        asec_roles,
        original,
        seals,
        projection,
        projection.attachment,
        projection.provenance,
        contract,
    )


def _implementation():
    return {
        module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in _modules()
    }


def _live():
    result = {}
    for module in _modules():
        for name, value in vars(module).items():
            if isinstance(value, FunctionType):
                result[module.__name__, name] = source._function_seal(value)
            elif isinstance(value, type) and value.__module__ == module.__name__:
                result[module.__name__, name] = value
                for key, function in vars(value).items():
                    if isinstance(function, property):
                        function = function.fget
                    if isinstance(function, FunctionType):
                        result[module.__name__, name, key] = source._function_seal(
                            function
                        )
    return result, (
        PROTOCOL,
        asdict(ASEC_2025_INCOME_2024_SPM_POLICY),
        _ASEC_FIELDS,
        repr(STRING),
        asec_roles.SPM_ROLE_RULE,
        asec_roles.SPM_ADULT_RULE,
        asec_roles._COUNTS,
        acs.ACS_SPM_AUTHORITY_ORDER,
        tuple(acs.ACS_SPM_ASSUMPTION_AUTHORITY.items()),
        tuple(acs.ACS_SPM_SECONDARY_AUTHORITY.items()),
        tuple(acs.ACS_SPM_LINK_AUTHORITY.items()),
        acs.ACS_SPM_OUTSIDE_AUTHORITY,
        acs.ACS_SPM_PROFILE_NAME,
        acs.ACS_SPM_METHODOLOGY,
        acs.ACS_SPM_LINEAGE,
        INCLUDED,
        UNRESOLVED,
        tuple(
            (module.__name__, name, getattr(module, name))
            for module in (contract, acs, projection)
            for name in (
                "INCLUDED",
                "OUTSIDE",
                "UNRESOLVED",
                "ROLE_INPUT",
                "UNIVERSE_INPUT",
                "UNIVERSE_STATUSES",
            )
            if hasattr(module, name)
        ),
        tuple(
            (name, value)
            for name, value in vars(projection.provenance).items()
            if name.isupper() and type(value) in (str, int)
        ),
        _REVALIDATE_CODE,
    )


def _policy(policy):
    _require(
        policy is None or policy is ASEC_2025_INCOME_2024_SPM_POLICY,
        "ASEC_SCOPE_POLICY",
    )


def _callback_seal(callback):
    # Keep the issuer's detached closure seal outside the editable closure.
    # Function identities become numeric IDs to avoid retaining the result via
    # the callback's self-reference in this weak issuance registry.
    def detach(value):
        if isinstance(value, FunctionType):
            return ("function", id(value), value.__code__)
        if isinstance(value, tuple):
            return tuple(detach(item) for item in value)
        return value

    return detach(source._function_seal(callback))


def _result_tables(value):
    return (
        value.origins,
        value.asec_raw,
        value.unit_evidence,
        value.roles.to_frame(),
        value.unit_status.to_frame(),
    )


def _check_retained_output(candidate, issued):
    """Pure final fence against mutations during owner/source revalidation."""
    _require(_live() == _LIVE, "IMPLEMENTATION_CHANGED")
    _require(
        type(candidate) is QualifiedNativeSpmInputs
        and issued is not None
        and _ISSUED.get(candidate) is issued
        and type(candidate._revalidate) is FunctionType
        and candidate._revalidate.__code__ is _REVALIDATE_CODE
        and issued[0]() is candidate._revalidate
        and issued[1] == _callback_seal(candidate._revalidate),
        "RETAINED_OWNER_REQUIRED",
    )
    _require(
        candidate.receipt == issued[2]
        and tuple(seals._table_seal(table) for table in _result_tables(candidate))
        == issued[3],
        "PROJECTION_CHANGED",
    )


def _scan_asec(stream, wanted, maximum):
    """Exhaust authenticated bytes, retaining only selected original people."""
    _require(source_csv_builtin.capture_csv_reader(csv) is not None, "CSV_BINDING")
    header, count, selected = None, 0, {}
    for raw in original.records._records(stream):
        values = original.records._decode_record(raw, first=header is None)
        if header is None:
            header = values
            _require(
                bool(header)
                and all(header)
                and len(set(header)) == len(header)
                and set(_ASEC_FIELDS) <= set(header),
                "ASEC_HEADER",
            )
            positions = [header.index(c) for c in _ASEC_FIELDS]
            continue
        count += 1
        _require(count <= maximum and len(values) == len(header), "ASEC_ROW_SHAPE")
        row = dict(zip(_ASEC_FIELDS, (values[p] for p in positions), strict=True))
        _require(all(len(v) <= 64 for v in row.values()), "ASEC_TOKEN_BOUND")
        key = original._key(row, "asec")
        if key in wanted:
            _require(key not in selected, "ASEC_DUPLICATE_KEY")
            selected[key] = row
    _require(header is not None and count == maximum, "ASEC_ROW_COUNT")
    _require(set(selected) == wanted, "ASEC_SELECTED_ROSTER")
    # A matching count is not proof of complete source membership. Exhaust the
    # same captured bytes again and retain all exact keys in selected units.
    rosters = {row["SPM_ID"]: set() for row in selected.values()}
    stream.seek(0)
    second_count = 0
    for position, raw in enumerate(original.records._records(stream)):
        values = original.records._decode_record(raw, first=position == 0)
        if position == 0:
            _require(values == header, "ASEC_HEADER_CHANGED")
            continue
        second_count += 1
        _require(
            second_count <= maximum and len(values) == len(header), "ASEC_ROW_SHAPE"
        )
        row = dict(zip(_ASEC_FIELDS, (values[p] for p in positions), strict=True))
        _require(all(len(v) <= 64 for v in row.values()), "ASEC_TOKEN_BOUND")
        if row["SPM_ID"] in rosters:
            key = original._key(row, "asec")
            _require(key not in rosters[row["SPM_ID"]], "ASEC_DUPLICATE_UNIT_MEMBER")
            rosters[row["SPM_ID"]].add(key)
    _require(second_count == maximum, "ASEC_ROW_COUNT")
    return selected, {key: frozenset(value) for key, value in rosters.items()}


def _integer(token, low, high, *, nullable=False):
    if nullable and token == "":
        return pd.NA
    _require(
        type(token) is str and re.fullmatch(r"[0-9]+", token, re.ASCII) is not None,
        "ASEC_INTEGER_TOKEN",
    )
    value = int(token)
    _require(low <= value <= high, "ASEC_INTEGER_RANGE")
    return value


def _asec_projection(frame, origins, keys, selected, source_rosters, policy):
    """Pure reconciliation; only the enclosing retained owner grants provenance."""
    _policy(policy)
    ids = origins.index[origins.source.eq("asec")]
    people = frame.person.set_index("person_id", drop=False).loc[ids]
    rows = {pid: selected[key] for (arm, key), pid in keys.items() if arm == "asec"}
    _require(set(rows) == set(ids), "ASEC_ORIGIN_COVERAGE")
    _require(
        all(
            original._key(selected[key], "asec") == key
            for arm, key in keys
            if arm == "asec"
        ),
        "ASEC_SOURCE_COORDINATE_CHANGED",
    )
    raw = pd.DataFrame(
        {
            name: pd.array([rows[pid][name] for pid in ids], dtype=STRING)
            for name in _ASEC_FIELDS
        },
        index=ids.copy(),
    )
    values = raw.copy()
    for name, limits in {
        "A_AGE": (0, 85),
        "SPM_HAGE": (15, 85),
        "SPM_HEAD": (0, 1),
        "A_FAMTYP": (1, 5),
        "A_FAMREL": (0, 4),
        "SPM_NUMADULTS": (0, 20),
        "SPM_NUMKIDS": (0, 20),
        "SPM_NUMPER": (1, 20),
    }.items():
        nullable = name in ("SPM_HEAD", "A_FAMTYP", "A_FAMREL")
        values[name] = pd.array(
            [_integer(v, *limits, nullable=nullable) for v in raw[name]], dtype="Int64"
        )
    _require(raw.SPM_ID.str.fullmatch(r"[0-9]+", flags=re.ASCII).all(), "ASEC_SPM_KEY")
    _require(
        people.index.equals(ids) and values.A_AGE.eq(people.A_AGE).all(),
        "ASEC_OBSERVED_AGE",
    )
    _require(raw.PERIDNUM.is_unique, "ASEC_DUPLICATE_PERSON")
    values["person_spm_unit_id"] = people.person_spm_unit_id
    values["person_household_id"] = people.person_household_id
    missing = values[["SPM_HEAD", "A_FAMTYP", "A_FAMREL"]].isna().any(axis=1)
    values["_role"] = asec_roles.independent_minor_role(values).astype("boolean")
    values.loc[missing, "_role"] = pd.NA
    values["_age_only_adult"] = values.A_AGE.ge(18)
    values["_adult"] = values._age_only_adult | (values.A_AGE.ge(15) & values._role)
    groups = values.groupby("person_spm_unit_id", sort=False, dropna=False)
    _require(
        groups[["SPM_ID", "PH_SEQ", "person_household_id"]].nunique().eq(1).all().all()
        and values.groupby("SPM_ID").person_spm_unit_id.nunique().eq(1).all(),
        "ASEC_UNIT_BIJECTION",
    )
    evidence = []
    for uid, unit in groups:
        actual_keys = frozenset(original._key(rows[pid], "asec") for pid in unit.index)
        _require(
            source_rosters.get(unit.SPM_ID.iloc[0]) == actual_keys,
            "ASEC_EXACT_SOURCE_UNIT_ROSTER",
        )
        _require(
            unit[list(asec_roles._COUNTS)].nunique(dropna=False).eq(1).all()
            and unit.SPM_NUMPER.eq(len(unit)).all()
            and (unit.SPM_NUMADULTS + unit.SPM_NUMKIDS).eq(len(unit)).all(),
            "ASEC_COMPLETE_UNIT",
        )
        unresolved_role = bool(unit._role.isna().any())
        if not unresolved_role:
            asec_roles._reconcile_units(unit, "person_spm_unit_id", "Native ASEC")
        status = INCLUDED if policy is not None and not unresolved_role else UNRESOLVED
        reason = (
            "missing_role_source"
            if unresolved_role
            else "annual_scope_policy_not_admitted"
            if policy is None
            else "authenticated_asec2025_person_membership_income2024"
        )
        evidence.append(
            {
                "spm_unit_id": uid,
                "source": "asec",
                "authority": "documented_relationship_inference",
                "status": status,
                "reason": reason,
                "source_membership": "authenticated_complete_census_person_unit",
                "source_spm_unit_id": unit.SPM_ID.iloc[0],
                "role_rule": asec_roles.SPM_ROLE_RULE,
            }
        )
    return raw, values._role.rename("role"), evidence


def _combine(frame, origins, document, classified, asec_raw, asec_role, asec_units):
    maps = {}
    for entity in ("person", "spm_unit", "household"):
        records = document["origins"]["entities"][entity]
        maps[entity] = {(arm, native): current for current, arm, native in records}
        _require(len(maps[entity]) == len(records), "ENTITY_ORIGIN_BIJECTION")
    people = frame.person.set_index("person_id", drop=False)
    _require(people.index.equals(origins.index), "PERSON_AXIS")
    role = pd.Series(pd.NA, index=origins.index.copy(), dtype="boolean", name="role")
    assigned = set()
    for row in classified.roles:
        pid = maps["person"]["acs", row.person_id]
        _require(
            pid not in assigned
            and origins.at[pid, "source"] == "acs"
            and people.at[pid, "person_spm_unit_id"]
            == maps["spm_unit"]["acs", row.spm_unit_id]
            and people.at[pid, "person_household_id"]
            == maps["household"]["acs", row.household_id],
            "ACS_NATIVE_MEMBERSHIP",
        )
        role.at[pid] = row.value
        assigned.add(pid)
    _require(
        assigned == set(origins.index[origins.source.eq("acs")]), "ACS_ROLE_COVERAGE"
    )
    role.loc[asec_role.index] = asec_role
    unit_rows = list(asec_units)
    for unit in classified.units:
        uid = maps["spm_unit"]["acs", unit.spm_unit_id]
        expected = {maps["person"]["acs", pid] for pid in unit.person_ids}
        _require(
            set(people.index[people.person_spm_unit_id.eq(uid)]) == expected,
            "ACS_UNIT_COVERAGE",
        )
        unit_rows.append(
            {
                "spm_unit_id": uid,
                "source": "acs",
                "authority": unit.authority,
                "status": unit.status,
                "reason": unit.reason,
                "source_membership": "retained_native_acs_construction",
                "source_spm_unit_id": unit.canonical_id,
                "role_rule": "retained_acs_role_rules",
            }
        )
    units = pd.DataFrame(unit_rows).set_index("spm_unit_id")
    order = pd.Index(frame.table("spm_unit").spm_unit_id, name="spm_unit_id")
    _require(
        units.index.is_unique and set(units.index) == set(order), "UNIT_SCOPE_COVERAGE"
    )
    units = units.loc[order]
    status = units.status.astype(STRING).rename("status")
    return role, status, units


@dataclass(frozen=True, eq=False)
class QualifiedNativeSpmInputs:
    source_frame: object
    origins: pd.DataFrame
    roles: pd.Series
    unit_status: pd.Series
    unit_evidence: pd.DataFrame
    asec_raw: pd.DataFrame
    receipt: bytes
    _receiving_run: object = field(default=None, repr=False, compare=False)
    _revalidate: object = field(default=None, repr=False, compare=False)

    def validate(self):
        check = _check_retained_output
        issued = _ISSUED.get(self)
        check(self, issued)
        self._revalidate(self)
        # No owner callbacks or source I/O may follow this external-seal check.
        check(self, issued)


def qualify_current_survey_spm(
    preparation, *, acs_profile, asec_scope_policy, _receiving_run=None
):
    """Capture original ASEC literals and borrow ACS evidence from a real issuer.

    Calling this function reads only the retained owner's existing sources.
    No caller paths, source pins, arbitrary evidence, fetching or engine inputs
    are accepted. The lower-level seam permits narrow real-source fixture tests.
    """
    _require(_live() == _LIVE, "IMPLEMENTATION_CHANGED")
    _policy(asec_scope_policy)
    _require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    if _receiving_run is not None:
        _check_receiving(_receiving_run, preparation)
    implementation = _implementation()
    entry = preparation._checked()
    state, document = entry[2], json.loads(entry[1])
    origins, keys = original._origins(state.frame, document)
    construction = preparation.acs_spm_construction_evidence
    _require(construction is not None, "ACS_CONSTRUCTION_REQUIRED")
    classified = acs.classify_acs_spm_scope(construction, acs_profile)
    construction_seal = source._digest(construction)
    native = state.native[1]
    issued = source.asec_native._ISSUED.get(id(native))
    _require(
        issued is not None and issued[0]() is native and issued[1] == native.payload,
        "ASEC_NATIVE_ISSUANCE",
    )
    native_document = json.loads(issued[1])
    _require(
        native_document["source_year"] == native_document["income_year"] == 2024
        and native_document["survey_year"] == 2025,
        "ASEC_NATIVE_PERIOD",
    )
    coverage = issued[2].coverage
    pins = tuple(original.asec._MEMBER_PINS)
    current = tuple(pin for pin in pins if pin[0] == 2024)
    _require(len(current) == 1, "ASEC_PIN_ROSTER")
    year, member, archive_digest, digest, rows, size = current[0]
    retained = tuple(
        row for row in coverage.receipt["sources"] if row["source_year"] == year
    )
    _require(
        len(retained) == 1
        and all(
            retained[0][key] == value
            for key, value in (
                ("member", member),
                ("archive_sha256", archive_digest),
                ("member_sha256", digest),
                ("rows", rows),
                ("member_bytes", size),
            )
        ),
        "ASEC_MEMBER_BINDING",
    )
    with tempfile.TemporaryDirectory(prefix="microcosm-spm-source-") as temporary:
        captured = Path(temporary) / member
        identity = original.asec._capture(
            state.root / "asec" / member,
            captured,
            size=size,
            digest=digest,
            budget=[original.asec._BODY_MAX],
        )
        with captured.open("rb") as stream:
            selected, source_rosters = _scan_asec(
                stream, {key for arm, key in keys if arm == "asec"}, rows
            )
        _require(
            original.asec._identity(captured.stat(follow_symlinks=False)) == identity
            and original.housing._persisted_sha(captured, size) == digest,
            "ASEC_CAPTURE_CHANGED",
        )
    raw, asec_role, asec_units = _asec_projection(
        state.frame, origins, keys, selected, source_rosters, asec_scope_policy
    )
    roles, statuses, units = _combine(
        state.frame, origins, document, classified, raw, asec_role, asec_units
    )
    tables = (origins, raw, units, roles.to_frame(), statuses.to_frame())
    table_seals = tuple(seals._table_seal(table) for table in tables)
    profile_seal = source._digest(asdict(acs_profile))
    receipt = source._encode(
        {
            "protocol": PROTOCOL,
            "implementation_sha256": implementation,
            "preparation_sha256": source._sha(entry[1]),
            "asec_native_sha256": source._sha(issued[1]),
            "acs_construction_sha256": construction_seal,
            "asec_person_member_sha256": digest,
            "asec_source_rows": rows,
            "asec_selected_unit_rosters_sha256": source._digest(
                {key: sorted(value) for key, value in source_rosters.items()}
            ),
            "acs_profile": asdict(acs_profile),
            "acs_minor_partner_role": classified.minor_partner_role,
            "asec_scope_policy": None
            if asec_scope_policy is None
            else asdict(asec_scope_policy),
            "income_year": 2024,
            "asec_survey_year": 2025,
            "acs_survey_year": 2024,
            "tables_sha256": table_seals,
            "persons": len(roles),
            "spm_units": len(units),
            "missing_roles": int(roles.isna().sum()),
            "scope_counts": {
                str(k): int(v) for k, v in statuses.value_counts().items()
            },
            "source_admission_issued": False,
            "release_eligible": False,
            "age_basis": "survey_observation_year_unchanged",
            "asec_scope_basis": "authenticated_person_membership_not_role_count_reconciliation",
        }
    )
    result = QualifiedNativeSpmInputs(
        state.frame, origins, roles, statuses, units, raw, receipt, _receiving_run
    )

    def revalidate(candidate):
        _require(
            candidate is result and candidate._revalidate is revalidate,
            "PROJECTION_OBJECT_CHANGED",
        )
        if _receiving_run is not None:
            _check_receiving(_receiving_run, preparation)
        _require(preparation._checked() is entry, "PREPARATION_CHANGED")
        coverage.validate()
        _require(_implementation() == implementation, "IMPLEMENTATION_BYTES_CHANGED")
        _require(
            _live() == _LIVE and tuple(original.asec._MEMBER_PINS) == pins,
            "IMPLEMENTATION_OR_PIN_CHANGED",
        )
        _require(
            source._digest(preparation.acs_spm_construction_evidence)
            == construction_seal,
            "ACS_EVIDENCE_CHANGED",
        )
        if _receiving_run is not None:
            _check_receiving(_receiving_run, preparation)
        source._pure_final(state)
        _require(
            source._ISSUED.get(id(preparation)) is entry
            and preparation.payload == entry[1]
            and source.asec_native._ISSUED.get(id(native)) is issued
            and native.payload == issued[1]
            and issued[2].coverage is coverage,
            "FINAL_OWNER",
        )
        _require(
            candidate.source_frame is state.frame
            and candidate.receipt == receipt
            and candidate.origins is origins
            and candidate.roles is roles
            and candidate.unit_status is statuses
            and candidate.unit_evidence is units
            and candidate.asec_raw is raw
            and candidate._receiving_run is _receiving_run
            and source._digest(asdict(acs_profile)) == profile_seal
            and tuple(
                seals._table_seal(table)
                for table in (
                    origins,
                    raw,
                    units,
                    roles.to_frame(),
                    statuses.to_frame(),
                )
            )
            == table_seals,
            "PROJECTION_CHANGED",
        )

    object.__setattr__(result, "_revalidate", revalidate)
    _ISSUED[result] = (
        weakref.ref(revalidate),
        _callback_seal(revalidate),
        receipt,
        table_seals,
    )
    result.validate()
    return result


def _check_receiving(run, preparation):
    from . import graph_us_survey_enrichment as host

    host.check_survey_enrichment_run(run)
    _require(
        run.parent_run.financial_run.prefix.preparation is preparation,
        "RECEIVING_PREPARATION",
    )


def qualify_native_spm_inputs(
    real_issued_enrichment_run, *, acs_profile, asec_scope_policy
):
    """Only a live issued receiving owner may enter the public bridge."""
    from . import graph_us_survey_enrichment as host

    host.check_survey_enrichment_run(real_issued_enrichment_run)
    preparation = real_issued_enrichment_run.parent_run.financial_run.prefix.preparation
    result = qualify_current_survey_spm(
        preparation,
        acs_profile=acs_profile,
        asec_scope_policy=asec_scope_policy,
        _receiving_run=real_issued_enrichment_run,
    )
    host.check_survey_enrichment_run(real_issued_enrichment_run)
    result.validate()
    return result


def project_qualified_spm_inputs(run, qualified, *, year, outside_role_placeholder):
    """Borrow inputs for the same live receiving owner, without modifying it."""
    _require(type(qualified) is QualifiedNativeSpmInputs, "RETAINED_OWNER_REQUIRED")
    _require(qualified._receiving_run is run and run is not None, "RECEIVING_OWNER")
    qualified.validate()
    result = projection.project_spm_inputs(
        qualified.source_frame,
        qualified.origins,
        qualified.roles,
        qualified.unit_status,
        run.population.frame,
        source_year=2024,
        year=year,
        outside_role_placeholder=outside_role_placeholder,
    )
    return projection.validate_spm_projection(result, qualified.validate)


_REVALIDATE_CODE = next(
    c
    for c in qualify_current_survey_spm.__code__.co_consts
    if type(c) is CodeType and c.co_name == "revalidate"
)
_LIVE = _live()
