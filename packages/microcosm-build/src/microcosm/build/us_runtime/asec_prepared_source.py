"""Full-source ASEC preparation for the current-money graph slice.

The graph source is one directory holding exactly the reviewed restoration
inputs: the raw-stage parent P, the household-observation attachment H, the
restored person-income attachment T with its receipt, the three housing cohort
HDFs, and the three official Census PERSON CSV members. No ZIP archive is
accepted and no file is discovered by pattern: the roster below is the contract.

The executor keys a directory source by relative name plus file bytes, which is
a cache identity, not an authentication. Authentication stays where it already
is — the pinned digests inside the reviewed loaders and the canonical byte
replay of T — and this module simply refuses to run unless those loaders issue
authority.

The authenticated ``ReadyCurrentMoney`` exists only inside this call. What
leaves is the canonical encoded body, a receipt of identities, and an
operator-free frame whose dtypes have been promoted to graph tokens.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.serialization_dtypes import canonicalize_frame_string_dtypes
from microcosm.frame import US_SCHEMA, Frame
from microcosm.graph.population import dtype_for_token, token_for_dtype

from . import asec_current_money_source as money_source
from . import asec_housing_status_source as housing_source
from . import asec_housing_universe as housing_universe
from . import asec_housing_universe_source as housing_universe_source
from . import asec_income_observations as income_source
from . import asec_person_income_source as person_income
from . import asec_student_controls as student_source
from ._asec_current_money_codec import (
    current_money_content_sha256,
    encode_current_money,
)
from .asec_checkpoint import ASEC_RAW_STAGE_CHECKPOINT_FILENAME
from .asec_current_money import (
    ENCODED_ZERO_POLICY,
    FIELDS,
    RESTORED_SOURCE_KIND,
    ZERO_POLICY,
    MoneyRefusalError,
    ReadyCurrentMoney,
    _json,
    _parse,
    _sha,
)
from .asec_current_money import (
    _digest as _digest_shape,
)
from .asec_current_money_units import reconstruct_current_money_tax_units
from .asec_housing_status_source import (
    attach_housing_status,
    load_authenticated_housing_status,
)
from .education_assistance_source import ASEC_EDUCATION_ASSISTANCE_ARCHIVES
from .operator_boundary import assert_operator_free_source_frame

PREPARED_SOURCE_KIND = "us_asec_prepared_current_money_v3"
PREPARED_RECEIPT_SCHEMA = "microcosm.us.asec_prepared_receipt.v3"
PREPARATION_BOUNDARY = "US graph ASEC prepared source"
SOURCE_YEARS: tuple[int, ...] = (2022, 2023, 2024)
PARENT_FILENAME = ASEC_RAW_STAGE_CHECKPOINT_FILENAME
HOUSEHOLD_ATTACHMENT_FILENAME = "asec_household_observations.checkpoint.h5"
PERSON_INCOME_FILENAME = person_income.CHECKPOINT_FILENAME
RESTORATION_RECEIPT_FILENAME = "restoration.receipt.json"
COHORT_FILENAMES: tuple[tuple[int, str], ...] = tuple(
    (year, f"asec_household_cohort_{year}.h5") for year in SOURCE_YEARS
)
MEMBER_FILENAMES: tuple[tuple[int, str], ...] = tuple(
    (year, ASEC_EDUCATION_ASSISTANCE_ARCHIVES[year].member) for year in SOURCE_YEARS
)
PREPARED_SOURCE_FILES: tuple[str, ...] = (
    PARENT_FILENAME,
    HOUSEHOLD_ATTACHMENT_FILENAME,
    PERSON_INCOME_FILENAME,
    RESTORATION_RECEIPT_FILENAME,
    *(name for _year, name in COHORT_FILENAMES),
    *(name for _year, name in MEMBER_FILENAMES),
)
RESTORATION_RECEIPT_MAX_BYTES = 256 * 1024
_INTEGER_WIDTH_TARGETS = frozenset({"int8", "int16", "uint8", "uint16", "uint32"})
_NULLABLE_INTEGER_TARGETS = frozenset(
    {"Int8", "Int16", "Int32", "UInt8", "UInt16", "UInt32", "UInt64"}
)


class PreparedSourceRefusalError(ValueError):
    """Sanitized preparation refusal; never carries rows, values or paths."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise PreparedSourceRefusalError(reason)


def _roster(source_dir) -> dict[str, Path]:
    directory = Path(source_dir)
    _require(directory.is_dir(), "SOURCE_DIRECTORY")
    present = sorted(entry.name for entry in directory.iterdir())
    _require(present == sorted(PREPARED_SOURCE_FILES), "SOURCE_FILE_ROSTER")
    paths = {}
    for name in PREPARED_SOURCE_FILES:
        path = directory / name
        _require(path.is_file() and not path.is_symlink(), "SOURCE_REGULAR_FILE")
        paths[name] = path
    return paths


RESTORATION_READER = "authenticated_census_csv_exact_integer_no_fill_v1"
RESTORATION_JOIN_KEYS = ["source_year", "PERIDNUM"]
RESTORATION_CROSSCHECK_COLUMNS = ["source_household_id", "A_LINENO", "A_AGE"]
RESTORATION_LOGICAL_FIELD = "PTOTVAL"
RESTORATION_NOMINAL_BASIS = "income_year_us_dollars"
RESTORATION_ZERO_CLAIM = "encoded_source_value_not_respondent_answer"
_RESTORATION_RECEIPT_KEYS = frozenset(
    {
        "artifact_kind",
        "crosscheck_columns",
        "dtype",
        "encoding_contract",
        "household_attachment_sha256",
        "implementation_sha256",
        "join_keys",
        "logical_field",
        "nominal_basis",
        "observation_sha256",
        "original_v4_sha256",
        "output_column",
        "output_file",
        "output_frame_sha256",
        "output_sha256",
        "parent_frame_sha256",
        "reader",
        "schema_version",
        "sources",
        "zero_claim",
        "zero_origin_policy",
    }
)
_RESTORATION_SOURCE_KEYS = frozenset(
    {
        "archive_sha256",
        "income_year",
        "incumbent_compared_rows",
        "incumbent_conflicts",
        "joined_rows",
        "member",
        "member_sha256",
        "native_key_or_age_conflicts",
        "source_rows",
        "survey_year",
        "unreferenced_source_rows",
    }
)


def _restoration_sources(rows, source) -> None:
    _require(
        type(rows) is list and len(rows) == len(SOURCE_YEARS), "RESTORATION_SOURCES"
    )
    for row, pins in zip(rows, person_income._MEMBER_PINS, strict=True):
        year, member, archive_pin, member_pin, source_rows, _size = pins
        positions = np.flatnonzero(np.asarray(source.scope.person_years) == year)
        known_incumbents = int(
            source.frame.person.PTOTVAL.iloc[positions].notna().sum()
        )
        _require(
            type(row) is dict and set(row) == _RESTORATION_SOURCE_KEYS,
            "RESTORATION_SOURCES",
        )
        _require(
            row["income_year"] == year
            and row["survey_year"] == year + 1
            and row["member"] == member
            and row["archive_sha256"] == archive_pin
            and row["member_sha256"] == member_pin
            and row["source_rows"] == source_rows,
            "RESTORATION_SOURCE_PINS",
        )
        _require(
            all(
                type(row[name]) is int and row[name] >= 0
                for name in (
                    "joined_rows",
                    "unreferenced_source_rows",
                    "incumbent_compared_rows",
                )
            )
            and row["joined_rows"] == len(positions) == source_rows
            and row["unreferenced_source_rows"] == 0
            and row["incumbent_compared_rows"] == known_incumbents
            and row["incumbent_conflicts"] == 0
            and row["native_key_or_age_conflicts"] == 0,
            "RESTORATION_SOURCE_COVERAGE",
        )


def _restoration_receipt(path: Path, evidence: dict, source) -> dict:
    """Bind the shipped restoration receipt to the authority the loader issued."""
    # Refuse special files even if the roster check raced with a replacement,
    # and bound allocation before parsing an untrusted sidecar.
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        _require(stat.S_ISREG(info.st_mode), "RESTORATION_RECEIPT")
        _require(
            0 < info.st_size <= RESTORATION_RECEIPT_MAX_BYTES, "RESTORATION_RECEIPT"
        )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(RESTORATION_RECEIPT_MAX_BYTES + 1)
    finally:
        os.close(descriptor)
    _require(0 < len(payload) <= RESTORATION_RECEIPT_MAX_BYTES, "RESTORATION_RECEIPT")
    _require(payload.endswith(b"\n"), "RESTORATION_RECEIPT")
    try:
        receipt = _parse(payload[:-1], RESTORATION_RECEIPT_MAX_BYTES)
    except MoneyRefusalError as error:
        raise PreparedSourceRefusalError("RESTORATION_RECEIPT") from error
    _require(_json(receipt) == payload[:-1], "RESTORATION_RECEIPT_CANONICAL")
    _require(set(receipt) == _RESTORATION_RECEIPT_KEYS, "RESTORATION_RECEIPT_SCHEMA")
    _require(
        receipt["schema_version"] == 1
        and receipt["artifact_kind"] == person_income.ARTIFACT_KIND
        and receipt["output_file"] == PERSON_INCOME_FILENAME
        and receipt["output_column"] == person_income.OBSERVED_COLUMN
        and receipt["encoding_contract"] == person_income._ENCODING_CONTRACT
        and receipt["reader"] == RESTORATION_READER
        and receipt["join_keys"] == RESTORATION_JOIN_KEYS
        and receipt["crosscheck_columns"] == RESTORATION_CROSSCHECK_COLUMNS
        and receipt["logical_field"] == RESTORATION_LOGICAL_FIELD
        and receipt["nominal_basis"] == RESTORATION_NOMINAL_BASIS
        and receipt["dtype"] == "int64"
        and receipt["zero_origin_policy"] == ENCODED_ZERO_POLICY
        and receipt["zero_claim"] == RESTORATION_ZERO_CLAIM,
        "RESTORATION_RECEIPT_SCHEMA",
    )
    _require(
        all(
            _digest_shape(receipt[name])
            for name in (
                "observation_sha256",
                "output_frame_sha256",
                "parent_frame_sha256",
            )
        ),
        "RESTORATION_RECEIPT_DIGEST",
    )
    _restoration_sources(receipt["sources"], source)
    # The reconstruction the loader just performed is the authority; the shipped
    # receipt must agree with the evidence it issued, digest for digest.
    _require(
        receipt["output_sha256"] == evidence["person_income_attachment_sha256"]
        and receipt["original_v4_sha256"] == evidence["parent_sha256"]
        and receipt["household_attachment_sha256"] == evidence["attachment_sha256"]
        and receipt["implementation_sha256"] == evidence["verification_sha256"],
        "RESTORATION_RECEIPT_BINDING",
    )
    # T1's pinned producer only appends the observed person-income column to an
    # owned copy of the authenticated parent. Recover that exact parent view
    # without reading another source or changing the original T1 authority.
    # The sidecar must describe those reconstructed facts, not merely contain
    # syntactically valid digest strings beside an authentic checkpoint.
    source.validate()
    observed = source.frame.person[person_income.OBSERVED_COLUMN]
    _require(observed.dtype == np.dtype("int64"), "RESTORATION_OBSERVATION_DTYPE")
    _require(
        receipt["observation_sha256"]
        == _sha(observed.to_numpy(dtype="<i8", copy=True).tobytes())
        and receipt["output_frame_sha256"] == evidence["frame_sha256"],
        "RESTORATION_RECEIPT_FACTS",
    )
    parent = housing_source._owned_frame(source.frame)
    del parent.person[person_income.OBSERVED_COLUMN]
    _require(
        receipt["parent_frame_sha256"] == money_source._frame_signature(parent),
        "RESTORATION_RECEIPT_FACTS",
    )
    source.validate()
    return receipt


def _target_token(dtype) -> str:
    try:
        return token_for_dtype(dtype)
    except Exception as error:  # PopulationError is a ValueError subclass
        del error
    name = getattr(dtype, "name", None)
    if isinstance(dtype, pd.api.extensions.ExtensionDtype):
        _require(name in _NULLABLE_INTEGER_TARGETS, "UNPROMOTABLE_DTYPE")
        return "Int64"
    normalized = np.dtype(dtype)
    _require(
        normalized.name in _INTEGER_WIDTH_TARGETS or normalized.name == "float16",
        "UNPROMOTABLE_DTYPE",
    )
    return "float64" if normalized.name == "float16" else "int64"


def _promote(frame: Frame) -> tuple[dict[str, str], ...]:
    """Promote every column to its graph dtype token, value- and null-preserving."""
    canonicalize_frame_string_dtypes(
        frame, boundary=PREPARATION_BOUNDARY, in_place=True
    )
    transitions: list[dict[str, str]] = []
    for entity in US_SCHEMA.entities:
        table = frame.table(entity)
        for column in list(table.columns):
            _require(isinstance(column, str), "COLUMN_NAME")
            original = table[column]
            token = _target_token(original.dtype)
            target = dtype_for_token(token)
            if original.dtype == target:
                continue
            promoted = original.astype(target)
            _require(original.isna().equals(promoted.isna()), "DTYPE_PROMOTION_NULLS")
            observed = original.notna()
            _require(
                np.array_equal(
                    original[observed].to_numpy(dtype=object),
                    promoted[observed].to_numpy(dtype=object),
                ),
                "DTYPE_PROMOTION_VALUES",
            )
            table[column] = promoted
            transitions.append(
                {
                    "entity": entity,
                    "column": column,
                    "from": str(original.dtype),
                    "to": token,
                }
            )
    for entity in US_SCHEMA.entities:
        table = frame.table(entity)
        for column in table.columns:
            token_for_dtype(table[column].dtype)
    return tuple(transitions)


def _entity_rows(frame: Frame) -> dict[str, int]:
    return {entity: frame.n(entity) for entity in US_SCHEMA.entities}


def prepared_source_implementation_identity() -> str:
    """Bind this preparation's own code plus the reviewed loaders it calls."""
    package = resources.files(__package__)
    modules = {
        name: _sha(package.joinpath(name).read_bytes())
        for name in (
            "asec_prepared_source.py",
            "asec_income_observations.py",
            "asec_current_money_selection.py",
            "cps_carried_current.py",
            "graph_housing_universe.py",
        )
    }
    return _sha(
        _json(
            {
                "schema": 3,
                "source_kind": PREPARED_SOURCE_KIND,
                "modules": modules,
                "money_source_verification": money_source._verification_identity(),
                "restoration_verification": person_income._implementation(),
                "student_controls_verification": student_source._implementation(),
                "income_observations_verification": income_source._implementation(),
                "housing_verification": housing_source._implementation(),
                "housing_universe_verification": housing_universe_source._implementation(),
                "file_roster": list(PREPARED_SOURCE_FILES),
                "boundary": PREPARATION_BOUNDARY,
            }
        )
    )


@dataclass(frozen=True)
class PreparedAsecPopulation:
    """An operator-free prepared frame with its money body and identity receipt."""

    frame: Frame
    money_payload: bytes
    receipt: dict
    field_entities: tuple[tuple[str, str], ...]
    housing_universe_payload: bytes
    income_observations_payload: bytes

    @property
    def receipt_payload(self) -> bytes:
        return _json(self.receipt)

    @property
    def receipt_sha256(self) -> str:
        return _sha(self.receipt_payload)


def prepare_asec_current_money_population(source_dir) -> PreparedAsecPopulation:
    """Load, reconstruct, attach and promote the full authenticated population."""
    paths = _roster(source_dir)
    identity = prepared_source_implementation_identity()
    source = person_income.load_authenticated_restored_current_money_source(
        paths[PARENT_FILENAME],
        paths[HOUSEHOLD_ATTACHMENT_FILENAME],
        paths[PERSON_INCOME_FILENAME],
        member_paths={year: paths[name] for year, name in MEMBER_FILENAMES},
    )
    evidence = _parse(source.source.identity)
    _require(evidence["source_kind"] == RESTORED_SOURCE_KIND, "RESTORED_SOURCE_KIND")
    _require(evidence["zero_origin_policy"] == ZERO_POLICY, "ZERO_ORIGIN_POLICY")
    _require(evidence["field_roster"] == list(FIELDS), "FIELD_ROSTER")
    restoration = _restoration_receipt(
        paths[RESTORATION_RECEIPT_FILENAME], evidence, source
    )
    ready = source.ready()
    _require(type(ready) is ReadyCurrentMoney, "AUTHENTICATED_READINESS_REQUIRED")
    income = income_source.load_authenticated_income_observations(
        source,
        ready,
        member_paths={year: paths[name] for year, name in MEMBER_FILENAMES},
    )
    # S is separate source authority. Its public loader authenticates the same
    # official members against the unchanged T1 and money parents; attachment
    # adds two aliases while retaining original missing controls untouched.
    controls = student_source.load_authenticated_student_controls(
        source,
        ready,
        member_paths={year: paths[name] for year, name in MEMBER_FILENAMES},
    )
    with_students = student_source.attach_student_controls(source, ready, controls)
    tax = reconstruct_current_money_tax_units(with_students, ready)
    # Housing still authenticates original T1, never the transient tax/S view.
    status = load_authenticated_housing_status(
        source, cohort_paths={year: paths[name] for year, name in COHORT_FILENAMES}
    )
    attached = attach_housing_status(tax, status)
    attached.validate()
    # HU authenticates original T1 and retains the complete S/tax/housing parent.
    # Its strict uint8 attachment validation runs before graph width promotion.
    universe = housing_universe_source.load_authenticated_housing_universe(source)
    with_universe = housing_universe_source.attach_housing_universe(attached, universe)
    with_universe.validate()
    frame = housing_source._owned_frame(with_universe.frame)
    pre_promotion = money_source._frame_signature(frame)
    _require(
        pre_promotion == with_universe.receipt["output_frame_sha256"], "PARENT_CAPTURE"
    )
    transitions = _promote(frame)
    post_promotion = money_source._frame_signature(frame)
    # The graph carries metadata as a typed context artifact, and the store's
    # frame codec does not persist Frame.metadata at all. A non-empty prepared
    # metadata mapping would be silently dropped, so refuse instead.
    _require(dict(frame.metadata) == {}, "PREPARED_FRAME_METADATA")
    _require(frame.mass_log == (), "PREPARED_FRAME_MASS_LOG")
    _require(frame.weighted_entities == ("household",), "PREPARED_FRAME_WEIGHTS")
    frame.revalidate()
    assert_operator_free_source_frame(frame, label=PREPARATION_BOUNDARY)
    _require(
        prepared_source_implementation_identity() == identity,
        "PREPARATION_IMPLEMENTATION_CHANGED",
    )
    money_payload = encode_current_money(ready)
    universe_payload = housing_universe.encode_housing_universe(universe)
    income_payload = income_source.encode_income_observations(income)
    field_entities = tuple(
        (domain.name, domain.entity) for domain in ready.bindings.spec.fields
    )
    receipt = {
        "schema": PREPARED_RECEIPT_SCHEMA,
        "source_kind": PREPARED_SOURCE_KIND,
        "restored_source_kind": evidence["source_kind"],
        "release_eligible": False,
        "all_current_money_consumers_wired": False,
        "source_evidence_sha256": _sha(source.source.identity),
        "t1_sha256": evidence["person_income_attachment_sha256"],
        "income_observations": {
            "parser_profile": income_source.parser_profile(),
            "header_sha256": _sha(income._header),
            "content_sha256": income.content_sha256,
            "payload_sha256": _sha(income_payload),
            "rows": income.receipt["rows"],
            "columns": list(income_source.COLUMNS),
            "kind": income_source.ARTIFACT_KIND,
            "scope_sha256": income.receipt["scope_sha256"],
        },
        "student_controls_receipt_sha256": _sha(_json(controls.receipt)),
        "student_controls_content_sha256": controls.content_sha256,
        "student_controls_attachment_receipt_sha256": _sha(
            _json(with_students.receipt)
        ),
        "student_controls_reference_period": controls.receipt["reference_period"],
        "annual_five_month_student_status_validated": False,
        "money_header_sha256": _sha(ready.header),
        "money_content_sha256": current_money_content_sha256(ready),
        "money_spec_sha256": ready.bindings.spec.sha256,
        "field_entities": [list(item) for item in field_entities],
        "scope_sha256": _parse(ready.header)["scope_sha256"],
        "tax_receipt_sha256": _sha(_json(tax.receipt)),
        "tax_old_partition_sha256": tax.receipt["old_partition_sha256"],
        "tax_new_partition_sha256": tax.receipt["new_partition_sha256"],
        "housing_content_sha256": status.content_sha256,
        "housing_attachment_receipt_sha256": _sha(_json(attached.receipt)),
        "pre_housing_universe_frame_sha256": attached.receipt["output_frame_sha256"],
        "housing_universe": {
            "header_sha256": _sha(universe.header),
            "content_sha256": universe.content_sha256,
            "payload_sha256": _sha(universe_payload),
            "definition_sha256": universe.header_data["definition_sha256"],
            "attachment_receipt_sha256": _sha(_json(with_universe.receipt)),
            "parent_kind": "HousingStatusAttachedAsec",
            "source_period_kind": "interview_household_universe",
            "native_dtype": "uint8",
            "graph_dtype": "int64",
            "aliases": list(housing_universe_source.ATTACHED_COLUMNS),
        },
        "restoration_receipt_sha256": _sha(_json(restoration)),
        "pre_promotion_frame_sha256": pre_promotion,
        "post_promotion_frame_sha256": post_promotion,
        "dtype_transitions": [dict(item) for item in transitions],
        "entity_rows": _entity_rows(frame),
        "implementation_sha256": identity,
        "file_roster": list(PREPARED_SOURCE_FILES),
    }
    return PreparedAsecPopulation(
        frame, money_payload, receipt, field_entities, universe_payload, income_payload
    )


def load_graph_asec_prepared(path, *, store=None) -> Frame:
    """Registered source codec: the prepared operator-free population only.

    The executor verifies that this loader is installed and content-keys the
    directory; it never calls it. The declared computation that invokes the
    preparation is the CREATE kernel, which needs the money body and receipt
    this signature cannot return, so the two paths share one implementation
    rather than preparing the population twice.
    """
    del store
    return prepare_asec_current_money_population(path).frame


__all__ = [
    "COHORT_FILENAMES",
    "HOUSEHOLD_ATTACHMENT_FILENAME",
    "MEMBER_FILENAMES",
    "PARENT_FILENAME",
    "PERSON_INCOME_FILENAME",
    "PREPARATION_BOUNDARY",
    "PREPARED_RECEIPT_SCHEMA",
    "PREPARED_SOURCE_FILES",
    "PREPARED_SOURCE_KIND",
    "PreparedAsecPopulation",
    "PreparedSourceRefusalError",
    "RESTORATION_RECEIPT_FILENAME",
    "load_graph_asec_prepared",
    "prepare_asec_current_money_population",
    "prepared_source_implementation_identity",
]
