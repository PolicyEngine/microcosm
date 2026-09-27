"""Authenticate the reviewed full ASEC v4 plus household-observation attachment.

No public pin override, file discovery, download, source selection, or unit fit.
Verification is source authority, not evidence that all downstream consumers are
corrected or that a population is eligible for release.
"""

import hashlib
import os
import tempfile
from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from importlib import metadata, resources
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build import frame_checkpoint as checkpoint
from microcosm.build.outer_stage_runtime import frame_identity
from microcosm.build.serialization_dtypes import canonicalize_frame_string_dtypes
from microcosm.frame import Frame

from . import asec_checkpoint, asec_household_observations, operator_boundary
from .asec_current_money import (
    _DTYPES,
    _SOURCE_TOKEN,
    FIELDS,
    ZERO_POLICY,
    AsecMoneyScope,
    AsecMoneyViews,
    AuthenticatedAsecSource,
    CurrentMoneySpec,
    MoneyRefusalError,
    _index_identity,
    _json,
    _parse,
    _require,
    _sha,
    classify_asec_money,
    compile_asec_current_money_spec,
    require_complete_current_money,
    restate_asec_current_money,
)
from .asec_current_money_resources import load_current_money_resources
from .education_assistance_source import ASEC_EDUCATION_ASSISTANCE_ARCHIVES

# Exact reviewed bytes from the accepted current-money source contract and the
# completed household-observation receipt. Tests monkeypatch this PRIVATE tuple
# only for invented files; production has no override/constructor for other pins.
_SOURCE_PINS = (
    "e2f2b7495bfcf1448dfb0acb0a17e93f86a6ab8bef8a70ec0a2981983028cbb5",
    "f7f086e262d9a1d9ec6fc1a0b7c32e577128578a4fc5b3f14e13fb8711f7ec2d",
    (
        (2022, "7ccca976284bb47815d84460cc4f75a0a65d26d7754ab0a0f417de351b3d474e"),
        (2023, "cb57817327799f42b741caed5f9be94d04021c2e6809c1ad7bd0686da5428d88"),
        (2024, "ec36604cb735a660b51b0b2f90be27d803b5878f3464fb30d0eacead59c1260d"),
    ),
)
_LOAD_TOKEN = object()
_ATTACHMENT_COLUMNS = tuple(
    "asec_" + c for c in asec_household_observations.ASEC_HOUSEHOLD_OBSERVATION_COLUMNS
)


def _stage_verified(path, expected, staging):
    """Load only a verified private copy, closing same-path replacement windows."""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as src, staging.open("xb") as dst:
            before = os.fstat(src.fileno())
            while chunk := src.read(1024 * 1024):
                digest.update(chunk)
                dst.write(chunk)
            after = os.fstat(src.fileno())
            _require(
                (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                == (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ),
                "SOURCE_BYTES_CHANGED",
            )
            _require(digest.hexdigest() == expected, "SOURCE_BYTES_MISMATCH")
    except OSError:
        raise MoneyRefusalError("SOURCE_BYTES_UNAVAILABLE") from None


def _series_digest(digest, series):
    """Use the existing checkpoint's closed scalar vocabulary, one column at a time."""
    spec = checkpoint._series_spec(series, label="authenticated source")
    digest.update(_json(spec))
    extension = series.array
    data, mask = getattr(extension, "_data", None), getattr(extension, "_mask", None)
    if isinstance(data, np.ndarray) and isinstance(mask, np.ndarray):
        for array in (data, mask):
            value = np.ascontiguousarray(array).tobytes()
            digest.update(len(value).to_bytes(8, "little"))
            digest.update(value)
    elif isinstance(series.dtype, np.dtype) and not series.dtype.hasobject:
        value = np.ascontiguousarray(series.to_numpy(copy=False)).tobytes()
        digest.update(len(value).to_bytes(8, "little"))
        digest.update(value)
    else:
        # Repeated literal strings are common in full survey source frames.
        # Cache their exact length-prefixed bytes within this column only; every
        # value is still visited and every later seal rereads the current data.
        # Both entry and payload bounds keep distinct or large strings bounded.
        string_tokens = {}
        token_bytes = 0
        for value in series.to_numpy(dtype=object, copy=False):
            if type(value) is str:
                token = string_tokens.get(value)
                if token is None:
                    encoded = checkpoint._encode_object_scalar(value)
                    token = len(encoded).to_bytes(8, "little") + encoded
                    if len(string_tokens) < 1024 and token_bytes + len(token) <= 262144:
                        string_tokens[value] = token
                        token_bytes += len(token)
                digest.update(token)
                continue
            encoded = checkpoint._encode_object_scalar(value)
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)


def _frame_signature(frame):
    digest = hashlib.sha256(b"microcosm/asec-current-money-source-frame/1\0")
    digest.update(_json(checkpoint._checkpoint_metadata(frame, {})))
    digest.update(_json(_plain_metadata(frame.metadata)))
    for _, table in checkpoint._frame_tables(frame):
        digest.update(
            _json(
                {
                    "columns_dtype": repr(table.columns.dtype),
                    "columns_type": type(table.columns).__name__,
                }
            )
        )
        _series_digest(
            digest, pd.Series(table.index.to_numpy(copy=False), dtype=table.index.dtype)
        )
        for column in table:
            _series_digest(digest, table[column])
    _series_digest(digest, frame.strata)
    digest.update(_json(frame.strata.name))
    digest.update(_json(checkpoint._index_spec(frame.strata.index, label="strata")))
    _series_digest(
        digest,
        pd.Series(
            frame.strata.index.to_numpy(copy=False), dtype=frame.strata.index.dtype
        ),
    )
    for entity in frame.weighted_entities:
        digest.update(frame.weights_for(entity).values.tobytes())
    return digest.hexdigest()


def _owned_index(index):
    """Detach axes, including pandas' materialized RangeIndex cache."""
    if isinstance(index, pd.RangeIndex):
        # RangeIndex.copy(deep=True) shares its cached ndarray. Reconstruct the
        # same logical range so a child cannot write through to a sealed parent.
        return pd.RangeIndex(index.start, index.stop, index.step, name=index.name)
    return index.copy(deep=True)


def _detach_frame_axes(frame):
    """Detach axes of an already copied Frame; preserve its data and axis types."""
    for _, table in checkpoint._frame_tables(frame):
        table.index = _owned_index(table.index)
        table.columns = _owned_index(table.columns)
    frame.strata.index = _owned_index(frame.strata.index)


def _plain_metadata(value):
    if isinstance(value, Mapping):
        return [
            "mapping",
            [[key, _plain_metadata(item)] for key, item in sorted(value.items())],
        ]
    if isinstance(value, tuple):
        return ["tuple", [_plain_metadata(item) for item in value]]
    if isinstance(value, frozenset):
        return [
            "frozenset",
            sorted((_plain_metadata(item) for item in value), key=_json),
        ]
    return ["scalar", value]


def _verification_identity():
    package = resources.files(__package__)
    names = (
        "asec_current_money.py",
        "asec_current_money_source.py",
        "asec_checkpoint.py",
        "asec_household_observations.py",
        "education_assistance_source.py",
        "reported_coverage_source.py",
        "operator_boundary.py",
    )
    shared = resources.files("microcosm.build")
    frame_package = resources.files("microcosm.frame")
    files = {name: _sha(package.joinpath(name).read_bytes()) for name in names}
    files.update(
        {
            "build/" + name: _sha(shared.joinpath(name).read_bytes())
            for name in (
                "frame_checkpoint.py",
                "outer_stage_runtime.py",
                "serialization_dtypes.py",
            )
        }
    )
    files.update(
        {
            "frame/" + path.name: _sha(path.read_bytes())
            for path in frame_package.iterdir()
            if path.name.endswith(".py")
        }
    )
    # This exact live projection is consumed by the existing source-only guard.
    projection = {
        family: {entity: sorted(columns) for entity, columns in by_entity.items()}
        for family, by_entity in operator_boundary.PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES.items()
    }
    return _sha(
        _json(
            {
                "modules": files,
                "operator_owned": projection,
                "pins": _SOURCE_PINS,
                "dependencies": {
                    n: metadata.version(n) for n in ("numpy", "pandas", "h5py")
                },
            }
        )
    )


def _scope(frame):
    person = frame.person
    keys = asec_household_observations._household_keys(frame)

    def ints(column):
        values = person[column]
        _require(
            values.dtype == np.dtype("int64") and not values.isna().any(),
            "SOURCE_COORDINATE_DTYPE",
        )
        return tuple(int(v) for v in values.to_numpy())

    years = ints("source_year")
    _require(set(years) == {2022, 2023, 2024}, "SOURCE_YEAR_MAPPING")
    return AsecMoneyScope(
        ints("person_id"),
        tuple(int(v) for v in keys.household_id),
        ints("person_household_id"),
        ints("person_spm_unit_id"),
        years,
        tuple(int(v) for v in keys.source_year),
        tuple(
            value.decode("utf-8") if isinstance(value, bytes) else value
            for value in person.PERIDNUM.tolist()
        ),
        tuple(str(int(v)) for v in keys.H_SEQ),
    )


def _views(frame, scope, *, restored=False):
    person = frame.person.loc[:, [f for f in FIELDS if f != "HTOTVAL"]].copy()
    if restored:
        person["PTOTVAL"] = frame.person["asec_PTOTVAL"].to_numpy(copy=True)
    return AsecMoneyViews(
        person,
        frame.table("household").loc[:, ["asec_HTOTVAL"]].copy(),
        scope,
    )


def _input_binding(views):
    scope_sha = _sha(
        _json(
            {
                "coordinates": _parse(views.scope.identity),
                "person_index": _parse(_index_identity(views.person.index)),
                "household_index": _parse(_index_identity(views.household.index)),
            }
        )
    )
    inputs = []
    for entity, table in (("person", views.person), ("household", views.household)):
        for column, series in table.items():
            field = "HTOTVAL" if entity == "household" else column
            _require(str(series.dtype) in _DTYPES, "UNSUPPORTED_DTYPE", field)
            values = series.to_numpy(dtype="<f8", na_value=np.nan, copy=True)
            valid = ~series.isna().to_numpy()
            values[~valid] = 0.0
            entry = {
                "field": field,
                "entity": entity,
                "column": column,
                "dtype": str(series.dtype),
                "values_sha256": _sha(values.tobytes()),
                "validity_sha256": _sha(valid.astype("u1").tobytes()),
            }
            if entity == "household":
                inputs.insert(FIELDS.index("HTOTVAL"), entry)
            else:
                inputs.append(entry)
    _require(tuple(entry["field"] for entry in inputs) == FIELDS, "FIELD_ROSTER")
    return scope_sha, _sha(_json(inputs))


def _same_table(left, right):
    try:
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    except AssertionError:
        raise MoneyRefusalError("PARENT_PROJECTION_CHANGED") from None
    for (_, left_series), (_, right_series) in zip(
        left.items(), right.items(), strict=True
    ):
        first, second = hashlib.sha256(), hashlib.sha256()
        _series_digest(first, left_series)
        _series_digest(second, right_series)
        _require(first.digest() == second.digest(), "PARENT_PROJECTION_CHANGED")


def _validate_attachment(parent, metadata, loaded, parent_sha):
    frame = loaded.frame
    document = loaded.metadata
    _require(
        set(document)
        == {
            "schema_version",
            "artifact_kind",
            "parent_checkpoint_sha256",
            "household_observations",
        }
        and document["schema_version"] == 1
        and document["artifact_kind"] == "microcosm.asec_household_observations_source"
        and document["parent_checkpoint_sha256"] == parent_sha,
        "ATTACHMENT_DOCUMENT",
    )
    receipt = document["household_observations"]
    _require(
        set(receipt)
        == {
            "schema_version",
            "artifact_kind",
            "input_checkpoint_sha256",
            "input_structural_identity_sha256",
            "reader",
            "join_keys",
            "sources",
            "household_identity",
            "outputs",
            "semantics",
        },
        "ATTACHMENT_RECEIPT",
    )
    _require(
        receipt["schema_version"] == 1
        and receipt["artifact_kind"]
        == "microcosm.asec_household_observation_attachment"
        and receipt["input_checkpoint_sha256"] == parent_sha
        and receipt["input_structural_identity_sha256"] == frame_identity(parent).sha256
        and receipt["reader"] == "pandas_fixed_hdf_numeric_int64_v1"
        and receipt["join_keys"] == ["income_year", "H_SEQ"],
        "ATTACHMENT_RECEIPT",
    )
    _require(
        frame.schema == parent.schema
        and frame.weighted_entities == parent.weighted_entities,
        "PARENT_STRUCTURE_CHANGED",
    )
    for entity in parent.entities:
        original = parent.table(entity)
        actual = frame.table(entity)
        expected = tuple(original.columns) + (
            _ATTACHMENT_COLUMNS if entity == "household" else ()
        )
        _require(tuple(actual.columns) == expected, "ATTACHMENT_COLUMN_ROSTER")
        _same_table(actual.loc[:, original.columns], original)
    _require(
        frame.mass_log == parent.mass_log and frame.metadata == parent.metadata,
        "PARENT_CONTEXT_CHANGED",
    )
    try:
        pd.testing.assert_series_equal(frame.strata, parent.strata, check_exact=True)
    except AssertionError:
        raise MoneyRefusalError("PARENT_STRATA_CHANGED") from None
    for entity in parent.weighted_entities:
        a, b = frame.weights_for(entity), parent.weights_for(entity)
        _require(
            a.kind == b.kind and a.values.tobytes() == b.values.tobytes(),
            "PARENT_WEIGHTS_CHANGED",
        )
    household = frame.table("household")
    keys = asec_household_observations._household_keys(parent)
    identity = {
        "rows": len(household),
        "id_dtype": "int64",
        "ordered_ids_sha256": _sha(
            household.household_id.to_numpy(dtype="<i8", copy=False).tobytes()
        ),
    }
    _require(
        receipt["household_identity"] == identity
        and set(receipt["outputs"]) == set(_ATTACHMENT_COLUMNS),
        "ATTACHMENT_IDENTITY",
    )
    for column in _ATTACHMENT_COLUMNS:
        values = household[column]
        _require(values.dtype == np.dtype("int64"), "ATTACHMENT_DTYPE")
        _require(
            receipt["outputs"][column]
            == {
                "source_column": column.removeprefix("asec_"),
                "dtype": "int64",
                "sha256": _sha(values.to_numpy(dtype="<i8", copy=False).tobytes()),
            },
            "ATTACHMENT_OUTPUT_DIGEST",
        )
    _require(
        np.array_equal(household.asec_H_SEQ.to_numpy(), keys.H_SEQ.to_numpy())
        and np.array_equal(
            household.asec_GESTFIPS.to_numpy(), household.state_fips.to_numpy()
        )
        and np.array_equal(
            household.asec_H_TENURE.to_numpy(), household.H_TENURE.to_numpy()
        ),
        "ATTACHMENT_SOURCE_JOIN",
    )
    cohorts = tuple(
        sorted(
            (row["year"], row["sha256"])
            for row in metadata["source_receipt"]["sources"]
        )
    )
    _require(cohorts == _SOURCE_PINS[2], "COHORT_PINS")
    rows = receipt["sources"]
    _require(len(rows) == 3, "ATTACHMENT_SOURCE_COVERAGE")
    for row, (year, pin) in zip(rows, cohorts, strict=True):
        _require(
            set(row)
            == {
                "income_year",
                "sha256",
                "source_rows",
                "joined_rows",
                "unreferenced_source_rows",
            }
            and row["income_year"] == year
            and row["sha256"] == pin
            and type(row["source_rows"]) is int
            and row["source_rows"] >= 0
            and row["joined_rows"] == int((keys.source_year == year).sum())
            and row["unreferenced_source_rows"]
            == row["source_rows"] - row["joined_rows"]
            >= 0,
            "ATTACHMENT_SOURCE_COVERAGE",
        )
    return cohorts


def _source_verification_identity(evidence):
    from .asec_current_money import RESTORED_SOURCE_KIND

    if evidence["source_kind"] == RESTORED_SOURCE_KIND:
        from .asec_person_income_source import _implementation

        return _implementation()
    return _verification_identity()


@dataclass(frozen=True)
class AuthenticatedCurrentMoneySource:
    frame: Frame
    scope: AsecMoneyScope
    source: AuthenticatedAsecSource
    spec: CurrentMoneySpec
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _LOAD_TOKEN, "SOURCE_CONSTRUCTOR_UNAVAILABLE")

    def validate(self):
        _require(
            type(self.source) is AuthenticatedAsecSource,
            "AUTHENTICATED_SOURCE_ISSUANCE",
        )
        self.source._validate()
        evidence = _parse(self.source.identity)
        _require(
            _source_verification_identity(evidence) == evidence["verification_sha256"],
            "SOURCE_IMPLEMENTATION_CHANGED",
        )
        try:
            _require(
                _frame_signature(self.frame) == evidence["frame_sha256"]
                and _scope(self.frame) == self.scope,
                "SOURCE_CHANGED",
            )
        except MoneyRefusalError:
            raise
        except (ValueError, TypeError, KeyError):
            raise MoneyRefusalError("SOURCE_CHANGED") from None

    def views(self):
        self.validate()
        from .asec_current_money import _restored_source

        return _views(self.frame, self.scope, restored=_restored_source(self.source))

    def ready(self):
        views = self.views()
        decoded = classify_asec_money(views, self.spec, self.scope)
        return require_complete_current_money(
            restate_asec_current_money(decoded, self.spec), self.spec, production=True
        )


def load_authenticated_current_money_source(parent_path, attachment_path):
    """Issue authority only for the two reviewed full-source files and exact scope."""
    try:
        return _load(parent_path, attachment_path)
    except MoneyRefusalError:
        raise
    except (ValueError, TypeError, KeyError, OSError):
        raise MoneyRefusalError("SOURCE_CONTRACT_REFUSAL") from None


def _load(parent_path, attachment_path):
    before = _verification_identity()
    with tempfile.TemporaryDirectory(prefix="microcosm-money-source-") as directory:
        directory = Path(directory)
        parent_copy = directory / "parent.h5"
        _stage_verified(parent_path, _SOURCE_PINS[0], parent_copy)
        parent, metadata = asec_checkpoint.load_asec_raw_stage_checkpoint_v4(
            parent_copy
        )
        parent_copy.unlink()
        attachment_copy = directory / "attachment.h5"
        _stage_verified(attachment_path, _SOURCE_PINS[1], attachment_copy)
        loaded = checkpoint.load_frame_checkpoint(attachment_copy)
        canonicalize_frame_string_dtypes(
            loaded.frame, boundary="ASEC money attachment", in_place=True
        )
    cohorts = _validate_attachment(parent, metadata, loaded, _SOURCE_PINS[0])
    sidecars = []
    actual_pins = metadata["raw_source_mappings"]["ED_VAL"]["source_pins"]
    _require(len(actual_pins) == 3, "ED_VAL_SOURCE_COVERAGE")
    for year in (2022, 2023, 2024):
        registered = ASEC_EDUCATION_ASSISTANCE_ARCHIVES[year]
        expected = {
            "income_year": year,
            "locator": registered.zip_url,
            "member": registered.member,
            "member_sha256": registered.member_sha256,
            "sha256": registered.zip_sha256,
        }
        _require(actual_pins.count(expected) == 1, "ED_VAL_REGISTERED_PIN")
        sidecars.append(
            {
                "income_year": year,
                "survey_year": registered.survey_year,
                "archive_sha256": registered.zip_sha256,
                "member": registered.member,
                "member_sha256": registered.member_sha256,
            }
        )
    frame = loaded.frame
    scope = _scope(frame)
    views = _views(frame, scope)
    scope_sha, input_sha = _input_binding(views)
    evidence = {
        "schema_version": 1,
        "source_kind": "asec_v4_with_household_observations_v1",
        "source_authentication": "checkpoint_bytes_verified",
        "parent_sha256": _SOURCE_PINS[0],
        "attachment_sha256": _SOURCE_PINS[1],
        "cohorts": cohorts,
        "sidecars": sidecars,
        "field_roster": FIELDS,
        "zero_origin_policy": ZERO_POLICY,
        "scope_sha256": scope_sha,
        "input_sha256": input_sha,
        "frame_sha256": _frame_signature(frame),
        "verification_sha256": before,
    }
    _require(_verification_identity() == before, "SOURCE_IMPLEMENTATION_CHANGED")
    authority = AuthenticatedAsecSource(_json(evidence), _token=_SOURCE_TOKEN)
    spec = compile_asec_current_money_spec(load_current_money_resources(), authority)
    return AuthenticatedCurrentMoneySource(
        frame, scope, authority, spec, _token=_LOAD_TOKEN
    )
