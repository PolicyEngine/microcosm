"""Pure ASEC current-money decoding and price restatement.

No resource discovery or source authentication happens here. Only the separate
resource loader reads installed code. Authenticated authority can only come from
the separate verified checkpoint loader; synthetic identities remain distinct.
"""

import json
import struct
import weakref
from dataclasses import InitVar, dataclass
from dataclasses import fields as dataclass_fields
from enum import IntEnum
from hashlib import sha256
from types import MappingProxyType

import numpy as np
import pandas as pd

RECIPE = "us_asec_current_money_ccpiu_2024_v1"
ZERO_POLICY = "frozen_fillna_zero_origin_unresolved"
RESTORED_SOURCE_KIND = "asec_v4_with_household_and_person_income_observations_v1"
ENCODED_ZERO_POLICY = "authenticated_census_csv_encoded_zero_v1"
RESTORED_FIELD_COLUMNS = MappingProxyType({"PTOTVAL": "asec_PTOTVAL"})
RESTORED_FIELD_ZERO_POLICY = MappingProxyType({"PTOTVAL": ENCODED_ZERO_POLICY})


class ZeroOrigin(IntEnum):
    """Source-encoding provenance; no code establishes a respondent answer."""

    NOT_ZERO = 0
    FROZEN_FILLNA_UNRESOLVED = 1
    AUTHENTICATED_CENSUS_ENCODED = 2


FIELDS = tuple(
    "ANN_VAL CAP_VAL CHSP_VAL CSP_VAL DIS_VAL1 DIS_VAL2 DIV_VAL DST_VAL1 DST_VAL1_YNG DST_VAL2 DST_VAL2_YNG ED_VAL FRSE_VAL HTOTVAL INT_VAL OI_VAL PHIP_VAL PMED_VAL PNSN_VAL POTC_VAL PTOTVAL RETCB_VAL RNT_VAL SEMP_VAL SPM_CAPHOUSESUB SPM_CHILDCAREXPNS SPM_ENGVAL SSI_VAL SS_VAL UC_VAL VET_VAL WC_VAL WSAL_VAL".split()
)
RESOURCE_PINS = (
    "09cefd4d08968cf1dc4dcdc787deef88c77b84549dfb0baa6bf9cbd9e3ce7352",
    "88cec207a3c73e5de93d677acf16e8db9ddae9761c644c06da92cacf0feb3f63",
    "c56058eaf08b68b2add4a5d2b366559ea6cc5238b039031be7094ef1f6457ff7",
)
MICROUNIT_VERSION = "0.1.0"
MICROUNIT_SHA256 = "79007065113f9600601433dc2633f4392c675e28b87a7383ab8a1f7c6279d47e"
RESOURCE_MAX_BYTES = 512 * 1024
HEADER_MAX_BYTES = 64 * 1024
MAX_PERSONS = 1_000_000
MAX_HOUSEHOLDS = 400_000
_STATE_TOKEN = object()
_SOURCE_TOKEN = object()
# Keep issuance outside publicly writable evidence objects. Keys use identity,
# not dataclass value equality: independently reconstructed evidence stays equal
# without lending its authority to an unissued copy. Weak references retire
# entries with their objects, and retained bytes are bounded by HEADER_MAX_BYTES.
_SOURCE_ISSUANCE: dict[int, tuple[weakref.ReferenceType, bytes]] = {}


class MoneyRefusalError(ValueError):
    """Sanitized field/reason refusal; never includes row identifiers or values."""

    def __init__(self, reason: str, field: str = "contract"):
        self.reason = reason
        self.field = field if field in FIELDS else "contract"
        super().__init__(f"{self.field}: {reason}")


def _require(condition: bool, reason: str, field: str = "contract") -> None:
    if not condition:
        raise MoneyRefusalError(reason, field)


def _sha(value: bytes) -> str:
    return sha256(value).hexdigest()


def _json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True
    ).encode("ascii")


def _parse(value: bytes, limit: int = RESOURCE_MAX_BYTES) -> dict:
    _require(type(value) is bytes and 0 < len(value) <= limit, "JSON_SIZE_OR_TYPE")

    def pairs(items):
        result = {}
        for key, item in items:
            _require(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = item
        return result

    def constant(_):
        raise MoneyRefusalError("NONFINITE_JSON")

    try:
        result = json.loads(value, object_pairs_hook=pairs, parse_constant=constant)
    except MoneyRefusalError:
        raise
    except (ValueError, RecursionError) as exc:
        raise MoneyRefusalError("MALFORMED_JSON") from exc
    _require(type(result) is dict, "JSON_OBJECT_REQUIRED")
    return result


def _digest(value: str) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


@dataclass(frozen=True)
class AuthenticatedAsecSource:
    """Immutable evidence issued only by the actual checkpoint verification loader."""

    evidence: bytes = b""
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _SOURCE_TOKEN, "AUTHENTICATED_SOURCE_UNAVAILABLE")
        self._validate_content()
        key = id(self)

        def discard(reference):
            issued = _SOURCE_ISSUANCE.get(key)
            if issued is not None and issued[0] is reference:
                del _SOURCE_ISSUANCE[key]

        _SOURCE_ISSUANCE[key] = (weakref.ref(self, discard), self.evidence)

    def _validate(self):
        issued = _SOURCE_ISSUANCE.get(id(self))
        evidence = self.evidence
        _require(
            type(self) is AuthenticatedAsecSource
            and issued is not None
            and issued[0]() is self
            and type(evidence) is bytes
            and evidence == issued[1],
            "AUTHENTICATED_SOURCE_ISSUANCE",
        )
        return evidence

    def _validate_content(self):
        data = _parse(self.evidence, HEADER_MAX_BYTES)
        restored = data.get("source_kind") == RESTORED_SOURCE_KIND
        extra = (
            {
                "person_income_attachment_sha256",
                "field_source_columns",
                "field_zero_origin_policy",
            }
            if restored
            else set()
        )
        _require(
            set(data)
            == extra
            | {
                "schema_version",
                "source_kind",
                "source_authentication",
                "parent_sha256",
                "attachment_sha256",
                "cohorts",
                "sidecars",
                "field_roster",
                "zero_origin_policy",
                "scope_sha256",
                "input_sha256",
                "frame_sha256",
                "verification_sha256",
            },
            "AUTHENTICATED_SOURCE_SCHEMA",
        )
        _require(
            data["schema_version"] == (2 if restored else 1)
            and data["source_kind"]
            == (
                RESTORED_SOURCE_KIND
                if restored
                else "asec_v4_with_household_observations_v1"
            )
            and data["source_authentication"] == "checkpoint_bytes_verified"
            and data["field_roster"] == list(FIELDS)
            and data["zero_origin_policy"] == ZERO_POLICY,
            "AUTHENTICATED_SOURCE_SCHEMA",
        )
        _require(
            all(
                _digest(data[name])
                for name in (
                    "parent_sha256",
                    "attachment_sha256",
                    "scope_sha256",
                    "input_sha256",
                    "frame_sha256",
                    "verification_sha256",
                )
            ),
            "SOURCE_PIN",
        )
        if restored:
            _require(
                _digest(data["person_income_attachment_sha256"])
                and data["field_source_columns"] == RESTORED_FIELD_COLUMNS
                and data["field_zero_origin_policy"] == RESTORED_FIELD_ZERO_POLICY,
                "RESTORED_SOURCE_SCHEMA",
            )
        _require(self.evidence == _json(data), "AUTHENTICATED_SOURCE_SCHEMA")

    @property
    def identity(self) -> bytes:
        # Return the same immutable bytes checked, not a second mutable lookup.
        return self._validate()


def _restored_source(source) -> bool:
    return (
        type(source) is AuthenticatedAsecSource
        and _parse(source.identity)["source_kind"] == RESTORED_SOURCE_KIND
    )


def _origin_code(spec, field: str) -> ZeroOrigin:
    return (
        ZeroOrigin.AUTHENTICATED_CENSUS_ENCODED
        if field == "PTOTVAL" and _restored_source(spec.source)
        else ZeroOrigin.FROZEN_FILLNA_UNRESOLVED
    )


def _header_provenance(spec) -> dict:
    if _restored_source(spec.source):
        return {
            "schema_version": 2,
            "field_zero_origin_policy": dict(RESTORED_FIELD_ZERO_POLICY),
        }
    return {"schema_version": 1}


def _authentication(source) -> str:
    return (
        "checkpoint_bytes_verified"
        if type(source) is AuthenticatedAsecSource
        else "synthetic_unverified"
    )


@dataclass(frozen=True)
class SyntheticAsecSource:
    """Explicit invented pins, never promoted to authenticated source authority."""

    parent_sha256: str
    attachment_sha256: str
    cohort_sha256: tuple[tuple[int, str], ...]
    sidecar_sha256: tuple[tuple[str, str], ...]
    selection_sha256: str
    source_kind: str = "asec_v4_with_household_observations_v1"
    field_roster: tuple[str, ...] = FIELDS
    zero_origin_policy: str = ZERO_POLICY

    def __post_init__(self):
        _require(
            all(
                _digest(v)
                for v in (
                    self.parent_sha256,
                    self.attachment_sha256,
                    self.selection_sha256,
                )
            ),
            "SOURCE_PIN",
        )
        _require(
            type(self.cohort_sha256) is tuple and len(self.cohort_sha256) == 3,
            "COHORT_PINS",
        )
        _require(
            all(
                type(p) is tuple and len(p) == 2 and type(p[0]) is int and _digest(p[1])
                for p in self.cohort_sha256
            ),
            "COHORT_PINS",
        )
        _require(
            tuple(p[0] for p in self.cohort_sha256) == (2022, 2023, 2024),
            "SOURCE_YEAR_MAPPING",
        )
        _require(
            type(self.sidecar_sha256) is tuple and len(self.sidecar_sha256) == 2,
            "SIDECAR_PINS",
        )
        _require(
            all(
                type(p) is tuple and len(p) == 2 and _digest(p[1])
                for p in self.sidecar_sha256
            ),
            "SIDECAR_PINS",
        )
        _require(
            tuple(p[0] for p in self.sidecar_sha256)
            == ("ED_VAL.archive", "ED_VAL.member"),
            "SIDECAR_PINS",
        )
        _require(
            self.source_kind == "asec_v4_with_household_observations_v1",
            "UNSUPPORTED_PARENT_KIND",
        )
        _require(
            type(self.field_roster) is tuple and self.field_roster == FIELDS,
            "FIELD_ROSTER",
        )
        _require(self.zero_origin_policy == ZERO_POLICY, "ZERO_ORIGIN_POLICY")

    @property
    def identity(self) -> bytes:
        return _json(
            {
                **{f.name: getattr(self, f.name) for f in dataclass_fields(self)},
                "source_authentication": "synthetic_unverified",
            }
        )


@dataclass(frozen=True)
class CurrentMoneyResources:
    """Explicit immutable resource bytes and inspected-code/runtime claims.

    Pure compilation checks these claims, not installed files. Use the separate
    loader for installed verification. Source authentication belongs to the
    separate closed checkpoint loader.
    """

    domains: bytes
    price: bytes
    consumers: bytes
    execution_identity: bytes
    inspected_version: str
    inspected_module_sha256: str

    def __post_init__(self):
        _require(
            all(
                type(x) is bytes
                for x in (
                    self.domains,
                    self.price,
                    self.consumers,
                    self.execution_identity,
                )
            ),
            "RESOURCE_BYTES_REQUIRED",
        )


@dataclass(frozen=True)
class MoneyDomain:
    name: str
    entity: str
    column: str
    grain: str
    minimum: int
    maximum: int
    zero_semantics: str


@dataclass(frozen=True)
class CurrentMoneySpec:
    """Closed v1 resources and issued source evidence; no mutable user mappings."""

    resources: CurrentMoneyResources
    source: SyntheticAsecSource | AuthenticatedAsecSource
    fields: tuple[MoneyDomain, ...]
    factors: tuple[tuple[int, str, str], ...]
    identity: bytes

    @property
    def sha256(self) -> str:
        return _sha(self.identity)


def compile_asec_current_money_spec(
    resources: CurrentMoneyResources,
    source_descriptor: SyntheticAsecSource | AuthenticatedAsecSource,
) -> CurrentMoneySpec:
    """Compile pinned declarations; this does not change live nominal consumers."""
    _require(
        type(resources) is CurrentMoneyResources
        and type(source_descriptor) in (SyntheticAsecSource, AuthenticatedAsecSource),
        "SYNTHETIC_SOURCE_REQUIRED",
    )
    if type(source_descriptor) is AuthenticatedAsecSource:
        source_descriptor._validate()
    else:
        source_descriptor.__post_init__()
    resources.__post_init__()
    documents = tuple(
        _parse(v) for v in (resources.domains, resources.price, resources.consumers)
    )
    # Immutable v1 pins close every nested key, domain, writer, year and address.
    # A changed convention requires a reviewed resource/recipe revision.
    _require(
        tuple(
            _sha(v) for v in (resources.domains, resources.price, resources.consumers)
        )
        == RESOURCE_PINS,
        "RESOURCE_FINGERPRINT",
    )
    _require(
        resources.inspected_version == MICROUNIT_VERSION
        and resources.inspected_module_sha256 == MICROUNIT_SHA256,
        "MICROUNIT_CONTRACT_MISMATCH",
    )
    execution = _parse(resources.execution_identity, HEADER_MAX_BYTES)
    _require(
        set(execution) == {"modules", "dependencies", "platform"}, "EXECUTION_IDENTITY"
    )
    _require(
        type(execution["modules"]) is dict
        and set(execution["modules"])
        == {
            "asec_current_money.py",
            "_asec_current_money_codec.py",
            "asec_current_money_resources.py",
        },
        "EXECUTION_MODULES",
    )
    _require(
        all(_digest(v) for v in execution["modules"].values()), "EXECUTION_MODULES"
    )
    _require(
        type(execution["dependencies"]) is dict
        and set(execution["dependencies"]) == {"python", "numpy", "pandas"},
        "EXECUTION_DEPENDENCIES",
    )
    _require(
        type(execution["platform"]) is dict
        and set(execution["platform"])
        == {"system", "machine", "byteorder", "python_implementation"},
        "EXECUTION_PLATFORM",
    )
    _require(
        all(
            type(v) is str and 0 < len(v) <= 128
            for group in (execution["dependencies"], execution["platform"])
            for v in group.values()
        ),
        "EXECUTION_IDENTITY",
    )
    domains, price, _ = documents
    fields = tuple(
        MoneyDomain(
            **{
                k: f[k]
                for k in (
                    "name",
                    "entity",
                    "column",
                    "grain",
                    "minimum",
                    "maximum",
                    "zero_semantics",
                )
            }
        )
        for f in domains["fields"]
    )
    factors = tuple(
        (
            y,
            f"174.4/{price['cells'][str(y)]}",
            struct.pack("<d", float("174.4") / float(price["cells"][str(y)])).hex(),
        )
        for y in (2022, 2023, 2024)
    )
    identity = _json(
        {
            "recipe": RECIPE,
            "source": _parse(source_descriptor.identity),
            "resource_sha256": RESOURCE_PINS,
            "execution": execution,
            "factors": factors,
            "numeric_scope": "platform_bitwise",
            "source_authentication": _authentication(source_descriptor),
        }
    )
    return CurrentMoneySpec(resources, source_descriptor, fields, factors, identity)


def _spec(spec: CurrentMoneySpec) -> None:
    _require(type(spec) is CurrentMoneySpec, "SPEC_TYPE")
    _require(
        spec == compile_asec_current_money_spec(spec.resources, spec.source),
        "SPEC_BINDING",
    )


@dataclass(frozen=True)
class AsecMoneyScope:
    """Ordered source coordinates supplied explicitly with projected tables.

    Pandas labels are representation only. IDs/native keys here belong to each
    positional row, and must be projected in the same permutation as the table.
    No original source authentication is inferred from this synthetic contract.
    """

    person_ids: tuple[int, ...]
    household_ids: tuple[int, ...]
    person_household_ids: tuple[int, ...]
    person_spm_ids: tuple[int, ...]
    person_years: tuple[int, ...]
    household_years: tuple[int, ...]
    person_native_keys: tuple[str, ...]
    household_native_keys: tuple[str, ...]

    def __post_init__(self):
        p, h = len(self.person_ids), len(self.household_ids)
        _require(0 < p <= MAX_PERSONS and 0 < h <= MAX_HOUSEHOLDS, "SOURCE_SIZE")
        for name in (
            "person_ids",
            "household_ids",
            "person_household_ids",
            "person_spm_ids",
            "person_years",
            "household_years",
        ):
            vector = getattr(self, name)
            _require(
                type(vector) is tuple
                and all(type(x) is int and -(2**63) <= x < 2**63 for x in vector),
                "SCOPE_INTEGER_VECTOR",
            )
            _require(
                len(vector) == (h if name.startswith("household") else p),
                "SCOPE_LENGTH",
            )
        _require(len(set(self.person_ids)) == p, "DUPLICATE_PERSON_ID")
        _require(len(set(self.household_ids)) == h, "DUPLICATE_HOUSEHOLD_ID")
        _require(
            set(self.person_household_ids) == set(self.household_ids),
            "HOUSEHOLD_MEMBERSHIP",
        )
        _require(
            set(self.person_years) <= {2022, 2023, 2024}
            and set(self.household_years) <= {2022, 2023, 2024},
            "SOURCE_YEAR_MAPPING",
        )
        expected = dict(zip(self.household_ids, self.household_years, strict=True))
        _require(
            all(
                expected[g] == y
                for g, y in zip(
                    self.person_household_ids, self.person_years, strict=True
                )
            ),
            "MIXED_HOUSEHOLD_YEAR",
        )
        spm_years = {}
        for spm, year in zip(self.person_spm_ids, self.person_years, strict=True):
            _require(spm not in spm_years or spm_years[spm] == year, "MIXED_SPM_YEAR")
            spm_years[spm] = year
        for name, years, count in (
            ("person_native_keys", self.person_years, p),
            ("household_native_keys", self.household_years, h),
        ):
            keys = getattr(self, name)
            _require(
                type(keys) is tuple
                and len(keys) == count
                and all(type(k) is str and 0 < len(k) <= 256 for k in keys),
                "NATIVE_KEYS",
            )
            _require(
                len(set(zip(years, keys, strict=True))) == count, "DUPLICATE_NATIVE_KEY"
            )

    @property
    def identity(self) -> bytes:
        # Digests bound the header, while the active scope retains real coordinates.
        return _json(
            {f.name: _sha(_json(getattr(self, f.name))) for f in dataclass_fields(self)}
        )


@dataclass(frozen=True)
class AsecMoneyViews:
    """Explicit raw projections plus their ordered coordinate association."""

    person: pd.DataFrame
    household: pd.DataFrame
    scope: AsecMoneyScope


class CodebookStatus(IntEnum):
    MISSING_NULL = 0
    AMOUNT_NONZERO = 1
    ZERO_NONE_OR_NIU = 2
    ZERO_DOLLARS_AS_CODED = 3
    DECLARED_NIU = 4


_ZERO_STATUS = {
    "none_or_niu_not_distinguishable_from_amount_alone": CodebookStatus.ZERO_NONE_OR_NIU,
    "valid_zero_dollars": CodebookStatus.ZERO_DOLLARS_AS_CODED,
    "valid_zero_dollars_or_none_as_described": CodebookStatus.ZERO_DOLLARS_AS_CODED,
    "niu": CodebookStatus.DECLARED_NIU,
}
_DTYPES = frozenset(
    (
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float64",
        "Int8",
        "UInt8",
        "Int16",
        "UInt16",
        "Int32",
        "UInt32",
        "Int64",
        "UInt64",
        "Float64",
    )
)


@dataclass(frozen=True)
class MoneyField:
    """Immutable canonical little-endian amounts and three uint8 evidence axes."""

    name: str
    amount_bytes: bytes
    status_bytes: bytes
    validity_bytes: bytes
    zero_origin_bytes: bytes

    @property
    def amounts(self) -> np.ndarray:
        return np.frombuffer(self.amount_bytes, dtype="<f8")

    @property
    def statuses(self) -> np.ndarray:
        return np.frombuffer(self.status_bytes, dtype="u1")

    @property
    def validity(self) -> np.ndarray:
        return np.frombuffer(self.validity_bytes, dtype="u1")

    @property
    def zero_origin(self) -> np.ndarray:
        return np.frombuffer(self.zero_origin_bytes, dtype="u1")


def _validate_field(
    field: MoneyField,
    domain: MoneyDomain,
    count: int,
    *,
    nominal: bool,
    zero_origin_code: ZeroOrigin,
) -> None:
    _require(type(field) is MoneyField and field.name == domain.name, "FIELD_IDENTITY")
    buffers = (
        field.amount_bytes,
        field.status_bytes,
        field.validity_bytes,
        field.zero_origin_bytes,
    )
    _require(
        all(type(v) is bytes for v in buffers)
        and tuple(map(len, buffers)) == (8 * count, count, count, count),
        "FIELD_BUFFER_SHAPE",
        domain.name,
    )
    a, s, v, z = field.amounts, field.statuses, field.validity, field.zero_origin
    _require(
        np.isfinite(a).all()
        and np.isin(s, [0, 1, 2, 3, 4]).all()
        and np.isin(v, [0, 1]).all()
        and np.isin(z, [0, 1, 2]).all(),
        "FIELD_ENCODING",
        domain.name,
    )
    _require(
        np.array_equal(v == 0, s == CodebookStatus.MISSING_NULL),
        "INVALID_SLOT",
        domain.name,
    )
    _require(
        np.array_equal(a != 0, s == CodebookStatus.AMOUNT_NONZERO)
        and not np.signbit(a[a == 0]).any(),
        "AMOUNT_STATUS_ENCODING",
        domain.name,
    )
    expected_zero = _ZERO_STATUS[domain.zero_semantics]
    allowed = {
        CodebookStatus.MISSING_NULL,
        CodebookStatus.AMOUNT_NONZERO,
        expected_zero,
    }
    if domain.name == "ANN_VAL":
        allowed.add(CodebookStatus.DECLARED_NIU)
    _require(np.isin(s, list(allowed)).all(), "FIELD_STATUS", domain.name)
    _require(
        type(zero_origin_code) is ZeroOrigin
        and zero_origin_code != ZeroOrigin.NOT_ZERO,
        "ZERO_ORIGIN_POLICY",
        domain.name,
    )
    expected_origin = (s == expected_zero).astype("u1") * int(zero_origin_code)
    _require(np.array_equal(z, expected_origin), "ZERO_ORIGIN_ENCODING", domain.name)
    factor_max = 1.0 if nominal else float("174.4") / float("163.6")
    lo = min(domain.minimum, domain.minimum * factor_max)
    hi = max(domain.maximum, domain.maximum * factor_max)
    _require(((a >= lo) & (a <= hi)).all(), "AMOUNT_DOMAIN", domain.name)
    if domain.name == "ANN_VAL":
        _require((a >= 0).all(), "AMOUNT_DOMAIN", domain.name)
    if nominal:
        _require((a == np.floor(a)).all(), "FRACTIONAL_SOURCE", domain.name)


@dataclass(frozen=True)
class MoneyBindings:
    """Expected replay identity, including the actual raw input fingerprint."""

    spec: CurrentMoneySpec
    header: bytes

    def __post_init__(self):
        _spec(self.spec)
        data = _parse(self.header, HEADER_MAX_BYTES)
        keys = {
            "schema_version",
            "recipe",
            "country",
            "semantic",
            "target_year",
            "source_authentication",
            "state",
            "spec_sha256",
            "scope_sha256",
            "input_sha256",
            "person_rows",
            "household_rows",
            "fields",
            "dtype",
            "evidence_dtype",
            "person_year_sha256",
            "household_year_sha256",
        }
        provenance = _header_provenance(self.spec)
        _require(
            set(data) == keys | set(provenance) and self.header == _json(data),
            "HEADER_SCHEMA",
        )
        fixed = {
            **provenance,
            "recipe": RECIPE,
            "country": "us",
            "semantic": "annual_current_money",
            "target_year": 2024,
            "source_authentication": _authentication(self.spec.source),
            "state": "target_current",
            "spec_sha256": self.spec.sha256,
            "fields": list(FIELDS),
            "dtype": "<f8",
            "evidence_dtype": "|u1",
        }
        _require(
            all(type(data[k]) is type(v) and data[k] == v for k, v in fixed.items()),
            "HEADER_BINDING",
        )
        _require(
            all(
                _digest(data[k])
                for k in (
                    "scope_sha256",
                    "input_sha256",
                    "person_year_sha256",
                    "household_year_sha256",
                )
            ),
            "HEADER_DIGEST",
        )
        if type(self.spec.source) is AuthenticatedAsecSource:
            evidence = _parse(self.spec.source.identity)
            _require(
                data["scope_sha256"] == evidence["scope_sha256"]
                and data["input_sha256"] == evidence["input_sha256"],
                "AUTHENTICATED_HEADER_BINDING",
            )
        _require(
            type(data["person_rows"]) is int
            and 0 < data["person_rows"] <= MAX_PERSONS
            and type(data["household_rows"]) is int
            and 0 < data["household_rows"] <= MAX_HOUSEHOLDS,
            "HEADER_ROW_BOUND",
        )


@dataclass(frozen=True)
class DecodedAsecMoney:
    bindings: MoneyBindings
    fields: tuple[MoneyField, ...]
    person_year_bytes: bytes
    household_year_bytes: bytes
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _STATE_TOKEN, "STATE_CONSTRUCTOR_UNAVAILABLE")

    def field(self, name: str) -> MoneyField:
        return _field(self.fields, name)


@dataclass(frozen=True)
class RestatedAsecMoney:
    bindings: MoneyBindings
    fields: tuple[MoneyField, ...]
    _token: InitVar[object] = None

    def __post_init__(self, _token):
        _require(_token is _STATE_TOKEN, "STATE_CONSTRUCTOR_UNAVAILABLE")

    @property
    def header(self) -> bytes:
        return self.bindings.header

    def field(self, name: str) -> MoneyField:
        return _field(self.fields, name)


@dataclass(frozen=True)
class SyntheticReadyCurrentMoney(RestatedAsecMoney):
    """Complete invented fixture amounts. This is NOT production authority."""


class ReadyCurrentMoney(RestatedAsecMoney):
    """Complete amounts bound to verified source bytes; not release certification."""

    def __new__(cls, *args, **kwargs):
        _require(
            kwargs.get("_token") is _STATE_TOKEN, "PRODUCTION_READINESS_UNAVAILABLE"
        )
        return super().__new__(cls)


def _field(fields: tuple[MoneyField, ...], name: str) -> MoneyField:
    _require(name in FIELDS, "UNKNOWN_FIELD")
    return fields[FIELDS.index(name)]


def _validate_money(value, spec: CurrentMoneySpec, *, nominal: bool) -> None:
    _spec(spec)
    _require(
        type(value.bindings) is MoneyBindings and value.bindings.spec == spec,
        "SPEC_MISMATCH",
    )
    value.bindings.__post_init__()
    header = _parse(value.bindings.header, HEADER_MAX_BYTES)
    _require(
        type(value.fields) is tuple and len(value.fields) == len(FIELDS), "FIELD_ROSTER"
    )
    for domain, field in zip(spec.fields, value.fields, strict=True):
        _validate_field(
            field,
            domain,
            header[domain.entity + "_rows"],
            nominal=nominal,
            zero_origin_code=_origin_code(spec, domain.name),
        )


def _index_identity(index: pd.Index) -> bytes:
    # An intentionally narrow, honest representation contract; duplicate/nondefault
    # signed-int64 labels are supported positionally. No label-based assignment.
    _require(
        type(index) in (pd.Index, pd.RangeIndex) and str(index.dtype) == "int64",
        "UNSUPPORTED_INDEX",
    )
    _require(index.name is None or type(index.name) is str, "UNSUPPORTED_INDEX_NAME")
    descriptor = {
        "type": type(index).__name__,
        "name": index.name,
        "values_sha256": _sha(index.to_numpy(dtype="<i8").tobytes()),
    }
    if type(index) is pd.RangeIndex:
        descriptor.update(start=index.start, stop=index.stop, step=index.step)
    return _json(descriptor)


def classify_asec_money(
    raw_views: AsecMoneyViews, spec: CurrentMoneySpec, source_scope: AsecMoneyScope
) -> DecodedAsecMoney:
    """Decode explicit raw columns positionally, preserving codebook evidence."""
    _require(type(raw_views) is AsecMoneyViews, "TYPED_STATE")
    _spec(spec)
    _require(
        type(source_scope) is AsecMoneyScope
        and type(raw_views.scope) is AsecMoneyScope,
        "SCOPE_TYPE",
    )
    source_scope.__post_init__()
    _require(raw_views.scope == source_scope, "SCOPE_MISMATCH")
    p, h = len(source_scope.person_ids), len(source_scope.household_ids)
    for table, count in ((raw_views.person, p), (raw_views.household, h)):
        _require(
            type(table) is pd.DataFrame
            and len(table) == count
            and table.columns.is_unique,
            "VIEW_SHAPE",
        )
    scope_identity = _json(
        {
            "coordinates": _parse(source_scope.identity),
            "person_index": _parse(_index_identity(raw_views.person.index)),
            "household_index": _parse(_index_identity(raw_views.household.index)),
        }
    )
    result = []
    inputs = []
    # Sorting native group IDs is only a temporary positional validation index.
    spm = np.asarray(source_scope.person_spm_ids, dtype="<i8")
    order = np.argsort(spm, kind="stable")
    same = np.diff(spm[order]) == 0
    for domain in spec.fields:
        table = getattr(raw_views, domain.entity)
        _require(domain.column in table, "MISSING_COLUMN", domain.name)
        series = table[domain.column]
        _require(str(series.dtype) in _DTYPES, "UNSUPPORTED_DTYPE", domain.name)
        values = series.to_numpy(dtype="<f8", na_value=np.nan, copy=True)
        valid = ~series.isna().to_numpy()
        _require(np.isfinite(values[valid]).all(), "NONFINITE_SOURCE", domain.name)
        _require(
            (np.floor(values[valid]) == values[valid]).all(),
            "FRACTIONAL_SOURCE",
            domain.name,
        )
        if domain.name == "SPM_ENGVAL":
            _require(
                not ((values >= 10001) & (values <= 99999) & valid).any(),
                "DISPUTED_CODEBOOK_RANGE",
                domain.name,
            )
        _require(
            not (((values < domain.minimum) | (values > domain.maximum)) & valid).any(),
            "OUTSIDE_CODEBOOK_RANGE",
            domain.name,
        )
        # Bind received numeric bytes (including -0) and validity before normalization.
        received = values.copy()
        received[~valid] = 0.0
        inputs.append(
            {
                "field": domain.name,
                "entity": domain.entity,
                "column": domain.column,
                "dtype": str(series.dtype),
                "values_sha256": _sha(received.tobytes()),
                "validity_sha256": _sha(valid.astype("u1").tobytes()),
            }
        )
        status = np.full(len(values), CodebookStatus.AMOUNT_NONZERO, dtype="u1")
        zeros = valid & (values == 0)
        status[zeros] = _ZERO_STATUS[domain.zero_semantics]
        if domain.name == "ANN_VAL":
            niu = valid & (values == -1)
            status[niu] = CodebookStatus.DECLARED_NIU
            values[niu] = 0.0
        status[~valid] = CodebookStatus.MISSING_NULL
        values[~valid | (values == 0)] = 0.0
        field = MoneyField(
            domain.name,
            values.tobytes(),
            status.tobytes(),
            valid.astype("u1").tobytes(),
            (zeros.astype("u1") * int(_origin_code(spec, domain.name))).tobytes(),
        )
        _validate_field(
            field,
            domain,
            len(table),
            nominal=True,
            zero_origin_code=_origin_code(spec, domain.name),
        )
        if domain.grain == "spm_unit_repeated_on_person":
            for vector in (values, status, valid, zeros):
                sorted_values = vector[order]
                _require(
                    not ((sorted_values[1:] != sorted_values[:-1]) & same).any(),
                    "INCONSISTENT_SPM_AMOUNT",
                    domain.name,
                )
        result.append(field)
    header = _json(
        {
            **_header_provenance(spec),
            "recipe": RECIPE,
            "country": "us",
            "semantic": "annual_current_money",
            "target_year": 2024,
            "source_authentication": _authentication(spec.source),
            "state": "target_current",
            "spec_sha256": spec.sha256,
            "scope_sha256": _sha(scope_identity),
            "input_sha256": _sha(_json(inputs)),
            "person_year_sha256": _sha(
                np.asarray(source_scope.person_years, dtype="<i8").tobytes()
            ),
            "household_year_sha256": _sha(
                np.asarray(source_scope.household_years, dtype="<i8").tobytes(),
            ),
            "person_rows": p,
            "household_rows": h,
            "fields": FIELDS,
            "dtype": "<f8",
            "evidence_dtype": "|u1",
        }
    )
    if type(spec.source) is AuthenticatedAsecSource:
        expected = _parse(spec.source.identity)
        _require(
            expected["scope_sha256"] == _sha(scope_identity)
            and expected["input_sha256"] == _sha(_json(inputs)),
            "AUTHENTICATED_INPUT_MISMATCH",
        )
    return DecodedAsecMoney(
        MoneyBindings(spec, header),
        tuple(result),
        np.asarray(source_scope.person_years, dtype="<i8").tobytes(),
        np.asarray(source_scope.household_years, dtype="<i8").tobytes(),
        _token=_STATE_TOKEN,
    )


def restate_asec_current_money(
    decoded: DecodedAsecMoney, spec: CurrentMoneySpec
) -> RestatedAsecMoney:
    """Restate once; target-year bytes and normalized zero bypass arithmetic."""
    _require(type(decoded) is DecodedAsecMoney, "TYPED_STATE")
    _validate_money(decoded, spec, nominal=True)
    header = _parse(decoded.bindings.header, HEADER_MAX_BYTES)
    years = {}
    for entity in ("person", "household"):
        data = getattr(decoded, entity + "_year_bytes")
        _require(
            type(data) is bytes and len(data) == header[entity + "_rows"] * 8,
            "YEAR_BUFFER",
        )
        _require(_sha(data) == header[entity + "_year_sha256"], "YEAR_BINDING")
        vector = np.frombuffer(data, dtype="<i8")
        _require(np.isin(vector, [2022, 2023, 2024]).all(), "SOURCE_YEAR_MAPPING")
        years[entity] = vector
    result = []
    for domain, field in zip(spec.fields, decoded.fields, strict=True):
        amounts = field.amounts.copy()
        for year, _, bits in spec.factors:
            if year == 2024:
                continue
            selection = (years[domain.entity] == year) & (amounts != 0)
            amounts[selection] = (
                amounts[selection] * struct.unpack("<d", bytes.fromhex(bits))[0]
            )
        result.append(
            MoneyField(
                field.name,
                amounts.tobytes(),
                field.status_bytes,
                field.validity_bytes,
                field.zero_origin_bytes,
            )
        )
    value = RestatedAsecMoney(decoded.bindings, tuple(result), _token=_STATE_TOKEN)
    _validate_money(value, spec, nominal=False)
    return value


def require_complete_current_money(
    restated: RestatedAsecMoney, spec: CurrentMoneySpec, *, production: bool = False
) -> SyntheticReadyCurrentMoney | ReadyCurrentMoney:
    """Require every amount; synthetic evidence cannot request verified readiness."""
    _require(type(production) is bool, "PRODUCTION_FLAG")
    verified = type(spec.source) is AuthenticatedAsecSource
    _require(not production or verified, "SYNTHETIC_NOT_PRODUCTION")
    _require(not verified or production, "AUTHENTICATED_READINESS_REQUIRED")
    _require(type(restated) is RestatedAsecMoney, "TYPED_STATE")
    _validate_money(restated, spec, nominal=False)
    for field in restated.fields:
        _require(field.validity.all(), "MISSING_REQUIRED_AMOUNT", field.name)
    ready_type = ReadyCurrentMoney if verified else SyntheticReadyCurrentMoney
    return ready_type(restated.bindings, restated.fields, _token=_STATE_TOKEN)
