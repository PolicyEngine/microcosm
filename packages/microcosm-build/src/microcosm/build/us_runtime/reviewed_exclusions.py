"""Reviewed-exclusion policy for the US stacked by-origin battery.

The schema and lifecycle rules deliberately mirror the UK schema-2 register.
US keys add one constraint: every entry names an exact ``leg#criterion`` so an
approval for one comparator outcome cannot suppress another outcome on the
same physical leg.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType

__all__ = [
    "US_BY_ORIGIN_EXCLUSION_REGISTER_RESOURCE",
    "USByOriginReviewedExclusionPolicy",
    "USReviewedExclusion",
    "coerce_us_reviewed_exclusions",
    "load_us_by_origin_reviewed_exclusion_policy",
    "require_authenticated_us_by_origin_reviewed_exclusion_policy",
    "us_default_by_origin_reviewed_exclusion_policy",
    "us_exclusion_evaluation_date",
    "validate_us_by_origin_reviewed_exclusion_source_provenance",
]

US_BY_ORIGIN_EXCLUSION_REGISTER_RESOURCE = "by_origin_reviewed_exclusions.json"
_REVIEWED_EXCLUSION_FIELDS = frozenset(
    {"reason", "approved_by", "adjudication", "approved_on", "expires_on"}
)
_ELIGIBLE_CRITERIA = frozenset({"categorical", "dead", "incidence", "quantile"})
_BATTERY_NAME_TOKEN = r"[a-z][a-z0-9_]*"
_BATTERY_TARGET_TOKEN = (
    rf"(?:{_BATTERY_NAME_TOKEN}|"
    rf"joint\[{_BATTERY_NAME_TOKEN}(?:,{_BATTERY_NAME_TOKEN})+\])"
)
_EXACT_CHECK_ID_PATTERN = re.compile(
    rf"^(?P<entity>{_BATTERY_NAME_TOKEN})/"
    rf"(?P<family>{_BATTERY_NAME_TOKEN})/"
    rf"(?P<target>{_BATTERY_TARGET_TOKEN})"
    r"\[clone_(?P<clone_index>0|[1-9][0-9]*)\]"
    r"(?P<sign>/positive|/negative)?#"
    r"(?P<criterion>categorical|dead|incidence|quantile)$"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_check_id(check_id: object, *, label: str) -> str:
    if (
        not isinstance(check_id, str)
        or not check_id.strip()
        or check_id != check_id.strip()
    ):
        raise ValueError(
            f"{label} reviewed exclusions need non-empty, trimmed exact "
            f"leg#criterion IDs; got {check_id!r}."
        )
    match = _EXACT_CHECK_ID_PATTERN.fullmatch(check_id)
    if match is None:
        raise ValueError(
            f"{label} reviewed exclusion key {check_id!r} must be one exact "
            "battery leg#criterion ID: entity/family/target[clone_N], with an "
            "optional /positive or /negative monetary leg, and criterion in "
            f"{sorted(_ELIGIBLE_CRITERIA)}."
        )
    sign = match.group("sign")
    criterion = match.group("criterion")
    target = match.group("target")
    if (sign is None and criterion == "quantile") or (
        sign is not None and criterion not in {"incidence", "quantile"}
    ):
        raise ValueError(
            f"{label} reviewed exclusion key {check_id!r} has an incompatible "
            "leg#criterion combination: quantile criteria require /positive or "
            "/negative, while signed legs permit only incidence or quantile."
        )
    if target.startswith("joint["):
        joint_columns = target.removeprefix("joint[").removesuffix("]").split(",")
        if sign is not None or criterion != "categorical":
            raise ValueError(
                f"{label} reviewed exclusion key {check_id!r} has an "
                "incompatible leg#criterion combination: joint targets permit "
                "only unsigned categorical criteria."
            )
        if len(set(joint_columns)) != len(joint_columns):
            raise ValueError(
                f"{label} reviewed exclusion key {check_id!r} must name unique "
                "columns in an exact joint leg#criterion ID."
            )
    return check_id


@dataclass(frozen=True)
class USReviewedExclusion:
    """One exact criterion exclusion and its complete approval receipt."""

    reason: str
    approved_by: str
    adjudication: str
    approved_on: str
    expires_on: str

    def __post_init__(self) -> None:
        for name in ("reason", "approved_by", "adjudication"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"US reviewed exclusion {name} must be a non-empty string."
                )
            if value != value.strip():
                raise ValueError(
                    f"US reviewed exclusion {name} must not carry surrounding "
                    "whitespace because the raw value is sealed into the policy "
                    f"digest; got {value!r}."
                )
        parsed: dict[str, date] = {}
        for name in ("approved_on", "expires_on"):
            value = getattr(self, name)
            try:
                parsed[name] = date.fromisoformat(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"US reviewed exclusion {name} must be an ISO date "
                    f"(YYYY-MM-DD), got {value!r}."
                ) from exc
            if parsed[name].isoformat() != value:
                raise ValueError(
                    f"US reviewed exclusion {name} must be an ISO date "
                    f"(YYYY-MM-DD), got {value!r}."
                )
        if parsed["expires_on"] <= parsed["approved_on"]:
            raise ValueError(
                "US reviewed exclusion expires_on must be after approved_on; "
                f"got approved_on={self.approved_on!r}, "
                f"expires_on={self.expires_on!r}."
            )

    def expired(self, now: date) -> bool:
        """Remain in force through the expiry date."""

        return now > date.fromisoformat(self.expires_on)

    def premature(self, now: date) -> bool:
        """Take force on the approval date."""

        return now < date.fromisoformat(self.approved_on)

    def policy_payload(self) -> dict[str, str]:
        """Return all five receipt fields for digesting and manifests."""

        return {
            "reason": self.reason,
            "approved_by": self.approved_by,
            "adjudication": self.adjudication,
            "approved_on": self.approved_on,
            "expires_on": self.expires_on,
        }


def coerce_us_reviewed_exclusions(
    values: object,
    *,
    label: str,
) -> dict[str, USReviewedExclusion]:
    """Validate raw schema-2 entries into sorted typed receipts."""

    if not isinstance(values, Mapping):
        raise TypeError(f"{label} reviewed exclusions must be a mapping.")
    records: dict[str, USReviewedExclusion] = {}
    for raw_check_id, entry in values.items():
        check_id = _validate_check_id(raw_check_id, label=label)
        if isinstance(entry, USReviewedExclusion):
            records[check_id] = entry
            continue
        if not isinstance(entry, Mapping):
            raise TypeError(
                f"{label} reviewed exclusion {check_id!r} must be an object "
                f"with fields {sorted(_REVIEWED_EXCLUSION_FIELDS)} (schema 2); "
                f"got {type(entry).__name__}."
            )
        if set(entry) != _REVIEWED_EXCLUSION_FIELDS:
            raise ValueError(
                f"{label} reviewed exclusion {check_id!r} fields must be "
                f"exactly {sorted(_REVIEWED_EXCLUSION_FIELDS)}, got "
                f"{sorted(entry)}."
            )
        try:
            records[check_id] = USReviewedExclusion(
                **{
                    field_name: entry[field_name]
                    for field_name in sorted(_REVIEWED_EXCLUSION_FIELDS)
                }
            )
        except ValueError as exc:
            raise ValueError(f"{label}: exclusion {check_id!r}: {exc}") from exc
    return dict(sorted(records.items()))


@dataclass(frozen=True)
class _USByOriginPolicySourceReceipt:
    """Path-free proof that one loader hashed and parsed the same bytes."""

    override_supplied: bool
    expected_sha256: str | None
    raw_source_sha256: str

    def __post_init__(self) -> None:
        if type(self.override_supplied) is not bool:
            raise TypeError("override_supplied must be a bool.")
        if not isinstance(self.raw_source_sha256, str) or not _SHA256_PATTERN.fullmatch(
            self.raw_source_sha256
        ):
            raise ValueError("raw_source_sha256 must be a lowercase SHA-256 digest.")
        if self.override_supplied:
            if (
                not isinstance(self.expected_sha256, str)
                or not _SHA256_PATTERN.fullmatch(self.expected_sha256)
                or self.expected_sha256 != self.raw_source_sha256
            ):
                raise ValueError(
                    "An override source receipt requires matching expected and "
                    "raw SHA-256 digests."
                )
        elif self.expected_sha256 is not None:
            raise ValueError(
                "The committed source receipt must not carry an expected SHA-256."
            )

    def policy_payload(self) -> dict[str, object]:
        return {
            "override_supplied": self.override_supplied,
            "expected_sha256": self.expected_sha256,
            "raw_source_sha256": self.raw_source_sha256,
        }


@dataclass(frozen=True)
class USByOriginReviewedExclusionPolicy:
    """One fully armed, immutable register snapshot."""

    description: str
    exclusions: Mapping[str, USReviewedExclusion]
    policy_source: str
    _source_receipt: _USByOriginPolicySourceReceipt | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError(
                "US by-origin exclusion register description must be a "
                "non-empty string."
            )
        if self.policy_source not in {"committed", "override"}:
            raise ValueError(
                "US by-origin exclusion policy_source must be 'committed' or "
                f"'override', got {self.policy_source!r}."
            )
        object.__setattr__(
            self,
            "exclusions",
            MappingProxyType(
                coerce_us_reviewed_exclusions(
                    self.exclusions,
                    label="US by-origin battery",
                )
            ),
        )

    def policy_payload(self) -> dict[str, object]:
        """Return the exact schema-2 content governed by this snapshot."""

        return {
            "schema_version": 2,
            "description": self.description,
            "exclusions": {
                check_id: record.policy_payload()
                for check_id, record in sorted(self.exclusions.items())
            },
        }

    @property
    def policy_sha256(self) -> str:
        """Canonical digest over the description and every receipt field."""

        return _canonical_sha256(self.policy_payload())

    @property
    def source_provenance(self) -> dict[str, object] | None:
        """Return loader-minted path-free provenance for the parsed bytes."""

        if self._source_receipt is None:
            return None
        return self._source_receipt.policy_payload()


def _strict_json_payload(raw: str, *, label: str) -> Mapping[str, object]:
    def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        duplicates: set[str] = set()
        for name, value in pairs:
            if name in result:
                duplicates.add(name)
            result[name] = value
        if duplicates:
            raise ValueError(f"{label}: duplicate JSON key(s): {sorted(duplicates)}.")
        return result

    def reject_nonfinite_json(value: str) -> object:
        raise ValueError(f"{label}: non-finite JSON value {value!r} is invalid.")

    try:
        payload = json.loads(
            raw,
            object_pairs_hook=strict_object,
            parse_constant=reject_nonfinite_json,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}: malformed JSON: {exc.msg}.") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label}: exclusion register must be a JSON object.")
    return payload


def _policy_from_raw(
    raw: str,
    *,
    label: str,
    policy_source: str,
) -> USByOriginReviewedExclusionPolicy:
    payload = _strict_json_payload(raw, label=label)
    expected_fields = {"schema_version", "description", "exclusions"}
    if set(payload) != expected_fields:
        raise ValueError(
            f"{label}: exclusion register fields must be exactly "
            f"{sorted(expected_fields)}, got {sorted(payload)}."
        )
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 2
    ):
        raise ValueError(
            f"{label}: exclusion register schema_version must be 2, got "
            f"{payload.get('schema_version')!r}."
        )
    description = payload.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(
            f"{label}: exclusion register description must be a non-empty string."
        )
    exclusions = payload.get("exclusions")
    if not isinstance(exclusions, Mapping):
        raise ValueError(
            f"{label}: exclusion register must carry an 'exclusions' object."
        )
    return USByOriginReviewedExclusionPolicy(
        description=description,
        exclusions=coerce_us_reviewed_exclusions(exclusions, label=label),
        policy_source=policy_source,
    )


def _decode_policy_bytes(raw_bytes: bytes, *, label: str) -> str:
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label}: exclusion register must be UTF-8 JSON.") from exc


def _mint_policy_source_receipt(
    policy: USByOriginReviewedExclusionPolicy,
    *,
    override_supplied: bool,
    expected_sha256: str | None,
    raw_source_sha256: str,
) -> USByOriginReviewedExclusionPolicy:
    receipt = _USByOriginPolicySourceReceipt(
        override_supplied=override_supplied,
        expected_sha256=expected_sha256,
        raw_source_sha256=raw_source_sha256,
    )
    object.__setattr__(policy, "_source_receipt", receipt)
    return policy


@functools.cache
def us_default_by_origin_reviewed_exclusion_policy() -> (
    USByOriginReviewedExclusionPolicy
):
    """Load and cache the committed policy of record."""

    raw_bytes = (
        files("microcosm.build.us")
        .joinpath(US_BY_ORIGIN_EXCLUSION_REGISTER_RESOURCE)
        .read_bytes()
    )
    return _mint_policy_source_receipt(
        _policy_from_raw(
            _decode_policy_bytes(
                raw_bytes,
                label=US_BY_ORIGIN_EXCLUSION_REGISTER_RESOURCE,
            ),
            label=US_BY_ORIGIN_EXCLUSION_REGISTER_RESOURCE,
            policy_source="committed",
        ),
        override_supplied=False,
        expected_sha256=None,
        raw_source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def load_us_by_origin_reviewed_exclusion_policy(
    source: str | Path | None = None,
    *,
    expected_sha256: str | None = None,
) -> USByOriginReviewedExclusionPolicy:
    """Resolve the committed register or an authenticated JSON override.

    An override path and its raw-file SHA-256 are one arming unit. Supplying
    only one is refused before any build work begins.
    """

    if (source is None) != (expected_sha256 is None):
        raise ValueError(
            "US by-origin exclusion override requires source and "
            "expected_sha256 together."
        )
    if source is None:
        return us_default_by_origin_reviewed_exclusion_policy()
    if not isinstance(expected_sha256, str) or not _SHA256_PATTERN.fullmatch(
        expected_sha256
    ):
        raise ValueError(
            "US by-origin exclusion expected_sha256 must be one lowercase "
            "SHA-256 digest."
        )
    path = Path(source)
    raw_bytes = path.read_bytes()
    observed_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if expected_sha256 != observed_sha256:
        raise ValueError(
            f"{path}: exclusion register SHA-256 mismatch; expected "
            f"{expected_sha256!r}, got {observed_sha256!r}."
        )
    resolved = _policy_from_raw(
        _decode_policy_bytes(raw_bytes, label=str(path)),
        label=str(path),
        policy_source="override",
    )
    committed = us_default_by_origin_reviewed_exclusion_policy()
    if resolved.policy_payload() == committed.policy_payload():
        resolved = USByOriginReviewedExclusionPolicy(
            description=resolved.description,
            exclusions=resolved.exclusions,
            policy_source="committed",
        )
    return _mint_policy_source_receipt(
        resolved,
        override_supplied=True,
        expected_sha256=expected_sha256,
        raw_source_sha256=observed_sha256,
    )


def validate_us_by_origin_reviewed_exclusion_source_provenance(
    policy: USByOriginReviewedExclusionPolicy,
    raw_provenance: object,
    *,
    boundary: str,
) -> None:
    """Validate path-free loader provenance against content and the default."""

    if not isinstance(policy, USByOriginReviewedExclusionPolicy):
        raise TypeError(
            f"{boundary}: policy must be a USByOriginReviewedExclusionPolicy."
        )
    expected_fields = {
        "override_supplied",
        "expected_sha256",
        "raw_source_sha256",
    }
    if (
        not isinstance(raw_provenance, Mapping)
        or set(raw_provenance) != expected_fields
    ):
        raise ValueError(f"{boundary}: policy source provenance schema mismatch.")
    try:
        receipt = _USByOriginPolicySourceReceipt(
            override_supplied=raw_provenance.get("override_supplied"),
            expected_sha256=raw_provenance.get("expected_sha256"),
            raw_source_sha256=raw_provenance.get("raw_source_sha256"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{boundary}: invalid policy source provenance: {exc}"
        ) from exc

    committed = us_default_by_origin_reviewed_exclusion_policy()
    same_as_committed = policy.policy_payload() == committed.policy_payload()
    if (policy.policy_source == "committed") != same_as_committed:
        raise ValueError(f"{boundary}: policy source does not match its content.")
    if not receipt.override_supplied:
        if (
            not same_as_committed
            or receipt.policy_payload() != committed.source_provenance
        ):
            raise ValueError(
                f"{boundary}: committed policy provenance does not match the "
                "packaged register bytes."
            )


def require_authenticated_us_by_origin_reviewed_exclusion_policy(
    policy: USByOriginReviewedExclusionPolicy,
    *,
    boundary: str,
) -> USByOriginReviewedExclusionPolicy:
    """Refuse policy objects that did not come through the byte loader."""

    if not isinstance(policy, USByOriginReviewedExclusionPolicy):
        raise TypeError(
            f"{boundary}: reviewed exclusion policy must be a fully armed "
            "USByOriginReviewedExclusionPolicy."
        )
    if policy.source_provenance is None:
        raise ValueError(
            f"{boundary}: reviewed exclusion policy lacks authenticated source "
            "provenance; resolve it through the path-plus-SHA loader."
        )
    validate_us_by_origin_reviewed_exclusion_source_provenance(
        policy,
        policy.source_provenance,
        boundary=boundary,
    )
    return policy


def us_exclusion_evaluation_date(now: date | None) -> date:
    """Resolve one plain-date clock for every exclusion lifecycle check."""

    if now is None:
        return datetime.now(UTC).date()
    if isinstance(now, datetime) or not isinstance(now, date):
        raise TypeError(
            "US exclusion evaluation date must be a datetime.date, got "
            f"{type(now).__name__}."
        )
    return now
