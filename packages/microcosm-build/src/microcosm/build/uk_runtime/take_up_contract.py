"""UK stochastic take-up contract loader."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any

__all__ = [
    "UK_ALLOWED_RATE_STATUSES",
    "UKRateEntry",
    "UKTakeUpContract",
    "load_uk_take_up_contract",
    "rate_at",
    "uk_take_up_contract_identity",
]

UK_ALLOWED_RATE_STATUSES = frozenset(
    {
        "sourced_administrative",
        "sourced_industry",
        "sourced_administrative_derived",
        "fitted_offline",
        "frozen_by_adjudication",
    }
)
UK_ALLOWED_RATE_ENTITIES = frozenset({"person", "benunit", "household"})


@dataclass(frozen=True)
class UKRateEntry:
    """One scalar rate or continuous distribution entry."""

    key: str
    values: Mapping[str, float]
    source: Mapping[str, Any]
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class UKTakeUpContract:
    """Checked-in UK stochastic take-up contract."""

    version: int
    country: str
    build_year: int
    resource_sha256: str
    policy: str
    programs: tuple[UKRateEntry, ...]
    stochastic: tuple[UKRateEntry, ...]
    continuous: tuple[Mapping[str, Any], ...]

    def entry(self, key: str) -> UKRateEntry:
        entries = {entry.key: entry for entry in (*self.programs, *self.stochastic)}
        try:
            return entries[key]
        except KeyError as exc:
            raise KeyError(f"unknown UK take-up contract key {key!r}") from exc

    def rate(self, key: str, build_year: int | None = None) -> float:
        return rate_at(
            self.entry(key), self.build_year if build_year is None else build_year
        )

    def continuous_entry(self, key: str) -> Mapping[str, Any]:
        for entry in self.continuous:
            if entry.get("key") == key:
                return entry
        raise KeyError(f"unknown UK continuous contract key {key!r}")


def _contract_path():
    return files("microcosm.build.uk").joinpath("take_up_contract.json")


def _canonical_sha256(resource: Mapping[str, object]) -> str:
    canonical = json.dumps(
        resource,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@lru_cache(maxsize=1)
def load_uk_take_up_contract() -> UKTakeUpContract:
    """Load and validate ``build/uk/take_up_contract.json``."""

    raw = json.loads(_contract_path().read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("UK take-up contract must contain a JSON object.")
    version = raw.get("version")
    country = raw.get("country")
    build_year = raw.get("build_year")
    if not isinstance(version, int) or version < 1:
        raise ValueError("UK take-up contract requires positive integer version.")
    if country != "uk":
        raise ValueError("UK take-up contract country must be 'uk'.")
    if not isinstance(build_year, int):
        raise ValueError("UK take-up contract requires integer build_year.")
    programs = _entries(raw.get("programs", ()), section="programs")
    stochastic = _entries(raw.get("stochastic", ()), section="stochastic")
    continuous = _continuous_entries(raw.get("continuous", ()))
    return UKTakeUpContract(
        version=version,
        country=country,
        build_year=build_year,
        resource_sha256=_canonical_sha256(raw),
        policy=str(raw.get("policy", "")),
        programs=programs,
        stochastic=stochastic,
        continuous=continuous,
    )


def rate_at(entry: UKRateEntry | Mapping[str, Any], build_year: int) -> float:
    """Return the latest date-keyed value with year <= ``build_year``."""

    values = entry.values if isinstance(entry, UKRateEntry) else entry.get("values", {})
    if not isinstance(values, Mapping) or not values:
        raise ValueError("rate entry requires non-empty date-keyed values.")
    selected: tuple[str, float] | None = None
    for date, value in sorted(values.items()):
        year = int(str(date).split("-", maxsplit=1)[0])
        if year <= int(build_year):
            selected = (str(date), float(value))
    if selected is None:
        raise ValueError(f"no rate value exists at or before build year {build_year}.")
    return selected[1]


def uk_take_up_contract_identity(
    contract: UKTakeUpContract | None = None,
) -> dict[str, object]:
    """Return the sidecar identity payload for the UK stochastic contract."""

    resolved = contract if contract is not None else load_uk_take_up_contract()
    return {
        "version": resolved.version,
        "country": resolved.country,
        "build_year": resolved.build_year,
        "resource_sha256": resolved.resource_sha256,
        "policy": resolved.policy,
        "programs": [deepcopy(dict(entry.raw)) for entry in resolved.programs],
        "stochastic": [deepcopy(dict(entry.raw)) for entry in resolved.stochastic],
        "continuous": [deepcopy(dict(entry)) for entry in resolved.continuous],
    }


def _entries(values: object, *, section: str) -> tuple[UKRateEntry, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"UK take-up contract {section} must be a list.")
    entries: list[UKRateEntry] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, Mapping):
            raise ValueError(f"UK take-up contract {section} entries must be objects.")
        key = item.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"UK take-up contract {section} entry requires key.")
        if key in seen:
            raise ValueError(f"duplicate UK take-up contract key {key!r}.")
        seen.add(key)
        entity = item.get("entity")
        if entity not in UK_ALLOWED_RATE_ENTITIES:
            raise ValueError(
                f"UK take-up contract {key!r} has invalid entity {entity!r}; "
                f"expected one of {sorted(UK_ALLOWED_RATE_ENTITIES)}."
            )
        entry_values = item.get("values")
        if not isinstance(entry_values, Mapping) or not entry_values:
            raise ValueError(f"UK take-up contract {key!r} requires values.")
        parsed_values = {
            str(date): float(value) for date, value in entry_values.items()
        }
        source = item.get("source")
        if not isinstance(source, Mapping):
            raise ValueError(f"UK take-up contract {key!r} requires source.")
        _validate_source(key, source)
        entries.append(
            UKRateEntry(
                key=key,
                values=parsed_values,
                source=dict(source),
                raw=dict(item),
            )
        )
    return tuple(entries)


def _continuous_entries(values: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("UK take-up contract continuous must be a list.")
    entries: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, Mapping):
            raise ValueError("UK take-up contract continuous entries must be objects.")
        key = item.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError("UK continuous contract entry requires key.")
        if key in seen:
            raise ValueError(f"duplicate UK continuous contract key {key!r}.")
        seen.add(key)
        source = item.get("source")
        if not isinstance(source, Mapping):
            raise ValueError(f"UK continuous contract {key!r} requires source.")
        _validate_source(key, source)
        if source.get("status") == "fitted_offline" and not item.get("fitting_receipt"):
            raise ValueError(
                f"UK continuous contract {key!r} requires fitting_receipt."
            )
        entries.append(dict(item))
    return tuple(entries)


def _validate_source(key: str, source: Mapping[str, Any]) -> None:
    status = source.get("status")
    if status not in UK_ALLOWED_RATE_STATUSES:
        raise ValueError(
            f"UK take-up contract {key!r} has unreviewed status {status!r}; "
            f"expected one of {sorted(UK_ALLOWED_RATE_STATUSES)}."
        )
    if not str(source.get("source") or ""):
        raise ValueError(f"UK take-up contract {key!r} requires source citation.")
    if status == "frozen_by_adjudication" and not source.get("freeze"):
        raise ValueError(
            f"UK take-up contract {key!r} requires freeze block for "
            "frozen_by_adjudication."
        )
