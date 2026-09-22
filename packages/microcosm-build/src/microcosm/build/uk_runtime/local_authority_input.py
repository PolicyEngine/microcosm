"""The engine's ``local_authority`` input, resolved from the ladder's GSS codes.

policyengine-uk reads a household enum input ``local_authority``
(``LocalAuthority``: one member per UK local authority district, default
``MAIDSTONE``). The rowwise geography ladder writes every household's April
2023 ONS code as ``local_authority_code``; this module resolves that code to
the enum member name the engine loader expects, through the committed ONS
names-and-codes resource ``local_authority_names.json`` (published data;
``geography_sources.LAD23_NAMES_SHA256`` pins the lookup bytes, and the
generator, the loader and this resource all assert it) and the mechanical
key rule below (code, with one declared alias).

Fail closed, never default: a code the resource does not carry raises at the
write point; :func:`uk_geography_ladder_gate` reports any row whose
``local_authority`` disagrees with its code; and the rowwise clone verifies
the assigned member names against the installed engine
(:func:`verify_local_authority_engine_domain`), with the whole roster asserted
by ``test_uk_local_authority_input.py``. No battery ``enum_domain`` gate is
declared for this column: that family needs a ``rules_engine`` evidence
artifact the rowwise local battery never supplies, and
``uk_local_geography_ladder_post_calibration`` already reruns the ladder gate,
consistency check included, on the calibrated candidate (microcosm#953).
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Iterable, Mapping
from importlib import resources as importlib_resources
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

import microcosm.build.uk_runtime.geography_sources as geography_sources

UK_LOCAL_AUTHORITY_INPUT_COLUMN = "local_authority"
UK_LOCAL_AUTHORITY_CODE_COLUMN = "local_authority_code"
UK_LOCAL_AUTHORITY_NAMES_RESOURCE = "local_authority_names.json"
UK_LOCAL_AUTHORITY_NAMES_KIND = "uk_local_authority_names"
UK_LOCAL_AUTHORITY_NAMES_SCHEMA_VERSION = 1
UK_LOCAL_AUTHORITY_VINTAGE = "2023_april_lad"

#: ONS display names whose ``LocalAuthority`` member name is not the
#: mechanical rule's output. The engine drops the apostrophe where the rule
#: would keep a separator.
LOCAL_AUTHORITY_ENGINE_KEY_ALIASES: Mapping[str, str] = MappingProxyType(
    {"King's Lynn and West Norfolk": "KINGS_LYNN_AND_WEST_NORFOLK"}
)

_NON_ALPHANUMERIC = re.compile(r"[^A-Z0-9]+")
_GSS_CODE = re.compile(r"^[ENSW]\d{8}$")


def local_authority_engine_key(name: object) -> str:
    """Return the ``LocalAuthority`` member name for an ONS display name.

    Upper-case, collapse every non-alphanumeric run to one underscore and strip
    the ends (``Bristol, City of`` -> ``BRISTOL_CITY_OF``, ``Na h-Eileanan
    Siar`` -> ``NA_H_EILEANAN_SIAR``). ``LOCAL_AUTHORITY_ENGINE_KEY_ALIASES``
    overrides the rule for the members the engine spells differently.
    """

    text = str(name).strip()
    alias = LOCAL_AUTHORITY_ENGINE_KEY_ALIASES.get(text)
    if alias is not None:
        return alias
    key = _NON_ALPHANUMERIC.sub("_", text.upper()).strip("_")
    if not key:
        raise ValueError(f"Local authority name {name!r} yields an empty engine key.")
    return key


@functools.cache
def load_uk_local_authority_names_resource() -> Mapping[str, Any]:
    """Load and validate the committed ONS April 2023 names resource."""

    payload = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath(UK_LOCAL_AUTHORITY_NAMES_RESOURCE)
        .read_text(encoding="utf-8")
    )
    label = UK_LOCAL_AUTHORITY_NAMES_RESOURCE
    if payload.get("schema_version") != UK_LOCAL_AUTHORITY_NAMES_SCHEMA_VERSION:
        raise ValueError(f"{label}: unsupported schema_version.")
    if payload.get("kind") != UK_LOCAL_AUTHORITY_NAMES_KIND:
        raise ValueError(f"{label}: kind must be {UK_LOCAL_AUTHORITY_NAMES_KIND!r}.")
    if payload.get("country") != "uk":
        raise ValueError(f"{label}: country must be 'uk'.")
    source = payload.get("source")
    if not isinstance(source, Mapping):
        raise ValueError(f"{label}: source block is missing.")
    if source.get("sha256") != geography_sources.LAD23_NAMES_SHA256:
        raise ValueError(
            f"{label}: source sha256 {source.get('sha256')!r} is not the pinned "
            f"LAD23 names digest {geography_sources.LAD23_NAMES_SHA256}."
        )
    if source.get("url") != geography_sources.LAD23_NAMES_URL:
        raise ValueError(f"{label}: source url is not LAD23_NAMES_URL.")
    if source.get("vintage") != UK_LOCAL_AUTHORITY_VINTAGE:
        raise ValueError(
            f"{label}: source vintage must be {UK_LOCAL_AUTHORITY_VINTAGE!r}."
        )
    areas = payload.get("areas")
    if not isinstance(areas, Mapping) or not areas:
        raise ValueError(f"{label}: areas must be a non-empty mapping.")
    if payload.get("area_count") != len(areas):
        raise ValueError(f"{label}: area_count disagrees with the areas mapping.")
    keys: dict[str, str] = {}
    for code, entry in areas.items():
        if not _GSS_CODE.match(str(code)):
            raise ValueError(f"{label}: {code!r} is not a GSS local authority code.")
        name = str(entry.get("name", "")).strip() if isinstance(entry, Mapping) else ""
        key = (
            str(entry.get("engine_key", "")).strip()
            if isinstance(entry, Mapping)
            else ""
        )
        if not name or not key:
            raise ValueError(f"{label}: {code} must carry a name and an engine_key.")
        if key != local_authority_engine_key(name):
            raise ValueError(
                f"{label}: {code} engine_key {key!r} is not the key rule's output "
                f"for {name!r}."
            )
        if key in keys:
            raise ValueError(
                f"{label}: engine_key {key!r} is shared by {keys[key]} and {code}."
            )
        keys[key] = str(code)
    return MappingProxyType(payload)


@functools.cache
def local_authority_engine_key_by_code() -> Mapping[str, str]:
    """The resource's derived view: GSS code -> ``LocalAuthority`` member name."""

    areas = load_uk_local_authority_names_resource()["areas"]
    return MappingProxyType(
        {str(code): str(entry["engine_key"]) for code, entry in sorted(areas.items())}
    )


def resolve_local_authority_engine_keys(codes: Iterable[object]) -> np.ndarray:
    """Map GSS local authority codes to engine member names, failing closed.

    Every code must sit on the April 2023 roster the resource carries; an
    unknown or blank code raises naming it. The engine input is never
    defaulted (microcosm#953).
    """

    values = pd.Series(
        np.asarray(pd.Series(codes, dtype=object).to_numpy(), dtype=object)
    )
    text = values.fillna("").astype(str).str.strip()
    mapped = text.map(local_authority_engine_key_by_code())
    unknown = mapped.isna().to_numpy()
    if unknown.any():
        bad = sorted(set(text.to_numpy()[unknown].tolist()))
        raise ValueError(
            "local_authority_code value(s) are not on the April 2023 local "
            f"authority roster ({len(bad)} distinct): {bad[:10]}; the engine "
            "input local_authority is never defaulted (microcosm#953)."
        )
    return mapped.to_numpy(dtype=object)


def local_authority_consistency_failures(
    household: pd.DataFrame,
    *,
    code_column: str = UK_LOCAL_AUTHORITY_CODE_COLUMN,
    input_column: str = UK_LOCAL_AUTHORITY_INPUT_COLUMN,
) -> list[str]:
    """Report rows whose engine input disagrees with their code (gate use).

    Returns failure messages and never raises, so the ladder gate can carry
    the verdict through the local battery.
    """

    missing = [c for c in (code_column, input_column) if c not in household.columns]
    if missing:
        return [f"household table is missing local authority column(s): {missing}"]
    total = len(household)
    codes = (
        household[code_column].fillna("").astype(str).str.strip().to_numpy(dtype=object)
    )
    keys = (
        household[input_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .to_numpy(dtype=object)
    )
    table = local_authority_engine_key_by_code()
    failures: list[str] = []
    blank = np.array([key == "" for key in keys], dtype=bool)
    if blank.any():
        failures.append(f"{input_column}: {int(blank.sum())}/{total} row(s) are blank")
    expected = [table.get(code) for code in codes]
    off_roster = np.array([value is None for value in expected], dtype=bool)
    if off_roster.any():
        examples = sorted(set(codes[off_roster].tolist()))[:5]
        failures.append(
            f"{code_column}: {int(off_roster.sum())}/{total} row(s) are not on "
            f"the April 2023 local authority roster; examples {examples}"
        )
    mismatched = np.array(
        [
            not off and not is_blank and want != key
            for off, is_blank, want, key in zip(
                off_roster, blank, expected, keys, strict=True
            )
        ],
        dtype=bool,
    )
    if mismatched.any():
        examples = sorted(
            {
                f"{code}->{key}"
                for code, key in zip(codes[mismatched], keys[mismatched], strict=True)
            }
        )[:5]
        failures.append(
            f"{input_column}: {int(mismatched.sum())}/{total} row(s) disagree with "
            f"{code_column}; examples {examples}"
        )
    return failures


def _engine_local_authority_members() -> tuple[str, frozenset[str]]:
    from importlib import metadata

    from policyengine_uk.variables.household.demographic.locations import (
        LocalAuthority,
    )

    try:
        version = metadata.version("policyengine-uk")
    except metadata.PackageNotFoundError:  # pragma: no cover - source checkout
        version = "unknown"
    return version, frozenset(LocalAuthority.__members__)


def local_authority_keys_missing_from_engine(
    keys: Iterable[object] | None = None,
) -> tuple[str, ...]:
    """Member names (the whole roster's by default) absent from the installed engine."""

    _version, members = _engine_local_authority_members()
    wanted = (
        set(local_authority_engine_key_by_code().values())
        if keys is None
        else {str(key) for key in keys}
    )
    return tuple(sorted(wanted - members))


def verify_local_authority_engine_domain(keys: Iterable[object]) -> None:
    """Refuse any member name that the installed engine's enum does not carry."""

    wanted = [str(key) for key in keys]
    missing = local_authority_keys_missing_from_engine(wanted)
    if missing:
        version, _members = _engine_local_authority_members()
        raise ValueError(
            f"{len(missing)} local_authority member name(s) are not in the "
            f"installed policyengine-uk {version} LocalAuthority enum: "
            f"{list(missing[:10])}; the engine pin must carry every roster "
            "authority before the input can ship (microcosm#953)."
        )


__all__ = [
    "LOCAL_AUTHORITY_ENGINE_KEY_ALIASES",
    "UK_LOCAL_AUTHORITY_CODE_COLUMN",
    "UK_LOCAL_AUTHORITY_INPUT_COLUMN",
    "UK_LOCAL_AUTHORITY_NAMES_KIND",
    "UK_LOCAL_AUTHORITY_NAMES_RESOURCE",
    "UK_LOCAL_AUTHORITY_NAMES_SCHEMA_VERSION",
    "UK_LOCAL_AUTHORITY_VINTAGE",
    "load_uk_local_authority_names_resource",
    "local_authority_consistency_failures",
    "local_authority_engine_key",
    "local_authority_engine_key_by_code",
    "local_authority_keys_missing_from_engine",
    "resolve_local_authority_engine_keys",
    "verify_local_authority_engine_domain",
]
