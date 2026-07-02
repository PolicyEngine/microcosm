"""Congressional-district geography-vintage translation for Ledger facts."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SOURCE_CONGRESSIONAL_DISTRICT_PREFIX = "5001700US"
CURRENT_CONGRESSIONAL_DISTRICT_PREFIX = "5001900US"
CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE = "119th_congress"
#: The packaged default crosswalk, built from Census sources by
#: ``tools/build_us_congressional_district_vintage_crosswalk.py`` and shipped
#: under ``populace.build.us_runtime.data``. See the sibling ``.provenance.json``
#: and ``CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK.md`` for exact source files and
#: per-state population conservation.
DEFAULT_CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_RESOURCE = (
    "congressional_district_vintage_crosswalk.csv"
)
CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR = (
    "populace_congressional_district_vintage_crosswalk_sha256"
)
CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR = (
    "populace_congressional_district_vintage_target"
)


def load_congressional_district_vintage_crosswalk(path: str | Path) -> pd.DataFrame:
    """Load a CD-vintage crosswalk artifact from CSV, JSON, or JSONL."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"CD vintage crosswalk artifact not found: {source}")
    if source.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with source.open() as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid CD vintage crosswalk JSONL row {line_number}: "
                        f"{exc.msg}"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise ValueError(
                        f"Invalid CD vintage crosswalk JSONL row {line_number}: "
                        f"expected object, got {type(row).__name__}."
                    )
                rows.append(dict(row))
        return _prepare_crosswalk(
            rows,
            source_prefix=SOURCE_CONGRESSIONAL_DISTRICT_PREFIX,
            target_prefix=CURRENT_CONGRESSIONAL_DISTRICT_PREFIX,
        )
    if source.suffix == ".json":
        payload = json.loads(source.read_text())
        if not isinstance(payload, list):
            raise ValueError("CD vintage crosswalk JSON artifact must be an array.")
        return _prepare_crosswalk(
            payload,
            source_prefix=SOURCE_CONGRESSIONAL_DISTRICT_PREFIX,
            target_prefix=CURRENT_CONGRESSIONAL_DISTRICT_PREFIX,
        )
    return _prepare_crosswalk(
        pd.read_csv(source),
        source_prefix=SOURCE_CONGRESSIONAL_DISTRICT_PREFIX,
        target_prefix=CURRENT_CONGRESSIONAL_DISTRICT_PREFIX,
    )


def default_congressional_district_vintage_crosswalk_path() -> Path:
    """Return the path to the packaged default CD-vintage crosswalk CSV."""

    return Path(
        str(
            files("populace.build.us_runtime.data").joinpath(
                DEFAULT_CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_RESOURCE
            )
        )
    )


def load_default_congressional_district_vintage_crosswalk() -> pd.DataFrame:
    """Load the packaged Census-built default 117th->119th CD crosswalk.

    This is the canonical, versioned crosswalk built by
    ``tools/build_us_congressional_district_vintage_crosswalk.py`` from Census
    Block Assignment Files, the 119th BEF, and 2020 P.L. 94-171 block
    populations. Builds may still pass an explicit crosswalk path to override it.
    """

    return load_congressional_district_vintage_crosswalk(
        default_congressional_district_vintage_crosswalk_path()
    )


def translate_congressional_district_facts_to_current_vintage(
    facts: Iterable[Mapping[str, Any]],
    crosswalk: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    source_prefix: str = SOURCE_CONGRESSIONAL_DISTRICT_PREFIX,
    target_prefix: str = CURRENT_CONGRESSIONAL_DISTRICT_PREFIX,
    target_vintage: str = CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
    crosswalk_basis: str = "population",
) -> tuple[dict[str, Any], ...]:
    """Translate old-vintage CD Ledger facts onto current CD geography.

    The crosswalk is a proportional source-to-target table. It may split one
    source district across multiple current districts, and it may merge several
    source districts into one current district. Derived pieces are therefore
    aggregated before returning so downstream target compilation sees one fact
    per current geography/measure slice.
    """

    prepared_crosswalk = _prepare_crosswalk(
        crosswalk,
        source_prefix=source_prefix,
        target_prefix=target_prefix,
    )
    materialized_facts = tuple(facts)
    materialized_facts = _with_state_total_proxy_source_districts(
        materialized_facts,
        prepared_crosswalk=prepared_crosswalk,
        source_prefix=source_prefix,
    )
    rows_by_source = {
        source_id: source_rows
        for source_id, source_rows in prepared_crosswalk.groupby(
            "source_geography_id", sort=False
        )
    }
    passthrough: list[dict[str, Any]] = []
    translated: dict[tuple[str, ...], dict[str, Any]] = {}
    contribution_order: list[tuple[str, ...]] = []
    for fact in materialized_facts:
        geography = _mapping_at(fact, "geography")
        geography_id = str(geography.get("id") or "")
        geography_level = str(geography.get("level") or "")
        if geography_level != "congressional_district":
            passthrough.append(copy.deepcopy(dict(fact)))
            continue
        if geography_id.startswith(target_prefix):
            passthrough.append(copy.deepcopy(dict(fact)))
            continue
        if not geography_id.startswith(source_prefix):
            raise ValueError(
                "Unsupported congressional-district geography id "
                f"{geography_id!r}; expected {source_prefix!r} or {target_prefix!r}."
            )
        source_rows = rows_by_source.get(geography_id)
        if source_rows is None:
            raise ValueError(
                f"CD vintage crosswalk is missing source geography {geography_id!r}."
            )
        source_value = _required_numeric_value(fact)
        for row in source_rows.to_dict("records"):
            target_geography_id = str(row["target_geography_id"])
            share = float(row["share"])
            key = _translated_fact_key(
                fact,
                target_geography_id=target_geography_id,
            )
            target_fact = translated.get(key)
            if target_fact is None:
                target_fact = _translated_fact_shell(
                    fact,
                    source_geography_id=geography_id,
                    target_geography_id=target_geography_id,
                    target_vintage=target_vintage,
                    crosswalk_basis=crosswalk_basis,
                    key=key,
                )
                translated[key] = target_fact
                contribution_order.append(key)
            target_fact["value"] = _numeric_value(target_fact) + source_value * share
            _append_lineage_contribution(
                target_fact,
                source_fact=fact,
                source_geography_id=geography_id,
                target_geography_id=target_geography_id,
                share=share,
                crosswalk_basis=crosswalk_basis,
            )
    return tuple([*passthrough, *(translated[key] for key in contribution_order)])


def _with_state_total_proxy_source_districts(
    facts: tuple[Mapping[str, Any], ...],
    *,
    prepared_crosswalk: pd.DataFrame,
    source_prefix: str,
) -> tuple[Mapping[str, Any], ...]:
    """Add source-vintage CD proxy facts for states absent from source CD rows."""

    source_ids_by_state: dict[str, set[str]] = {}
    for source_id in prepared_crosswalk["source_geography_id"].drop_duplicates():
        source_id = str(source_id)
        source_ids_by_state.setdefault(
            _source_state_fips(source_id, source_prefix=source_prefix), set()
        ).add(source_id)

    source_cd_states_with_facts = {
        _source_state_fips(
            str(_mapping_at(fact, "geography").get("id") or ""),
            source_prefix=source_prefix,
        )
        for fact in facts
        if _is_source_soi_congressional_district_fact(
            fact,
            source_prefix=source_prefix,
        )
    }
    source_cd_shapes = {
        _proxy_shape_key(fact)
        for fact in facts
        if _is_source_soi_congressional_district_fact(
            fact,
            source_prefix=source_prefix,
        )
    }
    proxy_source_by_state = {
        state_fips: next(iter(source_ids))
        for state_fips, source_ids in source_ids_by_state.items()
        if len(source_ids) == 1 and state_fips not in source_cd_states_with_facts
    }
    if not proxy_source_by_state:
        return facts

    proxy_facts = [
        _state_total_proxy_source_district_fact(
            fact,
            source_geography_id=source_geography_id,
            source_prefix=source_prefix,
        )
        for fact in facts
        for state_fips in [_state_fips_from_fact(fact)]
        for source_geography_id in [proxy_source_by_state.get(state_fips or "")]
        if source_geography_id is not None
        and _is_state_proxy_candidate_fact(fact)
        and source_cd_shapes
        and _proxy_shape_key(fact) in source_cd_shapes
    ]
    if not proxy_facts:
        return facts
    return (*facts, *proxy_facts)


def _state_total_proxy_source_district_fact(
    fact: Mapping[str, Any],
    *,
    source_geography_id: str,
    source_prefix: str,
) -> dict[str, Any]:
    proxy = copy.deepcopy(dict(fact))
    source_geoid = source_geography_id.removeprefix(source_prefix)
    state_geography_id = str(_mapping_at(fact, "geography").get("id") or "")

    geography = dict(_mapping_at(proxy, "geography"))
    geography["level"] = "congressional_district"
    geography["id"] = source_geography_id
    geography["vintage"] = "state_total_proxy_source_congressional_district"
    proxy["geography"] = geography

    layout = dict(_mapping_at(proxy, "layout"))
    layout["groupby_dimension"] = "irs_soi.congressional_district"
    layout["groupby_value_id"] = source_geoid
    layout["source_row_id"] = source_geoid
    proxy["layout"] = layout

    original_source_record_id = _source_record_id(fact)
    digest_payload = {
        "source_record_id": original_source_record_id,
        "source_geography_id": source_geography_id,
        "value": fact.get("value"),
    }
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    source_record_id = (
        f"{original_source_record_id}.state_total_proxy_cd."
        f"{source_geography_id}.{digest}"
    )
    proxy["source_record_id"] = source_record_id
    proxy["aggregate_fact_key"] = (
        f"populace.derived_fact.congressional_district_state_total_proxy.v1:{digest}"
    )
    proxy["semantic_fact_key"] = (
        f"populace.semantic_fact.congressional_district_state_total_proxy.v1:{digest}"
    )
    proxy.pop("legacy_fact_key", None)

    lineage = dict(_mapping_at(proxy, "lineage"))
    lineage["source_record_id"] = source_record_id
    lineage["congressional_district_state_total_proxy_source_record_id"] = (
        original_source_record_id
    )
    lineage["congressional_district_state_total_proxy_source_geography_id"] = (
        state_geography_id
    )
    lineage["congressional_district_state_total_proxy_cd_geography_id"] = (
        source_geography_id
    )
    proxy["lineage"] = lineage
    return proxy


def _is_source_congressional_district_fact(
    fact: Mapping[str, Any],
    *,
    source_prefix: str,
) -> bool:
    geography = _mapping_at(fact, "geography")
    return str(geography.get("level") or "") == "congressional_district" and str(
        geography.get("id") or ""
    ).startswith(source_prefix)


def _is_source_soi_congressional_district_fact(
    fact: Mapping[str, Any],
    *,
    source_prefix: str,
) -> bool:
    return _source_name(fact) == "irs_soi" and _is_source_congressional_district_fact(
        fact,
        source_prefix=source_prefix,
    )


def _is_state_proxy_candidate_fact(fact: Mapping[str, Any]) -> bool:
    geography = _mapping_at(fact, "geography")
    return (
        str(geography.get("level") or "") == "state"
        and _source_name(fact) == "irs_soi"
        and _state_fips_from_fact(fact) is not None
    )


def _state_fips_from_fact(fact: Mapping[str, Any]) -> str | None:
    geography_id = str(_mapping_at(fact, "geography").get("id") or "")
    if not geography_id.startswith("0400000US"):
        return None
    fips = geography_id.removeprefix("0400000US")
    if len(fips) != 2 or not fips.isdigit():
        return None
    return fips


def _source_state_fips(source_geography_id: str, *, source_prefix: str) -> str:
    source_geoid = source_geography_id.removeprefix(source_prefix)
    return source_geoid[:2]


def _proxy_shape_key(fact: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        _source_name(fact),
        _measure_id(fact),
        _str_at(fact, "entity", "name"),
        _str_at(fact, "aggregation", "method"),
        _soi_return_universe(fact),
        json.dumps(_mapping_at(fact, "dimensions"), sort_keys=True),
        json.dumps(_mapping_at(fact, "universe_constraints"), sort_keys=True),
    )


def _soi_return_universe(fact: Mapping[str, Any]) -> str:
    record_set_id = _str_at(fact, "layout", "record_set_id")
    normalized = ".".join(
        part for part in record_set_id.split(".") if not _is_period_token(part)
    )
    if "itemized_all_returns" in normalized:
        return "itemized_returns"
    if "all_returns_excluding_dependents" in normalized:
        return "returns_excluding_dependents"
    return "all_returns"


def _is_period_token(value: str) -> bool:
    normalized = value.lower().replace("-", "_")
    parts = normalized.split("_", maxsplit=1)
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return len(parts[0]) == 4 and len(parts[1]) in {1, 2}
    if normalized[:2] in {"ty", "cy", "fy"}:
        normalized = normalized[2:]
    return normalized.isdigit() and len(normalized) == 4


def _prepare_crosswalk(
    crosswalk: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    source_prefix: str,
    target_prefix: str,
) -> pd.DataFrame:
    frame = (
        crosswalk.copy()
        if isinstance(crosswalk, pd.DataFrame)
        else pd.DataFrame(crosswalk)
    )
    required = {"source_geography_id", "target_geography_id", "weight"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CD vintage crosswalk is missing column(s): {missing}.")
    prepared = frame.loc[
        :, ["source_geography_id", "target_geography_id", "weight"]
    ].copy()
    prepared["source_geography_id"] = prepared["source_geography_id"].map(str)
    prepared["target_geography_id"] = prepared["target_geography_id"].map(str)
    bad_source = prepared[
        ~prepared["source_geography_id"].str.startswith(source_prefix, na=False)
    ]
    if len(bad_source):
        raise ValueError(
            "CD vintage crosswalk source_geography_id must use prefix "
            f"{source_prefix!r}; invalid {bad_source['source_geography_id'].head(5).tolist()}."
        )
    bad_target = prepared[
        ~prepared["target_geography_id"].str.startswith(target_prefix, na=False)
    ]
    if len(bad_target):
        raise ValueError(
            "CD vintage crosswalk target_geography_id must use prefix "
            f"{target_prefix!r}; invalid {bad_target['target_geography_id'].head(5).tolist()}."
        )
    source_geoids = prepared["source_geography_id"].map(
        lambda value: value.removeprefix(source_prefix)
    )
    target_geoids = prepared["target_geography_id"].map(
        lambda value: value.removeprefix(target_prefix)
    )
    invalid_geoids = prepared[
        ~source_geoids.str.fullmatch(r"\d{4}") | ~target_geoids.str.fullmatch(r"\d{4}")
    ]
    if len(invalid_geoids):
        raise ValueError(
            "CD vintage crosswalk geography ids must end in four-digit state+district "
            f"geoids; invalid {invalid_geoids.head(5).to_dict('records')}."
        )
    cross_state = prepared[source_geoids.str[:2] != target_geoids.str[:2]]
    if len(cross_state):
        raise ValueError(
            "CD vintage crosswalk rows must stay within the source state; invalid "
            f"{cross_state.head(5).to_dict('records')}."
        )
    weights = pd.to_numeric(prepared["weight"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise ValueError("CD vintage crosswalk weight must be positive and finite.")
    prepared["weight"] = weights
    prepared = (
        prepared.groupby(
            ["source_geography_id", "target_geography_id"], as_index=False
        )["weight"]
        .sum()
        .sort_values(["source_geography_id", "target_geography_id"])
        .reset_index(drop=True)
    )
    totals = prepared.groupby("source_geography_id")["weight"].transform("sum")
    if (totals <= 0.0).any():
        raise ValueError("CD vintage crosswalk source weights must sum positive.")
    prepared["share"] = prepared["weight"] / totals
    return prepared


def _translated_fact_shell(
    fact: Mapping[str, Any],
    *,
    source_geography_id: str,
    target_geography_id: str,
    target_vintage: str,
    crosswalk_basis: str,
    key: tuple[str, ...],
) -> dict[str, Any]:
    translated = copy.deepcopy(dict(fact))
    target_geoid = target_geography_id.removeprefix(
        CURRENT_CONGRESSIONAL_DISTRICT_PREFIX
    )
    geography = dict(_mapping_at(translated, "geography"))
    geography["id"] = target_geography_id
    geography["level"] = "congressional_district"
    geography["vintage"] = target_vintage
    translated["geography"] = geography

    layout = dict(_mapping_at(translated, "layout"))
    layout["groupby_value_id"] = target_geoid
    layout["source_row_id"] = target_geoid
    translated["layout"] = layout

    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    source_record_id = _derived_source_record_id(
        translated,
        target_geography_id=target_geography_id,
        digest=digest,
    )
    translated["source_record_id"] = source_record_id
    translated["aggregate_fact_key"] = (
        f"populace.derived_fact.congressional_district_vintage.v1:{digest}"
    )
    translated["semantic_fact_key"] = (
        f"populace.semantic_fact.congressional_district_vintage.v1:{digest}"
    )
    translated.pop("legacy_fact_key", None)
    lineage = dict(_mapping_at(translated, "lineage"))
    lineage["source_record_id"] = source_record_id
    lineage["source_geography_vintage"] = str(
        _mapping_at(fact, "geography").get("vintage") or ""
    )
    lineage["target_geography_vintage"] = target_vintage
    lineage["congressional_district_vintage_crosswalk_basis"] = crosswalk_basis
    lineage["congressional_district_vintage_source_geography_id"] = source_geography_id
    lineage["congressional_district_vintage_target_geography_id"] = target_geography_id
    lineage["congressional_district_vintage_contributions"] = []
    translated["lineage"] = lineage
    translated["value"] = 0.0
    return translated


def _append_lineage_contribution(
    target_fact: dict[str, Any],
    *,
    source_fact: Mapping[str, Any],
    source_geography_id: str,
    target_geography_id: str,
    share: float,
    crosswalk_basis: str,
) -> None:
    lineage = dict(_mapping_at(target_fact, "lineage"))
    contributions = list(
        lineage.get("congressional_district_vintage_contributions") or []
    )
    contributions.append(
        {
            "source_record_id": _source_record_id(source_fact),
            "source_geography_id": source_geography_id,
            "target_geography_id": target_geography_id,
            "share": share,
            "basis": crosswalk_basis,
        }
    )
    lineage["congressional_district_vintage_contributions"] = contributions
    lineage["source_record_ids"] = sorted(
        {
            str(contribution["source_record_id"])
            for contribution in contributions
            if contribution.get("source_record_id")
        }
    )
    target_fact["lineage"] = lineage


def _translated_fact_key(
    fact: Mapping[str, Any],
    *,
    target_geography_id: str,
) -> tuple[str, ...]:
    return (
        _source_name(fact),
        _measure_id(fact),
        str(_mapping_at(fact, "period").get("value") or ""),
        target_geography_id,
        _str_at(fact, "entity", "name"),
        _str_at(fact, "aggregation", "method"),
        _str_at(fact, "layout", "record_set_id"),
        _str_at(fact, "layout", "groupby_dimension"),
        json.dumps(_mapping_at(fact, "dimensions"), sort_keys=True),
        json.dumps(_mapping_at(fact, "universe_constraints"), sort_keys=True),
    )


def _derived_source_record_id(
    fact: Mapping[str, Any],
    *,
    target_geography_id: str,
    digest: str,
) -> str:
    record_set_id = _str_at(fact, "layout", "record_set_id") or "ledger"
    measure_id = _measure_id(fact) or "measure"
    target_geoid = target_geography_id.removeprefix(
        CURRENT_CONGRESSIONAL_DISTRICT_PREFIX
    )
    return f"{record_set_id}.current_cd.{target_geoid}.{measure_id}.{digest}"


def _mapping_at(fact: Mapping[str, Any], *path: str) -> Mapping[str, Any]:
    current: Any = fact
    for key in path:
        if isinstance(current, Mapping):
            current = current.get(key)
        else:
            return {}
    return current if isinstance(current, Mapping) else {}


def _str_at(fact: Mapping[str, Any], *path: str) -> str:
    current: Any = fact
    for key in path:
        if isinstance(current, Mapping):
            current = current.get(key)
        else:
            return ""
    return "" if current is None else str(current)


def _numeric_value(fact: Mapping[str, Any]) -> float:
    try:
        return float(fact.get("value") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _required_numeric_value(fact: Mapping[str, Any]) -> float:
    value = fact.get("value")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"CD Ledger fact {_source_record_id(fact)!r} has invalid value {value!r}."
        ) from exc
    if not np.isfinite(numeric):
        raise ValueError(
            f"CD Ledger fact {_source_record_id(fact)!r} has invalid value {value!r}."
        )
    return numeric


def _source_record_id(fact: Mapping[str, Any]) -> str:
    return _str_at(fact, "source_record_id") or _str_at(
        fact, "lineage", "source_record_id"
    )


def _source_name(fact: Mapping[str, Any]) -> str:
    return _str_at(fact, "source", "source_name") or _str_at(
        fact, "observed_measure", "source_name"
    )


def _measure_id(fact: Mapping[str, Any]) -> str:
    return _str_at(fact, "observed_measure", "source_measure_id") or _str_at(
        fact, "layout", "measure_id"
    )
