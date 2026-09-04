"""Evaluate a UK candidate on the incumbent's target surface (microcosm#762 I9).

The scorer compares both artifacts on *our* register. This module answers the
opposite question: how does the candidate do on the surface the incumbent
(uk-data at the pinned ref) calibrates to — every grain, every row, signed
items included — so the parts we deliberately do not bind are still measured
and the reader can see what we do not do and why.

Inputs are the two vendored, hash-pinned fixtures of the incumbent's surface
(``registry_parity_fixture_2025.json`` for the national rows,
``local_registry_parity_fixture_2025.json`` for the local rows), the parity
register (``uk_data_target_parity.json``) for the status of families we route
or fence, the local reference membership for per-cell statuses, and the
measure-exclusion register. The candidate's estimates come from the caller
(engine-resolved metrics and a constraint matrix over the calibrated weights);
this module owns the joins, the classification and the summaries.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from importlib import resources as importlib_resources
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "INCUMBENT_LOCAL_METRIC_ALIASES",
    "classify_local_rows",
    "classify_national_rows",
    "evaluation_summary",
    "load_incumbent_local_fixture",
    "load_incumbent_national_fixture",
    "load_uk_data_target_parity",
    "match_national_rows",
    "national_family_status",
    "render_markdown",
]

#: Incumbent local metric name -> our metric name where the concept is the
#: same VOA band-count column under a different name. Every other incumbent
#: metric either shares our name exactly or has no counterpart in our surface.
INCUMBENT_LOCAL_METRIC_ALIASES: Mapping[str, str] = {
    f"voa/council_tax/{band}": f"council_tax/band_{band.lower()}" for band in "ABCDEFGH"
}

_AREA_TYPE_TO_LEVEL = {
    "constituency": "constituency",
    "local_authority": "local_authority",
}
_AREA_TYPE_TO_METRIC_GRAIN = {"constituency": "constituency", "local_authority": "la"}


def _resource_text(name: str) -> str:
    return (
        importlib_resources.files("microcosm.build.uk")
        .joinpath(name)
        .read_text(encoding="utf-8")
    )


def load_incumbent_national_fixture() -> dict[str, Any]:
    return json.loads(_resource_text("registry_parity_fixture_2025.json"))


def load_incumbent_local_fixture() -> dict[str, Any]:
    return json.loads(_resource_text("local_registry_parity_fixture_2025.json"))


def load_uk_data_target_parity() -> dict[str, Any]:
    return json.loads(_resource_text("uk_data_target_parity.json"))


def national_family_status(parity: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    """Map an incumbent source token (``fixture row['source']``) to the parity
    register's ``(status, concern_id)`` through the concern's ``covers``."""

    import re

    by_source: dict[str, tuple[str, str]] = {}
    for concern in parity.get("concerns", ()):
        for covered in concern.get("covers", ()):
            match = re.search(r"targets\.sources\.([a-z_0-9]+)$", str(covered))
            if match is None:
                continue
            source = match.group(1)
            # A ported status wins over a routed/excluded one for the same
            # source token: the family is on our surface, the concern narrows it.
            current = by_source.get(source)
            if current is None or (
                current[0] not in ("ported_national",)
                and concern["status"] == "ported_national"
            ):
                by_source[source] = (str(concern["status"]), str(concern["concern_id"]))
    # The incumbent's source tokens are shorter than the module names.
    aliases = {
        "hmrc_spi": "hmrc_spi",
        "hmrc": "hmrc_cgt",
        "ons": "ons_demographics",
        "voa": "voa_council_tax",
        "obr": "obr",
        "dwp": "dwp",
        "slc": "slc",
        "nts": "nts_vehicles",
        "ehs": "housing",
        "scottish_government": "scottish_government",
        "scotland_census": "ons_households",
    }
    return {
        token: by_source[module]
        for token, module in aliases.items()
        if module in by_source
    }


def _normalise(name: str) -> str:
    return name.replace("/", ".").lower()


def _with_nullable_objects(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Keep ``None`` (not ``NaN``) in optional columns so callers can test
    identity and JSON round-trips stay ``null``."""

    for column in columns:
        if column in frame.columns:
            values = frame[column].astype(object)
            frame[column] = values.where(values.notna(), None)
    return frame


def match_national_rows(
    fixture_rows: Iterable[Mapping[str, Any]],
    our_specs: Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    """Join incumbent national rows to our compiled specs.

    ``our_specs`` rows carry ``name`` (the compiled spec name, no period),
    ``contract_target_id`` and ``value`` (our target). Matching order: exact
    name, then name with ``/`` read as ``.``, then a contract id that names a
    single cell on both sides. Everything else is ``unmatched`` and keeps the
    incumbent's value so it is still listed.
    """

    ours = list(our_specs)
    by_name = {str(s["name"]): s for s in ours}
    by_norm = {_normalise(str(s["name"])): s for s in ours}
    by_cid: dict[str, list[Mapping[str, Any]]] = {}
    for s in ours:
        cid = s.get("contract_target_id")
        if cid:
            by_cid.setdefault(str(cid), []).append(s)
    inc_cid_counts: dict[str, int] = {}
    rows = list(fixture_rows)
    for r in rows:
        cid = r.get("contract_target_id")
        if cid:
            inc_cid_counts[str(cid)] = inc_cid_counts.get(str(cid), 0) + 1
    out = []
    for r in rows:
        name = str(r["name"])
        cid = r.get("contract_target_id")
        match, how = None, "unmatched"
        if name in by_name:
            match, how = by_name[name], "name"
        elif _normalise(name) in by_norm:
            match, how = by_norm[_normalise(name)], "normalised_name"
        elif (
            cid
            and inc_cid_counts.get(str(cid)) == 1
            and len(by_cid.get(str(cid), [])) == 1
        ):
            match, how = by_cid[str(cid)][0], "contract_id"
        out.append(
            {
                "incumbent_name": name,
                "contract_target_id": cid,
                "source": r.get("source"),
                "family": r.get("family"),
                "incumbent_target": float(r["value"]),
                "resolved_from_year": r.get("resolved_from_year"),
                "our_name": None if match is None else str(match["name"]),
                "our_target": None if match is None else float(match["value"]),
                "match": how,
            }
        )
    return _with_nullable_objects(
        pd.DataFrame(out), ("our_name", "our_target", "contract_target_id")
    )


def classify_national_rows(
    matched: pd.DataFrame,
    *,
    bound_names: set[str],
    exclusions: Mapping[str, Mapping[str, Any]],
    source_status: Mapping[str, tuple[str, str]],
) -> pd.DataFrame:
    """Attach our status to every incumbent national row."""

    statuses, reasons = [], []
    for row in matched.itertuples(index=False):
        if not isinstance(row.our_name, str):
            status, concern = source_status.get(str(row.source), ("not_ported", ""))
            statuses.append(
                f"not_ported:{status}" if status != "not_ported" else "not_ported"
            )
            reasons.append(concern)
        elif row.our_name in exclusions:
            rec = exclusions[row.our_name]
            statuses.append("measure_excluded")
            reasons.append(
                f"{rec.get('tracking', '')} expires {rec.get('expires_on', '')}"
            )
        elif row.our_name in bound_names:
            statuses.append("bound")
            reasons.append("")
        else:
            statuses.append("compiled_not_bound")
            reasons.append("")
    frame = matched.copy()
    frame["status"] = statuses
    frame["status_detail"] = reasons
    return frame


def classify_local_rows(
    fixture_rows: Iterable[Mapping[str, Any]],
    *,
    metric_target_ids: Mapping[str, str],
    membership: Mapping[str, Any],
    our_metric_names: Mapping[str, Iterable[str]],
    bound_names: set[str],
    unmapped_concern: Mapping[str, tuple[str, str]],
) -> pd.DataFrame:
    """Map incumbent local rows to our metrics and attach our per-cell status.

    ``metric_target_ids`` maps our metric name to the contract target id;
    ``membership`` is ``local_target_reference_membership.json``;
    ``our_metric_names`` maps grain (``constituency`` / ``la``) to the declared
    metric names; ``unmapped_concern`` maps an incumbent metric with no
    counterpart to its parity ``(status, concern_id)``.
    """

    candidates: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    targets = membership.get("targets", {})
    for target_id, payload in targets.items():
        for level, level_payload in (payload.get("geography_levels") or {}).items():
            for cand in level_payload.get("candidates") or ():
                candidates[
                    (str(target_id), str(level), str(cand.get("geography_id")))
                ] = cand
    deferral_reason: dict[tuple[str, str, str], str] = {}
    for d in membership.get("signed_deferrals", ()):
        for area in d.get("area_ids", ()):
            deferral_reason[
                (str(d["target_id"]), str(d["geography_level"]), str(area))
            ] = str(d["reason_id"])
    out = []
    for r in fixture_rows:
        area_type = str(r["area_type"])
        grain = _AREA_TYPE_TO_METRIC_GRAIN.get(area_type, area_type)
        level = _AREA_TYPE_TO_LEVEL.get(area_type, area_type)
        metric = str(r["metric"])
        our_metric = INCUMBENT_LOCAL_METRIC_ALIASES.get(metric, metric)
        declared = our_metric in set(our_metric_names.get(grain, ()))
        target_id = metric_target_ids.get(our_metric) if declared else None
        area = str(r["geography_id"])
        key = (str(target_id), level, area)
        if not declared:
            status, concern = unmapped_concern.get(metric, ("not_ported", ""))
            status = f"not_ported:{status}" if status != "not_ported" else "not_ported"
            detail = concern
        elif f"{target_id}@{area}@{r.get('period', 2025)}" in bound_names or (
            f"{target_id}@{area}" in bound_names
        ):
            status, detail = "bound", ""
        elif key in deferral_reason:
            status, detail = "signed_deferred", deferral_reason[key]
        elif key in candidates:
            status, detail = str(candidates[key].get("status")), ""
        else:
            status, detail = "no_reference", ""
        out.append(
            {
                "incumbent_name": r["name"],
                "area_type": area_type,
                "geography_id": area,
                "incumbent_metric": metric,
                "our_metric": our_metric if declared else None,
                "target_id": target_id,
                "incumbent_target": float(r["value"]),
                "incumbent_raw_value": float(r.get("raw_value", r["value"])),
                "adjustment_factor": r.get("adjustment_factor"),
                "boundary_mapped_from_2010": bool(
                    r.get("boundary_mapped_from_2010", False)
                ),
                "status": status,
                "status_detail": detail,
            }
        )
    return _with_nullable_objects(pd.DataFrame(out), ("our_metric", "target_id"))


def _relative_error(estimate: pd.Series, target: pd.Series) -> pd.Series:
    denom = target.where(target.abs() > 0, np.nan)
    return (estimate - target) / denom


def evaluation_summary(national: pd.DataFrame, local: pd.DataFrame) -> dict[str, Any]:
    """Counts and fit shares by grain, family/metric and status."""

    def block(
        frame: pd.DataFrame, group: str, estimate_col: str
    ) -> list[dict[str, Any]]:
        rows = []
        if frame.empty:
            return rows
        rel = _relative_error(frame[estimate_col], frame["incumbent_target"])
        frame = frame.assign(_rel=rel)
        for key, sub in frame.groupby(group, dropna=False):
            measured = sub["_rel"].notna()
            rows.append(
                {
                    group: key,
                    "rows": int(len(sub)),
                    "measured": int(measured.sum()),
                    "within_10pct": float(
                        (sub["_rel"].abs() <= 0.10).sum() / max(1, measured.sum())
                    ),
                    "within_25pct": float(
                        (sub["_rel"].abs() <= 0.25).sum() / max(1, measured.sum())
                    ),
                    "median_abs_rel": float(sub.loc[measured, "_rel"].abs().median())
                    if measured.any()
                    else None,
                    "by_status": {
                        str(k): int(v) for k, v in sub["status"].value_counts().items()
                    },
                }
            )
        return rows

    def overall(frame: pd.DataFrame, estimate_col: str) -> dict[str, Any]:
        if frame.empty:
            return {"rows": 0, "measured": 0}
        rel = _relative_error(frame[estimate_col], frame["incumbent_target"])
        measured = rel.notna()
        return {
            "rows": int(len(frame)),
            "measured": int(measured.sum()),
            "within_10pct": float((rel.abs() <= 0.10).sum() / max(1, measured.sum())),
            "within_25pct": float((rel.abs() <= 0.25).sum() / max(1, measured.sum())),
            "median_abs_rel": float(rel[measured].abs().median())
            if measured.any()
            else None,
            "by_status": {
                str(k): int(v) for k, v in frame["status"].value_counts().items()
            },
        }

    summary: dict[str, Any] = {
        "national": {
            "candidate": overall(national, "candidate_estimate"),
            "by_source": block(national, "source", "candidate_estimate"),
        },
        "local": {
            "candidate": overall(local, "candidate_estimate"),
            "by_area_type": block(local, "area_type", "candidate_estimate"),
            "by_metric": block(local, "incumbent_metric", "candidate_estimate"),
        },
    }
    if "incumbent_estimate" in local.columns:
        summary["local"]["incumbent"] = overall(local, "incumbent_estimate")
        summary["local"]["incumbent_by_metric"] = block(
            local, "incumbent_metric", "incumbent_estimate"
        )
    return summary


def render_markdown(
    summary: Mapping[str, Any], national: pd.DataFrame, local: pd.DataFrame
) -> str:
    lines = ["# Candidate on the incumbent's target surface", ""]
    nat = summary["national"]["candidate"]
    lines.append(
        f"**National** ({nat['rows']} incumbent rows, {nat['measured']} measurable on our side): "
        f"within 10 % {100 * nat.get('within_10pct', 0):.1f} %, within 25 % {100 * nat.get('within_25pct', 0):.1f} %, "
        f"median abs error {nat.get('median_abs_rel')}. By status: {nat.get('by_status')}."
    )
    loc = summary["local"]["candidate"]
    lines.append(
        f"**Local** ({loc['rows']} incumbent rows, {loc['measured']} measurable): "
        f"within 10 % {100 * loc.get('within_10pct', 0):.1f} %, within 25 % {100 * loc.get('within_25pct', 0):.1f} %, "
        f"median abs error {loc.get('median_abs_rel')}. By status: {loc.get('by_status')}."
    )
    if "incumbent" in summary["local"]:
        inc = summary["local"]["incumbent"]
        lines.append(
            f"**Incumbent on its own local surface** (same mapped rows): within 10 % "
            f"{100 * inc.get('within_10pct', 0):.1f} %, within 25 % {100 * inc.get('within_25pct', 0):.1f} %."
        )
    lines += [
        "",
        "## National by source",
        "",
        "| source | rows | measured | within 10 % | within 25 % | median abs | statuses |",
        "|---|---|---|---|---|---|---|",
    ]
    for b in summary["national"]["by_source"]:
        lines.append(
            f"| {b['source']} | {b['rows']} | {b['measured']} | {100 * b['within_10pct']:.0f} % | {100 * b['within_25pct']:.0f} % | {b['median_abs_rel'] if b['median_abs_rel'] is None else round(b['median_abs_rel'], 3)} | {b['by_status']} |"
        )
    lines += [
        "",
        "## Local by metric",
        "",
        "| metric | rows | measured | candidate within 10 % | incumbent within 10 % | statuses |",
        "|---|---|---|---|---|---|",
    ]
    inc_by_metric = {
        b["incumbent_metric"]: b
        for b in summary["local"].get("incumbent_by_metric", [])
    }
    for b in summary["local"]["by_metric"]:
        inc = inc_by_metric.get(b["incumbent_metric"], {})
        lines.append(
            f"| {b['incumbent_metric']} | {b['rows']} | {b['measured']} | {100 * b['within_10pct']:.0f} % | {100 * inc.get('within_10pct', 0):.0f} % | {b['by_status']} |"
        )
    worst = national.assign(
        rel_error=_relative_error(
            national["candidate_estimate"], national["incumbent_target"]
        )
    )
    worst = worst.reindex(
        worst["rel_error"].abs().sort_values(ascending=False).index
    ).head(15)
    lines += [
        "",
        "## The ugly part: worst national rows (measurable)",
        "",
        "| incumbent row | status | incumbent target | our estimate | rel. error |",
        "|---|---|---|---|---|",
    ]
    for r in worst.itertuples(index=False):
        if pd.isna(r.rel_error):
            continue
        lines.append(
            f"| {r.incumbent_name} | {r.status} | {r.incumbent_target:,.4g} | {r.candidate_estimate:,.4g} | {r.rel_error:+.1%} |"
        )
    unmeasured = national[national["candidate_estimate"].isna()]
    lines += ["", f"## Not measurable on our side: {len(unmeasured)} national rows", ""]
    for status, sub in unmeasured.groupby("status"):
        lines.append(
            f"- **{status}** ({len(sub)}): "
            + ", ".join(sorted(sub["incumbent_name"].astype(str))[:12])
            + (" …" if len(sub) > 12 else "")
        )
    return "\n".join(lines) + "\n"
