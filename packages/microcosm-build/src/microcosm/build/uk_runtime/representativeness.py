"""Per-area representativeness of a sized UK local release (microcosm#1124 item G).

A sized build ships S: a selection of the pool's households with refitted
weights. The dense solution D weights every pool household. Within each
constituency and local authority, this module compares S's distribution of
household characteristics no target binds with D's:
- household size, tenure, the head's age band, the ONS household type and the
  household's income decile;
- six joints of them.

The distance is the total variation distance (TVD). Beside it sits the TVD a
random draw from D of S's effective sample size would show by chance (the
noise null), so a TVD-to-noise ratio near one means S is as close to D as its
own size allows. A configuration is then judged on what it does to the joint,
not only to the bound rows.

The per-area table is unit-adjacent evidence and stays in a private directory
outside the repository. The summary keeps cross-area statistics only: medians,
the 90th percentile and counts of areas above a ratio. They are taken over
areas whose S support rests on at least ten distinct source households, and
carry no area code, maximum or minimum (UKDS disclosure control).

The pool profile and its variables are those of the #1163 size-experiment
scorecard, so that harness can read them from here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.local_rowwise import uk_area_support_summary
from microcosm.build.uk_runtime.local_targets import area_groups_from_codes
from microcosm.calibrate.artifacts import decode_problem, decode_solution
from microcosm.frame import Frame
from microcosm.graph.store import ContentStore

__all__ = [
    "UK_POOL_PROFILE_INCOME_COLUMNS",
    "UK_REPRESENTATIVENESS_GRAINS",
    "UK_REPRESENTATIVENESS_MINIMUM_AREAS",
    "UK_REPRESENTATIVENESS_MINIMUM_SOURCES",
    "UK_REPRESENTATIVENESS_RATIO_THRESHOLD",
    "UKLocalRunSelection",
    "UKPoolProfile",
    "build_uk_pool_profile",
    "read_uk_local_run",
    "uk_area_representativeness",
    "uk_area_tvd",
    "uk_representativeness_summary",
    "uk_representativeness_variables",
]

#: Person income inputs summed into the profile's gross household income
#: (those present on the frame; the profile records which were found). An
#: input-based gross income, not the engine's net income: it ranks
#: households for the representativeness block only.
UK_POOL_PROFILE_INCOME_COLUMNS = (
    "employment_income",
    "self_employment_income",
    "private_pension_income",
    "savings_interest_income",
    "dividend_income",
    "property_income",
    "maintenance_income",
    "miscellaneous_income",
)
#: Head-of-household age bands (lower bounds).
_HEAD_AGE_BANDS = (0, 30, 45, 65, 75)
_SIZE_CAP = 6
_N_DECILES = 10

#: The grains compared, with the profile attribute holding each area code.
UK_REPRESENTATIVENESS_GRAINS = (
    ("constituency", "constituency_code"),
    ("local_authority", "local_authority_code"),
)
#: An area enters the summary when S's support there rests on at least this
#: many distinct source households (the UKDS working floor).
UK_REPRESENTATIVENESS_MINIMUM_SOURCES = 10
#: A TVD this many times its noise null counts the area as departing from D.
UK_REPRESENTATIVENESS_RATIO_THRESHOLD = 2.0
#: Quantiles over fewer scored areas than this are withheld.
UK_REPRESENTATIVENESS_MINIMUM_AREAS = 10

_POOL_NODE = "uk.full.pool"
_PROBLEM_ARTIFACT = "uk.full.problem/problem"
_DENSE_ARTIFACT = "uk.full.dense/solution"
_REFIT_ARTIFACT = "uk.full.size_refit/solution"
_SIZE_ARTIFACT = "uk.full.size_refit/size"


@dataclass(frozen=True)
class UKPoolProfile:
    """Household metadata of one pool, in the problem's household order."""

    household_ids: np.ndarray
    source_household_ids: np.ndarray
    constituency_code: np.ndarray
    local_authority_code: np.ndarray
    nation: np.ndarray
    design_weights: np.ndarray
    household_size: np.ndarray
    head_age_band: np.ndarray
    tenure: np.ndarray
    household_type: np.ndarray
    gross_income: np.ndarray
    binding: Mapping[str, Any]

    def __len__(self) -> int:
        return len(self.household_ids)


def _ids_digest(ids: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(ids, dtype=np.int64)).tobytes()
    ).hexdigest()


def build_uk_pool_profile(
    pool: Frame,
    *,
    problem_sha256: str,
    income_columns: Sequence[str] = UK_POOL_PROFILE_INCOME_COLUMNS,
) -> UKPoolProfile:
    """Profile ``pool``'s households.

    Nation comes from the constituency code's prefix (the shared
    :func:`~microcosm.build.uk_runtime.local_targets.area_groups_from_codes`
    rule); household size counts persons; the head's age band reads the
    person flagged ``is_household_head``; tenure is ``tenure_type``; the
    household type is the ``frs_relationships`` stage's ``ons_household_type``
    (``"unknown"`` on a frame without it); gross income sums the person income
    inputs in ``income_columns`` that the frame carries.
    """

    household = pool.table("household")
    person = pool.table("person")
    ids = household["household_id"].to_numpy()
    membership = pool.schema.membership_column("household")
    position = pd.Index(ids).get_indexer(person[membership].to_numpy())
    if (position < 0).any():
        raise ValueError("pool persons reference households outside the pool.")
    n = len(ids)
    size = np.bincount(position, minlength=n)
    head = person["is_household_head"].to_numpy(dtype=bool)
    if np.bincount(position[head], minlength=n).max(initial=0) > 1:
        raise ValueError("a pool household has more than one household head.")
    head_age = np.full(n, np.nan)
    head_age[position[head]] = person.loc[head, "age"].to_numpy(dtype=np.float64)
    band_index = np.searchsorted(_HEAD_AGE_BANDS, head_age, side="right") - 1
    band_labels = np.asarray(
        [
            f"{lower}-{upper - 1}"
            for lower, upper in zip(
                _HEAD_AGE_BANDS, (*_HEAD_AGE_BANDS[1:], 200), strict=True
            )
        ],
        dtype=object,
    )
    head_age_band = np.where(
        np.isnan(head_age),
        "no_head",
        band_labels[np.clip(band_index, 0, len(band_labels) - 1)],
    ).astype(object)
    found = [column for column in income_columns if column in person.columns]
    income = np.zeros(n, dtype=np.float64)
    for column in found:
        np.add.at(income, position, person[column].to_numpy(dtype=np.float64))
    constituency = household["constituency_code"].astype(str).to_numpy(dtype=object)
    groups = area_groups_from_codes(sorted(set(constituency.tolist())))
    has_type = "ons_household_type" in household.columns
    household_type = (
        household["ons_household_type"].astype(str).to_numpy(dtype=object)
        if has_type
        else np.full(n, "unknown", dtype=object)
    )
    return UKPoolProfile(
        household_ids=np.asarray(ids, dtype=np.int64),
        source_household_ids=household["source_household_id"].to_numpy(dtype=object),
        constituency_code=constituency,
        local_authority_code=household["local_authority_code"]
        .astype(str)
        .to_numpy(dtype=object),
        nation=np.asarray([groups[code] for code in constituency], dtype=object),
        design_weights=np.asarray(
            pool.weights_for("household").values, dtype=np.float64
        ),
        household_size=np.minimum(size, _SIZE_CAP).astype(np.int64),
        head_age_band=head_age_band,
        tenure=household["tenure_type"].astype(str).to_numpy(dtype=object),
        household_type=household_type,
        gross_income=income,
        binding={
            "problem_sha256": str(problem_sha256),
            "household_ids_sha256": _ids_digest(ids),
            "households": int(n),
            "income_columns": found,
            "household_type_column": "ons_household_type" if has_type else None,
        },
    )


def _income_deciles(profile: UKPoolProfile, dense_weights: np.ndarray) -> np.ndarray:
    order = np.argsort(profile.gross_income, kind="stable")
    cumulative = np.cumsum(dense_weights[order])
    total = cumulative[-1] if cumulative.size else 0.0
    deciles = np.empty(len(profile), dtype=np.int64)
    deciles[order] = np.minimum(
        (cumulative / total * _N_DECILES).astype(np.int64) if total > 0 else 0,
        _N_DECILES - 1,
    )
    return deciles


def _joint(*arrays: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            "|".join(str(value) for value in values)
            for values in zip(*arrays, strict=True)
        ],
        dtype=object,
    )


def uk_representativeness_variables(
    profile: UKPoolProfile, dense_weights: np.ndarray
) -> dict[str, np.ndarray]:
    """The five marginals and six joints, per pool household.

    Income deciles are cut at D's weighted income distribution, so S and D
    are read on the same cut points.
    """

    deciles = _income_deciles(profile, np.asarray(dense_weights, dtype=np.float64))
    return {
        "household_size": profile.household_size,
        "tenure": profile.tenure,
        "head_age_band": profile.head_age_band,
        "household_type": profile.household_type,
        "income_decile": deciles,
        "size_x_tenure": _joint(profile.household_size, profile.tenure),
        "tenure_x_head_age": _joint(profile.tenure, profile.head_age_band),
        "size_x_income_decile": _joint(profile.household_size, deciles),
        "head_age_x_income_decile": _joint(profile.head_age_band, deciles),
        "type_x_tenure": _joint(profile.household_type, profile.tenure),
        "type_x_income_decile": _joint(profile.household_type, deciles),
    }


def uk_area_tvd(
    category: np.ndarray,
    area: np.ndarray,
    n_areas: int,
    dense_weights: np.ndarray,
    support: np.ndarray,
    weights: np.ndarray,
    area_ess: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-area TVD of S's distribution of ``category`` from D's.

    ``area`` holds each pool household's area index; S is ``weights`` on the
    pool rows in ``support``. The noise null is the expected TVD of a draw of
    S's effective sample size from D, ``0.5 * sum(sqrt(2 p (1 - p) / (pi n)))``
    for D's shares ``p``. An area where either side carries no mass is not
    valid; its distances are NaN.
    """

    codes, labels = pd.factorize(pd.Series(category, dtype=object), sort=True)
    k = len(labels)
    dense = (
        np.bincount(area * k + codes, weights=dense_weights, minlength=n_areas * k)
        .astype(np.float64)
        .reshape(n_areas, k)
    )
    # An empty S makes bincount return integers; the shares need floats.
    sample = (
        np.bincount(
            area[support] * k + codes[support], weights=weights, minlength=n_areas * k
        )
        .astype(np.float64)
        .reshape(n_areas, k)
    )
    dense_total = dense.sum(axis=1, keepdims=True)
    sample_total = sample.sum(axis=1, keepdims=True)
    valid = (dense_total[:, 0] > 0) & (sample_total[:, 0] > 0)
    p = np.divide(dense, dense_total, out=np.zeros_like(dense), where=dense_total > 0)
    q = np.divide(
        sample, sample_total, out=np.zeros_like(sample), where=sample_total > 0
    )
    tvd = 0.5 * np.abs(p - q).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        noise = 0.5 * np.sqrt(2.0 * p * (1.0 - p) / (np.pi * area_ess[:, None])).sum(
            axis=1
        )
    ratio = np.divide(tvd, noise, out=np.full_like(tvd, np.nan), where=noise > 0)
    nan = np.full(n_areas, np.nan)
    return {
        "valid": valid,
        "tvd": np.where(valid, tvd, nan),
        "noise": np.where(valid, noise, nan),
        "ratio": np.where(valid, ratio, nan),
    }


def uk_area_representativeness(
    profile: UKPoolProfile,
    *,
    dense_weights: np.ndarray,
    support: np.ndarray,
    weights: np.ndarray,
) -> pd.DataFrame:
    """One row per grain, area and variable: S's distances from D and its support.

    The table names areas and carries per-area support, so it stays private.
    """

    dense_weights = np.asarray(dense_weights, dtype=np.float64)
    support = np.asarray(support, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if dense_weights.shape != (len(profile),):
        raise ValueError("D's weights must align with the pool's households.")
    if support.shape != weights.shape or support.ndim != 1:
        raise ValueError("S's support and weights must align.")
    variables = uk_representativeness_variables(profile, dense_weights)
    frames = []
    for grain, attribute in UK_REPRESENTATIVENESS_GRAINS:
        codes = getattr(profile, attribute)
        area, roster = pd.factorize(pd.Series(codes, dtype=object), sort=True)
        n_areas = len(roster)
        support_table = uk_area_support_summary(
            codes[support],
            weights,
            area_codes=np.asarray(roster, dtype=object),
            source_household_ids=profile.source_household_ids[support],
        ).set_index("area_code")
        area_ess = (
            support_table["effective_sample_size"]
            .reindex(roster)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )
        sources = (
            support_table["nonzero_source_households"]
            .reindex(roster)
            .fillna(0)
            .to_numpy(dtype=np.int64)
        )
        for name, values in variables.items():
            distances = uk_area_tvd(
                values, area, n_areas, dense_weights, support, weights, area_ess
            )
            frames.append(
                pd.DataFrame(
                    {
                        "grain": grain,
                        "area_code": np.asarray(roster, dtype=object),
                        "variable": name,
                        "tvd": distances["tvd"],
                        "noise": distances["noise"],
                        "tvd_to_noise": distances["ratio"],
                        "valid": distances["valid"],
                        "s_effective_sample_size": area_ess,
                        "s_distinct_sources": sources,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def uk_representativeness_summary(
    table: pd.DataFrame,
    *,
    minimum_sources: int = UK_REPRESENTATIVENESS_MINIMUM_SOURCES,
    ratio_threshold: float = UK_REPRESENTATIVENESS_RATIO_THRESHOLD,
    minimum_areas: int = UK_REPRESENTATIVENESS_MINIMUM_AREAS,
) -> dict[str, Any]:
    """Cross-area statistics per grain and variable, safe to publish.

    An area is scored when it is valid and S rests on at least
    ``minimum_sources`` distinct source households there. Quantiles are taken
    over scored areas and withheld (``None``) below ``minimum_areas``. No area
    code, maximum or minimum is carried.
    """

    summary: dict[str, dict[str, Any]] = {}
    for (grain, variable), rows in table.groupby(["grain", "variable"], sort=True):
        valid = rows["valid"].to_numpy(dtype=bool)
        supported = rows["s_distinct_sources"].to_numpy() >= minimum_sources
        scored = rows[valid & supported]
        enough = len(scored) >= minimum_areas
        tvd = scored["tvd"].to_numpy(dtype=np.float64)
        ratio = scored["tvd_to_noise"].to_numpy(dtype=np.float64)
        finite_ratio = ratio[np.isfinite(ratio)]
        summary.setdefault(str(grain), {})[str(variable)] = {
            "areas": int(len(rows)),
            "areas_scored": int(len(scored)),
            "areas_below_source_floor": int((valid & ~supported).sum()),
            "tvd_median": float(np.median(tvd)) if enough else None,
            "tvd_p90": float(np.quantile(tvd, 0.9)) if enough else None,
            "tvd_to_noise_median": (
                float(np.median(finite_ratio)) if enough and finite_ratio.size else None
            ),
            "areas_above_ratio_threshold": (
                int((finite_ratio > ratio_threshold).sum()) if enough else None
            ),
        }
    return {
        "minimum_sources": int(minimum_sources),
        "ratio_threshold": float(ratio_threshold),
        "minimum_areas": int(minimum_areas),
        "grains": summary,
    }


@dataclass(frozen=True)
class UKLocalRunSelection:
    """D and S of one finished sized run, on its pool's profile."""

    profile: UKPoolProfile
    dense_weights: np.ndarray
    support: np.ndarray
    weights: np.ndarray
    artifacts: Mapping[str, str]


def _evidence(run_dir: Path) -> dict[str, Mapping[str, Any]]:
    path = run_dir / "evidence-index.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: not a finished graph build.")
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(run_dir: Path, evidence: Mapping[str, Any], key: str) -> bytes:
    if key not in evidence:
        raise ValueError(f"the run's evidence index has no {key!r} artifact.")
    entry = evidence[key]
    payload = (run_dir / str(entry["filename"])).read_bytes()
    if (
        len(payload) != int(entry["size_bytes"])
        or hashlib.sha256(payload).hexdigest() != entry["sha256"]
    ):
        raise ValueError(f"{entry['filename']} differs from its evidence-index digest.")
    return payload


def _pool_frame(run_dir: Path, graph_store: Path | None) -> Frame:
    manifest = json.loads((run_dir / "numerical.graph.json").read_text("utf-8"))
    nodes = manifest.get("content_addressed", manifest).get("nodes", {})
    record = nodes.get(_POOL_NODE)
    if not isinstance(record, Mapping) or not record.get("frame_key"):
        raise ValueError(f"the run's graph manifest records no {_POOL_NODE} frame.")
    key = str(record["frame_key"])
    store = run_dir / ".graph-store" if graph_store is None else Path(graph_store)
    return ContentStore.load_frame_path(store / "objects" / key[:2] / key)


def read_uk_local_run(
    run_dir: Path, *, graph_store: Path | None = None
) -> UKLocalRunSelection | None:
    """Read D and S of a finished sized UK local run, or ``None`` when unsized.

    Every artifact is checked against the run's evidence index. D is the dense
    solution over the pool; S is the size refit's solution on the rows the
    size receipt names, which must be the solution's own households. A run
    without a size refit ships D itself, so there is nothing to compare.
    """

    run_dir = Path(run_dir).resolve()
    evidence = _evidence(run_dir)
    if _REFIT_ARTIFACT not in evidence:
        return None
    problem = decode_problem(_artifact(run_dir, evidence, _PROBLEM_ARTIFACT))
    dense = decode_solution(
        _artifact(run_dir, evidence, _DENSE_ARTIFACT),
        problem_sha256=problem.sha256,
        entity_ids=problem.entity_ids,
    )
    refit = decode_solution(
        _artifact(run_dir, evidence, _REFIT_ARTIFACT), problem_sha256=problem.sha256
    )
    size = json.loads(_artifact(run_dir, evidence, _SIZE_ARTIFACT))
    support = np.asarray(size["pool_row_indices"], dtype=np.int64)
    pool_ids = np.asarray(problem.entity_ids, dtype=object)
    if support.size and (support.min() < 0 or support.max() >= len(pool_ids)):
        raise ValueError("the size receipt names rows outside the pool.")
    if list(pool_ids[support]) != list(refit.entity_ids):
        raise ValueError("the size receipt's rows are not the refit's households.")
    profile = build_uk_pool_profile(
        _pool_frame(run_dir, graph_store), problem_sha256=problem.sha256
    )
    if list(profile.household_ids) != [int(i) for i in problem.entity_ids]:
        raise ValueError("the pool frame's households are not the problem's.")
    return UKLocalRunSelection(
        profile=profile,
        dense_weights=np.asarray(dense.weights, dtype=np.float64),
        support=support,
        weights=np.asarray(refit.weights, dtype=np.float64),
        artifacts={
            key: str(evidence[key]["sha256"])
            for key in (
                _PROBLEM_ARTIFACT,
                _DENSE_ARTIFACT,
                _REFIT_ARTIFACT,
                _SIZE_ARTIFACT,
            )
        },
    )
