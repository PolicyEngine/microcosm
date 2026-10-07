"""Selection-only scorecard for UK dataset-size experiments (microcosm#1124).

One pool, one problem, three or more weight vectors: the dense reference D on
every pool row, the stored size run S0 on its support, and each experiment's
candidate on its support. Every block is computed the same way for each
vector so a delta is the selection's alone: household total and nation
shares; the household-composition rows and lone-person share; per-area
support (rows, Kish ESS, distinct FRS source households) at constituency and
local-authority grain against an absolute floor and a relative-collapse rule;
fit by grain and family; national rows past 25 %; the share of kept rows on
the stretch cap; chi-square distances to the stage's start, to uniform and to
the kept design; and a per-area representativeness block (total-variation
distance of S from D on unbound household distributions, beside the
sampling-noise distance of an ESS-sized draw from D).

The pool profile (:class:`UKPoolProfile`) is the household-level metadata the
blocks read, built once from a run's pool frame and bound to its problem and
household axis. Nothing here writes unit records: :func:`disclosure_controlled`
suppresses small unit counts before anything is published (UKDS licence).
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
from scipy import sparse

from microcosm.build.uk_runtime.diagnostics import uk_fit_by_family, uk_weight_summary
from microcosm.build.uk_runtime.local_rowwise import uk_area_support_summary
from microcosm.build.uk_runtime.local_targets import area_groups_from_codes
from microcosm.build.uk_runtime.target_weights import NATIONAL_GRAIN, UKTargetRows
from microcosm.calibrate import (
    chi_square_distance,
    default_target_loss_scales,
    relative_error_loss,
)
from microcosm.frame import Frame

__all__ = [
    "UK_POOL_PROFILE_INCOME_COLUMNS",
    "UK_SIZE_ACCEPTANCE",
    "UKPoolProfile",
    "UKSizeWeights",
    "build_uk_pool_profile",
    "disclosure_controlled",
    "load_uk_pool_profile",
    "uk_size_acceptance",
    "uk_size_scorecard",
    "write_uk_pool_profile",
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
#: Head-of-household age bands (lower bounds) of the representativeness block.
_HEAD_AGE_BANDS = (0, 30, 45, 65, 75)
_SIZE_CAP = 6
_N_DECILES = 10
_PROFILE_FILE = "pool_profile.npz"
_PROFILE_MANIFEST = "pool_profile.json"

#: The pre-registered acceptance thresholds of #1124 (María's rulings of
#: 2026-10-07 for the collapse rule): household total within 1 % of D; each
#: nation's weight share within 5 % relative of D's; lone-person share within
#: one point of D's; no local (grain, family) losing more than one point of
#: within-10 share against S0; national rows past 25 % at most D's count + 2;
#: no constituency or local authority below 25 % of its dense ESS.
UK_SIZE_ACCEPTANCE = {
    "household_total_relative": 0.01,
    "nation_share_relative": 0.05,
    "lone_person_share_points": 1.0,
    "local_family_within_10_points": 1.0,
    "national_past_25_extra": 2,
    "relative_collapse_share": 0.25,
}


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
    """Profile ``pool``'s households for the scorecard.

    Nation comes from the constituency code's prefix (the shared
    :func:`~microcosm.build.uk_runtime.local_targets.area_groups_from_codes`
    rule); household size counts persons; the head's age band reads the
    person flagged ``is_household_head``; tenure is ``tenure_type``; gross
    income sums the person income inputs in ``income_columns`` that the
    frame carries.
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
        gross_income=income,
        binding={
            "problem_sha256": str(problem_sha256),
            "household_ids_sha256": _ids_digest(ids),
            "households": int(n),
            "income_columns": found,
        },
    )


_PROFILE_ARRAYS = (
    "household_ids",
    "source_household_ids",
    "constituency_code",
    "local_authority_code",
    "nation",
    "design_weights",
    "household_size",
    "head_age_band",
    "tenure",
    "gross_income",
)


def write_uk_pool_profile(profile: UKPoolProfile, directory: Path) -> dict[str, Any]:
    """Persist ``profile`` under ``directory`` (no pickle; strings as unicode)."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name in _PROFILE_ARRAYS:
        values = getattr(profile, name)
        arrays[name] = values.astype(str) if values.dtype == object else values
    path = directory / _PROFILE_FILE
    np.savez(path, **arrays)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"binding": dict(profile.binding), "arrays_sha256": digest}
    (directory / _PROFILE_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_uk_pool_profile(
    directory: Path, *, problem_sha256: str, household_ids: Sequence[int] | np.ndarray
) -> UKPoolProfile:
    """Load a profile, refusing one bound to another problem or household axis."""

    directory = Path(directory)
    manifest = json.loads((directory / _PROFILE_MANIFEST).read_text(encoding="utf-8"))
    path = directory / _PROFILE_FILE
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["arrays_sha256"]:
        raise ValueError("pool profile arrays do not match their recorded digest.")
    binding = manifest["binding"]
    if binding["problem_sha256"] != problem_sha256:
        raise ValueError("pool profile belongs to another problem.")
    if binding["household_ids_sha256"] != _ids_digest(np.asarray(household_ids)):
        raise ValueError("pool profile belongs to another household axis.")
    with np.load(path, allow_pickle=False) as stored:
        arrays = {
            name: (
                stored[name].astype(object)
                if stored[name].dtype.kind == "U"
                else np.asarray(stored[name])
            )
            for name in _PROFILE_ARRAYS
        }
    return UKPoolProfile(**arrays, binding=binding)


@dataclass(frozen=True)
class UKSizeWeights:
    """One weight vector on a support of pool rows, with its stage's start.

    ``support`` holds pool row positions (every row for D); ``initial`` the
    stage's starting weights on that support (the refit's baseline, or the
    pool design for D), against which the stretch cap ``max_weight_ratio``
    is measured.
    """

    label: str
    support: np.ndarray
    weights: np.ndarray
    initial: np.ndarray | None = None
    max_weight_ratio: float | None = None
    training_loss_weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        support = np.asarray(self.support, dtype=np.int64)
        weights = np.asarray(self.weights, dtype=np.float64)
        if support.shape != weights.shape or support.ndim != 1:
            raise ValueError(f"{self.label}: support and weights must align.")
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError(f"{self.label}: weights must be finite and >= 0.")
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "weights", weights)
        if self.initial is not None:
            initial = np.asarray(self.initial, dtype=np.float64)
            if initial.shape != weights.shape:
                raise ValueError(f"{self.label}: initial weights must align.")
            object.__setattr__(self, "initial", initial)


def _tvd_block(
    category: np.ndarray,
    area: np.ndarray,
    n_areas: int,
    dense_weights: np.ndarray,
    kept: UKSizeWeights,
    area_ess: np.ndarray,
) -> dict[str, object]:
    """Per-area TVD of the kept rows' distribution from D's, with its noise null."""

    codes, labels = pd.factorize(pd.Series(category, dtype=object), sort=True)
    k = len(labels)
    dense = np.bincount(area * k + codes, weights=dense_weights, minlength=n_areas * k)
    dense = dense.reshape(n_areas, k)
    sample = np.bincount(
        area[kept.support] * k + codes[kept.support],
        weights=kept.weights,
        minlength=n_areas * k,
    ).reshape(n_areas, k)
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
    tvd, ratio = tvd[valid], ratio[valid]
    return {
        "categories": int(k),
        "areas": int(valid.sum()),
        "tvd_median": float(np.median(tvd)) if tvd.size else None,
        "tvd_p90": float(np.quantile(tvd, 0.9)) if tvd.size else None,
        "tvd_max": float(tvd.max()) if tvd.size else None,
        "tvd_to_noise_median": (
            float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else None
        ),
    }


def _joint(*arrays: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            "|".join(str(value) for value in values)
            for values in zip(*arrays, strict=True)
        ],
        dtype=object,
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


def _fit_rows(
    rows: UKTargetRows, estimates: np.ndarray, targets: np.ndarray
) -> pd.DataFrame:
    scales = default_target_loss_scales(targets)
    error = (estimates - targets) / scales
    return pd.DataFrame(
        {
            "target_name": rows.names,
            "family": [
                f"{grain}/{family}"
                for grain, family in zip(rows.grain, rows.family, strict=True)
            ],
            "grain": rows.grain,
            "estimate": estimates,
            "target": targets,
            "relative_error": error,
            "abs_relative_error": np.abs(error),
        }
    )


def _set_block(
    *,
    weights: UKSizeWeights,
    profile: UKPoolProfile,
    rows: UKTargetRows,
    matrix: sparse.csr_array,
    targets: np.ndarray,
    yardstick: np.ndarray,
    target_loss_cap: float,
    dense_area_ess: Mapping[str, pd.DataFrame] | None,
    ess_floor: float,
) -> dict[str, object]:
    support, values = weights.support, weights.weights
    estimates = np.asarray(matrix[:, support] @ values, dtype=np.float64)
    total = float(values.sum())
    persons = float((profile.household_size[support] * values).sum())
    nation_share = {
        nation: float(values[profile.nation[support] == nation].sum() / total)
        for nation in sorted(set(profile.nation.tolist()))
    }
    fit = _fit_rows(rows, estimates, targets)
    by_cell = {
        block["family"]: {
            "n_targets": block["n_targets"],
            "share_within_10pct": block["share_within_10pct"],
            "share_within_25pct": block["share_within_25pct"],
        }
        for block in uk_fit_by_family(fit)
    }
    national = fit[fit["grain"] == NATIONAL_GRAIN]
    past_25 = national[national["abs_relative_error"] > 0.25]
    composition = fit[
        (fit["grain"] == NATIONAL_GRAIN)
        & (np.asarray(rows.family, dtype=object) == "ons_household_composition")
    ]
    lone = composition[composition["target_name"].str.contains("lone_households")]
    sizes = np.bincount(
        profile.household_size[support], weights=values, minlength=_SIZE_CAP + 1
    )
    areas: dict[str, object] = {}
    for grain, codes in (
        ("constituency", profile.constituency_code),
        ("la", profile.local_authority_code),
    ):
        roster = np.asarray(sorted(set(codes.tolist())), dtype=object)
        summary = uk_area_support_summary(
            codes[support],
            values,
            area_codes=roster,
            source_household_ids=profile.source_household_ids[support],
        )
        ess = summary["effective_sample_size"].to_numpy(dtype=np.float64)
        rows_kept = summary["nonzero_households"].to_numpy(dtype=np.float64)
        block: dict[str, object] = {
            "areas": int(len(summary)),
            "ess_min": float(ess.min()) if ess.size else None,
            "ess_p5": float(np.quantile(ess, 0.05)) if ess.size else None,
            "ess_median": float(np.median(ess)) if ess.size else None,
            "below_ess_floor": int((ess < ess_floor).sum()),
            "rows_median": float(np.median(rows_kept)) if rows_kept.size else None,
            "rows_min": int(rows_kept.min()) if rows_kept.size else None,
            "ess_per_row_median": float(
                np.median(
                    np.divide(
                        ess, rows_kept, out=np.zeros_like(ess), where=rows_kept > 0
                    )
                )
            ),
            "sources_median": float(
                np.median(
                    summary["nonzero_source_households"].to_numpy(dtype=np.float64)
                )
            ),
            "sources_min": int(summary["nonzero_source_households"].min()),
        }
        if dense_area_ess is not None:
            reference = (
                dense_area_ess[grain]
                .set_index("area_code")["effective_sample_size"]
                .reindex(summary["area_code"])
                .to_numpy(dtype=np.float64)
            )
            ratio = np.divide(
                ess, reference, out=np.zeros_like(ess), where=reference > 0
            )
            block["ess_to_dense_median"] = float(np.median(ratio))
            block["below_relative_collapse"] = int(
                (ratio < UK_SIZE_ACCEPTANCE["relative_collapse_share"]).sum()
            )
        areas[grain] = block
    weights_block = uk_weight_summary(values)
    block: dict[str, object] = {
        "label": weights.label,
        "households": total,
        "persons": persons,
        "rows": int(len(values)),
        "distinct_source_households": int(
            len(set(profile.source_household_ids[support].tolist()))
        ),
        "nation_share": nation_share,
        "household_size_share": {
            str(size): float(sizes[size] / total) for size in range(1, _SIZE_CAP + 1)
        },
        "composition_rows": [
            {
                "target_name": row.target_name,
                "target": float(row.target),
                "estimate": float(row.estimate),
                "relative_error": float(row.relative_error),
            }
            for row in composition.itertuples(index=False)
        ],
        "lone_person_share": float(lone["estimate"].sum() / total)
        if len(lone)
        else None,
        "fit_by_grain_family": by_cell,
        "national_past_25": [
            {
                "target_name": row.target_name,
                "relative_error": float(row.relative_error),
            }
            for row in past_25.sort_values(
                "abs_relative_error", ascending=False
            ).itertuples(index=False)
        ],
        "loss_grain_equal_yardstick": float(
            relative_error_loss(
                estimates,
                targets,
                target_loss_weights=yardstick,
                target_loss_scales=default_target_loss_scales(targets),
                target_loss_cap=target_loss_cap,
            )
        ),
        "areas": areas,
        "weights": weights_block,
    }
    if weights.training_loss_weights is not None:
        block["loss_training_weights"] = float(
            relative_error_loss(
                estimates,
                targets,
                target_loss_weights=weights.training_loss_weights,
                target_loss_scales=default_target_loss_scales(targets),
                target_loss_cap=target_loss_cap,
            )
        )
    if weights.initial is not None:
        start = weights.initial
        if weights.max_weight_ratio is not None:
            cap = weights.max_weight_ratio * start
            block["share_at_stretch_cap"] = float(np.mean(values >= (1.0 - 1e-9) * cap))
        kept_design = profile.design_weights[support]
        scaled_design = kept_design * (profile.design_weights.sum() / kept_design.sum())
        block["chi_square_to_start"] = float(chi_square_distance(values, start))
        block["chi_square_to_uniform"] = float(
            chi_square_distance(values, np.full_like(start, start.mean()))
        )
        block["chi_square_to_kept_design"] = float(
            chi_square_distance(values, scaled_design)
        )
    return block


def uk_size_scorecard(
    *,
    profile: UKPoolProfile,
    rows: UKTargetRows,
    matrix: sparse.csr_array,
    targets: np.ndarray,
    yardstick: np.ndarray,
    target_loss_cap: float,
    dense: UKSizeWeights,
    control: UKSizeWeights,
    candidates: Sequence[UKSizeWeights] = (),
    ess_floor: float = 50.0,
) -> dict[str, object]:
    """Score D, the stored size run S0 and each candidate on one pool.

    ``matrix`` is the problem's target × pool-row constraint matrix and
    ``yardstick`` the fixed ``grain_equal`` loss weights every set is also
    scored under, whatever rule it trained on.
    """

    matrix = sparse.csr_array(matrix)
    targets = np.asarray(targets, dtype=np.float64)
    if matrix.shape != (len(rows), len(profile)):
        raise ValueError("scorecard matrix must be targets × pool households.")
    common = dict(
        profile=profile,
        rows=rows,
        matrix=matrix,
        targets=targets,
        yardstick=np.asarray(yardstick, dtype=np.float64),
        target_loss_cap=float(target_loss_cap),
        ess_floor=float(ess_floor),
    )
    dense_area_ess = {
        grain: uk_area_support_summary(
            codes[dense.support],
            dense.weights,
            area_codes=np.asarray(sorted(set(codes.tolist())), dtype=object),
            source_household_ids=profile.source_household_ids[dense.support],
        )[["area_code", "effective_sample_size"]]
        for grain, codes in (
            ("constituency", profile.constituency_code),
            ("la", profile.local_authority_code),
        )
    }
    sets = [dense, control, *candidates]
    blocks = {
        weights.label: _set_block(
            weights=weights,
            dense_area_ess=None if weights is dense else dense_area_ess,
            **common,
        )
        for weights in sets
    }
    dense_weights = np.zeros(len(profile), dtype=np.float64)
    dense_weights[dense.support] = dense.weights
    deciles = _income_deciles(profile, dense_weights)
    variables = {
        "household_size": profile.household_size,
        "tenure": profile.tenure,
        "head_age_band": profile.head_age_band,
        "income_decile": deciles,
    }
    variables["size_x_tenure"] = _joint(profile.household_size, profile.tenure)
    variables["tenure_x_head_age"] = _joint(profile.tenure, profile.head_age_band)
    variables["size_x_income_decile"] = _joint(profile.household_size, deciles)
    variables["head_age_x_income_decile"] = _joint(profile.head_age_band, deciles)
    representativeness: dict[str, object] = {}
    for weights in sets[1:]:
        per_grain = {}
        for grain, codes in (
            ("constituency", profile.constituency_code),
            ("la", profile.local_authority_code),
        ):
            area, roster = pd.factorize(pd.Series(codes, dtype=object), sort=True)
            n_areas = len(roster)
            sums = np.bincount(
                area[weights.support], weights=weights.weights, minlength=n_areas
            )
            squares = np.bincount(
                area[weights.support], weights=weights.weights**2, minlength=n_areas
            )
            area_ess = np.divide(
                sums**2, squares, out=np.zeros_like(sums), where=squares > 0
            )
            per_grain[grain] = {
                name: _tvd_block(
                    values, area, n_areas, dense_weights, weights, area_ess
                )
                for name, values in variables.items()
            }
        representativeness[weights.label] = per_grain
    return {
        "sets": blocks,
        "representativeness": representativeness,
        "ess_floor": float(ess_floor),
        "acceptance": {
            weights.label: uk_size_acceptance(
                blocks[weights.label],
                dense=blocks[dense.label],
                control=blocks[control.label],
            )
            for weights in sets[1:]
        },
    }


def uk_size_acceptance(
    candidate: Mapping[str, Any],
    *,
    dense: Mapping[str, Any],
    control: Mapping[str, Any],
    thresholds: Mapping[str, float] = UK_SIZE_ACCEPTANCE,
) -> dict[str, object]:
    """Evaluate the six pre-registered #1124 criteria for one scorecard set."""

    criteria: dict[str, dict[str, object]] = {}
    relative = abs(candidate["households"] - dense["households"]) / dense["households"]
    criteria["household_total"] = {
        "value": relative,
        "threshold": thresholds["household_total_relative"],
        "pass": relative <= thresholds["household_total_relative"],
    }
    worst_nation = max(
        (
            abs(candidate["nation_share"].get(nation, 0.0) - share) / share
            for nation, share in dense["nation_share"].items()
            if share > 0
        ),
        default=0.0,
    )
    criteria["nation_shares"] = {
        "value": worst_nation,
        "threshold": thresholds["nation_share_relative"],
        "pass": worst_nation <= thresholds["nation_share_relative"],
    }
    if candidate["lone_person_share"] is None or dense["lone_person_share"] is None:
        criteria["lone_person_share"] = {"value": None, "pass": None}
    else:
        points = 100.0 * abs(
            candidate["lone_person_share"] - dense["lone_person_share"]
        )
        criteria["lone_person_share"] = {
            "value": points,
            "threshold": thresholds["lone_person_share_points"],
            "pass": points <= thresholds["lone_person_share_points"],
        }
    losses = {
        cell: 100.0
        * (
            control["fit_by_grain_family"][cell]["share_within_10pct"]
            - block["share_within_10pct"]
        )
        for cell, block in candidate["fit_by_grain_family"].items()
        if not cell.startswith(f"{NATIONAL_GRAIN}/")
        and cell in control["fit_by_grain_family"]
    }
    worst_cell = max(losses, key=losses.get) if losses else None
    criteria["local_family_within_10"] = {
        "value": losses.get(worst_cell) if worst_cell else 0.0,
        "worst_cell": worst_cell,
        "threshold": thresholds["local_family_within_10_points"],
        "pass": (max(losses.values()) if losses else 0.0)
        <= thresholds["local_family_within_10_points"],
    }
    allowed = len(dense["national_past_25"]) + thresholds["national_past_25_extra"]
    criteria["national_past_25"] = {
        "value": len(candidate["national_past_25"]),
        "threshold": allowed,
        "pass": len(candidate["national_past_25"]) <= allowed,
    }
    collapse = sum(
        int(block.get("below_relative_collapse", 0))
        for block in candidate["areas"].values()
    )
    criteria["relative_collapse"] = {
        "value": collapse,
        "threshold": 0,
        "pass": collapse == 0,
        "below_ess_floor": {
            grain: block["below_ess_floor"]
            for grain, block in candidate["areas"].items()
        },
        "ess_min": {
            grain: block["ess_min"] for grain, block in candidate["areas"].items()
        },
    }
    return {
        "criteria": criteria,
        "passes": [
            name for name, block in criteria.items() if block.get("pass") is True
        ],
        "all_pass": all(block.get("pass") is True for block in criteria.values()),
    }


_UNIT_COUNT_KEY_PARTS = ("rows", "sources", "nonzero", "distinct")


def disclosure_controlled(value: Any, *, minimum_count: int = 10, key: str = "") -> Any:
    """Suppress small unit counts (survey rows, source households) for publication.

    Integer values under a key naming a unit count (``rows``, ``sources``,
    ``nonzero``, ``distinct``) below ``minimum_count`` become ``"<10"``;
    everything else is aggregate and passes through.
    """

    if isinstance(value, Mapping):
        return {
            name: disclosure_controlled(
                item, minimum_count=minimum_count, key=str(name)
            )
            for name, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [
            disclosure_controlled(item, minimum_count=minimum_count, key=key)
            for item in value
        ]
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and any(part in key for part in _UNIT_COUNT_KEY_PARTS)
        and value < minimum_count
    ):
        return f"<{minimum_count}"
    return value
