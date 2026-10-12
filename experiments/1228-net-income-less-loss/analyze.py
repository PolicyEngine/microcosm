"""Compare the #1228 re-solve arms of the Route A release.

Reads ``prepare.py``'s outputs (``--inputs``) and each arm's ``metrics.json``
and ``weights.npz`` (``--runs``), and writes ``results/summary.json`` and
``results/summary.md``:

* **Reproduction.** The release arm against the published weights: final
  loss, weight correlation and the largest per-household ratio.
* **Fit.** Each arm's own-surface loss, and, on the 5,382 rows every arm
  shares (everything but the 312 business, partnership and rental rows), the
  share within 10% and the capped error, overall and by SOI concept.
* **The 312 rows on both concepts.** Each arm's estimates against the
  published values, read as the release concepts and as the #1228 concepts.
* **Model totals.** National positive part, negative part, net, and counts of
  each income concept under each arm's weights, beside the IRS Table 1.4
  TY2022 legs aged by the rows' own factor (1.12035).
* **Concentration.** Kish ESS, and the chi-square distance from the initial
  weights.

Run: ``uv run --no-sync python experiments/1228-net-income-less-loss/analyze.py --inputs DIR --runs DIR``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
ARMS = (
    "release",
    "concept_exact",
    "concept_exact_no_business_count",
    "drop",
    "rebase_positive_part",
)
#: Table 1.4 TY2022 all-returns legs (23in14ar/22in14ar; amounts in dollars),
#: aged by the HT2 rows' factor so they sit beside the 2024 targets.
AGING = 1.12034597479376
IRS_TY2022 = {
    "schedule_c_total": {
        "pos_count": 21_969_832,
        "neg_count": 8_386_569,
        "pos_sum": 543_564_980e3 * AGING,
        "neg_sum": -133_172_174e3 * AGING,
    },
    "partnership_s_corp": {
        # Separate partnership and S-corp legs; their union bounds the
        # combined count, and the summed amounts bound the combined parts.
        "pos_count_upper": 3_311_926 + 4_027_461,
        "pos_sum_upper": (487_094_509e3 + 812_741_318e3) * AGING,
        "net": (487_094_509e3 - 175_602_300e3 + 812_741_318e3 - 93_025_607e3) * AGING,
    },
    "rental_plus_farm_rent": {
        "pos_count": 6_380_131,
        "neg_count": 3_313_901,
        "pos_sum": 148_879_455e3 * AGING,
        "neg_sum": -55_702_877e3 * AGING,
    },
}
CONCEPTS = (
    "schedule_c_total",
    "schedule_c_nonsstb",
    "sstb_only",
    "partnership_s_corp",
    "rental_only",
    "rental_plus_farm_rent",
    "farm_rent_only",
)


def capped_error(estimates: np.ndarray, targets: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.abs(targets), 1.0)
    return np.minimum(np.abs(estimates - targets) / scale, 1.0)


def kish(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / np.square(weights).sum())


def fit(estimates: np.ndarray, targets: np.ndarray) -> dict:
    error = capped_error(estimates, targets)
    return {
        "n": int(len(targets)),
        "within_10pct": float((error <= 0.10).mean()) if len(error) else None,
        "mean_capped_error": float(error.mean()) if len(error) else None,
    }


def concept_totals(totals: dict[str, np.ndarray], weights: np.ndarray) -> dict:
    out = {}
    for concept in CONCEPTS:
        pos = float(totals[f"{concept}__pos_sum"] @ weights)
        neg = float(totals[f"{concept}__neg_sum"] @ weights)
        pos_count = float(totals[f"{concept}__pos_count"] @ weights)
        neg_count = float(totals[f"{concept}__neg_count"] @ weights)
        out[concept] = {
            "pos_sum": pos,
            "neg_sum": neg,
            "net": pos + neg,
            "pos_count": pos_count,
            "neg_count": neg_count,
            "nonzero_count": pos_count + neg_count,
        }
    out["tax_units"] = float(totals["tax_units"] @ weights)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()

    meta = pd.read_parquet(args.inputs / "targets_meta.parquet")
    matrix = sparse.load_npz(args.inputs / "target_matrix.npz").tocsr()
    with np.load(args.inputs / "concept_rows.npz") as payload:
        concept_rows = payload["rows"]
        concept_matrix = sparse.csr_array(
            (payload["data"], payload["indices"], payload["indptr"]),
            shape=tuple(payload["shape"]),
        )
    weights_payload = dict(np.load(args.inputs / "weights.npz"))
    totals = dict(np.load(args.inputs / "concept_totals.npz"))
    initial = weights_payload["initial"]
    published = weights_payload["published"]
    values = meta["value"].to_numpy()
    common = ~meta["net_income_concept"].astype(bool).to_numpy()
    concept_values = values[concept_rows]
    concept_meta = meta.iloc[concept_rows].reset_index(drop=True)

    arms = {"published": {"weights": published, "metrics": None}}
    for arm in ARMS:
        path = args.runs / arm
        if not (path / "weights.npz").exists():
            print(f"missing arm {arm}")
            continue
        with np.load(path / "weights.npz") as payload:
            weights = payload["weights"]
        arms[arm] = {
            "weights": weights,
            "metrics": json.loads((path / "metrics.json").read_text()),
        }

    summary: dict = {"arms": {}}
    for arm, payload in arms.items():
        weights = payload["weights"]
        release_estimates = matrix @ weights
        concept_estimates = concept_matrix @ weights
        by_concept = {}
        for variable, group in meta[common].groupby("variable"):
            rows = group["row"].to_numpy()
            by_concept[str(variable)] = fit(release_estimates[rows], values[rows])
        rows_312 = {}
        for stem in (
            "schedule_c_income",
            "partnership_scorp_income",
            "rental_royalty_income",
        ):
            for kind in ("returns", "amount"):
                measure = f"{stem}_{kind}"
                mask = (concept_meta["measure_id"] == measure).to_numpy()
                us = mask & (concept_meta["geography_level"] == "country").to_numpy()
                states = mask & (concept_meta["geography_level"] == "state").to_numpy()
                rows_312[measure] = {
                    "us_target": float(concept_values[us][0]),
                    "us_release_concept": float(release_estimates[concept_rows[us]][0]),
                    "us_concept_exact": float(concept_estimates[us][0]),
                    "states_release_concept": fit(
                        release_estimates[concept_rows[states]],
                        concept_values[states],
                    ),
                    "states_concept_exact": fit(
                        concept_estimates[states], concept_values[states]
                    ),
                }
        own = None
        if payload["metrics"]:
            metrics = payload["metrics"]
            own_rows = np.asarray(metrics["own_rows"], dtype=np.int64)
            own_values = np.asarray(metrics["own_values"])
            own_estimates = np.asarray(metrics["own_estimates"])
            own_weights = np.asarray(metrics["own_loss_weights"])
            error = capped_error(own_estimates, own_values)
            in_312 = np.isin(own_rows, concept_rows)
            own = {
                **fit(own_estimates, own_values),
                "weighted_capped_error": float(
                    np.sum(own_weights * error) / np.sum(own_weights)
                ),
                "rows_312": fit(own_estimates[in_312], own_values[in_312]),
                "rows_312_by_measure": {
                    measure: fit(
                        own_estimates[in_312 & (own_measure == measure)],
                        own_values[in_312 & (own_measure == measure)],
                    )
                    for own_measure in [meta["measure_id"].to_numpy()[own_rows]]
                    for measure in sorted(set(own_measure[in_312]))
                },
            }
        record = {
            "own_surface": own,
            "own_final_loss": (
                payload["metrics"]["final_loss"] if payload["metrics"] else None
            ),
            "seconds": payload["metrics"]["seconds"] if payload["metrics"] else None,
            "common_rows": fit(release_estimates[common], values[common]),
            "common_rows_by_variable": by_concept,
            "rows_312": rows_312,
            "model_totals": concept_totals(totals, weights),
            "kish_ess": kish(weights),
            "chi_square_from_initial": float(
                np.sum(np.square(weights - initial) / initial) / initial.sum()
            ),
            "max_ratio_to_published": float(np.max(weights / published)),
            "min_ratio_to_published": float(np.min(weights / published)),
            "correlation_with_published": float(np.corrcoef(weights, published)[0, 1]),
        }
        summary["arms"][arm] = record
    summary["irs_ty2022_aged"] = IRS_TY2022
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "summary.json").write_text(json.dumps(summary, indent=1))
    (args.results / "summary.md").write_text(render(summary))
    print(render(summary))


def render(summary: dict) -> str:
    arms = summary["arms"]
    lines = ["# #1228 re-solve summary", ""]
    lines += [
        "| Arm | Own loss | Own rows within 10% | Own 312 rows within 10% | Common rows within 10% | Common mean capped error | Kish ESS | Corr. with published |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, record in arms.items():
        loss = record["own_final_loss"]
        own = record["own_surface"]
        own_within = "" if own is None else f"{own['within_10pct']:.4f}"
        own_312 = (
            ""
            if own is None or own["rows_312"]["within_10pct"] is None
            else f"{own['rows_312']['within_10pct']:.3f}"
        )
        lines.append(
            f"| {arm} | {'' if loss is None else f'{loss:.6f}'} | {own_within} | "
            f"{own_312} | "
            f"{record['common_rows']['within_10pct']:.4f} | "
            f"{record['common_rows']['mean_capped_error']:.5f} | "
            f"{record['kish_ess']:,.0f} | {record['correlation_with_published']:.4f} |"
        )
    lines += ["", "## Model totals (2024 weights; $B and millions)", ""]
    lines += [
        "| Arm | Concept | Positive $B | Negative $B | Net $B | Positive M | Negative M | Nonzero M |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, record in arms.items():
        for concept, totals in record["model_totals"].items():
            if concept == "tax_units":
                continue
            lines.append(
                f"| {arm} | {concept} | {totals['pos_sum'] / 1e9:,.1f} | "
                f"{totals['neg_sum'] / 1e9:,.1f} | {totals['net'] / 1e9:,.1f} | "
                f"{totals['pos_count'] / 1e6:.2f} | {totals['neg_count'] / 1e6:.2f} | "
                f"{totals['nonzero_count'] / 1e6:.2f} |"
            )
    lines += ["", "## The 312 rows: US value and state fit on each concept", ""]
    lines += [
        "| Arm | Measure | US target | US, release concept | US, #1228 concept | States within 10%, release concept | States within 10%, #1228 concept |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for arm, record in arms.items():
        for measure, row in record["rows_312"].items():
            lines.append(
                f"| {arm} | {measure} | {row['us_target']:,.0f} | "
                f"{row['us_release_concept']:,.0f} | {row['us_concept_exact']:,.0f} | "
                f"{row['states_release_concept']['within_10pct']:.2f} | "
                f"{row['states_concept_exact']['within_10pct']:.2f} |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
