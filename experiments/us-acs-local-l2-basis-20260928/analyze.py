"""Build the ESS-versus-fit frontier from the sweep's metrics.

Reads ``results/runs/*.json`` (one ``sweep.py`` payload per run) and writes
``results/frontier.csv``, ``results/frontier.md``, ``results/floors.md``,
``results/key_configs.md`` (also spliced into the README) and
``results/frontier.png``. ``dup_*`` runs repeat a first-head configuration at
a later head and are kept out of the frontier; ``results/rerun_variation.json``
compares each with its original: whether weights and trajectory are
byte-identical, the first epoch where the trajectories differ, and every
frontier metric's relative change (``weights.npz`` beside the checkpoint).

Run: ``uv run --with matplotlib python experiments/us-acs-local-l2-basis-20260928/analyze.py``
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE / "results" / "runs"
LOCAL_RUNS = Path(
    "/Users/maxghenis/PolicyEngine/_build_artifacts/acs-local-l2-basis-20260928/runs"
)
#: Reruns of a first-head run at a later kernel head, and the run they repeat.
#: The frontier metrics the rerun comparison reports.
RERUN_METRICS = (
    "final_loss",
    "within_10pct",
    "kish_ess",
    "top_1pct_share",
    "state_ess_min",
    "cd_ess_median",
    "cd_ess_min",
    "cds_below_50",
    "cds_below_0.25_of_prior",
    "ma_ess",
    "chi_square_distance_from_prior",
    "donor_mass_share",
    "within_10pct_soi",
)
HEAD_PAIRS = {
    "dup_release_repro": "release_repro",
    "dup_soft_chi_s050_0.03": "soft_chi_s050_0.03",
}
FAMILIES = ("soi", "snap", "medicaid", "pop_state", "pop_cd")
#: Candidate release-gate floors on Kish ESS, and the relative floor (share of
#: the prior's own ESS in the same geography).
STATE_FLOORS = (300, 1000)
CD_FLOORS = (50, 100, 200)
RELATIVE_FLOORS = (0.25, 0.5)
PRIOR_LABELS = {0.5: "0.5 (release)", 0.7: "0.7", 0.9: "0.9", 0.964: "0.964"}
#: Categorical slots 1-4 of the dataviz reference palette, in fixed order.
PRIOR_COLORS = {0.5: "#2a78d6", 0.7: "#eb6834", 0.9: "#1baf7a", 0.964: "#eda100"}


def _prior(payload: dict) -> float:
    share = payload["spec"].get("acs_share")
    return 0.5 if share is None else round(float(share), 3)


def _head(payload: dict) -> str | None:
    head = payload["provenance"]["git"].get("head_from_env")
    return head[:8] if head else None


def row(payload: dict) -> dict:
    spec = payload["spec"]
    metrics = payload["metrics"]
    concentration = metrics["concentration"]
    national = concentration["national"]
    state = concentration["state_summary"]
    district = concentration["cd_summary"]
    ma = concentration["massachusetts"]
    ma_districts = [entry["kish_ess"] for entry in ma["districts"].values()]
    fit = metrics["fit_train"]
    prior_fit = metrics["fit_train_design"]["overall"]
    state_ess = np.array([v["kish_ess"] for v in concentration["per_state"].values()])
    state_ratio = np.array(
        [v["kish_ess_ratio"] for v in concentration["per_state"].values()]
    )
    cd_ess = np.array([v["kish_ess"] for v in concentration["per_cd"].values()])
    cd_ratio = np.array([v["kish_ess_ratio"] for v in concentration["per_cd"].values()])
    out = {
        "run_id": spec["run_id"],
        "kernel_head": _head(payload),
        "prior_acs_share": _prior(payload),
        "mass_parametrization": spec["mass_parametrization"],
        "l2_basis": spec["l2_basis"],
        "l2_lambda": float(spec["l2_lambda"]),
        "holdout_fold": spec.get("holdout_fold"),
        "epochs": spec["epochs"],
        "kish_ess": national["kish_ess"],
        "ess_fraction": national["ess_fraction"],
        "top_1pct_share": national["top_1pct_weight_share"],
        "chi_square_distance_from_prior": national["chi_square_distance"],
        "chi_square_distance_from_release_design": metrics["prior"][
            "chi_square_distance_from_release_design"
        ],
        "prior_kish_ess": national["kish_ess_design"],
        "donor_mass_share": concentration["per_spine"]["asec_puf"]["mass_share"],
        "state_ess_median": state["kish_ess"]["median"],
        "state_ess_min": state["kish_ess"]["min"],
        "cd_ess_median": district["kish_ess"]["median"],
        "cd_ess_p10": district["kish_ess"]["p10"],
        "cd_ess_min": district["kish_ess"]["min"],
        "prior_cd_ess_median": district["kish_ess_design"]["median"],
        "ma_ess": ma["state"]["kish_ess"],
        "ma_prior_ess": ma["state"]["kish_ess_design"],
        "ma_cd_ess_min": float(min(ma_districts)),
        "ma_cd_ess_median": float(np.median(ma_districts)),
        "final_loss": payload["result"]["final_loss"],
        "within_10pct": fit["overall"]["fraction_within_10pct"],
        "mean_capped_error": fit["overall"]["mean_capped_scaled_error"],
        "prior_within_10pct": prior_fit["fraction_within_10pct"],
        "prior_mean_capped_error": prior_fit["mean_capped_scaled_error"],
        "n_states": int(state_ess.size),
        "n_cds": int(cd_ess.size),
    }
    for floor in STATE_FLOORS:
        out[f"states_below_{floor}"] = int((state_ess < floor).sum())
    for floor in CD_FLOORS:
        out[f"cds_below_{floor}"] = int((cd_ess < floor).sum())
    for floor in RELATIVE_FLOORS:
        out[f"states_below_{floor:g}_of_prior"] = int((state_ratio < floor).sum())
        out[f"cds_below_{floor:g}_of_prior"] = int((cd_ratio < floor).sum())
    for family in FAMILIES:
        block = fit["per_family"].get(family, {})
        out[f"within_10pct_{family}"] = block.get("fraction_within_10pct")
        out[f"mean_abs_rel_error_{family}"] = block.get("mean_abs_rel_error")
    if "fit_holdout" in metrics:
        holdout = metrics["fit_holdout"]
        prior_holdout = metrics["fit_holdout_design"]
        out["holdout_within_10pct"] = holdout["overall"]["fraction_within_10pct"]
        out["holdout_mean_capped_error"] = holdout["overall"][
            "mean_capped_scaled_error"
        ]
        out["prior_holdout_within_10pct"] = prior_holdout["overall"][
            "fraction_within_10pct"
        ]
        out["prior_holdout_mean_capped_error"] = prior_holdout["overall"][
            "mean_capped_scaled_error"
        ]
        for family in FAMILIES:
            block = holdout["per_family"].get(family, {})
            out[f"holdout_within_10pct_{family}"] = block.get("fraction_within_10pct")
    return out


#: Per-geography fields the committed receipts keep; the full payloads stay in
#: the artifacts directory and on the Modal volume.
GEOGRAPHY_FIELDS = (
    "n",
    "kish_ess",
    "kish_ess_design",
    "kish_ess_ratio",
    "n_records_holding_half_weight",
)


def compact(payload: dict) -> dict:
    """The receipt with per-state and per-district rows cut to GEOGRAPHY_FIELDS."""

    concentration = payload["metrics"]["concentration"]
    for key in ("per_state", "per_cd"):
        concentration[key] = {
            code: {field: row[field] for field in GEOGRAPHY_FIELDS}
            for code, row in concentration[key].items()
        }
    return payload


def load() -> pd.DataFrame:
    for path in sorted(RUNS.glob("*.json")):
        text = path.read_text()
        if text.startswith("{\n"):  # a full, indented payload from collect
            path.write_text(
                json.dumps(compact(json.loads(text)), separators=(",", ":")) + "\n"
            )
    rows = [
        row(json.loads(path.read_text()))
        for path in sorted(RUNS.glob("*.json"))
        if path.stem not in HEAD_PAIRS
    ]
    frame = pd.DataFrame(rows)
    frame["arm"] = [
        "full surface" if pd.isna(fold) else "holdout" for fold in frame.holdout_fold
    ]
    return frame.sort_values(
        ["arm", "mass_parametrization", "l2_basis", "prior_acs_share", "l2_lambda"]
    )


def _fmt(value, spec: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return format(value, spec)


#: The configurations the write-up compares, in order, with a short label.
KEY_CONFIGS = (
    ("release_repro", "Release (reproduced): share 0.5, projection, λ 0"),
    ("proj_chi_s050_0.03", "Share 0.5, projection, chi-square λ 0.03"),
    ("soft_chi_s050_0.03", "Share 0.5, softmax, chi-square λ 0.03"),
    ("proj_chi_s050_0.1", "Share 0.5, projection, chi-square λ 0.1"),
    ("soft_chi_s050_0.1", "Share 0.5, softmax, chi-square λ 0.1"),
    ("soft_chi_s050_1", "Share 0.5, softmax, chi-square λ 1"),
    ("soft_chi_s700_0.01", "Share 0.7, softmax, chi-square λ 0.01"),
    ("soft_chi_s900_0", "Share 0.9, softmax, no penalty"),
    ("soft_chi_s900_0.01", "Share 0.9, softmax, chi-square λ 0.01"),
    ("soft_chi_s964_0", "Share 0.964, softmax, no penalty"),
)


def _holdout_id(run_id: str, fold: int) -> str:
    if run_id == "release_repro":
        run_id = "proj_s050_0"
    return ("hold_" if fold == 0 else f"hold{fold}_") + run_id


def key_config_table(frame: pd.DataFrame) -> str:
    """The write-up's comparison: concentration, training fit and holdout."""

    by_id = frame.set_index("run_id")
    columns = (
        "Configuration",
        "National ESS",
        "Top-1% share",
        "State ESS min",
        "CD ESS median / min",
        "CDs < 50",
        "MA ESS",
        "Train within 10%",
        "Held-out error, fold 0 / 1",
        "Held-out within 10%, fold 0 / 1",
    )
    lines = [
        "| " + " | ".join(columns) + " |",
        "|---|" + "---:|" * (len(columns) - 1),
    ]
    for run_id, label in KEY_CONFIGS:
        if run_id not in by_id.index:
            continue
        r = by_id.loc[run_id]
        held, held_within = [], []
        for fold in (0, 1):
            hid = _holdout_id(run_id, fold)
            found = hid in by_id.index
            held.append(
                f"{by_id.loc[hid, 'holdout_mean_capped_error']:.4f}" if found else "–"
            )
            held_within.append(
                f"{by_id.loc[hid, 'holdout_within_10pct']:.1%}" if found else "–"
            )
        lines.append(
            f"| {label} | {r.kish_ess:,.0f} | {r.top_1pct_share:.1%} | "
            f"{r.state_ess_min:,.0f} | {r.cd_ess_median:,.0f} / {r.cd_ess_min:,.0f} | "
            f"{int(r.cds_below_50)} | {r.ma_ess:,.0f} | {r.within_10pct:.1%} | "
            f"{' / '.join(held)} | {' / '.join(held_within)} |"
        )
    return "\n".join(lines) + "\n"


def floor_table(frame: pd.DataFrame) -> str:
    """States and districts below each candidate ESS floor, per full-surface run."""

    full = frame[frame.arm == "full surface"]
    columns = [
        ("Run", "run_id", "s"),
        ("Within 10%", "within_10pct", ".1%"),
        ("State ESS min", "state_ess_min", ",.0f"),
        *[(f"States < {f:,}", f"states_below_{f}", "d") for f in STATE_FLOORS],
        ("CD ESS min", "cd_ess_min", ",.0f"),
        *[(f"CDs < {f}", f"cds_below_{f}", "d") for f in CD_FLOORS],
        *[
            (f"CDs < {f:.0%} of prior", f"cds_below_{f:g}_of_prior", "d")
            for f in RELATIVE_FLOORS
        ],
    ]
    lines = [
        "| " + " | ".join(name for name, _, _ in columns) + " |",
        "|" + "---:|" * len(columns),
    ]
    for _, r in full.iterrows():
        lines.append(
            "| " + " | ".join(_fmt(r[key], spec) for _, key, spec in columns) + " |"
        )
    return "\n".join(lines) + "\n"


def markdown(frame: pd.DataFrame) -> str:
    sections = []
    full = frame[frame.arm == "full surface"]
    columns = (
        ("Prior ACS share", "prior_acs_share", ".3g"),
        ("λ", "l2_lambda", "g"),
        ("National ESS", "kish_ess", ",.0f"),
        ("Top-1% share", "top_1pct_share", ".1%"),
        ("Donor mass", "donor_mass_share", ".1%"),
        ("χ² from prior", "chi_square_distance_from_prior", ".3f"),
        ("State ESS median", "state_ess_median", ",.0f"),
        ("CD ESS median / min", None, None),
        ("MA ESS", "ma_ess", ",.0f"),
        ("MA CD ESS min", "ma_cd_ess_min", ",.0f"),
        ("Loss", "final_loss", ".4f"),
        ("Within 10%", "within_10pct", ".1%"),
        ("SOI within 10%", "within_10pct_soi", ".1%"),
        ("SNAP", "within_10pct_snap", ".0%"),
        ("Medicaid", "within_10pct_medicaid", ".0%"),
        ("CD pop", "within_10pct_pop_cd", ".0%"),
    )
    for (parametrization, basis), group in full.groupby(
        ["mass_parametrization", "l2_basis"], sort=False
    ):
        lines = [
            f"### `{parametrization}` parametrization, `{basis}` basis",
            "",
            "| " + " | ".join(name for name, _, _ in columns) + " |",
            "|" + "---:|" * len(columns),
        ]
        for _, r in group.iterrows():
            cells = []
            for _name, key, spec in columns:
                if key is None:
                    cells.append(f"{r.cd_ess_median:,.0f} / {r.cd_ess_min:,.0f}")
                else:
                    cells.append(_fmt(r[key], spec))
            lines.append("| " + " | ".join(cells) + " |")
        sections.append("\n".join(lines))
    held = frame[frame.arm == "holdout"].sort_values(
        ["holdout_fold", "prior_acs_share", "mass_parametrization", "l2_lambda"]
    )
    if len(held):
        hold_columns = (
            ("Fold", "holdout_fold", ".0f"),
            ("Parametrization", "mass_parametrization", "s"),
            ("Prior ACS share", "prior_acs_share", ".3g"),
            ("λ", "l2_lambda", "g"),
            ("National ESS", "kish_ess", ",.0f"),
            ("Train within 10%", "within_10pct", ".1%"),
            ("Held-out within 10%", "holdout_within_10pct", ".1%"),
            ("Held-out capped error", "holdout_mean_capped_error", ".4f"),
            ("Held-out SOI within 10%", "holdout_within_10pct_soi", ".1%"),
            ("Held-out CD pop within 10%", "holdout_within_10pct_pop_cd", ".1%"),
            ("Prior: held-out within 10%", "prior_holdout_within_10pct", ".1%"),
        )
        lines = [
            "### Rotated holdout (each fold is 20% of targets, never seen by the solve)",
            "",
            "| " + " | ".join(name for name, _, _ in hold_columns) + " |",
            "|" + "---:|" * len(hold_columns),
        ]
        for _, r in held.iterrows():
            lines.append(
                "| "
                + " | ".join(_fmt(r[key], spec) for _, key, spec in hold_columns)
                + " |"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def chart(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    full = frame[frame.arm == "full surface"]
    softmax = full[
        (full.mass_parametrization == "softmax") & (full.l2_basis == "chi_square")
    ]
    release = full[
        (full.mass_parametrization == "projection")
        & (full.l2_lambda == 0.0)
        & (full.prior_acs_share == 0.5)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=surface)
    panels = (
        ("kish_ess", "National Kish ESS (log scale)"),
        ("cd_ess_median", "Median district Kish ESS (log scale)"),
    )
    for ax, (key, label) in zip(axes, panels, strict=True):
        ax.set_facecolor(surface)
        for prior, group in softmax.groupby("prior_acs_share"):
            group = group.sort_values("l2_lambda")
            color = PRIOR_COLORS.get(prior, muted)
            ax.plot(
                group.within_10pct,
                group[key],
                color=color,
                linewidth=2,
                marker="o",
                markersize=8,
                markeredgecolor=surface,
                markeredgewidth=2,
                label=f"ACS share {PRIOR_LABELS.get(prior, prior)}",
            )
            # Label only each line's ends; lambda rises right to left.
            for r in (group.iloc[0], group.iloc[-1]):
                ax.annotate(
                    f"λ={r.l2_lambda:g}",
                    (r.within_10pct, r[key]),
                    textcoords="offset points",
                    xytext=(6, -12) if r.l2_lambda == 0 else (-8, 8),
                    ha="left" if r.l2_lambda == 0 else "right",
                    fontsize=8,
                    color=muted,
                )
        if len(release):
            r = release.iloc[0]
            ax.scatter(
                [r.within_10pct],
                [r[key]],
                s=90,
                marker="D",
                color=ink,
                edgecolor=surface,
                linewidth=2,
                zorder=5,
                label="Published release (projection, λ=0)",
            )
        ax.set_yscale("log")
        ax.set_xlabel("Training targets within 10%", color=muted)
        ax.set_ylabel(label, color=muted)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(True, color=grid, linewidth=0.8)
        ax.tick_params(colors=muted)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=ink)
    fig.suptitle(
        "ACS local release: ESS vs fit under a chi-square penalty toward each "
        "prior (softmax mass, 800 epochs; λ = 0, 0.01, 0.03, 0.1, 0.3, 1, 3 "
        "from right to left)",
        color=ink,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)


def rerun_variation() -> dict:
    """Each rerun against its original: bytes, divergence, metric changes."""

    pairs = {}
    for dup, original in HEAD_PAIRS.items():
        paths = [LOCAL_RUNS / run / "weights.npz" for run in (dup, original)]
        receipts = [RUNS / f"{run}.json" for run in (dup, original)]
        if not all(path.exists() for path in (*paths, *receipts)):
            pairs[dup] = {"original": original, "status": "not collected"}
            continue
        payloads = [json.loads(path.read_text()) for path in receipts]
        heads = [_head(payload) for payload in payloads]
        rows = [row(payload) for payload in payloads]
        arrays = [np.load(path) for path in paths]
        weights = [a["weights"] for a in arrays]
        trajectories = [
            np.concatenate(
                [a[k] for k in sorted(a.files) if k.startswith("trajectory_")]
            )
            for a in arrays
        ]
        pairs[dup] = {
            "original": original,
            "heads": {"rerun": heads[0], "original": heads[1]},
            "weights_sha256": [
                hashlib.sha256(np.ascontiguousarray(w).tobytes()).hexdigest()
                for w in weights
            ],
            "weights_byte_identical": bool(
                weights[0].dtype == weights[1].dtype
                and weights[0].tobytes() == weights[1].tobytes()
            ),
            "trajectory_byte_identical": bool(
                trajectories[0].tobytes() == trajectories[1].tobytes()
            ),
            "first_trajectory_difference_epoch": (
                int(np.argmax(trajectories[0] != trajectories[1]))
                if (trajectories[0] != trajectories[1]).any()
                else None
            ),
            "trajectory_relative_difference_at_epochs_0_1": [
                float(abs(a - b) / abs(b))
                for a, b in zip(trajectories[0][:2], trajectories[1][:2], strict=True)
            ],
            "weight_relative_difference_median_p99": [
                float(q)
                for q in np.quantile(
                    np.abs(weights[0] - weights[1]) / weights[1], (0.5, 0.99)
                )
            ],
            "metrics": {
                name: {
                    "rerun": float(rows[0][name]),
                    "original": float(rows[1][name]),
                    "relative_change": (
                        float(rows[0][name] / rows[1][name] - 1.0)
                        if rows[1][name]
                        else None
                    ),
                }
                for name in RERUN_METRICS
            },
        }
    return pairs


def main() -> None:
    frame = load()
    out = HERE / "results"
    frame.to_csv(out / "frontier.csv", index=False)
    (out / "frontier.md").write_text(markdown(frame))
    (out / "floors.md").write_text(floor_table(frame))
    table = key_config_table(frame)
    (out / "key_configs.md").write_text(table)
    (out / "rerun_variation.json").write_text(
        json.dumps(rerun_variation(), indent=1) + "\n"
    )
    readme = HERE / "README.md"
    start, end = (
        "<!-- key-configs:start (written by analyze.py) -->",
        "<!-- key-configs:end -->",
    )
    text = readme.read_text()
    head, _, rest = text.partition(start)
    _, _, tail = rest.partition(end)
    readme.write_text(f"{head}{start}\n{table}{end}{tail}")
    chart(frame, out / "frontier.png")
    print(markdown(frame))


if __name__ == "__main__":
    main()
