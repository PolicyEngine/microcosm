"""Build the ESS-versus-fit frontier from the sweep's metrics.

Reads ``results/runs/*.json`` (one ``sweep.py`` payload per run) and writes
``results/frontier.csv``, ``results/frontier.md`` and
``results/frontier.png``.

Run: ``uv run --with matplotlib python experiments/us-acs-local-l2-basis-20260928/analyze.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE / "results" / "runs"
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


def load() -> pd.DataFrame:
    rows = [row(json.loads(path.read_text())) for path in sorted(RUNS.glob("*.json"))]
    frame = pd.DataFrame(rows)
    frame["arm"] = np.where(
        frame.holdout_fold.notna(),
        "holdout fold 0",
        "full surface",
    )
    return frame.sort_values(
        ["arm", "mass_parametrization", "l2_basis", "prior_acs_share", "l2_lambda"]
    )


def _fmt(value, spec: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return format(value, spec)


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
    held = frame[frame.arm == "holdout fold 0"]
    if len(held):
        hold_columns = (
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
            "### Rotated holdout, fold 0 (20% of targets never seen by the solve)",
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


def main() -> None:
    frame = load()
    out = HERE / "results"
    frame.to_csv(out / "frontier.csv", index=False)
    (out / "frontier.md").write_text(markdown(frame))
    (out / "floors.md").write_text(floor_table(frame))
    chart(frame, out / "frontier.png")
    print(markdown(frame))


if __name__ == "__main__":
    main()
