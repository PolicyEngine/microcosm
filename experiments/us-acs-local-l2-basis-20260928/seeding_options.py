"""Starting-weight concentration of the ACS local file under other seedings.

Before any calibration, how concentrated are the starting weights if the
staging seeds them differently? Two levers, alone and together:

``acs_share``
    The ACS rows' share of the household mass (``--acs-share`` in
    ``tools/_legacy/build_us_acs_multispine_base.py``, applied through
    ``base_pool._pooled_household_weights``; the release used 0.5). Each spine
    keeps its internal proportions (``sweep.reseed_prior``). 0.964 is the
    share proportional to record counts (1,531,614 of 1,588,854).
``k``
    Location clones of each donor (ASEC-by-PUF) record: ``k`` copies at
    weight/``k``, as location v1's draw supports (PolicyEngine/microcosm#1047;
    every line refuses k > 1 today). v1 draws a clone's block within the
    record's finest source geography, its CPS county when one is coded, else
    its state. This checkpoint carries no donor county, so here each clone's
    congressional district is drawn within the record's state with probability
    proportional to the ACS rows' starting mass in that district (a household
    count proxy). That is the state-only case, the widest spread v1 allows, so
    the per-district figures are an upper bound for donors with a coded county.

Clones of one record are not independent observations. Every figure is
therefore given twice: Kish ESS over rows, and over distinct source households
within the group (a record's clones summed first), which is at most the row
figure. Nationally and by state, cloning leaves the distinct figure unchanged,
since all of a record's clones stay in its state.

Run: ``uv run python experiments/us-acs-local-l2-basis-20260928/seeding_options.py``
Writes ``results/seeding_options.json`` and ``results/seeding_options.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sweep import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    MASSACHUSETTS,
    SPINES,
    kish,
    reseed_prior,
    top_share,
)

ACS_SHARES = (0.5, 0.7, 0.9, None)  # None: proportional to record counts
CLONES = (1, 4, 16)
SEED = 20260930


def grouped_kish(weights: np.ndarray, groups: np.ndarray) -> pd.Series:
    frame = pd.DataFrame({"g": groups, "w": weights, "w2": weights**2})
    sums = frame.groupby("g", observed=True)[["w", "w2"]].sum()
    return sums.w**2 / sums.w2


def distinct_kish(
    weights: np.ndarray, groups: np.ndarray, source: np.ndarray
) -> pd.Series:
    """Kish ESS per group after summing each source household's rows."""

    frame = pd.DataFrame({"g": groups, "s": source, "w": weights})
    per_source = frame.groupby(["g", "s"], observed=True).w.sum().reset_index()
    per_source["w2"] = per_source.w**2
    sums = per_source.groupby("g", observed=True)[["w", "w2"]].sum()
    return sums.w**2 / sums.w2


def summary(values: pd.Series) -> dict:
    return {
        "min": float(values.min()),
        "p10": float(values.quantile(0.10)),
        "median": float(values.median()),
        "max": float(values.max()),
    }


def cloned_rows(
    households: pd.DataFrame, prior: np.ndarray, k: int, rng: np.random.Generator
) -> pd.DataFrame:
    """ACS rows as they are; each donor row split into k clones at weight/k."""

    spine = households["spine"].astype(str).to_numpy()
    state = households["state_fips"].astype(str).to_numpy()
    district = households["congressional_district_geoid"].astype(str).to_numpy()
    acs = spine == SPINES[0]
    rows = pd.DataFrame(
        {
            "source": households["household_id"].to_numpy(),
            "state": state,
            "district": district,
            "spine": spine,
            "weight": prior,
        }
    )
    if k == 1:
        return rows
    # Placement probabilities: ACS starting mass by district within each state.
    acs_mass = (
        pd.DataFrame({"state": state[acs], "district": district[acs], "w": prior[acs]})
        .groupby(["state", "district"], observed=True)
        .w.sum()
    )
    donors = rows[~acs]
    parts = [rows[acs]]
    for state_code, group in donors.groupby("state", observed=True):
        options = acs_mass.loc[state_code]
        probabilities = (options / options.sum()).to_numpy()
        draws = rng.choice(len(options), size=(len(group), k), p=probabilities)
        parts.append(
            pd.DataFrame(
                {
                    "source": np.repeat(group["source"].to_numpy(), k),
                    "state": state_code,
                    "district": options.index.to_numpy()[draws.ravel()],
                    "spine": SPINES[1],
                    "weight": np.repeat(group["weight"].to_numpy() / k, k),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def measure(rows: pd.DataFrame) -> dict:
    weights = rows["weight"].to_numpy()
    source = rows["source"].to_numpy()
    donor = rows["spine"].to_numpy() == SPINES[1]
    national_distinct = kish(
        rows.groupby("source", observed=True).weight.sum().to_numpy()
    )
    by_state_rows = grouped_kish(weights, rows["state"].to_numpy())
    by_state_distinct = distinct_kish(weights, rows["state"].to_numpy(), source)
    by_cd_rows = grouped_kish(weights, rows["district"].to_numpy())
    by_cd_distinct = distinct_kish(weights, rows["district"].to_numpy(), source)
    ma_cds = sorted(c for c in by_cd_rows.index if str(c).startswith(MASSACHUSETTS))
    return {
        "rows": int(len(rows)),
        "donor_mass_share": float(weights[donor].sum() / weights.sum()),
        "national": {
            "kish_ess_rows": kish(weights),
            "kish_ess_distinct": national_distinct,
            "top_1pct_share_rows": top_share(weights),
        },
        "state": {
            "kish_ess_rows": summary(by_state_rows),
            "kish_ess_distinct": summary(by_state_distinct),
        },
        "district": {
            "kish_ess_rows": summary(by_cd_rows),
            "kish_ess_distinct": summary(by_cd_distinct),
        },
        "massachusetts": {
            "kish_ess_rows": float(by_state_rows[MASSACHUSETTS]),
            "kish_ess_distinct": float(by_state_distinct[MASSACHUSETTS]),
            "district_kish_ess_rows": {c: float(by_cd_rows[c]) for c in ma_cds},
            "district_kish_ess_distinct": {c: float(by_cd_distinct[c]) for c in ma_cds},
        },
    }


def main() -> None:
    households = pd.read_parquet(DEFAULT_CHECKPOINT / "households.parquet")
    design = households["design_weight"].to_numpy(np.float64)
    spine = households["spine"].astype(str).to_numpy()
    proportional = float((spine == SPINES[0]).mean())
    results = []
    for share in ACS_SHARES:
        acs_share = proportional if share is None else share
        prior = reseed_prior(design, spine, acs_share)
        for k in CLONES:
            rng = np.random.default_rng([SEED, k, int(round(acs_share * 1e6))])
            rows = cloned_rows(households, prior, k, rng)
            entry = {
                "acs_share": acs_share,
                "acs_share_label": "proportional" if share is None else str(share),
                "k": k,
                **measure(rows),
            }
            results.append(entry)
            print(
                f"acs_share={acs_share:.3f} k={k}: national ESS "
                f"{entry['national']['kish_ess_rows']:,.0f} rows / "
                f"{entry['national']['kish_ess_distinct']:,.0f} distinct; "
                f"MA {entry['massachusetts']['kish_ess_distinct']:,.0f}; "
                f"CD median {entry['district']['kish_ess_distinct']['median']:,.0f}",
                flush=True,
            )
    acs_only = measure(
        cloned_rows(
            households[spine == SPINES[0]].reset_index(drop=True),
            design[spine == SPINES[0]],
            1,
            np.random.default_rng(SEED),
        )
    )
    # The release design's Massachusetts split by spine, which the doc quotes.
    ma = households["state_fips"].astype(str).to_numpy() == MASSACHUSETTS
    districts = households["congressional_district_geoid"].astype(str).to_numpy()
    ma_by_spine = {}
    for label in SPINES:
        rows = ma & (spine == label)
        ma_by_spine[label] = {
            "n": int(rows.sum()),
            "design_mass_share_of_state": float(design[rows].sum() / design[ma].sum()),
            "kish_ess": kish(design[rows]),
            "district_kish_ess": {
                code: kish(design[rows & (districts == code)])
                for code in sorted(set(districts[ma]))
            },
        }
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    payload = {
        "massachusetts_by_spine_at_release_design": ma_by_spine,
        "checkpoint": str(DEFAULT_CHECKPOINT),
        "seed": SEED,
        "placement": (
            "donor clones: district drawn within the record's state with "
            "probability proportional to ACS starting mass in the district "
            "(state-only case of location v1; an upper bound on spread)"
        ),
        "proportional_acs_share": proportional,
        "options": results,
        "acs_rows_only_at_release_design": acs_only,
    }
    (out / "seeding_options.json").write_text(json.dumps(payload, indent=1) + "\n")

    lines = [
        "| ACS share | Donor clones k | National ESS (rows / distinct) "
        "| Top-1% share | State ESS median (distinct) | District ESS median, min "
        "(distinct) | Massachusetts ESS (distinct) | MA district ESS range "
        "(distinct) |",
        "|---|---:|---|---:|---:|---|---:|---|",
    ]
    for entry in results:
        ma = entry["massachusetts"]["district_kish_ess_distinct"].values()
        lines.append(
            f"| {entry['acs_share_label']} | {entry['k']} | "
            f"{entry['national']['kish_ess_rows']:,.0f} / "
            f"{entry['national']['kish_ess_distinct']:,.0f} | "
            f"{entry['national']['top_1pct_share_rows']:.1%} | "
            f"{entry['state']['kish_ess_distinct']['median']:,.0f} | "
            f"{entry['district']['kish_ess_distinct']['median']:,.0f}, "
            f"{entry['district']['kish_ess_distinct']['min']:,.0f} | "
            f"{entry['massachusetts']['kish_ess_distinct']:,.0f} | "
            f"{min(ma):,.0f}-{max(ma):,.0f} |"
        )
    ma_acs = acs_only["massachusetts"]["district_kish_ess_distinct"].values()
    lines.append(
        f"| ACS rows only | - | {acs_only['national']['kish_ess_rows']:,.0f} | "
        f"{acs_only['national']['top_1pct_share_rows']:.1%} | "
        f"{acs_only['state']['kish_ess_distinct']['median']:,.0f} | "
        f"{acs_only['district']['kish_ess_distinct']['median']:,.0f}, "
        f"{acs_only['district']['kish_ess_distinct']['min']:,.0f} | "
        f"{acs_only['massachusetts']['kish_ess_distinct']:,.0f} | "
        f"{min(ma_acs):,.0f}-{max(ma_acs):,.0f} |"
    )
    (out / "seeding_options.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
