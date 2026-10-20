"""Run the microcosm#1326 analysis on one released UK national file.

    python -m tools.uk_scotland_property_income \\
        --h5 <run>/microcosm_uk_2024_25.h5 \\
        --registry <run>/national_target_registry.json \\
        --out <private directory outside the repository>

The output directory holds three kinds of file:

- ``private/``: the per-band audit (rows, exact sources, shares), the reweight
  report and the drift on every SPI 2023-24 fact. Licensed working files;
  never shared.
- ``restricted/``: both releases in full, with exact record counts. Only for
  recipients whose licence covers the FRS and the SPI tape.
- ``public/``: both releases with bands under the floor withheld and flagged,
  and secondary suppression across the two.

Everything is derived from UK Data Service licensed microdata, so the tool
refuses an output directory inside a git working tree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from tools.uk_scotland_property_income import (
    backcast,
    measures,
    rake,
    registry,
    release,
)
from tools.uk_scotland_property_income.bands import (
    INTENTIONAL_EDGES,
    PERSONAL_ALLOWANCE_2023,
    REQUESTED_EDGES,
    SCOTTISH_RATE_STARTS_2023,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
AMOUNTS = ("taxable_property_income", "property_income")
#: Pounds of float32 rounding allowed in a non-Scottish taxpayer's tax.
OUTSIDE_SCOTLAND_TOLERANCE = 0.05
ENGINE_VARIABLES = (
    "person_id",
    "pays_scottish_income_tax",
    "income_tax",
    "property_income",
    "taxable_property_income",
    "taxable_property_income_after_allowances",
    "net_income",
    "taxable_savings_interest_income",
    "taxable_dividend_income",
    "total_income",
    "self_employment_income",
    "loss_relief",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _refuse_working_tree(out: Path) -> None:
    probe = out.resolve()
    while not probe.exists():
        probe = probe.parent
    inside = subprocess.run(
        ["git", "-C", str(probe), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
    )
    if inside.returncode == 0 and inside.stdout.strip() == "true":
        raise SystemExit(
            f"{out} is inside a git working tree; write licensed outputs elsewhere."
        )


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(
            git("status", "--porcelain", "--", "tools/uk_scotland_property_income")
        ),
    }


def _text(values: pd.Series) -> np.ndarray:
    return values.map(
        lambda value: value.decode() if isinstance(value, bytes) else str(value)
    ).to_numpy()


def person_frame(simulation: Any, dataset: Any, year: int) -> pd.DataFrame:
    """The person columns every step reads, at ``year``."""

    backcast.refuse_earlier_period(dataset, year)
    values = {
        name: np.asarray(simulation.calculate(name, year), dtype=float)
        for name in ENGINE_VARIABLES
    }
    person = dataset.person
    if not np.array_equal(
        values["person_id"], person["person_id"].to_numpy(dtype=float)
    ):
        raise ValueError("The engine's person order differs from the dataset's.")
    household = dataset.household
    position = pd.Series(
        np.arange(len(household)), index=household["household_id"].to_numpy()
    )
    household_row = position.loc[person["person_household_id"].to_numpy()].to_numpy()
    frame = pd.DataFrame(
        {name: values[name] for name in ENGINE_VARIABLES if name != "person_id"}
    )
    frame.insert(0, "household_row", household_row)
    frame.insert(1, "region", _text(household["region"])[household_row])
    frame["scottish"] = frame.pop("pays_scottish_income_tax") > 0
    frame["nsnd"] = (
        frame["net_income"]
        - frame["taxable_savings_interest_income"]
        - frame["taxable_dividend_income"]
    )
    frame["total_income_after_loss_relief"] = measures.total_income_after_loss_relief(
        frame["total_income"], frame["self_employment_income"], frame["loss_relief"]
    )
    return frame


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from policyengine_uk import CountryTaxBenefitSystem, Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    from tools.uk_scotland_property_income.reform import scottish_property_rate_reform

    out = Path(args.out)
    _refuse_working_tree(out)
    h5, registry_path = Path(args.h5), Path(args.registry)
    h5_sha = _sha256(h5)
    if args.expect_sha256 and not h5_sha.startswith(args.expect_sha256):
        raise SystemExit(f"{h5} has sha256 {h5_sha}, not {args.expect_sha256}.")

    released = UKSingleYearDataset(file_path=str(h5))
    if int(released.time_period) != args.year + 1:
        raise SystemExit(
            f"{h5} is for {released.time_period}; this back-casts one year to {args.year}."
        )
    sources = release.source_keys(released.person)
    parameters = CountryTaxBenefitSystem().parameters
    earlier, divided = backcast.backcast_one_year(released, parameters)
    w0 = earlier.household["household_weight"].to_numpy(dtype=float)
    persons = person_frame(
        Microsimulation(dataset=backcast.with_household_weights(earlier, w0)),
        earlier,
        args.year,
    )
    persons["source"] = sources

    # A4: reweight to the SPI 2023-24 facts, holding each region's households.
    facts = registry.spi_2023_facts(registry.load_registry(registry_path))
    raked_facts = [fact for fact in facts if fact.region is None] + [
        fact
        for fact in facts
        if fact.region == "SCOTLAND" and fact.measure in {"taxpayers", "total_income"}
    ]
    household_regions = _text(earlier.household["region"])
    regions = sorted(set(household_regions))
    n_households = len(w0)
    matrix = sparse.hstack(
        [
            measures.household_matrix(persons, raked_facts, n_households),
            measures.region_matrix(regions, household_regions),
        ],
        format="csr",
    )
    targets = np.concatenate(
        [
            [fact.value for fact in raked_facts],
            [w0[household_regions == region].sum() for region in regions],
        ]
    )
    raked = rake.rake(
        matrix, w0, targets, lower=args.ratio_bounds[0], upper=args.ratio_bounds[1]
    )
    w1 = raked.weights
    all_matrix = measures.household_matrix(persons, facts, n_households)
    raked_names = {fact.name for fact in raked_facts}
    drift = pd.DataFrame(
        {
            "name": [fact.name for fact in facts],
            "measure": [fact.measure for fact in facts],
            "region": [fact.region or "UK" for fact in facts],
            "lower": [fact.lower for fact in facts],
            "upper": [fact.upper for fact in facts],
            "published_2023_24": [fact.value for fact in facts],
            "backcast": all_matrix.T @ w0,
            "reweighted": all_matrix.T @ w1,
            "reweighted_to": [fact.name in raked_names for fact in facts],
        }
    )
    drift["backcast_error"] = drift["backcast"] / drift["published_2023_24"] - 1
    drift["reweighted_error"] = drift["reweighted"] / drift["published_2023_24"] - 1

    # A6: the two releases.
    persons["w"] = w1[persons["household_row"].to_numpy()]
    cells = persons[
        persons["scottish"]
        & (persons["income_tax"] > 0)
        & (persons["property_income"] > 0)
        & (persons["nsnd"] >= PERSONAL_ALLOWANCE_2023)
    ]
    tables = {}
    for key, edges in (
        ("release_1", REQUESTED_EDGES),
        ("release_2", INTENTIONAL_EDGES),
    ):
        tables[key] = (
            release.audit(cells, edges, AMOUNTS),
            release.bootstrap(
                cells, edges, AMOUNTS, replicates=args.replicates, seed=args.seed
            ),
        )
    statuses = dict(
        zip(
            ("release_1", "release_2"),
            release.disclosure_statuses(
                cells, REQUESTED_EDGES, INTENTIONAL_EDGES, AMOUNTS
            ),
            strict=True,
        )
    )
    for key, (table, intervals) in tables.items():
        audit = table.merge(intervals, on="band").assign(status=statuses[key])
        _write(audit, out / "private" / f"audit_{key}.csv")
        for mode in ("restricted", "public"):
            _write(
                release.render(table, intervals, statuses[key], AMOUNTS, mode=mode),
                out / mode / f"{key}.csv",
            )

    # B: the illustrative 1p, with and without the finance-cost reduction following.
    scottish = persons["scottish"].to_numpy()
    costing = {}
    for variant, relief_follows in (("rates_only", False), ("relief_follows", True)):
        reformed = Microsimulation(
            dataset=backcast.with_household_weights(earlier, w0),
            reform=scottish_property_rate_reform(
                args.delta, relief_follows=relief_follows
            ),
        )
        change = (
            np.asarray(reformed.calculate("income_tax", args.year), dtype=float)
            - persons["income_tax"].to_numpy()
        )
        # The engine stores float32, and the relief variant's rate is a float64
        # array, so tax outside Scotland can move by a rounding penny.
        if np.abs(change[~scottish]).max(initial=0.0) > OUTSIDE_SCOTLAND_TOLERANCE:
            raise ValueError("The Scottish reform changed tax outside Scotland.")
        change = np.where(scottish, change, 0.0)
        weighted = persons["w"].to_numpy() * change
        summary = release.bootstrap_total(
            weighted[scottish],
            persons.loc[scottish, "source"],
            replicates=args.replicates,
            seed=args.seed,
        )
        affected = scottish & (change > 0)
        summary["taxpayers_affected"] = float(persons.loc[affected, "w"].sum())
        summary["sources_affected"] = int(persons.loc[affected, "source"].nunique())
        costing[variant] = summary
    _write_costing(costing, out)

    manifest = {
        "issue": "PolicyEngine/microcosm#1326",
        "year": args.year,
        "inputs": {
            "h5": {"path": str(h5.resolve()), "sha256": h5_sha},
            "registry": {
                "path": str(registry_path.resolve()),
                "sha256": _sha256(registry_path),
            },
        },
        "engine": {
            name: importlib.metadata.version(name)
            for name in ("policyengine-uk", "policyengine-core")
        },
        "tool": _git_state(),
        "backcast": {"divided": divided, "not_inverted": list(backcast.NOT_INVERTED)},
        "reweight": {
            "facts": [fact.name for fact in raked_facts],
            "region_controls": regions,
            "ratio_bounds": list(args.ratio_bounds),
            "converged": raked.converged,
            "message": raked.message,
            "max_abs_error_before": float(np.abs(raked.relative_error_before).max()),
            "max_abs_error_after": float(np.abs(raked.relative_error_after).max()),
            "ratio_quantiles": dict(
                zip(
                    ("min", "p01", "median", "p99", "max"),
                    np.quantile(raked.ratio, [0, 0.01, 0.5, 0.99, 1]).tolist(),
                    strict=True,
                )
            ),
            "share_at_bound": float(
                np.mean(
                    (raked.ratio <= args.ratio_bounds[0] * 1.0001)
                    | (raked.ratio >= args.ratio_bounds[1] * 0.9999)
                )
            ),
            "effective_sample_size_ratio": raked.effective_sample_size_ratio,
        },
        "bands": {
            "release_1": list(REQUESTED_EDGES),
            "release_2": list(INTENTIONAL_EDGES),
            "scottish_rate_starts_2023": SCOTTISH_RATE_STARTS_2023,
        },
        "bootstrap": {
            "replicates": args.replicates,
            "seed": args.seed,
            "weights": "fixed",
        },
        "delta": args.delta,
    }
    (out / "private").mkdir(parents=True, exist_ok=True)
    drift.to_csv(out / "private" / "drift.csv", index=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _write_costing(costing: dict[str, dict[str, float]], out: Path) -> None:
    (out / "private").mkdir(parents=True, exist_ok=True)
    (out / "private" / "costing.json").write_text(json.dumps(costing, indent=2) + "\n")
    rows = []
    for variant, summary in costing.items():
        published = (
            summary["sources"] >= release.MIN_SOURCES
            and summary["top_source_share"] <= release.DOMINANCE_SHARE
        )
        rows.append(
            {
                "variant": variant,
                "change_in_scottish_income_tax": round(summary["estimate"] / 1e5) * 1e5,
                "low": round(summary["low"] / 1e5) * 1e5,
                "high": round(summary["high"] / 1e5) * 1e5,
                "cv": round(summary["cv"], 2),
                "taxpayers_affected": round(summary["taxpayers_affected"] / 100) * 100,
                "status": release.PUBLISHED if published else release.DOMINATED,
            }
        )
    table = pd.DataFrame(rows)
    _write(table, out / "restricted" / "costing.csv")
    public = table.copy()
    withheld = public["status"] != release.PUBLISHED
    public.loc[
        withheld,
        ["change_in_scottish_income_tax", "low", "high", "cv", "taxpayers_affected"],
    ] = None
    _write(public, out / "public" / "costing.csv")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--h5", required=True, help="The released 2024-25 national file."
    )
    parser.add_argument(
        "--registry", required=True, help="The build's national_target_registry.json."
    )
    parser.add_argument(
        "--out", required=True, help="A private directory outside any git working tree."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2023,
        help="The year to back-cast to (default 2023).",
    )
    parser.add_argument(
        "--expect-sha256", help="Refuse an H5 whose sha256 doesn't start with this."
    )
    parser.add_argument("--ratio-bounds", type=float, nargs=2, default=(0.25, 4.0))
    parser.add_argument("--replicates", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--delta",
        type=float,
        default=0.01,
        help="The change in each Scottish rate (default 1p).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    manifest = run(parse_args())
    reweight = manifest["reweight"]
    print(
        f"done: reweight converged={reweight['converged']}, "
        f"max |error| {reweight['max_abs_error_before']:.3f} -> {reweight['max_abs_error_after']:.4f}"
    )
