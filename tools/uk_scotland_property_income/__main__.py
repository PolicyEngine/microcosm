"""Run the microcosm#1326 analysis on one released UK national file.

    python -m tools.uk_scotland_property_income \\
        --h5 <run>/microcosm_uk_2024_25.h5 \\
        --registry <run>/national_target_registry.json \\
        --out <private directory outside the repository>

The output directory holds three kinds of file:

- ``private/``: the per-band audit (rows, exact records, households, shares,
  before and after the reweight), the reweight report, the drift on every SPI
  2023-24 fact, the check against HMRC's printed tables, the costing detail
  and the run manifest. Licensed working files; never shared.
- ``restricted/``: both releases in full, with exact record counts, the
  costing and a release note. Only for recipients whose licence covers the
  FRS and the SPI tape.
- ``public/``: both releases with withheld cells blank, the costing and a
  release note. Safe to send as a folder.

Everything is derived from UK Data Service licensed microdata, so the tool
refuses an output directory inside a git working tree. That check needs git on
the PATH, and it can't see a cloud-synced folder: choose a local, unsynced
private directory.
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
from packaging.version import Version
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
#: The headline amount first; profit before the property allowance beside it.
AMOUNTS = ("taxable_property_income", "property_income")
#: Pounds of float32 rounding allowed in a tax the reform shouldn't touch.
ROUNDING_TOLERANCE = 0.05
#: The engine the tests passed on. Older ones lack the private uprating map
#: the back-cast reads, and before it ``net_income`` didn't deduct trade-loss
#: relief, which moves the NSND bands.
MINIMUM_ENGINE = Version("2.134.4")
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


def _refuse_old_engine() -> dict[str, str]:
    versions = {
        name: importlib.metadata.version(name)
        for name in ("policyengine-uk", "policyengine-core")
    }
    if Version(versions["policyengine-uk"]) < MINIMUM_ENGINE:
        raise SystemExit(
            f"policyengine-uk {versions['policyengine-uk']} is older than "
            f"{MINIMUM_ENGINE}, the engine this tool was tested on."
        )
    return versions


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
    frame.insert(
        1, "region", release.as_text(household["region"]).to_numpy()[household_row]
    )
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


def _record_keys(released: Any) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Records and bootstrap clusters for every person in the released file."""

    person = released.person
    household = released.household
    key_by_household = pd.Series(
        release.as_text(household["source_household_key"]).to_numpy(),
        index=household["household_id"].to_numpy(),
    )
    household_keys = key_by_household.loc[person["person_household_id"]].to_numpy()
    return (
        release.source_keys(person),
        release.cluster_keys(person, household_keys),
        pd.DataFrame(
            {
                "person_source_id": person["person_source_id"]
                .astype(np.int64)
                .to_numpy(),
                "channel": release.as_text(person["person_support_channel"]).to_numpy(),
            }
        ),
    )


def _reweight(
    persons: pd.DataFrame,
    earlier: Any,
    facts: list[registry.SpiFact],
    args: argparse.Namespace,
) -> tuple[rake.RakeResult, list[registry.SpiFact], list[str], dict[str, float]]:
    raked_facts = [fact for fact in facts if fact.region is None] + [
        fact
        for fact in facts
        if fact.region == "SCOTLAND" and fact.measure in {"taxpayers", "total_income"}
    ]
    w0 = earlier.household["household_weight"].to_numpy(dtype=float)
    household_regions = release.as_text(earlier.household["region"]).to_numpy()
    regions = sorted(set(household_regions))
    matrix = sparse.hstack(
        [
            measures.household_matrix(persons, raked_facts, len(w0)),
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
    errors = np.abs(raked.relative_error_after)
    misses = {
        "facts": float(errors[: len(raked_facts)].max()),
        "region_households": float(errors[len(raked_facts) :].max()),
    }
    if not raked.converged or max(misses.values()) > args.max_miss:
        raise SystemExit(
            f"The reweight didn't settle: converged={raked.converged}, "
            f"largest misses {misses}, allowed {args.max_miss}."
        )
    return raked, raked_facts, regions, misses


def _with_weights(cells: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    return cells.assign(w=weights[cells["household_row"].to_numpy()])


def run(args: argparse.Namespace) -> dict[str, Any]:
    from policyengine_uk import CountryTaxBenefitSystem, Microsimulation
    from policyengine_uk.data import UKSingleYearDataset

    from tools.uk_scotland_property_income.reform import scottish_property_rate_reform

    out = Path(args.out)
    _refuse_working_tree(out)
    engine = _refuse_old_engine()
    h5, registry_path = Path(args.h5), Path(args.registry)
    h5_sha = _sha256(h5)
    if args.expect_sha256 and not h5_sha.startswith(args.expect_sha256):
        raise SystemExit(f"{h5} has sha256 {h5_sha}, not {args.expect_sha256}.")

    released = UKSingleYearDataset(file_path=str(h5))
    if int(released.time_period) != args.year + 1:
        raise SystemExit(
            f"{h5} is for {released.time_period}; this back-casts one year to {args.year}."
        )
    sources, clusters, identities = _record_keys(released)
    parameters = CountryTaxBenefitSystem().parameters
    earlier, divided = backcast.backcast_one_year(released, parameters)
    w0 = earlier.household["household_weight"].to_numpy(dtype=float)
    persons = person_frame(
        Microsimulation(dataset=backcast.with_household_weights(earlier, w0)),
        earlier,
        args.year,
    )
    persons["source"] = sources
    persons["cluster"] = clusters
    persons["person_source_id"] = identities["person_source_id"]
    persons["channel"] = identities["channel"]

    # A4: reweight to the SPI 2023-24 facts, checked against HMRC's tables.
    facts = registry.spi_2023_facts(registry.load_registry(registry_path))
    raked, raked_facts, regions, misses = _reweight(persons, earlier, facts, args)
    published_check = pd.DataFrame(registry.check_against_published(raked_facts))
    _write(published_check, out / "private" / "published_check.csv")
    if len(published_check) != len(raked_facts) or not published_check["agrees"].all():
        raise SystemExit(
            "Some reweighted facts don't match HMRC's printed SPI 2023-24 tables; "
            "see private/published_check.csv."
        )
    w1 = raked.weights
    all_matrix = measures.household_matrix(persons, facts, len(w0))
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
    _write(drift, out / "private" / "drift.csv")

    # A6: the two releases. The headline counts taxpayers with taxable property
    # income; landlords wholly under the property allowance are counted beside
    # it, in the restricted and private tables only.
    persons["w"] = w1[persons["household_row"].to_numpy()]
    scottish = persons["scottish"].to_numpy()
    landlords = persons[
        persons["scottish"]
        & (persons["income_tax"] > 0)
        & (persons["property_income"] > 0)
        & (persons["nsnd"] >= PERSONAL_ALLOWANCE_2023)
    ]
    cells = landlords[landlords["taxable_property_income"] > 0]
    universe = persons.loc[scottish, "cluster"].to_numpy()
    statuses = dict(
        zip(
            ("release_1", "release_2"),
            release.disclosure_statuses(
                cells, REQUESTED_EDGES, INTENTIONAL_EDGES, AMOUNTS
            ),
            strict=True,
        )
    )
    for key, edges in (
        ("release_1", REQUESTED_EDGES),
        ("release_2", INTENTIONAL_EDGES),
    ):
        table = release.audit(cells, edges, AMOUNTS)
        intervals = release.bootstrap(
            cells,
            edges,
            AMOUNTS,
            universe=universe,
            replicates=args.replicates,
            seed=args.seed,
        )
        before = release.audit(_with_weights(cells, w0), edges, AMOUNTS)
        everyone = release.audit(landlords, edges, ())
        audit = table.merge(intervals, on="band").assign(
            status=statuses[key],
            count_before_reweight=before["count"],
            **{f"{n}_before_reweight": before[n] for n in AMOUNTS},
            all_landlords_count=everyone["count"],
            all_landlords_records=everyone["sources"],
        )
        _write(audit, out / "private" / f"audit_{key}.csv")
        restricted = release.render(
            table, intervals, statuses[key], AMOUNTS, mode="restricted"
        )
        restricted.insert(
            restricted.columns.get_loc("status"),
            "all_landlords_count",
            (everyone["count"] / 100).round() * 100,
        )
        _write(restricted, out / "restricted" / f"{key}.csv")
        _write(
            release.render(table, intervals, statuses[key], AMOUNTS, mode="public"),
            out / "public" / f"{key}.csv",
        )

    # B: the illustrative 1p. The relief variant copies the engine's formula,
    # so check first that at zero it reproduces the engine on this file.
    def tax_change(delta: float, relief_follows: bool) -> np.ndarray:
        reformed = Microsimulation(
            dataset=backcast.with_household_weights(earlier, w0),
            reform=scottish_property_rate_reform(delta, relief_follows=relief_follows),
        )
        return (
            np.asarray(reformed.calculate("income_tax", args.year), dtype=float)
            - persons["income_tax"].to_numpy()
        )

    if np.abs(tax_change(0.0, True)).max() > ROUNDING_TOLERANCE:
        raise SystemExit("The relief variant at zero doesn't reproduce the engine.")
    costing = {}
    for variant, relief_follows, role in (
        ("rates_only", False, "headline"),
        ("relief_follows", True, "sensitivity"),
    ):
        change = tax_change(args.delta, relief_follows)
        if np.abs(change[~scottish]).max(initial=0.0) > ROUNDING_TOLERANCE:
            raise SystemExit("The Scottish reform changed tax outside Scotland.")
        change = np.where(scottish, change, 0.0)
        weighted = persons["w"].to_numpy() * change
        summary = release.bootstrap_total(
            weighted[scottish],
            persons.loc[scottish, "cluster"],
            replicates=args.replicates,
            seed=args.seed,
        )
        shares = release.record_shares(
            weighted[scottish], persons.loc[scottish, "source"]
        )
        affected = scottish & (change > 0)
        summary.update(
            shares,
            role=role,
            taxpayers_affected=float(persons.loc[affected, "w"].sum()),
            status=release.passes_primary(
                shares["records"],
                [shares["top_record_share"]],
                [shares["top_two_record_share"]],
            ),
        )
        costing[variant] = summary
    _write_costing(costing, out)

    spi_channel = (cells["channel"] == "spi") & cells["source"].str.startswith("frs:")
    manifest = {
        "issue": "PolicyEngine/microcosm#1326",
        "year": args.year,
        "inputs": {
            "h5": {
                "path": str(h5.resolve()),
                "sha256": h5_sha,
                "tag": args.release_tag,
            },
            "registry": {
                "path": str(registry_path.resolve()),
                "sha256": _sha256(registry_path),
            },
        },
        "engine": engine,
        "tool": _git_state(),
        "backcast": {
            "divided": divided,
            "not_inverted": list(backcast.NOT_INVERTED),
            "held_at_file_year": backcast.held_at_file_year(released, divided),
        },
        "reweight": {
            "facts": [fact.name for fact in raked_facts],
            "region_controls": regions,
            "ratio_bounds": list(args.ratio_bounds),
            "converged": raked.converged,
            "message": raked.message,
            "max_abs_error_before": float(np.abs(raked.relative_error_before).max()),
            "max_abs_error_after": misses,
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
            "published_check": {
                "facts": int(len(published_check)),
                "agree": int(published_check["agrees"].sum()),
            },
        },
        "records": {
            "headline_rows": int(len(cells)),
            "headline_records": int(cells["source"].nunique()),
            "headline_households_and_tape_records": int(cells["cluster"].nunique()),
            "rows_keyed_by_frs_person": int(spi_channel.sum()),
            "frs_persons_also_hosting_tape_records": int(
                len(
                    set(
                        cells.loc[
                            cells["source"].str.startswith("frs:"), "person_source_id"
                        ]
                    )
                    & set(
                        cells.loc[
                            cells["source"].str.startswith("spi:"), "person_source_id"
                        ]
                    )
                )
            ),
        },
        "bands": {
            "release_1": list(REQUESTED_EDGES),
            "release_2": list(INTENTIONAL_EDGES),
            "scottish_rate_starts_2023": SCOTTISH_RATE_STARTS_2023,
        },
        "bootstrap": {
            "replicates": args.replicates,
            "seed": args.seed,
            "clusters": "FRS households and tape records, over all Scottish persons",
            "weights": "fixed",
        },
        "delta": args.delta,
    }
    (out / "private" / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    for mode in ("restricted", "public"):
        (out / mode / "release_note.md").write_text(release_note(manifest, mode))
    return manifest


def _write_costing(costing: dict[str, dict[str, Any]], out: Path) -> None:
    (out / "private").mkdir(parents=True, exist_ok=True)
    (out / "private" / "costing.json").write_text(json.dumps(costing, indent=2) + "\n")
    rows = []
    for variant, summary in costing.items():
        rows.append(
            {
                "variant": variant,
                "role": summary["role"],
                "change_in_scottish_income_tax": round(summary["estimate"] / 1e5) * 1e5,
                "low": round(summary["low"] / 1e5) * 1e5,
                "high": round(summary["high"] / 1e5) * 1e5,
                "cv": round(summary["cv"], 2),
                "taxpayers_affected": round(summary["taxpayers_affected"] / 100) * 100,
                "status": summary["status"],
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


def release_note(manifest: dict[str, Any], mode: str) -> str:
    """The note that goes with the tables. It states definitions, method and
    caveats, and holds no figure from the tables."""

    engine = manifest["engine"]["policyengine-uk"]
    source = manifest["inputs"]["h5"]
    file_label = source["tag"] or f"sha256 {source['sha256'][:8]}…"
    region_miss = manifest["reweight"]["max_abs_error_after"]["region_households"]
    starts = manifest["bands"]["scottish_rate_starts_2023"]
    version = (
        "**Restricted version.** Every band is shown, with exact record counts. "
        "It is only for recipients covered by the UK Data Service licence for "
        "the Family Resources Survey and the SPI public use tape. Don't publish it."
        if mode == "restricted"
        else "**Public version.** Cells that fail the disclosure rules below are "
        "blank, with no record count."
    )
    all_landlords = (
        " The restricted tables also count them, in `all_landlords_count`."
        if mode == "restricted"
        else ""
    )
    return f"""# Scottish taxpayers with property income, by NSND income band, 2023-24

Prepared by PolicyEngine from the Microcosm UK national file {file_label}
(PolicyEngine/microcosm#1326), with policyengine-uk {engine}.

{version}

## What the tables count

- **Taxpayers:** people who pay Scottish income tax and have a positive income
  tax liability in 2023-24, after tax reductions such as the finance-cost
  reduction.
- **Property income:** the profit of a property business after allowable
  expenses and before finance costs, the concept the SPI uses. The headline
  amount, `taxable_property_income`, is after the £1,000 property allowance.
  `property_income`, before the allowance, is shown beside it.
- **Headline count:** taxpayers with taxable property income above zero.
  Landlords whose whole profit is under the property allowance are left out,
  since most of them don't report it to HMRC.{all_landlords}
- **Bands:** individual non-savings, non-dividend (NSND) income before
  allowances, after trade-loss relief and the property allowance. "PA" means
  £12,570 for everyone, so marriage allowance transferors and blind person's
  allowance holders are banded on £12,570 too. Taxpayers with NSND income
  below £12,570 are not in the tables.
- **Release 1** keeps the ten requested bands. **Release 2** uses four unions
  of them. Each edge is the requested edge nearest a 2023-24 Scottish rate
  start: basic £{starts["basic"]:,}, intermediate £{starts["intermediate"]:,}
  and higher £{starts["higher"]:,}. The bins are PA–£24,999 (starter and
  basic), £25,000–£39,999 (intermediate), £40,000–£89,999 (higher) and
  £90,000+ (higher and top).

## How the 2023-24 figures are made

- **Base:** the 2024-25 Microcosm file, back-cast one year. Each input is
  divided by the growth index policyengine-uk uses to project it forward:
  OBR national aggregates per person, and ONS population for the weights.
  Inputs the engine doesn't uprate, such as Gift Aid and losses brought
  forward, stay at their 2024-25 levels.
- **Reweight:** the back-cast weights are reweighted to the SPI 2023-24 tables
  HMRC published. The targets are UK property income by range of total income
  (counts and amounts, Table 3.7) and Scotland's taxpayers and total income by
  range (Table 3.11). Each region's household total is held to within
  {region_miss:.2%}. Because those figures are calibrated, they don't test the
  data. The split of Scottish property income across NSND bands is not
  targeted, so that split is what the outturn tests.
- **Law:** 2023-24 income tax.

## Uncertainty

Each band shows:
- the number of distinct survey records behind it;
- its effective number of records;
- a 90% interval, from resampling FRS households and SPI records across all
  Scottish taxpayers.

The weights are held fixed, so the intervals leave out calibration error and
are too narrow.

## The 1p costing

- **What it adds:** 1p on every Scottish rate charged on property income,
  meaning the slice of property income above the allowances, with the same
  band limits as the other NSND rates.
- **Basis:** static, on 2023-24 data and law, with no behavioural response.
- **Headline and sensitivity:** the headline keeps the finance-cost reduction
  at the basic rate. The sensitivity raises it by 1p as well.
- **Later years:** a 2027-28 figure, when the power applies, needs a separate
  run.

## Disclosure rules

A cell is shown only if:
- at least 10 distinct records sit behind it;
- no single record is over 50% of its weight or amounts;
- no two records together are over 85% of them.

Further cells are withheld so that none of these can be rebuilt by
subtraction. That includes subtraction from the costing, which reveals the
total.
"""


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
    parser.add_argument(
        "--release-tag", help="The release's cut tag, for the release note."
    )
    parser.add_argument("--ratio-bounds", type=float, nargs=2, default=(0.25, 4.0))
    parser.add_argument(
        "--max-miss",
        type=float,
        default=0.005,
        help="Refuse a reweight that misses any fact or region total by more (default 0.5%%).",
    )
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
        f"done: reweight converged={reweight['converged']}, largest miss "
        f"{reweight['max_abs_error_before']:.3f} -> {reweight['max_abs_error_after']}"
    )
