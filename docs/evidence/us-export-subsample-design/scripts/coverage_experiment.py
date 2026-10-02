"""How well do the probe's design-based standard errors cover, by sampling design?

Run on a real US export, engine-free. For each reform-coverage smoke probe the
proxy quantity is ``y_h`` = household total of the absolute values of the
probe's binding inputs (booleans as 1): the support and tails of the probe's
effect. Its full-export weighted total ``T = sum w_h y_h`` is known, so for
each design and seed the script

1. builds the certainty set with the sampler's own ``probe_carriers`` and
   ``size_certainty_households``,
2. draws with the sampler's own ``draw_households``,
3. estimates ``T`` with the drawn weights and its variance with the probe's
   own ``stratified_variance_terms``,
4. records ``z = (T_hat - T) / SE`` and the effective number of households the
   variance estimate rests on (``effective_variance_households``).

A draw is *eligible* for an authoritative verdict when the effective count is
at least ``--min-effective`` (or the probe is take-all and no drawn household
carries it). The output reports, per design and probe, the eligible share and
the share of eligible draws with ``|z| > k``: the rate at which the probe
would be confidently wrong.

Binding inputs come from a published ``reform_coverage_smoke.json``, so
neither policyengine-us nor ``microcosm.build.us_runtime`` is imported.

Usage::

    python coverage_experiment.py --export <populace_us_2024.h5> \
        --smoke <reform_coverage_smoke.json> --out coverage.json \
        --design 5:0 --design 30:2 [--seeds 200] [--fraction 0.05]

where each design is ``<certainty threshold>:<size certainty multiplier>``.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

TOOLS = Path(__file__).resolve().parents[4] / "tools"
K_VALUES = (3.0, 4.0, 5.0)


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--design", action="append", required=True)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--min-effective", type=float, default=30.0)
    args = parser.parse_args()
    started = time.time()

    def log(message):
        print(f"[{time.time() - started:7.1f}s] {message}", flush=True)

    sampler = load_tool("sample_us_export_households")
    probe = load_tool("probe_us_post_export")
    results = json.loads(args.smoke.read_text())["reform_coverage_smoke"]["details"][
        "results"
    ]
    probes = [
        SimpleNamespace(id=pid, binding_inputs=tuple(result["binding_inputs"]))
        for pid, result in results.items()
    ]
    leaves = sorted({leaf for p in probes for leaf in p.binding_inputs})

    with pd.HDFStore(str(args.export), mode="r") as store:
        columns = {e: sampler.table_columns(store, e) for e in sampler.US_ENTITIES}
        strata_columns = [
            c for c in sampler.DEFAULT_STRATUM_COLUMNS if c in columns["person"]
        ]
        household = sampler.read_table_rows(
            store,
            "household",
            columns=list(
                dict.fromkeys(
                    [
                        "household_id",
                        sampler.HOUSEHOLD_WEIGHT_COLUMN,
                        *[
                            c
                            for c in sampler.DEFAULT_STRATUM_COLUMNS
                            if c in columns["household"]
                        ],
                        *[c for c in leaves if c in columns["household"]],
                    ]
                )
            ),
        )
        person = sampler.read_table_rows(
            store,
            "person",
            columns=[
                *sampler.PERSON_MEMBERSHIP_COLUMNS,
                *strata_columns,
                *[c for c in leaves if c in columns["person"]],
            ],
        )
        groups = {"household": household} | {
            e: sampler.read_table_rows(
                store, e, columns=[f"{e}_id", *[c for c in leaves if c in columns[e]]]
            )
            for e in sampler.US_GROUP_ENTITIES[1:]
        }
    ids = household["household_id"].to_numpy()
    weights = household[sampler.HOUSEHOLD_WEIGHT_COLUMN].to_numpy(np.float64)
    labels, _ = sampler.household_stratum_labels(
        household, person, sampler.DEFAULT_STRATUM_COLUMNS
    )
    log(f"{len(ids):,} households, {len(person):,} persons, {len(probes)} probes")

    leaf_households = {}
    for leaf in leaves:
        for entity in sampler.US_ENTITIES:
            table = person if entity == "person" else groups[entity]
            if leaf in table.columns:
                rows = None if entity == "person" else table[f"{entity}_id"].to_numpy()
                leaf_households[leaf] = (
                    entity,
                    table[leaf].to_numpy(),
                    sampler.household_of_rows(entity, person, rows),
                )
                break
    index = pd.Index(ids)
    y_of = {}
    for p in probes:
        y = np.zeros(len(ids))
        for leaf in p.binding_inputs:
            if leaf in leaf_households:
                _, values, households = leaf_households[leaf]
                np.add.at(
                    y,
                    index.get_indexer(households),
                    np.abs(sampler.numeric_support_values(values)),
                )
        y_of[p.id] = y
    truth = {pid: float(np.dot(weights, y)) for pid, y in y_of.items()}
    position = pd.Series(np.arange(len(ids)), index=ids)

    report = {
        "export": str(args.export),
        "fraction": args.fraction,
        "seeds": args.seeds,
        "min_effective_households": args.min_effective,
        "proxy": "household total of |binding inputs|, weighted",
        "designs": {},
    }
    for design in args.design:
        threshold, multiplier = (float(part) for part in design.split(":"))
        records, reasons = sampler.probe_carriers(
            probes, leaf_households, fraction=args.fraction, threshold=threshold
        )
        size_reasons, _ = sampler.size_certainty_households(
            probes,
            leaf_households,
            ids,
            weights,
            fraction=args.fraction,
            multiplier=multiplier,
        )
        certain = sorted(set(reasons) | set(size_reasons))
        take_all = {r.probe_id for r in records if r.certainty}
        carriers = {r.probe_id: int(len(r.carrier_household_ids)) for r in records}
        z_of = {p.id: [] for p in probes}
        ok_of = {p.id: [] for p in probes}
        sizes = []
        for seed in range(args.seeds):
            draw = sampler.draw_households(
                ids, weights, labels, certain, fraction=args.fraction, seed=seed
            )
            sizes.append(len(draw.selected_ids))
            where = position.reindex(draw.selected_ids).to_numpy()
            for p in probes:
                if carriers[p.id] == 0:
                    continue
                y = y_of[p.id][where]
                terms = probe.stratified_variance_terms(
                    y,
                    source_weights=draw.source_weights,
                    labels=draw.labels,
                    certainty=draw.certainty,
                    strata=draw.strata,
                )
                error = float(np.dot(draw.adjusted_weights, y)) - truth[p.id]
                se = float(np.sqrt(terms.sum())) if terms is not None else np.nan
                drawn = int(((y != 0) & ~draw.certainty).sum())
                exact = p.id in take_all and drawn == 0
                effective = probe.effective_variance_households(terms)
                ok_of[p.id].append(bool(exact or effective >= args.min_effective))
                if se > 0:
                    z_of[p.id].append(abs(error) / se)
                else:
                    tiny = abs(error) <= 1e-9 * max(1.0, abs(truth[p.id]))
                    z_of[p.id].append(0.0 if tiny else np.inf)
        rows = {}
        for p in probes:
            if carriers[p.id] == 0:
                continue
            z = np.asarray(z_of[p.id])
            ok = np.asarray(ok_of[p.id])
            rows[p.id] = {
                "carrier_households": carriers[p.id],
                "take_all": p.id in take_all,
                "eligible_share": float(ok.mean()),
                "miss_share_of_all_draws": {
                    f"{k:g}": float((z > k).mean()) for k in K_VALUES
                },
                "miss_share_of_eligible": {
                    f"{k:g}": (float((z[ok] > k).mean()) if ok.any() else None)
                    for k in K_VALUES
                },
            }
        eligible = [r for r in rows.values() if r["eligible_share"] >= 0.5]
        report["designs"][design] = {
            "certainty_threshold": threshold,
            "size_certainty_multiplier": multiplier,
            "certainty_households": len(certain),
            "mean_sample_households": float(np.mean(sizes)),
            "sample_share": float(np.mean(sizes)) / len(ids),
            "probes_eligible_in_most_draws": len(eligible),
            "probes_missing_over_1pct_of_all_draws_at_k3": sum(
                r["miss_share_of_all_draws"]["3"] > 0.01 for r in rows.values()
            ),
            "worst_miss_share_of_eligible_at_k4": max(
                (r["miss_share_of_eligible"]["4"] or 0.0) for r in rows.values()
            ),
            "probes": rows,
        }
        summary = report["designs"][design]
        log(
            f"design {design}: n={summary['mean_sample_households']:.0f} "
            f"({summary['sample_share']:.1%}), certain={len(certain)}, "
            f"eligible probes={len(eligible)}, "
            f"unguarded >1% miss at k=3: "
            f"{summary['probes_missing_over_1pct_of_all_draws_at_k3']}, "
            f"worst guarded miss at k=4: "
            f"{summary['worst_miss_share_of_eligible_at_k4']:.3f}"
        )
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    log("done")


if __name__ == "__main__":
    main()
