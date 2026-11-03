"""The gambling top-up's income gradient under three propensity variants (microcosm#1359).

Usage, from the tree (its packages first on PYTHONPATH):
  <venv>/bin/python <this> <tree> <spine.h5> <gsgb consumer facts jsonl> <out.json>

Vahid's round 4 on microcosm#1381 asked how much of the top-up's income
gradient comes from the propensity's own features. This script runs the tree's
committed lcfs_consumption stage once on a finished spine, as declared, and
records the top-up's inputs. It then runs the top-up alone on those inputs for
each variant, so every other input is the declared run's:

- ``gross_recreation``: the propensity and the stake draws read recreation with
  its gambling stakes in, as at 3a20860fb;
- ``net_recreation``: recreation net of its stakes, as committed;
- ``chance_only``: the walk ordered by the published chance alone (the
  propensity set to one), with the committed features for the stakes.

level_excise_quantities shares each national base by the keys, so a decile's
share of a base is its share of the key; the script checks that on the
declared run. Records aggregates only, at prior weights: nationally and by
equivalised-income decile, the share of households carrying each key and each
decile's share of the key; by region, the share carrying each key, with GSGB's
regional rates (B.2, B.4) for the correlation; the before (diary draws only)
shares; and each variant's stake scales. Every cell rests on ten or more source
households, which the script checks. Licensed output: it stays under
data/ukds/acceptance until sanitize_outputs.py has run.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import microcosm.build.uk_runtime.gambling_participation as participation
import microcosm.build.uk_runtime.lcfs_consumption as consumption
from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.household_income import household_level
from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    uk_time_period,
)
from microcosm.build.uk_runtime.receipt_counts import distinct_count
from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_run_gsgb import KEYS, LCFS, _gsgb_regions, _tool  # noqa: E402

FLOOR = 10


def _gross_recreation_features(donor_encoded, recipient_encoded, drawn, predictors):
    """The features as at 3a20860fb: every spend target, recreation gross."""

    columns = list(consumption.UK_LCFS_CONSUMPTION_TARGET_COLUMNS)
    donor = donor_encoded[[*predictors, *columns]].reset_index(drop=True)
    recipient = recipient_encoded[list(predictors)].reset_index(drop=True)
    for column in columns:
        recipient[column] = drawn[column].to_numpy(dtype=float)
    return donor, recipient


def _ones(donor_features, donor_played, donor_weights, recipient_features, **_):
    played = np.asarray(donor_played, dtype=bool)
    weight = np.asarray(donor_weights, dtype=float)
    return np.ones(len(recipient_features)), {
        "donor_rate": float(weight[played].sum() / weight.sum()),
        "ordering": "published chance alone",
    }


def main() -> None:
    tree, spine, gsgb, out = (Path(arg) for arg in sys.argv[1:5])
    started = time.time()
    recorded: dict = {}
    top_up = consumption.top_up_gambling_participation
    features = consumption.gambling_participation_features

    def recording_top_up(keys, **kwargs):
        recorded["keys"] = {key: np.array(value) for key, value in keys.items()}
        recorded["kwargs"] = kwargs
        result = top_up(keys, **kwargs)
        recorded["result"] = result
        return result

    def recording_features(
        donor_encoded, donor_spending, recipient_encoded, drawn, *, predictors
    ):
        recorded["inputs"] = (
            donor_encoded.copy(),
            recipient_encoded.copy(),
            drawn.copy(),
            list(predictors),
        )
        return features(
            donor_encoded,
            donor_spending,
            recipient_encoded,
            drawn,
            predictors=predictors,
        )

    consumption.top_up_gambling_participation = recording_top_up
    consumption.gambling_participation_features = recording_features

    frame, _ = load_uk_national_frame(spine)
    frame = _tool(tree)._engine_safe_frame(frame)
    spec = load_country_spec("uk")
    stage = spec.sources.stage_map()["lcfs_consumption"]
    transform = consumption.UKLCFSConsumptionStageTransform(
        stage=stage,
        engine=PolicyEngineUKEngine(),
        lcfs_hh_tab_path=LCFS / "dvhh_uk_anon_v3_2024.tab",
        lcfs_person_tab_path=LCFS / "dvper_uk_anon_2024.tab",
    )
    after = transform(frame).table("household")
    consumption.top_up_gambling_participation = top_up
    consumption.gambling_participation_features = features
    evidence = transform.checkpoint_metadata()["evidence"]
    gates = {}
    for gate in spec.gates.gates:
        parameters = dict(gate.parameters)
        if gate.gate != "stage_health" or parameters.get("stage") != "lcfs_consumption":
            continue
        checked = uk_stage_health_gate(
            evidence=evidence,
            stage="lcfs_consumption",
            check=str(parameters["check"]),
            parameters=parameters,
        )
        gates[gate.id] = {"passed": checked.passed, "failures": list(checked.failures)}

    kwargs = recorded["kwargs"]
    donor_encoded, recipient_encoded, drawn, predictors = recorded["inputs"]
    variants = {"net_recreation": recorded["result"]}
    donor_gross, recipient_gross = _gross_recreation_features(
        donor_encoded, recipient_encoded, drawn, predictors
    )
    variants["gross_recreation"] = participation.top_up_gambling_participation(
        recorded["keys"],
        **{
            **kwargs,
            "donor_features": donor_gross,
            "recipient_features": recipient_gross,
        },
    )
    propensity = participation._propensity
    participation._propensity = _ones
    try:
        variants["chance_only"] = participation.top_up_gambling_participation(
            recorded["keys"], **kwargs
        )
    finally:
        participation._propensity = propensity

    household = frame.table("household")
    weight = frame.weights_for("household").values
    mass = float(weight.sum())
    sources = (
        household["source_household_id"].to_numpy()
        if "source_household_id" in household
        else np.arange(len(household))
    )
    if not np.array_equal(
        np.asarray(kwargs["household_ids"]), household["household_id"].to_numpy()
    ):
        raise SystemExit("the top-up's households are not the frame's, in order.")
    engine = PolicyEngineUKEngine()
    variable = "equiv_hbai_household_net_income"
    materialized = engine.materialize(frame, (variable,), uk_time_period(frame))
    income = np.asarray(
        household_level(frame, engine, variable, materialized[variable]), dtype=float
    )
    order = np.argsort(income, kind="stable")
    cumulative = np.cumsum(weight[order]) / mass
    decile = np.empty(len(income), dtype=int)
    decile[order] = np.minimum((cumulative * 10).astype(int), 9) + 1
    region = (
        household["region"].map(lambda v: getattr(v, "name", v)).astype(str).to_numpy()
    )

    def check(name: str, mask: np.ndarray) -> None:
        if 0 < distinct_count(sources, mask, weight) < FLOOR:
            raise SystemExit(
                f"cell {name} rests on fewer than {FLOOR} source households."
            )

    def summary(name: str, values: np.ndarray) -> dict:
        positive = values > 0
        total = float(weight @ values)
        check(f"{name}:positive", positive)
        by_decile = {}
        for d in range(1, 11):
            in_decile = decile == d
            check(f"{name}:decile{d}", in_decile & positive)
            check(f"decile{d}", in_decile)
            by_decile[d] = {
                "households_carrying": float(
                    weight[in_decile & positive].sum() / weight[in_decile].sum()
                ),
                "share_of_key": float(weight[in_decile] @ values[in_decile] / total),
            }
        by_region = {}
        for r in sorted(set(region)):
            in_region = region == r
            check(f"{name}:{r}", in_region & positive)
            check(r, in_region)
            by_region[r] = float(
                weight[in_region & positive].sum() / weight[in_region].sum()
            )
        return {
            "households_carrying": float(weight[positive].sum() / mass),
            "by_decile": by_decile,
            "by_region": by_region,
        }

    report: dict = {
        "tree": str(tree),
        "spine": str(spine),
        "households": int(len(household)),
        "gates": gates,
        "variants": {},
    }
    # The declared run's levelled inputs follow its keys.
    for key, column in KEYS.items():
        values = after[column].to_numpy(dtype=float)
        topped = np.asarray(variants["net_recreation"].keys[key], dtype=float)
        if not np.array_equal(values > 0, topped > 0):
            raise SystemExit(f"{column} is not carried where {key} is.")
        levelled = summary(f"{key}:levelled", values)
        keyed = summary(f"{key}:declared", topped)
        for d in range(1, 11):
            gap = abs(
                levelled["by_decile"][d]["share_of_key"]
                - keyed["by_decile"][d]["share_of_key"]
            )
            if gap > 1e-9:
                raise SystemExit(f"{column}'s decile {d} share is not {key}'s.")
    report["before"] = {
        key: summary(f"{key}:before", np.asarray(recorded["keys"][key], dtype=float))
        for key in KEYS
    }
    for name, result in variants.items():
        entry = {}
        for key in KEYS:
            keyed = result.receipt["keys"][key]
            entry[key] = {
                **summary(f"{name}:{key}", np.asarray(result.keys[key], dtype=float)),
                "amount_scale": keyed.get("amount_scale"),
                "stake_shares": keyed.get("stake_shares"),
                "propensity": keyed.get("propensity"),
            }
        report["variants"][name] = entry
    report["gsgb_regions"] = _gsgb_regions(gsgb)
    report["seconds"] = round(time.time() - started, 1)
    out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"wrote {out} in {report['seconds']} s")


if __name__ == "__main__":
    main()
