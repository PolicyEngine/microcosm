"""Fingerprint each step of the retirement-contribution QRF on one platform.

``run`` replays the exported stage input (see ``platform_stage_input.py``)
through the production stage code with four observers attached to
``microcosm.fit.qrf``: the weighted bootstrap sample each forest is grown on,
every fitted tree (structure, thresholds, node target means), each sign gate's
predicted probabilities on the PUF rows, and each forest's drawn values. The
observers only read; the stage output is unchanged.

``compare`` takes two such fingerprints from different platforms and reports
the first step at which they diverge: how many trees differ in structure, how
far the gate probabilities move, how many drawn values differ. It first
authenticates each side: the arrays file must have the sha256 its fingerprint
records, and every stored array (node arrays per tree, gate probabilities,
drawn values) must match its digest in the fingerprint.

Usage:
    python platform_forest_probe.py run --input INPUT.pkl --out PROBE.json \
        --arrays PROBE.npz
    python platform_forest_probe.py compare A.json A.npz B.json B.npz --out OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NODE_ARRAYS = (
    "feature",
    "threshold",
    "children_left",
    "children_right",
    "impurity",
    "n_node_samples",
    "weighted_n_node_samples",
)


def first_divergence(left: dict, right: dict) -> dict[str, object] | None:
    """Walk two trees from the root; return the shallowest node they split
    differently, with each side's weighted impurity decrease there."""

    frontier = [(0, 0, 0)]
    while frontier:
        next_frontier = []
        for a, b, depth in frontier:
            leaf_a = left["children_left"][a] == -1
            leaf_b = right["children_left"][b] == -1
            same = (
                leaf_a == leaf_b
                and left["feature"][a] == right["feature"][b]
                and left["threshold"][a] == right["threshold"][b]
            )
            if not same:

                def decrease(tree: dict, node: int) -> float | None:
                    lo, hi = tree["children_left"][node], tree["children_right"][node]
                    if lo == -1:
                        return None
                    weight, impurity = tree["weighted_n_node_samples"], tree["impurity"]
                    return float(
                        weight[node] * impurity[node]
                        - weight[lo] * impurity[lo]
                        - weight[hi] * impurity[hi]
                    )

                return {
                    "depth": depth,
                    "n_node_samples": [
                        int(left["n_node_samples"][a]),
                        int(right["n_node_samples"][b]),
                    ],
                    "parent_impurity": [
                        float(left["impurity"][a]),
                        float(right["impurity"][b]),
                    ],
                    "feature": [int(left["feature"][a]), int(right["feature"][b])],
                    "threshold": [
                        float(left["threshold"][a]),
                        float(right["threshold"][b]),
                    ],
                    "weighted_impurity_decrease": [
                        decrease(left, a),
                        decrease(right, b),
                    ],
                }
            if not leaf_a:
                next_frontier.append(
                    (left["children_left"][a], right["children_left"][b], depth + 1)
                )
                next_frontier.append(
                    (left["children_right"][a], right["children_right"][b], depth + 1)
                )
        frontier = next_frontier
    return None


def relative_gain_difference(divergence: dict) -> float | None:
    """Relative gap between the two platforms' impurity decreases at a node
    both split; ``None`` when one platform made the node a leaf."""

    x, y = divergence["weighted_impurity_decrease"]
    if x is None or y is None:
        return None
    scale = max(abs(x), abs(y))
    return abs(x - y) / scale if scale > 0 else 0.0


def summarize_divergences(found: list[dict]) -> dict[str, object]:
    """Describe a forest's first-divergent nodes, one per diverging tree."""

    def block(rows: list[dict]) -> dict[str, object]:
        gaps = [g for g in map(relative_gain_difference, rows) if g is not None]
        depths = [d["depth"] for d in rows]
        samples = [d["n_node_samples"][0] for d in rows]
        return {
            "trees": len(rows),
            "both_platforms_split": len(gaps),
            "one_platform_leaf_other_split": len(rows) - len(gaps),
            "depth_min": int(min(depths)),
            "depth_median": float(np.median(depths)),
            "depth_max": int(max(depths)),
            "trees_at_depth_0_or_1": sum(depth <= 1 for depth in depths),
            "node_samples_min": int(min(samples)),
            "node_samples_median": float(np.median(samples)),
            "node_samples_max": int(max(samples)),
            "gains_within_1e-12": sum(g <= 1e-12 for g in gaps),
            "gains_within_1e-9": sum(g <= 1e-9 for g in gaps),
            "gains_within_1e-6": sum(g <= 1e-6 for g in gaps),
            "gain_relative_difference_median": (
                float(np.median(gaps)) if gaps else None
            ),
            "gain_relative_difference_max": max(gaps) if gaps else None,
        }

    by_pair: dict[str, list[dict]] = {}
    for d in found:
        lo, hi = sorted(d["feature"])
        by_pair.setdefault(f"{lo}|{hi}", []).append(d)
    summary = block(found)
    summary.update(
        {
            "same_node_samples_both_platforms": sum(
                d["n_node_samples"][0] == d["n_node_samples"][1] for d in found
            ),
            "same_parent_impurity": sum(
                d["parent_impurity"][0] == d["parent_impurity"][1] for d in found
            ),
            "different_feature": sum(d["feature"][0] != d["feature"][1] for d in found),
            "same_feature_different_threshold": sum(
                d["feature"][0] == d["feature"][1]
                and d["threshold"][0] != d["threshold"][1]
                for d in found
            ),
            # sklearn marks a leaf's feature as -2, so a pair with -2 is a node
            # one platform split and the other left as a leaf.
            "by_feature_pair": {
                pair: block(rows) for pair, rows in sorted(by_pair.items())
            },
            "gain_relative_differences_sorted": sorted(
                g for g in map(relative_gain_difference, found) if g is not None
            ),
            "examples": found[:3],
        }
    )
    return summary


def authenticate_forest(event: dict, arrays) -> None:
    """Refuse a forest whose fingerprints and node arrays do not agree.

    Every per-tree list must have one entry per tree; the offsets must start at
    0, step by each tree's node count and end at the length of every node
    array; and each tree's ``feature``/``children_left``/``children_right``,
    ``threshold`` and ``impurity``/``n_node_samples``/``weighted_n_node_samples``
    slices must reproduce the structure, threshold and diagnostic digests the
    fingerprint recorded.
    """

    trees, index = event["trees"], event["index"]
    for key in (
        "node_counts",
        "structure_sha256",
        "threshold_sha256",
        "diagnostic_sha256",
        "value_sha256",
    ):
        if len(event[key]) != trees:
            raise SystemExit(f"Forest {index}: {key} has the wrong length.")
    for name in ("offsets", *NODE_ARRAYS):
        if f"forest_{index}_{name}" not in arrays:
            raise SystemExit(f"Forest {index}: node array {name!r} is missing.")
    offsets = arrays[f"forest_{index}_offsets"]
    if (
        len(offsets) != trees + 1
        or int(offsets[0]) != 0
        or list(np.diff(offsets)) != event["node_counts"]
    ):
        raise SystemExit(f"Forest {index}: node offsets disagree with node counts.")
    for name in NODE_ARRAYS:
        if len(arrays[f"forest_{index}_{name}"]) != int(offsets[-1]):
            raise SystemExit(
                f"Forest {index}: node array {name!r} does not end at the last offset."
            )
    for tree in range(trees):
        nodes = tree_arrays(arrays, index, tree)
        for key, value in (
            ("structure_sha256", structure_digest(nodes)),
            ("threshold_sha256", digest(nodes["threshold"])),
            ("diagnostic_sha256", diagnostic_digest(nodes)),
        ):
            if value != event[key][tree]:
                raise SystemExit(
                    f"Forest {index}, tree {tree}: stored arrays do not match {key}."
                )


def authenticate_probe(probe: dict, arrays, arrays_path: Path) -> None:
    """Refuse a probe whose arrays file is not the one its fingerprint names,
    or whose stored arrays do not match their recorded digests."""

    if file_sha256(arrays_path) != probe["arrays_sha256"]:
        raise SystemExit(f"{arrays_path.name} is not the arrays file of its probe.")
    for event in probe["events"]:
        if event["step"] == "forest":
            authenticate_forest(event, arrays)
            continue
        key, recorded = {
            "gate_fit": ("train_proba_key", "train_proba_sha256"),
            "gate_draw": ("proba_key", "proba_sha256"),
            "forest_draw": ("values_key", "values_sha256"),
        }.get(event["step"], (None, None))
        if key is None:
            continue
        if event[key] not in arrays or digest(arrays[event[key]]) != event[recorded]:
            raise SystemExit(f"Stored array {event[key]!r} does not match {recorded}.")


def tree_arrays(arrays, index: int, tree: int) -> dict[str, np.ndarray]:
    offsets = arrays[f"forest_{index}_offsets"]
    lo, hi = int(offsets[tree]), int(offsets[tree + 1])
    return {name: arrays[f"forest_{index}_{name}"][lo:hi] for name in NODE_ARRAYS}


def file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            sha.update(block)
    return sha.hexdigest()


def digest(*arrays: np.ndarray) -> str:
    sha = hashlib.sha256()
    for array in arrays:
        sha.update(np.ascontiguousarray(array).tobytes())
    return sha.hexdigest()


def structure_digest(tree) -> str:
    """``tree`` is a fitted sklearn ``Tree`` or a dict of its node arrays."""

    get = tree.__getitem__ if isinstance(tree, dict) else tree.__getattribute__
    return digest(get("feature"), get("children_left"), get("children_right"))


def diagnostic_digest(tree) -> str:
    get = tree.__getitem__ if isinstance(tree, dict) else tree.__getattribute__
    return digest(
        get("impurity"), get("n_node_samples"), get("weighted_n_node_samples")
    )


def run(args: argparse.Namespace) -> None:
    import microcosm.fit.qrf as qrf
    from microcosm.build.source_runtime import SourceRuntimeConfig, run_source_stage
    from microcosm.build.us_runtime import retirement_contributions as stage

    events: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {}
    original_bootstrap = qrf._weighted_bootstrap
    original_fit_forest = qrf._fit_forest
    original_fit_gate = qrf.RegimeGatedQRF._fit_gate
    original_gate_draw = qrf._gate_draw_with_rng
    original_draw = qrf._Forest.draw

    def bootstrap(x, y, weights, rng):
        x_fit, y_fit = original_bootstrap(x, y, weights, rng)
        events.append(
            {
                "step": "bootstrap",
                "rows": int(len(y_fit)),
                "x_sha256": digest(x_fit),
                "y_sha256": digest(y_fit),
                "weights_sha256": None if weights is None else digest(weights),
            }
        )
        return x_fit, y_fit

    def fit_forest(x, y, columns, weights, **kwargs):
        forest = original_fit_forest(x, y, columns, weights, **kwargs)
        trees = [estimator.tree_ for estimator in forest.model.estimators_]
        index = sum(1 for event in events if event["step"] == "forest")
        # Node arrays of every tree, concatenated; ``offsets`` splits them.
        arrays[f"forest_{index}_offsets"] = np.cumsum(
            [0, *(tree.node_count for tree in trees)]
        )
        for name in NODE_ARRAYS:
            arrays[f"forest_{index}_{name}"] = np.concatenate(
                [np.asarray(getattr(tree, name)) for tree in trees]
            )
        events.append(
            {
                "step": "forest",
                "index": index,
                "columns": list(columns),
                "seed": int(kwargs["seed"]),
                "trees": len(trees),
                "node_counts": [int(tree.node_count) for tree in trees],
                "structure_sha256": [structure_digest(tree) for tree in trees],
                "threshold_sha256": [digest(tree.threshold) for tree in trees],
                "diagnostic_sha256": [diagnostic_digest(tree) for tree in trees],
                # ``tree.value`` holds every node's target mean, internal
                # nodes included.
                "value_sha256": [digest(tree.value) for tree in trees],
            }
        )
        return forest

    def fit_gate(self, features, labels, weights, rng):
        gate = original_fit_gate(self, features, labels, weights, rng)
        probabilities = np.asarray(gate.predict_proba(features))
        key = f"gate_train_proba_{sum(1 for e in events if e['step'] == 'gate_fit')}"
        arrays[key] = probabilities
        events.append(
            {
                "step": "gate_fit",
                "features_sha256": digest(features),
                "labels_sha256": digest(labels),
                "train_proba_sha256": digest(probabilities),
                "train_proba_key": key,
                "iterations": int(gate.n_iter_),
            }
        )
        return gate

    def gate_draw(gate, features, columns, rng):
        x = features.loc[:, list(columns)].to_numpy(dtype=np.float64)
        probabilities = np.asarray(gate.predict_proba(x))
        signs = original_gate_draw(gate, features, columns, rng)
        key = f"gate_draw_proba_{sum(1 for e in events if e['step'] == 'gate_draw')}"
        arrays[key] = probabilities
        events.append(
            {
                "step": "gate_draw",
                "features_sha256": digest(x),
                "proba_sha256": digest(probabilities),
                "proba_key": key,
                "signs_sha256": digest(np.asarray(signs)),
                "positive_rows": int(np.count_nonzero(np.asarray(signs) == 1)),
            }
        )
        return signs

    def draw(self, frame, quantiles):
        values = original_draw(self, frame, quantiles)
        key = f"forest_draw_{sum(1 for e in events if e['step'] == 'forest_draw')}"
        arrays[key] = np.asarray(values)
        events.append(
            {
                "step": "forest_draw",
                "rows": int(len(values)),
                "features_sha256": digest(
                    frame.loc[:, list(self.columns)].to_numpy(dtype=np.float64)
                ),
                "quantiles_sha256": digest(np.asarray(quantiles)),
                "values_sha256": digest(np.asarray(values)),
                "values_key": key,
            }
        )
        return values

    qrf._weighted_bootstrap = bootstrap
    qrf._fit_forest = fit_forest
    qrf.RegimeGatedQRF._fit_gate = fit_gate
    qrf._gate_draw_with_rng = gate_draw
    qrf._Forest.draw = draw

    slim = pd.read_pickle(args.input)
    slim = slim.drop(columns=[c for c in slim.columns if c.startswith("stored_")])
    output = run_source_stage(
        stage.us_retirement_contributions_stage_spec(),
        tables={"person": slim},
        operation_handlers={
            "derive_retirement_contributions": (
                stage.derive_us_retirement_contributions_from_manifest
            ),
            "impute_retirement_contributions_to_puf_support": (
                stage.impute_us_retirement_contributions_to_puf_support_from_manifest
            ),
        },
        config=SourceRuntimeConfig(seed=0, target_year=2024),
    )
    receipt = {
        "input": Path(args.input).name,
        "input_sha256": file_sha256(Path(args.input)),
        "platform": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "sklearn": __import__("sklearn").__version__,
            "quantile_forest": __import__("quantile_forest").__version__,
        },
        "output_sha256": {
            column: digest(output[column].to_numpy(dtype=np.float64))
            for column in stage.US_RETIREMENT_CONTRIBUTION_OUTPUT_COLUMNS
        },
        "events": events,
    }
    # The arrays first, so the fingerprint can name them by sha256.
    np.savez_compressed(args.arrays, **arrays)
    receipt["arrays_sha256"] = file_sha256(Path(args.arrays))
    Path(args.out).write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps([event["step"] for event in events]))


def compare(args: argparse.Namespace) -> None:
    left = json.loads(args.left.read_text())
    right = json.loads(args.right.read_text())
    left_arrays, right_arrays = np.load(args.left_arrays), np.load(args.right_arrays)
    steps_left = [event["step"] for event in left["events"]]
    steps_right = [event["step"] for event in right["events"]]
    # Refuse a comparison that would be incomplete or misaligned.
    if steps_left != steps_right:
        raise SystemExit("The two probes recorded different step sequences.")
    if left.get("input_sha256") != right.get("input_sha256"):
        raise SystemExit("The two probes read different inputs.")
    for a, b in zip(left["events"], right["events"]):
        if a["step"] != "forest":
            continue
        if (
            a["columns"] != b["columns"]
            or a["trees"] != b["trees"]
            or a["index"] != b["index"]
            or a["seed"] != b["seed"]
        ):
            raise SystemExit("A forest differs in identity, seed or tree count.")
    authenticate_probe(left, left_arrays, args.left_arrays)
    authenticate_probe(right, right_arrays, args.right_arrays)
    report: dict[str, object] = {
        "inputs": {
            name: {"file": path.name, "sha256": file_sha256(path)}
            for name, path in (
                ("left", args.left),
                ("left_arrays", args.left_arrays),
                ("right", args.right),
                ("right_arrays", args.right_arrays),
            )
        },
        "probed_input": {"file": left["input"], "sha256": left["input_sha256"]},
        "left": left["platform"],
        "right": right["platform"],
        "same_step_sequence": True,
        "steps": [],
    }
    for index, (a, b) in enumerate(zip(left["events"], right["events"])):
        entry: dict[str, object] = {"index": index, "step": a["step"]}
        if a["step"] == "bootstrap":
            entry["identical"] = (
                a["x_sha256"] == b["x_sha256"] and a["y_sha256"] == b["y_sha256"]
            )
            entry["rows"] = a["rows"]
        elif a["step"] == "forest":
            entry["trees"] = a["trees"]
            entry["trees_differing_in_structure"] = sum(
                x != y for x, y in zip(a["structure_sha256"], b["structure_sha256"])
            )
            entry["trees_differing_in_thresholds"] = sum(
                x != y for x, y in zip(a["threshold_sha256"], b["threshold_sha256"])
            )
            entry["trees_differing_in_node_values"] = sum(
                x != y for x, y in zip(a["value_sha256"], b["value_sha256"])
            )
            entry["trees_differing_in_node_count"] = sum(
                x != y for x, y in zip(a["node_counts"], b["node_counts"])
            )
            divergences = [
                first_divergence(
                    tree_arrays(left_arrays, a["index"], tree),
                    tree_arrays(right_arrays, b["index"], tree),
                )
                for tree in range(a["trees"])
            ]
            found = [d for d in divergences if d is not None]
            entry["trees_with_a_divergent_split"] = len(found)
            if found:
                entry["first_divergence"] = summarize_divergences(found)
        elif a["step"] in ("gate_fit", "gate_draw"):
            key = a["train_proba_key"] if a["step"] == "gate_fit" else a["proba_key"]
            pa, pb = left_arrays[key], right_arrays[key]
            entry["features_identical"] = a["features_sha256"] == b["features_sha256"]
            entry["rows"] = int(len(pa))
            entry["rows_with_different_probability"] = int((pa != pb).any(axis=1).sum())
            entry["max_abs_probability_difference"] = float(np.max(np.abs(pa - pb)))
            if a["step"] == "gate_draw":
                entry["signs_identical"] = a["signs_sha256"] == b["signs_sha256"]
                entry["positive_rows"] = [a["positive_rows"], b["positive_rows"]]
            else:
                entry["iterations"] = [a["iterations"], b["iterations"]]
        elif a["step"] == "forest_draw":
            va, vb = left_arrays[a["values_key"]], right_arrays[b["values_key"]]
            entry["rows"] = [int(len(va)), int(len(vb))]
            entry["features_identical"] = a["features_sha256"] == b["features_sha256"]
            entry["quantiles_identical"] = (
                a["quantiles_sha256"] == b["quantiles_sha256"]
            )
            if len(va) == len(vb):
                difference = np.abs(va - vb)
                entry["rows_differing"] = int((va != vb).sum())
                entry["rows_differing_by_more_than_1e-9"] = int(
                    (difference > 1e-9).sum()
                )
                entry["rows_differing_by_more_than_1_dollar"] = int(
                    (difference > 1.0).sum()
                )
                entry["max_abs_difference"] = float(difference.max())
        report["steps"].append(entry)
    forests = [step for step in report["steps"] if step["step"] == "forest"]
    divergent = [
        step["first_divergence"] for step in forests if "first_divergence" in step
    ]
    report["forest_totals"] = {
        "trees": sum(step["trees"] for step in forests),
        "trees_differing_in_structure": sum(
            step["trees_differing_in_structure"] for step in forests
        ),
        **{
            key: sum(d[key] for d in divergent)
            for key in (
                "both_platforms_split",
                "one_platform_leaf_other_split",
                "different_feature",
                "same_feature_different_threshold",
                "same_node_samples_both_platforms",
                "gains_within_1e-12",
                "gains_within_1e-9",
                "gains_within_1e-6",
            )
        },
    }
    report["outputs_identical"] = {
        column: left["output_sha256"][column] == right["output_sha256"][column]
        for column in left["output_sha256"]
    }
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["steps"], indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    rn = sub.add_parser("run")
    rn.add_argument("--input", type=Path, required=True)
    rn.add_argument("--out", type=Path, required=True)
    rn.add_argument("--arrays", type=Path, required=True)
    cp = sub.add_parser("compare")
    cp.add_argument("left", type=Path)
    cp.add_argument("left_arrays", type=Path)
    cp.add_argument("right", type=Path)
    cp.add_argument("right_arrays", type=Path)
    cp.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"run": run, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
