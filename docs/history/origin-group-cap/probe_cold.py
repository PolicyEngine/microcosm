"""Bounded cold integration probe over freshly generated invented sources."""

import importlib.util
import json
import pathlib
import tempfile

p = (
    pathlib.Path(__file__).resolve().parents[3]
    / "packages/microcosm-build/tests/test_us_origin_group_calibration_graph.py"
)
s = importlib.util.spec_from_file_location("cap_test", p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
original = m.stage.budgets.verify_origin_group_budget
original_freeze = m.stage.budgets.freeze_origin_group_budget
initial = {}


def freeze(manifest, store, **kwargs):
    for name in ("native_population", "expanded_population"):
        frame = kwargs[name].frame
        snapshot = {
            "columns": {e: list(frame.table(e).columns) for e in frame.entities},
            "metadata": dict(frame.metadata),
        }
        if name not in initial:
            initial[name] = snapshot
        elif initial[name] != snapshot:
            print(
                "Frame shape metadata differences",
                name,
                json.dumps(
                    {
                        k: [initial[name][k], v]
                        for k, v in snapshot.items()
                        if initial[name][k] != v
                    },
                    default=str,
                ),
                flush=True,
            )
    return original_freeze(manifest, store, **kwargs)


def inspect(budget, manifest, store, **kwargs):
    expected = m.stage.budgets.freeze_origin_group_budget(manifest, store, **kwargs)
    if expected.payload != budget.payload:
        print(
            "Budget differences:",
            json.dumps(
                {
                    k: [budget.document.get(k), v]
                    for k, v in expected.document.items()
                    if budget.document.get(k) != v
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return original(budget, manifest, store, **kwargs)


m.stage.budgets.verify_origin_group_budget = inspect
m.stage.budgets.freeze_origin_group_budget = freeze
with tempfile.TemporaryDirectory(prefix="origin-cap-invented-") as tmp:
    root = pathlib.Path(tmp) / "fixture"
    m._fixture(root)
    print(json.dumps(m._run(root, observe=True), sort_keys=True))
