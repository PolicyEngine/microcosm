#!/usr/bin/env python3
"""Run selection-only UK dataset-size experiments on a finished graph build.

microcosm#1124: vary the target-weight rule, the per-stage L2 penalty and the
refit baseline floor on a stored ``--dataset-households`` build, with the
dense reference held fixed. Subcommands, each its own process:

``build-cache``  load the run's pool once; cache its skeleton and profile.
``control``      re-run the stored refit; it must reproduce the stored weights.
``census``       step 0's census of the stored support and its caps (no solve).
``run``          run the declared experiments (JSON list); resumable. A
                 configuration the solver chain refuses is recorded as a
                 failed receipt and the run moves on: no release gate runs
                 here, and a refusal is a result.
``score``        score D, the reproduced S0 and every finished experiment.
``plan-step1b``  write #1124's step 1b from the scored step 1a by the
                 pre-registered rules (``--stage ae``, then ``--stage holdout``).
``publish``      write the disclosure-controlled aggregates (no unit records).

Outputs go to ``--out``, which must lie outside the repository and the run
directory: experiment weights are unit-level licensed data. The compute
subcommands take a machine lock and require ``--confirm-exclusive`` (no other
solve is running on this machine).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from microcosm.build.uk_runtime.size_experiment import (
    UKSizeExperiment,
    build_uk_size_experiment_cache,
    load_uk_size_experiment_baseline,
    load_uk_size_scoring_inputs,
    load_uk_size_weights,
    parse_uk_size_experiments,
    run_uk_size_census,
    run_uk_size_control,
    run_uk_size_experiment,
    save_uk_size_weights,
    score_uk_size_experiments,
    uk_size_failed_receipt,
    uk_size_step1b_plan,
    uk_size_weights_of,
)
from microcosm.build.uk_runtime.size_experiment_scorecard import disclosure_controlled

_REPO = Path(__file__).resolve().parents[1]
_LOCK = Path.home() / ".cache" / "microcosm" / "uk-size-experiment.lock"
_CONTROL = "control"
_CENSUS = "census"


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=_default) + "\n"


def _default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _check_out(out: Path, run_dir: Path | None) -> Path:
    out = out.resolve()
    if out.is_relative_to(_REPO):
        raise SystemExit(
            f"--out {out} is inside the repository: weights are unit-level data."
        )
    if run_dir is not None and out.is_relative_to(run_dir.resolve()):
        raise SystemExit(
            f"--out {out} is inside the run directory; it must stay read-only."
        )
    out.mkdir(parents=True, exist_ok=True)
    return out


@contextmanager
def _machine_lock():
    _LOCK.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(
                "another size experiment holds the machine lock."
            ) from None
        yield


def _require_exclusive(args: argparse.Namespace) -> None:
    if not args.confirm_exclusive:
        raise SystemExit(
            "pass --confirm-exclusive to state that no other solve is running on "
            "this machine (24 GiB; one solve at a time)."
        )


def _check_rss(receipt: dict, limit_gib: float | None) -> None:
    if limit_gib is not None and receipt["peak_rss_bytes"] > limit_gib * 2**30:
        raise SystemExit(
            f"peak RSS {receipt['peak_rss_bytes'] / 2**30:.1f} GiB passed "
            f"--max-rss-gib {limit_gib}; stopping before the next experiment."
        )


def _build_cache(args: argparse.Namespace) -> int:
    _require_exclusive(args)
    with _machine_lock():
        manifest = build_uk_size_experiment_cache(args.run_dir, args.cache_dir)
    sys.stdout.write(_json(manifest))
    return 0


def _control(args: argparse.Namespace) -> int:
    _require_exclusive(args)
    out = _check_out(args.out, args.run_dir)
    with _machine_lock():
        baseline = load_uk_size_experiment_baseline(args.run_dir, args.cache_dir)
        result = run_uk_size_control(
            baseline, accept_drift_reason=args.accept_control_drift
        )
    sized = result.pop("sized")
    directory = out / _CONTROL
    directory.mkdir(parents=True, exist_ok=True)
    save_uk_size_weights(
        uk_size_weights_of(
            "S0", sized, max_weight_ratio=baseline.settings["max_weight_ratio"]
        ),
        directory / "weights.npz",
    )
    receipt = {
        **result,
        "run_rule": baseline.run_rule,
        "settings": dict(baseline.settings),
        "digests": dict(baseline.digests),
    }
    (directory / "receipt.json").write_text(_json(receipt), encoding="utf-8")
    sys.stdout.write(
        _json(
            {
                key: receipt[key]
                for key in (
                    "passed",
                    "exact",
                    "max_relative_weight_difference",
                    "receipt_differences",
                )
            }
        )
    )
    return 0 if result["passed"] else 1


def _census(args: argparse.Namespace) -> int:
    _require_exclusive(args)
    out = _check_out(args.out, args.run_dir)
    with _machine_lock():
        baseline = load_uk_size_experiment_baseline(args.run_dir, args.cache_dir)
        census = run_uk_size_census(baseline)
    directory = out / _CENSUS
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "census.json").write_text(_json(census), encoding="utf-8")
    stored = census["floors"][f"{census['stored_floor']:g}"]
    sys.stdout.write(
        _json(
            {
                "checks": census["checks"],
                "early_size_triggers": census["early_size_triggers"],
                "k_min": census["search"]["k_min"],
                "capacity_to_dense_at_run_floor": stored["capacity_to_dense"],
            }
        )
    )
    if not census["checks"]["passed"]:
        sys.stdout.write(
            "census self-check FAILED: its recomputed refit start disagrees with "
            "the stored run; read census.json before using its capacities.\n"
        )
    return 0


def _control_passed(out: Path) -> bool:
    path = out / _CONTROL / "receipt.json"
    return path.is_file() and json.loads(path.read_text("utf-8")).get("passed") is True


def _run(args: argparse.Namespace) -> int:
    _require_exclusive(args)
    out = _check_out(args.out, args.run_dir)
    if not _control_passed(out):
        raise SystemExit(
            "run `control` first: no variant is read before S0 reproduces."
        )
    experiments = parse_uk_size_experiments(
        json.loads(Path(args.experiments).read_text("utf-8"))
    )
    if args.only:
        unknown = set(args.only) - {experiment.name for experiment in experiments}
        if unknown:
            raise SystemExit(f"--only names unknown experiments: {sorted(unknown)}")
        experiments = tuple(e for e in experiments if e.name in set(args.only))
    failed = []
    with _machine_lock():
        baseline = load_uk_size_experiment_baseline(args.run_dir, args.cache_dir)
        for experiment in experiments:
            directory = out / experiment.name
            if (directory / "receipt.json").is_file() and not args.force:
                sys.stdout.write(f"skip {experiment.name}: receipt exists\n")
                continue
            sys.stdout.write(f"run {experiment.name} ({experiment.mode})\n")
            sys.stdout.flush()
            started = time.perf_counter()
            try:
                outcome = run_uk_size_experiment(baseline, experiment)
            except Exception as error:
                # A refusal of the solver chain is this configuration's result:
                # record it and move on (no weights, so it is never scored).
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "weights.npz").unlink(missing_ok=True)
                receipt = uk_size_failed_receipt(
                    baseline,
                    experiment,
                    error,
                    wall_seconds=time.perf_counter() - started,
                )
                (directory / "receipt.json").write_text(
                    _json(receipt), encoding="utf-8"
                )
                failed.append(experiment.name)
                sys.stdout.write(
                    f"failed {experiment.name}: {type(error).__name__}: {error}\n"
                )
                continue
            directory.mkdir(parents=True, exist_ok=True)
            refit = outcome["results"].get("refit")
            if refit is not None:
                save_uk_size_weights(
                    uk_size_weights_of(
                        experiment.name,
                        refit,
                        max_weight_ratio=baseline.settings["max_weight_ratio"],
                    ),
                    directory / "weights.npz",
                )
            (directory / "receipt.json").write_text(
                _json(outcome["receipt"]), encoding="utf-8"
            )
            _check_rss(outcome["receipt"], args.max_rss_gib)
    if failed:
        sys.stdout.write(f"{len(failed)} configuration(s) failed: {failed}\n")
    return 0


def _finished(out: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in out.glob("*/weights.npz")
        if path.parent.name != _CONTROL and (path.parent / "receipt.json").is_file()
    )


def _score(args: argparse.Namespace) -> int:
    out = _check_out(args.out, args.run_dir)
    if not _control_passed(out):
        raise SystemExit("score needs a passing control (S0) in --out.")
    inputs = load_uk_size_scoring_inputs(args.run_dir, args.cache_dir)
    ratio = inputs.max_weight_ratio
    control = load_uk_size_weights(
        "S0", out / _CONTROL / "weights.npz", max_weight_ratio=ratio
    )
    candidates = [
        load_uk_size_weights(
            directory.name, directory / "weights.npz", max_weight_ratio=ratio
        )
        for directory in _finished(out)
        if not args.only or directory.name in set(args.only)
    ]
    scorecard = score_uk_size_experiments(inputs, control, candidates)
    (out / "scorecard.json").write_text(_json(scorecard), encoding="utf-8")
    summary = {
        label: {"all_pass": block["all_pass"], "passes": block["passes"]}
        for label, block in scorecard["acceptance"].items()
    }
    sys.stdout.write(_json(summary))
    return 0


def _plan_step1b(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    scorecard = json.loads((out / "scorecard.json").read_text("utf-8"))
    receipts = {
        path.parent.name: json.loads(path.read_text("utf-8"))
        for path in sorted(out.glob("*/receipt.json"))
        if path.parent.name != _CONTROL
    }
    census = json.loads((out / _CENSUS / "census.json").read_text("utf-8"))
    plan = uk_size_step1b_plan(scorecard, receipts, census, stage=args.stage)
    (out / f"step1b_{args.stage}_plan.json").write_text(_json(plan), encoding="utf-8")
    destination = args.to.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_json(plan["experiments"]), encoding="utf-8")
    picks = {
        key: value
        for key, value in plan["picks"].items()
        if key != "ladders" and not key.endswith("candidates")
    }
    sys.stdout.write(
        _json(
            {
                "stage": plan["stage"],
                "picks": picks,
                "experiments": [
                    experiment["name"] for experiment in plan["experiments"]
                ],
            }
        )
    )
    return 0


def _publish(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    destination = args.to.resolve()
    scorecard = json.loads((out / "scorecard.json").read_text("utf-8"))
    receipts = {}
    for path in sorted(out.glob("*/receipt.json")):
        receipt = json.loads(path.read_text("utf-8"))
        receipt.get("size_receipt", {}).pop("household_ids", None)
        receipt.pop("traceback", None)
        receipts[path.parent.name] = receipt
    published = {"scorecard": scorecard, "receipts": receipts}
    census_path = out / _CENSUS / "census.json"
    if census_path.is_file():
        published["census"] = json.loads(census_path.read_text("utf-8"))
    plans = {
        path.name.removeprefix("step1b_").removesuffix("_plan.json"): json.loads(
            path.read_text("utf-8")
        )
        for path in sorted(out.glob("step1b_*_plan.json"))
    }
    if plans:
        published["step1b"] = plans
    destination.mkdir(parents=True, exist_ok=True)
    controlled = disclosure_controlled(published, minimum_count=args.minimum_count)
    (destination / "results.json").write_text(_json(controlled), encoding="utf-8")
    sys.stdout.write(f"wrote {destination / 'results.json'}\n")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser, *, out: bool = True) -> None:
        command.add_argument("--run-dir", type=Path, required=True)
        command.add_argument("--cache-dir", type=Path, required=True)
        if out:
            command.add_argument("--out", type=Path, required=True)

    def compute(command: argparse.ArgumentParser) -> None:
        command.add_argument("--confirm-exclusive", action="store_true")

    build = sub.add_parser("build-cache", help="cache the pool skeleton and profile")
    common(build, out=False)
    compute(build)
    control = sub.add_parser("control", help="reproduce the stored refit")
    common(control)
    compute(control)
    control.add_argument("--accept-control-drift", metavar="REASON", default=None)
    census = sub.add_parser("census", help="step 0's census of the stored support")
    common(census)
    compute(census)
    run = sub.add_parser("run", help="run declared experiments")
    common(run)
    compute(run)
    run.add_argument("--experiments", type=Path, required=True)
    run.add_argument("--only", nargs="+", default=None)
    run.add_argument("--force", action="store_true")
    run.add_argument("--max-rss-gib", type=float, default=None)
    score = sub.add_parser("score", help="score D, S0 and finished experiments")
    common(score)
    score.add_argument("--only", nargs="+", default=None)
    plan = sub.add_parser(
        "plan-step1b",
        help="write step 1b from the scored step 1a (pre-registered rules)",
    )
    plan.add_argument("--out", type=Path, required=True)
    plan.add_argument("--stage", choices=("ae", "holdout"), required=True)
    plan.add_argument("--to", type=Path, required=True)
    publish = sub.add_parser("publish", help="write disclosure-controlled aggregates")
    publish.add_argument("--out", type=Path, required=True)
    publish.add_argument("--to", type=Path, required=True)
    publish.add_argument("--minimum-count", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    handlers = {
        "build-cache": _build_cache,
        "control": _control,
        "census": _census,
        "run": _run,
        "score": _score,
        "plan-step1b": _plan_step1b,
        "publish": _publish,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["UKSizeExperiment", "main", "parse_args"]
