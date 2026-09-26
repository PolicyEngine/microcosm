#!/usr/bin/env python3
"""Check shipped post-export requests on a small written US release H5.

Run sequential workers with four reform passes each by default, so their reform
systems are discarded when each worker exits. This is a fixture guard sweep;
the release-frame memory and timing rehearsal remains a separate operation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


def _load_builder():
    path = Path(__file__).with_name("build_us_fiscal_refresh_release.py")
    spec = importlib.util.spec_from_file_location("post_export_sweep_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _record_requests(builder):
    """Record OBBBA stacks and validation's missing-estimate fallback passes."""
    specs = builder.load_default_reform_specs(period=builder.PERIOD)

    def validation(simulate):
        return builder.reform_validation_payload(
            specs,
            period=builder.PERIOD,
            simulate=simulate,
            baseline_levels=builder.default_baseline_level_specs(),
            release_id="post-export-guard-sweep",
        )

    consumers = {
        "reform_coverage_smoke": builder._reform_coverage_smoke_consumer,
        "reform_validation": validation,
        "demographics": builder._demographics_consumer,
    }
    plans, requests = {}, []
    for name, consume in consumers.items():
        baseline = {}

        class RecordingSimulation:
            def __init__(self, reform, *, name=name, baseline=baseline):
                self.reform = reform
                self.name = name
                self.baseline = baseline
                self.keys = set()

            def calculate(self, variable, period=None, map_to=None):
                key = builder._post_export_key(variable, period, map_to)
                if self.reform is None:
                    self.baseline.setdefault(key, None)
                elif key not in self.keys:
                    requests.append((self.name, self.reform, key))
                    self.keys.add(key)
                return builder._RecordingPostExportSimulation(None).calculate(*key)

        consume(RecordingSimulation)
        plans[name] = builder._ascending_period_plan(baseline)
    counts = {
        "smoke_probes": sum(name == "reform_coverage_smoke" for name, _, _ in requests),
        "validation_specs": sum(not spec.in_sample for spec in specs),
        "in_sample_fallback_specs": sum(spec.in_sample for spec in specs),
        "validation_reform_passes": sum(
            name == "reform_validation" for name, _, _ in requests
        ),
        "baseline_keys": {name: len(plan) for name, plan in plans.items()},
    }
    return plans, requests, counts


def _assert_same_values(observed, expected, key):
    import numpy as np

    np.testing.assert_array_equal(observed.weights, expected.weights, err_msg=str(key))
    values, reference = np.asarray(observed), np.asarray(expected)
    if values.dtype.kind in "biuf":
        np.testing.assert_allclose(
            values, reference, rtol=1e-12, atol=1e-6, err_msg=str(key)
        )
    else:
        np.testing.assert_array_equal(values, reference, err_msg=str(key))


def _score_baselines(builder, scorer, path, plans):
    from microcosm.build.us_runtime.reform_validation import default_simulate_factory

    whole = default_simulate_factory(path)(None)
    try:
        engine = builder._AscendingPeriodEngine(
            whole, label="sweep whole-file baseline"
        )
        reference = {
            key: builder._PostExportValues(
                *builder._post_export_values_and_weights(engine.calculate(key), key)
            )
            for key in builder._ascending_period_plan(
                key for plan in plans.values() for key in plan
            )
        }
    finally:
        builder.release_engine_simulation(whole)
    for name, plan in plans.items():
        consumer = scorer.open_consumer(name, plan)
        for key in plan:
            _assert_same_values(
                consumer.simulate(None).calculate(*key), reference[key], key
            )


def _comparison_indices(requests):
    """Compare the first two reform passes of each consumer to a whole file."""
    seen, selected = {}, set()
    for index, (name, _, _) in enumerate(requests):
        seen[name] = seen.get(name, 0) + 1
        if seen[name] <= 2:
            selected.add(index)
    return selected


def _score_reforms(builder, scorer, path, requests, start, count):
    from microcosm.build.us_runtime.reform_validation import default_simulate_factory

    consumer = scorer.open_consumer("shipped-reform-sweep", ())
    comparisons = _comparison_indices(requests)
    compared = []
    for index in range(start, min(start + count, len(requests))):
        name, reform, key = requests[index]
        observed = consumer.simulate(reform).calculate(*key)
        if index in comparisons:
            whole = default_simulate_factory(path)(reform)
            try:
                expected = whole.calculate(key[0], key[1], map_to=key[2])
                _assert_same_values(observed, expected, (name, index, key))
            finally:
                builder.release_engine_simulation(whole)
            compared.append(index)
        print(f"Guard sweep pass {index + 1}/{len(requests)}: {name} {key}", flush=True)
    return {"reform_passes": consumer.reform_passes, "whole_file_comparisons": compared}


def _peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _watch_worker_rss():
    def watch():
        while True:
            if _peak_rss_bytes() > 7 * 1024**3:
                try:
                    print(
                        "Guard sweep worker exceeded 7 GiB RSS",
                        file=sys.stderr,
                        flush=True,
                    )
                finally:
                    os._exit(2)
            time.sleep(0.1)

    threading.Thread(target=watch, daemon=True).start()


def _worker(args):
    _watch_worker_rss()
    started = time.monotonic()
    builder = _load_builder()
    plans, requests, counts = _record_requests(builder)
    scorer = builder._HouseholdBatchedPostExportScorer(
        args.dataset_path, maximum_microsim_batch_size=args.batch_size
    )
    if scorer.n_batches < 2:
        raise ValueError("The guard sweep needs at least two household batches")
    try:
        if args.worker_start is None:
            _score_baselines(builder, scorer, args.dataset_path, plans)
            report = {**counts, "total_reform_passes": len(requests)}
        else:
            report = _score_reforms(
                builder,
                scorer,
                args.dataset_path,
                requests,
                args.worker_start,
                args.reforms_per_worker,
            )
        report["n_batches"] = scorer.n_batches
        report["n_households"] = scorer.n_households
        report["dataset_sha256"] = scorer.dataset_sha256
        report["peak_rss_bytes"] = _peak_rss_bytes()
        report["elapsed_seconds"] = time.monotonic() - started
        return report
    finally:
        scorer.close()


def _run_workers(args, *, run=subprocess.run):
    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix="post-export-guard-sweep-", dir=args.output.parent
    ) as temporary:
        output = Path(temporary) / "worker.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--dataset-path",
            str(args.dataset_path),
            "--batch-size",
            str(args.batch_size),
            "--reforms-per-worker",
            str(args.reforms_per_worker),
            "--worker",
            "--output",
            str(output),
        ]
        run(command, check=True)
        report = json.loads(output.read_text())
        report["workers"] = [
            {
                "baseline": True,
                "peak_rss_bytes": report["peak_rss_bytes"],
                "elapsed_seconds": report["elapsed_seconds"],
            }
        ]
        report["scored_reform_passes"] = 0
        report["whole_file_comparisons"] = []
        for start in range(0, report["total_reform_passes"], args.reforms_per_worker):
            run([*command, "--worker-start", str(start)], check=True)
            worker = json.loads(output.read_text())
            report["workers"].append({"reform_start": start, **worker})
            report["scored_reform_passes"] += worker["reform_passes"]
            report["whole_file_comparisons"].extend(worker["whole_file_comparisons"])
        assert report["scored_reform_passes"] == report["total_reform_passes"]
        report["peak_rss_bytes"] = max(
            worker["peak_rss_bytes"] for worker in report["workers"]
        )
        report["elapsed_seconds"] = time.monotonic() - started
        return report


def _positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=_positive_integer, default=4)
    parser.add_argument(
        "--reforms-per-worker", type=_positive_integer, choices=range(1, 5), default=4
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-start", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = _worker(args) if args.worker else _run_workers(args)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
