#!/usr/bin/env python3
"""Evaluate one UK rowwise dataset-size candidate and write auditable receipts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from microcosm.build.uk_runtime.size_evaluation import (
    RowwiseRun,
    area_support_tables,
    dense_reference_deltas,
    fit_tables,
    footprint,
    frozen_vs_recomputed,
    gate_table,
    load_run,
    paired_targets,
    run_acceptance,
    summarize,
    weight_tables,
)

_REPO = Path(__file__).resolve().parents[1]
_STEPS = (
    "00-run-acceptance",
    "10-dense-reference",
    "20-vs-reference",
    "30-incumbent-score",
    "40-incumbent-surface",
    "50-downstream",
    "90-summary",
)
_CANDIDATE_NOTICE = (
    "These are candidate evaluations and untargeted diagnostics. No gate "
    "threshold is loosened or applied as a verdict; `--release-candidate` "
    "remains refused for size runs."
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--label")
    parser.add_argument("--reference", action="append", default=[])
    parser.add_argument("--spine", action="append", default=[])
    parser.add_argument("--incumbent-dir", type=Path)
    parser.add_argument(
        "--scoring-registry",
        type=Path,
        help=(
            "Frozen compiled scoring register (TargetRegistry JSON) for step 30. "
            "Defaults to <run>/scoring_registry_compiled.json when present, else the "
            "first reference's; candidate runs do not write one."
        ),
    )
    parser.add_argument("--ledger-facts", type=Path)
    parser.add_argument("--ledger-facts-sha256")
    parser.add_argument("--ledger-manifest-sha256")
    parser.add_argument("--eval-home", type=Path)
    parser.add_argument("--incumbent-h5", type=Path)
    parser.add_argument("--reforms-config", type=Path)
    parser.add_argument("--steps")
    parser.add_argument("--skip")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--skip-h5", action="store_true")
    return parser.parse_args(argv)


def _label_paths(values: Sequence[str], *, option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} must be LABEL=PATH: {value!r}")
        label, raw_path = value.split("=", 1)
        if not label or not raw_path or label in result:
            raise ValueError(f"invalid or duplicate {option}: {value!r}")
        result[label] = Path(raw_path).resolve()
    return result


def _selected_steps(args: argparse.Namespace) -> set[str]:
    selected = set(_STEPS)
    if args.steps:
        selected = {value.strip() for value in args.steps.split(",") if value.strip()}
    skipped: set[str] = set()
    if args.skip:
        skipped = {value.strip() for value in args.skip.split(",") if value.strip()}
    unknown = sorted((selected | skipped) - set(_STEPS))
    if unknown:
        raise ValueError(f"unknown evaluation step(s): {unknown}")
    return selected - skipped


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_markdown(path: Path, title: str, payload: object) -> None:
    path.write_text(
        f"# {title}\n\n{_CANDIDATE_NOTICE}\n\n```json\n"
        + json.dumps(payload, indent=2, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )


def _ensure_candidate_notice(path: Path) -> None:
    if not path.is_file():
        return
    content = path.read_text(encoding="utf-8")
    if _CANDIDATE_NOTICE not in content:
        path.write_text(_CANDIDATE_NOTICE + "\n\n" + content, encoding="utf-8")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _receipt(
    folder: Path,
    *,
    step: str,
    status: str,
    reason: str | None,
    argv: Sequence[str],
    started_at: str,
    started_clock: float,
    inputs: Mapping[str, Path],
    outputs: Mapping[str, Path],
    exit_code: int | None = 0,
    commands: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "step": step,
        "status": status,
        "reason": reason,
        "argv": list(argv),
        "cwd": str(_REPO),
        "started_at": started_at,
        "finished_at": _now(),
        "wall_seconds": time.perf_counter() - started_clock,
        "exit_code": exit_code,
        "inputs": {
            name: _artifact(path) for name, path in inputs.items() if path.is_file()
        },
        "outputs": {
            name: _artifact(path) for name, path in outputs.items() if path.is_file()
        },
    }
    if commands is not None:
        receipt["commands"] = list(commands)
    _write_json(folder / "receipt.json", receipt)
    return receipt


def _prepare(folder: Path) -> None:
    if folder.name not in _STEPS or folder.parent == Path(folder.anchor):
        raise ValueError(f"refusing to rewrite unsafe step folder: {folder}")
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)


def _h5_path(run: RowwiseRun) -> Path | None:
    outputs = run.manifest.get("outputs")
    if isinstance(outputs, Mapping):
        for record in outputs.values():
            if isinstance(record, Mapping):
                name = Path(str(record.get("path") or "")).name
                candidate = run.path / name
                if name.endswith(".h5") and candidate.is_file():
                    return candidate
    matches = sorted(run.path.glob("microcosm_uk_*_local.h5"))
    return matches[0] if matches else None


def _run_inputs(run: RowwiseRun, *, prefix: str) -> dict[str, Path]:
    inputs = {
        f"{prefix}_manifest": run.path / "rowwise_candidate_manifest.json",
        f"{prefix}_solve_diagnostics": run.path / "solve_diagnostics.csv",
        f"{prefix}_calibration_diagnostics": run.path / "calibration_diagnostics.json",
        f"{prefix}_area_support": run.path / "area_support_summary.csv",
    }
    gates = sorted(run.path.glob("*.local_gates.json"))
    if gates:
        inputs[f"{prefix}_gates"] = gates[0]
    h5 = _h5_path(run)
    if h5 is not None:
        inputs[f"{prefix}_h5"] = h5
    for name in ("dense_reference_diagnostics.csv", "dataset_size_selection.csv"):
        path = run.path / name
        if path.is_file():
            inputs[f"{prefix}_{path.stem}"] = path
    return inputs


def _spine(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return pd.read_hdf(path, "household", columns=["household_id", "household_weight"])


def _comparison(
    run: RowwiseRun,
    reference: RowwiseRun,
    *,
    run_spine: pd.DataFrame | None,
    reference_spine: pd.DataFrame | None,
) -> dict[str, Any]:
    return {
        "run": {
            "fit": fit_tables(run),
            "weights": weight_tables(run, spine_weights=run_spine),
            "areas": area_support_tables(run),
            "gates": gate_table(run),
            "footprint": footprint(run),
        },
        "reference": {
            "fit": fit_tables(reference),
            "weights": weight_tables(reference, spine_weights=reference_spine),
            "areas": area_support_tables(reference),
            "gates": gate_table(reference),
            "footprint": footprint(reference),
        },
        "paired_targets": paired_targets(run, reference),
    }


_RUSAGE_SHIM = (
    "import atexit, resource, runpy, sys\n"
    "script = sys.argv[1]\n"
    "sys.argv = [script, *sys.argv[2:]]\n"
    "def _report():\n"
    "    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
    "    if sys.platform != 'darwin':\n"
    "        peak *= 1024\n"
    "    sys.stderr.write(f'__MICROCOSM_MAXRSS_BYTES__={int(peak)}\\n')\n"
    "    sys.stderr.flush()\n"
    "atexit.register(_report)\n"
    "runpy.run_path(script, run_name='__main__')\n"
)
_MAXRSS_RE = re.compile(r"^__MICROCOSM_MAXRSS_BYTES__=([0-9]+)\s*$")


def _run_command(command: list[str], *, timed: bool = False) -> dict[str, Any]:
    """Run one external step; when timed, measure it portably.

    The timed form runs a Python script in a child interpreter through a
    small shim that executes the script in-process and reports the child's
    own peak resident set (bytes on every platform) on stderr at exit; wall
    time is measured here. No platform-specific ``time`` binary is involved,
    so the receipt is the same on macOS and on the linux CI runners, and the
    child's exit code is the script's own.
    """

    if timed:
        if len(command) < 2 or command[0] != sys.executable:
            raise ValueError("timed commands must be `<interpreter> <script> [args]`")
        argv = [sys.executable, "-c", _RUSAGE_SHIM, *command[1:]]
    else:
        argv = command
    started = time.perf_counter()
    completed = subprocess.run(
        argv,
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    wall = time.perf_counter() - started
    peak_rss: int | None = None
    stderr = completed.stderr
    if timed:
        kept: list[str] = []
        for line in stderr.splitlines():
            match = _MAXRSS_RE.match(line)
            if match:
                peak_rss = int(match.group(1))
            else:
                kept.append(line)
        stderr = "\n".join(kept)
    exit_code = completed.returncode
    return {
        "argv": command,
        "status": "ran" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "wall_seconds": wall,
        "peak_rss_bytes": peak_rss,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": stderr[-4000:],
    }


def _safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-") or "dataset"


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _engine_version() -> str | None:
    try:
        return importlib.metadata.version("policyengine-uk")
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    return payload if isinstance(payload, dict) else None


def _summary_markdown(summary: Mapping[str, Any], index: Mapping[str, Any]) -> str:
    lines = [
        "# UK dataset-size evaluation",
        "",
        _CANDIDATE_NOTICE,
        "",
        "## Identity",
        "",
    ]
    identity = summary["identity"]
    lines.extend(
        [
            f"- Label: `{identity['label']}`",
            f"- Run: `{identity['path']}`",
            f"- Git commit: `{identity.get('git_commit')}`",
            "",
            "## Pre-registered outcomes",
            "",
        ]
    )
    scorecard = summary["scorecard"]
    scorecard_columns = list(scorecard["columns"])
    lines.append(
        "| Metric | "
        + " | ".join(scorecard_columns)
        + " | Flag | Expectation | Red flag |"
    )
    lines.append("|---|" + "---:|" * len(scorecard_columns) + "---|---|---|")
    outcomes = {row["id"]: row for row in summary["pre_registered_outcomes"]}
    for scorecard_row in scorecard["rows"]:
        outcome = outcomes[scorecard_row["id"]]
        values = scorecard_row["values"]
        lines.append(
            f"| {scorecard_row['label']} | "
            + " | ".join(str(values[column]) for column in scorecard_columns)
            + f" | {outcome['flag']} | {outcome['expectation']} | "
            f"{outcome['red_flag']} |"
        )
    lines.extend(
        [
            "",
            "## Gates",
            "",
            "| Gate | Status | Criticality |",
            "|---|---|---|",
        ]
    )
    for row in summary["gates"]:
        lines.append(f"| {row['id']} | {row['status']} | {row['criticality']} |")
    lines.extend(
        [
            "",
            "## Per-grain fit",
            "",
            "| Grain | Rows | Within 10% | Within 25% | Median abs. error | Max abs. error |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for grain, row in summary["fit"]["by_grain"].items():
        lines.append(
            f"| {grain} | {row['n']} | {row['within_10pct']} | "
            f"{row['within_25pct']} | {row['median_abs']} | {row['max_abs']} |"
        )
    lines.extend(["", "## Paired deltas", ""])
    references = summary["references"]
    if references:
        lines.extend(
            [
                "| Reference | Common | Wins | Ties | Losses |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for label, row in references.items():
            lines.append(
                f"| {label} | {row['n_common']} | {row['wins']} | "
                f"{row['ties']} | {row['losses']} |"
            )
    else:
        lines.append("No external reference was evaluated.")
    lines.extend(
        [
            "",
            "## Area support breaches",
            "",
            "| Grain | Areas | Any breach | Codes |",
            "|---|---:|---:|---|",
        ]
    )
    for grain, row in summary["area_support"]["by_geography_level"].items():
        breaches = row["breaches"]
        lines.append(
            f"| {grain} | {row['n_areas']} | {breaches['any']} | "
            f"{', '.join(breaches['codes'])} |"
        )
    lines.extend(
        [
            "",
            "## Footprint",
            "",
            "```json",
            json.dumps(summary["footprint"], indent=2, sort_keys=True),
            "```",
            "",
            "## File index",
            "",
        ]
    )
    for name, record in sorted(index.items()):
        lines.append(f"- `{name}` — `{record['sha256']}` ({record['bytes']} bytes)")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(raw_argv)
    run_path = args.run.resolve()
    label = args.label or run_path.name
    reference_paths = _label_paths(args.reference, option="--reference")
    spine_paths = _label_paths(args.spine, option="--spine")
    selected = _selected_steps(args)
    out = (args.out or run_path / "evaluation").resolve()
    out.mkdir(parents=True, exist_ok=True)

    run = load_run(run_path, label=label, skip_h5=args.skip_h5)
    references = {
        reference_label: load_run(
            reference_path,
            label=reference_label,
            skip_h5=args.skip_h5,
        )
        for reference_label, reference_path in reference_paths.items()
    }
    spines = {spine_label: _spine(path) for spine_label, path in spine_paths.items()}
    previous_manifest = _load_optional_json(out / "evaluation_manifest.json") or {}
    step_state = dict(previous_manifest.get("steps") or {})

    def record(step: str, receipt: Mapping[str, Any]) -> None:
        step_state[step] = {
            "status": receipt["status"],
            "reason": receipt["reason"],
            "receipt": str((out / step / "receipt.json").resolve()),
        }

    if "00-run-acceptance" in selected:
        step = "00-run-acceptance"
        folder = out / step
        _prepare(folder)
        started_at, started_clock = _now(), time.perf_counter()
        solve = run.manifest.get("solve") or {}
        size = solve.get("dataset_size") if isinstance(solve, Mapping) else None
        size = size if isinstance(size, Mapping) else {}
        result = run_acceptance(
            run,
            expected_households=size.get("requested_households"),
            expected_pool=size.get("pool_households"),
            expected_epochs=(run.manifest.get("parameters") or {}).get("epochs"),
        )
        output = folder / "run_acceptance.json"
        _write_json(output, result)
        receipt = _receipt(
            folder,
            step=step,
            status="ran",
            reason=None,
            argv=raw_argv,
            started_at=started_at,
            started_clock=started_clock,
            inputs=_run_inputs(run, prefix="run"),
            outputs={"run_acceptance": output},
        )
        record(step, receipt)

    dense_result = dense_reference_deltas(run)
    if "10-dense-reference" in selected:
        step = "10-dense-reference"
        folder = out / step
        _prepare(folder)
        started_at, started_clock = _now(), time.perf_counter()
        outputs: dict[str, Path] = {}
        if dense_result is None:
            status, reason = "skipped", "run has no dense reference"
        else:
            status, reason = "ran", None
            json_path = folder / "size_only_deltas.json"
            md_path = folder / "size_only_deltas.md"
            _write_json(json_path, dense_result)
            _write_markdown(md_path, "Dense-reference deltas", dense_result)
            outputs = {"json": json_path, "markdown": md_path}
        receipt = _receipt(
            folder,
            step=step,
            status=status,
            reason=reason,
            argv=raw_argv,
            started_at=started_at,
            started_clock=started_clock,
            inputs={"dense_reference": run.path / "dense_reference_diagnostics.csv"},
            outputs=outputs,
        )
        record(step, receipt)

    if "20-vs-reference" in selected:
        step = "20-vs-reference"
        folder = out / step
        _prepare(folder)
        started_at, started_clock = _now(), time.perf_counter()
        outputs = {}
        if not references:
            status, reason = "skipped", "no --reference was supplied"
        else:
            status, reason = "ran", None
            for reference_label, reference in references.items():
                comparison = _comparison(
                    run,
                    reference,
                    run_spine=spines.get(label),
                    reference_spine=spines.get(reference_label),
                )
                reference_folder = folder / _safe_label(reference_label)
                reference_folder.mkdir()
                json_path = reference_folder / "comparison.json"
                md_path = reference_folder / "comparison.md"
                _write_json(json_path, comparison)
                _write_markdown(
                    md_path, f"Comparison with {reference_label}", comparison
                )
                outputs[f"{reference_label}_json"] = json_path
                outputs[f"{reference_label}_markdown"] = md_path
        receipt = _receipt(
            folder,
            step=step,
            status=status,
            reason=reason,
            argv=raw_argv,
            started_at=started_at,
            started_clock=started_clock,
            inputs={
                **_run_inputs(run, prefix="run"),
                **{
                    name: path
                    for key, value in references.items()
                    for name, path in _run_inputs(
                        value, prefix=f"reference_{key}"
                    ).items()
                },
                **{f"spine_{key}": value for key, value in spine_paths.items()},
            },
            outputs=outputs,
        )
        record(step, receipt)

    incumbent_score: dict[str, Any] | None = None
    if "30-incumbent-score" in selected:
        step = "30-incumbent-score"
        folder = out / step
        _prepare(folder)
        started_at, started_clock = _now(), time.perf_counter()
        registry_candidates = [
            *([args.scoring_registry] if args.scoring_registry is not None else []),
            run.path / "scoring_registry_compiled.json",
            *(
                reference.path / "scoring_registry_compiled.json"
                for reference in references.values()
            ),
        ]
        registry = next(
            (path for path in registry_candidates if path.is_file()),
            registry_candidates[-1]
            if registry_candidates
            else run.path / "scoring_registry_compiled.json",
        )
        outputs = {}
        command_rows: list[dict[str, Any]] = []
        score_inputs = {
            "candidate_diagnostics": run.path / "calibration_diagnostics.json",
            "registry": registry,
        }
        if args.incumbent_dir is None:
            status, reason, exit_code = (
                "skipped",
                "--incumbent-dir was not supplied",
                None,
            )
        elif not registry.is_file():
            status, reason, exit_code = (
                "skipped",
                "no frozen scoring register: pass --scoring-registry (candidate runs "
                "do not write scoring_registry_compiled.json)",
                None,
            )
        else:
            diagnostics = run.path / "calibration_diagnostics.json"
            incumbent_weights = args.incumbent_dir / "incumbent_local_weights.csv"
            incumbent_metrics = args.incumbent_dir / "incumbent_household_metrics.csv"
            score_inputs.update(
                {
                    "incumbent_weights": incumbent_weights,
                    "incumbent_metrics": incumbent_metrics,
                }
            )
            missing = [
                path.name
                for path in (incumbent_weights, incumbent_metrics)
                if not path.is_file()
            ]
            if missing:
                status, reason, exit_code = (
                    "failed",
                    f"missing incumbent input(s): {missing}",
                    1,
                )
            else:
                output = folder / "score_vs_incumbent.json"
                command = [
                    sys.executable,
                    str(_REPO / "tools/score_uk_local_candidate.py"),
                    "--candidate-diagnostics-json",
                    str(diagnostics),
                    "--candidate-diagnostics-sha256",
                    _sha256(diagnostics),
                    "--incumbent-weights-csv",
                    str(incumbent_weights),
                    "--incumbent-weights-sha256",
                    _sha256(incumbent_weights),
                    "--incumbent-household-metrics-csv",
                    str(incumbent_metrics),
                    "--incumbent-household-metrics-sha256",
                    _sha256(incumbent_metrics),
                    "--registry-json",
                    str(registry),
                    "--registry-sha256",
                    _sha256(registry),
                    "--output-json",
                    str(output),
                ]
                command_result = _run_command(command)
                command_rows.append(command_result)
                status = command_result["status"]
                exit_code = command_result["exit_code"]
                reason = (
                    command_result["stderr_tail"] or None
                    if status == "failed"
                    else None
                )
                if output.is_file():
                    outputs["score"] = output
                    incumbent_score = _load_optional_json(output)
        receipt = _receipt(
            folder,
            step=step,
            status=status,
            reason=reason,
            argv=raw_argv,
            started_at=started_at,
            started_clock=started_clock,
            inputs=score_inputs,
            outputs=outputs,
            exit_code=exit_code,
            commands=command_rows,
        )
        record(step, receipt)
    else:
        incumbent_score = _load_optional_json(
            out / "30-incumbent-score/score_vs_incumbent.json"
        )

    surface_result: dict[str, Any] | None = None
    if "40-incumbent-surface" in selected:
        step = "40-incumbent-surface"
        folder = out / step
        _prepare(folder)
        started_at, started_clock = _now(), time.perf_counter()
        outputs = {}
        command_rows = []
        candidate_h5 = _h5_path(run)
        surface_inputs = {
            "candidate_manifest": run.path / "rowwise_candidate_manifest.json"
        }
        if candidate_h5 is not None:
            surface_inputs["candidate_h5"] = candidate_h5
        if args.ledger_facts is not None:
            surface_inputs["ledger_facts"] = args.ledger_facts
        if args.incumbent_dir is not None:
            surface_inputs.update(
                {
                    "incumbent_manifest": args.incumbent_dir
                    / "incumbent_local_surface_manifest.json",
                    "incumbent_metrics": args.incumbent_dir
                    / "incumbent_household_metrics.csv",
                    "incumbent_weights": args.incumbent_dir
                    / "incumbent_local_weights.csv",
                }
            )
        if args.ledger_facts is None:
            status, reason, exit_code = (
                "skipped",
                "--ledger-facts was not supplied",
                None,
            )
        elif candidate_h5 is None:
            status, reason, exit_code = "failed", "run H5 is absent", 1
        elif not args.ledger_facts_sha256 or not args.ledger_manifest_sha256:
            status, reason, exit_code = (
                "failed",
                "ledger SHA-256 arguments are required",
                1,
            )
        else:
            output = folder / "incumbent_surface_evaluation.json"
            markdown = folder / "incumbent_surface_evaluation.md"
            command = [
                sys.executable,
                str(_REPO / "tools/evaluate_uk_incumbent_surface.py"),
                "--candidate-h5",
                str(candidate_h5),
                "--candidate-manifest",
                str(run.path / "rowwise_candidate_manifest.json"),
                "--ledger-facts",
                str(args.ledger_facts),
                "--ledger-facts-sha256",
                args.ledger_facts_sha256,
                "--ledger-manifest-sha256",
                args.ledger_manifest_sha256,
                "--engine-blocks",
                "1",
                "--out-json",
                str(output),
                "--out-md",
                str(markdown),
            ]
            if args.incumbent_dir is not None:
                command.extend(
                    [
                        "--incumbent-manifest",
                        str(
                            args.incumbent_dir / "incumbent_local_surface_manifest.json"
                        ),
                        "--incumbent-metrics-csv",
                        str(args.incumbent_dir / "incumbent_household_metrics.csv"),
                        "--incumbent-weights-csv",
                        str(args.incumbent_dir / "incumbent_local_weights.csv"),
                    ]
                )
            command_result = _run_command(command)
            command_rows.append(command_result)
            status = command_result["status"]
            exit_code = command_result["exit_code"]
            reason = (
                command_result["stderr_tail"] or None if status == "failed" else None
            )
            if status == "ran" and output.is_file():
                surface_result = frozen_vs_recomputed(run, output)
                frozen_path = folder / "frozen_vs_recomputed.json"
                _write_json(frozen_path, surface_result)
                _ensure_candidate_notice(markdown)
                outputs.update(
                    {"json": output, "markdown": markdown, "frozen": frozen_path}
                )
        receipt = _receipt(
            folder,
            step=step,
            status=status,
            reason=reason,
            argv=raw_argv,
            started_at=started_at,
            started_clock=started_clock,
            inputs=surface_inputs,
            outputs=outputs,
            exit_code=exit_code,
            commands=command_rows,
        )
        record(step, receipt)
    else:
        surface_result = _load_optional_json(
            out / "40-incumbent-surface/frozen_vs_recomputed.json"
        )

    downstream_result: dict[str, Any] | None = None
    if "50-downstream" in selected:
        step = "50-downstream"
        folder = out / step
        _prepare(folder)
        started_at, started_clock = _now(), time.perf_counter()
        outputs = {}
        command_rows = []
        downstream_inputs: dict[str, Path] = {}
        if args.eval_home is None:
            status, reason, exit_code = "skipped", "--eval-home was not supplied", None
        elif not references:
            status, reason, exit_code = (
                "skipped",
                "downstream comparison requires a --reference",
                None,
            )
        else:
            first_reference = next(iter(references.values()))
            candidate_h5 = _h5_path(run)
            reference_h5 = _h5_path(first_reference)
            if candidate_h5 is None or reference_h5 is None:
                status, reason, exit_code = (
                    "failed",
                    "candidate or reference H5 is absent",
                    1,
                )
            else:
                status, reason, exit_code = "ran", None, 0
                scripts = args.eval_home / "scripts"
                configs = args.eval_home / "config"
                downstream_inputs.update(
                    {
                        "candidate_h5": candidate_h5,
                        "reference_h5": reference_h5,
                        "compare_script": scripts / "compare_uk_datasets.py",
                        "admin_script": scripts / "t4_admin_benchmarks.py",
                        "scorecard_script": scripts / "scorecard_731.py",
                        "expected_divergences": configs / "expected_divergences.json",
                        "anchors": configs / "anchors_2025.json",
                    }
                )
                jobs: list[tuple[str, list[str], list[Path]]] = [
                    (
                        "compare_uk_datasets",
                        [
                            sys.executable,
                            str(scripts / "compare_uk_datasets.py"),
                            "--candidate-h5",
                            str(candidate_h5),
                            "--incumbent-h5",
                            str(reference_h5),
                            "--year",
                            "2025",
                            "--expected-divergences",
                            str(configs / "expected_divergences.json"),
                            "--out-json",
                            str(folder / "t3_distribution.json"),
                            "--out-md",
                            str(folder / "t3_distribution.md"),
                        ],
                        [
                            folder / "t3_distribution.json",
                            folder / "t3_distribution.md",
                        ],
                    ),
                    (
                        "t4_admin_benchmarks",
                        [
                            sys.executable,
                            str(scripts / "t4_admin_benchmarks.py"),
                            "--candidate-h5",
                            str(candidate_h5),
                            "--incumbent-h5",
                            str(reference_h5),
                            "--year",
                            "2025",
                            "--anchors",
                            str(configs / "anchors_2025.json"),
                            "--out",
                            str(folder / "t4_admin.json"),
                        ],
                        [folder / "t4_admin.json"],
                    ),
                ]
                if args.reforms_config is not None:
                    downstream_inputs.update(
                        {
                            "reforms_script": scripts / "t5_reforms.py",
                            "reforms_config": args.reforms_config,
                        }
                    )
                    jobs.append(
                        (
                            "t5_reforms",
                            [
                                sys.executable,
                                str(scripts / "t5_reforms.py"),
                                "--candidate-h5",
                                str(candidate_h5),
                                "--incumbent-h5",
                                str(reference_h5),
                                "--config",
                                str(args.reforms_config),
                                "--out",
                                str(folder / "t5_reforms.json"),
                            ],
                            [folder / "t5_reforms.json"],
                        )
                    )
                else:
                    command_rows.append(
                        {
                            "name": "t5_reforms",
                            "status": "skipped",
                            "reason": "--reforms-config was not supplied",
                            "wall_seconds": None,
                            "peak_rss_bytes": None,
                        }
                    )
                datasets: list[tuple[str, Path]] = [(label, candidate_h5)]
                datasets.extend(
                    (key, value)
                    for key, reference in references.items()
                    if (value := _h5_path(reference)) is not None
                )
                if args.incumbent_h5 is not None:
                    datasets.append(("incumbent", args.incumbent_h5.resolve()))
                    downstream_inputs["incumbent_h5"] = args.incumbent_h5.resolve()
                for dataset_label, h5 in datasets:
                    output = folder / f"scorecard_731_{_safe_label(dataset_label)}.json"
                    jobs.append(
                        (
                            f"scorecard_731:{dataset_label}",
                            [
                                sys.executable,
                                str(scripts / "scorecard_731.py"),
                                "--h5",
                                str(h5),
                                "--label",
                                dataset_label,
                                "--year",
                                "2026",
                                "--out",
                                str(output),
                            ],
                            [output],
                        )
                    )
                for job_name, command, job_outputs in jobs:
                    command_result = {
                        "name": job_name,
                        **_run_command(command, timed=True),
                    }
                    command_rows.append(command_result)
                    if command_result["status"] == "failed":
                        status, exit_code = "failed", int(command_result["exit_code"])
                        reason = f"{job_name} failed: {command_result['stderr_tail']}"
                    for path in job_outputs:
                        if path.is_file():
                            if path.suffix == ".md":
                                _ensure_candidate_notice(path)
                            outputs[path.name] = path
                downstream_result = {row["name"]: row for row in command_rows}
                footprint_path = folder / "footprint.json"
                _write_json(footprint_path, downstream_result)
                outputs["footprint"] = footprint_path
        receipt = _receipt(
            folder,
            step=step,
            status=status,
            reason=reason,
            argv=raw_argv,
            started_at=started_at,
            started_clock=started_clock,
            inputs=downstream_inputs,
            outputs=outputs,
            exit_code=exit_code,
            commands=command_rows,
        )
        record(step, receipt)
    else:
        downstream_result = _load_optional_json(out / "50-downstream/footprint.json")

    if "90-summary" in selected:
        step = "90-summary"
        folder = out / step
        _prepare(folder)
        started_at, started_clock = _now(), time.perf_counter()
        summary = summarize(
            run,
            references=references,
            dense_deltas=dense_result,
            incumbent_score=incumbent_score,
            surface=surface_result,
            downstream=downstream_result,
        )
        markdown_path = folder / "EVALUATION.md"
        existing_index = {
            str(path.relative_to(out)): _artifact(path)
            for path in out.rglob("*")
            if path.is_file() and path != folder / "evaluation_summary.json"
        }
        markdown_path.write_text(
            _summary_markdown(summary, existing_index), encoding="utf-8"
        )
        file_index = {
            str(path.relative_to(out)): _artifact(path)
            for path in out.rglob("*")
            if path.is_file() and path != folder / "evaluation_summary.json"
        }
        summary["files"] = file_index
        summary_path = folder / "evaluation_summary.json"
        _write_json(summary_path, summary)
        summary_inputs = {
            **_run_inputs(run, prefix="run"),
            **{
                name: path
                for key, value in references.items()
                for name, path in _run_inputs(value, prefix=f"reference_{key}").items()
            },
            **{
                f"evaluation_{str(path.relative_to(out)).replace('/', '_')}": path
                for path in out.rglob("*")
                if path.is_file() and folder not in path.parents
            },
        }
        receipt = _receipt(
            folder,
            step=step,
            status="ran",
            reason=None,
            argv=raw_argv,
            started_at=started_at,
            started_clock=started_clock,
            inputs=summary_inputs,
            outputs={"summary": summary_path, "markdown": markdown_path},
        )
        record(step, receipt)

    for step in _STEPS:
        step_state.setdefault(
            step,
            {"status": "skipped", "reason": "not selected", "receipt": None},
        )
    evaluation_manifest = {
        "schema_version": 1,
        "run": {"label": label, "path": str(run.path)},
        "references": [
            {"label": key, "path": str(value.path)} for key, value in references.items()
        ],
        "code_git_commit": _git_commit(),
        "engine_version": _engine_version(),
        "steps": {step: step_state[step] for step in _STEPS},
        "created_at": _now(),
    }
    _write_json(out / "evaluation_manifest.json", evaluation_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
