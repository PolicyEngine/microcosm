"""End-to-end tests for the UK dataset-size evaluation command."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_tool():
    path = Path(__file__).resolve().parents[3] / "tools/evaluate_uk_dataset_size.py"
    return _load_module("evaluate_uk_dataset_size", path)


def _write_run_dir(tmp_path: Path, label: str, *, size_run: bool) -> Path:
    fixtures = _load_module(
        "_uk_size_evaluation_fixtures",
        Path(__file__).with_name("test_uk_size_evaluation.py"),
    )
    return fixtures._write_run_dir(tmp_path, label, size_run=size_run)


def test_main_writes_selected_tree_receipts_manifest_and_summary(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    run = _write_run_dir(tmp_path, "candidate", size_run=True)
    reference = _write_run_dir(tmp_path, "reference", size_run=False)
    out = tmp_path / "evaluation"

    assert (
        tool.main(
            [
                "--run",
                str(run),
                "--reference",
                f"R17={reference}",
                "--steps",
                "00-run-acceptance,10-dense-reference,20-vs-reference,90-summary",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    expected = (
        out / "00-run-acceptance/run_acceptance.json",
        out / "10-dense-reference/size_only_deltas.json",
        out / "10-dense-reference/size_only_deltas.md",
        out / "20-vs-reference/R17/comparison.json",
        out / "20-vs-reference/R17/comparison.md",
        out / "90-summary/evaluation_summary.json",
        out / "90-summary/EVALUATION.md",
    )
    assert all(path.is_file() for path in expected)
    for step in (
        "00-run-acceptance",
        "10-dense-reference",
        "20-vs-reference",
        "90-summary",
    ):
        receipt = json.loads((out / step / "receipt.json").read_text())
        assert receipt["status"] == "ran"
        assert receipt["step"] == step
        assert receipt["argv"]
    manifest = json.loads((out / "evaluation_manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["run"]["label"] == "candidate"
    assert manifest["references"][0]["label"] == "R17"
    assert manifest["steps"]["30-incumbent-score"]["status"] == "skipped"
    summary = json.loads(expected[-2].read_text())
    assert summary["references"]["R17"]["n_common"] == 10
    assert "00-run-acceptance/run_acceptance.json" in summary["files"]
    markdown = expected[-1].read_text()
    assert "candidate evaluations and untargeted diagnostics" in markdown
    assert "--release-candidate" in markdown


def test_absent_incumbent_inputs_skip_steps_30_and_40(tmp_path: Path) -> None:
    tool = _load_tool()
    run = _write_run_dir(tmp_path, "candidate", size_run=True)
    out = tmp_path / "evaluation"

    assert (
        tool.main(
            [
                "--run",
                str(run),
                "--steps",
                "30-incumbent-score,40-incumbent-surface",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    score = json.loads((out / "30-incumbent-score/receipt.json").read_text())
    surface = json.loads((out / "40-incumbent-surface/receipt.json").read_text())
    assert score["status"] == surface["status"] == "skipped"
    assert "--incumbent-dir" in score["reason"]
    assert "--ledger-facts" in surface["reason"]


def _write_downstream_stubs(eval_home: Path) -> None:
    scripts = eval_home / "scripts"
    config = eval_home / "config"
    scripts.mkdir(parents=True)
    config.mkdir()
    (config / "expected_divergences.json").write_text("{}\n")
    (config / "anchors_2025.json").write_text("{}\n")
    body = """\
import json
import sys
from pathlib import Path
import pytest

# The evaluation reads and writes PyTables-format H5 through pandas; the wheels
# lane's venv has no pytables, so these tests skip there like the other H5 tests.
pytest.importorskip("tables", exc_type=ModuleNotFoundError)

for flag in ("--out-json", "--out-md", "--out"):
    if flag not in sys.argv:
        continue
    output = Path(sys.argv[sys.argv.index(flag) + 1])
    if output.suffix == ".md":
        output.write_text("# stub\\n")
    else:
        output.write_text(json.dumps({"script": Path(__file__).stem}) + "\\n")
if Path(__file__).stem == "t4_admin_benchmarks":
    print("intentional t4 failure", file=sys.stderr)
    raise SystemExit(7)
"""
    for name in (
        "compare_uk_datasets.py",
        "t4_admin_benchmarks.py",
        "t5_reforms.py",
        "scorecard_731.py",
    ):
        (scripts / name).write_text(body)


def test_downstream_continues_after_a_failing_stub(tmp_path: Path) -> None:
    tool = _load_tool()
    run = _write_run_dir(tmp_path, "candidate", size_run=True)
    reference = _write_run_dir(tmp_path, "reference", size_run=False)
    eval_home = tmp_path / "eval-home"
    _write_downstream_stubs(eval_home)
    reforms = tmp_path / "reforms.json"
    reforms.write_text("{}\n")
    out = tmp_path / "evaluation"

    assert (
        tool.main(
            [
                "--run",
                str(run),
                "--reference",
                f"R17={reference}",
                "--eval-home",
                str(eval_home),
                "--reforms-config",
                str(reforms),
                "--steps",
                "50-downstream",
                "--out",
                str(out),
            ]
        )
        == 0
    )

    receipt = json.loads((out / "50-downstream/receipt.json").read_text())
    footprint = json.loads((out / "50-downstream/footprint.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["exit_code"] == 7
    assert "intentional t4 failure" in receipt["reason"]
    assert footprint["t4_admin_benchmarks"]["status"] == "failed"
    assert footprint["t5_reforms"]["status"] == "ran"
    assert footprint["scorecard_731:candidate"]["status"] == "ran"
    assert footprint["scorecard_731:R17"]["status"] == "ran"
    assert (out / "50-downstream/t3_distribution.json").is_file()
    assert (out / "50-downstream/t5_reforms.json").is_file()
    assert (out / "50-downstream/scorecard_731_candidate.json").is_file()


def test_rerunning_one_step_rewrites_only_its_folder(tmp_path: Path) -> None:
    tool = _load_tool()
    run = _write_run_dir(tmp_path, "candidate", size_run=True)
    out = tmp_path / "evaluation"
    common = ["--run", str(run), "--out", str(out)]

    assert tool.main([*common, "--steps", "00-run-acceptance,10-dense-reference"]) == 0
    rewritten = out / "00-run-acceptance/rewrite-me"
    preserved = out / "10-dense-reference/preserve-me"
    rewritten.write_text("old")
    preserved.write_text("old")

    assert tool.main([*common, "--steps", "00-run-acceptance"]) == 0
    assert not rewritten.exists()
    assert preserved.read_text() == "old"
    manifest = json.loads((out / "evaluation_manifest.json").read_text())
    assert manifest["steps"]["10-dense-reference"]["status"] == "ran"
