"""The child-support gate diagnosis reads a base checkpoint and never writes it."""

import hashlib
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build.frame_checkpoint import write_frame_checkpoint
from microcosm.build.us_runtime.child_support import with_us_child_support_inputs
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


def _load_tool_module():
    path = _TEST_PATHS.repository / "tools" / "diagnose_us_child_support_gate.py"
    spec = importlib.util.spec_from_file_location(
        "diagnose_us_child_support_gate", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _pooled_frame(rows: int) -> Frame:
    rng = np.random.default_rng(0)
    ids = np.arange(1, rows + 1, dtype="int64")
    age = rng.integers(18, 80, rows)
    wages = rng.gamma(2.0, 25_000.0, rows)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids * 10,
            "person_tax_unit_id": ids * 100,
            "person_spm_unit_id": ids * 1_000,
            "person_family_id": ids * 10_000,
            "person_marital_unit_id": ids * 100_000,
            "CSP_VAL": np.where(rng.random(rows) < 0.03, 4_000.0, 0.0),
            "CHSP_VAL": np.where((rng.random(rows) < 0.04) & (age < 55), 3_000.0, 0.0),
            "WSAL_VAL": wages,
            "SEMP_VAL": np.zeros(rows),
            "employment_income_before_lsr": wages,
            "self_employment_income_before_lsr": np.zeros(rows),
            "age": age,
            "is_female": rng.random(rows) < 0.5,
            "has_esi": rng.random(rows) < 0.5,
            "tax_unit_role_input": ["HEAD"] * rows,
            "social_security_retirement": np.zeros(rows),
            "social_security_disability": np.zeros(rows),
            "social_security_dependents": np.zeros(rows),
            "social_security_survivors": np.zeros(rows),
            "source_year": rng.choice([2023, 2024, 2025], rows),
        }
    )
    tables = {
        entity: pd.DataFrame({f"{entity}_id": person[f"person_{entity}_id"]})
        for entity in ("household", "tax_unit", "spm_unit", "family", "marital_unit")
    }
    tables["person"] = person
    tables["tax_unit"]["filing_status_input"] = ["SINGLE"] * rows
    frame = Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(rng.uniform(500.0, 1_500.0, rows), WeightKind.DESIGN)},
    )
    direct = with_us_child_support_inputs(frame, seed=0, time_period=2024)
    return clone_us_frame_for_puf_support(direct)


def _checkpoint_directory(tmp_path, frame: Frame):
    root = tmp_path / "base-checkpoints"
    root.mkdir()
    path = root / "010.frame.h5"
    write_frame_checkpoint(path, frame)
    record = {
        "stage": "prior_year_income_post_clone",
        "checkpoint_filename": path.name,
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (root / "stage_run_context.json").write_text(
        json.dumps(
            {
                "run_config": {"seed": 3, "target_year": 2024},
                "stage_records": [record],
            }
        )
    )
    return root


def test_diagnosis_reruns_the_stage_and_leaves_the_checkpoints_alone(
    tmp_path,
) -> None:
    tool = _load_tool_module()
    root = _checkpoint_directory(tmp_path, _pooled_frame(1_200))
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    out = tmp_path / "diagnosis.json"

    assert tool.main([str(root), "--seeds", "1", "--out", str(out)]) == 0

    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
    report = json.loads(out.read_text())
    assert report["build_seed"] == 3
    runs = report["runs"]
    assert [(run["seed"], run["gate_calibration"]) for run in runs] == [
        (3, "in_sample (the build's)"),
        (4, "in_sample (the build's)"),
        (3, "cross_fitted"),
    ]
    for run in runs:
        assert run["training_sample"]["rows"] == 1_200
        for output in ("child_support_expense", "child_support_received"):
            for channel in ("asec", "puf_tax_detail"):
                entry = run[output][channel]
                assert 0.0 <= entry["share"] <= 1.0
                assert set(entry["by_income_year"]) == {"2023", "2024", "2025"}
    assert runs[0]["gate_outcomes"]["child_support_expense"] == "in_sample"
    # The ASEC channel carries the survey values, so no seed moves it.
    asec = {run["child_support_expense"]["asec"]["share"] for run in runs}
    assert len(asec) == 1


def test_diagnosis_refuses_a_checkpoint_that_changed(tmp_path) -> None:
    tool = _load_tool_module()
    root = _checkpoint_directory(tmp_path, _pooled_frame(300))
    with (root / "010.frame.h5").open("ab") as handle:
        handle.write(b"\0")

    with pytest.raises(SystemExit, match="differs from its run context"):
        tool.main([str(root), "--skip-cross-fitted", "--seeds", "0"])


def test_diagnosis_refuses_an_output_inside_the_checkpoints(tmp_path) -> None:
    tool = _load_tool_module()
    root = _checkpoint_directory(tmp_path, _pooled_frame(300))

    with pytest.raises(SystemExit, match="outside the checkpoint directory"):
        tool.main(
            [
                str(root),
                "--skip-cross-fitted",
                "--seeds",
                "0",
                "--out",
                str(root / "diagnosis.json"),
            ]
        )
