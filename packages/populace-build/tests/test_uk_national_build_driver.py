from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_builder_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_uk_national_dataset.py"
    spec = importlib.util.spec_from_file_location("build_uk_national_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_national_build_driver_uses_standalone_national_seam(
    monkeypatch, tmp_path, capsys
) -> None:
    builder = _load_builder_module()
    input_h5 = tmp_path / "base.h5"
    staging_h5 = tmp_path / "staging.h5"
    input_h5.write_bytes(b"base")
    calls = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        staging_h5.write_bytes(b"staged")
        kwargs["input_coverage_path"].write_text('{"passed": true}\n')
        return SimpleNamespace(
            input_h5=input_h5.resolve(),
            staging_h5=staging_h5.resolve(),
            stage_names=(),
            input_coverage=SimpleNamespace(
                passed=True,
                failures=(),
                details={"required_columns": 132},
            ),
        )

    monkeypatch.setattr(builder, "build_uk_national_dataset", fake_build)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_uk_national_dataset.py",
            "--input-h5",
            str(input_h5),
            "--staging-h5",
            str(staging_h5),
        ],
    )

    assert builder.main() == 0

    assert len(calls) == 1
    assert calls[0]["input_h5"] == input_h5
    assert calls[0]["staging_h5"] == staging_h5
    assert calls[0]["stages"] == ()
    assert calls[0]["input_coverage_path"] == staging_h5.with_suffix(
        ".input_coverage.json"
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["build_kind"] == "uk_national_staging_dataset"
    assert payload["stages"] == []
    assert payload["input_coverage"]["passed"] is True
    assert payload["artifacts"]["staging_h5"]["sha256"]
