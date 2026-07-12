from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


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
    spi_tab = tmp_path / "put2223uk.tab"
    hmrc_ods = tmp_path / "hmrc.ods"
    input_h5.write_bytes(b"base")
    spi_tab.write_bytes(b"spi")
    hmrc_ods.write_bytes(b"hmrc")
    calls = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        kwargs["stages"][0].transform.last_result = SimpleNamespace(
            evidence=lambda: {"stage": "hmrc_spi_income"}
        )
        staging_h5.write_bytes(b"staged")
        kwargs["input_coverage_path"].write_text('{"passed": true}\n')
        return SimpleNamespace(
            input_h5=input_h5.resolve(),
            staging_h5=staging_h5.resolve(),
            stage_names=("hmrc_spi_income",),
            input_coverage=SimpleNamespace(
                passed=True,
                failures=(),
                details={"required_columns": 132},
            ),
        )

    monkeypatch.setattr(builder, "build_uk_national_dataset", fake_build)
    monkeypatch.setattr(
        builder,
        "verify_certified_uk_candidate",
        lambda path: SimpleNamespace(
            path=Path(path).resolve(),
            filename="populace_uk_2023.h5",
            revision="test-revision",
            sha256="a" * 64,
            size_bytes=4,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_uk_national_dataset.py",
            "--input-h5",
            str(input_h5),
            "--staging-h5",
            str(staging_h5),
            "--spi-tab",
            str(spi_tab),
            "--hmrc-ods",
            str(hmrc_ods),
        ],
    )

    assert builder.main() == 0

    assert len(calls) == 1
    assert calls[0]["input_h5"] == input_h5
    assert calls[0]["staging_h5"] == staging_h5
    assert len(calls[0]["stages"]) == 1
    assert calls[0]["stages"][0].name == "hmrc_spi_income"
    transform = calls[0]["stages"][0].transform
    assert transform.spi_tab_path == spi_tab
    assert transform.hmrc_ods_path == hmrc_ods
    assert transform.certified_candidate.revision == "test-revision"
    assert calls[0]["input_coverage_path"] == staging_h5.with_suffix(
        ".input_coverage.json"
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["build_kind"] == "uk_national_staging_dataset"
    assert payload["stages"] == ["hmrc_spi_income"]
    assert payload["input_coverage"]["passed"] is True
    assert payload["artifacts"]["staging_h5"]["sha256"]
    evidence_path = staging_h5.with_suffix(".hmrc_income.json")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["base_candidate"]["revision"] == "test-revision"
    assert evidence["family"]["stage"] == "hmrc_spi_income"
    assert payload["artifacts"]["hmrc_evidence"]["sha256"]


@pytest.mark.parametrize(
    "removed_flag",
    [
        "--spi-donor-sample-size",
        "--max-weight-ratio",
        "--maximum-abs-relative-error",
    ],
)
def test_national_driver_rejects_unreviewed_release_overrides(
    monkeypatch,
    removed_flag,
) -> None:
    builder = _load_builder_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_uk_national_dataset.py",
            "--input-h5",
            "base.h5",
            "--staging-h5",
            "staging.h5",
            "--spi-tab",
            "put2223uk.tab",
            "--hmrc-ods",
            "hmrc.ods",
            removed_flag,
            "10",
        ],
    )

    with pytest.raises(SystemExit):
        builder._parse_args()
