from __future__ import annotations

import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH_ARGUMENTS = (
    "evidence_path",
    "coverage_path",
    "input_h5",
    "staging_h5",
    "spi_tab",
    "hmrc_ods",
)


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
                details={"required_columns": 145},
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
    ("left", "right"),
    list(combinations(_PATH_ARGUMENTS, 2)),
    ids=lambda value: value,
)
def test_national_driver_requires_every_input_output_path_to_be_distinct(
    tmp_path,
    left,
    right,
) -> None:
    builder = _load_builder_module()
    paths = {name: tmp_path / f"{name}.artifact" for name in _PATH_ARGUMENTS}
    collision = tmp_path / "collision.artifact"
    paths[left] = collision
    paths[right] = collision

    with pytest.raises(ValueError, match="pairwise distinct") as error:
        builder._validate_distinct_paths(**paths)

    message = str(error.value)
    assert collision.as_posix() in message


def test_national_driver_rejects_case_only_path_aliases(tmp_path) -> None:
    builder = _load_builder_module()
    candidate = tmp_path / "Candidate.H5"
    candidate.write_bytes(b"certified base")
    paths = {name: tmp_path / f"{name}.artifact" for name in _PATH_ARGUMENTS}
    paths["input_h5"] = candidate
    paths["coverage_path"] = tmp_path / "candidate.h5"

    with pytest.raises(ValueError, match="pairwise distinct"):
        builder._validate_distinct_paths(**paths)

    assert candidate.read_bytes() == b"certified base"


def test_national_driver_rejects_existing_hardlink_aliases(tmp_path) -> None:
    builder = _load_builder_module()
    candidate = tmp_path / "candidate.h5"
    alias = tmp_path / "coverage.json"
    candidate.write_bytes(b"certified base")
    alias.hardlink_to(candidate)
    paths = {name: tmp_path / f"{name}.artifact" for name in _PATH_ARGUMENTS}
    paths["input_h5"] = candidate
    paths["coverage_path"] = alias

    with pytest.raises(ValueError, match="pairwise distinct"):
        builder._validate_distinct_paths(**paths)


def test_national_driver_rejects_source_sidecar_collision_before_unlink(
    monkeypatch,
    tmp_path,
) -> None:
    builder = _load_builder_module()
    input_h5 = tmp_path / "base.h5"
    staging_h5 = tmp_path / "staging.h5"
    spi_tab = tmp_path / "put2223uk.tab"
    hmrc_ods = tmp_path / "hmrc.ods"
    evidence = tmp_path / "evidence.json"
    input_h5.write_bytes(b"certified base")
    staging_h5.write_bytes(b"previous staging")
    spi_tab.write_bytes(b"licensed donor")
    hmrc_ods.write_bytes(b"official surface")
    evidence.write_bytes(b"previous evidence")
    monkeypatch.setattr(
        builder,
        "verify_certified_uk_candidate",
        lambda _path: pytest.fail("path validation must precede candidate hashing"),
    )
    monkeypatch.setattr(
        builder,
        "build_uk_national_dataset",
        lambda **_kwargs: pytest.fail("a colliding path must not start the build"),
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
            "--input-coverage-json",
            str(spi_tab),
            "--hmrc-evidence-json",
            str(evidence),
        ],
    )

    with pytest.raises(ValueError, match="pairwise distinct"):
        builder.main()

    assert input_h5.read_bytes() == b"certified base"
    assert staging_h5.read_bytes() == b"previous staging"
    assert spi_tab.read_bytes() == b"licensed donor"
    assert hmrc_ods.read_bytes() == b"official surface"
    assert evidence.read_bytes() == b"previous evidence"


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
