from __future__ import annotations

import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from microcosm.build.uk_runtime.national_frame import (
    UKStagingProvenance,
    _uk_source_file_fingerprint,
    uk_national_frame,
)
from microcosm.frame import MassChangeRecord, WeightKind


def _toy_result_frame():
    """A real one-household frame satisfying the driver's evidence reads."""

    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2],
                "person_benunit_id": [1, 1],
                "person_household_id": [1, 1],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1]}),
        household=pd.DataFrame({"household_id": [1], "household_weight": [2.0]}),
        time_period="2023",
        weight_kind=WeightKind.IMPORTANCE,
        mass_log=(
            MassChangeRecord(
                entity="household",
                old_total=2.0,
                new_total=2.0,
                declared_factor=1.0,
                reason="reviewed test mass allocation",
            ),
        ),
    )


_PATH_ARGUMENTS = (
    "evidence_path",
    "replay_path",
    "terminal_gate_path",
    "input_h5",
    "staging_h5",
    "spi_tab",
    "hmrc_ods",
    "adult_tab",
    "benefits_tab",
    "build_record_path",
    "input_mass_reference_path",
    "input_mass_exclusions_path",
    "qrf_tail_exclusions_path",
    "rung_abort_path",
)
_IDENTITY_CLI_ARGUMENTS = (
    "--release-id",
    "populace-uk-2023-frs-k535080",
    "--calibration-diagnostics-sha256",
    "c" * 64,
)


def _gate_result(*, passed: bool) -> SimpleNamespace:
    return SimpleNamespace(
        passed=passed,
        failures=() if passed else ("seeded coverage failure",),
        details={"required_columns": 145},
    )


def _terminal_gates(input_coverage: SimpleNamespace) -> SimpleNamespace:
    manifest = {
        "passed": bool(input_coverage.passed),
        "gates": {
            "uk_release_input_coverage": {
                "passed": bool(input_coverage.passed),
                "failures": list(input_coverage.failures),
                "details": dict(input_coverage.details),
            },
            "weight_ess": {
                "passed": True,
                "failures": [],
                "details": {"ess_fraction": 0.5},
            },
        },
    }
    return SimpleNamespace(
        passed=bool(input_coverage.passed),
        to_manifest=lambda: manifest,
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
    frs_raw_dir = tmp_path / "frs_2023_24"
    build_record_path = tmp_path / "national_staging_build_record.json"
    adult_tab = frs_raw_dir / "adult.tab"
    benefits_tab = frs_raw_dir / "benefits.tab"
    frs_raw_dir.mkdir()
    input_h5.write_bytes(b"base")
    spi_tab.write_bytes(b"spi")
    hmrc_ods.write_bytes(b"hmrc")
    adult_tab.write_bytes(b"adult")
    benefits_tab.write_bytes(b"benefits")
    calls = []
    replay_writes = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        kwargs["stages"][0].transform.last_result = SimpleNamespace(
            evidence=lambda: {"stage": "frs_hmrc_retained_leaves"}
        )
        kwargs["stages"][1].transform.last_result = SimpleNamespace(
            evidence=lambda: {"stage": "hmrc_spi_income"},
            replay_report=SimpleNamespace(summary={"excluded_with_fence": 208}),
        )
        staging_h5.write_bytes(b"staged")
        kwargs["terminal_gate_path"].write_text('{"passed": true}\n')
        input_coverage = _gate_result(passed=True)
        return SimpleNamespace(
            frame=_toy_result_frame(),
            provenance=UKStagingProvenance(
                source_h5=input_h5.resolve(),
                fingerprint=_uk_source_file_fingerprint(input_h5.resolve()),
            ),
            input_h5=input_h5.resolve(),
            staging_h5=staging_h5.resolve(),
            stage_names=(
                "frs_hmrc_retained_leaves",
                "hmrc_spi_income",
            ),
            terminal_gates=_terminal_gates(input_coverage),
            input_coverage=input_coverage,
            sampling_receipt=None,
        )

    monkeypatch.setattr(builder, "build_uk_national_dataset", fake_build)
    monkeypatch.setattr(
        builder,
        "write_hmrc_replay_report",
        lambda report, path: (
            replay_writes.append((report, Path(path))),
            Path(path).write_text('{"excluded_with_fence": 208}\n'),
        )[1],
    )
    monkeypatch.setattr(
        builder,
        "verify_certified_uk_candidate",
        lambda path: SimpleNamespace(
            path=Path(path).resolve(),
            filename="populace_uk_2023.h5",
            tier="frs",
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
            *_IDENTITY_CLI_ARGUMENTS,
            "--input-h5",
            str(input_h5),
            "--staging-h5",
            str(staging_h5),
            "--frs-raw-dir",
            str(frs_raw_dir),
            "--spi-tab",
            str(spi_tab),
            "--hmrc-ods",
            str(hmrc_ods),
            "--build-record-json",
            str(build_record_path),
        ],
    )

    assert builder.main() == 0

    assert len(calls) == 1
    assert calls[0]["input_h5"] == input_h5
    assert calls[0]["staging_h5"] == staging_h5
    assert calls[0]["release_id"] == "populace-uk-2023-frs-k535080"
    assert calls[0]["calibration_diagnostics_sha256"] == "c" * 64
    assert len(calls[0]["stages"]) == 2
    assert calls[0]["stages"][0].name == "frs_hmrc_retained_leaves"
    retained_transform = calls[0]["stages"][0].transform
    assert retained_transform.adult_tab_path == adult_tab
    assert retained_transform.benefits_tab_path == benefits_tab
    assert calls[0]["stages"][1].name == "hmrc_spi_income"
    hmrc_transform = calls[0]["stages"][1].transform
    assert hmrc_transform.spi_tab_path == spi_tab
    assert hmrc_transform.hmrc_ods_path == hmrc_ods
    assert hmrc_transform.certified_candidate.revision == "test-revision"
    assert hmrc_transform.retained_leaves_transform is retained_transform
    assert calls[0]["terminal_gate_path"] == staging_h5.with_suffix(
        ".terminal_gates.json"
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["build_kind"] == "uk_national_staging_dataset"
    assert payload["stages"] == [
        "frs_hmrc_retained_leaves",
        "hmrc_spi_income",
    ]
    assert payload["input_coverage"]["passed"] is True
    assert payload["terminal_gates"]["passed"] is True
    assert payload["terminal_gates"]["gates"]["weight_ess"]["passed"] is True
    assert payload["hmrc_replay"]["summary"] == {"excluded_with_fence": 208}
    assert payload["artifacts"]["staging_h5"]["sha256"]
    evidence_path = staging_h5.with_suffix(".hmrc_income.json")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["base_candidate"]["tier"] == "frs"
    assert evidence["base_candidate"]["revision"] == "test-revision"
    assert evidence["retained_leaves"]["stage"] == "frs_hmrc_retained_leaves"
    assert evidence["family"]["stage"] == "hmrc_spi_income"
    assert payload["artifacts"]["hmrc_evidence"]["sha256"]
    replay_path = staging_h5.with_suffix(".hmrc_replay.json")
    assert replay_writes == [(hmrc_transform.last_result.replay_report, replay_path)]
    assert payload["artifacts"]["hmrc_replay"]["sha256"]
    assert payload["artifacts"]["frs_adult"]["sha256"]
    assert payload["artifacts"]["frs_benefits"]["sha256"]
    assert payload["artifacts"]["terminal_gates"]["sha256"]
    assert payload["artifacts"]["build_record"]["sha256"]
    record = json.loads(build_record_path.read_text(encoding="utf-8"))
    assert record["status"] == "passed"
    assert record["terminal_gates"]["gates"]["weight_ess"]["passed"] is True
    assert record["dataset"] == {
        "entity_rows": {"benunit": 1, "household": 1, "person": 2},
        "household_weight_kind": "importance",
        "household_weight_total": 2.0,
        "mass_changes": [
            {
                "declared_factor": 1.0,
                "entity": "household",
                "new_total": 2.0,
                "old_total": 2.0,
                "reason": "reviewed test mass allocation",
            }
        ],
        "time_period": "2023",
    }
    assert record["artifacts"]["staging_h5"]["retention"] == "local_untracked"
    assert all("path" not in artifact for artifact in record["artifacts"].values())


def test_national_driver_writes_aggregate_reports_before_reraising_final_gate(
    monkeypatch,
    tmp_path,
) -> None:
    builder = _load_builder_module()
    input_h5 = tmp_path / "base.h5"
    staging_h5 = tmp_path / "staging.h5"
    spi_tab = tmp_path / "put2223uk.tab"
    hmrc_ods = tmp_path / "hmrc.ods"
    frs_raw_dir = tmp_path / "frs_2023_24"
    frs_raw_dir.mkdir()
    for path, content in (
        (input_h5, b"base"),
        (spi_tab, b"spi"),
        (hmrc_ods, b"hmrc"),
        (frs_raw_dir / "adult.tab", b"adult"),
        (frs_raw_dir / "benefits.tab", b"benefits"),
    ):
        path.write_bytes(content)

    replay_report = object()

    def fake_build(**kwargs):
        kwargs["stages"][0].transform.last_result = SimpleNamespace(
            evidence=lambda: {"stage": "frs_hmrc_retained_leaves"}
        )
        kwargs["stages"][1].transform.last_result = SimpleNamespace(
            evidence=lambda: {"stage": "hmrc_spi_income"},
            replay_report=replay_report,
        )
        kwargs["terminal_gate_path"].write_text('{"passed": false}\n')
        raise RuntimeError(
            "Release gates failed: [uk_release_input_coverage] gift_aid remains "
            "a reviewed exclusion with positive effective-mass signal"
        )

    monkeypatch.setattr(builder, "build_uk_national_dataset", fake_build)
    monkeypatch.setattr(
        builder,
        "verify_certified_uk_candidate",
        lambda path: SimpleNamespace(
            path=Path(path).resolve(),
            filename="populace_uk_2023.h5",
            tier="frs",
            revision="test-revision",
            sha256="a" * 64,
            size_bytes=4,
        ),
    )
    replay_calls = []

    def fake_write_replay(report, path):
        replay_calls.append((report, Path(path)))
        Path(path).write_text('{"excluded_with_fence": 208}\n')
        return Path(path)

    monkeypatch.setattr(builder, "write_hmrc_replay_report", fake_write_replay)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_uk_national_dataset.py",
            *_IDENTITY_CLI_ARGUMENTS,
            "--input-h5",
            str(input_h5),
            "--staging-h5",
            str(staging_h5),
            "--frs-raw-dir",
            str(frs_raw_dir),
            "--spi-tab",
            str(spi_tab),
            "--hmrc-ods",
            str(hmrc_ods),
        ],
    )

    with pytest.raises(RuntimeError, match="Release gates failed"):
        builder.main()

    evidence = json.loads(
        staging_h5.with_suffix(".hmrc_income.json").read_text(encoding="utf-8")
    )
    assert evidence["retained_leaves"]["stage"] == ("frs_hmrc_retained_leaves")
    assert evidence["family"]["stage"] == "hmrc_spi_income"
    assert replay_calls == [
        (replay_report, staging_h5.with_suffix(".hmrc_replay.json"))
    ]
    assert staging_h5.with_suffix(".terminal_gates.json").is_file()
    assert not staging_h5.exists()
    assert not staging_h5.with_suffix(".build.json").exists()


def test_national_driver_does_not_write_reports_for_stage_failure(
    monkeypatch,
    tmp_path,
) -> None:
    builder = _load_builder_module()
    input_h5 = tmp_path / "base.h5"
    staging_h5 = tmp_path / "staging.h5"
    spi_tab = tmp_path / "put2223uk.tab"
    hmrc_ods = tmp_path / "hmrc.ods"
    frs_raw_dir = tmp_path / "frs_2023_24"
    frs_raw_dir.mkdir()
    for path in (
        input_h5,
        spi_tab,
        hmrc_ods,
        frs_raw_dir / "adult.tab",
        frs_raw_dir / "benefits.tab",
    ):
        path.write_bytes(b"source")

    monkeypatch.setattr(
        builder,
        "build_uk_national_dataset",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("SPI donor identity mismatch")
        ),
    )
    monkeypatch.setattr(
        builder,
        "verify_certified_uk_candidate",
        lambda path: SimpleNamespace(
            path=Path(path).resolve(),
            filename="populace_uk_2023.h5",
            tier="frs",
            revision="test-revision",
            sha256="a" * 64,
            size_bytes=6,
        ),
    )
    monkeypatch.setattr(
        builder,
        "write_hmrc_replay_report",
        lambda *_args: pytest.fail("stage errors must not emit replay reports"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_uk_national_dataset.py",
            *_IDENTITY_CLI_ARGUMENTS,
            "--input-h5",
            str(input_h5),
            "--staging-h5",
            str(staging_h5),
            "--frs-raw-dir",
            str(frs_raw_dir),
            "--spi-tab",
            str(spi_tab),
            "--hmrc-ods",
            str(hmrc_ods),
        ],
    )

    with pytest.raises(RuntimeError, match="SPI donor identity mismatch"):
        builder.main()

    assert not staging_h5.with_suffix(".hmrc_income.json").exists()
    assert not staging_h5.with_suffix(".hmrc_replay.json").exists()
    assert not staging_h5.with_suffix(".build.json").exists()


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
    paths["terminal_gate_path"] = tmp_path / "candidate.h5"

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
    paths["terminal_gate_path"] = alias

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
    frs_raw_dir = tmp_path / "frs_2023_24"
    evidence = tmp_path / "evidence.json"
    frs_raw_dir.mkdir()
    input_h5.write_bytes(b"certified base")
    staging_h5.write_bytes(b"previous staging")
    spi_tab.write_bytes(b"licensed donor")
    hmrc_ods.write_bytes(b"official surface")
    (frs_raw_dir / "adult.tab").write_bytes(b"raw adult")
    (frs_raw_dir / "benefits.tab").write_bytes(b"raw benefits")
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
            *_IDENTITY_CLI_ARGUMENTS,
            "--input-h5",
            str(input_h5),
            "--staging-h5",
            str(staging_h5),
            "--frs-raw-dir",
            str(frs_raw_dir),
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


def test_national_driver_accepts_legacy_input_coverage_path_alias(
    monkeypatch,
    tmp_path,
) -> None:
    builder = _load_builder_module()
    legacy_path = tmp_path / "legacy-coverage.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_uk_national_dataset.py",
            *_IDENTITY_CLI_ARGUMENTS,
            "--input-h5",
            "base.h5",
            "--staging-h5",
            "staging.h5",
            "--frs-raw-dir",
            "frs_2023_24",
            "--spi-tab",
            "put2223uk.tab",
            "--hmrc-ods",
            "hmrc.ods",
            "--input-coverage-json",
            str(legacy_path),
        ],
    )

    args = builder._parse_args()

    assert args.input_coverage_json == legacy_path
    assert args.terminal_gates_json is None
    assert args.release_id == "populace-uk-2023-frs-k535080"
    assert args.calibration_diagnostics_sha256 == "c" * 64


def test_national_driver_forwards_legacy_output_to_compatibility_serializer(
    monkeypatch,
    tmp_path,
) -> None:
    builder = _load_builder_module()
    legacy_path = tmp_path / "legacy-coverage.json"
    calls = []

    class StopAfterForwardingError(Exception):
        pass

    def fake_build(**kwargs):
        calls.append(kwargs)
        raise StopAfterForwardingError

    monkeypatch.setattr(builder, "build_uk_national_dataset", fake_build)
    monkeypatch.setattr(
        builder,
        "verify_certified_uk_candidate",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_uk_national_dataset.py",
            *_IDENTITY_CLI_ARGUMENTS,
            "--input-h5",
            str(tmp_path / "base.h5"),
            "--staging-h5",
            str(tmp_path / "staging.h5"),
            "--frs-raw-dir",
            str(tmp_path / "frs_2023_24"),
            "--spi-tab",
            str(tmp_path / "put2223uk.tab"),
            "--hmrc-ods",
            str(tmp_path / "hmrc.ods"),
            "--input-coverage-json",
            str(legacy_path),
        ],
    )

    with pytest.raises(StopAfterForwardingError):
        builder.main()

    assert len(calls) == 1
    assert calls[0]["input_coverage_path"] == legacy_path
    assert "terminal_gate_path" not in calls[0]


def test_national_driver_rejects_both_terminal_gate_cli_names(
    monkeypatch,
    tmp_path,
) -> None:
    builder = _load_builder_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_uk_national_dataset.py",
            *_IDENTITY_CLI_ARGUMENTS,
            "--input-h5",
            "base.h5",
            "--staging-h5",
            "staging.h5",
            "--frs-raw-dir",
            "frs_2023_24",
            "--spi-tab",
            "put2223uk.tab",
            "--hmrc-ods",
            "hmrc.ods",
            "--terminal-gates-json",
            str(tmp_path / "terminal.json"),
            "--input-coverage-json",
            str(tmp_path / "legacy.json"),
        ],
    )

    with pytest.raises(SystemExit):
        builder._parse_args()


@pytest.mark.parametrize(
    "removed_flag",
    [
        "--spi-donor-sample-size",
        "--max-weight-ratio",
        "--maximum-abs-relative-error",
        "--calibration-epochs",
        "--calibration-learning-rate",
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
            *_IDENTITY_CLI_ARGUMENTS,
            "--input-h5",
            "base.h5",
            "--staging-h5",
            "staging.h5",
            "--frs-raw-dir",
            "frs_2023_24",
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


def test_stage_reports_survive_a_checkpoint_resumed_spi_stage(tmp_path):
    """The adversarial-review blocker: a resumed run must still write sidecars.

    A checkpoint-resumed SPI transform carries the rehydrated evidence
    surface instead of a live report object; the driver's stage reports
    must consume it, and the resumed replay sidecar must be byte-identical
    to what the payload serializes to.
    """

    import json
    from types import SimpleNamespace

    builder = _load_builder_module()
    from microcosm.build.uk_runtime.frs_hmrc_leaves import (
        UKFRSHMRCRetainedLeavesStageTransform,
        _ResumedRetainedLeaves,
    )
    from microcosm.build.uk_runtime.hmrc_restoration import (
        UKHMRCIncomeStageTransform,
    )

    retained = UKFRSHMRCRetainedLeavesStageTransform(
        adult_tab_path=tmp_path / "adult.tab",
        benefits_tab_path=tmp_path / "benefits.tab",
    )
    retained.last_result = _ResumedRetainedLeaves(
        frame=None,
        evidence_payload={"stage": "frs_hmrc_retained_leaves"},
        input_content_identity="a" * 64,
        output_content_identity="b" * 64,
    )
    hmrc = UKHMRCIncomeStageTransform(
        spi_tab_path=tmp_path / "put2223uk.tab",
        hmrc_ods_path=tmp_path / "hmrc.ods",
        certified_candidate=SimpleNamespace(),
    )
    from microcosm.build.uk_runtime.content_identity import (
        uk_frame_content_identity,
    )

    resumed_frame = _toy_result_frame()
    replay_payload = {"summary": {"status": "comparisons_passed"}, "facts": {}}
    hmrc.resume_from_checkpoint(
        {
            "fit_weight_records": [
                {"fit_name": "uk_spi_fill_qrf", "weight_kind": "design"}
            ],
            "evidence": {"stage": "hmrc_spi_income"},
            "replay_payload": replay_payload,
            "output_content_identity": uk_frame_content_identity(resumed_frame),
        },
        resumed_frame,
    )

    evidence_path = tmp_path / "evidence.json"
    replay_path = tmp_path / "replay.json"
    builder._write_stage_reports(
        evidence_path=evidence_path,
        replay_path=replay_path,
        candidate=SimpleNamespace(
            path=tmp_path / "candidate.h5",
            filename="candidate.h5",
            tier="frs",
            revision="r",
            sha256="c" * 64,
            size_bytes=1,
        ),
        retained_leaves_transform=retained,
        hmrc_transform=hmrc,
    )
    written = json.loads(replay_path.read_text())
    assert written == replay_payload
    evidence = json.loads(evidence_path.read_text())
    assert evidence["family"] == {"stage": "hmrc_spi_income"}
    assert builder._replay_summary(hmrc.last_result) == replay_payload["summary"]


def test_national_driver_rejects_non_rung_sample_fractions(monkeypatch) -> None:
    builder = _load_builder_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_uk_national_dataset.py",
            *_IDENTITY_CLI_ARGUMENTS,
            "--input-h5",
            "base.h5",
            "--staging-h5",
            "staging.h5",
            "--frs-raw-dir",
            "frs_2023_24",
            "--spi-tab",
            "put2223uk.tab",
            "--hmrc-ods",
            "hmrc.ods",
            "--sample-fraction",
            "0.2",
        ],
    )

    with pytest.raises(SystemExit):
        builder._parse_args()


def test_national_driver_refuses_canonical_release_ids_for_rung_builds(
    monkeypatch, capsys
) -> None:
    builder = _load_builder_module()
    # _IDENTITY_CLI_ARGUMENTS carries the canonical populace-uk-...-k id; a
    # sampled rung build must refuse it — rung artifacts are receipts, never
    # releases (#627).
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_uk_national_dataset.py",
            *_IDENTITY_CLI_ARGUMENTS,
            "--input-h5",
            "base.h5",
            "--staging-h5",
            "staging.h5",
            "--frs-raw-dir",
            "frs_2023_24",
            "--spi-tab",
            "put2223uk.tab",
            "--hmrc-ods",
            "hmrc.ods",
            "--sample-fraction",
            "0.01",
        ],
    )

    with pytest.raises(SystemExit):
        builder._parse_args()
    assert "non-releasable" in capsys.readouterr().err


def test_staging_run_config_pins_the_sampling_identity(monkeypatch, tmp_path) -> None:
    builder = _load_builder_module()
    for name in ("adult.tab", "benefits.tab", "put2223uk.tab", "hmrc.ods"):
        (tmp_path / name).write_bytes(b"x")
    import microcosm.build.code_identity as code_identity_module

    monkeypatch.setattr(
        code_identity_module,
        "builder_code_identity",
        lambda *args, **kwargs: {"stub": True},
    )
    args = SimpleNamespace(
        release_id="uk-dev-rung",
        calibration_diagnostics_sha256="c" * 64,
        seed=42,
        qrf_estimators=100,
        sample_fraction=0.01,
        sample_seed=7,
    )
    retained = SimpleNamespace(
        adult_tab_path=tmp_path / "adult.tab",
        benefits_tab_path=tmp_path / "benefits.tab",
    )
    hmrc = SimpleNamespace(
        spi_tab_path=tmp_path / "put2223uk.tab",
        hmrc_ods_path=tmp_path / "hmrc.ods",
    )
    candidate = SimpleNamespace(sha256="a" * 64, size_bytes=3)

    config = builder._staging_run_config(
        args,
        candidate=candidate,
        retained_leaves_transform=retained,
        hmrc_transform=hmrc,
    )

    # The fraction is a string on purpose: run-config equality is exact over
    # canonical JSON, and float normalization is exactly the ambiguity a run
    # identity must not carry. Two rungs on one checkpoint directory refuse
    # instead of cross-resuming.
    assert config["sampling"] == {
        "sample_fraction": "0.01",
        "sample_seed": 7,
        "rung_token": "f001",
    }


def _rung_abort_argv(tmp_path: Path, *, fraction: str) -> list[str]:
    frs_raw_dir = tmp_path / "frs_2023_24"
    frs_raw_dir.mkdir(exist_ok=True)
    for name in ("adult.tab", "benefits.tab"):
        (frs_raw_dir / name).write_bytes(b"x")
    for name in ("base.h5", "put2223uk.tab", "hmrc.ods"):
        (tmp_path / name).write_bytes(b"x")
    return [
        "build_uk_national_dataset.py",
        "--release-id",
        "uk-dev-rung",
        "--calibration-diagnostics-sha256",
        "c" * 64,
        "--input-h5",
        str(tmp_path / "base.h5"),
        "--staging-h5",
        str(tmp_path / "staging.h5"),
        "--frs-raw-dir",
        str(frs_raw_dir),
        "--spi-tab",
        str(tmp_path / "put2223uk.tab"),
        "--hmrc-ods",
        str(tmp_path / "hmrc.ods"),
        "--sample-fraction",
        fraction,
    ]


def _named_edge_error() -> ValueError:
    return ValueError(
        "The least populated classes in y have only 1 member, which is too "
        "few. The minimum number of groups for any class cannot be less "
        "than 2. Classes with too few members are: [0.0]"
    )


def _install_rung_abort_seams(builder, monkeypatch, error: Exception) -> None:
    monkeypatch.setattr(
        builder,
        "verify_certified_uk_candidate",
        lambda path: SimpleNamespace(
            path=Path(path).resolve(),
            filename="populace_uk_2023.h5",
            tier="frs",
            revision="test-revision",
            sha256="a" * 64,
            size_bytes=1,
        ),
    )

    def raising_build(**_kwargs):
        raise error

    monkeypatch.setattr(builder, "build_uk_national_dataset", raising_build)


def test_rung_named_edge_aborts_with_a_receipt(monkeypatch, tmp_path) -> None:
    """The one named dev-scale edge (#657) receipts instead of crashing."""

    builder = _load_builder_module()
    _install_rung_abort_seams(builder, monkeypatch, _named_edge_error())
    monkeypatch.setattr(sys, "argv", _rung_abort_argv(tmp_path, fraction="0.10"))

    assert builder.main() == builder._RUNG_ABORT_EXIT_CODE

    receipt = json.loads((tmp_path / "staging.rung_abort.json").read_text())
    assert receipt["named_edge"] == "spi_split_singleton_class"
    assert receipt["disposition"] == "aborted_with_receipt"
    assert receipt["sampling"]["rung_token"] == "f010"
    assert "least populated classes" in receipt["error"]


def test_full_scale_named_edge_still_crashes(monkeypatch, tmp_path) -> None:
    builder = _load_builder_module()
    _install_rung_abort_seams(builder, monkeypatch, _named_edge_error())
    monkeypatch.setattr(sys, "argv", _rung_abort_argv(tmp_path, fraction="1.0"))

    with pytest.raises(ValueError, match="least populated classes"):
        builder.main()
    assert not (tmp_path / "staging.rung_abort.json").exists()


def test_rung_unknown_exception_still_crashes(monkeypatch, tmp_path) -> None:
    """Only the named edge is receipted — the path cannot absorb defects."""

    builder = _load_builder_module()
    _install_rung_abort_seams(
        builder, monkeypatch, ValueError("some entirely different failure")
    )
    monkeypatch.setattr(sys, "argv", _rung_abort_argv(tmp_path, fraction="0.10"))

    with pytest.raises(ValueError, match="entirely different"):
        builder.main()
    assert not (tmp_path / "staging.rung_abort.json").exists()
