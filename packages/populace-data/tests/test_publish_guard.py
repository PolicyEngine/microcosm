import json
from pathlib import Path

from populace.data.publish_cli import _reform_validation_skipped, main


def _write_rv(release_dir: Path, *, out_of_sample_simulated: bool | None) -> None:
    payload: dict = {"schema_version": 1, "reforms": []}
    if out_of_sample_simulated is not None:
        payload["out_of_sample_simulated"] = out_of_sample_simulated
    (release_dir / "reform_validation.json").write_text(json.dumps(payload))


def test_skipped_detects_false_flag(tmp_path):
    _write_rv(tmp_path, out_of_sample_simulated=False)
    assert _reform_validation_skipped(tmp_path) is True


def test_not_skipped_when_simulated(tmp_path):
    _write_rv(tmp_path, out_of_sample_simulated=True)
    assert _reform_validation_skipped(tmp_path) is False


def test_not_skipped_when_no_reform_validation(tmp_path):
    # e.g. a UK release, which carries no reform_validation.json
    assert _reform_validation_skipped(tmp_path) is False


def test_publish_refused_when_out_of_sample_skipped(tmp_path, capsys):
    _write_rv(tmp_path, out_of_sample_simulated=False)
    # The guard returns before publish_release is ever called (no HF upload).
    rc = main([str(tmp_path), "--repo-id", "policyengine/populace-us"])
    assert rc == 1
    assert "refusing to publish" in capsys.readouterr().err
