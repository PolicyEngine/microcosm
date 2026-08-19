import json
from pathlib import Path

import pytest

from microcosm.data.publish_cli import (
    _reform_validation_skipped,
    _staging_undelivered,
    main,
)


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


def _stub_publish(monkeypatch):
    import microcosm.data.publish_cli as cli

    monkeypatch.setattr(
        cli, "publish_release", lambda *a, **k: {"release_id": "r", "updated_at": None}
    )
    return cli


def test_allow_incomplete_reform_validation_publishes(tmp_path, capsys, monkeypatch):
    _write_rv(tmp_path, out_of_sample_simulated=False)
    cli = _stub_publish(monkeypatch)
    monkeypatch.delenv("SLACK_WEBHOOK_POPULACE_US", raising=False)
    rc = cli.main([str(tmp_path), "--allow-incomplete-reform-validation"])
    assert rc == 0
    assert "refusing to publish" not in capsys.readouterr().err


def _write_bm(release_dir: Path, manifest: dict) -> None:
    (release_dir / "build_manifest.json").write_text(json.dumps(manifest))


def test_staging_undelivered_reads_the_manifest_not_the_country(tmp_path):
    assert _staging_undelivered(tmp_path) is False

    # No staging key: a builder with no staging path, e.g. the ACS local-area
    # product. Key presence scopes the gate without an exception list.
    _write_bm(tmp_path, {"build_id": "x", "dataset": {"kind": "acs_local_area"}})
    assert _staging_undelivered(tmp_path) is False

    # Present and empty: the pre-provenance shape, or a lost destination.
    _write_bm(tmp_path, {"build_id": "x", "staging": None})
    assert _staging_undelivered(tmp_path) is True
    _write_bm(tmp_path, {"build_id": "x", "staging": {}})
    assert _staging_undelivered(tmp_path) is True

    # A declared opt-out is a statement, not a gap.
    _write_bm(
        tmp_path,
        {"build_id": "x", "staging": {"enabled": False, "reason": "--no-staging"}},
    )
    assert _staging_undelivered(tmp_path) is False

    # Enabled but nothing landed: what a run without a write token looks like.
    _write_bm(
        tmp_path,
        {
            "build_id": "x",
            "staging": {
                "enabled": True,
                "run_id": "r",
                "uploads_succeeded": 0,
            },
        },
    )
    assert _staging_undelivered(tmp_path) is True

    _write_bm(
        tmp_path,
        {
            "build_id": "x",
            "staging": {
                "enabled": True,
                "run_id": "r",
                "uploads_succeeded": 9,
            },
        },
    )
    assert _staging_undelivered(tmp_path) is False


def test_publish_refused_when_staging_never_delivered(tmp_path, capsys, monkeypatch):
    _write_bm(
        tmp_path,
        {
            "build_id": "x",
            "staging": {
                "enabled": True,
                "run_id": "r",
                "uploads_succeeded": 0,
            },
        },
    )
    monkeypatch.delenv("SLACK_WEBHOOK_POPULACE_US", raising=False)
    rc = main([str(tmp_path)])
    assert rc == 1
    assert "refusing to publish" in capsys.readouterr().err


def test_publish_refused_when_staging_block_is_null(tmp_path, capsys, monkeypatch):
    _write_bm(tmp_path, {"build_id": "x", "staging": None})
    monkeypatch.delenv("SLACK_WEBHOOK_POPULACE_US", raising=False)
    rc = main([str(tmp_path)])
    assert rc == 1
    assert "refusing to publish" in capsys.readouterr().err


def test_publish_allowed_for_a_declared_no_staging_build(tmp_path, capsys, monkeypatch):
    _write_bm(
        tmp_path,
        {"build_id": "x", "staging": {"enabled": False, "reason": "--no-staging"}},
    )
    cli = _stub_publish(monkeypatch)
    monkeypatch.delenv("SLACK_WEBHOOK_POPULACE_US", raising=False)
    rc = cli.main([str(tmp_path)])
    assert rc == 0
    assert "refusing to publish" not in capsys.readouterr().err


def test_publish_allowed_when_the_builder_has_no_staging_path(
    tmp_path, capsys, monkeypatch
):
    _write_bm(tmp_path, {"build_id": "microcosm-us-local-x", "dataset": {}})
    cli = _stub_publish(monkeypatch)
    monkeypatch.delenv("SLACK_WEBHOOK_POPULACE_US", raising=False)
    rc = cli.main([str(tmp_path), "--no-latest"])
    assert rc == 0
    assert "refusing to publish" not in capsys.readouterr().err


def test_allow_missing_staging_escape_hatch_publishes(tmp_path, capsys, monkeypatch):
    _write_bm(tmp_path, {"build_id": "x", "staging": None})
    cli = _stub_publish(monkeypatch)
    monkeypatch.delenv("SLACK_WEBHOOK_POPULACE_US", raising=False)
    rc = cli.main([str(tmp_path), "--allow-missing-staging"])
    assert rc == 0
    assert "refusing to publish" not in capsys.readouterr().err


def test_publish_proceeds_when_staging_delivered(tmp_path, capsys, monkeypatch):
    _write_bm(
        tmp_path,
        {
            "build_id": "x",
            "staging": {
                "enabled": True,
                "run_id": "r",
                "uploads_succeeded": 9,
            },
        },
    )
    cli = _stub_publish(monkeypatch)
    monkeypatch.delenv("SLACK_WEBHOOK_POPULACE_US", raising=False)
    rc = cli.main([str(tmp_path)])
    assert rc == 0
    assert "refusing to publish" not in capsys.readouterr().err


def test_tag_only_cli_forwards_no_main_publication_mode(tmp_path, monkeypatch):
    import microcosm.data.publish_cli as cli

    captured: dict = {}

    def fake_publish(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"release_id": "r", "updated_at": None}

    monkeypatch.setattr(cli, "publish_release", fake_publish)

    rc = cli.main([str(tmp_path), "--no-latest", "--tag-only"])

    assert rc == 0
    assert captured["kwargs"]["update_latest"] is False
    assert captured["kwargs"]["tag_only"] is True
    assert captured["kwargs"]["create_tag"] is True


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--tag-only"], "--tag-only requires --no-latest"),
        (
            ["--no-latest", "--tag-only", "--no-create-tag"],
            "--tag-only requires tag creation",
        ),
    ],
)
def test_tag_only_cli_rejects_unsafe_flag_combinations_before_publish(
    tmp_path, capsys, monkeypatch, arguments, message
):
    import microcosm.data.publish_cli as cli

    called = False

    def unexpected_publish(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(cli, "publish_release", unexpected_publish)

    with pytest.raises(SystemExit, match="2"):
        cli.main([str(tmp_path), *arguments])

    assert message in capsys.readouterr().err
    assert called is False
