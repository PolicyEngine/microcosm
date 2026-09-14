"""Declared file artifacts are atomically materialized and checked by bytes."""

import pytest

from microcosm.build.artifact_files import materialize_bytes, validate_file_inventory


def test_atomic_artifact_materialization_can_recreate_a_missing_output(tmp_path):
    path = tmp_path / "evidence.json"
    record = materialize_bytes(b'{"result":1}\n', path)
    path.unlink()
    assert materialize_bytes(b'{"result":1}\n', path) == record
    validate_file_inventory({"evidence": record}, root=tmp_path)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity"):
        validate_file_inventory({"evidence": record}, root=tmp_path)


def test_artifact_inventory_rejects_paths_outside_bundle(tmp_path):
    with pytest.raises(ValueError, match="filename"):
        validate_file_inventory({"escape": {"filename": "../other"}}, root=tmp_path)


def test_bundle_publication_replaces_existing_payloads_and_marker_last(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from microcosm.build.artifact_files import publish_staged_bundle

    stage = tmp_path / "stage"
    output = tmp_path / "output"
    stage.mkdir()
    output.mkdir()
    sources = {
        role: stage / name
        for role, name in (("dataset", "full.h5"), ("manifest", "full.build.json"))
    }
    targets = {role: output / path.name for role, path in sources.items()}
    for role in sources:
        sources[role].write_bytes(("new-" + role).encode())
        targets[role].write_bytes(("old-" + role).encode())
    moves = []
    original = Path.replace

    def record(path, destination):
        if Path(destination).parent == output:
            moves.append(Path(destination).name)
        return original(path, destination)

    monkeypatch.setattr(Path, "replace", record)
    inventory = publish_staged_bundle(sources, targets)
    assert moves == ["full.h5", "full.build.json"]
    assert targets["manifest"].read_bytes() == b"new-manifest"
    validate_file_inventory(inventory, root=output)


@pytest.mark.parametrize("interrupt", [RuntimeError, KeyboardInterrupt])
def test_bundle_publication_rolls_back_old_complete_bundle(
    tmp_path, monkeypatch, interrupt
):
    from pathlib import Path

    from microcosm.build.artifact_files import publish_staged_bundle

    stage = tmp_path / "stage"
    output = tmp_path / "output"
    stage.mkdir()
    output.mkdir()
    sources = {
        role: stage / name
        for role, name in (("dataset", "full.h5"), ("manifest", "full.build.json"))
    }
    targets = {role: output / path.name for role, path in sources.items()}
    for role in sources:
        sources[role].write_bytes(("new-" + role).encode())
        targets[role].write_bytes(("old-" + role).encode())
    original = Path.replace

    def fail(path, destination):
        if path == sources["manifest"]:
            assert not targets["manifest"].exists()
            raise interrupt("synthetic publication interruption")
        return original(path, destination)

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(interrupt):
        publish_staged_bundle(sources, targets)
    for role in targets:
        assert targets[role].read_bytes() == ("old-" + role).encode()
    assert not list(tmp_path.glob(".bundle-backup-*"))
