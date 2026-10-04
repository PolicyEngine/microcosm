"""The staged-dataset lane: bundle from a manifest, one-commit upload, fetch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from microcosm.build.staging_dataset import (
    SHA256SUMS_FILENAME,
    STAGED_MANIFEST_FILENAME,
    StagedDatasetBundle,
    StagedDatasetError,
    disabled_staged_dataset,
    fetch_bundle,
    local_only_staged_dataset,
    parse_sha256sums,
    refresh_sha256sums_entry,
    stage_bundle,
    validate_staged_dataset_delivery,
    write_sidecars,
)
from microcosm.build.staging_storage import HuggingFaceDatasetStorage

MANIFEST = "rowwise_candidate_manifest.json"
RUN_ID = "uk-local-candidate-f100-s7-20260917T180000Z-0badcafe"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_run(root: Path, *, manifest_extra: dict | None = None) -> Path:
    run_dir = root / "run"
    run_dir.mkdir()
    files = {
        "microcosm_uk_2025_local.h5": b"h5-bytes" * 100,
        "solve_diagnostics.csv": b"a,b\n1,2\n",
        "calibration_diagnostics.json": json.dumps({"targets": []}).encode(),
    }
    outputs = {}
    for name, data in files.items():
        (run_dir / name).write_bytes(data)
        outputs[name.split(".")[0]] = {
            "path": str(run_dir / name),
            "sha256": _sha(data),
            "bytes": len(data),
        }
    (run_dir / "run.log").write_text("not staged\n")
    manifest = {
        "build_kind": "uk_rowwise_calibrated_candidate",
        "releasable": False,
        "git_commit": "b" * 40,
        "outputs": outputs,
        **(manifest_extra or {}),
    }
    (run_dir / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return run_dir


class FakeHub:
    """Records commits and serves files back, like a private dataset repo."""

    def __init__(self, root: Path, *, fail_commit: bool = False) -> None:
        self.root = root
        self.files: dict[str, bytes] = {}
        self.commits: list[dict] = []
        self.fail_commit = fail_commit
        self.sha = "0" * 40

    def file_exists(self, *, repo_id, filename, repo_type):
        assert repo_type == "dataset"
        return filename in self.files

    def repo_info(self, *, repo_id, repo_type):
        assert repo_type == "dataset"
        return SimpleNamespace(sha=self.sha)

    def create_commit(
        self, *, repo_id, operations, commit_message, repo_type, parent_commit
    ):
        assert repo_type == "dataset"
        if self.fail_commit:
            raise RuntimeError("401 Unauthorized token=do-not-record")
        assert parent_commit == self.sha
        for operation in operations:
            self.files[operation.path_in_repo] = Path(
                operation.path_or_fileobj
            ).read_bytes()
        self.commits.append(
            {
                "message": commit_message,
                "paths": sorted(self.files),
                "parent": parent_commit,
            }
        )
        self.sha = _sha(commit_message.encode())[:40]
        return SimpleNamespace(oid=self.sha)

    def hf_hub_download(self, *, repo_id, filename, repo_type, **kwargs):
        if filename not in self.files:
            raise FileNotFoundError(filename)
        destination = self.root / "cache" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.files[filename])
        return str(destination)


def _storage(hub: FakeHub) -> HuggingFaceDatasetStorage:
    return HuggingFaceDatasetStorage("policyengine/populace-uk-private", api=hub)


def test_bundle_takes_exactly_the_manifest_outputs(tmp_path):
    run_dir = _write_run(tmp_path)
    bundle = StagedDatasetBundle.from_manifest(
        run_dir, run_id=RUN_ID, manifest_name=MANIFEST
    )
    assert [item.name for item in bundle.files] == [
        "calibration_diagnostics.json",
        "microcosm_uk_2025_local.h5",
        "solve_diagnostics.csv",
    ]
    assert "run.log" not in bundle.digests()
    assert bundle.summary == {
        "build_kind": "uk_rowwise_calibrated_candidate",
        "git_commit": "b" * 40,
        "releasable": False,
    }
    assert bundle.remote_prefix() == f"staged/{RUN_ID}"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda d: (d / "solve_diagnostics.csv").unlink(), "missing"),
        (lambda d: (d / "solve_diagnostics.csv").write_bytes(b"x"), "digest"),
    ],
)
def test_bundle_refuses_missing_or_changed_outputs(tmp_path, mutation, message):
    run_dir = _write_run(tmp_path)
    mutation(run_dir)
    with pytest.raises(StagedDatasetError, match=message):
        StagedDatasetBundle.from_manifest(
            run_dir, run_id=RUN_ID, manifest_name=MANIFEST
        )


def test_bundle_refuses_an_unsafe_run_id_and_an_empty_manifest(tmp_path):
    run_dir = _write_run(tmp_path)
    with pytest.raises(StagedDatasetError, match="run_id"):
        StagedDatasetBundle.from_manifest(
            run_dir, run_id="../escape", manifest_name=MANIFEST
        )
    (run_dir / MANIFEST).write_text(json.dumps({"outputs": {}}))
    with pytest.raises(StagedDatasetError, match="nothing to stage"):
        StagedDatasetBundle.from_manifest(
            run_dir, run_id=RUN_ID, manifest_name=MANIFEST
        )


def test_sidecars_list_every_file_and_verify_themselves(tmp_path):
    run_dir = _write_run(tmp_path)
    bundle = StagedDatasetBundle.from_manifest(
        run_dir, run_id=RUN_ID, manifest_name=MANIFEST
    )
    sums, staged_manifest = write_sidecars(
        bundle, repository="policyengine/populace-uk-private"
    )
    entries = dict((name, sha) for sha, name in parse_sha256sums(sums.read_text()))
    assert set(entries) == {
        "calibration_diagnostics.json",
        "microcosm_uk_2025_local.h5",
        "solve_diagnostics.csv",
        MANIFEST,
        STAGED_MANIFEST_FILENAME,
    }
    for name, sha in entries.items():
        assert _sha((run_dir / name).read_bytes()) == sha
    payload = json.loads(staged_manifest.read_text())
    assert payload["schema_name"] == "microcosm.staged-dataset.manifest"
    assert payload["prefix"] == f"staged/{RUN_ID}"
    assert payload["files"] == bundle.digests()
    assert payload["manifest"]["sha256"] == _sha((run_dir / MANIFEST).read_bytes())
    # Byte-identical on a second write: no timestamps, no absolute paths.
    before = staged_manifest.read_bytes()
    write_sidecars(bundle, repository="policyengine/populace-uk-private")
    assert staged_manifest.read_bytes() == before
    assert str(run_dir) not in staged_manifest.read_text()


def test_stage_bundle_commits_everything_once_under_the_run_prefix(tmp_path):
    run_dir = _write_run(tmp_path)
    bundle = StagedDatasetBundle.from_manifest(
        run_dir, run_id=RUN_ID, manifest_name=MANIFEST
    )
    write_sidecars(bundle, repository="policyengine/populace-uk-private")
    hub = FakeHub(tmp_path)
    delivery = stage_bundle(bundle, storage=_storage(hub))
    assert delivery["status"] == "uploaded"
    assert delivery["revision"] == hub.sha
    assert delivery["repository"] == "policyengine/populace-uk-private"
    assert delivery["prefix"] == f"staged/{RUN_ID}"
    assert len(hub.commits) == 1
    assert hub.commits[0]["parent"] == "0" * 40
    assert hub.commits[0]["paths"] == sorted(
        f"staged/{RUN_ID}/{name}"
        for name in (
            "calibration_diagnostics.json",
            "microcosm_uk_2025_local.h5",
            "solve_diagnostics.csv",
            MANIFEST,
            STAGED_MANIFEST_FILENAME,
            SHA256SUMS_FILENAME,
        )
    )
    assert validate_staged_dataset_delivery(delivery) == delivery


def test_stage_bundle_is_idempotent_on_the_output_digests(tmp_path):
    run_dir = _write_run(tmp_path)
    bundle = StagedDatasetBundle.from_manifest(
        run_dir, run_id=RUN_ID, manifest_name=MANIFEST
    )
    write_sidecars(bundle, repository="policyengine/populace-uk-private")
    hub = FakeHub(tmp_path)
    first = stage_bundle(bundle, storage=_storage(hub))
    # The driver appends evidence to the local manifest after the upload;
    # a re-stage must still recognise the same outputs.
    manifest = json.loads((run_dir / MANIFEST).read_text())
    manifest["staged_dataset"] = first
    (run_dir / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    again = StagedDatasetBundle.from_manifest(
        run_dir, run_id=RUN_ID, manifest_name=MANIFEST
    )
    write_sidecars(again, repository="policyengine/populace-uk-private")
    second = stage_bundle(again, storage=_storage(hub))
    assert second["status"] == "already_staged"
    assert second["revision"] == hub.sha
    assert len(hub.commits) == 1

    (run_dir / "solve_diagnostics.csv").write_bytes(b"a,b\n9,9\n")
    manifest["outputs"]["solve_diagnostics"]["sha256"] = _sha(b"a,b\n9,9\n")
    manifest["outputs"]["solve_diagnostics"]["bytes"] = len(b"a,b\n9,9\n")
    (run_dir / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    changed = StagedDatasetBundle.from_manifest(
        run_dir, run_id=RUN_ID, manifest_name=MANIFEST
    )
    write_sidecars(changed, repository="policyengine/populace-uk-private")
    refused = stage_bundle(changed, storage=_storage(hub))
    assert refused["status"] == "failed"
    assert refused["error_code"] == "REMOTE_DIFFERS"
    assert len(hub.commits) == 1


def test_stage_bundle_records_transport_failure_without_raising(tmp_path, capsys):
    run_dir = _write_run(tmp_path)
    bundle = StagedDatasetBundle.from_manifest(
        run_dir, run_id=RUN_ID, manifest_name=MANIFEST
    )
    write_sidecars(bundle, repository="policyengine/populace-uk-private")
    hub = FakeHub(tmp_path, fail_commit=True)
    delivery = stage_bundle(bundle, storage=_storage(hub))
    assert delivery["status"] == "failed"
    assert delivery["error_code"] == "UPLOAD_FAILED"
    assert delivery["revision"] is None
    assert delivery["files"] == bundle.digests()
    err = capsys.readouterr().err
    assert "staged dataset upload failed" in err
    assert "do-not-record" not in err and "do-not-record" not in json.dumps(delivery)


def test_stage_bundle_requires_the_sidecars_first(tmp_path):
    run_dir = _write_run(tmp_path)
    bundle = StagedDatasetBundle.from_manifest(
        run_dir, run_id=RUN_ID, manifest_name=MANIFEST
    )
    with pytest.raises(StagedDatasetError, match="write_sidecars"):
        stage_bundle(bundle, storage=_storage(FakeHub(tmp_path)))


def test_fetch_bundle_verifies_every_digest_and_copies_out_of_the_cache(tmp_path):
    run_dir = _write_run(tmp_path)
    bundle = StagedDatasetBundle.from_manifest(
        run_dir, run_id=RUN_ID, manifest_name=MANIFEST
    )
    write_sidecars(bundle, repository="policyengine/populace-uk-private")
    hub = FakeHub(tmp_path)
    stage_bundle(bundle, storage=_storage(hub))

    dest = tmp_path / "fetched"
    paths = fetch_bundle(storage=_storage(hub), run_id=RUN_ID, dest=dest)
    assert {path.name for path in paths} == {
        "calibration_diagnostics.json",
        "microcosm_uk_2025_local.h5",
        "solve_diagnostics.csv",
        MANIFEST,
        STAGED_MANIFEST_FILENAME,
    }
    assert (dest / "microcosm_uk_2025_local.h5").read_bytes() == b"h5-bytes" * 100
    assert (dest / SHA256SUMS_FILENAME).is_file()

    only = fetch_bundle(
        storage=_storage(hub), run_id=RUN_ID, dest=tmp_path / "h5", h5_only=True
    )
    assert [path.name for path in only] == ["microcosm_uk_2025_local.h5"]

    hub.files[f"staged/{RUN_ID}/solve_diagnostics.csv"] = b"tampered"
    with pytest.raises(StagedDatasetError, match="does not match"):
        fetch_bundle(storage=_storage(hub), run_id=RUN_ID, dest=tmp_path / "bad")
    assert not (tmp_path / "bad" / "solve_diagnostics.csv").exists()


def test_delivery_records_follow_their_mode(tmp_path):
    run_dir = _write_run(tmp_path)
    bundle = StagedDatasetBundle.from_manifest(
        run_dir, run_id=RUN_ID, manifest_name=MANIFEST
    )
    disabled = disabled_staged_dataset("--no-staging")
    assert disabled["mode"] == "disabled" and disabled["status"] == "skipped"
    assert disabled["files"] == {}
    local = local_only_staged_dataset(bundle)
    assert local["mode"] == "local_only" and local["status"] == "skipped"
    assert local["repository"] is None and local["files"] == bundle.digests()
    with pytest.raises(StagedDatasetError):
        disabled_staged_dataset("")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"status": "uploaded", "revision": None}, "records its revision"),
        ({"status": "failed", "error_code": None}, "error code"),
        ({"status": "skipped"}, "uploaded, already staged or failed"),
        ({"opt_out_reason": "x"}, "opt-out reason"),
        ({"error_code": "BOOM", "status": "failed"}, "unknown"),
        (
            {
                "mode": "local_only",
                "repository": None,
                "revision": None,
                "status": "uploaded",
            },
            "local-only",
        ),
        ({"extra": 1}, "keys differ"),
        ({"files": {"a.h5": {"sha256": "zz", "bytes": 1}}}, "invalid"),
    ],
)
def test_delivery_validation_refuses_contradictions(overrides, message):
    base = {
        "contract_version": 1,
        "mode": "local_and_remote",
        "repository": "policyengine/populace-uk-private",
        "prefix": f"staged/{RUN_ID}",
        "run_id": RUN_ID,
        "revision": "a" * 40,
        "status": "uploaded",
        "error_code": None,
        "opt_out_reason": None,
        "files": {"microcosm_uk_2025_local.h5": {"sha256": "0" * 64, "bytes": 3}},
    }
    assert validate_staged_dataset_delivery(base) == base
    with pytest.raises(StagedDatasetError, match=message):
        validate_staged_dataset_delivery({**base, **overrides})


def test_parse_sha256sums_rejects_paths_and_duplicates():
    good = f"{'a' * 64}  file.h5\n{'b' * 64}  other.json\n"
    assert parse_sha256sums(good) == [("a" * 64, "file.h5"), ("b" * 64, "other.json")]
    with pytest.raises(StagedDatasetError, match="malformed"):
        parse_sha256sums(f"{'a' * 64}  ../escape.h5\n")
    with pytest.raises(StagedDatasetError, match="twice"):
        parse_sha256sums(f"{'a' * 64}  file.h5\n{'b' * 64}  file.h5\n")
    with pytest.raises(StagedDatasetError, match="no files"):
        parse_sha256sums("\n")


def test_refresh_sha256sums_entry_redigests_one_listed_file(tmp_path):
    run_dir = _write_run(tmp_path)
    bundle = StagedDatasetBundle.from_manifest(
        run_dir, run_id=RUN_ID, manifest_name=MANIFEST
    )
    sums, _ = write_sidecars(bundle, repository=None)
    before = dict((name, sha) for sha, name in parse_sha256sums(sums.read_text()))
    (run_dir / MANIFEST).write_text(
        json.dumps(
            {**json.loads((run_dir / MANIFEST).read_text()), "staged_dataset": {}}
        )
    )
    digest = refresh_sha256sums_entry(run_dir, MANIFEST)
    after = dict((name, sha) for sha, name in parse_sha256sums(sums.read_text()))
    assert digest == _sha((run_dir / MANIFEST).read_bytes()) == after[MANIFEST]
    assert after[MANIFEST] != before[MANIFEST]
    assert {k: v for k, v in after.items() if k != MANIFEST} == {
        k: v for k, v in before.items() if k != MANIFEST
    }
    with pytest.raises(StagedDatasetError, match="does not list"):
        refresh_sha256sums_entry(run_dir, "run.log")


class RecordingHub(FakeHub):
    """FakeHub that also answers get_paths_info with each file's last commit."""

    def __init__(self, root: Path, **kwargs) -> None:
        super().__init__(root, **kwargs)
        self.commit_of: dict[str, str] = {}

    def create_commit(self, **kwargs):
        info = super().create_commit(**kwargs)
        for operation in kwargs["operations"]:
            self.commit_of[operation.path_in_repo] = self.sha
        return info

    def get_paths_info(self, *, repo_id, paths, expand, repo_type):
        assert expand and repo_type == "dataset"
        return [
            SimpleNamespace(path=p, last_commit=SimpleNamespace(oid=self.commit_of[p]))
            for p in paths
            if p in self.commit_of
        ]


def test_already_staged_records_the_bundles_own_commit_not_the_head(tmp_path):
    from microcosm.build.staging_dataset import delivery_covers_bundle

    run_dir = _write_run(tmp_path)
    bundle = StagedDatasetBundle.from_manifest(
        run_dir, run_id=RUN_ID, manifest_name=MANIFEST
    )
    write_sidecars(bundle, repository="policyengine/populace-uk-private")
    hub = RecordingHub(tmp_path)
    first = stage_bundle(bundle, storage=_storage(hub))
    staged_at = first["revision"]
    # Another commit moves the repository head after the bundle landed.
    hub.sha = "f" * 40
    again = stage_bundle(bundle, storage=_storage(hub))
    assert again["status"] == "already_staged"
    assert again["revision"] == staged_at != hub.sha
    assert delivery_covers_bundle(
        first, bundle, repository="policyengine/populace-uk-private"
    )
    assert delivery_covers_bundle(
        again, bundle, repository="policyengine/populace-uk-private"
    )
    assert not delivery_covers_bundle(first, bundle, repository="other/repo")
    assert not delivery_covers_bundle(
        {**first, "status": "failed", "revision": None, "error_code": "UPLOAD_FAILED"},
        bundle,
        repository="policyengine/populace-uk-private",
    )
    assert not delivery_covers_bundle(None, bundle, repository="x/y")
