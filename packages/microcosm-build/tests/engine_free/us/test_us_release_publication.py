"""``microcosm-publish-release-build`` for the US releases (microcosm#1570).

Once the US repository records its builds in ``releases.json``, every US
build publishes into one of the US releases through this command. These
tests never start an emitter or reach the Hub.
"""

from __future__ import annotations

import pytest

from microcosm.build import release_publication

#: The largest US builds, the ACS local-area ones, are about 10 GB.
LARGEST_US_BUILD_BYTES = 10e9
#: The upload rate the command's waits are sized for.
UPLOAD_BITS_PER_SECOND = 20e6
#: The emitter refuses a change still pending an hour after it was handed
#: over (REQUEST_EXPIRED).
EMITTER_EXPIRY_SECONDS = 3600


def test_the_publication_wait_covers_the_emitter_expiry_and_a_us_upload() -> None:
    upload_seconds = LARGEST_US_BUILD_BYTES * 8 / UPLOAD_BITS_PER_SECOND

    assert release_publication.PUBLICATION_WAIT_SECONDS >= (
        EMITTER_EXPIRY_SECONDS + upload_seconds
    )


def test_the_command_takes_a_us_release(tmp_path, monkeypatch, capsys) -> None:
    """A directory named after a US release passes the command's first check;
    with the Hub offline the command then refuses before any emitter starts."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    release_dir = tmp_path / "microcosm-us-2024-national"
    release_dir.mkdir()

    exit_code = release_publication.main(
        [str(release_dir), "--repo-id", "policyengine/populace-us", "--promote"]
    )

    error_output = capsys.readouterr().err
    assert "does not publish into build folders" not in error_output
    assert exit_code == 1
    assert "HF_HUB_OFFLINE" in error_output


def _prepared(release_dir):
    from types import SimpleNamespace

    return SimpleNamespace(
        release_dir=release_dir,
        repository="policyengine/populace-us",
        release=release_dir.name,
        build="populace-us-2024-recertified-20261201",
        release_role="national",
        contract_warnings=(),
    )


def test_the_command_passes_the_parent_h5_and_wheels(tmp_path, monkeypatch) -> None:
    release_dir = tmp_path / "microcosm-us-2024-national"
    release_dir.mkdir()
    passed = {}

    def prepare(path, **kwargs):
        passed.update(kwargs)
        return _prepared(path)

    monkeypatch.setattr(
        release_publication, "publication_refusal", lambda *_a, **_k: None
    )
    monkeypatch.setattr(release_publication, "prepare_build_folder_release", prepare)
    parent_h5 = tmp_path / "parent.h5"
    wheel = tmp_path / "policyengine_us-2.2.1-py3-none-any.whl"

    exit_code = release_publication.main(
        [
            str(release_dir),
            "--preflight-only",
            "--parent-h5",
            str(parent_h5),
            "--compatibility-wheel",
            str(wheel),
        ]
    )

    assert exit_code == 0
    assert passed["parent_h5"] == parent_h5
    assert passed["compatibility_wheels"] == (wheel,)


def test_a_recertification_is_handed_over_only_once_its_parent_is_published(
    tmp_path, monkeypatch
) -> None:
    from microcosm.data import compatibility_recertification

    release_dir = tmp_path / "microcosm-us-2024-national"
    release_dir.mkdir()
    monkeypatch.setattr(
        release_publication,
        "build_folder_identity",
        lambda path: ("policyengine/populace-us", path.name, "b"),
    )
    monkeypatch.setattr(
        release_publication, "_refuse_pointer_files", lambda _repo: None
    )
    monkeypatch.setattr(
        release_publication,
        "prepare_build_folder_release",
        lambda path, **_kwargs: _prepared(path),
    )
    monkeypatch.setattr(release_publication, "_hub_api", lambda: object())
    monkeypatch.setattr(
        compatibility_recertification, "is_recertification", lambda _path: True
    )

    def parent_not_published(_api, _path, **_kwargs):
        raise ValueError("the parent's published manifest is not the one carried")

    monkeypatch.setattr(
        compatibility_recertification,
        "require_parent_publication",
        parent_not_published,
    )

    class NoEmitter:
        def __getattr__(self, name):
            raise AssertionError(f"the build was handed over ({name})")

    with pytest.raises(ValueError, match="published manifest"):
        release_publication.publish_release_build(
            release_dir, artifact_root=tmp_path, client=NoEmitter()
        )


def test_the_command_takes_the_annual_release(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    release_dir = tmp_path / "microcosm-us-2024-annual"
    release_dir.mkdir()

    exit_code = release_publication.main(
        [str(release_dir), "--repo-id", "policyengine/populace-us", "--promote"]
    )

    error_output = capsys.readouterr().err
    assert "does not publish into build folders" not in error_output
    assert exit_code == 1
    assert "HF_HUB_OFFLINE" in error_output


#: The national build an annual build projects, and its dataset's SHA-256.
_BASE_BUILD = "populace-us-2024-9f1260b-20260611"
_BASE_SHA256 = "c" * 64
_US_REPOSITORY = "policyengine/populace-us"


def _annual_build_dir(tmp_path):
    """The release manifest of an annual build, the one file the Hub check reads."""
    import json

    release_dir = tmp_path / "microcosm-us-2024-annual"
    release_dir.mkdir()
    dataset = {
        "kind": "microdata",
        "path": "microcosm_us_2024.h5",
        "repo_id": _US_REPOSITORY,
        "sha256": _BASE_SHA256,
        "role": "dataset",
    }
    (release_dir / "release_manifest.json").write_text(
        json.dumps(
            {
                "release": "microcosm-us-2024-annual",
                "build": f"{_BASE_BUILD}-annual-20261110T120000Z-a1b2c3d4",
                "default_datasets": {"national": "microcosm_us_2024"},
                "metadata": {
                    "annual_projection_base": {
                        "repo_id": _US_REPOSITORY,
                        "release": "microcosm-us-2024-national",
                        "build": _BASE_BUILD,
                        "path": "microcosm_us_2024.h5",
                        "sha256": _BASE_SHA256,
                    }
                },
                "artifacts": {"microcosm_us_2024": dataset},
            }
        )
    )
    return release_dir


class _BaseOnTheHub:
    """A Hub that serves the base build's published release manifest."""

    def __init__(self, tmp_path, problem):
        self.directory = tmp_path / "hub"
        self.problem = problem
        self.requests = []

    def hf_hub_download(self, *, repo_id, filename, repo_type, revision):
        import json

        from test_support.microcosm_data.hub_errors import missing_on_hub

        self.requests.append((repo_id, filename, repo_type, revision))
        if self.problem == "absent":
            raise missing_on_hub(repo_id, filename, revision)
        if self.problem == "unreachable":
            raise RuntimeError("the Hub did not answer")
        sha256 = "d" * 64 if self.problem == "other_sha256" else _BASE_SHA256
        manifest = {
            "release": "microcosm-us-2024-national",
            "build": _BASE_BUILD,
            "default_datasets": {"national": "microcosm_us_2024"},
            "artifacts": {
                "microcosm_us_2024": {
                    "kind": "microdata",
                    "path": "microcosm_us_2024.h5",
                    "repo_id": repo_id,
                    "sha256": sha256,
                    "role": "dataset",
                }
            },
        }
        local = self.directory / filename
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(manifest))
        return str(local)


@pytest.mark.parametrize(
    ("problem", "refusal"),
    [
        (None, None),
        ("absent", "not published"),
        ("other_sha256", "sha256"),
        ("unreachable", "did not answer"),
    ],
)
def test_an_annual_build_is_handed_over_only_once_its_base_build_is_published(
    tmp_path, monkeypatch, problem, refusal
) -> None:
    """The base build's release manifest, read at main from its folder, must
    declare the base-year H5's SHA-256 as its dataset before anything is handed
    to the emitter. A Hub failure other than absence propagates."""
    release_dir = _annual_build_dir(tmp_path)
    hub = _BaseOnTheHub(tmp_path, problem)
    handed_over = []
    monkeypatch.setattr(
        release_publication, "_refuse_pointer_files", lambda _repo: None
    )
    monkeypatch.setattr(
        release_publication,
        "prepare_build_folder_release",
        lambda path, **_kwargs: _prepared(path),
    )
    monkeypatch.setattr(
        release_publication,
        "publication_request",
        lambda prepared, **_kwargs: prepared,
    )
    monkeypatch.setattr(release_publication, "_hub_api", lambda: hub)
    monkeypatch.setattr(
        release_publication,
        "_hand_over",
        lambda change, **_kwargs: handed_over.append(change) or "handed over",
    )

    def publish():
        return release_publication.publish_release_build(
            release_dir, artifact_root=tmp_path, client=object()
        )

    if refusal is None:
        assert publish() == "handed over"
        assert len(handed_over) == 1
    else:
        with pytest.raises((ValueError, RuntimeError), match=refusal):
            publish()
        assert handed_over == []
    assert hub.requests == [
        (
            _US_REPOSITORY,
            f"releases/microcosm-us-2024-national/{_BASE_BUILD}/release_manifest.json",
            "dataset",
            "main",
        )
    ]
