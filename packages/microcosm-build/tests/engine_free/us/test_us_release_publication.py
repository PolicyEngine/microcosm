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


class _NoEmitter:
    """A client that fails the test when anything is handed over."""

    def __getattr__(self, name):
        raise AssertionError(f"the build was handed over ({name})")


def _a_recertification_build(tmp_path, monkeypatch):
    """A national release directory whose build reads as a recertification,
    with its checks and its parent's publication passing."""
    from microcosm.data import compatibility_recertification

    release_dir = tmp_path / "microcosm-us-2024-national"
    release_dir.mkdir()
    monkeypatch.setattr(
        release_publication,
        "build_folder_identity",
        lambda path: (
            "policyengine/populace-us",
            path.name,
            "populace-us-2024-recertified-20261201",
        ),
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
    monkeypatch.setattr(
        compatibility_recertification,
        "require_parent_publication",
        lambda _api, _path, **_kwargs: None,
    )
    return release_dir


def test_a_recertification_is_never_published_as_current(tmp_path, monkeypatch) -> None:
    """A recertification republishes its parent's data under a wider
    compatibility claim; as with the tagged publisher, it never becomes the
    current build, and consumers load it with build=."""
    release_dir = _a_recertification_build(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="compatibility recertification"):
        release_publication.publish_release_build(
            release_dir, artifact_root=tmp_path, make_current=True, client=_NoEmitter()
        )


def test_a_recertification_is_never_promoted(tmp_path, monkeypatch) -> None:
    release_dir = _a_recertification_build(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="compatibility recertification"):
        release_publication.promote_release_build(release_dir, client=_NoEmitter())
