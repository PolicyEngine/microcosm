"""``microcosm-publish-release-build`` for the US releases (microcosm#1570).

Once the US repository records its builds in ``releases.json``, every US
build publishes into one of the US releases through this command. These
tests never start an emitter or reach the Hub.
"""

from __future__ import annotations

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
