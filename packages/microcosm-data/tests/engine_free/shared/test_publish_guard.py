import json
from pathlib import Path

import pytest

from microcosm.data.release import (
    _non_release_artifact,
    _reform_validation_skipped,
    _staging_undelivered,
    publication_refusal,
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


def test_publication_refused_when_out_of_sample_skipped(tmp_path):
    _write_rv(tmp_path, out_of_sample_simulated=False)

    refusal = publication_refusal(tmp_path)

    assert refusal is not None
    assert refusal.startswith("refusing to publish")
    assert "--allow-incomplete-reform-validation" in refusal


def test_allow_incomplete_reform_validation_lifts_the_refusal(tmp_path):
    _write_rv(tmp_path, out_of_sample_simulated=False)

    assert (
        publication_refusal(tmp_path, allow_incomplete_reform_validation=True) is None
    )


def _write_bm(release_dir: Path, manifest: dict) -> None:
    (release_dir / "build_manifest.json").write_text(json.dumps(manifest))


def test_publication_refuses_non_release_smoke_even_with_staging_override(
    tmp_path,
):
    _write_bm(
        tmp_path,
        {
            "build_id": "uk-frs-spine-h0100-s578-x",
            "non_release": True,
            "release_posture": "non_release_smoke",
            "staging": {"enabled": False, "reason": "--no-staging"},
        },
    )
    assert _non_release_artifact(tmp_path) is True

    refusal = publication_refusal(tmp_path, allow_missing_staging=True)

    assert refusal is not None
    assert "non-release smoke output" in refusal


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


def _version_2_delivery(**overrides) -> dict:
    payload = {
        "contract_version": 2,
        "enabled": True,
        "mode": "local_and_remote",
        "run_id": "uk-run",
        "configured_repository": "policyengine/populace-uk-staging",
        "upload_attempts": 4,
        "upload_successes": 4,
        "read_back": "not_requested",
        "last_error_code": None,
        "opt_out_reason": None,
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    ("delivery", "undelivered"),
    [
        (_version_2_delivery(), False),
        (_version_2_delivery(upload_attempts=0, upload_successes=0), True),
        (
            _version_2_delivery(
                mode="local_only",
                configured_repository=None,
                upload_attempts=0,
                upload_successes=0,
            ),
            True,
        ),
        (
            _version_2_delivery(
                enabled=False,
                mode="disabled",
                run_id=None,
                configured_repository=None,
                upload_attempts=0,
                upload_successes=0,
                opt_out_reason="--no-staging",
            ),
            False,
        ),
        (_version_2_delivery(contract_version=999), True),
        (_version_2_delivery(upload_attempts=1, upload_successes=2), True),
        (
            _version_2_delivery(
                upload_attempts=0,
                upload_successes=0,
                read_back="passed",
            ),
            True,
        ),
        (
            _version_2_delivery(
                read_back="failed",
                last_error_code="READ_BACK_FAILED",
            ),
            True,
        ),
        (
            _version_2_delivery(
                enabled=False,
                mode="disabled",
                run_id=None,
                configured_repository=None,
                upload_attempts=0,
                upload_successes=0,
                read_back="passed",
                opt_out_reason="--no-staging",
            ),
            True,
        ),
        (_version_2_delivery(last_error_code=["UPLOAD_FAILED"]), True),
        ({"contract_version": 2, "enabled": True}, True),
    ],
)
def test_version_2_staging_delivery_parser(tmp_path, delivery, undelivered):
    _write_bm(tmp_path, {"build_id": "x", "staging": delivery})

    assert _staging_undelivered(tmp_path) is undelivered


_STAGED = {"enabled": True, "run_id": "r", "repo_id": "o/s", "uploads_succeeded": 4}
_UNDELIVERED = {
    "enabled": True,
    "run_id": "r",
    "repo_id": "o/s",
    "uploads_succeeded": 0,
}
_OPTED_OUT = {"enabled": False, "reason": "--no-staging"}


@pytest.mark.parametrize(
    ("certification", "build", "undelivered"),
    [
        # Both staged, or both declared an opt-out.
        (_STAGED, _STAGED, False),
        ({"enabled": False, "reason": "the build did not stage"}, _OPTED_OUT, False),
        # Certification opted out where its build staged.
        ({"enabled": False, "reason": "--no-staging at certification"}, _STAGED, False),
        # The build's own uploads never arrived: refused, whatever its
        # certification did (review of microcosm#1672).
        (_STAGED, _UNDELIVERED, True),
        (
            {"enabled": False, "reason": "--no-staging at certification"},
            _UNDELIVERED,
            True,
        ),
        (_STAGED, None, True),
        (_STAGED, {}, True),
        (_STAGED, {"enabled": False}, True),
        # The certification's own uploads never arrived.
        (_UNDELIVERED, _STAGED, True),
    ],
)
def test_a_separately_certified_build_needs_its_own_staging_too(
    tmp_path, certification, build, undelivered
):
    """A US national build is certified as a step of its own (microcosm#1585
    item 11): the manifest's staging block is its certification's, with the
    build's own under ``build``, and the gate reads both."""
    _write_bm(tmp_path, {"build_id": "x", "staging": {**certification, "build": build}})
    assert _staging_undelivered(tmp_path) is undelivered


def test_publication_refuses_a_certified_build_whose_own_staging_never_arrived(
    tmp_path,
):
    _write_bm(
        tmp_path,
        {
            "build_id": "x",
            "staging": {
                "enabled": False,
                "reason": "--no-staging at certification; the build staged (run r)",
                "build": _UNDELIVERED,
            },
        },
    )

    refusal = publication_refusal(tmp_path)

    assert refusal is not None
    assert "for the build itself under staging.build" in refusal
    assert publication_refusal(tmp_path, allow_missing_staging=True) is None


def test_publication_refused_when_staging_never_delivered(tmp_path):
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

    refusal = publication_refusal(tmp_path)

    assert refusal is not None
    assert refusal.startswith("refusing to publish")


def test_publication_refused_when_staging_block_is_null(tmp_path):
    _write_bm(tmp_path, {"build_id": "x", "staging": None})

    refusal = publication_refusal(tmp_path)

    assert refusal is not None
    assert refusal.startswith("refusing to publish")


def test_publication_allowed_for_a_declared_no_staging_build(tmp_path):
    _write_bm(
        tmp_path,
        {"build_id": "x", "staging": {"enabled": False, "reason": "--no-staging"}},
    )

    assert publication_refusal(tmp_path) is None


def test_publication_allowed_when_the_builder_has_no_staging_path(tmp_path):
    _write_bm(tmp_path, {"build_id": "microcosm-us-local-x", "dataset": {}})

    assert publication_refusal(tmp_path) is None


def test_allow_missing_staging_lifts_the_staging_refusal(tmp_path):
    _write_bm(tmp_path, {"build_id": "x", "staging": None})

    assert publication_refusal(tmp_path, allow_missing_staging=True) is None


def test_publication_proceeds_when_staging_delivered(tmp_path):
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

    assert publication_refusal(tmp_path) is None


@pytest.mark.parametrize(
    ("delivery", "refused"),
    [
        (_version_2_delivery(), False),
        (_version_2_delivery(upload_attempts=0, upload_successes=0), True),
        (
            _version_2_delivery(
                mode="local_only",
                configured_repository=None,
                upload_attempts=0,
                upload_successes=0,
            ),
            True,
        ),
        (
            _version_2_delivery(
                enabled=False,
                mode="disabled",
                run_id=None,
                configured_repository=None,
                upload_attempts=0,
                upload_successes=0,
                opt_out_reason="--no-staging",
            ),
            False,
        ),
    ],
)
def test_version_2_publication_decisions(tmp_path, delivery, refused):
    _write_bm(tmp_path, {"build_id": "x", "staging": delivery})

    assert (publication_refusal(tmp_path) is not None) is refused


def test_version_2_local_only_can_use_explicit_override(tmp_path):
    _write_bm(
        tmp_path,
        {
            "build_id": "x",
            "staging": _version_2_delivery(
                mode="local_only",
                configured_repository=None,
                upload_attempts=0,
                upload_successes=0,
            ),
        },
    )

    assert publication_refusal(tmp_path) is not None
    assert publication_refusal(tmp_path, allow_missing_staging=True) is None
