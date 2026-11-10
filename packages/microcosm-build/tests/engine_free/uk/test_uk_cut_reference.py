"""A UK cut's reference: the current build of its own release line (#1095)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from microcosm.build.uk_runtime.cut_reference import (
    UKCutReference,
    check_uk_cut_reference,
    read_uk_cut_reference,
    resolve_uk_cut_reference,
    verify_uk_cut_reference_file,
    write_uk_cut_reference,
)
from test_support.microcosm_build.uk_cut_reference import (
    UK_DENSE_RELEASE,
    UK_NATIONAL_RELEASE,
    UK_REPO_ID,
    UKReferenceHub,
    dense_gate_summary,
    national_certification,
    sha256_of,
)

NATIONAL_BUILD = "uk-frs-calibration-attempt-20261107T020434Z-eeb6093e"
EARLIER_NATIONAL_BUILD = "uk-frs-calibration-attempt-20261025T091414Z-a6299504"
DENSE_BUILD = "uk-local-candidate-f100-s42-20261107T173210Z-38605444"
NATIONAL_BYTES = b"national current build"
DENSE_BYTES = b"dense current build"


def _fixed_now() -> datetime:
    return datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


@pytest.fixture
def hub(tmp_path: Path) -> UKReferenceHub:
    hub = UKReferenceHub(tmp_path)
    hub.publish(UK_NATIONAL_RELEASE, EARLIER_NATIONAL_BUILD, dataset=b"older")
    hub.publish(UK_NATIONAL_RELEASE, NATIONAL_BUILD, dataset=NATIONAL_BYTES)
    hub.publish(UK_DENSE_RELEASE, DENSE_BUILD, dataset=DENSE_BYTES)
    return hub


def test_a_national_cut_compares_with_the_national_current_build(hub) -> None:
    path, reference = resolve_uk_cut_reference(
        UK_NATIONAL_RELEASE, hub_download=hub, now=_fixed_now
    )

    assert path.read_bytes() == NATIONAL_BYTES
    assert reference.release == UK_NATIONAL_RELEASE
    assert reference.build == NATIONAL_BUILD
    assert reference.label == f"{UK_NATIONAL_RELEASE}@{NATIONAL_BUILD}"
    assert reference.repo_id == UK_REPO_ID
    assert reference.path_in_repo == (
        f"releases/{UK_NATIONAL_RELEASE}/{NATIONAL_BUILD}/microcosm_uk_2024_25.h5"
    )
    assert reference.filename == "microcosm_uk_2024_25.h5"
    assert reference.sha256 == sha256_of(NATIONAL_BYTES)
    assert reference.size_bytes == len(NATIONAL_BYTES)
    assert reference.built_with_model_package == {
        "name": "policyengine-uk",
        "version": "2.174.1",
    }
    assert reference.evidence["key"] == "release_certification"
    assert reference.resolved_at == "2026-11-10T12:00:00Z"


def test_a_dense_cut_compares_with_the_dense_current_build(hub) -> None:
    path, reference = resolve_uk_cut_reference(UK_DENSE_RELEASE, hub_download=hub)

    assert path.read_bytes() == DENSE_BYTES
    assert reference.build == DENSE_BUILD
    assert reference.filename == "microcosm_uk_2024_25_dense.h5"
    assert reference.evidence["key"] == "gate_summary"


def test_the_check_names_the_build_without_downloading_its_dataset(hub) -> None:
    summary = check_uk_cut_reference(UK_NATIONAL_RELEASE, hub_download=hub)

    assert summary["build"] == NATIONAL_BUILD
    assert summary["sha256"] == sha256_of(NATIONAL_BYTES)
    assert not any(call.endswith(".h5") for call in hub.calls)


def test_a_promotion_moves_what_the_next_fetch_resolves(hub) -> None:
    hub.publish(
        UK_NATIONAL_RELEASE,
        "uk-frs-calibration-attempt-20261112T000000Z-0new0000",
        dataset=b"newer",
    )

    _, reference = resolve_uk_cut_reference(UK_NATIONAL_RELEASE, hub_download=hub)

    assert reference.build == "uk-frs-calibration-attempt-20261112T000000Z-0new0000"


@pytest.mark.parametrize(
    ("evidence", "message"),
    [
        (
            national_certification(NATIONAL_BYTES, shippable=False),
            "not certified shippable",
        ),
        (
            national_certification(NATIONAL_BYTES, candidate={"sha256": "0" * 64}),
            "certifies another file",
        ),
        (
            national_certification(
                NATIONAL_BYTES, attestation={"signing_key_sha256": "1" * 64}
            ),
            "registered UK gate key",
        ),
        (
            national_certification(NATIONAL_BYTES, release_id=UK_DENSE_RELEASE),
            "certifies another release",
        ),
    ],
)
@pytest.mark.committed_gate_key_registry
def test_a_national_build_without_a_shippable_certification_is_refused(
    tmp_path: Path, evidence, message
) -> None:
    hub = UKReferenceHub(tmp_path)
    hub.publish(
        UK_NATIONAL_RELEASE, NATIONAL_BUILD, dataset=NATIONAL_BYTES, evidence=evidence
    )

    with pytest.raises(ValueError, match=message):
        resolve_uk_cut_reference(UK_NATIONAL_RELEASE, hub_download=hub)


def test_a_dense_build_with_a_failed_gate_is_refused(tmp_path: Path) -> None:
    hub = UKReferenceHub(tmp_path)
    hub.publish(
        UK_DENSE_RELEASE,
        DENSE_BUILD,
        dataset=DENSE_BYTES,
        evidence=dense_gate_summary(
            uk_local_target_fit={"failures": ["x"], "passed": False, "status": "failed"}
        ),
    )

    with pytest.raises(ValueError, match="did not pass uk_local_target_fit"):
        resolve_uk_cut_reference(UK_DENSE_RELEASE, hub_download=hub)


def test_a_release_without_a_current_build_is_refused(tmp_path: Path) -> None:
    hub = UKReferenceHub(tmp_path)
    hub.publish(
        UK_NATIONAL_RELEASE, NATIONAL_BUILD, dataset=NATIONAL_BYTES, current=False
    )

    with pytest.raises(ValueError, match="has no current build"):
        resolve_uk_cut_reference(UK_NATIONAL_RELEASE, hub_download=hub)


def test_a_withdrawn_current_build_is_refused(tmp_path: Path) -> None:
    hub = UKReferenceHub(tmp_path)
    hub.publish(UK_NATIONAL_RELEASE, NATIONAL_BUILD, dataset=NATIONAL_BYTES)
    record = json.loads((tmp_path / "hub" / "releases.json").read_text())
    record["releases"][UK_NATIONAL_RELEASE]["builds"][NATIONAL_BUILD]["withdrawn"] = (
        True
    )
    hub.add_file("releases.json", json.dumps(record).encode())

    with pytest.raises(ValueError, match="withdrawn"):
        resolve_uk_cut_reference(UK_NATIONAL_RELEASE, hub_download=hub)


def test_a_line_outside_the_cut_lines_is_refused(hub) -> None:
    with pytest.raises(ValueError, match="compares with the current build"):
        resolve_uk_cut_reference("populace-uk-2023", hub_download=hub)


def test_the_record_round_trips_and_pins_the_file(hub, tmp_path: Path) -> None:
    path, reference = resolve_uk_cut_reference(
        UK_NATIONAL_RELEASE, hub_download=hub, now=_fixed_now
    )
    record_path = write_uk_cut_reference(tmp_path / "cut" / "reference.json", reference)

    assert read_uk_cut_reference(record_path) == reference
    assert verify_uk_cut_reference_file(reference, path) == path

    other = tmp_path / "other.h5"
    other.write_bytes(b"x" * len(NATIONAL_BYTES))
    with pytest.raises(ValueError, match="has sha256"):
        verify_uk_cut_reference_file(reference, other)
    with pytest.raises(ValueError, match="is not at"):
        verify_uk_cut_reference_file(reference, tmp_path / "missing.h5")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: record.pop("evidence"),
        lambda record: record.update(kind="something_else"),
        lambda record: record.update(schema_version=2),
        lambda record: record.update(sha256="not-a-digest"),
        lambda record: record.update(size_bytes=0),
        lambda record: record.update(release="populace-uk-2023"),
        lambda record: record.update(
            built_with_model_package={"name": "policyengine-uk"}
        ),
    ],
)
def test_a_malformed_record_is_refused(hub, mutate) -> None:
    _, reference = resolve_uk_cut_reference(UK_NATIONAL_RELEASE, hub_download=hub)
    record = reference.to_record()
    mutate(record)

    with pytest.raises(ValueError):
        UKCutReference.from_record(record)


def test_a_build_withdrawn_after_the_fetch_no_longer_stands(hub) -> None:
    """The review of microcosm#1574 (S2): withdrawal is re-read, read-only."""

    from microcosm.build.uk_runtime.cut_reference import uk_cut_reference_withdrawal

    _, reference = resolve_uk_cut_reference(UK_NATIONAL_RELEASE, hub_download=hub)
    assert uk_cut_reference_withdrawal(reference, hub_download=hub) is None
    hub.publish(UK_NATIONAL_RELEASE, "uk-frs-calibration-attempt-later", dataset=b"x")
    # A promotion alone leaves the recorded build standing.
    assert uk_cut_reference_withdrawal(reference, hub_download=hub) is None
    hub.withdraw(UK_NATIONAL_RELEASE, reference.build)
    assert uk_cut_reference_withdrawal(reference, hub_download=hub) == (
        f"{reference.label} has been withdrawn since the cut resolved it"
    )
