"""S0101-only capture on invented response bytes, without network or inventories."""

import hashlib
import json
import os
from urllib.parse import parse_qs, urlsplit

import pytest

from microcosm.build.us_runtime import survey_age_sources as source


def test_only_two_fixed_national_subject_requests_are_declared():
    metadata, data = source.survey_age_requests()
    assert metadata["url"].endswith("/2024/acs/acs1/subject/groups/S0101.json")
    assert parse_qs(urlsplit(data["url"]).query) == {
        "get": ["group(S0101)"],
        "for": ["us:*"],
    }
    assert {r["kind"] for r in (metadata, data)} == {"metadata", "data"}
    assert all(r["table"] == "S0101" and r["year"] == 2024 for r in (metadata, data))
    data["table"] = "changed"
    assert source.survey_age_requests()[1]["table"] == "S0101"


@pytest.mark.parametrize(
    "change",
    [
        {"table": "B19001"},
        {"table": "B25003"},
        {"year": True},
        {"year": 2023},
        {"geography": "congressional district:*"},
        {"url": "https://example.invalid/data"},
        {"extra": "unsupported"},
    ],
)
def test_request_refuses_before_touching_output_root(tmp_path, change):
    request = {**source.survey_age_requests()[1], **change}
    root = tmp_path / "must-remain-absent"
    with pytest.raises(ValueError, match="SURVEY_AGE_SOURCE_REQUEST"):
        source.capture_survey_age_response(root, request, b"[]")
    assert not root.exists()


def test_capture_roundtrip_reads_only_two_named_response_files(tmp_path):
    request = source.survey_age_requests()[0]
    payload = b'{"variables":{}}'
    descriptor = source.capture_survey_age_response(tmp_path, request, payload)
    assert descriptor["sha256"] == hashlib.sha256(payload).hexdigest()
    assert (
        source.read_survey_age_response(tmp_path, "metadata", descriptor["sha256"])[0]
        == payload
    )
    assert source.capture_survey_age_response(tmp_path, request, payload) == descriptor
    assert not (tmp_path / "national-derived-inventory.json").exists()
    with pytest.raises(ValueError, match="IMMUTABLE"):
        source.capture_survey_age_response(tmp_path, request, b'{"variables":{"x":1}}')


@pytest.mark.parametrize("payload", [b'{"a":1,"a":2}', b'{"a":NaN}', b"not json", b""])
def test_invalid_response_never_creates_capture(tmp_path, payload):
    root = tmp_path / "absent"
    with pytest.raises(ValueError):
        source.capture_survey_age_response(
            root, source.survey_age_requests()[0], payload
        )
    assert not root.exists()


def test_self_rehashed_response_does_not_replace_reviewed_pin(tmp_path):
    request = source.survey_age_requests()[0]
    original = source.capture_survey_age_response(
        tmp_path, request, b'{"variables":{}}'
    )
    changed = b'{"variables":{"changed":true}}'
    digest = hashlib.sha256(changed).hexdigest()
    (tmp_path / "raw" / f"{digest}.json").write_bytes(changed)
    descriptor = {
        **original,
        "sha256": digest,
        "size_bytes": len(changed),
        "path": f"raw/{digest}.json",
    }
    key = hashlib.sha256(request["url"].encode()).hexdigest()
    (tmp_path / "requests" / f"{key}.json").write_text(json.dumps(descriptor))
    with pytest.raises(ValueError, match="PIN"):
        source.read_survey_age_response(tmp_path, "metadata", original["sha256"])


@pytest.mark.parametrize("member", ["root", "raw", "descriptor", "response"])
def test_symlink_capture_members_refuse(tmp_path, member):
    real = tmp_path / "real"
    request = source.survey_age_requests()[0]
    descriptor = source.capture_survey_age_response(real, request, b'{"variables":{}}')
    root = real
    if member == "root":
        root = tmp_path / "alias"
        root.symlink_to(real, target_is_directory=True)
    elif member == "raw":
        (real / "raw").rename(real / "original-raw")
        (real / "raw").symlink_to(real / "original-raw", target_is_directory=True)
    else:
        relative = (
            descriptor["path"]
            if member == "response"
            else "requests/"
            + hashlib.sha256(request["url"].encode()).hexdigest()
            + ".json"
        )
        path = real / relative
        saved = path.with_suffix(".saved")
        path.rename(saved)
        path.symlink_to(saved)
    with pytest.raises(ValueError, match="SYMLINK"):
        source.read_survey_age_response(root, "metadata", descriptor["sha256"])


def test_reader_rejects_unbounded_descriptor_before_decoding(tmp_path):
    request = source.survey_age_requests()[0]
    descriptor = source.capture_survey_age_response(
        tmp_path, request, b'{"variables":{}}'
    )
    key = hashlib.sha256(request["url"].encode()).hexdigest()
    path = tmp_path / "requests" / f"{key}.json"
    path.write_bytes(b" " * (source.MAX_DESCRIPTOR_BYTES + 1))
    with pytest.raises(ValueError, match="SIZE"):
        source.read_survey_age_response(tmp_path, "metadata", descriptor["sha256"])


@pytest.mark.parametrize("member", ["descriptor", "response"])
def test_fifo_capture_members_refuse_without_waiting_for_a_writer(
    tmp_path, monkeypatch, member
):
    request = source.survey_age_requests()[0]
    descriptor = source.capture_survey_age_response(
        tmp_path, request, b'{"variables":{}}'
    )
    key = hashlib.sha256(request["url"].encode()).hexdigest()
    path = tmp_path / (
        f"requests/{key}.json" if member == "descriptor" else descriptor["path"]
    )
    path.unlink()
    os.mkfifo(path)
    original_open = os.open

    def guarded_open(filename, flags, *args, **kwargs):
        if filename == path:
            # Fail promptly on regression instead of hanging the test process.
            assert flags & os.O_NONBLOCK
        return original_open(filename, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", guarded_open)
    with pytest.raises(ValueError, match="REGULAR_FILE"):
        source.read_survey_age_response(tmp_path, "metadata", descriptor["sha256"])
