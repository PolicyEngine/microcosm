"""Raw inputs checked against Chronicle registrations (microcosm#1524)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from microcosm.build.pinned_inputs import (
    licence_of,
    manifest_files,
    registration,
    sha256_of_file,
    verify_pinned_file,
)

_MANIFEST = """\
source_id: example
package_id: example-package-2026
license: Open Government Licence v3.0
license_url: https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/
attribution: 'Source: an example publisher.'
registration_kind: raw_only
files:
  2026:
    filename: data.csv
    source_url: https://example.org/data.csv
    sha256: {sha}
    size_bytes: {size}
    storage:
      local: false
      r2:
        key: raw/example/data.csv
  2025:
    filename: older.csv
    sha256: {sha}
    size_bytes: {size}
"""


def _write(tmp_path: Path, payload: bytes = b"a,b\n1,2\n") -> tuple[Path, Path]:
    data = tmp_path / "data.csv"
    data.write_bytes(payload)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        _MANIFEST.format(sha=hashlib.sha256(payload).hexdigest(), size=len(payload)),
        encoding="utf-8",
    )
    return data, manifest


def test_manifest_files_keys_the_entries_by_text(tmp_path: Path) -> None:
    _, path = _write(tmp_path)
    manifest, files = manifest_files(path)
    assert manifest["package_id"] == "example-package-2026"
    assert set(files) == {"2025", "2026"}
    assert files["2026"]["filename"] == "data.csv"


def test_manifest_files_refuses_a_manifest_without_files(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text("package_id: x\nfiles: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="declares no files"):
        manifest_files(path)


def test_verify_pinned_file_records_the_pin(tmp_path: Path) -> None:
    data, path = _write(tmp_path)
    _, files = manifest_files(path)
    record = verify_pinned_file(data, files["2026"])
    assert record == {
        "filename": "data.csv",
        "sha256": sha256_of_file(data),
        "size_bytes": data.stat().st_size,
        "source_url": "https://example.org/data.csv",
        "r2_key": "raw/example/data.csv",
    }
    # Without storage or a URL, both read as empty text.
    assert verify_pinned_file(data, files["2025"])["r2_key"] == ""


def test_verify_pinned_file_refuses_other_bytes(tmp_path: Path) -> None:
    data, path = _write(tmp_path)
    _, files = manifest_files(path)
    data.write_bytes(b"a,b\n1,3\n")  # same size, other bytes
    with pytest.raises(ValueError, match="sha256 differs"):
        verify_pinned_file(data, files["2026"])
    # Development runs may skip the digest, never the size.
    assert verify_pinned_file(data, files["2026"], skip=True)["filename"] == "data.csv"
    data.write_bytes(b"a,b\n")
    with pytest.raises(ValueError, match="size"):
        verify_pinned_file(data, files["2026"], skip=True)


def test_verify_pinned_file_refuses_a_link(tmp_path: Path) -> None:
    data, path = _write(tmp_path)
    _, files = manifest_files(path)
    link = tmp_path / "link.csv"
    link.symlink_to(data)
    with pytest.raises(ValueError, match="regular local file"):
        verify_pinned_file(link, files["2026"])
    with pytest.raises(ValueError, match="regular local file"):
        verify_pinned_file(tmp_path / "absent.csv", files["2026"])


def test_registration_orders_the_files(tmp_path: Path) -> None:
    _, path = _write(tmp_path)
    manifest, files = manifest_files(path)
    record = registration(manifest, {"2026": {"a": 1}, "2025": {"b": 2}})
    assert record["chronicle_package_id"] == "example-package-2026"
    assert record["registration_kind"] == "raw_only"
    assert list(record["files"]) == ["2025", "2026"]
    assert registration({}, {})["registration_kind"] == "committed"


def test_licence_of_quotes_the_manifest(tmp_path: Path) -> None:
    _, path = _write(tmp_path)
    manifest, _ = manifest_files(path)
    assert licence_of(manifest) == {
        "package_id": "example-package-2026",
        "license": "Open Government Licence v3.0",
        "license_url": (
            "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/"
        ),
        "attribution": "Source: an example publisher.",
        "license_source": "manifest",
    }
    with pytest.raises(ValueError, match="no licence"):
        licence_of({"package_id": "x"})
    with pytest.raises(ValueError, match="package_id"):
        licence_of({"license": "MIT"})


def test_a_declared_licence_fills_only_a_manifest_that_states_none() -> None:
    declared = {"license": "Open Government Licence v3.0", "attribution": "Ours."}
    assert licence_of({"package_id": "x"}, declared=declared) == {
        "package_id": "x",
        "license": "Open Government Licence v3.0",
        "attribution": "Ours.",
        "license_source": "declared",
    }
    # The manifest's own words win over a declaration.
    stated = {"package_id": "x", "license": "MIT"}
    assert licence_of(stated, declared=declared) == {
        "package_id": "x",
        "license": "MIT",
        "license_source": "manifest",
    }
    with pytest.raises(ValueError, match="but no licence"):
        licence_of({"package_id": "x", "attribution": "Theirs."}, declared=declared)
    with pytest.raises(ValueError, match="declared for x"):
        licence_of({"package_id": "x"}, declared={"attribution": "Ours."})
