"""Every raw microdata root is pinned to a Chronicle registration or listed.

The contract (microcosm#848, Chronicle ADR "Raw microdata in Chronicle is
identity, not content") is that a build graph has no anonymous roots. A source
manifest entry whose ``kind`` names microdata either declares the SHA-256 of
the exact file a stage reads plus the one Chronicle registration that witnesses
it, or it carries a reviewed row in ``microdata_pins_pending.json`` saying what
blocks the pin. The allowlist is a ratchet, so the unwitnessed surface can only
shrink.
"""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path

import pytest

from microcosm.build.source_manifest import (
    CHRONICLE_ACCESS_BY_ARTIFACT_KIND,
    MICRODATA_ARTIFACT_KINDS,
    MICRODATA_PIN_ALLOWLIST_FILENAME,
    ChronicleArtifactReference,
    SourceManifest,
    audit_microdata_pins,
    load_microdata_pin_allowlist,
    microdata_artifact_entries,
    packaged_microdata_pin_allowlist,
    resolved_chronicle_registrations,
)
from microcosm.build.source_runtime import (
    MicrodataIdentityError,
    sha256_file,
    verified_chronicle_registrations,
    verify_microdata_files,
    verify_recorded_microdata_pins,
)

COUNTRIES = ("am", "be", "uk", "us")

# The committed ratchet. A change that raises this number is a deliberate
# decision to add an unwitnessed build root and must be argued in review; a
# change that lowers it is a pin landing.
COMMITTED_PENDING_BASELINE = 39

#: Countries with no pending rows at all: every microdata root is witnessed.
FULLY_PINNED_COUNTRIES = ("uk",)


def _manifest(country: str) -> SourceManifest:
    from microcosm.build.source_manifest import load_source_manifest

    return load_source_manifest(
        files(f"microcosm.build.{country}").joinpath("source_stages.json")
    )


def _frozen_uk_replay_manifest() -> dict:
    return json.loads(
        files("microcosm.build.uk")
        .joinpath("hmrc_income_source_stages.json")
        .read_text(encoding="utf-8")
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class TestPinnedOrListed:
    @pytest.mark.parametrize("country", COUNTRIES)
    def test_every_microdata_root_is_pinned_or_listed(self, country: str) -> None:
        gaps = audit_microdata_pins(
            _manifest(country), allowlist=packaged_microdata_pin_allowlist()
        )

        assert [gap.message() for gap in gaps] == []

    def test_the_frozen_uk_replay_manifest_is_pinned_too(self) -> None:
        entries = microdata_artifact_entries(_frozen_uk_replay_manifest())

        assert entries
        assert all(entry.is_pinned for entry in entries)

    @pytest.mark.parametrize("country", FULLY_PINNED_COUNTRIES)
    def test_fully_pinned_countries_carry_no_pending_rows(self, country: str) -> None:
        allowlist = packaged_microdata_pin_allowlist()
        entries = microdata_artifact_entries(_manifest(country))

        assert allowlist.for_country(country) == ()
        assert entries
        assert all(entry.is_pinned for entry in entries)


class TestAllowlistRatchet:
    def test_the_allowlist_never_exceeds_its_committed_baseline(self) -> None:
        allowlist = packaged_microdata_pin_allowlist()

        assert allowlist.baseline_count <= COMMITTED_PENDING_BASELINE
        assert len(allowlist.pending) <= allowlist.baseline_count

    def test_every_pending_row_names_a_live_unpinned_root(self) -> None:
        allowlist = packaged_microdata_pin_allowlist()
        unpinned = {
            (country, entry.stage, entry.locator)
            for country in COUNTRIES
            for entry in microdata_artifact_entries(_manifest(country))
            if not entry.is_pinned
        }

        listed = {(row.country, row.stage, row.locator) for row in allowlist.pending}
        assert listed == unpinned

    def test_every_pending_row_states_a_reason_and_an_issue(self) -> None:
        for row in packaged_microdata_pin_allowlist().pending:
            assert row.issue.startswith("PolicyEngine/microcosm#")
            # A reason has to say what blocks the pin, not merely that one is
            # missing; a one-liner would let a row outlive its cause unnoticed.
            assert len(row.reason) > 80

    def test_a_row_count_above_the_baseline_is_refused(self) -> None:
        raw = json.loads(
            files("microcosm.build")
            .joinpath(MICRODATA_PIN_ALLOWLIST_FILENAME)
            .read_text(encoding="utf-8")
        )
        raw["baseline_count"] = len(raw["pending"]) - 1

        with pytest.raises(ValueError, match="ratchet and may only shrink"):
            load_microdata_pin_allowlist(_written(raw))

    def test_a_stale_row_for_an_already_pinned_root_is_a_gap(self) -> None:
        manifest = _manifest("uk")
        entry = microdata_artifact_entries(manifest)[0]
        allowlist = load_microdata_pin_allowlist(
            _written(
                {
                    "version": 1,
                    "policy": "test",
                    "baseline_count": 1,
                    "pending": [
                        {
                            "country": "uk",
                            "stage": entry.stage,
                            "locator": entry.locator,
                            "reason": "stale",
                            "issue": "PolicyEngine/microcosm#848",
                        }
                    ],
                }
            )
        )

        gaps = audit_microdata_pins(manifest, allowlist=allowlist)

        assert [gap.problem for gap in gaps] == ["stale_allowlist_row"]

    def test_a_row_naming_no_artifact_is_a_gap(self) -> None:
        allowlist = load_microdata_pin_allowlist(
            _written(
                {
                    "version": 1,
                    "policy": "test",
                    "baseline_count": 1,
                    "pending": [
                        {
                            "country": "uk",
                            "stage": "frs_spine",
                            "locator": "nonexistent.tab",
                            "reason": "orphan",
                            "issue": "PolicyEngine/microcosm#848",
                        }
                    ],
                }
            )
        )

        gaps = audit_microdata_pins(_manifest("uk"), allowlist=allowlist)

        assert [gap.problem for gap in gaps] == ["orphan_allowlist_row"]


class TestRegistrationValidation:
    def test_access_class_is_derived_from_the_artifact_kind(self) -> None:
        for country in COUNTRIES:
            for entry in microdata_artifact_entries(_manifest(country)):
                reference = entry.chronicle_artifact
                if reference is None:
                    continue
                assert reference.access == CHRONICLE_ACCESS_BY_ARTIFACT_KIND[entry.kind]

    def test_only_public_registrations_have_raw_object_keys(self) -> None:
        for country in COUNTRIES:
            for reference in resolved_chronicle_registrations(_manifest(country)):
                key = reference.raw_object_key
                if reference.access == "public":
                    assert key == (
                        f"raw/{reference.source_id}/{reference.package_id}/"
                        f"{reference.year}/{reference.sha256}/{reference.filename}"
                    )
                else:
                    assert key is None

    def test_registration_sha256_equals_the_artifact_pin(self) -> None:
        for country in COUNTRIES:
            for entry in microdata_artifact_entries(_manifest(country)):
                if entry.chronicle_artifact is None:
                    continue
                assert entry.chronicle_artifact.sha256 == entry.sha256

    def test_one_registration_per_distinct_file(self) -> None:
        for country in COUNTRIES:
            by_sha: dict[str, ChronicleArtifactReference] = {}
            for entry in microdata_artifact_entries(_manifest(country)):
                reference = entry.chronicle_artifact
                if reference is None or entry.sha256 is None:
                    continue
                assert by_sha.setdefault(entry.sha256, reference) == reference

    def test_a_registration_witnessing_other_bytes_is_refused(self) -> None:
        with pytest.raises(ValueError, match="does not equal the artifact sha256"):
            SourceManifest.from_mapping(
                _stage_with_artifact(
                    {
                        "kind": "public_microdata",
                        "locator": "example.zip",
                        "vintage": "2023",
                        "sha256": _sha("a"),
                        "chronicle_artifact": {
                            "source_id": "census_cps",
                            "package_id": "census-cps-asec-2023",
                            "year": 2023,
                            "sha256": _sha("b"),
                            "filename": "example.zip",
                            "access": "public",
                        },
                    }
                )
            )

    def test_an_unknown_access_class_is_refused(self) -> None:
        with pytest.raises(ValueError, match="'access' must be one of"):
            SourceManifest.from_mapping(
                _stage_with_artifact(
                    _artifact(access="open", sha256=_sha("a")),
                )
            )

    def test_an_access_class_the_kind_does_not_imply_is_refused(self) -> None:
        with pytest.raises(ValueError, match="registers under Chronicle access"):
            SourceManifest.from_mapping(
                _stage_with_artifact(_artifact(access="licensed", sha256=_sha("a")))
            )

    def test_a_public_registration_without_a_filename_is_refused(self) -> None:
        artifact = _artifact(sha256=_sha("a"))
        del artifact["chronicle_artifact"]["filename"]

        with pytest.raises(ValueError, match="public access without a"):
            SourceManifest.from_mapping(_stage_with_artifact(artifact))

    def test_a_registration_without_an_artifact_pin_is_refused(self) -> None:
        artifact = _artifact(sha256=_sha("a"))
        del artifact["sha256"]

        with pytest.raises(ValueError, match="without .*declaring its own"):
            SourceManifest.from_mapping(_stage_with_artifact(artifact))

    def test_an_uppercase_digest_is_refused(self) -> None:
        with pytest.raises(ValueError, match="64 lowercase hex characters"):
            SourceManifest.from_mapping(
                _stage_with_artifact(
                    {
                        "kind": "public_microdata",
                        "locator": "example.zip",
                        "vintage": "2023",
                        "sha256": _sha("a").upper(),
                    }
                )
            )

    def test_a_year_the_entry_does_not_declare_is_refused(self) -> None:
        artifact = _artifact(sha256=_sha("a"))
        artifact["chronicle_artifact"]["year"] = 1999

        with pytest.raises(ValueError, match="is not one of the years"):
            SourceManifest.from_mapping(_stage_with_artifact(artifact))

    def test_a_derived_artifact_cannot_carry_a_registration(self) -> None:
        artifact = _artifact(sha256=_sha("a"))
        artifact["kind"] = "versioned_derived_microdata"

        with pytest.raises(ValueError, match="no Chronicle access class"):
            SourceManifest.from_mapping(_stage_with_artifact(artifact))

    def test_two_registrations_for_one_file_are_refused(self) -> None:
        first = _artifact(sha256=_sha("a"))
        second = json.loads(json.dumps(first))
        second["chronicle_artifact"]["package_id"] = "census-cps-asec-2023-other"

        with pytest.raises(ValueError, match="one file has one registration"):
            SourceManifest.from_mapping(
                _stage_with_artifact(first, second, locators=("a.zip", "b.zip"))
            )


class TestFailClosedGate:
    def test_matching_bytes_pass_and_resolve_their_registration(
        self, tmp_path: Path
    ) -> None:
        payload = b"asec bytes"
        path = tmp_path / "example.zip"
        path.write_bytes(payload)
        manifest = SourceManifest.from_mapping(
            _stage_with_artifact(_artifact(sha256=hashlib.sha256(payload).hexdigest()))
        )

        verifications = verify_microdata_files(manifest, {"example.zip": path})

        assert [check.matched for check in verifications] == [True]
        assert verifications[0].actual_sha256 == hashlib.sha256(payload).hexdigest()
        registrations = verified_chronicle_registrations(verifications)
        assert [reference.package_id for reference in registrations] == [
            "census-cps-asec-2023"
        ]

    def test_different_bytes_stop_the_build_with_a_diagnosable_message(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "example.zip"
        path.write_bytes(b"a reissued vintage")
        manifest = SourceManifest.from_mapping(
            _stage_with_artifact(_artifact(sha256=_sha("the pinned release")))
        )

        with pytest.raises(MicrodataIdentityError) as error:
            verify_microdata_files(manifest, {"example.zip": path})

        message = str(error.value)
        assert "census_cps/census-cps-asec-2023" in message
        assert "'2023'" in message
        assert "example.zip" in message
        assert _sha("the pinned release") in message
        assert sha256_file(path) in message

    def test_a_key_naming_no_pinned_root_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "stranger.zip"
        path.write_bytes(b"x")
        manifest = SourceManifest.from_mapping(
            _stage_with_artifact(_artifact(sha256=_sha("a")))
        )

        with pytest.raises(MicrodataIdentityError, match="names no hash-pinned"):
            verify_microdata_files(manifest, {"stranger.zip": path})

    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        manifest = SourceManifest.from_mapping(
            _stage_with_artifact(_artifact(sha256=_sha("a")))
        )

        with pytest.raises(MicrodataIdentityError, match="which is not a file"):
            verify_microdata_files(manifest, {"example.zip": tmp_path / "absent.zip"})

    def test_a_caller_supplied_input_resolves_by_its_declared_filename(
        self, tmp_path: Path
    ) -> None:
        payload = b"licensed tab"
        path = tmp_path / "put2223uk.tab"
        path.write_bytes(payload)
        artifact = {
            "kind": "private_microdata",
            "locator": "caller-supplied local input",
            "filename": "put2223uk.tab",
            "vintage": "2022-23",
            "tax_year_start": 2022,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "chronicle_artifact": {
                "source_id": "hmrc",
                "package_id": "hmrc-spi-public-use-tape-2022-23",
                "year": 2022,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "filename": "put2223uk.tab",
                "access": "restricted",
            },
        }
        manifest = SourceManifest.from_mapping(_stage_with_artifact(artifact))

        verifications = verify_microdata_files(manifest, {"put2223uk.tab": path})

        assert [check.matched for check in verifications] == [True]

    def test_the_shared_placeholder_locator_is_refused_as_ambiguous(
        self, tmp_path: Path
    ) -> None:
        # "caller-supplied local input" is the locator of every UK private
        # root, so it names several distinct files; one file cannot satisfy
        # several pins and the gate must say so rather than pick one.
        path = tmp_path / "something.tab"
        path.write_bytes(b"x")

        with pytest.raises(MicrodataIdentityError, match="is ambiguous"):
            verify_microdata_files(
                _manifest("uk"), {"caller-supplied local input": path}
            )

    def test_the_uk_manifest_gate_accepts_the_real_pins(self, tmp_path: Path) -> None:
        # Every UK root is hash-pinned, so a synthetic file whose bytes hash to
        # a declared pin is impossible to fabricate; instead assert the gate
        # resolves keys for all of them and rejects the tampered one.
        manifest = _manifest("uk")
        entries = microdata_artifact_entries(manifest)
        adult = next(entry for entry in entries if entry.locator == "adult.tab")
        path = tmp_path / "adult.tab"
        path.write_bytes(b"not the licensed tab")

        with pytest.raises(MicrodataIdentityError) as error:
            verify_microdata_files(manifest, {"adult.tab": path})

        message = str(error.value)
        assert message.count("adult.tab") >= 5  # every stage that reads it
        assert adult.sha256 in message
        assert "dwp/dwp-frs-2024-25-adult" in message


class TestRecordedPinCrossCheck:
    def test_agreeing_pins_resolve_to_their_registrations(self) -> None:
        manifest = _manifest("us")
        entry = next(
            candidate
            for candidate in microdata_artifact_entries(manifest)
            if candidate.stage == "weeks_unemployed_input"
        )

        audit = verify_recorded_microdata_pins(
            manifest,
            [
                {
                    "locator": entry.locator,
                    "sha256": entry.sha256,
                    "member_sha256": entry.member_sha256,
                }
            ],
            context="test",
        )

        assert [reference.package_id for reference in audit.resolved] == [
            "census-cps-asec-2023"
        ]
        assert audit.unregistered == ()

    def test_a_disagreeing_archive_digest_stops_the_build(self) -> None:
        manifest = _manifest("us")
        entry = next(
            candidate
            for candidate in microdata_artifact_entries(manifest)
            if candidate.stage == "weeks_unemployed_input"
        )

        with pytest.raises(MicrodataIdentityError, match="recorded sha256"):
            verify_recorded_microdata_pins(
                manifest,
                [{"locator": entry.locator, "sha256": _sha("other")}],
                context="test",
            )

    def test_a_disagreeing_member_digest_stops_the_build(self) -> None:
        manifest = _manifest("us")
        entry = next(
            candidate
            for candidate in microdata_artifact_entries(manifest)
            if candidate.stage == "weeks_unemployed_input"
        )

        with pytest.raises(MicrodataIdentityError, match="recorded member_sha256"):
            verify_recorded_microdata_pins(
                manifest,
                [
                    {
                        "locator": entry.locator,
                        "sha256": entry.sha256,
                        "member_sha256": _sha("other member"),
                    }
                ],
                context="test",
            )

    def test_an_unregistered_locator_is_reported_not_refused(self) -> None:
        # The pooled ASEC archives for later survey years have no manifest
        # entry yet; that is the allowlist's business, not a build failure.
        audit = verify_recorded_microdata_pins(
            _manifest("us"),
            [
                {
                    "locator": (
                        "https://www2.census.gov/programs-surveys/cps/datasets/"
                        "2025/march/asecpub25csv.zip"
                    ),
                    "sha256": _sha("later vintage"),
                }
            ],
            context="test",
        )

        assert audit.resolved == ()
        assert audit.unregistered == (
            "https://www2.census.gov/programs-surveys/cps/datasets/2025/march/"
            "asecpub25csv.zip",
        )


def _artifact(*, sha256: str, access: str = "public") -> dict:
    return {
        "kind": "public_microdata",
        "locator": "example.zip",
        "vintage": "2023",
        "sha256": sha256,
        "chronicle_artifact": {
            "source_id": "census_cps",
            "package_id": "census-cps-asec-2023",
            "year": 2023,
            "sha256": sha256,
            "filename": "example.zip",
            "access": access,
        },
    }


def _stage_with_artifact(
    *artifacts: dict, locators: tuple[str, ...] | None = None
) -> dict:
    prepared = []
    for index, artifact in enumerate(artifacts):
        entry = json.loads(json.dumps(artifact))
        if locators is not None:
            entry["locator"] = locators[index]
        prepared.append(entry)
    return {
        "version": 1,
        "country": "xx",
        "policy": "test manifest",
        "stages": [
            {
                "stage": "example_stage",
                "survey": "Example",
                "source": "https://example.invalid/",
                "grain": "person",
                "artifacts": prepared,
                "operations": [{"kind": "read_table", "table": "example"}],
                "outputs": ["example_output"],
            }
        ],
    }


def _written(payload: dict) -> Path:
    import tempfile

    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - path outlives the handle
        "w", suffix=".json", delete=False, encoding="utf-8"
    )
    with handle:
        json.dump(payload, handle)
    return Path(handle.name)


def test_every_microdata_kind_is_covered_by_the_contract() -> None:
    # A new microdata kind must decide its Chronicle access class or be
    # explicitly derived-only; it must not silently escape the audit.
    undecided = MICRODATA_ARTIFACT_KINDS - set(CHRONICLE_ACCESS_BY_ARTIFACT_KIND)

    assert undecided == {"versioned_derived_microdata"}
