"""Generate or verify the release-contract golden fixtures (version 1).

docs/release-contract.md defines the release contract, C1 to C5. These fixtures
freeze two of its parts in bytes, the way ``generate_staging_contract_fixtures.py``
freezes the staging documents that calibration-diagnostics byte-checks. Each
golden comes from the writer that produces the same file in a real build, and
``writers.json`` names that writer for every file ``SHA256SUMS`` lists.

C1, the release manifest
------------------------
One build folder's ``build_manifest.json`` and ``release_manifest.json`` for the
UK national release and for the US national release:

* The UK pair. ``uk_national_release_manifests`` builds both documents and
  ``_write_json`` serializes them; both live in
  ``tools/assemble_uk_release_dir.py``.
* The US pair. ``_build_manifests`` in ``tools/build_us_fiscal_refresh_release.py``
  writes both manifests, and ``_finish_build_folder_release`` rewrites them into
  the build-folder layout by calling
  ``microcosm.data.build_folder_manifests.rewrite_release_dir``. The fixture
  freezes that function's serialization of the builder's content.
* ``c1/inventory.json`` lists the files and certification fields the release
  contract requires of the UK national release. The generator reads them from
  ``microcosm.data.contract``.

C5, publication
---------------
One ``releases.json`` for each country's dataset repository, and
``c5/registry.json``, a snapshot of ``microcosm.data.registry.REGISTRY``.

The emitter's release writer is the only writer of a live ``releases.json``. It
lives outside this repository, and ``microcosm_emitter_interface.releases``
holds its rules. The generator builds each record from three sources:

1. ``add_build`` with ``make_current=False``, the interface's change for a
   publication;
2. ``make_current``, the interface's change for a promotion;
3. ``dataclasses.replace(record, default=...)``, the step
   ``microcosm.build.build_folder_switch._switch_record`` takes to set
   ``default``. The interface has no function that sets it. Both goldens name
   the national release as the default.

``ReleasesRecord.serialize`` then gives the bytes.

Seams the generator replaces
----------------------------
The UK writer is a pure function, so it needs no seam: the clock, the runtime
versions and each certification part's census arrive as arguments.

The US builder reads the clock, the checkout and the installed packages through
module globals, and ``_build_manifests`` computes the calibration gate itself.
The generator replaces four globals of the builder's module and nothing else:

* ``_runtime_versions`` returns the synthetic versions in ``US_RUNTIME``;
* ``_git_output`` returns one synthetic commit for ``rev-parse HEAD``;
* ``datetime`` becomes a clock fixed at ``FIXTURE_TIME``;
* ``_release_gate_failures`` returns no failures, so the US build manifest
  records a passing calibration gate, as the UK inputs give each certification
  part a passing census.

Synthetic and live values
-------------------------
Every digest, version, identifier and gate count in the manifests and records
is synthetic. A digest is the SHA-256 of ``release-contract-fixture:<name>``, so
no reader can mistake one for a real artifact; the three JSON files the US
builder hashes in its release directory hold
``{"release_contract_fixture": "<file name>"}`` instead, because the builder
parses the coverage file. The SPI Public Use Tape marker names one synthetic
stage; its study number and conditions are the live constants of
``microcosm.build.uk_runtime.spi_dependency``.

Some values are live on purpose. ``c1/inventory.json`` and ``c5/registry.json``
freeze the contract's constants and the registry. The US manifests carry what
the US builder writes for this generator's call, which passes
``_build_manifests`` only its required arguments and a staging block: the
layout of ``_build_manifests`` and ``rewrite_release_dir``, the gate-evidence keys
(``US_RELEASE_GATE_EVIDENCE_FILES``), the documented-absent inputs
(``US_DOCUMENTED_ABSENT_INPUTS``), the calibration diagnostics' schema version
(``CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION``), the fiscal-target exclusion
receipt's key, the dataset and calibration keys and file names, and the
repository id. A pull request that changes one of those regenerates the
fixtures. A real build also writes its timing, its ledger provenance, its
optional gate blocks, the other optional blocks its run passes (``base_pool``
and ``post_export_scoring``, for example), and the blocks that its
gate-evidence files bind, and the fixtures freeze none of those.

What the fixtures omit
----------------------
No fixture holds a signed certification, a gate-battery report, calibration
diagnostics, a source-coverage file or a stored-input verdict. Those bind
digests of live registers, which most pull requests move. The tests in
``test_release_contract_fixtures.py`` pin exactly which refusals each omission
causes.

Regenerate after a deliberate contract change, in the same pull request that
updates docs/release-contract.md::

    uv run python tools/generate_release_contract_fixtures.py

``--check`` regenerates into a temporary directory and fails, naming each file
that differs, unless every byte matches the committed fixtures.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace

from microcosm_emitter_interface.releases import (
    ReleasesRecord,
    add_build,
    make_current,
)

from microcosm.build.staging_v2 import validate_staging_delivery
from microcosm.build.uk_runtime.spi_dependency import uk_spi_public_use_tape_marker
from microcosm.build.uk_runtime.spi_income import SPI_DONOR_UKDS_STUDY
from microcosm.data import contract
from microcosm.data.registry import DEFAULT_VARIANT, REGISTRY
from microcosm.data.releases_record import RELEASES_RECORD_PATH

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "packages"
    / "microcosm-data"
    / "tests"
    / "fixtures"
    / "release_contract"
    / "v1"
)
GENERATOR = "tools/generate_release_contract_fixtures.py"
ASSEMBLER = "tools/assemble_uk_release_dir.py"
US_BUILDER = "tools/build_us_fiscal_refresh_release.py"
INTERFACE_RELEASES = "microcosm_emitter_interface.releases"

#: One instant for every timestamp the fixtures carry.
FIXTURE_TIME = "2026-01-01T00:00:00+00:00"
UK_RELEASE = contract._UK_NATIONAL_RELEASE_ID
US_RELEASE = contract.US_NATIONAL_RELEASE_ID
#: The UK national build's name: a calibration attempt ID.
UK_BUILD = "uk-frs-calibration-attempt-20260101T000000Z-00000000"
#: The US national build's name: a build ID with the ``populace-us-`` prefix
#: the US builder requires (``_assert_us_release_id``).
US_BUILD = "populace-us-2024-0000000-20260101"
UK_LINE = "national"
#: Synthetic runtime versions. Each is a resolved PEP 440 version, which is all the
#: contract asks of a runtime pin.
UK_RUNTIME = {
    "python": "3.13.0",
    "policyengine-core": "3.0.0",
    "policyengine-uk": "2.0.0",
    "microcosm-data": "0.1.0",
    "microcosm-build": "0.1.0",
    "microcosm-calibrate": "0.1.0",
}
#: The packages the US builder's ``_runtime_versions`` reports, in its order.
US_RUNTIME = {
    "python": "3.13.0",
    "microcosm-build": "0.1.0",
    "microcosm-calibrate": "0.1.0",
    "microcosm-data": "0.1.0",
    "microcosm-frame": "0.1.0",
    "policyengine-core": "3.0.0",
    "policyengine-us": "1.0.0",
    "numpy": "2.0.0",
    "pandas": "2.0.0",
    "torch": "2.0.0",
}
#: A successful version 2 staging delivery, matching the staging fixtures'
#: ``delivery_success`` case apart from its run id.
UK_STAGING_DELIVERY = {
    "contract_version": 2,
    "enabled": True,
    "mode": "local_and_remote",
    "run_id": "uk-national-release-contract-fixture",
    "configured_repository": "policyengine/populace-uk-staging",
    "upload_attempts": 6,
    "upload_successes": 6,
    "read_back": "passed",
    "last_error_code": None,
    "opt_out_reason": None,
}
#: One synthetic spine stage that reads the SPI Public Use Tape, so the UK
#: manifests carry the marker a tape-dependent national release restates
#: (microcosm#1095): the flag on the microdata entry and ``licensed_data``.
UK_SPI_STAGE = SimpleNamespace(
    stage="release_contract_fixture_stage",
    artifacts=[{"ukds_study_number": SPI_DONOR_UKDS_STUDY}],
)
#: The release-directory files the US builder hashes into its manifests. The
#: fixtures ship none of them.
_US_HASHED_FILES = (
    "calibration_diagnostics.json",
    "us_source_coverage.json",
    "us_ssi_take_up.json",
)


def _writer(module: str, function: str) -> dict[str, str]:
    return {"module": module, "function": function}


_UK_MANIFEST_WRITERS = [
    _writer(ASSEMBLER, "uk_national_release_manifests"),
    _writer(ASSEMBLER, "_write_json"),
]
_US_MANIFEST_WRITERS = [
    _writer(US_BUILDER, "_build_manifests"),
    _writer(US_BUILDER, "_finish_build_folder_release"),
    _writer("microcosm.data.build_folder_manifests", "rewrite_release_dir"),
]
_RELEASES_RECORD_WRITERS = [
    _writer(INTERFACE_RELEASES, "add_build"),
    _writer(INTERFACE_RELEASES, "make_current"),
    _writer("dataclasses", "replace"),
    _writer(INTERFACE_RELEASES, "ReleasesRecord.serialize"),
]
#: Every fixture file ``SHA256SUMS`` lists, with the calls that produce its
#: bytes, in call order: the last one serializes. A module is a dotted name, or
#: a path from the repository root for a tool. ``_writers`` writes this table as
#: ``writers.json`` and refuses a run whose files it does not cover exactly.
WRITERS: dict[str, list[dict[str, str]]] = {
    "c1/inventory.json": [_writer(GENERATOR, "_c1_inventory")],
    f"c1/{UK_RELEASE}/build_manifest.json": _UK_MANIFEST_WRITERS,
    f"c1/{UK_RELEASE}/release_manifest.json": _UK_MANIFEST_WRITERS,
    f"c1/{US_RELEASE}/build_manifest.json": _US_MANIFEST_WRITERS,
    f"c1/{US_RELEASE}/release_manifest.json": _US_MANIFEST_WRITERS,
    "c5/registry.json": [_writer(GENERATOR, "_c5_registry")],
    f"c5/uk/{RELEASES_RECORD_PATH}": _RELEASES_RECORD_WRITERS,
    f"c5/us/{RELEASES_RECORD_PATH}": _RELEASES_RECORD_WRITERS,
    "writers.json": [_writer(GENERATOR, "_writers")],
}


def fixture_digest(name: str) -> str:
    """The synthetic SHA-256 the fixtures use for ``name``."""

    return hashlib.sha256(_fixture_bytes(name)).hexdigest()


def _fixture_bytes(name: str) -> bytes:
    """A stub file's bytes, whose SHA-256 is ``fixture_digest(name)``."""

    return f"release-contract-fixture:{name}".encode()


def _fixture_commit() -> str:
    return hashlib.sha1(b"release-contract-fixture").hexdigest()


def _load(tool: str) -> ModuleType:
    """Import one of the repository's tools from its file."""

    path = ROOT / tool
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _c1_uk_national(root: Path, assembler: ModuleType) -> None:
    """Write the UK national build folder's manifests with the UK assembler."""

    measured = {
        name: fixture_digest(name)
        for name in ("candidate", *(name for _, _, name in assembler._EVIDENCE))
    }
    build_manifest, release_manifest = assembler.uk_national_release_manifests(
        measured=measured,
        calibration_filename=f"{assembler._DATASET_KEY}_calibration.npz",
        calibration_sha=fixture_digest("calibration"),
        household_weight_year={"dataset_year": 2024, "basis": "dataset_year"},
        attempt_id=UK_BUILD,
        runtime=UK_RUNTIME,
        code_pin=_fixture_commit(),
        target_surface={"sha256": fixture_digest("target surface"), "n_targets": 2},
        target_registry={"version": "release-contract-fixture-v1", "n_specs": 2},
        # Each part's statuses are the certifier's census of its gates by status
        # (``_status_census``); a shippable part is all "passed". The counts are
        # synthetic, so the fixture does not move when a UK gate scope grows.
        spine_part={"statuses": {"passed": 3}},
        seam_part={"statuses": {"passed": 2}},
        release_cut_part={"statuses": {"passed": 4}},
        staging_delivery=validate_staging_delivery(UK_STAGING_DELIVERY),
        spi_marker=uk_spi_public_use_tape_marker([UK_SPI_STAGE]),
        created_at=FIXTURE_TIME,
        publisher_labels={"ons": "Office for National Statistics"},
    )
    release_dir = root / "c1" / UK_RELEASE
    assembler._write_json(release_dir / "build_manifest.json", build_manifest)
    assembler._write_json(release_dir / "release_manifest.json", release_manifest)


def _c1_inventory(root: Path, assembler: ModuleType) -> None:
    """Write what the live contract requires of the UK national release."""

    dataset_role = contract.dataset_role_for_line(UK_LINE)
    inventory = {
        "schema_name": "microcosm.release-contract.c1-inventory",
        "schema_version": 1,
        "release_id": UK_RELEASE,
        "line": contract.line_for_release_id(UK_RELEASE),
        "dataset_role": dataset_role,
        "release_role": contract.RELEASE_ROLES[dataset_role],
        "publishes_to_build_folder": contract.publishes_to_build_folder(UK_RELEASE),
        "retired_build_fields": list(contract.RETIRED_BUILD_FIELDS),
        "release_manifest_schema_version": contract.RELEASE_MANIFEST_SCHEMA_VERSION,
        "required_release_files": list(contract.required_release_files(UK_RELEASE)),
        "release_directory_files": sorted(
            [
                "build_manifest.json",
                "release_manifest.json",
                *(filename for _, filename, _ in assembler._EVIDENCE),
            ]
        ),
        "certification": {
            "file": contract._UK_RELEASE_CERTIFICATION_FILE,
            "schema_version": contract._UK_RELEASE_CERTIFICATION_SCHEMA_VERSION,
            "kind": contract._UK_RELEASE_CERTIFICATION_KIND,
            "required_fields": sorted(contract._UK_CERTIFICATION_REQUIRED_FIELDS),
            "part_evidence_files": dict(contract._UK_CERTIFICATION_PART_EVIDENCE_FILES),
            "score_receipt_file": contract._UK_CERTIFICATION_SCORE_RECEIPT_FILE,
            "gate_battery_phases": list(contract._UK_GATE_BATTERY_PHASES),
            "signing_key_env": contract._UK_GATE_BATTERY_SIGNING_KEY_ENV,
        },
    }
    path = root / "c1" / "inventory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(inventory))


class _FixtureClock(datetime):
    """``datetime`` whose ``now`` is the fixture instant."""

    @classmethod
    def now(cls, tz=None):
        return datetime.fromisoformat(FIXTURE_TIME).astimezone(tz)


class _FixtureTargetRegistry:
    """The two things ``_build_manifests`` reads from the target registry."""

    version = "release-contract-fixture-v1"

    def __len__(self) -> int:
        return 1


def _c1_us_national(root: Path) -> None:
    """Write the US national build folder's manifests with the US builder.

    The module docstring names the four globals of the builder's module this
    replaces. Everything else is the builder's own code: ``_build_manifests``,
    ``_staging_manifest_block`` and ``_finish_build_folder_release``.
    """

    builder = _load(US_BUILDER)
    builder._runtime_versions = lambda: dict(US_RUNTIME)
    builder._git_output = lambda *args: (
        _fixture_commit() if args == ("rev-parse", "HEAD") else ""
    )
    builder.datetime = _FixtureClock
    # Given a synthetic solve, the builder's own gate fails and lists every
    # critical fiscal target the solve's diagnostics lack, so a golden that
    # kept its verdict would move with the US critical-target list. This seam
    # returns no failures, and the solve result below then needs no diagnostics.
    builder._release_gate_failures = lambda *args, **kwargs: []
    with tempfile.TemporaryDirectory(prefix="release-contract-us-") as temporary:
        work = Path(temporary)
        release_dir = work / "releases" / US_RELEASE
        release_dir.mkdir(parents=True)
        artifact_root = work / "artifacts"
        artifact_root.mkdir()
        (artifact_root / builder.DATASET_FILENAME).write_bytes(
            _fixture_bytes("dataset")
        )
        (artifact_root / builder.CALIBRATION_FILENAME).write_bytes(
            _fixture_bytes("calibration")
        )
        for name in _US_HASHED_FILES:
            # JSON, because the builder parses the coverage file for the
            # fiscal-target exclusion receipt.
            (release_dir / name).write_bytes(
                json.dumps({"release_contract_fixture": name}).encode()
            )
        builder._build_manifests(
            release_id=US_BUILD,
            release_dir=release_dir,
            artifact_root=artifact_root,
            result=SimpleNamespace(
                initial_loss=2.0, final_loss=1.0, fraction_within_10pct=1.0
            ),
            target_surface={
                "sha256": fixture_digest("target surface"),
                "n_targets": 1,
            },
            registry=_FixtureTargetRegistry(),
            dropped={"dropped_target_names": []},
            target_profile_gate=builder.GateResult(
                name="target_profile_coverage",
                passed=True,
                details={"requirements_checked": 1},
            ),
            staging=builder._staging_manifest_block(
                SimpleNamespace(
                    staging_bundle=object(),
                    run_id="us-national-release-contract-fixture",
                    repo_id="policyengine/populace-us-staging",
                    uploads_succeeded=True,
                )
            ),
        )
        builder._finish_build_folder_release(
            release_dir,
            build=US_BUILD,
            artifact_root=artifact_root,
            evidence_release=False,
        )
        target = root / "c1" / US_RELEASE
        target.mkdir(parents=True, exist_ok=True)
        for name in ("build_manifest.json", "release_manifest.json"):
            (target / name).write_bytes((release_dir / name).read_bytes())


def _one_current_build(*, release: str, build: str, role: str) -> ReleasesRecord:
    """A repository's record after one build is published, then promoted.

    The module docstring names the source of each of the three steps.
    """

    published = add_build(
        ReleasesRecord.empty(),
        release=release,
        build=build,
        release_role=role,
        published_at=FIXTURE_TIME,
        make_current=False,
    )
    promoted = make_current(published, release=release, build=build)
    return dataclasses.replace(promoted, default=release)


def _c5_releases(root: Path) -> None:
    """Write each country's ``releases.json`` with the interface's serializer."""

    records = {
        "uk": _one_current_build(
            release=UK_RELEASE,
            build=UK_BUILD,
            role=contract.RELEASE_ROLES[contract.dataset_role_for_line(UK_LINE)],
        ),
        "us": _one_current_build(
            release=US_RELEASE,
            build=US_BUILD,
            role=contract.US_RELEASES[US_RELEASE],
        ),
    }
    for country, record in records.items():
        path = root / "c5" / country / RELEASES_RECORD_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(record.serialize())


def _c5_registry(root: Path) -> None:
    """Write a snapshot of the live dataset registry."""

    registry = {
        "schema_name": "microcosm.release-contract.registry",
        "schema_version": 1,
        "default_variant": DEFAULT_VARIANT,
        "entries": [
            {"key": list(key), **dataclasses.asdict(REGISTRY[key])}
            for key in sorted(REGISTRY)
        ],
    }
    path = root / "c5" / "registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(registry))


def _writers(root: Path) -> None:
    """Write ``writers.json``; refuse a run that ``WRITERS`` does not cover."""

    path = root / "writers.json"
    written = {*_tree(root), path.name}
    if written != set(WRITERS):
        raise SystemExit(
            "WRITERS must name a writer for exactly the files the generator "
            f"writes; it differs on {sorted(written ^ set(WRITERS))}"
        )
    path.write_bytes(
        _json_bytes(
            {
                "schema_name": "microcosm.release-contract.fixture-writers",
                "schema_version": 1,
                "files": {name: {"writers": WRITERS[name]} for name in sorted(WRITERS)},
            }
        )
    )


def _generate(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    assembler = _load(ASSEMBLER)
    _c1_uk_national(root, assembler)
    _c1_inventory(root, assembler)
    _c1_us_national(root)
    _c5_releases(root)
    _c5_registry(root)
    _writers(root)
    checksums = [
        f"{hashlib.sha256(content).hexdigest()}  {name}"
        for name, content in sorted(_tree(root).items())
    ]
    (root / "SHA256SUMS").write_text("\n".join(checksums) + "\n")


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify the release-contract golden fixtures."
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory(prefix="microcosm-release-contract-") as temporary:
        generated = Path(temporary) / "v1"
        _generate(generated)
        if args.check:
            fresh, committed = _tree(generated), _tree(args.output)
            differing = sorted(
                name
                for name in set(fresh) | set(committed)
                if fresh.get(name) != committed.get(name)
            )
            if differing:
                raise SystemExit(
                    f"release-contract fixtures are not reproducible: {differing} "
                    "differ. Regenerate with `uv run python "
                    "tools/generate_release_contract_fixtures.py` and update "
                    "docs/release-contract.md in the same change"
                )
            return 0
        if args.output.exists():
            shutil.rmtree(args.output)
        shutil.copytree(generated, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
