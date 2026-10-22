"""The release-contract fixture generator and the manifests it freezes.

``tools/generate_release_contract_fixtures.py`` writes the C1 and C5 golden
fixtures (docs/release-contract.md) by calling the UK national assembler's own
manifest builder. The data shard's ``test_release_contract_fixtures.py`` checks
the committed bytes; this module checks that the generator reproduces them and
that the builder's output satisfies the release contract for every input, not
only the fixture's.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.staging_v2 import validate_staging_delivery
from microcosm.data import contract, loader
from microcosm.data.registry import REGISTRY
from test_support.paths import paths_for

ROOT = paths_for("microcosm-build").repository


def _load_assembler():
    path = ROOT / "tools" / "assemble_uk_release_dir.py"
    spec = importlib.util.spec_from_file_location("assemble_uk_release_dir", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ASSEMBLER = _load_assembler()
UK_NATIONAL_SPEC = REGISTRY[("uk", 2025, "national")]


def test_release_contract_fixtures_are_reproducible():
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "generate_release_contract_fixtures.py"),
            "--check",
        ],
        cwd=ROOT,
        check=True,
    )


_HEX = "0123456789abcdef"
_digest = st.text(_HEX, min_size=64, max_size=64)
_version = st.from_regex(r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}", fullmatch=True)
_attempt_suffix = st.builds(
    "{}{:02d}{:02d}T{:02d}{:02d}{:02d}Z-{}".format,
    st.integers(2026, 2099).map(str),
    st.integers(1, 12),
    st.integers(1, 28),
    st.integers(0, 23),
    st.integers(0, 59),
    st.integers(0, 59),
    st.text(_HEX, min_size=8, max_size=8),
)
#: A shippable part's statuses: the certifier's census of its gates by status
#: (``_status_census``), which the contract requires to be all "passed".
_statuses = st.integers(1, 500).map(lambda count: {"passed": count})
_staging = st.sampled_from(
    [
        {
            "contract_version": 2,
            "enabled": True,
            "mode": "local_and_remote",
            "run_id": "uk-national-property",
            "configured_repository": "policyengine/populace-uk-staging",
            "upload_attempts": 3,
            "upload_successes": 3,
            "read_back": "passed",
            "last_error_code": None,
            "opt_out_reason": None,
        },
        {
            "contract_version": 2,
            "enabled": False,
            "mode": "disabled",
            "run_id": None,
            "configured_repository": None,
            "upload_attempts": 0,
            "upload_successes": 0,
            "read_back": "not_requested",
            "last_error_code": None,
            "opt_out_reason": "--no-staging",
        },
    ]
)


@st.composite
def _manifest_inputs(draw):
    evidence_names = [name for _, _, name in ASSEMBLER._EVIDENCE]
    attempt_id = ASSEMBLER._ATTEMPT_PREFIX + draw(_attempt_suffix)
    return {
        "measured": {name: draw(_digest) for name in ("candidate", *evidence_names)},
        "calibration_filename": f"{ASSEMBLER._DATASET_KEY}_calibration.npz",
        "calibration_sha": draw(_digest),
        "household_weight_year": {
            "dataset_year": draw(st.integers(2020, 2040)),
            "basis": draw(st.sampled_from(["dataset_year", "as_solved"])),
        },
        "cut_tag": ASSEMBLER._cut_tag(attempt_id, None),
        "attempt_id": attempt_id,
        "runtime": {package: draw(_version) for package in ASSEMBLER._RUNTIME_PACKAGES},
        "code_pin": draw(st.text(_HEX, min_size=40, max_size=40)),
        "target_surface": {
            "sha256": draw(_digest),
            "n_targets": draw(st.integers(1, 100_000)),
        },
        "target_registry": {
            "version": draw(st.text(_HEX, min_size=1, max_size=16)),
            "n_specs": draw(st.integers(1, 100_000)),
        },
        "spine_part": {"statuses": draw(_statuses)},
        "seam_part": {"statuses": draw(_statuses)},
        "release_cut_part": {"statuses": draw(_statuses)},
        "staging_delivery": validate_staging_delivery(draw(_staging)),
        "created_at": "2026-01-01T00:00:00+00:00",
        "publisher_labels": draw(
            st.dictionaries(st.text(min_size=1), st.text(min_size=1), max_size=3)
        ),
    }


@settings(max_examples=60, deadline=None)
@given(_manifest_inputs())
def test_national_manifests_satisfy_the_release_contract(inputs):
    """For any valid inputs, the assembler's manifests pass the contract's
    manifest checks, pin every artifact to the one cut tag, ship exactly the
    signed evidence, and resolve through the registry's national spec."""

    build_manifest, release_manifest = ASSEMBLER.uk_national_release_manifests(**inputs)
    release_id = contract._UK_NATIONAL_RELEASE_ID

    failures: list[str] = []
    contract._check_build_manifest(build_manifest, release_id, failures)
    contract._check_release_manifest(release_manifest, release_id, failures)
    contract._check_cross_manifest_consistency(
        build_manifest, release_manifest, None, failures
    )
    assert failures == []

    artifacts = release_manifest["artifacts"]
    assert {artifact["revision"] for artifact in artifacts.values()} == {
        inputs["cut_tag"]
    }
    assert {
        artifact["path"]
        for artifact in artifacts.values()
        if artifact["kind"] == "diagnostics"
    } == {filename for _, filename, _ in ASSEMBLER._EVIDENCE}
    assert set(contract.required_release_files(release_id)) <= {
        "build_manifest.json",
        "release_manifest.json",
        *(filename for _, filename, _ in ASSEMBLER._EVIDENCE),
    }

    certified = loader._certified_release_from_manifest(
        UK_NATIONAL_SPEC,
        release_id,
        release_manifest,
        expected_revision=inputs["cut_tag"],
    )
    assert certified.artifact_sha256 == inputs["measured"]["candidate"]
    assert certified.model.built_version == inputs["runtime"]["policyengine-uk"]
    assert certified.core.built_version == inputs["runtime"]["policyengine-core"]

    # The manifest records each part's census exactly as the certification signed it.
    assert build_manifest["gates"] == {
        "spine": inputs["spine_part"]["statuses"],
        "calibration_seam": inputs["seam_part"]["statuses"],
        "release_cut": inputs["release_cut_part"]["statuses"],
    }

    # Deterministic, key order included: the builder is a function of its inputs.
    again = ASSEMBLER.uk_national_release_manifests(**inputs)
    assert [json.dumps(document, indent=1) for document in again] == [
        json.dumps(document, indent=1)
        for document in (build_manifest, release_manifest)
    ]
