"""The non-default local-area release contract (microcosm#398).

A US local-area release must also carry the ACS package stage's stored-input
gate (microcosm#1026, #1034). Its tests are examples plus Hypothesis
properties; the wheels job installs no test extras, so the properties skip
there.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from microcosm.data import contract
from microcosm.data.contract import (
    NON_DEFAULT_LOCAL_AREA_DATASET_ROLE,
    US_LOCAL_AREA_SPM_MEASUREMENT_STAGE,
    US_LOCAL_AREA_STORED_INPUTS_STAGE,
    US_SPM_MEASUREMENT_EVIDENCE_KEY,
    US_STORED_INPUTS_EVIDENCE_KEY,
    ReleaseContractError,
    release_dataset_role,
    validate_release_dir,
)
from microcosm.data.release import prepare_release, publish_release
from microcosm.data.stored_inputs import register_sha256
from test_support.paths import REPOSITORY_ROOT

RELEASE_ID = "populace-us-2024-buildo-acs-local-abc1234-20260723T000000Z"
UK_LOCAL_RELEASE_ID = "microcosm-uk-2024-25-local-k55000"
REPO_ROOT = REPOSITORY_ROOT
ARTIFACT_BYTES = b"not-a-real-h5-but-hash-stable"
BUILT_WITH_US = {"name": "policyengine-us", "version": "2.2.1"}
_DEFAULT = object()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


ARTIFACT_SHA = _sha256_bytes(ARTIFACT_BYTES)


def _stored_inputs_gate() -> dict:
    """The gate the ACS package stage writes for ARTIFACT_BYTES."""
    return {
        "passed": True,
        "failures": [],
        "engine": "policyengine-us 2.2.1",
        "register_sha256": register_sha256(),
        "registered_non_variables": ["household_spine", "person_spine"],
        "artifact_sha256": ARTIFACT_SHA,
        "checked_at_stage": US_LOCAL_AREA_STORED_INPUTS_STAGE,
    }


def _spm_measurement_gate(engine: str = "policyengine-us 2.2.1") -> dict:
    """The SPM measurement gate the ACS package stage writes for ARTIFACT_BYTES
    under an engine that measures every SPM unit, as policyengine-us 2.2.1
    does (microcosm#1263)."""
    return {
        "passed": True,
        "failures": [],
        "engine": engine,
        "details": {
            "role_column_present": True,
            "role_null_values": 0,
            "role_non_boolean_values": 0,
            "engine_scopes_spm_universe": False,
            "universe_declared_in_release": False,
            "measured_scope": "all_units",
            "n_units": 1_589_568,
            "n_units_by_universe": {"INCLUDED": 1_408_308, "OUTSIDE": 181_260},
            "n_measured_units": 1_589_568,
            "n_measured_units_without_classified_adult": 0,
            "n_units_without_classified_adult": 0,
            "n_units_without_classified_adult_by_universe": {
                "INCLUDED": 0,
                "OUTSIDE": 0,
            },
        },
        "artifact_sha256": ARTIFACT_SHA,
        "checked_at_stage": US_LOCAL_AREA_SPM_MEASUREMENT_STAGE,
    }


def _write_local_bundle(
    root: Path,
    *,
    release_id: str = RELEASE_ID,
    gate_passed: bool = True,
    default_datasets: dict | None = None,
    pin_revisions: bool = True,
    drop_target_field: str | None = None,
    dataset_role: str = NON_DEFAULT_LOCAL_AREA_DATASET_ROLE,
    stored_inputs_gate: object = _DEFAULT,
    build_manifest_stored_inputs_gate: object = _DEFAULT,
    built_with_model_package: object = _DEFAULT,
    extra_gates: dict | None = None,
    spm_measurement_gate: object = _DEFAULT,
    build_manifest_spm_measurement_gate: object = _DEFAULT,
) -> Path:
    """A local-area release directory the contract accepts by default.

    A US id gets the package stage's stored-input and SPM measurement gates in
    ``gate_summary.json`` and ``build_manifest.json`` and the built-with
    engine they name; a UK id gets none of them, as the UK lanes write none.
    ``None`` omits a piece, and a value replaces it.
    """
    release_dir = root / "releases" / release_id
    release_dir.mkdir(parents=True)
    us = release_id.startswith("populace-us-")
    if stored_inputs_gate is _DEFAULT:
        stored_inputs_gate = _stored_inputs_gate() if us else None
    if build_manifest_stored_inputs_gate is _DEFAULT:
        build_manifest_stored_inputs_gate = copy.deepcopy(stored_inputs_gate)
    if built_with_model_package is _DEFAULT:
        built_with_model_package = dict(BUILT_WITH_US) if us else None
    if spm_measurement_gate is _DEFAULT:
        # The package stage grades against the engine it records as
        # built-with, so the default gate names that engine.
        version = (
            built_with_model_package.get("version")
            if isinstance(built_with_model_package, dict)
            else None
        )
        spm_measurement_gate = (
            _spm_measurement_gate(
                f"policyengine-us {version}"
                if isinstance(version, str) and version
                else "policyengine-us 2.2.1"
            )
            if us
            else None
        )
    if build_manifest_spm_measurement_gate is _DEFAULT:
        build_manifest_spm_measurement_gate = copy.deepcopy(spm_measurement_gate)

    diagnostics = {
        "households": 1_600_000,
        "n_targets": 2,
        "final_loss": 0.058,
        "fraction_within_10pct": 0.879,
        "targets": [
            {
                "name": "pop_state_06",
                "target": 100.0,
                "compiled_target": 100.0,
                "initial_estimate": 90.0,
                "final_estimate": 99.0,
            },
            {
                "name": "usda_snap_state_06_benefits",
                "target": 50.0,
                "compiled_target": 50.0,
                "initial_estimate": 40.0,
                "final_estimate": 49.0,
            },
        ],
    }
    if drop_target_field:
        del diagnostics["targets"][1][drop_target_field]
    gates = {
        "us_puma_ladder_gate": {"passed": True, "failures": []},
        "calibration": {"passed": gate_passed, "final_loss": 0.058},
        **(extra_gates or {}),
    }
    if stored_inputs_gate is not None:
        gates[US_STORED_INPUTS_EVIDENCE_KEY] = stored_inputs_gate
    if spm_measurement_gate is not None:
        gates[US_SPM_MEASUREMENT_EVIDENCE_KEY] = spm_measurement_gate
    gate_summary = {"gates": gates, "reviewed_limitations": []}
    coverage = {
        "schema_version": 1,
        "acs_sources": {"manifest": "pinned"},
        "geography_ladder": {"sha256": "f" * 64},
        "transfer_coverage": {"targets": 91},
        "donor_release": {"release_id": "populace-us-2024-buildo-sparse-x"},
    }
    build_manifest = {"build_id": release_id, "build_sha": "abc1234"}
    build_manifest_gates = {}
    if build_manifest_stored_inputs_gate is not None:
        build_manifest_gates[US_STORED_INPUTS_EVIDENCE_KEY] = (
            build_manifest_stored_inputs_gate
        )
    if build_manifest_spm_measurement_gate is not None:
        build_manifest_gates[US_SPM_MEASUREMENT_EVIDENCE_KEY] = (
            build_manifest_spm_measurement_gate
        )
    if build_manifest_gates:
        build_manifest["gates"] = build_manifest_gates

    files = {
        "calibration_diagnostics.json": diagnostics,
        "gate_summary.json": gate_summary,
        "us_source_coverage.json": coverage,
        "build_manifest.json": build_manifest,
    }
    payloads = {
        name: json.dumps(content, indent=1).encode() for name, content in files.items()
    }
    for name, payload in payloads.items():
        (release_dir / name).write_bytes(payload)

    artifact_bytes = ARTIFACT_BYTES
    (root / "populace_us_2024_acs_local.h5").write_bytes(artifact_bytes)

    revision = release_id if pin_revisions else "some-other-revision"
    manifest = {
        "schema_version": 1,
        "data_package": {"name": "microcosm-data", "version": "0.1.0"},
        "default_datasets": default_datasets if default_datasets is not None else {},
        "dataset_role": dataset_role,
        "is_default": False,
        "namespace": "buildo_acs_local",
        "build": {
            "build_id": release_id,
            **(
                {"built_with_model_package": built_with_model_package}
                if built_with_model_package is not None
                else {}
            ),
        },
        "artifacts": {
            "populace_us_2024_acs_local": {
                "kind": "microdata",
                "path": "populace_us_2024_acs_local.h5",
                "repo_id": "policyengine/populace-us",
                "revision": revision,
                "sha256": _sha256_bytes(artifact_bytes),
            },
            "gate_summary": {
                "kind": "diagnostics",
                "path": "gate_summary.json",
                "repo_id": "policyengine/populace-us",
                "revision": revision,
                "sha256": _sha256_bytes(payloads["gate_summary.json"]),
            },
        },
        "reviewed_limitations": [{"id": "cd_population_marginal_vintage_2020"}],
    }
    manifest_text = json.dumps(manifest, indent=1)
    (release_dir / "release_manifest.json").write_text(manifest_text)
    ledger = {name: _sha256_bytes(payload) for name, payload in payloads.items()}
    ledger["release_manifest.json"] = _sha256_bytes(manifest_text.encode())
    ledger["populace_us_2024_acs_local.h5"] = _sha256_bytes(artifact_bytes)
    (release_dir / "sha256sums.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(ledger.items()))
    )
    return release_dir


def test_valid_local_area_bundle_passes(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(tmp_path)

    assert release_dataset_role(release_dir) == NON_DEFAULT_LOCAL_AREA_DATASET_ROLE
    validate_release_dir(release_dir)


def test_explicit_diagnostics_failure_does_not_block_local_release(
    tmp_path: Path,
) -> None:
    release_dir = _write_local_bundle(tmp_path)
    diagnostics_path = release_dir / "calibration_diagnostics.json"
    diagnostics_path.unlink()
    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["calibration_diagnostics"] = {
        "status": "failed",
        "expected_schema_version": 8,
        "error_code": "validation_error",
        "message": "Target hierarchy is incomplete.",
    }
    manifest_text = json.dumps(manifest, indent=1)
    manifest_path.write_text(manifest_text)
    ledger_path = release_dir / "sha256sums.txt"
    ledger = {
        name: digest
        for digest, name in (
            line.split(maxsplit=1) for line in ledger_path.read_text().splitlines()
        )
        if name != "calibration_diagnostics.json"
    }
    ledger["release_manifest.json"] = _sha256_bytes(manifest_text.encode())
    ledger_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(ledger.items()))
    )

    validate_release_dir(release_dir)
    with pytest.warns(UserWarning, match="generation failed"):
        prepared = prepare_release(
            release_dir,
            artifact_root=tmp_path,
            update_latest=False,
        )
    assert prepared.calibration_diagnostics_available is False
    assert "calibration_diagnostics.json" not in prepared.filenames


def test_local_area_dispatch_does_not_bypass_uk_tier_identity(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(
        tmp_path,
        release_id="populace-uk-2023-public-k10",
    )

    with pytest.raises(ReleaseContractError, match="unratified tier"):
        validate_release_dir(release_dir)


def test_local_area_bundle_skips_national_critical_targets(tmp_path: Path) -> None:
    # The two-target local surface would fail every national critical-target
    # requirement; the role-keyed contract must not apply them.
    release_dir = _write_local_bundle(tmp_path)

    validate_release_dir(release_dir)


def test_failing_gate_blocks_local_area_publish(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(tmp_path, gate_passed=False)

    with pytest.raises(ReleaseContractError, match="did not pass"):
        validate_release_dir(release_dir)


def test_default_dataset_claim_blocks_local_area_publish(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(
        tmp_path,
        default_datasets={"us_2024": "populace_us_2024_acs_local"},
    )

    with pytest.raises(ReleaseContractError, match="empty\\s+object"):
        validate_release_dir(release_dir)


def test_unpinned_artifact_revision_blocks_local_area_publish(
    tmp_path: Path,
) -> None:
    release_dir = _write_local_bundle(tmp_path, pin_revisions=False)

    with pytest.raises(ReleaseContractError, match="not pinned to the release id"):
        validate_release_dir(release_dir)


def test_incomplete_target_row_blocks_local_area_publish(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(tmp_path, drop_target_field="final_estimate")

    with pytest.raises(ReleaseContractError, match="missing 'final_estimate'"):
        validate_release_dir(release_dir)


def test_unknown_dataset_role_is_rejected(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(tmp_path, dataset_role="research_preview")

    with pytest.raises(ReleaseContractError, match="unknown dataset_role"):
        validate_release_dir(release_dir)


def test_tampered_diagnostics_artifact_hash_is_rejected(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(tmp_path)
    (release_dir / "gate_summary.json").write_text(
        json.dumps(
            {
                "gates": {"calibration": {"passed": True}},
                "reviewed_limitations": [],
            }
        )
    )

    with pytest.raises(ReleaseContractError, match="observed"):
        validate_release_dir(release_dir)


def test_publish_refuses_pointer_update_for_local_area_role(
    tmp_path: Path, monkeypatch
) -> None:
    from microcosm.data import release as release_module
    from microcosm.data.publish_cli import main

    release_dir = _write_local_bundle(tmp_path)
    validate_release_dir(release_dir)
    monkeypatch.setattr(
        release_module, "_hf_api", lambda: pytest.fail("unexpected Hub construction")
    )

    with pytest.raises(ValueError, match="never update latest.json") as publication:
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=object(),
            update_latest=True,
        )

    with pytest.raises(type(publication.value)) as preflight:
        main([str(release_dir), "--preflight-only"])
    assert str(preflight.value) == str(publication.value)


def test_publish_without_pointer_passes_the_role_guard(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(tmp_path)

    # A dummy backend fails the atomic-publication capability check — which
    # sits AFTER the role guard and the artifact verification, proving
    # --no-latest publishes proceed to the Hub surface.
    with pytest.raises(TypeError, match="Hub backend|create_tag"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=object(),
            artifact_root=tmp_path,
            update_latest=False,
        )


def test_empty_checksum_ledger_is_rejected(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(tmp_path)
    (release_dir / "sha256sums.txt").write_text("")

    with pytest.raises(ReleaseContractError, match="does not cover required file"):
        validate_release_dir(release_dir)


def test_tampered_checksum_ledger_is_rejected(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(tmp_path)
    ledger = (release_dir / "sha256sums.txt").read_text().splitlines()
    flipped = []
    for line in ledger:
        digest, name = line.split(maxsplit=1)
        if name == "gate_summary.json":
            digest = ("0" if digest[0] != "0" else "1") + digest[1:]
        flipped.append(f"{digest}  {name}")
    (release_dir / "sha256sums.txt").write_text("\n".join(flipped) + "\n")

    with pytest.raises(ReleaseContractError, match="disagrees|does not match"):
        validate_release_dir(release_dir)


def test_unsafe_ledger_path_is_rejected(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(tmp_path)
    with (release_dir / "sha256sums.txt").open("a") as handle:
        handle.write("a" * 64 + "  ../escape.json\n")

    with pytest.raises(ReleaseContractError, match="unsafe path"):
        validate_release_dir(release_dir)


def test_explicit_null_dataset_role_is_rejected(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(tmp_path)
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    manifest["dataset_role"] = None
    (release_dir / "release_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ReleaseContractError, match="unknown dataset_role"):
        validate_release_dir(release_dir)


# ---------------------------------------------------------------------------
# The stored-input gate (microcosm#1026, #1034)
# ---------------------------------------------------------------------------


def _failures(release_dir: Path) -> list[str]:
    """The contract's failure lines for ``release_dir``; empty if it validates."""
    try:
        validate_release_dir(release_dir)
    except ReleaseContractError as error:
        return list(error.failures)
    return []


def _stored_input_failures(failures: list[str]) -> list[str]:
    return [
        failure
        for failure in failures
        if US_STORED_INPUTS_EVIDENCE_KEY in failure
        or "built_with_model_package" in failure
    ]


def test_us_local_area_release_requires_the_stored_input_gate(
    tmp_path: Path,
) -> None:
    """A US bundle whose gate summary does not list the gate is refused, even
    though every gate it does list passed (the hole microcosm#1034 closes)."""
    release_dir = _write_local_bundle(
        tmp_path, stored_inputs_gate=None, build_manifest_stored_inputs_gate=None
    )

    failures = _failures(release_dir)

    assert failures == [
        "gate_summary.json must carry the 'stored_inputs' gate for a US "
        "non-default local-area release (microcosm#1026, #1034): package it "
        "with tools/build_us_acs_local_release.py --stage package, which "
        "checks the calibrated H5's stored columns against the policyengine-us "
        "the release records as built-with."
    ]


@pytest.mark.parametrize(
    ("field", "value", "names"),
    [
        ("failures", ["stored column 'would_claim_wic' ..."], "empty 'failures'"),
        ("checked_at_stage", "finalize", "checked_at_stage"),
        ("checked_at_stage", _DEFAULT, "checked_at_stage"),
        ("engine", "policyengine-us 2.2.0", "built-with engine"),
        ("engine", _DEFAULT, "built-with engine"),
        ("register_sha256", "abc", "'register_sha256'"),
        ("register_sha256", "A" * 64, "'register_sha256'"),
        (
            "registered_non_variables",
            ["person_spine", "household_spine"],
            "'registered_non_variables'",
        ),
        (
            "registered_non_variables",
            ["household_spine", "household_spine"],
            "'registered_non_variables'",
        ),
        ("registered_non_variables", ["A_AGE"], "'registered_non_variables'"),
        ("registered_non_variables", "person_spine", "'registered_non_variables'"),
        ("artifact_sha256", "0" * 64, "artifact_sha256"),
        ("artifact_sha256", _DEFAULT, "artifact_sha256"),
        ("passed", False, "did not pass"),
    ],
)
def test_each_stored_input_gate_field_is_bound(
    tmp_path: Path, field: str, value: object, names: str
) -> None:
    """Each field of the gate is checked: a bad or missing value is refused by
    name, with the build manifest's copy kept equal so only the field fires."""
    gate = _stored_inputs_gate()
    if value is _DEFAULT:
        del gate[field]
    else:
        gate[field] = value
    release_dir = _write_local_bundle(tmp_path, stored_inputs_gate=gate)

    failures = _failures(release_dir)

    assert len(failures) == 1, failures
    assert names in failures[0]
    assert "stored_inputs" in failures[0]


@pytest.mark.parametrize(
    "build_copy",
    [None, {**_stored_inputs_gate(), "registered_non_variables": []}],
    ids=["missing", "different"],
)
def test_the_build_manifest_copy_of_the_gate_must_match(
    tmp_path: Path, build_copy: dict | None
) -> None:
    release_dir = _write_local_bundle(
        tmp_path, build_manifest_stored_inputs_gate=build_copy
    )

    assert _failures(release_dir) == [
        "build_manifest.json gates.stored_inputs must equal the "
        "gate_summary.json 'stored_inputs' gate; the package stage writes both "
        "from one entry."
    ]


@pytest.mark.parametrize(
    "built_with",
    [
        None,
        {"name": "policyengine-uk", "version": "2.2.1"},
        {"name": "policyengine-us", "version": "unknown"},
        {"name": "policyengine-us", "version": ""},
    ],
    ids=["missing", "other-package", "unknown", "empty"],
)
def test_the_gate_is_graded_against_a_recorded_built_with_engine(
    tmp_path: Path, built_with: dict | None
) -> None:
    """Without a resolved policyengine-us built-with version there is nothing
    to hold the gate's engine to, and that is itself a refusal."""
    release_dir = _write_local_bundle(tmp_path, built_with_model_package=built_with)

    failures = _failures(release_dir)

    assert len(failures) == 1, failures
    assert "build.built_with_model_package" in failures[0]


def test_uk_local_k_bundle_needs_no_stored_input_gate(tmp_path: Path) -> None:
    """The requirement is scoped to US ids: UK local-k (and UK dense, which
    shares _check_local_area_gates) releases have no stored-input seam."""
    release_dir = _write_local_bundle(tmp_path, release_id=UK_LOCAL_RELEASE_ID)
    gate_summary = json.loads((release_dir / "gate_summary.json").read_text())

    assert US_STORED_INPUTS_EVIDENCE_KEY not in gate_summary["gates"]
    validate_release_dir(release_dir)


def test_a_release_packaged_before_the_gate_is_refused() -> None:
    """Negative control on a real pre-#1031 package-stage output: the Build O
    hours rebuild's release files list eight green gates and no stored-input
    gate, which the old contract accepted and this one refuses."""
    source = REPO_ROOT / "experiments" / "us-acs-local-hours-rebuild-20260922"
    gate_summary = json.loads((source / "gate_summary.json").read_text())
    release_manifest = json.loads((source / "release_manifest.json").read_text())
    build_manifest = json.loads((source / "build_manifest.json").read_text())
    assert US_STORED_INPUTS_EVIDENCE_KEY not in gate_summary["gates"]
    old_failures: list[str] = []
    contract._check_local_area_gates(gate_summary, old_failures)
    assert old_failures == []

    failures: list[str] = []
    contract._check_us_local_area_stored_inputs_gate(
        gate_summary,
        release_manifest=release_manifest,
        build_manifest=build_manifest,
        failures=failures,
    )

    assert len(failures) == 1
    assert "must carry the 'stored_inputs' gate" in failures[0]


# ---------------------------------------------------------------------------
# The SPM measurement gate (microcosm#1263)
# ---------------------------------------------------------------------------


def test_us_local_area_release_requires_the_spm_measurement_gate(
    tmp_path: Path,
) -> None:
    """A US bundle without the gate is refused through the full validator,
    though every gate it does list passed."""
    release_dir = _write_local_bundle(
        tmp_path,
        spm_measurement_gate=None,
        build_manifest_spm_measurement_gate=None,
    )

    failures = _failures(release_dir)

    assert len(failures) == 1, failures
    assert "must carry the 'spm_measurement_composition' gate" in failures[0]


def test_a_release_with_a_measured_unit_without_an_adult_is_refused(
    tmp_path: Path,
) -> None:
    """The 2026-09-23 failure, as a record: under policyengine-us 2.2.1 every
    unit is measured, so 1,946 unclassified units outside the universe are
    measured units, and the release would crash every SPM measurement."""
    gate = _spm_measurement_gate()
    gate["details"].update(
        n_measured_units_without_classified_adult=1_946,
        n_units_without_classified_adult=1_946,
        n_units_without_classified_adult_by_universe={"INCLUDED": 0, "OUTSIDE": 1_946},
    )
    release_dir = _write_local_bundle(tmp_path, spm_measurement_gate=gate)

    failures = _failures(release_dir)

    assert any("SPM_COMPOSITION_REQUIRED" in failure for failure in failures)


def test_the_build_manifest_copy_of_the_spm_gate_must_match(tmp_path: Path) -> None:
    other = _spm_measurement_gate()
    other["details"]["n_units"] = 1
    release_dir = _write_local_bundle(
        tmp_path, build_manifest_spm_measurement_gate=other
    )

    failures = _failures(release_dir)

    assert any(
        "build_manifest.json gates.spm_measurement_composition" in failure
        for failure in failures
    ), failures


def test_uk_local_k_bundle_needs_no_spm_measurement_gate(tmp_path: Path) -> None:
    release_dir = _write_local_bundle(tmp_path, release_id=UK_LOCAL_RELEASE_ID)
    gate_summary = json.loads((release_dir / "gate_summary.json").read_text())

    assert US_SPM_MEASUREMENT_EVIDENCE_KEY not in gate_summary["gates"]
    validate_release_dir(release_dir)


def test_a_release_packaged_before_the_spm_gate_is_refused() -> None:
    """Negative control on a real pre-gate package-stage output: the Build O
    hours rebuild's release files carry no SPM measurement gate."""
    source = REPO_ROOT / "experiments" / "us-acs-local-hours-rebuild-20260922"
    gate_summary = json.loads((source / "gate_summary.json").read_text())
    release_manifest = json.loads((source / "release_manifest.json").read_text())
    build_manifest = json.loads((source / "build_manifest.json").read_text())
    assert US_SPM_MEASUREMENT_EVIDENCE_KEY not in gate_summary["gates"]

    failures: list[str] = []
    contract._check_us_local_area_spm_measurement_gate(
        gate_summary,
        release_manifest=release_manifest,
        build_manifest=build_manifest,
        failures=failures,
    )

    assert len(failures) == 1
    assert "must carry the 'spm_measurement_composition' gate" in failures[0]


def _hypothesis():
    pytest.importorskip("hypothesis")
    import hypothesis
    from hypothesis import strategies as st

    return hypothesis, st


_ORACLE_MODEL_NAMED = re.compile(r"[a-z][a-z0-9_]*")
_ORACLE_SHA256 = re.compile(r"[0-9a-f]{64}")
_OMIT = "<omitted>"


def _oracle_gate_is_valid(
    gate: dict | None, build_copy: object, built_with: dict | None
) -> bool:
    """An independent reading of the requirement, written from the spec."""
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        return False
    version = built_with.get("version") if isinstance(built_with, dict) else None
    registered = gate.get("registered_non_variables")
    return (
        isinstance(built_with, dict)
        and built_with.get("name") == "policyengine-us"
        and isinstance(version, str)
        and version not in {"", "unknown", "not-installed"}
        and gate.get("engine") == f"policyengine-us {version}"
        and gate.get("failures") == []
        and gate.get("checked_at_stage") == "package"
        and isinstance(gate.get("register_sha256"), str)
        and _ORACLE_SHA256.fullmatch(gate["register_sha256"]) is not None
        and isinstance(registered, list)
        and all(
            isinstance(column, str) and _ORACLE_MODEL_NAMED.fullmatch(column)
            for column in registered
        )
        and registered == sorted(set(registered))
        and gate.get("artifact_sha256") == ARTIFACT_SHA
        and build_copy == gate
    )


def _gate_strategy(st):
    """Gates near the valid one: each field valid, wrong, or missing."""
    valid = _stored_inputs_gate()
    columns = st.sampled_from(
        ["household_spine", "person_spine", "source_year", "A_AGE", "x", "Bad-x"]
    )
    fields = {
        "passed": st.sampled_from([True, True, True, False, "true", 1, _OMIT]),
        "failures": st.sampled_from([[], [], [], ["refused"], None, _OMIT]),
        "engine": st.sampled_from(
            [valid["engine"], valid["engine"], "policyengine-us 2.3.0", 7, _OMIT]
        ),
        "register_sha256": st.sampled_from(
            [valid["register_sha256"], "f" * 64, "F" * 64, "abc", None, _OMIT]
        ),
        "registered_non_variables": st.one_of(
            st.just(valid["registered_non_variables"]),
            st.lists(columns, max_size=4),
            st.lists(columns, max_size=4).map(lambda names: sorted(set(names))),
            st.just(_OMIT),
        ),
        "artifact_sha256": st.sampled_from(
            [ARTIFACT_SHA, ARTIFACT_SHA, "0" * 64, None, _OMIT]
        ),
        "checked_at_stage": st.sampled_from(
            ["package", "package", "finalize", "", _OMIT]
        ),
    }
    gate = st.fixed_dictionaries(fields).map(
        lambda drawn: {key: value for key, value in drawn.items() if value != _OMIT}
    )
    return st.one_of(gate, st.just(None))


def _build_copy_strategy(st, gate):
    return st.sampled_from(["same", "same", "same", "missing", "other"]).map(
        lambda mode: (
            copy.deepcopy(gate)
            if mode == "same"
            else None
            if mode == "missing"
            else {**(gate or {}), "engine": "policyengine-us 0.0.1"}
        )
    )


def _built_with_strategy(st):
    return st.sampled_from(
        [
            BUILT_WITH_US,
            BUILT_WITH_US,
            BUILT_WITH_US,
            {"name": "policyengine-us", "version": "2.3.0"},
            {"name": "policyengine-uk", "version": "2.2.1"},
            {"name": "policyengine-us", "version": "unknown"},
            None,
        ]
    )


def test_property_a_us_bundle_validates_iff_its_stored_input_gate_is_bound(
    tmp_path_factory,
) -> None:
    """For all gates, build-manifest copies and built-with records near the
    valid ones, a US local-area bundle validates if and only if its gate is
    present, green, failure-free, from the package stage, names the built-with
    engine, carries a sha256 register digest and a sorted list of distinct
    model-named registered columns, certifies the shipped microdata sha256,
    and equals its build-manifest copy (an independent oracle); and every
    refusal of an otherwise-valid bundle names the stored-input gate or the
    built-with record."""
    hypothesis, st = _hypothesis()

    @hypothesis.settings(max_examples=150, deadline=None)
    @hypothesis.given(data=st.data())
    def check(data):
        gate = data.draw(_gate_strategy(st), label="gate")
        build_copy = data.draw(_build_copy_strategy(st, gate), label="build_copy")
        built_with = data.draw(_built_with_strategy(st), label="built_with")
        release_dir = _write_local_bundle(
            tmp_path_factory.mktemp("bundle"),
            stored_inputs_gate=gate,
            build_manifest_stored_inputs_gate=build_copy,
            built_with_model_package=built_with,
        )

        failures = _failures(release_dir)

        assert (failures == []) == _oracle_gate_is_valid(gate, build_copy, built_with)
        if isinstance(gate, dict) and gate.get("passed") is True:
            # The generic green-gate rule is satisfied, so any refusal is the
            # stored-input requirement's own.
            assert _stored_input_failures(failures) == failures

    check()


def test_property_the_requirement_only_adds_failures(tmp_path_factory) -> None:
    """For all generated US bundles, the failures with the stored-input gate
    requirement are a superset of the failures without it: the change
    tightens the local-area contract and relaxes nothing."""
    hypothesis, st = _hypothesis()

    @hypothesis.settings(max_examples=80, deadline=None)
    @hypothesis.given(data=st.data())
    def check(data):
        gate = data.draw(_gate_strategy(st), label="gate")
        build_copy = data.draw(_build_copy_strategy(st, gate), label="build_copy")
        release_dir = _write_local_bundle(
            tmp_path_factory.mktemp("bundle"),
            stored_inputs_gate=gate,
            build_manifest_stored_inputs_gate=build_copy,
            gate_passed=data.draw(st.booleans(), label="calibration_passed"),
        )
        with_requirement = set(_failures(release_dir))
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                contract,
                "_check_us_local_area_stored_inputs_gate",
                lambda *args, **kwargs: None,
            )
            without_requirement = set(_failures(release_dir))

        assert without_requirement <= with_requirement

    check()


def test_property_uk_local_k_verdicts_ignore_the_stored_input_gate(
    tmp_path_factory,
) -> None:
    """For all gates a UK local-k bundle might list under the stored-input key
    (any green one, or none), the bundle validates: the requirement is scoped
    to US ids and the UK lanes have no stored-input seam."""
    hypothesis, st = _hypothesis()

    @hypothesis.settings(max_examples=60, deadline=None)
    @hypothesis.given(gate=_gate_strategy(st))
    def check(gate):
        if isinstance(gate, dict):
            gate = {**gate, "passed": True}
        release_dir = _write_local_bundle(
            tmp_path_factory.mktemp("uk"),
            release_id=UK_LOCAL_RELEASE_ID,
            stored_inputs_gate=gate,
        )

        assert _failures(release_dir) == []

    check()


def test_property_extra_green_gates_do_not_change_the_verdict(
    tmp_path_factory,
) -> None:
    """For all gates and all sets of extra green gates, a US bundle's failures
    are the same with and without the extra gates: only the stored-input gate
    decides the stored-input requirement."""
    hypothesis, st = _hypothesis()
    gate_names = st.from_regex(r"[a-z][a-z_]{0,12}", fullmatch=True).filter(
        lambda name: name not in {"stored_inputs", "calibration", "us_puma_ladder_gate"}
    )

    @hypothesis.settings(max_examples=60, deadline=None)
    @hypothesis.given(
        gate=_gate_strategy(st),
        extra_names=st.lists(gate_names, max_size=4, unique=True),
    )
    def check(gate, extra_names):
        extras = {name: {"passed": True, "failures": []} for name in extra_names}
        plain = _write_local_bundle(
            tmp_path_factory.mktemp("plain"), stored_inputs_gate=gate
        )
        extended = _write_local_bundle(
            tmp_path_factory.mktemp("extended"),
            stored_inputs_gate=gate,
            extra_gates=extras,
        )

        assert set(_failures(extended)) == set(_failures(plain))

    check()
