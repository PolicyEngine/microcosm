"""The dense release assembler stages a contract-valid bundle from a run."""

from __future__ import annotations

import base64
import hashlib
import hmac
import importlib.util
import json
from pathlib import Path

import pytest

from microcosm.build.gate_battery import _canonical_json_bytes, _canonical_sha256
from microcosm.build.uk_runtime.calibration_run import UK_LOCAL_GATE_SCOPE
from microcosm.build.uk_runtime.graph_terminal import (
    _gate_rows,
    _reviewed_exclusions_stood_on,
)
from microcosm.build.uk_runtime.local_doctrine import UK_LOCAL_DATASET_HOUSEHOLDS
from microcosm.build.uk_runtime.release_identity import UK_DENSE_RELEASE_ID
from microcosm.data import contract as dc
from test_support.microcosm_build.uk_full_build_cli import (
    patch_support_register,
    synthetic_geography_binding,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

REPO_ROOT = _TEST_PATHS.repository
KEY = base64.b64encode(b"\x09" * 32).decode()
KEY_BYTES = base64.b64decode(KEY)
ATTEMPT = "uk-local-candidate-f100-s42-20260903T105422Z-ca611e43"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "tools" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


#: The area the synthetic run's support gate excused, which the candidate
#: manifest's solve.area_support_exclusions records too: the release manifest
#: must list exactly the areas the signed report stood on (microcosm#1301).
_SIGNED_REPORT_DETAILS = {
    "uk_local_area_support": {
        "reviewed_exclusions": {
            "local_authority/E06000053": {
                "record": {"reason": "synthetic reviewed exclusion"}
            }
        }
    }
}


def _signed_report():
    gates = {}
    for entry_id in sorted(UK_LOCAL_GATE_SCOPE):
        blocking = entry_id in dc._UK_DENSE_RELEASE_BLOCKING_IDS
        gates[entry_id] = {
            "gate": entry_id.removeprefix("uk_local_"),
            "phase": "terminal",
            "criticality": "release_blocking" if blocking else "diagnostic",
            "status": "passed" if blocking else "failed",
            "failures": [] if blocking else ["a diagnostic miss"],
            "details": dict(_SIGNED_REPORT_DETAILS.get(entry_id, {})),
            "reason": None,
        }
    attestation = {
        "schema_version": dc._UK_GATE_BATTERY_ATTESTATION_SCHEMA_VERSION,
        "producer": dc._UK_GATE_BATTERY_PRODUCER,
        "country": "uk",
        "release_id": ATTEMPT,
        "release_candidate": True,
        "spec_fingerprint": dc._UK_DENSE_GATE_DIGESTS["spec_fingerprint"],
        "gates_manifest_sha256": dc._UK_DENSE_GATE_DIGESTS["gates_manifest_sha256"],
        "policy_sha256": dc._UK_DENSE_GATE_DIGESTS["policy_sha256"],
        "phases": ["terminal"],
        "phases_evaluated": ["terminal"],
        "blocked_at_phase": None,
        "release_evidence": {},
        "evidence_sha256": {},
        "gate_outcomes_sha256": _canonical_sha256(gates),
        "signature_algorithm": "hmac-sha256",
        "signing_key_sha256": hashlib.sha256(KEY_BYTES).hexdigest(),
        "signature": None,
    }
    report = {
        "schema_version": dc._UK_GATE_BATTERY_SCHEMA_VERSION,
        "country": "uk",
        "release_id": ATTEMPT,
        "release_candidate": True,
        "spec_fingerprint": attestation["spec_fingerprint"],
        "gates_manifest_sha256": attestation["gates_manifest_sha256"],
        "phases": ["terminal"],
        "phases_evaluated": ["terminal"],
        "blocked_at_phase": None,
        "shippable": True,
        "gates": gates,
        "policy_sha256": attestation["policy_sha256"],
        "release_evidence": {},
        "evidence_sha256": {},
        "attestation": attestation,
        "posture": "local_candidate",
        "scope_exclusions": {},
        "aggregate_admin_measurement": None,
    }
    attestation["signature"] = hmac.new(
        KEY_BYTES, _canonical_json_bytes(report), hashlib.sha256
    ).hexdigest()
    return report


@pytest.fixture(autouse=True)
def _registered_supports(monkeypatch) -> None:
    """The assembler runs the release pre-flight, which checks the candidate's
    geography support pins against the register; the synthetic candidates
    bind the synthetic pins, so register those."""
    patch_support_register(monkeypatch)


def _staging_delivery(**overrides) -> dict:
    """A validated version 2 telemetry receipt the run carried (local-only)."""

    return {
        "contract_version": 2,
        "enabled": True,
        "mode": "local_only",
        "run_id": ATTEMPT,
        "configured_repository": None,
        "upload_attempts": 0,
        "upload_successes": 0,
        "read_back": "not_requested",
        "last_error_code": None,
        "opt_out_reason": None,
        **overrides,
    }


#: A reviewed target-fit exclusion the synthetic full gate report stood on.
SIGNED_TARGET = "synthetic.target_fit_row@2025"


def _full_gate_report() -> dict:
    """A minimal full gate report: the outcomes the assembler names exclusions from.

    The target-fit row carries its measured error, which must never reach the
    release manifest; the input-mass gate lists one column it stood on and
    one dormant entry.
    """

    return {
        "schema_version": 1,
        "kind": "uk_full_gate_report",
        "report": {
            "schema_version": 1,
            "phase": "terminal",
            "outcomes": [
                {
                    "id": "uk_target_fit",
                    "status": "passed",
                    "details": {
                        "reviewed_exclusions": {
                            SIGNED_TARGET: {
                                "relative_error": 0.4321,
                                "reason": "synthetic reviewed gap",
                                "approved_by": "synthetic_reviewer",
                                "adjudication": "synthetic decision",
                                "expires_on": "2099-01-01",
                            }
                        },
                        "dormant_exclusions": [],
                    },
                },
                {
                    "id": "uk_input_mass_parity",
                    "status": "passed",
                    "details": {
                        "reviewed_exclusions": {
                            "synthetic_column": "synthetic input reason",
                            "synthetic_dormant": "never measured",
                        },
                        "dormant_exclusions": ["synthetic_dormant"],
                    },
                },
                {"id": "uk_weight_ratio", "status": "passed", "details": {}},
            ],
        },
    }


SKIPPED_HOLDOUT = {
    "report_only": True,
    "skipped": True,
    "reason": "Not requested: opt-in diagnostic; no out-of-sample claim.",
}


def _candidate_dir(
    root: Path, *, with_incumbent: bool = False, holdout: str = "skipped"
) -> tuple[Path, Path, Path | None]:
    """A synthetic finished release-candidate run.

    By default it requested neither the rotated holdout nor the incumbent
    comparison (microcosm#1301); ``holdout="measured"`` records a measured
    holdout and ``with_incumbent`` adds the score, the surface evaluation and
    the incumbent extraction manifest (returned third, else None).
    """
    candidate = root / "f100-k15-RC"
    (candidate / "logbook-spool").mkdir(parents=True)
    spine = root / "spine.h5"
    spine.write_bytes(b"spine-stand-in")
    h5 = candidate / "microcosm_uk_2024_25_local.h5"
    h5.write_bytes(b"dense-candidate-stand-in")
    diagnostics = {
        "schema_version": 6,
        "n_records": 30,
        "final_loss": 0.0156,
        "fraction_within_10pct": 0.98,
        "targets": [
            {
                "name": "ons.age.20_30@E14000001@2025",
                "target": 10.0,
                "compiled_target": 10.0,
                "initial_estimate": 9.0,
                "final_estimate": 10.0,
            },
            {
                "name": "dwp.uc.households@2025",
                "target": 5.0,
                "compiled_target": 5.0,
                "initial_estimate": 4.0,
                "final_estimate": 5.0,
            },
        ],
    }
    (candidate / "calibration_diagnostics.json").write_text(json.dumps(diagnostics))
    report = _signed_report()
    (candidate / "microcosm_uk_2024_25_local.local_gates.json").write_text(
        json.dumps(report)
    )
    full_report = candidate / "uk.full.gates.calibrated.gate_report.json"
    full_report.write_text(json.dumps(_full_gate_report()))
    if with_incumbent:
        (candidate / "score_vs_incumbent.json").write_text(
            json.dumps(
                {
                    "candidate_fitted_surface_loss": 0.0146,
                    "incumbent_fitted_surface_loss": 0.181,
                    "rows_compared": 2,
                    "incumbent_missing_areas": {},
                    # The scorer pins what it scored; the assembler hash-joins
                    # these to the candidate bytes and the incumbent manifest.
                    "artifacts": {
                        "candidate_diagnostics": {
                            "sha256": _sha(candidate / "calibration_diagnostics.json")
                        },
                        "incumbent_household_metrics": {"sha256": "6" * 64},
                        "incumbent_wide_weights": {"sha256": "7" * 64},
                    },
                }
            )
        )
    (candidate / "logbook-spool" / "row.json").write_text(
        json.dumps(
            {"build_id": ATTEMPT, "row_digest": "f" * 64, "prev_row_digest": "e" * 64}
        )
    )
    manifest = {
        "schema_version": 4,
        "release_role": "dense",
        "created_at": "2026-09-03T10:54:22+00:00",
        "git_commit": "b" * 40,
        "parameters": {
            "release_candidate": True,
            "epochs": 2000,
            "n_clones": 15,
            "skip_holdout": holdout == "skipped",
            "doctrine": {
                "max_weight_ratio": 10.0,
                "target_weight_rule": "nation_grain_family_equal_sqrt_count",
                "solve_epochs": 2000,
                "clone_count": 15,
                "target_loss_cap": 10.0,
                "scale_rule": "default_target_loss_scales",
            },
        },
        "releasable": True,
        "blocked_at_f100": False,
        "blocking_failures": [],
        "solve": {
            "measure_resolution": {"blocks": 1},
            "target_weight_rule_override": {},
            "n_households": 30,
            "final_loss": 0.0156,
            "n_targets_by_kind": {"local": 1, "ladder": 0, "national": 1},
            "binding_adjudications": {"stood_on": {"age_structure/constituency": {}}},
            "area_support_exclusions": {
                "entries_stood_on": ["local_authority/E06000053"]
            },
        },
        "fit": {
            "rotated_holdout": (
                dict(SKIPPED_HOLDOUT)
                if holdout == "skipped"
                else {
                    "report_only": True,
                    "n_folds": 5,
                    "mean_holdout_loss": 0.204,
                    "worst_holdout_loss": 0.213,
                    "folds": [{"fold": 0}],
                }
            )
        },
        "weights": {
            "calibration_mass_change": {"declared_factor": 0.989},
            "realized_max_weight_ratio_vs_design": 10.0,
        },
        "cross_grain": {
            "unbound_bridges": [
                {
                    "bridge_id": "national_household_composition_partition_vs_census_households"
                }
            ],
            "empty_legs_licensed": [1, 2],
        },
        "census_household_uprating": {
            "applied": True,
            "grains": {
                "constituency": {"factor": 1.0335597414671631},
                "local_authority": {"factor": 1.0335595204737118},
            },
            "household_cells": {
                "applied": True,
                "cells": 1,
                "total_cells": 1,
                "attempted_cells": 1,
                "eligible_cells": 1,
                "skipped_cells": 0,
                "holds": [
                    {
                        "target_name": "ons.census.households@E14000001",
                        "geography_level": "constituency",
                        "from_period": 2021,
                        "to_period": 2025,
                        "attempted": True,
                        "eligible": True,
                        "applied": True,
                        "skipped": False,
                        "reason": "census_vintage_hold_uprated",
                    }
                ],
            },
            "tenure_cells": {
                "applied": True,
                "cells": 1,
                "total_cells": 1,
                "attempted_cells": 1,
                "eligible_cells": 1,
                "skipped_cells": 0,
                "holds": [
                    {
                        "attempted": True,
                        "eligible": True,
                        "applied": True,
                        "skipped": False,
                        "target_name": "ons.tenure.owned_outright@E14000001",
                        "geography_level": "constituency",
                        "from_period": 2021,
                        "to_period": 2025,
                        "reason": "census_vintage_hold_uprated",
                    }
                ],
            },
        },
        "measure_exclusions": {
            "obr.housing_benefit": {
                "reason": "unreachable",
                "tracking": "microcosm#736",
                "expires_on": "2099-01-01",
                "approved_on": "2026-09-03",
                "approved_by": "synthetic_reviewer",
                "adjudication": "synthetic decision",
            }
        },
        # What the build's projection records from the full gate report
        # (microcosm#1301), which the contract holds the coverage to.
        "reviewed_exclusions_stood_on": _reviewed_exclusions_stood_on(
            _gate_rows(_full_gate_report())
        ),
        "staging_delivery": _staging_delivery(),
        "geography": {"assignment": synthetic_geography_binding()},
        "identity": {
            "spine": {
                "pin_verified": True,
                "sha256": _sha(spine),
                "path": str(spine),
                "bytes": spine.stat().st_size,
                "spine_provenance": {
                    "rules_engine": {"package": "policyengine-uk", "version": "2.92.1"},
                    "source_vintages": {"frs": "2024_25"},
                    "stages": ["frs_spine"],
                    "stochastic_contract_sha256": "9" * 64,
                },
            },
            "ladder": {
                "pin_verified": True,
                "sha256": "d" * 64,
                "bytes": 1,
                "layer_vintages": {"constituency": "2024_pcon"},
            },
            "targets": {
                "chronicle": {
                    "path_name": "chronicle-uk-artifact-1cab809",
                    "facts_sha256": "1" * 64,
                    "manifest_sha256": "2" * 64,
                    "fact_row_count": 3,
                    "schema_version": "v1",
                },
                "paired_ladder_sha256": "d" * 64,
            },
            "code": {"git_commit": "b" * 40, "git_dirty": False},
            "runtime": {
                "policyengine-core": "3.31.0",
                "policyengine-uk": "2.92.1",
                "microcosm-data": "0.1.0",
                "python": "3.13.14",
            },
        },
        "outputs": {
            "dataset": {"path": str(h5), "sha256": _sha(h5)},
            "calibration_diagnostics": {
                "path": str(candidate / "calibration_diagnostics.json"),
                "sha256": _sha(candidate / "calibration_diagnostics.json"),
            },
            "local_gate_report": {
                "path": str(candidate / "microcosm_uk_2024_25_local.local_gates.json"),
                "sha256": _sha(
                    candidate / "microcosm_uk_2024_25_local.local_gates.json"
                ),
            },
            "full_gate_report": {"path": str(full_report), "sha256": _sha(full_report)},
        },
    }
    (candidate / "rowwise_candidate_manifest.json").write_text(json.dumps(manifest))
    if not with_incumbent:
        return candidate, spine, None
    incumbent_manifest = root / "incumbent_local_surface_manifest.json"
    incumbent_manifest.write_text(
        json.dumps(
            {
                "period": 2025,
                "households": 52846,
                "inputs": {"incumbent_h5": {"sha256": "5" * 64}},
                "outputs": {
                    "metrics": {"sha256": "6" * 64},
                    "weights": {"sha256": "7" * 64},
                },
            }
        )
    )
    evaluation = {
        "schema_version": 2,
        "kind": "uk_incumbent_surface_evaluation",
        "measure_resolution": {"blocks": 1},
        **_surface_rows(),
        "identity": {
            "candidate_dataset_sha256": _sha(h5),
            "candidate_manifest_sha256": _sha(
                candidate / "rowwise_candidate_manifest.json"
            ),
            "candidate_diagnostics_sha256": _sha(
                candidate / "calibration_diagnostics.json"
            ),
            "ledger_facts_sha256": "1" * 64,
            "ledger_manifest_sha256": "2" * 64,
            "incumbent_manifest_sha256": _sha(incumbent_manifest),
            "incumbent_metrics_sha256": "6" * 64,
            "incumbent_weights_sha256": "7" * 64,
        },
    }
    evaluation["summary"] = dc.uk_incumbent_surface_assessment(evaluation)
    (candidate / "incumbent_surface_evaluation.json").write_text(json.dumps(evaluation))
    return candidate, spine, incumbent_manifest


def test_current_diagnostics_are_validated_then_copied_without_rewriting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    assembler = _load("assemble_uk_dense_release_dir")
    source = tmp_path / "source.json"
    destination = tmp_path / "release.json"
    source_bytes = b'{"schema_version":8,"source":"exact bytes"}\n'
    source.write_bytes(source_bytes)
    diagnostics = {"schema_version": 8, "source": "exact bytes"}
    captured: dict[str, object] = {}

    def canonical_parser(payload: object) -> object:
        captured["payload"] = payload
        return object()

    monkeypatch.setattr(assembler, "parse_calibration_diagnostics", canonical_parser)

    assembler._write_release_diagnostics(
        destination,
        source=source,
        diagnostics=diagnostics,
        measured_source_sha256=_sha(source),
        households=3,
    )

    assert captured["payload"] == diagnostics
    assert destination.read_bytes() == source_bytes


def test_current_diagnostics_are_not_copied_when_canonical_validation_fails(
    tmp_path: Path,
) -> None:
    assembler = _load("assemble_uk_dense_release_dir")
    source = tmp_path / "source.json"
    destination = tmp_path / "release.json"
    source.write_text('{"schema_version":8}\n')

    with pytest.raises(SystemExit, match="canonical schema"):
        assembler._write_release_diagnostics(
            destination,
            source=source,
            diagnostics={"schema_version": 8},
            measured_source_sha256=_sha(source),
            households=3,
        )

    assert not destination.exists()


def _limitations(release_dir: Path) -> dict[str, dict]:
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    return {item["id"]: item for item in manifest["reviewed_limitations"]}


def test_assembler_stages_a_contract_valid_dense_release(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """No holdout, no incumbent comparison: the release says what it does not
    claim (microcosm#1301)."""
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)
    assembler = _load("assemble_uk_dense_release_dir")
    candidate, spine, incumbent = _candidate_dir(tmp_path)
    assert incumbent is None
    out = tmp_path / "releases"
    assert assembler.main(_assemble_args(candidate, spine, None, out)) == 0
    summary = json.loads(capsys.readouterr().out)
    release_dir = out / UK_DENSE_RELEASE_ID
    assert summary["release_id"] == UK_DENSE_RELEASE_ID
    assert summary["cut_tag"] == f"{UK_DENSE_RELEASE_ID}-20260903T105422Z-ca611e43"
    assert release_dir.is_dir()
    for name in dc._UK_DENSE_REQUIRED_RELEASE_FILES:
        assert (release_dir / name).is_file(), name
    for name in dc._UK_DENSE_INCUMBENT_FILES:
        assert not (release_dir / name).exists(), name
    limitations = _limitations(release_dir)
    assert {
        "no_out_of_sample_claim",
        "no_incumbent_comparison",
        "engine_measurement",
    } <= set(limitations)
    assert not {
        "rotated_holdout",
        "incumbent_surface_diagnostic",
        "size_doctrine",
    } & set(limitations)
    assert limitations["engine_measurement"]["blocks"] == 1
    # The signed exclusions are named with their register reasons, copied
    # verbatim; the gate's measured error for the row is not copied.
    signed = limitations[f"reviewed_exclusion:uk_target_fit:{SIGNED_TARGET}"]
    assert signed["summary"] == "synthetic reviewed gap"
    assert signed["approved_by"] == "synthetic_reviewer"
    assert "relative_error" not in signed
    assert "0.4321" not in (release_dir / "release_manifest.json").read_text()
    assert "reviewed_exclusion:uk_input_mass_parity:synthetic_column" in limitations
    assert not any("synthetic_dormant" in key for key in limitations)
    coverage = json.loads((release_dir / "uk_source_coverage.json").read_text())
    assert coverage["holdout"] == {"run": False, "claim": "none"}
    assert coverage["incumbent"] == {"compared": False}
    assert coverage["size"]["sized"] is False
    # The names the limitations restate, which the contract checks
    # (microcosm#1301); the dormant input-mass entry is not one.
    assert coverage["signed_deferrals"]["reviewed_exclusions"] == {
        "uk_target_fit": [SIGNED_TARGET],
        "uk_local_target_fit": [],
        "uk_input_mass_parity": ["synthetic_column"],
    }
    # The full gate report does not ship; the candidate manifest's record of
    # the same names does, and the contract holds the coverage to it.
    assert not (release_dir / "uk.full.gates.calibrated.gate_report.json").exists()
    shipped = json.loads((release_dir / "rowwise_candidate_manifest.json").read_text())
    assert (
        shipped["reviewed_exclusions_stood_on"]
        == coverage["signed_deferrals"]["reviewed_exclusions"]
    )
    assert "--tag-only" in summary["publish_command"]
    published = candidate / "microcosm_uk_2024_25_dense.h5"
    assert (
        published.read_bytes()
        == (candidate / "microcosm_uk_2024_25_local.h5").read_bytes()
    )
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    assert manifest["dataset_role"] == "non_default_local_area"
    assert manifest["default_datasets"] == {} and manifest["is_default"] is False
    # The fixture's spine ran frs_spine alone, which reads no SPI tape; the
    # marker is rederived from the stages the provenance names (#1095).
    assert manifest["licensed_data"]["spi_public_use_tape"] == {
        "study": "SN 9422",
        "dependent": False,
        "stages": [],
        "conditions": [],
    }
    assert (
        manifest["artifacts"]["microcosm_uk_2024_25_dense"][
            "spi_public_use_tape_dependent"
        ]
        is False
    )
    assert (
        manifest["artifacts"]["microcosm_uk_2024_25_dense"]["revision"]
        == summary["cut_tag"]
    )
    assert (
        manifest["artifacts"]["microcosm_uk_2024_25_dense"]["repo_id"]
        == "policyengine/populace-uk-private"
    )
    assert any(
        item["id"].startswith("measure_exclusion:obr.housing_benefit")
        for item in manifest["reviewed_limitations"]
    )
    from microcosm.build.uk_runtime.battery_bindings import uk_area_support_floors

    # Codes only: the stood-on support exclusion, no counts (microcosm#1278).
    assert manifest["area_support"] == {
        "gate": "uk_local_area_support",
        "floors": uk_area_support_floors(),
        "below_floor": {"local_authority": ["E06000053"]},
    }
    diagnostics = json.loads((release_dir / "calibration_diagnostics.json").read_text())
    assert diagnostics["households"] == 30 and diagnostics["n_targets"] == 2
    gate_summary = json.loads((release_dir / "gate_summary.json").read_text())
    assert set(gate_summary["gates"]) == set(dc._UK_DENSE_RELEASE_BLOCKING_IDS)
    assert set(gate_summary["diagnostic_gates"]) == set(UK_LOCAL_GATE_SCOPE) - set(
        dc._UK_DENSE_RELEASE_BLOCKING_IDS
    )
    assert (
        "--no-latest" in summary["publish_command"]
        and "populace-uk-private" in summary["publish_command"]
    )
    # The national assembler's rule: the run's staging receipt rides into the
    # build manifest unchanged, where publication reads it.
    build_manifest = json.loads((release_dir / "build_manifest.json").read_text())
    assert build_manifest["staging"] == _staging_delivery()
    # The contract re-validates the finished directory, and so does the pre-flight.
    dc.validate_release_dir(release_dir)
    preflight = _load("preflight_uk_local_release_candidate")
    assert preflight.main(["--release-dir", str(release_dir)]) == 0


def test_assembler_warns_on_a_shipped_support_figure_and_still_assembles(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A per-area support figure in a file the assembler ships byte for byte
    is a warning, not a refusal: the directory is assembled, and the warning
    is printed and listed in the summary by file and path, never by value
    (review of microcosm#1307)."""
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)
    assembler = _load("assemble_uk_dense_release_dir")
    candidate, spine, _ = _candidate_dir(tmp_path)
    manifest_path = candidate / "rowwise_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # Synthetic figures beside a placeholder code no area carries.
    manifest["fit"]["weakest_areas"] = [
        {
            "area_code": "E06999999",
            "nonzero_households": 37,
            "effective_sample_size": 23.75,
        }
    ]
    manifest_path.write_text(json.dumps(manifest))
    out = tmp_path / "releases"

    assert assembler.main(_assemble_args(candidate, spine, None, out)) == 0
    captured = capsys.readouterr()
    (warning,) = json.loads(captured.out)["warnings"]
    assert warning.startswith(
        "rowwise_candidate_manifest.json carries 2 per-area support figure(s)"
    )
    assert "fit.weakest_areas[0].nonzero_households" in warning
    assert f"warning: {warning}" in captured.err
    for figure in ("37", "23.75"):
        assert figure not in warning
    release_dir = out / UK_DENSE_RELEASE_ID
    assert dc.validate_release_dir(release_dir) == [warning]


def test_assembler_override_assembles_a_pre_lane_run_with_a_recorded_opt_out(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)
    assembler = _load("assemble_uk_dense_release_dir")
    candidate, spine, incumbent = _candidate_dir(tmp_path)
    manifest_path = candidate / "rowwise_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["staging_delivery"]
    manifest_path.write_text(json.dumps(manifest))
    out = tmp_path / "releases"
    argv = _assemble_args(candidate, spine, incumbent, out)
    with pytest.raises(SystemExit, match="--allow-missing-staging"):
        assembler.main(argv)
    assert not out.exists()
    assert assembler.main([*argv, "--allow-missing-staging"]) == 0
    capsys.readouterr()
    build_manifest = json.loads(
        (out / UK_DENSE_RELEASE_ID / "build_manifest.json").read_text()
    )
    assert build_manifest["staging"]["mode"] == "disabled"
    assert build_manifest["staging"]["enabled"] is False
    assert "--allow-missing-staging" in build_manifest["staging"]["opt_out_reason"]
    dc.validate_release_dir(out / UK_DENSE_RELEASE_ID)


@pytest.mark.parametrize(
    ("staging_delivery", "message"),
    [
        (None, "missing valid staging-delivery evidence"),
        ({"contract_version": 2, "enabled": True}, "invalid staging-delivery evidence"),
    ],
)
def test_assembler_refuses_a_run_without_staging_evidence(
    tmp_path: Path, monkeypatch, staging_delivery, message
) -> None:
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)
    assembler = _load("assemble_uk_dense_release_dir")
    candidate, spine, incumbent = _candidate_dir(tmp_path)
    manifest_path = candidate / "rowwise_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if staging_delivery is None:
        del manifest["staging_delivery"]
    else:
        manifest["staging_delivery"] = staging_delivery
    manifest_path.write_text(json.dumps(manifest))
    argv = _assemble_args(candidate, spine, incumbent, tmp_path / "releases")
    with pytest.raises(SystemExit, match=message):
        assembler.main(argv)
    if staging_delivery is not None:
        # The override covers absent evidence only; invalid evidence stays refused.
        with pytest.raises(SystemExit, match=message):
            assembler.main([*argv, "--allow-missing-staging"])
    assert not (tmp_path / "releases").exists()


def test_assembler_refuses_a_dev_posture_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)
    assembler = _load("assemble_uk_dense_release_dir")
    candidate, spine, incumbent = _candidate_dir(tmp_path)
    manifest_path = candidate / "rowwise_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["parameters"]["release_candidate"] = False
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="pre-flight failed"):
        assembler.main(
            _assemble_args(candidate, spine, incumbent, tmp_path / "releases")
        )


def test_assembler_refuses_tampered_candidate_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)
    assembler = _load("assemble_uk_dense_release_dir")
    candidate, spine, incumbent = _candidate_dir(tmp_path)
    (candidate / "microcosm_uk_2024_25_local.h5").write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="candidate bytes"):
        assembler.main(
            _assemble_args(candidate, spine, incumbent, tmp_path / "releases")
        )


@pytest.mark.parametrize("tamper", ["missing", "bytes", "kind"])
def test_assembler_requires_the_full_gate_report(tmp_path, monkeypatch, tamper):
    """The full gate report names the reviewed exclusions the release stood on."""
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)
    candidate, spine, incumbent = _candidate_dir(tmp_path)
    path = candidate / "uk.full.gates.calibrated.gate_report.json"
    if tamper == "missing":
        path.unlink()
        expected = "full gate report missing"
    elif tamper == "bytes":
        path.write_text(json.dumps({**_full_gate_report(), "extra": 1}))
        expected = "full gate report bytes"
    else:
        path.write_text(json.dumps({**_full_gate_report(), "kind": "other"}))
        manifest_path = candidate / "rowwise_candidate_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["outputs"]["full_gate_report"]["sha256"] = _sha(path)
        manifest_path.write_text(json.dumps(manifest))
        expected = "not a uk_full_gate_report"
    with pytest.raises(SystemExit, match=expected):
        _load("assemble_uk_dense_release_dir").main(
            _assemble_args(candidate, spine, incumbent, tmp_path / "r")
        )


def test_assembly_refuses_names_the_candidate_did_not_record(
    tmp_path, monkeypatch
) -> None:
    """The coverage's national exclusion names come from the full gate report,
    which does not ship; the shipped candidate manifest's record of them is
    what the contract holds them to (microcosm#1301)."""
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)
    candidate, spine, incumbent = _candidate_dir(tmp_path)
    manifest_path = candidate / "rowwise_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["reviewed_exclusions_stood_on"]["uk_target_fit"] = []
    manifest_path.write_text(json.dumps(manifest))
    out = tmp_path / "releases"
    with pytest.raises(
        dc.ReleaseContractError,
        match="reviewed_exclusions.uk_target_fit does not match the reviewed "
        "exclusions the evaluated candidate recorded",
    ):
        _load("assemble_uk_dense_release_dir").main(
            _assemble_args(candidate, spine, incumbent, out)
        )
    assert not (out / UK_DENSE_RELEASE_ID).exists()


def _assemble_args(
    candidate: Path, spine: Path, incumbent: Path | None, out: Path
) -> list[str]:
    return [
        "--candidate-dir",
        str(candidate),
        "--spine-h5",
        str(spine),
        *(() if incumbent is None else ("--incumbent-manifest", str(incumbent))),
        "--out-dir",
        str(out),
    ]


def test_assembler_attaches_the_incumbent_comparison_on_request(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """--incumbent-manifest attaches all three files as a recorded diagnostic."""
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)
    candidate, spine, incumbent = _candidate_dir(tmp_path, with_incumbent=True)
    out = tmp_path / "releases"
    assembler = _load("assemble_uk_dense_release_dir")
    assert assembler.main(_assemble_args(candidate, spine, incumbent, out)) == 0
    capsys.readouterr()
    release_dir = out / UK_DENSE_RELEASE_ID
    for name in dc._UK_DENSE_INCUMBENT_FILES:
        assert (release_dir / name).is_file(), name
    limitations = _limitations(release_dir)
    assert "no_incumbent_comparison" not in limitations
    assert limitations["incumbent_surface_diagnostic"]["passed"] is True
    assert limitations["incumbent_surface_diagnostic"]["failure_count"] == 0
    coverage = json.loads((release_dir / "uk_source_coverage.json").read_text())
    assert coverage["incumbent"]["compared"] is True
    assert coverage["incumbent"]["snapshot"] == {"incumbent_h5": {"sha256": "5" * 64}}
    dc.validate_release_dir(release_dir)


def test_assembler_ignores_comparison_files_without_the_flag(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)
    candidate, spine, _ = _candidate_dir(tmp_path, with_incumbent=True)
    out = tmp_path / "releases"
    assembler = _load("assemble_uk_dense_release_dir")
    assert assembler.main(_assemble_args(candidate, spine, None, out)) == 0
    captured = capsys.readouterr()
    assert "not attached" in captured.err and "score_vs_incumbent.json" in captured.err
    release_dir = out / UK_DENSE_RELEASE_ID
    for name in dc._UK_DENSE_INCUMBENT_FILES:
        assert not (release_dir / name).exists(), name
    assert "no_incumbent_comparison" in _limitations(release_dir)
    dc.validate_release_dir(release_dir)


def test_a_measured_holdout_is_reported_not_claimed(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)
    candidate, spine, incumbent = _candidate_dir(tmp_path, holdout="measured")
    out = tmp_path / "releases"
    assembler = _load("assemble_uk_dense_release_dir")
    assert assembler.main(_assemble_args(candidate, spine, incumbent, out)) == 0
    capsys.readouterr()
    release_dir = out / UK_DENSE_RELEASE_ID
    limitations = _limitations(release_dir)
    assert "no_out_of_sample_claim" not in limitations
    assert limitations["rotated_holdout"]["claim"] == "report_only"
    assert limitations["rotated_holdout"]["mean_holdout_loss"] == 0.204
    coverage = json.loads((release_dir / "uk_source_coverage.json").read_text())
    assert coverage["holdout"]["run"] is True
    assert coverage["holdout"]["claim"] == "report_only"
    assert "folds" not in coverage["holdout"]
    dc.validate_release_dir(release_dir)


def test_assembler_refuses_a_score_computed_on_other_diagnostics(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)
    assembler = _load("assemble_uk_dense_release_dir")
    candidate, spine, incumbent = _candidate_dir(tmp_path, with_incumbent=True)
    score_path = candidate / "score_vs_incumbent.json"
    score = json.loads(score_path.read_text())
    score["artifacts"]["candidate_diagnostics"]["sha256"] = "9" * 64
    score_path.write_text(json.dumps(score))
    with pytest.raises(SystemExit, match="candidate_diagnostics"):
        assembler.main(_assemble_args(candidate, spine, incumbent, tmp_path / "r"))


def test_assembler_refuses_a_score_against_another_incumbent_extraction(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)
    assembler = _load("assemble_uk_dense_release_dir")
    candidate, spine, incumbent = _candidate_dir(tmp_path, with_incumbent=True)
    manifest = json.loads(incumbent.read_text())
    manifest["outputs"]["weights"]["sha256"] = "8" * 64
    incumbent.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="incumbent_wide_weights"):
        assembler.main(_assemble_args(candidate, spine, incumbent, tmp_path / "r"))


@pytest.mark.parametrize("dirty", [True, None, "false", 0, "missing"])
def test_assembler_refuses_unmeasured_or_dirty_code(tmp_path, monkeypatch, dirty):
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)
    assembler = _load("assemble_uk_dense_release_dir")
    candidate, spine, incumbent = _candidate_dir(tmp_path)
    manifest_path = candidate / "rowwise_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if dirty == "missing":
        manifest["identity"]["code"].pop("git_dirty")
    else:
        manifest["identity"]["code"]["git_dirty"] = dirty
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="code.git_dirty"):
        assembler.main(_assemble_args(candidate, spine, incumbent, tmp_path / "a"))


def _surface_rows():
    return {
        "national_rows": [
            {
                "incumbent_name": "synthetic_national",
                "incumbent_target": 100.0,
                "family": "synthetic_program",
                "candidate_estimate": 100.0,
                "status": "measure_excluded",
            }
        ],
        "local_rows": [
            {
                "incumbent_name": "synthetic_local",
                "incumbent_target": 100.0,
                "area_type": "local_authority",
                "geography_id": "SYNTHETIC",
                "incumbent_metric": "synthetic_metric",
                "candidate_estimate": 100.0,
                "incumbent_estimate": 99.0,
                "status": "signed_deferred",
            }
        ],
    }


@pytest.fixture(autouse=True)
def _synthetic_surface_inventory(monkeypatch):
    from datetime import date

    class FixedDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 9, 7)

    monkeypatch.setattr(dc, "date", FixedDate)
    monkeypatch.setattr(
        dc,
        "_UK_DENSE_SURFACE_INVENTORIES",
        {
            grain: {
                "rows": len(rows),
                "sha256": dc._canonical_sha256(
                    [dc._uk_incumbent_row_identity(r, grain) for r in rows]
                ),
            }
            for grain, rows in (
                ("national", _surface_rows()["national_rows"]),
                ("local", _surface_rows()["local_rows"]),
            )
        },
    )


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("missing", "incumbent surface missing"),
        ("missing_score", "score receipt missing"),
        ("summary", "summary does not match"),
        ("input", "identity.ledger_facts_sha256"),
    ],
)
def test_assembler_requires_a_complete_authenticated_incumbent_surface(
    tmp_path, monkeypatch, mutation, expected
):
    """An attached comparison's identities refuse (microcosm#1301)."""
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)
    candidate, spine, incumbent = _candidate_dir(tmp_path, with_incumbent=True)
    path = candidate / "incumbent_surface_evaluation.json"
    if mutation == "missing":
        path.unlink()
    elif mutation == "missing_score":
        (candidate / "score_vs_incumbent.json").unlink()
    else:
        payload = json.loads(path.read_text())
        if mutation == "summary":
            payload["summary"]["passed"] = False
        else:
            payload["identity"]["ledger_facts_sha256"] = "9" * 64
        path.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match=expected):
        _load("assemble_uk_dense_release_dir").main(
            _assemble_args(candidate, spine, incumbent, tmp_path / "r")
        )


def test_an_attached_comparison_records_a_failed_assessment(
    tmp_path, monkeypatch, capsys
):
    """The measured limits are recorded, not enforced: a deferred row that
    misses its target ships with the comparison marked failed."""
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)
    candidate, spine, incumbent = _candidate_dir(tmp_path, with_incumbent=True)
    path = candidate / "incumbent_surface_evaluation.json"
    payload = json.loads(path.read_text())
    payload["local_rows"][0]["candidate_estimate"] = 1.8
    payload["summary"] = dc.uk_incumbent_surface_assessment(payload)
    path.write_text(json.dumps(payload))
    out = tmp_path / "r"
    assert (
        _load("assemble_uk_dense_release_dir").main(
            _assemble_args(candidate, spine, incumbent, out)
        )
        == 0
    )
    capsys.readouterr()
    release_dir = out / UK_DENSE_RELEASE_ID
    diagnostic = _limitations(release_dir)["incumbent_surface_diagnostic"]
    assert diagnostic["passed"] is False and diagnostic["failure_count"] == 1
    coverage = json.loads((release_dir / "uk_source_coverage.json").read_text())
    assert coverage["incumbent"]["assessment_passed"] is False
    assert coverage["incumbent"]["assessment_failure_count"] == 1
    dc.validate_release_dir(release_dir)


def test_assembler_preserves_full_measure_approval_provenance(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)
    candidate, spine, incumbent = _candidate_dir(tmp_path)
    out = tmp_path / "release"
    assert (
        _load("assemble_uk_dense_release_dir").main(
            _assemble_args(candidate, spine, incumbent, out)
        )
        == 0
    )
    capsys.readouterr()
    original = json.loads((candidate / "rowwise_candidate_manifest.json").read_text())
    coverage = json.loads(
        (out / UK_DENSE_RELEASE_ID / "uk_source_coverage.json").read_text()
    )
    assert coverage["measure_exclusions"] == original["measure_exclusions"]
    assert (
        coverage["uprating"]["grains"]
        == original["census_household_uprating"]["grains"]
    )


@pytest.mark.parametrize("expiry", [None, "2026-09-01", "not-a-date"])
def test_assembler_rejects_expired_or_missing_measure_approval(
    tmp_path, monkeypatch, expiry
):
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)
    candidate, spine, incumbent = _candidate_dir(tmp_path)
    path = candidate / "rowwise_candidate_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["measure_exclusions"]["obr.housing_benefit"]["expires_on"] = expiry
    path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="measure exclusion"):
        _load("assemble_uk_dense_release_dir").main(
            _assemble_args(candidate, spine, incumbent, tmp_path / "r")
        )


def test_area_support_limitations_name_each_exclusion_by_its_grain() -> None:
    # microcosm#1301: a gate-only exclusion may excuse a constituency too.
    tool = _load("assemble_uk_dense_release_dir")
    solve = {
        "area_support_exclusions": {
            "entries_stood_on": ["constituency/PCON_A", "local_authority/LA_A"]
        }
    }
    assert tool._area_support_exclusion_limitations(solve) == [
        {
            "id": "area_support_exclusion:constituency/PCON_A",
            "summary": "reviewed exclusion of the constituency support floor.",
        },
        {
            "id": "area_support_exclusion:local_authority/LA_A",
            "summary": "reviewed exclusion of the local-authority support floor.",
        },
    ]
    assert tool._area_support_exclusion_limitations({}) == []


def test_assembler_refuses_a_national_role_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    """The dense assembler covers the dense line only (microcosm#823)."""

    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)
    assembler = _load("assemble_uk_dense_release_dir")
    candidate, spine, incumbent = _candidate_dir(tmp_path)
    manifest_path = candidate / "rowwise_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["release_role"] = "national"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="release_role is 'national'"):
        assembler.main(
            _assemble_args(candidate, spine, incumbent, tmp_path / "releases")
        )
    del manifest["release_role"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="release_role is None"):
        assembler.main(
            _assemble_args(candidate, spine, incumbent, tmp_path / "releases")
        )


@pytest.mark.parametrize(
    ("sized", "diagnostics"),
    [(False, False), (True, False), (False, True)],
    ids=["whole-pool", "doctrine-sized", "rotated-holdout-and-incumbent"],
)
def test_assembler_stages_a_graph_built_dense_candidate(
    tmp_path: Path, monkeypatch, capsys, sized: bool, diagnostics: bool
) -> None:
    """The graph driver's bundle assembles like a rowwise-tool bundle.

    A release candidate sized at the size doctrine (microcosm#1124, ruling of
    2026-10-13) assembles on the same contract as the whole pool.

    ``main`` runs the synthetic graph with local-only staging, so the manifest
    carries the staging receipt, the staged-dataset sidecars and the Logbook
    row the assembler hash-joins. The build signs its own local battery report,
    bound to the Logbook attempt. By default the run requests neither the
    rotated holdout nor the incumbent comparison (microcosm#1301); the
    ``diagnostics`` case opts into both, supplying the incumbent-surface
    companions as the scoring step supplies them.

    The area-support gate reports the real gate's details over the synthetic
    support table, whose every area sits below the floors under a reviewed
    exclusion: no shipped file may carry those areas' support figures
    (microcosm#1301).
    """

    pytest.importorskip("tables")
    import io

    import pandas as pd

    from microcosm.build.gates import area_support_gate
    from microcosm.build.logbook import load_spool_rows
    from microcosm.build.uk_runtime import full_build_cli as cli
    from microcosm.build.uk_runtime.battery_bindings import uk_area_support_floors
    from test_support.microcosm_build import uk_full_build_cli as harness
    from test_support.microcosm_build.uk_full_build_cli import (
        STEM,
        graph_dense_bundle,
    )

    support = pd.read_csv(io.BytesIO(harness.SUPPORT_CSV))
    excused = {
        f"{row.geography_level}/{row.area_code}": {"reason": "synthetic review"}
        for row in support.itertuples(index=False)
    }
    support_gate = area_support_gate(
        support,
        area_roster={
            level: tuple(rows["area_code"])
            for level, rows in support.groupby("geography_level")
        },
        geography_levels=("constituency", "local_authority"),
        reviewed_exclusions=excused,
        **uk_area_support_floors(),
    )
    assert support_gate.passed
    monkeypatch.setattr(harness, "AREA_SUPPORT_DETAILS", dict(support_gate.details))
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)
    monkeypatch.setattr(cli, "git_commit", lambda: "b" * 40)
    monkeypatch.setattr(cli, "git_dirty", lambda: False)
    size = ("--dataset-households", str(UK_LOCAL_DATASET_HOUSEHOLDS)) if sized else ()
    rotated = ("--rotated-holdout",) if diagnostics else ()
    candidate = graph_dense_bundle(
        tmp_path,
        monkeypatch,
        "--release-candidate",
        *size,
        *rotated,
        staging="--staging-local-only",
    )
    # The graph driver prints the finished manifest to stdout, as the tool does.
    printed = json.loads(capsys.readouterr().out)
    assert printed["release_role"] == "dense"
    spine = tmp_path / "spine.h5"
    row = load_spool_rows(candidate / "logbook-spool")[0]
    manifest_path = candidate / "rowwise_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["staging_delivery"]["mode"] == "local_only"
    assert manifest["staged_dataset"]["status"] == "skipped"
    assert manifest["releasable"] is True
    assert manifest["parameters"]["skip_holdout"] is (not diagnostics)
    if sized:
        assert manifest["parameters"]["dataset_households"] == 150_000
        assert manifest["release_posture"]["size_doctrine"] is True
        assert manifest["solve"]["dataset_size"]["realized_households"] == 150_000
    else:
        assert manifest["solve"]["dataset_size"] is None
        assert "size_doctrine" not in manifest["release_posture"]
    report_path = candidate / f"{STEM}.local_gates.json"
    report = json.loads(report_path.read_text())
    assert report["release_id"] == row.build_id and report["shippable"] is True
    assert manifest["outputs"]["local_gate_report"]["sha256"] == _sha(report_path)
    diagnostics_path = Path(manifest["outputs"]["calibration_diagnostics"]["path"])
    incumbent_manifest = None
    if diagnostics:
        (candidate / "score_vs_incumbent.json").write_text(
            json.dumps(
                {
                    "candidate_fitted_surface_loss": 0.0146,
                    "incumbent_fitted_surface_loss": 0.181,
                    "rows_compared": 2,
                    "incumbent_missing_areas": {},
                    "artifacts": {
                        "candidate_diagnostics": {"sha256": _sha(diagnostics_path)},
                        "incumbent_household_metrics": {"sha256": "6" * 64},
                        "incumbent_wide_weights": {"sha256": "7" * 64},
                    },
                }
            )
        )
        incumbent_manifest = tmp_path / "incumbent_local_surface_manifest.json"
        incumbent_manifest.write_text(
            json.dumps(
                {
                    "period": 2025,
                    "households": 52846,
                    "inputs": {"incumbent_h5": {"sha256": "5" * 64}},
                    "outputs": {
                        "metrics": {"sha256": "6" * 64},
                        "weights": {"sha256": "7" * 64},
                    },
                }
            )
        )
        chronicle = manifest["identity"]["targets"]["chronicle"]
        evaluation = {
            "schema_version": 2,
            "kind": "uk_incumbent_surface_evaluation",
            "measure_resolution": {"blocks": 1},
            **_surface_rows(),
            "identity": {
                "candidate_dataset_sha256": _sha(candidate / f"{STEM}.h5"),
                "candidate_manifest_sha256": _sha(manifest_path),
                "candidate_diagnostics_sha256": _sha(diagnostics_path),
                "ledger_facts_sha256": chronicle["facts_sha256"],
                "ledger_manifest_sha256": chronicle["manifest_sha256"],
                "incumbent_manifest_sha256": _sha(incumbent_manifest),
                "incumbent_metrics_sha256": "6" * 64,
                "incumbent_weights_sha256": "7" * 64,
            },
        }
        evaluation["summary"] = dc.uk_incumbent_surface_assessment(evaluation)
        (candidate / "incumbent_surface_evaluation.json").write_text(
            json.dumps(evaluation)
        )
    assembler = _load("assemble_uk_dense_release_dir")
    out = tmp_path / "releases"
    assert (
        assembler.main(_assemble_args(candidate, spine, incumbent_manifest, out)) == 0
    )
    summary = json.loads(capsys.readouterr().out)
    release_dir = out / UK_DENSE_RELEASE_ID
    assert summary["release_id"] == UK_DENSE_RELEASE_ID
    for name in dc._UK_DENSE_REQUIRED_RELEASE_FILES:
        assert (release_dir / name).is_file(), name
    for name in dc._UK_DENSE_INCUMBENT_FILES:
        assert (release_dir / name).is_file() is diagnostics, name
    build_manifest = json.loads((release_dir / "build_manifest.json").read_text())
    assert build_manifest["staging"]["mode"] == "local_only"
    coverage = json.loads((release_dir / "uk_source_coverage.json").read_text())
    assert coverage["doctrine"]["clone_count"] == 15
    assert coverage["size"]["sized"] is sized
    # The build records the names each signed-exclusion gate stood on, which
    # the coverage restates from the full gate report (microcosm#1301).
    assert (
        manifest["reviewed_exclusions_stood_on"]
        == coverage["signed_deferrals"]["reviewed_exclusions"]
        == {gate_id: [] for gate_id in dc._UK_DENSE_SIGNED_EXCLUSION_GATES}
    )
    limitations = _limitations(release_dir)
    assert "engine_measurement" in limitations
    assert ("size_doctrine" in limitations) is sized
    if sized:
        # microcosm#1301: the release names the L0 bracket its size search
        # ran in, as the pre-flight checked it.
        bracket = {"lower": 1e-10, "upper": 10.0}
        assert limitations["size_doctrine"]["search_bracket"] == bracket
        assert coverage["size"]["search_bracket"] == bracket
        assert coverage["size"]["dataset_households"] == 150_000
    if diagnostics:
        assert coverage["holdout"]["n_folds"] == 5
        assert coverage["holdout"]["claim"] == "report_only"
        assert coverage["incumbent"]["compared"] is True
        assert limitations["rotated_holdout"]["claim"] == "report_only"
        assert limitations["incumbent_surface_diagnostic"]["passed"] is True
    else:
        assert coverage["holdout"] == {"run": False, "claim": "none"}
        assert coverage["incumbent"] == {"compared": False}
        assert {"no_out_of_sample_claim", "no_incumbent_comparison"} <= set(limitations)
    # Disclosure (microcosm#1301): every area is below the floors, yet no
    # shipped document carries an area's rows, ESS or sources, so the
    # contract has nothing to warn about. The signed report names the floors
    # each excused area misses; the manifest counts areas; the release
    # manifest names the areas by code.
    assert dc.validate_release_dir(release_dir) == []
    assert summary["warnings"] == []
    for path in sorted(release_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        assert dc._uk_dense_disclosure_findings(path.name, payload) == [], path.name
    signed = json.loads((release_dir / "uk_local_gates.json").read_text())
    shipped_exclusions = signed["gates"]["uk_local_area_support"]["details"][
        "reviewed_exclusions"
    ]
    assert set(shipped_exclusions) == set(excused)
    assert all(
        set(entry) == {"shortfalls", "record"} for entry in shipped_exclusions.values()
    )
    shipped_manifest = json.loads(
        (release_dir / "rowwise_candidate_manifest.json").read_text()
    )
    assert shipped_manifest["support"] == {
        "areas_checked": 4,
        "areas_below_floor": 4,
        "by_geography_level": {
            level: {"areas_checked": 2, "areas_below_floor": 2}
            for level in ("constituency", "local_authority")
        },
    }
    release_manifest = json.loads((release_dir / "release_manifest.json").read_text())
    assert release_manifest["area_support"]["below_floor"] == {
        "constituency": ["E14000001", "E14000002"],
        "local_authority": ["E07000008", "E09000001"],
    }
