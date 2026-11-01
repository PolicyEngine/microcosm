"""The dense joint UK line's release contract (microcosm#762 A18)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from pathlib import Path

import pytest

from microcosm.data import contract as dc
from microcosm.data.contract import (
    ReleaseContractError,
    _canonical_json_bytes,
    _canonical_sha256,
    validate_release_dir,
)

DENSE_ID = "microcosm-uk-2024-25-dense"
CUT_TAG = f"{DENSE_ID}-20260903T105422Z-ca611e43"
KEY = base64.b64encode(b"\x07" * 32).decode()
KEY_BYTES = base64.b64decode(KEY)


@pytest.fixture(autouse=True)
def _trusted_key(monkeypatch) -> None:
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _signed_report(
    *,
    blocking_status: str = "passed",
    release_candidate: bool = True,
    area_support_details: dict | None = None,
):
    gates = {}
    for entry_id in sorted(dc._UK_DENSE_GATE_ENTRY_IDS):
        blocking = entry_id in dc._UK_DENSE_RELEASE_BLOCKING_IDS
        gates[entry_id] = {
            "gate": entry_id.removeprefix("uk_local_"),
            "phase": "terminal",
            "criticality": "release_blocking" if blocking else "diagnostic",
            "status": blocking_status if blocking else "failed",
            "failures": [] if blocking else ["diagnostic miss"],
            "details": (
                dict(area_support_details)
                if entry_id == "uk_local_area_support" and area_support_details
                else {}
            ),
            "reason": None,
        }
    attestation = {
        "schema_version": dc._UK_GATE_BATTERY_ATTESTATION_SCHEMA_VERSION,
        "producer": dc._UK_GATE_BATTERY_PRODUCER,
        "country": "uk",
        "release_id": "uk-local-candidate-f100-s42-20260903T105422Z-ca611e43",
        "release_candidate": release_candidate,
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
        "release_id": attestation["release_id"],
        "release_candidate": release_candidate,
        "spec_fingerprint": attestation["spec_fingerprint"],
        "gates_manifest_sha256": attestation["gates_manifest_sha256"],
        "phases": ["terminal"],
        "phases_evaluated": ["terminal"],
        "blocked_at_phase": None,
        "shippable": blocking_status == "passed" and release_candidate,
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


def _resigned(report: dict, gate_id: str, details: dict) -> dict:
    """``report`` with one gate's details replaced, re-signed with the test key."""
    report = json.loads(json.dumps(report))
    report["gates"][gate_id]["details"] = json.loads(json.dumps(details))
    attestation = report["attestation"]
    attestation["gate_outcomes_sha256"] = _canonical_sha256(report["gates"])
    attestation["signature"] = None
    attestation["signature"] = hmac.new(
        KEY_BYTES, _canonical_json_bytes(report), hashlib.sha256
    ).hexdigest()
    return report


def write_dense_bundle(
    root: Path,
    *,
    release_id: str = DENSE_ID,
    revision: str = CUT_TAG,
    namespace: str = "uk_dense",
    default_datasets: dict | None = None,
    report: dict | None = None,
    drop_coverage_key: str | None = None,
    drop_households: bool = False,
    attempt_id: str | None = None,
    with_incumbent: bool = False,
    holdout: str = "skipped",
    sized: bool = False,
    national_exclusions: tuple[str, ...] = (),
) -> Path:
    """A synthetic dense release directory.

    The core bundle claims neither the rotated holdout nor the incumbent
    comparison (microcosm#1301); ``with_incumbent`` attaches the three
    incumbent files, ``holdout="measured"`` reports a holdout, and ``sized``
    records a sized release, each with its matching limitation. The candidate
    manifest records the same holdout and size. Every reviewed exclusion the
    signed report's local target-fit and area-support gates stood on, and each
    of ``national_exclusions`` (a ``uk_target_fit`` name), is named in the
    coverage and the manifest's limitations, the areas in ``area_support``;
    the candidate manifest records the target-fit and input-mass names under
    ``reviewed_exclusions_stood_on``, as the build does.
    """
    release_dir = root / "releases" / release_id
    release_dir.mkdir(parents=True)
    report = report or _signed_report()
    stood_on = {
        gate_id: sorted(
            (report["gates"].get(gate_id, {}).get("details") or {}).get(
                "reviewed_exclusions"
            )
            or {}
        )
        for gate_id in ("uk_local_target_fit", "uk_local_area_support")
    }
    reviewed_exclusions = {
        "uk_target_fit": sorted(national_exclusions),
        "uk_local_target_fit": stood_on["uk_local_target_fit"],
        "uk_input_mass_parity": [],
    }
    below_floor: dict[str, list[str]] = {}
    for key in stood_on["uk_local_area_support"]:
        level, _, code = key.partition("/")
        below_floor.setdefault(level, []).append(code)
    measured_holdout = {
        "report_only": True,
        "n_folds": 5,
        "mean_holdout_loss": 0.2,
        "worst_holdout_loss": 0.3,
        "folds": [{"fold": 0}],
    }
    size_record = {
        "clone_count": 2,
        "dataset_households": 4 if sized else None,
        "realized_households": 4 if sized else None,
        "selection_initial_lambda": 2.5e-8 if sized else None,
        "selection_l0_lambda": 3e-9 if sized else None,
        "search_bracket": {"lower": 1e-10, "upper": 10.0} if sized else None,
    }
    diagnostics = {
        "schema_version": 6,
        "households": 4,
        "n_targets": 2,
        "n_records": 4,
        "final_loss": 0.01,
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
    if drop_households:
        del diagnostics["households"]
    gate_summary = {
        "gates": {
            entry_id: {
                "passed": report["gates"].get(entry_id, {}).get("status") == "passed"
            }
            for entry_id in dc._UK_DENSE_RELEASE_BLOCKING_IDS
        },
        "diagnostic_gates": {},
        "reviewed_limitations": [],
    }
    coverage = {
        "schema_version": 1,
        "spine": {"path_name": "spine-m.h5", "sha256": "a" * 64},
        "ledger_artifact": {
            "path_name": "chronicle-uk-artifact-1cab809",
            "facts_sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
        },
        "geography_ladder": {"sha256": "d" * 64},
        "incumbent": (
            {"compared": True, "snapshot": {"incumbent_h5": {"sha256": "e" * 64}}}
            if with_incumbent
            else {"compared": False}
        ),
        "doctrine": {"max_weight_ratio": 10.0},
        "measure_exclusions": {
            "obr.housing_benefit": {
                "reason": "synthetic gap",
                "tracking": "microcosm#869",
                "approved_by": "synthetic_reviewer",
                "adjudication": "synthetic decision",
                "approved_on": "2026-09-03",
                "expires_on": "2026-10-03",
            }
        },
        "signed_deferrals": {
            "binding_adjudications": {"x": {}},
            "reviewed_exclusions": json.loads(json.dumps(reviewed_exclusions)),
        },
        "holdout": (
            {
                **{k: v for k, v in measured_holdout.items() if k != "folds"},
                "run": True,
                "claim": "report_only",
            }
            if holdout == "measured"
            else {"run": False, "claim": "none"}
        ),
        "size": {"sized": sized, **size_record},
        "uprating": {"applied": True, "factor": 1.03},
    }
    if drop_coverage_key:
        del coverage[drop_coverage_key]
    build_manifest = {
        "build_id": release_id,
        "build_sha": "abc1234",
        "code": {"git_commit": "a" * 40, "git_dirty": False},
        "attempt_id": attempt_id if attempt_id is not None else report["release_id"],
    }
    score = {
        "candidate_fitted_surface_loss": 0.01,
        "incumbent_fitted_surface_loss": 0.2,
    }
    payloads = {
        "calibration_diagnostics.json": json.dumps(diagnostics, indent=1).encode(),
        "gate_summary.json": json.dumps(gate_summary, indent=1).encode(),
        "uk_source_coverage.json": json.dumps(coverage, indent=1).encode(),
        "uk_local_gates.json": json.dumps(report, indent=1).encode(),
        "build_manifest.json": json.dumps(build_manifest, indent=1).encode(),
    }
    source_diagnostics = payloads["calibration_diagnostics.json"]
    candidate = {
        "identity": {
            "code": {"git_commit": "a" * 40, "git_dirty": False},
            "targets": {
                "chronicle": {
                    "facts_sha256": "b" * 64,
                    "manifest_sha256": "c" * 64,
                }
            },
        },
        "outputs": {
            "dataset": {"sha256": _sha(b"dense-h5-stand-in")},
            "calibration_diagnostics": {"sha256": _sha(source_diagnostics)},
            "local_gate_report": {"sha256": _sha(payloads["uk_local_gates.json"])},
        },
    }
    candidate["measure_exclusions"] = coverage.get("measure_exclusions", {})
    candidate["reviewed_exclusions_stood_on"] = reviewed_exclusions
    candidate["parameters"] = {
        "skip_holdout": holdout != "measured",
        "n_clones": size_record["clone_count"],
        "dataset_households": size_record["dataset_households"],
        "selection_initial_lambda": size_record["selection_initial_lambda"],
    }
    candidate["fit"] = {
        "rotated_holdout": (
            measured_holdout
            if holdout == "measured"
            else {"report_only": True, "skipped": True, "reason": "Not requested."}
        )
    }
    candidate["solve"] = {
        "measure_resolution": {"blocks": 1},
        "dataset_size": (
            {
                "requested_households": size_record["dataset_households"],
                "realized_households": size_record["realized_households"],
                "selection_l0_lambda": size_record["selection_l0_lambda"],
                "selection_search_bracket": size_record["search_bracket"],
            }
            if sized
            else None
        ),
    }
    incumbent = {
        "inputs": {"incumbent_h5": {"sha256": "e" * 64}},
        "outputs": {"metrics": {"sha256": "6" * 64}, "weights": {"sha256": "7" * 64}},
    }
    payloads["rowwise_candidate_manifest.json"] = json.dumps(candidate).encode()
    incumbent_manifest = json.dumps(incumbent).encode()
    payloads["source_calibration_diagnostics.json"] = source_diagnostics
    evaluation = {
        "schema_version": 2,
        "kind": "uk_incumbent_surface_evaluation",
        "measure_resolution": {"blocks": 1},
        **_surface_rows(),
        "identity": {
            "candidate_dataset_sha256": _sha(b"dense-h5-stand-in"),
            "candidate_manifest_sha256": _sha(
                payloads["rowwise_candidate_manifest.json"]
            ),
            "candidate_diagnostics_sha256": _sha(source_diagnostics),
            "ledger_facts_sha256": "b" * 64,
            "ledger_manifest_sha256": "c" * 64,
            "incumbent_manifest_sha256": _sha(incumbent_manifest),
            "incumbent_metrics_sha256": "6" * 64,
            "incumbent_weights_sha256": "7" * 64,
        },
    }
    diagnostics["source_diagnostics_sha256"] = _sha(source_diagnostics)
    payloads["calibration_diagnostics.json"] = json.dumps(diagnostics).encode()
    score["artifacts"] = {
        "candidate_diagnostics": {"sha256": _sha(source_diagnostics)},
        "incumbent_household_metrics": {"sha256": "6" * 64},
        "incumbent_wide_weights": {"sha256": "7" * 64},
    }
    evaluation["summary"] = dc.uk_incumbent_surface_assessment(evaluation)
    limitations = [
        {"id": "engine_measurement", "blocks": 1},
        (
            {"id": "rotated_holdout", "claim": "report_only"}
            if holdout == "measured"
            else {"id": "no_out_of_sample_claim"}
        ),
        *([{"id": "size_doctrine", **size_record}] if sized else []),
        *(
            {"id": f"reviewed_exclusion:uk_target_fit:{name}"}
            for name in sorted(national_exclusions)
        ),
        *(
            {"id": f"reviewed_exclusion:uk_local_target_fit:{name}"}
            for name in stood_on["uk_local_target_fit"]
        ),
        *(
            {"id": f"area_support_exclusion:{key}"}
            for key in stood_on["uk_local_area_support"]
        ),
    ]
    if with_incumbent:
        payloads["score_vs_incumbent.json"] = json.dumps(score).encode()
        payloads["incumbent_manifest.json"] = incumbent_manifest
        payloads["incumbent_surface_evaluation.json"] = json.dumps(evaluation).encode()
        limitations.append(
            {"id": "incumbent_surface_diagnostic", "passed": True, "failure_count": 0}
        )
    else:
        limitations.append({"id": "no_incumbent_comparison"})
    for name, payload in payloads.items():
        (release_dir / name).write_bytes(payload)
    h5_bytes = b"dense-h5-stand-in"
    (root / "microcosm_uk_2024_25_dense.h5").write_bytes(h5_bytes)

    def artifact(kind, path, sha):
        return {
            "kind": kind,
            "path": path,
            "repo_id": "policyengine/populace-uk-private",
            "revision": revision,
            "sha256": sha,
        }

    manifest = {
        "schema_version": 1,
        "data_package": {"name": "microcosm-data", "version": "0.1.0"},
        "dataset_role": "non_default_local_area",
        "is_default": False,
        "default_datasets": default_datasets if default_datasets is not None else {},
        "namespace": namespace,
        "build": {"build_id": release_id},
        "artifacts": {
            "microcosm_uk_2024_25_dense": artifact(
                "microdata", "microcosm_uk_2024_25_dense.h5", _sha(h5_bytes)
            ),
            **{
                Path(name).stem: artifact("diagnostics", name, _sha(payload))
                for name, payload in payloads.items()
                if name != "build_manifest.json"
            },
        },
        "reviewed_limitations": limitations,
        "area_support": {
            "gate": "uk_local_area_support",
            "floors": {
                "minimum_rows": 50,
                "minimum_effective_sample_size": 50.0,
                "minimum_distinct_sources": 50,
            },
            "below_floor": below_floor,
        },
    }
    manifest_text = json.dumps(manifest, indent=1)
    (release_dir / "release_manifest.json").write_text(manifest_text)
    ledger = {name: _sha(payload) for name, payload in payloads.items()}
    ledger["release_manifest.json"] = _sha(manifest_text.encode())
    ledger["microcosm_uk_2024_25_dense.h5"] = _sha(h5_bytes)
    (release_dir / "sha256sums.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(ledger.items()))
    )
    return release_dir


def test_local_area_diagnostics_use_canonical_v8_count_fields() -> None:
    diagnostics = {
        "schema_version": 8,
        "n_records": 4,
        "target_surface": {"n_targets": 1},
        "final_loss": 0.01,
        "fraction_within_10pct": 1.0,
        "targets": [
            {
                "name": "target",
                "target": 1.0,
                "compiled_target": 1.0,
                "initial_estimate": 1.0,
                "final_estimate": 1.0,
            }
        ],
    }
    failures: list[str] = []

    dc._check_local_area_calibration_diagnostics(diagnostics, failures)

    assert failures == []


def _withheld_diagnostics(**row_overrides) -> dict:
    """A published row and a withheld BRMA row, with totals over the first."""

    withheld_row = {
        "name": "brma",
        "target": 50.0,
        "compiled_target": 50.0,
        "initial_estimate": None,
        "final_estimate": None,
        "relative_error": None,
        "target_loss_weight_share": 0.5,
        "target_loss_scale": 50.0,
        "metadata": {
            "model_figures_withheld": "model figures withheld",
            "area_type": "brma",
            "carrier_floor_met": False,
        },
        **row_overrides,
    }
    return {
        "schema_version": 8,
        "n_records": 4,
        "target_surface": {"n_targets": 2},
        "initial_loss": 0.1,
        "final_loss": 0.025,
        "fraction_within_10pct": 1.0,
        "target_loss_basis": {"cap": 10.0},
        "targets": [
            {
                "name": "target",
                "target": 1.0,
                "compiled_target": 1.0,
                "initial_estimate": 1.2,
                "final_estimate": 1.05,
                "relative_error": 0.05,
                "target_loss_weight_share": 0.5,
                "target_loss_scale": 1.0,
                "final_capped_scaled_error": 0.05,
                "final_loss_contribution": 0.025,
            },
            withheld_row,
        ],
        "past_cap_census": {
            "cap": 10.0,
            "n_targets": 1,
            "initial_past_cap": 0,
            "final_past_cap": 0,
            "escaped": 0,
            "frozen": 0,
            "pushed_out": 0,
            "pushed_out_rows": [],
        },
        "uk_diagnostics": {
            "model_figures_withheld": {"targets": 1},
            "target_pass_rates_by_geography_level": [
                {
                    "geography_level": "local_authority",
                    "n_targets": 1,
                    "n_scored": 1,
                    "n_skipped": 0,
                    "n_within_10pct": 1,
                },
                {
                    "geography_level": "brma",
                    "n_targets": 1,
                    "n_scored": 1,
                    "n_skipped": 0,
                    "n_within_10pct": 0,
                },
            ],
        },
        "loss_trajectory": [],
        "options": {
            "method": "adam",
            "budget_search": None,
            "iterate_selection_receipt": {
                "rule": "best_feasible_loss",
                "selected_epoch": 3,
                "selected_loss_float32": None,
                "closing_iterate_loss_float32": None,
            },
        },
    }


def test_a_withheld_row_may_leave_its_estimates_null() -> None:
    """microcosm#1262: a thin BRMA row keeps its place without its figures."""

    failures: list[str] = []
    dc._check_local_area_calibration_diagnostics(_withheld_diagnostics(), failures)
    assert failures == []


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"final_estimate": 40.0}, "withholds its model figures but carries one"),
        ({"relative_error": -0.2}, "withholds its model figures but carries one"),
        ({"metadata": {}}, "must be a finite number"),
    ],
)
def test_a_withheld_row_is_held_to_its_note(overrides, message) -> None:
    failures: list[str] = []
    dc._check_local_area_calibration_diagnostics(
        _withheld_diagnostics(**overrides), failures
    )
    assert any(message in failure for failure in failures), failures


@pytest.mark.parametrize(
    "field",
    ["within_tolerance", "final_capped_scaled_error", "final_loss_contribution"],
)
def test_a_withheld_row_carries_none_of_its_model_fields(field) -> None:
    """The third review of #1305: every model field, from the shared tuple."""

    failures: list[str] = []
    dc._check_local_area_calibration_diagnostics(
        _withheld_diagnostics(**{field: 0.05}), failures
    )
    assert any("withholds its model figures but carries one" in f for f in failures)


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda d: d.update(loss_trajectory=[0.4, 0.3]), "keeps a loss trajectory"),
        (
            lambda d: d["options"]["iterate_selection_receipt"].update(
                selected_loss_float32=0.3
            ),
            "iterate_selection_receipt keeps ['selected_loss_float32']",
        ),
        (
            lambda d: d["options"].update(budget_search={"probes": [{"loss": 0.4}]}),
            "options.budget_search is not null",
        ),
        (
            lambda d: d.update(past_cap_census={"pushed_out_rows": [{"name": "brma"}]}),
            "names 1 withheld row(s)",
        ),
        (
            lambda d: d["uk_diagnostics"].update(
                weakest_families=[{"family": "private_rent", "worst_target": "brma"}]
            ),
            "names 1 withheld row(s)",
        ),
        (
            lambda d: d["uk_diagnostics"].update(
                weakest_areas_by_fit={"bottom_by_fit": [{"worst_target": "brma"}]}
            ),
            "names 1 withheld row(s)",
        ),
    ],
    ids=["trajectory", "iterate-loss", "budget-search", "census", "rollup", "areas"],
)
def test_no_total_over_every_row_ships_beside_a_withheld_row(mutate, message) -> None:
    """While a row is withheld, the totals and rollups cannot give it back."""

    diagnostics = _withheld_diagnostics()
    failures: list[str] = []
    dc._check_local_area_calibration_diagnostics(diagnostics, failures)
    assert failures == []
    mutate(diagnostics)
    dc._check_local_area_calibration_diagnostics(diagnostics, failures)
    assert any(message in failure for failure in failures), failures


def _set_level_passes(diagnostics: dict) -> None:
    levels = diagnostics["uk_diagnostics"]["target_pass_rates_by_geography_level"]
    levels[1]["n_within_10pct"] = 1


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda d: d.update(final_loss=0.125), "its final_loss is not restated"),
        (lambda d: d.update(initial_loss=0.3), "its initial_loss is not restated"),
        (
            lambda d: d.update(fraction_within_10pct=0.5),
            "its fraction_within_10pct is not restated",
        ),
        (
            lambda d: d["past_cap_census"].update(n_targets=2),
            "past_cap_census does not count the published rows only",
        ),
        (_set_level_passes, "pass rates by geography level"),
        (
            lambda d: d["targets"][1]["metadata"].update(estimate=40.0),
            "its metadata carries one",
        ),
        (
            lambda d: d["targets"][0].pop("final_loss_contribution"),
            "lacks the fields its restated totals are recomputed from",
        ),
    ],
    ids=["final", "initial", "fraction", "census", "level", "metadata", "fields"],
)
def test_a_total_over_every_row_put_back_is_refused(mutate, message) -> None:
    """The fourth review of #1305: the contract recomputes the restated totals.

    Each put-back total or figure passed the shape checks; less the published
    rows, the final loss gave the withheld row's contribution back.
    """

    diagnostics = _withheld_diagnostics()
    failures: list[str] = []
    dc._check_local_area_calibration_diagnostics(diagnostics, failures)
    assert failures == []
    mutate(diagnostics)
    dc._check_local_area_calibration_diagnostics(diagnostics, failures)
    assert any(message in failure for failure in failures), failures


def test_the_candidate_manifest_restates_the_diagnostics_totals() -> None:
    diagnostics = _withheld_diagnostics()
    census = diagnostics["past_cap_census"]
    candidate = {
        "solve": {
            "model_figures_withheld": {"targets": 1, "totals_basis": "published_rows"},
            "initial_loss": diagnostics["initial_loss"],
            "final_loss": diagnostics["final_loss"],
            "past_cap": {
                "n_targets": census["n_targets"],
                "past_at_init": census["initial_past_cap"],
                "past_at_final": census["final_past_cap"],
                "escaped": census["escaped"],
                "frozen": census["frozen"],
                "pushed_out": census["pushed_out"],
            },
        }
    }
    failures: list[str] = []
    dc._check_uk_candidate_restated_losses(candidate, diagnostics, failures)
    assert failures == []
    candidate["solve"]["final_loss"] = 0.125
    candidate["solve"]["past_cap"]["n_targets"] = 2
    candidate["solve"].pop("model_figures_withheld")
    dc._check_uk_candidate_restated_losses(candidate, diagnostics, failures)
    assert len(failures) == 3, failures


def test_a_brma_row_needs_no_carrier_verdict() -> None:
    """María's ruling of 2026-10-26 dropped the withholding and its BRMA
    carrier verdict; an earlier release's verdicts still read, unchecked."""

    unflagged = _withheld_diagnostics()
    del unflagged["targets"][1]["metadata"]["carrier_floor_met"]
    failures: list[str] = []
    dc._check_local_area_calibration_diagnostics(unflagged, failures)
    assert failures == []
    assert not hasattr(dc, "_uk_brma_floor_verdict_warnings")


def test_the_dense_validator_runs_the_withheld_checks(tmp_path, monkeypatch) -> None:
    """The candidate cross-check still runs on a release, for the releases
    built while rows were withheld."""

    seen: list[str] = []

    def restated(candidate, diagnostics, failures) -> None:
        seen.append("restated")
        assert "solve" in candidate and "targets" in diagnostics

    monkeypatch.setattr(dc, "_check_uk_candidate_restated_losses", restated)
    validate_release_dir(write_dense_bundle(tmp_path))
    assert seen == ["restated"]


def test_the_withheld_block_must_count_the_withheld_rows() -> None:
    diagnostics = _withheld_diagnostics()
    diagnostics["uk_diagnostics"] = {}
    failures: list[str] = []
    dc._check_local_area_calibration_diagnostics(diagnostics, failures)
    assert any("counts 0 withheld rows" in failure for failure in failures)


def test_valid_dense_bundle_passes(tmp_path: Path) -> None:
    validate_release_dir(write_dense_bundle(tmp_path))


def test_dense_bundle_rejects_stale_ledger_identity_key(tmp_path: Path) -> None:
    release_dir = write_dense_bundle(tmp_path)
    candidate_path = release_dir / "rowwise_candidate_manifest.json"
    candidate = json.loads(candidate_path.read_text())
    candidate["identity"]["ledger"] = candidate["identity"].pop("targets")["chronicle"]
    candidate_path.write_text(json.dumps(candidate))
    assert "identity.targets.chronicle" in _failures(release_dir)


def test_dense_bundle_accepts_the_release_id_as_revision(tmp_path: Path) -> None:
    validate_release_dir(write_dense_bundle(tmp_path, revision=DENSE_ID))


def _failures(release_dir: Path) -> str:
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(release_dir)
    return str(excinfo.value)


def test_unpinned_revision_is_rejected(tmp_path: Path) -> None:
    assert "not pinned" in _failures(write_dense_bundle(tmp_path, revision="main"))


def test_wrong_namespace_is_rejected(tmp_path: Path) -> None:
    assert "namespace" in _failures(write_dense_bundle(tmp_path, namespace="uk_local"))


def test_default_slot_claim_is_rejected(tmp_path: Path) -> None:
    text = _failures(write_dense_bundle(tmp_path, default_datasets={"national": "x"}))
    assert "default_datasets" in text


def test_failing_release_blocking_gate_is_rejected(tmp_path: Path) -> None:
    text = _failures(
        write_dense_bundle(tmp_path, report=_signed_report(blocking_status="failed"))
    )
    assert "did not pass" in text and "shippable" in text


def test_dev_posture_report_is_rejected(tmp_path: Path) -> None:
    text = _failures(
        write_dense_bundle(tmp_path, report=_signed_report(release_candidate=False))
    )
    assert "release_candidate must be true" in text


def test_tampered_signature_is_rejected(tmp_path: Path) -> None:
    report = _signed_report()
    report["attestation"]["signature"] = "0" * 64
    assert "signature does not authenticate" in _failures(
        write_dense_bundle(tmp_path, report=report)
    )


def test_report_edited_after_signing_is_rejected(tmp_path: Path) -> None:
    report = _signed_report()
    report["gates"]["uk_local_target_fit"]["status"] = "failed"
    text = _failures(write_dense_bundle(tmp_path, report=report))
    assert "gate_outcomes_sha256" in text


def test_foreign_entry_ids_are_rejected(tmp_path: Path) -> None:
    report = _signed_report()
    report["gates"]["uk_terminal_something"] = report["gates"].pop(
        "uk_local_weight_ess"
    )
    assert "exactly the local battery entries" in _failures(
        write_dense_bundle(tmp_path, report=report)
    )


def test_missing_coverage_object_is_rejected(tmp_path: Path) -> None:
    assert "coverage object 'incumbent'" in _failures(
        write_dense_bundle(tmp_path, drop_coverage_key="incumbent")
    )


def test_diagnostics_without_households_are_rejected(tmp_path: Path) -> None:
    assert "households" in _failures(write_dense_bundle(tmp_path, drop_households=True))


def test_missing_key_env_names_the_hole(tmp_path: Path, monkeypatch) -> None:
    release_dir = write_dense_bundle(tmp_path)
    monkeypatch.delenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY")
    assert "MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY" in _failures(release_dir)


def test_other_local_area_ids_keep_the_generic_contract(tmp_path: Path) -> None:
    # A non-dense local-area id never reaches the dense validator.
    release_dir = write_dense_bundle(
        tmp_path,
        release_id="populace-uk-2025-frs-k9",
        revision="populace-uk-2025-frs-k9",
    )
    text = _failures(release_dir)
    assert "us_source_coverage.json" in text


def test_report_from_another_attempt_is_rejected(tmp_path: Path) -> None:
    release_dir = write_dense_bundle(
        tmp_path, attempt_id="uk-local-candidate-f100-s42-20260101T000000Z-00000000"
    )
    assert "does not belong to this run" in _failures(release_dir)


def test_build_manifest_without_attempt_id_is_rejected(tmp_path: Path) -> None:
    release_dir = write_dense_bundle(tmp_path)
    build_manifest = json.loads((release_dir / "build_manifest.json").read_text())
    del build_manifest["attempt_id"]
    (release_dir / "build_manifest.json").write_text(
        json.dumps(build_manifest, indent=1)
    )
    assert "must carry the calibration 'attempt_id'" in _failures(release_dir)


@pytest.mark.parametrize("dirty", [True, None, "false", 0, "missing"])
def test_dense_contract_requires_measured_clean_code(tmp_path, dirty):
    release_dir = write_dense_bundle(tmp_path)
    path = release_dir / "build_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["code"] = {} if dirty == "missing" else {"git_dirty": dirty}
    path.write_text(json.dumps(manifest))
    assert "code.git_dirty" in _failures(release_dir)


@pytest.mark.parametrize(
    "when,passes",
    [
        ("2026-09-02", False),
        ("2026-09-03", True),
        ("2026-10-03", True),
        ("2026-10-04", False),
    ],
)
def test_measure_exclusion_time_window_is_checked_at_validation(when, passes):
    from datetime import date

    record = {
        "name": "example",
        "reason": "measured gap",
        "tracking": "microcosm#869",
        "approved_by": "reviewer",
        "adjudication": "review decision",
        "approved_on": "2026-09-03",
        "expires_on": "2026-10-03",
    }
    failures = []
    dc._check_uk_measure_exclusions(
        {"example": record}, failures, today=date.fromisoformat(when)
    )
    assert (not failures) is passes


@pytest.mark.parametrize(
    "field,value",
    [
        ("expires_on", None),
        ("expires_on", ""),
        ("expires_on", "2026-99-03"),
        ("approved_on", None),
        ("approved_on", "20260903"),
        ("approved_by", ""),
    ],
)
def test_measure_exclusion_requires_full_approval_provenance(field, value):
    from datetime import date

    record = {
        "reason": "gap",
        "tracking": "microcosm#869",
        "approved_by": "reviewer",
        "adjudication": "decision",
        "approved_on": "2026-09-03",
        "expires_on": "2026-10-03",
    }
    record[field] = value
    failures = []
    dc._check_uk_measure_exclusions(
        {"example": record}, failures, today=date(2026, 9, 7)
    )
    assert any(field in failure for failure in failures)


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
        ("missing_file", "all together or not at all"),
        ("missing_row", "complete nonempty row list"),
        ("renamed_row", "pinned incumbent target inventory"),
        ("nonfinite_candidate", "not valid JSON"),
        ("changed_identity", "identity.candidate_dataset_sha256"),
        ("changed_summary", "summary does not match"),
    ],
)
def test_dense_release_refuses_bad_incumbent_surface_evidence(
    tmp_path, mutation, expected
):
    """An attached comparison's identities refuse (microcosm#1301)."""
    release_dir = write_dense_bundle(tmp_path, with_incumbent=True)
    path = release_dir / "incumbent_surface_evaluation.json"
    if mutation == "missing_file":
        path.unlink()
    else:
        payload = json.loads(path.read_text())
        if mutation == "missing_row":
            payload["national_rows"] = []
        elif mutation == "renamed_row":
            payload["local_rows"][0]["incumbent_name"] = "another_local"
            payload["summary"] = dc.uk_incumbent_surface_assessment(payload)
        elif mutation == "nonfinite_candidate":
            payload["national_rows"][0]["candidate_estimate"] = float("nan")
        elif mutation == "changed_identity":
            payload["identity"]["candidate_dataset_sha256"] = "9" * 64
        elif mutation == "changed_summary":
            payload["summary"]["grains"]["local"]["measured"] = 100
        path.write_text(json.dumps(payload))
    assert expected in _failures(release_dir)


def _record_incumbent_limitation(release_dir: Path) -> dict:
    """Re-derive the comparison's limitation as the assembler would."""
    path = release_dir / "incumbent_surface_evaluation.json"
    payload = json.loads(path.read_text())
    record = dc._check_uk_incumbent_surface_evaluation(
        payload, [], expected_identity=payload["identity"], enforce_limits=False
    )
    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["reviewed_limitations"]:
        if item["id"] == "incumbent_surface_diagnostic":
            item["passed"] = record["passed"]
            item["failure_count"] = len(record["limit_failures"])
    manifest_path.write_text(json.dumps(manifest))
    _refresh_packaging_checksums(release_dir)
    return record


@pytest.mark.parametrize(
    "mutation,failure",
    [
        ("missing_comparator", "incumbent_estimate"),
        ("deferred_miss", "absolute relative error"),
        ("two_blocks", "single-block engine measurement"),
    ],
)
def test_an_attached_comparison_records_its_limit_misses(tmp_path, mutation, failure):
    """The measured limits are the diagnostic's verdict, recorded not enforced.

    A failed assessment ships only when the manifest says so: the
    limitation must carry the recomputed verdict (microcosm#1301).
    """
    release_dir = write_dense_bundle(tmp_path, with_incumbent=True)
    path = release_dir / "incumbent_surface_evaluation.json"
    payload = json.loads(path.read_text())
    if mutation == "missing_comparator":
        payload["local_rows"][0]["incumbent_estimate"] = None
    elif mutation == "deferred_miss":
        payload["local_rows"][0]["candidate_estimate"] = 1.8
    else:
        payload["measure_resolution"]["blocks"] = 2
    payload["summary"] = dc.uk_incumbent_surface_assessment(payload)
    path.write_text(json.dumps(payload))
    _refresh_packaging_checksums(release_dir)
    # The manifest still claims a passing comparison: refused.
    assert "recomputes (False, 1)" in _failures(release_dir)
    record = _record_incumbent_limitation(release_dir)
    assert record["passed"] is False
    assert len(record["limit_failures"]) == 1
    assert failure in record["limit_failures"][0]
    validate_release_dir(release_dir)


def test_the_core_bundle_claims_neither_holdout_nor_incumbent(tmp_path) -> None:
    release_dir = write_dense_bundle(tmp_path)
    for name in dc._UK_DENSE_INCUMBENT_FILES:
        assert not (release_dir / name).exists()
    coverage = json.loads((release_dir / "uk_source_coverage.json").read_text())
    assert coverage["holdout"] == {"run": False, "claim": "none"}
    assert coverage["incumbent"] == {"compared": False}
    validate_release_dir(release_dir)


@pytest.mark.parametrize(
    "options",
    [
        {"with_incumbent": True},
        {"holdout": "measured"},
        {"sized": True},
        {"with_incumbent": True, "holdout": "measured", "sized": True},
    ],
)
def test_opt_in_diagnostics_validate_with_their_limitations(tmp_path, options):
    validate_release_dir(write_dense_bundle(tmp_path, **options))


@pytest.mark.parametrize("name", dc._UK_DENSE_INCUMBENT_FILES)
def test_incumbent_files_ship_all_together_or_not_at_all(tmp_path, name) -> None:
    release_dir = write_dense_bundle(tmp_path, with_incumbent=True)
    (release_dir / name).unlink()
    assert "all together or not at all" in _failures(release_dir)


def test_declared_incumbent_files_cannot_be_removed(tmp_path) -> None:
    """Deleting all three files does not detach a comparison the manifest
    declares: the coverage and the limitation still say it was attached."""
    release_dir = write_dense_bundle(tmp_path, with_incumbent=True)
    for name in dc._UK_DENSE_INCUMBENT_FILES:
        (release_dir / name).unlink()
    text = _failures(release_dir)
    assert "all together or not at all" in text
    assert "release_manifest.json must declare" not in text


def test_a_stray_incumbent_file_attaches_the_comparison(tmp_path) -> None:
    release_dir = write_dense_bundle(tmp_path)
    (release_dir / "score_vs_incumbent.json").write_text("{}")
    text = _failures(release_dir)
    assert "all together or not at all" in text
    assert "incumbent.compared is False" in text


@pytest.mark.parametrize(
    "tamper,expected",
    [
        ("shipped_diagnostics", "changed original evaluated"),
        ("gate_report_hash", "gate report does not match released attempt"),
        ("candidate_git_dirty", "identity.code.git_dirty must be false"),
        ("candidate_dataset_hash", "dataset hash does not match released evidence"),
        ("candidate_ledger", "Ledger facts_sha256"),
    ],
)
def test_core_joins_refuse_without_any_incumbent_file(tmp_path, tamper, expected):
    """The key guard of the split (microcosm#1301): with no incumbent file
    attached, the candidate's own evidence still joins the released bytes."""
    release_dir = write_dense_bundle(tmp_path)
    assert not any(
        (release_dir / name).exists() for name in dc._UK_DENSE_INCUMBENT_FILES
    )
    candidate_path = release_dir / "rowwise_candidate_manifest.json"
    candidate = json.loads(candidate_path.read_text())
    if tamper == "shipped_diagnostics":
        path = release_dir / "calibration_diagnostics.json"
        diagnostics = json.loads(path.read_text())
        diagnostics["targets"][0]["final_estimate"] = 11.0
        path.write_text(json.dumps(diagnostics))
    elif tamper == "gate_report_hash":
        candidate["outputs"]["local_gate_report"]["sha256"] = "9" * 64
    elif tamper == "candidate_git_dirty":
        candidate["identity"]["code"]["git_dirty"] = True
    elif tamper == "candidate_dataset_hash":
        candidate["outputs"]["dataset"]["sha256"] = "9" * 64
    else:
        candidate["identity"]["targets"]["chronicle"]["facts_sha256"] = "9" * 64
    if tamper != "shipped_diagnostics":
        candidate_path.write_text(json.dumps(candidate))
    # Unsigned packaging is refreshed, so only the join can catch the tamper.
    _refresh_packaging_checksums(release_dir)
    assert expected in _failures(release_dir)


def _rewrite_coverage(release_dir: Path, mutate) -> None:
    path = release_dir / "uk_source_coverage.json"
    coverage = json.loads(path.read_text())
    mutate(coverage)
    path.write_text(json.dumps(coverage))
    _refresh_packaging_checksums(release_dir)


@pytest.mark.parametrize(
    "options,mutate,expected",
    [
        (
            {},
            lambda c: c["holdout"].__setitem__("claim", "out_of_sample"),
            "holdout.claim must be 'none' or 'report_only'",
        ),
        (
            {},
            lambda c: c["holdout"].__setitem__("run", True),
            "holdout.run must be false",
        ),
        (
            {"holdout": "measured"},
            lambda c: c["holdout"].pop("mean_holdout_loss"),
            "holdout.mean_holdout_loss must be a finite number",
        ),
        (
            {},
            lambda c: c["incumbent"].__setitem__("compared", True),
            "incumbent.compared is True",
        ),
        (
            {"with_incumbent": True},
            lambda c: c["incumbent"].__setitem__("compared", False),
            "incumbent.compared is False",
        ),
        (
            {"with_incumbent": True},
            lambda c: c["incumbent"].pop("snapshot"),
            "private-repo snapshot",
        ),
        (
            {},
            lambda c: c["incumbent"].pop("compared"),
            "incumbent.compared must be true or false",
        ),
        (
            {},
            lambda c: c["size"].__setitem__("sized", "yes"),
            "size.sized must be true or false",
        ),
        (
            {"sized": True},
            lambda c: c["size"].__setitem__("dataset_households", None),
            "size.dataset_households must be a positive integer",
        ),
        ({}, lambda c: c.pop("size"), "coverage object 'size'"),
    ],
)
def test_coverage_discriminators_are_validated(tmp_path, options, mutate, expected):
    release_dir = write_dense_bundle(tmp_path, **options)
    _rewrite_coverage(release_dir, mutate)
    assert expected in _failures(release_dir)


def _rewrite_limitations(release_dir: Path, mutate) -> None:
    path = release_dir / "release_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["reviewed_limitations"] = mutate(manifest["reviewed_limitations"])
    path.write_text(json.dumps(manifest))
    _refresh_packaging_checksums(release_dir)


def _without(identifier: str):
    return lambda items: [item for item in items if item["id"] != identifier]


@pytest.mark.parametrize(
    "options,mutate,expected",
    [
        ({}, _without("engine_measurement"), "must carry 'engine_measurement'"),
        ({}, _without("no_out_of_sample_claim"), "must carry 'no_out_of_sample_claim'"),
        (
            {},
            lambda items: [*items, {"id": "rotated_holdout", "claim": "report_only"}],
            "carries 'rotated_holdout'",
        ),
        (
            {"holdout": "measured"},
            lambda items: [
                {**item, "claim": "out_of_sample"}
                if item["id"] == "rotated_holdout"
                else item
                for item in items
            ],
            "must say claim 'report_only'",
        ),
        (
            {},
            _without("no_incumbent_comparison"),
            "must carry 'no_incumbent_comparison'",
        ),
        (
            {"with_incumbent": True},
            _without("incumbent_surface_diagnostic"),
            "must carry 'incumbent_surface_diagnostic'",
        ),
        (
            {"with_incumbent": True},
            lambda items: [*items, {"id": "no_incumbent_comparison"}],
            "carries 'no_incumbent_comparison'",
        ),
        (
            {"with_incumbent": True},
            lambda items: [
                {**item, "failure_count": 3}
                if item["id"] == "incumbent_surface_diagnostic"
                else item
                for item in items
            ],
            "records passed/failure_count",
        ),
        ({"sized": True}, _without("size_doctrine"), "must carry 'size_doctrine'"),
        (
            {},
            lambda items: [*items, {"id": "size_doctrine"}],
            "'size_doctrine' for an unsized release",
        ),
    ],
)
def test_limitations_must_match_what_ran(tmp_path, options, mutate, expected):
    release_dir = write_dense_bundle(tmp_path, **options)
    _rewrite_limitations(release_dir, mutate)
    assert expected in _failures(release_dir)


#: A local cell and an area the signed gates excused (synthetic records).
_SIGNED_CELL = "synthetic_family/E06000053/synthetic_metric"
_SIGNED_AREA = "local_authority/E06000053"
_NATIONAL_EXCLUSION = "synthetic.target_fit_row@2025"


def _signed_bundle(tmp_path: Path, **options) -> Path:
    """A dense bundle whose gates stood on a reviewed local cell, a reviewed
    area and a reviewed national row (microcosm#1301)."""
    report = _resigned(
        _resigned(
            _signed_report(),
            "uk_local_target_fit",
            {"reviewed_exclusions": {_SIGNED_CELL: {"reason": "synthetic"}}},
        ),
        "uk_local_area_support",
        {"reviewed_exclusions": {_SIGNED_AREA: {"record": {"reason": "synthetic"}}}},
    )
    return write_dense_bundle(
        tmp_path,
        report=report,
        national_exclusions=(_NATIONAL_EXCLUSION,),
        **options,
    )


def test_a_bundle_naming_every_signed_exclusion_validates(tmp_path) -> None:
    release_dir = _signed_bundle(tmp_path)
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    ids = {item["id"] for item in manifest["reviewed_limitations"]}
    assert {
        f"reviewed_exclusion:uk_local_target_fit:{_SIGNED_CELL}",
        f"reviewed_exclusion:uk_target_fit:{_NATIONAL_EXCLUSION}",
        f"area_support_exclusion:{_SIGNED_AREA}",
    } <= ids
    assert manifest["area_support"]["below_floor"] == {"local_authority": ["E06000053"]}
    validate_release_dir(release_dir)


def _rewrite_manifest(release_dir: Path, mutate) -> None:
    path = release_dir / "release_manifest.json"
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_text(json.dumps(manifest))
    _refresh_packaging_checksums(release_dir)


def _drop_limitation(identifier: str):
    def mutate(manifest: dict) -> None:
        manifest["reviewed_limitations"] = _without(identifier)(
            manifest["reviewed_limitations"]
        )

    return mutate


def _drop_coverage_name(gate_id: str, name: str):
    def mutate(coverage: dict) -> None:
        names = coverage["signed_deferrals"]["reviewed_exclusions"][gate_id]
        names.remove(name)

    return mutate


def _add_coverage_name(gate_id: str, name: str):
    def mutate(coverage: dict) -> None:
        coverage["signed_deferrals"]["reviewed_exclusions"][gate_id].append(name)

    return mutate


@pytest.mark.parametrize(
    "manifest_edit,coverage_edit,expected",
    [
        (
            _drop_limitation(f"reviewed_exclusion:uk_local_target_fit:{_SIGNED_CELL}"),
            None,
            "must name each reviewed exclusion the gates stood on; 1 missing",
        ),
        (
            _drop_limitation(f"reviewed_exclusion:uk_target_fit:{_NATIONAL_EXCLUSION}"),
            None,
            "must name each reviewed exclusion the gates stood on; 1 missing",
        ),
        (
            lambda m: m["reviewed_limitations"].append(
                {"id": "reviewed_exclusion:uk_input_mass_parity:synthetic_column"}
            ),
            None,
            "exclusion(s) the gates did not stand on",
        ),
        (
            # The coverage and the manifest both drop the signed cell: the
            # signed report still names it.
            _drop_limitation(f"reviewed_exclusion:uk_local_target_fit:{_SIGNED_CELL}"),
            _drop_coverage_name("uk_local_target_fit", _SIGNED_CELL),
            "does not match the reviewed exclusions the signed uk_local_gates.json",
        ),
        (
            # The coverage and the manifest both drop the national row: the
            # candidate manifest's record still names it.
            _drop_limitation(f"reviewed_exclusion:uk_target_fit:{_NATIONAL_EXCLUSION}"),
            _drop_coverage_name("uk_target_fit", _NATIONAL_EXCLUSION),
            "reviewed_exclusions.uk_target_fit does not match the reviewed "
            "exclusions the evaluated candidate recorded",
        ),
        (
            # Both add an input-mass name the candidate's gates never stood on.
            lambda m: m["reviewed_limitations"].append(
                {"id": "reviewed_exclusion:uk_input_mass_parity:synthetic_column"}
            ),
            _add_coverage_name("uk_input_mass_parity", "synthetic_column"),
            "reviewed_exclusions.uk_input_mass_parity does not match the reviewed "
            "exclusions the evaluated candidate recorded",
        ),
        (
            None,
            lambda c: c["signed_deferrals"].pop("reviewed_exclusions"),
            "signed_deferrals.reviewed_exclusions must map each of",
        ),
        (
            lambda m: m.pop("area_support"),
            None,
            "must carry the 'area_support' block for a dense release",
        ),
        (
            # The block and the limitation both drop the excused area: the
            # signed report still names it.
            lambda m: (
                m["area_support"].__setitem__("below_floor", {}),
                _drop_limitation(f"area_support_exclusion:{_SIGNED_AREA}")(m),
            ),
            None,
            "must list exactly the areas whose reviewed exclusions",
        ),
        (
            lambda m: m["reviewed_limitations"].append(
                {"id": "area_support_exclusion:local_authority/E09000001"}
            ),
            None,
            "for areas not listed below the floor",
        ),
        (
            _drop_limitation(f"area_support_exclusion:{_SIGNED_AREA}"),
            None,
            "without a reviewed area_support_exclusion limitation",
        ),
    ],
)
def test_every_signed_exclusion_is_named_in_the_manifest(
    tmp_path, manifest_edit, coverage_edit, expected
):
    """After microcosm#1301 local cells and areas are signed at the gate only,
    so the manifest's limitations are where a consumer learns of them: every
    gate's names are checked against the candidate manifest's record, the
    local names against the signed report too."""
    release_dir = _signed_bundle(tmp_path)
    if coverage_edit is not None:
        _rewrite_coverage(release_dir, coverage_edit)
    if manifest_edit is not None:
        _rewrite_manifest(release_dir, manifest_edit)
    assert expected in _failures(release_dir)


@pytest.mark.parametrize(
    "record",
    [
        "missing",
        {"uk_target_fit": [_NATIONAL_EXCLUSION]},
        {
            "uk_target_fit": _NATIONAL_EXCLUSION,
            "uk_local_target_fit": [],
            "uk_input_mass_parity": [],
        },
    ],
    ids=["missing", "partial", "malformed"],
)
def test_a_candidate_must_record_the_exclusions_its_gates_stood_on(
    tmp_path, record
) -> None:
    """Without the build's record nothing shipped names the national
    exclusions the gates stood on, so the release refuses (microcosm#1301)."""

    release_dir = _signed_bundle(tmp_path)
    path = release_dir / "rowwise_candidate_manifest.json"
    candidate = json.loads(path.read_text())
    if record == "missing":
        del candidate["reviewed_exclusions_stood_on"]
    else:
        candidate["reviewed_exclusions_stood_on"] = record
    path.write_text(json.dumps(candidate))
    _refresh_packaging_checksums(release_dir)
    assert "must record, under 'reviewed_exclusions_stood_on'" in _failures(release_dir)


def _set_holdout_claim(claim: str, limitation: dict, **fields):
    """Rewrite the coverage's holdout claim and the matching limitation."""

    def coverage_edit(coverage: dict) -> None:
        coverage["holdout"] = {
            "run": claim == "report_only",
            "claim": claim,
            **fields,
        }

    def manifest_edit(manifest: dict) -> None:
        manifest["reviewed_limitations"] = [
            item
            for item in manifest["reviewed_limitations"]
            if item["id"] not in ("rotated_holdout", "no_out_of_sample_claim")
        ] + [limitation]

    return coverage_edit, manifest_edit


@pytest.mark.parametrize(
    "options,edits,expected",
    [
        (
            {},
            _set_holdout_claim(
                "report_only",
                {"id": "rotated_holdout", "claim": "report_only"},
                mean_holdout_loss=0.1,
                n_folds=5,
            ),
            "holdout is reported, but the evaluated candidate skipped",
        ),
        (
            {"holdout": "measured"},
            _set_holdout_claim("none", {"id": "no_out_of_sample_claim"}),
            "holdout claims nothing, but the evaluated candidate did not skip",
        ),
        (
            {"holdout": "measured"},
            (lambda c: c["holdout"].__setitem__("mean_holdout_loss", 0.01), None),
            "holdout.mean_holdout_loss does not match",
        ),
        (
            {"holdout": "measured"},
            (lambda c: c["holdout"].__setitem__("n_folds", 4), None),
            "holdout.n_folds does not match",
        ),
        (
            {"sized": True},
            (lambda c: c["size"].__setitem__("dataset_households", 5), None),
            "size ['dataset_households'] do not match the evaluated candidate",
        ),
        (
            {"sized": True},
            (lambda c: c["size"].__setitem__("clone_count", 3), None),
            "size ['clone_count'] do not match the evaluated candidate",
        ),
        (
            {},
            (
                lambda c: c["size"].update(sized=True, dataset_households=4),
                lambda m: m["reviewed_limitations"].append({"id": "size_doctrine"}),
            ),
            "size.sized is True, but the evaluated candidate requested no",
        ),
        (
            {},
            (
                None,
                lambda m: m["reviewed_limitations"][0].__setitem__("blocks", 2),
            ),
            "limitation 'engine_measurement' ['blocks'] do not match",
        ),
        (
            {"sized": True},
            (
                None,
                lambda m: [
                    item.__setitem__("dataset_households", 5)
                    for item in m["reviewed_limitations"]
                    if item["id"] == "size_doctrine"
                ],
            ),
            "limitation 'size_doctrine' ['dataset_households'] do not match",
        ),
    ],
)
def test_holdout_and_size_claims_restate_the_evaluated_candidate(
    tmp_path, options, edits, expected
):
    """The unsigned coverage cannot switch the release between
    no_out_of_sample_claim and rotated_holdout, invent a holdout loss or
    resize it: each claim is joined to the candidate manifest (microcosm#1301)."""
    release_dir = write_dense_bundle(tmp_path, **options)
    coverage_edit, manifest_edit = edits
    if coverage_edit is not None:
        _rewrite_coverage(release_dir, coverage_edit)
    if manifest_edit is not None:
        _rewrite_manifest(release_dir, manifest_edit)
    assert expected in _failures(release_dir)


def test_a_sized_candidate_must_realize_its_requested_size(tmp_path) -> None:
    release_dir = write_dense_bundle(tmp_path, sized=True)
    path = release_dir / "rowwise_candidate_manifest.json"
    candidate = json.loads(path.read_text())
    candidate["solve"]["dataset_size"]["realized_households"] = 3
    path.write_text(json.dumps(candidate))
    _refresh_packaging_checksums(release_dir)
    assert "did not realize the dataset size the candidate requested" in _failures(
        release_dir
    )


def test_incumbent_surface_limits_keep_exact_existing_boundaries():
    payload = _surface_rows()
    payload["national_rows"][0]["candidate_estimate"] = 125.0
    result = dc.uk_incumbent_surface_assessment(payload)
    # At 25% the row limit passes. The within-10 share remains diagnostic.
    assert not any(
        "absolute relative error" in failure for failure in result["failures"]
    )
    assert result["passed"] is True
    payload["national_rows"][0]["candidate_estimate"] = 125.0001
    assert any(
        "absolute relative error" in failure
        for failure in dc.uk_incumbent_surface_assessment(payload)["failures"]
    )
    payload["national_rows"][0]["candidate_estimate"] = 110.0
    assert dc.uk_incumbent_surface_assessment(payload)["passed"] is True


@pytest.mark.parametrize(
    "when,passes",
    [
        ("2026-09-02", False),
        ("2026-09-03", True),
        ("2026-10-03", True),
        ("2026-10-04", False),
    ],
)
def test_dense_directory_rechecks_exclusion_dates_on_each_validation(
    tmp_path, monkeypatch, when, passes
):
    from datetime import date

    release_dir = write_dense_bundle(tmp_path)

    class ValidationDate(date):
        @classmethod
        def today(cls):
            return date.fromisoformat(when)

    monkeypatch.setattr(dc, "date", ValidationDate)
    if passes:
        validate_release_dir(release_dir)
    else:
        assert "measure exclusion" in _failures(release_dir)


@pytest.mark.parametrize(
    "name,mutate,expected",
    [
        ("uk_source_coverage.json", "ledger", "Ledger facts_sha256"),
        ("uk_source_coverage.json", "incumbent", "incumbent snapshot"),
        ("calibration_diagnostics.json", "diagnostics", "changed original evaluated"),
        ("score_vs_incumbent.json", "score", "candidate_diagnostics does not match"),
    ],
)
def test_dense_release_joins_every_receipt_to_the_evaluated_sources(
    tmp_path, name, mutate, expected
):
    release_dir = write_dense_bundle(tmp_path, with_incumbent=True)
    path = release_dir / name
    data = json.loads(path.read_text())
    if mutate == "ledger":
        data["ledger_artifact"]["facts_sha256"] = "9" * 64
    elif mutate == "incumbent":
        data["incumbent"]["snapshot"] = {"other": "snapshot"}
    elif mutate == "diagnostics":
        data["targets"][0]["final_estimate"] = 11.0
    else:
        data["artifacts"]["candidate_diagnostics"]["sha256"] = "9" * 64
    path.write_text(json.dumps(data))
    assert expected in _failures(release_dir)


def _refresh_packaging_checksums(release_dir):
    """Unsigned packaging edits cannot replace original approval provenance."""
    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for artifact in manifest["artifacts"].values():
        local = release_dir / artifact["path"]
        if local.is_file():
            artifact["sha256"] = _sha(local.read_bytes())
    manifest_path.write_text(json.dumps(manifest))
    sums_path = release_dir / "sha256sums.txt"
    sums = {}
    for line in sums_path.read_text().splitlines():
        digest, name = line.split("  ", 1)
        local = release_dir / name
        sums[name] = _sha(local.read_bytes()) if local.is_file() else digest
    sums_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items()))
    )


@pytest.mark.parametrize("mutation", ["extend", "delete"])
def test_refreshing_unsigned_coverage_cannot_renew_original_approvals(
    tmp_path, monkeypatch, mutation
):
    from datetime import date

    release_dir = write_dense_bundle(tmp_path)

    class AfterExpiry(date):
        @classmethod
        def today(cls):
            return cls(2026, 10, 4)

    monkeypatch.setattr(dc, "date", AfterExpiry)
    path = release_dir / "uk_source_coverage.json"
    coverage = json.loads(path.read_text())
    if mutation == "extend":
        coverage["measure_exclusions"]["obr.housing_benefit"]["expires_on"] = (
            "2099-01-01"
        )
    else:
        coverage["measure_exclusions"] = {}
    path.write_text(json.dumps(coverage))
    _refresh_packaging_checksums(release_dir)
    failures = _failures(release_dir)
    assert "original candidate approvals" in failures
    assert "expired 2026-10-03" in failures


#: Placeholder area codes no area carries, so a fixture that pairs an area
#: with support figures never reads as a real area's figures.
_AREA = "E06999999"
_OTHER_AREA = "E06999998"
#: The synthetic support figures the walk tests inject; no finding may
#: repeat one.
_FIGURES = ("37", "23.75", "29", "41")


#: The area-support details a signed report may ship: the floors an excused
#: area misses and its signed record, and area counts by level.
_FLOOR_ONLY_DETAILS = {
    "floors": {
        "minimum_rows": 50,
        "minimum_effective_sample_size": 50.0,
        "minimum_distinct_sources": 50,
    },
    "by_geography_level": {
        "local_authority": {"areas_checked": 2, "areas_below_floor": 1}
    },
    "reviewed_exclusions": {
        f"local_authority/{_AREA}": {
            "shortfalls": ["rows < 50", "ESS < 50"],
            "record": {"reason": "synthetic reviewed exclusion"},
        }
    },
}


def test_floor_only_area_support_details_ship(tmp_path: Path) -> None:
    assert (
        validate_release_dir(
            write_dense_bundle(
                tmp_path,
                report=_signed_report(area_support_details=_FLOOR_ONLY_DETAILS),
            )
        )
        == []
    )


def _with_figures(details: dict) -> dict:
    figures = json.loads(json.dumps(details))
    figures["reviewed_exclusions"][f"local_authority/{_AREA}"].update(
        rows=37, ess=23.75, sources=29
    )
    return figures


@pytest.mark.parametrize("inject", ["summary_shortfall_text", "signed_report_figures"])
def test_a_dense_release_ships_per_area_support_figures(
    tmp_path: Path, inject: str
) -> None:
    """María's ruling of 2026-10-26 dropped the withholding of per-area
    support figures (microcosm#1301): a shipped document carrying an area's
    rows, ESS or sources draws no warning."""

    if inject == "signed_report_figures":
        release_dir = write_dense_bundle(
            tmp_path,
            report=_signed_report(
                area_support_details=_with_figures(_FLOOR_ONLY_DETAILS)
            ),
        )
    else:
        release_dir = write_dense_bundle(tmp_path)
        path = release_dir / "gate_summary.json"
        summary = json.loads(path.read_text())
        summary["diagnostic_gates"]["uk_local_area_support"] = {
            "failures": [f"local_authority/{_AREA}: rows 37 < 50"]
        }
        path.write_text(json.dumps(summary))
        _refresh_packaging_checksums(release_dir)

    assert validate_release_dir(release_dir) == []


_FIGURE = "per-area support figure(s)"
_EXTREME = "minimum(s), maximum(s) or median(s) of an area support figure"

#: The shapes an area's support figures could take in a shipped document
#: (microcosm#1301), each injected into the release manifest under one key.
_SUPPORT_FIGURE_SHAPES = {
    "record_by_area_code": ("areas", [{"area_code": _AREA, "sources": 29}], _FIGURE),
    "record_by_geography_id": (
        "areas",
        [{"geography_id": _AREA, "rows": 37}],
        _FIGURE,
    ),
    "record_by_code": ("areas", [{"code": _AREA, "ess": 23.75}], _FIGURE),
    "list_in_a_record": ("areas", [{"area_code": _AREA, "rows": [37]}], _FIGURE),
    "figure_as_text": (
        "areas",
        [{"area_code": _AREA, "nonzero_households": "37"}],
        _FIGURE,
    ),
    "code_keyed_map": ("support_by_area", {"ess": {_AREA: 23.75}}, _FIGURE),
    "code_keyed_map_by_level": (
        "support_by_area",
        {"sources": {"local_authority": {_AREA: 29}}},
        _FIGURE,
    ),
    "level_prefixed_code_keys": (
        "support_by_area",
        {"rows": {f"local_authority/{_AREA}": [37]}},
        _FIGURE,
    ),
    "parallel_lists": (
        "support_table",
        {
            "area_code": [_AREA, _OTHER_AREA],
            "nonzero_source_households": [29, 41],
        },
        _FIGURE,
    ),
    "pandas_columns": (
        "support_table",
        {"geography_id": {"0": _AREA}, "effective_sample_size": {"0": 23.75}},
        _FIGURE,
    ),
    "code_column": ("support_table", {"code": [_AREA], "rows": [37]}, _FIGURE),
    "minimum": ("support", {"min_nonzero_households": 37}, _EXTREME),
    "maximum": ("support", {"max_nonzero_source_households": 29}, _EXTREME),
    "median": ("support", {"median_effective_sample_size": 23.75}, _EXTREME),
    "minimum_by_level": (
        "support",
        {"minimum_rows": {"local_authority": 37}},
        _EXTREME,
    ),
    "maxima_in_a_list": ("support", {"maximum_ess": [23.75, 41]}, _EXTREME),
    "name_in_upper_case": ("areas", [{"area_code": _AREA, "ESS": 23.75}], _FIGURE),
    "n_rows": ("areas", [{"geography_id": _AREA, "n_rows": 37}], _FIGURE),
    "statistic_after_the_name": ("support", {"rows_min": 37}, _EXTREME),
    "smallest": ("support", {"smallest_nonzero_households": 29}, _EXTREME),
    "statistics_under_the_name": (
        "support",
        {"effective_sample_size": {"min": 23.75, "p10": 29}},
        _EXTREME,
    ),
    # Shapes the third review of microcosm#1307 found the walk missed.
    "statistic_above_the_name": ("support", {"min": {"rows": 37}}, _EXTREME),
    "distinct_sources": (
        "areas",
        [{"area_code": _AREA, "distinct_sources": 29}],
        _FIGURE,
    ),
    "sample_size": ("areas", [{"geography_id": _AREA, "sample_size": 41}], _FIGURE),
    "lad_code_column": (
        "support_table",
        {"lad_code": [_AREA, _OTHER_AREA], "rows": [37, 41]},
        _FIGURE,
    ),
    "geography_field": ("areas", [{"geography": _AREA, "ess": 23.75}], _FIGURE),
    "pandas_split": (
        "support_table",
        {
            "index": [0, 1],
            "columns": ["area_code", "rows"],
            "data": [[_AREA, 37], [_OTHER_AREA, 41]],
        },
        _FIGURE,
    ),
    "pandas_split_indexed_by_area": (
        "support_table",
        {"index": [_AREA], "columns": ["effective_sample_size"], "data": [[23.75]]},
        _FIGURE,
    ),
    "map_by_area": ("support", {"rows_by_area": {_AREA: 37}}, _FIGURE),
    "num_rows": ("areas", [{"area_code": _AREA, "num_rows": 37}], _FIGURE),
    "row_count": ("areas", [{"area_code": _AREA, "row_count": 37}], _FIGURE),
    "kish_ess": ("areas", [{"area_code": _AREA, "kish_ess": 23.75}], _FIGURE),
    "area_code_in_upper_case": ("areas", [{"AREA_CODE": _AREA, "rows": 37}], _FIGURE),
    "figure_as_text_with_separators": (
        "areas",
        [{"area_code": _AREA, "nonzero_households": "3,741"}],
        _FIGURE,
    ),
    "minimum_distinct_source_households": (
        "support",
        {"min_distinct_source_households": 29},
        _EXTREME,
    ),
}


@pytest.mark.parametrize("shape", sorted(_SUPPORT_FIGURE_SHAPES))
def test_the_walk_finds_support_figures_in_every_shape(shape: str) -> None:
    """Records keyed by any area identifier, code-keyed maps, columnar tables,
    lists and the extremes of a figure are found alike (microcosm#1301). The
    sign-off drafter walks the entries it drafts for a public PR with it."""

    key, value, expected = _SUPPORT_FIGURE_SHAPES[shape]
    findings = "\n".join(
        dc._uk_dense_disclosure_findings("release_manifest.json", {key: value})
    )
    assert "release_manifest.json carries" in findings and expected in findings
    # A finding names where the figures are, never their values.
    assert f"first at ['{key}" in findings
    for figure in _FIGURES:
        assert figure not in findings


def test_disclosure_walker_allows_floors_counts_weights_and_estimates() -> None:
    shipped = {
        "area_support": {
            "floors": {"minimum_rows": 50, "minimum_distinct_sources": 50},
            "below_floor": {"local_authority": [_AREA]},
        },
        "support": {"areas_checked": 2, "areas_below_floor": 1},
        "weights": {
            "effective_sample_size": 3.5,
            "max_weight": 4.0,
            "median_positive_weight": 1.0,
        },
        "worst_areas": [
            {
                "area_code": _AREA,
                "worst_abs_relative_error": 0.4,
                "support_below_floor": True,
            }
        ],
        "failures": [f"local_authority/{_AREA}: rows < 50, ESS < 50"],
        # Per-cell estimates ship, however the cell's area is keyed.
        "local_rows": [{"geography_id": _AREA, "candidate_estimate": 100.0}],
        "rows": [{"area_code": _AREA, "estimate": 5.0}],
        # A ``code`` that holds no area code identifies no area.
        "code": {"git_commit": "a" * 40, "git_dirty": False},
        "diagnostic_warnings": [{"code": "synthetic_warning", "rows": 3}],
        # Counts of target rows, and a declared bound, are not support figures.
        "fit": {"local_by_family": [{"family": "synthetic", "rows": 12}]},
        "parameters": {"maximum_max_to_median_ratio": 100.0},
        # Names that merely end in "ess" are not ESS, under an order
        # statistic too.
        "subset_relations": [{"max_excess": 0.0}],
        "details": {"maximum_owner_share_without_main_residence_excess": 0.05},
        "bounds": {"max": {"max_excess": 0.0}},
        # Target rows by family, as a map, a family named "..._by_area" or a
        # split table without an area column.
        "per_family": {
            "rows_by_family": {"synthetic": 12},
            "family_sizes": {"hmrc_income_by_area": 3},
            "table": {"index": [0], "columns": ["family", "rows"], "data": [["x", 12]]},
        },
        # A statistic of a weight is not one of a support figure.
        "weight_summary": {"max": {"weight": 4.0}},
        # Region, nation and UK support is a GOR-level aggregate, not an area's
        # (placeholder region, nation and UK codes).
        "support_by_region": {"ess": {"E12999999": 4100.0, "W92999999": 2300.0}},
        "regions": [{"area_code": "K02999999", "rows": 41000}],
        "region_rows": [{"region_code": "E12999999", "rows": 41000}],
        "region_table": {
            "index": ["E12999999", "W92999999"],
            "columns": ["ess"],
            "data": [[4100.0], [2300.0]],
        },
    }
    assert dc._uk_dense_disclosure_findings("shipped.json", shipped) == []


def test_a_real_shaped_candidate_manifest_draws_no_warning(tmp_path: Path) -> None:
    """The build's candidate manifest carries a cross-grain relation's
    ``max_excess`` and a spine gate's excess bound, which an over-broad
    pattern once read as ESS extremes; neither is an area support figure
    (microcosm#1301)."""

    release_dir = write_dense_bundle(tmp_path)
    path = release_dir / "rowwise_candidate_manifest.json"
    candidate = json.loads(path.read_text())
    relation = {
        "grain": "local_authority",
        "part": "synthetic.part",
        "whole": "synthetic.whole",
        "part_total": 90.0,
        "whole_total": 100.0,
        "max_excess": 0.0,
        "geographies_tested": 2,
        "status": "consistent",
    }
    candidate["cross_grain"] = {"subset_relations": [relation]}
    candidate["solve"]["cross_grain"] = {"subset_relations": [relation]}
    candidate["identity"]["spine"] = {
        "spine_provenance": {
            "spine_gate_report": {
                "payload": {
                    "gates": {
                        "uk_stage_was_wealth_coherence": {
                            "details": {
                                "maximum_owner_share_without_main_residence_excess": (
                                    0.05
                                ),
                                "owner_share_without_main_residence_value": 0.01,
                            }
                        }
                    }
                }
            }
        }
    }
    path.write_text(json.dumps(candidate))
    _refresh_packaging_checksums(release_dir)
    assert validate_release_dir(release_dir) == []


def test_the_area_support_block_names_only_excused_areas() -> None:
    excused = [{"id": f"area_support_exclusion:local_authority/{_AREA}"}]
    failures: list[str] = []
    dc._check_area_support_block(
        {
            "gate": "uk_local_area_support",
            "floors": {"minimum_rows": 50},
            "below_floor": {"local_authority": [_AREA]},
        },
        excused,
        failures,
    )
    assert failures == []

    failures = []
    dc._check_area_support_block(
        {"below_floor": {"local_authority": [_OTHER_AREA]}, "rows": {_OTHER_AREA: 3}},
        excused,
        failures,
    )
    assert any("holds area codes and the floors only" in f for f in failures)
    assert any("without a reviewed area_support_exclusion" in f for f in failures)

    failures = []
    dc._check_area_support_block({"below_floor": [_AREA]}, excused, failures)
    assert any("must map each geography level" in f for f in failures)
