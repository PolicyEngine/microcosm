"""The release-candidate pre-flight fails closed on every missing key or flag."""

from __future__ import annotations

import base64
import importlib.util
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from microcosm.build.uk_runtime import country_adapter
from test_support.microcosm_build.uk_full_build_cli import (
    patch_support_register,
    synthetic_geography_binding,
)
from test_support.paths import paths_for

#: The committed register reader, kept before the autouse fixture replaces it.
_REAL_REGISTER = country_adapter.uk_atomic_support_register

_TEST_PATHS = paths_for("microcosm-build")


REPO_ROOT = _TEST_PATHS.repository


def _load():
    spec = importlib.util.spec_from_file_location(
        "preflight_uk_local_release_candidate",
        REPO_ROOT / "tools" / "preflight_uk_local_release_candidate.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _good_candidate(tmp_path: Path) -> Path:
    candidate = tmp_path / "candidate"
    (candidate / "logbook-spool").mkdir(parents=True)
    (candidate / "logbook-spool" / "row.json").write_text("{}")
    h5 = candidate / "microcosm_uk_2024_25_local.h5"
    h5.write_bytes(b"h5")
    import hashlib

    manifest = {
        "release_role": "dense",
        "parameters": {
            "release_candidate": True,
            "epochs": 2000,
            "n_clones": 15,
            "skip_holdout": False,
            "doctrine": {
                "max_weight_ratio": 10.0,
                "target_weight_rule": "nation_grain_family_equal_sqrt_count",
                "solve_epochs": 2000,
                "clone_count": 15,
            },
        },
        "releasable": True,
        "blocked_at_f100": False,
        "solve": {
            "measure_resolution": {"blocks": 1},
            "target_weight_rule_override": {},
        },
        "fit": {"rotated_holdout": {"n_folds": 5, "mean_holdout_loss": 0.2}},
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
                "reason": "synthetic gap",
                "tracking": "microcosm#869",
                "approved_by": "synthetic_reviewer",
                "adjudication": "synthetic decision",
                "approved_on": "2026-09-03",
                "expires_on": "2026-10-03",
            }
        },
        "identity": {"spine": {"pin_verified": True}, "ladder": {"pin_verified": True}},
        "geography": {"assignment": synthetic_geography_binding()},
        "outputs": {
            "dataset": {"path": str(h5), "sha256": hashlib.sha256(b"h5").hexdigest()}
        },
    }
    (candidate / "rowwise_candidate_manifest.json").write_text(json.dumps(manifest))
    (candidate / "microcosm_uk_2024_25_local.local_gates.json").write_text(
        json.dumps(_signed_report())
    )
    return candidate


KEY = base64.b64encode(b"\x06" * 32).decode()


@pytest.fixture(autouse=True)
def _trusted_key(monkeypatch) -> None:
    monkeypatch.setenv("MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY", KEY)


@pytest.fixture(autouse=True)
def _registered_supports(monkeypatch) -> None:
    """The synthetic candidates bind synthetic support pins; register them."""
    patch_support_register(monkeypatch)


def _signed_report(area_support_details: dict | None = None) -> dict:
    """A complete schema-4 local report signed the way the battery signs.

    ``area_support_details`` joins the area-support gate's details before the
    report is signed, so a planted figure stays under a valid signature.
    """

    import hashlib
    import hmac

    from microcosm.build.gate_battery import _canonical_json_bytes, _canonical_sha256
    from microcosm.data import contract as dc

    key_bytes = base64.b64decode(KEY)
    gates = {}
    for entry_id in sorted(dc._UK_DENSE_GATE_ENTRY_IDS):
        blocking = entry_id in dc._UK_DENSE_RELEASE_BLOCKING_IDS
        details: dict = {"floors": {"minimum_effective_sample_size": 50.0}}
        if entry_id == "uk_local_area_support":
            details.update(area_support_details or {})
        gates[entry_id] = {
            "gate": entry_id.removeprefix("uk_local_"),
            "phase": "terminal",
            "criticality": "release_blocking" if blocking else "diagnostic",
            "status": "passed" if blocking else "failed",
            "failures": [] if blocking else ["a diagnostic miss"],
            "details": details,
            "reason": None,
        }
    attestation = {
        "schema_version": dc._UK_GATE_BATTERY_ATTESTATION_SCHEMA_VERSION,
        "producer": dc._UK_GATE_BATTERY_PRODUCER,
        "country": "uk",
        "release_id": "uk-local-candidate-f100-s42-20260903T160005Z-e39715fc",
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
        "signing_key_sha256": hashlib.sha256(key_bytes).hexdigest(),
        "signature": None,
    }
    report = {
        "schema_version": dc._UK_GATE_BATTERY_SCHEMA_VERSION,
        "country": "uk",
        "release_id": attestation["release_id"],
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
        key_bytes, _canonical_json_bytes(report), hashlib.sha256
    ).hexdigest()
    return report


def test_good_candidate_dir_passes(tmp_path: Path) -> None:
    module = _load()
    assert (
        module.check_candidate_dir(_good_candidate(tmp_path), today=date(2026, 9, 4))
        == []
    )


def _in_force_today(candidate: Path) -> None:
    """Centre the synthetic measure exclusion's window on today: ``main``
    checks the run on the real clock."""

    path = candidate / "rowwise_candidate_manifest.json"
    manifest = json.loads(path.read_text())
    today = date.today()
    manifest["measure_exclusions"]["obr.housing_benefit"].update(
        approved_on=(today - timedelta(days=1)).isoformat(),
        expires_on=(today + timedelta(days=30)).isoformat(),
    )
    path.write_text(json.dumps(manifest))


def test_the_candidate_preflight_warns_on_no_support_figure(
    tmp_path: Path, capsys
) -> None:
    """María's ruling of 2026-10-26: a release ships per-area support figures
    (microcosm#1301), so the candidate pre-flight no longer walks for them."""

    module = _load()
    candidate = _good_candidate(tmp_path)
    _in_force_today(candidate)
    manifest_path = candidate / "rowwise_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["support"] = {"min_nonzero_households": 37}
    manifest_path.write_text(json.dumps(manifest))
    assert not hasattr(module, "candidate_disclosure_warnings")
    assert module.main(["--candidate-dir", str(candidate)]) == 0
    err = capsys.readouterr().err
    assert "WARN" not in err and "FAIL" not in err


def test_a_named_incumbent_manifest_must_exist(tmp_path: Path, capsys) -> None:
    module = _load()
    candidate = _good_candidate(tmp_path)
    _in_force_today(candidate)
    missing = tmp_path / "no-such-manifest.json"
    argv = ["--candidate-dir", str(candidate), "--incumbent-manifest", str(missing)]
    assert module.main(argv) == 1
    assert f"FAIL incumbent manifest missing: {missing}" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        module.main(["--env", "--incumbent-manifest", str(missing)])


def test_release_dir_contract_warnings_print_without_failing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The contract's warnings never fail the pre-flight; its failures
    still do."""
    from microcosm.data import contract

    module = _load()
    monkeypatch.setattr(
        contract,
        "validate_release_dir",
        lambda _path: ["synthetic.json carries a synthetic finding"],
    )
    assert module.main(["--release-dir", str(tmp_path)]) == 0
    err = capsys.readouterr().err
    assert "WARN synthetic.json carries a synthetic finding" in err
    assert "preflight: OK, 1 warning(s)" in err

    def refuse(_path):
        raise contract.ReleaseContractError(tmp_path, ["synthetic failure"])

    monkeypatch.setattr(contract, "validate_release_dir", refuse)
    assert module.main(["--release-dir", str(tmp_path)]) == 1
    assert "FAIL synthetic failure" in capsys.readouterr().err


@pytest.mark.parametrize("role", ["national", None])
def test_preflight_refuses_a_national_role_manifest(tmp_path: Path, role) -> None:
    """This pre-flight is the dense line's; any other role is refused by name."""

    module = _load()
    candidate = _good_candidate(tmp_path)
    manifest_path = candidate / "rowwise_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if role is None:
        del manifest["release_role"]
    else:
        manifest["release_role"] = role
    manifest_path.write_text(json.dumps(manifest))
    failures = module.check_candidate_dir(candidate, today=date(2026, 9, 4))
    assert len(failures) == 1
    assert f"manifest.release_role is {role!r}" in failures[0]


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (
            lambda m, r: m["parameters"].__setitem__("release_candidate", False),
            "release_candidate",
        ),
        (lambda m, r: m.__setitem__("releasable", False), "releasable"),
        (
            lambda m, r: m["solve"]["measure_resolution"].__setitem__("blocks", 15),
            "blocks",
        ),
        (
            lambda m, r: m["parameters"]["doctrine"].__setitem__(
                "max_weight_ratio", 100.0
            ),
            "max_weight_ratio",
        ),
        (
            lambda m, r: m["parameters"].__setitem__(
                "refit_l2", {"lambda": 0.03, "anchor": "initial", "basis": "chi_square"}
            ),
            "refit L2 penalty",
        ),
        (
            lambda m, r: m["census_household_uprating"]["tenure_cells"].__setitem__(
                "applied", False
            ),
            "A17",
        ),
        (lambda m, r: m.__setitem__("measure_exclusions", {}), "measure_exclusions"),
        (
            lambda m, r: m["measure_exclusions"]["obr.housing_benefit"].__setitem__(
                "expires_on", "2026-09-01"
            ),
            "expired",
        ),
        (lambda m, r: r.__setitem__("release_candidate", False), "dev posture"),
        (lambda m, r: r.__setitem__("shippable", False), "shippable"),
        (
            lambda m, r: r["gates"]["uk_local_area_support"].__setitem__(
                "status", "failed"
            ),
            "release-blocking gate",
        ),
        (lambda m, r: r["attestation"].__setitem__("signature", "0" * 64), "signature"),
    ],
)
def test_each_missing_flag_is_named(tmp_path: Path, mutate, needle: str) -> None:
    module = _load()
    candidate = _good_candidate(tmp_path)
    manifest = json.loads((candidate / "rowwise_candidate_manifest.json").read_text())
    report = json.loads(
        (candidate / "microcosm_uk_2024_25_local.local_gates.json").read_text()
    )
    mutate(manifest, report)
    (candidate / "rowwise_candidate_manifest.json").write_text(json.dumps(manifest))
    (candidate / "microcosm_uk_2024_25_local.local_gates.json").write_text(
        json.dumps(report)
    )
    failures = module.check_candidate_dir(candidate, today=date(2026, 9, 4))
    assert any(needle in failure for failure in failures), failures


_SKIPPED = {"report_only": True, "skipped": True, "reason": "Not requested."}


def _rewrite_manifest(candidate: Path, mutate) -> None:
    path = candidate / "rowwise_candidate_manifest.json"
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_text(json.dumps(manifest))


def test_a_holdout_that_was_not_requested_passes(tmp_path: Path) -> None:
    """The rotated holdout is an opt-in diagnostic (microcosm#1301)."""
    module = _load()
    candidate = _good_candidate(tmp_path)

    def skip(manifest):
        manifest["parameters"]["skip_holdout"] = True
        manifest["fit"]["rotated_holdout"] = dict(_SKIPPED)

    _rewrite_manifest(candidate, skip)
    assert module.check_candidate_dir(candidate, today=date(2026, 9, 4)) == []


@pytest.mark.parametrize(
    "holdout",
    [
        {"n_folds": 4, "mean_holdout_loss": 0.2},
        {"n_folds": 5},
        {"n_folds": 5, "mean_holdout_loss": float("nan")},
        {"n_folds": 5, "mean_holdout_loss": True},
    ],
    ids=["four-folds", "no-mean", "nan-mean", "bool-mean"],
)
def test_a_measured_holdout_needs_five_folds_and_a_finite_mean(
    tmp_path: Path, holdout: dict
) -> None:
    module = _load()
    candidate = _good_candidate(tmp_path)
    _rewrite_manifest(
        candidate, lambda m: m["fit"].__setitem__("rotated_holdout", holdout)
    )
    failures = module.check_candidate_dir(candidate, today=date(2026, 9, 4))
    assert any("5 folds and a finite mean" in failure for failure in failures)


@pytest.mark.parametrize(
    ("skip_holdout", "holdout", "needle"),
    [
        (False, _SKIPPED, "requested (--rotated-holdout) but the run recorded it"),
        (True, {"n_folds": 5, "mean_holdout_loss": 0.2}, "not requested but"),
        (None, _SKIPPED, "must record whether the holdout was requested"),
        (True, None, "neither measured nor skipped"),
    ],
)
def test_the_holdout_record_must_match_the_request(
    tmp_path: Path, skip_holdout, holdout, needle: str
) -> None:
    module = _load()
    candidate = _good_candidate(tmp_path)

    def mutate(manifest):
        manifest["parameters"]["skip_holdout"] = skip_holdout
        if holdout is None:
            del manifest["fit"]["rotated_holdout"]
        else:
            manifest["fit"]["rotated_holdout"] = dict(holdout)

    _rewrite_manifest(candidate, mutate)
    failures = module.check_candidate_dir(candidate, today=date(2026, 9, 4))
    assert any(needle in failure for failure in failures), failures


def _per_clone(exact: bool = True, blocks: int = 15) -> dict:
    return {
        "blocks": blocks,
        "engine_population_representation": {
            "mode": "block_weights_scaled_to_pool",
            "blocks": blocks,
            "exact": exact,
        },
    }


@pytest.mark.parametrize(
    ("resolution", "passes"),
    [
        (_per_clone(), True),
        (_per_clone(exact=False), False),
        # One block per clone, so the count must be the run's K.
        (_per_clone(blocks=5), False),
        ({"blocks": 15}, False),
    ],
)
def test_one_engine_block_per_clone_passes_only_when_recorded_exact(
    tmp_path: Path, resolution: dict, passes: bool
) -> None:
    module = _load()
    candidate = _good_candidate(tmp_path)
    manifest_path = candidate / "rowwise_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["solve"]["measure_resolution"] = resolution
    manifest_path.write_text(json.dumps(manifest))
    failures = module.check_candidate_dir(candidate, today=date(2026, 9, 4))
    assert (failures == []) is passes, failures
    if not passes:
        assert any("without an exact pool representation" in f for f in failures)


def _sized(manifest: dict) -> None:
    """Make the fixture a sized run at the size doctrine."""

    manifest["parameters"].update(
        {
            "dataset_households": 150_000,
            "selection_pi_hi": 0.5,
            "baseline_pi_floor": 0.5,
            "refit_l2": {
                "lambda": 0.0215748772773509,
                "anchor": "uniform",
                "basis": "chi_square",
            },
        }
    )
    manifest["solve"]["dataset_size"] = {
        "requested_households": 150_000,
        "realized_households": 150_000,
        "selection_search_bracket": {"lower": 1e-10, "upper": 10.0},
    }
    manifest.setdefault("release_posture", {})["size_doctrine"] = True


def _sized_at_the_old_doctrine(manifest: dict) -> None:
    manifest["parameters"]["dataset_households"] = 100_000
    manifest["solve"]["dataset_size"].update(
        requested_households=100_000, realized_households=100_000
    )


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda m: None, None),
        (
            lambda m: m["parameters"].__setitem__("dataset_households", 60_000),
            "parameters.dataset_households",
        ),
        (
            lambda m: m["parameters"].__setitem__("selection_pi_hi", 0.95),
            "parameters.selection_pi_hi",
        ),
        (lambda m: m["parameters"].__setitem__("refit_l2", None), "refit_l2"),
        (
            lambda m: m["solve"]["dataset_size"].__setitem__(
                "realized_households", 149_998
            ),
            "realized",
        ),
        # microcosm#1301: a run consistently sized at the old doctrine fails.
        (_sized_at_the_old_doctrine, "parameters.dataset_households"),
        # The size search must have run in the doctrine's L0 bracket.
        (
            lambda m: m["solve"]["dataset_size"].pop("selection_search_bracket"),
            "selection_search_bracket",
        ),
        (
            lambda m: m["solve"]["dataset_size"].__setitem__(
                "selection_search_bracket", {"lower": 1e-7, "upper": 10.0}
            ),
            "selection_search_bracket",
        ),
        (
            lambda m: m["release_posture"].__setitem__("size_doctrine", False),
            "size_doctrine",
        ),
        (lambda m: m["solve"].pop("dataset_size"), "dataset_size is missing"),
        (
            lambda m: m["parameters"].__setitem__("resume_size_checkpoint", "ck"),
            "resumed from a size checkpoint",
        ),
        (
            lambda m: m["parameters"].__setitem__("selection_l2", {"lambda": 0.5}),
            "selection L2 penalty",
        ),
    ],
)
def test_a_sized_run_passes_only_at_the_size_doctrine(
    tmp_path: Path, mutate, needle
) -> None:
    module = _load()
    candidate = _good_candidate(tmp_path)
    manifest_path = candidate / "rowwise_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    _sized(manifest)
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest))
    failures = module.check_candidate_dir(candidate, today=date(2026, 9, 4))
    if needle is None:
        assert failures == []
    else:
        assert any(needle in failure for failure in failures), failures


def test_the_restated_size_doctrine_matches_the_ruled_one() -> None:
    """The pre-flight restates the size doctrine rather than importing it; the
    restatement and the doctrine module must agree (microcosm#1301)."""
    from microcosm.build.uk_runtime.dataset_size import uk_size_search_bracket
    from microcosm.build.uk_runtime.local_doctrine import uk_local_size_doctrine

    module = _load()
    doctrine = uk_local_size_doctrine()
    assert module.SIZE_DOCTRINE == {key: doctrine[key] for key in module.SIZE_DOCTRINE}
    assert module.REFIT_L2_DOCTRINE == {
        "lambda": doctrine["refit_l2_lambda"],
        "anchor": doctrine["refit_l2_anchor"],
        "basis": doctrine["refit_l2_basis"],
    }
    lower, upper = uk_size_search_bracket()
    assert module.SIZE_SEARCH_BRACKET == {"lower": lower, "upper": upper}


def test_an_unsized_run_carries_no_refit_penalty(tmp_path: Path) -> None:
    module = _load()
    candidate = _good_candidate(tmp_path)
    manifest_path = candidate / "rowwise_candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["parameters"]["refit_l2"] = {"lambda": 0.01}
    manifest_path.write_text(json.dumps(manifest))
    failures = module.check_candidate_dir(candidate, today=date(2026, 9, 4))
    assert any("without a size request" in failure for failure in failures)


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (
            lambda m: m["geography"]["assignment"].__setitem__("assignment", "legacy"),
            "geography.assignment.assignment is 'legacy', not 'atomic'",
        ),
        (
            lambda m: m["geography"]["assignment"]["support_pins"][
                "uk_ni_data_zone_2021"
            ].__setitem__("sha256", "9" * 64),
            "support_pins.uk_ni_data_zone_2021 is not the registered support",
        ),
        (
            lambda m: m["geography"]["assignment"]["support_pins"][
                "uk_ew_output_area_2021"
            ].__setitem__("size_bytes", 1),
            "support_pins.uk_ew_output_area_2021 is not the registered support",
        ),
        (lambda m: m.pop("geography"), "geography.assignment.assignment is None"),
    ],
)
def test_preflight_requires_the_registered_atomic_assignment(
    tmp_path: Path, mutate, needle: str
) -> None:
    """A release candidate is the identity-keyed assignment on the registered
    supports (microcosm#932 round 1): the pre-flight refuses the legacy law, a
    pre-#932 manifest with no binding, and a pin that is not the register's."""
    candidate = _good_candidate(tmp_path)
    path = candidate / "rowwise_candidate_manifest.json"
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_text(json.dumps(manifest))
    failures = _load().check_candidate_dir(candidate, today=date(2026, 9, 4))
    assert any(needle in line for line in failures), failures


def test_preflight_reads_the_committed_support_register(
    tmp_path: Path, monkeypatch
) -> None:
    """Against sources.yaml itself the synthetic pins are not the register's."""
    monkeypatch.setattr(country_adapter, "uk_atomic_support_register", _REAL_REGISTER)
    candidate = _good_candidate(tmp_path)
    failures = _load().check_candidate_dir(candidate, today=date(2026, 9, 4))
    refused = [line for line in failures if "support_pins" in line]
    assert len(refused) == 3, failures
    register = _REAL_REGISTER()
    assert set(register) == set(synthetic_geography_binding()["support_pins"])
    assert all(
        len(row["sha256"]) == 64 and row["size_bytes"] > 0 for row in register.values()
    )


def test_env_check_names_the_missing_key_and_pins(tmp_path: Path) -> None:
    module = _load()
    pins = tmp_path / "pins.txt"
    spine = tmp_path / "spine.h5"
    spine.write_bytes(b"s")
    ladder = tmp_path / "ladder.npz"
    ladder.write_bytes(b"l")
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "consumer_facts.jsonl").write_text("{}\n")
    (ledger / "manifest.json").write_text("{}")
    import hashlib

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    pins.write_text(
        f"spine={sha(spine)} ladder={sha(ladder)} facts={sha(ledger / 'consumer_facts.jsonl')} manifest={sha(ledger / 'manifest.json')}\n"
    )
    good_env = {module.SIGNING_KEY_ENV: base64.b64encode(b"\x07" * 32).decode()}
    assert (
        module.check_env(
            pins_path=pins,
            spine_h5=spine,
            ladder_npz=ladder,
            ledger_dir=ledger,
            environ=good_env,
        )
        == []
    )
    failures = module.check_env(
        pins_path=pins, spine_h5=spine, ladder_npz=ladder, ledger_dir=ledger, environ={}
    )
    assert any(module.SIGNING_KEY_ENV in f for f in failures)
    pins.write_text("spine=deadbeef ladder=x facts=y manifest=z\n")
    failures = module.check_env(
        pins_path=pins,
        spine_h5=spine,
        ladder_npz=ladder,
        ledger_dir=ledger,
        environ=good_env,
    )
    assert any("digest mismatch" in f for f in failures)


def test_preflight_does_not_infer_tenure_success_from_ladder_uprating(tmp_path):
    candidate = _good_candidate(tmp_path)
    path = candidate / "rowwise_candidate_manifest.json"
    manifest = json.loads(path.read_text())
    tenure = manifest["census_household_uprating"]["tenure_cells"]
    tenure["holds"][0].update(
        {
            "applied": False,
            "eligible": False,
            "skipped": True,
            "from_period": None,
            "to_period": None,
            "reason": "missing_ladder_oa_vintage",
        }
    )
    tenure.update(
        {"applied": False, "cells": 0, "eligible_cells": 0, "skipped_cells": 1}
    )
    path.write_text(json.dumps(manifest))
    failures = _load().check_candidate_dir(candidate, today=date(2026, 9, 7))
    assert any(
        "A17" in failure and "skipped attempted" in failure for failure in failures
    )
    assert not any("A15" in failure for failure in failures)


def test_graph_built_manifest_passes_the_candidate_preflight(
    tmp_path: Path, monkeypatch
) -> None:
    """The graph driver's projected manifest is the schema the pre-flight reads.

    The build signs its own local battery report under the posture's
    ``*.local_gates.json`` name, bound to its Logbook attempt; nothing is
    signed by hand.
    """

    pytest.importorskip("tables")
    from microcosm.build.logbook import load_spool_rows
    from test_support.microcosm_build.uk_full_build_cli import (
        STEM,
        graph_dense_bundle,
    )

    out = graph_dense_bundle(tmp_path, monkeypatch, "--release-candidate")
    module = _load()
    build_id = load_spool_rows(out / "logbook-spool")[0].build_id
    report = json.loads((out / f"{STEM}.local_gates.json").read_text())
    assert report["release_id"] == build_id and report["shippable"] is True
    manifest = json.loads((out / "rowwise_candidate_manifest.json").read_text())
    assert manifest["release_role"] == "dense"
    assert manifest["parameters"]["release_candidate"] is True
    assert module.check_candidate_dir(out, today=date(2026, 9, 4)) == []
    # The same projection is refused for what the pre-flight refuses: a size
    # run, or a run whose manifest lost its uprating receipt.
    manifest["census_household_uprating"]["applied"] = False
    (out / "rowwise_candidate_manifest.json").write_text(json.dumps(manifest))
    failures = module.check_candidate_dir(out, today=date(2026, 9, 4))
    assert any("A15" in line for line in failures)
