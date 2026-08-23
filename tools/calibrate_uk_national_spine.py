"""Calibrate a WS-E UK spine at national level, without the certified-input replay.

The certified-H5 driver (``build_uk_national_dataset.py``) encodes the June
convention: its input arrives with a zero-weight SPI channel, and the driver
re-derives the SPI and CGT surfaces in-run before calibrating. A WS-E spine
already carries those derivations — E7/E8 ported the same instruments (pinned
sources, QRF stages, reviewed fences, conservation receipts) into declarative
source stages, and the spine ships post-allocation. Re-running the replay here
would discard the spine's certified derivation and substitute an equivalent
one, so this runner deliberately does not.

What it does, in order:

1. sha-verifies the spine input and the Ledger consumer artifact;
2. compiles the packaged target references against the Ledger facts at the
   declared calibration year (FRS 2024-25 -> 2025);
3. repairs the spine's structural NaN columns (SPI-channel concepts undefined
   on the base-FRS channel) to zero, from a fixed allowlist, with a receipt —
   any NaN outside the allowlist aborts;
4. runs ``UKNationalCalibrationStage`` under the frozen national doctrine;
5. writes the staged H5 through the shared validated writer;
6. scores the candidate against the incumbent enhanced FRS on the same frozen
   register (#578 rule 1), so the score rides the diagnostics in one pass;
7. writes US-format calibration diagnostics (schema 6 + uk block) with the
   score already merged into the ``build`` block, and shas the final bytes;
8. evaluates the numeric gate fences that are honestly measurable in this
   posture against the thresholds declared in ``uk/gates.json``, and lists the
   entries that are not evaluable here and why;
9. writes an assessment build record and spools a Logbook row.

This is an assessment posture: the artifact is structurally non-releasable
(non-certified input), the full declared battery is not enforced, and the
build record says both things in as many words. Gate entries that encode the
certified-input convention (in-run stage census, in-run fit-weight records)
are reported as not-evaluated rather than silently passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.logbook import canonical_json_bytes
from microcosm.build.logbook_adoption import (
    AttemptState,
    append_phase,
    git_code_pin,
    local_artifact_reference,
    record_terminal_attempt,
    resolve_predecessor,
    role_pins_digest,
)
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
from microcosm.build.uk_runtime.ledger_targets import compile_uk_target_registry
from microcosm.build.uk_runtime.national_build import (
    load_uk_national_frame,
    write_uk_national_frame,
)
from microcosm.build.uk_runtime.national_calibration import UKNationalCalibrationStage
from microcosm.build.uk_runtime.national_doctrine import UK_NATIONAL_SOLVE_DOCTRINE
from microcosm.build.uk_runtime.diagnostics import write_uk_calibration_diagnostics
from microcosm.calibrate import effective_sample_size

_REPOSITORY = Path(__file__).resolve().parent.parent
_PIPELINE = "uk-frs-calibration"

# Structural NaN repair allowlist. These twelve person columns are
# SPI-channel concepts the spine leaves undefined (NaN) on the base-FRS
# channel; zero is their semantic value there — base-channel incomes are
# carried by the standard FRS components. Any NaN outside this list aborts.
_STRUCTURAL_NAN_COLUMNS = (
    "other_investment_income",
    "hmrc_spi_employment_benefits",
    "hmrc_spi_employment_expenses",
    "hmrc_spi_other_social_security_income",
    "hmrc_spi_taxable_termination_pay",
    "hmrc_spi_miscellaneous_employment_income",
    "hmrc_spi_other_income",
    "hmrc_spi_state_pension_income",
    "hmrc_spi_employed_income",
    "hmrc_spi_total_earned_income",
    "hmrc_spi_total_investment_income",
    "hmrc_spi_assessable_income",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_gate_parameters() -> dict[str, dict[str, Any]]:
    from importlib import resources

    payload = json.loads(
        resources.files("microcosm.build.uk")
        .joinpath("gates.json")
        .read_text(encoding="utf-8")
    )
    return {
        entry["id"]: dict(entry.get("parameters") or {})
        for entry in payload["gates"]
    }


def _repair_structural_nans(frame: Any) -> dict[str, Any]:
    """Zero the allowlisted NaN columns in place; abort on any other NaN.

    ``Frame.table`` documents its return as read-only, but it hands back the
    live DataFrame (the calibration adapter takes defensive copies for
    exactly that reason). This repair leans on that liveness and then
    verifies the mutation stuck — if a future Frame returns copies, the
    post-assert aborts the run instead of calibrating on unrepaired data.
    """

    receipt: dict[str, Any] = {"repaired": {}, "policy": "structural_zero_fill"}
    for entity in ("person", "benunit", "household"):
        table = frame.table(entity)
        for column in table.columns:
            series = table[column]
            if series.dtype.kind != "f":
                continue
            nan_count = int(series.isna().sum())
            if nan_count == 0:
                continue
            if entity != "person" or column not in _STRUCTURAL_NAN_COLUMNS:
                raise ValueError(
                    f"unexpected NaN outside the structural allowlist: "
                    f"{entity}.{column} has {nan_count} NaN rows."
                )
            table[column] = series.fillna(0.0)
            receipt["repaired"][column] = nan_count
    for column in receipt["repaired"]:
        if int(frame.table("person")[column].isna().sum()) != 0:
            raise RuntimeError(
                f"structural NaN repair did not persist for person.{column}; "
                "Frame.table returned a copy."
            )
    return receipt


def _uk_target_geography_levels(registry: Any) -> dict[str, str]:
    """Row-name -> geography level, from the value-free contract."""

    from importlib import resources

    payload = json.loads(
        resources.files("microcosm.build.uk")
        .joinpath("uk_national_targets.json")
        .read_text(encoding="utf-8")
    )
    targets = {row["target_id"]: row for row in payload["targets"]}
    levels: dict[str, str] = {}
    for spec in registry.specs:
        target_id = spec.metadata.get("contract_target_id")
        target = targets.get(str(target_id))
        if target is None:
            raise ValueError(
                f"UK calibration target {spec.name!r} references unknown "
                f"contract target {target_id!r}."
            )
        geography_levels = tuple(target.get("geography_levels") or ())
        if not geography_levels:
            raise ValueError(
                f"UK calibration target {spec.name!r} has no geography level."
            )
        levels[spec.to_target().row_name] = str(geography_levels[0])
    return levels


def _assessment_gate_receipt(
    stage: UKNationalCalibrationStage,
    frame: Any,
    gate_parameters: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate the honestly-measurable numeric fences from uk/gates.json."""

    weights = np.asarray(frame.weights_for("household").values, dtype=float)
    household = frame.table("household")
    evaluated: dict[str, Any] = {}

    fit_params = gate_parameters["uk_target_fit"]
    bound = float(fit_params["max_abs_relative_error"])
    errors = {
        str(row["name"]): float(row["relative_error"]) for row in stage.diagnostics
    }
    failing = {
        name: error for name, error in errors.items() if abs(error) > bound
    }
    evaluated["uk_target_fit"] = {
        "passed": not failing,
        "threshold": bound,
        "targets_checked": len(errors),
        "failing_count": len(failing),
        "worst": dict(
            sorted(failing.items(), key=lambda kv: -abs(kv[1]))[:20]
        ),
    }

    ratio_params = gate_parameters["uk_weight_ratio"]
    positive = weights[weights > 0]
    max_to_median = float(positive.max() / np.median(positive)) if positive.size else 0.0
    ratio_bound = float(ratio_params["maximum_max_to_median_ratio"])
    evaluated["uk_weight_ratio"] = {
        "passed": max_to_median <= ratio_bound,
        "threshold": ratio_bound,
        "measured_max_to_median": max_to_median,
    }

    ess_params = gate_parameters["uk_weight_ess"]
    ess = float(effective_sample_size(weights))
    ess_fraction = ess / len(weights) if len(weights) else 0.0
    ess_bound = float(ess_params["minimum_ess_fraction"])
    evaluated["uk_weight_ess"] = {
        "passed": ess_fraction >= ess_bound,
        "threshold": ess_bound,
        "measured_ess_fraction": ess_fraction,
        "effective_sample_size": ess,
    }

    strata_params = gate_parameters["uk_zero_weight_strata"]
    zero_mask = weights <= 0.0
    strata_results = []
    matched = np.zeros(len(household), dtype=bool)
    for declaration in strata_params.get("declarations", []):
        selector = declaration.get("selector", {})
        mask = np.ones(len(household), dtype=bool)
        for column, value in selector.items():
            mask &= household[column].to_numpy() == value
        matched |= mask
        zero_rows = int((mask & zero_mask).sum())
        strata_results.append(
            {
                "name": declaration.get("name"),
                "zero_weight_rows": zero_rows,
                "maximum": declaration.get("maximum_zero_weight_rows"),
                "passed": zero_rows <= int(declaration["maximum_zero_weight_rows"]),
            }
        )
    unmatched_zero = int((zero_mask & ~matched).sum())
    evaluated["uk_zero_weight_strata"] = {
        "passed": all(row["passed"] for row in strata_results)
        and unmatched_zero == 0,
        "strata": strata_results,
        "zero_weight_rows_outside_declared_strata": unmatched_zero,
    }

    admin_params = gate_parameters["uk_aggregate_admin"]
    default_rtol = float(admin_params.get("default_rtol", 0.5))
    admin_results = []
    for anchor in admin_params.get("anchors", []):
        measure = str(anchor.get("measure") or anchor.get("name"))
        value = float(anchor["value"])
        if measure not in household.columns:
            admin_results.append(
                {"anchor": anchor.get("name"), "status": "column_absent"}
            )
            continue
        column_values = household[measure].to_numpy(dtype=float)
        total = float(np.dot(column_values, weights))
        carriers = column_values != 0
        carrier_weight = float(weights[carriers].sum())
        mean_carriers = (
            float(np.dot(column_values[carriers], weights[carriers]) / carrier_weight)
            if carrier_weight
            else float("nan")
        )
        # The anchor value's magnitude tells its statistic: NEED per-household
        # means are hundreds of pounds, the NHS anchor is a national total.
        measured = mean_carriers if abs(value) < 1e6 else total
        scale = max(abs(value), 1.0)
        tolerance = anchor.get("tolerance")
        rtol = float(tolerance) / scale if tolerance is not None else default_rtol
        miss = abs(measured - value) / scale
        admin_results.append(
            {
                "anchor": anchor.get("name"),
                "measure": measure,
                "declared_value": value,
                "measured": measured,
                "weighted_total": total,
                "weighted_mean_carriers": mean_carriers,
                "relative_miss": miss,
                "rtol": rtol,
                "passed": miss <= rtol,
                "statistic_convention": "assessed_by_anchor_magnitude",
            }
        )
    evaluated["uk_aggregate_admin"] = {
        "passed": all(row.get("passed", False) for row in admin_results),
        "anchors": admin_results,
    }

    not_evaluated = {
        "uk_release_family_build_stages": (
            "counts in-run stage names; superseded for spine inputs — the "
            "spine's build record carries the source-stage census"
        ),
        "uk_weights_audit": (
            "consumes in-run FitWeightRecords from the replay stage; the "
            "spine's qrf_evidence sidecar holds the equivalent record"
        ),
        "uk_release_input_coverage": "manifest describes the certified-input pipeline",
        "uk_release_input_coverage_manifest_current": "same",
        "uk_ledger_compile_parity_production_2023": (
            "register-coverage receipt verified the 2025 compile pre-run"
        ),
        "uk_ledger_compile_parity_incumbent_2025": "same",
        "uk_export_surface": "deferred to T0 column-inventory evaluation",
        "uk_target_surface": (
            "register identity is enforced by the scorer refusing mismatched "
            "target rows"
        ),
        "uk_support": "support-bounds resources bind to the replay stages",
        "uk_take_up_signal": "take-up draws are spine stages; deferred to T-evals",
        "uk_qrf_tail_concentration": "recorded at spine build (L3 baselines)",
        "uk_degenerate_release_surface": "deferred to T0/T3 evaluations",
        "uk_nonnegative_columns": "spine stage-local gates already enforce",
        "uk_brma_enum_domain": "deferred to T0 load check in the model venv",
        "uk_student_loan_plan_enum_domain": "same",
        "uk_calibration_reference_coverage": (
            "register-coverage receipt carries the census"
        ),
    }
    return {
        "posture": "spine_assessment",
        "enforcement": "not_applied",
        "evaluated": evaluated,
        "not_evaluated": not_evaluated,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", required=True, type=Path)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--ledger-facts", required=True, type=Path)
    parser.add_argument("--ledger-facts-sha256", required=True)
    parser.add_argument("--ledger-manifest-sha256", required=True)
    parser.add_argument("--staging-h5", required=True, type=Path)
    parser.add_argument("--register-json", required=True, type=Path,
                        help="Frozen scoring register emitted by the coverage check;"
                        " re-derived here and byte-compared.")
    parser.add_argument("--incumbent-h5", required=True, type=Path)
    parser.add_argument("--diagnostics-json", required=True, type=Path)
    parser.add_argument("--build-record-json", required=True, type=Path)
    parser.add_argument("--logbook-prev-row-digest", default=None)
    args = parser.parse_args(argv)

    started_at = time.perf_counter()
    started_ts = datetime.now(timezone.utc)
    code_pin = git_code_pin(_REPOSITORY)

    # --- 1. verify inputs -------------------------------------------------
    input_sha = _sha256_file(args.input_h5)
    if input_sha != args.input_sha256:
        raise SystemExit(
            f"input H5 sha mismatch: measured {input_sha}, pinned {args.input_sha256}"
        )
    artifact = load_ledger_consumer_artifact(args.ledger_facts)
    if artifact.facts_sha256 != args.ledger_facts_sha256:
        raise SystemExit("ledger facts sha mismatch")
    if artifact.manifest_sha256 != args.ledger_manifest_sha256:
        raise SystemExit("ledger manifest sha mismatch")

    source_pins = {
        "input_h5": {
            "sha256": input_sha,
            "size_bytes": args.input_h5.stat().st_size,
        },
        "ledger_facts": {
            "sha256": artifact.facts_sha256,
            "size_bytes": (args.ledger_facts / "consumer_facts.jsonl").stat().st_size
            if args.ledger_facts.is_dir()
            else args.ledger_facts.stat().st_size,
        },
    }

    # --- 2. compile the register at the declared calibration year ---------
    calibration_year = load_uk_frs_release().calibration_year
    compilation = compile_uk_target_registry(
        artifact.facts, target_period=calibration_year
    )
    if compilation.unsupported:
        raise SystemExit(
            f"{len(compilation.unsupported)} target references failed to compile"
        )
    register_payload = {
        "country": compilation.registry.country,
        "version": compilation.registry.version,
        "specs": [
            {
                field: value
                for field, value in spec.__dict__.items()
                if value is not None
            }
            for spec in compilation.registry.specs
        ],
    }
    register_bytes = json.dumps(
        register_payload, indent=2, sort_keys=True
    ).encode("utf-8")
    register_sha = hashlib.sha256(register_bytes).hexdigest()
    frozen_sha = hashlib.sha256(args.register_json.read_bytes()).hexdigest()
    if register_sha != frozen_sha:
        raise SystemExit(
            "re-derived register differs from the frozen scoring register: "
            f"{register_sha} vs {frozen_sha}"
        )

    run_config = {
        "pipeline": _PIPELINE,
        "source_pins": source_pins,
        "register_sha256": register_sha,
        "calibration_year": calibration_year,
        "doctrine": {
            "epochs": UK_NATIONAL_SOLVE_DOCTRINE.epochs,
            "learning_rate": UK_NATIONAL_SOLVE_DOCTRINE.learning_rate,
            "max_weight_ratio": UK_NATIONAL_SOLVE_DOCTRINE.max_weight_ratio,
            "seed": UK_NATIONAL_SOLVE_DOCTRINE.seed,
            "target_loss_cap": UK_NATIONAL_SOLVE_DOCTRINE.target_loss_cap,
            "l0_lambda": UK_NATIONAL_SOLVE_DOCTRINE.l0_lambda,
            "mass_rule": UK_NATIONAL_SOLVE_DOCTRINE.mass_rule,
        },
        "structural_nan_allowlist": list(_STRUCTURAL_NAN_COLUMNS),
    }
    state = AttemptState(
        build_id=f"uk-frs-calibration-{started_ts.strftime('%Y%m%dT%H%M%SZ')}",
        identity_digest=hashlib.sha256(canonical_json_bytes(run_config)).hexdigest(),
        input_pins_digest=role_pins_digest(source_pins),
        phases_reached=["attempt_started"],
        gate_verdicts={},
    )

    # --- 3. load + repair -------------------------------------------------
    frame, provenance = load_uk_national_frame(args.input_h5)
    append_phase(state, "input_loaded")
    nan_receipt = _repair_structural_nans(frame)
    append_phase(state, "structural_nans_repaired")

    # --- 4. calibrate -----------------------------------------------------
    stage = UKNationalCalibrationStage(compilation, period=calibration_year)
    calibrated = stage(frame)
    append_phase(state, "national_calibration_solved")

    # --- 5. write the H5 --------------------------------------------------
    args.staging_h5.parent.mkdir(parents=True, exist_ok=True)
    write_uk_national_frame(calibrated, args.staging_h5)
    staged_sha = _sha256_file(args.staging_h5)
    append_phase(state, "staging_h5_written")

    # --- 6. score vs the incumbent on the same register -------------------
    import importlib.util

    scorer_spec = importlib.util.spec_from_file_location(
        "score_uk_national_candidate",
        Path(__file__).with_name("score_uk_national_candidate.py"),
    )
    scorer = importlib.util.module_from_spec(scorer_spec)
    scorer_spec.loader.exec_module(scorer)
    score = scorer.score_uk_national_candidate(
        candidate_h5=args.staging_h5,
        incumbent_h5=args.incumbent_h5,
        target_registry=compilation.registry,
    )
    append_phase(state, "scored_vs_incumbent")

    # --- 7. diagnostics with the score merged, single pass ----------------
    build_block = {
        "build_id": state.build_id,
        "ledger_facts": artifact.provenance(),
        "code_pin": code_pin,
        "source_pins": source_pins,
        "input_posture": {
            "posture": "staging_candidate_spine",
            "filename": args.input_h5.name,
            "tier": "staging_candidate",
            "revision": "non_certified_staging_candidate",
            "sha256": input_sha,
            "size_bytes": args.input_h5.stat().st_size,
        },
        "structural_nan_repair": nan_receipt,
        "score_vs_enhanced_frs": score,
    }
    write_uk_calibration_diagnostics(
        stage.solve_result,
        args.diagnostics_json,
        calibrated,
        target_geography_levels=_uk_target_geography_levels(compilation.registry),
        target_registry=compilation.registry,
        build=build_block,
    )
    diagnostics_sha = _sha256_file(args.diagnostics_json)
    append_phase(state, "diagnostics_written")

    # --- 8. assessment gate fences ----------------------------------------
    gate_receipt = _assessment_gate_receipt(
        stage, calibrated, _load_gate_parameters()
    )
    for gate_id, payload in gate_receipt["evaluated"].items():
        verdict = payload.get("passed")
        state.gate_verdicts[gate_id] = {
            "verdict": "passed"
            if verdict
            else ("failed" if verdict is not None else "measured"),
            "receipt": f"local://{args.build_record_json.name}#/assessment_gates/"
            f"evaluated/{gate_id}",
        }
    append_phase(state, "assessment_gates_measured")

    # --- 9. build record + logbook ----------------------------------------
    record = {
        "schema_version": 1,
        "build_kind": "uk_spine_calibration_assessment",
        "status": "completed",
        "shippable": False,
        "shippable_reason": (
            "non-certified spine input; declared battery not enforced in "
            "spine-assessment posture"
        ),
        "build_id": state.build_id,
        "code_pin": code_pin,
        "pipeline": _PIPELINE,
        "calibration_year": calibration_year,
        "run_config": run_config,
        "input_posture": build_block["input_posture"],
        "structural_nan_repair": nan_receipt,
        "register": {
            "sha256": register_sha,
            "n_specs": len(compilation.registry.specs),
        },
        "calibration": stage.manifest,
        "assessment_gates": gate_receipt,
        "score_vs_enhanced_frs": score,
        "artifacts": {
            "staging_h5": {
                "path": str(args.staging_h5),
                "sha256": staged_sha,
                "size_bytes": args.staging_h5.stat().st_size,
                "retention": "local_untracked",
            },
            "calibration_diagnostics": {
                "path": str(args.diagnostics_json),
                "sha256": diagnostics_sha,
            },
            "incumbent_h5": {
                "path": str(args.incumbent_h5),
                "sha256": _sha256_file(args.incumbent_h5),
            },
        },
    }
    args.build_record_json.parent.mkdir(parents=True, exist_ok=True)
    args.build_record_json.write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    state.artifact_location = local_artifact_reference(
        args.staging_h5, repository_hint=_REPOSITORY
    )
    append_phase(state, "build_record_written")
    spool = record_terminal_attempt(
        state=state,
        started_at=started_at,
        started_ts=started_ts,
        pipeline=_PIPELINE,
        rung="f100",
        seed=UK_NATIONAL_SOLVE_DOCTRINE.seed,
        code_pin=code_pin,
        disposition="iterating",
        predecessor=resolve_predecessor(args.logbook_prev_row_digest),
        spool_dir=args.staging_h5.parent / "logbook-spool",
    )

    summary = {
        "build_id": state.build_id,
        "code_pin": code_pin,
        "final_loss": stage.manifest["loss"],
        "targets": stage.manifest["matrix_target_count"],
        "score_rule_1": {
            "candidate_full_loss": score["candidate_full_loss"],
            "incumbent_full_loss": score["incumbent_full_loss"],
            "candidate_target_wins": score["candidate_target_wins"],
            "incumbent_target_wins": score["incumbent_target_wins"],
        },
        "uk_target_fit": gate_receipt["evaluated"]["uk_target_fit"],
        "staging_h5_sha256": staged_sha,
        "diagnostics_sha256": diagnostics_sha,
        "logbook_spool": str(spool),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
