"""Build the raw UK FRS spine Frame from pinned local tabs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from microcosm.build.country_spec import country_stage_plan, load_country_spec
from microcosm.build.frame_sampling import (
    normalize_sampled_household_mass,
    sample_frame_households,
)
from microcosm.build.logbook import canonical_json_bytes
from microcosm.build.logbook_adoption import (
    AttemptState,
    append_phase,
    apply_error_verdict,
    atomic_write_json,
    error_receipt_path,
    git_code_pin,
    local_artifact_reference,
    preflight_digest,
    record_terminal_attempt,
    resolve_predecessor,
    role_pins_digest,
    sha256_argument,
    write_error_receipt,
)
from microcosm.build.uk_runtime.age_tail import UKAgeTailStageTransform
from microcosm.build.uk_runtime.cgt_imputation import uk_cgt_spine_stage_transform
from microcosm.build.uk_runtime.cgt_structure import (
    UKCGTBandDonorStageTransform,
    UKCGTIncidenceCloneStageTransform,
)
from microcosm.build.uk_runtime.etb_services import UKETBServicesStageTransform
from microcosm.build.uk_runtime.etb_vat import UKETBVATStageTransform
from microcosm.build.uk_runtime.frs_brma import UKFRSBRMAStageTransform
from microcosm.build.uk_runtime.frs_council_tax import UKFRSCouncilTaxStageTransform
from microcosm.build.uk_runtime.frs_disability import UKFRSDisabilityStageTransform
from microcosm.build.uk_runtime.frs_education import UKFRSEducationStageTransform
from microcosm.build.uk_runtime.frs_education_grants import (
    FRS_EDUCATION_GRANT_REWRITES,
    UKFRSEducationGrantSplitStageTransform,
)
from microcosm.build.uk_runtime.frs_employment import UKFRSEmploymentStageTransform
from microcosm.build.uk_runtime.frs_household_draws import (
    UKFRSHouseholdDrawsStageTransform,
)
from microcosm.build.uk_runtime.frs_legacy_proxies import (
    UKFRSLegacyProxiesStageTransform,
)
from microcosm.build.uk_runtime.frs_person_draws import UKFRSPersonDrawsStageTransform
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
from microcosm.build.uk_runtime.frs_spine import (
    UKFRSSpineStageTransform,
    uk_frs_spine_seed_frame,
)
from microcosm.build.uk_runtime.frs_take_up import UKFRSTakeUpStageTransform
from microcosm.build.uk_runtime.hmrc_replay import write_hmrc_replay_report
from microcosm.build.uk_runtime.lcfs_consumption import (
    UKLCFSConsumptionStageTransform,
)
from microcosm.build.uk_runtime.national_build import write_uk_national_frame
from microcosm.build.uk_runtime.national_frame import uk_household_weight_kind
from microcosm.build.uk_runtime.national_sampling import (
    UK_SAMPLE_RUNG_TOKENS,
    UK_SAMPLE_SEED_DEFAULT,
)
from microcosm.build.uk_runtime.regional_uprating import (
    UKRegionalPropertyUpratingStageTransform,
)
from microcosm.build.uk_runtime.salary_sacrifice import UKSalarySacrificeStageTransform
from microcosm.build.uk_runtime.spi_spine import (
    UKFRSHMRCSpineLeavesStageTransform,
    UKSPIIncomeSpineStageTransform,
    UKSPISupportChannelStageTransform,
)
from microcosm.build.uk_runtime.student_loans import UKStudentLoansStageTransform
from microcosm.build.uk_runtime.take_up_contract import load_uk_take_up_contract
from microcosm.build.uk_runtime.was_wealth import UKWASWealthStageTransform
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

_PIPELINE = "uk-frs-spine"
_REPOSITORY = Path(__file__).resolve().parents[1]
_RUNG_NAMED_EDGE_SIGNATURE = "The least populated classes in y have only 1 member"
_RUNG_ABORT_EXIT_CODE = 3
_STAGE_NAMES = (
    "frs_spine",
    "frs_employment",
    "frs_council_tax",
    "frs_disability",
    "frs_education",
    "frs_legacy_proxies",
    "frs_education_grant_split",
    "frs_take_up",
    "frs_person_draws",
    "frs_household_draws",
    "frs_brma",
    "was_wealth",
    "regional_property_uprating",
    "lcfs_consumption",
    "etb_vat",
    "etb_services",
    "frs_hmrc_spine_leaves",
    "spi_support_channel",
    "hmrc_spi_income_spine",
    "cgt_incidence_clone",
    "cgt_band_donors",
    "hmrc_cgt_gains_spine",
    "salary_sacrifice",
    "student_loans",
    "age_tail",
)


def _rung_sample_fraction(value: str) -> float:
    """CLI rung policy (#624) over the permissive library validator."""

    try:
        fraction = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"sample fraction must be a number; got {value!r}."
        ) from error
    if fraction not in UK_SAMPLE_RUNG_TOKENS:
        raise argparse.ArgumentTypeError(
            "sample fraction must be one of 0.01, 0.10, or 1.0 (the #624 rungs)."
        )
    return fraction


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the deterministic UK FRS spine from pinned raw tabs. Every "
            "stochastic stage draws identity-keyed from seeds declared in the "
            "manifest, so two runs from the same inputs are payload-identical."
        )
    )
    parser.add_argument(
        "--frs-raw-dir",
        type=Path,
        required=True,
        help="Directory containing the 14 licensed FRS 2024-25 tab files.",
    )
    parser.add_argument(
        "--spine-h5",
        type=Path,
        required=True,
        help="Output H5 path for the raw FRS spine Frame.",
    )
    parser.add_argument(
        "--spi-tab",
        type=Path,
        required=True,
        help="Pinned local SPI 2022-23 put2223uk.tab path.",
    )
    parser.add_argument(
        "--hmrc-ods",
        type=Path,
        required=True,
        help="Pinned local HMRC collated ODS path.",
    )
    parser.add_argument(
        "--cgt-ods",
        type=Path,
        help="Pinned local HMRC Capital Gains Tax Table 3 ODS path.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Optional directory for a copy of the completed spine checkpoint.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=_rung_sample_fraction,
        default=1.0,
        help=(
            "Scale-ladder rung (#624): 0.01 smoke, 0.10 dev, or 1.0 full. "
            "Below 1.0 the raw FRS spine is sampled immediately after ingest, "
            "renormalized to full household mass, and treated as a receipt."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=UK_SAMPLE_SEED_DEFAULT,
        help=f"Raw FRS spine sampling seed (default: {UK_SAMPLE_SEED_DEFAULT}).",
    )
    parser.add_argument(
        "--was-tab",
        type=Path,
        help="Caller-supplied private WAS round-8 household tab for was_wealth.",
    )
    parser.add_argument(
        "--lcfs-hh-tab",
        type=Path,
        help="Caller-supplied private LCFS 2023-24 household tab for lcfs_consumption.",
    )
    parser.add_argument(
        "--lcfs-person-tab",
        type=Path,
        help="Caller-supplied private LCFS 2023-24 person tab for lcfs_consumption.",
    )
    parser.add_argument(
        "--etb-tab",
        type=Path,
        help="Caller-supplied private ETB 1977-2024 household tab for ETB stages.",
    )
    parser.add_argument(
        "--emit-nonzero-shares",
        type=Path,
        help="Optional JSON path for unweighted per-produced-column nonzero shares.",
    )
    parser.add_argument(
        "--logbook-prev-row-digest",
        type=sha256_argument,
        help="Optional current Logbook chain head.",
    )
    args = parser.parse_args(argv)
    if args.sample_seed < 0:
        parser.error("sample seed must be a non-negative integer.")
    if args.sample_fraction != 1.0 and args.checkpoint_dir is not None:
        parser.error(
            "sampled spine rungs refuse --checkpoint-dir; rung artifacts are "
            "receipts, never releases."
        )
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if not args.frs_raw_dir.is_dir():
        raise ValueError(
            f"--frs-raw-dir must be an existing directory: {args.frs_raw_dir}"
        )
    if args.spine_h5.suffix != ".h5":
        raise ValueError("--spine-h5 must end with '.h5'.")
    if not args.spi_tab.is_file():
        raise ValueError(f"--spi-tab must be an existing file: {args.spi_tab}")
    if args.spi_tab.name != "put2223uk.tab":
        raise ValueError("--spi-tab must name put2223uk.tab.")
    if not args.hmrc_ods.is_file():
        raise ValueError(f"--hmrc-ods must be an existing file: {args.hmrc_ods}")
    if args.hmrc_ods.suffix.lower() != ".ods":
        raise ValueError("--hmrc-ods must end with '.ods'.")
    if args.cgt_ods is not None:
        if not args.cgt_ods.is_file():
            raise ValueError(f"--cgt-ods must be an existing file: {args.cgt_ods}")
        if args.cgt_ods.suffix.lower() != ".ods":
            raise ValueError("--cgt-ods must end with '.ods'.")
    paths = {
        "spine_h5": args.spine_h5,
        "build_sidecar": args.spine_h5.with_suffix(".build.json"),
        "hmrc_replay_sidecar": args.spine_h5.with_suffix(".hmrc_replay.json"),
    }
    if args.emit_nonzero_shares is not None:
        paths["emit_nonzero_shares"] = args.emit_nonzero_shares
    resolved: dict[Path, str] = {}
    for label, path in paths.items():
        target = Path(path).expanduser().resolve()
        other = resolved.get(target)
        if other is not None:
            raise ValueError(f"{label} path collides with {other}: {target}.")
        resolved[target] = label


def _artifact_pins(stages) -> dict[str, dict[str, object]]:
    pins = {}
    for stage in stages:
        for artifact in stage.artifacts:
            key = artifact.get("table", artifact.get("filename"))
            if key is None:
                continue
            key = str(key)
            pin = {
                "locator": str(artifact["locator"]),
                "sha256": str(artifact["sha256"]),
                "size_bytes": int(artifact["size_bytes"]),
            }
            if key in pins and pins[key] != pin:
                raise ValueError(
                    f"UK source artifact {key!r} has inconsistent pins across stages."
                )
            pins[key] = pin
    return dict(sorted(pins.items()))


def _stage_artifact_pins(stage) -> dict[str, dict[str, object]]:
    return {
        str(artifact.get("table", artifact.get("filename"))): {
            "locator": str(artifact["locator"]),
            "sha256": str(artifact["sha256"]),
            "size_bytes": int(artifact["size_bytes"]),
        }
        for artifact in stage.artifacts
        if "table" in artifact or "filename" in artifact
    }


def _resource_pins(stages, spec) -> dict[str, str]:
    """Country-package resources the selected stages declare as inputs.

    Non-tab artifacts reference committed resources by filename; their bytes
    are hashed by load_country_spec, so the pin is the spec's recorded sha.
    """

    pins: dict[str, str] = {}
    for stage in stages:
        for artifact in stage.artifacts:
            if "resource" not in artifact:
                continue
            resource = str(artifact["resource"])
            sha256 = spec.resource_hashes.get(resource)
            if sha256 is None:
                raise ValueError(
                    f"stage {stage.stage!r} declares resource artifact "
                    f"{resource!r} which is not a declared country-package "
                    "resource."
                )
            pins[resource] = str(sha256)
    return dict(sorted(pins.items()))


def _input_artifact_pins(stages) -> dict[str, dict[str, object]]:
    """Caller-supplied private input artifacts, pinned by role.

    Non-table, non-resource artifacts (the SPI donor tab and the HMRC ODS)
    carry their own sha256/size pins in the manifest. Binding them here puts
    the pins in the build sidecar and the Logbook input-pins digest, so two
    runs with different high-impact source inputs can never share build-side
    provenance (adversarial-review finding on #717).
    """

    pins: dict[str, dict[str, object]] = {}
    for stage in stages:
        for artifact in stage.artifacts:
            if "table" in artifact or "resource" in artifact:
                continue
            if "sha256" not in artifact:
                continue
            role = str(artifact.get("role") or artifact.get("filename") or "")
            if not role:
                raise ValueError(
                    f"stage {stage.stage!r} declares a pinned input artifact "
                    "without a role or filename."
                )
            pin = {
                "filename": str(
                    artifact.get("filename") or artifact.get("locator") or ""
                ),
                "kind": str(artifact.get("kind", "")),
                "sha256": str(artifact["sha256"]),
                "size_bytes": int(artifact["size_bytes"]),
            }
            if role in pins and pins[role] != pin:
                raise ValueError(
                    f"input artifact role {role!r} has inconsistent pins across stages."
                )
            pins[role] = pin
    return dict(sorted(pins.items()))


def _role_pins(pins: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        table: {
            "sha256": str(pin["sha256"]),
            "size_bytes": int(pin["size_bytes"]),
        }
        for table, pin in pins.items()
    }


def _entity_row_counts(frame) -> dict[str, int]:
    return {entity: int(len(frame.table(entity))) for entity in frame.entities}


def _rules_engine() -> PolicyEngineUKEngine:
    try:
        import policyengine_uk  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "build_uk_frs_spine requires the microcosm-build 'uk' extra "
            "(policyengine-uk). Run: uv sync --all-packages --extra uk"
        ) from exc
    return PolicyEngineUKEngine()


def _rules_engine_provenance() -> dict[str, str]:
    try:
        version = metadata.version("policyengine-uk")
    except metadata.PackageNotFoundError:
        return {"package": "policyengine-uk", "version": "unavailable"}
    return {"package": "policyengine-uk", "version": version}


def _declared_seeds(stages) -> dict[str, dict[str, int]]:
    declared: dict[str, dict[str, int]] = {}
    for stage in stages:
        stage_seeds: dict[str, int] = {}
        for operation in stage.operations:
            output = operation.parameters.get("output")
            seed = operation.parameters.get("seed")
            if seed is None:
                seed = operation.parameters.get("seed_base")
            if isinstance(output, str) and isinstance(seed, int):
                stage_seeds[output] = seed
            elif isinstance(seed, int):
                if operation.kind == "stack_zero_weight_donors":
                    stage_seeds["stack_zero_weight_donors"] = seed
                elif operation.kind == "strict_read_private_table":
                    stage_seeds["donor_bootstrap"] = seed
                elif operation.kind == "fit_weighted_qrf_stage1":
                    stage_seeds["stage1"] = seed
                elif operation.kind == "fit_weighted_qrf_stage2":
                    stage_seeds["stage2"] = seed
                elif operation.kind == "bridge_donor_column_via_qrf":
                    stage_seeds["bridge_donor_column_via_qrf"] = seed
                elif operation.kind == "assign_binary_from_rate":
                    target = operation.parameters.get("target")
                    if isinstance(target, str):
                        stage_seeds[target] = seed
                    else:
                        stage_seeds["assign_binary_from_rate"] = seed
                elif operation.kind == "fit_weighted_qrf_chain":
                    stage_seeds[stage.stage] = seed
                elif operation.kind == "fit_weighted_qrf":
                    stage_seeds[stage.stage] = seed
                elif operation.kind == "draw_capital_gains_prior_from_banded_quantiles":
                    stage_seeds[str(operation.parameters["salt"])] = seed
                elif operation.kind == "stack_band_donor_households":
                    stage_seeds["stack_band_donor_households"] = seed
                elif operation.kind == "within_band_draws":
                    stage_seeds["within_band_draws"] = seed
                elif operation.kind == "convert_donors_to_target_stock":
                    stage_seeds[str(operation.parameters["salt"])] = seed
                elif operation.kind == "top_up_to_stock":
                    stage_seeds[str(operation.parameters["salt"])] = seed
        if stage_seeds:
            declared[stage.stage] = stage_seeds
    return declared


def _build_sidecar(
    *,
    frame,
    stages,
    records,
    artifact_pins,
    resource_pins: dict[str, str],
    input_artifact_pins: dict[str, dict[str, object]],
    hmrc_replay: dict[str, object],
    stochastic_contract_sha256: str,
    frs_vintage: str,
    sampling: dict[str, object] | None,
) -> dict[str, object]:
    household_weight = frame.weights_for("household")
    return {
        "schema_version": 2,
        "pipeline": _PIPELINE,
        "stages": [stage.stage for stage in stages],
        "time_period": str(frame.metadata["time_period"]),
        "household_weight_kind": uk_household_weight_kind(frame).value,
        "household_weight_total": float(household_weight.values.sum()),
        "entity_row_counts": _entity_row_counts(frame),
        "artifact_pins": artifact_pins,
        "resource_pins": resource_pins,
        "input_artifact_pins": input_artifact_pins,
        "hmrc_replay": hmrc_replay,
        "stage_artifact_pins": {
            stage.stage: _stage_artifact_pins(stage) for stage in stages
        },
        "stage_records": [
            {
                "stage": record.stage,
                "produced": list(record.produced),
                "nonzero_share": dict(record.nonzero_share),
                "seconds": record.seconds,
            }
            for record in records
        ],
        "operations": {
            stage.stage: [operation.kind for operation in stage.operations]
            for stage in stages
        },
        "declared_seeds": _declared_seeds(stages),
        "source_vintages": {"frs": frs_vintage},
        "sampling": sampling,
        "stochastic_contract_sha256": stochastic_contract_sha256,
        "rules_engine": _rules_engine_provenance(),
    }


def _nonzero_shares(frame, columns: list[str]) -> dict[str, float]:
    shares: dict[str, float] = {}
    for column in columns:
        for entity in frame.entities:
            table = frame.table(entity)
            if column not in table.columns:
                continue
            values = table[column]
            if values.dtype == object:
                shares[column] = float(values.astype(str).ne("").mean())
            else:
                shares[column] = float((values != 0).mean())
            break
    return shares


def _new_build_id(timestamp: datetime) -> str:
    return f"uk-frs-spine-{timestamp.strftime('%Y%m%dT%H%M%SZ')}"


def _record_attempt(
    *,
    state: AttemptState,
    started_at: float,
    started_ts: datetime,
    code_pin: str,
    disposition: str,
    predecessor: str | None,
    rung: str,
    spool_dir: Path,
) -> Path:
    return record_terminal_attempt(
        state=state,
        started_at=started_at,
        started_ts=started_ts,
        pipeline=_PIPELINE,
        rung=rung,
        seed=None,
        code_pin=code_pin,
        disposition=disposition,
        predecessor=predecessor,
        spool_dir=spool_dir,
    )


def _sample_spine_frame(
    frame,
    *,
    fraction: float,
    seed: int,
) -> tuple[object, dict[str, object] | None]:
    if fraction == 1.0:
        return frame, None
    household_weight = frame.weights_for("household")
    pre_households = int(len(frame.table("household")))
    sampled, receipt = sample_frame_households(
        frame,
        fraction=fraction,
        seed=seed,
        source_name="UK FRS spine",
    )
    normalized, factor = normalize_sampled_household_mass(
        sampled,
        target_mass=float(household_weight.total),
        source_name="UK FRS spine",
    )
    return normalized, {
        "fraction": float(fraction),
        "seed": int(seed),
        "rung_token": UK_SAMPLE_RUNG_TOKENS[fraction],
        "pre_household_count": pre_households,
        "post_household_count": int(len(normalized.table("household"))),
        "normalization_factor": float(factor),
        "receipt": dict(receipt),
    }


def _run_plan_with_spine_sampling(
    plan,
    *,
    sample_fraction: float,
    sample_seed: int,
) -> tuple[object, tuple[object, ...], dict[str, object] | None]:
    if not plan.stages or plan.stages[0].name != "frs_spine":
        frame, records = plan.run(uk_frs_spine_seed_frame())
        return frame, records, None

    from microcosm.build.plan import StagePlan

    spine_frame, spine_records = StagePlan(plan.stages[:1]).run(
        uk_frs_spine_seed_frame()
    )
    spine_frame, sampling = _sample_spine_frame(
        spine_frame,
        fraction=sample_fraction,
        seed=sample_seed,
    )
    if len(plan.stages) == 1:
        return spine_frame, spine_records, sampling
    frame, tail_records = StagePlan(plan.stages[1:]).run(spine_frame)
    return frame, (*spine_records, *tail_records), sampling


def _rung_abort_receipt(
    args: argparse.Namespace,
    *,
    error: BaseException,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_kind": "uk_frs_spine_rung_abort_receipt",
        "build_kind": "uk_frs_spine",
        "sampling": {
            "sample_fraction": float(args.sample_fraction),
            "sample_seed": int(args.sample_seed),
            "rung_token": UK_SAMPLE_RUNG_TOKENS[args.sample_fraction],
        },
        "named_edge": "spine_split_singleton_class",
        "stage": "frs_spine",
        "error": str(error),
        "disposition": "aborted_with_receipt",
        "remedy": (
            "Re-roll --sample-seed; accepted dev-scale statistical edge. "
            "The computation is never altered to avoid it."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rung = UK_SAMPLE_RUNG_TOKENS[args.sample_fraction]
    started_at = time.perf_counter()
    started_ts = datetime.now(UTC)
    predecessor = resolve_predecessor(args.logbook_prev_row_digest)
    digest = preflight_digest(_PIPELINE)
    state = AttemptState(
        build_id=_new_build_id(started_ts),
        identity_digest=digest,
        input_pins_digest=digest,
        phases_reached=["attempt_started"],
        gate_verdicts={
            "pipeline": {
                "verdict": "running",
                "receipt": "pending-build-scoped-spine-receipt",
            }
        },
    )
    code_pin = "unresolved-local-git-code-pin"
    spool_dir = args.spine_h5.parent / "logbook-spool"
    try:
        _validate_args(args)
        # A crash between the H5 write and the sidecar writes must never
        # leave a stale sidecar beside a fresh H5 (adversarial-review
        # finding on #717): clear every output up front, and treat the
        # build sidecar - written last, binding the replay hash - as the
        # marker that the bundle is complete.
        stale_outputs = [
            args.spine_h5,
            args.spine_h5.with_suffix(".build.json"),
            args.spine_h5.with_suffix(".hmrc_replay.json"),
            args.spine_h5.with_suffix(".rung_abort.json"),
        ]
        if args.emit_nonzero_shares is not None:
            stale_outputs.append(args.emit_nonzero_shares)
        for stale in stale_outputs:
            stale.unlink(missing_ok=True)
        code_pin = git_code_pin(_REPOSITORY)
        append_phase(state, "configured")
        spec = load_country_spec("uk")
        if spec.sources is None:
            raise ValueError("UK country spec has no source stages.")
        stages_by_name = spec.sources.stage_map()
        stage_names = tuple(name for name in _STAGE_NAMES if name in stages_by_name)
        if "hmrc_cgt_gains_spine" in stage_names and args.cgt_ods is None:
            raise ValueError(
                "--cgt-ods is required when hmrc_cgt_gains_spine is scheduled."
            )
        if "was_wealth" in stage_names and args.was_tab is None:
            raise ValueError(
                "--was-tab is required when the was_wealth stage is scheduled."
            )
        if "lcfs_consumption" in stage_names:
            missing_lcfs = [
                flag
                for flag, value in (
                    ("--lcfs-hh-tab", args.lcfs_hh_tab),
                    ("--lcfs-person-tab", args.lcfs_person_tab),
                    ("--was-tab", args.was_tab),
                )
                if value is None
            ]
            if missing_lcfs:
                raise ValueError(
                    "lcfs_consumption requires caller-supplied private inputs: "
                    f"{', '.join(missing_lcfs)}."
                )
        if (
            "etb_vat" in stage_names or "etb_services" in stage_names
        ) and args.etb_tab is None:
            raise ValueError(
                "--etb-tab is required when etb_vat or etb_services is scheduled."
            )
        stages = [stages_by_name[name] for name in stage_names]
        artifact_pins = _artifact_pins(stages)
        resource_pins = _resource_pins(stages, spec)
        input_artifact_pins = _input_artifact_pins(stages)
        overlapping_pin_roles = set(artifact_pins) & set(input_artifact_pins)
        if overlapping_pin_roles:
            raise ValueError(
                "input artifact roles collide with FRS tab names: "
                f"{sorted(overlapping_pin_roles)}."
            )
        state.input_pins_digest = role_pins_digest(
            _role_pins({**artifact_pins, **input_artifact_pins})
        )
        run_config = {
            "pipeline": _PIPELINE,
            "stages": list(stage_names),
            "artifact_pins_digest": state.input_pins_digest,
            "spine_h5": str(args.spine_h5),
        }
        state.identity_digest = hashlib.sha256(
            canonical_json_bytes(run_config)
        ).hexdigest()
        append_phase(state, "inputs_pinned")
        engine = _rules_engine()
        stochastic_contract = load_uk_take_up_contract()
        frs_release = load_uk_frs_release()
        hmrc_spine_transform = UKSPIIncomeSpineStageTransform(
            args.spi_tab,
            args.hmrc_ods,
            stage=stages_by_name["hmrc_spi_income_spine"],
            sampled_rung=args.sample_fraction != 1.0,
        )
        implementations = {
            "frs_spine": UKFRSSpineStageTransform(
                args.frs_raw_dir,
                stage=stages_by_name["frs_spine"],
            ),
            "frs_employment": UKFRSEmploymentStageTransform(
                args.frs_raw_dir,
                stage=stages_by_name["frs_employment"],
            ),
            "frs_council_tax": UKFRSCouncilTaxStageTransform(
                args.frs_raw_dir,
                stage=stages_by_name["frs_council_tax"],
            ),
            "frs_disability": UKFRSDisabilityStageTransform(
                stage=stages_by_name["frs_disability"],
            ),
            "frs_education": UKFRSEducationStageTransform(
                args.frs_raw_dir,
                stage=stages_by_name["frs_education"],
            ),
            "frs_legacy_proxies": UKFRSLegacyProxiesStageTransform(
                args.frs_raw_dir,
                stage=stages_by_name["frs_legacy_proxies"],
                engine=engine,
            ),
            "frs_education_grant_split": (
                UKFRSEducationGrantSplitStageTransform(
                    stage=stages_by_name["frs_education_grant_split"],
                    engine=engine,
                )
            ),
            "frs_take_up": UKFRSTakeUpStageTransform(
                contract=stochastic_contract,
                stage=stages_by_name["frs_take_up"],
            ),
            "frs_person_draws": UKFRSPersonDrawsStageTransform(
                contract=stochastic_contract,
                stage=stages_by_name["frs_person_draws"],
            ),
            "frs_household_draws": UKFRSHouseholdDrawsStageTransform(
                contract=stochastic_contract,
                stage=stages_by_name["frs_household_draws"],
            ),
            "frs_brma": UKFRSBRMAStageTransform(
                stage=stages_by_name["frs_brma"],
                engine=engine,
            ),
        }
        if "was_wealth" in stage_names:
            implementations["was_wealth"] = UKWASWealthStageTransform(
                stage=stages_by_name["was_wealth"],
                engine=engine,
                was_tab_path=args.was_tab,
            )
        if "regional_property_uprating" in stage_names:
            implementations["regional_property_uprating"] = (
                UKRegionalPropertyUpratingStageTransform(
                    stage=stages_by_name["regional_property_uprating"],
                )
            )
        if "lcfs_consumption" in stage_names:
            implementations["lcfs_consumption"] = UKLCFSConsumptionStageTransform(
                stage=stages_by_name["lcfs_consumption"],
                engine=engine,
                lcfs_hh_tab_path=args.lcfs_hh_tab,
                lcfs_person_tab_path=args.lcfs_person_tab,
                was_tab_path=args.was_tab,
            )
        if "etb_vat" in stage_names:
            implementations["etb_vat"] = UKETBVATStageTransform(
                stage=stages_by_name["etb_vat"],
                engine=engine,
                etb_tab_path=args.etb_tab,
            )
        if "etb_services" in stage_names:
            implementations["etb_services"] = UKETBServicesStageTransform(
                stage=stages_by_name["etb_services"],
                engine=engine,
                etb_tab_path=args.etb_tab,
            )
        implementations["frs_hmrc_spine_leaves"] = UKFRSHMRCSpineLeavesStageTransform(
            args.frs_raw_dir,
            stage=stages_by_name["frs_hmrc_spine_leaves"],
            sampled_rung=args.sample_fraction != 1.0,
        )
        implementations["spi_support_channel"] = UKSPISupportChannelStageTransform(
            stage=stages_by_name["spi_support_channel"],
            sample_fraction=args.sample_fraction,
        )
        implementations["hmrc_spi_income_spine"] = hmrc_spine_transform
        if "cgt_incidence_clone" in stage_names:
            implementations["cgt_incidence_clone"] = UKCGTIncidenceCloneStageTransform(
                stage=stages_by_name["cgt_incidence_clone"]
            )
        if "cgt_band_donors" in stage_names:
            implementations["cgt_band_donors"] = UKCGTBandDonorStageTransform(
                stage=stages_by_name["cgt_band_donors"]
            )
        if "hmrc_cgt_gains_spine" in stage_names:
            implementations["hmrc_cgt_gains_spine"] = uk_cgt_spine_stage_transform(
                stages_by_name["hmrc_cgt_gains_spine"],
                args.cgt_ods,
            )
        if "salary_sacrifice" in stage_names:
            implementations["salary_sacrifice"] = UKSalarySacrificeStageTransform(
                stage=stages_by_name["salary_sacrifice"]
            )
        if "student_loans" in stage_names:
            implementations["student_loans"] = UKStudentLoansStageTransform(
                stage=stages_by_name["student_loans"],
                calibration_year=frs_release.calibration_year,
            )
        if "age_tail" in stage_names:
            implementations["age_tail"] = UKAgeTailStageTransform(
                stage=stages_by_name["age_tail"]
            )
        plan = country_stage_plan(
            spec,
            implementations,
            stage_names=stage_names,
        )
        frame, records, sampling = _run_plan_with_spine_sampling(
            plan,
            sample_fraction=args.sample_fraction,
            sample_seed=args.sample_seed,
        )
        append_phase(state, "spine_built")
        output = write_uk_national_frame(frame, args.spine_h5)
        append_phase(state, "spine_written")
        if args.checkpoint_dir is not None:
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            write_uk_national_frame(frame, args.checkpoint_dir / "frs_spine.h5")
            append_phase(state, "checkpoint_written")
        sidecar_path = output.with_suffix(".build.json")
        replay_sidecar_path = output.with_suffix(".hmrc_replay.json")
        if hmrc_spine_transform.last_result is None:
            raise RuntimeError("HMRC SPI spine stage did not record replay evidence.")
        write_hmrc_replay_report(
            hmrc_spine_transform.last_result.replay_report,
            replay_sidecar_path,
        )
        append_phase(state, "hmrc_replay_sidecar_written")
        replay_bytes = replay_sidecar_path.read_bytes()
        replay_binding = {
            "filename": replay_sidecar_path.name,
            "report_kind": str(json.loads(replay_bytes).get("report_kind", "")),
            "sha256": hashlib.sha256(replay_bytes).hexdigest(),
        }
        sidecar = _build_sidecar(
            frame=frame,
            stages=stages,
            records=records,
            artifact_pins=artifact_pins,
            resource_pins=resource_pins,
            input_artifact_pins=input_artifact_pins,
            hmrc_replay=replay_binding,
            stochastic_contract_sha256=stochastic_contract.resource_sha256,
            frs_vintage=frs_release.vintage,
            sampling=sampling,
        )
        # E8 executed-effect receipts (#730/#684 two-arm rule, arm 2): the
        # clone/donor/salsac/student-loan transforms record their receipts on
        # last_result; persist them beside the declared seeds so the sidecar
        # carries evidence that every declared parameter shaped the output.
        e8_stage_evidence: dict[str, object] = {}
        for e8_stage_name in (
            "cgt_incidence_clone",
            "cgt_band_donors",
            "salary_sacrifice",
            "student_loans",
            "age_tail",
        ):
            e8_implementation = implementations.get(e8_stage_name)
            e8_last_result = getattr(e8_implementation, "last_result", None)
            if e8_last_result is not None:
                # age_tail's receipt is already the evidence mapping; the E8
                # transforms carry a result object that produces one.
                e8_stage_evidence[e8_stage_name] = (
                    e8_last_result
                    if isinstance(e8_last_result, dict)
                    else e8_last_result.evidence()
                )
        if e8_stage_evidence:
            sidecar["stage_evidence"] = e8_stage_evidence
        atomic_write_json(sidecar_path, sidecar)
        append_phase(state, "build_sidecar_written")
        if args.emit_nonzero_shares is not None:
            final_columns = list(
                dict.fromkeys(
                    [column for record in records for column in record.produced]
                    + list(FRS_EDUCATION_GRANT_REWRITES)
                )
            )
            atomic_write_json(
                args.emit_nonzero_shares,
                {
                    "stages": {
                        record.stage: dict(record.nonzero_share) for record in records
                    },
                    "final": _nonzero_shares(frame, final_columns),
                },
            )
            append_phase(state, "nonzero_shares_written")
        state.artifact_location = local_artifact_reference(
            output,
            repository_hint=_REPOSITORY,
        )
        state.gate_verdicts = {
            "pipeline": {
                "verdict": "passed",
                "receipt": local_artifact_reference(
                    sidecar_path, repository_hint=_REPOSITORY
                ),
            }
        }
        spool_path = _record_attempt(
            state=state,
            started_at=started_at,
            started_ts=started_ts,
            code_pin=code_pin,
            disposition="iterating",
            predecessor=predecessor,
            rung=rung,
            spool_dir=spool_dir,
        )
        print(f"Wrote FRS spine H5: {output}", file=sys.stderr)
        print(f"Wrote Logbook row: {spool_path}", file=sys.stderr)
        return 0
    except Exception as error:
        if (
            isinstance(error, ValueError)
            and args.sample_fraction != 1.0
            and _RUNG_NAMED_EDGE_SIGNATURE in str(error)
        ):
            rung_abort_path = args.spine_h5.with_suffix(".rung_abort.json")
            receipt = _rung_abort_receipt(args, error=error)
            atomic_write_json(rung_abort_path, receipt)
            state.gate_verdicts = {
                "uk_frs_spine_rung_abort": {
                    "verdict": "aborted",
                    "receipt": (
                        f"{local_artifact_reference(rung_abort_path, repository_hint=_REPOSITORY)}"
                        "#/named_edge"
                    ),
                }
            }
            append_phase(state, "rung_aborted")
            _record_attempt(
                state=state,
                started_at=started_at,
                started_ts=started_ts,
                code_pin=code_pin,
                disposition="discarded",
                predecessor=predecessor,
                rung=rung,
                spool_dir=spool_dir,
            )
            print(json.dumps(receipt, indent=2, sort_keys=True))
            return _RUNG_ABORT_EXIT_CODE
        try:
            receipt_path = write_error_receipt(
                error_receipt_path(args.spine_h5.parent, build_id=state.build_id),
                state=state,
                pipeline=_PIPELINE,
                error=error,
            )
            apply_error_verdict(
                state,
                local_artifact_reference(receipt_path, repository_hint=_REPOSITORY),
            )
            _record_attempt(
                state=state,
                started_at=started_at,
                started_ts=started_ts,
                code_pin=code_pin,
                disposition="failed",
                predecessor=predecessor,
                rung=rung,
                spool_dir=spool_dir,
            )
        except Exception:
            pass
        print(f"UK FRS spine build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
