"""Deterministic projections of the current US runtime configuration authorities.

This module is a migration seam, not a second configuration system.  Each
builder reads the constants, checked-in manifests, and public runtime identity
constructors used by the constants-era executor and returns plain JSON-shaped
objects.  The one-shot US bundle generator can therefore author the spec from
the current authority once, while the legacy adapter can compile the same
fields back for byte-for-byte comparison.

Run-request values are intentionally absent.  In particular, exact-k ``k``,
``pi_hi``, and ``seed`` have no current default and this module must not mint
one while extracting the runtime contract.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from functools import lru_cache
from importlib.resources import files
from typing import Any

from microcosm.build.spec_engine.battery_semantics import (
    derive_battery_registry_views as _derive_battery_registry_views,
)
from microcosm.build.spec_engine.battery_semantics import (
    project_battery_legacy_contract as _project_battery_legacy_contract,
)
from microcosm.build.spec_engine.typed_closure import take_up_scope_registry_wire
from microcosm.build.us_runtime.stacked_battery_contract import (
    build_live_stacked_battery_contract,
)

__all__ = [
    "build_battery",
    "build_battery_contract",
    "build_calibration",
    "build_calibration_contract",
    "build_selection",
    "build_selection_contract",
    "build_take_up",
    "build_take_up_contract",
    "build_us_runtime_contracts",
    "contract_sha256",
    "derive_battery_registry_views",
    "project_battery_legacy_contract",
    "project_legacy_calibration_contract",
    "project_legacy_take_up_contract",
    "resolve_calibration_tail_contracts",
]


_TAKE_UP_PROGRAM_VARIABLES = (
    ("snap", "takes_up_snap_if_eligible"),
    ("tanf", "takes_up_tanf_if_eligible"),
    ("eitc", "takes_up_eitc"),
    ("medicaid", "takes_up_medicaid_if_eligible"),
    ("chip", "takes_up_chip_if_eligible"),
    ("basic_health_program", "takes_up_basic_health_program_if_eligible"),
    ("medicare", "takes_up_medicare_if_eligible"),
    ("ssi", "takes_up_ssi_if_eligible"),
    ("dc_ptc", "takes_up_dc_ptc"),
    ("head_start", "takes_up_head_start_if_eligible"),
    ("early_head_start", "takes_up_early_head_start_if_eligible"),
    ("housing_assistance", "takes_up_housing_assistance_if_eligible"),
    ("aca", "takes_up_aca_if_eligible"),
)
_TAKE_UP_PROGRAM_COUNT = len(_TAKE_UP_PROGRAM_VARIABLES)
_TAKE_UP_SOURCE_RESOURCE = "source_stages.json"
_TAKE_UP_CONTRACT_RESOURCE = "take_up_contract.json"
_PUF_TAIL_SUPPORT_REF = {
    "domain": "spine",
    "support_role": "puf_tax_detail",
    "pointer": "/tail_support/legacy_contract",
}
_PUF_TAIL_EXECUTION_REF = {
    "domain": "imputation",
    "producer": "primary_puf_qrf",
    "resource": "tax_unit.@primary_puf_execution_config",
    "pointer": "/binding/capital_gains_tail",
}
_LEGACY_ENGINE_FACT_FIELDS = frozenset(
    {"variable", "entity", "value_type", "default", "engine_class"}
)
_TAKE_UP_DOCUMENTATION_FIELDS = frozenset(
    {"consumed_via", "engine_state_note", "followup", "notes", "scope_owner"}
)

# Compiler-facing executable references use this reviewed, closed kernel
# namespace.  Dotted Python names are extraction evidence only and are kept in
# the compatibility projections below; they are never executable identifiers
# in the generation-1 contract.
_KERNEL_ID_BY_IMPLEMENTATION = {
    "microcosm.build.gates.tail_concentration_gate": "tail_concentration_gate",
    "microcosm.build.us_runtime.exact_k_ladder.calibrate_exact_k_ladder": (
        "calibrate_exact_k_ladder"
    ),
    "microcosm.build.us_runtime.exact_k_ladder.exact_k_ladder_manifest_payload": (
        "exact_k_ladder_manifest_payload"
    ),
    "microcosm.build.us_runtime.puf_capital_gains_tail.assert_puf_capital_gains_tail_survives_selection": (
        "puf_capital_gains_tail_post_selection_gate"
    ),
    "microcosm.calibrate.exact_k.assert_exact_k_support": "assert_exact_k_support",
    "microcosm.calibrate.exact_k.select_exact_k": "select_exact_k",
    "microcosm.calibrate.solve.calibrate": "calibrate",
    "microcosm.calibrate.solve.calibrate_l0_refit": "calibrate_l0_refit",
    "microcosm.calibrate.solve.refit_l0_selection": "refit_l0_selection",
    "torch.optim.adam.Adam": "torch_adam",
}

# SourceManifest operation names are legacy evidence, not the step type
# vocabulary.  Normalize them through this total table so the typed bundle has
# a small closed pipeline algebra while retaining the exact old operation id
# and every operation parameter on the emitted row.
_SOURCE_TAKE_UP_STEP_BINDINGS = {
    "aggregate_person_to_tax_unit": ("measured_map", "marketplace_measured_map"),
    "assign_binary_from_rate": ("assignment", "binary_assignment"),
    "calibrate_binary_assignment": ("count_calibration", "count_calibration"),
    "calibrate_binary_assignment_joint_targets": (
        "count_calibration",
        "joint_count_calibration",
    ),
    "compute_ratio": ("assignment", "marketplace_assignment"),
    "derive_housing_tenure_inputs": ("measured_map", "housing_measured_map"),
    "derive_medicare_take_up": ("measured_map", "medicare_measured_map"),
    "derive_snap_take_up": ("probability_seed", "snap_probability_seed"),
    "fit_weighted_qrf": ("imputed_transfer", "weighted_qrf_imputed_transfer"),
    "impute_housing_assistance_to_puf_support": (
        "imputed_transfer",
        "housing_imputed_transfer",
    ),
    "support_clip": ("assignment", "marketplace_assignment"),
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def contract_sha256(value: object) -> str:
    """Return the strict canonical-JSON digest of one extracted contract."""

    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _json_ready(value: object) -> object:
    """Copy a runtime authority value into deterministic JSON-shaped data."""

    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(nested)
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(nested) for nested in value]
    if isinstance(value, (set, frozenset)):
        return [_json_ready(nested) for nested in sorted(value)]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _json_ready(item())
    raise TypeError(
        f"Runtime contract projections must be JSON-shaped; got {type(value).__name__}."
    )


def _mapping_like(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{location} must be a mapping.")
    return value


def _array_like(value: object, location: str) -> list[object]:
    if not isinstance(value, list):
        raise RuntimeError(f"{location} must be an array.")
    return value


def _callable_name(function: Callable[..., object]) -> str:
    return f"{function.__module__}.{function.__qualname__}"


def _kernel_ref(function: Callable[..., object]) -> str:
    implementation_id = _callable_name(function)
    try:
        kernel_id = _KERNEL_ID_BY_IMPLEMENTATION[implementation_id]
    except KeyError as error:
        raise RuntimeError(
            f"No reviewed F0 kernel id for implementation {implementation_id!r}."
        ) from error
    return f"kernel:{kernel_id}"


def _adam_arguments(
    adam: Callable[..., object], learning_rate: float
) -> dict[str, object]:
    """Resolve the reviewed Adam surface into a closed argument object.

    This deliberately refuses signature drift.  The generated contract names
    every behavior-relevant argument directly; it does not embed a generic
    callable-signature snapshot that could silently admit a new parameter.
    """

    signature = inspect.signature(adam)
    expected = (
        "params",
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "foreach",
        "maximize",
        "capturable",
        "differentiable",
        "fused",
        "decoupled_weight_decay",
    )
    if tuple(signature.parameters) != expected:
        raise RuntimeError(
            "torch.optim.Adam signature changed; review the calibration schema "
            f"before regenerating (got {tuple(signature.parameters)!r})."
        )
    defaults = {
        name: _json_ready(signature.parameters[name].default) for name in expected[2:]
    }
    return {
        "params": {"source": "trainable_log_weights_and_optional_l0_gates"},
        "lr": learning_rate,
        **defaults,
    }


@lru_cache(maxsize=1)
def _source_manifest_authority() -> dict[str, object]:
    """Load and validate the source-stage resource, retaining its exact rows."""

    from microcosm.build.source_manifest import load_source_manifest

    resource = files("microcosm.build.us").joinpath(_TAKE_UP_SOURCE_RESOURCE)
    raw_bytes = resource.read_bytes()
    raw = json.loads(raw_bytes)
    manifest = load_source_manifest(resource)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("stages"), list):
        raise RuntimeError("US source-stage authority is not an object with stages.")
    raw_names = [str(stage.get("stage")) for stage in raw["stages"]]
    parsed_names = [stage.stage for stage in manifest.stages]
    if raw_names != parsed_names:
        raise RuntimeError(
            "Validated US source-stage order differs from its raw authority rows."
        )
    return {
        "country": manifest.country,
        "version": manifest.version,
        "policy": manifest.policy,
        "file_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "canonical_resource_sha256": contract_sha256(raw),
        "stages": deepcopy(raw["stages"]),
    }


def _program_source_stages(
    variable: str,
    source_stages: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        deepcopy(dict(stage))
        for stage in source_stages
        if variable in stage.get("outputs", [])
    ]


def _program_ownership(
    *,
    populace_treatment: str,
    stages: Sequence[Mapping[str, object]],
) -> str:
    operation_kinds = {
        str(operation.get("kind"))
        for stage in stages
        for operation in stage.get("operations", [])
        if isinstance(operation, Mapping)
    }
    if not stages:
        if populace_treatment == "seed":
            return "modeled"
        if populace_treatment == "rate_unsourced":
            return "engine"
        raise RuntimeError(
            "A take-up program without a source stage must be either seeded by "
            "the generic authority or deliberately left at the engine default; "
            f"got treatment {populace_treatment!r}."
        )
    if {
        "derive_housing_tenure_inputs",
        "impute_housing_assistance_to_puf_support",
    } <= operation_kinds:
        return "mixed"
    if "fit_weighted_qrf" in operation_kinds:
        return "transferred"
    if "derive_medicare_take_up" in operation_kinds:
        return "measured"
    return "modeled"


def _seeded_program_step(
    *,
    variable: str,
    entity: str,
    rate: Mapping[str, object],
) -> dict[str, object]:
    from microcosm.build.us_runtime import take_up as take_up_runtime

    step: dict[str, object] = {
        "kind": "probability_seed",
        "operation_id": _callable_name(take_up_runtime.with_us_take_up_inputs),
        "kernel": "kernel:seeded_rate",
        "entity": entity,
        "output": variable,
        "rate": deepcopy(dict(rate)),
        "assignment": {
            "comparison": "stable_uniform_draw_less_than_rate",
            "hash": "blake2b",
            "digest_size_bytes": 8,
            "byte_order": "big",
            "uniform_denominator": 2**64,
            "message_fields_in_order": ["seed", "variable", "stable_unit_key"],
            "message_separator": ":",
            "stable_unit_key_precedence": [
                "support_source_id",
                ["source_year", "source_household_id", "source_person_id"],
                f"{entity}_id",
            ],
            "seed": {"source": "build_config", "argument": "seed"},
            "time_period_argument": "accepted_but_current_rates_are_fixed_vintages",
            "existing_assembled_values": "fill_missing_only",
        },
        "signal_gate_share_bounds": list(
            take_up_runtime.US_TAKE_UP_SHARE_BAND[variable]
        ),
    }
    if "values_by_num_children" in rate:
        step["rate_selector"] = {
            "kind": "approximated_eitc_qualifying_children",
            "person_age_column": "age",
            "membership_column": "person_tax_unit_id",
            "child_age_comparison": "age < max_age_exclusive",
            "max_age_exclusive": int(take_up_runtime._EITC_QUALIFYING_CHILD_MAX_AGE),
            "bins": ["0", "1", "2", "3+"],
            "student_under_24_extension": False,
        }
    return step


def _typed_source_stage_step(
    operation: Mapping[str, object],
) -> dict[str, object]:
    """Normalize one legacy source operation into the closed step algebra."""

    operation_id = str(operation.get("kind"))
    try:
        kind, kernel_id = _SOURCE_TAKE_UP_STEP_BINDINGS[operation_id]
    except KeyError as error:
        raise RuntimeError(
            f"Take-up source operation {operation_id!r} has no reviewed typed "
            "step binding."
        ) from error
    if operation.get("rate_target_role") == "ssa_ssi_age_band_recipients":
        if operation_id != "assign_binary_from_rate":
            raise RuntimeError(
                "The SSI target-derived prior must remain an "
                "assign_binary_from_rate source operation."
            )
        kind, kernel_id = "probability_seed", "ssi_probability_seed"
    return {"kind": kind, "kernel": f"kernel:{kernel_id}"}


def _source_stage_steps(
    *,
    variable: str,
    stages: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Project exact take-up operations with stable source locations."""

    steps: list[dict[str, object]] = []
    has_ssi_probability_seed = False
    for stage in stages:
        operations = stage.get("operations", [])
        housing_mixed_stage = any(
            isinstance(operation, Mapping)
            and operation.get("kind") == "impute_housing_assistance_to_puf_support"
            for operation in operations
        )
        for index, operation in enumerate(operations):
            if not isinstance(operation, Mapping):
                raise RuntimeError(
                    f"Source stage {stage.get('stage')!r} has a non-object operation."
                )
            kind = str(operation.get("kind"))
            if kind in {"read_table", "read_tables"}:
                continue
            if housing_mixed_stage:
                operation_text = json.dumps(operation, sort_keys=True)
                if variable not in operation_text and "take_up" not in operation_text:
                    continue
            step = _typed_source_stage_step(operation)
            has_ssi_probability_seed = has_ssi_probability_seed or (
                operation.get("rate_target_role")
                == "ssa_ssi_age_band_recipients"
            )
            step["source_operation_ref"] = {
                "stage": str(stage["stage"]),
                "operation_index": index,
                "operation_id": kind,
            }
            steps.append(step)

    if has_ssi_probability_seed:
        from microcosm.build.us_runtime import ssi_take_up

        steps.append(
            {
                "kind": "delivery_gate",
                "operation_id": _callable_name(
                    ssi_take_up.us_ssi_take_up_delivery_gate
                ),
                "kernel": "kernel:ssi_delivery_gate",
                "phase": ssi_take_up.US_SSI_TAKE_UP_PHASE_RELEASE_FINAL,
                "target_table": ssi_take_up.US_SSI_TAKE_UP_TARGET_TABLE_NAME,
                "target_period": ssi_take_up._TARGET_PERIOD,
                "target_measure": ssi_take_up._TARGET_MEASURE,
                "candidate_definition": ssi_take_up._CANDIDATE_DEFINITION,
                "assignment_prior_basis_default": (
                    ssi_take_up.US_SSI_TAKE_UP_PRIOR_BASIS_CURRENT_FRAME
                ),
                "assignment_prior_basis_optional": (
                    ssi_take_up.US_SSI_TAKE_UP_PRIOR_BASIS_RELEASE_ARTIFACT
                ),
                "enforced_age_bands": list(
                    ssi_take_up.US_SSI_TAKE_UP_ENFORCED_BAND_KEYS
                ),
                "relative_tolerance": (
                    ssi_take_up.US_SSI_TAKE_UP_BAND_DELIVERY_RELATIVE_TOLERANCE
                ),
                "unenforced_age_bands": [
                    target.key
                    for target in ssi_take_up.US_SSI_TAKE_UP_AGE_TARGETS
                    if target.key not in ssi_take_up.US_SSI_TAKE_UP_ENFORCED_BAND_KEYS
                ],
            }
        )
    return steps


def _mixed_take_up_segments(steps: Sequence[Mapping[str, object]]) -> list[object]:
    measured = [
        index
        for index, step in enumerate(steps)
        if step.get("kind") == "measured_map"
        and _mapping_like(
            step.get("source_operation_ref"), "measured source operation ref"
        ).get("operation_id")
        == "derive_housing_tenure_inputs"
    ]
    transferred = [
        index
        for index, step in enumerate(steps)
        if step.get("kind") == "imputed_transfer"
        and _mapping_like(
            step.get("source_operation_ref"), "transferred source operation ref"
        ).get("operation_id")
        == "impute_housing_assistance_to_puf_support"
    ]
    if len(measured) != 1 or len(transferred) != 1:
        raise RuntimeError(
            "Mixed housing take-up authority must have one measured and one "
            "transferred segment."
        )
    return [
        {
            "row_scope": "asec_rows",
            "ownership": "measured",
            "pipeline": [deepcopy(dict(steps[index])) for index in measured],
            "final_owner_stage": str(
                _mapping_like(
                    steps[measured[-1]].get("source_operation_ref"),
                    "measured source operation ref",
                )["stage"]
            ),
        },
        {
            "row_scope": "puf_support_rows",
            "ownership": "transferred",
            "pipeline": [deepcopy(dict(steps[index])) for index in transferred],
            "final_owner_stage": str(
                _mapping_like(
                    steps[transferred[-1]].get("source_operation_ref"),
                    "transferred source operation ref",
                )["stage"]
            ),
        },
    ]


def _take_up_review_target(
    *,
    steps: Sequence[dict[str, object]],
    ownership: str,
    source_stages: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Select the typed step that owns non-source compatibility evidence.

    Source-stage operation parameters stay on the copied typed operation row.
    This selector only attaches reviewed facts which that operation does not
    contain (for example, the provenance status of SNAP's source rate).  It
    therefore cannot create a second authored copy of a source-stage value.
    """

    if ownership == "engine":
        return None
    delivery_seed = next(
        (
            step
            for step in steps
            if step.get("kind") == "probability_seed"
            and any(candidate.get("kind") == "delivery_gate" for candidate in steps)
        ),
        None,
    )
    if delivery_seed is not None:
        return delivery_seed
    if len(steps) == 2 and steps[-1].get("kind") == "count_calibration":
        return steps[-1]
    source_rate = next(
        (
            step
            for step in steps
            if "take_up_rate"
            in (
                _referenced_source_operation(step, source_stages=source_stages)
                or step
            )
        ),
        None,
    )
    if source_rate is not None:
        return source_rate
    assignment = next(
        (
            step
            for step in steps
            if step.get("kind") == "assignment"
            and "rate_key"
            in (
                _referenced_source_operation(step, source_stages=source_stages)
                or step
            )
        ),
        None,
    )
    if assignment is not None:
        return assignment
    return None


def _referenced_source_operation(
    step: Mapping[str, object],
    *,
    source_stages: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    reference_value = step.get("source_operation_ref")
    if reference_value is None:
        return None
    reference = _mapping_like(reference_value, "take-up source operation ref")
    stage_id = str(reference.get("stage"))
    matches = [stage for stage in source_stages if stage.get("stage") == stage_id]
    if len(matches) != 1:
        raise RuntimeError(
            f"Take-up source operation stage {stage_id!r} must resolve exactly once."
        )
    operations = matches[0].get("operations")
    if not isinstance(operations, list):
        raise RuntimeError(f"Source stage {stage_id!r} has no operations array.")
    operation_index = reference.get("operation_index")
    if not isinstance(operation_index, int) or isinstance(operation_index, bool):
        raise RuntimeError("Take-up source operation index must be an integer.")
    try:
        operation = operations[operation_index]
    except IndexError as error:
        raise RuntimeError(
            f"Take-up source operation index {operation_index} is out of range "
            f"for stage {stage_id!r}."
        ) from error
    if not isinstance(operation, Mapping):
        raise RuntimeError("Referenced take-up source operation is not an object.")
    if operation.get("kind") != reference.get("operation_id"):
        raise RuntimeError(
            "Take-up source operation id differs from its referenced operation."
        )
    return operation


def _attach_take_up_review_evidence(
    *,
    steps: Sequence[dict[str, object]],
    ownership: str,
    source_stages: Sequence[Mapping[str, object]],
    legacy_rate: Mapping[str, object],
    legacy_calibration: Mapping[str, object] | None,
) -> None:
    """Attach only compatibility facts not already owned by a typed step."""

    rate = deepcopy(dict(legacy_rate))
    effective_steps = [
        _referenced_source_operation(step, source_stages=source_stages) or step
        for step in steps
    ]
    if ownership in {"measured", "transferred", "mixed"}:
        expected = {"status": "not_used_measured_source"}
        if rate != expected:
            raise RuntimeError(
                "Measured/transferred take-up rate review changed: "
                f"expected={expected!r}, actual={rate!r}."
            )
    elif ownership == "engine":
        engine_steps = [step for step in steps if step.get("kind") == "engine_default"]
        if len(engine_steps) != 1:
            raise RuntimeError("Engine-owned take-up must have one engine-default step.")
        debt = _mapping_like(engine_steps[0].get("debt"), "take-up engine debt")
        if _json_ready(debt.get("rate_review")) != _json_ready(rate):
            raise RuntimeError(
                "Engine-default debt and reviewed take-up rate evidence differ."
            )
    elif any(step.get("rate") == rate for step in effective_steps):
        # Generic TANF/EITC seeds already own the complete reviewed rate.
        pass
    else:
        target = _take_up_review_target(
            steps=steps,
            ownership=ownership,
            source_stages=source_stages,
        )
        if target is None:
            raise RuntimeError("Take-up rate evidence has no typed owning step.")
        source_operation = _referenced_source_operation(
            target, source_stages=source_stages
        )
        source_rate = (
            source_operation.get("take_up_rate")
            if source_operation is not None
            else target.get("take_up_rate")
        )
        if isinstance(source_rate, Mapping):
            residual = deepcopy(rate)
            for key, value in source_rate.items():
                if residual.pop(str(key), object()) != value:
                    raise RuntimeError(
                        "Take-up source-stage rate and reviewed evidence differ at "
                        f"{key!r}."
                    )
            if not residual:
                raise RuntimeError(
                    "A source-stage rate review must not duplicate the source row."
                )
            target["rate_review"] = residual
        else:
            target["rate_review"] = rate

    if legacy_calibration is None:
        return
    calibration = deepcopy(dict(legacy_calibration))
    if any(step.get("kind") == "delivery_gate" for step in steps):
        seed = next(step for step in steps if step.get("kind") == "probability_seed")
        effective_seed = (
            _referenced_source_operation(seed, source_stages=source_stages) or seed
        )
        delivery = next(step for step in steps if step.get("kind") == "delivery_gate")
        derived = {
            "age_bands": effective_seed.get("age_bands"),
            "target_measure": effective_seed.get("target_measure"),
            "target_period": effective_seed.get("target_period"),
            "target_role": effective_seed.get("rate_target_role"),
            "target_source": effective_seed.get("target_source"),
            "target_table": delivery.get("target_table"),
            "targets": [delivery.get("target_table")],
        }
        # The frozen inventory named the source column while the executable
        # predicate is ``SSI_VAL > 0``.  Keep that reviewed semantic as a
        # typed calibration fact rather than parsing an expression string.
        seed["anchor_column"] = calibration.pop("anchor")
        for key, value in derived.items():
            if calibration.pop(key, object()) != value:
                raise RuntimeError(
                    f"SSI calibration evidence differs from typed steps at {key!r}."
                )
        seed["calibration_review"] = calibration
        return

    calibration_steps = [
        step for step in steps if step.get("kind") == "count_calibration"
    ]
    if not calibration_steps:
        raise RuntimeError("Reviewed take-up calibration has no typed calibration step.")
    step = calibration_steps[-1]
    effective_step = (
        _referenced_source_operation(step, source_stages=source_stages) or step
    )
    derived = {
        "anchor": effective_step.get("preserve_true_anchor"),
        "targets": effective_step.get("targets"),
    }
    for key, value in derived.items():
        if calibration.pop(key, object()) != value:
            raise RuntimeError(
                f"Take-up calibration evidence differs from typed step at {key!r}."
            )
    step["calibration_review"] = calibration


def project_legacy_take_up_contract(
    typed_contract: Mapping[str, object],
    *,
    sources_document: Mapping[str, object],
) -> dict[str, object]:
    """Reconstruct the frozen legacy contract from typed fields and ABI."""

    from microcosm.build.spec_engine.take_up_semantics import (
        project_legacy_take_up_contract as project_from_typed_authorities,
    )
    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

    engine_abi = PolicyEngineUSEngine().take_up_contract()
    programs = typed_contract.get("programs")
    if not isinstance(programs, list):
        raise RuntimeError("Typed take-up contract needs programs.")
    lock_programs: dict[str, object] = {}
    for row in programs:
        if not isinstance(row, Mapping):
            raise RuntimeError("Typed take-up program is not an object.")
        program_id = row.get("id")
        variable = row.get("variable")
        if not isinstance(program_id, str) or not isinstance(variable, str):
            raise RuntimeError("Typed take-up program needs id and variable.")
        try:
            abi = engine_abi[variable]
        except KeyError as error:
            raise RuntimeError(
                f"Typed take-up variable {variable!r} is absent from engine ABI."
            ) from error
        lock_programs[program_id] = {
            "variable": variable,
            "entity": str(abi["entity"]),
            "value_type": str(abi["value_type"]),
            "default": _json_ready(abi["default"]),
            "engine_class": str(abi["engine_class"]),
            "consumers": list(abi.get("consumers", [])),
        }
    return project_from_typed_authorities(
        typed_contract,
        engine_abi_lock={"programs": lock_programs},
        sources_document=sources_document,
    )


def project_legacy_calibration_contract(
    typed_contract: Mapping[str, object],
) -> dict[str, object]:
    """Restore the constants-era solver summary from normalized contracts."""

    from microcosm.build.spec_engine.calibration_semantics import (
        project_legacy_calibration_contract as project_from_normalized_contract,
    )

    return project_from_normalized_contract(typed_contract)


@lru_cache(maxsize=1)
def _build_take_up_contract_cached() -> dict[str, object]:
    from microcosm.build.spec_engine.engine_abi import (
        scoped_take_up_manifest_program_bindings,
    )
    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

    raw_resource = json.loads(
        files("microcosm.build.us")
        .joinpath(_TAKE_UP_CONTRACT_RESOURCE)
        .read_text(encoding="utf-8")
    )
    if not isinstance(raw_resource, Mapping):
        raise RuntimeError("US take-up compatibility resource is not an object.")
    raw_programs = raw_resource.get("programs")
    if not isinstance(raw_programs, list):
        raise RuntimeError("US take-up compatibility programs are not an array.")
    bootstrap_bindings = tuple(
        (
            str(row["variable"]),
            str(row["entity"]),
            str(row["populace_treatment"]),
        )
        for row in raw_programs
        if isinstance(row, Mapping)
    )
    if len(bootstrap_bindings) != _TAKE_UP_PROGRAM_COUNT:
        raise RuntimeError("US take-up bootstrap bindings are incomplete.")
    with scoped_take_up_manifest_program_bindings(bootstrap_bindings):
        from microcosm.build.us_runtime.take_up_contract import (
            assert_take_up_contract_current,
            assert_take_up_treatments_consistent,
            load_legacy_take_up_contract_evidence,
            take_up_contract_identity,
        )

    engine = PolicyEngineUSEngine()
    contract = load_legacy_take_up_contract_evidence()
    assert_take_up_contract_current(engine, contract=contract)
    assert_take_up_treatments_consistent(contract=contract)
    legacy_identity = take_up_contract_identity(contract)
    if legacy_identity["resource_sha256"] != contract_sha256(raw_resource):
        raise RuntimeError(
            "US take-up compatibility resource differs from the validated "
            "contract identity."
        )
    engine_abi = engine.take_up_contract()
    source_authority = _source_manifest_authority()
    all_source_stages = source_authority["stages"]
    if not isinstance(all_source_stages, list):  # pragma: no cover - guarded above
        raise RuntimeError("US source-stage projection lost its stage list.")

    contract_variables = [program.variable for program in contract.programs]
    if len(contract_variables) != _TAKE_UP_PROGRAM_COUNT:
        raise RuntimeError(
            "Current take-up contract count changed: "
            f"{len(contract_variables)} != {_TAKE_UP_PROGRAM_COUNT}; review the "
            "typed ownership and step extraction before regenerating the bundle."
        )
    if set(contract_variables) != set(engine_abi):
        raise RuntimeError(
            "Take-up checked-in contract and fresh engine ABI variable sets differ."
        )

    program_id_by_variable = {
        variable: program_id for program_id, variable in _TAKE_UP_PROGRAM_VARIABLES
    }
    if (
        len(program_id_by_variable) != len(_TAKE_UP_PROGRAM_VARIABLES)
        or len({program_id for program_id, _ in _TAKE_UP_PROGRAM_VARIABLES})
        != len(_TAKE_UP_PROGRAM_VARIABLES)
        or set(program_id_by_variable) != set(contract_variables)
    ):
        raise RuntimeError(
            "Reviewed take-up program-to-variable mapping is not a total and "
            "injective projection of the checked-in contract."
        )
    program_ids = [program_id_by_variable[variable] for variable in contract_variables]

    programs: list[dict[str, object]] = []
    relevant_stage_names: set[str] = set()
    for program_id, program in zip(program_ids, contract.programs, strict=True):
        raw_contract = deepcopy(dict(program.raw))
        legacy_fields = {
            key: deepcopy(value)
            for key, value in raw_contract.items()
            if key not in _LEGACY_ENGINE_FACT_FIELDS
        }
        legacy_treatment = legacy_fields.pop("populace_treatment")
        legacy_rate = _mapping_like(
            legacy_fields.pop("rate"),
            f"take-up program {program_id!r} reviewed rate",
        )
        legacy_calibration_value = legacy_fields.pop("calibration", None)
        legacy_calibration = (
            _mapping_like(
                legacy_calibration_value,
                f"take-up program {program_id!r} reviewed calibration",
            )
            if legacy_calibration_value is not None
            else None
        )
        legacy_seed_method = legacy_fields.pop("seed_method", None)
        documentation = {
            key: legacy_fields.pop(key)
            for key in tuple(legacy_fields)
            if key in _TAKE_UP_DOCUMENTATION_FIELDS
        }
        if legacy_fields:
            raise RuntimeError(
                f"Take-up program {program_id!r} has untyped compatibility "
                f"fields {sorted(legacy_fields)!r}."
            )
        stages = _program_source_stages(program.variable, all_source_stages)
        relevant_stage_names.update(str(stage["stage"]) for stage in stages)
        ownership = _program_ownership(
            populace_treatment=program.populace_treatment,
            stages=stages,
        )
        if stages:
            steps = _source_stage_steps(variable=program.variable, stages=stages)
        elif program.populace_treatment == "seed":
            steps = [
                _seeded_program_step(
                    variable=program.variable,
                    entity=program.entity,
                    rate=program.rate,
                )
            ]
        else:
            steps = [
                {
                    "kind": "engine_default",
                    "operation_id": "legacy_seeded_input_default",
                    "kernel": "kernel:engine_default",
                    "debt": {
                        "populace_treatment": program.populace_treatment,
                        "rate_review": deepcopy(raw_contract.get("rate", {})),
                    },
                }
            ]
        if not steps:
            raise RuntimeError(f"Take-up program {program_id!r} has no typed steps.")
        _attach_take_up_review_evidence(
            steps=steps,
            ownership=ownership,
            source_stages=all_source_stages,
            legacy_rate=legacy_rate,
            legacy_calibration=legacy_calibration,
        )
        row: dict[str, object] = {
            "id": program_id,
            "variable": program.variable,
            "ownership": ownership,
            "documentation": documentation,
        }
        if ownership == "mixed":
            row["segments"] = _mixed_take_up_segments(steps)
        else:
            row["pipeline"] = steps
            row["final_owner_stage"] = (
                str(stages[-1]["stage"])
                if stages
                else (
                    "policyengine_us_default"
                    if ownership == "engine"
                    else "generic_take_up_seeder"
                )
            )
        if program.populace_treatment == "seed":
            row["dependence"] = {"group": "generic_seeder_batch"}
        projected_treatment = (
            "rate_unsourced" if ownership == "engine" else legacy_treatment
        )
        if legacy_treatment != projected_treatment:
            raise RuntimeError(
                f"Take-up program {program_id!r} treatment review changed."
            )
        if legacy_seed_method is not None:
            expected_seed_method = (
                "calibrated_bernoulli_by_children"
                if any("rate_selector" in step for step in steps)
                else "calibrated_bernoulli"
            )
            if legacy_seed_method != expected_seed_method:
                raise RuntimeError(
                    f"Take-up program {program_id!r} seed method differs from "
                    "its typed probability step."
                )
        programs.append(row)

    ownership_counts = Counter(str(row["ownership"]) for row in programs)
    expected_ownership_counts = {
        "engine": 4,
        "measured": 1,
        "mixed": 1,
        "modeled": 6,
        "transferred": 1,
    }
    if dict(sorted(ownership_counts.items())) != expected_ownership_counts:
        raise RuntimeError(
            "Take-up ownership extraction changed; review before generation: "
            f"{dict(sorted(ownership_counts.items()))!r}."
        )

    declared_stage_names = {
        str(stage["stage"]) for stage in all_source_stages if isinstance(stage, Mapping)
    }
    if not relevant_stage_names <= declared_stage_names:
        raise RuntimeError("Take-up source-stage references are not closed.")
    result = {
        "contract_id": "us_take_up_runtime_authority",
        "version": contract.version,
        "country": contract.country,
        "scope_registry": take_up_scope_registry_wire(),
        "programs": programs,
        "legacy_contract_metadata": {
            "policy": deepcopy(raw_resource["policy"]),
            "asserted_engine": deepcopy(raw_resource["asserted_engine"]),
            "doctrine": deepcopy(raw_resource["doctrine"]),
        },
    }
    projected = project_legacy_take_up_contract(
        result,
        sources_document={"stages": all_source_stages},
    )
    if projected != _json_ready(raw_resource):
        raise RuntimeError(
            "Typed take-up contract does not reconstruct the frozen legacy "
            "resource exactly."
        )
    if legacy_identity["resource_sha256"] != contract_sha256(projected):
        raise RuntimeError("Typed take-up projection changed its legacy identity.")
    return result


def build_take_up_contract() -> dict[str, object]:
    """Build all 13 take-up ownership, operation, ABI, and receipt rows."""

    return deepcopy(_build_take_up_contract_cached())


def build_take_up() -> dict[str, object]:
    """Short generator-facing alias for :func:`build_take_up_contract`."""

    return build_take_up_contract()


def derive_battery_registry_views(
    document: Mapping[str, object],
) -> dict[str, object]:
    return _derive_battery_registry_views(document)


def project_battery_legacy_contract(
    document: Mapping[str, object],
    *,
    authority_receipt: Mapping[str, object],
) -> dict[str, object]:
    return _project_battery_legacy_contract(
        document,
        authority_receipt=authority_receipt,
    )


def build_battery_contract() -> dict[str, object]:
    """Build and validate the live stacked battery migration document."""

    result = build_live_stacked_battery_contract()
    stacked = importlib.import_module("microcosm.build.us_runtime.stacked_spine")
    authority_receipt = _json_ready(stacked.stacked_spine_authority_receipt())
    if not isinstance(authority_receipt, Mapping):  # pragma: no cover - live invariant
        raise RuntimeError("Stacked authority receipt must be an object.")
    derived = derive_battery_registry_views(result)
    if derived["declared_surface"] != _json_ready(
        stacked.CANONICAL_STACKED_DECLARED_SURFACE
    ):
        raise RuntimeError(
            "Battery metric registry no longer derives the canonical surface."
        )
    project_battery_legacy_contract(
        result,
        authority_receipt=authority_receipt,
    )
    return result


def build_battery() -> dict[str, object]:
    """Short generator-facing alias for :func:`build_battery_contract`."""

    return build_battery_contract()


def _calibration_loss_weighting(release: Any) -> dict[str, object]:
    return {
        "id": release.US_FISCAL_TARGET_LOSS_WEIGHTING,
        "target_value_basis": {
            "count_when": [
                "measure_mode in {indicator_sum, less_than_indicator_sum}",
                "source_measure_id contains enrollment or recipients",
                "source_measure_id contains both return and count",
            ],
            "otherwise": "amount",
        },
        "raw_value_weight": {
            "formula": "max(abs(target_value), 1) ** value_power",
            "value_power": float(release.US_FISCAL_TARGET_VALUE_WEIGHT_POWER),
        },
        "within_basis_normalization": "divide_by_basis_mean",
        "concept_budget": {
            "group_budget": "maximum_member_raw_weight",
            "allocation": "proportional_member_weights_sum_to_group_budget",
            "metadata_exclusions": sorted(
                release.US_FISCAL_TARGET_CONCEPT_METADATA_EXCLUSIONS
            ),
        },
        "objective_basis_budget": {
            "amount_share": 0.5,
            "count_share": 0.5,
            "implementation": "equal_total_weight_per_present_basis",
        },
        "target_family_multipliers": {
            "default": {},
            "application_order": "sorted_family_id",
            "post_multiplier_normalization": "mean_one",
        },
        "final_normalization": "mean_one",
    }


def _calibration_tail_contracts(release: Any) -> dict[str, object]:
    tail = importlib.import_module("microcosm.build.us_runtime.puf_capital_gains_tail")
    return {
        "puf_capital_gains_tail": {
            "support_contract_ref": deepcopy(_PUF_TAIL_SUPPORT_REF),
            "execution_binding_ref": deepcopy(_PUF_TAIL_EXECUTION_REF),
            "post_selection_gate": _kernel_ref(
                tail.assert_puf_capital_gains_tail_survives_selection
            ),
        },
        "qrf_imputed_export_tail_concentration": {
            "kernel": "kernel:tail_concentration_gate",
            "top_k": int(release.US_QRF_TAIL_CONCENTRATION_TOP_K),
            "max_top_share": float(release.US_QRF_TAIL_CONCENTRATION_MAX_TOP_SHARE),
            "min_nonzero_records": int(
                release.US_QRF_TAIL_CONCENTRATION_MIN_NONZERO_RECORDS
            ),
            "sparse_nonzero_share_max": float(release.US_QRF_SPARSE_NONZERO_SHARE_MAX),
            "qrf_imputed_outputs": sorted(release._qrf_imputed_source_outputs()),
            "reviewed_exclusions_default": {},
        },
    }


def _require_contract_ref(
    value: object,
    expected: Mapping[str, str],
    *,
    location: str,
) -> None:
    reference = _mapping_like(value, location)
    if dict(reference) != dict(expected):
        raise RuntimeError(
            f"{location} must be the reviewed typed reference; "
            f"expected={dict(expected)!r}, actual={dict(reference)!r}."
        )


def resolve_calibration_tail_contracts(
    document: Mapping[str, object],
    *,
    spine_document: Mapping[str, object],
    imputation_document: Mapping[str, object],
) -> dict[str, object]:
    """Resolve calibration tail references into the constants-era projection."""

    tail_contracts = deepcopy(
        dict(_mapping_like(document["tail_contracts"], "calibration tail contracts"))
    )
    puf = dict(
        _mapping_like(
            tail_contracts["puf_capital_gains_tail"],
            "calibration PUF capital-gains tail",
        )
    )
    _require_contract_ref(
        puf.pop("support_contract_ref"),
        _PUF_TAIL_SUPPORT_REF,
        location="calibration tail support_contract_ref",
    )
    _require_contract_ref(
        puf.pop("execution_binding_ref"),
        _PUF_TAIL_EXECUTION_REF,
        location="calibration tail execution_binding_ref",
    )

    support_roles = [
        _mapping_like(value, "spine support role")
        for value in _array_like(spine_document["support_roles"], "spine support roles")
        if _mapping_like(value, "spine support role").get("id")
        == _PUF_TAIL_SUPPORT_REF["support_role"]
    ]
    if len(support_roles) != 1:
        raise RuntimeError("Calibration tail support reference must resolve once.")
    tail_support = _mapping_like(
        support_roles[0].get("tail_support"), "spine PUF tail support"
    )

    graph = _mapping_like(
        imputation_document["producer_graph"], "imputation producer graph"
    )
    primary_nodes = [
        _mapping_like(value, "imputation producer node")
        for value in _array_like(graph["nodes"], "imputation producer nodes")
        if _mapping_like(value, "imputation producer node").get("name")
        == _PUF_TAIL_EXECUTION_REF["producer"]
    ]
    if len(primary_nodes) != 1:
        raise RuntimeError("Calibration tail execution producer must resolve once.")
    resources = [
        _mapping_like(value, "imputation virtual resource")
        for value in _array_like(
            primary_nodes[0]["virtual_resources"],
            "primary PUF virtual resources",
        )
        if _mapping_like(value, "imputation virtual resource").get("id")
        == _PUF_TAIL_EXECUTION_REF["resource"]
    ]
    if len(resources) != 1:
        raise RuntimeError("Calibration tail execution resource must resolve once.")
    binding = _mapping_like(resources[0].get("binding"), "primary PUF binding")
    execution_tail = _mapping_like(
        binding.get("capital_gains_tail"), "primary PUF capital-gains tail"
    )
    soi = _mapping_like(
        execution_tail.get("soi_e19200_agi_bands"),
        "primary PUF SOI E19200 inputs",
    )
    runtime_agi_bands = dict(
        _mapping_like(soi.get("runtime_agi_bands"), "runtime SOI AGI bands")
    )
    runtime_sha256 = contract_sha256(runtime_agi_bands)
    if soi.get("agi_bands") != runtime_agi_bands.get("agi_bands"):
        raise RuntimeError("SOI parsed and runtime AGI-band rows differ.")
    puf.update(
        {
            "support_contract": deepcopy(tail_support["legacy_contract"]),
            "disaggregation_spec": deepcopy(execution_tail["spec"]),
            "concentration_controls": deepcopy(
                execution_tail["concentration_gate"]
            ),
            "soi_e19200_agi_bands": {
                "agi_bands": deepcopy(soi["agi_bands"]),
                "all_returns": deepcopy(soi["all_returns"]),
                "asset": soi["asset"],
                "asset_sha256": soi["asset_sha256"],
                "runtime_schema_version": runtime_agi_bands["schema_version"],
                "runtime_sha256": runtime_sha256,
            },
        }
    )
    tail_contracts["puf_capital_gains_tail"] = puf
    return tail_contracts


def _fiscal_release_cli_defaults(release: Any) -> dict[str, object]:
    """Read solver defaults from the current release parser without running it."""

    args = release._parse_args(
        [
            "--ledger-facts",
            "__contract_projection_ledger_facts__",
            "--out",
            "__contract_projection_output__",
        ]
    )
    if any(
        value is not None
        for value in (args.exact_k, args.exact_k_pi_hi, args.refit_l2_lambda)
    ):
        raise RuntimeError(
            "The default fiscal release parse unexpectedly selected exact-k or "
            "a refit-only L2 override."
        )
    return {
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "max_weight_ratio": float(args.max_weight_ratio),
        "l0_refit_lambda_share": float(args.l0_refit_lambda_share),
        "l2_lambda": float(args.l2_lambda),
        "seed": int(args.seed),
        "target_family_loss_multipliers": dict(args.target_family_loss_multipliers),
        "include_congressional_district_targets": bool(
            args.include_congressional_district_targets
        ),
        "gate_congressional_district_targets": bool(
            args.gate_congressional_district_targets
        ),
    }


def build_calibration_contract() -> dict[str, object]:
    """Build the fully resolved current US fiscal calibration contract."""

    release = importlib.import_module("tools.build_us_fiscal_refresh_release")
    solve = importlib.import_module("microcosm.calibrate.solve")
    gates = importlib.import_module("microcosm.calibrate.gates")
    torch = importlib.import_module("torch")
    ladder = importlib.import_module("microcosm.build.us_runtime.exact_k_ladder")
    cli = _fiscal_release_cli_defaults(release)
    epochs = int(cli["epochs"])
    learning_rate = float(cli["learning_rate"])
    max_weight_ratio = float(cli["max_weight_ratio"])
    l0_refit_lambda_share = float(cli["l0_refit_lambda_share"])
    l2_lambda = float(cli["l2_lambda"])
    legacy_seed = int(cli["seed"])

    loss_weights = _calibration_loss_weighting(release)
    family_multipliers = dict(cli["target_family_loss_multipliers"])
    loss_weights["target_family_multipliers"]["default"] = family_multipliers
    target_loss_scales = {
        "kind": "default_target",
        "formula": "max(abs(target), 1)",
    }
    loss_contract = {
        "formula_id": str(release.US_FISCAL_TARGET_LOSS_WEIGHTING),
        "params": {
            "formula": (
                "weighted_mean(min(abs((A@w-target)/target_scale), target_loss_cap))"
            ),
            "target_scaling": target_loss_scales,
            "target_loss_cap": float(release.US_FISCAL_TARGET_LOSS_CAP),
            "weighting": loss_weights,
        },
    }
    adam_arguments = _adam_arguments(torch.optim.Adam, learning_rate)
    return {
        "contract_id": "us_fiscal_calibration_runtime",
        "version": 1,
        "targets": {
            # Approved RFC v3 grammar literal.  The compiler owns the mapping
            # to the constants-era ledger-consumer module name.
            "source": "chronicle_facts",
            "facts_sha256": {"source": "run_request", "default": None},
            "manifest_sha256": {"source": "run_request", "default": None},
            "geography_layers": ["national", "state"],
            "cd_policy": "always_present_report_attainment",
            "default_geography_layers": ["national", "state"],
            "congressional_district": {
                "included_by_default": bool(
                    cli["include_congressional_district_targets"]
                ),
                "gate_by_default": bool(cli["gate_congressional_district_targets"]),
                "requires_vintage_crosswalk": True,
            },
            "county": {"included": False},
            "negative_target_policy": "require_target_spec_signed_true",
            "zero_target_policy": "retain_with_scale_one",
            "matrix": {
                "format": "scipy_csr_array",
                "weight_entity": "household",
                "person_target_projection": "sum_members_to_weight_entity",
                "multi_period_rule": "one_weight_per_trajectory",
                "uncompilable_target": "skip_with_reason",
                "all_targets_uncompilable": "refuse",
            },
        },
        "solver": {
            "kernel": "kernel:calibrate_l0_refit",
            "loss": loss_contract,
            "row_weighting": "household_calibration_weight",
            # Keep the approved binding fields and their types intact.  The
            # fully resolved extensions live in closed sibling contracts.
            "optimizer": {
                "name": "adam",
                "dtype": "float32",
                "kernel": "kernel:torch_adam",
                "parameterization": "log_weights",
                "returned_weight_dtype": "float64",
                "schedule": {
                    "kind": "constant_learning_rate",
                    "learning_rate": learning_rate,
                    "optimizer_steps_per_epoch": 1,
                    "warmup_epochs": 0,
                    "decay": "none",
                },
                "arguments": adam_arguments,
            },
            "l0": {"mode": "exact_k_projection"},
            "initialization_contract": {
                "policy_id": "resolved_weights_optional_validated_warm_start",
                "weights": "resolved_household_weights",
                "warm_start_default": None,
                "warm_start_source": "validated_calibration_npz",
                "l0_gate_init_mean": 0.999,
                "l0_gate_temperature": 0.25,
                "hard_concrete_gamma": float(gates._GAMMA),
                "hard_concrete_zeta": float(gates._ZETA),
                "training_uniform_lower_exclusive": 1e-8,
                "training_uniform_upper_exclusive": 1.0 - 1e-8,
                "positive_gate_open_probability_threshold": float(
                    gates.hard_concrete_open_probability_threshold(0.25)
                ),
                "rng": "kernel:torch_manual_seed",
            },
            "stopping_contract": {
                "kind": "fixed_epochs",
                "max_epochs": epochs,
                "early_stopping": False,
                "tolerance": None,
                "patience": None,
                "null_fields_reason": (
                    "current_solver_has_no_early_stopping_or_tolerance_check"
                ),
            },
            "hard_constraints": {
                "mass": solve.CONSERVE_MASS,
                "mass_total": "input_weight_total",
                "max_weight_ratio": max_weight_ratio,
                "per_step_order": ["ratio_cap", "mass_projection", "ratio_recap"],
                "closing_projection": "float64_cap_and_mass_redistribution",
                "pruned_records_may_absorb_mass": False,
            },
            "infeasibility_contract": {
                "structural_cap_mass_or_empty_support": "refuse",
                "soft_target_miss": "report_and_continue",
            },
            "target_priority_contract": {
                "policy_id": "single_weighted_objective_no_lexicographic_priority",
                "ordering": "compiled_target_registry_order",
                "priority_groups": [],
                "empty_groups_reason": (
                    "current_authority_uses_loss_weights_not_priority_tiers"
                ),
            },
            "l0_contract": {
                "default_mode": "fixed_l0_then_refit",
                "l0_refit_lambda_share": l0_refit_lambda_share,
                "l0_lambda_formula": ("l0_refit_lambda_share / n_candidate_households"),
                "target_records": None,
                "prune_relative_atol": float(solve._PRUNE_REL_ATOL),
                "budget_search": {
                    "used_by_default": False,
                    "lower": float(solve._L0_SEARCH_LO),
                    "upper": float(solve._L0_SEARCH_HI),
                    "iterations": int(solve._DEFAULT_BUDGET_ITERS),
                    "scale": "log10_bisection",
                },
            },
            "attainment": {
                "verdict_stage": "release_gates",
                "solver_success_is_attainment": False,
                "per_target_error": (
                    "relative_error_or_absolute_error_when_target_is_zero"
                ),
                "per_target_tolerance_source": "compiled_target_spec",
                "missing_tolerance": "diagnostic_only_with_null_verdict",
                "dropped_or_required_skipped_target": "fail",
                "positive_target_zero_support": "fail",
                "critical_target_policy": "declared_fit_requirement",
                "nonfinite_loss": "fail",
                "final_loss_worse_than_initial": "fail",
            },
            "implicit_calibrate_parameters": {
                "method": "adam",
                "mass_reason": None,
                "target_records": None,
                "l1_lambda": 0.0,
                "l2_anchor_weights": None,
            },
        },
        "refit_contracts": {
            "default_sparse": {
                "support": "selection_weight_above_prune_relative_atol",
                "subset_source": "selection_stage_frame",
                "starting_weights": "selection_stage_calibrated_weights",
                "l0_lambda": 0.0,
                "l2_lambda": l2_lambda,
                "l2_anchor": "initial",
                "mass": "conserve",
                "max_weight_ratio": max_weight_ratio,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "seed": legacy_seed,
            },
            "exact_k": {
                "support": "sorted_exact_k_indices",
                "subset_source": "original_pool_frame",
                "starting_weights": "normalized_original_weight_over_selected_q",
                "l0_lambda": 0.0,
                "l2_lambda": l2_lambda,
                "l2_anchor": "initial",
                "mass": "conserve",
                "max_weight_ratio": max_weight_ratio,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "seed": {"source": "run_request", "default": None},
            },
        },
        "release_modes": {
            "default_sparse_l0_refit": {
                "kernel": _kernel_ref(solve.calibrate_l0_refit),
                "solver_ref": "solver",
                "selection_contract_ref": "solver.l0_contract",
                "refit_contract_ref": "refit_contracts.default_sparse",
            },
            "dense_diagnostic": {
                "kernel": _kernel_ref(solve.calibrate),
                "solver_ref": "solver",
                "l0_mode": "none",
                "refit_contract_ref": None,
            },
            "exact_k_or_full_pool_refit": {
                "kernel": _kernel_ref(ladder.calibrate_exact_k_ladder),
                "solver_ref": "solver",
                "selection_contract_ref": "selection.exact_k",
                "refit_contract_ref": "refit_contracts.exact_k",
            },
        },
        "tail_contracts": _calibration_tail_contracts(release),
    }


def build_calibration() -> dict[str, object]:
    """Short generator-facing alias for :func:`build_calibration_contract`."""

    return build_calibration_contract()


def build_selection_contract() -> dict[str, object]:
    """Build the exact-k algorithm contract without inventing run values."""

    release = importlib.import_module("tools.build_us_fiscal_refresh_release")
    exact_k = importlib.import_module("microcosm.calibrate.exact_k")
    ladder = importlib.import_module("microcosm.build.us_runtime.exact_k_ladder")
    ratified_counts = sorted(int(value) for value in release.RATIFIED_EXACT_K_COUNTS)
    return {
        "contract_id": "us_exact_k_selection_runtime",
        "version": 1,
        "exact_k": {
            "kernel": _kernel_ref(exact_k.select_exact_k),
            "cardinality_gate": _kernel_ref(exact_k.assert_exact_k_support),
            "k": {
                "required": True,
                "default": None,
                "surface": "run_request",
                "precedence": "run_request_overrides_default",
                "allowed": ["N", *ratified_counts],
                "N": "authenticated_pool_household_count",
                "minimum": 1,
                "maximum": "pool_household_count",
            },
            "pi_hi": {
                "required": True,
                "default": None,
                "surface": "run_request",
                "precedence": "run_request_overrides_default",
                "minimum_inclusive": 0.0,
                "maximum_inclusive": 1.0,
            },
            "seed": {
                "required": True,
                "default": None,
                "surface": "run_request",
                "precedence": "run_request_overrides_default",
                "kind": "nonnegative_integer",
            },
            "group_ids": "none",
            "on_infeasible": "refuse",
            "post_selection_weights": "selected_keep_calibrated",
        },
        "inputs": {
            "selection_score": "hard_concrete_gate_open_probability_pi",
            "group_ids": {
                "current_ladder_value": None,
                "runtime_seam": "one_unique_id_per_record_only",
                "duplicate_policy": "refuse_pending_spine_grouping_policy",
            },
            "required_authenticated_bindings": [
                "pool_manifest_sha256",
                "pool_release_id",
                "incumbent_diagnostics_sha256",
                "frozen_target_surface_sha256",
                "ledger_facts_sha256",
                "ledger_manifest_sha256",
            ],
        },
        "algorithm": {
            "certainty": {
                "comparison": "pi >= pi_hi",
                "action": "include_first",
                "k_below_certainty_count": "refuse",
            },
            "boundary": {
                "remaining_draw": "k - certainty_count",
                "positive_score_filter": "pi > 0",
                "first_order_inclusion_probability": (
                    "q_i = remaining_draw * pi_i / sum(pi_boundary)"
                ),
                "probability_one_tolerance": float(exact_k._PROBABILITY_ONE_TOLERANCE),
                "q_above_one": "refuse_without_clamping",
                "exact_one": "deterministic_take_all_before_fractional_draw",
            },
            "design": {
                "name": "sampford",
                "without_replacement": True,
                "rng": "kernel:exact_k_pcg64_rng",
                "support_order": "ascending_stable",
                "index_dtype": str(exact_k._INDEX_DTYPE),
                "subset_law": (
                    "P(S) proportional to sum(1-q_i for i in S) "
                    "* product(q_i/(1-q_i) for i in S)"
                ),
                "majority_draw": "sample_complementary_sampford_design",
                "dynamic_program": {
                    "always_max_cells": int(exact_k._SAMPFORD_DP_ALWAYS_MAX_CELLS),
                    "max_bytes": int(exact_k._SAMPFORD_DP_MAX_BYTES),
                    "max_cells": int(exact_k._SAMPFORD_DP_MAX_CELLS),
                },
                "rejection": {
                    "max_attempts": int(exact_k._SAMPFORD_MAX_REJECTION_ATTEMPTS),
                    "safety_factor": float(exact_k._SAMPFORD_REJECTION_SAFETY_FACTOR),
                    "attempt_limit": "fail_closed",
                },
            },
            "full_pool": {
                "condition": "k == pool_size",
                "support": "identity",
                "inclusion_probabilities": 1.0,
                "design_receipt": "full-pool",
                "selection_epochs": 0,
            },
        },
        "post_selection_contract": {
            "subset_source": "original_pool_frame",
            "initial_formula": "original_weight_i / selected_q_i",
            "normalization": "project_to_known_original_pool_weight_total",
            "refit": "ordinary_adam_calibration_with_l0_removed",
            "final_weights": "refit_calibrated_weights",
            "no_post_selection_keep_without_refit": True,
            "diagnostic_methods": [
                "normalized_horvitz_thompson_w_over_q",
                "full_pool_original_frame_weights",
            ],
        },
        "receipt": {
            "selection_receipt_fields": [
                "k",
                "pi_hi",
                "seed",
                "certainty_count",
                "boundary_pool_size",
                "design",
            ],
            "manifest_kernel": _kernel_ref(ladder.exact_k_ladder_manifest_payload),
            "manifest_fields": [
                "k",
                "seed",
                "selection_receipt",
                "refit_baseline_diagnostics",
                "pool",
                "agreement_gate_reference",
                "frozen_target_register",
            ],
        },
    }


def build_selection() -> dict[str, object]:
    """Short generator-facing alias for :func:`build_selection_contract`."""

    return build_selection_contract()


def build_us_runtime_contracts() -> dict[str, object]:
    """Build all four deterministic US bundle-generation authority projections."""

    return {
        "take_up": build_take_up_contract(),
        "battery": build_battery_contract(),
        "calibration": build_calibration_contract(),
        "selection": build_selection_contract(),
    }
