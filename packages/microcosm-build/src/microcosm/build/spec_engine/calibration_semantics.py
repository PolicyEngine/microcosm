"""Pure normalization and constants-era projection for calibration contracts."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from .canonical import sha256_json
from .errors import SpecValidationError

CALIBRATION_SUMMARY_ALIASES = frozenset(
    {
        "target_scaling",
        "zero_negative_target_policy",
        "objective_aggregation",
        "mass_constraints",
        "initialization",
        "stopping",
        "infeasibility_policy",
        "target_priority",
        "max_weight_ratio",
    }
)

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


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SpecValidationError(f"{location}: object required")
    return value


def _array(value: object, location: str) -> list[object]:
    if not isinstance(value, list):
        raise SpecValidationError(f"{location}: array required")
    return value


def _require_contract_ref(
    value: object,
    expected: Mapping[str, str],
    *,
    location: str,
) -> None:
    reference = _mapping(value, location)
    if dict(reference) != dict(expected):
        raise SpecValidationError(
            f"{location}: expected reviewed typed reference {dict(expected)!r}, "
            f"got {dict(reference)!r}"
        )


def derive_calibration_summary_aliases(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Derive the retired flat solver summary from normalized contracts."""

    targets = _mapping(document.get("targets"), "calibration/targets")
    solver = _mapping(document.get("solver"), "calibration/solver")
    loss = _mapping(solver.get("loss"), "calibration/solver/loss")
    params = _mapping(loss.get("params"), "calibration/solver/loss/params")
    scaling = _mapping(
        params.get("target_scaling"),
        "calibration/solver/loss/params/target_scaling",
    )
    if dict(scaling) != {
        "kind": "default_target",
        "formula": "max(abs(target), 1)",
    }:
        raise SpecValidationError(
            "calibration/solver/loss/params/target_scaling: no reviewed "
            "constants-era projection"
        )
    if (
        targets.get("negative_target_policy") != "require_target_spec_signed_true"
        or targets.get("zero_target_policy") != "retain_with_scale_one"
    ):
        raise SpecValidationError(
            "calibration/targets: zero/negative policies have no reviewed "
            "constants-era projection"
        )
    hard = _mapping(
        solver.get("hard_constraints"), "calibration/solver/hard_constraints"
    )
    stopping = _mapping(
        solver.get("stopping_contract"), "calibration/solver/stopping_contract"
    )
    initialization = _mapping(
        solver.get("initialization_contract"),
        "calibration/solver/initialization_contract",
    )
    infeasibility = _mapping(
        solver.get("infeasibility_contract"),
        "calibration/solver/infeasibility_contract",
    )
    priority = _mapping(
        solver.get("target_priority_contract"),
        "calibration/solver/target_priority_contract",
    )
    if hard.get("mass") != "conserve" or hard.get("mass_total") != "input_weight_total":
        raise SpecValidationError(
            "calibration/solver/hard_constraints: unsupported mass summary"
        )
    return {
        "target_scaling": "max_abs_target_or_one",
        "zero_negative_target_policy": "retain_zero_reject_unsigned_negative",
        "objective_aggregation": (
            "concept_budget_weighted_mean_capped_absolute_relative_error"
        ),
        "mass_constraints": [
            "conserve_input_weight_total",
            "per_step_ratio_cap",
            "closing_float64_projection",
        ],
        "initialization": deepcopy(initialization.get("policy_id")),
        "stopping": {"max_epochs": deepcopy(stopping.get("max_epochs"))},
        "infeasibility_policy": deepcopy(infeasibility.get("soft_target_miss")),
        "target_priority": deepcopy(priority.get("policy_id")),
        "max_weight_ratio": deepcopy(hard.get("max_weight_ratio")),
    }


def project_legacy_calibration_contract(
    document: Mapping[str, object],
) -> dict[str, object]:
    """Recreate the pre-normalization calibration object without dual authority."""

    result = deepcopy(dict(document))
    solver = _mapping(result.get("solver"), "calibration/solver")
    collisions = sorted(CALIBRATION_SUMMARY_ALIASES.intersection(solver))
    if collisions:
        raise SpecValidationError(
            "calibration/solver: retired derived aliases must not be authored: "
            f"{collisions!r}"
        )
    solver.update(derive_calibration_summary_aliases(result))
    return result


def resolve_calibration_tail_contracts(
    document: Mapping[str, object],
    *,
    spine_document: Mapping[str, object],
    imputation_document: Mapping[str, object],
) -> dict[str, object]:
    """Resolve typed tail references into the constants-era execution object."""

    tail_contracts = deepcopy(
        dict(_mapping(document.get("tail_contracts"), "calibration/tail_contracts"))
    )
    puf = dict(
        _mapping(
            tail_contracts.get("puf_capital_gains_tail"),
            "calibration/tail_contracts/puf_capital_gains_tail",
        )
    )
    _require_contract_ref(
        puf.pop("support_contract_ref", None),
        _PUF_TAIL_SUPPORT_REF,
        location="calibration/tail_contracts/puf_capital_gains_tail/support_contract_ref",
    )
    _require_contract_ref(
        puf.pop("execution_binding_ref", None),
        _PUF_TAIL_EXECUTION_REF,
        location=(
            "calibration/tail_contracts/puf_capital_gains_tail/"
            "execution_binding_ref"
        ),
    )

    support_roles = [
        _mapping(value, "spine/support_roles row")
        for value in _array(spine_document.get("support_roles"), "spine/support_roles")
        if _mapping(value, "spine/support_roles row").get("id")
        == _PUF_TAIL_SUPPORT_REF["support_role"]
    ]
    if len(support_roles) != 1:
        raise SpecValidationError(
            "spine/support_roles: calibration tail support reference must resolve once"
        )
    tail_support = _mapping(
        support_roles[0].get("tail_support"),
        "spine/support_roles/puf_tax_detail/tail_support",
    )

    graph = _mapping(
        imputation_document.get("producer_graph"),
        "imputation/producer_graph",
    )
    primary_nodes = [
        _mapping(value, "imputation/producer_graph/nodes row")
        for value in _array(graph.get("nodes"), "imputation/producer_graph/nodes")
        if _mapping(value, "imputation/producer_graph/nodes row").get("name")
        == _PUF_TAIL_EXECUTION_REF["producer"]
    ]
    if len(primary_nodes) != 1:
        raise SpecValidationError(
            "imputation/producer_graph/nodes: calibration tail producer must "
            "resolve once"
        )
    resources = [
        _mapping(value, "primary_puf_qrf/virtual_resources row")
        for value in _array(
            primary_nodes[0].get("virtual_resources"),
            "primary_puf_qrf/virtual_resources",
        )
        if _mapping(value, "primary_puf_qrf/virtual_resources row").get("id")
        == _PUF_TAIL_EXECUTION_REF["resource"]
    ]
    if len(resources) != 1:
        raise SpecValidationError(
            "primary_puf_qrf/virtual_resources: calibration tail resource must "
            "resolve once"
        )
    binding = _mapping(
        resources[0].get("binding"),
        "primary_puf_qrf/virtual_resources/tail/binding",
    )
    execution_tail = _mapping(
        binding.get("capital_gains_tail"),
        "primary_puf_qrf/binding/capital_gains_tail",
    )
    soi = _mapping(
        execution_tail.get("soi_e19200_agi_bands"),
        "primary_puf_qrf/binding/capital_gains_tail/soi_e19200_agi_bands",
    )
    runtime_agi_bands = dict(
        _mapping(
            soi.get("runtime_agi_bands"),
            "primary_puf_qrf/capital_gains_tail/runtime_agi_bands",
        )
    )
    if soi.get("agi_bands") != runtime_agi_bands.get("agi_bands"):
        raise SpecValidationError(
            "primary_puf_qrf/capital_gains_tail: parsed and runtime AGI bands differ"
        )
    puf.update(
        {
            "support_contract": deepcopy(tail_support.get("legacy_contract")),
            "disaggregation_spec": deepcopy(execution_tail.get("spec")),
            "concentration_controls": deepcopy(
                execution_tail.get("concentration_gate")
            ),
            "soi_e19200_agi_bands": {
                "agi_bands": deepcopy(soi.get("agi_bands")),
                "all_returns": deepcopy(soi.get("all_returns")),
                "asset": soi.get("asset"),
                "asset_sha256": soi.get("asset_sha256"),
                "runtime_schema_version": runtime_agi_bands.get("schema_version"),
                "runtime_sha256": sha256_json(runtime_agi_bands),
            },
        }
    )
    tail_contracts["puf_capital_gains_tail"] = puf
    return tail_contracts


__all__ = [
    "CALIBRATION_SUMMARY_ALIASES",
    "derive_calibration_summary_aliases",
    "project_legacy_calibration_contract",
    "resolve_calibration_tail_contracts",
]
