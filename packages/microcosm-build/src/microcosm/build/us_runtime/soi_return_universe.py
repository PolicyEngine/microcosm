"""Explicit modeled filer universes for IRS return-based target design rows.

This measures the actual engine predicate. It does not establish equivalence
to the IRS processing-year population or authenticate separate dependent returns.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from microcosm.calibrate import TargetSpec
    from microcosm.frame import Frame

FILING_PROPENSITY_INPUTS = (
    "would_file_if_eligible_for_refundable_credit",
    "would_file_taxes_voluntarily",
)
_DECISION_VARIABLES = (
    "tax_unit_is_filer",
    "tax_unit_is_required_to_file",
    "eligible_for_refundable_credits",
)


def _require(value, reason):
    if not value:
        raise ValueError(f"SOI return universe: {reason}")


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def soi_target_universe_contract(
    target_specs: Sequence[TargetSpec], *, period: int
) -> dict[str, Any]:
    """Validate source declarations before constructing any model simulation."""
    rows = []
    unsupported = []
    for spec in target_specs:
        if spec.family != "irs_soi":
            continue
        details = spec.metadata
        universe = details.get("soi_return_universe")
        _require(
            universe
            in {"all_returns", "itemized_returns", "returns_excluding_dependents"},
            f"unsupported or missing source universe for {spec.name!r}",
        )
        _require(
            details.get("target_period", str(period)) == str(period),
            f"target period differs from modeled period for {spec.name!r}",
        )
        itemized = details.get("itemized_only")
        _require(itemized in {None, "true", "false"}, "invalid itemization declaration")
        _require(
            not (universe == "itemized_returns" and itemized == "false"),
            "itemized universe contradicts itemized_only=false",
        )
        record_set = details.get("ledger_layout_record_set_id", "")
        domain = details.get("ledger_domain", "")
        _require(
            not (
                "excluding_dependents" in record_set or "excluding_dependents" in domain
            )
            or universe == "returns_excluding_dependents",
            "source layout contradicts declared return universe",
        )
        _require(
            not (
                (
                    "itemized_all_returns" in record_set
                    or "with_itemized_deductions" in domain
                )
                and universe != "itemized_returns"
            ),
            "source layout contradicts declared return universe",
        )
        _require(
            not (
                universe == "returns_excluding_dependents"
                and "all_returns" in record_set
                and "excluding_dependents" not in record_set
            ),
            "source layout contradicts declared return universe",
        )
        row = {
            "name": spec.name,
            "measure": spec.measure,
            "source_period": details.get("source_period"),
            "target_period": str(period),
            "source_universe": universe,
            "requires_itemization": universe == "itemized_returns"
            or itemized == "true",
            "target_value_universe_bridge": details.get("soi_return_universe_bridge"),
        }
        if universe == "returns_excluding_dependents":
            unsupported.append(
                {
                    **row,
                    "reason": "dependent_return_input_contract_not_established",
                }
            )
        else:
            rows.append(row)
    return {
        "schema_version": 2,
        "model_period": period,
        "targets": rows,
        "unsupported_targets": unsupported,
        "measurement": "modeled_target_year_filers",
        "dependent_return_completeness": "not_established",
        "source_processing_year_and_geography_equivalence": "not_established",
    }


def _bool_variable(system, name):
    variable = system.variables.get(name)
    _require(variable is not None, f"missing engine variable {name}")
    _require(
        variable.entity.key == "tax_unit"
        and variable.value_type is bool
        and str(variable.definition_period) == "year",
        f"invalid engine entity/type/period for {name}",
    )
    return variable


def _bool_vector(values, n, name):
    result = np.asarray(values)
    _require(
        result.dtype == np.dtype("bool") and result.shape == (n,),
        f"invalid boolean vector for {name}",
    )
    return result


def _input_origin(frame, name, default):
    table = frame.table("tax_unit")
    if name in table:
        values = table[name]
        _require(
            pd.api.types.is_bool_dtype(values.dtype) and not values.isna().any(),
            f"supplied input {name} must be nonnullable boolean",
        )
        return {
            "origin": "supplied_frame_column",
            "sha256": _digest(values.to_numpy(dtype=bool).tobytes()),
        }
    _require(type(default) is bool, f"nonboolean model default for {name}")
    return {
        "origin": "declared_model_default",
        "value": default,
        "interpretation": "model_assumption_not_observed_response",
    }


def _model_source_identity(system, names):
    distribution = metadata.distribution("policyengine-us")
    package_root = Path(distribution.locate_file("policyengine_us")).resolve()
    declarations = {}
    for name in names:
        variable = _bool_variable(system, name)
        path = Path(inspect.getsourcefile(type(variable))).resolve()
        _require(
            path.is_relative_to(package_root),
            "engine declaration origin differs from installed package",
        )
        declarations[name] = {
            "path": path.relative_to(package_root).as_posix(),
            "sha256": _digest(path.read_bytes()),
        }
    return {
        "package": "policyengine-us",
        "version": distribution.version,
        "core_version": metadata.version("policyengine-core"),
        "direct_variable_declarations": declarations,
        "scope": "direct_filing_declarations_not_transitive_model_closure",
    }


def materialize_soi_filer_universe(
    frame: Frame, simulation: Any, system: Any, *, contract: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return actual tax-unit masks and a compilation-provenance receipt."""
    period = contract["model_period"]
    n = frame.n("tax_unit")
    names = _DECISION_VARIABLES + FILING_PROPENSITY_INPUTS
    needs_itemizer = any(row["requires_itemization"] for row in contract["targets"])
    if needs_itemizer:
        names += ("tax_unit_itemizes",)
    variables = {name: _bool_variable(system, name) for name in names}
    origins = {
        name: _input_origin(frame, name, variables[name].default_value)
        for name in FILING_PROPENSITY_INPUTS
    }
    masks = {
        name: _bool_vector(simulation.calculate(name, period=period), n, name)
        for name in ("tax_unit_is_filer",)
        + (("tax_unit_itemizes",) if needs_itemizer else ())
    }
    receipt = {
        "schema_version": 1,
        "kind": "microcosm.soi_modeled_return_universe",
        "contract": contract,
        "tax_unit_rows": n,
        "tax_unit_ids_sha256": _digest(
            frame.table("tax_unit").tax_unit_id.to_numpy(dtype="<i8").tobytes()
        ),
        "filing_input_origins": origins,
        "actual_mask_sha256": {
            name: _digest(value.tobytes()) for name, value in masks.items()
        },
        "model": _model_source_identity(system, names),
        "provenance_scope": "local_target_compilation_not_source_authentication",
    }
    return masks, receipt


def validate_cached_soi_filer_receipt(
    receipt: object, frame: Frame, *, contract: dict[str, Any]
) -> None:
    """Check declared cache provenance; do not claim to rerun or authenticate it."""
    _require(
        type(receipt) is dict and receipt.get("schema_version") == 1,
        "missing cached filer receipt",
    )
    _require(
        receipt.get("kind") == "microcosm.soi_modeled_return_universe"
        and receipt.get("provenance_scope")
        == "local_target_compilation_not_source_authentication",
        "invalid cached provenance scope",
    )
    _require(
        receipt.get("contract") == contract, "cached source-universe contract changed"
    )
    _require(
        receipt.get("tax_unit_rows") == frame.n("tax_unit")
        and receipt.get("tax_unit_ids_sha256")
        == _digest(frame.table("tax_unit").tax_unit_id.to_numpy(dtype="<i8").tobytes()),
        "cached tax-unit identity changed",
    )
    origins = receipt.get("filing_input_origins", {})
    _require(
        set(origins) == set(FILING_PROPENSITY_INPUTS),
        "cached filing input roster changed",
    )
    for name, recorded in origins.items():
        _require(
            recorded == _input_origin(frame, name, recorded.get("value")),
            "cached filing input origin or content changed",
        )
    distribution = metadata.distribution("policyengine-us")
    model = receipt.get("model", {})
    _require(
        model.get("package") == "policyengine-us"
        and model.get("scope")
        == "direct_filing_declarations_not_transitive_model_closure"
        and model.get("version") == distribution.version
        and model.get("core_version") == metadata.version("policyengine-core"),
        "cached engine identity changed",
    )
    declarations = model.get("direct_variable_declarations", {})
    expected_names = set(_DECISION_VARIABLES + FILING_PROPENSITY_INPUTS)
    needs_itemizer = any(row["requires_itemization"] for row in contract["targets"])
    if needs_itemizer:
        expected_names.add("tax_unit_itemizes")
    _require(set(declarations) == expected_names, "cached declaration roster changed")
    package_root = Path(distribution.locate_file("policyengine_us")).resolve()
    for declaration in declarations.values():
        relative = Path(declaration["path"])
        path = (package_root / relative).resolve()
        _require(
            not relative.is_absolute() and path.is_relative_to(package_root),
            "invalid cached declaration path",
        )
        _require(
            _digest(path.read_bytes()) == declaration["sha256"],
            "cached engine declaration changed",
        )
    expected_masks = {"tax_unit_is_filer"} | (
        {"tax_unit_itemizes"} if needs_itemizer else set()
    )
    mask_hashes = receipt.get("actual_mask_sha256", {})
    _require(
        set(mask_hashes) == expected_masks
        and all(
            type(value) is str
            and len(value) == 64
            and all(c in "0123456789abcdef" for c in value)
            for value in mask_hashes.values()
        ),
        "invalid cached mask identities",
    )
    # Ownership and comparison above validate ordinary cached provenance only.
    # The legacy target checkpoint has no independently authenticated row seal.
    json.dumps(receipt, allow_nan=False)
