"""Pure constants-era projections for normalized imputation domains.

This module is a compiler-front-end compatibility layer.  It consumes only
validated JSON-shaped domain mappings and immutable compiler registries; it
does not import the executor, CountrySpec, frozen legacy resources, or the
one-shot migration generator.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path

from .seeds import LEGACY_V1_PROTOCOL
from .typed_closure import compile_producer_outputs

_PUF_ATTACHMENT_REF: dict[str, str] = {
    "domain": "spine",
    "support_role": "puf_tax_detail",
    "pointer": "/attachment",
}
_PUF_TAIL_SUPPORT_REF: dict[str, str] = {
    "domain": "spine",
    "support_role": "puf_tax_detail",
    "pointer": "/tail_support/legacy_contract",
}
_BUILD_MODEL_SEED_REF: dict[str, str] = {
    "domain": "seed_protocol",
    "value_source": "run_request.build_model_seed",
}
_TARGET_PERIOD_REF: dict[str, str] = {
    "domain": "bundle",
    "pointer": "/dataset_run/target_period",
}
_SOURCE_OPERATOR_REGISTRY_REF: dict[str, str] = {
    "domain": "spine",
    "pointer": "/pipeline_contract/post_clone_source_operator_order",
}

_NO_WRITE_ACTIONS = {
    "consume_only_byte_exact_noop",
    "origin_projection_masked_noop",
    "producer_masked_byte_exact_noop",
    "scope_masked_noop",
}


def _mapping_like(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{location} must be an object.")
    return value


def _array_like(value: object, location: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(f"{location} must be an array.")
    return value


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _legacy_float(value: object, *, location: str) -> float:
    """Restore a reviewed generation-0 JSON number's required float wire type."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{location} must be numeric.")
    return float(value)


def _restore_primary_tail_legacy_numbers(tail: dict[str, object]) -> None:
    """Re-inflate only the reviewed float-valued capital-gains fields."""

    concentration = _mapping_like(
        tail["concentration_gate"], "capital-gains concentration gate"
    )
    for key in ("positive_mass_five_x_target", "recipient_capital_gains_topcode"):
        concentration[key] = _legacy_float(  # type: ignore[index]
            concentration[key], location=f"capital-gains concentration gate/{key}"
        )

    spec = _mapping_like(tail["spec"], "capital-gains disaggregation spec")
    for index, bucket_value in enumerate(
        _array_like(spec["buckets"], "capital-gains disaggregation buckets")
    ):
        bucket = _mapping_like(bucket_value, f"capital-gains bucket {index}")
        for key in ("agi_lower", "agi_upper", "synthetic_agi_upper"):
            if bucket.get(key) is not None:
                bucket[key] = _legacy_float(  # type: ignore[index]
                    bucket[key], location=f"capital-gains bucket {index}/{key}"
                )

    soi = _mapping_like(tail["soi_e19200_agi_bands"], "SOI E19200 AGI bands")
    runtime = _mapping_like(soi["runtime_agi_bands"], "runtime SOI AGI bands")
    for label, rows_value in (
        ("parsed", soi["agi_bands"]),
        ("runtime", runtime["agi_bands"]),
    ):
        for index, row_value in enumerate(
            _array_like(rows_value, f"{label} AGI bands")
        ):
            row = _mapping_like(row_value, f"{label} AGI band {index}")
            for key in ("lower_bound", "upper_bound"):
                if row.get(key) is not None:
                    row[key] = _legacy_float(  # type: ignore[index]
                        row[key], location=f"{label} AGI band {index}/{key}"
                    )
    runtime["sha256"] = _canonical_sha256(  # type: ignore[index]
        {key: value for key, value in runtime.items() if key != "sha256"}
    )


def _restore_source_stage_legacy_numbers(stage: dict[str, object]) -> None:
    """Restore the sole reviewed integral float in the source-stage payload."""

    for index, operation_value in enumerate(
        _array_like(stage["operations"], "resolved source-stage operations")
    ):
        operation = _mapping_like(operation_value, f"source operation {index}")
        if operation.get("kind") != "derive_wic_claim":
            continue
        parameters = _mapping_like(
            operation["parameters"], f"source operation {index} parameters"
        )
        category_rates = _mapping_like(
            parameters["category_rates"],
            f"source operation {index} category rates",
        )
        values = _mapping_like(
            category_rates["values"],
            f"source operation {index} category-rate values",
        )
        for category, value in values.items():
            values[category] = _legacy_float(  # type: ignore[index]
                value,
                location=f"source operation {index}/category_rates/{category}",
            )


def _project_transfer_execution_identity(
    transfer_execution: Mapping[str, object],
    predictor_blocks: Mapping[str, object],
    *,
    profile_id: str,
    targets: Sequence[str],
) -> dict[str, object]:
    """Rehydrate one exact constants-era ACS-transfer ABI identity."""

    result = {
        key: deepcopy(value)
        for key, value in transfer_execution.items()
        if key not in {"predictor_bindings", "post_transfer_features", "profiles"}
    }
    bindings = _mapping_like(
        transfer_execution["predictor_bindings"], "transfer predictor bindings"
    )

    def block_columns(block_id: object) -> list[object]:
        block = _mapping_like(
            predictor_blocks[str(block_id)], f"predictor block {block_id}"
        )
        return deepcopy(list(_array_like(block["columns"], f"{block_id} columns")))

    result["person_required_predictors"] = block_columns(bindings["person_required"])
    result["person_optional_predictors"] = [
        column
        for block_id in _array_like(
            bindings["person_optional"], "person optional predictor block refs"
        )
        for column in block_columns(block_id)
    ]
    result["group_required_predictors"] = block_columns(bindings["group_required"])
    housing = deepcopy(dict(_mapping_like(result["housing"], "transfer housing")))
    housing["mandatory_features"] = block_columns(bindings["housing_mandatory"])
    result["housing"] = housing
    tenure_codes = _mapping_like(result["tenure_codes"], "transfer tenure codes")
    result["tenure_codes"] = {
        key: _legacy_float(value, location=f"transfer tenure_codes/{key}")
        for key, value in tenure_codes.items()
    }

    profiles = _mapping_like(transfer_execution["profiles"], "transfer profiles")
    profile = _mapping_like(profiles[profile_id], f"transfer profile {profile_id}")
    derive_schedule_d = bool(profile["derive_schedule_d"])
    target_set = set(targets)
    post_transfer: dict[str, object] = {}
    features = _mapping_like(
        transfer_execution["post_transfer_features"], "post-transfer features"
    )
    for feature_id, feature_value in features.items():
        feature = _mapping_like(feature_value, f"post-transfer feature {feature_id}")
        activation = _mapping_like(
            feature["activation"], f"post-transfer feature {feature_id} activation"
        )
        enabled = set(activation.get("all_targets", [])).issubset(target_set)
        if activation.get("derive_schedule_d") is True:
            enabled = enabled and derive_schedule_d
        contract = deepcopy(
            dict(
                _mapping_like(
                    feature["contract"], f"post-transfer feature {feature_id} contract"
                )
            )
        )
        if enabled:
            contract.update(deepcopy(dict(feature.get("enabled_overrides", {}))))
        contract["enabled"] = enabled
        post_transfer[feature_id] = contract
    schedule_d = _mapping_like(
        post_transfer["schedule_d_capital_gain_distributions"],
        "Schedule D post-transfer structure",
    )
    share_asset_value = schedule_d.get("share_asset")
    if share_asset_value is not None:
        share_asset = _mapping_like(share_asset_value, "Schedule D share asset")
        anchor = _mapping_like(
            share_asset["national_anchor_ty2015"],
            "Schedule D share asset national anchor",
        )
        anchor["pub1304_direct_1040_route_k"] = _legacy_float(  # type: ignore[index]
            anchor["pub1304_direct_1040_route_k"],
            location="Schedule D share asset/pub1304_direct_1040_route_k",
        )
    result["post_transfer_structure"] = post_transfer
    result["sha256"] = _canonical_sha256(result)
    return result


def _worker_execution_template() -> dict[str, object]:
    """Describe exact runtime resolution with a closed resolver-op algebra."""

    return {
        "surface": "execution_profile",
        "resolve_as": "worker_execution",
        "template": {
            "module": "microcosm.build.us_runtime.puf_qrf_worker",
            "argv_template": [
                {"resolver_op": "sys_executable"},
                "-m",
                "microcosm.build.us_runtime.puf_qrf_worker",
                "--checkpoint-dir",
                "{checkpoint_dir}",
                "--target-index",
                "{target_index}",
            ],
            "interpreter": {
                "executable": {"resolver_op": "sys_executable"},
                "resolved_executable": {"resolver_op": "resolved_sys_executable"},
                "implementation": {"resolver_op": "python_implementation"},
                "cache_tag": {"resolver_op": "python_cache_tag"},
                "version": {"resolver_op": "python_version_triplet"},
            },
            "environment": {
                "policy": "inherit_parent_environment_with_bound_fit_controls",
                "overrides": {},
                "semantic_controls": {
                    "POPULACE_FIT_N_JOBS": {
                        "configured": {
                            "resolver_op": "environment_value",
                            "name": "POPULACE_FIT_N_JOBS",
                        },
                        "resolved": {
                            "resolver_op": "env_canonical_positive_int_or_default",
                            "name": "POPULACE_FIT_N_JOBS",
                            "default": -1,
                        },
                    },
                    "POPULACE_FIT_PREDICT_WORKERS": {
                        "configured": {
                            "resolver_op": "environment_value",
                            "name": "POPULACE_FIT_PREDICT_WORKERS",
                        },
                        "resolved": {
                            "resolver_op": "env_positive_int_or_cpu_count",
                            "name": "POPULACE_FIT_PREDICT_WORKERS",
                            "fallback_minimum": 1,
                        },
                        "resolution": {
                            "resolver_op": "env_or_cpu_count_resolution_label",
                            "name": "POPULACE_FIT_PREDICT_WORKERS",
                        },
                    },
                },
                "bound_names": [
                    "POPULACE_FIT_N_JOBS",
                    "POPULACE_FIT_PREDICT_WORKERS",
                ],
            },
        },
    }


def _resolve_worker_execution(value: Mapping[str, object]) -> dict[str, object]:
    """Resolve the one reviewed worker template, refusing an extended mini-language."""

    expected = _worker_execution_template()
    if value != expected:
        raise RuntimeError(
            "Primary-QRF worker template differs from the closed reviewed resolver."
        )
    fit_jobs_raw = os.environ.get("POPULACE_FIT_N_JOBS")
    if fit_jobs_raw is None:
        fit_jobs = -1
    else:
        try:
            fit_jobs = int(fit_jobs_raw)
        except ValueError as error:
            raise ValueError(
                "POPULACE_FIT_N_JOBS must be a positive integer for the "
                "primary-QRF worker binding."
            ) from error
        if fit_jobs < 1 or str(fit_jobs) != fit_jobs_raw:
            raise ValueError(
                "POPULACE_FIT_N_JOBS must be a canonical positive integer for "
                "the primary-QRF worker binding."
            )
    predict_workers_raw = os.environ.get("POPULACE_FIT_PREDICT_WORKERS")
    if predict_workers_raw is None or not predict_workers_raw.strip():
        predict_workers = os.cpu_count() or 1
        predict_workers_source = "os_cpu_count_fallback"
    else:
        try:
            predict_workers = int(predict_workers_raw)
        except ValueError as error:
            raise ValueError(
                "POPULACE_FIT_PREDICT_WORKERS must be a positive integer for the "
                "primary-QRF worker binding."
            ) from error
        if predict_workers < 1:
            raise ValueError(
                "POPULACE_FIT_PREDICT_WORKERS must be positive for the "
                "primary-QRF worker binding."
            )
        predict_workers_source = "environment_override"
    executable = Path(sys.executable)
    module = "microcosm.build.us_runtime.puf_qrf_worker"
    return {
        "module": module,
        "argv_template": [
            str(executable),
            "-m",
            module,
            "--checkpoint-dir",
            "{checkpoint_dir}",
            "--target-index",
            "{target_index}",
        ],
        "interpreter": {
            "executable": str(executable),
            "resolved_executable": str(executable.resolve()),
            "implementation": sys.implementation.name,
            "cache_tag": sys.implementation.cache_tag,
            "version": list(sys.version_info[:3]),
        },
        "environment": {
            "policy": "inherit_parent_environment_with_bound_fit_controls",
            "overrides": {},
            "semantic_controls": {
                "POPULACE_FIT_N_JOBS": {
                    "configured": fit_jobs_raw,
                    "resolved": fit_jobs,
                },
                "POPULACE_FIT_PREDICT_WORKERS": {
                    "configured": predict_workers_raw,
                    "resolved": predict_workers,
                    "resolution": predict_workers_source,
                },
            },
            "bound_names": [
                "POPULACE_FIT_N_JOBS",
                "POPULACE_FIT_PREDICT_WORKERS",
            ],
        },
    }


def _compile_node_outputs(
    document: Mapping[str, object],
    graph: Mapping[str, object],
) -> dict[str, list[dict[str, object]]]:
    """Expand family-owned outputs and restore canonical node output order."""

    if graph is not document.get("producer_graph"):
        raise RuntimeError("Producer graph must be the document's authored graph.")
    return {
        producer: [deepcopy(dict(row)) for row in rows]
        for producer, rows in compile_producer_outputs({"imputation": document}).items()
    }


def _node_capabilities(kind: str) -> dict[str, object]:
    if kind == "acs_earnings_universe":
        return {
            "determinism": "deterministic",
            "numeric_reproducibility": "bitwise",
            "effects": ["declared_source_read"],
            "structural_delta": "none",
            "retry_safety": "idempotent",
        }
    if kind == "primary_puf":
        return {
            "determinism": "seeded",
            "numeric_reproducibility": "tolerance_bound",
            "effects": ["declared_source_read", "declared_sink_write"],
            "structural_delta": "expand",
            "retry_safety": "attempt_scoped",
        }
    if kind == "post_clone_source":
        return {
            "determinism": "seeded",
            "numeric_reproducibility": "tolerance_bound",
            "effects": ["declared_source_read"],
            "structural_delta": "none",
            "retry_safety": "idempotent",
        }
    if kind == "source_finalizer":
        return {
            "determinism": "deterministic",
            "numeric_reproducibility": "bitwise",
            "effects": ["none"],
            "structural_delta": "none",
            "retry_safety": "idempotent",
        }
    if kind == "late_transfer":
        return {
            "determinism": "seeded",
            "numeric_reproducibility": "tolerance_bound",
            "effects": ["none"],
            "structural_delta": "none",
            "retry_safety": "idempotent",
        }
    raise RuntimeError(f"Unexpected producer kind {kind!r}.")


def _node_mutations(kind: str) -> dict[str, object]:
    """Declare the structural Frame contract observed at the current seam."""

    if kind != "primary_puf":
        return {
            "entity_keys": {
                "operation": "preserve",
                "precondition": "entity_keys_valid",
                "postcondition": "entity_keys_unchanged",
            },
            "cardinality": {
                "operation": "preserve",
                "precondition": "entity_cardinality_valid",
                "postcondition": "entity_cardinality_unchanged",
            },
            "links": {
                "operation": "preserve",
                "precondition": "links_valid",
                "postcondition": "links_unchanged",
            },
            "memberships": {
                "operation": "preserve",
                "precondition": "memberships_valid",
                "postcondition": "memberships_unchanged",
            },
            "order": {
                "operation": "preserve",
                "precondition": "entity_order_valid",
                "postcondition": "entity_order_unchanged",
            },
            "weights": {
                "operation": "preserve",
                "precondition": "weights_valid",
                "postcondition": "weights_unchanged",
            },
            "mass_history": {
                "operation": "preserve",
                "precondition": "mass_history_valid",
                "postcondition": "mass_history_unchanged",
            },
        }
    return {
        "entity_keys": {
            "operation": "append_remapped_clone_keys",
            "precondition": "native_entity_keys_unique",
            "postcondition": "all_entity_keys_unique",
        },
        "cardinality": {
            "operation": "expand_complete_household_graphs",
            "precondition": "native_clone_index_zero",
            "postcondition": "clone_roles_materialized",
        },
        "links": {
            "operation": "preserve_absent",
            "precondition": "link_tables_absent",
            "postcondition": "link_tables_absent",
        },
        "memberships": {
            "operation": "append_relinked_clone_memberships",
            "precondition": "native_memberships_valid",
            "postcondition": "clone_memberships_reference_remapped_keys",
        },
        "order": {
            "operation": "append_clone_blocks_preserving_native_order",
            "precondition": "native_entity_order_valid",
            "postcondition": "clone_blocks_follow_native_rows",
        },
        "weights": {
            "operation": "split_mass_across_clone_descendants",
            "precondition": "native_household_mass_finite",
            "postcondition": "household_mass_conserved",
        },
        "mass_history": {
            "operation": "preserve",
            "precondition": "mass_history_valid",
            "postcondition": "mass_history_unchanged",
        },
    }


def _write_scope(
    *,
    producer: str,
    output: Mapping[str, object],
    ownership_rows: Sequence[object],
) -> dict[str, object]:
    entity = str(output["entity"])
    column = str(output["column"])
    segments = []
    matching_rows = [
        _mapping_like(value, "overlap ownership row")
        for value in ownership_rows
        if _mapping_like(value, "overlap ownership row")["entity"] == entity
        and _mapping_like(value, "overlap ownership row")["target"] == column
    ]
    for row in matching_rows:
        actions = [
            _mapping_like(value, "overlap producer action")
            for value in _array_like(row["producer_actions"], "producer actions")
            if _mapping_like(value, "overlap producer action")["producer"] == producer
        ]
        if len(actions) != 1:
            raise RuntimeError(
                f"Ownership row {entity}.{column} does not name {producer!r} once."
            )
        action = str(actions[0]["action"])
        if action not in _NO_WRITE_ACTIONS:
            segments.append(
                {
                    "predicate": "origin_clone",
                    "origin": row["origin"],
                    "clone_index": row["clone_index"],
                    "write_policy": action,
                }
            )
    if not matching_rows:
        segments.append(
            {
                "predicate": "coverage_scope",
                "coverage_scope": output["coverage_scope"],
                "write_policy": "declared_output_write",
            }
        )
    if not segments:
        raise RuntimeError(
            f"Producer {producer!r} declares {entity}.{column} but owns no cells."
        )
    if column == "@resolved_weight":
        mode = "resolved_weight"
    elif column.startswith("@"):
        mode = "virtual_receipt"
    elif column.endswith("_id") or "support_" in column:
        mode = "structural_column"
    else:
        mode = "column_cells"
    return {
        "entity": entity,
        "column": column,
        "row_scope": output["coverage_scope"],
        "mode": mode,
        "cell_segments": segments,
    }


def _segments_overlap(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    scope_coverage: Mapping[str, object],
) -> bool:
    if left["predicate"] == right["predicate"] == "origin_clone":
        return (left["origin"], left["clone_index"]) == (
            right["origin"],
            right["clone_index"],
        )
    declared = _mapping_like(scope_coverage["declared"], "scope coverage")

    def atoms(segment: Mapping[str, object]) -> set[tuple[str, int]]:
        if segment["predicate"] == "origin_clone":
            return {(str(segment["origin"]), int(segment["clone_index"]))}
        scope = str(segment["coverage_scope"])
        scopes = set(_array_like(declared.get(scope, [scope]), f"scope {scope}"))
        result: set[tuple[str, int]] = set()
        if "whole_pool" in scopes:
            return {
                (origin, clone_index)
                for origin in ("asec", "acs")
                for clone_index in (0, 1, 2)
            }
        if "asec_source" in scopes:
            result.add(("asec", 0))
        if "acs_source" in scopes:
            result.add(("acs", 0))
        if "puf_clone" in scopes:
            result.update(
                (origin, clone_index)
                for origin in ("asec", "acs")
                for clone_index in (1, 2)
            )
        if "receipt" in scopes:
            result.add(("receipt", 0))
        return result

    return bool(atoms(left) & atoms(right))


def _incomparable_proof(
    *,
    nodes: Sequence[Mapping[str, object]],
    edges: Sequence[object],
    scope_coverage: Mapping[str, object],
) -> dict[str, object]:
    names = [str(node["name"]) for node in nodes]
    reachable = {name: set() for name in names}
    for value in edges:
        edge = _array_like(value, "producer graph edge")
        reachable[str(edge[0])].add(str(edge[1]))
    changed = True
    while changed:
        changed = False
        for name in names:
            expanded = set(reachable[name])
            for child in tuple(reachable[name]):
                expanded.update(reachable[child])
            if expanded != reachable[name]:
                reachable[name] = expanded
                changed = True
    incomparable = 0
    disjoint = 0
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            left_name = str(left["name"])
            right_name = str(right["name"])
            if right_name in reachable[left_name] or left_name in reachable[right_name]:
                continue
            incomparable += 1
            overlap = False
            for left_scope_value in _array_like(
                left["write_scopes"], f"{left_name} write scopes"
            ):
                left_scope = _mapping_like(left_scope_value, "left write scope")
                for right_scope_value in _array_like(
                    right["write_scopes"], f"{right_name} write scopes"
                ):
                    right_scope = _mapping_like(right_scope_value, "right write scope")
                    if (left_scope["entity"], left_scope["column"]) != (
                        right_scope["entity"],
                        right_scope["column"],
                    ):
                        continue
                    if any(
                        _segments_overlap(
                            _mapping_like(a, "left cell segment"),
                            _mapping_like(b, "right cell segment"),
                            scope_coverage=scope_coverage,
                        )
                        for a in _array_like(
                            left_scope["cell_segments"], "left cell segments"
                        )
                        for b in _array_like(
                            right_scope["cell_segments"], "right cell segments"
                        )
                    ):
                        overlap = True
                        break
                if overlap:
                    break
            if overlap:
                raise RuntimeError(
                    "Incomparable producer nodes have overlapping exact writes: "
                    f"{left_name!r}, {right_name!r}."
                )
            disjoint += 1
    return {
        "requirement": "commute_or_disjoint_writes",
        "proof_method": "transitive_closure_and_closed_cell_segment_intersection",
        "overlap_rule": "explicit_commutativity_proof_required",
        "commutativity_proofs": [],
        "incomparable_pair_count": incomparable,
        "disjoint_write_pair_count": disjoint,
    }


def _project_gap_fill_plan(
    document: Mapping[str, object],
) -> list[dict[str, object]]:
    families = [
        _mapping_like(value, "imputation family")
        for value in _array_like(document["families"], "imputation families")
        if _mapping_like(value, "imputation family").get("stage")
        == "gap_fill_stacked_spine"
    ]
    schedule = _mapping_like(document["gap_fill_schedule"], "gap-fill schedule")
    result: list[dict[str, object]] = []
    for direction_value in _array_like(
        schedule["directions"], "gap-fill schedule directions"
    ):
        direction = _mapping_like(direction_value, "gap-fill schedule direction")
        name = str(direction["name"])
        direction_families = [
            family for family in families if family.get("direction") == name
        ]
        if not direction_families:
            raise RuntimeError(f"Gap-fill direction {name!r} has no families.")
        donor_channels = {
            str(family["donor"]["channel"]) for family in direction_families
        }
        recipient_channels = {
            str(family["recipient"]["channel"]) for family in direction_families
        }
        if len(donor_channels) != 1 or len(recipient_channels) != 1:
            raise RuntimeError(
                f"Gap-fill direction {name!r} has inconsistent channels."
            )
        target_families: dict[str, dict[str, list[str]]] = {}
        absence_rules: list[object] = []
        seen_absence_rules: set[str] = set()
        for family in direction_families:
            _, family_direction, entity, family_name = str(family["id"]).split("/", 3)
            if family_direction != name:
                raise RuntimeError(
                    f"Gap-fill family {family['id']!r} direction disagrees."
                )
            target_families.setdefault(entity, {})[family_name] = [
                str(target["name"])
                for target in _array_like(family["targets"], "family targets")
            ]
            for target in _array_like(family["targets"], "family targets"):
                target_row = _mapping_like(target, "family target")
                for rule in target_row.get("recipient_absence_rules", []):
                    key = json.dumps(rule, sort_keys=True, separators=(",", ":"))
                    if key not in seen_absence_rules:
                        seen_absence_rules.add(key)
                        absence_rules.append(deepcopy(rule))
        result.append(
            {
                "name": name,
                "donor_channel": donor_channels.pop(),
                "recipient_channel": recipient_channels.pop(),
                "target_families": target_families,
                "recipient_absence_rules": absence_rules,
            }
        )
    return result


def _project_gap_fill_schedule(
    document: Mapping[str, object],
    gap_fill_plan: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    schedule = _mapping_like(document["gap_fill_schedule"], "gap-fill schedule")
    plans = {str(plan["name"]): plan for plan in gap_fill_plan}
    directions = []
    total_targets = 0
    for index, direction_value in enumerate(
        _array_like(schedule["directions"], "gap-fill schedule directions")
    ):
        direction = _mapping_like(direction_value, "gap-fill schedule direction")
        plan = plans[str(direction["name"])]
        targets = []
        direction_families = [
            _mapping_like(value, "gap-fill family")
            for value in _array_like(document["families"], "imputation families")
            if _mapping_like(value, "imputation family").get("stage")
            == "gap_fill_stacked_spine"
            and _mapping_like(value, "imputation family").get("direction")
            == direction["name"]
        ]
        for family in direction_families:
            _, family_direction, family_entity, family_name = str(family["id"]).split(
                "/", 3
            )
            if family_direction != direction["name"]:
                raise RuntimeError(
                    f"Gap-fill family {family['id']!r} direction disagrees."
                )
            for target_value in _array_like(family["targets"], "family targets"):
                target_row = _mapping_like(target_value, "family target")
                producer = _mapping_like(
                    target_row["producer_binding"], "gap-fill producer binding"
                )
                targets.append(
                    {
                        "entity": family_entity,
                        "family": family_name,
                        "column": target_row["name"],
                        "producer": producer["operator"],
                        "producer_order_index": producer["order_index"],
                        "execution_scope": producer["execution_scope"],
                        "produced_channel": plan["donor_channel"],
                        "producer_stage": producer["stage"],
                    }
                )
        total_targets += len(targets)
        directions.append(
            {
                "name": direction["name"],
                "order_index": index,
                "donor_channel": plan["donor_channel"],
                "activation_stage": direction["activation_stage"],
                "target_count": len(targets),
                "targets": targets,
            }
        )
    result: dict[str, object] = {
        "status": schedule["activation_policy"],
        "direction_count": len(directions),
        "target_count": total_targets,
        "directions": directions,
    }
    result["sha256"] = _canonical_sha256(result)
    return result


def _primary_family_from_document(
    document: Mapping[str, object],
) -> Mapping[str, object]:
    matches = [
        _mapping_like(value, "primary imputation family")
        for value in _array_like(document["families"], "imputation families")
        if _mapping_like(value, "imputation family").get("stage") == "primary_puf_qrf"
    ]
    if len(matches) != 1:
        raise RuntimeError("Imputation document must have exactly one primary family.")
    return matches[0]


def _require_typed_ref(
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


def _spine_support_role(
    spine_document: Mapping[str, object],
    *,
    role_id: str,
) -> Mapping[str, object]:
    matches = [
        _mapping_like(value, "spine support role")
        for value in _array_like(spine_document["support_roles"], "spine support roles")
        if _mapping_like(value, "spine support role").get("id") == role_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Spine must declare exactly one support role {role_id!r}.")
    return matches[0]


def _build_model_seed_default(value: object, *, location: str) -> int:
    _require_typed_ref(value, _BUILD_MODEL_SEED_REF, location=location)
    defaults = {
        site.default
        for site in LEGACY_V1_PROTOCOL.sites
        if site.value_source == _BUILD_MODEL_SEED_REF["value_source"]
        and site.default is not None
    }
    if len(defaults) != 1:
        raise RuntimeError(
            "legacy-v1 build_model_seed sites no longer share one default: "
            f"{sorted(defaults)!r}."
        )
    return defaults.pop()


def _target_period(
    bundle_document: Mapping[str, object],
    value: object,
    *,
    location: str,
) -> int:
    _require_typed_ref(value, _TARGET_PERIOD_REF, location=location)
    dataset_run = _mapping_like(bundle_document["dataset_run"], "bundle dataset_run")
    period = dataset_run.get("target_period")
    if isinstance(period, bool) or not isinstance(period, int):
        raise RuntimeError("bundle dataset_run target_period must be an integer.")
    return period


def _post_clone_operator_order(
    spine_document: Mapping[str, object],
    value: object,
    *,
    location: str,
) -> list[str]:
    _require_typed_ref(value, _SOURCE_OPERATOR_REGISTRY_REF, location=location)
    pipeline = _mapping_like(
        spine_document["pipeline_contract"], "spine pipeline contract"
    )
    return [
        str(operator)
        for operator in _array_like(
            pipeline["post_clone_source_operator_order"],
            "spine post-clone source operator order",
        )
    ]


def _inflate_virtual_resource(
    *,
    document: Mapping[str, object],
    node: Mapping[str, object],
    resource_row: Mapping[str, object],
    late_families: Mapping[str, Mapping[str, object]],
    sources_document: Mapping[str, object],
    spine_document: Mapping[str, object],
    bundle_document: Mapping[str, object],
    schedule_sha256: str,
    schedule_payload_sha256: str,
) -> tuple[str, dict[str, object]]:
    resource = {
        key: deepcopy(value)
        for key, value in resource_row.items()
        if key not in {"id", "kind"}
    }
    resource_id = str(resource_row["id"])
    binding = deepcopy(
        dict(_mapping_like(resource["binding"], f"resource {resource_id} binding"))
    )
    resource_kind = str(binding["resource_kind"])
    predictor_blocks = _mapping_like(
        document["predictor_blocks"], "imputation predictor blocks"
    )
    transfer_execution = _mapping_like(
        document["transfer_execution"], "imputation transfer execution"
    )
    models = _mapping_like(document["models"], "imputation models")
    qrf_model = _mapping_like(models["regime_gated_qrf"], "regime-gated QRF model")
    qrf_params = _mapping_like(qrf_model["params"], "regime-gated QRF parameters")
    graph = _mapping_like(document["producer_graph"], "producer graph")
    resource_semantics = _mapping_like(
        graph["resource_semantics"], "producer resource semantics"
    )
    resolution = _mapping_like(
        resource_semantics["resolution"], "producer resource resolution"
    )
    if resolution.get("digest_rule") != (
        "canonical_sha256(resolved_payload_without_sha256)"
    ):
        raise RuntimeError("Producer resource digest rule is unsupported.")
    build_model_seed = _build_model_seed_default(
        resolution["build_model_seed_ref"],
        location="resource semantics build_model_seed_ref",
    )

    if resource_kind == "late_transfer_model_config":
        producer = str(node["name"])
        family = late_families[producer]
        targets = [
            str(target["name"])
            for target in _array_like(family["targets"], "late family targets")
        ]
        entity = str(_array_like(family["entities"], "late family entities")[0])
        donor_contract = _mapping_like(
            family["donor_contract"], "late family donor contract"
        )
        binding.update(
            {
                "producer": producer,
                "n_estimators": qrf_params["n_estimators"],
                "entity": entity,
                "family": producer.split("/", 1)[1],
                "ordered_targets": targets,
                "max_targets_per_fit": family["max_targets_per_fit"],
                "donor_spine": donor_contract["spine"],
                "donor_channel": donor_contract["resource_binding_channel"],
                "donor_selection": donor_contract["selection"],
                "donor_projection": donor_contract["projection"],
                "transfer_execution_contract": _project_transfer_execution_identity(
                    transfer_execution,
                    predictor_blocks,
                    profile_id=str(family["execution_contract"]),
                    targets=targets,
                ),
            }
        )
        binding["seed"] = build_model_seed
    elif resource_kind == "primary_puf_execution_config":
        family = _primary_family_from_document(document)
        targets = [
            _mapping_like(value, "primary target")
            for value in _array_like(family["targets"], "primary targets")
        ]
        qrf = deepcopy(dict(_mapping_like(binding["qrf"], "primary QRF binding")))
        predictor_ref = str(_array_like(family["predictors"], "primary predictors")[0])
        qrf.update(
            {
                "n_estimators": qrf_params["n_estimators"],
                "predictors": deepcopy(
                    list(
                        _array_like(
                            _mapping_like(
                                predictor_blocks[predictor_ref],
                                f"predictor block {predictor_ref}",
                            )["columns"],
                            f"predictor block {predictor_ref} columns",
                        )
                    )
                ),
                "person_outputs": [
                    target["name"] for target in targets if target["entity"] == "person"
                ],
                "tax_unit_outputs": [
                    target["name"]
                    for target in targets
                    if target["entity"] == "tax_unit"
                ],
            }
        )
        qrf["seed"] = build_model_seed
        tail = deepcopy(
            dict(_mapping_like(binding["capital_gains_tail"], "capital-gains tail"))
        )
        tail["seed"] = build_model_seed
        _require_typed_ref(
            tail.pop("support_contract_ref"),
            _PUF_TAIL_SUPPORT_REF,
            location=f"resource {resource_id} tail support_contract_ref",
        )
        support_role = _spine_support_role(
            spine_document,
            role_id=_PUF_TAIL_SUPPORT_REF["support_role"],
        )
        tail_support = _mapping_like(support_role["tail_support"], "spine tail support")
        tail["support_contract"] = deepcopy(tail_support["legacy_contract"])
        soi_bands = deepcopy(
            dict(
                _mapping_like(
                    tail["soi_e19200_agi_bands"],
                    "capital-gains-tail SOI AGI bands",
                )
            )
        )
        runtime_bands = deepcopy(
            dict(
                _mapping_like(
                    soi_bands["runtime_agi_bands"],
                    "capital-gains-tail runtime AGI bands",
                )
            )
        )
        runtime_bands["sha256"] = _canonical_sha256(runtime_bands)
        soi_bands["runtime_agi_bands"] = runtime_bands
        tail["soi_e19200_agi_bands"] = soi_bands
        _restore_primary_tail_legacy_numbers(tail)
        _require_typed_ref(
            resolution["clone_attachment_ref"],
            _PUF_ATTACHMENT_REF,
            location="resource semantics clone_attachment_ref",
        )
        attachment_role = _spine_support_role(
            spine_document,
            role_id=_PUF_ATTACHMENT_REF["support_role"],
        )
        attachment = _mapping_like(
            attachment_role["attachment"], "spine PUF attachment"
        )
        fraction = _mapping_like(
            attachment["fraction"], "spine PUF attachment fraction"
        )
        seed = _mapping_like(attachment["seed"], "spine PUF attachment seed")
        assembly = _mapping_like(spine_document["assembly"], "spine assembly")
        binding["clone_attachment"] = {
            "fraction": _legacy_float(
                fraction["default"], location="spine PUF attachment fraction/default"
            ),
            "seed": seed["default"],
            "support_channels": [
                assembly["mass_anchor_channel"],
                attachment_role["id"],
            ],
            "puf_clone_index": attachment_role["clone_index"],
        }
        doctrines = _mapping_like(binding["doctrines"], "primary PUF doctrines")
        output_universes = _mapping_like(
            doctrines["whole_pool_output_universes"],
            "primary PUF whole-pool output universes",
        )
        s_corp = _mapping_like(
            output_universes["person.s_corp_income"],
            "primary PUF person.s_corp_income doctrine",
        )
        s_corp["materialized_value"] = _legacy_float(  # type: ignore[index]
            s_corp["materialized_value"],
            location="primary PUF person.s_corp_income materialized_value",
        )
        binding["qrf"] = qrf
        binding["capital_gains_tail"] = tail
    elif resource_kind == "late_transfer_target_bank":
        dynamic_field = deepcopy(
            dict(
                _mapping_like(
                    resource["dynamic_field"],
                    f"late-transfer target-bank dynamic field for {node['name']}",
                )
            )
        )
        derivation = deepcopy(
            dict(
                _mapping_like(
                    dynamic_field["derivation"],
                    f"late-transfer target-bank derivation for {node['name']}",
                )
            )
        )
        derivation["late_producer_dag_sha256"] = schedule_sha256
        derivation["late_producer_schedule_sha256"] = schedule_payload_sha256
        dynamic_field["derivation"] = derivation
        resource["dynamic_field"] = dynamic_field
    elif resource_kind == "post_clone_source_execution_config":
        metadata = _mapping_like(
            graph["resource_semantics"], "producer resource semantics"
        )
        defaults = _mapping_like(
            metadata["source_execution_defaults"], "source execution defaults"
        )
        operator_registry = _post_clone_operator_order(
            spine_document,
            defaults["operator_registry_ref"],
            location="resource semantics source operator_registry_ref",
        )
        time_period = _target_period(
            bundle_document,
            defaults["time_period_ref"],
            location="resource semantics source time_period_ref",
        )
        operator = str(node["name"]).removeprefix("source:")
        if node["name"] != f"source:{operator}":
            raise RuntimeError(f"Malformed source producer name {node['name']!r}.")
        removed = _mapping_like(
            binding["formula_owned_outputs_removed"],
            f"formula-owned outputs for {operator}",
        )
        output_family: dict[str, set[str]] = {}
        for output_value in _array_like(node["outputs"], f"{operator} outputs"):
            output = _mapping_like(output_value, f"{operator} output")
            if not str(output["column"]).startswith("@"):
                output_family.setdefault(str(output["entity"]), set()).add(
                    str(output["column"])
                )
        for entity, columns in removed.items():
            output_family.setdefault(str(entity), set()).update(
                str(column)
                for column in _array_like(columns, f"removed {entity} outputs")
            )
        binding.update(
            {
                "operator": operator,
                "phase": defaults["phase"],
                "operator_registry": operator_registry,
                "declared_output_family": {
                    entity: sorted(columns)
                    for entity, columns in sorted(output_family.items())
                },
                "seed": build_model_seed,
                "time_period": (
                    None
                    if operator == "impute_us_housing_assistance_to_puf_support"
                    else time_period
                ),
            }
        )
        stage_ref = binding.pop("source_stage_ref")
        if stage_ref is None:
            binding["source_stage_spec"] = None
        else:
            ref = _mapping_like(stage_ref, f"source stage ref for {operator}")
            stage_id = str(ref["stage_id"])
            stages = {
                str(stage["stage"]): stage
                for value in _array_like(sources_document["stages"], "source stages")
                for stage in [_mapping_like(value, "source stage")]
            }
            if len(stages) != len(
                _array_like(sources_document["stages"], "source stages")
            ):
                raise RuntimeError("Source document repeats a stage id.")
            if stage_id not in stages:
                raise RuntimeError(
                    f"Source producer {operator!r} references missing stage {stage_id!r}."
                )
            asset = _mapping_like(
                sources_document["stage_asset"], "source document stage asset"
            )
            if asset["id"] != ref["asset_id"]:
                raise RuntimeError(
                    f"Source producer {operator!r} references source-stage asset "
                    f"{ref['asset_id']!r}, but sources declares {asset['id']!r}."
                )
            manifest = _mapping_like(
                sources_document["stage_manifest"], "source stage manifest"
            )
            resolver = _mapping_like(ref["runtime_resolver"], "source runtime resolver")
            stage_row = _mapping_like(stages[stage_id], f"source stage {stage_id}")
            stage = {
                "stage": stage_row["stage"],
                "survey": stage_row["survey"],
                "source": stage_row["source"],
                "grain": stage_row["grain"],
                "artifacts": deepcopy(list(stage_row.get("artifacts", []))),
                "operations": [
                    {
                        "kind": operation["kind"],
                        "parameters": {
                            key: deepcopy(value)
                            for key, value in operation.items()
                            if key != "kind"
                        },
                    }
                    for value in _array_like(
                        stage_row["operations"], f"source stage {stage_id} operations"
                    )
                    for operation in [
                        _mapping_like(value, f"source stage {stage_id} operation")
                    ]
                ],
                "outputs": deepcopy(list(stage_row["outputs"])),
                "nonnegative_outputs": deepcopy(
                    list(stage_row.get("nonnegative_outputs", []))
                ),
                "notes": stage_row.get("notes", ""),
            }
            _restore_source_stage_legacy_numbers(stage)
            binding["source_stage_spec"] = {
                "asset": asset["path"],
                "asset_sha256": asset["sha256"],
                "manifest": deepcopy(dict(manifest)),
                "stage_name": stage_id,
                "resolved_stage_spec": stage,
                "resolved_stage_spec_sha256": _canonical_sha256(stage),
                "runtime_stage_spec_resolver": {
                    "module": resolver["module"],
                    "callable": resolver["callable"],
                },
                "runtime_stage_spec_verified": True,
            }
    elif resource_kind == "primary_qrf_checkpoint":
        family = _primary_family_from_document(document)
        target_order = [
            target["name"]
            for target in _array_like(family["targets"], "primary targets")
        ]
        checkpoint = _mapping_like(document["primary_checkpoint"], "primary checkpoint")
        binding.update(
            {
                "checkpoint_schema_version": checkpoint["schema_version"],
                "target_order": target_order,
                "target_order_sha256": _canonical_sha256(target_order),
            }
        )
    elif resource_kind == "acs_pums_earnings_universe_execution_config":
        identity = deepcopy(
            dict(
                _mapping_like(
                    binding["contract_identity"],
                    "ACS earnings-universe contract identity",
                )
            )
        )
        identity["sha256"] = _canonical_sha256(identity)
        binding["contract_identity"] = identity

    resource["binding"] = binding
    return resource_id, resource


def _derive_canonical_schedule(
    document: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    """Compile and validate RAW dependencies and family-owned outputs."""

    graph = _mapping_like(document["producer_graph"], "producer graph")
    nodes = [
        _mapping_like(value, "producer graph node")
        for value in _array_like(graph["nodes"], "producer graph nodes")
    ]
    by_name = {str(node["name"]): node for node in nodes}
    if len(by_name) != len(nodes):
        raise RuntimeError("Producer graph repeats a node name.")
    if any(node["id"] != node["name"] for node in nodes):
        raise RuntimeError("Producer graph node ids must equal legacy names.")
    external = set(
        str(stage) for stage in _array_like(graph["external_stages"], "external stages")
    )
    if external & set(by_name):
        raise RuntimeError("Producer and external stage ids overlap.")
    scope_coverage = _mapping_like(graph["scope_coverage"], "scope coverage")
    declared_scopes = _mapping_like(scope_coverage["declared"], "declared scopes")
    node_rank = {str(node["name"]): index for index, node in enumerate(nodes)}
    compiled_outputs = _compile_node_outputs(document, graph)
    edges: set[tuple[str, str]] = set()
    contracts = []
    predecessors = {name: set() for name in by_name}
    proof_nodes: list[dict[str, object]] = []
    for name in sorted(by_name):
        node = by_name[name]
        outputs = [
            {
                "entity": output["entity"],
                "column": output["column"],
                "coverage_scope": output["coverage_scope"],
            }
            for value in compiled_outputs[name]
            for output in [_mapping_like(value, f"{name} output")]
        ]
        expected_write_scopes = [
            _write_scope(
                producer=name,
                output=output,
                ownership_rows=_array_like(
                    graph["ownership_matrix"], "overlap ownership"
                ),
            )
            for output in outputs
        ]
        proof_node = deepcopy(dict(node))
        proof_node["write_scopes"] = expected_write_scopes
        proof_nodes.append(proof_node)
        inputs = deepcopy(list(_array_like(node["inputs"], f"{name} inputs")))
        for value in inputs:
            item = _mapping_like(value, f"{name} input")
            producer_name = str(item["producing_stage"])
            if producer_name in external:
                continue
            if producer_name not in by_name:
                raise RuntimeError(
                    f"Producer node {name!r} references unknown stage {producer_name!r}."
                )
            producer_outputs = [
                _mapping_like(value, f"{producer_name} output")
                for value in _array_like(
                    compiled_outputs[producer_name], f"{producer_name} outputs"
                )
                if _mapping_like(value, f"{producer_name} output")["entity"]
                == item["entity"]
                and _mapping_like(value, f"{producer_name} output")["column"]
                == item["column"]
            ]
            required_scope = str(item["required_scope"])
            if not any(
                required_scope
                in set(
                    _array_like(
                        declared_scopes.get(
                            str(output["coverage_scope"]),
                            [output["coverage_scope"]],
                        ),
                        "covered scopes",
                    )
                )
                for output in producer_outputs
            ):
                raise RuntimeError(
                    f"Producer input {name}/{item['entity']}.{item['column']} has "
                    "no scope-compatible producing output."
                )
            edges.add((producer_name, name))
            predecessors[name].add(producer_name)
        contracts.append(
            {
                "name": name,
                "kind": node["kind"],
                "inputs": inputs,
                "outputs": outputs,
            }
        )
    adjacency = {name: set() for name in by_name}
    indegree = {name: 0 for name in by_name}
    for producer, consumer in edges:
        adjacency[producer].add(consumer)
        indegree[consumer] += 1
    remaining = set(by_name)
    waves: list[list[str]] = []
    while remaining:
        ready = sorted(
            (name for name in remaining if indegree[name] == 0),
            key=node_rank.__getitem__,
        )
        if not ready:
            raise RuntimeError("Producer graph contains a dependency cycle.")
        waves.append(ready)
        remaining.difference_update(ready)
        for producer in ready:
            for consumer in adjacency[producer]:
                indegree[consumer] -= 1
    order = [name for wave in waves for name in wave]
    sorted_edges = [
        list(edge)
        for edge in sorted(
            edges,
            key=lambda edge: (node_rank[edge[0]], node_rank[edge[1]]),
        )
    ]
    _incomparable_proof(
        nodes=proof_nodes,
        edges=sorted_edges,
        scope_coverage=scope_coverage,
    )
    payload = {
        "schema_version": graph["graph_schema_version"],
        "external_stages": sorted(external),
        "scope_coverage": deepcopy(dict(scope_coverage)),
        "contracts": contracts,
        "edges": sorted_edges,
        "waves": waves,
        "order": order,
    }
    return payload, _canonical_sha256(payload)


def _project_resource_semantics(
    document: Mapping[str, object],
    *,
    sources_document: Mapping[str, object],
    spine_document: Mapping[str, object],
    bundle_document: Mapping[str, object],
) -> dict[str, object]:
    graph = _mapping_like(document["producer_graph"], "producer graph")
    late_families = {
        str(family["runtime_name"]): family
        for value in _array_like(document["families"], "imputation families")
        for family in [_mapping_like(value, "imputation family")]
        if family.get("stage") == "late_producer_dag"
    }
    nodes_by_name = {
        str(node["name"]): node
        for value in _array_like(graph["nodes"], "producer graph nodes")
        for node in [_mapping_like(value, "producer graph node")]
    }
    metadata = _mapping_like(graph["resource_semantics"], "producer resource semantics")
    compiled_schedule, schedule_sha256 = _derive_canonical_schedule(document)
    ownership = _project_overlap_ownership(graph)
    schedule_receipt = _project_late_schedule(document, graph, ownership)
    producers = []
    for producer_name in _array_like(
        compiled_schedule["order"], "compiled producer graph order"
    ):
        node = nodes_by_name[str(producer_name)]
        resources = dict(
            _inflate_virtual_resource(
                document=document,
                node=node,
                resource_row=_mapping_like(resource, "virtual resource"),
                late_families=late_families,
                sources_document=sources_document,
                spine_document=spine_document,
                bundle_document=bundle_document,
                schedule_sha256=schedule_sha256,
                schedule_payload_sha256=str(schedule_receipt["payload_sha256"]),
            )
            for resource in _array_like(
                node["virtual_resources"], "node virtual resources"
            )
        )
        producers.append(
            {
                "producer": node["name"],
                "kind": node["kind"],
                "resources": resources,
            }
        )
    primary = next(
        producer for producer in producers if producer["producer"] == "primary_puf_qrf"
    )
    primary_binding = primary["resources"]["tax_unit.@primary_puf_execution_config"][
        "binding"
    ]
    primary_binding["qrf"]["worker_execution"] = _resolve_worker_execution(
        _mapping_like(
            primary_binding["qrf"]["worker_execution"],
            "primary-QRF worker template",
        )
    )
    result = {
        "artifact_kind": metadata["artifact_kind"],
        "schema_version": metadata["schema_version"],
        "producer_schedule_sha256": schedule_sha256,
        "producer_schedule_payload_sha256": schedule_receipt["payload_sha256"],
        "producer_count": len(producers),
        "producers": producers,
    }
    result["sha256"] = _canonical_sha256(result)
    return result


def _project_overlap_ownership(
    graph: Mapping[str, object],
) -> dict[str, object]:
    result = deepcopy(
        dict(_mapping_like(graph["ownership_contract"], "ownership contract"))
    )
    ownership = deepcopy(list(graph["ownership_matrix"]))
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for value in ownership:
        row = _mapping_like(value, "overlap ownership row")
        grouped.setdefault((str(row["entity"]), str(row["target"])), []).append(row)
    targets = []
    # The matrix is the sole authority for both membership and legacy row
    # order.  Preserve each target's first appearance: sorting here changes
    # the byte-attested overlap receipt and therefore the schedule payload.
    for (entity, target), rows in grouped.items():
        actions = [
            _mapping_like(value, "overlap producer action")
            for row in rows
            for value in _array_like(row["producer_actions"], "producer actions")
        ]
        source_producers = {
            str(action["producer"])
            for action in actions
            if str(action["producer"]).startswith("source:")
        }
        transfer_producers = {
            str(action["producer"])
            for action in actions
            if str(action["producer"]).startswith("transfer:")
        }
        if len(source_producers) != 1 or len(transfer_producers) != 1:
            raise RuntimeError(
                f"Ownership rows for {entity}.{target} do not identify one source "
                "and transfer producer."
            )
        source_actions = {
            str(action["action"])
            for action in actions
            if str(action["producer"]) in source_producers
        }
        if "consume_only_byte_exact_noop" in source_actions:
            source_touch = "consume_only_byte_exact_noop"
        elif "final_write" in source_actions:
            source_touch = "persisted_owner_last_write"
        else:
            raise RuntimeError(
                f"Ownership rows for {entity}.{target} have no reviewed source touch."
            )
        targets.append(
            {
                "entity": entity,
                "target": target,
                "source_producer": next(iter(source_producers)),
                "source_touch": source_touch,
                "transfer_producer": next(iter(transfer_producers)),
            }
        )
    result["targets"] = targets
    result["ownership"] = ownership
    result["sha256"] = _canonical_sha256(result)
    return result


def _derive_input_inventory(
    node: Mapping[str, object],
    *,
    operator: str,
    default_scope: str,
) -> dict[str, object]:
    requirements = []
    node_name = str(node["name"])
    for value in _array_like(node["inputs"], f"{node_name} inputs"):
        input_row = _mapping_like(value, f"{node_name} input")
        column = str(input_row["column"])
        if not column.startswith("@effective:"):
            continue
        label = column.removeprefix("@effective:")
        optional_receipt = f"optional_input:{node_name}:{label}"
        absence_receipts = list(
            _array_like(
                input_row["tolerated_absence_receipts"],
                f"{node_name}/{label} tolerated absence receipts",
            )
        )
        if absence_receipts not in ([], [optional_receipt]):
            raise RuntimeError(
                f"Effective input {node_name}/{label} has unsupported absence receipts."
            )
        required_scope = str(input_row["required_scope"])
        requirements.append(
            {
                "label": label,
                "optional": absence_receipts == [optional_receipt],
                "required_scope": (
                    None if required_scope == default_scope else required_scope
                ),
                "alternatives": deepcopy(list(input_row["alternatives"])),
            }
        )
    requirements.sort(key=lambda row: str(row["label"]))
    return {"operator": operator, "requirements": requirements}


def _derive_input_inventories(graph: Mapping[str, object]) -> dict[str, object]:
    inventories: dict[str, object] = {"sources": [], "transfers": []}
    for value in _array_like(graph["nodes"], "producer graph nodes"):
        node = _mapping_like(value, "producer graph node")
        name = str(node["name"])
        kind = str(node["kind"])
        if kind == "acs_earnings_universe":
            inventories["acs_earnings_universe"] = _derive_input_inventory(
                node, operator=name, default_scope="acs_source"
            )
        elif kind == "primary_puf":
            inventories["primary_puf"] = _derive_input_inventory(
                node, operator=name, default_scope="whole_pool"
            )
        elif kind == "post_clone_source":
            if not name.startswith("source:"):
                raise RuntimeError(
                    f"Source producer {name!r} lacks the source: prefix."
                )
            inventories["sources"].append(
                _derive_input_inventory(
                    node,
                    operator=name.removeprefix("source:"),
                    default_scope="asec_source",
                )
            )
        elif kind == "late_transfer":
            inventories["transfers"].append(
                _derive_input_inventory(node, operator=name, default_scope="whole_pool")
            )
    if set(inventories) != {
        "acs_earnings_universe",
        "primary_puf",
        "sources",
        "transfers",
    }:
        raise RuntimeError("Producer nodes do not define all legacy input inventories.")
    for key in ("sources", "transfers"):
        inventories[key].sort(key=lambda row: str(row["operator"]))
    return inventories


def _derive_transfer_groups(document: Mapping[str, object]) -> list[dict[str, object]]:
    groups = []
    for value in _array_like(document["families"], "imputation families"):
        family = _mapping_like(value, "imputation family")
        if family.get("stage") != "late_producer_dag":
            continue
        entities = list(_array_like(family["entities"], "late family entities"))
        if len(entities) != 1:
            raise RuntimeError(f"Late family {family['id']!r} must have one entity.")
        entity = str(entities[0])
        prefix = f"late/{entity}/"
        family_id = str(family["id"])
        if not family_id.startswith(prefix):
            raise RuntimeError(f"Late family {family_id!r} does not match {entity!r}.")
        groups.append(
            {
                "name": family["runtime_name"],
                "entity": entity,
                "family": family_id.removeprefix(prefix),
                "targets": [
                    _mapping_like(target, "late family target")["name"]
                    for target in _array_like(family["targets"], "late family targets")
                ],
            }
        )
    return groups


def _project_late_schedule(
    document: Mapping[str, object],
    graph: Mapping[str, object],
    ownership: Mapping[str, object],
) -> dict[str, object]:
    inventories = _derive_input_inventories(graph)
    compiled_schedule, schedule_sha256 = _derive_canonical_schedule(document)
    transfers = deepcopy(list(inventories["transfers"]))
    sources = deepcopy(list(inventories["sources"]))
    groups = _derive_transfer_groups(document)
    payload = {
        "schema_version": graph["schedule_payload_schema_version"],
        "overlap_ownership": deepcopy(dict(ownership)),
        "execution_receipt_contract": deepcopy(graph["execution_receipt_contract"]),
        "schedule_sha256": schedule_sha256,
        "external_stages": deepcopy(compiled_schedule["external_stages"]),
        "order": deepcopy(compiled_schedule["order"]),
        "waves": deepcopy(compiled_schedule["waves"]),
        "edges": deepcopy(compiled_schedule["edges"]),
        "transfer_groups": groups,
        "source_input_inventories": sources,
        "primary_puf_input_inventory": deepcopy(inventories["primary_puf"]),
        "acs_earnings_universe_input_inventory": deepcopy(
            inventories["acs_earnings_universe"]
        ),
        "transfer_input_inventories": transfers,
    }
    payload_sha256 = _canonical_sha256(payload)
    return {
        **payload,
        "payload_sha256": payload_sha256,
        "producer_count": len(graph["nodes"]),
        "source_producer_count": len(sources),
        "transfer_group_count": len(groups),
        "transfer_target_count": sum(len(group["targets"]) for group in groups),
        "status": "derived_and_import_validated",
    }


def _assert_compiler_semantics(document: Mapping[str, object]) -> None:
    """Consume compiler-only references fail-closed."""

    graph = _mapping_like(document["producer_graph"], "producer graph")
    node_ids = {
        str(node["id"])
        for value in _array_like(graph["nodes"], "producer graph nodes")
        for node in [_mapping_like(value, "producer graph node")]
    }
    models = _mapping_like(document["models"], "imputation models")
    profiles = _mapping_like(
        _mapping_like(document["transfer_execution"], "transfer execution")["profiles"],
        "transfer profiles",
    )
    primary = _primary_family_from_document(document)
    if primary["execution_contract"] not in node_ids:
        raise RuntimeError(
            "Primary family execution_contract must resolve to its producer node."
        )
    for value in _array_like(document["families"], "imputation families"):
        family = _mapping_like(value, "imputation family")
        if family["model"] not in models:
            raise RuntimeError(
                f"Imputation family {family['id']!r} references an unknown model."
            )
        if (
            family["stage"] != "primary_puf_qrf"
            and family["execution_contract"] not in profiles
        ):
            raise RuntimeError(
                f"Imputation family {family['id']!r} references an unknown transfer profile."
            )
    derive_primary_effective_predictor_tuples(document)


def derive_primary_effective_predictor_tuples(
    document: Mapping[str, object],
) -> list[dict[str, object]]:
    """Compile the primary chain from its sole predictor block and target order."""

    primary = _primary_family_from_document(document)
    if primary.get("chaining") != "base_plus_preceding_declared_targets":
        raise RuntimeError("Primary family has an unsupported chaining contract.")
    predictor_blocks = _mapping_like(document["predictor_blocks"], "predictor blocks")
    predictor_refs = _array_like(primary["predictors"], "primary predictors")
    if len(predictor_refs) != 1:
        raise RuntimeError(
            "Primary family must reference exactly one base predictor block."
        )
    predictor_ref = str(predictor_refs[0])
    if predictor_ref not in predictor_blocks:
        raise RuntimeError(
            f"Primary family references missing predictor block {predictor_ref!r}."
        )
    base = list(
        _array_like(
            _mapping_like(predictor_blocks[predictor_ref], "primary predictor block")[
                "columns"
            ],
            "primary predictor columns",
        )
    )
    preceding: list[object] = []
    result = []
    for value in _array_like(primary["targets"], "primary targets"):
        target = _mapping_like(value, "primary target")
        result.append(
            {
                "target": target["name"],
                "entity": target["entity"],
                "predictors": [*base, *preceding],
            }
        )
        preceding.append(target["name"])
    return result


def project_imputation_legacy_payloads(
    document: Mapping[str, object],
    *,
    sources_document: Mapping[str, object],
    spine_document: Mapping[str, object],
    bundle_document: Mapping[str, object],
) -> dict[str, object]:
    """Reconstruct every constants-era imputation identity from typed fields."""

    _assert_compiler_semantics(document)
    gap_fill_plan = _project_gap_fill_plan(document)
    graph = _mapping_like(document["producer_graph"], "producer graph")
    ownership = _project_overlap_ownership(graph)
    primary_family = _primary_family_from_document(document)
    primary_targets = [
        _mapping_like(value, "primary target")
        for value in _array_like(primary_family["targets"], "primary targets")
    ]
    target_order = [target["name"] for target in primary_targets]
    predictor_blocks = _mapping_like(document["predictor_blocks"], "predictor blocks")
    transfer_execution = _mapping_like(
        document["transfer_execution"], "transfer execution"
    )
    profiles = _mapping_like(transfer_execution["profiles"], "transfer profiles")
    early_contracts = []
    for direction in gap_fill_plan:
        ordered_targets = [
            target
            for entity_families in direction["target_families"].values()
            for targets in entity_families.values()
            for target in targets
        ]
        profile_id = "acs_transfer_early"
        early_contracts.append(
            {
                "id": f"early/{direction['name']}",
                "direction": direction["name"],
                "derive_schedule_d": profiles[profile_id]["derive_schedule_d"],
                "ordered_targets": ordered_targets,
                "identity": _project_transfer_execution_identity(
                    transfer_execution,
                    predictor_blocks,
                    profile_id=profile_id,
                    targets=ordered_targets,
                ),
            }
        )
    late_contracts = []
    for value in _array_like(document["families"], "imputation families"):
        family = _mapping_like(value, "imputation family")
        if family.get("stage") != "late_producer_dag":
            continue
        ordered_targets = [
            target["name"]
            for target in _array_like(family["targets"], "late family targets")
        ]
        profile_id = str(family["execution_contract"])
        late_contracts.append(
            {
                "id": family["id"],
                "producer": family["runtime_name"],
                "derive_schedule_d": profiles[profile_id]["derive_schedule_d"],
                "ordered_targets": ordered_targets,
                "identity": _project_transfer_execution_identity(
                    transfer_execution,
                    predictor_blocks,
                    profile_id=profile_id,
                    targets=ordered_targets,
                ),
            }
        )
    checkpoint = _mapping_like(document["primary_checkpoint"], "primary checkpoint")
    predictor_ref = str(
        _array_like(primary_family["predictors"], "primary predictors")[0]
    )
    return {
        "gap_fill_plan": gap_fill_plan,
        "gap_fill_producer_schedule_receipt": _project_gap_fill_schedule(
            document, gap_fill_plan
        ),
        "primary_qrf": {
            "predictors": deepcopy(predictor_blocks[predictor_ref]["columns"]),
            "person_outputs": [
                target["name"]
                for target in primary_targets
                if target["entity"] == "person"
            ],
            "tax_unit_outputs": [
                target["name"]
                for target in primary_targets
                if target["entity"] == "tax_unit"
            ],
            "target_order": target_order,
            "target_order_sha256": _canonical_sha256(target_order),
            "target_order_digest_rule": checkpoint["target_order_digest_rule"],
            "checkpoint_schema_version": checkpoint["schema_version"],
        },
        "late_producer_schedule_receipt": _project_late_schedule(
            document, graph, ownership
        ),
        "late_producer_resource_semantics": _project_resource_semantics(
            document,
            sources_document=sources_document,
            spine_document=spine_document,
            bundle_document=bundle_document,
        ),
        "overlap_ownership": ownership,
        "transfer_execution_contract_identities": {
            "base": _project_transfer_execution_identity(
                transfer_execution,
                predictor_blocks,
                profile_id="acs_transfer_default",
                targets=[],
            ),
            "early": early_contracts,
            "late": late_contracts,
        },
    }


__all__ = [
    "derive_primary_effective_predictor_tuples",
    "project_imputation_legacy_payloads",
]
