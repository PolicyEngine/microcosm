"""Version-two receipt validation and assignment, independent of country packs."""

from __future__ import annotations

import itertools
import json
import math
from collections import deque
from collections.abc import Mapping
from fractions import Fraction
from numbers import Real

import numpy as np

from microcosm.build.calibration_predicates import calibration_predicate_mask
from microcosm.frame.concepts import derive_take_up_draws
from microcosm.frame.input_closure import UndeterminedAction, resolve_judgments

RECEIPT_RELATIVE_TOLERANCE = 1 / 10**12
RECEIPT_UNIT_FIELDS = ("entitlement_entity", "payee", "draw_unit", "projection")


class ReceiptAssignmentRefused(ValueError):  # noqa: N818
    """An assignment refusal retaining its diagnostic result, including order."""

    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


def _text(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"Receipt {name} must be non-empty text.")
    return value


def _strings(values, name, *, nonempty=False):
    if not isinstance(values, list | tuple) or (nonempty and not values):
        raise ValueError(
            f"Receipt {name} must be a {'non-empty ' if nonempty else ''}list."
        )
    result = [_text(value, name) for value in values]
    if len(set(result)) != len(result):
        raise ValueError(f"Receipt {name} must contain unique names.")
    return result


def receipt_draw_key(program, suffix):
    """A tuple encoding: even adversarial delimiters cannot alias another pair."""
    return json.dumps([program, suffix], ensure_ascii=True, separators=(",", ":"))


def reversed_receipt_groups(groups, order):
    if order not in {"declared", "reversed"}:
        raise ValueError("Receipt exclusion_order must be declared or reversed.")
    return [
        list(group) if order == "declared" else list(reversed(group))
        for group in groups
    ]


def receipt_dispatch_tables(tables, row):
    """Dispatch through each declared unit, ready for the declared-unit extension."""
    # [BE] Reserved fields drive the declared-unit receipt extension's dispatch.
    return {field: tables[row[field]] for field in RECEIPT_UNIT_FIELDS}


def validate_receipt_v2(contract):
    required = {
        "contract_version",
        "seed_column",
        "default_assignment_profile",
        "assignment_knob",
        "assignment_profiles",
        "programs",
        "exclusion_groups",
    }
    if not required <= set(contract) or set(contract) - required - {
        "unmodelled_programs"
    }:
        raise ValueError("Receipt v2 contract has missing or unknown fields.")
    if (
        type(contract["contract_version"]) is not int
        or contract["contract_version"] != 2
    ):
        raise ValueError("Receipt contract_version must be 2.")
    seed = _text(contract["seed_column"], "seed_column")
    default = _text(
        contract["default_assignment_profile"], "default_assignment_profile"
    )
    _text(contract["assignment_knob"], "assignment_knob")
    profiles = contract["assignment_profiles"]
    if not isinstance(profiles, Mapping) or not profiles:
        raise ValueError("Receipt assignment_profiles must be a non-empty object.")
    for name, profile in profiles.items():
        _text(name, "profile")
        if (
            not isinstance(profile, Mapping)
            or not {"draw_key_suffix"} <= set(profile)
            or set(profile) - {"draw_key_suffix", "exclusion_order"}
        ):
            raise ValueError("Receipt profile has missing or unknown fields.")
        if not isinstance(profile["draw_key_suffix"], str):
            raise ValueError("Receipt draw_key_suffix must be text.")
        reversed_receipt_groups([], profile.get("exclusion_order", "declared"))
    if default not in profiles:
        raise ValueError(
            "Receipt default_assignment_profile must name an assignment profile."
        )
    programs = contract["programs"]
    if not isinstance(programs, list | tuple) or not programs:
        raise ValueError("Receipt programs must be a non-empty ordered list.")
    parsed, identities, draws = [], set(), {}
    for row in programs:
        required_row = {
            "program",
            "output",
            "judgment_columns",
            "judgment_reduction",
            "payment_column",
        }
        allowed = required_row | {
            "target",
            "strata",
            "outside_strata",
            "exclude_columns",
            *RECEIPT_UNIT_FIELDS,
        }
        if (
            not isinstance(row, Mapping)
            or not required_row <= set(row)
            or set(row) - allowed
        ):
            raise ValueError("Receipt v2 program has missing or unknown fields.")
        if ("target" in row) == ("strata" in row):
            raise ValueError(
                "Receipt v2 program declares exactly one target or strata."
            )
        item = dict(row)
        for field in ("program", "output", "payment_column"):
            item[field] = _text(row[field], field)
        item["judgment_columns"] = _strings(
            row["judgment_columns"], "judgment_columns", nonempty=True
        )
        excludes = row.get("exclude_columns", [])
        if not isinstance(excludes, list | tuple) or len(set(excludes)) != len(
            excludes
        ):
            raise ValueError("Receipt exclude_columns must be distinct columns.")
        item["exclude_columns"] = [
            _text(column, "exclude column") for column in excludes
        ]
        if row["judgment_reduction"] not in {"any", "all"}:
            raise ValueError("Receipt judgment_reduction must be any or all.")
        for field in RECEIPT_UNIT_FIELDS:
            item[field] = row.get(field, "person")
            if item[field] != "person":
                raise ValueError(
                    f"Receipt non-person {field} reserved for the declared-unit receipt extension."
                )
        if (
            item["output"] == f"{item['projection']}_id"
            or item["output"].startswith("person_")
            and item["output"].endswith("_id")
        ):
            raise ValueError("Receipt output cannot be a structural column.")
        if "strata" in row:
            if not isinstance(row["strata"], list | tuple) or not row["strata"]:
                raise ValueError("Receipt strata must be a non-empty list.")
            if row.get("outside_strata") != "not_assigned":
                raise ValueError(
                    "Receipt outside_strata must be declared as not_assigned."
                )
            for stratum in row["strata"]:
                if not isinstance(stratum, Mapping) or set(stratum) != {
                    "target",
                    "filter",
                }:
                    raise ValueError("Receipt stratum has missing or unknown fields.")
                _text(stratum["target"], "target")
                _text(stratum["filter"], "filter")
            item["strata"] = [dict(stratum) for stratum in row["strata"]]
        else:
            if "outside_strata" in row:
                raise ValueError("Receipt outside_strata requires strata.")
            item["target"] = _text(row["target"], "target")
        names = [
            item["program"],
            item["output"],
            *[s["target"] for s in receipt_strata(item)],
        ]
        for name in names:
            if name in identities:
                raise ValueError(
                    f"Receipt program, output and target identities must be unique: {name!r}."
                )
            identities.add(name)
        for profile in profiles.values():
            pair = (item["program"], profile["draw_key_suffix"])
            key = receipt_draw_key(*pair)
            if key in draws and draws[key] != pair:
                raise ValueError("Receipt draw key collision.")
            draws[key] = pair
        parsed.append(item)
    outputs = {row["output"] for row in parsed}
    groups = contract["exclusion_groups"]
    if not isinstance(groups, list | tuple):
        raise ValueError("Receipt exclusion_groups must be a list.")
    for group in groups:
        names = _strings(group, "exclusion group", nonempty=True)
        if len(names) < 2 or not set(names) <= outputs:
            raise ValueError(
                "Each receipt exclusion group names distinct declared outputs."
            )
    _ordered_receipt_outputs([row["output"] for row in parsed], groups)
    unmodelled = _strings(
        contract.get("unmodelled_programs", []), "unmodelled_programs"
    )
    if set(unmodelled) & identities:
        raise ValueError(
            "Receipt unmodelled_programs must not name produced outputs or programs."
        )
    return seed, parsed, groups


def receipt_strata(row):
    return row.get("strata", [{"target": row.get("target"), "filter": None}])


def _mass(weights, mask):
    return math.fsum(weights[mask].tolist())


def _ratio(target, mass):
    return target / mass if mass else (0 if target == 0 else None)


def _ordered_receipt_outputs(outputs, groups):
    pending, ordered = list(outputs), []
    while pending:
        available = [
            output
            for output in pending
            if all(
                not any(prior in pending for prior in group[: group.index(output)])
                for group in groups
                if output in group
            )
        ]
        if not available:
            raise ValueError("Receipt exclusion groups declare a cyclic order.")
        output = available[0]
        ordered.append(output)
        pending.remove(output)
    return ordered


def hall_diagnostics(demands, weights):
    """Report one maximally deficient demand set using exact rational max-flow.

    A demand is (programme, filter, target, eligibility mask). Split people into
    equal-neighbourhood weighted supplies. Join source to supplies, supplies to
    eligible demands, and demands to sink. Convert each float weight, target and
    relative tolerance r with Fraction, and scale demand capacities by (1 - r).
    For T > N, the relative Hall rule permits a subset iff N >= (1 - r) * T.
    Thus max-flow saturates every demand iff all subsets satisfy that rule.

    The demand nodes on the sink side of a minimum cut maximise (1 - r) * T - N.
    When infeasible, report this one set with its unscaled target and supply.
    Edmonds-Karp uses O(V * E**2) rational operations, never demand subsets.
    A flow divided by people's weights gives mutually exclusive receipt
    probabilities, and multiplying those probabilities gives a flow, proving
    equivalence for expected counts. This is diagnostic only; draws and priority
    may still refuse even when the fractional allocation is feasible.
    """
    grouped_weights = {}
    if demands:
        packed = np.packbits(
            np.column_stack([item[3] for item in demands]), axis=1, bitorder="little"
        )
        for packed_pattern, weight in zip(packed, weights, strict=True):
            pattern = int.from_bytes(packed_pattern.tobytes(), "little")
            if pattern:
                grouped_weights[pattern] = grouped_weights.get(
                    pattern, Fraction()
                ) + Fraction(float(weight))
    supplies = sorted(grouped_weights.items())
    scale = 1 - Fraction(RECEIPT_RELATIVE_TOLERANCE)
    targets = [Fraction(item[2]) for item in demands]
    required = sum(targets, Fraction()) * scale
    source, supply_start = 0, 1
    demand_start = supply_start + len(supplies)
    sink = demand_start + len(demands)
    graph = [[] for _ in range(sink + 1)]

    def edge(left, right, capacity):
        graph[left].append([right, capacity, len(graph[right])])
        graph[right].append([left, Fraction(), len(graph[left]) - 1])

    for index, (pattern, mass) in enumerate(supplies):
        supply = supply_start + index
        edge(source, supply, mass)
        for demand in range(len(demands)):
            if pattern & (1 << demand):
                edge(supply, demand_start + demand, required)
    for index, target in enumerate(targets):
        edge(demand_start + index, sink, scale * target)
    flow = Fraction()
    while True:
        parents = {source: None}
        queue = deque([source])
        while queue and sink not in parents:
            left = queue.popleft()
            for index, (right, capacity, _) in enumerate(graph[left]):
                if capacity > 0 and right not in parents:
                    parents[right] = (left, index)
                    queue.append(right)
        if sink not in parents:
            break
        capacity, right = required, sink
        while right != source:
            left, index = parents[right]
            capacity = min(capacity, graph[left][index][1])
            right = left
        right = sink
        while right != source:
            left, index = parents[right]
            arc = graph[left][index]
            arc[1] -= capacity
            graph[right][arc[2]][1] += capacity
            right = left
        flow += capacity
    violations = []
    if flow < required:
        indices = [
            index
            for index in range(len(demands))
            if demand_start + index not in parents
        ]
        subset = [demands[index] for index in indices]
        subset_pattern = sum(1 << index for index in indices)
        filters = {item[1] for item in subset}
        violations.append(
            {
                "programs": [item[0] for item in subset],
                "filter": next(iter(filters)) if len(filters) == 1 else None,
                "filters": [item[1] for item in subset],
                "target": float(sum((targets[index] for index in indices), Fraction())),
                "eligible_mass": float(
                    sum(
                        (
                            mass
                            for pattern, mass in supplies
                            if pattern & subset_pattern
                        ),
                        Fraction(),
                    )
                ),
            }
        )
    return {
        "violations": violations,
        "feasible": not violations,
        "relative_tolerance": RECEIPT_RELATIVE_TOLERANCE,
    }


def _refuse_assignment(message, audit, out):
    audit["status"] = "refused"
    for group_audit in audit["exclusion_groups"]:
        group_audit["status"] = (
            "completed"
            if all(name in out for name in group_audit["order"])
            else "refused"
        )
    raise ReceiptAssignmentRefused(message, audit)


def _unit_views_align(tables, row, person):
    """Check what the [BE] dispatch seam returns, not the raw unit tables.

    Each unit's ids are matched through the person table's membership column, so
    a declared unit broadcast to person rows at the seam aligns.
    """
    if any(row[field] not in tables for field in RECEIPT_UNIT_FIELDS):
        return False
    for field, view in receipt_dispatch_tables(tables, row).items():
        entity = row[field]
        membership = "person_id" if entity == "person" else f"person_{entity}_id"
        if (
            len(view) != len(person)
            or f"{entity}_id" not in view
            or membership not in person
            or not np.array_equal(
                view[f"{entity}_id"].to_numpy(), person[membership].to_numpy()
            )
        ):
            return False
    return True


def assign_receipts_v2(
    person,
    contract,
    *,
    weights=None,
    targets=None,
    bindings=None,
    knobs=None,
    tables=None,
):
    seed, rows, declared_groups = validate_receipt_v2(contract)
    if not isinstance(knobs, Mapping) or contract["assignment_knob"] not in knobs:
        raise ValueError(
            "Receipt assignment_knob must be present in the resolved knobs."
        )
    profile_name = knobs[contract["assignment_knob"]]
    if profile_name not in contract["assignment_profiles"]:
        raise ValueError(f"Receipt unknown assignment profile {profile_name!r}.")
    profile = contract["assignment_profiles"][profile_name]
    order = profile.get("exclusion_order", "declared")
    groups = reversed_receipt_groups(declared_groups, order)
    if weights is None or targets is None:
        raise ValueError("Receipt v2 targets need a surface and weights.")
    weights = np.asarray(weights, dtype=np.float64)
    if (
        weights.shape != (len(person),)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
    ):
        raise ValueError("Receipt weights must be aligned, finite and non-negative.")
    try:
        total_weight = math.fsum(weights.tolist())
    except OverflowError as error:
        raise ValueError("Receipt weights must have a finite sum.") from error
    if not math.isfinite(total_weight):
        raise ValueError("Receipt weights must have a finite sum.")
    tables = (
        dict(tables)
        if tables is not None
        else {row[field]: person for row in rows for field in RECEIPT_UNIT_FIELDS}
    )
    if not all(_unit_views_align(tables, row, person) for row in rows):
        raise ValueError("Receipt entity tables must align with person length and ids.")
    eligible, masks, target_values = {}, {}, {}
    before_payment = {}
    for row in rows:
        dispatch = receipt_dispatch_tables(tables, row)
        judgments = [
            resolve_judgments(
                dispatch["entitlement_entity"][name].to_numpy(),
                UndeterminedAction.REFUSE,
            )
            for name in row["judgment_columns"]
        ]
        reduction = (
            np.logical_or if row["judgment_reduction"] == "any" else np.logical_and
        )
        base = reduction.reduce(judgments)
        payment = dispatch["payee"][row["payment_column"]].to_numpy(dtype=np.float64)
        if not np.isfinite(payment).all():
            raise ValueError("Receipt payment values must be finite.")
        base &= payment > 0
        before_payment[row["output"]] = base.copy()
        for column in row["exclude_columns"]:
            values = dispatch["entitlement_entity"][column]
            if str(values.dtype) not in {"bool", "boolean"} or values.isna().any():
                raise ValueError(
                    "Existing receipt exclusions must be non-null booleans."
                )
            base &= ~values.to_numpy(dtype=bool)
        eligible[row["output"]] = base
        previous = []
        for stratum in receipt_strata(row):
            name, filter_name = stratum["target"], stratum["filter"]
            if name not in targets:
                raise ValueError(
                    f"Receipt target {name!r} needs a surface and weights."
                )
            value = targets[name]
            if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
                raise ValueError("Receipt target must be finite and non-negative.")
            try:
                value = float(value)
            except (OverflowError, ValueError) as error:
                raise ValueError(
                    "Receipt target must be finite and non-negative."
                ) from error
            if not math.isfinite(value) or value < 0:
                raise ValueError("Receipt target must be finite and non-negative.")
            target_values[name] = value
            if filter_name is None:
                mask = np.ones(len(person), dtype=bool)
            else:
                if bindings is None:
                    raise ValueError("Receipt strata need calibration_bindings.")
                mask = calibration_predicate_mask(
                    tables, bindings, entity=row["projection"], name=filter_name
                )
            for previous_name, previous_mask in previous:
                if np.any(mask & previous_mask):
                    raise ValueError(
                        f"Receipt overlapping strata {previous_name!r} and {filter_name!r} for {row['program']!r}."
                    )
            previous.append((filter_name, mask))
            masks[name] = mask
    audit = {
        "assignment_profile": profile_name,
        "exclusion_order": order,
        "status": "completed",
        "programs": {},
        "exclusion_groups": [],
    }
    for group in groups:
        group_rows = [
            row for output in group for row in rows if row["output"] == output
        ]
        overlaps = []
        filters = {s["filter"] for row in group_rows for s in receipt_strata(row)} - {
            None
        }
        for left, right in itertools.permutations(group_rows, 2):
            both = eligible[left["output"]] & eligible[right["output"]]
            for filter_name in [None, *sorted(filters)]:
                mask = both.copy()
                if filter_name is not None:
                    mask &= calibration_predicate_mask(
                        tables,
                        bindings,
                        entity=left["projection"],
                        name=filter_name,
                    )
                overlaps.append(
                    {
                        "programs": [left["output"], right["output"]],
                        "filter": filter_name,
                        "n_eligible": int(mask.sum()),
                        "eligible_mass": _mass(weights, mask),
                    }
                )
        demands = [
            (
                row["output"],
                s["filter"],
                target_values[s["target"]],
                eligible[row["output"]] & masks[s["target"]],
            )
            for row in group_rows
            for s in receipt_strata(row)
        ]
        # One flow includes all strata and mixed-filter demand cuts.
        feasibility = hall_diagnostics(demands, weights)
        audit["exclusion_groups"].append(
            {
                "order": group,
                "status": "completed",
                "overlaps": overlaps,
                "feasibility": feasibility,
            }
        )
    by_output = {row["output"]: row for row in rows}
    ordered = [
        by_output[output] for output in _ordered_receipt_outputs(by_output, groups)
    ]
    id_column = f"{rows[0]['projection']}_id"
    out = person.loc[:, [id_column]].copy()
    for row in ordered:
        output = row["output"]
        remaining = eligible[output].copy()
        for group in groups:
            if output in group:
                for prior in group[: group.index(output)]:
                    remaining &= ~out[prior].to_numpy(dtype=bool)
        draws = derive_take_up_draws(
            receipt_dispatch_tables(tables, row)["draw_unit"][seed].to_numpy(),
            receipt_draw_key(row["program"], profile["draw_key_suffix"]),
        )
        assigned, covered = (
            np.zeros(len(person), dtype=bool),
            np.logical_or.reduce(
                [masks[stratum["target"]] for stratum in receipt_strata(row)]
            ),
        )
        stratum_audits = []
        total_target = math.fsum(
            target_values[stratum["target"]] for stratum in receipt_strata(row)
        )
        total_before = _mass(weights, before_payment[output] & covered)
        total_remaining = _mass(weights, remaining & covered)
        audit["programs"][output] = {
            "program": row["program"],
            "target": total_target,
            "eligible_mass_before_exclusions": total_before,
            "eligible_mass": total_remaining,
            "target_to_eligible_mass_before_exclusions": _ratio(
                total_target, total_before
            ),
            "target_to_remaining_eligible_mass": _ratio(total_target, total_remaining),
            "strata": stratum_audits,
            "outside_strata": {
                "n_eligible": int((remaining & ~covered).sum()),
                "eligible_mass": _mass(weights, remaining & ~covered),
            },
        }
        for stratum in receipt_strata(row):
            name, filter_name = stratum["target"], stratum["filter"]
            mask, target = masks[name], target_values[name]
            pool = remaining & mask
            mass = _mass(weights, pool)
            before = _mass(weights, before_payment[output] & mask)
            rate = _ratio(target, mass)
            detail = {
                "target_name": name,
                "target": target,
                "filter": filter_name,
                "eligible_mass": mass,
                "eligible_mass_before_exclusions": before,
                "target_to_eligible_mass_before_exclusions": _ratio(target, before),
                "target_to_remaining_eligible_mass": rate,
                "rate": rate,
            }
            stratum_audits.append(detail)
            if target > mass:
                _refuse_assignment(
                    f"Receipt target {name!r} exceeds eligible weighted mass in {order} order.",
                    audit,
                    out,
                )
            chosen = pool & (draws < rate)
            assigned |= chosen
            expected = math.fsum([rate * mass])
            detail.update(
                {
                    "realized_mass": _mass(weights, chosen),
                    "n_receipts": int(chosen.sum()),
                    "numerical_check": {
                        "passed": math.isclose(
                            expected,
                            target,
                            rel_tol=RECEIPT_RELATIVE_TOLERANCE,
                            abs_tol=0,
                        ),
                        "relative_tolerance": RECEIPT_RELATIVE_TOLERANCE,
                        "expected_mass": expected,
                    },
                }
            )
            if not detail["numerical_check"]["passed"]:
                _refuse_assignment(
                    f"Receipt target {name!r} failed the numerical expected-mass check in {order} order.",
                    audit,
                    out,
                )
        audit["programs"][output]["realized_mass"] = _mass(weights, assigned)
        audit["programs"][output]["n_receipts"] = int(assigned.sum())
        out[output] = assigned
    return out.loc[:, [id_column, *by_output]], audit
