"""Whole original-arm PUF55 finalization over chain-consistent units.

This is the explicit successor to the conservative development placement in
:mod:`.puf55_original_placement`. It shares that module's precondition checks
and ownership rules, and adds three things the conservative cut leaves open:

**Mixed-known-member units.** The observation-conditioned chain uses a fixed
input as a target's conditioning value only when *every* member of the unit is
source-qualified for it; otherwise the raw draw conditions later targets. A
target's draw is therefore consistent with the retained population exactly
when every fixed target *earlier in the chain* is known for every member. The
conservative cut required all twelve fixed targets for all eight outputs; this
policy applies the exact per-target prefix. Fixed targets are never rewritten:
known members keep their values and unknown members of a mixed unit stay
unresolved. No residual, constrained total or person-grain completion is
invented for a mixed unit.

**Whole-arm attachment.** Every PUF55 output introduced by the checked arm-one
attachment (absent from the pre-PUF parent) is a candidate, not only the three
unit and five singleton outputs. Person outputs are allocated inside each
eligible unit with the maintained arm-one helpers
(:func:`~.puf_support._write_person_tax_unit_totals` and
:func:`~.puf_support._write_person_tax_unit_boolean_counts`) and the maintained
distribution bases, restricted to arm-zero members. Two arm-zero refusals are
stricter than arm one: a multi-member unit whose declared basis columns are all
absent from the population (tuition without a student flag) or whose basis is
unresolved for an allocation member stays unresolved instead of falling back
to the first member. Earnings-universe outputs keep the maintained age-15
universe: under-15 members of an eligible unit receive the receipted universe
zero, and a nonzero draw with no eligible member stays unresolved.

**Finalization and pruning policy.** Arm-one finalization clips, snaps to donor
values, caps the configured tail, prunes sparse outputs to the donor's weighted
positive rate and aligns signed mass. None of those population-level donor
alignments is applied here: the recipients are survey units with modeled
non-filer roles, so the PUF filer donor's rates are not a qualified target for
this arm. A negative draw for a nonnegative output stays unresolved (no
clipping). Boolean QBI counts keep the maintained representation: rounded and
capped at the allocation members. Descriptive arm-zero rates and totals are
recorded for later comparison only; they are not a calibration, selection or
release criterion.

**Own-tail copies.** Arm-zero rows are selected by clone index 0. A
capital-gains/AGI own-tail copy (clone index 2) keeps its clone-one twin's
source IDs and PUF channel, so a ``(source ID, role)`` key cannot tell the two
apart; this arm keys copies by clone index only. A whole-household tail copy in
the receiving population is split off before the shared checks, which then run
on the exact two-clone core, and every tail cell is carried unchanged. Tail
placement belongs to the tail owner, not to this arm.

The result is detached columns and a descriptive document. It issues no
source, owner, model or release authority, and every host duty in
:mod:`.puf55_original_placement` still applies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import puf55_original_placement as placement

values, codec, application = placement.values, placement.codec, placement.application
recipients = values.recipients
support = recipients.support
provenance = recipients.provenance

POLICY = "microcosm.us.puf55-original-finalization.v1"
POLICIES = (POLICY,)
FIXED_TARGETS = (*values.FINANCIAL_TARGETS, *values.DEVELOPMENT_TARGETS)
BOOLEAN_OUTPUTS = support._PUF_TAX_DETAIL_BOOLEAN_PERSON_OUTPUTS
NONNEGATIVE_OUTPUTS = support._PUF_TAX_DETAIL_NONNEGATIVE_OUTPUTS
EARNINGS_UNIVERSE_OUTPUTS = support._PUF_EARNINGS_UNIVERSE_PERSON_OUTPUTS
DISTRIBUTION_BASIS = support._PERSON_OUTPUT_DISTRIBUTION_BASIS
MODELED = "modeled"
REASONS = (
    MODELED,
    "fixed_chain_unresolved",
    "draw_domain_unresolved",
    "allocation_universe_empty",
    "allocation_basis_absent",
    "allocation_basis_unresolved",
)
# Immutable so a graph node can carry it as an exact parameter.
NUMERICAL_POLICY = (
    ("boolean_counts", "maintained_round_and_cap_at_allocation_members"),
    ("declared_basis_absent", "multi_member_unit_unresolved"),
    ("donor_positive_rate_pruning", "not_applied"),
    ("donor_value_snapping", "not_applied"),
    ("earnings_universe", "maintained_age_15_universe_zero"),
    ("first_member_fallback", "maintained_when_basis_present_but_zero_or_undeclared"),
    ("mixed_known_members", "per_target_fixed_chain_prefix_else_unresolved"),
    ("nonnegative_domain", "unresolved_not_clipped"),
    ("person_allocation", "maintained_arm_one_distribution_basis"),
    ("signed_mass_alignment", "not_applied"),
    ("tail_bound_caps", "not_applied"),
    ("unresolved_basis", "multi_member_unit_unresolved"),
)
TAIL_CLONE_INDEX = placement.TAIL_CLONE_INDEX


def require(condition, reason):
    if not condition:
        raise ValueError("PUF55_ORIGINAL_FINALIZATION_" + reason)


def require_policy(policy):
    """Accept only an exact named policy; never a truthy switch or alias."""
    require(type(policy) is str and policy in POLICIES, "POLICY")
    return policy


def chain_prefixes(profile):
    """Fixed targets strictly before each target in the profile's chain order."""
    placement.output_policy(profile)
    positions = {target: i for i, target in enumerate(profile.targets)}
    require(
        len(positions) == len(profile.targets)
        and all(target in positions for target in FIXED_TARGETS),
        "FIXED_ROSTER",
    )
    return {
        target: tuple(f for f in FIXED_TARGETS if positions[f] < positions[target])
        for target in profile.targets
    }


def chain_consistency(qualified, selected, profile):
    """Unit x target: every earlier fixed conditioning value is the unit total.

    ``tax_unit_known`` is the qualifier's whole-member rule, so a True cell
    means the conditioning value was the preserved total of all members and a
    False cell means the raw draw conditioned every later target instead.
    """
    known = qualified.tax_unit_known.loc[selected]
    require(tuple(known.columns) == FIXED_TARGETS, "FIXED_ROSTER")
    result = {}
    for target, prefix in chain_prefixes(profile).items():
        result[target] = (
            known[list(prefix)].all(axis=1).to_numpy(dtype=bool)
            if prefix
            else np.ones(len(selected), dtype=bool)
        )
    return pd.DataFrame(result, index=selected)


def fixed_knownness(known, current, selected):
    """Count all-known, mixed and all-unknown units for each fixed target."""
    result = {}
    unit = current.person_tax_unit_id
    for target in FIXED_TARGETS:
        grouped = known[target].groupby(unit, sort=False)
        every = grouped.all().reindex(selected).to_numpy(dtype=bool)
        some = grouped.any().reindex(selected).to_numpy(dtype=bool)
        result[target] = {
            "all_known": int(every.sum()),
            "mixed": int((some & ~every).sum()),
            "all_unknown": int((~some).sum()),
        }
    return result


def candidate_outputs(inputs, profile):
    """Every non-fixed PUF55 output introduced by the checked arm-one attach."""
    placement.output_policy(profile)
    names = {
        "person": tuple(t for t in profile.person_outputs if t not in FIXED_TARGETS),
        "tax_unit": tuple(profile.tax_unit_outputs),
    }
    result = placement._owned_candidates(inputs, names)
    require(not any(name in FIXED_TARGETS for _, name, _ in result), "FIXED_CANDIDATE")
    return result


def read_columns(inputs, profile):
    """Person columns read for allocation, besides fixed targets and candidates.

    Presence is taken from the checked arm-one population, whose columns every
    later receiving version carries, so a graph declaration made before the
    receiving terminal exists names exactly the columns the result reads. A
    later version that introduces a declared basis column cannot silently add
    it to an allocation that was declared without it.
    """
    candidates = candidate_outputs(inputs, profile)
    names = {name for _, name, _ in candidates}
    person = {name for entity, name, _ in candidates if entity == "person"}
    columns = inputs.arm_one.frame.person.columns
    result = []
    if person & EARNINGS_UNIVERSE_OUTPUTS:
        require("age" in columns, "AGE_COLUMN")
        result.append("age")
    for name in sorted(person):
        result.extend(
            b
            for b in DISTRIBUTION_BASIS.get(name, ())
            if b in columns and b not in names and b not in FIXED_TARGETS
        )
    return tuple(dict.fromkeys(result))


def _valid_draws(draws, name):
    valid = np.isfinite(draws)
    if name in NONNEGATIVE_OUTPUTS or name in BOOLEAN_OUTPUTS:
        valid &= np.where(valid, draws, 0.0) >= 0.0
    return valid


def _basis_score(work, mask, name, columns):
    """The maintained fallback score, summed per unit; for diagnostics only."""
    score = np.zeros(int(mask.sum()), dtype=np.float64)
    for column in columns:
        if column in work.columns:
            score += support._nonnegative_allocation_basis_values(
                work.loc[mask, column], output_column=name, basis_column=column
            )
    return (
        pd.Series(score, index=work.index[mask.to_numpy()])
        .groupby(work.loc[mask, "person_tax_unit_id"], sort=False)
        .sum()
    )


def finalization_result(
    qualified, inputs, conditioning, merge_receipt, *, profile, policy
):
    """Compute exact full-axis columns and a descriptive document; no issuance.

    The caller supplies the complete strictly merged arm-zero conditioning
    table. Actual source and 55-model producer authentication, and the
    receiving owner's before/after checks, remain the host's obligation.
    Arm-zero rows are selected by clone index 0. An own-tail copy (clone index
    2) in the receiving population is split off by clone index, never by a
    (source ID, role) key, and every one of its cells is carried unchanged.
    """
    require_policy(policy)
    full_inputs, full_stamp = inputs, placement._stamp(inputs)
    inputs, tail = placement._receiving_core(full_inputs)
    basis = placement._checked_basis(
        qualified, inputs, conditioning, merge_receipt, profile=profile
    )
    selected, current = basis.selected, basis.current
    units = basis.units
    arm = provenance.support_clone_index_column("tax_unit")
    member_arm = provenance.support_clone_index_column("person")
    require(
        units.loc[selected, arm].eq(0).all() and current[member_arm].eq(0).all(),
        "ARM_ZERO_AXIS",
    )
    consistency = chain_consistency(qualified, selected, profile)
    candidates = candidate_outputs(inputs, profile)
    candidate_names = {name for _, name, _ in candidates}
    positions = {target: i for i, target in enumerate(profile.targets)}
    prefixes = chain_prefixes(profile)

    # Ownership integrity: arm-zero cells of every candidate are still the
    # arm-one nulls and no later writer changed the column.
    targets = {}
    for entity, name, _ in candidates:
        id_column = entity + "_id"
        prior = inputs.arm_one.frame.table(entity).set_index(id_column)[name]
        target = inputs.receiving.frame.table(entity).set_index(id_column)[name]
        axis = current.index if entity == "person" else selected
        require(
            prior.loc[axis].isna().all() and target.loc[axis].isna().all(),
            "PRESERVE_NULL_OWNERSHIP_CHANGED",
        )
        require(target.equals(prior), "LATER_OUTPUT_WRITER")
        targets[entity, name] = target

    person_candidates = {n for e, n, _ in candidates if e == "person"}
    reads = read_columns(inputs, profile)
    arm_one_columns = inputs.arm_one.frame.person.columns
    needed = ["person_tax_unit_id", *reads]
    for name in sorted(person_candidates):
        needed.append(name)
        # Fixed and earlier-candidate bases; presence comes from arm one.
        needed.extend(
            b for b in DISTRIBUTION_BASIS.get(name, ()) if b in arm_one_columns
        )
    needed = list(dict.fromkeys(needed))
    require(all(column in current for column in needed), "READ_COLUMN")
    # A detached arm-zero working table in receiving row order. Float outputs
    # work in float64 and are converted back losslessly below.
    work = current.loc[:, needed].copy(deep=True)
    for name in person_candidates:
        if name not in BOOLEAN_OUTPUTS:
            work[name] = work[name].astype("float64")
    unit_of = work.person_tax_unit_id

    statuses = pd.DataFrame(index=selected)
    statuses["members"] = basis.counts.to_numpy()
    written, allocation_evidence, unit_totals = {}, {}, {}
    for name in profile.targets:
        if name not in candidate_names:
            continue
        draws = conditioning[name].to_numpy(dtype=np.float64)
        valid = _valid_draws(draws, name)
        consistent = consistency[name].to_numpy()
        reason = np.where(
            ~consistent,
            "fixed_chain_unresolved",
            np.where(~valid, "draw_domain_unresolved", MODELED),
        ).astype(object)
        eligible = consistent & valid
        if name not in person_candidates:
            ids = selected[eligible]
            written[name] = ids
            unit_totals[name] = pd.Series(draws[eligible], index=ids)
            statuses[name] = pd.array(reason.tolist(), dtype="string")
            continue
        in_eligible = unit_of.isin(selected[eligible])
        if name in EARNINGS_UNIVERSE_OUTPUTS:
            allocation = support._puf_earnings_allocation_mask(
                work, person_puf_mask=in_eligible
            )
            universe_zero = in_eligible & ~allocation
        else:
            allocation = in_eligible
            universe_zero = pd.Series(False, index=work.index)
        members = (
            allocation.groupby(unit_of, sort=False)
            .sum()
            .reindex(selected, fill_value=0)
            .to_numpy(dtype=np.int64)
        )
        boolean = name in BOOLEAN_OUTPUTS
        nonzero = (np.rint(draws) if boolean else draws) != 0.0
        reason[eligible & (members == 0) & nonzero] = "allocation_universe_empty"
        declared = DISTRIBUTION_BASIS.get(name, ())
        present = tuple(b for b in declared if b in work.columns)
        multi = members > 1
        if declared and not present:
            reason[(reason == MODELED) & multi] = "allocation_basis_absent"
        # A later candidate basis is not yet placed in the maintained pass
        # order and contributes zero, exactly as on arm one. Every other basis
        # must be resolved for each allocation member of a multi-member unit.
        checked = tuple(
            b
            for b in present
            if not (b in candidate_names and positions[b] > positions[name])
        )
        unresolved_units = set()
        for column in checked:
            unresolved_units.update(unit_of[allocation & work[column].isna()].tolist())
        unresolved = np.isin(selected, list(unresolved_units))
        reason[(reason == MODELED) & multi & unresolved] = "allocation_basis_unresolved"
        resolved = reason == MODELED
        in_resolved = unit_of.isin(selected[resolved])
        mask = allocation & in_resolved
        zero_mask = universe_zero & in_resolved
        totals = pd.Series(draws[resolved], index=selected[resolved])
        evidence = {
            "allocation_members": int(mask.sum()),
            "universe_zero_persons": int(zero_mask.sum()),
        }
        if mask.any():
            score = _basis_score(work, mask, name, declared)
            multi_resolved = pd.Series(members, index=selected)[resolved].gt(1)
            multi_ids = multi_resolved.index[multi_resolved.to_numpy()]
            zero_score = score.reindex(multi_ids, fill_value=0.0).eq(0.0)
            key = (
                "rank_ties_by_position_units"
                if boolean
                else ("first_member_fallback_units")
            )
            evidence[key] = int(zero_score.sum())
        if zero_mask.any():
            work.loc[zero_mask, name] = 0.0
        if boolean:
            if mask.any():
                support._write_person_tax_unit_boolean_counts(
                    work,
                    mask=mask,
                    column=name,
                    totals=totals,
                    preserve_boolean_dtype=True,
                    fallback_basis_columns=declared,
                )
            capacity = pd.Series(members, index=selected)[resolved].to_numpy()
            evidence["count_adjusted_units"] = int(
                (
                    (np.rint(totals.to_numpy()) != totals.to_numpy())
                    | (np.rint(totals.to_numpy()) > capacity)
                ).sum()
            )
        elif mask.any():
            support._write_person_tax_unit_totals(
                work,
                mask=mask,
                column=name,
                totals=totals,
                nonnegative=name in NONNEGATIVE_OUTPUTS,
                allow_legacy_numeric_boolean_basis=False,
                fallback_basis_columns=declared,
            )
        persons = work.index[(mask | zero_mask).to_numpy()]
        require(bool(work.loc[persons, name].notna().all()), "ALLOCATION_NULL")
        written[name] = persons
        per_unit = work.loc[persons, name].astype("float64")
        unit_totals[name] = (
            per_unit.groupby(unit_of.loc[persons], sort=False)
            .sum()
            .reindex(selected[resolved], fill_value=0.0)
        )
        allocation_evidence[name] = evidence
        statuses[name] = pd.array(reason.tolist(), dtype="string")

    columns, result_seal = {}, []
    for entity, name, dtype in candidates:
        output = targets[entity, name].copy(deep=True)
        ids = written[name]
        if entity == "person":
            assigned = work.loc[ids, name]
        else:
            assigned = unit_totals[name].loc[ids]
        if name in BOOLEAN_OUTPUTS:
            require(bool(assigned.notna().all()), "BOOLEAN_WRITE")
            output.loc[ids] = assigned.to_numpy(dtype=bool)
        else:
            raw = assigned.to_numpy(dtype=np.float64)
            converted = raw.astype(dtype)
            require(
                np.isfinite(raw).all()
                and np.array_equal(
                    converted.astype("float64").view("u8"), raw.view("u8")
                ),
                "LOSSY_WRITE",
            )
            output.loc[ids] = converted
        require(output.dtype == targets[entity, name].dtype, "OUTPUT_DTYPE")
        columns[entity, name] = output
    columns = placement._carry_tail(full_inputs, columns, tail)
    for key, output in columns.items():
        result_seal.append((key, recipients._table_digest(output.to_frame())))

    weights = pd.Series(
        support._tax_unit_household_weights(inputs.receiving.frame, selected),
        index=selected,
    )
    diagnostics = {}
    for name, totals in unit_totals.items():
        w = weights.loc[totals.index]
        total_weight = float(w.sum())
        diagnostics[name] = {
            "modeled_units": int(len(totals)),
            "weighted_units": total_weight,
            "weighted_positive_share": (
                float(w[totals.gt(0.0).to_numpy()].sum()) / total_weight
                if total_weight > 0.0
                else None
            ),
            "weighted_total": float((w * totals).sum()),
        }
    reason_counts = {
        name: {
            reason: int(statuses[name].eq(reason).sum())
            for reason in REASONS
            if int(statuses[name].eq(reason).sum())
        }
        for name in statuses.columns
        if name != "members"
    }
    document = codec.encode_json(
        {
            "protocol": POLICY,
            "profile": profile.value,
            "numerical_policy": dict(NUMERICAL_POLICY),
            "qualification_sha256": codec.sha(qualified.receipt),
            "merge_sha256": codec.sha(merge_receipt),
            "input_population_versions": [
                inputs.financial_parent.version,
                inputs.arm_one.version,
                inputs.receiving.version,
            ],
            "candidate_outputs": [list(c) for c in candidates],
            "fixed_targets": list(FIXED_TARGETS),
            "fixed_chain_prefix": {
                name: list(prefixes[name]) for _, name, _ in candidates
            },
            "fixed_knownness": fixed_knownness(basis.known, current, selected),
            "recipient_units": int(len(selected)),
            "reason_counts": reason_counts,
            "unit_status_sha256": codec.sha(statuses.to_json(orient="table").encode()),
            "write_counts": {name: int(len(ids)) for name, ids in written.items()},
            "allocation": allocation_evidence,
            "descriptive_diagnostics": diagnostics,
            "diagnostics_use": "comparison_only_not_calibration_selection_or_gate",
            "output_hashes": [[*key, digest] for key, digest in result_seal],
            "missingness": "unresolved_cells_keep_arm_one_nulls",
            "clone_axis": "arm_zero_by_clone_index_0",
            "own_tail_copies_carried": tail or {},
            "source_observation_claim": False,
            "source_admission_issued": False,
            "population_admission_issued": False,
            "release_eligible": False,
        }
    )
    require(
        placement._stamp(full_inputs) == full_stamp
        and placement._stamp(inputs) == basis.source_stamp
        and values.fixed_input_stamp(qualified) == basis.fixed_stamp
        and recipients._table_digest(conditioning) == basis.table_stamp
        and [
            ((e, n), recipients._table_digest(s.to_frame()))
            for (e, n), s in columns.items()
        ]
        == result_seal,
        "FINAL_INPUT_CHANGED",
    )
    return columns, document
