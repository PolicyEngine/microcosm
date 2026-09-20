"""Conservative original-arm development placement; detached values, not authority.

PUF55 has 52 person and 3 tax-unit outputs. This cut preserves 12 qualified fixed
leaves and defers 35 other person outputs; it places only 5 new numeric person
leaves on complete singletons and the 3 unit leaves on eligible original units.
The host authenticates both retained ancestors and all 55 actual apply artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from microcosm.frame import Frame
from microcosm.graph import population as population_ops
from microcosm.graph.serialize import _node_payload

from . import puf55_original_application as application

values, codec = application.values, application.codec
attachment = application.attachment
PROTOCOL = "microcosm.us.puf55-original-placement.v1"
SINGLETON_OUTPUTS = (
    "long_term_capital_gains_on_collectibles",
    "non_sch_d_capital_gains",
    "partnership_income",
    "s_corp_income",
    "partnership_self_employment_net_earnings",
)
UNIT_OUTPUTS = (
    "domestic_production_ald",
    "unrecaptured_section_1250_gain",
    "health_savings_account_ald",
)


def require(condition, reason):
    if not condition:
        raise ValueError("PUF55_ORIGINAL_PLACEMENT_" + reason)


def output_policy(profile):
    require(profile in attachment.PROFILES, "PROFILE")
    require(
        len(profile.person_outputs) == 52
        and tuple(profile.tax_unit_outputs) == UNIT_OUTPUTS
        and len(profile.targets) == 55
        and set(SINGLETON_OUTPUTS) <= set(profile.person_outputs),
        "PROFILE_ROSTER",
    )
    fixed = (*values.FINANCIAL_TARGETS, *values.DEVELOPMENT_TARGETS)
    return {
        "person": tuple(x for x in profile.person_outputs if x in SINGLETON_OUTPUTS),
        "tax_unit": tuple(profile.tax_unit_outputs),
        "preserved_fixed_person": fixed,
        "deferred_person": tuple(
            x for x in profile.person_outputs if x not in (*fixed, *SINGLETON_OUTPUTS)
        ),
        "transport": "puf55_original_development_transport",
        "partnership_net_earnings": "maintained_raw_partnership_nonpassive_net_proxy",
        "source_observation_claim": False,
        "legal_eligibility_claim": False,
        "release_science_accepted": False,
    }


@dataclass(frozen=True)
class PlacementInputs:
    """Detached descriptions; constructing this object establishes no authority."""

    financial_parent: population_ops.Population
    arm_one: population_ops.Population
    receiving: population_ops.Population
    arm_one_node: object


def _stamp(inputs):
    require(type(inputs) is PlacementInputs, "INPUT_TYPE")
    return (
        *(
            attachment.physical._population_stamp(p)
            for p in (inputs.financial_parent, inputs.arm_one, inputs.receiving)
        ),
        codec.encode_json(_node_payload(inputs.arm_one_node)),
    )


def _capture_checked_puf_ancestors(puf_run, receiving):
    """Private host building block: check only the supplied PUF ancestry.

    This is not a complete receiving-owner capture boundary. The integrating
    enrichment host must admit/recheck its actual receiving owner and Population
    before and after I/O. This returns no retained lease or issuer.
    """
    from . import graph_survey_puf55 as parent

    entry = parent._run_entry(puf_run)
    parent.check_survey_puf55_run(puf_run)
    financial = attachment.financial._run_entry(puf_run.financial_run)
    result = PlacementInputs(
        financial[2].financial_population,
        puf_run.population,
        receiving,
        puf_run.compiled.graph.node(attachment.ATTACH_NODE),
    )
    stamp = _stamp(result)
    parent.check_survey_puf55_run(puf_run)
    require(
        parent._run_entry(puf_run) is entry and _stamp(result) == stamp,
        "CAPTURE_CHANGED",
    )
    return result


def _axes(inputs, qualified):
    frames = tuple(
        p.frame for p in (inputs.financial_parent, inputs.arm_one, inputs.receiving)
    )
    require(
        all(type(f) is Frame and f.schema == frames[0].schema for f in frames), "SCHEMA"
    )
    for entity in frames[0].entities:
        column = frames[0].schema.entity_id_column(entity)
        ids = frames[0].table(entity)[column]
        require(
            ids.dtype == np.dtype("int64") and ids.is_unique and len(ids), "ID_AXIS"
        )
        require(
            all(
                np.array_equal(f.table(entity)[column].to_numpy(), ids.to_numpy())
                for f in frames[1:]
            ),
            "ID_AXIS",
        )
    for group in frames[0].schema.group_entities:
        column = frames[0].schema.membership_column(group)
        require(
            all(frames[0].person[column].equals(f.person[column]) for f in frames[1:]),
            "MEMBERSHIP",
        )
    people = frames[-1].person.set_index("person_id", drop=False)
    units = frames[-1].table("tax_unit").set_index("tax_unit_id", drop=False)
    require(
        qualified.person_values.index.equals(people.index)
        and qualified.tax_unit_values.index.equals(units.index),
        "QUALIFIED_AXIS",
    )
    require(set(people.person_tax_unit_id) == set(units.index), "COMPLETE_UNIT_AXIS")
    return frames, people, units


def _selected_ids(qualified):
    matrices = tuple(
        application.fixed_graph.parent.model_input.decode_recipient_matrix(p)
        for _, p in qualified.recipients.matrices
    )
    ids = [int(i) for m in matrices for i in m.features.index]
    require(len(ids) == len(set(ids)) and ids, "SELECTED_AXIS")
    ordered = qualified.tax_unit_values.index
    selected = ordered[ordered.isin(ids)]
    require(set(selected) == set(ids), "SELECTED_AXIS")
    return selected


def candidate_outputs(inputs, profile):
    """Only columns absent in the pre-PUF parent may be owned by this cut."""
    policy = output_policy(profile)
    node = inputs.arm_one_node
    require(
        node.id == attachment.ATTACH_NODE
        and node.kernel == attachment.SurveyPuf55AttachKernel.ref,
        "ARM_ONE_DECLARATION",
    )
    declared = {(x.entity, x.column): x for x in node.outputs}
    result = []
    for entity in ("person", "tax_unit"):
        for name in policy[entity]:
            if name in inputs.financial_parent.frame.table(entity):
                continue
            key = (entity, name)
            require(
                key in declared
                and inputs.arm_one.owners.get(key) == node.id
                and inputs.receiving.owners.get(key) == node.id,
                "ARM_ONE_OUTPUT_OWNER",
            )
            require(name in inputs.receiving.frame.table(entity), "RECEIVING_OUTPUT")
            dtype = inputs.receiving.frame.table(entity)[name].dtype
            require(dtype in (np.dtype("float32"), np.dtype("float64")), "OUTPUT_DTYPE")
            require(
                inputs.arm_one.frame.table(entity)[name].dtype == dtype, "OUTPUT_DTYPE"
            )
            require(declared[key].dtype == str(dtype), "OUTPUT_DECLARED_DTYPE")
            result.append((entity, name, str(dtype)))
    return tuple(result)


def placement_result(qualified, inputs, conditioning, merge_receipt, *, profile):
    """Compute exact full-axis columns/reasons without issuing a population.

    A canonical missing producer is graph ownership evidence only. It never
    means source NIU, source-observed zero, or legal eligibility. Actual source
    and 55-model producer authentication remains the caller's host obligation.
    """
    values.check_fixed_input_binding(qualified)
    require(
        qualified.recipients.arm == 0 and qualified.rules == values.DEVELOPMENT_RULES,
        "FIXED_POLICY",
    )
    policy = output_policy(profile)
    source_stamp, fixed_stamp = _stamp(inputs), values.fixed_input_stamp(qualified)
    frames, people, units = _axes(inputs, qualified)
    selected = _selected_ids(qualified)
    require(
        type(conditioning) is pd.DataFrame
        and conditioning.index.equals(selected)
        and tuple(conditioning) == profile.targets
        and all(d == np.dtype("float64") for d in conditioning.dtypes),
        "CONDITIONING_AXIS",
    )
    table_stamp = values.recipients._table_digest(conditioning)
    require(type(merge_receipt) is bytes, "MERGE_RECEIPT")
    receipt = codec.decode_json(merge_receipt)
    require(
        receipt["protocol"] == application.PROTOCOL
        and receipt["recipient_arm"] == 0
        and receipt["qualification_sha256"] == codec.sha(qualified.receipt)
        and receipt["conditioning_table_sha256"] == table_stamp,
        "MERGE_BINDING",
    )
    original = people.person_tax_unit_id.isin(selected)
    known = qualified.person_known.loc[original]
    current = people.loc[original]
    for name in qualified.person_values:
        take = known[name].to_numpy()
        wanted = qualified.person_values.loc[original, name].to_numpy()
        for frame in frames:
            require(name in frame.person, "FIXED_COLUMN")
            actual = (
                frame.person.set_index("person_id")
                .loc[current.index, name]
                .to_numpy(dtype="float64", na_value=np.nan)
            )
            require(
                np.array_equal(actual[take].view("u8"), wanted[take].view("u8")),
                "KNOWN_VALUE_CHANGED",
            )
    all_known = (
        known.all(axis=1).groupby(current.person_tax_unit_id).all().reindex(selected)
    )
    for name in qualified.person_values:
        expected_known = known[name].groupby(current.person_tax_unit_id).all()
        require(
            expected_known.reindex(selected).equals(
                qualified.tax_unit_known.loc[selected, name]
            ),
            "UNIT_KNOWNNESS",
        )
        take = expected_known.reindex(selected).to_numpy()
        wanted = qualified.tax_unit_values.loc[selected, name].to_numpy()
        actual = conditioning[name].to_numpy()
        require(
            np.array_equal(actual[take].view("u8"), wanted[take].view("u8")),
            "CONDITIONING_FIXED_VALUE",
        )
    counts = current.groupby("person_tax_unit_id", sort=False).size().reindex(selected)
    require(counts.gt(0).all(), "EMPTY_UNIT")
    statuses = pd.DataFrame(index=selected)
    statuses["members"] = counts
    statuses["all_fixed_members_known"] = all_known
    statuses["singleton"] = counts.eq(1)
    columns, writes = {}, {}
    candidates = candidate_outputs(inputs, profile)
    for entity, name, dtype in candidates:
        prior = inputs.arm_one.frame.table(entity).set_index(entity + "_id")[name]
        target = inputs.receiving.frame.table(entity).set_index(entity + "_id")[name]
        axis = current.index if entity == "person" else selected
        require(
            prior.loc[axis].isna().all() and target.loc[axis].isna().all(),
            "PRESERVE_NULL_OWNERSHIP_CHANGED",
        )
        require(target.equals(prior), "LATER_OUTPUT_WRITER")
        raw = conditioning[name]
        valid = np.isfinite(raw.to_numpy())
        if (
            name
            in application.values.recipients.support._PUF_TAX_DETAIL_NONNEGATIVE_OUTPUTS
        ):
            valid &= raw.to_numpy() >= 0
        enabled = all_known.to_numpy() & valid
        if entity == "person":
            enabled &= counts.eq(1).to_numpy()
        unit_ids = selected[enabled]
        output = target.copy(deep=True)
        if entity == "tax_unit":
            ids = unit_ids
            assigned = raw.loc[ids].to_numpy()
        else:
            ids = current.index[current.person_tax_unit_id.isin(unit_ids)]
            assigned = raw.loc[current.loc[ids, "person_tax_unit_id"]].to_numpy()
        converted = assigned.astype(dtype)
        require(
            np.array_equal(converted.astype("float64").view("u8"), assigned.view("u8")),
            "LOSSY_WRITE",
        )
        output.loc[ids] = converted
        columns[entity, name] = output
        writes[name] = int(len(ids))
        reason = np.where(
            ~all_known.to_numpy(),
            "fixed_values_unresolved",
            np.where(~valid, "draw_domain_unresolved", "modeled"),
        )
        if entity == "person":
            reason = np.where(
                all_known.to_numpy() & valid & ~counts.eq(1).to_numpy(),
                "person_allocation_deferred",
                reason,
            )
        statuses[name] = pd.array(reason, dtype="string")
    for entity in ("person", "tax_unit"):
        for name in policy[entity]:
            if name not in statuses:
                statuses[name] = pd.array(
                    ["preexisting_parent_preserved"] * len(statuses), dtype="string"
                )
    result_seal = tuple(
        (key, values.recipients._table_digest(series.to_frame()))
        for key, series in columns.items()
    )
    document = codec.encode_json(
        {
            "protocol": PROTOCOL,
            "profile": profile.value,
            "policy": policy,
            "qualification_sha256": codec.sha(qualified.receipt),
            "merge_sha256": codec.sha(merge_receipt),
            "input_population_stamps": list(source_stamp[:3]),
            "candidate_outputs": candidates,
            "eligibility_table": statuses.to_json(orient="table"),
            "write_counts": writes,
            "output_hashes": [[*key, digest] for key, digest in result_seal],
            "missingness": "absent_canonical_producer_before_checked_arm_one_preserve_nulls",
            "source_admission_issued": False,
            "release_eligible": False,
            "person_counts": {
                "new_singleton_policy": 5,
                "preserved_fixed": 12,
                "other_deferred": 35,
            },
            "tail_caps": False,
            "donor_positive_rate_alignment": False,
            "signed_mass_alignment": False,
        }
    )
    require(
        _stamp(inputs) == source_stamp
        and values.fixed_input_stamp(qualified) == fixed_stamp
        and values.recipients._table_digest(conditioning) == table_stamp
        and tuple(
            (key, values.recipients._table_digest(s.to_frame()))
            for key, s in columns.items()
        )
        == result_seal,
        "FINAL_INPUT_CHANGED",
    )
    return columns, document
