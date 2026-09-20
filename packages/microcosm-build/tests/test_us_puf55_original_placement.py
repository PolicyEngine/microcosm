"""Invented eligibility and typed codecs; no source owner or 55-fit claim."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_us_puf55_observed_recipients import _invented_values
from test_us_puf55_original_application import _codec_only_full55, real_chain

from microcosm.build.us_runtime import graph_puf55_original_placement as graph
from microcosm.build.us_runtime import puf55_original_placement as placement
from microcosm.frame import Frame
from microcosm.graph import (
    ArtifactInput,
    ArtifactType,
    ArtifactValue,
    KernelResult,
    Node,
    Numeric,
    NumericScope,
    Owned,
    platform_fingerprint,
)
from microcosm.graph import population as populations
from microcosm.graph.keys import opaque_artifact_key

# The real_chain fixture supplies only three real fits for existing envelope
# templates. The 55-target envelopes below remain explicitly synthetic.
assert real_chain
values, application, codec = placement.values, placement.application, placement.codec
PROFILE = placement.attachment.PROFILES[0]
AFTER = ArtifactInput(
    "terminal", "invented.late", "binding", ArtifactType("invented", 1)
)
SEEDS = dict(clone_one_seed=31, original_application_seed=73)


def _frame(frame, tables):
    return Frame(
        tables,
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        frame.strata,
        metadata=frame.metadata,
    )


def fixture(*, preexisting=None, dtype="float64"):
    qualified, parent = _invented_values(rules=values.DEVELOPMENT_RULES)
    if preexisting:
        entity, name = preexisting
        parent.table(entity)[name] = np.full(parent.n(entity), 321.0)
    financial = populations.Population.from_frame(parent, "invented.financial")
    tables = {e: parent.table(e).copy(deep=True) for e in parent.entities}
    outputs = []
    for entity, names in (
        ("person", placement.SINGLETON_OUTPUTS),
        ("tax_unit", placement.UNIT_OUTPUTS),
    ):
        for name in names:
            if name in tables[entity]:
                continue
            tables[entity][name] = np.where(
                tables[entity][entity + "_id"] < 1000, np.nan, 777.0
            ).astype(dtype)
            outputs.append(Owned(entity, name, dtype))
    node = Node(
        placement.attachment.ATTACH_NODE,
        placement.attachment.SurveyPuf55AttachKernel.ref,
        population="invented.arm_one_version",
        outputs=tuple(outputs),
    )
    owners = dict(financial.owners)
    owners.update({(o.entity, o.column): node.id for o in outputs})
    arm_one = populations.Population.from_frame(
        _frame(parent, tables), node.population, owners
    )
    receiving = populations.Population.from_frame(
        _frame(parent, {e: t.copy(deep=True) for e, t in tables.items()}),
        "invented.late_version",
        owners,
    )
    inputs = placement.PlacementInputs(financial, arm_one, receiving, node)
    index = placement._selected_ids(qualified)
    conditioning = pd.DataFrame(20.25, index=index, columns=PROFILE.targets)
    for name in qualified.tax_unit_values:
        mask = qualified.tax_unit_known.loc[index, name]
        conditioning.loc[mask, name] = qualified.tax_unit_values.loc[index[mask], name]
    return qualified, inputs, conditioning


def result(qualified, inputs, table):
    receipt = codec.encode_json(
        {
            "protocol": application.PROTOCOL,
            "recipient_arm": 0,
            "qualification_sha256": codec.sha(qualified.receipt),
            "conditioning_table_sha256": values.recipients._table_digest(table),
        }
    )
    return placement.placement_result(
        qualified, inputs, table, receipt, profile=PROFILE
    )


def routes(qualified):
    return tuple(
        placement.attachment.route_nodes(p, seed=31, n_estimators=3, zero_atol=0.0)
        for p in placement.attachment.PROFILES
        if p.value in dict(qualified.recipients.matrices)
    )


def test_actual_roster_is_52_plus_3_with_disjoint_5_12_35_person_policy():
    for profile in placement.attachment.PROFILES:
        policy = placement.output_policy(profile)
        names = [
            *policy["person"],
            *policy["preserved_fixed_person"],
            *policy["deferred_person"],
        ]
        assert len(names) == len(set(names)) == 52
        assert [
            len(policy[x])
            for x in ("person", "preserved_fixed_person", "deferred_person")
        ] == [5, 12, 35]
        assert set(names) == set(profile.person_outputs)
        assert policy["tax_unit"] == placement.UNIT_OUTPUTS
        assert not any("mortgage" in x for x in profile.tax_unit_outputs)
        assert policy["release_science_accepted"] is False


def test_exact_singletons_and_knownness_preserve_mixed_units_and_clone_one():
    qualified, inputs, table = fixture()
    table.loc[40, "partnership_income"] = -72.5
    before = placement._stamp(inputs)
    columns, payload = result(qualified, inputs, table)
    document = codec.decode_json(payload)
    assert document["person_counts"] == {
        "new_singleton_policy": 5,
        "preserved_fixed": 12,
        "other_deferred": 35,
    }
    assert document["source_admission_issued"] is document["release_eligible"] is False
    for (entity, name), column in columns.items():
        incumbent = inputs.receiving.frame.table(entity).set_index(entity + "_id")[name]
        if entity == "person":
            changed = [8]  # actual unit 40 has exactly one person; all others blocked.
        else:
            changed = [
                20,
                30,
                40,
                50,
            ]  # unit 10 includes an unqualified underage member.
        assert (
            column.loc[changed].tolist()
            == table.loc[[40] if entity == "person" else changed, name].tolist()
        )
        pd.testing.assert_series_equal(column.drop(changed), incumbent.drop(changed))
        assert document["write_counts"][name] == len(changed)
    assert columns["person", "partnership_income"].loc[8] == -72.5
    assert placement._stamp(inputs) == before
    # Returned columns are detached and cannot alter inherited cells.
    columns["person", placement.SINGLETON_OUTPUTS[0]].iloc[-1] = -999.0
    assert placement._stamp(inputs) == before


def test_preexisting_output_is_preserved_even_if_null_and_has_no_new_owner():
    name = placement.SINGLETON_OUTPUTS[0]
    qualified, inputs, table = fixture(preexisting=("person", name))
    for population in (inputs.financial_parent, inputs.arm_one, inputs.receiving):
        population.frame.person[name] = np.nan
    columns, payload = result(qualified, inputs, table)
    assert ("person", name) not in columns
    assert name not in codec.decode_json(payload)["write_counts"]
    assert all(
        o.column != name
        for o in graph.original_placement_nodes(
            qualified, inputs, routes(qualified), after=AFTER, **SEEDS
        )[1].outputs
    )


@pytest.mark.parametrize("where", ("financial_parent", "arm_one", "receiving"))
def test_known_person_value_change_refuses(where):
    qualified, inputs, table = fixture()
    getattr(inputs, where).frame.person.loc[0, values.FINANCIAL_TARGETS[0]] += 1
    with pytest.raises(ValueError, match="KNOWN_VALUE_CHANGED"):
        result(qualified, inputs, table)


@pytest.mark.parametrize(
    "change",
    (
        "membership",
        "axis",
        "owner",
        "late_owner",
        "declaration",
        "nonnull",
        "late_clone",
    ),
)
def test_invalid_ancestry_descriptions_refuse(change):
    qualified, inputs, table = fixture()
    if change == "membership":
        inputs.receiving.frame.person.loc[0, "person_tax_unit_id"] = 40
    elif change == "axis":
        inputs.receiving.frame.person.loc[0, "person_id"] = 123456
    elif change == "owner":
        owners = dict(inputs.arm_one.owners)
        owners["person", placement.SINGLETON_OUTPUTS[0]] = "invented.other"
        inputs = replace(inputs, arm_one=replace(inputs.arm_one, owners=owners))
    elif change == "late_owner":
        owners = dict(inputs.receiving.owners)
        owners["person", placement.SINGLETON_OUTPUTS[0]] = "invented.later"
        inputs = replace(inputs, receiving=replace(inputs.receiving, owners=owners))
    elif change == "declaration":
        inputs = replace(
            inputs, arm_one_node=replace(inputs.arm_one_node, kernel="invented.wrong@1")
        )
    elif change == "nonnull":
        inputs.receiving.frame.person.loc[0, placement.SINGLETON_OUTPUTS[0]] = 0.0
    else:
        inputs.receiving.frame.person.loc[19, placement.SINGLETON_OUTPUTS[0]] = 0.0
    with pytest.raises(ValueError, match="PUF55_ORIGINAL_PLACEMENT"):
        result(qualified, inputs, table)


def test_fixed_conditioning_change_refuses_before_placement():
    qualified, inputs, table = fixture()
    table.loc[40, values.FINANCIAL_TARGETS[0]] += 1
    with pytest.raises(ValueError, match="CONDITIONING_FIXED_VALUE"):
        result(qualified, inputs, table)


def test_float32_retained_dtype_requires_exact_no_loss_assignment():
    qualified, inputs, table = fixture(dtype="float32")
    columns, _ = result(qualified, inputs, table)
    assert all(s.dtype == np.dtype("float32") for s in columns.values())
    table.loc[40, placement.SINGLETON_OUTPUTS[0]] = 0.1
    with pytest.raises(ValueError, match="LOSSY_WRITE"):
        result(qualified, inputs, table)


def test_negative_nonnegative_draw_is_unresolved_without_clipping():
    qualified, inputs, table = fixture()
    name = "long_term_capital_gains_on_collectibles"
    table.loc[40, name] = -1.0
    columns, payload = result(qualified, inputs, table)
    assert np.isnan(columns["person", name].loc[8])
    assert codec.decode_json(payload)["write_counts"][name] == 0


def test_genuine_owner_capture_refuses_descriptive_objects():
    with pytest.raises(ValueError, match="UNISSUED_PUF55_RUN"):
        placement._capture_checked_puf_ancestors(object(), object())


def test_typed_fragment_has_every_apply_step_and_exact_whole_population_patch():
    qualified, inputs, table = fixture()
    keep, attach = graph.original_placement_nodes(
        qualified, inputs, routes(qualified), after=AFTER, **SEEDS
    )
    assert keep.base == inputs.receiving.version and keep.mass == "conserve"
    assert attach.population == keep.id and len(attach.outputs) == 8
    assert all(o.rewrite for o in attach.outputs)
    for r in (0, 1):
        assert {
            e.name for e in attach.artifact_inputs if e.name.startswith(f"r{r}_t")
        } == {
            f"r{r}_t{t}_{name}"
            for t in range(55)
            for name in (
                "model",
                "training_state",
                "raw_draw",
                "conditioning",
                "apply_state",
            )
        }
    retained = graph.keep_all_population(inputs, keep)
    columns, payload = result(qualified, inputs, table)
    final = populations.patch(
        retained,
        attach,
        KernelResult(columns=columns, artifacts={"placement": payload}),
    )
    for entity in inputs.receiving.frame.entities:
        expected = inputs.receiving.frame.table(entity).copy(deep=True)
        for (e, name), column in columns.items():
            if e == entity:
                expected[name] = column.to_numpy()
                assert final.owners[e, name] == attach.id
        pd.testing.assert_frame_equal(final.frame.table(entity), expected)
    for entity in inputs.receiving.frame.weighted_entities:
        old, new = (
            inputs.receiving.frame.weights_for(entity),
            final.frame.weights_for(entity),
        )
        assert old.kind == new.kind and old.values.tobytes() == new.values.tobytes()
    pd.testing.assert_series_equal(final.frame.strata, inputs.receiving.frame.strata)
    assert final.mass_ledger[-1].policy == "conserve"


def _edge(payload, kind, name):
    producer = codec.sha(("invented:" + name).encode())
    return ArtifactValue(
        payload,
        kind,
        opaque_artifact_key(producer, name),
        producer,
        NumericScope(Numeric.PLATFORM_BITWISE, platform=platform_fingerprint()),
    )


@pytest.mark.parametrize("late_change", ("frame", "artifact"))
def test_typed_result_consumes_strict_full55_codecs_without_claiming_55_fits(
    real_chain,
    monkeypatch,
    late_change,
):
    qualified, transports = _codec_only_full55(real_chain)
    _, inputs, _ = fixture()
    declarations = routes(qualified)
    keep, node = graph.original_placement_nodes(
        qualified, inputs, declarations, after=AFTER, **SEEDS
    )
    fixed = graph.fixed_graph._payloads(
        qualified,
        {
            route.profile.value: (route.matrix.payload, route.matrix.producer_key)
            for route in transports
        },
    )
    artifacts = {
        "qualification": _edge(
            qualified.receipt, graph.fixed_graph.QUALIFICATION_TYPE, "qualification"
        ),
        "source_basis": _edge(
            fixed["source_basis"], graph.fixed_graph.SOURCE_BASIS_TYPE, "source_basis"
        ),
    }
    for r, route in enumerate(transports):
        artifacts[f"r{r}_matrix"] = route.matrix
        artifacts.update(
            {f"r{r}_fixed_{target}": edge for target, edge in route.fixed_inputs}
        )
        for t, step in enumerate(route.steps):
            for name in (
                "model",
                "training_state",
                "raw_draw",
                "conditioning",
                "apply_state",
            ):
                artifacts[f"r{r}_t{t}_{name}"] = getattr(step, name)
    actual = graph.original_placement_result(
        node, qualified, inputs, declarations, artifacts, after=AFTER, **SEEDS
    )
    assert (
        codec.decode_json(actual.artifacts["placement"])["source_admission_issued"]
        is False
    )
    final = populations.patch(graph.keep_all_population(inputs, keep), node, actual)
    assert np.isfinite(final.frame.person.loc[7, placement.SINGLETON_OUTPUTS[0]])
    # Missing and changed downstream steps cannot be bypassed just because this
    # cut only attaches eight outputs; the strict chain still consumes all 55.
    missing = artifacts.pop("r0_t54_raw_draw")
    with pytest.raises(ValueError, match="ARTIFACT_ROSTER"):
        graph.original_placement_result(
            node, qualified, inputs, declarations, artifacts, after=AFTER, **SEEDS
        )

    artifacts["r0_t54_raw_draw"] = missing
    original_merge = graph.application.merge_puf55_original_conditioning

    def changed_after_merge(*args, **kwargs):
        merged = original_merge(*args, **kwargs)
        # Even an unrelated retained cell changed by a callback must prevent
        # return; source stamps precede all decoding/callbacks.
        if late_change == "frame":
            inputs.receiving.frame.person.loc[0, "age"] += 1
        else:
            artifacts["r0_t54_raw_draw"] = replace(missing, payload=b"changed")
        return merged

    monkeypatch.setattr(
        graph.application, "merge_puf55_original_conditioning", changed_after_merge
    )
    with pytest.raises(ValueError, match="RESULT_INPUT_CHANGED"):
        graph.original_placement_result(
            node, qualified, inputs, declarations, artifacts, after=AFTER, **SEEDS
        )


@pytest.mark.parametrize(
    "module", ("puf55_original_placement.py", "graph_puf55_original_placement.py")
)
def test_new_fragment_uses_reviewed_dynamic_selectors_without_provenance_exemption(
    module,
):
    import test_us_spine_blindness as scanner

    assert module not in scanner._SOURCE_SPINE_PROVENANCE_OWNERS
    assert (
        scanner._non_owner_source_spine_accesses(
            module, (scanner._US_RUNTIME / module).read_text()
        )
        == ()
    )
    assert scanner._non_owner_source_spine_accesses(
        module, 'x = table["person_spine_source_id"]'
    )
