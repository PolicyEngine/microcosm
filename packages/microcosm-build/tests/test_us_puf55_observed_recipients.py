"""Invented arm and qualified-value arithmetic; no false financial-run issuance."""

import numpy as np
import pandas as pd
import pytest
from test_us_puf55_survey_recipients import _frames

from microcosm.build.us_runtime import puf55_survey_observed as observed
from microcosm.build.us_runtime import puf55_survey_recipients as recipients


def test_arm_zero_and_default_clone_one_have_disjoint_exact_routes():
    native, frame, report = _frames()
    measured, _ = recipients.ss._measure(
        native.person, native.table("tax_unit"), report
    )
    default = recipients._project(native, frame, report, measured)
    explicit = recipients._project(native, frame, report, measured, arm=1)
    zero = recipients._project(native, frame, report, measured, arm=0)
    assert default[2] == explicit[2]
    one_ids = []
    zero_ids = []
    for (_, a), (_, b) in zip(default[2], zero[2], strict=True):
        one = recipients.model_input.decode_recipient_matrix(a)
        original = recipients.model_input.decode_recipient_matrix(b)
        one_ids.extend(one.entity_ids)
        zero_ids.extend(original.entity_ids)
        np.testing.assert_array_equal(
            one.features.to_numpy(), original.features.to_numpy()
        )
    assert set(one_ids).isdisjoint(zero_ids)
    assert set(one_ids) == {1010, 1020, 1030, 1040, 1050}
    assert set(zero_ids) == {10, 20, 30, 40, 50}


@pytest.mark.parametrize("arm", [True, False, -1, 2, "0", None])
def test_arm_is_an_explicit_closed_integer(arm):
    with pytest.raises(ValueError, match="RECIPIENT_ARM"):
        recipients.recipient_protocol(arm)


def test_development_rule_metadata_names_maintained_assumptions():
    rules = observed.development_rule_metadata(observed.DEVELOPMENT_RULES)
    assert rules["release_science_accepted"] is False
    assert rules["observed_taxable_amount_claim"] is False
    assert rules["pension_fraction"] == observed.leaves.TAXABLE_PENSION_FRACTION
    assert rules["regular_ira_code"] == observed.leaves._IRA_DISTRIBUTION_CODE
    assert rules["source_component_statuses_reinterpreted"] is False


def _invented_values(*, arm=0, rules=()):
    """Codec/arithmetic fixture only: deliberately no issued financial owner."""
    native, frame, report = _frames()
    report["source"] = "asec"
    report["native_person_id"] = report.index.to_numpy()
    measured, _ = recipients.ss._measure(
        native.person, native.table("tax_unit"), report
    )
    person, units, matrices, _ = recipients._project(
        native, frame, report, measured, arm=arm
    )
    basis = None
    if rules:
        from test_us_current_asec_income_routing import _pure_rows, _set_amount

        raw, ages = _pure_rows(
            10,
            [70.0] * 10,
            PEN_YN=["1"] * 10,
            ANN_YN=["1"] * 10,
            DST_YN=["1"] * 10,
            DST_SC1=["4"] * 10,
            RNT_YN=["1"] * 10,
            ERN_YN=["1"] * 10,
            FRSE_YN=["1"] * 10,
        )
        for name, amount in (
            ("PNSN_VAL", 100.0),
            ("ANN_VAL", 20.0),
            ("DST_VAL1", 70.0),
            ("RNT_VAL", -40.0),
            ("FRSE_VAL", -90.0),
        ):
            _set_amount(raw, name, [amount] * 10)
        # Member 2 is underage for pension/annuity/IRA/net property. Farm has
        # its own ERN_YN/FRMOTR universe and remains source-known here.
        ages[1] = 14
        basis = observed.routing.project_income_routing(raw, ages)
        basis.index = pd.Index(np.arange(1, 11, dtype=np.int64), name="person_id")
        basis["native_person_id"] = basis.index.to_numpy()
        amounts, _ = observed._development_person_values(basis)
        for name in observed.DEVELOPMENT_TARGETS:
            # Both clone arms carry the same maintained values. Unknown source
            # reports deliberately have an unrelated finite inherited cell.
            frame.person[name] = np.tile(amounts[name].fillna(987654.0).to_numpy(), 2)
    projections = observed._project_values(
        frame, person, source_basis=basis, rules=rules
    )
    recipient = recipients.Puf55SurveyRecipients(
        object(), person, units, matrices, b"invented-not-an-owner", arm
    )
    source_evidence = observed.codec.encode_json({"invented": True})
    receipt = observed.codec.encode_json(
        {
            "protocol": observed.PROTOCOL,
            "recipient_arm": arm,
            "recipient_projection_sha256": observed.codec.sha(recipient.receipt),
            "financial_run_sha256": "a" * 64,
            "source_qualification_sha256": observed.codec.sha(source_evidence),
            "source_basis_sha256": None
            if basis is None
            else recipients._table_digest(basis),
            "financial_targets": list(observed.FINANCIAL_TARGETS),
            "development_targets": list(observed.DEVELOPMENT_TARGETS) if rules else [],
            "rule_metadata": observed.development_rule_metadata(rules),
            **{
                name + "_sha256": recipients._table_digest(table)
                for name, table in zip(
                    (
                        "person_values",
                        "person_known",
                        "tax_unit_values",
                        "tax_unit_known",
                    ),
                    projections,
                    strict=True,
                )
            },
        }
    )
    qualified = observed.Puf55SurveyFixedInputs(
        recipient, rules, *projections, basis, source_evidence, receipt
    )
    return qualified, frame


@pytest.mark.parametrize("arm", (0, 1))
@pytest.mark.parametrize("rules", ((), observed.DEVELOPMENT_RULES))
def test_typed_fixed_inputs_bind_selected_arm_matrix_and_preserve_parent(arm, rules):
    from microcosm.build.us_runtime import graph_puf55_survey_observed as graph

    value, frame = _invented_values(arm=arm, rules=rules)
    before = recipients.source._frame_identity(frame)
    edges = {
        profile: (payload, "b" * 64) for profile, payload in value.recipients.matrices
    }
    artifacts = graph._payloads(value, edges)
    assert artifacts["qualification"] == value.receipt
    source = observed.codec.decode_json(artifacts["source_basis"])
    assert source["source_taxability_or_component_split_resolved"] is False
    for profile, payload in value.recipients.matrices:
        matrix = recipients.model_input.decode_recipient_matrix(payload)
        for target, name in graph.observed_artifact_names(value, profile).items():
            actual, known = observed.observed.read_observed_target(
                artifacts[name],
                target=target,
                index=matrix.features.index,
                matrix_sha256=observed.codec.sha(payload),
                matrix_producer_key="b" * 64,
            )
            np.testing.assert_array_equal(
                actual, value.tax_unit_values.loc[matrix.features.index, target]
            )
            np.testing.assert_array_equal(
                known, value.tax_unit_known.loc[matrix.features.index, target]
            )
        assert all((identity < 1000) == (arm == 0) for identity in matrix.entity_ids)
    assert recipients.source._frame_identity(frame) == before
    assert tuple(value.person_values) == observed.FINANCIAL_TARGETS + (
        observed.DEVELOPMENT_TARGETS if rules else ()
    )


def test_partial_source_knownness_poisoning_is_all_member_and_never_inherited_nonnull():
    value, frame = _invented_values(rules=observed.DEVELOPMENT_RULES)
    pension, ira, rental, farm = observed.DEVELOPMENT_TARGETS
    for unit in (10, 1010):
        for name in (pension, ira, rental):
            assert not value.tax_unit_known.loc[unit, name]
            assert np.isnan(value.tax_unit_values.loc[unit, name])
        assert value.tax_unit_known.loc[unit, farm]
        assert value.tax_unit_values.loc[unit, farm] == -180.0
    assert frame.person.loc[frame.person.person_id.eq(2), pension].iloc[0] == 987654
    assert (
        value.tax_unit_values.loc[20, pension]
        == 3 * 120.0 * observed.leaves.TAXABLE_PENSION_FRACTION
    )
    assert value.tax_unit_values.loc[20, ira] == 210.0
    assert value.tax_unit_values.loc[20, rental] == -120.0
    assert not value.source_basis.pension_annuity_taxable_amount_known.any()
    assert not value.source_basis.net_property_component_split_known.any()


def test_acs_development_masks_stay_unknown_and_mixed_origin_units_refuse():
    value, frame = _invented_values(rules=observed.DEVELOPMENT_RULES)
    person = value.recipients.person.copy(deep=True)
    acs = person.native_person_id.isin((9, 10))
    person.loc[acs, "source"] = "acs"
    basis = value.source_basis.iloc[:8].copy()
    _, _, amounts, known = observed._project_values(
        frame, person, source_basis=basis, rules=value.rules
    )
    assert (
        not known.loc[[50, 1050], list(observed.DEVELOPMENT_TARGETS)].to_numpy().any()
    )
    assert (
        amounts.loc[[50, 1050], list(observed.DEVELOPMENT_TARGETS)].isna().all().all()
    )
    person.loc[person.index == 10, "source"] = "asec"
    with pytest.raises(ValueError, match="MIXED_OR_EMPTY_ORIGIN_UNIT"):
        observed._project_values(frame, person, source_basis=basis, rules=value.rules)


@pytest.mark.parametrize(
    "defect",
    ("known_value", "known_dtype", "amount_dtype", "axis", "source", "receipt"),
)
def test_detached_mutations_refuse_before_artifact_encoding(defect, monkeypatch):
    from dataclasses import replace

    value, _ = _invented_values(rules=observed.DEVELOPMENT_RULES)
    if defect == "known_value":
        value.tax_unit_values.iloc[0, 0] += 1
    elif defect == "known_dtype":
        value.tax_unit_known[value.tax_unit_known.columns[0]] = 1
    elif defect == "amount_dtype":
        value.tax_unit_values[value.tax_unit_values.columns[0]] = (
            value.tax_unit_values.iloc[:, 0].astype("float32")
        )
    elif defect == "axis":
        value.tax_unit_values.index = value.tax_unit_values.index[::-1]
    elif defect == "source":
        value.source_basis.iloc[0, 0] += 1
    else:
        value = replace(value, source_evidence=b"changed")
    monkeypatch.setattr(
        observed.observed,
        "encode_observed_target",
        lambda *a, **k: pytest.fail("encoded mutated values"),
    )
    profile, payload = value.recipients.matrices[0]
    with pytest.raises(ValueError, match="VALUE_"):
        observed.observed_target_artifacts(
            value, profile=profile, matrix_payload=payload, matrix_producer_key="b" * 64
        )


@pytest.mark.parametrize(
    "defect",
    ("carried", "native_id", "unresolved_flag", "nullable_flag", "known_amount"),
)
def test_development_projection_refuses_source_or_carried_contradictions(defect):
    value, frame = _invented_values(rules=observed.DEVELOPMENT_RULES)
    basis = value.source_basis.copy(deep=True)
    if defect == "carried":
        frame.person.loc[0, observed.DEVELOPMENT_TARGETS[0]] += 1
    elif defect == "native_id":
        basis.loc[1, "native_person_id"] = 999
    elif defect == "unresolved_flag":
        basis.loc[1, "net_property_component_split_known"] = True
    elif defect == "nullable_flag":
        basis["net_property_component_split_known"] = pd.array(
            [pd.NA] * len(basis), dtype="boolean"
        )
    else:
        basis.loc[1, "farm_known_amount"] = np.nan
    with pytest.raises(ValueError):
        observed._project_values(
            frame, value.recipients.person, source_basis=basis, rules=value.rules
        )


def test_authenticated_original_routing_is_reused_without_resolving_taxability(
    tmp_path, monkeypatch
):
    from test_us_current_asec_income_routing import _qualified

    _, source = _qualified(tmp_path, monkeypatch)
    before = recipients._table_digest(source.person)
    amounts, known = observed._development_person_values(source.person)
    native = source.person.native_person_id
    position = source.person.index[native.eq(105)][0]
    assert known.loc[position, "taxable_ira_distributions"]
    assert amounts.loc[position, "taxable_ira_distributions"] == 7000.0
    assert known.loc[position, "rental_income"]
    assert amounts.loc[position, "rental_income"] == -400.0
    assert not known.loc[position, "taxable_private_pension_income"]  # ANN_VAL NIU
    child = source.person.index[native.eq(106)][0]
    assert not known.loc[child].any()
    assert recipients._table_digest(source.person) == before


def test_public_qualification_and_replay_never_accept_a_detached_projection():
    from microcosm.build.us_runtime import graph_puf55_survey_observed as graph

    value, _ = _invented_values()
    with pytest.raises(ValueError):
        observed.qualify_puf55_survey_fixed_inputs(value)
    with pytest.raises(ValueError):
        graph.verify_materialized_puf55_fixed_inputs(value, artifacts={}, matrices={})


def test_new_source_and_graph_keep_protected_provenance_scanner():
    import inspect

    import test_us_spine_blindness as guard

    from microcosm.build.us_runtime import graph_puf55_survey_observed as graph

    for module in (observed, graph):
        name = module.__name__.split(".")[-1] + ".py"
        assert name in guard._US_LAUNCH_GRAPH_RUNTIME_MODULES
        assert name not in guard._SOURCE_SPINE_PROVENANCE_OWNERS
        assert (
            guard._non_owner_source_spine_accesses(name, inspect.getsource(module))
            == ()
        )
        assert guard._non_owner_source_spine_accesses(
            name, 'def bad(table):\n return table["person_support_channel"]\n'
        )


@pytest.mark.parametrize("arm", (0, 1))
def test_declarations_bind_full_receiving_slices_actual_edges_and_rule_metadata(
    arm, monkeypatch
):
    """Declaration component fixture; never replace a financial/source issuer."""
    from types import SimpleNamespace

    from microcosm.build.us_runtime import graph_puf55_survey_observed as graph

    value, frame = _invented_values(arm=arm, rules=observed.DEVELOPMENT_RULES)
    entry = (
        None,
        b"invented-run",
        SimpleNamespace(
            financial_population=SimpleNamespace(frame=frame, version="invented"),
            source_items=(("invented_source", "invented_path"),),
        ),
    )
    monkeypatch.setattr(graph.parent, "_check_values", lambda x: (entry, {}))
    monkeypatch.setattr(graph.parent, "_pins", lambda x: {})
    monkeypatch.setattr(graph.parent, "_edges", lambda x: ())
    (node,) = graph.puf55_survey_fixed_input_nodes(value)
    assert node.id == graph.fixed_input_node_id(arm)
    assert node.params["recipient_arm"] == arm
    metadata = observed.codec.decode_json(node.params["rule_metadata"].encode())
    assert metadata == observed.development_rule_metadata(value.rules)
    assert node.inputs == graph.parent.financial.financial._inputs(frame)
    assert set(observed.DEVELOPMENT_TARGETS) <= set(
        next(s for s in node.inputs if s.entity == "person").columns
    )
    assert {e.producer for e in node.artifact_inputs} == set(
        graph.parent.recipient_node_ids(arm)
    )
    assert all(
        e.type == observed.observed.OBSERVED_TARGET_TYPE
        for e in node.artifact_outputs
        if e.name not in ("qualification", "source_basis")
    )


@pytest.mark.parametrize("defect", (None, "payload", "type", "key", "sibling_key"))
def test_matrix_transport_checks_actual_typed_keys_and_both_profiles(
    defect, monkeypatch
):
    """Only the typed-edge check is isolated; no source/run authority is minted."""
    from dataclasses import replace
    from types import SimpleNamespace

    from microcosm.build.us_runtime import graph_puf55_survey_observed as graph
    from microcosm.graph import (
        ArtifactInput,
        ArtifactType,
        ArtifactValue,
        Node,
        Numeric,
        NumericScope,
    )
    from microcosm.graph.keys import opaque_artifact_key

    value, _ = _invented_values()
    projection_id, matrix_id = graph.parent.recipient_node_ids(0)
    edges = [
        ArtifactInput(
            "projection", projection_id, "projection", graph.parent.projection_type(0)
        )
    ]
    artifacts = {
        "projection": ArtifactValue(
            value.recipients.receipt,
            edges[0].type,
            opaque_artifact_key("a" * 64, "projection"),
            "a" * 64,
            NumericScope(Numeric.BITWISE),
        )
    }
    for profile, payload in value.recipients.matrices:
        name = graph.parent._NAMES[profile]
        edges.append(
            ArtifactInput(
                name, matrix_id, name, graph.parent.model_input.RECIPIENT_MATRIX_TYPE
            )
        )
        artifacts[name] = ArtifactValue(
            payload,
            edges[-1].type,
            opaque_artifact_key("b" * 64, name),
            "b" * 64,
            NumericScope(Numeric.BITWISE),
        )
    target = graph.parent._NAMES[value.recipients.matrices[0][0]]
    if defect == "payload":
        artifacts[target] = replace(artifacts[target], payload=b"changed")
    elif defect == "type":
        artifacts[target] = replace(
            artifacts[target], type=ArtifactType("incorrect", 1)
        )
    elif defect == "key":
        artifacts[target] = replace(artifacts[target], key="c" * 64)
    elif defect == "sibling_key":
        artifacts[target] = replace(
            artifacts[target],
            producer_key="c" * 64,
            key=opaque_artifact_key("c" * 64, target),
        )
    context = SimpleNamespace(
        node=Node(
            graph.fixed_input_node_id(0),
            graph.Puf55SurveyFixedInputKernel.ref,
            artifact_inputs=tuple(edges),
        ),
        artifacts=artifacts,
    )
    # Leave shared.artifact/siblings and _matrices live. The retained-owner
    # context gate is outside this source-free transport test's scope.
    monkeypatch.setattr(
        graph.Puf55SurveyFixedInputKernel, "_context", lambda *a, **k: None
    )
    monkeypatch.setattr(
        graph, "puf55_survey_fixed_input_nodes", lambda q: (context.node,)
    )
    kernel = graph.Puf55SurveyFixedInputKernel(object())
    if defect is None:
        assert kernel._matrices(context, value) == {
            p: (data, "b" * 64) for p, data in value.recipients.matrices
        }
    else:
        with pytest.raises(ValueError):
            kernel._matrices(context, value)
