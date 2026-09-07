"""Closure admission, drift refusal, and a real PolicyEngine-US evaluation."""

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import asec_engine_evaluation as evaluation
from microcosm.build.us_runtime.asec_engine_evaluation import (
    ADMITTED_ENGINE_OUTPUT_CONTRACTS,
    ADMITTED_ROOTS,
    BLOCKED_ENGINE_OUTPUTS,
    EngineAdmissionError,
    EngineOutputContract,
    admit_engine_outputs,
    decode_engine_evaluation,
    encode_engine_evaluation,
    load_engine_defaults,
    materialize_engine_outputs,
)
from microcosm.build.us_runtime.cps_carried_current import (
    CPS_CARRIED_CURRENT_PERSON_LEAVES,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.frame.adapters.policyengine_us import (
    PolicyEngineUSEngine,
    PolicyEngineUSVariableMetadataIndex,
)

pytestmark = pytest.mark.requires_us

LEAF_VALUES = {
    "employment_income_before_lsr": [30000.0, 0.0, 12000.0, 0.0, 45000.0, 0.0],
    "qualified_dividend_income": [448.0, 0.0, 0.0, 0.0, 89.6, 0.0],
    "non_qualified_dividend_income": [552.0, 0.0, 0.0, 0.0, 110.4, 0.0],
    "social_security_retirement": [0.0, 0.0, 0.0, 0.0, 0.0, 9000.0],
    "social_security_disability": [0.0, 7000.0, 0.0, 0.0, 0.0, 0.0],
    "social_security_survivors": [0.0, 0.0, 5000.0, 0.0, 0.0, 0.0],
    "social_security_dependents": [0.0, 0.0, 0.0, 3000.0, 0.0, 0.0],
    "long_term_capital_gains_before_response": [880.0, 0.0, 0.0, 0.0, 176.0, 0.0],
    "short_term_capital_gains": [120.0, 0.0, 0.0, 0.0, 24.0, 0.0],
}


def index():
    return PolicyEngineUSVariableMetadataIndex()


def tiny_frame():
    """Six invented persons in three households; no real microdata is read."""
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 7, dtype=np.int64),
            **{
                US_SCHEMA.membership_column(group): np.repeat(
                    np.arange(1, 4, dtype=np.int64), 2
                )
                for group in US_SCHEMA.group_entities
            },
        }
    )
    for leaf in CPS_CARRIED_CURRENT_PERSON_LEAVES:
        person[leaf] = np.asarray(LEAF_VALUES.get(leaf, [0.0] * 6), dtype="float64")
    person["age"] = np.asarray([40.0, 45.0, 12.0, 9.0, 67.0, 70.0])
    tables = {"person": person}
    for group in US_SCHEMA.group_entities:
        tables[group] = pd.DataFrame(
            {US_SCHEMA.id_column(group): np.arange(1, 4, dtype=np.int64)}
        )
    tables["spm_unit"]["spm_unit_pre_subsidy_childcare_expenses"] = np.asarray(
        [0.0, 1200.0, 0.0]
    )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.asarray([11.0, 12.0, 13.0]), WeightKind.DESIGN)},
    )


def test_defaults_resource_is_pinned_with_an_empty_allowlist():
    document = load_engine_defaults()
    assert document["allowlist"] == []
    assert document["release_eligible"] is False
    assert tuple(sorted(document["admitted_roots"])) == ADMITTED_ROOTS
    assert tuple(sorted(document["blocked_roots"])) == BLOCKED_ENGINE_OUTPUTS


def test_admission_accepts_only_the_four_complete_closures():
    contracts = admit_engine_outputs(
        index(), produced_leaves=CPS_CARRIED_CURRENT_PERSON_LEAVES
    )
    assert tuple(contract.root for contract in contracts) == ADMITTED_ROOTS
    assert contracts == ADMITTED_ENGINE_OUTPUT_CONTRACTS
    for contract in contracts:
        assert set(contract.input_leaves) <= set(CPS_CARRIED_CURRENT_PERSON_LEAVES)


def test_blocked_roots_stay_blocked_by_a_real_unproduced_leaf():
    metadata = index()
    produced = set(CPS_CARRIED_CURRENT_PERSON_LEAVES)
    for root in BLOCKED_ENGINE_OUTPUTS:
        closure = metadata.variable_dependency_closure(root)
        assert set(closure.input_leaves) - produced
    assert "sstb_self_employment_income_before_lsr" in set(
        metadata.variable_dependency_closure("self_employment_income").input_leaves
    )
    assert "tax_exempt_interest_income" in set(
        metadata.variable_dependency_closure("interest_income").input_leaves
    )
    assert "taxable_public_pension_income" in set(
        metadata.variable_dependency_closure("pension_income").input_leaves
    )


def test_a_missing_produced_leaf_blocks_its_root():
    produced = set(CPS_CARRIED_CURRENT_PERSON_LEAVES) - {"qualified_dividend_income"}
    with pytest.raises(EngineAdmissionError, match="unproduced closure leaves"):
        admit_engine_outputs(index(), produced_leaves=produced)


def test_closure_drift_refuses_before_any_simulation(monkeypatch):
    drifted = tuple(
        EngineOutputContract(
            root=contract.root,
            entity=contract.entity,
            engine_version=contract.engine_version,
            input_leaves=contract.input_leaves,
            formula_node_count=contract.formula_node_count + 1,
            edge_count=contract.edge_count,
            sha256=contract.sha256,
        )
        for contract in ADMITTED_ENGINE_OUTPUT_CONTRACTS
    )
    monkeypatch.setattr(evaluation, "ADMITTED_ENGINE_OUTPUT_CONTRACTS", drifted)
    with pytest.raises(EngineAdmissionError, match="closure drifted"):
        admit_engine_outputs(index(), produced_leaves=CPS_CARRIED_CURRENT_PERSON_LEAVES)


def test_a_nonempty_allowlist_refuses():
    document = dict(load_engine_defaults())
    document["allowlist"] = [{"leaf": "tax_exempt_interest_income", "value": 0.0}]
    with pytest.raises(EngineAdmissionError, match="ENGINE_DEFAULTS_ALLOWLIST"):
        admit_engine_outputs(
            index(),
            produced_leaves=CPS_CARRIED_CURRENT_PERSON_LEAVES,
            defaults=document,
        )


def test_real_policyengine_us_evaluates_the_four_admitted_outputs():
    """The real adapter runs; this test is never skipped in the engine lane."""
    frame = tiny_frame()
    contracts = admit_engine_outputs(
        index(), produced_leaves=CPS_CARRIED_CURRENT_PERSON_LEAVES
    )
    result = materialize_engine_outputs(
        frame,
        PolicyEngineUSEngine(),
        contracts,
        context_bindings={"fixture": "invented six-person frame"},
    )
    assert result.roots == ADMITTED_ROOTS
    for root in ADMITTED_ROOTS:
        assert result.values[root].shape == (6,)
    # employment_income has no behavioural response without an LSR reform.
    assert (
        result.values["employment_income"].tolist()
        == LEAF_VALUES["employment_income_before_lsr"]
    )
    assert result.values["dividend_income"].tolist() == pytest.approx(
        [
            a + b
            for a, b in zip(
                LEAF_VALUES["qualified_dividend_income"],
                LEAF_VALUES["non_qualified_dividend_income"],
                strict=True,
            )
        ]
    )
    assert result.values["social_security"].tolist() == pytest.approx(
        [0.0, 7000.0, 5000.0, 3000.0, 0.0, 9000.0]
    )
    assert result.values["capital_gains"].tolist() == pytest.approx(
        [1000.0, 0.0, 0.0, 0.0, 200.0, 0.0]
    )
    header = result.header
    assert header["release_eligible"] is False
    assert header["representative"] is False
    assert header["recorded_defaults"] == []
    assert header["engine_version"] == "1.819.0"
    assert header["period"] == 2024
    assert header["blocked_roots"] == list(BLOCKED_ENGINE_OUTPUTS)
    assert header["aggregates"]["social_security"]["unweighted_total"] == 24000.0


def test_evaluation_payload_round_trips_and_refuses_tampering():
    frame = tiny_frame()
    contracts = admit_engine_outputs(
        index(), produced_leaves=CPS_CARRIED_CURRENT_PERSON_LEAVES
    )
    result = materialize_engine_outputs(frame, PolicyEngineUSEngine(), contracts)
    payload = encode_engine_evaluation(result)
    decoded = decode_engine_evaluation(payload)
    assert decoded.header == result.header
    for root in ADMITTED_ROOTS:
        assert decoded.values[root].tobytes() == result.values[root].tobytes()
    assert encode_engine_evaluation(decoded) == payload
    tampered = bytearray(payload)
    tampered[-40] ^= 0xFF
    with pytest.raises(EngineAdmissionError, match="PAYLOAD_CHECKSUM"):
        decode_engine_evaluation(bytes(tampered))
    with pytest.raises(EngineAdmissionError, match="PAYLOAD_VERSION"):
        decode_engine_evaluation(b"X" + payload[1:])


def test_outputs_are_never_returned_as_population_cells():
    frame = tiny_frame()
    contracts = admit_engine_outputs(
        index(), produced_leaves=CPS_CARRIED_CURRENT_PERSON_LEAVES
    )
    result = materialize_engine_outputs(frame, PolicyEngineUSEngine(), contracts)
    # The evaluation holds arrays only; the frame it ran on is unchanged.
    assert set(result.values) == set(ADMITTED_ROOTS)
    for root in ADMITTED_ROOTS:
        assert root not in frame.person.columns


def test_baseline_runtime_refuses_same_version_source_or_parameter_drift(monkeypatch):
    expected = load_engine_defaults()["baseline_runtime"]

    def changed(package, module):
        result = dict(expected[package])
        if package == "policyengine-us":
            result["source_and_parameters_sha256"] = "f" * 64
        return result

    monkeypatch.setattr(evaluation, "_runtime_package_identity", changed)
    with pytest.raises(EngineAdmissionError, match="ENGINE_RUNTIME_DRIFT"):
        evaluation.engine_runtime_identity()


def test_runtime_identity_binds_code_and_baseline_parameters_without_data(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    installed = SimpleNamespace(version="test", locate_file=lambda module: tmp_path)
    monkeypatch.setattr(evaluation, "distribution", lambda package: installed)
    (tmp_path / "formula.py").write_text("result = x + y\n")
    params = tmp_path / "parameters"
    params.mkdir()
    parameter = params / "baseline.yaml"
    parameter.write_text("value: 0\n")
    original = evaluation._runtime_package_identity("test", "test")
    assert original["files"] == 2
    # Source-only inspection must not read data: even an unreadable or invalid
    # population payload is outside the exact source/parameter roster.
    (tmp_path / "population.h5").write_bytes(b"not a source input")
    assert evaluation._runtime_package_identity("test", "test") == original
    parameter.write_text("value: 1\n")
    altered = evaluation._runtime_package_identity("test", "test")
    assert altered != original
    parameter.write_text("value: 0\n")
    (tmp_path / "formula.py").write_text("result = x - y\n")
    assert evaluation._runtime_package_identity("test", "test") != original


def test_forged_contract_does_not_reach_engine():
    contracts = ADMITTED_ENGINE_OUTPUT_CONTRACTS[:-1]
    with pytest.raises(EngineAdmissionError, match="ENGINE_CONTRACT_ROSTER"):
        materialize_engine_outputs(tiny_frame(), PolicyEngineUSEngine(), contracts)


def test_runtime_refuses_shadow_package_before_any_engine_call(tmp_path, monkeypatch):
    from types import SimpleNamespace

    shadow = SimpleNamespace(
        origin=str(tmp_path / "__init__.py"), submodule_search_locations=[str(tmp_path)]
    )
    monkeypatch.setattr(evaluation, "find_spec", lambda name: shadow)
    with pytest.raises(EngineAdmissionError, match="ENGINE_RUNTIME_ORIGIN"):
        evaluation.engine_runtime_identity()


def test_runtime_refuses_loaded_submodule_outside_pinned_distribution(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    monkeypatch.setitem(
        evaluation.sys.modules,
        "policyengine_us.unpinned_fixture",
        SimpleNamespace(__file__=str(tmp_path / "formula.py")),
    )
    with pytest.raises(EngineAdmissionError, match="ENGINE_RUNTIME_ORIGIN"):
        evaluation.engine_runtime_identity()
