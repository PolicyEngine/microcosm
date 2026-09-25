"""Tests split from packages/microcosm-frame/tests/test_policyengine_uk_adapter.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_frame.policyengine_uk_adapter import *


def test_policyengine_uk_adapter_satisfies_rules_protocol_without_importing_engine() -> (
    None
):
    adapter = PolicyEngineUKEngine()

    assert isinstance(adapter, RulesEngine)
    assert adapter.country == "uk"
    assert adapter.entity_schema() == UK_SCHEMA


def test_policyengine_uk_adapter_export_side_is_not_implemented() -> None:
    adapter = PolicyEngineUKEngine()

    with pytest.raises(NotImplementedError, match="write_uk_national_frame"):
        adapter.write_dataset(object(), "unused.h5", period=2023)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "defect", ["missing", "not_loaded", "changed", "entity", "dtype", "period"]
)
def test_adapter_refuses_unconsumed_explicit_uc_roles(monkeypatch, defect):
    definition = SimpleNamespace(
        entity=SimpleNamespace(key="person"), value_type=bool, definition_period="year"
    )
    variables = {
        "is_uc_claimant": definition,
        "universal_credit": SimpleNamespace(
            entity=SimpleNamespace(key="benunit"), value_type=float
        ),
    }
    if defect == "missing":
        del variables["is_uc_claimant"]
    elif defect == "entity":
        definition.entity.key = "benunit"
    elif defect == "dtype":
        definition.value_type = float
    elif defect == "period":
        definition.definition_period = "month"
    calls = []

    def calculate(name, *args, **kwargs):
        calls.append(name)
        return np.array([False if defect == "changed" else True])

    simulation = SimpleNamespace(
        tax_benefit_system=SimpleNamespace(variables=variables),
        input_variables=[] if defect == "not_loaded" else ["is_uc_claimant"],
        calculate=calculate,
    )
    engine = PolicyEngineUKEngine()
    engine._system = simulation.tax_benefit_system
    monkeypatch.setattr(engine, "_build_dataset", lambda *args: object())
    monkeypatch.setattr(
        engine,
        "_import_policyengine_uk",
        lambda: SimpleNamespace(Microsimulation=lambda **kwargs: simulation),
    )
    bundle = SimpleNamespace(
        table=lambda entity: pd.DataFrame({"is_uc_claimant": [True]}),
        n=lambda entity: 1,
    )
    with pytest.raises(ValueError, match="is_uc_claimant"):
        engine.materialize(bundle, ["universal_credit"], 2025)
    assert "universal_credit" not in calls
