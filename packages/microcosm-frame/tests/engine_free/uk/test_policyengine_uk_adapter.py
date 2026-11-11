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


def _person_role():
    return SimpleNamespace(
        entity=SimpleNamespace(key="person"), value_type=bool, definition_period="year"
    )


_ROLES = ("is_uc_claimant", "is_claimant_or_partner", "is_hbai_dependent_child")
_SUPPLIED = {
    "is_uc_claimant": np.array([True, False]),
    "is_claimant_or_partner": np.array([True, False]),
    "is_hbai_dependent_child": np.array([False, True]),
}


def _role_engine(monkeypatch, variables, *, loaded, values):
    calls = []

    def calculate(name, *args, **kwargs):
        calls.append(name)
        return np.asarray(values[name]) if name in values else np.zeros(2)

    simulation = SimpleNamespace(
        tax_benefit_system=SimpleNamespace(variables=variables),
        input_variables=loaded,
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
        table=lambda entity: pd.DataFrame(_SUPPLIED), n=lambda entity: 2
    )
    return engine, bundle, calls


_OUTPUT = SimpleNamespace(entity=SimpleNamespace(key="person"), value_type=float)


def test_the_validated_roles_are_the_supplied_person_types() -> None:
    assert tuple(name for name, _ in SUPPLIED_ROLE_INPUTS) == _ROLES


@pytest.mark.parametrize("role", ["is_claimant_or_partner", "is_hbai_dependent_child"])
@pytest.mark.parametrize(
    "defect", ["missing", "not_loaded", "changed", "entity", "dtype", "period"]
)
def test_adapter_refuses_unconsumed_person_type_roles(monkeypatch, role, defect):
    # policyengine-uk 2.107.0 reads both person types across its programmes;
    # a supplied role must reach the model as an input, unchanged.
    variables = {name: _person_role() for name in _ROLES} | {"income_tax": _OUTPUT}
    if defect == "missing":
        del variables[role]
    elif defect == "entity":
        variables[role].entity.key = "benunit"
    elif defect == "dtype":
        variables[role].value_type = float
    elif defect == "period":
        variables[role].definition_period = "month"
    loaded = [name for name in _ROLES if not (defect == "not_loaded" and name == role)]
    values = dict(_SUPPLIED)
    if defect == "changed":
        values[role] = ~values[role]
    engine, bundle, calls = _role_engine(
        monkeypatch, variables, loaded=loaded, values=values
    )
    with pytest.raises(ValueError, match=role):
        engine.materialize(bundle, ["income_tax"], 2025)
    assert "income_tax" not in calls


def test_adapter_accepts_person_type_roles_the_model_reads(monkeypatch):
    variables = {name: _person_role() for name in _ROLES} | {"income_tax": _OUTPUT}
    engine, bundle, calls = _role_engine(
        monkeypatch, variables, loaded=list(_ROLES), values=_SUPPLIED
    )
    engine.materialize(bundle, ["income_tax"], 2025)
    assert calls == [*_ROLES, "income_tax"]
