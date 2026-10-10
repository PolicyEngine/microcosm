"""Reform-coverage probes stay in step with the installed PolicyEngine-US.

The 2026-09-27 Route A export failed its post-export smoke on four probes whose
definitions predated PolicyEngine-US 2.2.1, not on missing inputs:

- Three SNAP source-exclusion probes pinned a person-level unearned-source list
  that still named ``tanf``. 2.2.1 counts TANF once on ``unearned_spm_unit``, and
  a group variable named in a person-level ``adds`` list is projected onto every
  member, so the reform re-added TANF once per SPM-unit member and cut SNAP.
- The alimony-expense probe flipped ``divorce_year_threshold``, which since
  PolicyEngine-US a8e8a2e2ab also gates recipients' ``taxable_alimony_income``,
  so the "abolition" untaxed receipts and lowered income tax.

Probes no longer pin whole lists. Each list change is a declared
``ListParameterEdit`` (the items removed or added over a bounded period), and
``_build_reform`` resolves it against the installed engine's baseline list when
the reform is built. A later engine change to that list is kept rather than
reverted, and an edit that no longer fits the baseline (a removed item the
engine dropped, an added item it already carries, a baseline that changes
inside the period) raises instead of scoring a different reform.

These tests pin, against the installed engine:

- each list-edit probe declares exactly the one item it means to change, and
  resolves to the engine's baseline list with only that item changed;
- the reform ``_build_reform`` returns holds the resolved list inside its
  bounded period and the baseline outside it, survives the repeated application
  the engine's own constructors perform, and refuses a system whose list
  already differs from the baseline it was resolved against;
- edits that no longer fit the engine fail loudly, and the pure resolver agrees
  with the list the engine-applied reform holds (a differential property);
- each source/deduction probe moves only its own leaf on a household that also
  carries the neighbouring channels.
"""

from datetime import date, timedelta

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.us_runtime.reform_coverage_smoke import _build_reform
from microcosm.build.us_runtime.release_input_coverage import (
    ListParameterEdit,
    ReformCoverageProbe,
    resolve_list_parameter_edit,
    resolve_probe_parameter_changes,
    us_release_reform_coverage_probes,
)

# Every list-edit probe and the one item it changes relative to the installed
# engine's baseline list: removed, or added for a reactivation probe. These
# maps state the intent independently of the manifest's declared edits.
_REMOVED_ITEM = {
    "qbi_farm_operations_income_exclusion": "farm_operations_income",
    "qbi_farm_rent_income_exclusion": "farm_rent_income",
    "child_support_received_snap_exclusion": "child_support_received",
    "child_support_expense_snap_deduction_abolition": "snap_child_support_deduction",
    "disability_benefits_snap_exclusion": "disability_benefits",
    "workers_compensation_snap_exclusion": "workers_compensation",
    "educator_expense_ald_abolition": "educator_expense",
    "alimony_expense_ald_abolition": "alimony_expense_ald",
}
_ADDED_ITEM = {"domestic_production_ald_reactivation": "domestic_production_ald"}

_SNAP_UNEARNED = "gov.usda.snap.income.sources.unearned"
_SNAP_SOURCE_PROBES = {
    "child_support_received_snap_exclusion": "child_support_received",
    "disability_benefits_snap_exclusion": "disability_benefits",
    "workers_compensation_snap_exclusion": "workers_compensation",
}

# The list-valued parameters the shipped probes edit, plus the SPM-unit SNAP
# source list the TANF failure turned on.
_LIST_PARAMETERS = (
    "gov.irs.deductions.qbi.income_definition",
    "gov.irs.ald.deductions",
    _SNAP_UNEARNED,
    "gov.usda.snap.income.sources.unearned_spm_unit",
    "gov.usda.snap.income.deductions.allowed",
)


def _probes():
    return {probe.id: probe for probe in us_release_reform_coverage_probes()}


def _list_edits():
    return [
        (probe.id, path, edit)
        for probe in us_release_reform_coverage_probes()
        for path, edit in probe.list_edits.items()
    ]


def _one_list_edit_per_parameter():
    # The resolved-reform class is shared by every probe, so the constructor
    # paths below run once per edited parameter rather than once per probe.
    chosen = {}
    for probe_id, path, edit in sorted(_list_edits(), key=lambda item: item[0]):
        chosen.setdefault(path, (probe_id, path, edit))
    return sorted(chosen.values(), key=lambda item: item[0])


def _ids(cases):
    return [f"{probe_id}-{path}" for probe_id, path, _ in cases]


def _probe_with_edit(path: str, edit: ListParameterEdit) -> ReformCoverageProbe:
    return ReformCoverageProbe(
        id="list_edit_probe",
        name="A list edit under test",
        parameter_changes={},
        list_edits={path: edit},
        budget_measure="snap",
        binding_inputs=("employment_income",),
        min_abs_effect=1.0,
        reason="Exercises list-edit resolution against the installed engine.",
        issue="PolicyEngine/microcosm#1046",
    )


def _shifted(instant: str, days: int) -> str:
    return (date.fromisoformat(instant) + timedelta(days=days)).isoformat()


def _assert_edit_holds_only_inside_its_period(
    system, reformed, path: str, edit: ListParameterEdit, resolved: list
) -> None:
    edited = reformed.parameters.get_child(path)
    baseline = system.parameters.get_child(path)
    assert list(edited(edit.start)) == resolved
    assert list(edited(edit.stop)) == resolved
    for outside in (_shifted(edit.start, -1), _shifted(edit.stop, 1)):
        assert edited(outside) == baseline(outside), (
            f"{path} changed at {outside}, outside the edit period {edit.period}"
        )


@pytest.fixture(scope="module")
def system():
    from policyengine_us import CountryTaxBenefitSystem

    return CountryTaxBenefitSystem()


@pytest.fixture(scope="module")
def reformed_systems(system):
    # Each reform is applied once, to a clone of the baseline system's
    # parameters (``Reform.__init__``), which is far cheaper than building a
    # full reformed system per probe. The engine's own constructor path is
    # pinned separately below.
    probes = _probes()
    return {
        probe_id: _build_reform(probes[probe_id])(system)
        for probe_id in (*_SNAP_SOURCE_PROBES, "alimony_expense_ald_abolition")
    }


def test_every_list_edit_probe_declares_its_one_item() -> None:
    # Guards the parametrizations below against silently collecting nothing,
    # and makes a new list-edit probe state the item it means to change.
    probes = _probes()
    list_edit_probes = {probe_id for probe_id, *_ in _list_edits()}
    assert list_edit_probes == set(_REMOVED_ITEM) | set(_ADDED_ITEM)
    assert set(_SNAP_SOURCE_PROBES) <= set(_REMOVED_ITEM)
    for probe_id in sorted(list_edit_probes):
        probe = probes[probe_id]
        # The declared item is the probe's whole reform.
        assert probe.parameter_changes == {}, probe_id
        ((path, edit),) = probe.list_edits.items()
        expected = (
            ((), (_ADDED_ITEM[probe_id],))
            if probe_id in _ADDED_ITEM
            else ((_REMOVED_ITEM[probe_id],), ())
        )
        assert (edit.remove, edit.add) == expected, f"{probe_id}: {path}"
    for probe_id, item in _SNAP_SOURCE_PROBES.items():
        assert probes[probe_id].list_edits[_SNAP_UNEARNED].remove == (item,)


@pytest.mark.parametrize(
    ("probe_id", "path", "edit"), _list_edits(), ids=_ids(_list_edits())
)
def test_list_edit_resolves_to_engine_baseline_with_only_its_item_changed(
    system, probe_id: str, path: str, edit: ListParameterEdit
) -> None:
    resolved = resolve_probe_parameter_changes(_probes()[probe_id])[path][edit.period]
    baseline = list(system.parameters.get_child(path)(edit.start))

    assert len(resolved) == len(set(resolved)), f"{probe_id}: duplicate items"
    removed = set(baseline) - set(resolved)
    added = set(resolved) - set(baseline)
    expected = (
        (set(), {_ADDED_ITEM[probe_id]})
        if probe_id in _ADDED_ITEM
        else ({_REMOVED_ITEM[probe_id]}, set())
    )
    assert (removed, added) == expected, (
        f"{probe_id}: {path} must resolve to the installed engine's baseline "
        f"with only {expected} changed (removed, added); got removed "
        f"{sorted(removed)}, added {sorted(added)}."
    )
    # The kept items stay in the engine's order.
    assert [item for item in resolved if item in baseline] == [
        item for item in baseline if item in resolved
    ]


@pytest.mark.parametrize(
    ("probe_id", "path", "edit"), _list_edits(), ids=_ids(_list_edits())
)
def test_built_reform_holds_the_resolved_list_only_inside_its_period(
    system, probe_id: str, path: str, edit: ListParameterEdit
) -> None:
    probe = _probes()[probe_id]
    resolved = resolve_probe_parameter_changes(probe)[path][edit.period]
    baseline = list(system.parameters.get_child(path)(edit.start))
    reform = _build_reform(probe)

    assert reform.resolved_list_edits == {path: resolved}
    reformed = reform(system)
    _assert_edit_holds_only_inside_its_period(system, reformed, path, edit, resolved)
    # The reform edits a clone: the baseline system keeps its list.
    assert list(system.parameters.get_child(path)(edit.start)) == baseline


@pytest.mark.parametrize(
    ("probe_id", "path", "edit"),
    _one_list_edit_per_parameter(),
    ids=_ids(_one_list_edit_per_parameter()),
)
def test_list_edit_reform_is_idempotent(
    system, probe_id: str, path: str, edit: ListParameterEdit
) -> None:
    # Invariant: re-applying a probe's reform to a system it already reformed
    # changes nothing. The engine re-applies reforms as a matter of course (see
    # the constructor test below), so a reform that refuses its own resolved
    # list cannot be scored.
    reform = _build_reform(_probes()[probe_id])
    once = reform(system)
    twice = reform(once)

    _assert_edit_holds_only_inside_its_period(
        system, twice, path, edit, reform.resolved_list_edits[path]
    )


@pytest.mark.parametrize(
    ("probe_id", "path", "edit"),
    _one_list_edit_per_parameter(),
    ids=_ids(_one_list_edit_per_parameter()),
)
def test_list_edit_reform_builds_a_country_system(
    system, probe_id: str, path: str, edit: ListParameterEdit
) -> None:
    # The release's post-export scorer builds each reform's system with
    # ``Microsimulation.default_tax_benefit_system(reform=...)``, which is
    # ``CountryTaxBenefitSystem(reform=...)``. That constructor applies its
    # reform set twice (before and after backdating parameters), so the
    # reform has to accept the system it has already reformed.
    from policyengine_us import CountryTaxBenefitSystem

    reform = _build_reform(_probes()[probe_id])
    reformed = CountryTaxBenefitSystem(reform=(reform,))

    _assert_edit_holds_only_inside_its_period(
        system, reformed, path, edit, reform.resolved_list_edits[path]
    )


def test_list_edit_reform_refuses_a_list_changed_under_it(system) -> None:
    from policyengine_core.reforms import Reform
    from policyengine_us import CountryTaxBenefitSystem

    probe = _probes()["child_support_received_snap_exclusion"]
    edit = probe.list_edits[_SNAP_UNEARNED]
    baseline = list(system.parameters.get_child(_SNAP_UNEARNED)(edit.start))
    readds_tanf = Reform.from_dict(
        {_SNAP_UNEARNED: {edit.period: [*baseline, "tanf"]}}, country_id="us"
    )
    reform = _build_reform(probe)
    refusal = rf"'{probe.id}': {_SNAP_UNEARNED} was resolved against"

    # Applied over a system whose list already re-adds TANF, the resolved list
    # would silently drop that change, so the reform refuses to apply.
    with pytest.raises(ValueError, match=refusal) as refused:
        reform(readds_tanf(system))
    assert "'tanf']" in str(refused.value)
    # The same refusal through the engine's reform tuple.
    with pytest.raises(ValueError, match=refusal):
        CountryTaxBenefitSystem(reform=(readds_tanf, reform))


def test_unchanged_list_at_a_breakpoint_does_not_split_an_edit(system) -> None:
    # The QBI income definition has a 2018-01-01 breakpoint that restates the
    # same list, so an edit spanning it still has one baseline to edit.
    path = "gov.irs.deductions.qbi.income_definition"
    parameter = system.parameters.get_child(path)
    assert "2018-01-01" in {str(value.instant_str) for value in parameter.values_list}
    assert parameter("2017-01-01") == parameter("2018-01-01")
    edit = ListParameterEdit("2017-01-01.2018-12-31", remove=("farm_rent_income",))

    resolved = resolve_probe_parameter_changes(_probe_with_edit(path, edit))
    assert resolved[path][edit.period] == [
        item for item in parameter("2017-01-01") if item != "farm_rent_income"
    ]


@pytest.mark.parametrize(
    ("path", "edit", "failure"),
    [
        (
            _SNAP_UNEARNED,
            ListParameterEdit("2024-01-01.2024-12-31", remove=("not_a_snap_source",)),
            r"removes \['not_a_snap_source'\], which the baseline lacks",
        ),
        (
            _SNAP_UNEARNED,
            ListParameterEdit("2024-01-01.2024-12-31", add=("ssi",)),
            r"adds \['ssi'\], which the baseline already has",
        ),
        (
            "gov.irs.ald.deductions",
            ListParameterEdit("2017-01-01.2018-12-31", remove=("educator_expense",)),
            r"the list changes at 2018-01-01",
        ),
        (
            "gov.irs.deductions.standard.amount.SINGLE",
            ListParameterEdit("2024-01-01.2024-12-31", remove=("single",)),
            r"is not a list",
        ),
        (
            "gov.usda.snap.income.sources.not_a_parameter",
            ListParameterEdit("2024-01-01.2024-12-31", remove=("ssi",)),
            r"not a parameter of the engine",
        ),
        (
            "gov.usda.snap.income.sources",
            ListParameterEdit("2024-01-01.2024-12-31", remove=("ssi",)),
            r"not a leaf parameter",
        ),
    ],
    ids=[
        "removed-item-missing",
        "added-item-present",
        "baseline-changes-inside-period",
        "scalar-parameter",
        "unknown-path",
        "parameter-node",
    ],
)
def test_list_edit_that_does_not_fit_the_engine_fails_loudly(
    path: str, edit: ListParameterEdit, failure: str
) -> None:
    probe = _probe_with_edit(path, edit)

    with pytest.raises(
        ValueError, match=rf"^reform-coverage probe '{probe.id}': .*{failure}"
    ):
        resolve_probe_parameter_changes(probe)
    with pytest.raises(ValueError, match=failure):
        _build_reform(probe)


def _constant_years(system, path: str) -> list[int]:
    # Years whose list holds one non-empty value all year: no breakpoint after
    # 1 January and on or before 31 December.
    parameter = system.parameters.get_child(path)
    breakpoints = {str(value.instant_str) for value in parameter.values_list}
    return [
        year
        for year in range(2015, 2031)
        if parameter(f"{year}-01-01")
        and not any(
            f"{year}-01-01" < instant <= f"{year}-12-31" for instant in breakpoints
        )
    ]


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
def test_resolver_matches_the_list_the_engine_applied_reform_holds(
    system, data
) -> None:
    # Invariant (differential): for any removal subset of a real baseline list
    # and any fresh additions, the list the pure resolver returns is the list
    # the engine-applied reform holds at every instant of the edit period, and
    # outside the period the reformed system keeps the engine's baseline.
    path = data.draw(st.sampled_from(_LIST_PARAMETERS), label="path")
    year = data.draw(st.sampled_from(_constant_years(system, path)), label="year")
    period = f"{year}-01-01.{year}-12-31"
    baseline = list(system.parameters.get_child(path)(f"{year}-01-01"))
    remove = data.draw(
        st.lists(st.sampled_from(baseline), min_size=1, unique=True), label="remove"
    )
    add = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=99).map(lambda n: f"probe_item_{n}"),
            max_size=2,
            unique=True,
        ),
        label="add",
    )
    day = data.draw(st.integers(min_value=0, max_value=364), label="day")
    edit = ListParameterEdit(period, remove=tuple(remove), add=tuple(add))

    expected = resolve_list_parameter_edit(path, edit, baseline)
    reformed = _build_reform(_probe_with_edit(path, edit))(system)

    instant = _shifted(edit.start, day)
    assert list(reformed.parameters.get_child(path)(instant)) == expected
    _assert_edit_holds_only_inside_its_period(system, reformed, path, edit, expected)


@pytest.mark.parametrize("probe_id", sorted(_SNAP_SOURCE_PROBES))
def test_snap_source_probe_does_not_readd_spm_unit_sources(
    system, reformed_systems, probe_id: str
) -> None:
    edit = _probes()[probe_id].list_edits[_SNAP_UNEARNED]
    spm_unit_sources = system.parameters.gov.usda.snap.income.sources.unearned_spm_unit(
        edit.start
    )
    scored = reformed_systems[probe_id].parameters.get_child(_SNAP_UNEARNED)

    assert set(spm_unit_sources), "the engine counts no SPM-unit unearned sources"
    for instant in (edit.start, edit.stop):
        assert not set(scored(instant)) & set(spm_unit_sources), instant


def _snap_household(leaf: str, amount: float, tanf: float) -> dict:
    members = ["adult", "child1", "child2"]
    return {
        "people": {
            "adult": {
                "age": {"2024": 35},
                "employment_income": {"2024": 8_000.0},
                leaf: {"2024": amount},
            },
            "child1": {"age": {"2024": 8}},
            "child2": {"age": {"2024": 5}},
        },
        "tax_units": {
            "tax_unit": {
                "members": members,
                "filing_status": {"2024": "HEAD_OF_HOUSEHOLD"},
            }
        },
        "spm_units": {"spm_unit": {"members": members, "tanf": {"2024": tanf}}},
        "households": {"household": {"members": members, "state_code": {"2024": "TX"}}},
    }


@pytest.mark.parametrize("probe_id", sorted(_SNAP_SOURCE_PROBES))
def test_snap_source_probe_raises_snap_when_the_unit_also_gets_tanf(
    system, reformed_systems, probe_id: str
) -> None:
    from policyengine_us import Simulation

    situation = _snap_household(_SNAP_SOURCE_PROBES[probe_id], 3_000.0, 4_800.0)
    baseline = Simulation(tax_benefit_system=system, situation=situation)
    reformed = Simulation(
        tax_benefit_system=reformed_systems[probe_id], situation=situation
    )

    assert baseline.calculate("snap_unearned_income", 2024)[0] == pytest.approx(7_800.0)
    assert reformed.calculate("snap_unearned_income", 2024)[0] == pytest.approx(4_800.0)
    # SNAP phases out at 30 cents per dollar of net income: excluding $3,000
    # a year raises the benefit by $900 while the unit stays in the phase-out.
    effect = reformed.calculate("snap", 2024)[0] - baseline.calculate("snap", 2024)[0]
    assert effect == pytest.approx(900.0)


@pytest.mark.parametrize("probe_id", sorted(_SNAP_SOURCE_PROBES))
@settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    leaf_amount=st.integers(min_value=0, max_value=24_000),
    tanf=st.integers(min_value=0, max_value=12_000),
)
def test_snap_source_probe_removes_exactly_its_leaf(
    system, reformed_systems, probe_id: str, leaf_amount: int, tanf: int
) -> None:
    # Invariant: for any leaf and TANF amount, the reform lowers SNAP countable
    # unearned income by exactly the leaf and never lowers SNAP.
    from policyengine_us import Simulation

    situation = _snap_household(
        _SNAP_SOURCE_PROBES[probe_id], float(leaf_amount), float(tanf)
    )
    baseline = Simulation(tax_benefit_system=system, situation=situation)
    reformed = Simulation(
        tax_benefit_system=reformed_systems[probe_id], situation=situation
    )

    drop = (
        baseline.calculate("snap_unearned_income", 2024)[0]
        - reformed.calculate("snap_unearned_income", 2024)[0]
    )
    assert drop == pytest.approx(float(leaf_amount), abs=0.01)
    assert (
        reformed.calculate("snap", 2024)[0]
        >= baseline.calculate("snap", 2024)[0] - 0.01
    )


def _tax_household(alimony_expense: float, alimony_income: float) -> dict:
    return {
        "people": {
            "adult": {
                "age": {"2024": 45},
                "employment_income": {"2024": 90_000.0},
                "alimony_expense": {"2024": alimony_expense},
                "alimony_income": {"2024": alimony_income},
            }
        },
        "tax_units": {
            "tax_unit": {"members": ["adult"], "filing_status": {"2024": "SINGLE"}}
        },
        "spm_units": {"spm_unit": {"members": ["adult"]}},
        "households": {
            "household": {"members": ["adult"], "state_code": {"2024": "TX"}}
        },
    }


def _income_tax_change(system, reformed_systems, situation: dict) -> float:
    from policyengine_us import Simulation

    baseline = Simulation(tax_benefit_system=system, situation=situation)
    reformed = Simulation(
        tax_benefit_system=reformed_systems["alimony_expense_ald_abolition"],
        situation=situation,
    )
    return float(
        baseline.calculate("income_tax", 2024)[0]
        - reformed.calculate("income_tax", 2024)[0]
    )


def test_alimony_probe_taxes_payers_and_leaves_recipients_alone(
    system, reformed_systems
) -> None:
    payer = _income_tax_change(system, reformed_systems, _tax_household(20_000.0, 0.0))
    recipient = _income_tax_change(
        system, reformed_systems, _tax_household(0.0, 20_000.0)
    )

    # baseline_minus_reform, as the probe scores it: negative for payers.
    assert payer < -1_000.0
    assert recipient == 0.0


@settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    alimony_expense=st.integers(min_value=0, max_value=40_000),
    alimony_income=st.integers(min_value=0, max_value=40_000),
)
def test_alimony_probe_raises_agi_by_exactly_the_payer_deduction(
    system, reformed_systems, alimony_expense: int, alimony_income: int
) -> None:
    # Invariant: the reform adds back exactly the baseline alimony ALD and
    # leaves the recipient side of AGI untouched.
    from policyengine_us import Simulation

    situation = _tax_household(float(alimony_expense), float(alimony_income))
    baseline = Simulation(tax_benefit_system=system, situation=situation)
    reformed = Simulation(
        tax_benefit_system=reformed_systems["alimony_expense_ald_abolition"],
        situation=situation,
    )

    agi_change = (
        reformed.calculate("adjusted_gross_income", 2024)[0]
        - baseline.calculate("adjusted_gross_income", 2024)[0]
    )
    assert agi_change == pytest.approx(
        baseline.calculate("alimony_expense_ald", 2024)[0], abs=0.01
    )
