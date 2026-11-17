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
  the engine's own constructors perform (``CountryTaxBenefitSystem`` and
  ``Simulation``), extends an earlier application of the same edit over an
  overlapping period, and refuses a system where any stretch of the period
  already differs from the baseline it was resolved against;
- an edit that starts on or before a list's first breakpoint at the
  constructor's first reform pass is refused, at resolution and at apply time,
  because the engine backdates that first value after applying reforms;
- edits that no longer fit the engine fail loudly, and the pure resolver agrees
  with the list the engine-applied reform holds (a differential property);
- each source/deduction probe moves only its own leaf on a household that also
  carries the neighbouring channels.
"""

from datetime import date, timedelta

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import microcosm.build.us_runtime.reform_coverage_smoke as smoke_module
import microcosm.build.us_runtime.release_input_coverage as coverage_module
from microcosm.build.us_runtime.reform_coverage_smoke import _build_reform
from microcosm.build.us_runtime.release_input_coverage import (
    ListParameterEdit,
    ReformCoverageProbe,
    list_parameter_baseline,
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
    "niit_passive_pass_through_exclusion": "passive_partnership_s_corp_income",
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
    "gov.irs.investment.income.sources",
)
_QBI = "gov.irs.deductions.qbi.income_definition"
_ALD = "gov.irs.ald.deductions"


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


def _probe_with_edit(
    path: str, edit: ListParameterEdit, probe_id: str = "list_edit_probe"
) -> ReformCoverageProbe:
    return ReformCoverageProbe(
        id=probe_id,
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


@pytest.mark.parametrize(
    ("probe_id", "path", "edit"),
    _one_list_edit_per_parameter(),
    ids=_ids(_one_list_edit_per_parameter()),
)
def test_list_edit_reform_builds_a_simulation(
    system, probe_id: str, path: str, edit: ListParameterEdit
) -> None:
    # Simulation(reform=...) builds CountryTaxBenefitSystem(reform=...) (two
    # applications) and then applies the reform set once more.
    from policyengine_us import Simulation

    reform = _build_reform(_probes()[probe_id])
    simulation = Simulation(
        situation=_snap_household("child_support_received", 0.0, 0.0), reform=reform
    )

    _assert_edit_holds_only_inside_its_period(
        system,
        simulation.tax_benefit_system,
        path,
        edit,
        reform.resolved_list_edits[path],
    )


def test_overlapping_applications_of_one_edit_cover_the_union(system) -> None:
    # Sol review of microcosm#1197, finding 1: each stretch of the period is
    # checked on its own, so an earlier application of the same edit over an
    # overlapping period is accepted (in either order) and the two together
    # hold the resolved list over the union of their periods.
    from policyengine_us import CountryTaxBenefitSystem

    item = "child_support_received"
    first = _build_reform(
        _probe_with_edit(
            _SNAP_UNEARNED,
            ListParameterEdit("2024-01-01.2024-06-30", remove=(item,)),
            "first_half",
        )
    )
    second = _build_reform(
        _probe_with_edit(
            _SNAP_UNEARNED,
            ListParameterEdit("2024-04-01.2024-12-31", remove=(item,)),
            "april_on",
        )
    )
    union = ListParameterEdit("2024-01-01.2024-12-31", remove=(item,))
    resolved = first.resolved_list_edits[_SNAP_UNEARNED]
    assert resolved == second.resolved_list_edits[_SNAP_UNEARNED]
    assert item not in resolved

    for reformed in (
        second(first(system)),
        first(second(system)),
        CountryTaxBenefitSystem(reform=(first, second)),
    ):
        _assert_edit_holds_only_inside_its_period(
            system, reformed, _SNAP_UNEARNED, union, resolved
        )
        edited = reformed.parameters.get_child(_SNAP_UNEARNED)
        for day in ("2024-03-31", "2024-04-01", "2024-06-30", "2024-07-01"):
            assert list(edited(day)) == resolved, day


def test_overlapping_different_edit_is_refused_at_the_first_stretch_it_changed(
    system,
) -> None:
    first = _build_reform(
        _probe_with_edit(
            _SNAP_UNEARNED,
            ListParameterEdit(
                "2024-01-01.2024-06-30", remove=("workers_compensation",)
            ),
            "first_half",
        )
    )
    second = _build_reform(
        _probe_with_edit(
            _SNAP_UNEARNED,
            ListParameterEdit(
                "2024-04-01.2024-12-31", remove=("child_support_received",)
            ),
            "april_on",
        )
    )

    # From April to June the list already lacks workers_compensation, which
    # the second edit's resolution would put back.
    with pytest.raises(
        ValueError,
        match=r"^reform-coverage probe 'april_on': .* from 2024-04-01, inside "
        r"2024-04-01\.2024-12-31; the resolved list would revert",
    ):
        second(first(system))


@pytest.mark.parametrize(
    "period",
    ["2015-01-01.2015-12-31", "2017-01-01.2017-12-31", "2018-01-01.2018-12-31"],
)
def test_edit_reaching_a_backdated_first_value_is_refused(period: str) -> None:
    # Sol review of microcosm#1197, finding 2: CountryTaxBenefitSystem applies
    # reforms before it backdates each parameter's first value (to 2015 in
    # PolicyEngine-US 2.2.1). The QBI income definition first appears on
    # 2018-01-01 at that first pass, so an edit starting on or before it would
    # become the first value and be backdated outside its period. Resolution
    # refuses it before any system is built.
    assert coverage_module._installed_parameter_stages()[0][_QBI] == "2018-01-01"
    probe = _probe_with_edit(
        _QBI, ListParameterEdit(period, remove=("farm_rent_income",))
    )
    refusal = (
        r"starts on or before 2018-01-01, the list's first breakpoint when "
        r"the installed engine first applies reforms"
    )

    with pytest.raises(
        ValueError, match=rf"^reform-coverage probe '{probe.id}': .*{refusal}"
    ):
        resolve_probe_parameter_changes(probe)
    with pytest.raises(ValueError, match=refusal):
        _build_reform(probe)


def test_constructors_refuse_an_edit_that_reaches_the_first_value(
    system, monkeypatch
) -> None:
    # The apply-time check holds even when a resolver reads only the finished
    # tree, where 2017 is defined (backdated): the reform applies to the
    # finished system, but both constructors refuse it at their first pass
    # rather than let backdating carry the edited list back to 2015.
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    monkeypatch.setattr(
        smoke_module,
        "installed_list_parameter_baseline",
        lambda path, edit: list_parameter_baseline(system.parameters, path, edit),
    )
    edit = ListParameterEdit("2017-01-01.2017-12-31", remove=("farm_rent_income",))
    reform = _build_reform(_probe_with_edit(_QBI, edit))
    _assert_edit_holds_only_inside_its_period(
        system, reform(system), _QBI, edit, reform.resolved_list_edits[_QBI]
    )

    refusal = (
        r"starts on or before 2018-01-01, the list's first breakpoint in the "
        r"tree it is read from"
    )
    with pytest.raises(ValueError, match=refusal):
        CountryTaxBenefitSystem(reform=(reform,))
    with pytest.raises(ValueError, match=refusal):
        Simulation(
            situation=_snap_household("child_support_received", 0.0, 0.0),
            reform=reform,
        )


def test_edit_after_a_backdated_first_value_holds_only_inside_its_period(
    system,
) -> None:
    # Past the first breakpoint both constructors apply the edit to the same
    # baseline it resolved on, and the backdated years keep that baseline.
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    edit = ListParameterEdit("2019-01-01.2019-12-31", remove=("farm_rent_income",))
    reform = _build_reform(_probe_with_edit(_QBI, edit))
    resolved = reform.resolved_list_edits[_QBI]
    simulation = Simulation(
        situation=_snap_household("child_support_received", 0.0, 0.0), reform=reform
    )
    baseline = system.parameters.get_child(_QBI)

    for reformed in (
        CountryTaxBenefitSystem(reform=(reform,)),
        simulation.tax_benefit_system,
    ):
        _assert_edit_holds_only_inside_its_period(
            system, reformed, _QBI, edit, resolved
        )
        edited = reformed.parameters.get_child(_QBI)
        for day in ("2015-01-01", "2016-06-01", "2018-01-01", "2018-12-31"):
            assert edited(day) == baseline(day), day


def test_unchanged_list_at_a_breakpoint_does_not_split_an_edit(system) -> None:
    # The above-the-line deduction list has a 2026-01-01 breakpoint that
    # restates the 2021 list, so an edit spanning it still has one baseline.
    parameter = system.parameters.get_child(_ALD)
    assert "2026-01-01" in {str(value.instant_str) for value in parameter.values_list}
    assert parameter("2025-01-01") == parameter("2026-01-01")
    edit = ListParameterEdit("2025-01-01.2026-12-31", remove=("educator_expense",))

    resolved = resolve_probe_parameter_changes(_probe_with_edit(_ALD, edit))
    assert resolved[_ALD][edit.period] == [
        item for item in parameter("2025-01-01") if item != "educator_expense"
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
    # Years whose list holds one non-empty value all year (no breakpoint after
    # 1 January and on or before 31 December) and that start after the list's
    # first breakpoint at the constructor's first reform pass, so an edit over
    # the year can be scored.
    parameter = system.parameters.get_child(path)
    breakpoints = {str(value.instant_str) for value in parameter.values_list}
    first = coverage_module._installed_parameter_stages()[0][path]
    return [
        year
        for year in range(2015, 2031)
        if f"{year}-01-01" > first
        and parameter(f"{year}-01-01")
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
