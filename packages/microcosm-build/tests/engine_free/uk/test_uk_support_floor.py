"""The zero-support floor (microcosm#1095 U3; María's rule of 2026-10-21)."""

from __future__ import annotations

import json
from datetime import date
from importlib import resources as importlib_resources

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.measure_simulation import (
    load_uk_calibration_measure_exclusions,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.support_floor import (
    UK_SUPPORT_FLOOR_KIND,
    apply_uk_support_floor,
    load_uk_support_floor_declaration,
    narrow_uk_registry_by_support_floor,
    uk_household_source_ids,
    uk_row_support,
    uk_support_floor_from_payload,
    uk_support_floor_names,
)
from microcosm.calibrate import TargetRegistry, TargetSpec, build_constraint_matrix

SCOPE = "dwp.uc.payment_distribution_single"
SCOPE_NAME = "uc_bands"
EVALUATED = date(2026, 10, 20)


def _scope(**overrides):
    return {
        "scope": SCOPE_NAME,
        "unit": "benefit_units",
        "contract_target_ids": [SCOPE],
        "max_dropped_rows": 2,
        "max_dropped_published_share": 0.5,
        **overrides,
    }


def _block(scope=None, **overrides):
    return {
        "metric": "distinct_source_households_with_positive_prior_weight",
        "floor": 1,
        "scopes": [_scope(**(scope or {}))],
        "reason": "Sparse bands are dropped, not pooled.",
        "tracking": "microcosm#1095",
        "approved_by": "juaristi22",
        "adjudication": "microcosm#1095 U3",
        "approved_on": "2026-10-16",
        "expires_on": "2027-01-08",
        **overrides,
    }


def _frame():
    # Six households from four source households (two are clones), one with a
    # zero prior weight.
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4, 5, 6],
            "region": ["LONDON"] * 6,
            "household_source_id": [10, 10, 20, 30, 40, 40],
            "band_low": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            "band_thin": [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "band_zero_weight": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            "other_thin": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
    )
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2, 3, 4, 5, 6],
                "person_benunit_id": [1, 2, 3, 4, 5, 6],
                "person_household_id": [1, 2, 3, 4, 5, 6],
                "age": [30] * 6,
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1, 2, 3, 4, 5, 6]}),
        household=household,
        household_weights=np.array([10.0, 10.0, 10.0, 10.0, 0.0, 0.0]),
        time_period="2025",
    )


def _spec(name, measure, value, target=SCOPE):
    return TargetSpec(
        name=name,
        entity="household",
        value=value,
        measure=measure,
        period=2025,
        source="test",
        family="dwp_universal_credit",
        metadata={"contract_target_id": target},
    )


def _registry(*specs):
    return TargetRegistry(list(specs), country="uk")


def _compiled(registry):
    problem = build_constraint_matrix(_frame(), registry.to_target_set(), "household")
    assert not problem.skipped
    return problem


def _apply(registry, **kwargs):
    floor = uk_support_floor_from_payload(_block(**kwargs.pop("block", {})))
    return apply_uk_support_floor(
        registry,
        _compiled(registry),
        source_ids=uk_household_source_ids(_frame()),
        floor=floor,
        evaluated_on=kwargs.pop("evaluated_on", EVALUATED),
        **kwargs,
    )


def test_support_counts_distinct_source_households_with_a_positive_prior():
    frame = _frame()
    registry = _registry(
        _spec("low", "band_low", 40.0),
        _spec("thin", "band_thin", 30.0),
        _spec("zero_weight", "band_zero_weight", 5.0),
    )
    problem = _compiled(registry)
    support = uk_row_support(
        problem.matrix,
        problem.initial_weights.values,
        uk_household_source_ids(frame),
    )
    # Clones of one source household count once; a zero prior counts never.
    assert support.tolist() == [3, 1, 0]


def test_the_floor_drops_only_rows_no_household_carries():
    """A row on one source household stays bound; a row on none leaves."""

    registry = _registry(
        _spec("low", "band_low", 40.0),
        _spec("thin", "band_thin", 30.0),
        _spec("zero_weight", "band_zero_weight", 5.0),
        _spec("other", "other_thin", 7.0, target="dwp.uc.households"),
    )
    result = _apply(registry)

    assert [spec.name for spec in result.registry.specs] == ["low", "thin", "other"]
    assert result.problem.names == ("low@2025", "thin@2025", "other@2025")
    assert result.problem.target_vector.tolist() == [40.0, 30.0, 7.0]
    assert result.problem.matrix.shape == (3, 6)
    assert set(result.exclusions) == {"zero_weight"}
    assert result.exclusions["zero_weight"]["kind"] == UK_SUPPORT_FLOOR_KIND
    assert result.exclusions["zero_weight"]["evaluated_on"] == "2026-10-20"
    receipt = result.receipt
    assert receipt["dropped"] == ["zero_weight"]
    assert receipt["dropped_by_target"] == {SCOPE: {"rows": 1, "published_total": 5.0}}
    assert receipt["scopes"] == {
        SCOPE_NAME: {
            "unit": "benefit_units",
            "in_scope_rows": 3,
            "dropped_rows": 1,
            "dropped_published_share": pytest.approx(5.0 / 75.0),
        }
    }
    assert receipt["solved_register_sha256"] == result.registry.version
    # The receipt carries no row's support: support counts stay local.
    assert set(receipt) == {
        "metric",
        "floor",
        "tracking",
        "approved_on",
        "expires_on",
        "evaluated_on",
        "in_scope_rows",
        "probed_row_exclusions",
        "dropped_rows",
        "scopes",
        "dropped_by_target",
        "dropped",
        "solved_register_sha256",
    }


def test_the_floor_refuses_outside_its_window_and_past_its_ceilings():
    registry = _registry(
        _spec("low", "band_low", 40.0),
        _spec("thin", "band_thin", 30.0),
        _spec("zero_weight", "band_zero_weight", 5.0),
        _spec("zero_weight_too", "band_zero_weight", 5.0),
    )
    with pytest.raises(ValueError, match="outside its reviewed window"):
        _apply(registry, evaluated_on=date(2026, 10, 15))
    with pytest.raises(ValueError, match="outside its reviewed window"):
        _apply(registry, evaluated_on=date(2027, 1, 9))
    with pytest.raises(ValueError, match="drops 2 rows in scope 'uc_bands'"):
        _apply(registry, block={"scope": {"max_dropped_rows": 1}})
    with pytest.raises(ValueError, match="published total"):
        _apply(registry, block={"scope": {"max_dropped_published_share": 0.1}})


def test_each_scope_meets_its_own_ceilings():
    """A share sums published values, so it is taken within one unit's scope."""

    other = "hmrc.spi.dividend_income.amount_by_total_income_band"
    registry = _registry(
        _spec("low", "band_low", 40.0),
        _spec("zero_weight", "band_zero_weight", 5.0),
        _spec("pounds", "band_low", 9_000.0, target=other),
        _spec("pounds_none", "band_zero_weight", 1_000.0, target=other),
    )
    pounds = {
        "scope": "spi_amounts",
        "unit": "GBP",
        "contract_target_ids": [other],
        "max_dropped_rows": 1,
        "max_dropped_published_share": 0.2,
    }
    floor = uk_support_floor_from_payload({**_block(), "scopes": [_scope(), pounds]})
    result = apply_uk_support_floor(
        registry,
        _compiled(registry),
        source_ids=uk_household_source_ids(_frame()),
        floor=floor,
        evaluated_on=EVALUATED,
    )
    assert result.receipt["dropped"] == ["pounds_none", "zero_weight"]
    shares = {
        name: scope["dropped_published_share"]
        for name, scope in result.receipt["scopes"].items()
    }
    # Pooled, the two would read 1,005 of 10,045; each scope reads its own.
    assert shares == {
        SCOPE_NAME: pytest.approx(5.0 / 45.0),
        "spi_amounts": pytest.approx(0.1),
    }
    tight = {**pounds, "max_dropped_published_share": 0.05}
    with pytest.raises(ValueError, match="scope 'spi_amounts'"):
        apply_uk_support_floor(
            registry,
            _compiled(registry),
            source_ids=uk_household_source_ids(_frame()),
            floor=uk_support_floor_from_payload(
                {**_block(), "scopes": [_scope(), tight]}
            ),
            evaluated_on=EVALUATED,
        )


def test_a_per_row_exclusion_the_floor_would_cover_refuses():
    registry = _registry(
        _spec("low", "band_low", 40.0),
        _spec("zero_weight", "band_zero_weight", 5.0),
    )
    with pytest.raises(ValueError, match=r"retire those entries .*\['zero_weight'\]"):
        _apply(registry, row_excluded={"zero_weight"})
    # A per-row entry on a row some household carries, however few, is
    # measured and leaves the problem.
    result = _apply(
        _registry(_spec("low", "band_low", 40.0), _spec("thin", "band_thin", 30.0)),
        row_excluded={"thin"},
    )
    assert [spec.name for spec in result.registry.specs] == ["low"]
    assert result.exclusions == {}
    assert result.receipt["probed_row_exclusions"] == 1
    with pytest.raises(ValueError, match="name no compiled row"):
        _apply(registry, row_excluded={"absent"})


def test_lineage_is_read_only_when_a_row_in_scope_needs_it():
    registry = _registry(_spec("other", "other_thin", 7.0, target="dwp.uc.households"))
    result = apply_uk_support_floor(
        registry,
        _compiled(registry),
        source_ids=lambda: pytest.fail("lineage read with nothing in scope"),
        floor=uk_support_floor_from_payload(_block()),
        evaluated_on=EVALUATED,
    )
    assert result.registry is registry
    assert result.receipt["dropped_rows"] == 0
    frame = _frame()
    no_lineage = uk_national_frame(
        person=frame.table("person"),
        benunit=frame.table("benunit"),
        household=frame.table("household").drop(columns=["household_source_id"]),
        household_weights=frame.weights_for("household").values,
        time_period="2025",
    )
    with pytest.raises(ValueError, match="household_source_id"):
        uk_household_source_ids(no_lineage)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"metric": "rows"}, "metric"),
        ({"floor": 0}, "must be 1"),
        ({"floor": 3}, "must be 1"),
        ({"floor": 10}, "must be 1"),
        ({"floor": True}, "must be 1"),
        ({"scope": {"max_dropped_rows": -1}}, "count"),
        ({"scope": {"max_dropped_published_share": 1.0}}, r"\[0, 1\)"),
        ({"scope": {"contract_target_ids": [SCOPE, SCOPE]}}, "distinct"),
        ({"scope": {"unit": ""}}, "unit is empty"),
        ({"scope": {"extra": 1}}, "closed-world"),
        ({"scopes": []}, "non-empty"),
        ({"scopes": [_scope(), _scope(contract_target_ids=["other"])]}, "distinct"),
        ({"scopes": [_scope(), _scope(scope="again")]}, "disjoint"),
        ({"approved_by": " "}, "approved_by is empty"),
        ({"expires_on": "2027-1-8"}, "canonical ISO"),
        ({"expires_on": "2026-10-16"}, "after approved_on"),
        ({"extra": 1}, "closed-world"),
    ],
)
def test_the_declaration_is_validated_closed_world(override, message):
    with pytest.raises(ValueError, match=message):
        uk_support_floor_from_payload(_block(**override))


def test_the_packaged_floor_drops_only_unsupported_bands_and_binds_thin_ones():
    """María's rule of 2026-10-21: fewer than ten records is no reason to leave a row out."""

    declaration = load_uk_support_floor_declaration()
    floor = uk_support_floor_from_payload(declaration)
    bands = (
        "up_to_100",
        *(f"{low}_01_to_{low + 100}" for low in range(100, 1300, 100)),
        "1300_01_and_above",
    )
    spi = (
        "employment_income",
        "self_employment_income",
        "dividend_income",
        "property_income",
        "private_pension_income",
        "state_pension",
        "savings_interest_income",
    )
    scopes = {scope.scope: scope for scope in floor.scopes}
    assert {name: scope.contract_target_ids for name, scope in scopes.items()} == {
        "uc_payment_and_benefit_cap_bands": {
            *(
                f"dwp.uc.payment_distribution_{family}"
                for family in (
                    "single",
                    "lone_parent",
                    "couple_no_children",
                    "couple_with_children",
                )
            ),
            *(f"dwp.benefit_cap.capped_households_{band}" for band in bands),
        },
        # Extended on María's ruling of 2026-10-22 (the next policyengine-uk bump).
        "hmrc_spi_income_band_counts": {
            f"hmrc.spi.{family}.count_by_total_income_band" for family in spi
        },
        "hmrc_spi_income_band_amounts": {
            f"hmrc.spi.{family}.amount_by_total_income_band" for family in spi
        },
        "council_tax_band_h_by_region": {"mhclg.council_tax_stock.band_h"},
    }
    assert floor.floor == 1
    assert (floor.approved_by, floor.approved_on, floor.expires_on) == (
        "juaristi22",
        date(2026, 10, 22),
        date(2027, 1, 8),
    )
    # The rows once left out for thin support are bound again; the floor
    # covers the ones no household carries.
    named = {entry["name"] for entry in load_uk_calibration_measure_exclusions()}
    assert not {name for name in named if name.startswith("dwp.benefit_cap.")}
    # The HMRC £1m-and-over cells were bound on arm 1 and distorted their
    # neighbours, so they are held on that measurement, not for thin support.
    held_top = [
        entry
        for entry in load_uk_calibration_measure_exclusions()
        if "income_band_1_000_000_to_inf" in entry["name"]
    ]
    assert len(held_top) == 10
    assert all(entry["reason"].startswith("Measured distortion") for entry in held_top)
    assert not {name for name in named if name.startswith("mhclg.council_tax_stock.")}
    assert not named & {
        "hmrc/dividend_income_income_band_500_000_to_1_000_000",
        "hmrc/property_income_count_income_band_500_000_to_1_000_000",
    }


def test_each_packaged_scope_holds_one_unit():
    """A share ceiling sums published values, so no scope mixes units."""

    targets = {
        target["target_id"]: target
        for target in json.loads(
            importlib_resources.files("microcosm.build.uk")
            .joinpath("uk_population_targets.json")
            .read_text()
        )["targets"]
    }
    floor = uk_support_floor_from_payload(load_uk_support_floor_declaration())
    for scope in floor.scopes:
        units = {
            (
                targets[target_id]["measurement"]["entity"],
                targets[target_id]["measurement"]["concept"].rsplit(".", 1)[-1],
            )
            for target_id in scope.contract_target_ids
        }
        assert len(units) == 1, (scope.scope, units)


def test_only_schema_4_registers_carry_a_floor(tmp_path):
    path = tmp_path / "exclusions.json"
    for version in (2, 3):
        path.write_text(
            json.dumps(
                {"schema_version": version, "support_floor": _block(), "exclusions": []}
            )
        )
        with pytest.raises(ValueError, match="support_floor|schema_version"):
            load_uk_calibration_measure_exclusions(path)
    path.write_text(
        json.dumps({"schema_version": 4, "support_floor": _block(), "exclusions": []})
    )
    assert load_uk_calibration_measure_exclusions(path) == ()
    assert load_uk_support_floor_declaration(path) == _block()


def test_readers_narrow_by_the_recorded_drops_under_the_declared_floor():
    floor = uk_support_floor_from_payload(_block())
    registry = _registry(
        _spec("low", "band_low", 40.0),
        _spec("thin", "band_thin", 30.0),
        _spec("other", "other_thin", 7.0, target="dwp.uc.households"),
    )
    record = floor.record(EVALUATED)
    per_row = {"per_row": {"reason": "reviewed", "approved_on": "2026-09-01"}}
    narrowed = narrow_uk_registry_by_support_floor(
        registry, {**per_row, "thin": record}, floor
    )
    assert [spec.name for spec in narrowed.specs] == ["low", "other"]
    assert uk_support_floor_names({**per_row, "thin": record}) == {"thin"}
    assert narrow_uk_registry_by_support_floor(registry, per_row, None) is registry
    with pytest.raises(ValueError, match="declares no support floor"):
        narrow_uk_registry_by_support_floor(registry, {"thin": record}, None)
    with pytest.raises(ValueError, match="outside the floor's scope"):
        narrow_uk_registry_by_support_floor(registry, {"other": record}, floor)
    with pytest.raises(ValueError, match="not a row of the register"):
        narrow_uk_registry_by_support_floor(registry, {"absent": record}, floor)
    with pytest.raises(ValueError, match="declared signature"):
        narrow_uk_registry_by_support_floor(
            registry, {"thin": {**record, "approved_by": "someone"}}, floor
        )
    tight = uk_support_floor_from_payload(_block(scope={"max_dropped_rows": 0}))
    with pytest.raises(ValueError, match="in scope 'uc_bands', more than its signed"):
        narrow_uk_registry_by_support_floor(registry, {"thin": record}, tight)


def _encode(monkeypatch, registry, *, band_edge=None, exclusions=None, **block):
    """The national problem node's route, the engine replaced by the frame's
    own columns (the measures are household columns here)."""

    from microcosm.build.uk_runtime import graph_national
    from microcosm.build.uk_runtime.local_rowwise import UKRowwiseNationalRows
    from microcosm.build.uk_runtime.national_doctrine import UK_NATIONAL_SOLVE_DOCTRINE

    materialized = []

    def materialize(frame, compiled, **kwargs):
        materialized.append([spec.name for spec in compiled.specs])
        rows = UKRowwiseNationalRows(
            targets=compiled.to_target_set(),
            registry=compiled,
            families=tuple(sorted({spec.family for spec in compiled.specs})),
        )
        return frame, (lambda prepared: prepared), rows, {"mode": "frame_only"}

    monkeypatch.setattr(graph_national, "materialize_uk_national_rows", materialize)
    doctrine = UK_NATIONAL_SOLVE_DOCTRINE
    payload = graph_national.encode_uk_national_problem(
        _frame(),
        national_registry=registry,
        band_edge_registry=registry if band_edge is None else band_edge,
        period=2025,
        doctrine={
            "target_weight_rule": doctrine.target_weight_rule,
            "target_loss_cap": doctrine.target_loss_cap,
            "max_weight_ratio": doctrine.max_weight_ratio,
            "l0_lambda": doctrine.l0_lambda,
            "mass_rule": doctrine.mass_rule,
            "scale_rule": doctrine.scale_rule,
        },
        measure_exclusions=exclusions,
        support_floor={"declaration": _block(**block), "evaluated_on": "2026-10-20"},
        resolver_factory=None,
    )
    from microcosm.calibrate.artifacts import decode_problem

    return decode_problem(payload), materialized


def _national_spec(name, measure, value, target=SCOPE):
    spec = _spec(name, measure, value, target=target)
    return TargetSpec(
        **{
            **spec.to_dict(),
            "metadata": {**spec.metadata, "geography_level": "country"},
        }
    )


def test_the_national_problem_drops_unsupported_rows_and_binds_thin_ones(
    monkeypatch,
):
    registry = _registry(
        _national_spec("low", "band_low", 40.0),
        _national_spec("thin", "band_thin", 30.0),
        _national_spec("zero_weight", "band_zero_weight", 5.0),
        _national_spec("other", "other_thin", 7.0, target="dwp.uc.households"),
    )
    problem, materialized = _encode(monkeypatch, registry)

    assert materialized == [["low", "thin", "zero_weight", "other"]]
    assert problem.problem.names == ("low@2025", "thin@2025", "other@2025")
    assert [row["contract_target_id"] for row in problem.target_metadata] == [
        SCOPE,
        SCOPE,
        "dwp.uc.households",
    ]
    bindings = problem.bindings
    assert len(bindings["target_loss_weights"]) == 3
    assert bindings["activated_reference_count"] == 3
    assert bindings["resolved_reference_count"] == 3
    assert (
        bindings["measure_exclusions"]["zero_weight"]["kind"] == UK_SUPPORT_FLOOR_KIND
    )
    assert bindings["support_floor"]["dropped"] == ["zero_weight"]
    # The register the targets node froze stays the approved one.
    assert bindings["register_sha256"] == registry.version


def test_the_national_problem_measures_per_row_exclusions_in_scope(monkeypatch):
    approved = _registry(_national_spec("low", "band_low", 40.0))
    band_edges = _registry(
        _national_spec("low", "band_low", 40.0),
        _national_spec("zero_weight", "band_zero_weight", 30.0),
        _national_spec("high", "band_thin", 9.0),
    )
    record = {
        "reason": "reviewed",
        "tracking": "uk-data#452",
        "approved_by": "juaristi22",
        "adjudication": "microcosm#757",
        "approved_on": "2026-08-26",
        "expires_on": "2026-11-26",
        "evaluated_on": "2026-10-20",
    }
    with pytest.raises(ValueError, match=r"retire those entries .*\['zero_weight'\]"):
        _encode(
            monkeypatch,
            approved,
            band_edge=band_edges,
            exclusions={"zero_weight": record, "high": record},
        )
    problem, materialized = _encode(
        monkeypatch, approved, band_edge=band_edges, exclusions={"high": record}
    )
    # The entry on a thin row was compiled to be measured and left again.
    assert materialized == [["low", "high"]]
    assert problem.problem.names == ("low@2025",)
    assert problem.bindings["measure_exclusions"] == {"high": record}
    assert problem.bindings["support_floor"]["probed_row_exclusions"] == 1


def test_readers_take_the_solved_register_from_the_problem_bindings():
    from microcosm.build.uk_runtime.graph_national import (
        solved_national_measure_exclusions,
        solved_national_registry,
    )
    from microcosm.build.uk_runtime.graph_targets import registry_payload

    registry = _registry(
        _national_spec("low", "band_low", 40.0),
        _national_spec("thin", "band_thin", 30.0),
    )
    per_row = {"per_row": {"reason": "reviewed"}}
    document = {
        "national_registry": registry_payload(registry),
        "measure_exclusions": per_row,
        "support_floor": {"declaration": _block(), "evaluated_on": "2026-10-20"},
    }
    record = uk_support_floor_from_payload(_block()).record(EVALUATED)
    bound = {**per_row, "thin": record}
    exclusions = solved_national_measure_exclusions(
        document, {"measure_exclusions": bound}
    )
    assert exclusions == bound
    solved = solved_national_registry(document, exclusions)
    assert [spec.name for spec in solved.specs] == ["low"]
    with pytest.raises(ValueError, match="beyond the support floor"):
        solved_national_measure_exclusions(
            document, {"measure_exclusions": {"low": {"reason": "x"}, **per_row}}
        )
    with pytest.raises(ValueError, match="beyond the support floor"):
        solved_national_measure_exclusions(
            document, {"measure_exclusions": {"thin": record}}
        )
