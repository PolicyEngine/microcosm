"""Landlords' receipts and finance costs on the UK spine (microcosm#1106)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build import load_country_spec
from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.stochastic_assignment import (
    binomial_share_tolerance,
    stable_identity_uniforms,
)
from microcosm.build.uk_runtime import copy_id_table
from microcosm.build.uk_runtime import property_components as components
from microcosm.build.uk_runtime.hmrc_property_rental import (
    HMRC_PROPERTY_RENTAL_EXPENSE_TYPES,
    HMRC_PROPERTY_RENTAL_RECEIPTS_BAND_LOWER_BOUNDS,
    HMRCPropertyRentalBand,
    HMRCPropertyRentalFacts,
)
from microcosm.build.uk_runtime.property_components import (
    ALLOCATE_PROPERTY_RENTAL_INCOME_KIND,
    IMPUTE_FRS_PROPERTY_FINANCE_COSTS_KIND,
    UK_PROPERTY_COMPONENTS_OUTPUT_COLUMNS,
    UK_PROPERTY_COMPONENTS_REWRITE_COLUMNS,
    UK_PROPERTY_COMPONENTS_STAGE_NAME,
    FinanceCostModelSpec,
    PropertyComponentsError,
    ReceiptsAllocationSpec,
    _incidence_edges,
    allocate_property_rental_income,
    frs_concept_rental_profit,
    impute_frs_property_finance_costs,
    spi_tape_finance_cost_donor,
)

#: PRIS 2024-25 Table 13 counts (thousands, all tax entities).
_BAND_COUNTS = (1300, 860, 330, 150, 80, 50, 30, 20, 10, 10, 40)


def _stage():
    spec = load_country_spec("uk")
    assert spec.sources is not None
    return {stage.stage: stage for stage in spec.sources.stages}[
        UK_PROPERTY_COMPONENTS_STAGE_NAME
    ]


def _parameters(kind: str) -> dict:
    (operation,) = [op for op in _stage().operations if op.kind == kind]
    return dict(operation.parameters)


def _facts(
    *, landlords: float, receipts: float, counts=None
) -> HMRCPropertyRentalFacts:
    bounds = HMRC_PROPERTY_RENTAL_RECEIPTS_BAND_LOWER_BOUNDS
    counts = list(_BAND_COUNTS if counts is None else counts)
    total = float(sum(counts))
    all_landlords = landlords / 0.99
    return HMRCPropertyRentalFacts(
        tax_year=2024,
        landlords={
            "individual": landlords,
            "partnership": all_landlords - landlords,
            "all": all_landlords,
        },
        receipts={"individual": receipts, "partnership": 0.0, "all": receipts},
        expenses={
            "individual": 0.6 * receipts,
            "partnership": 0.0,
            "all": 0.6 * receipts,
        },
        expenses_by_type={
            kind: 0.6 * receipts / len(HMRC_PROPERTY_RENTAL_EXPENSE_TYPES)
            for kind in HMRC_PROPERTY_RENTAL_EXPENSE_TYPES
        },
        expense_landlords_by_type={},
        receipts_bands=tuple(
            HMRCPropertyRentalBand(
                lower,
                bounds[index + 1] if index + 1 < len(bounds) else None,
                all_landlords * count / total,
            )
            for index, (lower, count) in enumerate(zip(bounds, counts, strict=True))
        ),
        resource_sha256="test",
    )


def _landlords(
    n: int = 3_000, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    profit = np.exp(rng.normal(np.log(6_000.0), 1.0, n))
    profit[: n // 10] = 0.0
    weights = rng.uniform(500.0, 1_500.0, n)
    ids = np.arange(1, n + 1) * 101
    return ids, profit, weights


class TestDeclaredParameters:
    def test_the_stage_declares_the_runtime_constants(self) -> None:
        stage = _stage()
        assert tuple(stage.outputs) == UK_PROPERTY_COMPONENTS_OUTPUT_COLUMNS
        assert tuple(stage.rewrites) == UK_PROPERTY_COMPONENTS_REWRITE_COLUMNS
        finance = FinanceCostModelSpec.from_parameters(
            _parameters(IMPUTE_FRS_PROPERTY_FINANCE_COSTS_KIND)
        )
        receipts = ReceiptsAllocationSpec.from_parameters(
            _parameters(ALLOCATE_PROPERTY_RENTAL_INCOME_KIND)
        )
        assert finance.tape_preparation_seed == 42
        assert finance.predictors == ("profit_after_finance_costs", "age")
        assert (
            receipts.band_lower_bounds
            == HMRC_PROPERTY_RENTAL_RECEIPTS_BAND_LOWER_BOUNDS
        )

    def test_the_tape_seed_is_the_income_stages_stage_1_seed(self) -> None:
        spec = load_country_spec("uk")
        assert spec.sources is not None
        (income,) = [
            stage
            for stage in spec.sources.stages
            if stage.stage == "hmrc_spi_income_spine"
        ]
        (read,) = [
            op for op in income.operations if op.kind == "strict_read_private_table"
        ]
        assert (
            read.parameters["seed"]
            == FinanceCostModelSpec.from_parameters(
                _parameters(IMPUTE_FRS_PROPERTY_FINANCE_COSTS_KIND)
            ).tape_preparation_seed
        )

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("income_uprating_variables", {"property_income": "property_income"}),
            ("donor_income_period", 2023),
            ("tape_preparation_seed", 7),
            ("amount_salt", "property_components:has_finance_costs"),
            ("add_back_to", "property_rental_income"),
        ],
    )
    def test_a_drifted_finance_cost_declaration_is_refused(self, key, value) -> None:
        parameters = _parameters(IMPUTE_FRS_PROPERTY_FINANCE_COSTS_KIND)
        parameters[key] = value
        with pytest.raises(PropertyComponentsError):
            FinanceCostModelSpec.from_parameters(parameters)

    def test_receipts_bands_must_be_table_13s(self) -> None:
        parameters = _parameters(ALLOCATE_PROPERTY_RENTAL_INCOME_KIND)
        parameters["band_lower_bounds"] = [0, 10_000, 100_000]
        with pytest.raises(PropertyComponentsError, match="Table 13"):
            ReceiptsAllocationSpec.from_parameters(parameters)

    def test_the_walk_declares_a_tie_salt_of_its_own(self) -> None:
        # The receipts walk keys its tie-break on person ids under walk_seed
        # and its own salt (microcosm#931): empty or missing is refused, and so
        # is a salt the finance-cost draws key the same person ids on.
        parameters = _parameters(ALLOCATE_PROPERTY_RENTAL_INCOME_KIND)
        finance = FinanceCostModelSpec.from_parameters(
            _parameters(IMPUTE_FRS_PROPERTY_FINANCE_COSTS_KIND)
        )
        in_use = (finance.incidence_salt, finance.amount_salt)
        spec = ReceiptsAllocationSpec.from_parameters(parameters, salts_in_use=in_use)
        assert spec.tie_salt == "allocate_property_rental_income:tie"
        assert spec.walk_seed == 0
        for bad in ("", None):
            with pytest.raises(PropertyComponentsError, match="tie_salt"):
                ReceiptsAllocationSpec.from_parameters({**parameters, "tie_salt": bad})
        without = {key: value for key, value in parameters.items() if key != "tie_salt"}
        with pytest.raises(PropertyComponentsError, match="tie_salt"):
            ReceiptsAllocationSpec.from_parameters(without)
        for salt in in_use:
            with pytest.raises(PropertyComponentsError, match="tie_salt"):
                ReceiptsAllocationSpec.from_parameters(
                    {**parameters, "tie_salt": salt}, salts_in_use=in_use
                )


class TestFRSConcept:
    def test_the_reference_persons_sublet_rent_leaves_the_rental_profit(self) -> None:
        person = pd.DataFrame(
            {
                "person_household_id": [1, 1, 2, 3],
                "is_household_head": [True, False, True, True],
                "property_income": [5_000.0, 2_000.0, 0.0, 800.0],
            }
        )
        household = pd.DataFrame(
            {"household_id": [1, 2, 3], "subrent": [1_200.0, 0.0, 800.0]}
        )
        sublet, rental = frs_concept_rental_profit(person, household)
        np.testing.assert_allclose(sublet, [1_200.0, 0.0, 0.0, 800.0])
        np.testing.assert_allclose(rental, [3_800.0, 2_000.0, 0.0, 0.0])


class TestTapeDonor:
    def _prepared(self) -> pd.DataFrame:
        rng = np.random.default_rng(3)
        n = 4_000
        profit = np.exp(rng.normal(np.log(8_000.0), 0.9, n))
        profit[:400] = 0.0
        costs = np.where(rng.random(n) < 0.4, 0.5 * profit * rng.random(n), 0.0)
        return pd.DataFrame(
            {
                "property_income": profit,
                "property_finance_costs": costs,
                "age": rng.uniform(25, 85, n),
                "FACT": rng.uniform(50, 150, n),
            }
        )

    def test_the_analog_landlords_are_rebased_and_keep_a_profit_after_costs(
        self,
    ) -> None:
        prepared = self._prepared()
        spec = FinanceCostModelSpec.from_parameters(
            _parameters(IMPUTE_FRS_PROPERTY_FINANCE_COSTS_KIND)
        )
        factors = {"property_income": 1.1, "property_finance_costs": 1.9}
        donor = spi_tape_finance_cost_donor(
            prepared, uprating_factors=factors, spec=spec
        )
        profit = prepared["property_income"].to_numpy() * 1.1
        costs = prepared["property_finance_costs"].to_numpy() * 1.9
        analog = (profit > 0) & (profit - costs > 0)
        assert len(donor.analog) == int(analog.sum())
        np.testing.assert_allclose(
            np.sort(donor.analog["profit_after_finance_costs"].to_numpy()),
            np.sort((profit - costs)[analog]),
        )
        assert len(donor.incidence) == len(donor.edges) + 1 <= spec.incidence_cells
        assert all(cell["tape_weight"] > 0 for cell in donor.incidence)
        # The cells' shares reproduce the analog landlords' overall share.
        mass = sum(cell["tape_weight"] for cell in donor.incidence)
        share = (
            sum(
                cell["tape_weight"] * cell["share_with_finance_costs"]
                for cell in donor.incidence
            )
            / mass
        )
        weight = prepared["FACT"].to_numpy()[analog]
        assert share == pytest.approx(
            weight[costs[analog] > 0].sum() / weight.sum(), rel=1e-12
        )

    def test_incidence_edges_never_leave_a_cell_empty(self) -> None:
        values = np.array([0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 9.0])
        weights = np.ones_like(values)
        edges = _incidence_edges(values, weights, 10)
        cells = np.searchsorted(edges, values, side="right")
        assert set(cells) == set(range(len(edges) + 1))


class TestFinanceCostDraw:
    def _donor(self):
        prepared = TestTapeDonor()._prepared()
        spec = FinanceCostModelSpec.from_parameters(
            _parameters(IMPUTE_FRS_PROPERTY_FINANCE_COSTS_KIND)
        )
        factors = {"property_income": 1.0, "property_finance_costs": 1.0}
        return spi_tape_finance_cost_donor(
            prepared, uprating_factors=factors, spec=spec
        ), spec

    def test_costs_are_keyed_on_person_id_and_zero_off_landlords(self) -> None:
        donor, spec = self._donor()
        ids, profit, weights = _landlords(800)
        age = np.linspace(20, 90, len(ids))
        first = impute_frs_property_finance_costs(
            donor, person_ids=ids, profit=profit, age=age, weights=weights, spec=spec
        )
        order = np.random.default_rng(1).permutation(len(ids))
        second = impute_frs_property_finance_costs(
            donor,
            person_ids=ids[order],
            profit=profit[order],
            age=age[order],
            weights=weights[order],
            spec=spec,
        )
        np.testing.assert_array_equal(first.costs[order], second.costs)
        assert (first.costs[profit <= 0.0] == 0.0).all()
        assert (first.costs >= 0.0).all()
        assert first.receipt["regime"] == "positive_only"
        assert 0 < first.receipt["drawn"] < int((profit > 0).sum())


class TestReceiptsWalk:
    def _allocate(self, ids, profit, weights, *, receipts_per_profit: float = 1.65):
        landlords = float(weights[profit > 0].sum())
        total_profit = float((profit * weights)[profit > 0].sum())
        facts = _facts(landlords=landlords, receipts=receipts_per_profit * total_profit)
        spec = ReceiptsAllocationSpec.from_parameters(
            _parameters(ALLOCATE_PROPERTY_RENTAL_INCOME_KIND)
        )
        return (
            allocate_property_rental_income(
                person_ids=ids, profit=profit, weights=weights, facts=facts, spec=spec
            ),
            facts,
        )

    def test_receipts_never_fall_below_profit_and_vanish_off_landlords(self) -> None:
        ids, profit, weights = _landlords()
        allocation, _ = self._allocate(ids, profit, weights)
        landlord = profit > 0.0
        assert (allocation.receipts[landlord] >= profit[landlord]).all()
        assert (allocation.receipts[~landlord] == 0.0).all()

    def test_the_walk_fills_every_band_above_the_lowest_to_its_count(self) -> None:
        ids, profit, weights = _landlords()
        allocation, facts = self._allocate(ids, profit, weights)
        cells = allocation.receipt["bands_top_down"]
        largest = float(weights.max())
        for cell in cells[:-1]:
            assert cell["walk_landlords_weight"] == pytest.approx(
                cell["target_landlords"], abs=largest
            )
        lowest = cells[-1]
        assert lowest["walk_landlords_weight"] == pytest.approx(
            float(weights[profit > 0].sum())
            - sum(cell["walk_landlords_weight"] for cell in cells[:-1])
        )

    def test_the_top_band_multiplier_reaches_the_receipts_total(self) -> None:
        # Receipts at twice the profit put the top band's solve inside its
        # bounds on these landlords (k near 3.8).
        ids, profit, weights = _landlords()
        allocation, facts = self._allocate(
            ids, profit, weights, receipts_per_profit=2.0
        )
        top = allocation.receipt["bands_top_down"][0]
        assert 1.0 < top["profit_multiplier"] < 5.0
        assert top["mean_receipts_gbp"] == pytest.approx(
            top["target_mean_receipts_gbp"], rel=1e-9
        )

    def test_the_top_band_holds_its_floor_and_records_the_overshoot(self) -> None:
        # Receipts equal to profit in total leave the top band nothing after
        # the bounded bands' midpoints: the multiplier stays at its floor of 1,
        # the band's receipts are its profit floor and the overshoot is
        # recorded rather than hidden (microcosm#1145 review, item 7).
        ids, profit, weights = _landlords()
        allocation, _ = self._allocate(ids, profit, weights, receipts_per_profit=1.0)
        top = allocation.receipt["bands_top_down"][0]
        assert top["profit_multiplier"] == 1.0
        assert top["multiplier_at_floor"] is True
        assert top["mean_receipts_at_floor_gbp"] == top["mean_receipts_gbp"]
        assert top["mean_receipts_shortfall_gbp"] < 0.0
        assert top["target_mean_nonpositive"] is True

    def test_every_bounded_band_sits_within_its_recorded_rounding_allowance(
        self,
    ) -> None:
        ids, profit, weights = _landlords()
        # Receipts at twice the profit keep the top band's solve off its floor.
        allocation, _ = self._allocate(ids, profit, weights, receipts_per_profit=2.0)
        cells = allocation.receipt["bands_top_down"]
        for cell in cells[:-1]:
            error = abs(cell["walk_landlords_weight"] - cell["target_landlords"])
            assert error <= cell["boundary_allowance_weight"]
            # Two straddlers at most, each no heavier than the heaviest landlord.
            assert cell["boundary_allowance_weight"] <= 2.0 * float(weights.max())
        assert cells[0]["multiplier_at_floor"] is False
        assert cells[0]["target_mean_nonpositive"] is False

    def test_a_positive_target_below_the_floor_holds_the_multiplier_at_one(
        self,
    ) -> None:
        # Receipts at 1.65 times profit leave the top band a positive target
        # mean that its profit floor already exceeds: k stays 1 and the
        # overshoot is recorded (microcosm#1145 re-review, item 7).
        ids, profit, weights = _landlords()
        allocation, _ = self._allocate(ids, profit, weights, receipts_per_profit=1.65)
        top = allocation.receipt["bands_top_down"][0]
        assert top["target_mean_receipts_gbp"] > 0.0
        assert top["target_mean_nonpositive"] is False
        assert top["profit_multiplier"] == 1.0
        assert top["multiplier_at_floor"] is True
        assert top["mean_receipts_shortfall_gbp"] < 0.0

    def test_an_empty_band_carries_one_straddlers_allowance_and_passes_the_gate(
        self,
    ) -> None:
        # A band whose share is smaller than one landlord's weight is missed
        # whole; both its boundaries fall between the same two people, so its
        # allowance is one straddler's weight, which covers the miss
        # (microcosm#1145 re-review, item M2).
        from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate

        ids, profit, weights = _landlords()
        counts = list(_BAND_COUNTS)
        counts[6] = sum(_BAND_COUNTS) * 1e-4  # the £60,000 band, under one weight
        landlords = float(weights[profit > 0].sum())
        total_profit = float((profit * weights)[profit > 0].sum())
        facts = _facts(landlords=landlords, receipts=2.0 * total_profit, counts=counts)
        spec = ReceiptsAllocationSpec.from_parameters(
            _parameters(ALLOCATE_PROPERTY_RENTAL_INCOME_KIND)
        )
        allocation = allocate_property_rental_income(
            person_ids=ids, profit=profit, weights=weights, facts=facts, spec=spec
        )
        cells = {
            cell["lower_bound"]: cell for cell in allocation.receipt["bands_top_down"]
        }
        thin = cells[60_000]
        assert thin["target_landlords"] < float(weights.min())
        assert thin["walk_records"] <= 1
        if thin["walk_records"] == 0:
            assert thin["boundary_allowance_weight"] <= float(weights.max())
        assert (
            abs(thin["walk_landlords_weight"] - thin["target_landlords"])
            <= (thin["boundary_allowance_weight"])
        )
        result = uk_stage_health_gate(
            evidence={
                "stage": "property_components",
                "receipts": {**allocation.receipt, "rows_receipts_below_profit": 0},
            },
            stage="property_components",
            check="property_components",
            parameters={
                "stage": "property_components",
                "check": "property_components",
                "maximum_rows_receipts_below_profit": 0,
                "maximum_rows_with_profit_above_band": 0,
                "maximum_band_error_in_boundary_weights": 1.0,
            },
        )
        assert result.passed, result.failures

    def test_bands_take_their_pris_share_of_the_landlords_here(self) -> None:
        # microcosm#1106: before calibration the spine carries fewer landlords
        # than PRIS. Each band takes its share of the landlords present, so a
        # frame with half PRIS's landlords fills every band to half its count;
        # filling the published counts would leave the lowest band empty.
        ids, profit, weights = _landlords()
        landlords = float(weights[profit > 0].sum())
        total_profit = float((profit * weights)[profit > 0].sum())
        facts = _facts(landlords=2.0 * landlords, receipts=4.0 * total_profit)
        spec = ReceiptsAllocationSpec.from_parameters(
            _parameters(ALLOCATE_PROPERTY_RENTAL_INCOME_KIND)
        )
        allocation = allocate_property_rental_income(
            person_ids=ids, profit=profit, weights=weights, facts=facts, spec=spec
        )
        cells = allocation.receipt["bands_top_down"]
        largest = float(weights.max())

        assert allocation.receipt["share_scale"] == pytest.approx(0.5)
        for cell in cells:
            assert cell["target_landlords"] == pytest.approx(
                0.5 * cell["published_landlords"]
            )
            assert cell["walk_landlords_weight"] == pytest.approx(
                cell["target_landlords"], abs=2.0 * largest
            )
        assert cells[-1]["walk_landlords_weight"] > 0.4 * landlords

    def test_receipts_do_not_depend_on_row_order(self) -> None:
        ids, profit, weights = _landlords()
        first, _ = self._allocate(ids, profit, weights)
        order = np.random.default_rng(5).permutation(len(ids))
        second, _ = self._allocate(ids[order], profit[order], weights[order])
        np.testing.assert_allclose(first.receipts[order], second.receipts)

    @staticmethod
    def _by_id(monkeypatch):
        """The walk's order before microcosm#931: ties by person id alone."""

        monkeypatch.setattr(
            components,
            "keyed_tie_order",
            lambda primary, ids, **_: np.lexsort((np.asarray(ids), primary)),
        )

    def test_distinct_profits_take_the_receipts_of_the_id_order(
        self, monkeypatch
    ) -> None:
        # The tie-break only reorders exact ties, so on distinct profits
        # every receipt and every cell of the receipt is the id order's bit
        # for bit; only the declared order's wording moves (microcosm#931).
        ids, profit, weights = _landlords()
        keyed, _ = self._allocate(ids, profit, weights, receipts_per_profit=2.0)
        self._by_id(monkeypatch)
        by_id, _ = self._allocate(ids, profit, weights, receipts_per_profit=2.0)

        np.testing.assert_array_equal(keyed.receipts, by_id.receipts)
        assert {**keyed.receipt, "order": None} == {**by_id.receipt, "order": None}
        assert keyed.receipt["order"] == (
            "profit descending, landlords tied on profit in ascending order of a "
            "uniform keyed on person_id under walk_seed and tie_salt, then "
            "person_id ascending"
        )

    def test_tied_landlords_take_their_receipts_in_keyed_order(
        self, monkeypatch
    ) -> None:
        # Six landlords of equal weight tie on profit inside one bounded band:
        # the walk places them in the order of their keyed uniforms, which is
        # not their id order, so the first takes the highest receipt. Their
        # receipts are the id order's, among themselves permuted, and every
        # other landlord's is unchanged (microcosm#931).
        ids, profit, weights = _landlords()
        tied = np.flatnonzero(profit > 0.0)[100:106]
        profit[tied] = float(np.median(profit[profit > 0.0]))
        weights[tied] = 1_000.0
        spec = ReceiptsAllocationSpec.from_parameters(
            _parameters(ALLOCATE_PROPERTY_RENTAL_INCOME_KIND)
        )
        uniforms = stable_identity_uniforms(
            ids[tied], seed=spec.walk_seed, salt=spec.tie_salt
        )
        keyed_order = tied[np.lexsort((ids[tied], uniforms))]
        assert keyed_order.tolist() != sorted(keyed_order.tolist())

        keyed, _ = self._allocate(ids, profit, weights, receipts_per_profit=2.0)
        receipts = keyed.receipts[keyed_order]
        assert len(set(receipts.tolist())) == len(tied)
        assert (np.diff(receipts) < 0.0).all()
        self._by_id(monkeypatch)
        by_id, _ = self._allocate(ids, profit, weights, receipts_per_profit=2.0)
        np.testing.assert_array_equal(
            np.sort(keyed.receipts[tied]), np.sort(by_id.receipts[tied])
        )
        others = np.setdiff1d(np.arange(len(ids)), tied)
        np.testing.assert_array_equal(keyed.receipts[others], by_id.receipts[others])
        order = np.random.default_rng(5).permutation(len(ids))
        monkeypatch.undo()
        again, _ = self._allocate(
            ids[order], profit[order], weights[order], receipts_per_profit=2.0
        )
        np.testing.assert_allclose(keyed.receipts[order], again.receipts)

    def test_clone_zero_takes_the_top_receipt_of_tied_clones_at_its_share(
        self, monkeypatch
    ) -> None:
        # Sixty source landlords, each as three geographic clones of equal
        # weight tied on profit, among the synthetic landlords. Walked by id,
        # clone 0 took the highest receipt of its family every time; keyed on
        # each clone's own id, at about a third of the families, inside the
        # four-sigma binomial band (design verifier P10; microcosm#931).
        ids, profit, weights = _landlords()
        families, clone_count = 60, 3
        sources = np.arange(1, families + 1, dtype=np.int64) * 7 + 900_000
        clone_ids = np.concatenate(
            [
                sources,
                *(
                    copy_id_table.mint(sources, "geographic_support", k)
                    for k in range(1, clone_count)
                ),
            ]
        )
        ids = np.concatenate([ids, clone_ids])
        profit = np.concatenate(
            [profit, np.tile(np.linspace(3_000.0, 9_000.0, families), clone_count)]
        )
        weights = np.concatenate([weights, np.full(families * clone_count, 900.0)])
        clones = slice(len(ids) - families * clone_count, len(ids))

        def tops(allocation) -> np.ndarray:
            receipts = allocation.receipts[clones].reshape(clone_count, families)
            assert all(len(set(column)) == clone_count for column in receipts.T)
            return np.argmax(receipts, axis=0)

        keyed, _ = self._allocate(ids, profit, weights, receipts_per_profit=2.0)
        share = float((tops(keyed) == 0).mean())
        band = binomial_share_tolerance(
            target=1 / clone_count, rows=families, sigma=4.0
        )
        assert abs(share - 1 / clone_count) <= band
        self._by_id(monkeypatch)
        by_id, _ = self._allocate(ids, profit, weights, receipts_per_profit=2.0)
        assert float((tops(by_id) == 0).mean()) == 1.0

    def test_unequal_weight_ties_move_no_untied_landlord_across_a_band(
        self, monkeypatch
    ) -> None:
        # Half the landlords tie on 60 round profits at weights that differ,
        # as different households reporting one amount do. The keyed order
        # puts another tied landlord at a band boundary than the id order did,
        # so tied landlords cross it and the bands' record counts move. An
        # untied landlord has the same landlords above it in either order, so
        # it keeps its band, though its receipt can move with its band's
        # members (microcosm#931).
        rng = np.random.default_rng((2800, 0))
        n = 2_400
        atoms = np.round(np.exp(rng.normal(np.log(9_000.0), 1.0, 60)), -2)
        profit = np.where(
            rng.random(n) < 0.5,
            atoms[rng.integers(0, atoms.size, n)],
            np.exp(rng.normal(np.log(9_000.0), 1.0, n)),
        )
        weights = rng.uniform(50.0, 900.0, n)
        ids = np.arange(1, n + 1, dtype=np.int64) * 7
        landlord = profit > 0.0
        tied = pd.Series(profit).duplicated(keep=False).to_numpy() & landlord
        assert tied.sum() > n // 3

        def walked_bands() -> np.ndarray:
            seen: dict[str, np.ndarray] = {}
            order, walk = components.keyed_tie_order, components._walk_bands

            def recording_order(primary, ids, **kwargs):
                seen["order"] = order(primary, ids, **kwargs)
                return seen["order"]

            def recording_walk(ranked_weights, masses, offset):
                seen["band"] = walk(ranked_weights, masses, offset)
                return seen["band"]

            with monkeypatch.context() as patch:
                patch.setattr(components, "keyed_tie_order", recording_order)
                patch.setattr(components, "_walk_bands", recording_walk)
                self._allocate(ids, profit, weights, receipts_per_profit=2.0)
            band = np.full(n, -1)
            band[np.flatnonzero(landlord)[seen["order"]]] = seen["band"]
            return band

        keyed = walked_bands()
        self._by_id(monkeypatch)
        by_id = walked_bands()

        crossed = keyed != by_id
        assert not crossed[landlord & ~tied].any()
        assert crossed[tied].sum() > 10
        bands = len(HMRC_PROPERTY_RENTAL_RECEIPTS_BAND_LOWER_BOUNDS)
        assert (
            np.bincount(keyed[landlord], minlength=bands)
            != np.bincount(by_id[landlord], minlength=bands)
        ).any()


class TestStageTransform:
    def _frame(self):
        from microcosm.build.uk_runtime.national_frame import uk_national_frame

        rng = np.random.default_rng(11)
        households = 60
        household = pd.DataFrame(
            {
                "household_id": np.arange(1, households + 1),
                "subrent": np.where(np.arange(households) % 9 == 0, 2_600.0, 0.0),
                "other_residential_property_value": np.where(
                    np.arange(households) % 2 == 0, 250_000.0, 0.0
                ),
                "non_residential_property_value": 0.0,
                "household_support_channel": np.where(
                    np.arange(households) < 40, "frs", "spi"
                ),
            }
        )
        person = pd.DataFrame(
            {
                "person_id": np.arange(1, 2 * households + 1),
                "person_benunit_id": np.repeat(np.arange(1, households + 1), 2),
                "person_household_id": np.repeat(np.arange(1, households + 1), 2),
                "age": np.tile([45.0, 12.0], households),
                "is_household_head": np.tile([True, False], households),
                "is_uc_claimant": np.tile([True, False], households),
                "person_support_channel": np.repeat(
                    np.where(np.arange(households) < 40, "frs", "spi"), 2
                ),
            }
        )
        profit = np.where(
            person["is_household_head"], rng.uniform(500.0, 40_000.0, len(person)), 0.0
        )
        sublet = (
            person["person_household_id"]
            .map(household.set_index("household_id")["subrent"])
            .to_numpy()
            * person["is_household_head"].to_numpy()
        )
        person["property_income"] = profit + sublet
        spi_rows = person["person_support_channel"].eq("spi") & person["is_uc_claimant"]
        person["person_is_spi_income_recipient"] = spi_rows.to_numpy()
        person["property_finance_costs"] = np.where(spi_rows, 0.2 * profit, 0.0)
        benunit = pd.DataFrame({"benunit_id": np.arange(1, households + 1)})
        return uk_national_frame(
            person=person,
            benunit=benunit,
            household=household,
            time_period="2024",
            household_weights=rng.uniform(100.0, 300.0, households),
        )

    def test_the_stage_refuses_a_frame_off_the_declared_build_period(self) -> None:
        from microcosm.build.uk_runtime.national_frame import uk_national_frame
        from microcosm.build.uk_runtime.property_components import (
            PropertyComponentsError,
            UKPropertyComponentsStageTransform,
        )

        frame = self._frame()
        moved = uk_national_frame(
            person=frame.table("person"),
            benunit=frame.table("benunit"),
            household=frame.table("household"),
            time_period="2025",
            household_weights=frame.weights_for("household").values,
        )
        transform = UKPropertyComponentsStageTransform(
            stage=_stage(),
            uprating_factors={"property_income": 1.0, "property_finance_costs": 1.0},
            facts=_facts(landlords=1_000.0, receipts=1_000_000.0),
        )
        with pytest.raises(PropertyComponentsError, match="build period 2024"):
            transform(moved)

    def test_the_stage_refuses_a_tie_salt_its_finance_costs_draw_on(self) -> None:
        # The receipts walk and the finance-cost draws key the same person
        # ids, so the stage refuses a tie salt equal to either draw's salt
        # (microcosm#931).
        from microcosm.build.uk_runtime.property_components import (
            UKPropertyComponentsStageTransform,
        )

        stage = _stage()
        finance = FinanceCostModelSpec.from_parameters(
            _parameters(IMPUTE_FRS_PROPERTY_FINANCE_COSTS_KIND)
        )
        for salt in (finance.incidence_salt, finance.amount_salt):
            operations = tuple(
                SourceOperationSpec(
                    kind=op.kind,
                    parameters=(
                        {**op.parameters, "tie_salt": salt}
                        if op.kind == ALLOCATE_PROPERTY_RENTAL_INCOME_KIND
                        else dict(op.parameters)
                    ),
                )
                for op in stage.operations
            )
            transform = UKPropertyComponentsStageTransform(
                stage=replace(stage, operations=operations),
                uprating_factors={
                    "property_income": 1.0,
                    "property_finance_costs": 1.0,
                },
                facts=_facts(landlords=1_000.0, receipts=1_000_000.0),
            )
            with pytest.raises(PropertyComponentsError, match="tie_salt"):
                transform(self._frame())

    def test_the_stage_adds_frs_costs_back_and_gives_every_landlord_receipts(
        self, tmp_path
    ) -> None:
        from microcosm.build.uk_runtime.property_components import (
            UKPropertyComponentsStageTransform,
        )
        from microcosm.build.uk_runtime.spi_income import SPIDonorAgeModel
        from test_support.microcosm_build.uk_spi_income import _write_donor

        _write_donor(tmp_path / "donor.tab")
        frame = self._frame()
        person = frame.table("person")
        weights = (
            person["person_household_id"]
            .map(
                pd.Series(
                    frame.weights_for("household").values,
                    index=frame.table("household")["household_id"].to_numpy(),
                )
            )
            .to_numpy()
        )
        transform = UKPropertyComponentsStageTransform(
            stage=_stage(),
            donor_table=pd.read_csv(tmp_path / "donor.tab", sep="\t"),
            age_model=SPIDonorAgeModel(
                populations={
                    (sex, age): 1.0 for sex in ("MALE", "FEMALE") for age in range(91)
                },
                state_pension_age=66,
                source="flat test populations",
            ),
            uprating_factors={"property_income": 1.0, "property_finance_costs": 1.0},
            facts=_facts(
                landlords=float(weights[person["property_income"] > 0].sum()),
                receipts=1.65 * float((person["property_income"] * weights).sum()),
            ),
        )
        result = transform(frame)
        out = result.table("person")

        assert list(out.columns)[-1] == "property_rental_income"
        assert (out["property_rental_income"] >= out["property_income"] - 1e-6).all()
        assert (
            out.loc[out["property_income"] == 0.0, "property_rental_income"] == 0.0
        ).all()
        spi = person["person_support_channel"].eq("spi") & person["is_uc_claimant"]
        # The SPI channel keeps its profit and the costs the income stage drew.
        np.testing.assert_array_equal(
            out.loc[spi, "property_income"], person.loc[spi, "property_income"]
        )
        np.testing.assert_array_equal(
            out.loc[spi, "property_finance_costs"],
            person.loc[spi, "property_finance_costs"],
        )
        # FRS landlords' profit gains exactly the costs they were given.
        frs = ~spi
        np.testing.assert_allclose(
            out.loc[frs, "property_income"] - person.loc[frs, "property_income"],
            out.loc[frs, "property_finance_costs"],
        )
        assert (out.loc[frs, "property_finance_costs"] > 0.0).any()
        evidence = transform.checkpoint_metadata()["evidence"]
        assert evidence["stage"] == "property_components"
        assert evidence["receipts"]["rows_receipts_below_profit"] == 0
        assert evidence["residuals"]["sublet_receipts"]["rows"] > 0
        assert result.mass_log[-1].reason == (
            "Landlords' receipts and finance costs on the source spine: household "
            "weights pass through unchanged and total household mass is conserved."
        )
