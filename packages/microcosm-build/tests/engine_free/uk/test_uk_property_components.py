"""Landlords' receipts and finance costs on the UK spine (microcosm#1106)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build import load_country_spec
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


def _facts(*, landlords: float, receipts: float) -> HMRCPropertyRentalFacts:
    bounds = HMRC_PROPERTY_RENTAL_RECEIPTS_BAND_LOWER_BOUNDS
    total = float(sum(_BAND_COUNTS))
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
            for index, (lower, count) in enumerate(
                zip(bounds, _BAND_COUNTS, strict=True)
            )
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
        person["property_finance_costs"] = np.where(spi_rows, 0.2 * profit, 0.0)
        benunit = pd.DataFrame({"benunit_id": np.arange(1, households + 1)})
        return uk_national_frame(
            person=person,
            benunit=benunit,
            household=household,
            time_period="2024",
            household_weights=rng.uniform(100.0, 300.0, households),
        )

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
