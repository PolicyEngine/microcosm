"""The UC new-claimant health element draw, by claim recency (microcosm#1095)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime import uc_new_claimant_draw as stage_module
from microcosm.build.uk_runtime.frs_release import resolve_uk_year_rule
from microcosm.build.uk_runtime.frs_spine import (
    FRS_UC_CLAIM_MONTHS_COLUMN,
    FRS_UC_CLAIM_MONTHS_NOT_LINKED,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.uc_new_claimant_draw import (
    UC_NEW_CLAIMANT_DRAW_OUTPUT,
    UC_NEW_CLAIMANT_DRAW_STAGE_NAME,
    UC_NEW_CLAIMANT_ENGINE_VARIABLES,
    UC_NEW_CLAIMANT_MASS_CHANGE_REASON,
    UKUCNewClaimantDrawStageTransform,
    assign_uc_new_claimant_draw,
    claim_month_pools,
    weighted_quantile_draws,
    weighted_rank_midpoints,
)
from microcosm.frame import WeightKind

NOT_LINKED = FRS_UC_CLAIM_MONTHS_NOT_LINKED

# One row per benefit unit, each in its own household: claim months, the
# youngest claimant's age, whether the engine pays the award with the LCWRA
# element, and whether the household is an SPI-channel copy.
_UNITS = [
    # 1-4 and 6: linked claims in the population.
    (24.0, 40, True, False),
    (3.0, 30, True, False),
    (3.0, 22, True, False),
    (60.0, 50, True, False),
    # 5: outside the population (no award with the element).
    (1.0, 19, False, False),
    (0.0, 45, True, False),
]


def _frame(units=_UNITS, *, weights=None, carrier: bool = True, role: bool = True):
    count = len(units)
    ids = np.arange(1, count + 1)
    benunit = pd.DataFrame({"benunit_id": ids})
    if carrier:
        benunit[FRS_UC_CLAIM_MONTHS_COLUMN] = [row[0] for row in units]
    # Each unit has its claimant and a child, who is never a claimant.
    person = pd.DataFrame(
        {
            "person_id": np.concatenate([ids * 10, ids * 10 + 1]),
            "person_benunit_id": np.concatenate([ids, ids]),
            "person_household_id": np.concatenate([ids, ids]),
            "age": np.concatenate([[row[1] for row in units], np.full(count, 5)]),
        }
    )
    # Without ``role`` nobody is flagged a claimant or partner.
    person["is_claimant_or_partner"] = np.concatenate(
        [np.full(count, role, dtype=bool), np.zeros(count, dtype=bool)]
    )
    household = pd.DataFrame(
        {
            "household_id": ids,
            "region": ["WALES"] * count,
            "household_source_id": ids,
            # The true source each floor counts (microcosm#931).
            "source_household_id": ids,
            "household_is_spi_synthetic": [row[3] for row in units],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        household_weights=(
            np.full(count, 1000.0) if weights is None else np.asarray(weights)
        ),
        weight_kind=WeightKind.IMPORTANCE,
        time_period="2024",
    )


def _answers(units=_UNITS) -> dict[str, np.ndarray]:
    paid = np.asarray([row[2] for row in units])
    return {
        "universal_credit": np.where(paid, 6_000.0, 0.0),
        "uc_LCWRA_element": np.where(paid, 5_000.0, 0.0),
    }


def _identity(ids) -> np.ndarray:
    return stable_identity_uniforms(
        np.asarray(ids), seed=0, salt=UC_NEW_CLAIMANT_DRAW_OUTPUT
    )


def test_the_population_is_ordered_by_recency_then_youth() -> None:
    result = assign_uc_new_claimant_draw(_frame(), engine_answers=_answers())
    draws = result.frame.table("benunit")[UC_NEW_CLAIMANT_DRAW_OUTPUT].to_numpy()

    # Five units of equal weight: unit 6 (0 months), unit 3 (3 months, aged
    # 22), unit 2 (3 months, aged 30), unit 1 (24) and unit 4 (60) take the
    # midpoints of fifths, in that order.
    np.testing.assert_allclose(draws[[5, 2, 1, 0, 3]], [0.1, 0.3, 0.5, 0.7, 0.9])
    # The unit outside the population keeps its identity uniform.
    assert draws[4] == _identity([5])[0]
    assert result.frame.weights_for("household").total == 6000.0
    assert result.frame.mass_log[-1].reason == UC_NEW_CLAIMANT_MASS_CHANGE_REASON


def test_the_draw_is_uniform_within_the_population_at_its_weights() -> None:
    rng = np.random.default_rng(7)
    count = 400
    units = [
        (
            float(rng.integers(0, 120)),
            int(rng.integers(18, 65)),
            bool(rng.random() < 0.6),
            False,
        )
        for _ in range(count)
    ]
    weights = rng.uniform(100.0, 3000.0, count)
    result = assign_uc_new_claimant_draw(
        _frame(units, weights=weights), engine_answers=_answers(units)
    )
    draws = result.frame.table("benunit")[UC_NEW_CLAIMANT_DRAW_OUTPUT].to_numpy()
    population = np.asarray([row[2] for row in units])
    total = weights[population].sum()
    for probe in np.linspace(0.05, 0.95, 19):
        share = weights[population & (draws < probe)].sum() / total
        assert abs(share - probe) <= weights[population].max() / total
    assert ((draws >= 0.0) & (draws < 1.0)).all()
    np.testing.assert_array_equal(
        draws[~population], _identity(np.flatnonzero(~population) + 1)
    )
    # Recency leads: within the population the draw rises with the months.
    months = np.asarray([row[0] for row in units])[population]
    order = np.argsort(draws[population], kind="stable")
    assert (np.diff(months[order]) >= 0).all()


def test_units_without_a_linked_start_take_months_from_their_age_band(
    monkeypatch,
) -> None:
    monkeypatch.setattr(stage_module, "UC_NEW_CLAIMANT_MINIMUM_POOL_SOURCES", 1)
    units = [
        (2.0, 20, True, False),  # young donors
        (5.0, 23, True, False),
        (NOT_LINKED, 21, True, False),  # a young entrant
        (80.0, 40, True, False),  # older donors
        (90.0, 41, True, False),
        (NOT_LINKED, 42, True, False),  # an older entrant
        # An SPI-channel copy carrying its twin's months: never linked.
        (1.0, 44, True, True),
    ]
    result = assign_uc_new_claimant_draw(_frame(units), engine_answers=_answers(units))

    assert result.claim_months[2] in (2.0, 5.0)
    assert result.claim_months[5] in (80.0, 90.0)
    assert result.claim_months[6] in (80.0, 90.0)
    draws = result.frame.table("benunit")[UC_NEW_CLAIMANT_DRAW_OUTPUT].to_numpy()
    # The young entrant sits among the young claims, below every older one.
    assert draws[2] < draws[[3, 4, 5, 6]].min()


def test_a_band_without_enough_linked_claims_pools_with_its_neighbour() -> None:
    bands = np.array([0, 0, 1, 2, 3, 4])
    donors = np.ones(6, dtype=bool)
    sources = np.array([1, 2, 3, 4, 5, 6])
    weights = np.ones(6)

    assert claim_month_pools(
        bands, donors, sources=sources, weights=weights, minimum_sources=2
    ) == ((0,), (1, 2), (3, 4))
    assert claim_month_pools(
        bands, donors, sources=sources, weights=weights, minimum_sources=99
    ) == ((0, 1, 2, 3, 4),)
    # Copies of one source household count once.
    assert claim_month_pools(
        bands,
        donors,
        sources=np.array([1, 1, 2, 2, 3, 3]),
        weights=weights,
        minimum_sources=2,
    ) == ((0, 1), (2, 3, 4))


def test_the_weighted_quantile_and_rank_helpers() -> None:
    values = np.array([30.0, 10.0, 20.0, 40.0])
    weights = np.array([1.0, 1.0, 2.0, 0.0])
    draws = weighted_quantile_draws(values, weights, np.array([0.1, 0.3, 0.74, 0.75]))
    assert draws.tolist() == [10.0, 20.0, 20.0, 30.0]
    assert 40.0 not in weighted_quantile_draws(
        values, weights, np.linspace(0, 0.99, 50)
    )
    with pytest.raises(ValueError, match="at least one"):
        weighted_quantile_draws(np.array([]), np.array([]), np.array([0.5]))

    midpoints = weighted_rank_midpoints(np.array([2, 0, 1]), np.array([1.0, 1.0, 2.0]))
    np.testing.assert_allclose(midpoints, [0.625, 0.875, 0.25])
    even = weighted_rank_midpoints(np.array([1, 0]), np.zeros(2))
    np.testing.assert_allclose(even, [0.75, 0.25])


def test_the_receipt_bands_its_cells() -> None:
    receipt = assign_uc_new_claimant_draw(_frame(), engine_answers=_answers()).receipt

    assert receipt["population"]["benefit_units"] == "3-9"
    assert receipt["population"]["support"] == "3-9"
    # Every unit of the population is linked: the unlinked part is empty, so
    # the share withholds nothing.
    assert receipt["linked_share"] == 1.0
    assert receipt["pools"] == [
        {
            "age_bands": ["16-24", "25-34", "35-44", "45-54", "55+"],
            "linked_support": "3-9",
            "imputed_benefit_units": 0,
            "imputed_support": 0,
        }
    ]
    assert receipt["first_band_support"] == "<3"
    assert receipt["first_band_share_below"] == ["<3", "<3", "<3"]


def test_the_stage_refuses_what_it_cannot_read() -> None:
    with pytest.raises(KeyError, match=FRS_UC_CLAIM_MONTHS_COLUMN):
        assign_uc_new_claimant_draw(_frame(carrier=False), engine_answers=_answers())
    stored = assign_uc_new_claimant_draw(_frame(), engine_answers=_answers()).frame
    with pytest.raises(ValueError, match="only writer"):
        assign_uc_new_claimant_draw(stored, engine_answers=_answers())
    for odd in (2.5, -3.0, np.nan):
        units = [(odd, 30, True, False), *_UNITS[1:]]
        with pytest.raises(ValueError, match="whole months"):
            assign_uc_new_claimant_draw(_frame(units), engine_answers=_answers(units))
    with pytest.raises(ValueError, match="no claimant"):
        assign_uc_new_claimant_draw(_frame(role=False), engine_answers=_answers())
    short = {name: values[:-1] for name, values in _answers().items()}
    with pytest.raises(ValueError, match="shape"):
        assign_uc_new_claimant_draw(_frame(), engine_answers=short)


class _Engine:
    def __init__(self, answers: dict[str, np.ndarray]) -> None:
        self.answers = answers
        self.calls: list[tuple[tuple[str, ...], object, object]] = []

    def materialize(self, frame, variables, period, *, dataset_period=None):
        self.calls.append((tuple(variables), period, dataset_period))
        return {name: self.answers[name] for name in variables}


def _stage():
    return load_country_spec("uk").sources.stage_map()[UC_NEW_CLAIMANT_DRAW_STAGE_NAME]


def test_the_transform_reads_the_award_at_the_calibration_year() -> None:
    engine = _Engine(_answers())
    transform = UKUCNewClaimantDrawStageTransform(stage=_stage(), engine=engine)

    out = transform(_frame())

    assert engine.calls == [
        (
            UC_NEW_CLAIMANT_ENGINE_VARIABLES,
            resolve_uk_year_rule("calibration_year"),
            "2024",
        )
    ]
    assert transform.output_columns() == (UC_NEW_CLAIMANT_DRAW_OUTPUT,)
    assert UC_NEW_CLAIMANT_DRAW_OUTPUT in out.table("benunit").columns
    evidence = transform.checkpoint_metadata()["evidence"]
    assert evidence["stage"] == UC_NEW_CLAIMANT_DRAW_STAGE_NAME
    assert evidence["entitlement_year"] == resolve_uk_year_rule("calibration_year")


def test_the_committed_stage_binds_its_operations_closed_world() -> None:
    stage = _stage()
    operations = list(stage.operations)
    assert [operation.kind for operation in operations] == [
        "materialize_rules_engine_predictors",
        "assign_uc_new_claimant_draw",
        "record_mass_conservation_receipt",
    ]
    assert tuple(stage.outputs) == (UC_NEW_CLAIMANT_DRAW_OUTPUT,)
    drifted = replace(
        operations[1],
        parameters={**dict(operations[1].parameters), "minimum_pool_sources": 10},
    )
    with pytest.raises(ValueError, match="parameters drifted"):
        UKUCNewClaimantDrawStageTransform(
            stage=replace(stage, operations=(operations[0], drifted, operations[2])),
            engine=_Engine(_answers()),
        )
    # The last stage of the spine, after every writer of an award input.
    names = [entry.stage for entry in load_country_spec("uk").sources.stages]
    assert names[-1] == UC_NEW_CLAIMANT_DRAW_STAGE_NAME
