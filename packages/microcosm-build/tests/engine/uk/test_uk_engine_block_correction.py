"""Per-clone engine blocks corrected to the pool's share denominator (microcosm#931).

The dense measure step resolves one engine block per clone, its household
weights scaled to the pool's mass. Once clones differ (in their keys, pay and
weights, and in which copies they hold), policyengine-uk's share formulas
would see a block's denominator, so each block's resolver sets them to the
pool's value in every simulation it builds and judges the whole-population
formulas the block ran. On a synthetic pool of three clones that differ, the
national problem and the local metrics resolved in three blocks are held to
one block over the pool: the household-local rows (the annual-income row
among them) and the local metrics bitwise, the land rows within the float32
bound, and the representation exact on its new basis. Negative controls show
what the correction and the guard catch, and identical copies resolve
bitwise as they did without the correction. Every household here is made up.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import full_measure
from microcosm.build.uk_runtime.full_measure import (
    resolve_uk_full_measures,
    resolve_uk_full_national_problem,
)
from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.rowwise_dataset import ladder_clone_index_column
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.calibrate.matrix import build_constraint_matrix
from microcosm.frame import WeightKind
from microcosm.frame.adapters import engine_population
from microcosm.frame.adapters.engine_population import (
    pool_share_atol,
    pool_share_rtol,
)
from microcosm.frame.adapters.policyengine_uk_population import UK_ENGINE_POPULATION

K = 3
SOURCES = 8
D, P = 2024, 2025
REGIONS = ("LONDON", "NORTH_WEST", "SCOTLAND", "WALES", "SOUTH_EAST")
COICOP = UK_ENGINE_POPULATION.key_adds["consumption"]
CLONE = ladder_clone_index_column("household")
#: The registry's rows: the two land rows, which read the corrected shares,
#: a household-local one, and an annual-income one (the annual simulation).
LAND = ("ons.land.corporate_land_value", "ons.land.land_value")
LOCAL = ("obr.ni_employee", "obr.income_tax")
#: Everything that reads a share formula: none may be known before the
#: shares are set.
SHARE_READERS = (
    "shareholding",
    "corporate_land_value",
    "consumption_shareholding",
    "business_rates",
    "corporate_sdlt",
    "land_value",
    "LVT",
    "household_tax",
    "household_net_income",
    "gov_tax",
)


def _pool(*, differ: bool):
    """Three clones of eight made-up households. With ``differ`` each clone
    draws its own keys, pay and weights, and clone 1 holds a ninth household,
    a copy only it carries (as an SPI copy would be)."""

    persons, benunits, households, weights = [], [], [], []
    for clone in range(K):
        sources = [*range(SOURCES), *([99] if differ and clone == 1 else [])]
        for slot, source in enumerate(sources):
            household_id = (clone + 1) * 100 + slot
            shared = np.random.default_rng(1000 + source)
            drawn = np.random.default_rng(5000 + source + (37 * clone if differ else 0))
            weight = 1000.0 * (1 + source % 4) / K
            if differ:
                weight *= float(drawn.uniform(0.8, 1.2))
            weights.append(weight)
            row = {
                "household_id": household_id,
                "household_source_id": source,
                CLONE: clone,
                "region": REGIONS[source % len(REGIONS)],
                "council_tax": 1_500.0 + 100.0 * (source % 4),
                "rent": 9_000.0 if source % 2 else 0.0,
                "tenure_type": "RENT_PRIVATELY" if source % 2 else "OWNED_OUTRIGHT",
                "main_residence_value": (
                    0.0 if source % 2 else 300_000.0 + 10_000.0 * source
                ),
                "corporate_wealth": (
                    float(drawn.lognormal(10, 1.5)) if source % 2 == 0 else 0.0
                ),
                "private_pension_wealth": (
                    float(drawn.lognormal(11, 1.0)) if source % 3 else 0.0
                ),
            }
            for column in COICOP:
                row[column] = float(drawn.uniform(300, 6_000))
            households.append(row)
            people = 1 + source % 3
            benunits.append(
                {
                    "benunit_id": household_id * 10,
                    "dependent_children": 1 if people == 3 else 0,
                    "is_married": people >= 2,
                }
            )
            for p in range(people):
                age = (44, 41, 9)[p]
                pay = np.random.default_rng(
                    9000 + 10 * source + p + (11 * clone if differ else 0)
                )
                works = age >= 18 and (source + p) % 3 != 0
                self_employed = p == 0 and source % 4 == 1
                persons.append(
                    {
                        "person_id": household_id * 10 + p,
                        "person_benunit_id": household_id * 10,
                        "person_household_id": household_id,
                        "age": age,
                        "is_benunit_head": p == 0,
                        "is_parent": people == 3 and p < 2,
                        "employment_income": (
                            float(pay.uniform(0, 70_000)) if works else 0.0
                        ),
                        "self_employment_income": (
                            float(pay.uniform(2_000, 30_000)) if self_employed else 0.0
                        ),
                        # Stored as the spine stores them (the root stage and
                        # stage 52).
                        "months_since_last_birthday": float(shared.integers(0, 12)),
                        "attends_private_school": False,
                        "attends_private_school_random_draw": float(shared.uniform()),
                        # Each adult's year: half-year entrants, and taxable
                        # other spells in every clone (job changers and a
                        # leaver, businesses), which the annual simulation
                        # carries by the engine's factor: in a block and in
                        # the pool alike only if the factor is the engine's
                        # own parameter, not a statistic of the rows.
                        "employment_income_annual_share": (
                            0.5 if p == 0 and source % 5 == 2 else 1.0
                        ),
                        "employment_income_other_spells": (
                            8_000.0 + 4_000.0 * (source % 8)
                            if p == 0 and source % 2
                            else 0.0
                        ),
                        "self_employment_income_annual_share": 1.0,
                        "self_employment_income_other_spells": (
                            12_000.0 if self_employed else 0.0
                        ),
                        "private_pension_income_annual_share": 1.0,
                        "contributory_benefits_annual_share": 1.0,
                    }
                )
    return uk_national_frame(
        person=pd.DataFrame(persons),
        benunit=pd.DataFrame(benunits),
        household=pd.DataFrame(households),
        time_period=D,
        household_weights=np.asarray(weights, dtype=float),
        weight_kind=WeightKind.IMPORTANCE,
    )


def _registry() -> TargetRegistry:
    def spec(target_id: str, entity: str, measure: str, family: str) -> TargetSpec:
        return TargetSpec(
            name=target_id,
            entity=entity,
            value=1.0e9,
            measure=measure,
            period=P,
            source="synthetic",
            family=family,
            metadata={
                "contract_target_id": target_id,
                "measure_kind": "prepared_column",
            },
        )

    return TargetRegistry(
        [
            spec(LAND[0], "household", "ons/corporate_land_value", "ons_land"),
            spec(LAND[1], "household", "ons/land_value", "ons_land"),
            spec(LOCAL[0], "person", "obr/ni_employee", "obr"),
            spec(LOCAL[1], "person", "obr/income_tax", "obr"),
        ],
        country="uk",
    )


def _without_correction(**kwargs):
    """A resolver as it was before the correction: no pool shares."""

    kwargs.pop("engine_pool_shares", None)
    return UKMeasureResolver(**kwargs)


def _rows(problem) -> dict[str, np.ndarray]:
    matrix = problem.matrix.toarray()
    return {name.split("@")[0]: matrix[row] for row, name in enumerate(problem.names)}


def _largest_block(frame) -> int:
    return int(frame.table("household")[CLONE].value_counts().max())


def _land_rtol(frame) -> float:
    """The float32 bound for corporate_land_value, one year from the data."""

    key = UK_ENGINE_POPULATION.corrected["corporate_land_value"]
    return pool_share_rtol(
        _largest_block(frame),
        frame.n("household"),
        key_inputs=len(UK_ENGINE_POPULATION.key_adds[key]),
        projection_years=P - D,
    )


@pytest.fixture(scope="module")
def differing(tmp_path_factory):
    """The differing pool resolved in one block and in three, with the
    resolvers each run built (kept, so their engines can be read after)."""

    frame = _pool(differ=True)
    runs, resolvers = {}, {}
    for blocks in (1, K):
        built = resolvers.setdefault(blocks, [])

        def keep(*, _built=built, **kwargs):
            resolver = UKMeasureResolver(**kwargs)
            _built.append(resolver)
            return resolver

        runs[blocks] = resolve_uk_full_national_problem(
            frame,
            _registry(),
            period=P,
            scratch_dir=tmp_path_factory.mktemp(f"differing-{blocks}"),
            blocks=blocks,
            resolver_factory=keep,
        )
    return frame, runs, resolvers


def test_household_local_rows_and_local_metrics_do_not_depend_on_the_blocks(
    differing,
):
    _frame, runs, _resolvers = differing
    (one, _, one_metrics, _), (three, _, three_metrics, _) = runs[1], runs[K]
    assert one.names == three.names
    np.testing.assert_array_equal(
        one.initial_weights.values, three.initial_weights.values
    )
    pool, blocked = _rows(one), _rows(three)
    for target_id in LOCAL:
        # Bitwise: income tax on each person's year included, whose other
        # spells grow by the engine's own growth parameters in every block.
        np.testing.assert_array_equal(blocked[target_id], pool[target_id])
        assert np.any(pool[target_id] != 0)
    assert set(one_metrics) == set(three_metrics) == {"constituency", "la"}
    for grain, metrics in one_metrics.items():
        pd.testing.assert_frame_equal(three_metrics[grain], metrics, check_exact=True)


def test_land_rows_are_one_engine_over_the_pool_to_float32_rounding(differing):
    frame, runs, _resolvers = differing
    pool, blocked = _rows(runs[1][0]), _rows(runs[K][0])
    rtol = _land_rtol(frame)
    corporate = pool["ons.land.corporate_land_value"]
    assert np.any(corporate != 0)
    np.testing.assert_allclose(
        blocked["ons.land.corporate_land_value"], corporate, rtol=rtol, atol=0
    )
    # land_value adds each household's own land value to its corporate share:
    # bounded per row, by the share term and the additions.
    atol = pool_share_atol(
        np.abs(corporate),
        np.abs(pool["ons.land.land_value"]) + 2 * np.abs(corporate),
        share_rtol=rtol,
        additions=2,
    )
    gap = np.abs(blocked["ons.land.land_value"] - pool["ons.land.land_value"])
    assert np.all(gap <= atol)


def test_corrected_blocks_are_exact_and_say_on_what_basis(differing):
    _frame, runs, _resolvers = differing
    receipt = runs[K][3]
    representation = receipt["engine_population_representation"]
    assert representation["exact"] is True
    assert representation["exact_basis"] == "pool_share_injection_with_population_guard"
    assert representation["mode"] == "block_weights_scaled_to_pool"
    assert representation["blocks"] == K
    # The clones are not identical copies: the old test of exactness fails,
    # and stays as a diagnostic.
    checks = representation["checks"]
    assert checks["equal_household_counts"] is False
    assert checks["weight_share_inputs_match"] is False
    correction = representation["share_correction"]
    assert correction["applied_in_every_block"] is True
    assert correction["key_sums_verified"] is True
    assert correction["uniform_growth_verified"] is True
    assert correction["growth_factors_agree_across_blocks"] is True
    assert correction["siblings_match"] is True
    assert correction["periods"] == [str(D), str(P)]
    corrections = [
        block["corporate_land_value"]
        for block in correction["corrections_by_block"].values()
    ]
    assert max(abs(c - 1.0) for c in corrections) > 1e-3
    reach = representation["population_reach"]
    assert reach["clean_in_every_block"] is True
    assert reach["refused"] == []
    assert reach["window"] == {"first_year": 2026, "violations": []}
    assert reach["classes"]["corrected"] == sorted(UK_ENGINE_POPULATION.corrected)
    assert set(reach["classes"]["stored"]) == {
        "attends_private_school",
        "months_since_last_birthday",
    }
    assert representation["annual_income_projection_factors"] == {
        "agree_across_blocks": True,
        "max_rel_spread": 0.0,
    }
    sensitivity = receipt["block_sensitivity"]
    assert sensitivity["exact"] is True and sensitivity["caveat"] is None
    assert sensitivity["present_in_this_run"] == ["corporate_land_value", "land_value"]
    # Each block's own resolution receipt carries its correction; the
    # single-block run's carries none of it.
    for resolution in receipt["resolution"]:
        assert resolution["provider"]["engine_pool_shares"]["applied"] is True
    single = runs[1][3]
    assert "engine_population_representation" not in single
    for resolution in single["resolution"]:
        assert "engine_pool_shares" not in resolution["provider"]
        assert "engine_population_reach" not in resolution["provider"]


#: The corrected formulas and two values that multiply a share by a national
#: total, each with the allocation key its share divides by.
SHARE_VALUES = (
    ("shareholding", "corporate_sector_wealth"),
    ("consumption_shareholding", "consumption"),
    ("business_rates", "corporate_sector_wealth"),
    ("corporate_sdlt", "corporate_sector_wealth"),
)


def test_every_corrected_formula_takes_the_pools_value_in_every_block(differing):
    """Each block's correction is ``f_b * S_b / S`` for every share formula,
    and in each block's engine the three share formulas, and what multiplies
    a share by a national total, take the pool's value for every household
    to float32 rounding (microcosm#931)."""

    _frame, runs, resolvers = differing
    representation = runs[K][3]["engine_population_representation"]
    factors = representation["factor_by_block"]
    inputs = representation["checks"]["weight_share_inputs"]
    corrections = representation["share_correction"]["corrections_by_block"]
    assert set(corrections) == set(factors)
    for block, by_formula in corrections.items():
        assert set(by_formula) == set(UK_ENGINE_POPULATION.corrected)
        for formula, correction in by_formula.items():
            sums = inputs[formula]["sum_by_block"]
            expected = factors[block] * sums[block] / math.fsum(sums.values())
            assert correction == pytest.approx(expected, rel=1e-12)
    (pool,) = resolvers[1]
    assert len(resolvers[K]) == K
    pool_ids = np.asarray(pool.simulation.calculate("household_id", P))
    for name, key in SHARE_VALUES:
        reference = pd.Series(
            np.asarray(pool.simulation.calculate(name, P), dtype=np.float64),
            index=pool_ids,
        )
        assert np.any(reference.to_numpy() != 0), name
        for resolver in resolvers[K]:
            ids = np.asarray(resolver.simulation.calculate("household_id", P))
            values = np.asarray(
                resolver.simulation.calculate(name, P), dtype=np.float64
            )
            rtol = pool_share_rtol(
                len(ids),
                len(pool_ids),
                key_inputs=len(UK_ENGINE_POPULATION.key_adds[key]),
                projection_years=P - D,
            )
            np.testing.assert_allclose(
                values, reference.loc[ids].to_numpy(), rtol=rtol, atol=0, err_msg=name
            )


def test_the_measures_entry_point_corrects_its_blocks_too(differing, tmp_path):
    frame, runs, _resolvers = differing
    registry = _registry()
    prepared, _restore, _rows_, metrics, receipt = resolve_uk_full_measures(
        frame, registry, period=P, scratch_dir=tmp_path, blocks=K
    )
    assert receipt["engine_population_representation"]["exact"] is True
    problem = build_constraint_matrix(prepared, registry.to_target_set(), "household")
    pool, measured = _rows(runs[1][0]), _rows(problem)
    for target_id in LOCAL:
        np.testing.assert_array_equal(measured[target_id], pool[target_id])
    np.testing.assert_allclose(
        measured["ons.land.corporate_land_value"],
        pool["ons.land.corporate_land_value"],
        rtol=_land_rtol(frame),
        atol=0,
    )
    for grain, values in runs[1][2].items():
        pd.testing.assert_frame_equal(metrics[grain], values, check_exact=True)


def test_without_the_correction_the_land_rows_miss_the_pool(differing, tmp_path):
    frame, runs, _resolvers = differing
    _problem, _rows_, _metrics, receipt = resolve_uk_full_national_problem(
        frame,
        _registry(),
        period=P,
        scratch_dir=tmp_path,
        blocks=K,
        resolver_factory=_without_correction,
        local_grains=(),
    )
    representation = receipt["engine_population_representation"]
    assert representation["exact"] is False
    assert representation["share_correction"]["applied_in_every_block"] is False
    pool = _rows(runs[1][0])["ons.land.corporate_land_value"]
    blocked = _rows(_problem)["ons.land.corporate_land_value"]
    carried = pool != 0
    assert np.max(np.abs(blocked[carried] / pool[carried] - 1.0)) > 1e-2


@pytest.fixture(scope="module")
def identical(tmp_path_factory):
    frame = _pool(differ=False)
    return {
        label: resolve_uk_full_national_problem(
            frame,
            _registry(),
            period=P,
            scratch_dir=tmp_path_factory.mktemp(f"identical-{label}"),
            blocks=K,
            resolver_factory=factory,
        )
        for label, factory in (
            ("corrected", UKMeasureResolver),
            ("uncorrected", _without_correction),
        )
    }


def test_identical_copies_resolve_bitwise_as_they_did_without_the_correction(
    identical,
):
    """Today's dense line: clones are identical copies, so every block's
    correction is 1 and the arrays it sets are the engine's own, bitwise; the
    matrix and the local metrics do not move (microcosm#931)."""

    corrected, uncorrected = identical["corrected"], identical["uncorrected"]
    assert corrected[0].names == uncorrected[0].names
    assert (corrected[0].matrix != uncorrected[0].matrix).nnz == 0
    np.testing.assert_array_equal(
        corrected[0].target_vector, uncorrected[0].target_vector
    )
    for grain, metrics in uncorrected[2].items():
        pd.testing.assert_frame_equal(corrected[2][grain], metrics, check_exact=True)
    representation = corrected[3]["engine_population_representation"]
    assert representation["exact"] is True
    assert representation["checks"]["identical_source_weight_multisets"] is True
    for block in representation["share_correction"]["corrections_by_block"].values():
        for correction in block.values():
            assert correction == pytest.approx(1.0, abs=1e-12)


def _block_resolver(frame, tmp_path, *, clone=0, year=P, resolver=UKMeasureResolver):
    block_frames = dict(full_measure._engine_block_frames(frame, K))
    representation = full_measure._engine_population_representation(
        frame, list(block_frames.items())
    )
    shares = full_measure._block_pool_shares(representation)
    return resolver(
        simulation_source=None,
        scratch_dir=tmp_path,
        year=year,
        frame=block_frames[clone],
        engine_weight_scale=representation["factor_by_block"][str(clone)],
        engine_identifier_universe=frame,
        engine_pool_shares=shares[str(clone)],
    )


def test_no_share_reader_is_known_before_the_shares_are_set(monkeypatch, tmp_path):
    frame = _pool(differ=True)
    known_before: list[dict[str, list]] = []
    inject = engine_population.inject_pool_shares

    def watching(simulation, shares, periods, *, weight_variable):
        names = [
            name
            for name in SHARE_READERS
            if name in simulation.tax_benefit_system.variables
        ]
        assert set(names) == set(SHARE_READERS)
        for target in (simulation, simulation.baseline):
            known_before.append(
                {
                    name: list(target.get_holder(name).get_known_periods())
                    for name in names
                    if target.get_holder(name).get_known_periods()
                }
            )
        return inject(simulation, shares, periods, weight_variable=weight_variable)

    # The shared EngineBlock injects (microcosm#931).
    monkeypatch.setattr(engine_population, "inject_pool_shares", watching)
    resolver = _block_resolver(frame, tmp_path)
    assert known_before == [{}, {}]
    shares = resolver.receipt()["engine_pool_shares"]
    assert shares["applied"] is True and shares["refusals"] == []
    # A counterfactual that moves an allocation key is refused outright.
    with pytest.raises(ValueError, match="pool-share correction reads"):
        resolver.counterfactual_delta(
            {
                "kind": "input_substitution_counterfactual",
                "zeroed_input": "corporate_wealth",
                "output_variable": "income_tax",
            },
            P,
        )
    resolver.compute("household", "corporate_land_value")
    resolver.close_engine_block()
    reach = resolver.receipt()["engine_population_reach"]
    assert reach["clean"] is True, reach["violations"]


class _ReadsAnIncomeDecile(UKMeasureResolver):
    """A block whose measures reached a rank over its own households."""

    def close_engine_block(self):
        self.compute("household", "household_income_decile")
        super().close_engine_block()


def test_a_block_that_reads_a_rank_is_not_exact_and_names_it(tmp_path):
    frame = _pool(differ=True)
    _problem, _rows_, _metrics, receipt = resolve_uk_full_national_problem(
        frame,
        TargetRegistry([], country="uk"),
        period=P,
        scratch_dir=tmp_path,
        blocks=K,
        resolver_factory=_ReadsAnIncomeDecile,
        local_grains=(),
    )
    representation = receipt["engine_population_representation"]
    assert representation["exact"] is False
    reach = representation["population_reach"]
    assert reach["clean_in_every_block"] is False
    assert reach["refused"] == ["household_income_decile"]
    assert receipt["block_sensitivity"]["caveat"] is not None


def test_a_block_measured_in_a_rebalancing_year_is_refused(tmp_path):
    """policyengine-uk draws which Universal Credit claimants are new by
    benefit-unit position from 2026, so a block measured there is refused,
    corrected or not: recorded when the block is built and by its verdict."""

    resolver = _block_resolver(_pool(differ=True), tmp_path, year=2026)
    shares = resolver.receipt()["engine_pool_shares"]
    assert shares["periods"] == [str(D), "2026"]
    assert len(shares["refusals"]) == 1
    assert "from 2026 on" in shares["refusals"][0]
    resolver.close_engine_block()
    receipt = resolver.receipt()
    reach = receipt["engine_population_reach"]
    assert reach["clean"] is False
    assert {"kind": "window_period", "name": None, "period": "2026"} in reach["window"][
        "violations"
    ]
