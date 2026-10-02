"""BRMA clone-spreading invariants, without importing a country engine."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.atomic_household_identity import household_draw_key
from microcosm.build.uk_runtime.brma_spread import (
    assign_clone_brmas,
    brmas_at_quantiles,
    household_brma_probabilities,
    lha_rate_keys,
    prepare_brma_spread,
    spaced_quantiles,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.rowwise_dataset import (
    expand_uk_geographic_pool,
    ladder_clone_index_column,
)
from microcosm.frame import engine_tables

_DISTRIBUTIONS = (
    (1.0,),
    (0.1, 0.6, 0.3, 0.0),
    (0.0, 0.2, 0.0, 0.8, 0.0),
    (0.7, 0.0, 0.1, 0.0, 0.2),
    (0.0, 1e-13, 1.0 - 1e-13, 0.0),
    (0.03, 0.08, 0.89, 0.0),
)


def _frame():
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2, 3, 4],
                "person_benunit_id": [10, 10, 20, 30],
                "person_household_id": [5, 5, 5, 9],
                "age": [40, 39, 60, 50],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [10, 20, 30], "would_claim_uc": True}),
        household=pd.DataFrame(
            {
                "household_id": [5, 9],
                "region": ["LONDON", "SCOTLAND"],
                "brma": ["LON_A", "SCOT_A"],
                "rent": [100.0, 200.0],
            }
        ),
        household_weights=[7.0, 13.0],
        time_period=2025,
    )


def _resource():
    return {
        "cells": {
            "LONDON": {
                "A": {"LON_C": 3, "LON_A": 1, "LON_ZERO": 0},
                "E": {"LON_B": 1, "LON_A": 0},
            },
            "SCOTLAND": {"B": {"SCOT_B": 2, "SCOT_A": 2}},
        }
    }


class _Engine:
    country = "uk"

    def __init__(self, *, uncapped=True):
        self.uncapped = uncapped
        self.calls = []

    def variables(self):
        # Formula-owned rates are deliberately absent from this adapter API.
        return ["brma"]

    def variable_metadata(self, name):
        if name == "uncapped_BRMA_LHA_rate" and not self.uncapped:
            raise ValueError(f"Unknown PolicyEngine-UK variable {name!r}.")
        return SimpleNamespace(entity="benunit")

    def materialize(self, frame, variables, period):
        names = frame.table("household")["brma"].to_list()
        self.calls.append((list(variables), names, period))
        if variables == ["LHA_category"]:
            return {"LHA_category": np.asarray(["A", "E", "B"])}
        rates = {"LON_A": 60, "LON_B": 30, "LON_C": 10, "SCOT_A": 100, "SCOT_B": 50}
        return {
            variables[0]: np.asarray(
                [rates[names[0]], rates[names[0]] + 20, rates[names[1]] + 7],
                dtype=float,
            )
        }


def _lineage_keys():
    return [
        household_draw_key(
            source="frs",
            source_vintage="2023_24",
            source_household_id=5,
            clone_path=(("spi_support_channel", spi), ("cgt_support_split", 0)),
        )
        for spi in (0, 1)
    ]


@pytest.mark.parametrize("probabilities", _DISTRIBUTIONS)
@pytest.mark.parametrize("k", [1, 2, 4, 15])
@pytest.mark.parametrize("ordered", [False, True])
def test_each_clone_has_exact_household_marginals_and_positive_support(
    probabilities, k, ordered
):
    grid = 6_000
    p = np.asarray(probabilities)[None, :]
    keys = np.arange(p.shape[1], dtype=float)[None, ::-1] if ordered else None
    uniforms = (np.arange(grid) + 0.5) / grid
    quantiles = spaced_quantiles(uniforms, k)
    p_grid = np.repeat(p, grid, axis=0)
    keys_grid = None if keys is None else np.repeat(keys, grid, axis=0)
    for clone in range(k):
        columns = brmas_at_quantiles(p_grid, keys_grid, quantiles[:, clone])
        frequencies = np.bincount(columns, minlength=p.shape[1]) / grid
        np.testing.assert_allclose(frequencies, p[0], rtol=0, atol=2 / grid)
        assert (p[0, columns] > 0).all()
        assert (frequencies[p[0] == 0] == 0).all()


@pytest.mark.parametrize("probabilities", _DISTRIBUTIONS)
@pytest.mark.parametrize("quantile", [0.0, 0.37, np.nextafter(1.0, 0.0)])
@pytest.mark.parametrize("ordered", [False, True])
def test_boundary_quantiles_never_choose_zero_probability(
    probabilities, quantile, ordered
):
    p = np.asarray(probabilities)[None, :]
    keys = np.arange(p.shape[1], dtype=float)[None, ::-1] if ordered else None
    columns = brmas_at_quantiles(p, keys, np.asarray([quantile]))
    assert p[0, columns[0]] > 0


@pytest.mark.parametrize("probabilities", _DISTRIBUTIONS)
@pytest.mark.parametrize("k", [1, 2, 4, 15])
@pytest.mark.parametrize("uniform", [0.0, 0.113, 0.491, 0.997])
def test_spaced_clones_bound_monotone_outcome_error(probabilities, k, uniform):
    p = np.asarray(probabilities)[None, :]
    keys = np.arange(p.shape[1], dtype=float)[None, ::-1]
    quantiles = spaced_quantiles(np.asarray([uniform]), k)
    gaps = np.diff(np.concatenate([np.sort(quantiles[0]), [quantiles.min() + 1]]))
    np.testing.assert_allclose(gaps, 1 / k, rtol=0, atol=1e-15)
    assert sorted(np.floor(quantiles[0] * k).astype(int)) == list(range(k))
    columns = brmas_at_quantiles(
        np.repeat(p, k, axis=0), np.repeat(keys, k, axis=0), quantiles.ravel()
    )
    # A non-linear monotone transform also obeys the range/K bound.
    outcome = keys[0] ** 3
    expected = p[0] @ outcome
    support = outcome[p[0] > 0]
    assert abs(outcome[columns].mean() - expected) <= np.ptp(support) / k + 1e-12


def test_walk_uses_increasing_rate_and_breaks_ties_by_column():
    p = np.asarray([[0.25, 0.25, 0.25, 0.0, 0.25]])
    keys = np.asarray([[3, 1, 1, -100, 2]])
    quantiles = np.asarray([0, 0.25, 0.5, 0.75])
    columns = brmas_at_quantiles(
        np.repeat(p, 4, axis=0), np.repeat(keys, 4, axis=0), quantiles
    )
    np.testing.assert_array_equal(columns, [1, 2, 4, 0])


def test_household_distribution_averages_units_not_people_or_cell_counts():
    frame = _frame()
    resource = _resource()
    original = copy.deepcopy(resource)
    brmas, probabilities = household_brma_probabilities(
        frame, lha_category=["A", "E", "B"], count_resource=resource
    )
    assert brmas == {
        "LONDON": ["LON_A", "LON_B", "LON_C"],
        "SCOTLAND": ["SCOT_A", "SCOT_B"],
    }
    np.testing.assert_allclose(probabilities, [[0.125, 0.5, 0.375], [0.5, 0.5, 0]])
    assert resource == original


@pytest.mark.parametrize("uncapped", [True, False])
def test_lha_rate_order_uses_household_mean_and_uncapped_fallback(uncapped):
    frame = _frame()
    original = frame.table("household").copy()
    engine = _Engine(uncapped=uncapped)
    brmas, _ = household_brma_probabilities(
        frame, lha_category=["A", "E", "B"], count_resource=_resource()
    )
    keys = lha_rate_keys(frame, engine=engine, brmas=brmas)
    np.testing.assert_array_equal(keys, [[70, 40, 20], [107, 57, np.inf]])
    variable = "uncapped_BRMA_LHA_rate" if uncapped else "BRMA_LHA_rate"
    assert [call[0] for call in engine.calls] == [[variable]] * 3
    assert [call[1] for call in engine.calls] == [
        ["LON_A", "SCOT_A"],
        ["LON_B", "SCOT_B"],
        ["LON_C", "SCOT_B"],
    ]
    pd.testing.assert_frame_equal(frame.table("household"), original)


def test_preparation_uses_pregeographic_lineage_keys_and_serializes():
    frame = _frame()
    lineage = _lineage_keys()
    payload = prepare_brma_spread(
        frame, engine=_Engine(), lineage_keys=lineage, count_resource=_resource()
    )
    assert payload["household_ids"] == [5, 9]
    assert payload["ordered_brmas"] == [
        ["LON_C", "LON_B", "LON_A"],
        ["SCOT_B", "SCOT_A", "SCOT_B"],
    ]
    np.testing.assert_allclose(
        payload["cumulative_probabilities"], [[0.375, 0.875, 1], [0.5, 1, 1]]
    )
    np.testing.assert_array_equal(
        payload["uniforms"],
        stable_identity_uniforms(lineage, seed=0, salt="brma:clone_spread"),
    )
    assert payload["uniforms"][0] != payload["uniforms"][1]
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


@pytest.mark.parametrize("k", [1, 4, 15])
def test_clone_assignment_has_regional_support_conserves_and_ignores_row_order(k):
    frame = _frame()
    payload = prepare_brma_spread(
        frame,
        engine=_Engine(),
        lineage_keys=_lineage_keys(),
        count_resource=_resource(),
    )
    tables = engine_tables(frame, weighted_entities=("household",))
    pool = expand_uk_geographic_pool(
        **tables, n_clones=k, time_period=2025, id_multiplier=1_000
    )
    household = pool.frame.table("household")
    original = household.copy()
    weights = pool.frame.weights_for("household").values.copy()
    kwargs = {
        "n_clones": k,
        "id_multiplier": pool.id_multiplier,
        "clone_index_column": ladder_clone_index_column("household"),
    }
    assigned = assign_clone_brmas(household, payload, **kwargs)
    for region, supported in {
        "LONDON": {"LON_A", "LON_B", "LON_C"},
        "SCOTLAND": {"SCOT_A", "SCOT_B"},
    }.items():
        assert set(assigned[household.region == region]) <= supported
    assert len(assigned) == len(household) == 2 * k
    pd.testing.assert_frame_equal(household, original)
    np.testing.assert_array_equal(pool.frame.weights_for("household").values, weights)
    rewritten = household.assign(brma=assigned)
    pd.testing.assert_frame_equal(
        rewritten.drop(columns="brma"), original.drop(columns="brma")
    )
    reordered = household.iloc[::-1]
    np.testing.assert_array_equal(
        assign_clone_brmas(reordered, payload, **kwargs), assigned[::-1]
    )
    # Reorder the prepared rows as well: parent lookup is by exact ID.
    reverse_payload = {key: value[::-1] for key, value in payload.items()}
    np.testing.assert_array_equal(
        assign_clone_brmas(household, reverse_payload, **kwargs), assigned
    )
    brmas, probabilities = household_brma_probabilities(
        frame, lha_category=["A", "E", "B"], count_resource=_resource()
    )
    keys = lha_rate_keys(frame, engine=_Engine(), brmas=brmas)
    quantiles = spaced_quantiles(np.asarray(payload["uniforms"]), k)
    for clone in range(k):
        columns = brmas_at_quantiles(probabilities, keys, quantiles[:, clone])
        expected = [
            brmas[region][column]
            for region, column in zip(
                frame.table("household").region, columns, strict=True
            )
        ]
        np.testing.assert_array_equal(assigned[2 * clone : 2 * (clone + 1)], expected)


def test_lineage_draws_and_quantile_assignments_are_independent_of_row_order():
    lineage = _lineage_keys()
    p = np.asarray([[0.125, 0.5, 0.375], [0.5, 0.5, 0]])
    keys = np.asarray([[70, 40, 20], [107, 57, np.inf]])
    u = stable_identity_uniforms(lineage, seed=0, salt="brma:clone_spread")
    reversed_u = stable_identity_uniforms(
        lineage[::-1], seed=0, salt="brma:clone_spread"
    )
    np.testing.assert_array_equal(reversed_u, u[::-1])
    np.testing.assert_array_equal(
        brmas_at_quantiles(p[::-1], keys[::-1], reversed_u),
        brmas_at_quantiles(p, keys, u)[::-1],
    )


@pytest.mark.parametrize("quantile", [-1, 1, np.nan, np.inf])
def test_quantiles_reject_invalid_values(quantile):
    with pytest.raises(ValueError, match="Quantiles"):
        brmas_at_quantiles(np.asarray([[1.0]]), None, np.asarray([quantile]))


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_spaced_quantiles_require_a_positive_integer(k):
    with pytest.raises(ValueError, match="positive integer"):
        spaced_quantiles(np.asarray([0.5]), k)


@pytest.mark.parametrize(
    "probabilities", [[0, 0], [-0.1, 1.1], [0.1, 0.2], [np.nan, 1]]
)
def test_inverse_cdf_rejects_invalid_probabilities(probabilities):
    with pytest.raises(ValueError, match="Probabilities"):
        brmas_at_quantiles(np.asarray([probabilities]), None, np.asarray([0.5]))


@pytest.mark.parametrize("bad_count", [-1, np.nan, np.inf])
def test_household_probabilities_reject_invalid_counts(bad_count):
    resource = _resource()
    resource["cells"]["LONDON"]["A"]["LON_A"] = bad_count
    with pytest.raises(ValueError, match="finite and non-negative"):
        household_brma_probabilities(
            _frame(), lha_category=["A", "E", "B"], count_resource=resource
        )


def test_household_probabilities_refuse_missing_and_zero_only_cells():
    with pytest.raises(KeyError, match="missing BRMA"):
        household_brma_probabilities(
            _frame(), lha_category=["A", "C", "B"], count_resource=_resource()
        )
    resource = _resource()
    resource["cells"]["LONDON"]["E"] = {"LON_B": 0}
    with pytest.raises(ValueError, match="no positive counts"):
        household_brma_probabilities(
            _frame(), lha_category=["A", "E", "B"], count_resource=resource
        )


def test_preparation_refuses_duplicate_lineage_keys():
    with pytest.raises(ValueError, match="distinct lineage key"):
        prepare_brma_spread(
            _frame(),
            engine=_Engine(),
            lineage_keys=["same", "same"],
            count_resource=_resource(),
        )


@pytest.mark.parametrize("error", ["unknown_parent", "invalid_clone", "invalid_cdf"])
def test_clone_assignment_refuses_malformed_inputs(error):
    payload = prepare_brma_spread(
        _frame(),
        engine=_Engine(),
        lineage_keys=_lineage_keys(),
        count_resource=_resource(),
    )
    household = pd.DataFrame({"household_id": [5, 9], "household_clone_index": [0, 0]})
    if error == "unknown_parent":
        household.loc[0, "household_id"] = 7
    elif error == "invalid_clone":
        household.loc[0, "household_clone_index"] = 2
    else:
        payload["cumulative_probabilities"][0][-1] = 0.99
    with pytest.raises(ValueError):
        assign_clone_brmas(
            household,
            payload,
            n_clones=2,
            id_multiplier=1_000,
            clone_index_column="household_clone_index",
        )
