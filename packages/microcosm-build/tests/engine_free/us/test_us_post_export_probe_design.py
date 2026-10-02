"""The post-export probe's design statistics (``tools/probe_us_post_export.py``).

Engine-free and release-tool-free: these tests load only the probe and the
sampler. Invariants, each checked for every generated input:

- unbiasedness, by exact enumeration: with equal weights inside each stratum
  the sampler's ratio estimator is the stratified expansion estimator, so over
  every equally likely sample its mean is the population total and the mean of
  ``stratified_ratio_variance`` is the estimator's exact design variance;
- the variance is non-negative, zero for a census or certainty-only sample,
  ``nan`` for a stratum that drew one household of several, and scales as the
  square of the values;
- a differential check against the sampler: ``draw_households`` and the
  variance agree with Monte Carlo over many seeds (unequal weights);
- the design rebuilt from a written subsample and its receipt verifies, and
  recovers every sampled household's source weight;
- ``choose_batch_size`` always yields at least three batches and never exceeds
  the requested size;
- ``classify_probe``: a census is authoritative, a probe without a standard
  error or with too few sampled carriers is informational, and otherwise the
  verdict is authoritative exactly when the margin to the floor is at least
  ``k`` standard errors.
"""

# ruff: noqa: F403, F405
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from test_support.microcosm_build.us_export_subsample import *

_SETTINGS = settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

_FINITE = st.floats(min_value=-1_000.0, max_value=1_000.0, allow_nan=False)


# ---------------------------------------------------------------------------
# stratified_ratio_variance: exact enumeration
# ---------------------------------------------------------------------------


@st.composite
def equal_weight_designs(draw):
    """One or two strata of N households with one weight each, n of N drawn,
    plus a few certainty households in the first stratum."""
    strata = []
    for _ in range(draw(st.integers(min_value=1, max_value=2))):
        size = draw(st.integers(min_value=2, max_value=6))
        drawn = draw(st.integers(min_value=2, max_value=size))
        weight = draw(st.floats(min_value=0.5, max_value=100.0))
        values = draw(st.lists(_FINITE, min_size=size, max_size=size))
        strata.append((size, drawn, weight, values))
    certainty = draw(
        st.lists(
            st.tuples(st.floats(min_value=0.1, max_value=50.0), _FINITE),
            max_size=3,
        )
    )
    return strata, certainty


def _sample_arrays(strata, certainty, chosen):
    """Arrays of one sample: certainty households, then each stratum's draw."""
    labels = [f"h{index}" for index in range(len(strata))]
    values = [y for _, y in certainty]
    weights = [w for w, _ in certainty]
    stratum_of = [labels[0]] * len(certainty)
    is_certain = [True] * len(certainty)
    adjusted = list(weights)
    records = {}
    for label, (size, drawn, weight, population), members in zip(
        labels, strata, chosen, strict=True
    ):
        factor = size / drawn
        for member in members:
            values.append(population[member])
            weights.append(weight)
            stratum_of.append(label)
            is_certain.append(False)
            adjusted.append(weight * factor)
        records[label] = {
            "eligible_noncertainty_households": size,
            "drawn_noncertainty_households": drawn,
            "noncertainty_weight_factor": factor,
        }
    return (
        np.asarray(values, np.float64),
        np.asarray(weights, np.float64),
        np.asarray(adjusted, np.float64),
        np.asarray(stratum_of, dtype=object),
        np.asarray(is_certain, dtype=bool),
        records,
    )


@_SETTINGS
@given(equal_weight_designs())
def test_variance_is_the_exact_design_variance_with_equal_weights(
    probe_tool, design
) -> None:
    strata, certainty = design
    population_total = sum(w * y for w, y in certainty) + sum(
        weight * sum(values) for _, _, weight, values in strata
    )
    estimates = []
    variances = []
    for chosen in itertools.product(
        *(itertools.combinations(range(size), drawn) for size, drawn, _, _ in strata)
    ):
        values, weights, adjusted, labels, is_certain, records = _sample_arrays(
            strata, certainty, chosen
        )
        estimates.append(float(np.dot(adjusted, values)))
        variances.append(
            probe_tool.stratified_ratio_variance(
                values,
                source_weights=weights,
                labels=labels,
                certainty=is_certain,
                strata=records,
            )
        )
    estimates = np.asarray(estimates)
    scale = 1.0 + abs(population_total) + float(np.abs(estimates).max())
    assert estimates.mean() == pytest.approx(
        population_total, rel=1e-9, abs=1e-9 * scale
    )
    design_variance = float(np.mean((estimates - population_total) ** 2))
    assert float(np.mean(variances)) == pytest.approx(
        design_variance, rel=1e-7, abs=1e-9 * scale**2
    )


# ---------------------------------------------------------------------------
# stratified_ratio_variance: structural properties
# ---------------------------------------------------------------------------


@st.composite
def drawn_samples(draw):
    """Unequal weights; every stratum draws n of N households."""
    labels, weights, values, records = [], [], [], {}
    for index in range(draw(st.integers(min_value=1, max_value=3))):
        label = f"s{index}"
        size = draw(st.integers(min_value=2, max_value=40))
        drawn = draw(st.integers(min_value=1, max_value=size))
        records[label] = {
            "eligible_noncertainty_households": size,
            "drawn_noncertainty_households": drawn,
        }
        labels += [label] * drawn
        weights += draw(
            st.lists(
                st.floats(min_value=0.01, max_value=1e4), min_size=drawn, max_size=drawn
            )
        )
        values += draw(st.lists(_FINITE, min_size=drawn, max_size=drawn))
    return (
        np.asarray(values, np.float64),
        np.asarray(weights, np.float64),
        np.asarray(labels, dtype=object),
        records,
    )


@_SETTINGS
@given(drawn_samples(), st.floats(min_value=-50.0, max_value=50.0))
def test_variance_structure(probe_tool, sample, scale) -> None:
    values, weights, labels, records = sample
    certainty = np.zeros(len(values), dtype=bool)

    def variance(y, certain=certainty, strata=records):
        return probe_tool.stratified_ratio_variance(
            y, source_weights=weights, labels=labels, certainty=certain, strata=strata
        )

    base = variance(values)
    single = any(
        1
        == record["drawn_noncertainty_households"]
        < record["eligible_noncertainty_households"]
        for record in records.values()
    )
    if single:
        assert math.isnan(base)
        return
    assert base >= 0.0
    assert variance(scale * values) == pytest.approx(
        scale**2 * base, rel=1e-9, abs=1e-12 * (1.0 + base)
    )
    census = {
        label: {
            **record,
            "eligible_noncertainty_households": record["drawn_noncertainty_households"],
        }
        for label, record in records.items()
    }
    assert variance(values, strata=census) == 0.0
    assert variance(values, certain=np.ones(len(values), dtype=bool)) == 0.0


def test_variance_matches_monte_carlo_over_sampler_draws(sampler, probe_tool) -> None:
    """Differential: over 600 seeded draws of ``draw_households`` (unequal
    weights, two strata, three certainty households), the ratio estimator is
    nearly unbiased and the mean linearized variance tracks the realized one."""
    rng = np.random.default_rng(20261002)
    n = 400
    ids = np.arange(1, n + 1, dtype=np.int64)
    weights = rng.lognormal(mean=4.0, sigma=0.6, size=n)
    labels = np.where(ids <= n // 2, "asec", "puf").astype(object)
    values = rng.gamma(shape=0.6, scale=900.0, size=n) * (rng.random(n) < 0.4)
    certain = [7, 211, 399]
    total = float(np.dot(weights, values))
    estimates, variances = [], []
    for seed in range(600):
        drawn = sampler.draw_households(
            ids, weights, labels, certain, fraction=0.1, seed=seed
        )
        position = pd.Series(np.arange(n), index=ids).reindex(drawn.selected_ids)
        y = values[position.to_numpy()]
        design = probe_tool.SampleDesign(
            household_ids=drawn.selected_ids,
            labels=drawn.labels,
            certainty=drawn.certainty,
            adjusted_weights=drawn.adjusted_weights,
            source_weights=drawn.source_weights,
            strata=drawn.strata,
            fraction=0.1,
        )
        assert design.verify() == []
        effects = pd.Series(y, index=drawn.selected_ids)
        estimates.append(design.total(effects))
        variances.append(design.variance(effects))
    estimates = np.asarray(estimates)
    spread = float(estimates.std(ddof=1))
    assert abs(float(estimates.mean()) - total) < 4.0 * spread / math.sqrt(600)
    ratio = float(np.mean(variances)) / float(estimates.var(ddof=1))
    assert 0.8 < ratio < 1.25, ratio


# ---------------------------------------------------------------------------
# The design rebuilt from a written subsample
# ---------------------------------------------------------------------------


@settings(max_examples=8, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.integers(min_value=0, max_value=10_000),
    st.floats(min_value=0.1, max_value=0.9),
    st.integers(min_value=0, max_value=99),
)
def test_design_rebuilt_from_a_subsample_verifies(
    sampler, probe_tool, tmp_path_factory, frame_seed, fraction, seed
) -> None:
    frame = synthetic_export_frame(30, seed=frame_seed, rare_households=(2, 17))
    path, receipt = sample_synthetic(
        sampler,
        tmp_path_factory.mktemp("design"),
        frame,
        fraction=fraction,
        seed=seed,
    )
    written = load_table_h5(path)
    design = probe_tool.design_from_sample(
        written.table("household"),
        written.table("person"),
        written.weights_for("household").values,
        receipt,
        sampler=sampler,
    )
    assert design.verify() == []
    source = pd.Series(
        frame.weights_for("household").values,
        index=frame.table("household")["household_id"].to_numpy(),
    )
    assert np.allclose(
        design.source_weights,
        source.reindex(design.household_ids).to_numpy(),
        rtol=1e-12,
        atol=0.0,
    )
    assert set(design.household_ids[design.certainty].tolist()) == set(
        receipt["certainty"]["household_ids"]
    )
    ones = pd.Series(1.0, index=design.household_ids)
    assert design.total(ones) == pytest.approx(source.sum(), rel=1e-9)
    assert design.is_census == all(
        record["drawn_noncertainty_households"]
        == record["eligible_noncertainty_households"]
        for record in receipt["strata"].values()
    )
    census = probe_tool.design_from_sample(
        written.table("household"),
        written.table("person"),
        written.weights_for("household").values,
        None,
        sampler=sampler,
    )
    assert census.is_census and census.variance(ones) == 0.0
    with pytest.raises(ValueError, match="not in the sample design"):
        design.total(pd.Series([1.0], index=[10**9]))


# ---------------------------------------------------------------------------
# Batch size and authority rules
# ---------------------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(
    st.integers(min_value=3, max_value=2_000_000),
    st.one_of(st.none(), st.integers(min_value=1, max_value=100_000)),
)
def test_batch_size_always_gives_three_batches(probe_tool, n, requested) -> None:
    size = probe_tool.choose_batch_size(n, requested)
    assert size >= 1
    assert requested is None or size <= requested
    assert probe_tool.batch_count(n, size) >= probe_tool.MINIMUM_BATCHES
    if requested is None or size < requested:
        # Maximal: one more household per batch would lose the third batch.
        assert probe_tool.batch_count(n, size + 1) < probe_tool.MINIMUM_BATCHES


def test_batch_size_refuses_too_few_households(probe_tool) -> None:
    with pytest.raises(ValueError, match="cannot fill 3 batches"):
        probe_tool.choose_batch_size(2, 2_000)
    with pytest.raises(ValueError, match="batch size must be positive"):
        probe_tool.choose_batch_size(10, 0)


@settings(max_examples=400, deadline=None)
@given(
    magnitude=st.floats(min_value=0.0, max_value=1e9),
    floor=st.floats(min_value=0.0, max_value=1e9),
    standard_error=st.one_of(
        st.none(),
        st.just(math.nan),
        st.just(math.inf),
        st.just(0.0),
        st.floats(min_value=0.0, max_value=1e8),
    ),
    take_all=st.booleans(),
    drawn=st.one_of(st.none(), st.integers(min_value=0, max_value=10_000)),
    effective=st.one_of(st.none(), st.floats(min_value=0.0, max_value=10_000.0)),
    census=st.booleans(),
    k=st.floats(min_value=0.5, max_value=8.0),
)
def test_classify_probe_rules(
    probe_tool, magnitude, floor, standard_error, take_all, drawn, effective, census, k
) -> None:
    authority, reason = probe_tool.classify_probe(
        signed_magnitude=magnitude,
        floor=floor,
        standard_error=standard_error,
        take_all=take_all,
        drawn_effect_households=drawn,
        effective_households=effective,
        census=census,
        se_multiplier=k,
        min_effective_households=100.0,
    )
    assert authority in (probe_tool.AUTHORITATIVE, probe_tool.INFORMATIONAL)
    assert reason
    if census:
        assert authority == probe_tool.AUTHORITATIVE
    elif standard_error is None or not math.isfinite(standard_error):
        assert authority == probe_tool.INFORMATIONAL
    elif take_all and not drawn:
        # Every pool carrier is certain and no drawn household carries an
        # effect: the effect is the certainty households' exactly.
        assert authority == probe_tool.AUTHORITATIVE
    elif (effective or 0.0) < 100.0:
        # A variance resting on too few effective households (none included)
        # bounds nothing, whatever it came out as.
        assert authority == probe_tool.INFORMATIONAL
    else:
        expected = abs(magnitude - floor) >= k * standard_error
        assert (authority == probe_tool.AUTHORITATIVE) == expected


def test_a_sample_without_effect_bearing_draws_is_not_authoritative(
    probe_tool,
) -> None:
    """The false-authoritative case: a rare effect none of whose households
    was drawn estimates SE 0 and an effect of 0 (far below the floor)."""
    authority, reason = probe_tool.classify_probe(
        signed_magnitude=0.0,
        floor=1e6,
        standard_error=0.0,
        take_all=False,
        drawn_effect_households=0,
        effective_households=0.0,
        census=False,
    )
    assert authority == probe_tool.INFORMATIONAL
    assert "0.0 effective household" in reason


@_SETTINGS
@given(drawn_samples())
def test_effective_households_bounds(probe_tool, sample) -> None:
    """The effective count is between 0 and the number of drawn households,
    equals 1 when one household carries all the variance, and the terms sum
    to the variance."""
    values, weights, labels, records = sample
    certainty = np.zeros(len(values), dtype=bool)
    terms = probe_tool.stratified_variance_terms(
        values,
        source_weights=weights,
        labels=labels,
        certainty=certainty,
        strata=records,
    )
    variance = probe_tool.stratified_ratio_variance(
        values,
        source_weights=weights,
        labels=labels,
        certainty=certainty,
        strata=records,
    )
    effective = probe_tool.effective_variance_households(terms)
    if terms is None:
        assert math.isnan(variance) and effective == 0.0
        return
    assert (terms >= 0).all()
    assert float(terms.sum()) == pytest.approx(variance, rel=1e-12, abs=0.0)
    assert 0.0 <= effective <= len(values) * (1 + 1e-9)
    assert probe_tool.effective_variance_households(np.asarray([0.0, 7.0, 0.0])) == 1.0
    assert probe_tool.effective_variance_households(np.ones(12)) == pytest.approx(12.0)


@pytest.mark.parametrize(
    ("n", "expected"), [(3, 1), (4, 1), (5, 2), (6, 2), (7, 3), (9, 4), (20_000, 9_999)]
)
def test_batch_size_examples(probe_tool, n, expected) -> None:
    assert probe_tool.choose_batch_size(n) == expected
    assert probe_tool.choose_batch_size(n, 2_000) == min(expected, 2_000)


@pytest.mark.parametrize(
    ("effect", "sign", "expected"),
    [(-5.0, "either", 5.0), (5.0, "positive", 5.0), (5.0, "negative", -5.0)],
)
def test_signed_magnitude_matches_the_gate(probe_tool, effect, sign, expected) -> None:
    assert probe_tool.signed_magnitude(effect, sign) == expected


def test_take_up_failure_split_and_stale_check(probe_tool) -> None:
    missing, constant, band = probe_tool.split_take_up_gate_failures(
        [
            "snap: missing seeded take-up column takes_up_snap_if_eligible",
            "wic: constant column takes_up_wic_if_eligible",
            "eitc: share 0.40 outside [0.70, 0.90]",
        ]
    )
    assert len(missing) == len(constant) == len(band) == 1
    payload = {
        "programs": [
            {
                "variable": "a",
                "populace_treatment": "count_calibrated",
                "ships_at_engine_default": True,
            },
            {
                "variable": "b",
                "populace_treatment": "count_calibrated",
                "ships_at_engine_default": False,
            },
            {
                "variable": "c",
                "populace_treatment": "seeded",
                "ships_at_engine_default": True,
            },
        ]
    }
    assert probe_tool.stale_count_calibrated(payload) == ["a"]


def test_a_constant_effect_has_exactly_zero_variance(sampler, probe_tool) -> None:
    """A value that is the same for every household has a total equal to the
    (conserved) weight total times it: zero error, and so exactly zero
    variance and no effective households. Rounding-level residuals once
    passed for a tiny variance spread over thousands of households, which
    made a conserved total look like a confident miss."""
    rng = np.random.default_rng(5)
    n = 4_000
    ids = np.arange(1, n + 1, dtype=np.int64)
    weights = rng.lognormal(mean=5.0, sigma=1.5, size=n)
    labels = np.where(ids % 3 == 0, "a", "b").astype(object)
    for seed in range(20):
        draw = sampler.draw_households(
            ids, weights, labels, [], fraction=0.05, seed=seed
        )
        terms = probe_tool.stratified_variance_terms(
            np.ones(len(draw.selected_ids)),
            source_weights=draw.source_weights,
            labels=draw.labels,
            certainty=draw.certainty,
            strata=draw.strata,
        )
        assert float(terms.sum()) == 0.0
        assert probe_tool.effective_variance_households(terms) == 0.0
        assert float(draw.adjusted_weights.sum()) == pytest.approx(
            float(weights.sum()), rel=1e-12
        )
