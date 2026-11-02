"""``grouped_pareto_mle@1`` and ``transport.quantile_map@2`` (G8-r2c hunk B).

The estimator fits a Pareto alpha to grouped counts; ``@2`` fits it to the
anchor facts on its own band surface and then maps exactly as ``@1`` does
with that alpha as a literal. Every count here is synthetic except
Belgium's two RSZ 2026-Q1 counts, which the hub's ruling R2C-9 quotes.
"""

from __future__ import annotations

import copy
import json
import math
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.tail_estimators import (
    GROUPED_PARETO_MLE_REF,
    TailEstimatorRefusalError,
    _anchors,
    grouped_pareto_mle,
    grouped_pareto_score,
    parse_grouped_pareto_declaration,
)
from microcosm.build.transport import compose
from microcosm.build.transport.artifact_types import TARGET_SURFACE_TYPE
from microcosm.build.transport.evidence_contract import (
    DONOR_RANGES,
    DONOR_RANGES_REF,
    DONOR_RANGES_TYPE,
    DONOR_SUPPORT_BOUNDS,
    SUPPORT_BOUNDS_REF,
    SUPPORT_BOUNDS_TYPE,
    decode_support_bounds,
    quantile_bounds,
    quantile_support,
)
from microcosm.build.transport.evidence_kernels import (
    TRANSPORT_DONOR_RANGES,
    TRANSPORT_SUPPORT_BOUNDS,
)
from microcosm.build.transport.population_kernels import TRANSPORT_QUANTILE_MAP
from microcosm.build.transport.quantile_kernels import (
    QUANTILE_MAP_V2_REF,
    TRANSPORT_QUANTILE_MAP_V2,
)
from microcosm.build.transport.registry import register_transport_population_kernels
from microcosm.build.transport.target_kernels import compile_target_surface
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    KernelRegistry,
    Node,
    Owned,
    Slice,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.executor import Population, _project_context
from microcosm.graph.kernel import ArtifactValue, NumericScope
from test_support.microcosm_build.transport_graph import (
    COUNTRY,
    reference_document,
    reference_row,
    toy_fact,
    write_facts,
)
from test_support.microcosm_build.transport_population import digest

#: The bracket the r2c design declares, [1 + 2**-20, 64], as exact text.
BRACKET_TEXT = ["1.00000095367431640625", "64"]
BRACKET = (Fraction(BRACKET_TEXT[0]), Fraction(BRACKET_TEXT[1]))
ITERATIONS = 64
#: Recovery tolerance on exact expected counts. Rounding to six decimals
#: contributes at most 5e-7; 64 halvings of [1 + 2**-20, 64] leave a bracket
#: of 63 / 2**64 (about 3e-18); and the binary64 score's root moves by its
#: summation error over l'' (about 1e-13 relative here). 1e-6 covers the
#: three with room, and no closer tolerance survives the half-unit rounding.
TOLERANCE = 1e-6

PROPERTY = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
)


def _fit(rows, *, bracket=BRACKET, iterations=ITERATIONS):
    return grouped_pareto_mle(rows, bracket=bracket, iterations=iterations)


def _expected_rows(alpha, edges, total=1e6):
    """Exact Pareto expected counts on contiguous ``edges`` plus the open top."""
    origin = edges[0]

    def survival(x):
        return (x / origin) ** -alpha

    rows = [
        (f"b{index}", lower, upper, total * (survival(lower) - survival(upper)))
        for index, (lower, upper) in enumerate(zip(edges, edges[1:], strict=False))
    ]
    rows.append(("top", edges[-1], None, total * survival(edges[-1])))
    return rows


@st.composite
def anchors(draw):
    origin = draw(st.floats(min_value=1, max_value=1e6))
    ratios = draw(
        st.lists(st.floats(min_value=1.01, max_value=3), min_size=1, max_size=10)
    )
    edges = [origin]
    for ratio in ratios:
        edges.append(edges[-1] * ratio)
    return edges


# ---------------------------------------------------------------------------
# The estimator: examples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [1.2, 1.5, 2.093, 3.348955, 5.0])
def test_recovers_a_known_alpha_from_exact_pareto_counts(alpha) -> None:
    edges = [500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0, 1000.0]
    fit = _fit(_expected_rows(alpha, edges))
    assert abs(fit.alpha_float - alpha) <= TOLERANCE
    assert fit.denominator == 10**6
    assert fit.alpha == Fraction(fit.numerator, 10**6)


def test_refuses_a_bracket_without_a_sign_change_by_name() -> None:
    rows = _expected_rows(1.5, [100.0, 200.0, 400.0])
    with pytest.raises(TailEstimatorRefusalError, match="bracket_without_sign_change"):
        _fit(rows, bracket=(Fraction(2), Fraction(64)))
    with pytest.raises(TailEstimatorRefusalError) as caught:
        _fit(rows, bracket=(Fraction(1, 10), Fraction(1)))
    assert caught.value.code == "bracket_without_sign_change"


def test_refuses_an_alpha_at_or_below_one_by_name() -> None:
    rows = _expected_rows(0.8, [100.0, 200.0, 400.0])
    with pytest.raises(TailEstimatorRefusalError) as caught:
        _fit(rows, bracket=(Fraction(1, 2), Fraction(64)))
    assert caught.value.code == "alpha_not_above_one"
    # Alpha exactly one after rounding is refused too: its mean is infinite.
    rows = _expected_rows(1.0000001, [100.0, 200.0, 400.0])
    with pytest.raises(TailEstimatorRefusalError, match="alpha_not_above_one"):
        _fit(rows, bracket=(Fraction(1, 2), Fraction(64)))


def test_degenerate_counts_fall_out_as_no_sign_change() -> None:
    no_top = [("a", 100.0, 200.0, 10.0), ("t", 200.0, None, 0.0)]
    no_body = [("a", 100.0, 200.0, 0.0), ("t", 200.0, None, 10.0)]
    for rows in (no_top, no_body):
        with pytest.raises(TailEstimatorRefusalError, match="no_sign_change|sign"):
            _fit(rows)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([("t", 100.0, None, 1.0)], "at least one finite band"),
        ([("a", 100.0, 200.0, 1.0), ("t", 200.0, 300.0, 1.0)], "open top"),
        ([("a", 100.0, None, 1.0), ("t", 200.0, None, 1.0)], "only the top"),
        ([("a", 100.0, 200.0, 1.0), ("t", 250.0, None, 1.0)], "contiguous"),
        ([("a", 0.0, 200.0, 1.0), ("t", 200.0, None, 1.0)], "positive"),
        ([("a", 100.0, 200.0, -1.0), ("t", 200.0, None, 1.0)], "nonnegative"),
    ],
)
def test_refuses_malformed_anchors(rows, message) -> None:
    with pytest.raises(ValueError, match=message):
        _fit(rows)


def test_the_declaration_is_parsed_exactly() -> None:
    declared = {
        "id": GROUPED_PARETO_MLE_REF,
        "anchor_references": ["a", "t"],
        "bracket": BRACKET_TEXT,
        "iterations": ITERATIONS,
    }
    references, bracket, iterations = parse_grouped_pareto_declaration(declared)
    assert references == ("a", "t")
    assert bracket == (1 + Fraction(1, 2**20), Fraction(64))
    assert iterations == ITERATIONS
    for change, message in (
        ({"id": "grouped_pareto_mle@2"}, "no implementation"),
        ({"bracket": [1.5, 64]}, "decimal text"),
        ({"bracket": ["64", "2"]}, "low < high"),
        ({"iterations": 0}, "iterations"),
        ({"iterations": 201}, "iterations"),
        ({"anchor_references": ["t"]}, "at least one"),
    ):
        with pytest.raises(ValueError, match=message):
            parse_grouped_pareto_declaration({**declared, **change})
    with pytest.raises(ValueError, match="exactly id"):
        parse_grouped_pareto_declaration({**declared, "alpha": 2})


def test_the_receipt_records_the_fit() -> None:
    fit = _fit([("a", 200, 300, 1_066_243), ("t", 300, None, 369_203)])
    receipt = fit.receipt()
    assert receipt["estimator"] == GROUPED_PARETO_MLE_REF
    assert receipt["estimator_id"] == "grouped_pareto_mle"
    assert receipt["estimator_version"] == 1
    assert receipt["anchor"] == "200"
    assert receipt["anchors"] == [
        {"reference": "a", "lower": "200", "upper": "300", "count": 1_066_243.0},
        {"reference": "t", "lower": "300", "upper": None, "count": 369_203.0},
    ]
    assert receipt["bracket"] == BRACKET_TEXT
    assert receipt["iterations"] == ITERATIONS
    low, high = receipt["final_bracket"]
    assert low <= fit.midpoint <= high
    score_low, score_high = receipt["score_at_final_bracket"]
    assert score_low > 0 >= score_high
    assert receipt["alpha"] == {"numerator": 3_348_955, "denominator": 10**6}
    assert receipt["alpha_decimal"] == "3.348955"
    mean = Fraction(3_348_955, 10**6) * 300 / (Fraction(3_348_955, 10**6) - 1)
    assert (
        Fraction(
            receipt["top_band_mean"]["numerator"],
            receipt["top_band_mean"]["denominator"],
        )
        == mean
    )
    assert receipt["top_band_mean_decimal"] == "427.716367"
    json.dumps(receipt, allow_nan=False)


# ---------------------------------------------------------------------------
# Belgium's minimal anchor set: the closed form
# ---------------------------------------------------------------------------


def _closed_form(n_band, n_top, lower, upper):
    return math.log((n_band + n_top) / n_top) / math.log(upper / lower)


def test_belgium_rsz_counts_give_the_closed_form_alpha() -> None:
    n_band, n_top = 1_066_243, 369_203
    fit = _fit([("rsz_200_300", 200, 300, n_band), ("rsz_over_300", 300, None, n_top)])
    assert fit.receipt()["alpha_decimal"] == "3.348955"
    assert f"{_closed_form(n_band, n_top, 200, 300):.6f}" == "3.348955"


@PROPERTY
@given(
    n_band=st.integers(min_value=1, max_value=10**8),
    n_top=st.integers(min_value=1, max_value=10**8),
    lower=st.floats(min_value=1, max_value=1e6),
    ratio=st.floats(min_value=1.05, max_value=10),
)
def test_property_one_band_and_the_top_match_the_closed_form(
    n_band, n_top, lower, ratio
) -> None:
    upper = lower * ratio
    expected = _closed_form(n_band, n_top, lower, upper)
    if not 1.01 < expected < 63:
        return
    fit = _fit([("a", lower, upper, n_band), ("t", upper, None, n_top)])
    assert abs(fit.alpha_float - expected) <= TOLERANCE


# ---------------------------------------------------------------------------
# The estimator: properties
# ---------------------------------------------------------------------------


@PROPERTY
@given(alpha=st.floats(min_value=1.2, max_value=5), edges=anchors())
def test_property_recovers_alpha_within_the_declared_tolerance(alpha, edges) -> None:
    fit = _fit(_expected_rows(alpha, edges))
    assert abs(fit.alpha_float - alpha) <= TOLERANCE


@PROPERTY
@given(
    alpha=st.floats(min_value=1.2, max_value=5),
    edges=anchors(),
    k=st.integers(min_value=2, max_value=10**6),
    power=st.integers(min_value=-20, max_value=20),
)
def test_property_scaling_every_count_leaves_alpha_unchanged(
    alpha, edges, k, power
) -> None:
    rows = _expected_rows(alpha, edges)
    base = _fit(rows)
    # A power of two scales every float exactly, so the fit is bitwise equal.
    exact = _fit([(r, lo, up, n * 2.0**power) for r, lo, up, n in rows])
    assert exact.numerator == base.numerator
    # Any other factor rounds each count once; alpha moves by at most a unit
    # in the sixth decimal, and only at a rounding tie.
    scaled = _fit([(r, lo, up, n * k) for r, lo, up, n in rows])
    assert abs(scaled.numerator - base.numerator) <= 1


@PROPERTY
@given(
    alpha=st.floats(min_value=1.2, max_value=5),
    edges=anchors(),
    data=st.data(),
)
def test_property_moving_mass_upward_lowers_alpha(alpha, edges, data) -> None:
    rows = _expected_rows(alpha, edges)
    source = data.draw(st.integers(min_value=0, max_value=len(rows) - 2))
    target = data.draw(st.integers(min_value=source + 1, max_value=len(rows) - 1))
    share = data.draw(st.floats(min_value=0.01, max_value=0.99))
    moved = rows[source][3] * share
    counts = [row[3] for row in rows]
    counts[source] -= moved
    counts[target] += moved
    shifted = [(r, lo, up, n) for (r, lo, up, _), n in zip(rows, counts, strict=True)]
    before = _fit(rows)
    shifted_anchors = _anchors(shifted)
    # Before rounding, the move lowers the score at every alpha, so the root
    # moves down.
    assert grouped_pareto_score(shifted_anchors, before.midpoint) < 0
    try:
        after = _fit(shifted)
    except TailEstimatorRefusalError as refusal:
        # The root fell below the bracket: alpha is lower still.
        assert refusal.code in {"bracket_without_sign_change", "alpha_not_above_one"}
        assert grouped_pareto_score(shifted_anchors, float(BRACKET[0])) <= 0
        return
    assert after.numerator <= before.numerator


@PROPERTY
@given(
    counts=st.lists(
        st.floats(min_value=0, max_value=1e7), min_size=2, max_size=8
    ).filter(lambda c: c[-1] > 0 and sum(c[:-1]) > 0),
    edges=anchors(),
)
def test_property_the_score_changes_sign_once_on_the_bracket(counts, edges) -> None:
    size = min(len(counts), len(edges))
    counts, edges = counts[-size:], edges[:size]
    rows = [
        (f"b{index}", lower, upper, count)
        for index, (lower, upper, count) in enumerate(
            zip(edges, edges[1:], counts, strict=False)
        )
    ]
    rows.append(("top", edges[-1], None, counts[-1]))
    if len(rows) < 2 or sum(row[3] for row in rows[:-1]) <= 0:
        return
    fit_anchors = _anchors(rows)
    grid = np.linspace(float(BRACKET[0]), float(BRACKET[1]), 257)
    scores = [grouped_pareto_score(fit_anchors, float(a)) for a in grid]
    # Nonincreasing (strictly, until expm1 overflows to infinity), hence
    # at most one sign change.
    assert all(b <= a for a, b in zip(scores, scores[1:], strict=False))
    signs = [s > 0 for s in scores]
    assert sum(1 for a, b in zip(signs, signs[1:], strict=False) if a != b) <= 1


# ---------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------

SCHEMA = EntitySchema(group_entities=("household",))
#: Synthetic band rows: a uniform body, two anchor bands and the open top.
BAND_ROWS = (
    ("toy_wage_0", 1.0, 1000.0),
    ("toy_wage_1", 1000.0, 2000.0),
    ("toy_wage_2", 2000.0, 3000.0),
    ("toy_wage_top", 3000.0, None),
)
COUNTS = {"toy_wage_0": 500.0, "toy_wage_1": 300.0, "toy_wage_2": 120.0}
ANCHORS = ["toy_wage_1", "toy_wage_2", "toy_wage_top"]


def _frame(values) -> Frame:
    count = len(values)
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, count + 1, dtype=np.int64),
            "person_household_id": np.arange(1, count + 1, dtype=np.int64),
            "employment_income": np.asarray(values, dtype=np.float64),
        }
    )
    household = pd.DataFrame({"household_id": np.arange(1, count + 1, dtype=np.int64)})
    weights = np.linspace(1, 3, count)
    return Frame(
        {"person": person, "household": household},
        SCHEMA,
        {"household": Weights(weights, WeightKind.DESIGN)},
    )


def _surface(root: Path, counts: dict[str, float]) -> ArtifactValue:
    document = reference_document(
        [
            reference_row(name, entity="person", measure="person_count")
            for name, _, _ in BAND_ROWS
        ]
    )
    facts = write_facts(
        root / "band-facts.jsonl",
        [toy_fact(name, counts[name], entity="person") for name, _, _ in BAND_ROWS],
    )
    payload, _ = compile_target_surface(
        facts, document, country=COUNTRY, references_sha256=digest(document)
    )
    return ArtifactValue(
        payload=payload,
        type=TARGET_SURFACE_TYPE,
        key="a" * 64,
        producer_key="b" * 64,
        numerics=NumericScope(),
    )


def _bands():
    return {
        "bands": [
            {"lower": lower, "upper": upper, "reference": name}
            for name, lower, upper in BAND_ROWS
        ]
    }


def _estimator(**change):
    return {
        "id": GROUPED_PARETO_MLE_REF,
        "anchor_references": ANCHORS,
        "bracket": BRACKET_TEXT,
        "iterations": ITERATIONS,
        **change,
    }


def _node(kernel, interpolation, bands=None) -> Node:
    bands = _bands() if bands is None else bands
    return Node(
        "toy.wage.map",
        kernel,
        inputs=(Slice("person", ("employment_income",)),),
        outputs=(Owned("person", "employment_income", "float64", rewrite=True),),
        params={
            "entity": "person",
            "column": "employment_income",
            "bands": canonical_json(bands).decode(),
            "interpolation": canonical_json(interpolation).decode(),
            "bands_sha256": digest(bands),
        },
        artifact_inputs=(
            ArtifactInput("surface", "toy.bands", "surface", TARGET_SURFACE_TYPE),
        ),
    )


def _run(kernel, node, frame, surface):
    context = _project_context(
        node,
        Population.from_frame(frame, "toy.population"),
        key="0" * 64,
        sources={},
        tolerances={},
        numerics={},
        artifacts={"surface": surface},
    )
    return kernel.run(context)


def _counts(top=60.0, **change):
    return {**COUNTS, "toy_wage_top": top, **change}


def _values(seed=7, size=40):
    rng = np.random.default_rng(seed)
    values = np.round(rng.lognormal(7, 1, size), 2)
    values[::9] = 0
    return values


def _pareto(estimator=None):
    return {"method": "pareto", "estimator": estimator or _estimator()}


def test_v2_equals_v1_with_its_estimated_alpha_as_a_literal(tmp_path) -> None:
    frame, surface = _frame(_values()), _surface(tmp_path, _counts())
    v2 = _run(
        TRANSPORT_QUANTILE_MAP_V2,
        _node(QUANTILE_MAP_V2_REF, _pareto()),
        frame,
        surface,
    )
    decimal = v2.receipt["tail_estimator"]["alpha_decimal"]
    literal = {"method": "pareto", "alpha": float(decimal)}
    v1 = _run(
        TRANSPORT_QUANTILE_MAP,
        _node("transport.quantile_map@1", literal),
        frame,
        surface,
    )
    ((key, mapped),) = v2.columns.items()
    assert key == ("person", "employment_income")
    assert mapped.to_numpy().tobytes() == v1.columns[key].to_numpy().tobytes()
    pd.testing.assert_index_equal(mapped.index, v1.columns[key].index)


@PROPERTY
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    top=st.floats(min_value=5, max_value=400),
    middle=st.floats(min_value=20, max_value=800),
)
def test_property_v2_is_bitwise_v1_at_the_fitted_alpha(tmp_path, seed, top, middle):
    with TemporaryDirectory(dir=tmp_path) as temporary:
        frame = _frame(_values(seed))
        surface = _surface(Path(temporary), _counts(top, toy_wage_2=middle))
        try:
            v2 = _run(
                TRANSPORT_QUANTILE_MAP_V2,
                _node(QUANTILE_MAP_V2_REF, _pareto()),
                frame,
                surface,
            )
        except TailEstimatorRefusalError:
            return
        alpha = v2.receipt["tail_estimator"]["alpha"]
        literal = {
            "method": "pareto",
            "alpha": alpha["numerator"] / alpha["denominator"],
        }
        v1 = _run(
            TRANSPORT_QUANTILE_MAP,
            _node("transport.quantile_map@1", literal),
            frame,
            surface,
        )
        key = ("person", "employment_income")
        assert v2.columns[key].to_numpy().tobytes() == (
            v1.columns[key].to_numpy().tobytes()
        )


def test_anchor_counts_are_the_band_rows_counts(tmp_path) -> None:
    """One source: the anchors read the same surface facts as the shares."""

    counts = _counts()
    frame, surface = _frame(_values()), _surface(tmp_path / "a", counts)
    receipt = _run(
        TRANSPORT_QUANTILE_MAP_V2,
        _node(QUANTILE_MAP_V2_REF, _pareto()),
        frame,
        surface,
    ).receipt["tail_estimator"]
    assert [(row["reference"], row["count"]) for row in receipt["anchors"]] == [
        (name, counts[name]) for name in ANCHORS
    ]
    expected = grouped_pareto_mle(
        [(name, lower, upper, counts[name]) for name, lower, upper in BAND_ROWS[1:]],
        bracket=BRACKET,
        iterations=ITERATIONS,
    )
    assert receipt["alpha"]["numerator"] == expected.numerator
    # Changing one band row's count changes alpha.
    changed = _run(
        TRANSPORT_QUANTILE_MAP_V2,
        _node(QUANTILE_MAP_V2_REF, _pareto()),
        frame,
        _surface(tmp_path / "b", _counts(toy_wage_2=counts["toy_wage_2"] * 2)),
    ).receipt["tail_estimator"]
    assert changed["alpha"]["numerator"] != receipt["alpha"]["numerator"]
    # A count below the anchors changes the shares but not alpha.
    body = _run(
        TRANSPORT_QUANTILE_MAP_V2,
        _node(QUANTILE_MAP_V2_REF, _pareto()),
        frame,
        _surface(tmp_path / "c", _counts(toy_wage_0=counts["toy_wage_0"] * 2)),
    ).receipt["tail_estimator"]
    assert body["alpha"] == receipt["alpha"]


@pytest.mark.parametrize(
    ("interpolation", "bands", "message"),
    [
        ({"method": "pareto", "alpha": 2.0}, None, "literal alpha belongs"),
        ({"method": "uniform"}, None, "literal alpha belongs"),
        (
            _pareto(_estimator(anchor_references=["toy_wage_1", "toy_wage_2"])),
            None,
            "top contiguous bands",
        ),
        (
            _pareto(_estimator(anchor_references=["toy_wage_top", "toy_wage_2"])),
            None,
            "top contiguous bands",
        ),
        (
            _pareto(),
            {"positive": _bands()["bands"], "negative": _bands()["bands"]},
            "bare band list",
        ),
        (_pareto(_estimator(bracket=["4", "64"])), None, "bracket_without_sign_change"),
    ],
)
def test_v2_refusals(tmp_path, interpolation, bands, message) -> None:
    with pytest.raises(ValueError, match=message):
        _run(
            TRANSPORT_QUANTILE_MAP_V2,
            _node(QUANTILE_MAP_V2_REF, interpolation, bands),
            _frame(_values()),
            _surface(tmp_path, _counts()),
        )


def test_v2_refuses_a_count_that_is_not_on_the_ranked_unit(tmp_path) -> None:
    node = _node(QUANTILE_MAP_V2_REF, _pareto())
    node = Node(
        node.id,
        node.kernel,
        inputs=node.inputs,
        outputs=node.outputs,
        params={**node.params, "aggregate_entity": "household"},
        artifact_inputs=node.artifact_inputs,
    )
    with pytest.raises(ValueError, match="unfiltered household count facts"):
        _run(
            TRANSPORT_QUANTILE_MAP_V2,
            node,
            _frame(_values()),
            _surface(tmp_path, _counts()),
        )


# ---------------------------------------------------------------------------
# Registration and support
# ---------------------------------------------------------------------------


def test_v2_is_registered_beside_v1() -> None:
    registry = register_transport_population_kernels(KernelRegistry())
    assert registry.get(QUANTILE_MAP_V2_REF) is TRANSPORT_QUANTILE_MAP_V2
    assert registry.get("transport.quantile_map@1") is TRANSPORT_QUANTILE_MAP
    assert {"transport.quantile_map@1", QUANTILE_MAP_V2_REF} <= compose.SKELETON_KERNELS
    assert compose._SUPPORT_PRODUCERS[QUANTILE_MAP_V2_REF] == "quantile_mapped"
    assert QUANTILE_MAP_V2_REF in compose._MONEY_PRODUCERS
    assert TRANSPORT_QUANTILE_MAP_V2.implementation_hash() != (
        TRANSPORT_QUANTILE_MAP.implementation_hash()
    )


def _spec_with_map(kernel):
    spec = copy.deepcopy(compose.load_transport_spec("nz"))
    spec["resources"]["transport_graph"]["nodes"].append(
        {
            "id": "toy.wage.map",
            "kernel": kernel,
            "population": "nz.central",
            "inputs": [{"entity": "person", "columns": ["employment_income"]}],
            "outputs": [
                {
                    "entity": "person",
                    "column": "employment_income",
                    "dtype": "float64",
                    "rewrite": True,
                }
            ],
        }
    )
    return spec


def test_composition_accepts_v2_as_the_quantile_mapped_money_producer() -> None:
    """A map in ``nz.central`` satisfies the central encode's wage read."""

    def wage_gaps(spec):
        return [
            gap
            for gap in compose._money_producer_gaps(spec)
            if "person.employment_income" in gap
        ]

    def central_readers(gaps):
        return [gap for gap in gaps if "'nz.encode.person'" in gap]

    assert central_readers(wage_gaps(compose.load_transport_spec("nz")))
    for kernel in (QUANTILE_MAP_V2_REF, "transport.quantile_map@1"):
        gaps = wage_gaps(_spec_with_map(kernel))
        assert central_readers(gaps) == []
        assert not any("produced by" in gap for gap in gaps)
    (gap,) = central_readers(wage_gaps(_spec_with_map("transport.currency@1")))
    assert "transport.quantile_map@2 may produce it" in gap


def _support_params():
    bands = _bands()
    return {"bands": bands, "interpolation": _pareto()}


def test_the_w4_support_producer_reads_v2_params_with_an_open_top() -> None:
    params = _support_params()
    support = quantile_support(params["bands"], params["interpolation"])
    assert support.positive == (
        (1.0, 1000.0),
        (1000.0, 2000.0),
        (2000.0, 3000.0),
        (3000.0, None),
    )
    # A bare band list feeds both signs, as quantile_map applies it.
    assert support.negative == support.positive
    low, high = quantile_bounds(support.positive, support.negative)
    assert high is None  # expected_max: null


def _support_row(params, values):
    """Run w4's donor-ranges and support-bounds producers on one roster row."""
    roster = {
        "columns": [
            {
                "entity": "person",
                "column": "employment_income",
                "class": "quantile_mapped",
                "owner": "toy.wage.map",
                "producer_params": params,
            }
        ]
    }
    frame = _frame(values)
    text = canonical_json(roster).decode()

    def node(ref, alias, artifact_type, inputs=(), artifact_inputs=()):
        return Node(
            f"toy.evidence.{alias}",
            ref,
            inputs=inputs,
            params={"support_roster": text},
            artifact_inputs=artifact_inputs,
            artifact_outputs=(ArtifactOutput(alias, artifact_type),),
        )

    def context(item, artifacts):
        return _project_context(
            item,
            Population.from_frame(frame, "toy.terminal"),
            key="0" * 64,
            sources={},
            tolerances={},
            numerics={},
            artifacts=artifacts,
        )

    ranges_node = node(DONOR_RANGES_REF, DONOR_RANGES, DONOR_RANGES_TYPE)
    ranges = TRANSPORT_DONOR_RANGES.run(context(ranges_node, {})).artifacts[
        DONOR_RANGES
    ]
    bounds_node = node(
        SUPPORT_BOUNDS_REF,
        DONOR_SUPPORT_BOUNDS,
        SUPPORT_BOUNDS_TYPE,
        inputs=(Slice("person", ("employment_income",)),),
        artifact_inputs=(
            ArtifactInput(
                DONOR_RANGES, ranges_node.id, DONOR_RANGES, DONOR_RANGES_TYPE
            ),
        ),
    )
    value = ArtifactValue(ranges, DONOR_RANGES_TYPE, "c" * 64, "d" * 64, NumericScope())
    bounds = TRANSPORT_SUPPORT_BOUNDS.run(
        context(bounds_node, {DONOR_RANGES: value})
    ).artifacts[DONOR_SUPPORT_BOUNDS]
    (row,) = decode_support_bounds(bounds).document["columns"]
    return row


def test_the_support_bounds_kernel_reports_a_null_expected_max_for_v2() -> None:
    row = _support_row(_support_params(), [0.0, 5.0, 4e6])
    assert row["class"] == "quantile_mapped"
    assert row["expected_max"] is None
    assert row["records_outside"] == 0


def test_v2_support_bounds_both_signs_for_a_bare_band_list() -> None:
    row = _support_row(_support_params(), [0.0, 5.0, 4e6])
    assert row["expected_min"] is None
    assert row["expected_max"] is None
