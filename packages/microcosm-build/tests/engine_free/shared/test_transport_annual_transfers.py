"""Exact minor-unit bridge, annual transfers, and reporting-only weekly amounts."""

from __future__ import annotations

import copy
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.transport.compose import (
    _check_money_bridge_owners,
    _check_single_money_normalization,
)
from microcosm.build.transport.money_kernels import (
    MONEY_ANNUAL_TRANSFERS,
    MONEY_MINOR_UNITS,
    MONEY_NORMALIZE_REF,
    weekly_gross_cents_for_reporting,
)
from microcosm.graph import (
    Graph,
    KernelContext,
    Node,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
    compile_graph,
)
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.money_normalization import graph_declaration


def _context(kernel, declaration, columns, *, params=None, node_change=None):
    entity = declaration["entity"]
    rows = declaration.get("columns", declaration.get("transfers"))
    sources = {row["source_column"] for row in rows}
    sources.update(row["flag_column"] for row in rows if "flag_column" in row)
    params = {"declaration": canonical_json(declaration).decode(), **(params or {})}
    node = Node(
        "money.test",
        kernel.ref,
        inputs=(Slice(entity, tuple(sorted(sources))),),
        outputs=tuple(
            Owned(entity, column, "int64")
            for column in dict.fromkeys(row["output_column"] for row in rows)
        ),
        params=params,
        population="money.population",
    )
    if node_change:
        node = replace(node, **node_change)
    length = len(next(iter(columns.values())))
    return KernelContext(
        node=node,
        tables={
            entity: pd.DataFrame({f"{entity}_id": range(11, 11 + length), **columns})
        },
        weights={},
        strata=pd.Series([], dtype=object),
        params=params,
        rng=np.random.default_rng(0),
        sources={},
    )


def _minor_declaration(*, rounding="exact"):
    return {
        "entity": "person",
        "columns": [
            {
                "source_column": "annual_gross",
                "output_column": "annual_gross_cents",
                "quantum": "0.01",
                "rounding": rounding,
            }
        ],
    }


def _transfer_declaration(*, exposure=1):
    return {
        "entity": "person",
        "weeks_per_year": 52,
        "exposure": {"central": exposure, "convention": "full_year_stock_equivalent"},
        "transfers": [
            {
                "flag_column": "receives_main",
                "source_column": "annual_gross_cents",
                "source_basis": "annual_gross_cents",
                "output_column": "annual_transfer_cents",
            },
            {
                "flag_column": "receives_pension",
                "source_column": "pension_weekly_cents",
                "source_basis": "weekly_before_tax_cents",
                "source_tax_basis": "before_tax",
                "output_column": "pension_annual_transfer_cents",
            },
        ],
    }


def _transfers(*, exposure=1, annual=None, weekly=None, flags=None, declaration=None):
    return _context(
        MONEY_ANNUAL_TRANSFERS,
        declaration or _transfer_declaration(exposure=exposure),
        {
            "annual_gross_cents": [2863999, 2864001] if annual is None else annual,
            "pension_weekly_cents": [53827, 53827] if weekly is None else weekly,
            "receives_main": [1, 0] if flags is None else flags,
            "receives_pension": [0, 1],
        },
    )


def test_quantized_bridge_derives_authoritative_integer_cents_without_float_multiply():
    result = MONEY_MINOR_UNITS.run(
        _context(
            MONEY_MINOR_UNITS,
            _minor_declaration(),
            {"annual_gross": [28639.99, 0.29]},
            params={"engine_ref": "axiom:test", "period": "test"},
        )
    )
    assert result.columns["person", "annual_gross_cents"].tolist() == [2863999, 29]
    assert result.columns["person", "annual_gross_cents"].dtype == np.int64
    assert result.receipt["columns"][0]["sum_minor_units"] == 2864028


@pytest.mark.parametrize("quantum,multiplier", [("1", 1), ("0.01", 100)])
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_minor_units_preserve_integer_sources_above_float_precision(
    quantum, multiplier, dtype
):
    declaration = _minor_declaration()
    declaration["columns"][0]["quantum"] = quantum
    amount = 9007199254740993
    result = MONEY_MINOR_UNITS.run(
        _context(
            MONEY_MINOR_UNITS,
            declaration,
            {"annual_gross": np.asarray([amount], dtype=dtype)},
        )
    )
    assert result.columns["person", "annual_gross_cents"].tolist() == [
        amount * multiplier
    ]
    assert result.receipt["columns"][0]["sum_minor_units"] == amount * multiplier


@pytest.mark.parametrize("value", [True, "0.29", object()])
def test_minor_units_refuse_values_without_exact_numeric_entry(value):
    with pytest.raises(ValueError, match="Exact money"):
        MONEY_MINOR_UNITS.run(
            _context(
                MONEY_MINOR_UNITS,
                _minor_declaration(),
                {"annual_gross": [value]},
            )
        )


def test_minor_units_preserve_exact_fraction_sources():
    result = MONEY_MINOR_UNITS.run(
        _context(
            MONEY_MINOR_UNITS,
            _minor_declaration(rounding="half_even"),
            {"annual_gross": [Fraction(201, 200), Fraction(203, 200)]},
        )
    )
    assert result.columns["person", "annual_gross_cents"].tolist() == [100, 102]


def test_annual_cents_are_used_directly_never_weekly_rounded_then_52():
    result = MONEY_ANNUAL_TRANSFERS.run(_transfers())
    assert result.columns["person", "annual_transfer_cents"].tolist() == [2863999, 0]
    assert weekly_gross_cents_for_reporting(2863999, 52) * 52 != 2863999
    assert result.columns["person", "annual_transfer_cents"].index.tolist() == [11, 12]
    assert not any("weekly" in column for _, column in result.columns)


def test_weekly_before_tax_payment_is_annualized_without_inversion():
    result = MONEY_ANNUAL_TRANSFERS.run(_transfers())
    assert result.columns["person", "pension_annual_transfer_cents"].tolist() == [
        0,
        53827 * 52,
    ]
    assert result.receipt["transfers"][1]["source_tax_basis"] == "before_tax"


def test_noninteger_exposure_rounds_half_even_once_at_the_end():
    result = MONEY_ANNUAL_TRANSFERS.run(
        _transfers(
            exposure="0.5",
            annual=[1, 3],
            flags=[1, 1],
            weekly=[1, 1],
        )
    )
    assert result.columns["person", "annual_transfer_cents"].tolist() == [0, 2]
    # Annualize first: rounding 1/2 cent weekly would incorrectly give zero.
    assert result.columns["person", "pension_annual_transfer_cents"].tolist() == [0, 26]


def test_zero_exposure_decimal_text_matches_integer_zero():
    integer = MONEY_ANNUAL_TRANSFERS.run(_transfers(exposure=0))
    decimal_text = MONEY_ANNUAL_TRANSFERS.run(_transfers(exposure="0"))
    for column in integer.columns:
        pd.testing.assert_series_equal(
            integer.columns[column], decimal_text.columns[column]
        )
        assert decimal_text.columns[column].tolist() == [0, 0]


def test_reporting_weekly_gross_is_half_even_and_cannot_be_a_kernel_output():
    assert weekly_gross_cents_for_reporting(26, 52) == 0
    assert weekly_gross_cents_for_reporting(78, 52) == 2
    assert weekly_gross_cents_for_reporting(2863999, 52) == 55077
    context = _transfers()
    node = replace(
        context.node,
        outputs=context.node.outputs + (Owned("person", "weekly_gross", "int64"),),
    )
    with pytest.raises(ValueError, match="owns exactly"):
        MONEY_ANNUAL_TRANSFERS.run(replace(context, node=node))


def test_exposure_is_required_never_implicitly_full_year():
    declaration = _transfer_declaration()
    del declaration["exposure"]
    with pytest.raises(ValueError, match="requires exactly"):
        MONEY_ANNUAL_TRANSFERS.run(_transfers(declaration=declaration))


@pytest.mark.parametrize(
    "exposure", [None, 0.5, float("nan"), float("inf"), True, -1, "NaN", "Infinity"]
)
def test_invalid_or_nonfinite_exposure_refuses(exposure):
    with pytest.raises(ValueError):
        MONEY_ANNUAL_TRANSFERS.run(_transfers(exposure=exposure))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_bridge_money_refuses(value):
    with pytest.raises(ValueError, match="NaN and infinite"):
        MONEY_MINOR_UNITS.run(
            _context(
                MONEY_MINOR_UNITS,
                _minor_declaration(),
                {"annual_gross": [value]},
            )
        )


@pytest.mark.parametrize("values", [[0.0, 1.0], [float("nan"), 0], [2, 0], [-1, 0]])
def test_nonbinary_or_nonfinite_receipt_flag_refuses(values):
    with pytest.raises(ValueError, match="finite binary"):
        MONEY_ANNUAL_TRANSFERS.run(_transfers(flags=values))


@pytest.mark.parametrize(
    "amount", [float("nan"), float("inf"), 1.5, 2**63, -(2**63) - 1, -1, True, False]
)
def test_invalid_minor_unit_input_refuses(amount):
    with pytest.raises(ValueError):
        MONEY_ANNUAL_TRANSFERS.run(_transfers(annual=[amount, amount]))


def test_annual_transfer_int64_overflow_refuses_and_receipt_sum_stays_exact():
    with pytest.raises(ValueError, match="exceeds int64"):
        MONEY_ANNUAL_TRANSFERS.run(_transfers(weekly=[2**63 - 1, 2**63 - 1]))
    context = _transfers(annual=[2**63 - 1, 2**63 - 1], flags=[1, 1])
    result = MONEY_ANNUAL_TRANSFERS.run(context)
    assert result.receipt["transfers"][0]["sum_minor_units"] == 2 * (2**63 - 1)


def test_minor_unit_int64_overflow_refuses():
    with pytest.raises(ValueError, match="exceeds int64"):
        MONEY_MINOR_UNITS.run(
            _context(
                MONEY_MINOR_UNITS,
                _minor_declaration(),
                {"annual_gross": [1e20]},
            )
        )


def test_exact_quantized_bridge_refuses_uncertified_fractional_cents():
    with pytest.raises(ValueError, match="not an exact multiple"):
        MONEY_MINOR_UNITS.run(
            _context(
                MONEY_MINOR_UNITS,
                _minor_declaration(),
                {"annual_gross": [1.005]},
            )
        )


def test_before_tax_minor_units_use_exact_half_even():
    declaration = _minor_declaration(rounding="half_even")
    declaration["columns"][0]["source_tax_basis"] = "before_tax"
    result = MONEY_MINOR_UNITS.run(
        _context(
            MONEY_MINOR_UNITS,
            declaration,
            {"annual_gross": [1.005, 1.015]},
        )
    )
    assert result.columns["person", "annual_gross_cents"].tolist() == [100, 102]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(source_basis="weekly_after_tax_cents"),
        lambda row: row.pop("source_tax_basis"),
        lambda row: row.update(source_tax_basis="after_tax"),
        lambda row: row.update(extra=True),
    ],
)
def test_weekly_before_tax_declaration_is_closed_and_explicit(mutation):
    declaration = _transfer_declaration()
    mutation(declaration["transfers"][1])
    with pytest.raises(ValueError):
        MONEY_ANNUAL_TRANSFERS.run(_transfers(declaration=declaration))


@pytest.mark.parametrize("weeks", [None, 0, -1, 52.0, True])
def test_weeks_are_explicit_positive_integer(weeks):
    declaration = _transfer_declaration()
    declaration["weeks_per_year"] = weeks
    with pytest.raises(ValueError, match="weeks_per_year"):
        MONEY_ANNUAL_TRANSFERS.run(_transfers(declaration=declaration))


def test_exact_money_output_has_one_owner_and_whole_row_slice():
    context = _transfers()
    with pytest.raises(ValueError, match="whole-row"):
        MONEY_ANNUAL_TRANSFERS.run(
            replace(
                context,
                node=replace(
                    context.node,
                    inputs=(
                        Slice(
                            "person",
                            (*context.tables["person"].columns[1:], "mask"),
                            rows="mask",
                        ),
                    ),
                ),
            )
        )
    extra = context.tables["person"].copy()
    extra["annual_transfer_cents"] = 0
    with pytest.raises(ValueError, match="existing output columns"):
        MONEY_ANNUAL_TRANSFERS.run(replace(context, tables={"person": extra}))


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    amount=st.integers(min_value=0, max_value=10**12),
    numerator=st.integers(min_value=1, max_value=1000),
    decimal_places=st.integers(min_value=0, max_value=4),
)
def test_transfer_matches_fraction_oracle_and_is_deterministic(
    amount, numerator, decimal_places
):
    exposure = (
        str(numerator)
        if not decimal_places
        else f"{numerator // (10**decimal_places)}.{numerator % (10**decimal_places):0{decimal_places}d}"
    )
    context = _transfers(exposure=exposure, annual=[amount, amount], flags=[1, 1])
    first = MONEY_ANNUAL_TRANSFERS.run(context)
    second = MONEY_ANNUAL_TRANSFERS.run(context)
    expected = round(Fraction(amount * numerator, 10**decimal_places))
    assert first.columns["person", "annual_transfer_cents"].tolist() == [
        expected,
        expected,
    ]
    assert canonical_json(first.receipt) == canonical_json(second.receipt)
    for column in first.columns:
        pd.testing.assert_series_equal(first.columns[column], second.columns[column])


def test_closed_documents_reject_unknown_and_duplicate_columns():
    declaration = _minor_declaration()
    declaration["columns"].append(copy.deepcopy(declaration["columns"][0]))
    with pytest.raises(ValueError, match="unique sources and outputs"):
        MONEY_MINOR_UNITS.run(
            _context(MONEY_MINOR_UNITS, declaration, {"annual_gross": [1]})
        )


BRIDGES = (MONEY_MINOR_UNITS.ref, MONEY_ANNUAL_TRANSFERS.ref)


def _lineage(kernel, column, *, normalizes=False):
    """A writer one population version above the column's first owner."""
    owned = [Owned("person", "major", "float64"), Owned("person", "units", "int64")]
    create = Node(
        "create",
        "toy.create@1",
        structural=StructuralDelta.CREATE,
        sources=("donor",),
        outputs=tuple(owned),
    )
    opened = Node(
        "open", "toy.open@1", structural=StructuralDelta.FILTER, base="create"
    )
    nodes = [create, opened]
    if normalizes:
        nodes.append(
            Node(
                "normalize",
                MONEY_NORMALIZE_REF,
                population="open",
                outputs=(Owned("person", "interest_income_cents", "int64"),),
            )
        )
    nodes.append(
        Node("next", "toy.open@1", structural=StructuralDelta.FILTER, base="open")
    )
    nodes.append(
        Node(
            "writer",
            kernel,
            population="next",
            inputs=(Slice("person", ("major",)),),
            outputs=(Owned("person", column, "int64"),),
        )
    )
    return Graph("xx", (SourceRef("donor", "fixture@1"),), tuple(nodes))


def _check(graph, *, declared=False):
    resources = {"money_normalization": graph_declaration()} if declared else {}
    compiled = compile_graph(graph)
    _check_single_money_normalization(
        {"country": "xx", "resources": resources}, graph, compiled
    )
    _check_money_bridge_owners(graph, compiled)


@pytest.mark.parametrize("kernel", BRIDGES)
@pytest.mark.parametrize("declared", [False, True])
def test_money_bridge_refuses_a_column_its_lineage_owns(kernel, declared):
    with pytest.raises(ValueError, match="owns person.units, which 'create' owns"):
        _check(_lineage(kernel, "units"), declared=declared)


@pytest.mark.parametrize("kernel", BRIDGES)
@pytest.mark.parametrize("declared", [False, True])
def test_money_bridge_owns_a_new_column(kernel, declared):
    _check(_lineage(kernel, "new_cents"), declared=declared)


@pytest.mark.parametrize("kernel", (*BRIDGES, "toy.rules@1"))
def test_only_normalization_writes_a_normalized_companion(kernel):
    with pytest.raises(ValueError, match="interest_income_cents"):
        _check(
            _lineage(kernel, "interest_income_cents", normalizes=True), declared=True
        )
