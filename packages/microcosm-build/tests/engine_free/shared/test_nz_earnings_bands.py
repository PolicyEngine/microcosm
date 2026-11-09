"""G8-r2c E: the P1 wage bands scale exactly and map like scaled values.

Two invariants of ``earnings_bands.json`` beyond the example tests in
``test_nz_transport_declarations.py``:

* for any positive factor, cut × factor / 100 keeps the 235 bands contiguous
  and strictly increasing from 0 to the open top (Hypothesis);
* mapping onto the scaled edges equals mapping onto the unscaled edges and
  multiplying by the factor, up to binary64 rounding (differential).
"""

from __future__ import annotations

import json
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.frame.transport import quantile_map
from test_support.paths import paths_for
from tools import build_nz_transport_graph as builder

_SPEC = paths_for("microcosm-build").package / "src/microcosm/build/nz"
_WAGE_TABLE = "ird_wage_salary_national_ty2026"

#: Differential tolerance, relative. The scaled path rounds each edge once
#: (≤ 2^-53 relative) and then interpolates; the unscaled path interpolates
#: and multiplies by the factor once more. The uniform branch's
#: ``lower + u(upper − lower)`` takes three roundings, the Pareto branch's
#: ``lower·exp(offset)`` two, so the two results differ by a handful of
#: half-ulps. 2^-49 (about 1.8e-15, eight units of 2^-52) bounds that with
#: margin; design 80 measured at most 3.9e-16 on 2,000 lognormal values.
_RELATIVE_TOLERANCE = 2.0**-49


def _table():
    intervals = json.loads((_SPEC / "reference_intervals.json").read_text())
    (table,) = [row for row in intervals["tables"] if row["id"] == _WAGE_TABLE]
    return intervals, table


def _aging_with_factor(factor: Fraction) -> dict:
    aging = json.loads((_SPEC / "reference_aging.json").read_text())
    for entry in aging["factors"]:
        if entry["receipt"]["factor_id"].startswith("p1_ty2026_to_ty2027_"):
            entry["operational_factor"] = {
                "numerator": factor.numerator,
                "denominator": factor.denominator,
            }
    return aging


@settings(max_examples=60, deadline=None)
@given(
    numerator=st.integers(min_value=1, max_value=10**9),
    denominator=st.integers(min_value=1, max_value=10**9),
)
def test_property_edges_stay_contiguous_and_increasing_for_any_positive_factor(
    numerator, denominator
):
    factor = Fraction(numerator, denominator)
    intervals, table = _table()
    document = builder.earnings_bands_document(intervals, _aging_with_factor(factor))
    for profile in document["profiles"].values():
        bands = profile["map_bands"]["bands"]
        assert len(bands) == 235
        assert bands[0]["lower"] == 0.0 and bands[-1]["upper"] is None
        for left, right in zip(bands, bands[1:], strict=False):
            assert left["upper"] == right["lower"]
        finite = [row["upper"] for row in bands[:-1]]
        assert all(b > a for a, b in zip([0.0, *finite], finite, strict=False))
        assert bands[-1]["lower"] == float(Fraction(100_000_000) * factor / 100)


def _bands(edges, shares):
    return [
        {"lower": edges[index], "upper": edges[index + 1], "share": share}
        for index, share in enumerate(shares)
    ]


@pytest.mark.parametrize("profile", ["low", "central", "high"])
def test_scaled_edge_map_equals_unscaled_map_times_the_factor(profile):
    intervals, table = _table()
    document = json.loads((_SPEC / "earnings_bands.json").read_text())
    declared = document["profiles"][profile]
    factor = Fraction(
        declared["operational_factor"]["numerator"],
        declared["operational_factor"]["denominator"],
    )
    cuts = [table["first_lower_cut_cents"]] + [b["upper_cent"] for b in table["bands"]]
    unscaled = [None if cut is None else cut / 100 for cut in cuts]
    scaled = [declared["map_bands"]["bands"][0]["lower"]] + [
        row["upper"] for row in declared["map_bands"]["bands"]
    ]
    # Synthetic shares: geometric decay so the open top band holds mass.
    raw = np.geomspace(1.0, 1e-3, len(table["bands"]))
    shares = (raw / raw.sum()).tolist()
    shares[-1] = 1.0 - sum(shares[:-1])
    generator = np.random.default_rng(20261102)
    values = generator.lognormal(mean=10.8, sigma=0.9, size=4000)
    weights = generator.uniform(0.5, 2.0, size=4000)
    interpolation = {"method": "pareto", "alpha": 2.4}
    on_scaled = quantile_map(
        values, weights, _bands(scaled, shares), interpolation=interpolation
    )
    on_unscaled = quantile_map(
        values, weights, _bands(unscaled, shares), interpolation=interpolation
    ) * float(factor)
    relative = np.abs(on_scaled - on_unscaled) / on_unscaled
    assert relative.max() <= _RELATIVE_TOLERANCE, relative.max()
    # The map ranks identically on both edge sets.
    assert (
        np.argsort(on_scaled, kind="stable") == np.argsort(on_unscaled, kind="stable")
    ).all()
