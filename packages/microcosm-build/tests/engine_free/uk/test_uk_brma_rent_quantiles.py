"""The committed rent quantiles by BRMA and category, and their builder (microcosm#1262)."""

from __future__ import annotations

import copy
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.uk_runtime import brma_area_layer as bal
from microcosm.build.uk_runtime import private_rent_localisation as prl
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
_UK = _TEST_PATHS.package / "src/microcosm/build/uk"
_TOOL = _TEST_PATHS.repository / "tools/build_uk_brma_rent_quantiles.py"
_SPEC = importlib.util.spec_from_file_location("build_uk_brma_rent_quantiles", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)


def _payload() -> bytes:
    return (_UK / "brma_rent_quantiles.json").read_bytes()


def _data() -> dict:
    return json.loads(_payload())


# ----------------------------------------------------------------------------
# The committed resource


def test_committed_table_loads_and_covers_every_layer_brma() -> None:
    table = prl.load_rent_quantiles(_payload())
    layer = bal.load_brma_area_layer((_UK / "brma_area_layer.json").read_bytes())
    names = set(layer.brma_by_area["ew_lsoa21"].values()) | set(
        layer.brma_by_area["scotland_oa22"].values()
    )
    assert {brma for brma, _ in table.knots} == names
    assert set(table.knots) == {(brma, c) for brma in names for c in prl.CATEGORIES}
    receipts = _data()["receipts"]
    assert receipts["brmas"] == {"england": 152, "wales": 22, "scotland": 18}


def test_bedrooms_order_the_medians_in_most_brmas() -> None:
    """A sanity check on the parsing: bigger homes cost more almost everywhere."""

    table = prl.load_rent_quantiles(_payload())
    brmas = {brma for brma, _ in table.knots}
    ordered = sum(
        all(
            np.interp(0.5, table.grid, table.knots[(brma, a)])
            <= np.interp(0.5, table.grid, table.knots[(brma, b)])
            for a, b in zip(prl.CATEGORIES, prl.CATEGORIES[1:], strict=False)
        )
        for brma in brmas
    )
    assert ordered / len(brmas) > 0.9


def test_scottish_cells_reproduce_their_published_quartiles() -> None:
    grid = np.asarray(prl.GRID)
    for brma, cells in _data()["brmas"].items():
        if cells["nation"] != "scotland":
            continue
        for category in prl.CATEGORIES:
            cell = cells[category]
            knots = np.asarray(cell["quantiles"])
            published = cell["published_weekly"]
            for p, measure in (
                (0.25, "lower_quartile"),
                (0.5, "median"),
                (0.75, "upper_quartile"),
            ):
                assert np.interp(p, grid, knots) == pytest.approx(
                    published[measure], rel=1e-3
                )
            gap = cell["mean"] / published["mean"] - 1.0
            assert abs(gap) <= tool.SCOTTISH_MEAN_TOLERANCE, (brma, category)


def test_sources_are_the_pinned_registrations() -> None:
    sources = _data()["sources"]
    for name, pin in tool.REGISTRATIONS.items():
        assert sources[name]["sha256"] == pin["sha256"]
        assert sources[name]["registration_kind"] == "raw_only"
    assert (
        sources["scotland_statistics"]["source_sha256"]
        == (tool.SCOTTISH_STATISTICS["source_sha256"])
    )
    assert _data()["licence"]["licence"] == "Open Government Licence v3.0"


# ----------------------------------------------------------------------------
# The loader


def _small() -> dict:
    knots = [10.0 * (i + 1) for i in range(len(prl.GRID))]
    mean = prl.knot_mean(prl.GRID, knots)
    cell = {"lets": 60, "basis": "own_lets", "quantiles": knots, "mean": mean}
    return {
        "schema_version": 1,
        "kind": prl.QUANTILES_KIND,
        "method": prl.QUANTILES_METHOD,
        "unit": "gbp_per_week",
        "grid": list(prl.GRID),
        "categories": list(prl.CATEGORIES),
        "brmas": {
            "CARDIFF": {"nation": "wales", **{c: dict(cell) for c in prl.CATEGORIES}}
        },
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(kind="x"), "schema-1"),
        (lambda d: d.update(method="v0"), "method"),
        (lambda d: d.update(unit="gbp_per_month"), "weekly"),
        (lambda d: d["grid"].pop(), "grid"),
        (lambda d: d["brmas"]["CARDIFF"].pop("E"), "no category E"),
        (
            lambda d: d["brmas"]["CARDIFF"]["B"]["quantiles"].reverse(),
            "must not decrease",
        ),
        (
            lambda d: d["brmas"]["CARDIFF"]["B"]["quantiles"].__setitem__(0, -1.0),
            "positive",
        ),
        (lambda d: d["brmas"]["CARDIFF"]["C"].update(mean=1.0), "mean"),
    ],
)
def test_loader_refuses_malformed_tables(mutate, message) -> None:
    payload = copy.deepcopy(_small())
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        prl.load_rent_quantiles(json.dumps(payload).encode())


@settings(max_examples=50, deadline=None)
@given(
    st.lists(
        st.floats(min_value=1.0, max_value=5_000.0, allow_nan=False),
        min_size=len(prl.GRID),
        max_size=len(prl.GRID),
    )
)
def test_knot_mean_is_the_integral_of_the_quantile_function(values) -> None:
    knots = np.sort(np.asarray(values))
    u = (np.arange(200_000) + 0.5) / 200_000
    numeric = float(prl.quantile(np.asarray(prl.GRID), knots, u).mean())
    assert prl.knot_mean(prl.GRID, knots) == pytest.approx(numeric, rel=1e-6)


# ----------------------------------------------------------------------------
# The builder's cell rules


def _lets(cells: dict[tuple[str, str], int], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for (brma, category), n in cells.items():
        level = {"B": 100.0, "C": 130.0, "D": 160.0, "E": 220.0}[category]
        level *= {"X": 1.0, "Y": 1.5}[brma]
        frames.append(
            pd.DataFrame(
                {
                    "brma": brma,
                    "category": category,
                    "rent": level * np.exp(rng.normal(0.0, 0.2, n)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_list_cells_choose_their_basis_by_lets() -> None:
    lets = _lets(
        {("X", c): 80 for c in prl.CATEGORIES}
        | {("Y", "B"): 80, ("Y", "C"): 80, ("Y", "D"): 30, ("Y", "E"): 5}
    )
    cells = tool.list_cells(lets, "england", {"X", "Y"})
    assert cells["X"]["B"]["basis"] == "own_lets"
    assert cells["Y"]["D"]["basis"] == "own_median_nation_shape"
    assert cells["Y"]["E"]["basis"] == "modelled_median_nation_shape"
    # The modelled median carries Y's level and E's relativity.
    grid = np.asarray(prl.GRID)
    modelled = np.interp(0.5, grid, cells["Y"]["E"]["quantiles"])
    assert modelled == pytest.approx(220.0 * 1.5, rel=0.15)
    for brma in ("X", "Y"):
        for category in prl.CATEGORIES:
            assert np.all(np.diff(cells[brma][category]["quantiles"]) >= 0)


def test_category_parser_reads_both_spellings() -> None:
    assert tool._category("Cat C") == "C"
    assert tool._category("E") == "E"
    with pytest.raises(SystemExit):
        tool._category("Cat F")
