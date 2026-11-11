from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceManifest, SourceStageSpec
from microcosm.build.uk_runtime.frs_council_tax import (
    SCOTTISH_BAND_D_EQUIVALENTS_CONCEPT,
    SCOTTISH_BAND_D_EQUIVALENTS_RECORD_SET,
    SCOTTISH_BAND_D_LIABILITY_RESOURCE,
    SCOTTISH_BAND_D_RATE_CONCEPT,
    SCOTTISH_BAND_D_RATES_RECORD_SET,
    SCOTTISH_CHARGES_IN_RECIPIENT_BILL,
    ScottishCouncilTaxLevel,
    assert_frs_council_tax_stage_parameters,
    derive_council_tax,
    level_scottish_council_tax,
    load_scottish_council_tax_level,
    scottish_council_tax_level_from_rows,
)
from microcosm.build.uk_runtime.frs_spine import WEEKS_IN_YEAR
from test_support.paths import paths_for

UK_PACKAGE = (
    paths_for("microcosm-build").repository
    / "packages/microcosm-build/src/microcosm/build/uk"
)


def _raw(**columns) -> pd.DataFrame:
    rows = len(columns["ctannual"])
    defaults = {
        "household_id": list(range(1, rows + 1)),
        # CTPAID 3, the household does not have to pay: a zero bill stays zero.
        "ctpaid": [3] * rows,
        "ctreb": [2] * rows,
        "ctrebamt": [0.0] * rows,
        "gvtregno": [1] * rows,
        "ctband": [4] * rows,
        "adulth": [2] * rows,
        "cwatamt1": [0.0] * rows,
        "csewamt1": [0.0] * rows,
        "cwatamtd": [0.0] * rows,
    }
    return pd.DataFrame({**defaults, **columns})


def _derive(raw: pd.DataFrame) -> pd.Series:
    return derive_council_tax(pd.DataFrame({"household_id": raw["household_id"]}), raw)


def test_council_tax_imputes_raw_missing_from_raw_cells() -> None:
    raw = _raw(
        gvtregno=[12, 12, 1, 1, 2, 2],
        ctband=[1, 1, np.nan, np.nan, 2, 2],
        adulth=[1, 1, 1, 1, 2, 2],
        ctannual=[1000.0, -1.0, 600.0, np.nan, -1.0, 0.0],
        # FRS 2024-25 shape: the gross Scottish cells CWATAMT1/CSEWAMT1.
        cwatamtd=[3.0, 3.0, 0.0, 0.0, 0.0, 0.0],
        cwatamt1=[4.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        csewamt1=[5.0, 5.0, 0.0, 0.0, 0.0, 0.0],
    )

    result = _derive(raw)

    # A non-recipient's bill carries the gross charges in full.
    scottish_tax = 1000.0 - (4.0 + 5.0) * WEEKS_IN_YEAR
    assert np.isclose(result.loc[1], scottish_tax)
    assert np.isclose(result.loc[2], scottish_tax)
    assert result.loc[4] == 600.0
    # No positive non-recipient bill in the cell: nothing to impute from.
    assert result.loc[5] == 0.0
    # A non-recipient's zero bill (an exempt dwelling) stays zero.
    assert result.loc[6] == 0.0


def test_paying_household_with_a_zero_bill_takes_the_cell_mean() -> None:
    raw = _raw(
        gvtregno=[1, 1, 1, 1, 1, 1, 1, 13, 2],
        ctannual=[1200.0, 1400.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ctpaid=[1, 1, 1, 1, 2, 3, np.nan, np.nan, 1],
        ctreb=[2, 2, 2, 1, 2, 2, 2, 2, 2],
        ctrebamt=[0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    result = _derive(raw)

    # Someone pays and no reduction is reported, so the household owes a bill.
    assert result.loc[3] == pytest.approx(1300.0)
    # A reported reduction is added back, as for any recipient.
    assert result.loc[4] == pytest.approx(20.0 * WEEKS_IN_YEAR)
    # Council-paid, exempt, blank and unasked (Northern Ireland) zeros stay zero.
    assert result.loc[5] == 0.0
    assert result.loc[6] == 0.0
    assert result.loc[7] == 0.0
    assert result.loc[8] == 0.0
    # No positive non-recipient bill in the cell: nothing to impute from.
    assert result.loc[9] == 0.0


def test_reported_reduction_is_added_back_to_the_bill() -> None:
    raw = _raw(
        ctannual=[0.0, 500.0, 1300.0],
        ctreb=[1, 1, 2],
        ctrebamt=[20.0, 10.0, 0.0],
    )

    result = _derive(raw)

    # A full reduction leaves CTANNUAL at zero; the liability is the reduction.
    assert result.loc[1] == pytest.approx(20.0 * WEEKS_IN_YEAR)
    assert result.loc[2] == pytest.approx(500.0 + 10.0 * WEEKS_IN_YEAR)
    assert result.loc[3] == pytest.approx(1300.0)


def test_unknown_reduction_takes_the_larger_of_bill_and_cell_mean() -> None:
    raw = _raw(
        ctannual=[1200.0, 1400.0, 300.0, 1500.0],
        ctreb=[2, 2, 1, 1],
        ctrebamt=[0.0, 0.0, 0.0, np.nan],
    )

    result = _derive(raw)

    assert result.loc[3] == pytest.approx(1300.0)
    assert result.loc[4] == pytest.approx(1500.0)


def test_cell_means_pool_non_recipients_only() -> None:
    raw = _raw(
        ctannual=[1200.0, 100.0, np.nan],
        ctreb=[2, 1, 2],
        ctrebamt=[0.0, 15.0, 0.0],
    )

    result = _derive(raw)

    # The recipient's net bill (100) would pull the cell mean down.
    assert result.loc[3] == pytest.approx(1200.0)
    assert result.loc[2] == pytest.approx(100.0 + 15.0 * WEEKS_IN_YEAR)


def test_scottish_recipients_net_the_reduced_share_of_the_gross_charges() -> None:
    raw = _raw(
        gvtregno=[12, 12],
        ctannual=[800.0, 1600.0],
        ctreb=[1, 2],
        ctrebamt=[5.0, 0.0],
        cwatamt1=[4.0, 4.0],
        csewamt1=[5.0, 5.0],
        cwatamtd=[3.0, 3.0],
    )

    result = _derive(raw)

    charges = (4.0 + 5.0) * WEEKS_IN_YEAR
    assert result.loc[1] == pytest.approx(
        800.0 - SCOTTISH_CHARGES_IN_RECIPIENT_BILL * charges + 5.0 * WEEKS_IN_YEAR
    )
    assert result.loc[2] == pytest.approx(1600.0 - charges)


def test_scottish_household_without_a_gross_cell_nets_its_recorded_charge() -> None:
    raw = _raw(
        gvtregno=[12],
        ctannual=[900.0],
        cwatamtd=[3.0],
    )

    assert _derive(raw).loc[1] == pytest.approx(900.0 - 3.0 * WEEKS_IN_YEAR)


def test_bills_outside_scotland_keep_ctannual_or_add_the_reduction() -> None:
    rng = np.random.default_rng(1095)
    rows = 400
    raw = _raw(
        gvtregno=rng.integers(1, 12, rows),
        ctband=rng.integers(1, 9, rows),
        adulth=rng.integers(1, 3, rows),
        ctannual=rng.choice([0.0, 250.0, 900.0, 1800.0, 2600.0], rows),
        ctreb=rng.choice([1, 2], rows),
        ctrebamt=rng.choice([0.0, 4.0, 18.0, 30.0], rows),
    )

    result = _derive(raw).to_numpy()
    ctannual = raw["ctannual"].to_numpy()
    recipient = raw["ctreb"].to_numpy() == 1
    reduction = raw["ctrebamt"].to_numpy() * WEEKS_IN_YEAR

    assert np.isfinite(result).all() and (result >= 0).all()
    np.testing.assert_allclose(result[~recipient], ctannual[~recipient])
    known = recipient & (reduction > 0)
    np.testing.assert_allclose(result[known], ctannual[known] + reduction[known])
    assert (result[recipient] >= ctannual[recipient] - 1e-9).all()


@pytest.mark.parametrize("column", ["ctreb", "ctpaid"])
def test_missing_raw_cells_refuse(column) -> None:
    raw = _raw(ctannual=[100.0]).drop(columns=column)

    with pytest.raises(KeyError, match=column):
        _derive(raw)


def _committed_stage() -> SourceStageSpec:
    manifest = SourceManifest.from_mapping(
        json.loads((UK_PACKAGE / "source_stages.json").read_text(encoding="utf-8"))
    )
    return next(stage for stage in manifest.stages if stage.stage == "frs_council_tax")


def test_committed_manifest_declares_the_implemented_rule() -> None:
    assert_frs_council_tax_stage_parameters(_committed_stage())


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("donor_filter", "raw ctannual > 0"),
        ("missing", "raw ctannual < 0 or NaN"),
        ("value", "Scottish-water-netted CTANNUAL"),
        ("cells", ["gvtregno", "ctband"]),
    ],
)
def test_manifest_drift_from_the_implemented_rule_refuses(parameter, value) -> None:
    stage = _committed_stage()
    operations = [
        {"kind": operation.kind, **dict(operation.parameters)}
        for operation in stage.operations
    ]
    operations[1][parameter] = value
    drifted = SourceStageSpec.from_mapping({**stage.__dict__, "operations": operations})
    with pytest.raises(ValueError, match=parameter):
        assert_frs_council_tax_stage_parameters(drifted)


@pytest.mark.parametrize(
    ("operation", "parameter", "value"),
    [
        (2, "record_sets", [SCOTTISH_BAND_D_EQUIVALENTS_RECORD_SET]),
        (2, "mapped_build_period", 2025),
        (3, "level", "Band D equivalents after council tax reduction"),
        (3, "weights", "calibrated_weights"),
    ],
)
def test_scottish_level_declaration_drift_refuses(operation, parameter, value) -> None:
    stage = _committed_stage()
    operations = [
        {"kind": item.kind, **dict(item.parameters)} for item in stage.operations
    ]
    operations[operation][parameter] = value
    drifted = SourceStageSpec.from_mapping({**stage.__dict__, "operations": operations})
    with pytest.raises(ValueError, match=parameter):
        assert_frs_council_tax_stage_parameters(drifted)


def test_a_stage_without_the_band_d_resource_refuses() -> None:
    stage = _committed_stage()
    artifacts = [
        dict(artifact)
        for artifact in stage.artifacts
        if artifact.get("resource") != SCOTTISH_BAND_D_LIABILITY_RESOURCE
    ]
    operations = [
        {"kind": item.kind, **dict(item.parameters)} for item in stage.operations
    ]
    missing = SourceStageSpec.from_mapping(
        {**stage.__dict__, "artifacts": artifacts, "operations": operations}
    )
    with pytest.raises(ValueError, match="scotgov_band_d_liability_facts"):
        assert_frs_council_tax_stage_parameters(missing)


def _area(index: int) -> str:
    return f"S12{index:06d}"


def _band_d_rows(
    equivalents: dict[str, float], rates: dict[str, float]
) -> list[dict[str, object]]:
    def row(record_set: str, concept: str, area: str, value: float) -> dict:
        return {
            "concept": concept,
            "geography": {"id": area, "level": "local_authority"},
            "layout": {"record_set_id": record_set},
            "value": value,
        }

    return [
        row(
            SCOTTISH_BAND_D_EQUIVALENTS_RECORD_SET,
            SCOTTISH_BAND_D_EQUIVALENTS_CONCEPT,
            area,
            value,
        )
        for area, value in equivalents.items()
    ] + [
        row(SCOTTISH_BAND_D_RATES_RECORD_SET, SCOTTISH_BAND_D_RATE_CONCEPT, area, value)
        for area, value in rates.items()
    ]


def _thirty_two() -> tuple[dict[str, float], dict[str, float]]:
    equivalents = {_area(index): 1_000.0 + index for index in range(32)}
    rates = {_area(index): 1_300.0 + 10.0 * index for index in range(32)}
    return equivalents, rates


def test_the_level_sums_each_council_areas_band_d_equivalents_times_its_band_d() -> (
    None
):
    equivalents, rates = _thirty_two()
    # Scotland's own row and the after-reduction table are not the level's.
    rows = _band_d_rows(equivalents, rates) + [
        {
            "concept": SCOTTISH_BAND_D_EQUIVALENTS_CONCEPT,
            "geography": {"id": "S92000003", "level": "country"},
            "layout": {"record_set_id": SCOTTISH_BAND_D_EQUIVALENTS_RECORD_SET},
            "value": 1.0,
        },
        {
            "concept": "scotgov.council_tax.band_d_equivalents_after_council_tax_reduction",
            "geography": {"id": _area(0), "level": "local_authority"},
            "layout": {
                "record_set_id": "scotgov.ctaxbase2025.band_d_equivalents_after_council_tax_reduction.y2024"
            },
            "value": 1.0,
        },
    ]

    level = scottish_council_tax_level_from_rows(rows)

    expected = sum(equivalents[area] * rates[area] for area in equivalents)
    assert level.total == pytest.approx(expected)
    assert level.receipt["council_areas"] == 32
    assert level.receipt["band_d_equivalents"] == pytest.approx(
        sum(equivalents.values())
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("drop_rate", "same 32 council areas"),
        ("drop_both", "same 32 council areas"),
        ("duplicate", "twice"),
        ("english", "non-Scottish"),
        ("zero", "non-positive"),
        ("concept", "carries concept"),
    ],
)
def test_the_level_refuses_rows_that_do_not_cover_the_councils_once(
    change, message
) -> None:
    equivalents, rates = _thirty_two()
    if change == "drop_rate":
        rates.pop(_area(5))
    if change == "drop_both":
        equivalents.pop(_area(5))
        rates.pop(_area(5))
    if change == "zero":
        rates[_area(3)] = 0.0
    rows = _band_d_rows(equivalents, rates)
    if change == "duplicate":
        rows.append(dict(rows[0]))
    if change == "english":
        rows[0] = {
            **rows[0],
            "geography": {"id": "E06000001", "level": "local_authority"},
        }
    if change == "concept":
        rows[0] = {**rows[0], "concept": "scotgov.council_tax.average_band_d"}
    with pytest.raises(ValueError, match=message):
        scottish_council_tax_level_from_rows(rows)


def test_the_level_scales_only_the_scottish_bills_to_the_published_total() -> None:
    bills = np.array([1_000.0, 1_500.0, 0.0, 2_000.0, 900.0])
    scottish = np.array([True, True, True, False, False])
    weights = np.array([10.0, 20.0, 5.0, 7.0, 3.0])
    stored = 10.0 * 1_000.0 + 20.0 * 1_500.0
    level = ScottishCouncilTaxLevel(
        total=1.08 * stored, receipt={"level_gbp": 1.08 * stored}
    )

    levelled, receipt = level_scottish_council_tax(
        bills, scottish=scottish, weights=weights, level=level
    )

    np.testing.assert_allclose(levelled[scottish], bills[scottish] * 1.08)
    np.testing.assert_array_equal(levelled[~scottish], bills[~scottish])
    assert float(np.dot(weights[scottish], levelled[scottish])) == pytest.approx(
        level.total
    )
    assert receipt["levelled"] is True
    assert receipt["factor"] == pytest.approx(1.08)
    assert receipt["stored_gbp"] == pytest.approx(stored)
    # Support as a band, never a count (three Scottish households here).
    assert receipt["households"] == "3-9"
    np.testing.assert_array_equal(bills, [1_000.0, 1_500.0, 0.0, 2_000.0, 900.0])


def test_a_frame_with_no_scottish_bill_keeps_its_bills() -> None:
    bills = np.array([0.0, 2_000.0])
    scottish = np.array([True, False])
    level = ScottishCouncilTaxLevel(total=3.4e9, receipt={})

    levelled, receipt = level_scottish_council_tax(
        bills, scottish=scottish, weights=np.ones(2), level=level
    )

    np.testing.assert_array_equal(levelled, bills)
    assert receipt["levelled"] is False
    assert receipt["factor"] is None


def test_the_committed_resource_gives_scotlands_2024_25_liability() -> None:
    level = load_scottish_council_tax_level()

    assert level.receipt["council_areas"] == 32
    # CTAXBASE x CTAS gives about GBP 3.4bn before council tax reduction for
    # 2024-25; a units or period slip would land far outside this range.
    assert 3.0e9 < level.total < 3.8e9
