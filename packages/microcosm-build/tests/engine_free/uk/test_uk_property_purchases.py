"""Annual property purchases, ported from uk-data#573 (microcosm#1095)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime import property_purchases as pp
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind

_HRAD = "hmrc.stamp_taxes_2024_25.table_6b.hrad.fy2024"
_RES_IND = "hmrc.stamp_taxes_2024_25.table_8a.individuals.fy2024"
_RES_OTHER = "hmrc.stamp_taxes_2024_25.table_8a.other_buyers.fy2024"
_NONRES_IND = "hmrc.stamp_taxes_2024_25.table_8b.individuals.fy2024"
#: Table 6b's ten bands (count, value in GBP m), and the five above GBP 2m.
_HRAD_BANDS = {
    "gbp_250k_or_less": (1_000.0, 150.0),
    "gbp_250k_to_500k": (500.0, 180.0),
    "gbp_500k_to_1m": (200.0, 140.0),
    "gbp_1m_to_1_5m": (40.0, 48.0),
    "gbp_1_5m_to_2m": (20.0, 35.0),
    "gbp_2m_to_3m": (8.0, 20.0),
    "gbp_3m_to_4m": (4.0, 14.0),
    "gbp_4m_to_5m": (2.0, 9.0),
    "gbp_5m_to_10m": (2.0, 14.0),
    "over_gbp_10m": (1.0, 15.0),
}
_NONRES_BANDS = {
    "gbp_250k_or_less": (300.0, 25.0),
    "gbp_250k_to_500k": (60.0, 21.0),
    "gbp_500k_to_1m": (25.0, 18.0),
    "gbp_1m_to_1_5m": (8.0, 10.0),
    "gbp_1_5m_to_2m": (4.0, 7.0),
    "over_gbp_2m": (6.0, 30.0),
}


def _row(record_set: str, concept: str, band: str, value: float) -> dict:
    return {
        "concept": concept,
        "layout": {"record_set_id": record_set, "groupby_value_id": band},
        "value": value,
    }


def _rows() -> list[dict]:
    rows = []
    for band, (count, value) in _HRAD_BANDS.items():
        rows.append(_row(_HRAD, "hmrc.sdlt.hrad_transactions", band, count))
        rows.append(_row(_HRAD, "hmrc.sdlt.hrad_property_value", band, value * 1e6))
    rows.append(_row(_HRAD, "hmrc.sdlt.hrad_transactions", "all_price_bands", 1_800.0))
    rows.append(_row(_HRAD, "hmrc.sdlt.hrad_property_value", "all_price_bands", 625e6))
    for record_set, count, value in (
        (_RES_IND, 9_000.0, 3_000e6),
        (_RES_OTHER, 1_000.0, 1_000e6),
    ):
        rows.append(
            _row(record_set, "hmrc.sdlt.transactions", "all_price_bands", count)
        )
        rows.append(
            _row(record_set, "hmrc.sdlt.property_value", "all_price_bands", value)
        )
    for band, (count, value) in _NONRES_BANDS.items():
        rows.append(_row(_NONRES_IND, "hmrc.sdlt.transactions", band, count))
        rows.append(_row(_NONRES_IND, "hmrc.sdlt.property_value", band, value * 1e6))
    rows.append(_row(_NONRES_IND, "hmrc.sdlt.transactions", "all_price_bands", 400.0))
    rows.append(_row(_NONRES_IND, "hmrc.sdlt.property_value", "all_price_bands", 110e6))
    return rows


def test_targets_allocate_hrad_by_individuals_shares_and_combine_top_bands() -> None:
    targets = pp.property_purchase_targets_from_rows(_rows())
    additional = targets.additional_residential
    # uk-data#573: Table 6b's all-buyer HRAD times individuals' Table 8a shares.
    assert additional.transactions == pytest.approx(1_800.0 * 9_000 / 10_000)
    assert additional.value == pytest.approx(625e6 * 3_000 / 4_000)
    assert additional.band_ids == pp.PROPERTY_PURCHASE_BAND_IDS
    # The five bands above GBP 2m become one: 8+4+2+2+1 transactions.
    assert additional.band_transactions[-1] == 17.0
    assert additional.band_values[-1] == pytest.approx(72e6)
    assert additional.minimum_price == pp.HMRC_SDLT_MINIMUM_CONSIDERATION
    non_residential = targets.non_residential
    assert non_residential.transactions == 400.0
    assert non_residential.value == pytest.approx(110e6)
    assert non_residential.band_transactions == tuple(
        count for count, _ in _NONRES_BANDS.values()
    )
    assert targets.derivation["residential_individuals_transaction_share"] == 0.9
    assert targets.derivation["residential_individuals_value_share"] == 0.75


def test_targets_round_trip_through_the_payload() -> None:
    targets = pp.property_purchase_targets_from_rows(_rows())
    assert pp.UKPropertyPurchaseTargets.from_payload(targets.payload()) == targets


def test_targets_refuse_missing_totals_duplicates_and_odd_bands() -> None:
    rows = [row for row in _rows() if row["layout"]["record_set_id"] != _RES_OTHER]
    with pytest.raises(ValueError, match="carries no"):
        pp.property_purchase_targets_from_rows(rows)
    rows = _rows()
    rows.append(_row(_NONRES_IND, "hmrc.sdlt.transactions", "gbp_250k_or_less", 1.0))
    with pytest.raises(ValueError, match="repeats band"):
        pp.property_purchase_targets_from_rows(rows)
    rows = [
        row
        for row in _rows()
        if not (
            row["layout"]["record_set_id"] == _HRAD
            and row["layout"]["groupby_value_id"] == "over_gbp_10m"
        )
    ]
    with pytest.raises(ValueError, match="bands to combine are missing"):
        pp.property_purchase_targets_from_rows(rows)


def _target(
    transactions: float = 30.0, value: float = 9e6
) -> pp.UKPropertyPurchaseTarget:
    return pp.UKPropertyPurchaseTarget(
        property_type="additional_residential",
        transactions=transactions,
        value=value,
        band_ids=pp.PROPERTY_PURCHASE_BAND_IDS,
        band_transactions=(50.0, 30.0, 10.0, 5.0, 3.0, 2.0),
        band_values=(10e6, 12e6, 8e6, 6e6, 5e6, 8e6),
        minimum_price=40_000.0,
    )


def _inputs(n: int = 400, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    stock = np.where(rng.random(n) < 0.6, rng.lognormal(12.5, 1.0, n), 0.0)
    stock[:5] = 30_000.0  # below the floor: never eligible
    weights = rng.uniform(0.5, 2.0, n)
    weights[5] = 0.0  # zero weight: never buys
    stock[5] = 500_000.0
    sdlt = rng.random(n) < 0.85
    return {
        "stock": stock,
        "weights": weights,
        "sdlt_mask": sdlt,
        "event_uniforms": rng.random(n),
        "price_uniforms": rng.random(n),
    }


def test_prices_fit_the_count_and_value_within_the_floor_and_the_stock() -> None:
    inputs = _inputs()
    target = _target()
    result = pp.impute_property_purchase_prices(**inputs, target=target)
    prices = result.prices
    bought = prices > 0
    fitted = bought & inputs["sdlt_mask"]
    weights = inputs["weights"]
    # The closest attainable count, and the value fitted exactly.
    assert abs(weights[fitted].sum() - target.transactions) <= weights.max()
    assert float(np.dot(prices, weights * inputs["sdlt_mask"])) == pytest.approx(
        target.value, rel=1e-8
    )
    assert np.all(prices[bought] >= target.minimum_price)
    assert np.all(prices <= inputs["stock"] + 1e-6)
    assert not bought[:6].any()  # below the floor, or at zero weight
    assert result.receipt["fitted_value"] == pytest.approx(target.value, rel=1e-8)
    assert result.receipt["purchaser_support"] in {"<10", "10-29", "30+"}


def test_the_threshold_applies_outside_the_fit_regions() -> None:
    inputs = _inputs()
    inputs["sdlt_mask"] = np.ones_like(inputs["sdlt_mask"])
    inputs["sdlt_mask"][-50:] = False
    result = pp.impute_property_purchase_prices(**inputs, target=_target())
    outside = result.prices[-50:] > 0
    # Scotland and Wales buy at the same hazard and threshold, unfitted.
    assert outside.any()
    assert result.receipt["outside_sdlt_regions_support"] in {"<10", "10-29", "30+"}


def test_receipt_cells_on_fewer_than_ten_sources_are_banded() -> None:
    # UKDS disclosure control: a weighted figure one to nine source
    # households carry is "<10", and so is anything derived from it.
    inputs = _inputs()
    inputs["sdlt_mask"] = np.ones_like(inputs["sdlt_mask"])
    inputs["sdlt_mask"][-50:] = False
    n = len(inputs["stock"])
    # Every row outside the fit regions belongs to one of three households.
    sources = np.arange(n)
    sources[-50:] = 10_000 + np.arange(50) % 3
    result = pp.impute_property_purchase_prices(
        **inputs, target=_target(), source_ids=sources
    )
    receipt = result.receipt
    assert (result.prices[-50:] > 0).any()
    assert receipt["outside_sdlt_regions_support"] == "<10"
    assert receipt["purchases_outside_sdlt_regions_weight"] == "<10"
    assert receipt["fitted_value"] == pytest.approx(9e6, rel=1e-8)
    # Every fit-region row belongs to one of five households.
    few = np.arange(n) % 5
    result = pp.impute_property_purchase_prices(
        **inputs, target=_target(), source_ids=few
    )
    for field in (
        "fitted_transactions",
        "fitted_value",
        "value_shortfall_share",
        "eligible_owner_weight",
        "event_rate",
        "price_scale",
    ):
        assert result.receipt[field] == "<10", field
    assert result.receipt["purchaser_support"] == "<10"


def _capped_inputs(capped: int, uncapped: int) -> dict[str, np.ndarray]:
    """Every row buys in the fit regions: ``capped`` rows hold the £40,000
    floor, so they buy at their stock at any scale, and ``uncapped`` rows hold
    more than any band draw."""

    n = capped + uncapped
    rng = np.random.default_rng(13)
    return {
        "stock": np.r_[np.full(capped, 40_000.0), np.full(uncapped, 1e9)],
        "weights": np.ones(n),
        "sdlt_mask": np.ones(n, dtype=bool),
        "event_uniforms": rng.random(n),
        "price_uniforms": rng.random(n),
    }


def _capped_receipt(capped: int, uncapped: int, sources: np.ndarray) -> dict:
    # A value the uncapped rows' scaled band draws make up beside the floor.
    target = _target(
        transactions=float(capped + uncapped),
        value=capped * 40_000.0 + uncapped * 480_000.0,
    )
    receipt = pp.impute_property_purchase_prices(
        **_capped_inputs(capped, uncapped), target=target, source_ids=sources
    ).receipt
    assert receipt["value_fit"] == "scaled_band_draws"
    return receipt


def test_the_stock_cap_share_counts_each_side_on_its_sources() -> None:
    # Twelve capped rows past ten, but on two sources: the share is withheld,
    # though counting rows would give it.
    receipt = _capped_receipt(12, 28, np.r_[np.arange(12) % 2, 100 + np.arange(28)])
    assert receipt["stock_cap_support"] == "<10"
    assert receipt["weighted_share_at_stock_cap"] == "<10"
    # Three uncapped rows that copy capped sources: the uncapped side rests on
    # three sources, so the share, which would give its weight back, is
    # withheld too.
    receipt = _capped_receipt(37, 3, np.r_[np.arange(37), np.arange(3)])
    assert receipt["stock_cap_support"] == "10-49"
    assert receipt["weighted_share_at_stock_cap"] == "<10"
    # Both sides on ten or more sources: the share is given.
    receipt = _capped_receipt(12, 28, np.arange(40))
    assert receipt["weighted_share_at_stock_cap"] == pytest.approx(12 / 40)


def test_a_fit_on_fewer_than_ten_sources_withholds_the_share_at_the_cap() -> None:
    # At the stock cap every purchaser buys at its stock, so a share of 1.0
    # beside a "<10" fit would still say so: it is withheld.
    result = pp.impute_property_purchase_prices(
        **_capped_inputs(12, 0),
        target=_target(transactions=12.0, value=12 * 40_000.0 / 0.9),
        source_ids=np.arange(12) % 5,
    )
    assert result.receipt["value_fit"] == "stock_cap"
    assert result.receipt["purchaser_support"] == "<10"
    assert result.receipt["weighted_share_at_stock_cap"] == "<10"


def test_an_empty_cell_reads_zero() -> None:
    # No purchase outside the fit regions: "0" beside a zero weight, not "<10".
    receipt = _capped_receipt(12, 28, np.arange(40))
    assert receipt["purchases_outside_sdlt_regions_weight"] == 0.0
    assert receipt["outside_sdlt_regions_support"] == "0"


def test_infeasible_targets_refuse() -> None:
    inputs = _inputs()
    with pytest.raises(ValueError, match="less weight than the purchase target"):
        pp.impute_property_purchase_prices(**inputs, target=_target(transactions=1e9))
    with pytest.raises(ValueError, match="carries less than 80%"):
        pp.impute_property_purchase_prices(**inputs, target=_target(value=1e15))
    with pytest.raises(ValueError, match="below the minimum transaction price"):
        pp.impute_property_purchase_prices(**inputs, target=_target(value=1.0))


def test_a_short_stock_buys_at_the_stock_and_records_the_shortfall() -> None:
    # María's ruling of 2026-10-16: where the purchasers' stock cannot carry
    # HMRC's value, each buys at its stock and the shortfall is recorded; the
    # stage refuses only below PURCHASE_MINIMUM_VALUE_SHARE of the value.
    inputs = _inputs()
    mask, _ = pp.select_property_purchasers(
        stock=inputs["stock"],
        weights=inputs["weights"],
        sdlt_mask=inputs["sdlt_mask"],
        event_uniforms=inputs["event_uniforms"],
        transactions=30.0,
        minimum_price=40_000.0,
        property_type="additional_residential",
    )
    fitted = mask & inputs["sdlt_mask"]
    capacity = float(np.dot(inputs["stock"], inputs["weights"] * fitted))
    result = pp.impute_property_purchase_prices(
        **inputs, target=_target(transactions=30.0, value=capacity / 0.9)
    )
    np.testing.assert_allclose(result.prices[mask], inputs["stock"][mask])
    assert not result.prices[~mask].any()
    receipt = result.receipt
    assert receipt["value_fit"] == "stock_cap"
    assert receipt["price_scale"] is None
    assert receipt["fitted_value"] == pytest.approx(capacity)
    assert receipt["value_shortfall_share"] == pytest.approx(0.1)
    assert receipt["weighted_share_at_stock_cap"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="carries less than 80%"):
        pp.impute_property_purchase_prices(
            **inputs, target=_target(transactions=30.0, value=capacity / 0.75)
        )
    fitted_in_full = pp.impute_property_purchase_prices(**inputs, target=_target())
    assert fitted_in_full.receipt["value_fit"] == "scaled_band_draws"
    assert fitted_in_full.receipt["value_shortfall_share"] == pytest.approx(0.0)


def _frame(n: int = 300, seed: int = 3, time_period: str = "2024"):
    rng = np.random.default_rng(seed)
    regions = np.array(
        ["LONDON", "SOUTH_EAST", "NORTHERN_IRELAND", "SCOTLAND", "WALES", "NORTH_WEST"]
    )
    household = pd.DataFrame(
        {
            "household_id": np.arange(1, n + 1, dtype="int64"),
            "household_source_id": np.arange(1, n + 1, dtype="int64"),
            "region": pd.array(regions[rng.integers(0, 6, n)], dtype="string"),
            "other_residential_property_value": np.where(
                rng.random(n) < 0.5, rng.lognormal(12.6, 0.8, n), 0.0
            ),
            "non_residential_property_value": np.where(
                rng.random(n) < 0.4, rng.lognormal(12.4, 0.9, n), 0.0
            ),
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.arange(1, n + 1, dtype="int64")})
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, n + 1, dtype="int64"),
            "person_household_id": np.arange(1, n + 1, dtype="int64"),
            "person_benunit_id": np.arange(1, n + 1, dtype="int64"),
            "age": np.full(n, 50, dtype="int64"),
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=time_period,
        weight_kind=WeightKind.DESIGN,
        household_weights=rng.uniform(1.0, 3.0, n),
    )


def _frame_targets(non_residential_value: float = 10e6) -> pp.UKPropertyPurchaseTargets:
    residential = _target(transactions=40.0, value=18e6)
    return pp.UKPropertyPurchaseTargets(
        additional_residential=residential,
        non_residential=pp.UKPropertyPurchaseTarget(
            property_type="non_residential",
            transactions=30.0,
            value=non_residential_value,
            band_ids=residential.band_ids,
            band_transactions=residential.band_transactions,
            band_values=residential.band_values,
            minimum_price=40_000.0,
        ),
    )


def test_the_stage_writes_both_purchases_and_conserves_mass() -> None:
    frame = _frame()
    result = pp.impute_property_purchases(frame, targets=_frame_targets())
    household = result.frame.table("household")
    for column in pp.PROPERTY_PURCHASES_OUTPUT_COLUMNS:
        assert column in household.columns
    assert np.all(
        household["additional_residential_property_purchased"]
        <= household["other_residential_property_value"] + 1e-6
    )
    assert np.all(
        household["non_residential_property_purchased"]
        <= household["non_residential_property_value"] + 1e-6
    )
    # Both kinds buy: positive prices, and the fit reaches its targets.
    for column in pp.PROPERTY_PURCHASES_OUTPUT_COLUMNS:
        assert (household[column] > 0).any(), column
    for kind in result.evidence()["kinds"]:
        assert kind["fitted_transactions"] > 0
        assert kind["fitted_value"] > 0
    record = result.frame.mass_log[-1]
    assert record.declared_factor == 1.0
    assert record.reason == pp.PROPERTY_PURCHASES_MASS_CHANGE_REASON
    assert record.old_total == pytest.approx(record.new_total)
    evidence = result.evidence()
    assert [kind["property_type"] for kind in evidence["kinds"]] == [
        "additional_residential",
        "non_residential",
    ]


def test_the_stage_fits_one_kind_at_its_stock_beside_one_it_scales() -> None:
    # As on the national spine: the non-residential purchasers' stock carries
    # nine tenths of the value, so they buy at it, while additional dwellings
    # fit their value with scaled band draws.
    frame = _frame()
    household = frame.table("household")
    weights = np.asarray(frame.weights_for("household").values)
    sdlt = pp.sdlt_region_mask(household["region"])
    kind = next(k for k in pp.PROPERTY_PURCHASE_KINDS if k.name == "non_residential")
    stock = household[kind.stock].to_numpy(dtype=float)
    event_uniforms, _ = pp.property_purchase_uniforms(
        household["household_id"].to_numpy(), kind
    )
    purchased, _ = pp.select_property_purchasers(
        stock=stock,
        weights=weights,
        sdlt_mask=sdlt,
        event_uniforms=event_uniforms,
        transactions=30.0,
        minimum_price=40_000.0,
        property_type=kind.name,
    )
    capacity = float(np.dot(stock, weights * (purchased & sdlt)))
    result = pp.impute_property_purchases(
        frame, targets=_frame_targets(non_residential_value=capacity / 0.9)
    )
    kinds = {k["property_type"]: k for k in result.evidence()["kinds"]}
    assert kinds["additional_residential"]["value_fit"] == "scaled_band_draws"
    assert kinds["non_residential"]["value_fit"] == "stock_cap"
    assert kinds["non_residential"]["value_shortfall_share"] == pytest.approx(0.1)
    after = result.frame.table("household")
    np.testing.assert_allclose(
        after[kind.output].to_numpy()[purchased], stock[purchased]
    )


def test_the_packaged_targets_reproduce_uk_data_573() -> None:
    # The vendored HMRC Stamp Tax rows give uk-data#573's household figures.
    targets = pp.load_property_purchase_targets()
    additional = targets.additional_residential
    assert additional.transactions == pytest.approx(186_588.8624, rel=1e-9)
    assert additional.value == pytest.approx(68_805_261_766.68, rel=1e-9)
    non_residential = targets.non_residential
    assert non_residential.transactions == pytest.approx(25_000.0)
    assert non_residential.value == pytest.approx(7.64e9)
    assert additional.band_ids == pp.PROPERTY_PURCHASE_BAND_IDS


def test_draws_are_keyed_by_household_identity() -> None:
    ids = np.arange(1, 501, dtype="int64")
    subset = ids[::3]
    for kind in pp.PROPERTY_PURCHASE_KINDS:
        full_event, full_price = pp.property_purchase_uniforms(ids, kind)
        part_event, part_price = pp.property_purchase_uniforms(subset, kind)
        # A household's uniforms do not depend on which others are drawn.
        np.testing.assert_array_equal(full_event[::3], part_event)
        np.testing.assert_array_equal(full_price[::3], part_price)
        assert not np.allclose(full_event, full_price)
    residential, non_residential = (
        pp.property_purchase_uniforms(ids, kind)[0]
        for kind in pp.PROPERTY_PURCHASE_KINDS
    )
    # Each kind reads its own stream.
    assert not np.allclose(residential, non_residential)


def test_the_stage_is_deterministic() -> None:
    frame = _frame()
    first = pp.impute_property_purchases(frame, targets=_frame_targets())
    second = pp.impute_property_purchases(frame, targets=_frame_targets())
    columns = list(pp.PROPERTY_PURCHASES_OUTPUT_COLUMNS)
    pd.testing.assert_frame_equal(
        first.frame.table("household")[columns],
        second.frame.table("household")[columns],
    )


def test_the_stage_refuses_a_stored_output_or_a_missing_lineage() -> None:
    frame = _frame()
    stored = pp.impute_property_purchases(frame, targets=_frame_targets()).frame
    with pytest.raises(ValueError, match="already stored"):
        pp.impute_property_purchases(stored, targets=_frame_targets())
    household = frame.table("household").drop(columns="household_source_id")
    bare = uk_national_frame(
        person=frame.table("person"),
        benunit=frame.table("benunit"),
        household=household,
        time_period="2024",
        weight_kind=WeightKind.DESIGN,
        household_weights=np.asarray(frame.weights_for("household").values),
    )
    with pytest.raises(ValueError, match="household_source_id"):
        pp.impute_property_purchases(bare, targets=_frame_targets())


def test_the_stage_refuses_a_frame_outside_the_mapped_period() -> None:
    # The 2024-25 Stamp Tax facts are mapped to build period 2024 only.
    with pytest.raises(ValueError, match="build period 2024"):
        pp.impute_property_purchases(
            _frame(time_period="2030"), targets=_frame_targets()
        )


def test_the_packaged_stage_carries_the_reviewed_operations_in_order() -> None:
    stages = {stage.stage: stage for stage in load_country_spec("uk").sources.stages}
    stage = stages["property_purchases"]
    # Closed world: the transform refuses any drift from the code's constants.
    pp.UKPropertyPurchasesStageTransform(stage=stage)
    order = list(stages)
    position = order.index("property_purchases")
    # After the uprating and the support copies, before any CGT clone.
    assert order.index("regional_property_uprating") < position
    assert order.index("high_value_home_support") < position
    assert order.index("property_components") < position
    assert position < order.index("cgt_support_split")
