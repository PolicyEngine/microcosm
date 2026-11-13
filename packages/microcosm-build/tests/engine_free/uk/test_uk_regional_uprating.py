from __future__ import annotations

import pandas as pd
import pytest

from microcosm.build.uk_runtime.regional_uprating import (
    uprate_household_property_by_region,
)


def test_regional_property_uprating_scales_owners_only_and_skips_missing_regions():
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4, 5],
            "region": ["LONDON", "LONDON", "SCOTLAND", "SCOTLAND", "NORTHERN_IRELAND"],
            "main_residence_value": [100.0, 0.0, 200.0, 400.0, 500.0],
            "property_wealth": [150.0, 10.0, 300.0, 600.0, 700.0],
        }
    )
    resource = {
        "values": [
            {"region": "LONDON", "avg_house_price": 200.0, "dwellings": 1},
            {"region": "SCOTLAND", "avg_house_price": 600.0, "dwellings": 1},
        ]
    }

    uprated = uprate_household_property_by_region(household, resource)

    assert uprated.loc[0, "main_residence_value"] == pytest.approx(200.0)
    assert uprated.loc[0, "property_wealth"] == pytest.approx(300.0)
    assert uprated.loc[1, "main_residence_value"] == 0.0
    assert uprated.loc[1, "property_wealth"] == 10.0
    assert uprated.loc[2, "main_residence_value"] == pytest.approx(400.0)
    assert uprated.loc[3, "main_residence_value"] == pytest.approx(800.0)
    assert uprated.loc[4, "main_residence_value"] == 500.0


def test_regional_property_uprating_requires_columns() -> None:
    with pytest.raises(KeyError, match="property-uprating"):
        uprate_household_property_by_region(pd.DataFrame({"region": ["LONDON"]}), {})


def test_stage_transform_rewrites_property_and_records_its_receipt() -> None:
    import numpy as np

    from microcosm.build.source_manifest import SourceStageSpec
    from microcosm.build.uk_runtime.national_frame import uk_national_frame
    from microcosm.build.uk_runtime.regional_uprating import (
        UK_REGIONAL_PROPERTY_UPRATING_MASS_CONSERVATION_REASON,
        UKRegionalPropertyUpratingStageTransform,
    )
    from microcosm.frame import WeightKind

    frame = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2],
                "person_benunit_id": [1, 2],
                "person_household_id": [1, 2],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1, 2]}),
        household=pd.DataFrame(
            {
                "household_id": [1, 2],
                "region": ["LONDON", "SCOTLAND"],
                "main_residence_value": [100.0, 200.0],
                "property_wealth": [150.0, 300.0],
            }
        ),
        time_period="2024",
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=np.array([10.0, 30.0]),
    )
    stage = SourceStageSpec.from_mapping(
        {
            "stage": "regional_property_uprating",
            "survey": "test",
            "source": "test",
            "grain": "household",
            "artifacts": [],
            "operations": [{"kind": "uprate_to_regional_reference"}],
            "outputs": [],
            "rewrites": ["main_residence_value", "property_wealth"],
        }
    )
    transform = UKRegionalPropertyUpratingStageTransform(
        stage=stage,
        resource={
            "values": [
                {"region": "LONDON", "avg_house_price": 200.0, "dwellings": 1},
                {"region": "SCOTLAND", "avg_house_price": 600.0, "dwellings": 1},
            ]
        },
    )

    result = transform(frame)

    household = result.table("household")
    assert household["main_residence_value"].tolist() == pytest.approx([200.0, 600.0])
    assert result.weights_for("household").values.tolist() == [10.0, 30.0]
    receipt = result.mass_log[-1]
    assert receipt.reason == UK_REGIONAL_PROPERTY_UPRATING_MASS_CONSERVATION_REASON
    assert receipt.old_total == receipt.new_total == 40.0
    assert receipt.declared_factor == 1.0


def test_regional_property_uprating_factor_excludes_spi_support_rows() -> None:
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4],
            "region": ["LONDON", "LONDON", "LONDON", "LONDON"],
            "household_support_channel": ["frs", "frs", "spi", "spi"],
            "main_residence_value": [100.0, 300.0, 1_000.0, 0.0],
            "property_wealth": [100.0, 300.0, 1_500.0, 50.0],
        }
    )
    resource = {
        "values": [{"region": "LONDON", "avg_house_price": 400.0, "dwellings": 1}]
    }

    uprated = uprate_household_property_by_region(household, resource)

    # The FRS-base owner mean is 200, so the factor is 2 for every owner,
    # the SPI owner included; the SPI non-owner is untouched.
    assert uprated["main_residence_value"].tolist() == pytest.approx(
        [200.0, 600.0, 2_000.0, 0.0]
    )
    assert uprated["property_wealth"].tolist() == pytest.approx(
        [200.0, 600.0, 3_000.0, 50.0]
    )


def test_regional_property_uprating_keeps_the_total_the_sum_of_its_components():
    """The factor scales the total and every component of it the frame carries
    (microcosm#1063), so the total an owner holds stays the sum of its parts;
    a renter's other property is untouched, as before."""

    household = pd.DataFrame(
        {
            "household_id": [1, 2],
            "region": ["LONDON", "LONDON"],
            "main_residence_value": [100.0, 0.0],
            "owned_land": [10.0, 4.0],
            "other_residential_property_value": [30.0, 5.0],
            "non_residential_property_value": [10.0, 1.0],
            "property_wealth": [150.0, 10.0],
        }
    )
    resource = {"values": [{"region": "LONDON", "avg_house_price": 300.0}]}

    uprated = uprate_household_property_by_region(household, resource)

    components = [
        "main_residence_value",
        "owned_land",
        "other_residential_property_value",
        "non_residential_property_value",
    ]
    assert uprated.loc[0, components].tolist() == [300.0, 30.0, 90.0, 30.0]
    assert uprated.loc[0, "property_wealth"] == pytest.approx(450.0)
    assert uprated.loc[1, [*components, "property_wealth"]].tolist() == [
        0.0,
        4.0,
        5.0,
        1.0,
        10.0,
    ]


def _legacy_uprate(household: pd.DataFrame, resource: dict) -> pd.DataFrame:
    """The stage's factor loop as it stood before microcosm#931 factored
    ``regional_reference_factors`` out of it, kept here to hold the
    refactor to it bit for bit."""

    import numpy as np

    from microcosm.build.uk_runtime.regional_uprating import (
        UK_REGIONAL_PROPERTY_REWRITES,
    )

    hpi_prices = {
        str(entry["region"]): float(entry["avg_house_price"])
        for entry in resource["values"]
    }
    result = household.copy()
    base_rows = (
        result["household_support_channel"] == "frs"
        if "household_support_channel" in result.columns
        else pd.Series(True, index=result.index)
    )
    for region, hpi_price in hpi_prices.items():
        region_mask = result["region"].astype(str) == region
        owners_mask = region_mask & (
            pd.to_numeric(result["main_residence_value"], errors="coerce") > 0
        )
        base_owners = owners_mask & base_rows
        if not base_owners.any():
            continue
        owner_values = np.sort(
            result.loc[base_owners, "main_residence_value"].to_numpy(dtype=float)
        )
        imputed_mean = float(owner_values.sum() / len(owner_values))
        if imputed_mean <= 0:
            continue
        factor = hpi_price / imputed_mean
        for column in UK_REGIONAL_PROPERTY_REWRITES:
            if column in result.columns:
                result.loc[owners_mask, column] *= factor
    return result


def _random_household(n: int = 300, seed: int = 931) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(seed)
    regions = np.asarray(["LONDON", "WALES", "SCOTLAND", "NORTHERN_IRELAND"])
    value = np.where(rng.uniform(size=n) < 0.3, 0.0, rng.uniform(1e4, 2e6, n))
    land = rng.uniform(0, 1e5, n)
    other = rng.uniform(0, 3e5, n)
    building = rng.uniform(0, 1e5, n)
    return pd.DataFrame(
        {
            "household_id": np.arange(1, n + 1),
            "region": rng.choice(regions, n),
            "household_support_channel": rng.choice(["frs", "spi"], n, p=[0.8, 0.2]),
            "main_residence_value": value,
            "owned_land": land,
            "other_residential_property_value": other,
            "non_residential_property_value": building,
            "property_wealth": value + land + other + building,
        }
    )


_RESOURCE = {
    "values": [
        {"region": "LONDON", "avg_house_price": 523_456.7, "dwellings": 1},
        {"region": "WALES", "avg_house_price": 211_111.1, "dwellings": 1},
        # Listed but without an FRS-base owner below: no factor.
        {"region": "NORTH_EAST", "avg_house_price": 160_000.0, "dwellings": 1},
        {"region": "SCOTLAND", "avg_house_price": 190_000.3, "dwellings": 1},
    ]
}


def test_the_default_denominator_is_the_stage_as_it_was_bit_for_bit() -> None:
    from microcosm.build.uk_runtime.regional_uprating import (
        regional_reference_factors,
    )

    household = _random_household()

    uprated = uprate_household_property_by_region(household, _RESOURCE)
    legacy = _legacy_uprate(household, _RESOURCE)

    pd.testing.assert_frame_equal(uprated, legacy, check_exact=True)
    factors = regional_reference_factors(household, _RESOURCE)
    assert list(factors) == ["LONDON", "WALES", "SCOTLAND"]
    london = (household["region"] == "LONDON") & (household["main_residence_value"] > 0)
    assert (
        uprated.loc[london, "main_residence_value"]
        == household.loc[london, "main_residence_value"] * factors["LONDON"]
    ).all()


def test_a_declared_denominator_gives_the_factor_its_mean_and_nothing_else() -> None:
    """The factor's mean is taken over the declared column; the owners are
    still the rows with a positive main residence, and every rewrite is
    scaled by the factor (microcosm#931)."""

    from microcosm.build.uk_runtime.property_localisation import RANK_BASIS_COLUMN
    from microcosm.build.uk_runtime.regional_uprating import (
        regional_reference_factors,
    )

    household = _random_household()
    # The kept draw, against a main residence moved on every owner.
    basis = household["main_residence_value"].to_numpy().copy()
    moved = household.assign(
        main_residence_value=household["main_residence_value"] * 1.7,
        **{RANK_BASIS_COLUMN: basis},
    )

    on_basis = regional_reference_factors(
        moved, _RESOURCE, denominator_column=RANK_BASIS_COLUMN
    )
    assert on_basis == regional_reference_factors(household, _RESOURCE)
    uprated = uprate_household_property_by_region(
        moved, _RESOURCE, denominator_column=RANK_BASIS_COLUMN
    )
    owners = (moved["region"] == "WALES") & (moved["main_residence_value"] > 0)
    assert (
        uprated.loc[owners, "main_residence_value"]
        == moved.loc[owners, "main_residence_value"] * on_basis["WALES"]
    ).all()
    assert (
        uprated.loc[owners, "owned_land"]
        == moved.loc[owners, "owned_land"] * on_basis["WALES"]
    ).all()
    with pytest.raises(KeyError, match=RANK_BASIS_COLUMN):
        uprate_household_property_by_region(
            household, _RESOURCE, denominator_column=RANK_BASIS_COLUMN
        )


def test_the_stage_reads_its_declared_denominator() -> None:
    from microcosm.build.source_manifest import SourceStageSpec
    from microcosm.build.uk_runtime.property_localisation import RANK_BASIS_COLUMN
    from microcosm.build.uk_runtime.regional_uprating import (
        declared_regional_reference_denominator,
    )

    def stage(*operations: dict) -> SourceStageSpec:
        return SourceStageSpec.from_mapping(
            {
                "stage": "regional_property_uprating",
                "survey": "test",
                "source": "test",
                "grain": "household",
                "artifacts": [],
                "operations": list(operations),
                "outputs": [],
                "rewrites": ["main_residence_value", "property_wealth"],
            }
        )

    kind = "uprate_to_regional_reference"
    assert declared_regional_reference_denominator(stage({"kind": kind})) == (
        "main_residence_value"
    )
    assert declared_regional_reference_denominator(stage()) == "main_residence_value"
    assert (
        declared_regional_reference_denominator(
            stage({"kind": kind, "denominator": RANK_BASIS_COLUMN})
        )
        == RANK_BASIS_COLUMN
    )
    with pytest.raises(ValueError, match="divides by one of"):
        declared_regional_reference_denominator(
            stage({"kind": kind, "denominator": "property_wealth"})
        )
    with pytest.raises(ValueError, match="more than once"):
        declared_regional_reference_denominator(stage({"kind": kind}, {"kind": kind}))
    # The packaged stage divides by the draw the WAS chain's area home values
    # keep, which the WAS stage declares among its outputs (microcosm#931).
    from microcosm.build.country_spec import load_country_spec

    stages = load_country_spec("uk").sources.stage_map()
    packaged = stages["regional_property_uprating"]
    assert declared_regional_reference_denominator(packaged) == RANK_BASIS_COLUMN
    assert RANK_BASIS_COLUMN in stages["was_wealth"].outputs


def test_the_stage_refuses_the_kept_draw_unless_it_divides_by_it() -> None:
    """Form F's two halves come together (C-15): households carrying the
    draw the WAS chain's area home values keep are uprated only by a stage
    that declares that draw its denominator; the stage that divides by the
    main residence refuses them, by a text that carries no count or value
    (microcosm#931)."""

    import numpy as np

    from microcosm.build.source_manifest import SourceStageSpec
    from microcosm.build.uk_runtime.national_frame import uk_national_frame
    from microcosm.build.uk_runtime.property_localisation import RANK_BASIS_COLUMN
    from microcosm.build.uk_runtime.regional_uprating import (
        UKRegionalPropertyUpratingStageTransform,
    )

    def stage(*operations: dict) -> SourceStageSpec:
        return SourceStageSpec.from_mapping(
            {
                "stage": "regional_property_uprating",
                "survey": "test",
                "source": "test",
                "grain": "household",
                "artifacts": [],
                "operations": list(operations),
                "outputs": [],
                "rewrites": ["main_residence_value", "property_wealth"],
            }
        )

    frame = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2],
                "person_benunit_id": [1, 2],
                "person_household_id": [1, 2],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1, 2]}),
        household=pd.DataFrame(
            {
                "household_id": [1, 2],
                "region": ["LONDON", "LONDON"],
                "main_residence_value": [300.0, 500.0],
                "property_wealth": [350.0, 500.0],
                RANK_BASIS_COLUMN: [100.0, 300.0],
            }
        ),
        time_period="2024",
        household_weights=np.array([1.0, 1.0]),
    )
    resource = {"values": [{"region": "LONDON", "avg_house_price": 400.0}]}
    kind = "uprate_to_regional_reference"

    for declared in (stage(), stage({"kind": kind})):
        with pytest.raises(ValueError, match="denominator") as caught:
            UKRegionalPropertyUpratingStageTransform(stage=declared, resource=resource)(
                frame
            )
        message = str(caught.value).replace("microcosm#931", "")
        assert not any(character.isdigit() for character in message)
    uprated = UKRegionalPropertyUpratingStageTransform(
        stage=stage({"kind": kind, "denominator": RANK_BASIS_COLUMN}),
        resource=resource,
    )(frame).table("household")
    # The factor is 400 over the kept draw's mean of 200.
    assert uprated["main_residence_value"].tolist() == pytest.approx([600.0, 1000.0])
