"""Local taxes reach households placed through the block-ladder line.

Synthetic households are pinned to fixture blocks in Yonkers, Wilmington,
Denver and Philadelphia (and a neighbour of each), placed by
``with_household_us_geography_ladder``, and run through
``PolicyEngineUSEngine.materialize``. Each local tax or benefit is nonzero
for the resident and zero (or at the nonresident rate) for the neighbour,
and stripping the derived flags returns the resident to the neighbour's
result, so the flags, not other geography, drive the difference.

The release builder's export keeps the inputs its targets were computed from:
after a legacy pool becomes block-only through L0 selection, a retained
Wilmington itemizer's ``income_tax`` on the written support is the value the
refit used, though writing ``in_wilmington`` would lower it. The standalone
L0/refit export likewise withholds ``in_wilmington`` when the rows its saved
weights keep decide it and the base it loads does not.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from microcosm.build.us_local_jurisdictions import (
    rederive_us_local_jurisdiction_flags,
)
from microcosm.build.us_runtime import l0_refit_export
from microcosm.build.us_runtime.geography_ladder import (
    with_household_us_geography_ladder,
)
from microcosm.build.us_runtime.l0_refit_export import (
    attach_l0_refit_entity_weights,
)
from microcosm.frame import Frame
from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine
from test_support.microcosm_build.us_local_jurisdictions import (
    block_ladder,
    households_on_blocks,
    us_frame,
)
from test_support.paths import paths_for

PERIOD = 2026
FLAGS = {
    "household": ("in_denver", "in_wilmington", "in_yonkers"),
    "person": ("pa_philadelphia_wage_tax_resident",),
}
# Every person carries every input column, so no table holds a null the
# engine would refuse.
NO_INPUTS = {
    "age": 40,
    "employment_income": 0.0,
    "rent": 0.0,
    "social_security_retirement": 0.0,
    "pa_philadelphia_wage_tax_taxable_wages": 0.0,
}
WORKER = {**NO_INPUTS, "employment_income": 100_000.0}
PHILADELPHIA_WORKER = {
    **NO_INPUTS,
    "employment_income": 50_000.0,
    "pa_philadelphia_wage_tax_taxable_wages": 50_000.0,
}
# Elderly low-income renter: Denver's renter relief needs rent, an elderly or
# disabled member, and income under the AMI-based limit.
DENVER_RENTER = {
    **NO_INPUTS,
    "age": 72,
    "rent": 14_400.0,
    "social_security_retirement": 9_000.0,
}
CASES = (
    ("yonkers", WORKER),
    ("new_rochelle_westchester", WORKER),
    ("wilmington", WORKER),
    ("new_castle_unincorporated", WORKER),
    ("denver", DENVER_RENTER),
    ("aurora_arapahoe", DENVER_RENTER),
    ("philadelphia", PHILADELPHIA_WORKER),
    ("delaware_county_pa", PHILADELPHIA_WORKER),
)
ROW = {name: position for position, (name, _) in enumerate(CASES)}
OUTPUTS = (
    "ny_yonkers_income_tax",
    "de_wilmington_earned_income_tax",
    "co_denver_property_tax_relief_renter_eligible",
    "co_denver_property_tax_relief",
    "pa_philadelphia_wage_tax",
    "in_denver",
    "in_wilmington",
    "in_yonkers",
    "pa_philadelphia_wage_tax_resident",
)


@pytest.fixture(scope="module")
def placed() -> Frame:
    frame = us_frame(
        households_on_blocks([name for name, _ in CASES]),
        person_inputs=[[dict(person)] for _, person in CASES],
    )
    return with_household_us_geography_ladder(frame, block_ladder(), seed=0)


def _without_flags(frame: Frame) -> Frame:
    tables = {entity: frame.table(entity) for entity in frame.entities}
    for entity, flags in FLAGS.items():
        tables[entity] = tables[entity].drop(columns=list(flags))
    return Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
    )


@pytest.fixture(scope="module")
def engine() -> PolicyEngineUSEngine:
    return PolicyEngineUSEngine()


@pytest.fixture(scope="module")
def with_flags(engine, placed) -> dict[str, np.ndarray]:
    return dict(engine.materialize(placed, OUTPUTS, PERIOD))


@pytest.fixture(scope="module")
def without_flags(engine, placed) -> dict[str, np.ndarray]:
    return dict(engine.materialize(_without_flags(placed), OUTPUTS, PERIOD))


def test_the_engine_loads_the_derived_flags(with_flags) -> None:
    for flag, resident in (
        ("in_yonkers", "yonkers"),
        ("in_wilmington", "wilmington"),
        ("in_denver", "denver"),
        ("pa_philadelphia_wage_tax_resident", "philadelphia"),
    ):
        assert np.flatnonzero(with_flags[flag]).tolist() == [ROW[resident]], flag


def test_yonkers_resident_surcharge_is_nonzero_only_in_yonkers(
    with_flags, without_flags
) -> None:
    tax = with_flags["ny_yonkers_income_tax"]

    assert tax[ROW["yonkers"]] > 0
    assert tax[ROW["new_rochelle_westchester"]] == 0
    assert without_flags["ny_yonkers_income_tax"][ROW["yonkers"]] == 0


def test_wilmington_earned_income_tax_is_nonzero_only_in_wilmington(
    with_flags, without_flags
) -> None:
    tax = with_flags["de_wilmington_earned_income_tax"]

    assert tax[ROW["wilmington"]] > 0
    assert tax[ROW["new_castle_unincorporated"]] == 0
    assert without_flags["de_wilmington_earned_income_tax"][ROW["wilmington"]] == 0


def test_denver_renter_relief_reaches_only_the_denver_renter(
    with_flags, without_flags
) -> None:
    eligible = with_flags["co_denver_property_tax_relief_renter_eligible"]
    relief = with_flags["co_denver_property_tax_relief"]

    assert eligible[ROW["denver"]] > 0
    assert relief[ROW["denver"]] > 0
    assert eligible[ROW["aurora_arapahoe"]] == 0
    assert relief[ROW["aurora_arapahoe"]] == 0
    assert without_flags["co_denver_property_tax_relief"][ROW["denver"]] == 0


def test_philadelphia_wage_tax_uses_the_resident_rate_for_residents(
    with_flags, without_flags
) -> None:
    from policyengine_us import CountryTaxBenefitSystem

    rates = CountryTaxBenefitSystem().parameters(PERIOD).gov.local.pa.philadelphia
    resident_rate = rates.tax.income.resident_rate
    nonresident_rate = rates.tax.income.nonresident_rate
    wages = PHILADELPHIA_WORKER["pa_philadelphia_wage_tax_taxable_wages"]
    tax = with_flags["pa_philadelphia_wage_tax"]

    assert resident_rate != nonresident_rate
    assert tax[ROW["philadelphia"]] == pytest.approx(wages * resident_rate)
    assert tax[ROW["delaware_county_pa"]] == pytest.approx(wages * nonresident_rate)
    assert without_flags["pa_philadelphia_wage_tax"][
        ROW["philadelphia"]
    ] == pytest.approx(wages * nonresident_rate)


# A Wilmington worker who itemizes: the city's earned-income tax enters SALT,
# so in_wilmington moves this household's federal income tax. The ACS row has
# no block, as a legacy pool's ACS rows do.
WILMINGTON_ITEMIZER = {
    **NO_INPUTS,
    "employment_income": 100_000.0,
    "charitable_cash_donations": 20_000.0,
}
ACS_RESIDENT = {**NO_INPUTS, "charitable_cash_donations": 0.0}
PARITY_OUTPUTS = (
    "income_tax",
    "de_wilmington_earned_income_tax",
    "salt_deduction",
    "in_wilmington",
)


def _release_builder():
    path = (
        paths_for("microcosm-build").repository
        / "tools"
        / "build_us_fiscal_refresh_release.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_a_block_only_l0_export_keeps_the_income_tax_its_weights_were_fit_to(
    engine,
) -> None:
    # Review round 2 of the adopted microcosm#1453. The builder materializes
    # its targets from the pool, whose ACS row withholds the place rung, and
    # the refit fits the weights to those values. The L0 support keeps only
    # the Wilmington donor, whose block decides in_wilmington; the export
    # keeps the pool's flags, so the target the H5 carries is the one fitted.
    builder = _release_builder()
    pool = us_frame(
        [
            {
                "state_fips": 10,
                "county_fips": "10003",
                "place_fips": "77580",
                "block_geoid": "100030001001000",
            },
            {
                "state_fips": 8,
                "county_fips": "08031",
                "place_fips": None,
                "block_geoid": None,
            },
        ],
        person_inputs=[[WILMINGTON_ITEMIZER], [ACS_RESIDENT]],
    )
    receipts: dict[str, object] = {}
    calibrated = builder._with_local_jurisdiction_flags(
        pool, support="base", receipts=receipts, telemetry=None
    )
    kept = attach_l0_refit_entity_weights(
        calibrated,
        weight_entity="household",
        selected_entity_ids=np.asarray([1]),
        selected_weights=np.asarray([5.0]),
        reason="L0 support without the ACS row",
    )
    exported = builder._with_local_jurisdiction_flags(
        kept, support="export", receipts=receipts, telemetry=None
    )
    # What the export wrote before this fix: the kept row's own flags.
    rewritten, _receipt = rederive_us_local_jurisdiction_flags(kept)

    fitted = dict(engine.materialize(calibrated, PARITY_OUTPUTS, PERIOD))
    written = dict(engine.materialize(exported, PARITY_OUTPUTS, PERIOD))
    flagged = dict(engine.materialize(rewritten, PARITY_OUTPUTS, PERIOD))

    for output in ("income_tax", "salt_deduction", "in_wilmington"):
        np.testing.assert_array_equal(written[output], fitted[output][:1])
    assert not written["in_wilmington"].any()
    # The parity is not vacuous: the flag's own geography decides it, and
    # writing it moves the fitted target.
    assert flagged["in_wilmington"].tolist() == [True]
    assert flagged["de_wilmington_earned_income_tax"][0] == pytest.approx(1_250.0)
    assert flagged["salt_deduction"][0] > fitted["salt_deduction"][0]
    assert flagged["income_tax"][0] < fitted["income_tax"][0]


def test_a_block_only_saved_weight_export_keeps_the_income_tax_they_were_fit_to(
    engine, monkeypatch, tmp_path
) -> None:
    # Review round 3 of the adopted microcosm#1453: the standalone export
    # (export_us_l0_refit_h5). Two New Castle County itemizers with the same
    # wages and gifts, a Wilmington donor on a block and an ACS row with no
    # block, so the base decides no place-rule flag. Saved weights fit on
    # that base keep only the donor. The export writes the base's inputs, so
    # the H5 gives the income tax the weights were fit to.
    pool = us_frame(
        [
            {
                "state_fips": 10,
                "county_fips": "10003",
                "place_fips": "77580",
                "block_geoid": "100030001001000",
            },
            {
                "state_fips": 10,
                "county_fips": "10003",
                "place_fips": None,
                "block_geoid": None,
            },
        ],
        person_inputs=[[WILMINGTON_ITEMIZER], [WILMINGTON_ITEMIZER]],
    )
    weights_npz = tmp_path / "weights.npz"
    np.savez_compressed(weights_npz, weights=np.asarray([2.0, 0.0]))
    base_h5 = tmp_path / "base.h5"
    base_h5.write_text("base")
    captured: dict[str, Frame] = {}

    class CapturingWriter:
        def write_dataset(self, frame, path, period):
            captured["frame"] = frame
            Path(path).write_text("sentinel")

    monkeypatch.setattr(l0_refit_export, "load_us_frame", lambda path: pool)
    monkeypatch.setattr(
        l0_refit_export, "copy_microcosm_root_attrs", lambda source, destination: ()
    )
    monkeypatch.setattr(l0_refit_export, "PolicyEngineUSEngine", CapturingWriter)

    summary = l0_refit_export.export_us_l0_refit_h5(
        base_h5=base_h5,
        weights_npz=weights_npz,
        output_h5=tmp_path / "populace_us_2026.h5",
        require_source_columns=False,
        # Two households cannot satisfy national NYC-share bounds.
        require_geography_ladder=False,
    )
    fitted_on, _receipt = rederive_us_local_jurisdiction_flags(pool)
    # What the export wrote before this fix: the kept row's own flags.
    rewritten, _receipt = rederive_us_local_jurisdiction_flags(captured["frame"])

    fitted = dict(engine.materialize(fitted_on, PARITY_OUTPUTS, PERIOD))
    written = dict(engine.materialize(captured["frame"], PARITY_OUTPUTS, PERIOD))
    flagged = dict(engine.materialize(rewritten, PARITY_OUTPUTS, PERIOD))

    # The input-mass gate passes either way: a column only the export
    # carries is not a failure.
    assert summary["input_mass_parity"]["passed"] is True
    for output in ("income_tax", "salt_deduction", "in_wilmington"):
        np.testing.assert_array_equal(written[output], fitted[output][:1])
    assert not written["in_wilmington"].any()
    assert flagged["in_wilmington"].tolist() == [True]
    assert flagged["de_wilmington_earned_income_tax"][0] == pytest.approx(1_250.0)
    assert flagged["salt_deduction"][0] > fitted["salt_deduction"][0]
    assert flagged["income_tax"][0] < fitted["income_tax"][0]
