"""The spine's BRMA, private-rent and council-tax stages on all three systems,
through the installed engine.

microcosm#931 (C-37). H2's households cycle through four Great Britain
regions, so the parity fixture cannot show the BRMA stages on Northern
Ireland's households. This stand-in carries households of England and Wales,
Scotland and Northern Ireland: test support's invented ``TOY_HOUSEHOLDS``,
cloned and keyed by the spine's own expansion and identity steps
(``toy_spine_frame``) and placed on the toy conditioned supports by
atomic_area_assign, local_authority and rfdr, each stage run as the packaged
spec declares it. area_brma then runs on the installed policyengine-uk
engine, and spi_copy_areas on invented SPI copies
(``with_invented_support_channel``), whose Northern Ireland copies take the
stage's one engine call over their households alone: the copy stage's first
run on the real engine. private_rent then reads the rents of the frame the
two left, with the housing spi_housing_shell would give it invented: H2's
root households all own their homes, so this stand-in is where the spine's
rent re-draw meets the BRMAs its own stages drew. council_tax_localise then
prices the council tax of the frame private_rent left, invented as stage 5
and the housing shell would leave it (``with_toy_council_tax``), at the
committed 2024-25 Band D: H2 has no Northern Ireland household, so this
stand-in is where the localisation meets Northern Ireland's households and
their SPI copies (microcosm#931).

At one geographic clone and at two:

- every ``local_authority`` the stages write, the households' and their
  copies', is a member of the installed engine's LocalAuthority enum;
- area_brma's Northern Ireland BRMAs are those frs_brma's draw gives the same
  households on the same engine's LHA_category, and Great Britain's are the
  area-to-BRMA layer's for the drawn areas; its one engine call runs one
  simulation per clone;
- spi_copy_areas makes one engine call, of one simulation, over the Northern
  Ireland copies' households alone, and gives those copies the BRMAs
  frs_brma's draw gives them on that call's frame, keyed on their own benefit
  units; Great Britain's copies read the layer for their new areas;
- private_rent re-draws every Great Britain private renter's rent, the
  households' and their copies', off the list of the BRMA those stages gave
  it, so every such BRMA has a list for the renter's bedrooms, and leaves
  Northern Ireland's rents and every other column as they were;
- council_tax_localise leaves Northern Ireland's council tax, rebate and
  reduction bit for bit, an SPI copy of a Northern Ireland household among
  the rows; gives every Great Britain row on a ratio path its bill times the
  ratio of its draw cell, the SPI copies at L3, and moves some; holds every
  Great Britain reduction and rebate within the new bill; keeps the paid bill
  from before it as ``council_tax_frs_bill``; and changes no other column.

Invented data only; assertion messages give stages, columns and K, never a
value, a key or an id.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.brma_area_layer import NI_SYSTEM, area_brmas
from microcosm.build.uk_runtime.frs_brma import UKFRSBRMAStageTransform
from microcosm.build.uk_runtime.local_authority_input import (
    local_authority_keys_missing_from_engine,
)
from microcosm.build.uk_runtime.spi_support import (
    SPI_SYNTHETIC_SUPPORT_CHANNEL,
    support_channel_column,
)
from microcosm.build.uk_runtime.spine_council_tax import (
    COUNCIL_TAX_FRS_BILL_COLUMN,
    COUNCIL_TAX_LOCALISE_STAGE_NAME,
    UKCouncilTaxLocaliseStageTransform,
    uk_council_tax_references,
)
from microcosm.build.uk_runtime.spine_geography import (
    AREA_BRMA_STAGE_NAME,
    ATOMIC_AREA_ASSIGN_STAGE_NAME,
    LOCAL_AUTHORITY_STAGE_NAME,
    RFDR_STAGE_NAME,
    SPI_COPY_AREAS_STAGE_NAME,
    UKAreaBRMAStageTransform,
    UKAtomicAreaAssignStageTransform,
    UKLocalAuthorityStageTransform,
    UKRFDRStageTransform,
    UKSPICopyAreasStageTransform,
    load_committed_brma_layer,
)
from microcosm.build.uk_runtime.spine_private_rent import (
    PRIVATE_RENT_STAGE_NAME,
    UKPrivateRentStageTransform,
)
from microcosm.frame import Frame
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine
from test_support.microcosm_build.uk_conditioned_support_fixtures import (
    toy_conditioned_support_payloads,
    toy_spine_frame,
    with_invented_support_channel,
)
from test_support.microcosm_build.uk_council_tax_fixtures import with_toy_council_tax

_SYSTEM = "atomic_area_system"
_CHANNEL = support_channel_column("household")


@dataclass
class _RecordingEngine:
    """The installed UK engine, keeping each call's frame and simulations."""

    clones: int
    engine: PolicyEngineUKEngine = field(init=False)
    calls: list[tuple[Frame, int]] = field(default_factory=list)
    country: str = "uk"

    def __post_init__(self) -> None:
        self.engine = PolicyEngineUKEngine(clones=self.clones)

    def materialize(self, frame, variables, period):
        self.calls.append((frame, self.engine.simulation_batches(frame)))
        return self.engine.materialize(frame, variables, period)


def _stage(name: str):
    return load_country_spec("uk").sources.stage_map()[name]


def _brmas(frame: Frame) -> pd.Series:
    household = frame.table("household")
    return pd.Series(
        household["brma"].to_numpy(dtype=object),
        index=household["household_id"].to_numpy(),
    )


def _layer_brmas(frame: Frame, rows: np.ndarray) -> np.ndarray:
    """The area-to-BRMA layer's BRMA for the drawn areas at ``rows``."""

    household = frame.table("household").loc[rows]
    layer, _ = load_committed_brma_layer()
    brmas, _ = area_brmas(
        layer=layer,
        area_system=household[_SYSTEM].to_numpy(dtype=object),
        lsoa_code=household["lsoa_code"].to_numpy(dtype=object),
        oa_code=household["oa_code"].to_numpy(dtype=object),
        region=household["region"].to_numpy(dtype=object),
        brma=np.full(len(household), None, dtype=object),
    )
    return np.asarray(brmas, dtype=object)


def _with_engine_inputs(frame: Frame) -> Frame:
    """``frame`` with the few inputs the engine's dataset reads that the
    stand-in lacks, all invented: every person's age and
    ``months_since_last_birthday`` (an input the spine stores from its root
    on, which an engine call in batches refuses to compute over part of the
    pool), and each household's council tax, tenure and rent, which
    policyengine-uk's dataset uprating reads."""

    tables = {entity: frame.table(entity) for entity in frame.entities}
    person = tables["person"].copy()
    person["age"] = 20 + np.arange(len(person), dtype=np.int64) % 60
    person["months_since_last_birthday"] = 0.0
    tables["person"] = person
    household = tables["household"].copy()
    household["council_tax"] = 1_200.0
    household["tenure_type"] = "RENT_PRIVATELY"
    household["rent"] = 6_000.0
    tables["household"] = household
    return Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


#: An invented annual private rent, the level of the engine-free private-rent
#: tests, at which every region's scale on the lists is inside the stage's
#: band.
_RENT = 14_000.0


def _housed(frame: Frame) -> Frame:
    """``frame`` with the housing spi_housing_shell would leave, invented:
    every household a private renter at ``_RENT`` a year, with one to four
    bedrooms in turn."""

    tables = {entity: frame.table(entity) for entity in frame.entities}
    household = tables["household"].copy()
    household["tenure_type"] = "RENT_PRIVATELY"
    household["rent"] = _RENT
    household["num_bedrooms"] = 1 + np.arange(len(household), dtype=np.int64) % 4
    tables["household"] = household
    return Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


@pytest.fixture(scope="module", params=[1, 2], ids=lambda clones: f"K={clones}")
def run(request) -> dict[str, object]:
    """The stand-in through the area stages, area_brma, spi_copy_areas,
    private_rent on invented housing and council_tax_localise on invented
    council tax, each engine call recorded."""

    clones = request.param
    payloads = toy_conditioned_support_payloads()
    placed = _with_engine_inputs(toy_spine_frame(clones=clones))
    for transform in (
        UKAtomicAreaAssignStageTransform(
            stage=_stage(ATOMIC_AREA_ASSIGN_STAGE_NAME), supports=payloads
        ),
        UKLocalAuthorityStageTransform(stage=_stage(LOCAL_AUTHORITY_STAGE_NAME)),
        UKRFDRStageTransform(stage=_stage(RFDR_STAGE_NAME), supports=payloads),
    ):
        placed = transform(placed)
    brma_engine = _RecordingEngine(clones)
    brma = UKAreaBRMAStageTransform(
        stage=_stage(AREA_BRMA_STAGE_NAME), engine=brma_engine
    )(placed)
    copied = with_invented_support_channel(brma)
    copy_engine = _RecordingEngine(clones)
    copy_areas = UKSPICopyAreasStageTransform(
        stage=_stage(SPI_COPY_AREAS_STAGE_NAME), engine=copy_engine, supports=payloads
    )(copied)
    housed = _housed(copy_areas)
    rents = UKPrivateRentStageTransform(stage=_stage(PRIVATE_RENT_STAGE_NAME))
    rented = rents(housed)
    priced = with_toy_council_tax(rented)
    localiser = UKCouncilTaxLocaliseStageTransform(
        stage=_stage(COUNCIL_TAX_LOCALISE_STAGE_NAME), supports=payloads
    )
    return {
        "clones": clones,
        "placed": placed,
        "brma": brma,
        "brma_engine": brma_engine,
        "copied": copied,
        "copy_areas": copy_areas,
        "copy_engine": copy_engine,
        "housed": housed,
        "private_rent": rented,
        "private_rent_evidence": rents.checkpoint_metadata()["evidence"],
        "priced": priced,
        "localiser": localiser,
        "council_tax_localise": localiser(priced),
        "council_tax_evidence": localiser.checkpoint_metadata()["evidence"],
    }


def test_the_stand_in_holds_all_three_systems(run) -> None:
    systems = set(run["placed"].table("household")[_SYSTEM])
    assert systems == {
        "uk_ew_output_area_2021",
        "uk_scotland_output_area_2022",
        NI_SYSTEM,
    }, f"K={run['clones']}"


def test_every_local_authority_is_a_member_of_the_installed_engines_enum(
    run,
) -> None:
    for name in ("placed", "copy_areas"):
        keys = run[name].table("household")["local_authority"].tolist()
        missing = local_authority_keys_missing_from_engine(keys)
        assert missing == (), (
            f"After {name} at K={run['clones']}, {len(missing)} local_authority "
            "member(s) are not in the installed engine's enum."
        )


def test_area_brma_draws_northern_ireland_as_frs_brma_and_reads_the_layer(
    run,
) -> None:
    clones = run["clones"]
    ((_, simulations),) = run["brma_engine"].calls
    assert simulations == clones, f"area_brma's call at K={clones}"
    placed, brma = run["placed"], run["brma"]
    household = brma.table("household")
    northern_ireland = household[_SYSTEM].eq(NI_SYSTEM).to_numpy()
    assert northern_ireland.any() and not northern_ireland.all()
    # frs_brma's draw on the frame area_brma read, on the same engine's
    # LHA_category: the same keys, seed and salts for every household.
    reference = _brmas(
        UKFRSBRMAStageTransform(
            stage=_stage(AREA_BRMA_STAGE_NAME),
            engine=PolicyEngineUKEngine(clones=clones),
        )(placed)
    )
    drawn = _brmas(brma)
    ids = household["household_id"].to_numpy()
    assert drawn.loc[ids[northern_ireland]].equals(
        reference.loc[ids[northern_ireland]]
    ), f"Northern Ireland's BRMAs at K={clones}"
    np.testing.assert_array_equal(
        drawn.loc[ids[~northern_ireland]].to_numpy(dtype=object),
        _layer_brmas(brma, ~northern_ireland),
        err_msg=f"Great Britain's BRMAs at K={clones}",
    )


def test_the_copy_stage_draws_northern_ireland_copies_on_one_engine_call(
    run,
) -> None:
    clones = run["clones"]
    copied, copy_areas = run["copied"], run["copy_areas"]
    household = copied.table("household")
    copies = household[_CHANNEL].eq(SPI_SYNTHETIC_SUPPORT_CHANNEL).to_numpy()
    northern_ireland = household[_SYSTEM].eq(NI_SYSTEM).to_numpy()
    assert (copies & northern_ireland).any() and (copies & ~northern_ireland).any()
    # One call, one simulation, over the Northern Ireland copies' households
    # alone, every clone's together.
    ((scratch, simulations),) = run["copy_engine"].calls
    assert simulations == 1, f"spi_copy_areas' call at K={clones}"
    ids = household["household_id"].to_numpy()
    assert set(scratch.table("household")["household_id"]) == set(
        ids[copies & northern_ireland]
    ), f"spi_copy_areas' call frame at K={clones}"
    # Each copy's BRMA is frs_brma's draw on that call's frame, on the
    # engine's LHA_category for the copies' own benefit units.
    reference = _brmas(
        UKFRSBRMAStageTransform(
            stage=_stage(AREA_BRMA_STAGE_NAME),
            engine=PolicyEngineUKEngine(clones=clones),
        )(scratch)
    )
    drawn = _brmas(copy_areas)
    copied_ids = ids[copies & northern_ireland]
    assert drawn.loc[copied_ids].equals(reference.loc[copied_ids]), (
        f"Northern Ireland copies' BRMAs at K={clones}"
    )
    # Great Britain's copies read the layer for the areas the stage drew them.
    rewritten = copy_areas.table("household")
    rows = (
        rewritten[_CHANNEL].eq(SPI_SYNTHETIC_SUPPORT_CHANNEL)
        & rewritten[_SYSTEM].ne(NI_SYSTEM)
    ).to_numpy()
    np.testing.assert_array_equal(
        rewritten.loc[rows, "brma"].to_numpy(dtype=object),
        _layer_brmas(copy_areas, rows),
        err_msg=f"Great Britain copies' BRMAs at K={clones}",
    )
    # Every other row keeps the BRMA area_brma gave it.
    base = ids[~copies]
    assert drawn.loc[base].equals(_brmas(run["brma"]).loc[base]), f"K={clones}"


def test_private_rent_re_draws_great_britain_renters_on_the_stages_brmas(
    run,
) -> None:
    """Every Great Britain private renter, a household or its SPI copy, takes
    a rent of the list of the BRMA area_brma and spi_copy_areas gave it, on a
    full-scale spine; Northern Ireland's rents, and every other column, stay
    bit for bit."""

    clones = run["clones"]
    before = run["housed"].table("household")
    after = run["private_rent"].table("household")
    great_britain = before[_SYSTEM].ne(NI_SYSTEM).to_numpy()
    copies = before[_CHANNEL].eq(SPI_SYNTHETIC_SUPPORT_CHANNEL).to_numpy()
    assert (great_britain & copies).any() and (great_britain & ~copies).any()
    assert (~great_britain).any()
    # The BRMAs private_rent read are those area_brma and spi_copy_areas gave.
    assert before["brma"].equals(run["copy_areas"].table("household")["brma"])
    old = before["rent"].to_numpy(dtype=np.float64)
    new = after["rent"].to_numpy(dtype=np.float64)
    assert (new[great_britain] != old[great_britain]).all(), (
        f"Great Britain's private rents at K={clones}"
    )
    np.testing.assert_array_equal(
        new[~great_britain].view(np.int64),
        old[~great_britain].view(np.int64),
        err_msg=f"Northern Ireland's private rents at K={clones}",
    )
    pd.testing.assert_frame_equal(
        after.drop(columns="rent"), before.drop(columns="rent"), check_exact=True
    )
    evidence = run["private_rent_evidence"]
    assert evidence["clone_count"] == clones
    assert evidence["strict"] is True


_NORTHERN_IRELAND = "NORTHERN_IRELAND"
_REDUCTION = "council_tax_benefit_reported"
_REWRITTEN = ("council_tax", "council_tax_rebate")


def _bits(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).view(np.int64)


def test_council_tax_localise_prices_great_britain_and_leaves_northern_ireland(
    run,
) -> None:
    """council_tax_localise, run as the packaged spec declares it on the frame
    the area stages, the copy stage and the private rents left, with invented
    council tax, at the committed Band D (microcosm#931). Northern Ireland's
    households and their SPI copies keep their council tax, rebate and
    reduction bit for bit, and one reduction above its zero bill stays there.
    Every Great Britain row on a ratio path takes its bill times the ratio of
    its own draw cell, the SPI copies at L3, and some ratios differ from one;
    the statute rows are priced, every Great Britain reduction and rebate lies
    within the new bill, the kept bill is the paid bill from before the stage,
    and no other column moves."""

    clones = run["clones"]
    before, after = run["priced"], run["council_tax_localise"]
    household_before = before.table("household")
    household_after = after.table("household")
    person_before = before.table("person")
    person_after = after.table("person")
    northern_ireland = household_before["region"].eq(_NORTHERN_IRELAND).to_numpy()
    copies = household_before[_CHANNEL].eq(SPI_SYNTHETIC_SUPPORT_CHANNEL).to_numpy()
    assert (northern_ireland & copies).any() and (~northern_ireland & copies).any()
    assert (northern_ireland & ~copies).any() and (~northern_ireland & ~copies).any()

    # Northern Ireland, the copies of its households among its rows: bit for
    # bit.
    for column in _REWRITTEN:
        np.testing.assert_array_equal(
            _bits(household_after.loc[northern_ireland, column]),
            _bits(household_before.loc[northern_ireland, column]),
            err_msg=f"Northern Ireland's {column} at K={clones}",
        )
    regions = pd.Series(
        household_before["region"].to_numpy(dtype=object),
        index=household_before["household_id"].to_numpy(),
    )
    person_northern_ireland = (
        person_before["person_household_id"]
        .map(regions)
        .eq(_NORTHERN_IRELAND)
        .to_numpy()
    )
    np.testing.assert_array_equal(
        _bits(person_after.loc[person_northern_ireland, _REDUCTION]),
        _bits(person_before.loc[person_northern_ireland, _REDUCTION]),
        err_msg=f"Northern Ireland's {_REDUCTION} at K={clones}",
    )
    # No other column moves, and the kept bill is the only new one.
    pd.testing.assert_frame_equal(
        household_after.drop(columns=[*_REWRITTEN, COUNCIL_TAX_FRS_BILL_COLUMN]),
        household_before.drop(columns=list(_REWRITTEN)),
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        person_after.drop(columns=_REDUCTION),
        person_before.drop(columns=_REDUCTION),
        check_exact=True,
    )
    # The kept bill is WAS's paid bill from the columns before the rewrite.
    bill_before = household_before["council_tax"].to_numpy(dtype=np.float64)
    paid = np.maximum(
        bill_before - household_before["council_tax_rebate"].to_numpy(dtype=np.float64),
        0.0,
    )
    np.testing.assert_array_equal(
        _bits(household_after[COUNCIL_TAX_FRS_BILL_COLUMN]),
        _bits(paid),
        err_msg=f"The kept bill at K={clones}",
    )

    # Great Britain: each ratio row takes the ratio of its own draw cell.
    localiser = run["localiser"]
    references = uk_council_tax_references(
        household_before,
        definition=localiser.definition,
        supports=localiser.decoded_supports,
        band_d=localiser.band_d,
    )
    great_britain = ~northern_ireland
    assert np.array_equal(
        np.asarray(references.great_britain, dtype=bool), great_britain
    ), f"K={clones}"
    levels = np.asarray(references.levels, dtype=object)
    assert (levels[copies & great_britain] == "L3").all(), f"K={clones}"
    provenance = household_before["council_tax_bill_provenance"].to_numpy(dtype=object)
    observed = household_before["council_tax_band_observed"].to_numpy(dtype=bool)
    statute = (
        great_britain
        & ~copies
        & np.isin(provenance, ["missing", "unknown_reduction"])
        & observed
    )
    ratio = great_britain & ~statute
    assert statute.any() and ratio.any(), f"K={clones}"
    ratios = np.asarray(references.ratios, dtype=np.float64)
    bill_after = household_after["council_tax"].to_numpy(dtype=np.float64)
    np.testing.assert_array_equal(
        _bits(bill_after[ratio]),
        _bits(ratios[ratio] * bill_before[ratio]),
        err_msg=f"Great Britain's ratio rows at K={clones}",
    )
    assert (ratios[ratio & ~copies] != 1.0).any(), f"K={clones}"
    assert (ratios[ratio & copies] != 1.0).any(), f"K={clones}"
    assert np.isfinite(bill_after).all() and (bill_after >= 0.0).all()

    # Every Great Britain reduction and rebate lies within the new bill.
    rebate_after = household_after["council_tax_rebate"].to_numpy(dtype=np.float64)
    assert (rebate_after[great_britain] <= bill_after[great_britain]).all(), (
        f"Great Britain's rebates at K={clones}"
    )
    bills = pd.Series(bill_after, index=household_after["household_id"].to_numpy())
    person_bill = (
        person_after["person_household_id"].map(bills).to_numpy(dtype=np.float64)
    )
    reduction_after = person_after[_REDUCTION].to_numpy(dtype=np.float64)
    assert (
        reduction_after[~person_northern_ireland]
        <= person_bill[~person_northern_ireland]
    ).all(), f"Great Britain's reductions at K={clones}"

    evidence = run["council_tax_evidence"]
    assert evidence["stage"] == COUNCIL_TAX_LOCALISE_STAGE_NAME
    assert evidence["clone_count"] == clones
    checks = evidence["checks"]
    for name in (
        "reduction_within_bill",
        "rebate_within_bill",
        "rebate_equals_reduction",
        "northern_ireland_unchanged",
    ):
        assert checks[name] is True, (name, clones)
    # An invented Northern Ireland reduction sits above its zero bill and is
    # not capped.
    assert checks["northern_ireland_reductions_within_bill"] is False
