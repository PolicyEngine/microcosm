"""The spine's BRMA and private-rent stages on all three systems, through the
installed engine.

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
rent re-draw meets the BRMAs its own stages drew.

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
  Northern Ireland's rents and every other column as they were.

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
    """The stand-in through the area stages, area_brma, spi_copy_areas and,
    on invented housing, private_rent, each engine call recorded."""

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
    return {
        "clones": clones,
        "placed": placed,
        "brma": brma,
        "brma_engine": brma_engine,
        "copied": copied,
        "copy_areas": copy_areas,
        "copy_engine": copy_engine,
        "housed": housed,
        "private_rent": rents(housed),
        "private_rent_evidence": rents.checkpoint_metadata()["evidence"],
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
