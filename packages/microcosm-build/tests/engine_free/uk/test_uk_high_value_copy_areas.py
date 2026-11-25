"""The high-value home support copies' own places and council tax (microcosm#931).

María's ruling of 2026-11-25: each copy the high-value home support stage
makes takes a place of its own, as another clone of its household would. The
stage splits the top value stratum, then gives each copy its own geography
key over its copy path, an area drawn from its draw inputs, which are its
parent's, its local authority, BRMA and fuel-relief flag, and, in Great
Britain, its council tax rescaled to its new authority's Band D. Its value
and every other column stay its parent's.

The frames are test_support's toy households on the toy conditioned
supports, at K = 1 and K = 3: keyed and cloned by the spine's own steps,
placed by the packaged G2, G3, G5 and G4, copied by the support channel's
stand-in and re-placed by the packaged SPI-copy area stage. Every household
is an owner with an invented value, council tax and reduction, each the same
for a household's clones and copies. The stage runs with a lower stratum and
three copies so that every region is split, a stand-in for the rules engine
and the invented Band D of the toy authorities.

They hold:

- each copy's key is its own, over its copy path with the high-value step,
  and a root keeps its key;
- each copy's area is an area of its parent's draw cell with mass in the
  weight it samples by, and its local authority, BRMA and flag are its
  area's;
- every root and every column but the rewritten ones keep their values;
- each Great Britain copy that changed authority takes its parent's bill,
  rebate and reduction times the Band D ratio, one rounding of the exact
  product, and every other copy keeps them bit for bit;
- Northern Ireland's copies take one engine call over their households;
- clone 0's copies keep their keys, places and council tax as K grows;
- the evidence counts distinct sources, rounded, and prints no key or code;
- a path with a later copy step is refused by name.
"""

from __future__ import annotations

import functools
import json
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_lineage import copy_paths
from microcosm.build.uk_runtime import copy_id_table
from microcosm.build.uk_runtime import high_value_home_support as hv
from microcosm.build.uk_runtime.atomic_area_support import IDENTITY_COLUMN
from microcosm.build.uk_runtime.atomic_household_identity import household_draw_key
from microcosm.build.uk_runtime.geographic_clones import ladder_clone_index_column
from microcosm.build.uk_runtime.geographic_identity import (
    GEOGRAPHIC_IDENTITY_SOURCE,
    frs_vintages_by_source_year,
)
from microcosm.build.uk_runtime.household_lineage import (
    HIGH_VALUE_SUPPORT_COPY_COLUMN,
    HIGH_VALUE_SUPPORT_INDEX_COLUMN,
    UK_COPY_LINEAGE,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
)
from microcosm.build.uk_runtime.receipt_counts import UK_WITHHELD
from microcosm.build.uk_runtime.spine_council_tax import (
    relocalise_uk_council_tax,
    uk_council_tax_references,
)
from microcosm.build.uk_runtime.spine_geography import (
    HIGH_VALUE_SUPPORT_BRANCH,
    UKAreaBRMAStageTransform,
    UKAtomicAreaAssignStageTransform,
    UKLocalAuthorityStageTransform,
    UKRFDRStageTransform,
    UKSPICopyAreasStageTransform,
    load_committed_brma_counts,
    load_committed_brma_layer,
    load_committed_rfdr_layer,
    uk_area_brmas,
    uk_local_authorities,
    uk_rfdr_flags,
)
from microcosm.frame import Frame
from test_support.microcosm_build.uk_conditioned_support_fixtures import (
    NI,
    TOY_HOUSEHOLDS,
    invented_copy_picks,
    toy_conditioned_support_payloads,
    toy_conditioned_supports,
    toy_spine_frame,
    with_invented_support_channel,
)
from test_support.microcosm_build.uk_council_tax_fixtures import (
    invented_band_d,
    toy_area_draw_definition,
)

_CLONE = ladder_clone_index_column("household")
_REDUCTION = "council_tax_benefit_reported"
_LHA_CATEGORIES = ("A", "B", "C", "D", "E")
_STRATUM = 0.5
_COPIES = 3
#: The columns the stage rewrites on a copy, and the ids and lineage it mints.
_REWRITTEN = set(hv.UK_HIGH_VALUE_SUPPORT_REWRITES)
_MINTED = {
    "household_id",
    HIGH_VALUE_SUPPORT_COPY_COLUMN,
    HIGH_VALUE_SUPPORT_INDEX_COLUMN,
    hv.HIGH_VALUE_SUPPORT_COPIES_COLUMN,
}


class _Engine:
    """A stand-in for the UK rules engine that keeps every call; each benefit
    unit's ``LHA_category`` is invented from its own id, modulo five."""

    def __init__(self) -> None:
        self.country = "uk"
        self.calls: list[tuple[Frame, tuple[str, ...], str]] = []

    def materialize(self, frame, variables, period):
        self.calls.append((frame, tuple(variables), str(period)))
        ids = frame.table("benunit")["benunit_id"].to_numpy(dtype=np.int64)
        return {
            "LHA_category": np.asarray(
                [_LHA_CATEGORIES[int(i) % 5] for i in ids], dtype=object
            )
        }


@functools.cache
def _stages() -> dict:
    return {stage.stage: stage for stage in load_country_spec("uk").sources.stages}


def _rebuilt(frame: Frame, household: pd.DataFrame, person: pd.DataFrame) -> Frame:
    return uk_national_frame(
        person=person,
        benunit=frame.table("benunit"),
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )


def _owners(frame: Frame) -> Frame:
    """Every household an owner, with an invented value, council tax and
    reduction by its source alone, so a household's clones and copies hold
    the same; the rebate equals the reduction its first person carries."""

    household = frame.table("household").copy()
    person = frame.table("person").copy()
    sources = household["source_household_id"].to_numpy(dtype=np.int64)
    household["tenure_type"] = pd.array(
        ["OWNED_OUTRIGHT"] * len(household), dtype="string"
    )
    household["main_residence_value"] = 150_000.0 + 12_500.0 * sources
    bill = np.round(1_000.0 + 37.25 * sources, 2)
    reduction = np.where(sources % 3 == 0, np.round(0.25 * bill, 2), 0.0)
    household["council_tax"] = bill.astype(np.float64)
    household["council_tax_rebate"] = reduction.astype(np.float64)
    first = ~person["person_household_id"].duplicated().to_numpy(dtype=bool)
    by_household = pd.Series(reduction, index=household["household_id"].to_numpy())
    carried = person["person_household_id"].map(by_household).to_numpy(dtype=float)
    person[_REDUCTION] = np.where(first, carried, 0.0).astype(np.float64)
    return _rebuilt(frame, household, person)


@functools.cache
def _spine(clones: int) -> Frame:
    """The toy households at the high-value stage's place, at ``clones``."""

    stages, payloads = _stages(), toy_conditioned_support_payloads()
    frame = toy_spine_frame(clones=clones, households=TOY_HOUSEHOLDS)
    for transform in (
        UKAtomicAreaAssignStageTransform(
            stage=stages["atomic_area_assign"], supports=payloads
        ),
        UKLocalAuthorityStageTransform(stage=stages["local_authority"]),
        UKRFDRStageTransform(stage=stages["rfdr"], supports=payloads),
        UKAreaBRMAStageTransform(stage=stages["area_brma"], engine=_Engine()),
    ):
        frame = transform(frame)
    frame = with_invented_support_channel(frame, invented_copy_picks)
    frame = UKSPICopyAreasStageTransform(
        stage=stages["spi_copy_areas"], engine=_Engine(), supports=payloads
    )(frame)
    return _owners(frame)


def _support(frame: Frame, engine: object | None = None):
    brma_layer, brma_sha = load_committed_brma_layer()
    counts, counts_sha = load_committed_brma_counts()
    rfdr_layer, rfdr_sha = load_committed_rfdr_layer()
    return hv.support_uk_high_value_homes(
        frame,
        definition=toy_area_draw_definition(),
        supports=toy_conditioned_supports(),
        engine=_Engine() if engine is None else engine,
        brma_layer=brma_layer,
        brma_layer_sha256=brma_sha,
        count_resource=counts,
        count_resource_sha256=counts_sha,
        rfdr_layer=rfdr_layer,
        rfdr_layer_sha256=rfdr_sha,
        band_d=invented_band_d(),
        stratum_lower=_STRATUM,
        copies=_COPIES,
    )


@functools.cache
def _run(clones: int):
    engine = _Engine()
    return _support(_spine(clones), engine), engine


@functools.cache
def _split(clones: int):
    return hv.split_high_value_support_households(
        _spine(clones), stratum_lower=_STRATUM, copies=_COPIES
    )


def _copies(household: pd.DataFrame) -> np.ndarray:
    return household[HIGH_VALUE_SUPPORT_COPY_COLUMN].to_numpy(dtype=bool)


def _parents(household: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """The immediate parent of each household at ``rows``, in their order."""

    by_id = household.set_index("household_id", drop=False)
    ids = household.loc[rows, "household_id"].to_numpy(dtype=np.int64)
    return by_id.loc[copy_id_table.parent(ids, hv.HIGH_VALUE_SUPPORT_COPY_ID_FIELD)]


def _bits(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64).view(np.int64)


@pytest.mark.parametrize("clones", [1, 3])
def test_every_region_is_split_and_northern_ireland_has_copies(clones) -> None:
    household = _run(clones)[0].frame.table("household")
    copies = _copies(household)
    regions = set(household.loc[copies, "region"])
    assert regions == set(household["region"])
    assert (household.loc[copies, "atomic_area_system"] == NI).any()


@pytest.mark.parametrize("clones", [1, 3])
def test_each_copy_takes_a_key_of_its_own_over_its_copy_path(clones) -> None:
    household = _run(clones)[0].frame.table("household")
    split = _split(clones).frame.table("household")
    copies = _copies(household)
    keys = household[IDENTITY_COLUMN]
    assert keys.is_unique
    # A root and every household left whole keep their keys.
    assert (keys[~copies] == split.loc[~copies, IDENTITY_COLUMN]).all()
    parents = _parents(household, copies)
    assert (keys[copies].to_numpy() != parents[IDENTITY_COLUMN].to_numpy()).all()
    sources, paths = copy_paths(household, UK_COPY_LINEAGE)
    vintages = frs_vintages_by_source_year()
    years = household["source_year"].to_numpy(dtype=np.int64)
    for row in np.flatnonzero(copies).tolist():
        assert dict(paths[row]).get(HIGH_VALUE_SUPPORT_BRANCH, 0) > 0
        assert keys.iloc[row] == household_draw_key(
            source=GEOGRAPHIC_IDENTITY_SOURCE,
            source_vintage=vintages[int(years[row])],
            source_household_id=int(sources[row]),
            clone_path=paths[row],
        )


@pytest.mark.parametrize("clones", [1, 3])
def test_each_copy_is_drawn_in_its_parents_cell_and_takes_its_areas_lookups(
    clones,
) -> None:
    result, _ = _run(clones)
    household = result.frame.table("household")
    copies = _copies(household)
    # The draw-cell guard: every Great Britain row's area is an area of the
    # cell its draw inputs give it, with mass in the weight it samples by.
    uk_council_tax_references(
        household,
        definition=toy_area_draw_definition(),
        supports=toy_conditioned_supports(),
        band_d=invented_band_d(),
    )
    parents = _parents(household, copies)
    for column in ("region", "imd_decile", "council_tax_band"):
        assert (
            household.loc[copies, column].astype(str).to_numpy()
            == parents[column].astype(str).to_numpy()
        ).all(), column
    placed = household.loc[copies]
    assert (
        placed["local_authority"].to_numpy(dtype=object)
        == uk_local_authorities(placed["local_authority_code"]).to_numpy(dtype=object)
    ).all()
    rfdr_layer, _ = load_committed_rfdr_layer()
    assert np.array_equal(
        placed["in_rural_fuel_duty_relief_area"].to_numpy(dtype=bool),
        uk_rfdr_flags(placed, layer=rfdr_layer, supports=toy_conditioned_supports()),
    )
    # A Great Britain copy reads its BRMA off its own area through the layer,
    # as G4 reads a household's; G4 reads no LHA_category there.
    brma_layer, _ = load_committed_brma_layer()
    counts, _ = load_committed_brma_counts()
    great_britain = placed.loc[(placed["atomic_area_system"] != NI).to_numpy()]
    benunit = result.frame.table("benunit")
    expected = uk_area_brmas(
        result.frame.table("person"),
        benunit,
        great_britain,
        lha_category=np.full(len(benunit), _LHA_CATEGORIES[0], dtype=object),
        layer=brma_layer,
        count_resource=counts,
    )
    assert (great_britain["brma"].to_numpy(dtype=object) == expected).all()


@pytest.mark.parametrize("clones", [1, 3])
def test_every_root_and_every_other_column_keep_their_values(clones) -> None:
    household = _run(clones)[0].frame.table("household")
    split = _split(clones).frame.table("household")
    copies = _copies(household)
    pd.testing.assert_frame_equal(
        household.loc[~copies].reset_index(drop=True),
        split.loc[~copies].reset_index(drop=True),
    )
    kept = [c for c in household.columns if c not in _REWRITTEN | _MINTED]
    parents = _parents(household, copies)
    for column in kept:
        assert (
            household.loc[copies, column].astype(str).to_numpy()
            == parents[column].astype(str).to_numpy()
        ).all(), column


@pytest.mark.parametrize("clones", [1, 3])
def test_moved_copies_take_their_parents_council_tax_times_the_band_d_ratio(
    clones,
) -> None:
    result, _ = _run(clones)
    household = result.frame.table("household")
    person = result.frame.table("person")
    copies = _copies(household)
    parents = _parents(household, copies)
    values = invented_band_d().band_d
    new = household.loc[copies, "local_authority_code"].to_numpy(dtype=object)
    old = parents["local_authority_code"].to_numpy(dtype=object)
    great_britain = household.loc[copies, "region"].to_numpy() != "NORTHERN_IRELAND"
    moved = great_britain & (new != old)
    assert moved.any() and (~moved).any()
    for column in ("council_tax", "council_tax_rebate"):
        got = household.loc[copies, column].to_numpy(dtype=np.float64)
        was = parents[column].to_numpy(dtype=np.float64)
        expected = np.asarray(
            [
                float(Fraction(float(w)) * values[n] / values[o]) if m else w
                for w, n, o, m in zip(was, new, old, moved, strict=True)
            ],
            dtype=np.float64,
        )
        assert np.array_equal(_bits(got), _bits(expected)), column
    reduction = person.set_index("person_id")[_REDUCTION]
    by_household = person.groupby("person_household_id")[_REDUCTION].sum()
    copy_ids = household.loc[copies, "household_id"].to_numpy(dtype=np.int64)
    parent_ids = parents["household_id"].to_numpy(dtype=np.int64)
    for copy_id, parent_id, n, o, m in zip(
        copy_ids, parent_ids, new, old, moved, strict=True
    ):
        expected = float(by_household[parent_id])
        if m:
            expected = float(Fraction(expected) * values[n] / values[o])
        assert by_household[copy_id] == expected
    assert len(reduction) == len(person)
    # The reductions stay within the bill wherever they were.
    bill = household.set_index("household_id")["council_tax"]
    rebate = household.set_index("household_id")["council_tax_rebate"]
    gb = household.loc[
        household["region"] != "NORTHERN_IRELAND", "household_id"
    ].to_numpy()
    assert (rebate.loc[gb].to_numpy() <= bill.loc[gb].to_numpy()).all()
    assert (
        by_household.reindex(gb).fillna(0.0).to_numpy() <= bill.loc[gb].to_numpy()
    ).all()


def test_northern_irelands_copies_take_one_engine_call_over_their_households() -> None:
    result, engine = _run(3)
    household = result.frame.table("household")
    copies = _copies(household) & (household["atomic_area_system"] == NI).to_numpy()
    ((frame, variables, _),) = engine.calls
    assert variables == ("LHA_category",)
    assert set(frame.table("household")["household_id"]) == set(
        household.loc[copies, "household_id"]
    )


def test_clone_zero_copies_keep_their_keys_places_and_council_tax_as_k_grows() -> None:
    columns = [
        "household_id",
        IDENTITY_COLUMN,
        "atomic_area_code",
        "local_authority_code",
        "constituency_code",
        "brma",
        "council_tax",
        "council_tax_rebate",
    ]

    def clone_zero(clones: int) -> pd.DataFrame:
        household = _run(clones)[0].frame.table("household")
        rows = _copies(household) & (household[_CLONE] == 0).to_numpy()
        return household.loc[rows, columns].set_index("household_id")

    # A clone of a household tied on value with its other clones enters the
    # stratum at its share, so the selected clone-0 households can differ at
    # the stratum's edge; every copy both clone counts make is the same.
    one, three = clone_zero(1), clone_zero(3)
    shared = one.index.intersection(three.index)
    assert len(shared) >= len(one) // 2 > 0
    pd.testing.assert_frame_equal(one.loc[shared], three.loc[shared])


def test_the_evidence_counts_sources_rounded_and_prints_no_key_or_code() -> None:
    result, _ = _run(3)
    evidence = result.evidence()
    areas = evidence["copy_areas"]
    assert areas["counted_among"] == "clone_0"
    assert areas["validation"]["outcome"] == "pass"
    assert areas["levels"] == ["L0", "L1", "L2", "L3"]
    counts = areas["copy_source_households"]
    assert set(counts) == {
        "uk_ew_output_area_2021",
        "uk_scotland_output_area_2022",
        "uk_ni_data_zone_2021",
    }
    assert all(
        isinstance(v, int) or v in ("<3", "3-9", UK_WITHHELD) for v in counts.values()
    )
    council_tax = evidence["council_tax"]
    assert council_tax["counted_among"] == "clone_0"
    assert evidence["rewrites"] == list(hv.UK_HIGH_VALUE_SUPPORT_REWRITES)
    text = json.dumps(evidence, sort_keys=True)
    household = result.frame.table("household")
    for column in (IDENTITY_COLUMN, "atomic_area_code", "oa_code", "lsoa_code"):
        for value in household[column].dropna().astype(str).unique().tolist():
            assert value not in text, column


def test_a_path_with_a_later_copy_step_is_refused_by_name() -> None:
    frame = _spine(1)
    household = frame.table("household").copy()
    household["household_is_cgt_support_copy"] = False
    household["cgt_support_copy_index"] = np.zeros(len(household), dtype=np.int64)
    household.loc[household.index[0], "household_is_cgt_support_copy"] = True
    household.loc[household.index[0], "cgt_support_copy_index"] = 1
    with pytest.raises(ValueError, match="branch other than"):
        _support(_rebuilt(frame, household, frame.table("person")))


def test_the_packaged_stage_builds_the_transform_and_runs_it() -> None:
    stage = _stages()[hv.HIGH_VALUE_SUPPORT_STAGE_NAME]
    transform = hv.UKHighValueHomeSupportStageTransform(
        stage=stage,
        engine=_Engine(),
        supports=toy_conditioned_support_payloads(),
        band_d=invented_band_d(),
    )
    out = transform(_spine(1))
    household = out.table("household")
    assert household[HIGH_VALUE_SUPPORT_COPY_COLUMN].any()
    evidence = transform.checkpoint_metadata()["evidence"]
    assert evidence["stage"] == hv.HIGH_VALUE_SUPPORT_STAGE_NAME
    assert set(evidence) >= {"copy_areas", "council_tax", "rewrites"}


def test_the_council_tax_rescale_refuses_what_it_cannot_price_naming_columns() -> None:
    frame = _spine(1)
    household = frame.table("household")
    person = frame.table("person")
    rows = (household["region"] != "NORTHERN_IRELAND").to_numpy(dtype=bool)
    before = household["local_authority_code"].to_numpy(dtype=object)
    band_d = invented_band_d()
    unchanged = relocalise_uk_council_tax(
        household, person, rows=rows, authorities_before=before, band_d=band_d
    )
    assert not unchanged.moved.any()
    assert np.array_equal(_bits(unchanged.council_tax), _bits(household["council_tax"]))
    with pytest.raises(ValueError, match="align"):
        relocalise_uk_council_tax(
            household, person, rows=rows[1:], authorities_before=before, band_d=band_d
        )
    unpriced = before.copy()
    unpriced[np.flatnonzero(rows)[0]] = "E99999999"
    with pytest.raises(ValueError, match="has none"):
        relocalise_uk_council_tax(
            household, person, rows=rows, authorities_before=unpriced, band_d=band_d
        )
    missing = before.copy()
    missing[np.flatnonzero(rows)[0]] = None
    with pytest.raises(ValueError, match="held none"):
        relocalise_uk_council_tax(
            household, person, rows=rows, authorities_before=missing, band_d=band_d
        )
    with pytest.raises(TypeError, match="UKCouncilTaxBandD"):
        relocalise_uk_council_tax(
            household, person, rows=rows, authorities_before=before, band_d=object()
        )
