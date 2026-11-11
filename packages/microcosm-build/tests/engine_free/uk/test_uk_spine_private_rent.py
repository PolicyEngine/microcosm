"""The clone-first spine's private-rent stage, G7 (microcosm#931).

G7 reads each Great Britain private renter's rent off its BRMA's list of rents
with every row its own spine household, keyed on its geography key, tie-broken
on the stage's own stream. No stage declares it yet, so the transform here
runs as a stage built from the reviewed declaration spine_private_rent states,
on the committed lists of rents and census counts.

The frames are invented households (toy rents, tenures and bedrooms), cloned
and keyed by the spine's own expansion and identity steps, placed by the
spine's area draw on test_support's toy conditioned supports, whose areas sit
in BRMAs the committed lists cover, given their BRMA through the committed
area-to-BRMA layer as the BRMA stage gives it, and given SPI copies keyed over
their copy path as the SPI-copy area stage keys them.

They hold: on a copy-free one-clone frame the transform equals
``localise_rents`` given the same keys and stream; per-row keys run where the
dense line's spine keys refuse; clone 0's tie uniforms at K = 3 are K = 1's;
two FRS renters at one rent in one cell take distinct ranks; the sampled
rung's region floor counts true sources; Northern Ireland, other tenures and
zero rents keep their rents bit for bit; the re-draw does not depend on row
order; every refusal names no key or BRMA of a household; and the evidence,
counted among clone-0 rows, is byte-equal at K = 1 and K = 3.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_lineage import copy_paths
from microcosm.build.source_manifest import (
    SourceOperationSpec,
    SourceStageSpec,
    _reject_executable_parameter_keys,
    _reject_incumbent_dependencies,
)
from microcosm.build.uk_runtime import copy_id_table
from microcosm.build.uk_runtime import private_rent_localisation as prl
from microcosm.build.uk_runtime.atomic_household_identity import household_draw_key
from microcosm.build.uk_runtime.brma_area_layer import (
    area_brmas,
    load_brma_area_layer,
)
from microcosm.build.uk_runtime.geographic_clones import (
    GEOGRAPHY_HOUSEHOLD_KEY_COLUMN,
    ladder_clone_index_column,
)
from microcosm.build.uk_runtime.geographic_identity import (
    GEOGRAPHIC_IDENTITY_SOURCE,
    frs_vintages_by_source_year,
)
from microcosm.build.uk_runtime.household_lineage import (
    GEOGRAPHIC_SUPPORT_BRANCH,
    UK_COPY_LINEAGE,
    without_geographic_branch,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
)
from microcosm.build.uk_runtime.property_localisation import spine_household_keys
from microcosm.build.uk_runtime.receipt_counts import rounded_count
from microcosm.build.uk_runtime.spine_geography import (
    ATOMIC_AREA_ASSIGN_STAGE_NAME,
    UK_ATOMIC_AREA_ASSIGN_OUTPUT_COLUMNS,
    UKAtomicAreaAssignStageTransform,
    atomic_area_assign_operation_parameters,
)
from microcosm.build.uk_runtime.spine_private_rent import (
    COUNTS_ARTIFACT_ROLE,
    COUNTS_RESOURCE,
    PRIVATE_RENT_OPERATION_KIND,
    PRIVATE_RENT_STAGE_NAME,
    QUANTILES_ARTIFACT_ROLE,
    QUANTILES_RESOURCE,
    RENT_RANK_PROCESS,
    UK_PRIVATE_RENT_INPUTS,
    UK_PRIVATE_RENT_MASS_CONSERVATION_REASON,
    UK_PRIVATE_RENT_REWRITES,
    UK_SPINE_PRIVATE_RENT_SEED,
    UK_SPINE_PRIVATE_RENT_STREAM,
    UK_SPINE_PRIVATE_RENT_STREAM_EXPERIMENT,
    UKPrivateRentStageTransform,
    draw_uk_spine_private_rents,
    load_committed_private_rent_resources,
    localise_uk_spine_private_rents,
    private_rent_operation_parameters,
    uk_spine_rent_tie_uniforms,
)
from microcosm.frame import Frame
from microcosm.graph.randomness import keyed_uniform
from test_support.microcosm_build.uk_conditioned_support_fixtures import (
    ToyHousehold,
    toy_conditioned_support_payloads,
    toy_spine_frame,
)

_KEY = GEOGRAPHY_HOUSEHOLD_KEY_COLUMN
_CLONE = ladder_clone_index_column("household")
_SPI_FIELD = "spi_support_channel"
#: The design's stream for G7, spelled out here rather than read from the
#: module, so a drift in the module's constant fails.
_DESIGN_STREAM = ("sha256-u53-v1", "uk-spine-private-rent-v1", 0, 0)
#: An invented annual rent level; every toy rent is a factor of it.
_LEVEL = 14_000.0
#: Northern Ireland's BRMA before the BRMA stage: the layer keeps it.
_NI_BRMA = "BELFAST"


@dataclass(frozen=True)
class _Home:
    """An invented household's region and housing; its rent is ``factor``
    times the invented level, rounded to pence."""

    region: str
    tenure: str
    bedrooms: int
    factor: float

    @property
    def rent(self) -> float:
        return round(_LEVEL * self.factor, 2)


#: Invented households, one source each (position + 1). London holds two FRS
#: renters at one rent in one cell (0 and 1) and three households the stage
#: keeps (an owner, a social renter with a rent, a private renter with none);
#: every region with a re-drawn row holds FRS renters, so a full-scale spine
#: runs; Northern Ireland's private renter keeps its rent.
_HOMES: tuple[_Home, ...] = (
    _Home("LONDON", "RENT_PRIVATELY", 2, 1.0),
    _Home("LONDON", "RENT_PRIVATELY", 2, 1.0),
    _Home("LONDON", "RENT_PRIVATELY", 1, 0.8),
    _Home("LONDON", "RENT_PRIVATELY", 3, 1.3),
    _Home("LONDON", "RENT_PRIVATELY", 2, 0.9),
    _Home("LONDON", "OWNED_OUTRIGHT", 3, 0.0),
    _Home("LONDON", "RENT_FROM_COUNCIL", 2, 0.6),
    _Home("LONDON", "RENT_PRIVATELY", 2, 0.0),
    _Home("NORTH_EAST", "RENT_PRIVATELY", 2, 1.0),
    _Home("NORTH_EAST", "RENT_PRIVATELY", 1, 0.9),
    _Home("NORTH_EAST", "RENT_PRIVATELY", 3, 1.1),
    _Home("NORTH_EAST", "OWNED_WITH_MORTGAGE", 3, 0.0),
    _Home("YORKSHIRE", "RENT_PRIVATELY", 2, 1.0),
    _Home("YORKSHIRE", "RENT_PRIVATELY", 4, 1.2),
    _Home("WALES", "RENT_PRIVATELY", 2, 1.0),
    _Home("SCOTLAND", "RENT_PRIVATELY", 2, 1.0),
    _Home("SCOTLAND", "RENT_PRIVATELY", 1, 0.85),
    _Home("NORTHERN_IRELAND", "RENT_PRIVATELY", 2, 0.55),
    _Home("NORTHERN_IRELAND", "OWNED_OUTRIGHT", 3, 0.0),
)
#: The SPI copies (ordinal 1 in the support channel's field) of four
#: households, in every clone, with the housing the shell would draw for each
#: clone apart: London's 1-bedroom renter's copy rents four bedrooms, a cell
#: no London FRS renter holds; London's 2-bedroom renter's copy rents at a
#: rent of its clone's own; the North East owner's copy owns in clone 0 and
#: rents four bedrooms in every later clone; Scotland's copy rents one
#: bedroom.
_Copy = Callable[[int], _Home]
_COPIES: Mapping[int, _Copy] = {
    2: lambda clone: _Home("LONDON", "RENT_PRIVATELY", 4, 1.4),
    4: lambda clone: _Home("LONDON", "RENT_PRIVATELY", 2, 0.95 + 0.1 * clone),
    11: lambda clone: (
        _Home("NORTH_EAST", "OWNED_WITH_MORTGAGE", 3, 0.0)
        if clone == 0
        else _Home("NORTH_EAST", "RENT_PRIVATELY", 4, 1.0 + 0.05 * clone)
    ),
    15: lambda clone: _Home("SCOTLAND", "RENT_PRIVATELY", 1, 0.7),
}
#: Wales's FRS renter turned owner, with an SPI copy that rents: a region
#: whose only re-drawn renter is SPI-channel.
_WALES_OWNS = tuple(
    _Home("WALES", "OWNED_OUTRIGHT", 2, 0.0) if position == 14 else home
    for position, home in enumerate(_HOMES)
)
_WALES_COPY: Mapping[int, _Copy] = {
    14: lambda clone: _Home("WALES", "RENT_PRIVATELY", 2, 1.0),
}


# --- Building the frames ------------------------------------------------------------


def _stage(**overrides: object) -> SourceStageSpec:
    """G7's stage built from the reviewed declaration spine_private_rent
    states, with ``overrides`` replacing fields."""

    stage = SourceStageSpec(
        stage=PRIVATE_RENT_STAGE_NAME,
        survey="Family Resources Survey 2024-25",
        source="Microcosm structural operator (microcosm#931)",
        grain="household",
        artifacts=tuple(
            {
                "role": role,
                "kind": "public_aggregate_reference",
                "resource": resource,
                "format": "json",
                "runtime_sha256_required": True,
            }
            for role, resource in (
                (QUANTILES_ARTIFACT_ROLE, QUANTILES_RESOURCE),
                (COUNTS_ARTIFACT_ROLE, COUNTS_RESOURCE),
            )
        ),
        operations=tuple(
            SourceOperationSpec(kind=kind, parameters=dict(parameters))
            for kind, parameters in private_rent_operation_parameters()
        ),
        outputs=(),
        rewrites=UK_PRIVATE_RENT_REWRITES,
    )
    return replace(stage, **overrides)


def _transform(*, sampled_rung: bool = False) -> UKPrivateRentStageTransform:
    return UKPrivateRentStageTransform(stage=_stage(), sampled_rung=sampled_rung)


@functools.cache
def _area_draw() -> UKAtomicAreaAssignStageTransform:
    """The spine's area draw (G2) on the toy conditioned supports."""

    stage = SourceStageSpec(
        stage=ATOMIC_AREA_ASSIGN_STAGE_NAME,
        survey="Family Resources Survey 2024-25",
        source="Microcosm structural operator (microcosm#931)",
        grain="household",
        artifacts=(),
        operations=tuple(
            SourceOperationSpec(kind=kind, parameters=dict(parameters))
            for kind, parameters in atomic_area_assign_operation_parameters()
        ),
        outputs=UK_ATOMIC_AREA_ASSIGN_OUTPUT_COLUMNS,
    )
    return UKAtomicAreaAssignStageTransform(
        stage=stage, supports=toy_conditioned_support_payloads()
    )


@functools.cache
def _layer():
    return load_brma_area_layer(
        files("microcosm.build.uk").joinpath("brma_area_layer.json").read_bytes()
    )


@functools.cache
def _resources():
    return load_committed_private_rent_resources()


def _brmas(household: pd.DataFrame) -> np.ndarray:
    """Each row's BRMA: off its drawn area in Great Britain, as the BRMA stage
    reads it; Northern Ireland's kept."""

    brma, _ = area_brmas(
        layer=_layer(),
        area_system=household["atomic_area_system"].to_numpy(dtype=object),
        lsoa_code=household["lsoa_code"].to_numpy(dtype=object),
        oa_code=household["oa_code"].to_numpy(dtype=object),
        region=household["region"].to_numpy(dtype=object),
        brma=np.full(len(household), _NI_BRMA, dtype=object),
    )
    return brma


def _keys(household: pd.DataFrame, *, pooled: bool = False) -> pd.Series:
    """Each row's geography key over its copy path, as the geography identity
    and the SPI-copy area stage key them; ``pooled`` drops the geographic
    branch, as the dense line's spine keys did."""

    sources, paths = copy_paths(
        household, UK_COPY_LINEAGE, require=(_SPI_FIELD, GEOGRAPHIC_SUPPORT_BRANCH)
    )
    if pooled:
        paths = without_geographic_branch(paths)
    vintages = frs_vintages_by_source_year()
    keys = [
        household_draw_key(
            source=GEOGRAPHIC_IDENTITY_SOURCE,
            source_vintage=vintages[int(year)],
            source_household_id=int(source),
            clone_path=path,
        )
        for source, year, path in zip(
            sources.tolist(), household["source_year"].tolist(), paths, strict=True
        )
    ]
    return pd.Series(keys, index=household.index, dtype=pd.StringDtype("python"))


def _house(household: pd.DataFrame, homes: list[_Home]) -> pd.DataFrame:
    """``household`` with each row's tenure, bedrooms, rent and mortgage."""

    housed = household.copy()
    housed["tenure_type"] = pd.Series(
        [home.tenure for home in homes], index=housed.index, dtype=object
    )
    housed["num_bedrooms"] = np.asarray([home.bedrooms for home in homes], np.int64)
    housed["rent"] = np.asarray([home.rent for home in homes], dtype=np.float64)
    housed["mortgage_interest_repayment"] = np.asarray(
        [3_000.0 if home.tenure == "OWNED_WITH_MORTGAGE" else 0.0 for home in homes]
    )
    return housed


def _with_spi_copies(frame: Frame, copies: Mapping[int, _Copy]) -> Frame:
    """``frame`` with an SPI copy of each named household in every clone, at
    half its parent's weight, keyed over its copy path."""

    household = frame.table("household")
    person = frame.table("person")
    benunit = frame.table("benunit")
    weights = np.asarray(frame.weights_for("household").values, dtype=float).copy()
    positions = household["source_household_id"].to_numpy() - 1
    picked = np.isin(positions, list(copies))
    parents = household.loc[picked]
    parent_ids = parents["household_id"].to_numpy()
    homes = [
        copies[int(position)](int(clone))
        for position, clone in zip(
            positions[picked], parents[_CLONE].to_numpy(), strict=True
        )
    ]
    copied = _house(parents, homes)
    copied["household_id"] = copy_id_table.mint(parent_ids, _SPI_FIELD, 1)
    copied["household_support_channel"] = "spi"
    copied["household_support_clone_index"] = np.ones(len(copied), dtype=np.int64)
    members = person.loc[person["person_household_id"].isin(parent_ids)].copy()
    units = benunit.loc[benunit["benunit_id"].isin(members["person_benunit_id"])].copy()
    for column in ("person_id", "person_household_id", "person_benunit_id"):
        members[column] = copy_id_table.mint(members[column], _SPI_FIELD, 1)
    units["benunit_id"] = copy_id_table.mint(units["benunit_id"], _SPI_FIELD, 1)
    half = weights[picked] / 2.0
    weights[picked] = half
    households = pd.concat([household, copied], ignore_index=True)
    order = np.argsort(households["household_id"].to_numpy(), kind="stable")
    households = households.iloc[order].reset_index(drop=True)
    households[_KEY] = _keys(households)
    return uk_national_frame(
        person=pd.concat([person, members], ignore_index=True)
        .sort_values("person_id", kind="stable")
        .reset_index(drop=True),
        benunit=pd.concat([benunit, units], ignore_index=True)
        .sort_values("benunit_id", kind="stable")
        .reset_index(drop=True),
        household=households,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=np.concatenate([weights, half])[order],
        mass_log=frame.mass_log,
    )


def _build(clones: int, homes: tuple[_Home, ...], copies: Mapping[int, _Copy]) -> Frame:
    cases = [ToyHousehold(home.region, 0, "D", False, "L3") for home in homes]
    placed = _area_draw()(toy_spine_frame(clones=clones, households=cases))
    household = placed.table("household")
    positions = household["source_household_id"].to_numpy() - 1
    housed = _house(household, [homes[int(position)] for position in positions])
    housed["brma"] = _brmas(housed)
    housed["household_support_channel"] = "frs"
    housed["household_support_clone_index"] = np.zeros(len(housed), dtype=np.int64)
    person = placed.table("person").copy()
    first = ~person["person_household_id"].duplicated()
    for column, amount in (
        ("housing_benefit_reported", 2_100.0),
        ("universal_credit_reported", 4_200.0),
    ):
        person[column] = np.where(first, amount, 0.0)
    frame = uk_national_frame(
        person=person,
        benunit=placed.table("benunit"),
        household=housed,
        time_period=uk_time_period(placed),
        weight_kind=uk_household_weight_kind(placed),
        household_weights=placed.weights_for("household").values,
        mass_log=placed.mass_log,
    )
    return _with_spi_copies(frame, copies) if copies else frame


@functools.cache
def _frame(clones: int, variant: str = "copies") -> Frame:
    """The toy spine at G7's position: ``variant`` "copies" (the default),
    "base" (no SPI copy) or "wales" (Wales's renter only an SPI copy)."""

    homes, copies = {
        "copies": (_HOMES, _COPIES),
        "base": (_HOMES, {}),
        "wales": (_WALES_OWNS, _WALES_COPY),
    }[variant]
    return _build(clones, homes, copies)


def _household(clones: int, variant: str = "copies") -> pd.DataFrame:
    return _frame(clones, variant).table("household")


def _weights(clones: int, variant: str = "copies") -> np.ndarray:
    return np.asarray(_frame(clones, variant).weights_for("household").values)


def _draw(clones: int, variant: str = "copies", *, strict: bool = True):
    return draw_uk_spine_private_rents(
        _household(clones, variant),
        _weights(clones, variant),
        resources=_resources(),
        strict=strict,
    )


def _redrawn(household: pd.DataFrame) -> np.ndarray:
    return prl.redraw_mask(
        household["region"].to_numpy(dtype=object),
        household["tenure_type"].to_numpy(dtype=object),
        household["rent"].to_numpy(dtype=float),
    )


def _localise(household, weights, *, spine_key, source, tie_uniform, strict=True):
    """``localise_rents`` on ``household`` with the given keys, sources and
    tie-breaks and the committed resources."""

    return prl.localise_rents(
        spine_key=np.asarray(spine_key, dtype=object),
        source=np.asarray(source),
        region=household["region"].to_numpy(dtype=object),
        tenure_type=household["tenure_type"].to_numpy(dtype=object),
        rent=household["rent"].to_numpy(dtype=float),
        num_bedrooms=household["num_bedrooms"].to_numpy(),
        brma=household["brma"].to_numpy(dtype=object),
        channel=household["household_support_channel"].to_numpy(dtype=object),
        weight=weights,
        tie_uniform=tie_uniform,
        quantiles=_resources().quantiles,
        counts=_resources().counts,
        strict=strict,
    )


def _ranks(household, weights, keys) -> pd.Series:
    """Each re-drawn spine household's rank, as ``localise_rents`` ranks it
    under ``keys``, with the stage's tie-breaks of those keys."""

    redrawn = _redrawn(household)
    spine = prl._spine_rows(
        spine_key=np.asarray(keys, dtype=object)[redrawn],
        source=household["source_household_id"].to_numpy()[redrawn],
        region=household["region"].to_numpy(dtype=object)[redrawn],
        category=prl.rent_categories(household["num_bedrooms"].to_numpy()[redrawn]),
        rent=household["rent"].to_numpy(dtype=float)[redrawn],
        channel=household["household_support_channel"].to_numpy(dtype=object)[redrawn],
        weight=np.asarray(weights)[redrawn],
    )
    ties = dict(
        zip(spine.index, uk_spine_rent_tie_uniforms(list(spine.index)), strict=True)
    )
    ranks, _ = prl.rent_uniforms(spine, ties)
    return ranks


def _bits(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).view(np.int64)


# --- The reviewed declaration -------------------------------------------------------


def test_the_reviewed_operations_are_declarative_and_name_the_stages_stream() -> None:
    for kind, parameters in private_rent_operation_parameters():
        _reject_executable_parameter_keys(parameters, context=kind)
        _reject_incumbent_dependencies(parameters, context=kind)
        assert json.loads(json.dumps(parameters)) == parameters
    (kind, rents), (receipt_kind, receipt) = private_rent_operation_parameters()
    assert kind == PRIVATE_RENT_OPERATION_KIND == "localise_private_rents"
    assert UK_SPINE_PRIVATE_RENT_STREAM == _DESIGN_STREAM
    assert (
        rents["stream"] == UK_SPINE_PRIVATE_RENT_STREAM_EXPERIMENT == _DESIGN_STREAM[1]
    )
    assert rents["seed"] == UK_SPINE_PRIVATE_RENT_SEED == _DESIGN_STREAM[3] == 0
    assert rents["identity"] == _KEY and rents["process"] == RENT_RANK_PROCESS
    # The stage's stream is its own, not the retired dense node's.
    assert UK_SPINE_PRIVATE_RENT_STREAM_EXPERIMENT != prl.STREAM_EXPERIMENT
    assert rents["sampled_region_renters"] == prl.MIN_SAMPLED_REGION_RENTERS
    assert rents["scale_band"] == list(prl.SCALE_BAND)
    assert rents["weeks_in_year"] == prl.WEEKS_IN_YEAR
    assert rents["method"] == prl.METHOD_VERSION
    assert receipt_kind == "record_mass_conservation_receipt"
    assert receipt["reason"] == UK_PRIVATE_RENT_MASS_CONSERVATION_REASON
    assert receipt["declared_factor"] == 1.0 and receipt["entity"] == "household"


def test_the_transform_runs_only_as_its_reviewed_stage() -> None:
    stage = _stage()
    UKPrivateRentStageTransform(stage=stage)
    rents, receipt = stage.operations
    drifted = SourceOperationSpec(
        kind=rents.kind, parameters={**rents.parameters, "seed": 1}
    )
    (quantiles, counts) = stage.artifacts
    for overrides, message in (
        ({"stage": "frs_brma"}, "Expected stage"),
        ({"grain": "benunit"}, "grain must be 'household'"),
        ({"outputs": ("rent",)}, "rewrites exactly the rent"),
        ({"rewrites": ("rent", "subrent")}, "rewrites exactly the rent"),
        ({"rewrites": ()}, "rewrites exactly the rent"),
        ({"operations": (drifted, receipt)}, "drifted"),
        ({"operations": (rents,)}, "operation order drifted"),
        ({"artifacts": (quantiles,)}, "and no other"),
        (
            {"artifacts": (quantiles, {**counts, "runtime_sha256_required": False})},
            "runtime_sha256_required",
        ),
        (
            {
                "artifacts": (
                    quantiles,
                    counts,
                    {
                        "role": "other",
                        "resource": "x.json",
                        "runtime_sha256_required": True,
                    },
                )
            },
            "and no other",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            UKPrivateRentStageTransform(stage=_stage(**overrides))
    with pytest.raises(TypeError, match="SourceStageSpec"):
        UKPrivateRentStageTransform(stage=object())
    with pytest.raises(TypeError, match="a bool"):
        UKPrivateRentStageTransform(stage=stage, sampled_rung="no")
    with pytest.raises(TypeError, match="a bool"):
        localise_uk_spine_private_rents(
            _frame(1), resources=_resources(), sampled_rung=1
        )


def test_the_transform_reads_the_committed_resources_and_needs_a_run() -> None:
    transform = _transform()
    package = files("microcosm.build.uk")
    assert (
        transform.resources.quantiles_sha256
        == hashlib.sha256(package.joinpath(QUANTILES_RESOURCE).read_bytes()).hexdigest()
    )
    assert (
        transform.resources.counts_sha256
        == hashlib.sha256(package.joinpath(COUNTS_RESOURCE).read_bytes()).hexdigest()
    )
    assert transform.output_columns() == ()
    with pytest.raises(RuntimeError, match="completed stage run"):
        transform.checkpoint_metadata()
    with pytest.raises(TypeError, match="Frame"):
        transform(_household(1))


# --- G7 against localise_rents ------------------------------------------------------


def test_a_copy_free_one_clone_frame_equals_localise_rents_on_the_same_keys() -> None:
    frame = _frame(1, "base")
    household = frame.table("household")
    assert set(household["household_support_channel"]) == {"frs"}
    rents = _transform()(frame).table("household")["rent"]
    keys = household[_KEY].to_numpy(dtype=object)
    redrawn = _redrawn(household)
    ties = keyed_uniform(
        stream=_DESIGN_STREAM, keys=[(key, "rent_rank") for key in keys[redrawn]]
    )
    expected, details = _localise(
        household,
        _weights(1, "base"),
        spine_key=keys,
        source=household["source_household_id"].to_numpy(),
        tie_uniform=dict(zip(keys[redrawn], ties.tolist(), strict=True)),
    )
    assert np.array_equal(_bits(rents), _bits(expected))
    assert details["spine_households_redrawn"] == details["rows_redrawn"] > 0
    # Every Great Britain private renter with a rent moved.
    assert (rents.to_numpy()[redrawn] != household["rent"].to_numpy()[redrawn]).all()


def test_per_row_keys_run_where_the_dense_lines_spine_keys_refuse() -> None:
    """The dense node's spine keys read the CGT lineage, which the spine
    writes only after G7, and pool the clones of an SPI copy whose housing the
    shell drew for each clone apart; G7's per-row keys run on both."""

    household = _household(1)
    with pytest.raises(ValueError, match="lacks lineage"):
        spine_household_keys(
            household,
            source=GEOGRAPHIC_IDENTITY_SOURCE,
            source_vintage=frs_vintages_by_source_year()[
                int(household["source_year"].iloc[0])
            ],
        )
    _transform()(_frame(1))
    two = _household(2)
    pooled = _keys(two, pooled=True)
    with pytest.raises(ValueError, match="disagree"):
        _localise(
            two,
            _weights(2),
            spine_key=pooled.to_numpy(dtype=object),
            source=two["source_household_id"].to_numpy(),
            tie_uniform={key: 0.5 for key in pooled},
        )
    out = _transform()(_frame(2)).table("household")
    redrawn = _redrawn(two)
    assert (out["rent"].to_numpy()[redrawn] != two["rent"].to_numpy()[redrawn]).all()


def test_clone_zero_takes_k_1s_tie_uniforms_at_k_3() -> None:
    single, triple = _household(1), _household(3)
    one, three = _draw(1), _draw(3)
    zero = triple[_CLONE].to_numpy() == 0
    # Clone 0's rows are K = 1's rows: the same ids and keys.
    assert triple.loc[zero, "household_id"].tolist() == single["household_id"].tolist()
    assert triple.loc[zero, _KEY].tolist() == single[_KEY].tolist()

    def by_id(draw, household) -> pd.Series:
        ids = household.loc[draw.tie_uniforms.index, "household_id"].to_numpy()
        return pd.Series(draw.tie_uniforms.to_numpy(), index=ids)

    k1, k3 = by_id(one, single), by_id(three, triple)
    assert np.array_equal(_bits(k3.loc[k1.index]), _bits(k1))
    # The other clones draw their own.
    others = k3.drop(index=k1.index)
    assert len(others) and not set(others.tolist()) & set(k1.tolist())
    # Ranks and scales pool over clones, so clone 0's rents nest approximately.
    rents_k3 = pd.Series(three.rent.to_numpy(), index=triple["household_id"].to_numpy())
    np.testing.assert_allclose(
        rents_k3.loc[single["household_id"].to_numpy()].to_numpy(),
        one.rent.to_numpy(),
        rtol=1e-9,
    )


def test_two_frs_renters_at_one_rent_in_one_cell_take_distinct_ranks() -> None:
    household, weights = _household(1, "base"), _weights(1, "base")
    a, b = household.index[household["source_household_id"].isin([1, 2])]
    assert household.loc[a, "rent"] == household.loc[b, "rent"]
    assert household.loc[a, "num_bedrooms"] == household.loc[b, "num_bedrooms"]
    ranks = _ranks(household, weights, household[_KEY])
    first, second = ranks[household.loc[a, _KEY]], ranks[household.loc[b, _KEY]]
    assert first != second
    # Both sit inside the one weight step their shared rent spans.
    redrawn = _redrawn(household)
    cell = (
        redrawn
        & (household["region"] == "LONDON").to_numpy()
        & (household["num_bedrooms"] == 2).to_numpy()
    )
    rent, total = household.loc[a, "rent"], weights[cell].sum()
    below = weights[cell & (household["rent"] < rent).to_numpy()].sum() / total
    at_or_below = weights[cell & (household["rent"] <= rent).to_numpy()].sum() / total
    for rank in (first, second):
        assert below <= rank <= at_or_below
    # At K = 2 the two clones of one renter take ranks of their own; keyed on
    # the path without the geographic branch, as the dense line keyed them,
    # they would share one.
    two, two_weights = _household(2, "base"), _weights(2, "base")
    clones = two.index[two["source_household_id"] == 1]
    per_row = _ranks(two, two_weights, two[_KEY])
    assert len({per_row[two.loc[row, _KEY]] for row in clones}) == 2
    pooled = _keys(two, pooled=True)
    assert len({pooled[row] for row in clones}) == 1
    assert len(_ranks(two, two_weights, pooled)) == len(per_row) // 2


def test_the_sampled_rungs_region_floor_counts_true_sources() -> None:
    """Every toy region holds fewer FRS renter sources than
    ``MIN_SAMPLED_REGION_RENTERS`` (a statistical floor, not the disclosure
    floor), so a sampled rung ranks every cell against Great Britain at every
    K. London's five FRS renters are fifteen rows at K = 3: counted by row,
    London would rank on its own."""

    renters = _household(1)
    frs = _redrawn(renters) & (renters["household_support_channel"] == "frs").to_numpy()
    held = renters.loc[frs].groupby("region")["source_household_id"].nunique()
    assert held.max() < prl.MIN_SAMPLED_REGION_RENTERS
    assert held["LONDON"] * 3 >= prl.MIN_SAMPLED_REGION_RENTERS
    for clones in (1, 3):
        draw = _draw(clones, strict=False)
        assert draw.cells_ranked_against_their_region == 0
        assert draw.cells_ranked_against_great_britain > 0
    household, weights = _household(3), _weights(3)
    draw = _draw(3, strict=False)
    keys = household[_KEY].to_numpy(dtype=object)
    redrawn = _redrawn(household)
    ties = dict(
        zip(
            keys[redrawn],
            uk_spine_rent_tie_uniforms(list(keys[redrawn])).tolist(),
            strict=True,
        )
    )
    by_source, sourced = _localise(
        household,
        weights,
        spine_key=keys,
        source=household["source_household_id"].to_numpy(),
        tie_uniform=ties,
        strict=False,
    )
    by_row, rowed = _localise(
        household,
        weights,
        spine_key=keys,
        source=household["household_id"].to_numpy(),
        tie_uniform=ties,
        strict=False,
    )
    assert np.array_equal(_bits(draw.rent), _bits(by_source))
    assert rowed["great_britain_cells"] < sourced["great_britain_cells"]
    london = redrawn & (household["region"] == "LONDON").to_numpy()
    assert not np.array_equal(by_row[london], by_source[london])


def test_a_full_scale_spine_refuses_a_region_without_frs_renters() -> None:
    """Wales's only re-drawn renter is an SPI copy: a full-scale spine
    refuses, and a sampled rung ranks and scales Wales on Great Britain's."""

    frame = _frame(1, "wales")
    with pytest.raises(ValueError, match="No FRS private renter"):
        _transform()(frame)
    transform = _transform(sampled_rung=True)
    out = transform(frame).table("household")
    wales = (out["region"] == "WALES").to_numpy() & _redrawn(frame.table("household"))
    assert wales.any()
    assert transform.checkpoint_metadata()["evidence"]["strict"] is False


def test_northern_ireland_other_tenures_and_zero_rents_keep_their_rents() -> None:
    frame = _frame(3)
    transform = _transform()
    out = transform(frame)
    before, after = frame.table("household"), out.table("household")
    redrawn = _redrawn(before)
    kept = ~redrawn
    assert np.array_equal(
        _bits(after["rent"].to_numpy()[kept]), _bits(before["rent"].to_numpy()[kept])
    )
    held = before.loc[kept]
    assert (held["region"] == "NORTHERN_IRELAND").any()
    assert ((held["tenure_type"] == "RENT_PRIVATELY") & (held["rent"] > 0)).any()
    assert (held["tenure_type"] == "RENT_FROM_COUNCIL").any()
    assert ((held["tenure_type"] == "RENT_PRIVATELY") & (held["rent"] == 0)).any()
    assert (
        (held["household_support_channel"] == "spi")
        & (held["tenure_type"] == "OWNED_WITH_MORTGAGE")
    ).any()
    assert after["rent"].dtype == before["rent"].dtype == np.float64
    assert after.drop(columns="rent").equals(before.drop(columns="rent"))
    # Reported Housing Benefit, Universal Credit and mortgages, and every
    # other table, stay as they were.
    for entity in ("person", "benunit"):
        assert out.table(entity).equals(frame.table(entity))
    assert np.array_equal(
        out.weights_for("household").values, frame.weights_for("household").values
    )
    assert out.mass_log[:-1] == frame.mass_log
    receipt = out.mass_log[-1]
    assert receipt.reason == UK_PRIVATE_RENT_MASS_CONSERVATION_REASON
    assert receipt.old_total == receipt.new_total and receipt.declared_factor == 1.0


def test_the_rents_do_not_depend_on_row_order() -> None:
    household, weights = _household(3), _weights(3)
    order = np.random.default_rng(931).permutation(len(household))
    shuffled = household.iloc[order]
    draw = _draw(3)
    permuted = draw_uk_spine_private_rents(
        shuffled, weights[order], resources=_resources(), strict=True
    )
    assert permuted.rent.index.equals(shuffled.index)
    assert np.array_equal(_bits(permuted.rent.loc[household.index]), _bits(draw.rent))
    assert np.array_equal(
        _bits(permuted.tie_uniforms.loc[draw.tie_uniforms.index]),
        _bits(draw.tie_uniforms),
    )
    for name in (
        "source_households_redrawn",
        "cells_ranked_against_their_region",
        "cells_ranked_against_great_britain",
    ):
        assert getattr(permuted, name) == getattr(draw, name)


# --- Evidence -----------------------------------------------------------------------


def test_the_evidence_counts_clone_zero_sources_and_cells() -> None:
    transform = _transform()
    transform(_frame(1))
    evidence = transform.checkpoint_metadata()["evidence"]
    household = _household(1)
    redrawn = _redrawn(household)
    sources = household.loc[redrawn, "source_household_id"].nunique()
    assert evidence == {
        "stage": PRIVATE_RENT_STAGE_NAME,
        "clone_count": 1,
        "counted_among": "clone_0",
        "source_households_redrawn": rounded_count(sources),
        # London's four-bedroom cell, where only an SPI copy rents.
        "cells_ranked_against_their_region": 1,
        "cells_ranked_against_great_britain": 0,
        "strict": True,
        "seed": 0,
        "stream": list(_DESIGN_STREAM),
        "quantiles_sha256": transform.resources.quantiles_sha256,
        "counts_sha256": transform.resources.counts_sha256,
        "method": prl.METHOD_VERSION,
        "weights": "unchanged",
        "rewrites": ["rent"],
    }
    # At K = 1 every row is clone 0: the cells are those localise_rents
    # routes, on a full-scale spine and on a sampled rung.
    keys = household[_KEY].to_numpy(dtype=object)
    ties = dict(
        zip(
            keys[redrawn],
            uk_spine_rent_tie_uniforms(list(keys[redrawn])).tolist(),
            strict=True,
        )
    )
    for strict in (True, False):
        _, details = _localise(
            household,
            _weights(1),
            spine_key=keys,
            source=household["source_household_id"].to_numpy(),
            tie_uniform=ties,
            strict=strict,
        )
        draw = _draw(1, strict=strict)
        assert (
            draw.cells_ranked_against_their_region,
            draw.cells_ranked_against_great_britain,
        ) == (details["fallback_cells"], details["great_britain_cells"])


def test_the_evidence_counts_clone_zero_where_later_clones_differ() -> None:
    """The North East owner's SPI copy rents four bedrooms in clones 1 and 2
    only, so the re-drawn sources and the cells over every clone grow with K;
    the evidence counts clone 0."""

    household, weights = _household(3), _weights(3)
    draw = _draw(3)
    redrawn = _redrawn(household)
    zero = (household[_CLONE] == 0).to_numpy()
    sources = household["source_household_id"].to_numpy()
    in_clone_zero = len(set(sources[redrawn & zero].tolist()))
    in_every_clone = len(set(sources[redrawn].tolist()))
    assert in_every_clone > in_clone_zero
    assert draw.source_households_redrawn == in_clone_zero
    keys = household[_KEY].to_numpy(dtype=object)
    _, details = _localise(
        household,
        weights,
        spine_key=keys,
        source=sources,
        tie_uniform=dict(
            zip(
                keys[redrawn],
                uk_spine_rent_tie_uniforms(list(keys[redrawn])).tolist(),
                strict=True,
            )
        ),
    )
    assert details["fallback_cells"] > draw.cells_ranked_against_their_region == 1


@pytest.mark.parametrize("sampled_rung", [False, True])
def test_every_evidence_count_is_byte_equal_at_k_1_and_k_3(sampled_rung) -> None:
    """The plan's "K clones count as one source": the evidence is the same
    bytes at K = 1 and K = 3 but for the clone count it states."""

    evidence = {}
    for clones in (1, 3):
        transform = _transform(sampled_rung=sampled_rung)
        transform(_frame(clones))
        evidence[clones] = dict(transform.checkpoint_metadata()["evidence"])
    assert (evidence[1].pop("clone_count"), evidence[3].pop("clone_count")) == (1, 3)
    assert json.dumps(evidence[1], sort_keys=True) == json.dumps(
        evidence[3], sort_keys=True
    )


def test_the_evidence_prints_no_key_and_no_brma_of_a_household() -> None:
    transform = _transform()
    transform(_frame(3))
    text = json.dumps(transform.checkpoint_metadata(), sort_keys=True)
    household = _household(3)
    for column in (_KEY, "brma", "atomic_area_code", "lsoa_code"):
        values = {str(value) for value in household[column]}
        assert not any(value in text for value in values), column


# --- Refusals -----------------------------------------------------------------------


@pytest.mark.parametrize("column", UK_PRIVATE_RENT_INPUTS)
def test_a_missing_input_column_is_refused_by_name(column) -> None:
    with pytest.raises(ValueError, match="frame lacks them") as refused:
        draw_uk_spine_private_rents(
            _household(1).drop(columns=column),
            _weights(1),
            resources=_resources(),
            strict=True,
        )
    assert column in str(refused.value)


def test_a_rent_other_than_float64_and_misaligned_weights_are_refused() -> None:
    household = _household(1)
    narrow = household.assign(rent=household["rent"].astype("float32"))
    with pytest.raises(ValueError, match="float64"):
        draw_uk_spine_private_rents(
            narrow, _weights(1), resources=_resources(), strict=True
        )
    with pytest.raises(ValueError, match="one household weight"):
        draw_uk_spine_private_rents(
            household, _weights(1)[:-1], resources=_resources(), strict=True
        )


def _refusal(household: pd.DataFrame) -> str:
    with pytest.raises(ValueError) as refused:
        draw_uk_spine_private_rents(
            household, _weights(1), resources=_resources(), strict=True
        )
    return str(refused.value)


def test_a_repeated_or_missing_key_among_re_drawn_rows_is_refused_naming_none() -> None:
    """An SPI copy keyed as its parent (the SPI-copy area stage never ran)
    shares its key; a re-drawn row without a key has none."""

    household = _household(1)
    redrawn = _redrawn(household)
    copy = household.index[
        redrawn & (household["household_support_channel"] == "spi").to_numpy()
    ][0]
    (parent,) = household.index[
        (household["household_support_channel"] == "frs").to_numpy()
        & (
            household["source_household_id"]
            == household.loc[copy, "source_household_id"]
        ).to_numpy()
    ]
    assert redrawn[household.index.get_loc(parent)]
    shared = household.copy()
    shared.loc[copy, _KEY] = household.loc[parent, _KEY]
    message = _refusal(shared)
    assert "share one" in message
    assert str(household.loc[parent, _KEY]) not in message
    missing = household.copy()
    missing.loc[copy, _KEY] = pd.NA
    assert "needs its" in _refusal(missing)
    # A row the stage does not re-draw needs no key of its own.
    owner = household.index[~redrawn][0]
    unkeyed = household.copy()
    unkeyed.loc[owner, _KEY] = pd.NA
    draw_uk_spine_private_rents(
        unkeyed, _weights(1), resources=_resources(), strict=True
    )


@pytest.mark.parametrize("brma", ["MARS_AND_VENUS", None])
def test_a_brma_without_a_list_is_refused_naming_no_brma(brma) -> None:
    household = _household(1)
    row = household.index[_redrawn(household)][0]
    placed = household.copy()
    placed.loc[row, "brma"] = brma
    message = _refusal(placed)
    assert "no list of rents" in message
    assert "MARS" not in message
    for value in set(household["brma"]):
        assert str(value) not in message
