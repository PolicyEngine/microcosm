"""The clone-first spine's area stages (microcosm#931).

G2 draws each geographic clone's area, G3 resolves its local authority, G4
its BRMA and G5 its fuel-relief flag, and the SPI-copy area stage re-keys and
re-places the rows the SPI support channel copied. No stage declares them
yet, so each transform here runs as a stage built from the reviewed
declaration spine_geography states. The tests run them on toy conditioned
supports whose codes resolve against the committed resources (test_support's
uk_conditioned_support_fixtures), on invented households cloned and keyed by
the spine's own expansion and identity steps. G4 and the copy stage read
``LHA_category`` from a recording stand-in for the rules engine, which gives
each benefit unit an invented category from its id and keeps every frame it
is handed; the copy stage runs on a stand-in for spi_support_channel and
spi_income_band_donors that copies invented households at the support
channel's two ordinals.

They hold: each household in its region at the level its inputs give it; the
written dtypes; clone 0's areas at K = 1, 2 and 3; a draw that does not
depend on row order; the validation, which passes on the prepared frame,
fails on a fallback row's raw decile and fails the stage where the written
ladder disagrees with the mapping; Great Britain's BRMAs off the layer and
Northern Ireland's as frs_brma draws them; every copy's own key, area, local
authority, BRMA (Northern Ireland's drawn on the copy's own benefit units,
after one engine call over those copies alone, which the UK adapter's clone
rule takes as one simulation at every K) and flag, with every other row left
as it was; every refusal, count-free and naming no key or code of a
household, the frame's refusals before any engine call; and every evidence
count byte-equal at K = 1 and K = 3.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

from microcosm.build.atomic_geography import (
    assign_atomic,
    derive_geography,
    validate_geography,
)
from microcosm.build.source_manifest import (
    SourceOperationSpec,
    SourceStageSpec,
    _reject_executable_parameter_keys,
    _reject_incumbent_dependencies,
)
from microcosm.build.uk_runtime import copy_id_table, spine_geography
from microcosm.build.uk_runtime.atomic_area_conditioning import (
    DECILE_INPUT_COLUMN,
    IDENTITY_COLUMN,
    UK_SPINE_AREA_COLUMNS,
    UK_SPINE_ATOMIC_AREA_SEED,
    UK_SPINE_LADDER_COLUMNS,
    uk_conditioned_assignment_definition,
    uk_draw_frame,
    uk_draw_inputs,
    uk_draw_levels,
)
from microcosm.build.uk_runtime.atomic_area_support import IMD_DECILE_COLUMN, _profile
from microcosm.build.uk_runtime.atomic_household_identity import household_draw_key
from microcosm.build.uk_runtime.brma_area_layer import (
    GRAINS as BRMA_GRAINS,
)
from microcosm.build.uk_runtime.brma_area_layer import (
    load_brma_area_layer,
)
from microcosm.build.uk_runtime.frs_brma import (
    UK_BRMA_BENUNIT_SALT,
    UK_BRMA_HOUSEHOLD_PICK_SALT,
    add_frs_brma,
    assign_brma_by_cell,
    collapse_benunit_brma_to_household,
    load_brma_count_resource,
)
from microcosm.build.uk_runtime.geographic_clones import ladder_clone_index_column
from microcosm.build.uk_runtime.household_lineage import uk_household_clone_paths
from microcosm.build.uk_runtime.local_authority_input import (
    local_authority_engine_key_by_code,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    uk_time_period,
)
from microcosm.build.uk_runtime.receipt_counts import rounded_count
from microcosm.build.uk_runtime.rfdr_area_layer import (
    RFDR_COLUMN,
    load_rfdr_area_layer,
)
from microcosm.build.uk_runtime.spi_band_donors import (
    SPI_INCOME_BAND_DONOR_CLONE_INDEX,
)
from microcosm.build.uk_runtime.spine_geography import (
    _COPY_KINDS,
    _NORTHERN_IRELAND_REGION,
    AREA_BRMA_STAGE_NAME,
    ATOMIC_AREA_ASSIGN_STAGE_NAME,
    BRMA_COUNT_ARTIFACT_ROLE,
    BRMA_LAYER_ARTIFACT_ROLE,
    BRMA_LAYER_RESOURCE,
    LOCAL_AUTHORITY_ARTIFACT_ROLE,
    LOCAL_AUTHORITY_STAGE_NAME,
    RFDR_ARTIFACT_ROLE,
    RFDR_LAYER_RESOURCE,
    RFDR_STAGE_NAME,
    SPI_COPY_AREAS_STAGE_NAME,
    UK_AREA_BRMA_OUTPUT_COLUMNS,
    UK_ATOMIC_AREA_ASSIGN_OUTPUT_COLUMNS,
    UK_LOCAL_AUTHORITY_OUTPUT_COLUMNS,
    UK_RFDR_OUTPUT_COLUMNS,
    UK_SPI_COPY_AREAS_REWRITE_COLUMNS,
    UKAreaBRMAStageTransform,
    UKAtomicAreaAssignStageTransform,
    UKLocalAuthorityStageTransform,
    UKRFDRStageTransform,
    UKSPICopyAreasStageTransform,
    area_brma_operation_parameters,
    assign_uk_spi_copy_areas,
    atomic_area_assign_operation_parameters,
    draw_uk_spine_areas,
    flag_uk_spine_rfdr,
    load_committed_rfdr_layer,
    local_authority_operation_parameters,
    rfdr_operation_parameters,
    spi_copy_areas_operation_parameters,
    uk_spi_copy_keys,
)
from microcosm.frame import Frame, MassChangeRecord
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine
from test_support.microcosm_build.uk_atomic_support_fixtures import toy_support_payloads
from test_support.microcosm_build.uk_conditioned_support_fixtures import (
    EW,
    NI,
    SCOT,
    TOY_AREAS,
    TOY_FLAGGED_AREAS,
    TOY_HOUSEHOLDS,
    ToyHousehold,
    committed_rfdr_areas,
    toy_conditioned_support_payloads,
    toy_conditioned_supports,
    toy_spine_frame,
)

_STRING = pd.StringDtype(storage="python")
_CLONE = ladder_clone_index_column("household")
_SYSTEM_OF_REGION = {"NORTHERN_IRELAND": NI, "SCOTLAND": SCOT}
#: The toy households the committed RFDR layer flags: drawn into the Hawes
#: and the Highlands-and-Islands toy areas, the only areas with mass in their
#: cells.
_FLAGGED = [
    position
    for position, case in enumerate(TOY_HOUSEHOLDS)
    if (case.region, case.imd_decile) in {("YORKSHIRE", 6), ("SCOTLAND", 4)}
]


def _stage(name: str, **overrides: object) -> SourceStageSpec:
    """A stage built from the reviewed declaration spine_geography states,
    with ``overrides`` replacing fields."""

    brma_artifacts = (
        (BRMA_LAYER_ARTIFACT_ROLE, BRMA_LAYER_RESOURCE),
        (BRMA_COUNT_ARTIFACT_ROLE, "brma_private_rented_households.json"),
    )
    reviewed = {
        ATOMIC_AREA_ASSIGN_STAGE_NAME: (
            atomic_area_assign_operation_parameters(),
            UK_ATOMIC_AREA_ASSIGN_OUTPUT_COLUMNS,
            (),
            (),
        ),
        LOCAL_AUTHORITY_STAGE_NAME: (
            local_authority_operation_parameters(),
            UK_LOCAL_AUTHORITY_OUTPUT_COLUMNS,
            ((LOCAL_AUTHORITY_ARTIFACT_ROLE, "local_authority_names.json"),),
            (),
        ),
        AREA_BRMA_STAGE_NAME: (
            area_brma_operation_parameters(),
            UK_AREA_BRMA_OUTPUT_COLUMNS,
            brma_artifacts,
            (),
        ),
        RFDR_STAGE_NAME: (
            rfdr_operation_parameters(),
            UK_RFDR_OUTPUT_COLUMNS,
            ((RFDR_ARTIFACT_ROLE, RFDR_LAYER_RESOURCE),),
            (),
        ),
        SPI_COPY_AREAS_STAGE_NAME: (
            spi_copy_areas_operation_parameters(),
            (),
            (
                ("frs_release", "frs_release.json"),
                (LOCAL_AUTHORITY_ARTIFACT_ROLE, "local_authority_names.json"),
                *brma_artifacts,
                (RFDR_ARTIFACT_ROLE, RFDR_LAYER_RESOURCE),
            ),
            UK_SPI_COPY_AREAS_REWRITE_COLUMNS,
        ),
    }
    operations, outputs, artifacts, rewrites = reviewed[name]
    stage = SourceStageSpec(
        stage=name,
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
            for role, resource in artifacts
        ),
        operations=tuple(
            SourceOperationSpec(kind=kind, parameters=dict(parameters))
            for kind, parameters in operations
        ),
        outputs=tuple(outputs),
        rewrites=tuple(rewrites),
    )
    return replace(stage, **overrides)


@pytest.fixture(scope="module")
def payloads() -> dict[str, bytes]:
    return toy_conditioned_support_payloads()


@pytest.fixture(scope="module")
def supports() -> dict:
    return toy_conditioned_supports()


@pytest.fixture(scope="module")
def definition(payloads) -> dict:
    return uk_conditioned_assignment_definition(
        payloads, seed=UK_SPINE_ATOMIC_AREA_SEED
    )


def _g2(payloads) -> UKAtomicAreaAssignStageTransform:
    return UKAtomicAreaAssignStageTransform(
        stage=_stage(ATOMIC_AREA_ASSIGN_STAGE_NAME), supports=payloads
    )


def _g3() -> UKLocalAuthorityStageTransform:
    return UKLocalAuthorityStageTransform(stage=_stage(LOCAL_AUTHORITY_STAGE_NAME))


def _g5(payloads) -> UKRFDRStageTransform:
    return UKRFDRStageTransform(stage=_stage(RFDR_STAGE_NAME), supports=payloads)


def _run(payloads, clones: int, households=TOY_HOUSEHOLDS) -> dict[str, object]:
    """The three stages in order on ``clones`` clones of ``households``."""

    g2, g3, g5 = _g2(payloads), _g3(), _g5(payloads)
    frames = {"keyed": toy_spine_frame(clones=clones, households=households)}
    frames["g2"] = g2(frames["keyed"])
    frames["g3"] = g3(frames["g2"])
    frames["g5"] = g5(frames["g3"])
    return {
        "frames": frames,
        "evidence": {
            stage: transform.checkpoint_metadata()["evidence"]
            for stage, transform in (("g2", g2), ("g3", g3), ("g5", g5))
        },
    }


@pytest.fixture(scope="module")
def runs(payloads) -> dict[int, dict[str, object]]:
    return {clones: _run(payloads, clones) for clones in (1, 2, 3)}


def _household(runs, clones: int, stage: str = "g5") -> pd.DataFrame:
    return runs[clones]["frames"][stage].table("household")


def _clone_zero(household: pd.DataFrame) -> pd.DataFrame:
    return household.loc[household[_CLONE] == 0].reset_index(drop=True)


# --- The toy supports ---------------------------------------------------------------


def _committed(name: str) -> dict:
    return json.loads(files("microcosm.build.uk").joinpath(name).read_text("utf-8"))


def test_the_toy_supports_resolve_against_the_committed_resources() -> None:
    """Every code G3, G4, G5, G7 and G8 read resolves in its committed resource."""

    roster = local_authority_engine_key_by_code()
    layer = _committed("brma_area_layer.json")["brmas"]
    rents = set(_committed("brma_rent_quantiles.json")["brmas"])
    knots = set(
        _committed("high_value_home_value_knots.json")["levels"]["constituency"]
    )
    flagged = committed_rfdr_areas()
    brma_of = {
        grain: {
            code: (region, brma)
            for region, by_brma in regions.items()
            for brma, codes in by_brma.items()
            for code in codes
        }
        for grain, regions in layer.items()
    }
    for area in TOY_AREAS:
        assert area.local_authority_code in roster
        if area.system == NI:
            continue
        key, grain = (
            (area.lsoa_code, "ew_lsoa21")
            if area.system == EW
            else (area.oa_code, "scotland_oa22")
        )
        region, brma = brma_of[grain][key]
        assert region == area.region
        assert brma in rents
        if area.system == EW:
            assert area.constituency_code in knots
        assert (area.oa_code in flagged[area.system]) is (
            area.oa_code == TOY_FLAGGED_AREAS[area.system]
        )
    # Every system is exercised, Northern Ireland's districts on the roster.
    assert {area.system for area in TOY_AREAS} == {EW, SCOT, NI}


def test_each_great_britain_support_holds_every_flagged_output_area(supports) -> None:
    for system, codes in committed_rfdr_areas().items():
        areas = set(supports[system].arrays["area"].tolist())
        assert set(codes) <= areas
        # The fillers carry no households, so the draw never meets them.
        filler = np.isin(
            supports[system].arrays["area"],
            [code for code in codes if code != TOY_FLAGGED_AREAS[system]],
        )
        assert not np.asarray(supports[system].arrays["households"])[filler].any()
    small = toy_conditioned_supports(rfdr_roster=False)
    assert len(small[SCOT].arrays["area"]) == sum(a.system == SCOT for a in TOY_AREAS)


# --- The reviewed declarations ------------------------------------------------------


def test_the_reviewed_operations_are_declarative() -> None:
    for operations in (
        atomic_area_assign_operation_parameters(),
        local_authority_operation_parameters(),
        rfdr_operation_parameters(),
    ):
        for kind, parameters in operations:
            _reject_executable_parameter_keys(parameters, context=kind)
            _reject_incumbent_dependencies(parameters, context=kind)
            assert json.loads(json.dumps(parameters)) == parameters
    ((_, g2),) = atomic_area_assign_operation_parameters()
    assert g2["seed"] == UK_SPINE_ATOMIC_AREA_SEED == 0
    assert g2["layers"] == list(UK_SPINE_LADDER_COLUMNS)
    assert [level["name"] for level in g2["fallback_levels"]] == [
        "L0",
        "L1",
        "L2",
        "L3",
    ]
    ((_, g5),) = rfdr_operation_parameters()
    assert g5["require_flagged"] is False


@pytest.mark.parametrize(
    ("name", "build"),
    [
        (ATOMIC_AREA_ASSIGN_STAGE_NAME, lambda stage, p: _g2_with(stage, p)),
        (
            LOCAL_AUTHORITY_STAGE_NAME,
            lambda stage, p: UKLocalAuthorityStageTransform(stage=stage),
        ),
        (
            RFDR_STAGE_NAME,
            lambda stage, p: UKRFDRStageTransform(stage=stage, supports=p),
        ),
    ],
)
def test_each_transform_runs_only_as_its_reviewed_stage(name, build, payloads) -> None:
    stage = _stage(name)
    build(stage, payloads)
    operation = stage.operations[0]
    drifted = SourceOperationSpec(
        kind=operation.kind, parameters={**operation.parameters, "extra": 1}
    )
    for overrides, message in (
        ({"stage": "frs_brma"}, "Expected stage"),
        ({"grain": "benunit"}, "grain must be 'household'"),
        ({"outputs": stage.outputs[:-1]}, "writes exactly its own columns"),
        ({"rewrites": stage.outputs[:1]}, "writes exactly its own columns"),
        ({"operations": (drifted,)}, "drifted"),
        ({"operations": ()}, "operation order drifted"),
        (
            {
                "artifacts": (
                    *stage.artifacts,
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
            build(_stage(name, **overrides), payloads)
    with pytest.raises(TypeError, match="SourceStageSpec"):
        build(object(), payloads)


def _g2_with(stage, payloads):
    return UKAtomicAreaAssignStageTransform(stage=stage, supports=payloads)


def test_the_artifact_must_require_its_runtime_digest(payloads) -> None:
    stage = _stage(RFDR_STAGE_NAME)
    (artifact,) = stage.artifacts
    loose = replace(stage, artifacts=({**artifact, "runtime_sha256_required": False},))
    with pytest.raises(ValueError, match="runtime_sha256_required"):
        UKRFDRStageTransform(stage=loose, supports=payloads)


def test_checkpoint_metadata_needs_a_completed_run(payloads) -> None:
    for transform in (_g2(payloads), _g3(), _g5(payloads)):
        with pytest.raises(RuntimeError, match="completed stage run"):
            transform.checkpoint_metadata()


# --- G2 -----------------------------------------------------------------------------


def test_g2_writes_the_assignment_and_ladder_columns_as_python_strings(
    runs, supports
) -> None:
    household = _household(runs, 3, "g2")
    assert UK_ATOMIC_AREA_ASSIGN_OUTPUT_COLUMNS == UK_SPINE_AREA_COLUMNS
    for column in UK_ATOMIC_AREA_ASSIGN_OUTPUT_COLUMNS:
        assert household[column].dtype == _STRING, column
        assert household[column].notna().all()
    assert set(household["atomic_area_basis"]) == {"assigned"}
    # Every ladder column is the support's own row for the drawn area.
    for _, row in household.iterrows():
        support = supports[row["atomic_area_system"]]
        position = int(
            np.flatnonzero(support.arrays["area"] == row["atomic_area_code"])[0]
        )
        assert row["oa_code"] == row["atomic_area_code"]
        for column in UK_SPINE_LADDER_COLUMNS[1:]:
            assert row[column] == support.arrays[column][position], column


def test_g2_keeps_every_row_id_weight_and_earlier_column(runs) -> None:
    keyed = runs[3]["frames"]["keyed"]
    placed = runs[3]["frames"]["g2"]
    for entity in keyed.entities:
        before, after = keyed.table(entity), placed.table(entity)
        assert after[before.columns].equals(before)
    assert np.array_equal(
        keyed.weights_for("household").values, placed.weights_for("household").values
    )
    assert placed.mass_log == keyed.mass_log
    assert list(placed.table("household").columns) == [
        *keyed.table("household").columns,
        *UK_ATOMIC_AREA_ASSIGN_OUTPUT_COLUMNS,
    ]


def test_every_household_stays_in_its_region_at_its_level(runs, supports) -> None:
    household = _household(runs, 3, "g2")
    levels = np.tile([case.level for case in TOY_HOUSEHOLDS], 3)
    cases = TOY_HOUSEHOLDS * 3
    for (_, row), case, level in zip(household.iterrows(), cases, levels, strict=True):
        system = _SYSTEM_OF_REGION.get(case.region, EW)
        assert row["atomic_area_system"] == system
        support = supports[system]
        position = int(
            np.flatnonzero(support.arrays["area"] == row["atomic_area_code"])[0]
        )
        assert support.arrays["frs_region"][position] == case.region
        if level in {"L0", "L1"}:
            assert support.arrays[IMD_DECILE_COLUMN][position] == str(case.imd_decile)
    # London's band-H household in decile 1 falls to L1: its area is a
    # decile-1 area, none of which holds band H.
    band_h = household.iloc[1]
    support = supports[EW]
    position = int(
        np.flatnonzero(support.arrays["area"] == band_h["atomic_area_code"])[0]
    )
    assert support.arrays[IMD_DECILE_COLUMN][position] == "1"
    assert int(support.arrays["households_band_h"][position]) == 0


def test_g2_evidence_counts_sources_by_system_level_and_reason(runs) -> None:
    evidence = runs[1]["evidence"]["g2"]
    by_level: dict[str, int] = {}
    for case in TOY_HOUSEHOLDS:
        by_level[case.level] = by_level.get(case.level, 0) + 1
    assert evidence["source_households"] == rounded_count(len(TOY_HOUSEHOLDS))
    assert evidence["source_households_by_level"] == {
        level: rounded_count(by_level.get(level, 0))
        for level in ("L0", "L1", "L2", "L3")
    }
    systems = [_SYSTEM_OF_REGION.get(case.region, EW) for case in TOY_HOUSEHOLDS]
    assert evidence["source_households_by_system"] == {
        system: rounded_count(systems.count(system)) for system in (EW, SCOT, NI)
    }
    assert set(evidence["source_households_by_fallback_reason"]) == {"decile", "band"}
    assert evidence["validation"] == {
        "outcome": "pass",
        "scope": "atomic_geography_mapping_integrity",
    }
    assert evidence["seed"] == 0
    assert evidence["stream"][1] == "uk-spine-atomic-area-v1"
    # No exact count: the validation's row count is replaced by sources.
    assert "households" not in evidence["validation"]


def test_clone_zero_draws_at_k_1_2_and_3_what_k_1_draws(runs) -> None:
    columns = [IDENTITY_COLUMN, *UK_ATOMIC_AREA_ASSIGN_OUTPUT_COLUMNS]
    single = _clone_zero(_household(runs, 1))[columns]
    for clones in (2, 3):
        assert _clone_zero(_household(runs, clones))[columns].equals(single)
    # Each clone draws its own key, so the areas vary across clones.
    household = _household(runs, 3)
    assert household[IDENTITY_COLUMN].is_unique
    assert (
        household.groupby("source_household_id")["atomic_area_code"].nunique() > 1
    ).any()


def test_the_draw_does_not_depend_on_row_order(runs, definition, supports) -> None:
    household = runs[3]["frames"]["keyed"].table("household")
    shuffled = household.sample(frac=1.0, random_state=931)
    ordered = draw_uk_spine_areas(household, definition=definition, supports=supports)
    permuted = draw_uk_spine_areas(shuffled, definition=definition, supports=supports)
    assert permuted.columns.index.equals(shuffled.index)
    assert permuted.columns.loc[household.index].equals(ordered.columns)
    assert np.array_equal(
        pd.Series(permuted.levels, index=shuffled.index)
        .loc[household.index]
        .to_numpy(),
        ordered.levels,
    )


def test_validation_passes_on_the_prepared_frame_and_fails_on_a_raw_decile(
    runs, definition, supports
) -> None:
    household = runs[1]["frames"]["keyed"].table("household")
    inputs = uk_draw_inputs(household)
    levels = uk_draw_levels(inputs, definition, supports)
    prepared = uk_draw_frame(inputs, levels)
    placed = pd.concat(
        [prepared, assign_atomic(prepared, definition, supports)], axis=1
    )
    frame = pd.concat([placed, derive_geography(placed, definition, supports)], axis=1)
    assert validate_geography(frame, definition, supports)["outcome"] == "pass"
    # London's decile-9 household falls to L2: no area holds its decile, so
    # its area violates the raw decile the prepared frame dropped.
    (fallback,) = [
        position
        for position, case in enumerate(TOY_HOUSEHOLDS)
        if (case.region, case.imd_decile) == ("LONDON", 9)
    ]
    assert levels[fallback] == "L2"
    raw = frame.copy()
    raw.iloc[fallback, raw.columns.get_loc(DECILE_INPUT_COLUMN)] = inputs[
        DECILE_INPUT_COLUMN
    ].iloc[fallback]
    with pytest.raises(ValueError, match="violates observed geography") as refused:
        validate_geography(raw, definition, supports)
    assert household[IDENTITY_COLUMN].iloc[fallback] not in str(refused.value)


def _corrupt_first_msoa(monkeypatch) -> None:
    """Make the area draw write, for its first row, another row's MSOA: the
    shared validation, which re-derives the ladder itself, must refuse."""

    real = spine_geography.derive_geography

    def corrupt(households, spec, supports):
        derived = real(households, spec, supports)
        values = derived["msoa_code"].tolist()
        other = next(value for value in values if value != values[0])
        derived.iloc[0, derived.columns.get_loc("msoa_code")] = other
        return derived

    monkeypatch.setattr(spine_geography, "derive_geography", corrupt)


def test_g2_and_the_copy_stage_refuse_a_ladder_the_mapping_does_not_give(
    monkeypatch, payloads, copy_runs
) -> None:
    keyed = toy_spine_frame(clones=1)
    copied = copy_runs[1]["frames"]["copied"]
    g2 = _g2(payloads)
    engine = _RecordingEngine()
    copy_stage = _copy_stage(payloads, engine)
    _corrupt_first_msoa(monkeypatch)
    with pytest.raises(ValueError, match="derived geography differs") as refused:
        g2(keyed)
    assert g2.last_result is None
    assert not any(c.isdigit() for c in str(refused.value))
    with pytest.raises(ValueError, match="derived geography differs"):
        copy_stage(copied)
    assert copy_stage.last_result is None
    assert engine.calls == []


def test_g2_refuses_a_frame_that_already_carries_an_output(runs, payloads) -> None:
    placed = runs[1]["frames"]["g2"]
    with pytest.raises(ValueError, match="already carries") as refused:
        _g2(payloads)(placed)
    assert "atomic_area_code" in str(refused.value)


def test_g2_refuses_a_frame_without_the_geography_key(runs, payloads) -> None:
    keyed = runs[1]["frames"]["keyed"]
    tables = {entity: keyed.table(entity) for entity in keyed.entities}
    tables["household"] = tables["household"].drop(columns=[IDENTITY_COLUMN])
    bare = Frame(
        tables,
        keyed.schema,
        {"household": keyed.weights_for("household")},
        keyed.strata,
        mass_log=keyed.mass_log,
        metadata=keyed.metadata,
    )
    with pytest.raises(ValueError, match="lacks") as refused:
        _g2(payloads)(bare)
    assert IDENTITY_COLUMN in str(refused.value)


def test_g2_refuses_supports_that_are_not_conditioned() -> None:
    with pytest.raises(ValueError, match="conditioned support columns differ"):
        _g2(toy_support_payloads())


# --- G3 -----------------------------------------------------------------------------


def test_g3_resolves_each_drawn_code_to_its_engine_name(runs) -> None:
    household = _household(runs, 3, "g3")
    roster = local_authority_engine_key_by_code()
    assert household["local_authority"].dtype == _STRING
    assert household["local_authority"].tolist() == [
        roster[code] for code in household["local_authority_code"]
    ]
    assert list(household.columns)[-1] == "local_authority"


def test_g3_refuses_an_off_roster_code_naming_no_code(runs) -> None:
    placed = runs[1]["frames"]["g2"]
    tables = {entity: placed.table(entity) for entity in placed.entities}
    household = tables["household"].copy()
    invented = "E06999999"
    assert invented not in local_authority_engine_key_by_code()
    household["local_authority_code"] = household["local_authority_code"].mask(
        household.index == household.index[0], invented
    )
    tables["household"] = household
    frame = Frame(
        tables,
        placed.schema,
        {"household": placed.weights_for("household")},
        placed.strata,
        mass_log=placed.mass_log,
        metadata=placed.metadata,
    )
    with pytest.raises(ValueError, match="off the April 2023") as refused:
        _g3()(frame)
    assert invented not in str(refused.value)


def test_g3_refuses_a_frame_without_the_code_or_already_resolved(runs) -> None:
    with pytest.raises(ValueError, match="lacks"):
        _g3()(runs[1]["frames"]["keyed"])
    with pytest.raises(ValueError, match="already carries"):
        _g3()(runs[1]["frames"]["g3"])


# --- G5 -----------------------------------------------------------------------------


def test_g5_flags_the_households_drawn_into_flagged_areas(runs) -> None:
    flagged_areas = committed_rfdr_areas()
    for clones in (1, 3):
        household = _household(runs, clones)
        assert household[RFDR_COLUMN].dtype == np.dtype(bool)
        systems = household["atomic_area_system"].to_numpy(dtype=object)
        codes = household["atomic_area_code"].to_numpy(dtype=object)
        expected = [
            system in flagged_areas and code in flagged_areas[system]
            for system, code in zip(systems, codes, strict=True)
        ]
        assert household[RFDR_COLUMN].tolist() == expected
        # The households whose cells hold one area, a flagged one, are flagged
        # in every clone; Northern Ireland never is.
        for clone in range(clones):
            for position in _FLAGGED:
                assert household[RFDR_COLUMN].iloc[
                    position + len(TOY_HOUSEHOLDS) * clone
                ]
        assert not household.loc[systems == NI, RFDR_COLUMN].any()


def test_g5_refuses_a_layer_that_flags_an_area_off_its_support_roster(runs) -> None:
    small = toy_conditioned_supports(rfdr_roster=False)
    layer = load_rfdr_area_layer(
        files("microcosm.build.uk").joinpath(RFDR_LAYER_RESOURCE).read_bytes()
    )
    placed = runs[1]["frames"]["g3"]
    with pytest.raises(ValueError, match="does not hold") as refused:
        flag_uk_spine_rfdr(placed, layer=layer, layer_sha256="0" * 64, supports=small)
    text = str(refused.value)
    assert not any(
        code in text for codes in committed_rfdr_areas().values() for code in codes
    )
    with pytest.raises(ValueError, match="does not hold"):
        UKRFDRStageTransform(
            stage=_stage(RFDR_STAGE_NAME),
            supports=toy_conditioned_support_payloads(rfdr_roster=False),
        )(placed)


def test_g5_reports_a_system_with_no_flagged_household_without_refusing(
    payloads,
) -> None:
    unflagged = [
        case
        for case in TOY_HOUSEHOLDS
        if (case.region, case.imd_decile) != ("YORKSHIRE", 6)
    ]
    run = _run(payloads, 1, unflagged)
    household = run["frames"]["g5"].table("household")
    england_wales = household["atomic_area_system"] == EW
    assert england_wales.any() and not household.loc[england_wales, RFDR_COLUMN].any()
    evidence = run["evidence"]["g5"]
    assert evidence["no_flagged_household"] == {EW: True, SCOT: False}
    assert evidence["flagged_source_households"][EW] == 0
    assert evidence["require_flagged"] is False


def test_g5_still_refuses_a_malformed_drawn_code(runs, payloads) -> None:
    placed = runs[1]["frames"]["g3"]
    tables = {entity: placed.table(entity) for entity in placed.entities}
    household = tables["household"].copy()
    first = household.index[household["atomic_area_system"] == EW][0]
    malformed = household.loc[first, "atomic_area_code"].lower()
    household.loc[first, "atomic_area_code"] = malformed
    tables["household"] = household
    frame = Frame(
        tables,
        placed.schema,
        {"household": placed.weights_for("household")},
        placed.strata,
        mass_log=placed.mass_log,
        metadata=placed.metadata,
    )
    with pytest.raises(ValueError, match="well-formed") as refused:
        _g5(payloads)(frame)
    assert malformed not in str(refused.value)


def test_g5_evidence_counts_flagged_sources_among_clone_zero(runs) -> None:
    single = _clone_zero(_household(runs, 1))
    expected = {
        system: rounded_count(
            single.loc[
                single[RFDR_COLUMN] & (single["atomic_area_system"] == system),
                "source_household_id",
            ].nunique()
        )
        for system in (EW, SCOT)
    }
    for clones in (1, 3):
        evidence = runs[clones]["evidence"]["g5"]
        assert evidence["counted_among"] == "clone_0"
        assert evidence["flagged_source_households"] == expected
        assert evidence["no_flagged_household"] == {EW: False, SCOT: False}
        assert evidence["roster"] == "pass"
        assert evidence["require_flagged"] is False
        assert evidence["method"] == "uk-rfdr-area-layer-v1"


def test_g5_counts_flagged_sources_in_clone_zero_where_later_clones_differ(
    payloads,
) -> None:
    """Region-level draws land in a flagged area in some clones and not in
    others, so a union over clones grows with K; the evidence counts clone 0."""

    cases = [ToyHousehold("YORKSHIRE", 0, "D", False, "L3")] * 12 + [
        ToyHousehold("SCOTLAND", 0, "D", False, "L3")
    ] * 12
    g2, g3, g5 = _g2(payloads), _g3(), _g5(payloads)
    g5(g3(g2(toy_spine_frame(clones=3, households=cases))))
    household = g5.last_result.frame.table("household")
    flagged = household[RFDR_COLUMN].to_numpy()
    zero = (household[_CLONE] == 0).to_numpy()
    systems = household["atomic_area_system"].to_numpy(dtype=object)
    sources = household["source_household_id"].to_numpy()

    def distinct(rows) -> int:
        return len(set(sources[rows].tolist()))

    in_clone_zero = {s: distinct(zero & flagged & (systems == s)) for s in (EW, SCOT)}
    in_every_clone = {s: distinct(flagged & (systems == s)) for s in (EW, SCOT)}
    # The invented frame tells the two counts apart.
    assert in_clone_zero != in_every_clone
    assert dict(g5.last_result.flagged_by_system) == in_clone_zero
    assert dict(g5.last_result.no_flagged_household) == {
        system: count == 0 for system, count in in_clone_zero.items()
    }


# --- G4 and the SPI-copy area stage: stand-ins --------------------------------------

#: The LHA categories the recording engine hands out: the count table's five.
_LHA_CATEGORIES = ("A", "B", "C", "D", "E")
_SUPPORT_CHANNEL = "household_support_channel"
_SUPPORT_ORDINAL = "household_support_clone_index"


class _RecordingEngine:
    """A stand-in for the UK rules engine that keeps every call.

    Each benefit unit's ``LHA_category`` is invented from its own id, modulo
    five; a copy's ids add whole multiples of the copy-id table's radices, so
    a copy takes its parent's category, as a whole-row copy takes it from
    the engine. ``misaligned`` answers one benefit unit short.
    """

    def __init__(self, *, misaligned: bool = False, country: str = "uk") -> None:
        self.country = country
        self.misaligned = misaligned
        self.calls: list[tuple[Frame, tuple[str, ...], str]] = []

    def materialize(self, frame, variables, period):
        self.calls.append((frame, tuple(variables), str(period)))
        ids = frame.table("benunit")["benunit_id"].to_numpy(dtype=np.int64)
        values = np.asarray([_LHA_CATEGORIES[int(i) % 5] for i in ids], dtype=object)
        return {"LHA_category": values[:-1] if self.misaligned else values}


def _lha(frame: Frame) -> np.ndarray:
    return _RecordingEngine().materialize(
        frame, ("LHA_category",), uk_time_period(frame)
    )["LHA_category"]


def _g4(engine: object) -> UKAreaBRMAStageTransform:
    return UKAreaBRMAStageTransform(stage=_stage(AREA_BRMA_STAGE_NAME), engine=engine)


def _copy_stage(payloads, engine: object) -> UKSPICopyAreasStageTransform:
    return UKSPICopyAreasStageTransform(
        stage=_stage(SPI_COPY_AREAS_STAGE_NAME), engine=engine, supports=payloads
    )


def _replace_household(frame: Frame, household: pd.DataFrame) -> Frame:
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["household"] = household
    return Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _copy_picks(source: int, clone: int) -> tuple[int, ...]:
    """The invented copies spi_support_channel and spi_income_band_donors
    make of a household, by ordinal.

    In clone 0 every household is copied (1) and every third one is a band
    donor too (2), by its source alone, so clone 0's copies are the same at
    every clone count; a later clone copies other households, so a union of
    copied sources over clones grows with K.
    """

    if clone == 0:
        return (1, 2) if source % 3 == 0 else (1,)
    return tuple(
        ordinal
        for ordinal, picked in (
            (1, (source + clone) % 2 == 0),
            (2, (source + clone) % 3 == 0),
        )
        if picked
    )


def _with_support_channel(frame: Frame, picks=_copy_picks) -> Frame:
    """``frame`` as spi_support_channel and spi_income_band_donors leave it,
    for these tests alone.

    Every row is on the support channel's FRS side at ordinal 0; each
    household ``picks`` names is copied whole at the channel's ordinals,
    every id minted in the copy-id table's ``spi_support_channel`` field, each
    copy at half its parent's invented weight, with a mass record for the
    copies. No other lineage column is added.
    """

    person, benunit = frame.table("person"), frame.table("benunit")
    household = frame.table("household").copy()
    household[_SUPPORT_CHANNEL] = "frs"
    household[_SUPPORT_ORDINAL] = np.zeros(len(household), dtype=np.int64)
    weights = frame.weights_for("household").values
    persons, benunits, households = [person], [benunit], [household]
    masses = [weights]
    for ordinal in (1, 2):
        positions = [
            position
            for position, (source, clone) in enumerate(
                zip(
                    household["source_household_id"].tolist(),
                    household[_CLONE].tolist(),
                    strict=True,
                )
            )
            if ordinal in picks(int(source), int(clone))
        ]
        if not positions:
            continue
        copied = household.iloc[positions].copy()
        members = person.loc[person["person_household_id"].isin(copied["household_id"])]
        members = members.copy()
        units = benunit.loc[benunit["benunit_id"].isin(members["person_benunit_id"])]
        units = units.copy()
        for column in ("person_id", "person_household_id", "person_benunit_id"):
            members[column] = copy_id_table.mint(
                members[column], "spi_support_channel", ordinal
            )
        units["benunit_id"] = copy_id_table.mint(
            units["benunit_id"], "spi_support_channel", ordinal
        )
        copied["household_id"] = copy_id_table.mint(
            copied["household_id"], "spi_support_channel", ordinal
        )
        copied[_SUPPORT_CHANNEL] = "spi"
        copied[_SUPPORT_ORDINAL] = np.int64(ordinal)
        persons.append(members)
        benunits.append(units)
        households.append(copied)
        masses.append(weights[positions] * 0.5)
    values = np.concatenate(masses)
    record = MassChangeRecord(
        entity="household",
        old_total=float(weights.sum()),
        new_total=float(values.sum()),
        declared_factor=None,
        reason="invented copies standing in for the SPI support channel",
    )
    return uk_national_frame(
        person=pd.concat(persons, ignore_index=True),
        benunit=pd.concat(benunits, ignore_index=True),
        household=pd.concat(households, ignore_index=True),
        time_period=uk_time_period(frame),
        household_weights=values,
        mass_log=(*frame.mass_log, record),
    )


def _copy_run(payloads, base: Frame, picks=_copy_picks) -> dict[str, object]:
    """G4 on ``base`` (G5's frame), the stand-in for the support channel's
    two stages, then the copy stage, each with an engine of its own."""

    engines = {"g4": _RecordingEngine(), "copy": _RecordingEngine()}
    g4, copy_stage = _g4(engines["g4"]), _copy_stage(payloads, engines["copy"])
    frames = {"g4": g4(base)}
    frames["copied"] = _with_support_channel(frames["g4"], picks)
    frames["copy"] = copy_stage(frames["copied"])
    return {
        "frames": frames,
        "engines": engines,
        "results": {"g4": g4.last_result, "copy": copy_stage.last_result},
        "evidence": {
            "g4": g4.checkpoint_metadata()["evidence"],
            "copy": copy_stage.checkpoint_metadata()["evidence"],
        },
    }


@pytest.fixture(scope="module")
def copy_runs(payloads, runs) -> dict[int, dict[str, object]]:
    return {
        clones: _copy_run(payloads, runs[clones]["frames"]["g5"])
        for clones in (1, 2, 3)
    }


def _copies(household: pd.DataFrame) -> np.ndarray:
    return (household[_SUPPORT_CHANNEL] == "spi").to_numpy(dtype=bool)


def _committed_brma_layer():
    return load_brma_area_layer(
        files("microcosm.build.uk").joinpath(BRMA_LAYER_RESOURCE).read_bytes()
    )


# --- G4 and the copy stage: their reviewed declarations -----------------------------


def test_the_g4_and_copy_stage_operations_are_declarative_and_agree() -> None:
    for operations in (
        area_brma_operation_parameters(),
        spi_copy_areas_operation_parameters(),
    ):
        for kind, parameters in operations:
            _reject_executable_parameter_keys(parameters, context=kind)
            _reject_incumbent_dependencies(parameters, context=kind)
            assert json.loads(json.dumps(parameters)) == parameters
    predictors = (
        "materialize_rules_engine_predictors",
        {"predictors": ["LHA_category"], "consumed_only": True},
    )
    (g4_predictors, (_, g4)) = area_brma_operation_parameters()
    (copy_predictors, (_, copy)) = spi_copy_areas_operation_parameters()
    assert g4_predictors == copy_predictors == predictors
    # frs_brma's seed and salts, which the Northern Ireland draws keep.
    assert (g4["output"], g4["seed"]) == ("brma", 0)
    assert (
        g4["salts"]
        == {
            "benunit": UK_BRMA_BENUNIT_SALT,
            "household_pick": UK_BRMA_HOUSEHOLD_PICK_SALT,
        }
        == {"benunit": "brma", "household_pick": "brma:household_pick"}
    )
    # The copy stage names G2's definition with G2's stream and seed, and
    # G4's count-table draw.
    ((_, g2),) = atomic_area_assign_operation_parameters()
    assert copy["area_draw"] == {
        "definition_of": ATOMIC_AREA_ASSIGN_STAGE_NAME,
        "level": g2["fallback_levels"][-1]["name"],
        "stream": g2["stream"],
        "seed": g2["seed"],
    }
    assert copy["area_draw"]["level"] == "L3"
    shared = set(copy["brma"]) & set(g4)
    assert shared >= {"seed", "salts", "count_resource", "cell_columns"}
    assert {key: copy["brma"][key] for key in shared} == {
        key: g4[key] for key in shared
    }
    # The engine reads Northern Ireland's copies alone, in one simulation over
    # every clone's copies (C-9).
    assert (copy["brma"]["engine_households"], copy["brma"]["engine_simulations"]) == (
        "northern_ireland_copies",
        "one",
    )
    assert copy["rewrites"] == list(UK_SPI_COPY_AREAS_REWRITE_COLUMNS)


def test_the_copy_kinds_and_northern_irelands_region_match_their_owners() -> None:
    """The ordinals and the region the stages spell for themselves."""

    bound = copy_id_table.ordinal_bound("spi_support_channel")
    assert set(_COPY_KINDS) == set(range(1, bound))
    assert _COPY_KINDS[SPI_INCOME_BAND_DONOR_CLONE_INDEX] == "income_band_donor"
    assert _profile(NI)[2] == (_NORTHERN_IRELAND_REGION,)


@pytest.mark.parametrize("name", [AREA_BRMA_STAGE_NAME, SPI_COPY_AREAS_STAGE_NAME])
def test_g4_and_the_copy_stage_run_only_as_their_reviewed_stages(
    name, payloads
) -> None:
    def build(stage):
        if name == AREA_BRMA_STAGE_NAME:
            return UKAreaBRMAStageTransform(stage=stage, engine=_RecordingEngine())
        return UKSPICopyAreasStageTransform(
            stage=stage, engine=_RecordingEngine(), supports=payloads
        )

    stage = _stage(name)
    build(stage)
    *head, last = stage.operations
    drifted = SourceOperationSpec(
        kind=last.kind, parameters={**last.parameters, "extra": 1}
    )
    loose = {**stage.artifacts[-1], "runtime_sha256_required": False}
    for overrides, message in (
        ({"stage": "frs_brma"}, "Expected stage"),
        ({"grain": "benunit"}, "grain must be 'household'"),
        ({"outputs": ("brma_extra",)}, "writes exactly its own columns"),
        ({"rewrites": ("brma_extra",)}, "writes exactly its own columns"),
        ({"operations": (*head, drifted)}, "drifted"),
        ({"operations": tuple(head)}, "operation order drifted"),
        ({"artifacts": stage.artifacts[:-1]}, "and no other"),
        ({"artifacts": (*stage.artifacts[:-1], loose)}, "runtime_sha256_required"),
    ):
        with pytest.raises(ValueError, match=message):
            build(_stage(name, **overrides))
    with pytest.raises(TypeError, match="SourceStageSpec"):
        build(object())


def test_g4_and_the_copy_stage_need_a_run_and_the_uk_engine(payloads, runs) -> None:
    for transform in (
        _g4(_RecordingEngine()),
        _copy_stage(payloads, _RecordingEngine()),
    ):
        with pytest.raises(RuntimeError, match="completed stage run"):
            transform.checkpoint_metadata()
    with pytest.raises(ValueError, match="does not match"):
        _g4(_RecordingEngine(country="us"))(runs[1]["frames"]["g5"])


# --- G4 -----------------------------------------------------------------------------


def test_g4_reads_great_britain_brmas_off_the_committed_layer(copy_runs) -> None:
    layer = _committed_brma_layer()
    for clones in (1, 3):
        household = copy_runs[clones]["frames"]["g4"].table("household")
        assert household["brma"].dtype == np.dtype(object)
        for grain, spec in BRMA_GRAINS.items():
            rows = (household["atomic_area_system"] == spec.system).to_numpy()
            assert rows.any()
            assert household.loc[rows, "brma"].tolist() == [
                layer.brma_by_area[grain][code]
                for code in household.loc[rows, spec.key]
            ]


def test_g4_draws_northern_ireland_as_frs_brma_does(runs, copy_runs) -> None:
    """Northern Ireland's benefit units draw with frs_brma's keys, so each
    of its households takes the BRMA frs_brma gives it, at every K."""

    for clones in (1, 3):
        base = runs[clones]["frames"]["g5"]
        reference = add_frs_brma(
            base, lha_category=_lha(base), count_resource=load_brma_count_resource()
        ).table("household")
        household = copy_runs[clones]["frames"]["g4"].table("household")
        northern_ireland = (household["atomic_area_system"] == NI).to_numpy()
        assert northern_ireland.sum() == 4 * clones
        assert (
            household["brma"].to_numpy()[northern_ireland].tolist()
            == reference["brma"].to_numpy()[northern_ireland].tolist()
        )


def test_g4_makes_one_engine_call_over_the_whole_frame(runs, copy_runs) -> None:
    for clones in (1, 3):
        (call,) = copy_runs[clones]["engines"]["g4"].calls
        frame, variables, period = call
        base = runs[clones]["frames"]["g5"]
        assert variables == ("LHA_category",)
        assert period == uk_time_period(base)
        assert frame is base


def test_g4_keeps_every_row_and_appends_brma(runs, copy_runs) -> None:
    base, placed = runs[3]["frames"]["g5"], copy_runs[3]["frames"]["g4"]
    for entity in base.entities:
        before, after = base.table(entity), placed.table(entity)
        assert after[before.columns].equals(before)
    assert list(placed.table("household").columns) == [
        *base.table("household").columns,
        "brma",
    ]
    assert np.array_equal(
        base.weights_for("household").values, placed.weights_for("household").values
    )
    assert placed.mass_log == base.mass_log


def test_g4_clone_zero_takes_at_k_3_the_brmas_of_k_1(copy_runs) -> None:
    single, triple = (
        _clone_zero(copy_runs[clones]["frames"]["g4"].table("household"))["brma"]
        for clones in (1, 3)
    )
    assert triple.tolist() == single.tolist()


def _g4_refusal(frame: Frame, engine: object | None = None) -> str:
    with pytest.raises(ValueError) as refused:
        _g4(engine if engine is not None else _RecordingEngine())(frame)
    return str(refused.value)


def test_g4_refuses_an_area_off_the_layer_naming_no_code(runs) -> None:
    base = runs[1]["frames"]["g5"]
    household = base.table("household").copy()
    first = household.index[household["atomic_area_system"] == EW][0]
    invented = "E01999999"
    assert invented not in _committed_brma_layer().brma_by_area["ew_lsoa21"]
    household.loc[first, "lsoa_code"] = invented
    engine = _RecordingEngine()
    text = _g4_refusal(_replace_household(base, household), engine)
    assert "not on the layer" in text
    assert invented not in text
    # Refused before the engine call.
    assert engine.calls == []


def test_g4_refuses_an_area_the_layer_places_in_another_region(runs) -> None:
    base = runs[1]["frames"]["g5"]
    household = base.table("household").copy()
    first = household.index[household["region"] == "LONDON"][0]
    household.loc[first, "region"] = "SOUTH_EAST"
    engine = _RecordingEngine()
    text = _g4_refusal(_replace_household(base, household), engine)
    assert "another FRS region" in text
    assert household.loc[first, "lsoa_code"] not in text
    # Refused by G4's own check, before the engine call.
    assert engine.calls == []


def test_g4_refuses_a_system_or_region_it_cannot_place(runs) -> None:
    base = runs[1]["frames"]["g5"]
    household = base.table("household")
    unknown = household.copy()
    unknown["atomic_area_system"] = unknown["atomic_area_system"].mask(
        unknown.index == unknown.index[0], "uk_invented_system"
    )
    engine = _RecordingEngine()
    text = _g4_refusal(_replace_household(base, unknown), engine)
    assert "holds another value" in text and "uk_invented_system" not in text
    moved = household.copy()
    first = moved.index[moved["atomic_area_system"] == NI][0]
    moved.loc[first, "region"] = "WALES"
    assert "disagree" in _g4_refusal(_replace_household(base, moved), engine)
    assert engine.calls == []


def test_g4_refuses_misaligned_engine_values_and_a_written_or_missing_column(
    runs, copy_runs
) -> None:
    base = runs[1]["frames"]["g5"]
    assert "not aligned" in _g4_refusal(base, _RecordingEngine(misaligned=True))
    # A frame G4 cannot place is refused before the engine call.
    engine = _RecordingEngine()
    assert "already carries" in _g4_refusal(copy_runs[1]["frames"]["g4"], engine)
    household = base.table("household").drop(columns=["lsoa_code"])
    text = _g4_refusal(_replace_household(base, household), engine)
    assert "lacks" in text and "lsoa_code" in text
    assert engine.calls == []


def test_g4_evidence_counts_sources_by_grain_and_northern_ireland_drawn(
    copy_runs,
) -> None:
    systems = [_SYSTEM_OF_REGION.get(case.region, EW) for case in TOY_HOUSEHOLDS]
    digest = hashlib.sha256(
        files("microcosm.build.uk").joinpath(BRMA_LAYER_RESOURCE).read_bytes()
    ).hexdigest()
    for clones in (1, 3):
        evidence = copy_runs[clones]["evidence"]["g4"]
        assert evidence["source_households_by_grain"] == {
            "ew_lsoa21": rounded_count(systems.count(EW)),
            "scotland_oa22": rounded_count(systems.count(SCOT)),
        }
        # Sources, not benefit units.
        assert evidence["northern_ireland_source_households_drawn"] == rounded_count(
            systems.count(NI)
        )
        assert evidence["layer_sha256"] == digest
        assert evidence["method"] == "uk-brma-area-layer-v1"
        assert (evidence["seed"], evidence["output"]) == (0, "brma")


# --- The SPI-copy area stage --------------------------------------------------------


def test_the_copy_stage_runs_on_the_support_channel_lineage_alone(copy_runs) -> None:
    """The frame carries the support channel's lineage and the clone index
    alone; the full-path reader, which needs the CGT steps, refuses it."""

    copied = copy_runs[3]["frames"]["copied"].table("household")
    assert not {"household_is_cgt_support_copy", "cgt_support_copy_index"} & set(
        copied.columns
    )
    with pytest.raises(ValueError, match="lacks lineage column"):
        uk_household_clone_paths(copied)
    assert copy_runs[3]["frames"]["copy"].table("household")[IDENTITY_COLUMN].is_unique


def _expected_copy_keys(household: pd.DataFrame) -> list[str]:
    """Each copy's key over its source, vintage and copy path, written out."""

    copies = household.loc[_copies(household)]
    return [
        household_draw_key(
            source="frs",
            source_vintage="2024_25",
            source_household_id=int(source),
            clone_path=(("spi_support_channel", int(ordinal)),)
            + ((("geographic_support", int(clone)),) if clone else ()),
        )
        for source, ordinal, clone in zip(
            copies["source_household_id"],
            copies[_SUPPORT_ORDINAL],
            copies[_CLONE],
            strict=True,
        )
    ]


def test_every_copy_takes_a_key_of_its_own_over_its_copy_path(copy_runs) -> None:
    for clones in (1, 3):
        before = copy_runs[clones]["frames"]["copied"].table("household")
        after = copy_runs[clones]["frames"]["copy"].table("household")
        copies = _copies(before)
        assert copies.any()
        # Each copy came in with its parent's key, and leaves with its own.
        assert not before[IDENTITY_COLUMN].is_unique
        assert after[IDENTITY_COLUMN].is_unique
        assert after.loc[copies, IDENTITY_COLUMN].tolist() == _expected_copy_keys(
            before
        )
        assert uk_spi_copy_keys(before)[1] == _expected_copy_keys(before)
        assert after.loc[~copies, IDENTITY_COLUMN].equals(
            before.loc[~copies, IDENTITY_COLUMN]
        )


def test_a_copys_key_is_the_full_path_readers_once_later_steps_exist(
    copy_runs,
) -> None:
    """With the CGT steps' columns at ordinal 0, the full-path reader gives
    every household, base row or copy, the key it carries."""

    household = copy_runs[3]["frames"]["copy"].table("household").copy()
    zeros = np.zeros(len(household), dtype=np.int64)
    for column in ("cgt_support_copy_index", "cgt_residential_clone_index"):
        household[column] = zeros
    for column in (
        "household_is_cgt_support_copy",
        "household_is_capital_gains_clone",
        "household_is_cgt_residential_clone",
    ):
        household[column] = zeros.astype(bool)
    sources, paths = uk_household_clone_paths(household)
    assert [
        household_draw_key(
            source="frs",
            source_vintage="2024_25",
            source_household_id=source,
            clone_path=path,
        )
        for source, path in zip(sources, paths, strict=True)
    ] == household[IDENTITY_COLUMN].tolist()


def test_clone_zero_copies_keep_their_keys_and_places_as_k_grows(copy_runs) -> None:
    columns = [IDENTITY_COLUMN, *UK_SPINE_AREA_COLUMNS, "local_authority", "brma"]

    def clone_zero_copies(clones: int) -> pd.DataFrame:
        household = copy_runs[clones]["frames"]["copy"].table("household")
        rows = household.loc[_copies(household) & (household[_CLONE] == 0).to_numpy()]
        return (
            rows.set_index(["source_household_id", _SUPPORT_ORDINAL])[
                [*columns, RFDR_COLUMN]
            ]
            .sort_index()
            .astype({column: object for column in columns})
        )

    for clones in (2, 3):
        assert clone_zero_copies(clones).equals(clone_zero_copies(1)), clones


def test_the_copy_stage_leaves_every_other_row_and_table_as_it_was(copy_runs) -> None:
    for clones in (1, 3):
        before_frame = copy_runs[clones]["frames"]["copied"]
        after_frame = copy_runs[clones]["frames"]["copy"]
        before = before_frame.table("household")
        after = after_frame.table("household")
        copies = _copies(before)
        assert list(after.columns) == list(before.columns)
        assert after.loc[~copies].equals(before.loc[~copies])
        kept = [c for c in before.columns if c not in UK_SPI_COPY_AREAS_REWRITE_COLUMNS]
        assert after[kept].equals(before[kept])
        for entity in ("person", "benunit"):
            assert after_frame.table(entity).equals(before_frame.table(entity))
        assert np.array_equal(
            after_frame.weights_for("household").values,
            before_frame.weights_for("household").values,
        )
        assert after_frame.mass_log == before_frame.mass_log
        assert after_frame.metadata == before_frame.metadata


def test_each_rewrite_keeps_its_columns_dtype(copy_runs) -> None:
    before = copy_runs[3]["frames"]["copied"].table("household")
    after = copy_runs[3]["frames"]["copy"].table("household")
    assert after.dtypes.equals(before.dtypes)
    for column in (IDENTITY_COLUMN, *UK_SPINE_AREA_COLUMNS, "local_authority"):
        assert after[column].dtype == _STRING, column
    assert after["brma"].dtype == np.dtype(object)
    assert after[RFDR_COLUMN].dtype == np.dtype(bool)


def test_every_copy_draws_its_area_at_l3_in_its_region(
    copy_runs, definition, supports
) -> None:
    for clones in (1, 3):
        before = copy_runs[clones]["frames"]["copied"].table("household")
        after = copy_runs[clones]["frames"]["copy"].table("household")
        copies = _copies(after)
        rows = after.loc[copies]
        levels = uk_draw_levels(uk_draw_inputs(rows), definition, supports)
        assert set(levels.tolist()) == {"L3"}
        for _, row in rows.iterrows():
            support = supports[row["atomic_area_system"]]
            position = int(
                np.flatnonzero(support.arrays["area"] == row["atomic_area_code"])[0]
            )
            assert support.arrays["frs_region"][position] == row["region"]
            assert row["oa_code"] == row["atomic_area_code"]
            for column in UK_SPINE_LADDER_COLUMNS[1:]:
                assert row[column] == support.arrays[column][position], column
        # Each copy's area is the draw of its own key, not its parent's area.
        drawn = draw_uk_spine_areas(rows, definition=definition, supports=supports)
        assert rows[list(UK_SPINE_AREA_COLUMNS)].equals(drawn.columns)
    assert (
        after.loc[copies, "atomic_area_code"].to_numpy()
        != before.loc[copies, "atomic_area_code"].to_numpy()
    ).any()


def test_every_copy_takes_its_areas_local_authority_brma_and_flag(copy_runs) -> None:
    roster = local_authority_engine_key_by_code()
    layer = _committed_brma_layer()
    flagged = committed_rfdr_areas()
    for clones in (1, 3):
        household = copy_runs[clones]["frames"]["copy"].table("household")
        rows = household.loc[_copies(household)]
        assert rows["local_authority"].tolist() == [
            roster[code] for code in rows["local_authority_code"]
        ]
        for grain, spec in BRMA_GRAINS.items():
            great_britain = rows.loc[rows["atomic_area_system"] == spec.system]
            assert len(great_britain)
            assert great_britain["brma"].tolist() == [
                layer.brma_by_area[grain][code] for code in great_britain[spec.key]
            ]
        assert rows[RFDR_COLUMN].tolist() == [
            system in flagged and code in flagged[system]
            for system, code in zip(
                rows["atomic_area_system"], rows["atomic_area_code"], strict=True
            )
        ]


def test_northern_ireland_copies_draw_their_own_brma_on_their_own_benefit_units(
    copy_runs,
) -> None:
    """C-9, María's ruling of 2026-11-11: one engine call over the Northern
    Ireland copies' households alone, then frs_brma's draw keyed on the
    copies' own benefit units and households, with its seed and salts."""

    differs = []
    for clones in (1, 3):
        run = copy_runs[clones]
        frame = run["frames"]["copy"]
        household = frame.table("household")
        person, benunit = frame.table("person"), frame.table("benunit")
        northern_ireland = _copies(household) & (
            household["atomic_area_system"] == NI
        ).to_numpy(dtype=bool)
        ids = household.loc[northern_ireland, "household_id"]
        assert len(ids)
        (call,) = run["engines"]["copy"].calls
        seen, variables, period = call
        assert (variables, period) == (("LHA_category",), uk_time_period(frame))
        assert seen.table("household")["household_id"].tolist() == ids.tolist()
        # One simulation over every clone's copies: no clone index to batch by.
        assert _CLONE not in seen.table("household").columns
        members = person.loc[person["person_household_id"].isin(ids)]
        assert sorted(seen.table("person")["person_id"]) == sorted(members["person_id"])
        units = benunit.loc[benunit["benunit_id"].isin(members["person_benunit_id"])]
        assert (
            seen.table("benunit")["benunit_id"].tolist() == units["benunit_id"].tolist()
        )
        cells = units[["benunit_id"]].assign(
            region="NORTHERN_IRELAND",
            LHA_category=[_LHA_CATEGORIES[int(i) % 5] for i in units["benunit_id"]],
        )
        drawn = assign_brma_by_cell(
            cells, count_resource=load_brma_count_resource(), seed=0
        )
        expected = collapse_benunit_brma_to_household(
            person,
            cells[["benunit_id"]].assign(brma=drawn),
            household.loc[northern_ireland],
            seed=0,
        )
        assert household.loc[northern_ireland, "brma"].tolist() == expected.tolist()
        # The copies arrived with their parents' BRMAs.
        before = run["frames"]["copied"].table("household")
        differs += (
            before.loc[northern_ireland, "brma"].to_numpy() != expected
        ).tolist()
    # Drawn, not inherited: some copy's BRMA is not its parent's.
    assert any(differs)


def test_the_copy_stage_calls_no_engine_without_northern_ireland_copies(
    payloads, runs
) -> None:
    base = runs[1]["frames"]["g5"]
    household = base.table("household")
    northern_ireland = set(
        household.loc[household["region"] == "NORTHERN_IRELAND", "source_household_id"]
    )
    run = _copy_run(
        payloads,
        base,
        picks=lambda source, clone: () if source in northern_ireland else (1,),
    )
    assert run["engines"]["copy"].calls == []
    evidence = run["evidence"]["copy"]
    assert evidence["copy_source_households"]["spi_support_copy"][NI] == 0
    assert run["frames"]["copy"].table("household")[IDENTITY_COLUMN].is_unique


class _GuardedEngine(_RecordingEngine):
    """The recording engine, refusing as the UK adapter built with
    ``clones=K`` refuses: its own batching decision, which builds no engine,
    runs on every frame it is handed and records the simulations it would
    run."""

    def __init__(self, clones: int) -> None:
        super().__init__()
        self.adapter = PolicyEngineUKEngine(clones=clones)
        self.simulations: list[int] = []

    def materialize(self, frame, variables, period):
        self.simulations.append(self.adapter.simulation_batches(frame))
        return super().materialize(frame, variables, period)


@pytest.mark.parametrize("clones", [2, 3])
def test_the_copy_stages_engine_call_is_one_simulation_at_every_clone_count(
    payloads, runs, clones
) -> None:
    """Stage 16 draws its SPI share per clone, so a later clone may hold no
    Northern Ireland copy. The adapter refuses a call over part of its pool,
    so the call over the copies alone carries no clone index and runs as one
    simulation over every clone's copies, as at one clone (C-9)."""

    base = runs[clones]["frames"]["g5"]
    household = base.table("household")
    northern_ireland = set(
        household.loc[household["region"] == "NORTHERN_IRELAND", "source_household_id"]
    )

    def picks(source: int, clone: int) -> tuple[int, ...]:
        if clone > 0 and source in northern_ireland:
            return ()
        return _copy_picks(source, clone)

    g4_engine = _GuardedEngine(clones)
    copied = _with_support_channel(_g4(g4_engine)(base), picks)
    assert g4_engine.simulations == [clones]
    copies = copied.table("household")
    in_clone_zero = (
        _copies(copies)
        & (copies["atomic_area_system"] == NI).to_numpy()
        & (copies[_CLONE] == 0).to_numpy()
    )
    assert in_clone_zero.any()
    # Kept, the clone index would put the copies of clone 0 alone before an
    # engine built for ``clones`` clones, which refuses them.
    subset = copied.select(
        copied.table("person")["person_household_id"]
        .isin(copies.loc[in_clone_zero, "household_id"])
        .to_numpy(dtype=bool)
    )
    with pytest.raises(ValueError, match="over a sub-pool"):
        PolicyEngineUKEngine(clones=clones).simulation_batches(subset)
    engine = _GuardedEngine(clones)
    copy_stage = _copy_stage(payloads, engine)
    placed = copy_stage(copied).table("household")
    assert engine.simulations == [1]
    (call,) = engine.calls
    assert call[0].table("household")["household_id"].tolist() == (
        copies.loc[in_clone_zero, "household_id"].tolist()
    )
    assert placed.loc[in_clone_zero, "brma"].notna().all()
    assert copy_stage.last_result.evidence()["brma"]["engine_simulations"] == "one"


def test_the_copy_stage_says_its_checks_did_not_run_without_a_copy(
    payloads, runs
) -> None:
    run = _copy_run(payloads, runs[1]["frames"]["g5"], picks=lambda source, clone: ())
    evidence = run["evidence"]["copy"]
    assert evidence["validation"] == {
        "outcome": "not_run",
        "scope": "atomic_geography_mapping_integrity",
    }
    assert evidence["rfdr"]["roster"] == "not_run"
    assert run["engines"]["copy"].calls == []
    assert (
        run["frames"]["copy"]
        .table("household")
        .equals(run["frames"]["copied"].table("household"))
    )
    assert all(
        count == 0
        for counts in evidence["copy_source_households"].values()
        for count in counts.values()
    )


def _layer_without_toy_areas(*, misplace: bool) -> object:
    """The committed BRMA layer with every England-and-Wales toy LSOA dropped,
    or, with ``misplace``, placed in another FRS region than its own."""

    layer = _committed_brma_layer()
    lsoas = {area.lsoa_code for area in TOY_AREAS if area.system == EW}
    brmas = {grain: dict(codes) for grain, codes in layer.brma_by_area.items()}
    regions = {grain: dict(codes) for grain, codes in layer.region_by_area.items()}
    for code in lsoas:
        if misplace:
            regions["ew_lsoa21"][code] = (
                "SOUTH_EAST" if regions["ew_lsoa21"][code] != "SOUTH_EAST" else "LONDON"
            )
        else:
            del brmas["ew_lsoa21"][code]
            del regions["ew_lsoa21"][code]
    return replace(layer, brma_by_area=brmas, region_by_area=regions)


@pytest.mark.parametrize(
    ("misplace", "message"),
    [(False, "not on the layer"), (True, "another FRS region")],
)
def test_the_copy_stage_checks_the_copies_areas_before_its_engine_call(
    payloads, definition, supports, copy_runs, misplace, message
) -> None:
    copied = copy_runs[1]["frames"]["copied"]
    household = copied.table("household")
    # Northern Ireland copies are present, so an engine call would follow.
    assert (
        _copies(household) & (household["atomic_area_system"] == NI).to_numpy()
    ).any()
    rfdr_layer, rfdr_sha256 = load_committed_rfdr_layer()
    engine = _RecordingEngine()
    with pytest.raises(ValueError, match=message) as refused:
        assign_uk_spi_copy_areas(
            copied,
            definition=definition,
            supports=supports,
            engine=engine,
            brma_layer=_layer_without_toy_areas(misplace=misplace),
            brma_layer_sha256="0" * 64,
            count_resource=load_brma_count_resource(),
            count_resource_sha256="0" * 64,
            rfdr_layer=rfdr_layer,
            rfdr_layer_sha256=rfdr_sha256,
        )
    assert engine.calls == []
    text = str(refused.value)
    assert not any(area.lsoa_code in text for area in TOY_AREAS)


def _copy_refusal(payloads, frame: Frame, engine: object | None = None) -> str:
    with pytest.raises(ValueError) as refused:
        _copy_stage(payloads, engine if engine is not None else _RecordingEngine())(
            frame
        )
    return str(refused.value)


def test_the_copy_stage_refuses_lineage_it_cannot_read(payloads, copy_runs) -> None:
    # No support channel: the stage runs after spi_income_band_donors.
    text = _copy_refusal(payloads, copy_runs[1]["frames"]["g4"])
    assert "'spi_support_channel' step" in text
    copied = copy_runs[1]["frames"]["copied"]
    household = copied.table("household")
    # A later copy step has run.
    later = household.copy()
    marker = np.zeros(len(later), dtype=bool)
    marker[0] = True
    later["household_is_cgt_support_copy"] = marker
    later["cgt_support_copy_index"] = marker.astype(np.int64)
    text = _copy_refusal(payloads, _replace_household(copied, later))
    assert "before any later copy step" in text
    # An ordinal the support channel never mints.
    other = household.copy()
    donor = other.index[other[_SUPPORT_ORDINAL] == 2][0]
    other.loc[donor, _SUPPORT_ORDINAL] = 3
    assert "another ordinal" in _copy_refusal(
        payloads, _replace_household(copied, other)
    )


def test_the_copy_stage_refuses_copies_whose_keys_repeat(payloads, runs) -> None:
    """Two copies of one source on one path would share a key."""

    placed = _g4(_RecordingEngine())(runs[1]["frames"]["g5"])
    household = placed.table("household").copy()
    sources = household["source_household_id"].to_numpy().copy()
    sources[1] = sources[0]
    household["source_household_id"] = sources
    copied = _with_support_channel(_replace_household(placed, household))
    text = _copy_refusal(payloads, copied)
    assert "keys repeat" in text
    keys = set(copied.table("household")[IDENTITY_COLUMN])
    assert not any(key in text for key in keys)
    # No id or count: the issue reference is its only number.
    assert not any(c.isdigit() for c in text.replace("microcosm#931", ""))


def test_the_copy_stage_refuses_a_missing_column_or_misaligned_engine_values(
    payloads, copy_runs
) -> None:
    copied = copy_runs[1]["frames"]["copied"]
    without = copied.table("household").drop(columns=["brma"])
    text = _copy_refusal(payloads, _replace_household(copied, without))
    assert "lacks" in text and "brma" in text
    text = _copy_refusal(payloads, copied, _RecordingEngine(misaligned=True))
    assert "not aligned" in text


def test_the_copy_stage_counts_copies_donors_and_flags_among_clone_zero(
    copy_runs,
) -> None:
    single = copy_runs[1]["frames"]["copy"].table("household")
    rows = single.loc[_copies(single)]

    def sources(mask) -> int:
        return rows.loc[mask, "source_household_id"].nunique()

    expected = {
        kind: {
            system: rounded_count(
                sources(
                    (rows[_SUPPORT_ORDINAL] == ordinal)
                    & (rows["atomic_area_system"] == system)
                )
            )
            for system in (EW, SCOT, NI)
        }
        for ordinal, kind in ((1, "spi_support_copy"), (2, "income_band_donor"))
    }
    flagged = {
        system: rounded_count(
            sources(rows[RFDR_COLUMN] & (rows["atomic_area_system"] == system))
        )
        for system in (EW, SCOT)
    }
    for clones in (1, 3):
        evidence = copy_runs[clones]["evidence"]["copy"]
        assert evidence["counted_among"] == "clone_0"
        assert evidence["copy_source_households"] == expected
        assert evidence["flagged_copy_source_households"] == flagged
        assert evidence["no_flagged_copy"] == {
            system: count == 0 for system, count in flagged.items()
        }
        assert evidence["level"] == "L3"
        assert evidence["rfdr"]["roster"] == "pass"
        assert evidence["validation"]["outcome"] == "pass"
        assert evidence["brma"]["engine_households"] == "northern_ireland_copies"
        assert evidence["brma"]["engine_simulations"] == "one"
        assert evidence["rewrites"] == list(UK_SPI_COPY_AREAS_REWRITE_COLUMNS)


def test_the_copy_stage_counts_clone_zero_where_later_clones_copy_others(
    copy_runs,
) -> None:
    """Later clones copy other households, so a union of copied sources over
    clones grows with K; the evidence counts clone 0's."""

    household = copy_runs[3]["frames"]["copy"].table("household")
    donors = _copies(household) & (household[_SUPPORT_ORDINAL] == 2).to_numpy()
    zero = (household[_CLONE] == 0).to_numpy()
    systems = household["atomic_area_system"].to_numpy(dtype=object)
    sources = household["source_household_id"].to_numpy()

    def distinct(rows) -> dict[str, int]:
        return {
            system: len(set(sources[rows & (systems == system)].tolist()))
            for system in (EW, SCOT, NI)
        }

    # The invented frame tells the two counts apart.
    assert distinct(donors & zero) != distinct(donors)
    result = copy_runs[3]["results"]["copy"]
    assert dict(result.by_kind["income_band_donor"]) == distinct(donors & zero)


def test_the_copy_stage_draws_with_g2s_definition_and_supports(runs, copy_runs) -> None:
    g2 = runs[1]["evidence"]["g2"]
    for clones in (1, 3):
        copy = copy_runs[clones]["evidence"]["copy"]
        for key in ("definition_sha256", "support_sha256", "stream", "seed"):
            assert copy[key] == g2[key], key
        assert copy["validation"] == g2["validation"]


# --- Evidence across clone counts ---------------------------------------------------


def test_g4_and_the_copy_stage_evidence_is_byte_equal_at_k_1_and_k_3(
    copy_runs,
) -> None:
    for stage in ("g4", "copy"):
        single, triple = ({**copy_runs[clones]["evidence"][stage]} for clones in (1, 3))
        assert (single.pop("clone_count"), triple.pop("clone_count")) == (1, 3)
        assert json.dumps(single, sort_keys=True) == json.dumps(triple, sort_keys=True)


def test_g4_and_the_copy_stage_evidence_prints_no_key_and_no_code(copy_runs) -> None:
    household = copy_runs[3]["frames"]["copy"].table("household")
    text = json.dumps(copy_runs[3]["evidence"], sort_keys=True)
    for column in (
        IDENTITY_COLUMN,
        "atomic_area_code",
        "lsoa_code",
        "local_authority_code",
        "local_authority",
        "brma",
    ):
        values = {str(value) for value in household[column]}
        assert not any(value in text for value in values), column


def test_every_evidence_count_is_byte_equal_at_k_1_and_k_3(runs) -> None:
    """The plan's "K clones count as one source": each stage's evidence is
    the same bytes at K = 1 and K = 3 but for the clone count it states."""

    for stage in ("g2", "g3", "g5"):
        single, triple = ({**runs[clones]["evidence"][stage]} for clones in (1, 3))
        assert (single.pop("clone_count"), triple.pop("clone_count")) == (1, 3)
        assert json.dumps(single, sort_keys=True) == json.dumps(triple, sort_keys=True)


def test_the_evidence_prints_no_key_and_no_code_of_a_household(runs) -> None:
    household = _household(runs, 3)
    text = json.dumps(runs[3]["evidence"], sort_keys=True)
    for column in (
        IDENTITY_COLUMN,
        *UK_SPINE_AREA_COLUMNS[:1],
        *UK_SPINE_LADDER_COLUMNS,
    ):
        values = {str(value) for value in household[column]}
        # The region codes the ladder writes are the regions' own GSS codes,
        # which the evidence never names either.
        assert not any(value in text for value in values), column
