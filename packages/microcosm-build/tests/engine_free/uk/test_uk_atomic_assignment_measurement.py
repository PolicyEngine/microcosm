"""The spine area-draw measurement tool on invented households (microcosm#931).

No spine build, publisher file or country engine: toy conditioned supports
(test_support's uk_conditioned_support_fixtures) stand in for the registered
ones, and invented households cloned and keyed by the spine's own expansion
and identity steps stand in for a bound spine. The tool's functions run
directly, and its command runs on synthetic checkpoints written as the spine
writes one, with their authentication stood in for or refused.

They hold: the base rows and their weights, with SPI copies and band donors
left out and later copies' mass kept; every clone keyed as the spine keys
it; the region-only law drawing today's v1 draws; the conditioned law drawing
what G2 draws, the spine's own areas included; each law's expected rows read
off the definition its draw runs, and z against its Poisson-binomial spread;
Northern Ireland's placements on household weights and census households;
the draws stable on reversed rows, on the clone-0 subset and as K grows; the
disclosure rules (rule 1, under which the committed #931 evidence stays a
fixed point, and rule 2, which the tool writes, Northern Ireland's districts
among its outputs); and the command's evidence, receipt and refusals.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build import atomic_geography as geo
from microcosm.build.source_manifest import SourceOperationSpec, SourceStageSpec
from microcosm.build.uk_runtime import copy_id_table
from microcosm.build.uk_runtime.atomic_area_conditioning import (
    IDENTITY_COLUMN,
    UK_SPINE_AREA_COLUMNS,
    UK_SPINE_ATOMIC_AREA_SEED,
    uk_band_weight_map,
    uk_conditioned_assignment_definition,
)
from microcosm.build.uk_runtime.atomic_area_support import (
    CONDITIONED_SOURCES,
    IMD_DECILE_COLUMN,
    SYSTEMS,
    uk_atomic_assignment_definition,
)
from microcosm.build.uk_runtime.atomic_household_identity import household_draw_key
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
from microcosm.build.uk_runtime.geographic_clones import ladder_clone_index_column
from microcosm.build.uk_runtime.local_authority_input import (
    resolve_local_authority_engine_keys,
)
from microcosm.build.uk_runtime.local_doctrine import UK_LOCAL_CLONE_COUNT
from microcosm.build.uk_runtime.national_frame import write_uk_national_frame
from microcosm.build.uk_runtime.receipt_counts import (
    UK_BELOW_TEN_BAND,
    UK_DISCLOSURE_FLOOR,
    rounded_count,
)
from microcosm.build.uk_runtime.rowwise_geography import FRS_REGION_TO_REGION_CODE
from microcosm.build.uk_runtime.rowwise_posture import UK_DENSE_SEED
from microcosm.build.uk_runtime.spine_geography import (
    ATOMIC_AREA_ASSIGN_STAGE_NAME,
    UK_ATOMIC_AREA_ASSIGN_OUTPUT_COLUMNS,
    UKAtomicAreaAssignStageTransform,
    atomic_area_assign_operation_parameters,
)
from microcosm.frame import Frame, Weights
from test_support.microcosm_build.uk_conditioned_support_fixtures import (
    NI,
    SCOT,
    TOY_HOUSEHOLDS,
    ToyHousehold,
    toy_conditioned_support_payloads,
    toy_household_table,
    toy_spine_frame,
)
from test_support.paths import paths_for

CLONE = ladder_clone_index_column("household")
SOURCE = "source_household_id"
WEIGHT = "household_weight"
LAWS = ("region_only", "conditioned")
M = UK_DISCLOSURE_FLOOR
#: Every toy case ten times over, for the statistical checks.
POOL = TOY_HOUSEHOLDS * 10
_SYSTEM_OF_REGION = {"NORTHERN_IRELAND": NI, "SCOTLAND": SCOT}
_TOOL = (
    paths_for("microcosm-build").repository / "tools/measure_uk_atomic_assignment.py"
)


@pytest.fixture(scope="module")
def harness():
    name = "uk_spine_area_draw_measurement_harness"
    spec = importlib.util.spec_from_file_location(name, _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The tool's dataclasses read their annotations through sys.modules.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def payloads() -> dict[str, bytes]:
    return toy_conditioned_support_payloads()


@pytest.fixture(scope="module")
def laws(harness, payloads):
    return harness.Laws.build(
        payloads, harness.base_supports(payloads), region_only_seed=UK_DENSE_SEED
    )


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


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


def _with_copies(frame: Frame, *, spi: dict[int, int], cgt: dict[int, float] = ()):
    """``frame`` with SPI-channel copies and, optionally, CGT support arms.

    ``spi`` maps a household row's position to its support-channel ordinal
    (1 an SPI copy, 2 an income band donor); each copy keeps its parent's
    inputs, key and source, takes ids minted in the channel's copy-id field
    and a weight of 500. ``cgt`` maps a row's position to the share of its
    weight its one CGT support arm takes, the arm keeping its key and area.
    """

    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    household = frame.table("household").copy()
    weights = np.asarray(frame.weights_for("household").values, dtype=float).copy()
    household["household_support_channel"] = pd.Series(
        ["frs"] * len(household), dtype=object, index=household.index
    )
    household["household_support_clone_index"] = np.zeros(len(household), np.int64)
    household["household_is_cgt_support_copy"] = np.zeros(len(household), bool)
    household["cgt_support_copy_index"] = np.zeros(len(household), np.int64)
    persons, benunits, households, extra = [person], [benunit], [household], []

    def copy_rows(positions, field, ordinal, edit, weight_of):
        rows = household.iloc[positions].copy()
        members = person[person["person_household_id"].isin(set(rows["household_id"]))]
        members = members.copy()
        units = benunit[benunit["benunit_id"].isin(set(members["person_benunit_id"]))]
        units = units.copy()
        for column in ("person_id", "person_household_id", "person_benunit_id"):
            members[column] = copy_id_table.mint(members[column], field, ordinal)
        units["benunit_id"] = copy_id_table.mint(units["benunit_id"], field, ordinal)
        rows["household_id"] = copy_id_table.mint(rows["household_id"], field, ordinal)
        edit(rows)
        persons.append(members)
        benunits.append(units)
        households.append(rows)
        extra.extend(weight_of(position) for position in positions)

    for ordinal in sorted(set(spi.values())):
        positions = [position for position, value in spi.items() if value == ordinal]

        def spi_edit(rows, ordinal=ordinal):
            rows["household_support_channel"] = "spi"
            rows["household_support_clone_index"] = ordinal

        copy_rows(positions, "spi_support_channel", ordinal, spi_edit, lambda _: 500.0)
    for position, share in dict(cgt).items():

        def cgt_edit(rows):
            rows["household_is_cgt_support_copy"] = True
            rows["cgt_support_copy_index"] = 1

        moved = weights[position] * share
        copy_rows(
            [position], "cgt_support_split", 1, cgt_edit, lambda _, mass=moved: mass
        )
        weights[position] -= moved
    tables = {
        "person": pd.concat(persons, ignore_index=True),
        "benunit": pd.concat(benunits, ignore_index=True),
        "household": pd.concat(households, ignore_index=True),
    }
    order = np.argsort(tables["household"]["household_id"].to_numpy(), kind="stable")
    every_weight = np.concatenate([weights, np.asarray(extra, dtype=float)])
    tables["household"] = tables["household"].iloc[order].reset_index(drop=True)
    tables["benunit"] = (
        tables["benunit"].sort_values("benunit_id").reset_index(drop=True)
    )
    return Frame(
        tables,
        frame.schema,
        {
            "household": Weights(
                values=every_weight[order], kind=frame.weights_for("household").kind
            )
        },
        None,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _g2(payloads) -> UKAtomicAreaAssignStageTransform:
    """G2's transform, run as a stage built from its reviewed declaration."""

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
        rewrites=(),
    )
    return UKAtomicAreaAssignStageTransform(stage=stage, supports=payloads)


def _measured(harness, laws, law: str, clones: int, households=TOY_HOUSEHOLDS):
    base = harness.spine_base(toy_spine_frame(clones=1, households=households))
    table = harness.expand_base(base, clones)
    drawn = laws.draw(law, table)
    return table, drawn, pd.concat([table, drawn.columns], axis=1)


# --- the base rows and their clones -------------------------------------------


def test_the_base_rows_are_the_households_with_their_clones_weight(harness) -> None:
    one, three = toy_spine_frame(clones=1), toy_spine_frame(clones=3)
    base_one, base_three = harness.spine_base(one), harness.spine_base(three)
    assert (base_one.clone_count, base_three.clone_count) == (1, 3)
    survey_year = str(load_uk_frs_release().survey_year)
    assert base_one.time_period == base_three.time_period == survey_year
    assert base_one.spine_areas is None and base_three.spine_areas is None
    assert base_one.household[SOURCE].tolist() == list(
        range(1, len(TOY_HOUSEHOLDS) + 1)
    )
    pd.testing.assert_frame_equal(
        base_one.household.drop(columns=WEIGHT),
        base_three.household.drop(columns=WEIGHT),
    )
    inputs = toy_household_table(TOY_HOUSEHOLDS)
    for column in inputs.columns:
        assert base_one.household[column].tolist() == inputs[column].tolist()
    # Each clone carries an equal share; the shares sum back to the weight.
    np.testing.assert_allclose(
        base_one.household[WEIGHT].to_numpy(), one.weights_for("household").values
    )
    np.testing.assert_allclose(
        base_three.household[WEIGHT].to_numpy(),
        base_one.household[WEIGHT].to_numpy(),
        rtol=1e-12,
    )


def test_the_base_rows_leave_out_spi_copies_and_band_donors(harness) -> None:
    plain = toy_spine_frame(clones=2)
    n = len(TOY_HOUSEHOLDS)
    # SPI copies of six households' clone-0 and clone-1 rows, band donors of four.
    spi = {position: 1 for position in (0, 3, 24, 25, n + 0, n + 25)}
    spi.update({position: 2 for position in (5, 26, n + 5, n + 27)})
    copied = _with_copies(plain, spi=spi)
    assert len(copied.table("household")) == 2 * n + len(spi)
    actual, expected = harness.spine_base(copied), harness.spine_base(plain)
    pd.testing.assert_frame_equal(actual.household, expected.household)
    assert actual.clone_count == 2


def test_a_households_weight_is_its_mass_on_every_row_that_keeps_its_areas(
    harness,
) -> None:
    plain = toy_spine_frame(clones=2)
    n = len(TOY_HOUSEHOLDS)
    design = np.asarray(plain.weights_for("household").values, dtype=float)
    sources = plain.table("household")[SOURCE].to_numpy()
    # A Northern Ireland household's clone-0 root gives 75% of its weight to a
    # CGT support arm, which keeps its key and areas; SPI copies and a band
    # donor, which draw areas of their own, sit beside it.
    assert TOY_HOUSEHOLDS[24].region == "NORTHERN_IRELAND"
    copied = _with_copies(plain, spi={24: 1, n + 24: 1, 25: 2}, cgt={24: 0.75})
    base = harness.spine_base(copied).household.set_index(SOURCE)[WEIGHT]
    expected = pd.Series(design, index=sources).groupby(level=0).sum()
    assert base.index.tolist() == expected.index.tolist()
    np.testing.assert_allclose(base.to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_a_spine_without_clones_or_with_clones_that_disagree_is_refused(
    harness,
) -> None:
    frame = toy_spine_frame(clones=2)
    household = frame.table("household")
    with pytest.raises(ValueError, match="lacks \\['household_clone_index'\\]"):
        harness.spine_base(_replace_household(frame, household.drop(columns=CLONE)))
    moved = household.copy()
    first_clone = int(np.flatnonzero(moved[CLONE].to_numpy() == 1)[0])
    moved.loc[moved.index[first_clone], "council_tax_band"] = "I"
    with pytest.raises(ValueError, match="disagree on 'council_tax_band'") as refused:
        harness.spine_base(_replace_household(frame, moved))
    assert not any(
        character.isdigit() for character in str(refused.value).split("(")[0]
    )


def test_each_clone_is_keyed_as_the_spines_identity_keys_it(harness) -> None:
    base = harness.spine_base(toy_spine_frame(clones=1))
    weights = dict(zip(base.household[SOURCE], base.household[WEIGHT], strict=True))
    for clones in (1, 3):
        table = harness.expand_base(base, clones)
        spine = toy_spine_frame(clones=clones).table("household")

        def keys(frame: pd.DataFrame) -> dict:
            return dict(
                zip(
                    zip(frame[SOURCE].tolist(), frame[CLONE].tolist(), strict=True),
                    frame[IDENTITY_COLUMN].astype(str).tolist(),
                    strict=True,
                )
            )

        assert keys(table) == keys(spine)
        np.testing.assert_allclose(
            table[WEIGHT].to_numpy(),
            np.asarray([weights[source] / clones for source in table[SOURCE]]),
        )


# --- the two laws ---------------------------------------------------------------


def test_the_region_only_law_reproduces_todays_v1_draws(
    harness, payloads, laws
) -> None:
    """Today's v1 definition at the dense role's seed, drawn by the shared
    operator on keys made straight from household_draw_key (the dense line's
    identity rule), places every clone where the region-only law does."""

    base_payloads = harness.base_supports(payloads)
    v1 = uk_atomic_assignment_definition(base_payloads, seed=UK_DENSE_SEED)
    assert laws.definitions["region_only"] == v1
    assert (
        harness._parse_args(
            ["--input-h5", "x.h5", "--supports-dir", "s", "--evidence-out", "e.json"]
        ).region_only_seed
        == UK_DENSE_SEED
    )
    table, drawn, _ = _measured(harness, laws, "region_only", 3)
    vintage = load_uk_frs_release().vintage
    keys = [
        household_draw_key(
            source="frs",
            source_vintage=vintage,
            source_household_id=int(source),
            clone_path=(("geographic_support", int(clone)),) if clone else (),
        )
        for source, clone in zip(table[SOURCE], table[CLONE], strict=True)
    ]
    pool = pd.DataFrame(
        {IDENTITY_COLUMN: pd.array(keys, dtype="string"), "region": table["region"]}
    )
    supports = {s: geo.decode_atomic_support(p) for s, p in base_payloads.items()}
    assigned = geo.assign_atomic(pool, v1, supports)
    derived = geo.derive_geography(pd.concat([pool, assigned], axis=1), v1, supports)
    expected = pd.concat([assigned, derived], axis=1)
    for column in UK_SPINE_AREA_COLUMNS:
        assert drawn.columns[column].astype(str).tolist() == (
            expected[column].astype(str).tolist()
        ), column
    assert drawn.columns["local_authority"].tolist() == list(
        resolve_local_authority_engine_keys(expected["local_authority_code"])
    )
    assert drawn.levels is None
    assert list(drawn.cells.columns) == ["system", "region"]


def test_the_conditioned_law_is_g2_on_the_same_rows(harness, payloads, laws) -> None:
    assert laws.definitions["conditioned"] == uk_conditioned_assignment_definition(
        payloads, seed=UK_SPINE_ATOMIC_AREA_SEED
    )
    base = harness.spine_base(toy_spine_frame(clones=1))
    for clones in (1, 3):
        spine = _g2(payloads)(toy_spine_frame(clones=clones))
        drawn_by_g2 = spine.table("household").set_index(IDENTITY_COLUMN)
        table = harness.expand_base(base, clones)
        drawn = laws.draw("conditioned", table)
        mine = drawn.columns.copy()
        mine.index = pd.Index(table[IDENTITY_COLUMN].astype(str))
        for column in UK_SPINE_AREA_COLUMNS:
            assert mine[column].astype(str).to_dict() == (
                drawn_by_g2[column].astype(str).to_dict()
            ), column
        by_source = dict(zip(table[SOURCE], drawn.levels, strict=True))
        assert [by_source[source] for source in range(1, len(TOY_HOUSEHOLDS) + 1)] == [
            case.level for case in TOY_HOUSEHOLDS
        ]
        # A checkpoint that carries G2's columns hands them to the tool, and
        # the tool's draw is the spine's own; a moved area is seen.
        carried = harness.spine_base(spine)
        assert carried.spine_areas is not None
        redraw = lambda rows: laws.draw("conditioned", rows).columns  # noqa: E731
        stability = harness.identity_stability(
            table, redraw, drawn.columns, spine_areas=carried.spine_areas
        )
        assert stability == {
            "reversed_rows_equal": True,
            "clone_zero_subset_equal": True,
            "keys_unique": True,
            "spine_areas_equal": True,
        }
        moved = carried.spine_areas.copy()
        moved.iloc[0, moved.columns.get_loc("atomic_area_code")] = "E00000099"
        assert harness.spine_areas_equal(table, drawn.columns, moved) is False


def _cases(table: pd.DataFrame) -> list:
    return [POOL[int(source) - 1] for source in table[SOURCE]]


def _independent_expectation(laws, law: str, table: pd.DataFrame) -> dict:
    """Expected rows per constituency from each case's declared level, read
    off the toy supports row by row."""

    expected: dict[str, float] = {}
    for case in _cases(table):
        system = _SYSTEM_OF_REGION.get(case.region, SYSTEMS[0])
        arrays = laws.supports[law][system].arrays
        mask = np.asarray(arrays["frs_region"]) == case.region
        column = "households"
        if law == "conditioned":
            if case.level in ("L0", "L1"):
                mask &= np.asarray(arrays[IMD_DECILE_COLUMN]) == str(case.imd_decile)
            if case.level in ("L0", "L2"):
                column = uk_band_weight_map(system)[case.council_tax_band]
        weights = np.asarray(arrays[column], dtype=np.float64) * mask
        for code, weight in zip(
            np.asarray(arrays["constituency_code"]).tolist(),
            (weights / weights.sum()).tolist(),
            strict=True,
        ):
            expected[code] = expected.get(code, 0.0) + weight
    return expected


def _row_probabilities(laws, law: str, table: pd.DataFrame) -> list[dict]:
    """Each row's probability of each constituency and local authority, read
    off the toy supports at the row's declared level."""

    out = []
    for case in _cases(table):
        system = _SYSTEM_OF_REGION.get(case.region, SYSTEMS[0])
        arrays = laws.supports[law][system].arrays
        mask = np.asarray(arrays["frs_region"]) == case.region
        column = "households"
        if law == "conditioned":
            if case.level in ("L0", "L1"):
                mask &= np.asarray(arrays[IMD_DECILE_COLUMN]) == str(case.imd_decile)
            if case.level in ("L0", "L2"):
                column = uk_band_weight_map(system)[case.council_tax_band]
        weights = np.asarray(arrays[column], dtype=np.float64) * mask
        probabilities = weights / weights.sum()
        row: dict[tuple[str, str], float] = {}
        for level, code_column in (
            ("constituency", "constituency_code"),
            ("la", "local_authority_code"),
        ):
            for code, value in zip(
                np.asarray(arrays[code_column]).tolist(),
                probabilities.tolist(),
                strict=True,
            ):
                row[(level, code)] = row.get((level, code), 0.0) + value
        out.append(row)
    return out


def test_each_laws_expectation_is_read_off_its_own_cells(harness, laws) -> None:
    for law in LAWS:
        table, drawn, _ = _measured(harness, laws, law, 2, households=POOL)
        expected = harness.expected_area_support(
            drawn.cells, laws.definitions[law], laws.supports[law]
        )
        constituency = expected[expected["area_type"] == "constituency"].set_index(
            "area_code"
        )
        independent = _independent_expectation(laws, law, table)
        for code, value in independent.items():
            assert constituency.at[code, "expected_rows"] == pytest.approx(value)
        roster = harness.support_roster(laws.supports["conditioned"])
        regions = harness._area_regions(roster)
        rows_by_region = (
            table["region"].map(FRS_REGION_TO_REGION_CODE).value_counts().to_dict()
        )
        for area_type, column in harness.LEVELS:
            level = expected[expected["area_type"] == area_type]
            assert level["expected_rows"].sum() == pytest.approx(len(table))
            by_region = level.groupby(level["area_code"].map(regions[column]))[
                "expected_rows"
            ].sum()
            for region, count in rows_by_region.items():
                assert by_region[region] == pytest.approx(count)
        if law == "region_only":
            # One cell per region: n p (1 - p), the binomial the old z read.
            region_rows = np.asarray(
                [
                    rows_by_region.get(regions["constituency_code"][code], 0)
                    for code in constituency.index
                ],
                dtype=float,
            )
            expected_rows = constituency["expected_rows"].to_numpy()
            share = np.divide(
                expected_rows,
                region_rows,
                out=np.zeros_like(expected_rows),
                where=region_rows > 0,
            )
            np.testing.assert_allclose(
                constituency["variance"].to_numpy(), expected_rows * (1 - share)
            )


def test_the_expectation_follows_the_declared_law(
    harness, payloads, laws, monkeypatch
) -> None:
    """A later edit of the declared law (here the default weight becomes
    population) moves the draws through the shared core; the expectation
    reads the same definition, so it moves with them, row by row."""

    original = harness.uk_conditioned_assignment_definition

    def declared(supports, *, seed):
        definition = original(supports, seed=seed)
        for system in definition["systems"]:
            (stage,) = system["stages"]
            if "weight_by" in stage:
                stage["weight_by"]["default"] = "population"
            else:
                stage["weight"] = "population"
        return geo.validate_assignment_spec(definition)

    monkeypatch.setattr(harness, "uk_conditioned_assignment_definition", declared)
    edited = harness.Laws.build(
        payloads, harness.base_supports(payloads), region_only_seed=UK_DENSE_SEED
    )
    base = harness.spine_base(toy_spine_frame(clones=1, households=POOL))
    table = harness.expand_base(base, 2)
    drawn = edited.draw("conditioned", table)
    expected = harness.expected_area_support(
        drawn.cells, edited.definitions["conditioned"], edited.supports["conditioned"]
    )
    by_code = expected[expected["area_type"] == "constituency"].set_index("area_code")[
        "expected_rows"
    ]
    implied: dict[str, float] = {}
    for source, level in zip(table[SOURCE], drawn.levels, strict=True):
        case = POOL[int(source) - 1]
        system = _SYSTEM_OF_REGION.get(case.region, SYSTEMS[0])
        arrays = edited.supports["conditioned"][system].arrays
        mask = np.asarray(arrays["frs_region"]) == case.region
        column = "population"
        if level in ("L0", "L1"):
            mask &= np.asarray(arrays[IMD_DECILE_COLUMN]) == str(case.imd_decile)
        if level in ("L0", "L2"):
            column = uk_band_weight_map(system)[case.council_tax_band]
        weights = np.asarray(arrays[column], dtype=float) * mask
        for code, value in zip(
            np.asarray(arrays["constituency_code"]).tolist(),
            (weights / weights.sum()).tolist(),
            strict=True,
        ):
            implied[code] = implied.get(code, 0.0) + value
    for code, value in implied.items():
        assert by_code[code] == pytest.approx(value), code
    # The edit matters: the unedited law gives other expected rows here.
    unedited = harness.expected_area_support(
        drawn.cells, laws.definitions["conditioned"], laws.supports["conditioned"]
    )
    unedited = unedited[unedited["area_type"] == "constituency"].set_index("area_code")[
        "expected_rows"
    ]
    assert max(abs(unedited[code] - by_code[code]) for code in implied) > 1e-6


def test_the_z_scores_divide_by_the_poisson_binomial_spread(harness, laws) -> None:
    roster = harness.support_roster(laws.supports["conditioned"])
    regions = harness._area_regions(roster)
    checked = 0
    for law in LAWS:
        table, drawn, measured = _measured(harness, laws, law, 3, households=POOL)
        expected = harness.expected_area_support(
            drawn.cells, laws.definitions[law], laws.supports[law]
        )
        summaries = harness.realized_area_support(measured, roster)
        probabilities = _row_probabilities(laws, law, table)
        for row in harness._cell_rows(expected, summaries, regions):
            key = (row["level"], row["area_code"])
            p = np.asarray([probs.get(key, 0.0) for probs in probabilities])
            mean, spread = p.sum(), (p * (1 - p)).sum()
            assert row["expected_rows"] == pytest.approx(mean, abs=1e-9)
            if spread > 1e-12:
                checked += 1
                assert row["z"] == pytest.approx(
                    (row["rows"] - mean) / math.sqrt(spread), rel=1e-9
                ), (law, key)
            else:
                assert row["z"] is None
    assert checked > 10


def test_the_realized_rows_stay_inside_their_laws_spread(harness, laws) -> None:
    roster = harness.support_roster(laws.supports["conditioned"])
    regions = harness._area_regions(roster)
    for law in LAWS:
        _, drawn, measured = _measured(harness, laws, law, 3, households=POOL)
        expected = harness.expected_area_support(
            drawn.cells, laws.definitions[law], laws.supports[law]
        )
        summaries = harness.realized_area_support(measured, roster)
        rows = harness._cell_rows(expected, summaries, regions)
        tested = [row for row in rows if row["expected_rows"] >= 5.0]
        assert tested and all(
            abs(row["z"]) < 5.0 for row in tested if row["z"] is not None
        ), law
        # An area that holds every area of its rows' cells has no spread: it
        # takes exactly its expected rows.
        for row in tested:
            if row["z"] is None:
                assert row["rows"] == pytest.approx(row["expected_rows"]), law
        for level in ("constituency", "la"):
            total = sum(row["rows"] for row in rows if row["level"] == level)
            assert total == len(measured)


def test_northern_ireland_placements_read_weights_and_census_households(
    harness, laws
) -> None:
    support = laws.supports["conditioned"][NI]
    codes = np.asarray(support.arrays["local_authority_code"]).astype(str)
    households = np.asarray(support.arrays["households"], dtype=float)
    census = {
        code: households[codes == code].sum() / households.sum()
        for code in sorted(set(codes.tolist()))
    }
    assert harness.ni_census_shares(support) == pytest.approx(census)
    for law in LAWS:
        _, _, measured = _measured(harness, laws, law, 3, households=POOL)
        block = harness._ni_placements(measured, census)
        assert block["level"] == "la"
        ni = measured[measured["region"] == "NORTHERN_IRELAND"]
        weights = ni[WEIGHT].to_numpy(dtype=float)
        placed = ni["local_authority_code"].astype(str).to_numpy()
        assert len(set(np.round(weights, 9).tolist())) > 1, "needs unequal weights"
        assert [district["area_code"] for district in block["districts"]] == sorted(
            census
        )
        for district in block["districts"]:
            mask = placed == district["area_code"]
            assert district["weighted_share"] == pytest.approx(
                weights[mask].sum() / weights.sum()
            ), law
            assert district["census_share"] == pytest.approx(
                census[district["area_code"]]
            )
    # The toy districts' households, zones and population give one set of
    # shares; this stand-in tells them apart.
    stand_in = SimpleNamespace(
        arrays={
            "local_authority_code": np.asarray(["N09000001"] * 3 + ["N09000002"]),
            "households": np.asarray([10, 20, 30, 40]),
            "population": np.asarray([50, 20, 30, 40]),
        }
    )
    assert harness.ni_census_shares(stand_in) == pytest.approx(
        {"N09000001": 0.6, "N09000002": 0.4}
    )


def test_the_draws_hold_on_reordered_rows_and_nest_as_k_grows(harness, laws) -> None:
    base = harness.spine_base(toy_spine_frame(clones=1))
    clone_counts = (1, 3, UK_LOCAL_CLONE_COUNT)
    out = harness.measure(base, laws, n_clones=clone_counts, exclusions={})
    assert [(cell["law"], cell["n_clones"]) for cell in out["cells"]] == [
        (law, k) for k in clone_counts for law in LAWS
    ]
    for cell in out["cells"]:
        assert cell["identity_stability"] == {
            "reversed_rows_equal": True,
            "clone_zero_subset_equal": True,
            "keys_unique": True,
            "spine_areas_equal": None,
        }
        assert cell["mapping_integrity"] == "pass"
        assert cell["source_households"] == rounded_count(len(TOY_HOUSEHOLDS))
    assert out["k_growth_draws_nested"] == {
        law: {"lower": [1, 3], "high": UK_LOCAL_CLONE_COUNT, "nested": True}
        for law in LAWS
    }
    conditioned = [cell for cell in out["cells"] if cell["law"] == "conditioned"]
    levels = {case.level for case in TOY_HOUSEHOLDS}
    for cell in conditioned:
        by_level = cell["source_households_by_fallback_level"]
        assert set(by_level) == {"L0", "L1", "L2", "L3"} == levels
        assert by_level == {
            level: rounded_count(sum(case.level == level for case in TOY_HOUSEHOLDS))
            for level in by_level
        }
    # A clone drawn at a lower K whose area moved at the highest is seen.
    draws = {1: {"a": "x"}, 3: {"a": "x", "b": "y"}, 15: {"a": "z", "b": "y"}}
    assert harness._nested(draws) == {"lower": [1, 3], "high": 15, "nested": False}


# --- disclosure: the rule the committed #931 evidence was written under ---------


def test_breach_tables_and_disclosure_suppression(harness, laws) -> None:
    _, _, measured = _measured(
        harness, laws, "region_only", 1, households=TOY_HOUSEHOLDS[:6]
    )
    roster = harness.support_roster(laws.supports["conditioned"])
    summaries = harness.realized_area_support(measured, roster)
    breaches = harness.breach_tables(summaries, {})
    assert set(breaches["by_geography_level"]) == {"constituency", "la"}
    assert breaches["by_geography_level"]["la"]["breaches"]["any"] >= 1
    assert harness._suppress(2.0, 3) == "<3"
    assert harness._suppress(0.0, 3) == 0.0
    assert harness._suppress(3.0, 3) == 3.0
    # z is a function of the suppressed count, so it is blanked with it
    # (#932 round 1: the counts were recoverable from expected_rows and z).
    row = {"rows": 2.0, "ess": 1.5, "sources": 2.0, "z": 0.3, "expected_rows": 1.0}
    assert harness._suppressed_area(row, 3) == {
        **row,
        "rows": "<3",
        "ess": "<3",
        "sources": "<3",
        "z": None,
    }
    assert harness._suppressed_area({**row, "rows": 0.0}, 3)["z"] == 0.3


def _level_areas(level, counts, expected):
    return [
        {
            "level": level,
            "area_code": f"{level}-{index:03d}",
            "region_code": "E12000001",
            "expected_rows": expected,
            "rows": float(count),
            "ess": float(count),
            "sources": float(count),
            "z": 0.1,
        }
        for index, count in enumerate(counts)
    ]


def test_complementary_suppression_leaves_no_suppressed_count_determined(harness):
    # Round 2 of microcosm#1057, item 8: with zeros published and the level total
    # known (the expected rows sum to it), two suppressed counts of 2 were pinned
    # by the residual. The complement adds an unknown of at least minimum + 1.
    counts = [2, 2, 0, 3, 7, 12, 30]
    areas = [
        harness._suppressed_area(row, 3)
        for row in _level_areas("constituency", counts, sum(counts) / len(counts))
    ]
    protected = harness._complement_suppressed(areas, 3)
    withheld = [area for area in protected if area["rows"] == harness.WITHHELD]
    assert len(withheld) == 1
    assert withheld[0]["z"] is None and withheld[0]["ess"] == harness.WITHHELD
    assert harness._complement_suppressed(protected, 3) == protected
    total = sum(area["expected_rows"] for area in protected)
    published = sum(a["rows"] for a in protected if not isinstance(a["rows"], str))
    suppressed = sum(1 for a in protected if a["rows"] == "<3")
    residual = total - published
    # The reader's feasible sums for the suppressed counts span more than one value.
    assert min(2 * suppressed, residual - 4) >= suppressed + 1


def test_resuppression_withholds_summaries_that_carry_a_suppressed_value(harness):
    counts = [1, 2, 9, 10, 11]
    areas = [
        harness._suppressed_area({**row, "z": float(index)}, 3)
        for index, row in enumerate(_level_areas("la", counts, 6.0))
    ]
    evidence = {
        "minimum_count": 3,
        "cells": [
            {
                "areas": areas,
                "levels": {
                    "la": {
                        "min_rows": 1.0,
                        "min_ess": 1.0,
                        "min_sources": 1.0,
                        "max_abs_z": 4.0,
                        "share_abs_z_gt_3": 0.2,
                    }
                },
                "breaches": {"la": {"rows": {"min": 1.0, "p10": 1.0, "median": 9.0}}},
            }
        ],
    }
    out = harness.resuppress_evidence(evidence)
    level = out["cells"][0]["levels"]["la"]
    assert level["min_rows"] is None and level["min_sources"] is None
    # The largest |z| belonged to a suppressed or withheld area; the summary is
    # over published areas only.
    published_z = [abs(a["z"]) for a in out["cells"][0]["areas"] if a["z"] is not None]
    assert level["max_abs_z"] == max(published_z)
    assert out["cells"][0]["breaches"]["la"]["rows"] == {
        "min": None,
        "p10": None,
        "median": 9.0,
    }
    assert harness.resuppress_evidence(out) == out


# --- disclosure: rule 2, which the tool writes -----------------------------------


def _areas(level, region, counts, expected, *, start=0, sources=None):
    return [
        {
            "level": level,
            "area_code": f"{region}-{level}-{start + index:03d}",
            "region_code": region,
            "expected_rows": expected,
            "rows": float(count),
            "ess": float(count),
            "sources": float(count if sources is None else sources[index]),
            "z": 0.5 + index,
        }
        for index, count in enumerate(counts)
    ]


def _rule_2(harness, areas, *, ni=None, breaches=None) -> dict:
    return {
        "disclosure_rule": harness.DISCLOSURE_RULE,
        "minimum_count": 3,
        "cells": [
            {
                "areas": areas,
                "levels": {},
                "breaches": breaches or {},
                "ni_lgd_placements": ni,
            }
        ],
    }


def _published(areas, level):
    return [
        area
        for area in areas
        if area["level"] == level
        and all(_number(area[field]) for field in ("rows", "ess", "sources"))
    ]


def test_rule_2_withholds_within_each_level_and_region(harness) -> None:
    england = _areas("la", "E12000001", [2, 0, 7, 12, 30], 10.2)
    wales = _areas("la", "W99999999", [1, 2, 0], 1.0, start=10)
    scotland = _areas("la", "S99999999", [5, 8], 6.5, start=20)
    constituencies = _areas("constituency", "E12000001", [20, 31], 25.5, start=30)
    evidence = _rule_2(
        harness,
        england + wales + scotland + constituencies,
        breaches={"la": {"rows": {"min": 0.0, "p10": 1.0, "median": 6.0}}},
    )
    out = harness.resuppress_evidence(evidence)
    (cell,) = out["cells"]
    by_region = {}
    for area in cell["areas"]:
        by_region.setdefault((area["level"], area["region_code"]), []).append(area)
    rows = {key: [area["rows"] for area in areas] for key, areas in by_region.items()}
    # England: the suppressed count, then one complement among those above the
    # minimum; its zero stays published.
    assert rows[("la", "E12000001")].count("<3") == 1
    assert rows[("la", "E12000001")].count(harness.WITHHELD) == 1
    assert 0.0 in rows[("la", "E12000001")]
    # Wales has no count above the minimum: every count there goes, zeros too.
    assert rows[("la", "W99999999")] == [harness.WITHHELD] * 3
    # Scotland and the constituencies hold no suppressed count and are untouched.
    assert rows[("la", "S99999999")] == [5.0, 8.0]
    assert rows[("constituency", "E12000001")] == [20.0, 31.0]
    for area in cell["areas"]:
        if isinstance(area["rows"], str):
            assert area["z"] is None and area["ess"] == area["sources"] == area["rows"]
    la = cell["levels"]["la"]
    assert la["min_rows"] is None and la["min_ess"] is None and la["areas"] == 10
    published = [
        abs(area["z"])
        for area in cell["areas"]
        if area["level"] == "la"
        and area["z"] is not None
        and area["expected_rows"] >= 5
    ]
    assert la["max_abs_z"] == max(published)
    assert la["areas_with_expected_ge_5"] == len(published)
    assert cell["levels"]["constituency"]["min_rows"] == 20.0
    # The breach tables are rebuilt over the published areas: the cell's own
    # table is not kept.
    shown = pd.Series([area["rows"] for area in _published(cell["areas"], "la")])
    breaches = cell["breaches"]["la"]
    assert breaches["rows"] == {
        "min": None,
        "p10": pytest.approx(float(shown.quantile(0.1))),
        "median": pytest.approx(float(shown.median())),
    }
    assert breaches["n_areas"] == 10
    assert breaches["unpublished_areas"] == 10 - len(shown)
    assert cell["breaches"]["constituency"]["rows"]["min"] == 20.0
    assert harness.resuppress_evidence(out) == out


def test_rule_2_withholds_rows_that_rest_on_too_few_source_households(
    harness,
) -> None:
    # Rows resting on fewer source households than the minimum are withheld
    # whatever their number: a household's clones are one household.
    row = {"rows": 6.0, "ess": 1.5, "sources": 2.0, "z": 0.3, "expected_rows": 1.0}
    assert harness._suppressed_area(row, 3) == {
        **row,
        "rows": "<3",
        "ess": "<3",
        "sources": "<3",
        "z": None,
    }
    assert harness._suppressed_area({**row, "rows": 3.0, "sources": 3.0}, 3) == {
        **row,
        "rows": 3.0,
        "sources": 3.0,
        "ess": "<3",
    }
    # Three clones of two households fill an area with six rows.
    areas = _areas("la", "E12000007", [6, 9, 15, 18], 12.0, sources=[2, 4, 5, 6])
    (cell,) = harness.resuppress_evidence(_rule_2(harness, areas))["cells"]
    first, *rest = cell["areas"]
    assert first["rows"] == first["sources"] == first["ess"] == "<3"
    assert first["z"] is None
    assert sum(area["rows"] == harness.WITHHELD for area in rest) == 1
    # The relative error and the z summary leave out every withheld area.
    big = _areas("la", "E12000007", [2, 150, 160], 150.0, start=10)
    (cell,) = harness.resuppress_evidence(_rule_2(harness, big))["cells"]
    published = [area for area in cell["areas"] if not isinstance(area["rows"], str)]
    assert len(published) == 1
    level = cell["levels"]["la"]
    assert level["areas_with_expected_ge_100"] == 1
    assert level["max_relative_error_expected_ge_100"] == pytest.approx(
        abs(published[0]["rows"] - 150.0) / 150.0
    )


def test_rule_2_withholds_a_region_whose_total_would_pin_its_suppressed_counts(
    harness,
) -> None:
    # The total (the expected rows' sum) less the published zero leaves 5 for
    # a count of 1 or 2 beside a complement above 3: only 1 + 4 fits.
    pinned = _areas("la", "E12000002", [1, 0, 4], 5.0 / 3)
    (cell,) = harness.resuppress_evidence(_rule_2(harness, pinned))["cells"]
    assert [area["rows"] for area in cell["areas"]] == [harness.WITHHELD] * 3
    # With 7 left, 1 + 6 and 2 + 5 both fit: one complement is enough.
    loose = _areas("la", "E12000002", [1, 0, 6], 7.0 / 3)
    (cell,) = harness.resuppress_evidence(_rule_2(harness, loose))["cells"]
    assert [area["rows"] for area in cell["areas"]] == ["<3", 0.0, harness.WITHHELD]
    assert harness.resuppress_evidence(_rule_2(harness, loose)) == (
        harness.resuppress_evidence(
            harness.resuppress_evidence(_rule_2(harness, loose))
        )
    )
    # A region's rows, where the cell records them, are the total the audit
    # reads, as the conditioned law's withheld expected rows give none.
    evidence = _rule_2(harness, _areas("la", "E12000002", [1, 0, 4], None))
    evidence["cells"][0]["region_rows"] = {"E12000002": 5.0}
    (cell,) = harness.resuppress_evidence(evidence)["cells"]
    assert [area["rows"] for area in cell["areas"]] == [harness.WITHHELD] * 3


def test_northern_ireland_districts_withhold_what_their_level_withholds(
    harness,
) -> None:
    areas = _areas("la", "N99999999", [2, 9, 12, 0], 5.75) + _areas(
        "la", "E12000001", [4, 5], 4.5, start=10
    )
    codes = [area["area_code"] for area in areas[:4]]
    block = {
        "region": "NORTHERN_IRELAND",
        "level": "la",
        "source_households": 20,
        "districts": [
            {"area_code": code, "census_share": share, "weighted_share": weighted}
            for code, share, weighted in zip(
                codes, (0.1, 0.4, 0.4, 0.1), (0.08, 0.39, 0.53, 0.0), strict=True
            )
        ],
    }
    (cell,) = harness.resuppress_evidence(_rule_2(harness, areas, ni=block))["cells"]
    districts = cell["ni_lgd_placements"]["districts"]
    la = {area["area_code"]: area for area in cell["areas"]}
    for district in districts:
        area = la[district["area_code"]]
        assert {key: district[key] for key in ("expected_rows", "rows", "sources")} == {
            key: area[key] for key in ("expected_rows", "rows", "sources")
        }
        assert district["z"] == area["z"]
        if isinstance(area["rows"], str):
            assert district["weighted_share"] == area["rows"]
    assert [district["census_share"] for district in districts] == [0.1, 0.4, 0.4, 0.1]
    marked = [d for d in districts if isinstance(d["weighted_share"], str)]
    assert [d["weighted_share"] for d in marked].count("<3") == 1
    assert [d["weighted_share"] for d in marked].count(harness.WITHHELD) == 1
    published = [d for d in districts if not isinstance(d["rows"], str)]
    assert cell["ni_lgd_placements"]["max_abs_share_gap"] == pytest.approx(
        max(abs(d["weighted_share"] - d["census_share"]) for d in published)
    )
    assert harness.resuppress_evidence(
        harness.resuppress_evidence(_rule_2(harness, areas, ni=block))
    ) == harness.resuppress_evidence(_rule_2(harness, areas, ni=block))


def _breach_summary(codes, rows, sources):
    return pd.DataFrame(
        {
            "area_code": codes,
            "assigned_households": np.asarray(rows, dtype=np.int64),
            "nonzero_households": np.asarray(rows, dtype=np.int64),
            "nonzero_source_households": np.asarray(sources, dtype=np.int64),
            "weight_sum": np.asarray(rows, dtype=float),
            "max_weight": np.ones(len(codes)),
            "effective_sample_size": np.asarray(sources, dtype=float),
        }
    )


def _breach_cell(harness, rows, sources):
    """A cell written as the tool writes one, on invented local authorities of
    one region whose first code, in the complement's order, holds the first
    count; one constituency holds every row."""

    region = "E12000007"
    codes = sorted(
        (f"E0900{index:04d}" for index in range(100, 140)),
        key=lambda code: hashlib.sha256(f"la:{code}".encode()).hexdigest(),
    )[: len(rows)]
    total = sum(rows)
    expected = pd.DataFrame(
        [
            {
                "area_type": "la",
                "area_code": code,
                "expected_rows": total / len(codes),
                "variance": 1.0,
            }
            for code in codes
        ]
        + [
            {
                "area_type": "constituency",
                "area_code": "E14099999",
                "expected_rows": float(total),
                "variance": 0.0,
            }
        ]
    )
    summaries = {
        "constituency": _breach_summary(["E14099999"], [total], [max(sources) + 10]),
        "la": _breach_summary(codes, rows, sources),
    }
    regions = {
        "constituency_code": {"E14099999": region},
        "local_authority_code": dict.fromkeys(codes, region),
    }
    cell = {
        "levels": {},
        "breaches": harness.breach_tables(summaries, {})["by_geography_level"],
        "areas": [
            harness._area_record(row, M, {region: 1000})
            for row in harness._cell_rows(expected, summaries, regions)
        ],
        "ni_lgd_placements": None,
    }
    harness._resuppress_cell(cell, M)
    return cell


def test_rule_2_breach_tables_cover_published_areas_only(harness) -> None:
    # A source-suppressed area (6 rows on 2 households, as at K = 3) and its
    # complement (15) beside published 9, 18 and 40: every written breach
    # statistic is the published areas' own.
    cell = _breach_cell(harness, [15, 6, 9, 18, 40], [15, 2, 9, 18, 40])
    la = [area for area in cell["areas"] if area["level"] == "la"]
    assert [area["rows"] for area in la].count(harness.WITHHELD) == 1
    for field in ("rows", "ess", "sources"):
        shown = pd.Series([area[field] for area in _published(la, "la")], dtype=float)
        written = cell["breaches"]["la"][field]
        assert written["min"] is None
        assert written["p10"] == pytest.approx(float(shown.quantile(0.1)))
        assert written["median"] == pytest.approx(float(shown.median()))
    # One row beside a complement of 50 and a published 60: the breach codes
    # and counts must not say on which side of 50 the complement lies.
    cell = _breach_cell(harness, [50, 1, 60], [50, 1, 60])
    la = [area for area in cell["areas"] if area["level"] == "la"]
    unpublished = {area["area_code"] for area in la if isinstance(area["rows"], str)}
    breaches = cell["breaches"]["la"]["breaches"]
    assert len(unpublished) == 2 and not unpublished & set(breaches["codes"])
    assert breaches["rows_lt_50"] == sum(
        1 for area in _published(la, "la") if area["rows"] < 50
    )
    assert cell["breaches"]["la"]["unpublished_areas"] == 2
    assert harness.published_breach_tables(cell["areas"]) == cell["breaches"]


def _twins(laws) -> dict:
    """Each area whose atomic areas, on the public supports, are exactly those
    of an area at the other level, with that area's code."""

    atoms: dict[tuple[str, str], set] = {}
    for system, support in laws.supports["conditioned"].items():
        arrays = support.arrays
        for area, la, pcon in zip(
            np.asarray(arrays["area"]).astype(str),
            np.asarray(arrays["local_authority_code"]).astype(str),
            np.asarray(arrays["constituency_code"]).astype(str),
            strict=True,
        ):
            atoms.setdefault(("la", la), set()).add((system, area))
            atoms.setdefault(("constituency", pcon), set()).add((system, area))
    return {
        (level, code): other_code
        for (level, code), held in atoms.items()
        for (other, other_code), other_held in atoms.items()
        if other != level and other_held == held
    }


def test_rule_2_withholds_constituencies_where_a_piece_is_small(harness, laws) -> None:
    # The pieces: two rows of two households, three of three, six of two.
    pieces = pd.DataFrame(
        {
            "region_code": ["R1"] * 2 + ["R2"] * 3 + ["R3"] * 6,
            "constituency_code": ["C1"] * 2 + ["C2"] * 3 + ["C3"] * 6,
            "local_authority_code": ["L1"] * 2 + ["L2"] * 3 + ["L3"] * 6,
            SOURCE: [1, 2, 3, 4, 5, 6, 7, 6, 7, 6, 7],
            WEIGHT: [1.0] * 11,
        }
    )
    assert harness._unsafe_piece_regions(pieces, M) == ["R1", "R3"]
    linked = _areas("constituency", "E12000001", [20, 31], 25.5) + _areas(
        "la", "E12000001", [21, 30], 25.5, start=5
    )
    other = _areas("constituency", "E12000002", [10, 11], 10.5, start=10)
    evidence = _rule_2(harness, linked + other)
    evidence["cells"][0]["constituencies_withheld_in"] = ["E12000001"]
    (cell,) = harness.resuppress_evidence(evidence)["cells"]
    rows = {area["area_code"]: area["rows"] for area in cell["areas"]}
    assert [rows[area["area_code"]] for area in linked] == [
        harness.WITHHELD,
        harness.WITHHELD,
        21.0,
        30.0,
    ]
    assert [rows[area["area_code"]] for area in other] == [10.0, 11.0]
    # A reader holding a written cell and the public twin map reads no
    # suppressed count back through the other level: each withheld complement
    # off its published twin, then the count as the region's total less the
    # published counts and the complement.
    households = tuple(
        ToyHousehold("LONDON", 0, "D", False, "L3") for _ in range(11)
    ) + tuple(ToyHousehold("NORTH_EAST", 0, "D", False, "L3") for _ in range(3))
    base = harness.spine_base(toy_spine_frame(clones=1, households=households))
    out = harness.measure(base, laws, n_clones=(1,), exclusions={})
    twin = _twins(laws)
    recovered = []
    for cell in out["cells"]:
        rows = {(a["level"], a["area_code"]): a["rows"] for a in cell["areas"]}
        for area in cell["areas"]:
            if (
                area["level"] == "constituency"
                and area["region_code"] in cell["constituencies_withheld_in"]
            ):
                assert area["rows"] == harness.WITHHELD
        groups: dict[tuple, list] = {}
        for area in cell["areas"]:
            groups.setdefault((area["level"], area["region_code"]), []).append(area)
        for (level, region), group in groups.items():
            suppressed = [a for a in group if a["rows"] == f"<{M}"]
            withheld = [a for a in group if a["rows"] == harness.WITHHELD]
            if len(suppressed) != 1 or len(withheld) != 1:
                continue
            other_level = "constituency" if level == "la" else "la"
            code = twin.get((level, withheld[0]["area_code"]))
            if code is None or isinstance(rows[(other_level, code)], str):
                continue
            total = cell["region_rows"][region]
            published = sum(a["rows"] for a in group if _number(a["rows"]))
            recovered.append(total - published - rows[(other_level, code)])
    assert not recovered


def test_the_conditioned_laws_expectations_feed_its_summaries_only(
    harness, laws, monkeypatch
) -> None:
    base = harness.spine_base(toy_spine_frame(clones=1, households=POOL))
    out = harness.measure(base, laws, n_clones=(3,), exclusions={})
    region_only, conditioned = out["cells"]
    assert region_only["expectation"] == harness.PER_AREA
    assert any(_number(area["expected_rows"]) for area in region_only["areas"])
    assert conditioned["expectation"] == harness.SUMMARIES_ONLY
    assert all(
        area["expected_rows"] is None and area["z"] is None
        for area in conditioned["areas"]
    )
    assert all(
        district["expected_rows"] is None and district["z"] is None
        for district in conditioned["ni_lgd_placements"]["districts"]
    )
    # The summaries are the ones its published areas gave before the
    # expectations were withheld.
    with monkeypatch.context() as patched:
        patched.setattr(harness, "_withhold_expectations", lambda cell: None)
        (whole,) = harness.measure(
            base, laws, law_names=("conditioned",), n_clones=(3,), exclusions={}
        )["cells"]
    assert conditioned["levels"] == whole["levels"]
    assert any(summary["max_abs_z"] is not None for summary in whole["levels"].values())
    assert (
        conditioned["ni_lgd_placements"]["max_abs_z"]
        == whole["ni_lgd_placements"]["max_abs_z"]
    )
    # A written file is a fixed point; where a rewrite withholds more of a
    # level, its kept summaries are withheld with it.
    evidence = {
        "disclosure_rule": harness.DISCLOSURE_RULE,
        "minimum_count": M,
        "cells": out["cells"],
    }
    assert harness.resuppress_evidence(evidence) == evidence
    assert conditioned["levels"]["constituency"]["areas_with_expected_ge_5"] > 0
    edited = json.loads(json.dumps(evidence))
    cell = edited["cells"][1]
    region = next(
        area["region_code"]
        for area in cell["areas"]
        if area["level"] == "constituency"
        and _number(area["rows"])
        and area["rows"] > 0
    )
    cell["constituencies_withheld_in"] = sorted(
        {*cell["constituencies_withheld_in"], region}
    )
    (_, rewritten) = harness.resuppress_evidence(edited)["cells"]
    assert all(
        rewritten["levels"]["constituency"][key] is None
        for key in harness._EXPECTATION_SUMMARY_FIELDS
    )
    assert rewritten["levels"]["la"] == conditioned["levels"]["la"]
    assert (
        rewritten["ni_lgd_placements"]["max_abs_z"]
        == conditioned["ni_lgd_placements"]["max_abs_z"]
    )


def test_no_written_expected_rows_give_back_draw_cell_counts(harness, laws) -> None:
    """Northern Ireland's district block: from its written expected rows and
    the public NI support, the households in each draw cell (L1 by decile,
    L3) must not come back as one integer solution."""

    support = laws.supports["conditioned"][NI].arrays
    deciles = np.asarray(support["imd_decile"]).astype(str)
    districts = np.asarray(support["local_authority_code"]).astype(str)
    households = np.asarray(support["households"], dtype=float)
    codes = sorted(set(districts.tolist()))

    def shares(mask):
        weights = households * mask
        return (
            np.asarray([weights[districts == code].sum() for code in codes])
            / weights.sum()
        )

    columns = [shares(deciles == decile) for decile in sorted(set(deciles.tolist()))]
    columns.append(shares(np.ones_like(households, dtype=bool)))
    matrix = np.vstack(columns).T
    base = harness.spine_base(toy_spine_frame(clones=1))
    out = harness.measure(base, laws, n_clones=(1,), exclusions={})
    for cell in out["cells"]:
        expected = [d["expected_rows"] for d in cell["ni_lgd_placements"]["districts"]]
        if cell["law"] == "conditioned":
            assert expected == [None] * len(codes)
            continue
        total = round(sum(expected))
        solutions = [
            counts
            for counts in itertools.product(range(total + 1), repeat=len(columns))
            if sum(counts) == total
            and np.allclose(matrix @ np.asarray(counts, dtype=float), expected)
        ]
        assert len(solutions) != 1, solutions


def _clone_cell(law, k, region, rows, sources):
    return {
        "law": law,
        "n_clones": k,
        "areas": _areas("la", region, rows, sum(rows) / len(rows), sources=sources),
        "levels": {},
        "breaches": {},
        "ni_lgd_placements": None,
    }


def test_rule_2_reads_the_laws_other_clone_counts_as_bounds(harness) -> None:
    # At one clone, a count of 2 beside a complement of 4 is not pinned by the
    # region's total: 1 + 5 and 2 + 4 both fit beside the published 0 and 3.
    low = _clone_cell("region_only", 1, "E12000001", [2, 4, 0, 3], [2, 4, 0, 3])
    (alone,) = harness.resuppress_evidence(
        {"disclosure_rule": 2, "minimum_count": M, "cells": [low]}
    )["cells"]
    assert [area["rows"] for area in alone["areas"]] == [
        "<3",
        harness.WITHHELD,
        0.0,
        3.0,
    ]
    # At two clones the complement's area holds 4 rows, so at one it held at
    # most 4: with a total of 9, the count is 2. The group is withheld whole.
    high = _clone_cell("region_only", 2, "E12000001", [3, 4, 0, 9], [3, 4, 0, 9])
    evidence = {"disclosure_rule": 2, "minimum_count": M, "cells": [low, high]}
    out = harness.resuppress_evidence(evidence)
    assert [area["rows"] for area in out["cells"][0]["areas"]] == [harness.WITHHELD] * 4
    assert [area["rows"] for area in out["cells"][1]["areas"]] == [3.0, 4.0, 0.0, 9.0]
    assert harness.resuppress_evidence(out) == out
    # An area published alike at a lower and a higher K pins its count between
    # them even when every count of the group is withheld: the group goes in
    # the law's other cells too, and only there.
    cells = [
        _clone_cell("region_only", 1, "E12000002", [0, 5], [0, 5]),
        _clone_cell("region_only", 2, "E12000002", [2, 5], [2, 5]),
        _clone_cell("region_only", 3, "E12000002", [4, 5], [4, 5]),
        _clone_cell("conditioned", 3, "E12000002", [4, 5], [4, 5]),
    ]
    out = harness.resuppress_evidence(
        {"disclosure_rule": 2, "minimum_count": M, "cells": cells}
    )
    assert [[area["rows"] for area in cell["areas"]] for cell in out["cells"]] == [
        [harness.WITHHELD] * 2
    ] * 3 + [[4.0, 5.0]]
    assert harness.resuppress_evidence(out) == out


def test_a_region_on_too_few_households_withholds_its_expected_rows(
    harness, laws
) -> None:
    # Yorkshire holds two invented households, four rows at two clones.
    base = harness.spine_base(toy_spine_frame(clones=1))
    out = harness.measure(base, laws, n_clones=(2,), exclusions={})
    for cell in out["cells"]:
        yorkshire = [a for a in cell["areas"] if a["region_code"] == "E12000003"]
        assert yorkshire
        marker = f"<{M}" if cell["law"] == "region_only" else None
        assert all(area["expected_rows"] == marker for area in yorkshire)
        assert all(isinstance(area["rows"], str) for area in yorkshire)
        assert cell["region_rows"]["E12000003"] == f"<{M}"


def test_londons_share_and_its_check_are_withheld_on_too_few_households(
    harness, laws
) -> None:
    households = (
        ToyHousehold("LONDON", 0, "D", False, "L3"),
        ToyHousehold("LONDON", 0, "D", False, "L3"),
    ) + tuple(ToyHousehold("NORTH_EAST", 0, "D", False, "L3") for _ in range(4))
    base = harness.spine_base(toy_spine_frame(clones=1, households=households))
    out = harness.measure(base, laws, n_clones=(2,), exclusions={})
    for cell in out["cells"]:
        # The verdict leaves out the London check, whose outcome would say on
        # which side of its bounds the withheld share lies.
        assert cell["ladder_gate"] == {
            "outcome": "pass",
            "london_weighted_household_share": f"<{M}",
            "london_share_checked": False,
        }
    base = harness.spine_base(toy_spine_frame(clones=1))
    (cell, _) = harness.measure(base, laws, n_clones=(1,), exclusions={})["cells"]
    assert cell["ladder_gate"]["london_share_checked"] is True
    assert _number(cell["ladder_gate"]["london_weighted_household_share"])


def test_the_ni_block_counts_households_not_clone_rows(harness, laws) -> None:
    base = harness.spine_base(toy_spine_frame(clones=1))
    out = harness.measure(base, laws, n_clones=(3,), exclusions={})
    households = sum(case.region == "NORTHERN_IRELAND" for case in TOY_HOUSEHOLDS)
    for cell in out["cells"]:
        assert cell["ni_lgd_placements"]["source_households"] == rounded_count(
            households
        )


def test_area_sources_count_each_households_clones_once(harness, laws) -> None:
    base = harness.spine_base(toy_spine_frame(clones=1, households=POOL))
    out = harness.measure(base, laws, n_clones=(3,), exclusions={})
    for cell in out["cells"]:
        table = harness.expand_base(base, 3)
        measured = pd.concat([table, laws.draw(cell["law"], table).columns], axis=1)
        counted = {}
        for level, column in harness.LEVELS:
            for code, group in measured.groupby(measured[column].astype(str)):
                positive = group[group[WEIGHT] > 0]
                counted[(level, code)] = (len(group), positive[SOURCE].nunique())
        for area in cell["areas"]:
            rows, sources = counted.get((area["level"], area["area_code"]), (0, 0))
            if _number(area["sources"]):
                assert area["sources"] == sources, (cell["law"], area["area_code"])
            if rows and sources < M:
                assert area["rows"] in (f"<{M}", harness.WITHHELD), area


def test_a_higher_minimum_withholds_every_count_below_it(harness, laws) -> None:
    minimum = 5
    base = harness.spine_base(toy_spine_frame(clones=1))
    out = harness.measure(base, laws, n_clones=(1,), exclusions={}, minimum=minimum)
    markers = {f"<{minimum}", harness.WITHHELD}
    for cell in out["cells"]:
        # Northern Ireland holds four households.
        assert cell["ni_lgd_placements"]["source_households"] == f"<{minimum}"
        for area in cell["areas"]:
            for field in ("rows", "sources"):
                value = area[field]
                assert (
                    value in markers
                    if isinstance(value, str)
                    else (value == 0 or value >= minimum)
                )
    by_level = {
        level: sum(case.level == level for case in TOY_HOUSEHOLDS)
        for level in ("L0", "L1", "L2", "L3")
    }
    assert out["cells"][1]["source_households_by_fallback_level"] == {
        level: f"<{minimum}" if 0 < count < minimum else rounded_count(count)
        for level, count in by_level.items()
    }
    evidence = {"disclosure_rule": 2, "minimum_count": minimum, "cells": out["cells"]}
    assert harness.resuppress_evidence(evidence) == evidence


def test_an_unknown_rule_or_a_low_minimum_is_refused(harness) -> None:
    with pytest.raises(ValueError, match="Unknown disclosure_rule 3"):
        harness.resuppress_evidence({"disclosure_rule": 3, "minimum_count": 3})
    with pytest.raises(ValueError, match="minimum_count of at least 3"):
        harness.resuppress_evidence(
            {"disclosure_rule": 2, "minimum_count": 2, "cells": []}
        )


def test_an_area_in_two_regions_is_refused(harness, laws) -> None:
    assert harness._area_regions(harness.support_roster(laws.supports["conditioned"]))
    straddling = SimpleNamespace(
        region_code=np.asarray(["E12000001", "E12000002"]),
        constituency_code=np.asarray(["E14000001", "E14000001"]),
        local_authority_code=np.asarray(["E06000001", "E06000002"]),
    )
    with pytest.raises(ValueError, match="constituency area in more than one"):
        harness._area_regions(straddling)


# --- the command ----------------------------------------------------------------


def _write_supports(directory: Path, payloads: dict[str, bytes]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for system, payload in payloads.items():
        (directory / f"{CONDITIONED_SOURCES[system]}.npz").write_bytes(payload)
    return directory


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _counts_follow_the_rule(evidence: dict) -> None:
    minimum = evidence["minimum_count"]
    markers = {f"<{minimum}", "withheld"}
    for cell in evidence["cells"]:
        for area in cell["areas"]:
            for field in ("rows", "ess", "sources"):
                value = area[field]
                if isinstance(value, str):
                    assert value in markers
                elif field != "ess":
                    assert value == 0 or value >= minimum
            if isinstance(area["rows"], str):
                assert area["z"] is None
            assert SOURCE not in area and IDENTITY_COLUMN not in area
        for district in cell["ni_lgd_placements"]["districts"]:
            if isinstance(district["rows"], str):
                assert district["weighted_share"] in markers


def test_the_command_measures_a_bound_spine_into_rule_2_evidence(
    harness, payloads, tmp_path, monkeypatch, capsys
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    h5 = run / "spine.h5"
    write_uk_national_frame(toy_spine_frame(clones=2), h5)
    supports = _write_supports(tmp_path / "supports", payloads)
    calls = []

    def bound(path, frame, *, gate_report_path=None, input_sha256=None):
        calls.append((Path(path), Path(gate_report_path), input_sha256))
        return {
            "geographic_clone_count": 2,
            "sampling": {
                "fraction": 0.1,
                "seed": 5,
                "rung_token": "f010",
                "pre_household_count": 1234,
                "post_household_count": 123,
            },
        }

    monkeypatch.setattr(harness, "load_bound_spine_checkpoint", bound)
    evidence_out, receipt_out = run / "cells.json", run / "receipt.json"
    harness.main(
        [
            *("--input-h5", str(h5)),
            *("--supports-dir", str(supports)),
            "--no-pin-supports",
            *("--n-clones", "1,2"),
            *("--evidence-out", str(evidence_out)),
            *("--receipt-out", str(receipt_out)),
        ]
    )
    assert calls == [
        (
            h5.with_suffix(".build.json"),
            h5.with_suffix(".spine_gates.json"),
            _sha256(h5),
        )
    ]
    evidence = json.loads(evidence_out.read_text(encoding="utf-8"))
    assert evidence["disclosure_rule"] == harness.DISCLOSURE_RULE
    assert evidence["publishable"] is False
    assert evidence["spine"] == {
        "file": "spine.h5",
        "sha256": _sha256(h5),
        "geographic_clone_count": 2,
        "source_households": rounded_count(len(TOY_HOUSEHOLDS)),
        "sampling": {"fraction": 0.1, "seed": 5, "rung_token": "f010"},
        "carries_areas": False,
    }
    assert evidence["supports"]["pinned"] is False
    assert evidence["supports"]["conditioned"] == {
        system: hashlib.sha256(payload).hexdigest()
        for system, payload in payloads.items()
    }
    assert [(cell["law"], cell["n_clones"]) for cell in evidence["cells"]] == [
        (law, k) for k in (1, 2) for law in LAWS
    ]
    assert evidence["definitions"]["conditioned"]["stream"][1:] == [
        "uk-spine-atomic-area-v1",
        0,
        UK_SPINE_ATOMIC_AREA_SEED,
    ]
    _counts_follow_the_rule(evidence)
    assert harness.resuppress_evidence(evidence) == evidence
    for cell in evidence["cells"]:
        assert cell["expectation"] == (
            harness.SUMMARIES_ONLY if cell["law"] == "conditioned" else harness.PER_AREA
        )
        if cell["law"] == "conditioned":
            assert all(area["expected_rows"] is None for area in cell["areas"])
        placements = cell["ni_lgd_placements"]
        assert placements["region"] == "NORTHERN_IRELAND"
        assert [d["area_code"] for d in placements["districts"]] == sorted(
            {"N09000003", "N09000011"}
        )
        assert sum(d["census_share"] for d in placements["districts"]) == (
            pytest.approx(1.0)
        )
    receipt = json.loads(receipt_out.read_text(encoding="utf-8"))
    assert receipt == harness.receipt_of(evidence, _sha256(evidence_out))
    assert receipt["publishable"] is False
    # The terminal shows code facts and rounded counts only.
    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == (
        f"spine: K=2, {rounded_count(len(TOY_HOUSEHOLDS))} source households"
    )
    assert all(line.startswith("  cell law=") for line in lines[1:-1])
    assert lines[-1] == f"evidence: {evidence_out.resolve()}"


def test_the_receipt_bands_minima_below_ten(harness) -> None:
    summary = {
        "areas": 4,
        "min_rows": 5.0,
        "min_ess": 4.5,
        "min_sources": 3.0,
        "max_abs_z": 1.2,
        "share_abs_z_gt_3": 0.0,
        "areas_with_expected_ge_5": 2,
        "max_relative_error_expected_ge_100": None,
        "areas_with_expected_ge_100": 0,
    }
    cell = {
        "law": "region_only",
        "n_clones": 1,
        "source_households": 20,
        "mapping_integrity": "pass",
        "ladder_gate": {"outcome": "pass"},
        "levels": {"la": summary, "constituency": {**summary, "min_rows": 12.0}},
        "identity_stability": {},
        "source_households_by_fallback_level": None,
        "ni_lgd_placements": {
            "source_households": "3-9",
            "max_abs_z": None,
            "max_abs_share_gap": 0.1,
        },
    }
    evidence = {
        "disclosure_rule": 2,
        "spine": {},
        "cells": [cell],
        "k_growth_draws_nested": {},
    }
    receipt = harness.receipt_of(evidence, "0" * 64)
    assert receipt["publishable"] is False
    levels = receipt["cells"][0]["levels"]
    assert levels["la"]["min_rows"] == levels["la"]["min_sources"] == UK_BELOW_TEN_BAND
    assert levels["la"]["min_ess"] == 4.5
    assert levels["constituency"]["min_rows"] == 12.0
    assert levels["constituency"]["min_sources"] == UK_BELOW_TEN_BAND
    assert evidence["cells"][0]["levels"]["la"]["min_rows"] == 5.0


def test_refusals_name_checks_not_counts(harness, payloads, tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    frame = toy_spine_frame(clones=2)
    h5 = run / "spine.h5"
    write_uk_national_frame(frame, h5)
    supports = _write_supports(tmp_path / "supports", payloads)
    h5.with_suffix(".build.json").write_text(
        json.dumps({"entity_row_counts": {"person": 1, "benunit": 1, "household": 1}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not bind: input H5 build sidecar") as e:
        harness.main(
            [
                *("--input-h5", str(h5)),
                *("--supports-dir", str(supports)),
                "--no-pin-supports",
                *("--evidence-out", str(run / "cells.json")),
            ]
        )
    message = str(e.value)
    assert "row-count mismatch" in message
    for entity in frame.entities:
        assert str(len(frame.table(entity))) not in message
    assert e.value.__suppress_context__ and not (run / "cells.json").exists()
    # The geography identity counts the households it refuses; the tool names
    # the step.
    stale = dataclasses.replace(harness.spine_base(frame), time_period="1999")
    with pytest.raises(ValueError, match="geography identity refused") as e:
        harness.expand_base(stale, 2)
    assert not any(character.isdigit() for character in str(e.value))


def test_the_command_writes_nothing_inside_a_checkout(harness, tmp_path) -> None:
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    absent = tmp_path / "absent"
    for outputs in (
        ("--evidence-out", str(checkout / "cells.json")),
        (
            *("--evidence-out", str(tmp_path / "cells.json")),
            *("--receipt-out", str(checkout / "out" / "receipt.json")),
        ),
    ):
        with pytest.raises(SystemExit) as refused:
            harness.main(
                [
                    *("--input-h5", str(absent / "spine.h5")),
                    *("--supports-dir", str(absent / "supports")),
                    *outputs,
                ]
            )
        assert "outside every checkout" in str(refused.value)
    assert sorted(path.name for path in checkout.iterdir()) == [".git"]
    assert not absent.exists() and not (tmp_path / "cells.json").exists()


def test_the_supports_are_held_to_their_registers(
    harness, payloads, tmp_path, monkeypatch
) -> None:
    supports = _write_supports(tmp_path / "supports", payloads)
    with pytest.raises(ValueError, match="not the register's") as refused:
        harness.load_supports(supports, pin=True)
    assert "--no-pin-supports" in str(refused.value)
    absent = tmp_path / "absent"
    with pytest.raises(ValueError, match="not the register's"):
        harness.main(
            [
                *("--input-h5", str(absent / "spine.h5")),
                *("--supports-dir", str(supports)),
                *("--evidence-out", str(tmp_path / "cells.json")),
            ]
        )
    assert not (tmp_path / "cells.json").exists()
    base = harness.base_supports(payloads)

    def register(*, conditioned: bool = False) -> dict:
        held = payloads if conditioned else base
        return {
            system: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for system, payload in held.items()
        }

    monkeypatch.setattr(harness, "uk_atomic_support_register", register)
    assert harness.load_supports(supports, pin=True) == (payloads, base)
    (supports / f"{CONDITIONED_SOURCES[SYSTEMS[2]]}.npz").unlink()
    with pytest.raises(ValueError, match="holds no uk_ni_data_zone_2021_conditioned"):
        harness.load_supports(supports, pin=False)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--minimum-count", "2"), "at least 3"),
        (("--laws", "legacy"), "--laws must name"),
        (("--n-clones", "1,1"), "distinct clone counts"),
        (("--n-clones", "0"), "distinct clone counts"),
        (("--n-clones", "1,x"), "comma-separated"),
    ],
)
def test_the_command_line_is_held_to_its_rules(harness, capsys, extra, message):
    with pytest.raises(SystemExit):
        harness._parse_args(
            [
                *("--input-h5", "x.h5"),
                *("--supports-dir", "s"),
                *("--evidence-out", "e.json"),
                *extra,
            ]
        )
    assert message in capsys.readouterr().err
    args = harness._parse_args(
        ["--input-h5", "x.h5", "--supports-dir", "s", "--evidence-out", "e.json"]
    )
    assert args.n_clones == (1, UK_LOCAL_CLONE_COUNT)
    assert args.laws == LAWS and args.minimum_count == UK_DISCLOSURE_FLOOR


def test_the_tool_reads_no_retired_dense_node() -> None:
    source = _TOOL.read_text(encoding="utf-8")
    assert "uk.full." not in source and "ContentStore" not in source


# --- the committed #931 evidence, written under rule 1 ----------------------------


def test_committed_931_evidence_is_internally_consistent(harness):
    root = paths_for("microcosm-build").repository
    path = root / "docs/evidence/uk-931/atomic-assignment-cells.json"
    if not path.is_file():
        pytest.skip("committed #931 evidence is not present in this checkout")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["minimum_count"] >= 3
    assert set(evidence["supports"]) == set(SYSTEMS)
    assert evidence["pool"]["sample_fraction"] <= 0.01
    assert {(c["law"], c["n_clones"]) for c in evidence["cells"]} == {
        (law, k) for law in evidence["laws"] for k in evidence["n_clones"]
    }
    for cell in evidence["cells"]:
        assert cell["gate_outcome"] == "pass"
        for area in cell["areas"]:
            for field in ("rows", "ess", "sources"):
                value = area[field]
                if isinstance(value, str):
                    assert value in {f"<{evidence['minimum_count']}", harness.WITHHELD}
                else:
                    assert value == 0 or value >= evidence["minimum_count"]
            if isinstance(area["rows"], str):
                assert area["z"] is None
            assert "household_id" not in area and "source_household_id" not in area
        if cell["law"] == "keyed":
            assert all(cell["identity_stability"].values())
        minimum = evidence["minimum_count"]
        for level, summary in cell["levels"].items():
            areas = [area for area in cell["areas"] if area["level"] == level]
            suppressed = sum(1 for area in areas if area["rows"] == f"<{minimum}")
            if not suppressed:
                continue
            # One complement per level, and no summary carries a suppressed value.
            assert sum(1 for area in areas if area["rows"] == harness.WITHHELD) == 1
            assert summary["min_rows"] is None and summary["min_sources"] is None
            # A reader who knows the level total (the expected rows sum to it)
            # and the rule can bound the suppressed counts' sum, but not pin it.
            total = round(sum(area["expected_rows"] for area in areas))
            published = sum(
                area["rows"] for area in areas if not isinstance(area["rows"], str)
            )
            highest = min((minimum - 1) * suppressed, total - published - minimum - 1)
            assert highest >= suppressed + 1, (cell["law"], cell["n_clones"], level)
    # The committed file is a fixed point of the harness's disclosure pass.
    assert harness.resuppress_evidence(evidence) == evidence
    assert np.isfinite(sum(c["wall_seconds"] for c in evidence["cells"]))


def test_committed_931_receipts_render_the_level_summaries_from_the_evidence(harness):
    # Round 2 of the microcosm#1059 review, item 7: a hand-copied receipts table
    # printed the K=15 constituency minimum that the evidence withholds, which
    # pinned the suppressed counts again. The R1 table's summary cells must be the
    # harness's rendering of the committed evidence.
    root = paths_for("microcosm-build").repository
    evidence_path = root / "docs/evidence/uk-931/atomic-assignment-cells.json"
    receipts_path = root / "experiments/931-uk-atomic-assignment-receipts.md"
    if not evidence_path.is_file() or not receipts_path.is_file():
        pytest.skip("committed #931 evidence or receipts are not present")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    section = receipts_path.read_text(encoding="utf-8").split("## R1", 1)[1]
    section = section.split("## R2", 1)[0]
    table = [line for line in section.splitlines() if line.startswith("|")]
    header = [cell.strip() for cell in table[0].strip("|").split("|")]
    columns = {
        ("constituency", "min"): header.index("const. min rows / ESS / sources"),
        ("constituency", "z"): header.index(
            "const. max abs z / share abs z>3 (n published)"
        ),
        ("la", "min"): header.index("LA min rows / ESS / sources"),
        ("la", "z"): header.index("LA max abs z / share>3 (n published)"),
    }
    cells = {(cell["law"], cell["n_clones"]): cell for cell in evidence["cells"]}
    rows = [[cell.strip() for cell in line.strip("|").split("|")] for line in table[2:]]
    assert {(row[0], int(row[1])) for row in rows} == set(cells)
    for row in rows:
        cell = cells[(row[0], int(row[1]))]
        for level in ("constituency", "la"):
            minima, z = harness.level_summary_cells(cell, level)
            assert row[columns[(level, "min")]] == minima, (row[:2], level)
            assert row[columns[(level, "z")]] == z, (row[:2], level)
