"""Shared fixtures for the US local-jurisdiction flag tests.

A synthetic block ladder whose blocks sit in the registry jurisdictions, in a
neighbour of each, and in near misses (a registry place code in another
state), plus a small US frame builder. Each block has its own congressional
district, so assigning a household to that district pins it to that block and
the ladder assignment is deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.geography_ladder import UsBlockLadder
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


@dataclass(frozen=True)
class FixtureBlock:
    name: str
    block_geoid: int
    place_fips: int
    congressional_district_geoid: int
    population: float

    @property
    def state_fips(self) -> int:
        return self.block_geoid // 10**13

    @property
    def county_fips(self) -> str:
        return f"{self.block_geoid:015d}"[:5]


FIXTURE_BLOCKS: tuple[FixtureBlock, ...] = (
    FixtureBlock("denver", 80310001001000, 20000, 801, 700.0),
    FixtureBlock("aurora_arapahoe", 80050001001000, 4000, 806, 300.0),
    FixtureBlock("yonkers", 361190001001000, 84000, 3616, 210.0),
    FixtureBlock("new_rochelle_westchester", 361190002001000, 50617, 3617, 80.0),
    FixtureBlock("wilmington", 100030001001000, 77580, 1000, 70.0),
    FixtureBlock("new_castle_unincorporated", 100030002001000, 0, 1001, 50.0),
    FixtureBlock("philadelphia", 421010001001000, 60000, 4203, 1600.0),
    FixtureBlock("delaware_county_pa", 420450001001000, 0, 4205, 500.0),
    # Near misses: a registry place code in another state.
    FixtureBlock("place_84000_in_california", 60370001001000, 84000, 630, 40.0),
    FixtureBlock("place_77580_in_north_carolina", 371290001001000, 77580, 3707, 30.0),
)
FIXTURE_BLOCK_BY_NAME = {block.name: block for block in FIXTURE_BLOCKS}
WAGE_INPUT = "employment_income_before_lsr"


def block_ladder_metadata() -> dict[str, object]:
    layers = {
        "congressional_district": {"vintage": "119th_congress"},
        "sldu": {"vintage": "2020_baf"},
        "sldl": {"vintage": "2020_baf"},
        "place": {"vintage": "2020_census"},
        "cbsa": {"vintage": "omb_2023_delineations"},
    }
    return {
        "schema_version": 1,
        "kind": "us_block_ladder",
        "block_vintage": "2020_tabulation_blocks",
        "sampling_basis": "population",
        "layers": layers,
    }


def block_ladder(blocks: Sequence[FixtureBlock] = FIXTURE_BLOCKS) -> UsBlockLadder:
    """A ladder holding ``blocks``, one congressional district per block."""

    count = len(blocks)
    return UsBlockLadder(
        block_geoid=np.asarray([block.block_geoid for block in blocks], dtype=np.int64),
        population=np.asarray([block.population for block in blocks], dtype=np.float64),
        congressional_district_geoid=np.asarray(
            [block.congressional_district_geoid for block in blocks], dtype=np.int64
        ),
        sldu=np.asarray(["001"] * count, dtype="U3"),
        sldl=np.asarray(["001"] * count, dtype="U3"),
        place_fips=np.asarray([block.place_fips for block in blocks], dtype=np.int32),
        cbsa_code=np.asarray([0] * count, dtype=np.int32),
        metadata=block_ladder_metadata(),
    )


def us_frame(
    households: Sequence[Mapping[str, object]],
    *,
    person_inputs: Sequence[Sequence[Mapping[str, object]]] | None = None,
    weights: Sequence[float] | None = None,
) -> Frame:
    """A US frame with one group unit of each kind per household.

    ``households`` gives each household's columns (``household_id`` is
    assigned 1..n when absent); ``person_inputs`` gives each household's
    people, one mapping of person columns each (one empty person by default).
    Every person has ``employment_income_before_lsr`` (0 unless given), as
    every built frame does, so the local-tax inputs can be derived.
    """

    household = pd.DataFrame(list(households))
    if "household_id" not in household:
        household.insert(0, "household_id", np.arange(1, len(household) + 1))
    people = person_inputs or [[{}] for _ in range(len(household))]
    if len(people) != len(household):
        raise ValueError("person_inputs must give one list per household.")
    rows: list[dict[str, object]] = []
    for household_id, members in zip(household["household_id"], people, strict=True):
        for member in members:
            row: dict[str, object] = {}
            for group in US_SCHEMA.group_entities:
                row[US_SCHEMA.membership_column(group)] = int(household_id)
            row[WAGE_INPUT] = 0.0
            row.update(member)
            rows.append(row)
    person = pd.DataFrame(rows)
    person.insert(0, US_SCHEMA.person_id_column, np.arange(1, len(person) + 1))
    tables: dict[str, pd.DataFrame] = {"person": person, "household": household}
    for group in US_SCHEMA.group_entities:
        if group == "household":
            continue
        tables[group] = pd.DataFrame(
            {US_SCHEMA.id_column(group): household["household_id"].to_numpy()}
        )
    values = np.ones(len(household)) if weights is None else np.asarray(weights)
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=values.astype(np.float64), kind=WeightKind.DESIGN
            )
        },
    )


def households_on_blocks(
    names: Sequence[str],
) -> list[dict[str, object]]:
    """Household rows whose district pins each to the named fixture block."""

    return [
        {
            "state_fips": FIXTURE_BLOCK_BY_NAME[name].state_fips,
            "congressional_district_geoid": (
                FIXTURE_BLOCK_BY_NAME[name].congressional_district_geoid
            ),
        }
        for name in names
    ]
