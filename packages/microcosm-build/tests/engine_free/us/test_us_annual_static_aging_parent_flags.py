"""The annual static-aging lane refuses a parent that lacks its local flags.

The lane copies its parent's base year byte for byte and keeps the parent's
columns in every projected year, so a local-jurisdiction flag the parent lacks
stays absent from every annual file and the engine defaults it to ``False``
for every resident. The lane checks the parent before the model provenance
check, and these tests stop every build there, so they need no engine.

Invariants:

- a parent that lacks a flag its own geography decides, or carries one that
  disagrees with that geography, fails before the model provenance check and
  before the output directory exists;
- a parent that carries exactly the flags its geography decides reaches the
  model provenance check, and its receipt lists those flags as carried;
- a parent with no county decides no flag, so it needs none.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
import pytest

from microcosm.build import us_annual_static_aging as annual
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.frame.materialize import put_frame_table, read_frame_table
from test_support.microcosm_build.us_local_jurisdictions import FIXTURE_BLOCK_BY_NAME

# One household per block, in this order; each household has one person.
BLOCKS = ("yonkers", "denver", "philadelphia", "wilmington", "aurora_arapahoe")
# The flags these blocks decide, written out independently of the registry.
FLAGS = {
    "in_yonkers": ("household", [True, False, False, False, False]),
    "in_denver": ("household", [False, True, False, False, False]),
    "in_wilmington": ("household", [False, False, False, True, False]),
    "pa_philadelphia_wage_tax_resident": (
        "person",
        [False, False, True, False, False],
    ),
}
COUNTY_FLAGS = ("in_denver", "pa_philadelphia_wage_tax_resident")


class _ReachedModelProvenanceError(Exception):
    """Stands in for the model provenance check: the parent passed."""


def _tables(
    *, geography: str, flags: Mapping[str, Sequence[bool]]
) -> dict[str, pd.DataFrame]:
    """A five-household parent on ``BLOCKS`` with the given flag columns.

    ``geography`` is ``block`` (county, place and block on every row),
    ``county`` (a county, with the place and block an ACS row leaves empty)
    or ``none`` (a state only).
    """

    ids = list(range(1, len(BLOCKS) + 1))
    tables = {
        entity: pd.DataFrame({US_SCHEMA.entity_id_column(entity): ids})
        for entity in US_SCHEMA.entities
    }
    for group in US_SCHEMA.group_entities:
        tables["person"][US_SCHEMA.membership_column(group)] = ids
    tables["person"]["age"] = [40] * len(ids)
    tables["person"]["is_female"] = [index % 2 == 0 for index in ids]
    household = tables["household"]
    household["household_weight"] = [float(index) for index in ids]
    blocks = [FIXTURE_BLOCK_BY_NAME[name] for name in BLOCKS]
    household["state_fips"] = [block.state_fips for block in blocks]
    if geography in {"block", "county"}:
        household["county_fips"] = [block.county_fips for block in blocks]
        household["place_fips"] = [
            ""
            if geography == "county" or block.place_fips == 0
            else f"{block.place_fips:05d}"
            for block in blocks
        ]
        household["block_geoid"] = [
            "" if geography == "county" else f"{block.block_geoid:015d}"
            for block in blocks
        ]
    for flag, values in flags.items():
        entity = FLAGS[flag][0]
        tables[entity][flag] = list(values)
    return tables


def _write_parent(path: Path, tables: Mapping[str, pd.DataFrame]) -> Path:
    with pd.HDFStore(path, "w") as store:
        for entity, table in tables.items():
            put_frame_table(
                store, entity, table, preferred_format="table", data_columns=True
            )
        store.put("_time_period", pd.Series([2024]), format="table")
    return path


def _inputs(tmp_path: Path, tables: Mapping[str, pd.DataFrame]) -> dict:
    base = _write_parent(tmp_path / "base.h5", tables)
    ssa = tmp_path / "ssa.csv"
    pd.DataFrame(
        {"Year": [2024, 2025], "Age": [40, 40], "M Tot": [4, 5], "F Tot": [2, 3]}
    ).to_csv(ssa, index=False)
    return dict(
        base_h5=base,
        base_sha256=annual._sha256(base),
        parent_release="pinned-parent",
        ssa_csv=ssa,
        ssa_sha256=annual._sha256(ssa),
        model_source=tmp_path,
        model_commit="a" * 40,
        model_version="test",
        output_dir=tmp_path / "candidate",
        base_year=2024,
        end_year=2025,
        epochs=1,
    )


def _complete(geography: str) -> dict[str, list[bool]]:
    decided = {"block": tuple(FLAGS), "county": COUNTY_FLAGS, "none": ()}[geography]
    return {flag: FLAGS[flag][1] for flag in decided}


def _flipped(flags: Mapping[str, list[bool]], flag: str) -> dict[str, list[bool]]:
    return {**flags, flag: [not value for value in flags[flag]]}


def _without(flags: Mapping[str, list[bool]], *names: str) -> dict[str, list[bool]]:
    return {flag: values for flag, values in flags.items() if flag not in names}


@pytest.mark.parametrize(
    ("geography", "flags", "failure"),
    [
        pytest.param(
            "block", {}, "in_yonkers: absent", id="block-parent-without-flags"
        ),
        pytest.param(
            "block",
            _without(_complete("block"), "in_yonkers", "in_wilmington"),
            "in_wilmington: absent",
            id="block-parent-with-county-flags-only",
        ),
        pytest.param(
            "block",
            _flipped(_complete("block"), "in_yonkers"),
            "in_yonkers: 5/5 household row(s) disagree",
            id="block-parent-with-a-wrong-place-flag",
        ),
        pytest.param(
            "block",
            _flipped(_complete("block"), "pa_philadelphia_wage_tax_resident"),
            "pa_philadelphia_wage_tax_resident: 5/5 person row(s) disagree",
            id="block-parent-with-a-wrong-person-flag",
        ),
        pytest.param(
            "county", {}, "in_denver: absent", id="county-parent-without-flags"
        ),
        pytest.param(
            "county",
            {**_complete("county"), "in_yonkers": FLAGS["in_yonkers"][1]},
            "in_yonkers: present, but this artifact's rungs ['county'] do not",
            id="county-parent-with-a-place-flag-it-cannot-decide",
        ),
    ],
)
def test_parent_without_its_flags_is_refused_before_model_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    geography: str,
    flags: dict[str, list[bool]],
    failure: str,
) -> None:
    inputs = _inputs(tmp_path, _tables(geography=geography, flags=flags))
    monkeypatch.setattr(
        annual, "_model_provenance", lambda **kwargs: pytest.fail("must refuse first")
    )
    monkeypatch.setattr(
        annual, "_runtime_provenance", lambda: pytest.fail("must refuse first")
    )

    with pytest.raises(
        ValueError, match="needs a parent that already carries"
    ) as error:
        annual.build_annual_static_aging(**inputs)

    assert failure in str(error.value)
    assert not inputs["output_dir"].exists()


@pytest.mark.parametrize("geography", ["block", "county", "none"])
def test_parent_carrying_its_flags_reaches_model_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, geography: str
) -> None:
    inputs = _inputs(tmp_path, _tables(geography=geography, flags=_complete(geography)))

    def reached(**kwargs: object) -> dict:
        raise _ReachedModelProvenanceError

    monkeypatch.setattr(annual, "_model_provenance", reached)

    with pytest.raises(_ReachedModelProvenanceError):
        annual.build_annual_static_aging(**inputs)

    assert not inputs["output_dir"].exists()


@pytest.mark.parametrize("geography", ["block", "county", "none"])
def test_parent_receipt_lists_the_flags_it_carries(
    tmp_path: Path, geography: str
) -> None:
    expected = _complete(geography)
    path = _write_parent(
        tmp_path / "base.h5", _tables(geography=geography, flags=expected)
    )
    with pd.HDFStore(path, "r") as store:
        tables = {
            entity: read_frame_table(store, entity) for entity in US_SCHEMA.entities
        }
    weights = tables["household"].pop("household_weight").to_numpy(dtype=float)
    frame = Frame(
        tables, US_SCHEMA, {"household": Weights(weights, WeightKind.CALIBRATED)}
    )

    receipt = annual._parent_local_jurisdiction_flags(frame)

    assert receipt["carried_flags"] == sorted(expected)
    assert (
        receipt["rungs"]
        == {
            "block": ["county", "place"],
            "county": ["county"],
            "none": [],
        }[geography]
    )
    for flag, (_entity, values) in FLAGS.items():
        entry = receipt["flags"][flag]
        if flag in expected:
            assert entry["status"] == "derived"
            assert entry["households_in_jurisdiction"] == sum(values)
        else:
            assert entry["status"] == "unavailable"
