"""The household measures sidecar: batching, keying, coverage and its receipt.

Engine-free: a fake ``Microsimulation`` (``test_support.microcosm_build.
us_county_files``) gives every household known values, so each invariant is
checked for generated populations, batch sizes and row orders.
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import json
import math
import os
import struct
import sys

import numpy as np
import pandas as pd
import pytest

from test_support.microcosm_build.us_county_files import (
    FakeDataset,
    expected_household_values,
    fake_calculator,
    fake_microsimulation_class,
    fake_system,
    load_tool,
    population_tables,
    write_population_h5,
)
from test_support.paths import paths_for

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

_SETTINGS = settings(
    max_examples=40,
    deadline=None,
    derandomize=True,
    # Generation is slow only on a loaded host (too_slow fired at load 200).
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
    ],
)


@pytest.fixture(scope="module")
def tool():
    return load_tool("us_household_measures")


@st.composite
def _populations(draw, max_households: int = 24):
    sizes = draw(st.lists(st.integers(1, 6), min_size=1, max_size=max_households))
    splits = draw(st.lists(st.booleans(), min_size=len(sizes), max_size=len(sizes)))
    n_persons = sum(sizes)
    household_order = draw(st.permutations(range(len(sizes))))
    person_order = draw(st.permutations(range(n_persons)))
    seed = draw(st.integers(0, 2**16))
    year = draw(st.sampled_from([2026, 2027]))
    batch = draw(st.integers(1, len(sizes) + 3))
    id_base = draw(st.integers(-2000, 2000))
    tables = population_tables(
        sizes,
        seed=seed,
        tax_unit_splits=splits,
        household_order=household_order,
        person_order=person_order,
        id_base=id_base,
        shuffle_groups=True,
    )
    return tables, year, batch


def _valid_sidecar(tool, tables, year):
    calculate, _, _ = fake_calculator(tool, multi_batch=False)
    table, _ = tool.compute_household_measures(
        tables, year, calculate=calculate, batch_households=len(tables["household"])
    )
    return table


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


@_SETTINGS
@given(_populations(), st.booleans())
@example(
    (
        population_tables(
            [2, 3, 1, 4],
            id_base=-31,
            seed=109,
            tax_unit_splits=[True] * 4,
            household_order=[2, 0, 3, 1],
            person_order=[9, 4, 0, 7, 2, 8, 1, 6, 3, 5],
            shuffle_groups=True,
        ),
        2027,
        1,
    ),
    True,
)
def test_every_household_once_in_its_own_order_whatever_the_batches(
    tool, population, reverse
):
    tables, year, batch = population
    n = len(tables["household"])
    calculate, cls, run = fake_calculator(
        tool, multi_batch=n > batch, reverse_households=reverse
    )
    table, info = tool.compute_household_measures(
        tables, year, calculate=calculate, batch_households=batch
    )

    household_ids = tables["household"]["household_id"].to_numpy()
    assert table["household_id"].tolist() == household_ids.tolist()
    assert table["household_id"].is_unique
    assert (table["year"] == year).all()
    assert info["batches"] == math.ceil(n / batch) == run.batches == len(cls.instances)
    assert info["checks"]["every_household_exactly_once"] is True
    for instance in cls.instances:
        assert instance.spm == {"geography_kind": "county"}
        assert instance.year == year

    # Batch invariance: each household gets its whole-file value.
    expected = expected_household_values(tables, year).loc[household_ids]
    for name in tool.COMPONENT_VARIABLES:
        np.testing.assert_array_equal(table[name].to_numpy(), expected[name].to_numpy())
    np.testing.assert_array_equal(
        table["net_income"].to_numpy(), expected["household_net_income"].to_numpy()
    )
    earnings = sum(expected[name].to_numpy() for name in tool.EARNINGS_VARIABLES)
    np.testing.assert_allclose(
        table["earnings"].to_numpy(), earnings, rtol=0, atol=1e-6
    )
    np.testing.assert_array_equal(
        table["gross_income"].to_numpy(),
        (
            expected["household_market_income"] + expected["household_benefits"]
        ).to_numpy(),
    )


@_SETTINGS
@given(_populations())
def test_batches_partition_every_entity_into_whole_households(tool, population):
    tables, _, batch = population
    seen = {entity: [] for entity in tool.US_ENTITIES}
    for part in tool.household_batches(tables, batch):
        households = set(part["household"]["household_id"])
        assert 1 <= len(households) <= batch
        assert set(part["person"]["person_household_id"]) <= households
        for group in tool.US_ENTITIES[2:]:
            members = set(part["person"][f"person_{group}_id"])
            assert set(part[group][f"{group}_id"]) == members
        for entity in tool.US_ENTITIES:
            seen[entity].extend(part[entity].to_dict("records"))
    for entity in tool.US_ENTITIES:
        key = list(tables[entity].columns)
        got = pd.DataFrame(seen[entity], columns=key).sort_values(key[0])
        want = tables[entity].sort_values(key[0])
        pd.testing.assert_frame_equal(
            got.reset_index(drop=True), want.reset_index(drop=True), check_dtype=False
        )


@_SETTINGS
@given(st.integers(2, 16), st.integers(0, 2**16))
def test_compute_refuses_batches_that_change_household_order(
    tool, monkeypatch, count, seed
):
    tables = population_tables([1] * count, seed=seed, shuffle_groups=True)
    batcher = tool.household_batches

    def reversed_batches(tables, batch_households):
        yield from reversed(list(batcher(tables, batch_households)))

    calculate, _, _ = fake_calculator(tool, multi_batch=True)
    with monkeypatch.context() as patch:
        patch.setattr(tool, "household_batches", reversed_batches)
        with pytest.raises(tool.HouseholdMeasuresError, match="its own order"):
            tool.compute_household_measures(
                tables, 2026, calculate=calculate, batch_households=count - 1
            )


@_SETTINGS
@given(
    sizes=st.lists(st.integers(1, 6), min_size=2, max_size=8),
    seed=st.integers(0, 2**16),
)
@pytest.mark.parametrize("group", ["tax_unit", "spm_unit", "family", "marital_unit"])
def test_a_group_spanning_two_households_refuses(tool, sizes, seed, group):
    tables = population_tables(sizes, seed=seed, shuffle_groups=True)
    person = tables["person"]
    person.loc[person.index[-1], f"person_{group}_id"] = person.loc[
        person.index[0], f"person_{group}_id"
    ]
    tables[group] = pd.DataFrame(
        {f"{group}_id": pd.unique(person[f"person_{group}_id"])}
    )
    with pytest.raises(tool.HouseholdMeasuresError, match="span more than one"):
        list(tool.household_batches(tables, 1))


def _orphan_person_household(tables):
    person = tables["person"]
    person["person_household_id"] = person["person_household_id"].replace({100: 99_999})


def _orphan_family(tables):
    tables["family"] = pd.concat([tables["family"], pd.DataFrame({"family_id": [5]})])


def _drop_spm_unit_row(tables):
    tables["spm_unit"] = tables["spm_unit"].iloc[1:]


def _repeat_household(tables):
    household = tables["household"]
    tables["household"] = pd.concat([household, household.iloc[:1]])


def _memberless_household(tables):
    household = tables["household"]
    tables["household"] = pd.concat(
        [household.iloc[:1].assign(household_id=101), household]
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_orphan_person_household, "names no household row"),
        (_orphan_family, "has no member person"),
        (_drop_spm_unit_row, "names no spm_unit row"),
        (_repeat_household, "appears more than once"),
        (_memberless_household, "have no member person"),
    ],
)
def test_batching_refuses_a_broken_membership(tool, mutate, message):
    tables = population_tables([1, 3, 2])
    mutate(tables)
    with pytest.raises(tool.HouseholdMeasuresError, match=message):
        list(tool.household_batches(tables, 2))


def test_batch_size_must_be_a_positive_integer(tool):
    tables = population_tables([1])
    for bad in (0, -1, 1.5, True):
        with pytest.raises(tool.HouseholdMeasuresError, match="positive integer"):
            list(tool.household_batches(tables, bad))


@_SETTINGS
@given(
    _populations(), st.sampled_from(["tax_unit", "spm_unit", "family", "marital_unit"])
)
@example((population_tables([1], id_base=-1, shuffle_groups=True), 2026, 1), "tax_unit")
@example((population_tables([1], id_base=-1, shuffle_groups=True), 2026, 1), "spm_unit")
@example((population_tables([1], id_base=-1, shuffle_groups=True), 2026, 1), "family")
@example(
    (population_tables([1], id_base=-1, shuffle_groups=True), 2026, 1), "marital_unit"
)
def test_every_group_id_must_be_unique(tool, population, group):
    tables, _, batch = population
    frame = tables[group]
    tables[group] = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    with pytest.raises(tool.HouseholdMeasuresError, match="appears more than once"):
        list(tool.household_batches(tables, batch))


@_SETTINGS
@given(st.integers(2, 12), st.integers(0, 2**16), st.booleans())
def test_medicaid_claims_must_stay_inside_the_household_when_batched(
    tool, count, seed, unresolved
):
    tables = population_tables([2] * count, seed=seed, tax_unit_splits=[True] * count)
    person = tables["person"]
    # Positive local claims are safe, including a different tax unit at home.
    local = person.groupby("person_household_id")["person_tax_unit_id"].transform(
        "last"
    )
    person["medicaid_claiming_tax_unit_id"] = local
    assert len(list(tool.household_batches(tables, 1))) == count
    # Nonpositive values are the engine's unknown-claim sentinels.
    person["medicaid_claiming_tax_unit_id"] = np.where(
        np.arange(len(person)) % 2, -1, 0
    )
    assert len(list(tool.household_batches(tables, 1))) == count
    person.loc[0, "medicaid_claiming_tax_unit_id"] = (
        int(person["person_tax_unit_id"].max()) + 1
        if unresolved
        else int(person.loc[2, "person_tax_unit_id"])
    )
    with pytest.raises(
        tool.HouseholdMeasuresError, match="medicaid_claiming_tax_unit_id"
    ):
        list(tool.household_batches(tables, count - 1))
    # Whole-file scoring still has the same lookup universe as the engine.
    assert len(list(tool.household_batches(tables, count))) == 1


@_SETTINGS
@given(st.integers(-10000, 10000), st.sampled_from([0.25, 0.5, 0.75]))
def test_nonintegral_household_ids_refuse_before_scoring(tool, base, fraction):
    tables = population_tables([2, 1], id_base=base)
    for frame, column in (
        (tables["household"], "household_id"),
        (tables["person"], "person_household_id"),
    ):
        frame[column] = frame[column].astype("float64") + fraction
    with pytest.raises(tool.HouseholdMeasuresError, match="integer|integral|int64"):
        tool.compute_household_measures(
            tables,
            2026,
            calculate=lambda *args: pytest.fail("calculated invalid ids"),
            batch_households=1,
        )


# ---------------------------------------------------------------------------
# The engine seam
# ---------------------------------------------------------------------------


def test_a_household_the_engine_did_not_compute_refuses(tool):
    tables = population_tables([1, 2, 3])
    dropped = int(tables["household"]["household_id"].iloc[1])
    calculate, _, _ = fake_calculator(tool, multi_batch=False, drop_household=dropped)
    with pytest.raises(tool.HouseholdMeasuresError, match="returned 2 households"):
        tool.compute_household_measures(
            tables, 2026, calculate=calculate, batch_households=3
        )


def test_a_multi_batch_engine_refuses_a_population_aggregate(tool):
    tables = population_tables([1, 2, 3])
    calculate, _, run = fake_calculator(
        tool, multi_batch=True, aggregate_on=("household_income_decile",)
    )
    with pytest.raises(tool.HouseholdMeasuresError, match="microcosm#956"):
        tool.compute_household_measures(
            tables, 2026, calculate=calculate, batch_households=1
        )
    assert run.aggregate_variables_computed == ("household_income_decile",)


def test_a_single_batch_may_compute_a_population_aggregate(tool):
    tables = population_tables([1, 2, 3])
    calculate, _, run = fake_calculator(
        tool, multi_batch=False, aggregate_on=("spm_unit_income_decile",)
    )
    table, _ = tool.compute_household_measures(
        tables, 2026, calculate=calculate, batch_households=3
    )
    assert len(table) == 3
    assert run.aggregate_variables_computed == ("spm_unit_income_decile",)


@pytest.mark.parametrize(
    "name",
    [
        "household_income_decile",
        "medicaid_slcsp_state_average_cost_index",
        "medicaid_slcsp_state_denominator",
        "spm_unit_income_decile",
    ],
)
def test_a_guarded_variable_computed_only_in_a_branch_refuses(tool, name):
    calculate, _, _ = fake_calculator(
        tool, multi_batch=True, branch_aggregate_on=(name,)
    )
    with pytest.raises(tool.HouseholdMeasuresError, match="microcosm#956"):
        tool.compute_household_measures(
            population_tables([1, 1]), 2026, calculate=calculate, batch_households=1
        )


@pytest.mark.parametrize("rename", [False, True])
def test_every_guarded_name_must_still_exist(tool, rename):
    for name in tool.POPULATION_AGGREGATE_VARIABLES:
        system = fake_system(drop=[name])
        if rename:
            system.variables[name + "_renamed"] = fake_system().variables[name]
        with pytest.raises(tool.HouseholdMeasuresError, match="defines no variable"):
            tool.check_engine_definitions(system)


def test_the_installed_definitions_are_checked(tool):
    tool.check_engine_definitions(fake_system())
    with pytest.raises(tool.HouseholdMeasuresError, match="adds"):
        tool.check_engine_definitions(
            fake_system(net_income_adds=["household_market_income"])
        )
    with pytest.raises(tool.HouseholdMeasuresError, match="defines no variable"):
        tool.check_engine_definitions(fake_system(drop=["farm_operations_income"]))
    with pytest.raises(tool.HouseholdMeasuresError, match="person variables"):
        tool.check_engine_definitions(
            fake_system(entities={"employment_income": "tax_unit"})
        )
    with pytest.raises(tool.HouseholdMeasuresError, match="household ones"):
        tool.check_engine_definitions(
            fake_system(entities={"household_benefits": "spm_unit"})
        )


@pytest.mark.parametrize(
    "changed",
    [
        {"market_income_adds": "gov.household.another_source"},
        {"market_income_adds": ""},
        {"market_income_subtracts": ["employment_income"]},
        {"net_income_subtracts": []},
        {"net_income_subtracts": ["household_health_costs"]},
        {"drop": ["household_id"]},
    ],
)
def test_each_documented_engine_definition_is_required(tool, changed):
    with pytest.raises(
        tool.HouseholdMeasuresError, match="adds|subtracts|defines no variable"
    ):
        tool.check_engine_definitions(fake_system(**changed))


def test_resolved_parameters_are_recorded(tool):
    resolved = tool.resolved_engine_parameters(fake_system(), 2026)
    assert resolved == {
        "gov.household.market_income_sources": ["employment_income", "dividend_income"],
        "gov.household.household_benefits": ["social_security", "snap"],
        "gov.simulation.include_health_benefits_in_net_income": False,
        "gov.hud.abolition": False,
        "gov.simulation.include_head_start_benefits_in_net_income": False,
    }


# ---------------------------------------------------------------------------
# The coverage check
# ---------------------------------------------------------------------------

_COMPONENTS = (
    "household_market_income",
    "household_benefits",
    "household_refundable_tax_credits",
    "household_tax_before_refundable_credits",
    "household_health_costs",
)
_MUTATIONS = (
    "duplicate",
    "drop",
    "extra",
    "year",
    "nan",
    "net_income",
    "gross_income",
)


def _mutate(table: pd.DataFrame, mutation: str, row: int) -> pd.DataFrame:
    table = table.copy()
    if mutation == "duplicate":
        return pd.concat([table, table.iloc[[row]]], ignore_index=True)
    if mutation == "drop":
        return table.drop(index=table.index[row]).reset_index(drop=True)
    if mutation == "extra":
        extra = table.iloc[[row]].copy()
        extra["household_id"] = int(table["household_id"].max()) + 1
        return pd.concat([table, extra], ignore_index=True)
    if mutation == "year":
        table.loc[table.index[row], "year"] = int(table["year"].iloc[row]) + 1
        return table
    if mutation == "nan":
        table.loc[table.index[row], "earnings"] = np.nan
        return table
    if mutation == "net_income":
        scale = sum(abs(float(table[name].iloc[row])) for name in _COMPONENTS)
        table.loc[table.index[row], "net_income"] += 0.02 + 2e-6 * scale
        return table
    if mutation == "gross_income":
        table.loc[table.index[row], "gross_income"] += 1e-3
        return table
    raise AssertionError(mutation)


@_SETTINGS
@given(_populations(max_households=12), st.sampled_from(_MUTATIONS), st.data())
def test_the_check_refuses_every_coverage_and_identity_break(
    tool, population, mutation, data
):
    tables, year, _ = population
    table = _valid_sidecar(tool, tables, year)
    household_ids = tables["household"]["household_id"].to_numpy()
    summary = tool.check_household_measures(table, household_ids, year)
    assert summary["households"] == summary["rows"] == len(household_ids)

    row = data.draw(st.integers(0, len(table) - 1))
    broken = _mutate(table, mutation, row)
    with pytest.raises(tool.HouseholdMeasuresError):
        tool.check_household_measures(broken, household_ids, year)


_AMOUNTS = st.floats(-2e7, 2e7, allow_nan=False, width=32)


@_SETTINGS
@given(st.lists(st.tuples(*[_AMOUNTS] * 5), min_size=1, max_size=30))
def test_the_identity_allows_the_engines_float32_sum_and_nothing_more(tool, components):
    """policyengine-core adds household_net_income's terms in float32.

    The engine stores float variables as float32 and builds the sum from 0 in
    float32: ``0 + m + b + c - t - h``. Every such sum passes the identity
    check, and moving net income by more than the tolerance fails it.
    """
    values = np.asarray(components, dtype=np.float32)
    net = np.zeros(len(values), dtype=np.float32)
    for column, sign in zip(range(5), (1, 1, 1, -1, -1), strict=True):
        net = net + np.float32(sign) * values[:, column]
    ids = np.arange(1, len(values) + 1, dtype=np.int64) * 3
    engine = {name: np.zeros(len(values)) for name in tool.EARNINGS_VARIABLES}
    engine.update(
        {
            name: values[:, index].astype(np.float64)
            for index, name in enumerate(tool.COMPONENT_VARIABLES)
        }
    )
    engine[tool.NET_INCOME_VARIABLE] = net.astype(np.float64)
    table = tool.assemble_household_measures(ids, 2026, engine)
    summary = tool.check_household_measures(table, ids, 2026)
    scale = np.abs(values.astype(np.float64)).sum(axis=1)
    assert summary["net_income_identity_max_abs_residual"] <= float(
        (3e-7 * scale).max() + 1e-9
    )

    moved = table.copy()
    moved.loc[0, "net_income"] += 0.011 + 1.3e-6 * scale[0]
    with pytest.raises(tool.HouseholdMeasuresError, match="off its components"):
        tool.check_household_measures(moved, ids, 2026)


def test_the_check_refuses_a_year_other_than_the_files(tool):
    tables = population_tables([2, 1])
    table = _valid_sidecar(tool, tables, 2026)
    with pytest.raises(tool.HouseholdMeasuresError, match="not the H5's 2027"):
        tool.check_household_measures(
            table, tables["household"]["household_id"].to_numpy(), 2027
        )


def test_the_check_refuses_other_columns_or_dtypes(tool):
    tables = population_tables([2, 1])
    table = _valid_sidecar(tool, tables, 2026)
    ids = tables["household"]["household_id"].to_numpy()
    with pytest.raises(tool.HouseholdMeasuresError, match="columns"):
        tool.check_household_measures(table.drop(columns="earnings"), ids, 2026)
    with pytest.raises(tool.HouseholdMeasuresError, match="dtype"):
        tool.check_household_measures(table.astype({"year": "float64"}), ids, 2026)


@_SETTINGS
@given(_populations(max_households=6), st.data())
def test_the_check_refuses_every_repeated_source_household_id(tool, population, data):
    tables, year, _ = population
    table = _valid_sidecar(tool, tables, year)
    ids = tables["household"]["household_id"].to_numpy()
    row = data.draw(st.integers(0, len(ids) - 1))
    repeated_ids = np.append(ids, ids[row])
    with pytest.raises(tool.HouseholdMeasuresError, match="repeats a household_id"):
        tool.check_household_measures(table, repeated_ids, year)


# ---------------------------------------------------------------------------
# The sidecar file
# ---------------------------------------------------------------------------


@_SETTINGS
@given(_populations(max_households=16))
def test_the_sidecar_reads_back_exactly_and_byte_reproducibly(
    tool, tmp_path_factory, population
):
    tables, year, _ = population
    table = _valid_sidecar(tool, tables, year)
    directory = tmp_path_factory.mktemp("sidecar")
    first, second = directory / "a.csv.gz", directory / "b.csv.gz"
    tool.write_sidecar(table, first)
    tool.write_sidecar(table, second)
    assert first.read_bytes() == second.read_bytes()
    reread = tool.read_sidecar(first)
    pd.testing.assert_frame_equal(reread, table, check_exact=True)
    # Plain pandas, with no hints, reads the dtypes and values to within one
    # ulp: its default float parser is not round-trip exact.
    plain = pd.read_csv(first)
    assert [str(dtype) for dtype in plain.dtypes] == [
        dtype for _, dtype in tool.SIDECAR_COLUMNS
    ]
    np.testing.assert_allclose(plain.to_numpy(), table.to_numpy(), rtol=1e-15, atol=0)
    with gzip.open(first, "rt") as handle:
        assert handle.readline().strip() == ",".join(
            name for name, _ in tool.SIDECAR_COLUMNS
        )


def test_writing_never_overwrites(tool, tmp_path):
    tables = population_tables([1])
    table = _valid_sidecar(tool, tables, 2026)
    path = tmp_path / "s.csv.gz"
    tool.write_sidecar(table, path)
    with pytest.raises(FileExistsError):
        tool.write_sidecar(table, path)
    assert not list(tmp_path.glob("*.partial"))


def test_gzip_header_has_zero_mtime_and_no_filename(tool, tmp_path):
    table = _valid_sidecar(tool, population_tables([1]), 2026)
    path = tmp_path / "named.csv.gz"
    tool.write_sidecar(table, path)
    header = path.read_bytes()[:10]
    assert header[:3] == b"\x1f\x8b\x08"
    assert header[3] & 0x08 == 0
    assert struct.unpack("<I", header[4:8])[0] == 0


@pytest.mark.parametrize("mask", [0o022, 0o027, 0o077, 0o002])
def test_every_written_file_respects_the_umask(tool, tmp_path, mask):
    table = _valid_sidecar(tool, population_tables([1]), 2026)
    sidecar, receipt = tmp_path / "out.csv.gz", tmp_path / "out.json"
    previous = os.umask(mask)
    try:
        tool.write_sidecar(table, sidecar)
        tool._write_json(receipt, {"kind": "fixture"})
    finally:
        os.umask(previous)
    for path in (sidecar, receipt):
        assert path.stat().st_mode & 0o777 == 0o666 & ~mask


def test_exclusive_writer_refuses_before_calling_the_writer(tool, tmp_path):
    path = tmp_path / "exists"
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        tool._write_exclusive(
            path, lambda target: pytest.fail("writer called for existing output")
        )
    assert path.read_bytes() == b"original"


def test_exclusive_writer_refuses_a_dangling_symlink_before_writing(tool, tmp_path):
    path = tmp_path / "exists"
    target = tmp_path / "absent"
    path.symlink_to(target)
    with pytest.raises(FileExistsError):
        tool._write_exclusive(
            path, lambda output: pytest.fail("writer called for existing symlink")
        )
    assert path.is_symlink()
    assert not target.exists()


@_SETTINGS
@given(st.binary(max_size=256))
def test_temporary_file_creation_never_overwrites_a_colliding_name(
    tool, tmp_path, monkeypatch, contents
):
    path = tmp_path / "out"
    identifier = tool.uuid.UUID(int=1)
    temporary = path.with_name(f".{path.name}.{identifier.hex}.partial")
    temporary.write_bytes(contents)
    with monkeypatch.context() as patch:
        patch.setattr(tool.uuid, "uuid4", lambda: identifier)
        with pytest.raises(FileExistsError):
            tool._write_exclusive(
                path, lambda output: pytest.fail("writer called for existing temporary")
            )
    assert temporary.read_bytes() == contents
    assert not path.exists()


def test_exclusive_link_preserves_a_file_created_during_the_write(tool, tmp_path):
    path = tmp_path / "raced"

    def write(temporary):
        temporary.write_bytes(b"new")
        path.write_bytes(b"other writer")

    with pytest.raises(FileExistsError):
        tool._write_exclusive(path, write)
    assert path.read_bytes() == b"other writer"
    assert not list(tmp_path.glob(".*.partial"))


# ---------------------------------------------------------------------------
# End to end through an H5 and the receipt
# ---------------------------------------------------------------------------


def _fake_engine_factory():
    return fake_microsimulation_class(), FakeDataset, fake_system()


def test_compute_writes_a_sidecar_and_receipt_that_verify(tool, tmp_path):
    tables = population_tables([3, 1, 2, 4, 1], tax_unit_splits=[True] * 5)
    h5 = write_population_h5(tables, tmp_path / "county_2026.h5", year=2026)
    receipt = tool.run_compute(
        h5,
        expect_year=2026,
        batch_households=2,
        engine_factory=_fake_engine_factory,
        log=lambda message: None,
    )
    sidecar, receipt_path = tool.sidecar_paths(h5)
    assert sidecar.name == "county_2026.household_measures.csv.gz"
    assert receipt_path.name == "county_2026.household_measures.json"
    assert json.loads(receipt_path.read_text()) == receipt
    assert receipt["kind"] == tool.SIDECAR_KIND
    assert receipt["year"] == receipt["source_h5"]["time_period"] == 2026
    assert receipt["source_h5"]["sha256"] == hashlib.sha256(h5.read_bytes()).hexdigest()
    assert (
        receipt["sidecar"]["sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    )
    assert receipt["batching"]["batches"] == 3
    assert receipt["engine"]["spm_selection"] == {"geography_kind": "county"}
    assert receipt["measures"]["earnings"]["engine_variables"] == list(
        tool.EARNINGS_VARIABLES
    )
    verified = tool.verify_sidecar(h5)
    assert verified["checks"]["households"] == 5

    with pytest.raises(FileExistsError):
        tool.run_compute(h5, engine_factory=_fake_engine_factory, log=lambda m: None)


def test_compute_refuses_a_file_of_another_year(tool, tmp_path):
    h5 = write_population_h5(population_tables([1, 2]), tmp_path / "x.h5", year=2027)
    with pytest.raises(tool.HouseholdMeasuresError, match="not the expected 2026"):
        tool.run_compute(
            h5, expect_year=2026, engine_factory=_fake_engine_factory, log=print
        )
    assert not any(tmp_path.glob("x.household_measures*"))


@pytest.mark.parametrize("batch", [1, 2, 3])
def test_run_compute_uses_the_exact_multibatch_boundary(tool, tmp_path, batch):
    h5 = write_population_h5(
        population_tables([1, 1]), tmp_path / "boundary.h5", year=2026
    )

    def engine_factory():
        return (
            fake_microsimulation_class(aggregate_on=("household_income_decile",)),
            FakeDataset,
            fake_system(),
        )

    if batch < 2:
        with pytest.raises(tool.HouseholdMeasuresError, match="microcosm#956"):
            tool.run_compute(
                h5,
                batch_households=batch,
                engine_factory=engine_factory,
                log=lambda message: None,
            )
        assert not any(tmp_path.glob("boundary.household_measures*"))
    else:
        receipt = tool.run_compute(
            h5,
            batch_households=batch,
            engine_factory=engine_factory,
            log=lambda message: None,
        )
        assert receipt["engine"]["aggregate_variables_computed"] == [
            "household_income_decile"
        ]


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_invalid_batch_size_refuses_before_loading_any_engine(tool, tmp_path, bad):
    h5 = write_population_h5(population_tables([1]), tmp_path / "batch.h5", year=2026)
    with pytest.raises(tool.HouseholdMeasuresError, match="positive integer"):
        tool.run_compute(
            h5,
            batch_households=bad,
            engine_factory=lambda: pytest.fail("engine loaded for invalid batch size"),
        )


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("dangling", [False, True])
def test_run_compute_refuses_each_existing_output_before_loading_engine(
    tool, tmp_path, which, dangling
):
    h5 = write_population_h5(
        population_tables([1]), tmp_path / "existing.h5", year=2026
    )
    paths = tool.sidecar_paths(h5)
    if dangling:
        paths[which].symlink_to(tmp_path / "absent")
    else:
        paths[which].write_bytes(b"original")
    with pytest.raises(FileExistsError):
        tool.run_compute(
            h5,
            engine_factory=lambda: pytest.fail("engine loaded despite existing output"),
        )
    if dangling:
        assert paths[which].is_symlink()
    else:
        assert paths[which].read_bytes() == b"original"
    assert not paths[1 - which].exists()


def test_compute_refuses_if_the_h5_changes_during_scoring(tool, tmp_path):
    h5 = write_population_h5(
        population_tables([1]), tmp_path / "changing.h5", year=2026
    )

    def engine_factory():
        with h5.open("ab") as handle:
            handle.write(b"changed during scoring")
        return _fake_engine_factory()

    with pytest.raises(
        tool.HouseholdMeasuresError, match="changed while it was scored"
    ):
        tool.run_compute(h5, engine_factory=engine_factory, log=lambda message: None)
    assert not any(tmp_path.glob("changing.household_measures*"))


def test_compute_refuses_and_cleans_up_an_inexact_readback(tool, tmp_path, monkeypatch):
    h5 = write_population_h5(
        population_tables([1]), tmp_path / "readback.h5", year=2026
    )
    real_read = tool.read_sidecar

    def inexact_read(path):
        table = real_read(path)
        table.loc[0, "earnings"] += 1
        return table

    monkeypatch.setattr(tool, "read_sidecar", inexact_read)
    with pytest.raises(tool.HouseholdMeasuresError, match="does not read back exactly"):
        tool.run_compute(
            h5, engine_factory=_fake_engine_factory, log=lambda message: None
        )
    assert not any(tmp_path.glob("readback.household_measures*"))


def test_compute_removes_the_sidecar_if_receipt_writing_fails(
    tool, tmp_path, monkeypatch
):
    h5 = write_population_h5(
        population_tables([1]), tmp_path / "receipt_failure.h5", year=2026
    )

    def fail_receipt(*args):
        raise OSError("fixture receipt failure")

    monkeypatch.setattr(tool, "_write_json", fail_receipt)
    with pytest.raises(OSError, match="fixture receipt failure"):
        tool.run_compute(
            h5, engine_factory=_fake_engine_factory, log=lambda message: None
        )
    assert not any(tmp_path.glob("receipt_failure.household_measures*"))


@pytest.fixture
def computed_sidecar(tool, tmp_path):
    h5 = write_population_h5(
        population_tables([1]), tmp_path / "identity.h5", year=2026
    )
    receipt = tool.run_compute(
        h5, engine_factory=_fake_engine_factory, log=lambda message: None
    )
    _, receipt_path = tool.sidecar_paths(h5)
    return h5, receipt, receipt_path


_JSON_VALUES = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.text(max_size=80),
    st.lists(st.one_of(st.none(), st.integers(), st.text(max_size=20)), max_size=5),
)


@_SETTINGS
@given(value=_JSON_VALUES)
@pytest.mark.parametrize("field", ["kind", "schema_version", "year"])
def test_verify_refuses_each_changed_receipt_identity(
    tool, computed_sidecar, field, value
):
    h5, receipt, receipt_path = computed_sidecar
    assume(value != receipt[field])
    receipt_path.write_text(json.dumps({**receipt, field: value}))
    with pytest.raises(tool.HouseholdMeasuresError, match=f"receipt {field}"):
        tool.verify_sidecar(h5)


def test_verify_binds_even_an_identity_preserving_edit_to_the_sidecar_hash(
    tool, tmp_path
):
    h5 = write_population_h5(population_tables([1, 2]), tmp_path / "hash.h5", year=2026)
    tool.run_compute(h5, engine_factory=_fake_engine_factory, log=lambda message: None)
    sidecar, _ = tool.sidecar_paths(h5)
    table = tool.read_sidecar(sidecar)
    table.loc[0, "earnings"] += 1
    ids, year = tool.read_h5_household_ids(h5)
    tool.check_household_measures(table, ids, year)
    sidecar.unlink()
    tool.write_sidecar(table, sidecar)
    with pytest.raises(tool.HouseholdMeasuresError, match="sidecar's sha256"):
        tool.verify_sidecar(h5)


def test_verify_refuses_a_changed_sidecar_or_another_h5(tool, tmp_path):
    tables = population_tables([2, 2, 1])
    h5 = write_population_h5(tables, tmp_path / "f.h5", year=2026)
    tool.run_compute(h5, engine_factory=_fake_engine_factory, log=lambda m: None)
    sidecar, receipt = tool.sidecar_paths(h5)

    other = write_population_h5(
        population_tables([2, 2, 1], seed=9), tmp_path / "g.h5", year=2026
    )
    with pytest.raises(tool.HouseholdMeasuresError, match="H5's sha256"):
        tool.verify_sidecar(other, sidecar, receipt)

    table = tool.read_sidecar(sidecar)
    sidecar.unlink()
    tool.write_sidecar(table.iloc[:-1], sidecar)
    with pytest.raises(tool.HouseholdMeasuresError, match="have no row"):
        tool.verify_sidecar(h5)


def test_a_file_without_a_year_refuses(tool, tmp_path):
    tables = population_tables([1])
    path = tmp_path / "noyear.h5"
    with pd.HDFStore(str(path), mode="w") as store:
        for entity, frame in tables.items():
            store.put(entity, frame, format="table")
    with pytest.raises(tool.HouseholdMeasuresError, match="states no year"):
        tool.read_h5_tables(path)


def test_the_cli_parses_both_commands(tool, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(tool, "run_compute", lambda *a, **k: calls.append((a, k)))
    assert (
        tool.main(["compute", "--h5", str(tmp_path / "a.h5"), "--expect-year", "2027"])
        == 0
    )
    assert calls[0][1]["expect_year"] == 2027
    assert calls[0][1]["batch_households"] == tool.DEFAULT_BATCH_HOUSEHOLDS


# ---------------------------------------------------------------------------
# Constants held equal to their owners, without importing the engine
# ---------------------------------------------------------------------------


def test_spm_selection_matches_the_release_scorers(tool):
    source = (
        paths_for("microcosm-build").package
        / "src/microcosm/build/us_runtime/reform_validation.py"
    ).read_text()
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.AnnAssign)
            and getattr(node.target, "id", None) == "US_RELEASE_SPM_SELECTION"
        ):
            assert ast.literal_eval(node.value) == dict(tool.US_RELEASE_SPM_SELECTION)
            break
    else:  # pragma: no cover - the owner moved
        pytest.fail("reform_validation.US_RELEASE_SPM_SELECTION not found")


def test_population_aggregates_match_the_recertification_list(tool):
    from microcosm.data.compatibility_recertification import (
        POPULATION_AGGREGATE_VARIABLES,
    )

    assert tool.POPULATION_AGGREGATE_VARIABLES == POPULATION_AGGREGATE_VARIABLES


def _engine_modules() -> set[str]:
    return {
        name
        for name in sys.modules
        if name.split(".")[0] == "policyengine_us"
        or name.startswith("microcosm.build.us_runtime")
    }


def test_the_engine_free_path_imports_no_engine(tool, tmp_path):
    before = _engine_modules()
    tables = population_tables([2, 1])
    calculate, _, _ = fake_calculator(tool, multi_batch=True)
    tool.compute_household_measures(
        tables, 2026, calculate=calculate, batch_households=1
    )
    h5 = write_population_h5(tables, tmp_path / "e.h5", year=2026)
    tool.run_compute(h5, engine_factory=_fake_engine_factory, log=lambda m: None)
    tool.verify_sidecar(h5)
    assert _engine_modules() - before == set()
