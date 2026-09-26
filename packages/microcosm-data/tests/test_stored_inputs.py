"""The stored-input contract (microcosm#1026, decision d271).

A US release is refused when a stored table carries a column that looks like a
policyengine-us variable, is not a variable of the certified engine, and is
not in the reviewed register. These tests pin that rule three ways:

- example tests, including the ``would_claim_wic`` regression against an
  engine that defines only ``takes_up_wic_if_eligible``;
- Hypothesis properties for the invariants the PR states (refusal if and only
  if, exact naming, monotonicity, register consistency, order independence,
  and an H5 metadata round trip). Hypothesis comes from the workspace sync;
  the wheels job installs no test extras, so those tests skip there;
- ``requires_us`` tests against the installed engine: the naming convention,
  the register, and the stored-column inventories of the two H5 files examined
  for #1026 (``fixtures/stored_input_inventories.json``).

The source-enrichment probe's use of the contract is pinned here too, with the
country and wrapper modules replaced by fakes so the order of its checks is
observable without either engine.
"""

from __future__ import annotations

import json
import random
import re
import string
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

from microcosm.data import source_enrichment as enrichment
from microcosm.data import stored_inputs
from microcosm.data.stored_inputs import (
    US_STORED_NON_VARIABLE_COLUMNS,
    CertifiedEngine,
    StoredInputRefusalError,
    StoredTableLayoutError,
    h5_stored_tables,
    h5_verdict,
    is_model_named,
    register_consistency_failures,
    register_sha256,
    require_h5_stored_inputs,
    stored_input_failures,
    undefined_stored_inputs,
)

_INVENTORIES = Path(__file__).parent / "fixtures" / "stored_input_inventories.json"
_PUBLISHED_DEFAULT = "populace-us-2024-spm-20260915"
_REHEARSAL_EXPORT = "route-a-r4-rehearsal-export"
_RECEIPT_CHILD = "populace-us-2024-spm-receipts-20260923"
_STACKED_POOL = "buildq-stacked-pool-f010-s578"
_ACS_LOCAL_RELEASE = "populace-us-2024-buildo-acs-local-767312d60-20260923T074941Z"
_QUOTED = re.compile(r"'([^']*)'")
#: The postal codes of the 50 states, DC, PR and VI: policyengine-us 2.2.1's
#: only variable names outside the lowercase convention.
_US_POSTAL_CODES = frozenset(
    "AK AL AR AZ CA CO CT DC DE FL GA HI IA ID IL IN KS KY LA MA MD ME MI MN MO "
    "MS MT NC ND NE NH NJ NM NV NY OH OK OR PA PR RI SC SD TN TX UT VA VI VT WA "
    "WI WV WY".split()
)
_LOWER = string.ascii_lowercase
_MODEL_CHARS = frozenset(string.ascii_lowercase + string.digits + "_")


def _oracle_model_named(column: object) -> bool:
    """An independent spelling of the convention, for the property tests."""

    return (
        isinstance(column, str)
        and column[:1] in _LOWER
        and column[:1] != ""
        and all(character in _MODEL_CHARS for character in column)
    )


def _engine(*variables: str, label: str = "policyengine-us 9.9.9") -> CertifiedEngine:
    return CertifiedEngine(label=label, variables=frozenset(variables))


def _inventories() -> dict:
    return json.loads(_INVENTORIES.read_text())["files"]


def _write_pandas_layout(
    path: Path,
    tables: dict[str, list[str]],
    *,
    fixed: tuple[str, ...] = (),
    bytes_attrs: bool = False,
    metadata: dict[str, str] | None = None,
) -> None:
    """Write the HDF layout pandas uses, with h5py and no pandas.

    A ``format="table"`` frame is a group whose ``table`` dataset has the
    ``index`` field followed by one field per data column; a ``format="fixed"``
    frame keeps its column labels in ``axis0``. ``metadata`` maps each
    top-level metadata key to its ``pandas_type``; by default it is the
    ``_time_period`` table series the country loader reads.
    """

    def attr(text: str):
        return np.bytes_(text) if bytes_attrs else text

    with h5py.File(path, "w") as h5:
        for key, columns in tables.items():
            group = h5.create_group(key)
            if key in fixed:
                group.attrs["pandas_type"] = attr("frame")
                group.create_dataset(
                    "axis0",
                    data=np.array([column.encode() for column in columns], dtype="S"),
                )
            else:
                group.attrs["pandas_type"] = attr("frame_table")
                dtype = np.dtype([("index", "<i8"), *((c, "<f8") for c in columns)])
                group.create_dataset("table", data=np.zeros(2, dtype=dtype))
        for key, pandas_type in (
            {"_time_period": "series_table"} if metadata is None else metadata
        ).items():
            h5.create_group(key).attrs["pandas_type"] = attr(pandas_type)


def _hypothesis():
    pytest.importorskip("hypothesis")
    import hypothesis
    from hypothesis import strategies as st

    return hypothesis, st


def _column_strategies(st):
    model_named = st.from_regex(r"[a-z][a-z0-9_]{0,10}", fullmatch=True)
    upper = st.from_regex(r"[A-Z][A-Z0-9_]{0,8}", fullmatch=True)
    mixed = st.from_regex(r"[a-z]{1,4}[A-Z][A-Za-z0-9_]{0,4}", fullmatch=True)
    odd = st.from_regex(r"[_0-9][a-z0-9_]{0,6}", fullmatch=True)
    anything = st.text(max_size=6)
    column = st.one_of(model_named, model_named, upper, mixed, odd, anything)
    return model_named, column


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


def test_would_claim_wic_is_refused_by_an_engine_that_defines_only_its_new_name():
    """The #1026 regression: the draw stored under its retired name is refused
    by name, with both remedies, and the renamed table passes."""

    engine = _engine("takes_up_wic_if_eligible", label="policyengine-us 2.2.1")
    stored = {"person": ("would_claim_wic", "A_AGE", "person_source_id")}

    failures = stored_input_failures(stored, engine=engine)

    assert len(failures) == 1
    (line,) = failures
    assert _QUOTED.findall(line) == ["would_claim_wic"]
    assert "(person table)" in line
    assert "policyengine-us 2.2.1" in line
    assert "Rename it to the live input" in line
    assert "add a reviewed entry with its reason" in line
    assert "microcosm.data.stored_inputs.US_STORED_NON_VARIABLE_COLUMNS" in line

    renamed = {"person": ("takes_up_wic_if_eligible", "A_AGE", "person_source_id")}
    assert stored_input_failures(renamed, engine=engine) == []


def test_the_regression_holds_on_an_h5_read_from_metadata(tmp_path):
    engine = _engine("takes_up_wic_if_eligible", "person_id", "household_id")
    path = tmp_path / "release.h5"
    _write_pandas_layout(
        path,
        {
            "person": ["person_id", "would_claim_wic", "A_AGE", "person_source_id"],
            "household": ["household_id", "H_TENURE"],
        },
    )

    with pytest.raises(StoredInputRefusalError) as refusal:
        require_h5_stored_inputs(path, engine=engine)

    assert [_QUOTED.findall(line) for line in refusal.value.failures] == [
        ["would_claim_wic"]
    ]
    assert isinstance(refusal.value, ValueError)
    verdict = h5_verdict(path, engine=engine)
    assert verdict["passed"] is False
    assert verdict["refused"] == ["would_claim_wic"]
    assert verdict["registered_non_variables"] == ["person_source_id"]


def test_a_column_stored_in_two_tables_is_one_line_naming_both():
    failures = stored_input_failures(
        {"person": ("stale_input",), "household": ("stale_input", "household_id")},
        engine=_engine("household_id"),
    )

    assert len(failures) == 1
    assert "(household, person tables)" in failures[0]


@pytest.mark.parametrize(
    "column",
    ["A_AGE", "H_TENURE", "SPM_WICVAL", "Mixed_case", "_hidden", "2024_value", "CA"],
)
def test_columns_outside_the_convention_pass_by_rule(column):
    assert not is_model_named(column)
    assert stored_input_failures({"person": (column,)}, engine=_engine()) == []


@pytest.mark.parametrize(
    "column", ["would_claim_wic", "a", "x1", "snake_case_2", "tax_unit_role_input"]
)
def test_lowercase_snake_columns_are_model_named(column):
    assert is_model_named(column)


@pytest.mark.parametrize("column", ["", "é", "café", "with space", "dash-ed", 7])
def test_non_ascii_or_non_identifier_columns_are_not_model_named(column):
    assert not is_model_named(column)


def test_register_consistency_names_each_unsound_entry():
    failures = register_consistency_failures(
        {
            "dead_entry": "the engine reads it",
            "UPPER": "never consulted",
            "blank_reason": "  ",
            "sound_entry": "provenance",
        },
        engine_variables={"dead_entry"},
    )

    assert failures == [
        "register entry 'UPPER' is not model-named, so the check never consults it.",
        "register entry 'blank_reason' has no reason.",
        "register entry 'dead_entry' is a variable of the engine, which reads the "
        "stored column as an input: remove the entry.",
    ]


def test_the_shipped_register_is_model_named_and_every_entry_has_a_reason():
    assert US_STORED_NON_VARIABLE_COLUMNS
    assert (
        register_consistency_failures(
            US_STORED_NON_VARIABLE_COLUMNS, engine_variables=frozenset()
        )
        == []
    )
    assert "would_claim_wic" not in US_STORED_NON_VARIABLE_COLUMNS
    assert "medicare_part_b_premiums" not in US_STORED_NON_VARIABLE_COLUMNS


def test_every_register_entry_is_stored_by_a_file_examined_for_1026():
    """The register holds evidence, not guesses: each entry is a column one of
    the examined H5 files actually stores."""

    stored = {
        column
        for inventory in _inventories().values()
        for columns in inventory["tables"].values()
        for column in columns
    }

    assert set(US_STORED_NON_VARIABLE_COLUMNS) <= stored


def test_the_examined_files_store_no_state_code_column():
    """policyengine-us's only variables outside the convention are the 53
    state and territory codes (pinned against the engine below). No examined
    file stores one, so none of their uppercase columns can collide. The only
    two-letter columns they store are the ACS PUMS fields ST and NP."""

    two_letter = set()
    for inventory in _inventories().values():
        for columns in inventory["tables"].values():
            assert not set(columns) & _US_POSTAL_CODES
            two_letter |= {c for c in columns if re.fullmatch(r"[A-Z]{2}", c)}
    assert two_letter == {"ST", "NP"}


#: The role columns policyengine-core's ``build_from_dataset`` reads to build
#: the group entities, for policyengine-us's five group entities. They are not
#: engine variables (pinned against the engine below).
_CORE_ROLE_COLUMNS = frozenset(
    {"role"}
    | {
        f"person_{group}_role"
        for group in ("household", "tax_unit", "spm_unit", "family", "marital_unit")
    }
)


def test_the_check_refuses_core_role_columns_and_no_examined_file_stores_one():
    """Core reads its role columns although no engine variable names them.
    The check does not exempt them, so it errs closed: a file storing one is
    refused by name. No examined file stores one; the only examined column
    ending in _role is the engine variable is_spm_independent_minor_role."""

    assert set(
        undefined_stored_inputs(
            _CORE_ROLE_COLUMNS,
            engine_variables={"is_spm_independent_minor_role"},
        )
    ) == set(_CORE_ROLE_COLUMNS)
    role_like = set()
    for inventory in _inventories().values():
        for columns in inventory["tables"].values():
            assert not set(columns) & _CORE_ROLE_COLUMNS
            role_like |= {c for c in columns if c == "role" or c.endswith("_role")}
    assert role_like == {"is_spm_independent_minor_role"}


def test_register_sha256_is_canonical_and_moves_with_the_register():
    register = {"b_entry": "reason b", "a_entry": "reason a"}
    reordered = dict(reversed(list(register.items())))

    assert register_sha256(register) == register_sha256(reordered)
    assert register_sha256(register) != register_sha256(
        {**register, "c_entry": "reason c"}
    )
    assert register_sha256(register) != register_sha256(
        {**register, "a_entry": "another reason"}
    )
    assert re.fullmatch(r"[0-9a-f]{64}", register_sha256())


def test_h5_tables_skip_the_index_and_the_period(tmp_path):
    path = tmp_path / "release.h5"
    _write_pandas_layout(
        path,
        {"person": ["person_id", "age"], "household": ["household_id"]},
        bytes_attrs=True,
    )

    assert h5_stored_tables(path) == {
        "person": ("person_id", "age"),
        "household": ("household_id",),
    }


def test_h5_fixed_frames_are_read_from_their_column_axis(tmp_path):
    path = tmp_path / "release.h5"
    _write_pandas_layout(
        path,
        {"person": ["person_id", "would_claim_wic"], "household": ["household_id"]},
        fixed=("person",),
    )

    assert h5_stored_tables(path) == {
        "person": ("person_id", "would_claim_wic"),
        "household": ("household_id",),
    }


def test_h5_tables_that_hide_their_columns_are_refused(tmp_path):
    path = tmp_path / "release.h5"
    _write_pandas_layout(path, {"person": ["values_block_0"]})

    with pytest.raises(StoredTableLayoutError, match="data_columns=True"):
        h5_stored_tables(path)


def test_h5_objects_that_are_not_frames_are_refused(tmp_path):
    path = tmp_path / "release.h5"
    _write_pandas_layout(path, {"person": ["person_id"]})
    with h5py.File(path, "a") as h5:
        h5.create_dataset("stray_input", data=np.zeros(3))

    with pytest.raises(StoredTableLayoutError, match="stray_input"):
        h5_stored_tables(path)


def test_h5_metadata_series_are_not_tables(tmp_path):
    """The nullable writer's ``_populace_staging_metadata`` series (the ACS
    local-area release and the multispine pools store one) holds no columns;
    fixed and table series both pass."""

    path = tmp_path / "release.h5"
    _write_pandas_layout(
        path,
        {"person": ["person_id", "would_claim_wic"]},
        fixed=("person",),
        bytes_attrs=True,
        metadata={
            "_time_period": "series",
            "_populace_staging_metadata": "series_table",
        },
    )

    assert h5_stored_tables(path) == {"person": ("person_id", "would_claim_wic")}


@pytest.mark.parametrize("key", ["_time_period", "_populace_staging_metadata"])
@pytest.mark.parametrize("pandas_type", ["frame", "frame_table", "", "wide"])
def test_h5_metadata_keys_that_are_not_series_are_refused(tmp_path, key, pandas_type):
    """A metadata key is skipped only as a series: stored as a frame, it could
    hold columns, so the check refuses rather than skip it."""

    path = tmp_path / "release.h5"
    _write_pandas_layout(
        path,
        {"person": ["person_id"]},
        metadata={"_time_period": "series_table", key: pandas_type},
    )

    with pytest.raises(StoredTableLayoutError, match=f"metadata object '{key}'"):
        h5_stored_tables(path)


def test_h5_series_under_an_unknown_key_are_refused(tmp_path):
    path = tmp_path / "release.h5"
    _write_pandas_layout(
        path,
        {"person": ["person_id"]},
        metadata={"_time_period": "series_table", "_other_metadata": "series"},
    )

    with pytest.raises(StoredTableLayoutError, match="_other_metadata"):
        h5_stored_tables(path)


def test_h5_tables_without_an_index_field_are_refused(tmp_path):
    path = tmp_path / "release.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("person")
        group.attrs["pandas_type"] = "frame_table"
        group.create_dataset("table", data=np.zeros(1, dtype=[("person_id", "<i8")]))

    with pytest.raises(StoredTableLayoutError, match="no pandas index field"):
        h5_stored_tables(path)


def test_pandas_written_tables_round_trip(tmp_path):
    """The layout above is the one pandas writes, in both formats."""

    pd = pytest.importorskip("pandas")
    pytest.importorskip("tables")
    path = tmp_path / "release.h5"
    person = pd.DataFrame(
        {"person_id": [1, 2], "would_claim_wic": [True, False], "A_AGE": [30, 4]}
    )
    household = pd.DataFrame({"household_id": [1], "household_weight": [2.5]})
    with pd.HDFStore(path, mode="w") as store:
        store.put("person", person, format="table", data_columns=True)
        store.put("household", household, format="fixed")
        store.put("_time_period", pd.Series([2024]), format="table")
        store.put(
            "_populace_staging_metadata",
            pd.Series(['{"artifact_kind": "calibrated_local_area_artifact"}']),
            format="table",
        )

    assert h5_stored_tables(path) == {
        "person": ("person_id", "would_claim_wic", "A_AGE"),
        "household": ("household_id", "household_weight"),
    }


def test_frames_nested_under_an_entity_frame_are_not_listed(tmp_path):
    """The reader lists top-level frames only, as ``USSingleYearDataset``
    reads them: a frame pandas nests inside the ``person`` frame's group
    (``person/extra``) is never an input, so its columns are not listed."""

    pd = pytest.importorskip("pandas")
    pytest.importorskip("tables")
    path = tmp_path / "release.h5"
    with pd.HDFStore(path, mode="w") as store:
        for key, frame in (
            ("person", pd.DataFrame({"person_id": [1]})),
            ("person/extra", pd.DataFrame({"hidden_col": [1.0]})),
        ):
            store.put(key, frame, format="table", data_columns=True)
        store.put("_time_period", pd.Series([2024]), format="table")

    assert h5_stored_tables(path) == {"person": ("person_id",)}


@pytest.mark.filterwarnings("ignore:object name is not a valid Python identifier")
def test_year_keyed_entity_frames_are_refused(tmp_path):
    """``USMultiYearDataset``'s ``person/2024`` layout leaves each top-level
    entity group untyped, so the reader refuses it rather than skip it."""

    pd = pytest.importorskip("pandas")
    pytest.importorskip("tables")
    path = tmp_path / "release.h5"
    with pd.HDFStore(path, mode="w") as store:
        store.put(
            "person/2024",
            pd.DataFrame({"person_id": [1], "would_claim_wic": [True]}),
            format="table",
            data_columns=True,
        )

    with pytest.raises(StoredTableLayoutError, match="'person' is not a pandas"):
        h5_stored_tables(path)


def test_the_cli_prints_one_verdict_per_file_and_exits_by_the_worst(
    tmp_path, monkeypatch, capsys
):
    engine = _engine("takes_up_wic_if_eligible", "person_id")
    monkeypatch.setattr(stored_inputs, "installed_us_engine", lambda: engine)
    clean = tmp_path / "clean.h5"
    stale = tmp_path / "stale.h5"
    broken = tmp_path / "broken.h5"
    _write_pandas_layout(clean, {"person": ["person_id", "takes_up_wic_if_eligible"]})
    _write_pandas_layout(stale, {"person": ["person_id", "would_claim_wic"]})
    _write_pandas_layout(broken, {"person": ["values_block_0"]})

    assert stored_inputs.main([str(clean)]) == 0
    assert json.loads(capsys.readouterr().out)[0]["passed"] is True

    assert stored_inputs.main([str(clean), str(stale), str(broken)]) == 1
    verdicts = json.loads(capsys.readouterr().out)
    assert [verdict["passed"] for verdict in verdicts] == [True, False, False]
    assert verdicts[1]["refused"] == ["would_claim_wic"]
    assert verdicts[2]["error"].startswith("StoredTableLayoutError")


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_property_refusal_if_and_only_if_and_the_error_names_exactly_those():
    """For any stored tables, engine and register: the check refuses if and
    only if some column is model-named, not an engine variable and not
    registered, and its lines quote exactly those columns, once each, sorted."""

    hypothesis, st = _hypothesis()
    model_named, column = _column_strategies(st)
    table_name = st.from_regex(r"[a-z_]{1,10}", fullmatch=True)

    @st.composite
    def cases(draw):
        tables = draw(
            st.dictionaries(table_name, st.lists(column, max_size=8), max_size=4)
        )
        pool = sorted(
            {c for cs in tables.values() for c in cs if isinstance(c, str)}
            | set(draw(st.lists(model_named, max_size=4)))
        )
        engine = (
            draw(st.sets(st.sampled_from(pool), max_size=len(pool))) if pool else set()
        )
        registered = (
            draw(st.sets(st.sampled_from(pool), max_size=len(pool))) if pool else set()
        )
        register = {name: f"reason for {name}" for name in registered}
        return tables, frozenset(engine), register

    @hypothesis.settings(max_examples=300, deadline=None)
    @hypothesis.given(cases())
    def check(case):
        tables, engine_variables, register = case
        columns = {c for cs in tables.values() for c in cs}
        expected = sorted(
            c
            for c in columns
            if _oracle_model_named(c)
            and c not in engine_variables
            and c not in register
        )
        engine = CertifiedEngine("policyengine-us 0.0.0", engine_variables)

        failures = stored_input_failures(tables, engine=engine, register=register)

        assert bool(failures) == bool(expected)
        quoted = [_QUOTED.findall(line) for line in failures]
        assert all(len(tokens) == 1 for tokens in quoted)
        assert [tokens[0] for tokens in quoted] == expected
        assert (
            list(
                undefined_stored_inputs(
                    columns, engine_variables=engine_variables, register=register
                )
            )
            == expected
        )

    check()


def test_property_adding_a_variable_or_register_entry_never_refuses_more():
    """Monotonicity: the refused set can only shrink when the engine gains a
    variable or the register gains an entry (so a pass stays a pass), and can
    only grow when the release stores one more column.

    The engine and the register are drawn from the stored columns (plus a few
    other model-named names), as in the refusal property, so they overlap the
    stored columns and a rule whose verdict on a registered or defined column
    depends on the rest of the register or engine is caught here too."""

    hypothesis, st = _hypothesis()
    model_named, column = _column_strategies(st)

    @st.composite
    def cases(draw):
        columns = draw(st.sets(column, max_size=10))
        extra = draw(column)
        pool = sorted(
            {c for c in columns | {extra} if isinstance(c, str)}
            | set(draw(st.lists(model_named, max_size=4)))
        )
        engine = (
            draw(st.frozensets(st.sampled_from(pool), max_size=len(pool)))
            if pool
            else frozenset()
        )
        registered = (
            draw(st.sets(st.sampled_from(pool), max_size=len(pool))) if pool else set()
        )
        return columns, engine, dict.fromkeys(registered, "reason"), extra, pool

    @hypothesis.settings(max_examples=300, deadline=None)
    @hypothesis.given(cases())
    def check(case):
        columns, engine, register, extra, pool = case

        def refused(cs, variables, entries):
            return set(
                undefined_stored_inputs(
                    cs, engine_variables=variables, register=entries
                )
            )

        before = refused(columns, engine, register)
        for name in pool:
            more_variables = refused(columns, engine | {name}, register)
            more_register = refused(columns, engine, {**register, name: "reason"})
            assert more_variables <= before
            assert more_register <= before
            if not before:
                assert not more_variables and not more_register
        assert before <= refused(columns | {extra}, engine, register)

    check()


def test_property_register_consistency_flags_exactly_the_dead_or_bare_entries():
    """Register consistency: an entry is sound if and only if it is model-named,
    not an engine variable and has a non-blank reason. An entry that is an
    engine variable is dead: dropping it changes no verdict."""

    hypothesis, st = _hypothesis()
    model_named, column = _column_strategies(st)
    reason = st.one_of(st.just(""), st.just("   "), st.text(min_size=1, max_size=12))

    @hypothesis.settings(max_examples=300, deadline=None)
    @hypothesis.given(
        register=st.dictionaries(column, reason, max_size=6),
        engine=st.frozensets(model_named, max_size=6),
        stored=st.sets(column, max_size=8),
    )
    def check(register, engine, stored):
        engine = engine | frozenset(c for c in list(register)[:2] if isinstance(c, str))
        failures = register_consistency_failures(register, engine_variables=engine)
        for entry, text in register.items():
            sound = (
                _oracle_model_named(entry)
                and entry not in engine
                and bool(text.strip())
            )
            flagged = any(
                line.startswith(f"register entry {entry!r} ") for line in failures
            )
            assert flagged == (not sound)
        dead = {e for e in register if isinstance(e, str) and e in engine}
        live = {e: r for e, r in register.items() if e not in dead}
        columns = stored | set(register)
        assert undefined_stored_inputs(
            columns, engine_variables=engine, register=register
        ) == undefined_stored_inputs(columns, engine_variables=engine, register=live)

    check()


def test_property_the_verdict_ignores_table_and_column_order_and_repeats():
    hypothesis, st = _hypothesis()
    model_named, column = _column_strategies(st)
    table_name = st.from_regex(r"[a-z_]{1,10}", fullmatch=True)

    @hypothesis.settings(max_examples=200, deadline=None)
    @hypothesis.given(
        tables=st.dictionaries(table_name, st.lists(column, max_size=6), max_size=4),
        engine=st.frozensets(model_named, max_size=6),
        seed=st.integers(0, 2**32 - 1),
    )
    def check(tables, engine, seed):
        shuffle = random.Random(seed)
        items = list(tables.items())
        shuffle.shuffle(items)
        permuted = {}
        for name, columns in items:
            columns = list(columns) * 2
            shuffle.shuffle(columns)
            permuted[name] = columns
        certified = CertifiedEngine("policyengine-us 0.0.0", engine)
        assert stored_input_failures(tables, engine=certified) == stored_input_failures(
            permuted, engine=certified
        )

    check()


def test_property_h5_metadata_round_trips_the_stored_columns(tmp_path_factory):
    hypothesis, st = _hypothesis()
    name = st.from_regex(r"[A-Za-z][A-Za-z0-9_]{0,10}", fullmatch=True).filter(
        lambda text: text != "index" and not text.startswith("values_block_")
    )
    entity = st.sampled_from(
        ("person", "household", "tax_unit", "spm_unit", "family", "marital_unit")
    )

    @hypothesis.settings(max_examples=40, deadline=None)
    @hypothesis.given(
        tables=st.dictionaries(
            entity, st.lists(name, min_size=1, max_size=8, unique=True), max_size=6
        ),
        fixed=st.sets(entity, max_size=3),
        metadata=st.fixed_dictionaries(
            {"_time_period": st.sampled_from(("series", "series_table"))},
            optional={
                "_populace_staging_metadata": st.sampled_from(
                    ("series", "series_table")
                )
            },
        ),
    )
    def check(tables, fixed, metadata):
        path = tmp_path_factory.mktemp("round_trip") / "release.h5"
        _write_pandas_layout(path, tables, fixed=tuple(fixed), metadata=metadata)
        assert h5_stored_tables(path) == {
            key: tuple(columns) for key, columns in tables.items()
        }

    check()


_LAYOUT_DEFECTS = (
    "values_block",
    "index_not_first",
    "no_index",
    "stray_dataset",
    "untyped_group",
    "unknown_series",
    "metadata_frame",
    "fixed_without_axis",
)


def test_property_h5_layouts_the_reader_cannot_see_are_refused(tmp_path_factory):
    """Invariant 5's refusal half: take any layout the reader round-trips and
    add one defect anywhere. The reader then refuses rather than guess. The
    defects are a ``values_block_*`` field at any position of any table, an
    index field that is missing or not first, a top-level object that is not a
    pandas frame (a bare dataset, an untyped group, a series under an unknown
    key), a metadata key stored as a frame, and a fixed frame with no column
    axis."""

    hypothesis, st = _hypothesis()
    name = st.from_regex(r"[A-Za-z][A-Za-z0-9_]{0,10}", fullmatch=True).filter(
        lambda text: text != "index" and not text.startswith("values_block_")
    )
    entity = st.sampled_from(
        ("person", "household", "tax_unit", "spm_unit", "family", "marital_unit")
    )

    @hypothesis.settings(max_examples=120, deadline=None)
    @hypothesis.given(
        tables=st.dictionaries(
            entity,
            st.lists(name, min_size=1, max_size=8, unique=True),
            min_size=1,
            max_size=6,
        ),
        fixed=st.sets(entity, max_size=3),
        defect=st.sampled_from(_LAYOUT_DEFECTS),
        data=st.data(),
    )
    def check(tables, fixed, defect, data):
        path = tmp_path_factory.mktemp("refused") / "release.h5"
        victim = data.draw(st.sampled_from(sorted(tables)), label="victim")
        stray_key = name.filter(lambda text: text not in tables)
        fixed = set(fixed)
        if defect in {"values_block", "index_not_first", "no_index"}:
            fixed.discard(victim)
        elif defect == "fixed_without_axis":
            fixed.add(victim)
        _write_pandas_layout(path, tables, fixed=tuple(fixed))
        with h5py.File(path, "a") as h5:
            if defect in {"values_block", "index_not_first", "no_index"}:
                fields = ["index", *tables[victim]]
                if defect == "values_block":
                    block = f"values_block_{data.draw(st.integers(0, 99))}"
                    fields.insert(data.draw(st.integers(1, len(fields))), block)
                elif defect == "index_not_first":
                    fields.remove("index")
                    fields.insert(data.draw(st.integers(1, len(fields))), "index")
                else:
                    fields.remove("index")
                del h5[victim]["table"]
                h5[victim].create_dataset(
                    "table",
                    data=np.zeros(2, dtype=[(field, "<f8") for field in fields]),
                )
            elif defect == "stray_dataset":
                h5.create_dataset(data.draw(stray_key, label="stray"), data=np.zeros(3))
            elif defect == "untyped_group":
                group = h5.create_group(data.draw(stray_key, label="stray"))
                group.create_dataset("axis0", data=np.array([b"hidden_input"]))
            elif defect == "unknown_series":
                stray = h5.create_group(data.draw(stray_key, label="stray"))
                stray.attrs["pandas_type"] = "series_table"
            elif defect == "metadata_frame":
                key = data.draw(
                    st.sampled_from(("_time_period", "_populace_staging_metadata"))
                )
                if key in h5:
                    del h5[key]
                frame = h5.create_group(key)
                frame.attrs["pandas_type"] = data.draw(
                    st.sampled_from(("frame", "frame_table"))
                )
            else:
                del h5[victim]["axis0"]

        with pytest.raises(StoredTableLayoutError):
            h5_stored_tables(path)

    check()


# ---------------------------------------------------------------------------
# The source-enrichment probe runs the contract before either loader
# ---------------------------------------------------------------------------


class _FakeCountryLoader:
    """Stands in for ``USSingleYearDataset``; reaching it ends the probe."""

    def __init__(self, *args, **kwargs):
        raise _LoaderReachedError


class _FakeWrapperLoader:
    pass


class _FakeRoleVariable:
    entity = types.SimpleNamespace(key="person")
    value_type = bool


class _LoaderReachedError(Exception):
    pass


@pytest.fixture
def fake_probe_runtime(monkeypatch):
    """The probe's imports resolve to fakes, so its ordering is observable."""

    system = types.SimpleNamespace(
        variables={enrichment.ROLE_VARIABLE: _FakeRoleVariable()}
    )
    modules = {
        "policyengine": types.ModuleType("policyengine"),
        "policyengine.tax_benefit_models": types.ModuleType(
            "policyengine.tax_benefit_models"
        ),
        "policyengine.tax_benefit_models.us": types.ModuleType(
            "policyengine.tax_benefit_models.us"
        ),
        "policyengine.tax_benefit_models.us.datasets": types.ModuleType(
            "policyengine.tax_benefit_models.us.datasets"
        ),
        "policyengine_us": types.ModuleType("policyengine_us"),
        "policyengine_us.data": types.ModuleType("policyengine_us.data"),
        "policyengine_us.system": types.ModuleType("policyengine_us.system"),
    }
    modules[
        "policyengine.tax_benefit_models.us.datasets"
    ].PolicyEngineUSDataset = _FakeWrapperLoader
    modules["policyengine_us.data"].USSingleYearDataset = _FakeCountryLoader
    modules["policyengine_us.system"].system = system
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(enrichment, "_runtime_package_identities", lambda *a, **k: {})
    engine = _engine(
        "takes_up_wic_if_eligible",
        "person_id",
        enrichment.ROLE_VARIABLE,
        label="policyengine-us 2.2.1",
    )
    monkeypatch.setattr(stored_inputs, "installed_us_engine", lambda: engine)


def test_the_probe_refuses_a_stale_input_before_either_loader_runs(
    tmp_path, fake_probe_runtime
):
    candidate = tmp_path / "populace_us_2024.h5"
    _write_pandas_layout(
        candidate,
        {"person": ["person_id", enrichment.ROLE_VARIABLE, "would_claim_wic"]},
    )

    with pytest.raises(StoredInputRefusalError, match="'would_claim_wic'"):
        enrichment.run_native_loader_compatibility(candidate)


def test_the_probe_reaches_the_loaders_once_the_contract_passes(
    tmp_path, fake_probe_runtime
):
    candidate = tmp_path / "populace_us_2024.h5"
    _write_pandas_layout(
        candidate,
        {"person": ["person_id", enrichment.ROLE_VARIABLE, "takes_up_wic_if_eligible"]},
    )

    with pytest.raises(_LoaderReachedError):
        enrichment.run_native_loader_compatibility(candidate)


def test_the_probe_receipt_records_the_contract():
    """The receipt carries the contract's summary and names the check, so a
    replay compares the register the certification consulted."""

    import ast
    import inspect

    source = inspect.getsource(enrichment.run_native_loader_compatibility)
    tree = ast.parse(source)
    returned = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
    ]
    assert len(returned) == 1
    receipt = returned[0]
    keys = [ast.literal_eval(key) for key in receipt.keys]
    assert ast.unparse(receipt.values[keys.index("stored_inputs")]) == "stored_inputs"
    checks = ast.unparse(receipt.values[keys.index("checks")])
    assert "'country:defines_every_stored_model_input'" in checks
    assert source.index("_require_stored_inputs(candidate_h5)") < source.index(
        "USSingleYearDataset(file_path="
    )


@pytest.mark.parametrize(
    ("installed_us", "built_with_us"),
    [("2.2.1", "2.2.1"), ("2.3.0", "2.2.1"), (None, "2.2.1")],
    ids=["certified-runtime", "other-runtime", "no-runtime"],
)
def test_a_refusing_probe_fails_the_contract_by_name(
    monkeypatch, tmp_path, installed_us, built_with_us
):
    """Validation reports the refusal as a compatibility failure that names the
    column (the probe's ``ValueError`` path). When the installed engine is not
    the bundle's built-with engine, the runtime mismatch a passing probe would
    report is reported first, so the refusal is not read as a verdict against
    the certified engine."""

    from importlib import metadata

    installed = {"policyengine-us": installed_us, "policyengine-core": "3.32.5"}

    def version(name):
        if installed.get(name) is None:
            raise metadata.PackageNotFoundError(name)
        return installed[name]

    monkeypatch.setattr(metadata, "version", version)
    failures: list[str] = []
    candidate = tmp_path / "populace_us_2024.h5"
    candidate.write_bytes(b"")
    receipt = tmp_path / enrichment.COMPATIBILITY_FILE
    receipt.write_text("{}")
    manifest = {
        "build": {
            "built_with_model_package": {
                "name": "policyengine-us",
                "version": built_with_us,
            },
            "built_with_core_package": {
                "name": "policyengine-core",
                "version": "3.32.5",
            },
        }
    }

    def refusing_probe(*args, **kwargs):
        raise StoredInputRefusalError(
            stored_input_failures(
                {"person": ("would_claim_wic",)},
                engine=_engine(label=f"policyengine-us {installed_us}"),
            )
        )

    monkeypatch.setattr(enrichment, "run_native_loader_compatibility", refusing_probe)
    enrichment._check_compatibility(
        tmp_path,
        manifest,
        {"filename": enrichment.COMPATIBILITY_FILE},
        candidate,
        False,
        (),
        failures,
    )

    assert failures[0] == (
        "source enrichment compatibility must hash-bind the actual test receipt"
    )
    mismatch = (
        []
        if installed_us == built_with_us
        else ["compatibility built-with policyengine-us must match tested runtime"]
    )
    assert failures[1:-1] == mismatch
    assert failures[-1].startswith(
        "native loader compatibility failed: stored-input contract refused "
        "the release: stored column 'would_claim_wic'"
    )
    assert len(failures) == 2 + len(mismatch)


def test_the_probe_helper_summarizes_a_passing_candidate(tmp_path, monkeypatch):
    engine = _engine("person_id", label="policyengine-us 2.2.1")
    monkeypatch.setattr(stored_inputs, "installed_us_engine", lambda: engine)
    candidate = tmp_path / "populace_us_2024.h5"
    _write_pandas_layout(
        candidate, {"person": ["person_id", "person_source_id", "A_AGE"]}
    )

    assert enrichment._require_stored_inputs(candidate) == {
        "register_sha256": register_sha256(),
        "registered_non_variables": ["person_source_id"],
    }


# ---------------------------------------------------------------------------
# Against the installed engine
# ---------------------------------------------------------------------------


@pytest.mark.requires_us
def test_the_naming_convention_holds_for_the_installed_engine():
    """Every variable is model-named except household Boolean formulas named by
    a two-letter code. Those are formula-owned, so the release writer refuses
    to store them, and uppercase stored columns may pass by rule. An engine
    that breaks this fails here rather than slipping past the check."""

    from importlib import metadata

    from policyengine_us.system import system

    names = set(system.variables)
    outside = {name for name in names if not is_model_named(name)}

    assert not [name for name in names if name[:1] in "_0123456789"]
    assert all(re.fullmatch(r"[A-Z]{2}", name) for name in outside)
    for name in outside:
        variable = system.variables[name]
        assert variable.entity.key == "household"
        assert variable.value_type is bool
        assert variable.formulas
    if metadata.version("policyengine-us") == "2.2.1":
        assert (len(names), outside) == (6167, _US_POSTAL_CODES)


@pytest.mark.requires_us
def test_the_register_is_consistent_with_the_installed_engine():
    engine = stored_inputs.installed_us_engine()

    assert (
        register_consistency_failures(
            US_STORED_NON_VARIABLE_COLUMNS, engine_variables=engine.variables
        )
        == []
    )


@pytest.mark.requires_us
def test_the_1026_premises_hold_for_the_installed_engine():
    """The two retired inputs, the construction columns the register treats
    and core's role columns, as the installed engine defines them."""

    from policyengine_us.system import system

    variables = system.variables
    assert "would_claim_wic" not in variables
    wic = variables["takes_up_wic_if_eligible"]
    assert (wic.entity.key, wic.value_type, wic.default_value) == ("person", bool, True)
    assert not wic.formulas
    for retired_or_construction in (
        "medicare_part_b_premiums",
        "tax_unit_role_input",
        "filing_status_input",
    ):
        assert retired_or_construction not in variables
    # medicare_part_b_premiums is a Person, YEAR, float input with no formula
    # in every policyengine-us version read from 1.452.0 to 1.670.2; its
    # replacement has that shape.
    part_b = variables["medicare_part_b_premiums_reported"]
    assert (part_b.entity.key, part_b.definition_period, part_b.value_type) == (
        "person",
        "year",
        float,
    )
    assert not part_b.formulas
    # Core reads these to build the group entities; none is a variable.
    assert {entity.key for entity in system.group_entities} == {
        "household",
        "tax_unit",
        "spm_unit",
        "family",
        "marital_unit",
    }
    assert not _CORE_ROLE_COLUMNS & set(variables)
    # puma_geoid is registered as an alias of the puma input.
    assert "puma_geoid" not in variables
    puma = variables["puma"]
    assert (puma.entity.key, bool(puma.formulas)) == ("household", False)
    for derived in (
        "filing_status",
        "is_tax_unit_head",
        "is_tax_unit_spouse",
        "is_tax_unit_dependent",
    ):
        assert variables[derived].formulas


#: Each examined file's expected verdict: the refused columns, and how many
#: register entries it stores. The published default, its receipt child and
#: the ACS local-area release store two retired engine inputs: the #1026 WIC
#: draw as would_claim_wic, and the Medicare Part B target as
#: medicare_part_b_premiums (an input in every policyengine-us version read
#: from 1.452.0 to 1.670.2, replaced by medicare_part_b_premiums_reported by
#: 1.690.7). So each is refused naming exactly those two. The files built from
#: main's tools pass.
_EXPECTED_VERDICTS = {
    _PUBLISHED_DEFAULT: (["medicare_part_b_premiums", "would_claim_wic"], 24),
    _RECEIPT_CHILD: (["medicare_part_b_premiums", "would_claim_wic"], 24),
    _ACS_LOCAL_RELEASE: (["medicare_part_b_premiums", "would_claim_wic"], 37),
    _REHEARSAL_EXPORT: ([], 30),
    _STACKED_POOL: ([], 43),
}


def test_every_examined_file_has_an_expected_verdict():
    assert set(_EXPECTED_VERDICTS) == set(_inventories())


@pytest.mark.requires_us
@pytest.mark.parametrize("name", sorted(_EXPECTED_VERDICTS))
def test_each_examined_file_gets_its_expected_verdict(name):
    """The examined files, by their stored-column inventories, against the
    installed engine: each refusal names exactly the stale inputs, one person
    table line each, and every other model-named non-variable column the file
    stores is a register entry."""

    engine = stored_inputs.installed_us_engine()
    tables = _inventories()[name]["tables"]
    refused, registered_count = _EXPECTED_VERDICTS[name]

    failures = stored_input_failures(tables, engine=engine)

    assert [_QUOTED.findall(line) for line in failures] == [
        [column] for column in refused
    ]
    assert all("(person table)" in line for line in failures)
    registered = {
        column
        for columns in tables.values()
        for column in columns
        if is_model_named(column)
        and column not in engine.variables
        and column in US_STORED_NON_VARIABLE_COLUMNS
    }
    assert len(registered) == registered_count
    uppercase = {
        column
        for columns in tables.values()
        for column in columns
        if not is_model_named(column)
    }
    assert not uppercase & engine.variables


def test_only_the_acs_lane_spine_tags_rest_on_refused_files_alone():
    """Every register entry is stored by one of the two files that pass (the
    rehearsal export and the stacked pool), except the six *_spine tags. Those
    are stored only by the ACS local-area release, which is itself refused
    for the same two retired inputs as its donor. They stay registered on the
    strength of their live producer, base_pool.spine_column, which
    test_us_stored_input_register.py binds, not on a passing file."""

    inventories = _inventories()

    def stored_by(*names):
        return {
            column
            for name in names
            for columns in inventories[name]["tables"].values()
            for column in columns
        }

    spine_tags = {
        "person_spine",
        "household_spine",
        "tax_unit_spine",
        "spm_unit_spine",
        "family_spine",
        "marital_unit_spine",
    }
    unbacked = set(US_STORED_NON_VARIABLE_COLUMNS) - stored_by(
        _REHEARSAL_EXPORT, _STACKED_POOL
    )
    assert unbacked == spine_tags
    assert unbacked <= stored_by(_ACS_LOCAL_RELEASE)
    assert not unbacked & stored_by(_PUBLISHED_DEFAULT, _RECEIPT_CHILD)
