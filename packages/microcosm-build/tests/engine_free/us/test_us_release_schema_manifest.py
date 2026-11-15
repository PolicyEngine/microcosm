"""The release schema manifest: description, round trips, diffs and Markdown."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import materialize
from test_support.microcosm_build.us_county_files import (
    FakeDataset,
    fake_microsimulation_class,
    fake_system,
    load_tool,
    population_tables,
    write_population_h5,
)

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

_SETTINGS = settings(
    max_examples=40,
    deadline=None,
    # Generation is slow only on a loaded host.
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
    ],
)


@pytest.fixture(scope="module")
def manifest_tool():
    return load_tool("us_release_schema_manifest")


@pytest.fixture(scope="module")
def measures_tool():
    return load_tool("us_household_measures")


# ---------------------------------------------------------------------------
# Generated tables and manifests
# ---------------------------------------------------------------------------

_DTYPES = ("int64", "float64", "bool", "str", "float32", "int16")


@st.composite
def _frames(draw, max_rows: int = 30):
    rows = draw(st.integers(1, max_rows))
    n_columns = draw(st.integers(1, 6))
    data = {}
    for index in range(n_columns):
        kind = draw(st.sampled_from(("int64", "float64", "bool", "str")))
        name = f"c{index}_{kind}"
        if kind == "int64":
            data[name] = np.asarray(
                draw(st.lists(st.integers(-1000, 1000), min_size=rows, max_size=rows)),
                dtype=np.int64,
            )
        elif kind == "bool":
            data[name] = np.asarray(
                draw(st.lists(st.booleans(), min_size=rows, max_size=rows)), dtype=bool
            )
        elif kind == "float64":
            values = draw(
                st.lists(
                    st.one_of(st.none(), st.floats(-1e6, 1e6)),
                    min_size=rows,
                    max_size=rows,
                )
            )
            data[name] = np.asarray(
                [np.nan if value is None else value for value in values], np.float64
            )
        else:
            values = draw(
                st.lists(
                    st.one_of(st.none(), st.sampled_from(["06001", "36061", "x"])),
                    min_size=rows,
                    max_size=rows,
                )
            )
            data[name] = pd.Series(values, dtype="str")
    return pd.DataFrame(data)


@st.composite
def _manifests(draw):
    tables = draw(
        st.lists(
            st.sampled_from(["person", "household", "tax_unit", "spm_unit", "family"]),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    entry_tables = []
    for table in tables:
        names = draw(
            st.lists(
                st.from_regex(r"[a-z][a-z_]{0,11}", fullmatch=True),
                min_size=1,
                max_size=8,
                unique=True,
            )
        )
        entry_tables.append(
            {
                "name": table,
                "storage": "table",
                "rows": draw(st.integers(0, 10**7)),
                "columns": [
                    {
                        "name": name,
                        "dtype": draw(st.sampled_from(_DTYPES)),
                        "null_count": draw(st.integers(0, 100)),
                    }
                    for name in names
                ],
            }
        )
    return {
        "name": "release.h5",
        "format": "hdf5",
        "sha256": "0" * 64,
        "size_bytes": draw(st.integers(0, 10**10)),
        "tables": entry_tables,
        "other_keys": [
            {
                "key": "_time_period",
                "pandas_type": "series_table",
                "rows": 1,
                "value": 2026,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Describing files
# ---------------------------------------------------------------------------


@_SETTINGS
@given(_frames(), _frames(), st.integers(1, 7))
def test_h5_description_matches_pandas_whatever_the_chunking(
    manifest_tool, tmp_path_factory, person, household, chunk
):
    path = tmp_path_factory.mktemp("h5") / "f.h5"
    with pd.HDFStore(str(path), mode="w") as store:
        store.put("household", household, format="table", data_columns=True)
        store.put("person", person, format="table", data_columns=True)
        store.put("_time_period", pd.Series([2027]), format="table")
    entry = manifest_tool.describe_h5(path, chunk_rows=chunk)

    assert [table["name"] for table in entry["tables"]] == ["person", "household"]
    assert entry["other_keys"] == [
        {"key": "_time_period", "pandas_type": "series_table", "rows": 1, "value": 2027}
    ]
    for table in entry["tables"]:
        whole = pd.read_hdf(path, table["name"])
        assert table["rows"] == len(whole)
        assert table["storage"] == "table"
        assert [column["name"] for column in table["columns"]] == list(whole.columns)
        for column in table["columns"]:
            assert column["dtype"] == str(whole[column["name"]].dtype)
            assert column["null_count"] == int(whole[column["name"]].isna().sum())
            assert "logical_dtype" not in column


def test_frame_codec_booleans_get_a_logical_dtype(manifest_tool, tmp_path):
    table = pd.DataFrame(
        {
            "person_id": np.arange(5, dtype=np.int64),
            "flag": pd.array([True, None, False, None, True], dtype="boolean"),
        }
    )
    path = tmp_path / "codec.h5"
    with pd.HDFStore(str(path), mode="w") as store:
        materialize.put_frame_table(
            store, "person", table, preferred_format="table", data_columns=True
        )
    entry = manifest_tool.describe_h5(path, chunk_rows=2)
    (person,) = entry["tables"]
    assert person["storage"] == "fixed"
    assert person["rows"] == 5
    flag = person["columns"][1]
    assert flag == {
        "name": "flag",
        "dtype": "object",
        "null_count": 2,
        "logical_dtype": "boolean",
    }


@_SETTINGS
@given(
    st.lists(st.one_of(st.none(), st.booleans()), min_size=1, max_size=25).filter(
        lambda values: None in values
    ),
    _frames(max_rows=1),
    st.integers(1, 9),
)
def test_fixed_tables_count_rows_whatever_block_comes_first(
    manifest_tool, tmp_path_factory, flags, extra, chunk
):
    """A fixed frame whose first block holds objects still counts every row.

    pandas' fixed storer reports its first block's leading dimension as
    ``shape[0]``, which for an object block is its column count.
    """
    rows = len(flags)
    table = pd.DataFrame({"flag": pd.array(flags, dtype="boolean")})
    for name in extra.columns:
        table[name] = np.resize(extra[name].to_numpy(), rows)
    path = tmp_path_factory.mktemp("fixed") / "f.h5"
    with pd.HDFStore(str(path), mode="w") as store:
        materialize.put_frame_table(
            store, "person", table, preferred_format="table", data_columns=True
        )
    (person,) = manifest_tool.describe_h5(path, chunk_rows=chunk)["tables"]
    whole = pd.read_hdf(path, "person")
    assert person["storage"] == "fixed"
    assert person["rows"] == rows == len(whole)
    for column in person["columns"]:
        assert column["dtype"] == str(whole[column["name"]].dtype)
        assert column["null_count"] == int(whole[column["name"]].isna().sum())
    assert person["columns"][0]["logical_dtype"] == "boolean"
    assert person["columns"][0]["null_count"] == flags.count(None)


@pytest.mark.parametrize("chunk", [1000, 1500, 4000, 7])
def test_a_long_run_of_missing_codec_booleans_reads_as_object(
    manifest_tool, tmp_path, chunk
):
    """A chunk where the column is all missing reads as ``str`` in pandas 3.

    The manifest takes the dtype from chunks with values, as a whole read
    does, instead of refusing the table.
    """
    flags = [True] * 1000 + [None] * 2000 + [False] * 1000
    table = pd.DataFrame(
        {"flag": pd.array(flags, dtype="boolean"), "x": np.arange(4000)}
    )
    path = tmp_path / "codec.h5"
    with pd.HDFStore(str(path), mode="w") as store:
        materialize.put_frame_table(
            store, "person", table, preferred_format="table", data_columns=True
        )
    (person,) = manifest_tool.describe_h5(path, chunk_rows=chunk)["tables"]
    assert str(pd.read_hdf(path, "person")["flag"].dtype) == "object"
    assert person["rows"] == 4000
    assert person["columns"][0] == {
        "name": "flag",
        "dtype": "object",
        "null_count": 2000,
        "logical_dtype": "boolean",
    }


def test_the_local_release_layout(manifest_tool, tmp_path):
    """The layout ``h5_io.write_nullable_us_h5`` gives the local release.

    Fixed-format entity tables (nullable booleans under the frame codec),
    ``_time_period`` and the JSON ``_populace_staging_metadata`` row as
    table-format series, and string root attributes. Only the entities are
    tables; the rest is file metadata.
    """
    tables = population_tables([3, 1, 2], tax_unit_splits=[True, False, True])
    tables["person"]["is_blind"] = pd.array(
        [True, None, False, None, False, True], dtype="boolean"
    )
    metadata = json.dumps({"artifact_kind": "calibrated_local_area_artifact"})
    path = tmp_path / "microcosm_us_2026_acs_local.h5"
    with pd.HDFStore(str(path), mode="w") as store:
        for entity, frame in tables.items():
            materialize.put_frame_table(store, entity, frame, preferred_format="fixed")
        store.put("_time_period", pd.Series([2026]), format="table")
        store.put("_populace_staging_metadata", pd.Series([metadata]), format="table")
        store.get_node("/")._v_attrs["block_v1_crosswalk_sha256"] = "ab" * 32
        store.get_node("/")._v_attrs["long_note"] = "x" * 5_000
    entry = manifest_tool.describe_h5(path, chunk_rows=2)

    assert [table["name"] for table in entry["tables"]] == list(
        manifest_tool.ENTITY_ORDER
    )
    assert entry["other_keys"] == [
        {
            "key": "_populace_staging_metadata",
            "pandas_type": "series_table",
            "rows": 1,
            "value": metadata,
        },
        {
            "key": "_time_period",
            "pandas_type": "series_table",
            "rows": 1,
            "value": 2026,
        },
    ]
    assert entry["root_attributes"] == {
        "block_v1_crosswalk_sha256": "ab" * 32,
        "long_note": "<str, not shown>",
    }
    for table in entry["tables"]:
        whole = pd.read_hdf(path, table["name"])
        assert table["storage"] == "fixed"
        assert table["rows"] == len(whole)
        assert [c["name"] for c in table["columns"]] == list(whole.columns)
        for column in table["columns"]:
            assert column["dtype"] == str(whole[column["name"]].dtype)
            assert column["null_count"] == int(whole[column["name"]].isna().sum())
    person = entry["tables"][0]
    blind = next(c for c in person["columns"] if c["name"] == "is_blind")
    assert blind["logical_dtype"] == "boolean" and blind["null_count"] == 2
    markdown = manifest_tool.render_markdown(manifest_tool.build_manifest([entry]))
    assert "## Metadata" in markdown
    assert "root attribute `block_v1_crosswalk_sha256`" in markdown


def test_the_codec_attribute_is_the_frames(manifest_tool):
    assert (
        manifest_tool.FRAME_TABLE_CODEC_ATTR
        == materialize._PYTABLES_FRAME_TABLE_CODEC_ATTR
    )


def test_a_measures_sidecar_is_described_with_its_declared_dtypes(
    manifest_tool, measures_tool, tmp_path
):
    tables = population_tables([2, 1, 3])
    h5 = write_population_h5(tables, tmp_path / "c.h5", year=2026)
    measures_tool.run_compute(
        h5,
        engine_factory=lambda: (
            fake_microsimulation_class(),
            FakeDataset,
            fake_system(),
        ),
        log=lambda message: None,
    )
    sidecar, _ = measures_tool.sidecar_paths(h5)
    entry = manifest_tool.describe_csv(sidecar)
    (table,) = entry["tables"]
    assert table["name"] == "household_measures"
    assert table["rows"] == 3
    assert [(c["name"], c["dtype"]) for c in table["columns"]] == list(
        measures_tool.SIDECAR_COLUMNS
    )
    assert all(column["null_count"] == 0 for column in table["columns"])


# ---------------------------------------------------------------------------
# Round trips and diffs
# ---------------------------------------------------------------------------


@_SETTINGS
@given(_manifests())
def test_a_manifest_round_trips_and_diffs_empty_against_itself(
    manifest_tool, tmp_path_factory, entry
):
    manifest = manifest_tool.build_manifest([entry])
    text = manifest_tool.manifest_json(manifest)
    assert json.loads(text) == manifest
    path = tmp_path_factory.mktemp("m") / "m.json"
    path.write_text(text)
    loaded = manifest_tool.load_manifest(path)
    assert loaded == manifest
    assert manifest_tool.manifest_json(loaded) == text

    diff = manifest_tool.diff_manifests(manifest, loaded, renames=())
    assert manifest_tool.diff_is_empty(diff)
    assert diff["declared_renames_not_applied"] == []

    column_list = tmp_path_factory.mktemp("c") / "columns.csv"
    column_list.write_text(manifest_tool.manifest_column_list(manifest))
    prior = manifest_tool.load_prior(column_list)
    assert manifest_tool.diff_is_empty(
        manifest_tool.diff_manifests(prior, manifest, renames=())
    )


@_SETTINGS
@given(_manifests(), st.data())
def test_the_diff_reports_exactly_the_set_differences(manifest_tool, entry, data):
    current = manifest_tool.build_manifest([entry])
    index = manifest_tool.column_index(current)
    keys = sorted(index)
    dropped = set(data.draw(st.lists(st.sampled_from(keys), unique=True)))
    retyped = set(data.draw(st.lists(st.sampled_from(keys), unique=True))) - dropped
    prior_entry = json.loads(json.dumps(entry))
    invented = set()
    for table in prior_entry["tables"]:
        columns = []
        for column in table["columns"]:
            key = (table["name"], column["name"])
            if key in dropped:
                continue
            if key in retyped:
                column["dtype"] = column["dtype"] + "_old"
            columns.append(column)
        name = f"zz_prior_only_{table['name']}"
        columns.append({"name": name, "dtype": "int64", "null_count": 0})
        invented.add((table["name"], name))
        table["columns"] = columns
    prior = manifest_tool.build_manifest([prior_entry])

    diff = manifest_tool.diff_manifests(prior, current, renames=())
    assert {(i["table"], i["column"]) for i in diff["added"]} == dropped
    assert {(i["table"], i["column"]) for i in diff["removed"]} == invented
    assert {(i["table"], i["column"]) for i in diff["dtype_changed"]} == retyped
    assert diff["renamed"] == []
    for item in diff["dtype_changed"]:
        assert item["dtype_before"] == item["dtype_after"] + "_old"


def _entry(tables):
    return {
        "name": "f.h5",
        "format": "hdf5",
        "sha256": None,
        "size_bytes": 0,
        "tables": [
            {
                "name": name,
                "storage": "table",
                "rows": 1,
                "columns": [
                    {"name": column, "dtype": dtype, "null_count": 0}
                    for column, dtype in columns
                ],
            }
            for name, columns in tables.items()
        ],
        "other_keys": [],
    }


def test_the_wic_rename_and_the_weeks_worked_removal(manifest_tool, tmp_path):
    prior_csv = tmp_path / "sample_columns.csv"
    prior_csv.write_text(
        "table,column,dtype\n"
        "household,household_id,int64\n"
        "person,person_id,int64\n"
        "person,would_claim_wic,bool\n"
        "person,weeks_worked,float64\n"
        "person,age,float32\n"
    )
    current = manifest_tool.build_manifest(
        [
            _entry(
                {
                    "household": [("household_id", "int64")],
                    "person": [
                        ("person_id", "int64"),
                        ("takes_up_wic_if_eligible", "bool"),
                        ("age", "float64"),
                    ],
                }
            )
        ]
    )
    diff = manifest_tool.diff_manifests(manifest_tool.load_prior(prior_csv), current)
    assert diff["renamed"] == [
        {
            "table": "person",
            "from": "would_claim_wic",
            "to": "takes_up_wic_if_eligible",
            "dtype_before": "bool",
            "dtype_after": "bool",
            "reason": manifest_tool.DECLARED_RENAMES[0]["reason"],
        }
    ]
    assert diff["removed"] == [
        {"table": "person", "column": "weeks_worked", "dtype": "float64"}
    ]
    assert diff["added"] == []
    assert diff["dtype_changed"] == [
        {
            "table": "person",
            "column": "age",
            "dtype_before": "float32",
            "dtype_after": "float64",
        }
    ]
    markdown = manifest_tool.render_markdown({**current, "diff": diff})
    assert "### Columns renamed" in markdown
    assert "`would_claim_wic`" in markdown and "`takes_up_wic_if_eligible`" in markdown
    assert "### Columns removed" in markdown and "`weeks_worked`" in markdown


@pytest.mark.parametrize(
    ("prior", "current", "why"),
    [
        ({"person": [("a", "bool")]}, {"person": [("a", "bool")]}, "no column 'from'"),
        (
            {"person": [("would_claim_wic", "bool")]},
            {"person": [("would_claim_wic", "bool")]},
            "still has 'from'",
        ),
        (
            {
                "person": [
                    ("would_claim_wic", "bool"),
                    ("takes_up_wic_if_eligible", "bool"),
                ]
            },
            {"person": [("takes_up_wic_if_eligible", "bool")]},
            "already has 'to'",
        ),
    ],
)
def test_a_declared_rename_that_does_not_apply_says_why(
    manifest_tool, prior, current, why
):
    diff = manifest_tool.diff_manifests(
        manifest_tool.build_manifest([_entry(prior)]),
        manifest_tool.build_manifest([_entry(current)]),
    )
    assert diff["renamed"] == []
    (item,) = diff["declared_renames_not_applied"]
    assert why in item["why"]


@pytest.mark.parametrize(
    "renames",
    [
        [{"table": "person", "from": "a", "to": "a"}],
        [
            {"table": "person", "from": "a", "to": "b"},
            {"table": "person", "from": "a", "to": "c"},
        ],
        [
            {"table": "person", "from": "a", "to": "b"},
            {"table": "person", "from": "b", "to": "c"},
        ],
        [{"table": "person", "from": "a"}],
        [{"table": "person", "from": "a", "to": "b", "extra": "x"}],
    ],
)
def test_malformed_rename_declarations_refuse(manifest_tool, renames):
    with pytest.raises(manifest_tool.ManifestError):
        manifest_tool.validate_renames(renames)


def test_extra_renames_load_from_a_file(manifest_tool, tmp_path):
    path = tmp_path / "renames.json"
    path.write_text(json.dumps([{"table": "tax_unit", "from": "x", "to": "y"}]))
    renames = manifest_tool.load_renames(path)
    assert [item["from"] for item in renames] == ["would_claim_wic", "x"]


def test_a_table_named_in_two_files_refuses(manifest_tool):
    entry = _entry({"person": [("a", "int64")]})
    with pytest.raises(manifest_tool.ManifestError, match="appears in both"):
        manifest_tool.build_manifest([entry, {**entry, "name": "g.csv"}])


@pytest.mark.parametrize(
    "text",
    [
        "table,column\nperson,a\n",
        "table,column,dtype\nperson,,int64\n",
        "table,column,dtype\nperson,a,int64,extra\n",
        "table,column,dtype\nperson,a\n",
    ],
)
def test_a_malformed_column_list_refuses(manifest_tool, tmp_path, text):
    path = tmp_path / "bad.csv"
    path.write_text(text)
    with pytest.raises(manifest_tool.ManifestError):
        manifest_tool.load_column_list(path)


def test_a_manifest_of_another_kind_refuses(manifest_tool, tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"kind": "something_else", "schema_version": 1}))
    with pytest.raises(manifest_tool.ManifestError, match="is not a"):
        manifest_tool.load_manifest(path)


# ---------------------------------------------------------------------------
# Markdown and the command line
# ---------------------------------------------------------------------------


@_SETTINGS
@given(_manifests())
def test_the_markdown_lists_every_column(manifest_tool, entry):
    manifest = manifest_tool.build_manifest([entry])
    markdown = manifest_tool.render_markdown(manifest)
    assert markdown == manifest_tool.render_markdown(manifest)
    for table in entry["tables"]:
        assert f"### `{table['name']}`" in markdown
        for column in table["columns"]:
            assert f"| `{column['name']}` | {column['dtype']} |" in markdown
    assert "No prior schema was given." in markdown


def test_the_cli_writes_json_and_markdown_and_never_overwrites(
    manifest_tool, measures_tool, tmp_path
):
    tables = population_tables([1, 2])
    h5 = write_population_h5(tables, tmp_path / "county_2027.h5", year=2027)
    measures_tool.run_compute(
        h5,
        engine_factory=lambda: (
            fake_microsimulation_class(),
            FakeDataset,
            fake_system(),
        ),
        log=lambda message: None,
    )
    sidecar, _ = measures_tool.sidecar_paths(h5)
    prior = tmp_path / "prior.csv"
    prior.write_text(
        "table,column,dtype\nperson,person_id,int64\nperson,weeks_worked,float64\n"
    )
    out_json, out_md = tmp_path / "out" / "m.json", tmp_path / "out" / "m.md"
    arguments = [
        "--h5", str(h5),
        "--sidecar", str(sidecar),
        "--prior", str(prior),
        "--output-json", str(out_json),
        "--output-md", str(out_md),
    ]  # fmt: skip
    assert manifest_tool.main(arguments) == 0
    manifest = manifest_tool.load_manifest(out_json)
    assert [entry["name"] for entry in manifest["files"]] == [
        "county_2027.h5",
        "county_2027.household_measures.csv.gz",
    ]
    tables_described = [t["name"] for e in manifest["files"] for t in e["tables"]]
    assert tables_described[-1] == "household_measures"
    assert {
        "table": "person",
        "column": "weeks_worked",
        "dtype": "float64",
    } in manifest["diff"]["removed"]
    assert out_md.read_text() == manifest_tool.render_markdown(manifest)
    with pytest.raises(FileExistsError):
        manifest_tool.main(arguments)
