"""Invented CSV/H5 contracts for numeric-only ACS source recovery."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import zipfile

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from tools import acs_numeric_source_recovery as recovery  # noqa: E402


def _archive(path, role, rows):
    prefix = "psam_hus" if role == "household" else "psam_pus"
    with zipfile.ZipFile(path, "w") as archive:
        # Deliberately inverted ZIP directory order and unsorted CSV records.
        for suffix, selected in (("b", rows[::2]), ("a", rows[1::2])):
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(selected)
            archive.writestr(prefix + suffix + ".csv", stream.getvalue())
    return {
        "role": role,
        "filename": path.name,
        "sha256": recovery.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _household(serial, n, kind=1):
    return dict(
        SERIALNO=serial,
        STATE="06",
        PUMA="00123",
        WGTP="0" if kind > 1 else "10",
        NP=str(n),
        ADJHSG="1000000",
        TEN="" if kind > 1 else "1",
        RNTP="",
        GRNTP="",
        TAXAMT="",
        TYPEHUGQ=str(kind),
    )


def _person(serial, sporder, age, hours):
    row = {name: "0" for name in (*recovery.PERSON_ANCHORS, *recovery.HOURS)}
    row.update(
        SERIALNO=serial,
        SPORDER=str(sporder),
        AGEP=str(age),
        SEX="1",
        ADJINC="1000000",
        PWGTP="1",
        MAR="5",
        RELSHIPP="20",
        WKHP=str(hours),
        WKL="1" if hours else "",
        FWKHP="0" if hours else "",
    )
    return row


def _numeric(row, names):
    return [float(row[name]) if row[name] else np.nan for name in names]


@pytest.fixture
def invented(tmp_path):
    # Lexical SERIALNO gives GQ '10' before HU '2'. Vacancy '00' has no rank.
    households = [
        _household("2", 2),
        _household("00", 0),
        _household("30", 1),
        _household("10", 1, kind=2),
    ]
    people = [
        _person("2", 10, 20, 30),
        _person("30", 1, 50, 45),
        _person("2", 2, 40, 35),
        _person("10", 1, 12, ""),
    ]
    hzip, pzip, parent = (
        tmp_path / "hus.zip",
        tmp_path / "pus.zip",
        tmp_path / "parent.h5",
    )
    sources = {
        "vintage": 2024,
        "artifacts": [
            _archive(hzip, "household", households),
            _archive(pzip, "person", people),
        ],
    }
    hs = [households[3], households[0], households[2]]
    ps = [people[3], people[2], people[0], people[1]]
    person_h = [1, 2, 2, 3]
    # The donor's year/source IDs collide, but its ACS raw marker is missing.
    # Current IDs use offsets unrelated to donor counts; one donor clone exists.
    person_keys = [[1, 1, 30, 2024, 1, 0, 1, 0, 0, 4]]
    for r, (row, h) in enumerate(zip(ps, person_h, strict=True)):
        person_keys.append(
            [
                h,
                int(row["SPORDER"]),
                int(row["AGEP"]),
                2024,
                h,
                r,
                h + 100,
                r + 200,
                r,
                0,
            ]
        )
    household_keys = [[1, 6, 1, 4]] + [[h + 100, 6, h, 0] for h in range(1, 4)]
    contract = {
        "parent_build_id": "invented",
        "reviewed_packaging_revision": "invented",
        "acs_people": 4,
        "acs_households": 3,
        "acs_housing_units": 2,
        "acs_group_quarters": 1,
        "total_people": 5,
        "total_households": 4,
        "blocks": {},
    }
    with h5py.File(parent, "w") as handle:
        for path, columns, values, dtype in (
            (
                "/person/block8",
                recovery.PERSON_ANCHORS,
                [[np.nan] * 13]
                + [_numeric(row, recovery.PERSON_ANCHORS) for row in ps],
                "float64",
            ),
            ("/person/block9", recovery.PERSON_KEYS, person_keys, "int64"),
            (
                "/household/block15",
                recovery.HOUSEHOLD_ANCHORS,
                [[np.nan] * 7]
                + [_numeric(row, recovery.HOUSEHOLD_ANCHORS) for row in hs],
                "float64",
            ),
            ("/household/block16", recovery.HOUSEHOLD_KEYS, household_keys, "int64"),
        ):
            array = np.array(values, dtype=dtype)
            handle.create_dataset(path + "_items", data=np.array(columns, dtype="S40"))
            handle.create_dataset(path + "_values", data=array)
            contract["blocks"][path] = dict(
                columns=list(columns), shape=list(array.shape), dtype=dtype
            )
        # An unreadable object-like external link beside allowed blocks. Any
        # recursive H5 payload traversal would try to open this missing target.
        handle["/person/block0_values"] = h5py.ExternalLink(
            "never-open-object-payload.h5", "/pickle"
        )
    contract["parent_sha256"] = recovery.sha256_file(parent)
    return (
        dict(
            household_zip=hzip,
            person_zip=pzip,
            parent_h5=parent,
            output_dir=tmp_path / "result",
            contract=contract,
            sources=sources,
            chunk_rows=2,
        ),
        households,
        people,
    )


def _mutate_parent(args, entity, field, row, value):
    with h5py.File(args["parent_h5"], "r+") as handle:
        for path, block in args["contract"]["blocks"].items():
            if path.startswith("/" + entity + "/") and field in block["columns"]:
                handle[path + "_values"][row, block["columns"].index(field)] = value
                break
        else:
            raise AssertionError("Invented field not found")
    args["contract"]["parent_sha256"] = recovery.sha256_file(args["parent_h5"])


def test_full_recovery_orders_sources_and_preserves_raw_hours(invented):
    args, _, _ = invented
    receipt = recovery.recover(**args)
    sidecar = np.load(args["output_dir"] / "acs-source-hours.npy", allow_pickle=False)
    assert sidecar.dtype.hasobject is False
    assert sidecar["SPORDER"].tolist() == [1, 2, 10, 1]
    assert sidecar["source_household_id"].tolist() == [1, 2, 2, 3]
    assert sidecar["source_row_id"].tolist() == [0, 1, 2, 3]
    assert sidecar["person_id"].tolist() == [200, 201, 202, 203]
    assert sidecar["parent_person_row"].tolist() == [1, 2, 3, 4]
    assert np.isnan(sidecar["WKHP"][0])
    assert sidecar["WKHP"][1:].tolist() == [35, 30, 45]
    assert receipt["full_source_bijection"] is True
    assert receipt["current_id_offsets"] == {"household": 100, "person": 200}
    assert receipt["original_staging_revision"] is None
    assert receipt["puma_compared"] is False
    assert receipt["sidecar_sha256"] == recovery.sha256_file(
        args["output_dir"] / "acs-source-hours.npy"
    )
    assert json.loads((args["output_dir"] / "RECOVERY.json").read_text()) == receipt
    assert {path.name for path in args["output_dir"].iterdir()} == {
        "RECOVERY.json",
        "acs-source-hours.npy",
    }


@pytest.mark.parametrize(
    "selector,expected_ranks,expected_rows",
    [
        ({"pilot_households": 2}, [1, 3], [0, 3]),
        ({"household_ranks": (2,)}, [2], [1, 2]),
    ],
)
def test_pilot_keeps_global_ranks_and_complete_households(
    invented, selector, expected_ranks, expected_rows
):
    args, _, _ = invented
    receipt = recovery.recover(**args, **selector)
    sidecar = np.load(args["output_dir"] / "acs-source-hours.npy", allow_pickle=False)
    assert receipt["full_source_bijection"] is False
    assert receipt["selected_source_bijection"] is True
    assert receipt["selected_household_ranks"] == expected_ranks
    assert sidecar["source_row_id"].tolist() == expected_rows


def test_spread_pilot_includes_housing_when_both_endpoints_are_gq():
    ranks = recovery._select_ranks(5, np.array([0, 2, 2, 1, 2, 3]), 2, None)
    assert ranks.tolist() == [1, 3]


@pytest.mark.parametrize(
    "entity,field,row,value,error",
    [
        ("person", "AGEP", 2, 20, "anchor mismatch"),
        ("person", "source_household_id", 2, 3, "joins disagree"),
        ("person", "source_row_id", 2, 0, "Duplicate target"),
        ("person", "person_support_clone_index", 2, 1, "clone present"),
        ("person", "source_year", 2, 2023, "source year mismatch"),
        ("person", "person_source_id", 2, 999, "source ID mismatch"),
        ("household", "household_support_clone_index", 2, 1, "clone present"),
        ("person", "person_household_id", 2, 101, "membership mismatch"),
        ("person", "person_id", 2, 202, "offset|Duplicate current"),
        ("person", "SPORDER", 2, np.nan, "Missing target"),
        ("household", "state_fips", 2, 7, "anchor mismatch"),
        ("household", "RNTP", 2, 0, "anchor mismatch"),
    ],
)
def test_bad_target_has_no_committed_output(invented, entity, field, row, value, error):
    args, _, _ = invented
    _mutate_parent(args, entity, field, row, value)
    with pytest.raises(ValueError, match=error):
        recovery.recover(**args)
    assert not args["output_dir"].exists()
    assert not list(args["output_dir"].parent.glob(".acs-recovery-*"))


@pytest.mark.parametrize(
    "change,error",
    [
        ("duplicate_person", "Duplicate ACS composite"),
        ("orphan", "Orphan ACS"),
        ("missing_person", "counts do not equal"),
        ("permuted_anchors", "anchor mismatch"),
    ],
)
def test_bad_source_has_no_committed_output(invented, change, error):
    args, _, people = invented
    if change == "duplicate_person":
        people[0]["SPORDER"] = "2"
    elif change == "orphan":
        people[0]["SERIALNO"] = "absent"
    elif change == "missing_person":
        people.pop(0)
    else:
        people[0]["AGEP"], people[2]["AGEP"] = people[2]["AGEP"], people[0]["AGEP"]
    args["sources"]["artifacts"][1] = _archive(args["person_zip"], "person", people)
    with pytest.raises(ValueError, match=error):
        recovery.recover(**args)
    assert not args["output_dir"].exists()


def test_duplicate_household_refused(invented):
    args, households, _ = invented
    households[0]["SERIALNO"] = households[2]["SERIALNO"]
    args["sources"]["artifacts"][0] = _archive(
        args["household_zip"], "household", households
    )
    with pytest.raises(sqlite3.IntegrityError):
        recovery.recover(**args)
    assert not args["output_dir"].exists()


def test_wrong_archive_pin_refused(invented):
    args, _, _ = invented
    args["sources"]["artifacts"][1]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        recovery.recover(**args)
    assert not args["output_dir"].exists()


def test_wrong_parent_pin_refused(invented):
    args, _, _ = invented
    args["contract"]["parent_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        recovery.recover(**args)
    assert not args["output_dir"].exists()


def test_household_st_takes_precedence_over_state(invented):
    args, households, _ = invented
    for row in households:
        row["ST"], row["STATE"] = "06", "07"
    args["sources"]["artifacts"][0] = _archive(
        args["household_zip"], "household", households
    )
    assert recovery.recover(**args)["full_source_bijection"] is True


def test_existing_output_not_changed(invented):
    args, _, _ = invented
    args["output_dir"].mkdir()
    keep = args["output_dir"] / "keep"
    keep.write_text("original")
    with pytest.raises(ValueError, match="already exists"):
        recovery.recover(**args)
    assert keep.read_text() == "original"


def test_object_labels_refused_before_read(invented):
    args, _, _ = invented
    with h5py.File(args["parent_h5"], "r+") as handle:
        del handle["/person/block8_items"]
        handle.create_dataset("/person/block8_items", (13,), dtype=h5py.string_dtype())
    args["contract"]["parent_sha256"] = recovery.sha256_file(args["parent_h5"])
    with pytest.raises(ValueError, match="Unsafe H5 labels"):
        recovery.recover(**args)
    assert not args["output_dir"].exists()


def test_required_external_block_refused_without_following_link(invented):
    args, _, _ = invented
    with h5py.File(args["parent_h5"], "r+") as handle:
        del handle["/person/block8_values"]
        handle["/person/block8_values"] = h5py.ExternalLink("never-open.h5", "/values")
    args["contract"]["parent_sha256"] = recovery.sha256_file(args["parent_h5"])
    with pytest.raises(ValueError, match="dataset link is not local"):
        recovery.recover(**args)
    assert not args["output_dir"].exists()
