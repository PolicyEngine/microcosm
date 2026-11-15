"""Annual candidates of a local-area build: the County Files' 2026 and 2027.

The builder projects a build of ``microcosm-us-2024-local`` from its folder,
its H5 checked against the build's manifest, and writes each requested year in
the base's own fixed nullable layout. These tests run the real builder on a
real fixed-format base with every engine touchpoint stood in for
(``test_support.microcosm_build.us_annual_local``); static aging is the real
operator unless a test says otherwise.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build import us_annual_static_aging as annual
from microcosm.build.us_runtime.h5_io import read_nullable_us_h5_metadata
from test_support.microcosm_build.us_annual_local import (
    HOUSEHOLD_GEOGRAPHY,
    LOCAL_BUILD,
    LOCAL_DATASET,
    LOCAL_FILE,
    LOCAL_RELEASE,
    MONETARY,
    PERSON_ATTRIBUTES,
    ROOT_ATTRIBUTES,
    Household,
    default_households,
    install_engine_stand_ins,
    local_build_kwargs,
    local_frame,
    read_tables,
    scaling_static_aging,
    sha256,
    uprating_map,
    write_local_base,
    write_local_build_folder,
)

_PROPERTY = settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)


def _root_attributes(path: Path) -> dict[str, str]:
    with pd.HDFStore(path, "r") as store:
        return annual.local_root_attributes(store)


def _period(path: Path) -> list[int]:
    with pd.HDFStore(path, "r") as store:
        return store["_time_period"].tolist()


# --- The candidate --------------------------------------------------------------


def test_a_local_build_writes_exactly_the_requested_years_in_the_base_layout(
    tmp_path, monkeypatch
):
    native = install_engine_stand_ins(monkeypatch)
    kwargs = local_build_kwargs(tmp_path)
    base_h5 = kwargs["base_artifact_root"] / LOCAL_FILE
    base_sha = sha256(base_h5)

    manifest = annual.build_annual_static_aging(**kwargs, years=[2027, 2026])

    output = kwargs["output_dir"]
    assert {p.name for p in output.iterdir()} == {
        "annual_manifest.json",
        "build_status.json",
        "microcosm_us_2026_acs_local.h5",
        "microcosm_us_2027_acs_local.h5",
        "projection_2026.json",
        "projection_2027.json",
    }
    assert json.loads((output / "annual_manifest.json").read_text()) == manifest
    assert manifest["layout"] == "fixed_nullable"
    assert manifest["base"] == {
        "dataset": LOCAL_DATASET,
        "year": 2024,
        "parent_release": LOCAL_BUILD,
        "release": LOCAL_RELEASE,
        "path": str(base_h5),
        "sha256": base_sha,
    }
    assert manifest["settings"]["years"] == [2026, 2027]
    assert manifest["metadata"]["dataset_years"] == {
        LOCAL_DATASET: {
            "2024": LOCAL_DATASET,
            "2026": "microcosm_us_2026_acs_local",
            "2027": "microcosm_us_2027_acs_local",
        }
    }
    base_artifact = manifest["artifacts"][LOCAL_DATASET]
    assert base_artifact["source"] == "base_build"
    assert (base_artifact["path"], base_artifact["sha256"], base_artifact["year"]) == (
        LOCAL_FILE,
        base_sha,
        2024,
    )
    assert set(manifest["artifacts"]) == {
        LOCAL_DATASET,
        "microcosm_us_2026_acs_local",
        "microcosm_us_2027_acs_local",
    }
    base = read_tables(base_h5)
    for year in (2026, 2027):
        name = f"microcosm_us_{year}_acs_local"
        record = manifest["artifacts"][name]
        path = output / record["path"]
        assert record["sha256"] == sha256(path)
        assert record["year"] == year
        assert record["projection_receipt"]["sha256"] == sha256(
            output / record["projection_receipt"]["path"]
        )
        assert read_nullable_us_h5_metadata(path) == dict(annual.LOCAL_BASE_METADATA)
        assert _period(path) == [year]
        assert _root_attributes(path) == ROOT_ATTRIBUTES
        projected = read_tables(path)
        for entity, table in base.items():
            assert list(projected[entity].columns) == list(table.columns)
            assert len(projected[entity]) == len(table)
        for column in HOUSEHOLD_GEOGRAPHY:
            pd.testing.assert_series_equal(
                projected["household"][column], base["household"][column]
            )
        for column in PERSON_ATTRIBUTES:
            pd.testing.assert_series_equal(
                projected["person"][column], base["person"][column]
            )
        weights = projected["household"]["household_weight"].to_numpy()
        assert np.isfinite(weights).all() and (weights > 0).all()
        assert not np.array_equal(
            projected["person"]["employment_income_before_lsr"],
            base["person"]["employment_income_before_lsr"],
        )
        receipt = json.loads((output / f"projection_{year}.json").read_text())
        shares = receipt["geography_shares"]
        assert set(shares) == {"state_fips", "county_fips_2023"}
        assert sum(
            v["base_share"] for v in shares["state_fips"]["values"].values()
        ) == (pytest.approx(1.0))
    assert [(p.name, year) for p, year, _ in native] == [
        ("microcosm_us_2026_acs_local.h5", 2026),
        ("microcosm_us_2027_acs_local.h5", 2027),
    ]
    assert json.loads((output / "build_status.json").read_text()) == {
        "status": "complete",
        "completed_years": [2026, 2027],
    }


def test_without_years_a_local_build_writes_every_year_after_its_base(
    tmp_path, monkeypatch
):
    install_engine_stand_ins(monkeypatch, static_aging=scaling_static_aging())
    kwargs = local_build_kwargs(tmp_path, root_attributes=None)

    manifest = annual.build_annual_static_aging(**kwargs)

    assert sorted(p.name for p in kwargs["output_dir"].glob("*.h5")) == [
        "microcosm_us_2025_acs_local.h5",
        "microcosm_us_2026_acs_local.h5",
        "microcosm_us_2027_acs_local.h5",
    ]
    assert manifest["settings"]["years"] == [2025, 2026, 2027]
    assert (
        _root_attributes(kwargs["output_dir"] / "microcosm_us_2025_acs_local.h5") == {}
    )


# --- Properties -------------------------------------------------------------------

_households = st.lists(
    st.builds(
        Household,
        state=st.sampled_from([1, 6, 9, 36, 48]),
        county=st.integers(1, 999),
        weight=st.floats(0.5, 5_000.0, allow_nan=False, allow_infinity=False),
        members=st.lists(
            st.tuples(
                st.integers(0, 99),
                st.booleans(),
                st.booleans(),
                st.integers(0, 9999),
                st.integers(0, 9999),
                st.floats(0, 250_000, allow_nan=False, allow_infinity=False),
                st.floats(-50_000, 50_000, allow_nan=False, allow_infinity=False),
            ),
            min_size=1,
            max_size=3,
        ).map(tuple),
    ),
    min_size=1,
    max_size=6,
)
_years = st.sets(st.sampled_from([2025, 2026, 2027]), min_size=1)


@_PROPERTY
@given(households=_households, years=_years)
def test_for_all_households_geography_ids_and_rows_carry_and_weights_stay_positive(
    monkeypatch, households, years
):
    """For every household, each projected year's geography equals the base's.

    Row counts, ids and memberships are identical across years, the written
    years are exactly the requested ones, and projected weights are positive
    and finite. Static aging is the real operator.
    """
    install_engine_stand_ins(monkeypatch)
    with tempfile.TemporaryDirectory() as directory:
        kwargs = local_build_kwargs(Path(directory), households)
        manifest = annual.build_annual_static_aging(**kwargs, years=sorted(years))
        output = kwargs["output_dir"]
        assert sorted(p.name for p in output.glob("*.h5")) == sorted(
            f"microcosm_us_{year}_acs_local.h5" for year in years
        )
        assert manifest["settings"]["years"] == sorted(years)
        base = read_tables(kwargs["base_artifact_root"] / LOCAL_FILE)
        for year in years:
            projected = read_tables(output / f"microcosm_us_{year}_acs_local.h5")
            for entity, table in base.items():
                assert len(projected[entity]) == len(table)
                identity = [
                    c
                    for c in table.columns
                    if c == f"{entity}_id" or c.startswith("person_")
                ]
                pd.testing.assert_frame_equal(
                    projected[entity][identity], table[identity], check_exact=True
                )
            pd.testing.assert_frame_equal(
                projected["household"][list(HOUSEHOLD_GEOGRAPHY)],
                base["household"][list(HOUSEHOLD_GEOGRAPHY)],
                check_exact=True,
            )
            pd.testing.assert_frame_equal(
                projected["person"][list(PERSON_ATTRIBUTES)],
                base["person"][list(PERSON_ATTRIBUTES)],
                check_exact=True,
            )
            weights = projected["household"]["household_weight"].to_numpy()
            assert np.isfinite(weights).all()
            assert (weights > 0).all()


@_PROPERTY
@given(data=st.data(), ledger=st.booleans())
def test_a_base_whose_manifest_or_ledger_differs_from_its_h5_is_refused(
    monkeypatch, data, ledger
):
    """Any one changed byte of the H5, against the digest the build's
    manifest (or its ledger) records, refuses the build before anything is
    written."""
    install_engine_stand_ins(monkeypatch)
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        kwargs = local_build_kwargs(tmp)
        h5 = kwargs["base_artifact_root"] / LOCAL_FILE
        original = sha256(h5)
        if ledger:
            # The manifest matches; the ledger lists another digest.
            other = data.draw(
                st.text("0123456789abcdef", min_size=64, max_size=64).filter(
                    lambda digest: digest != original
                )
            )
            shutil.rmtree(kwargs["base_build_dir"].parent)
            write_local_build_folder(
                kwargs["base_artifact_root"], h5, ledger_sha256=other
            )
            match = "sha256sums.txt"
        else:
            # The manifest pins the original bytes; one byte changes.
            raw = bytearray(h5.read_bytes())
            offset = data.draw(st.integers(0, len(raw) - 1))
            raw[offset] ^= data.draw(st.integers(1, 255))
            h5.write_bytes(bytes(raw))
            assert sha256(h5) != original
            match = "do not match"
        with pytest.raises(ValueError, match=match):
            annual.build_annual_static_aging(**kwargs, years=[2026, 2027])
        assert not kwargs["output_dir"].exists()


@settings(max_examples=200, deadline=None)
@given(
    base_year=st.integers(1990, 2100),
    span=st.integers(1, 15),
    data=st.data(),
)
def test_projection_years_are_exactly_the_requested_years(base_year, span, data):
    end_year = base_year + span
    window = list(range(base_year + 1, end_year + 1))
    chosen = data.draw(st.lists(st.sampled_from(window), min_size=1, unique=True))
    assert annual.projection_years(base_year, end_year, chosen) == tuple(sorted(chosen))
    assert annual.projection_years(base_year, end_year) == tuple(window)
    names = annual.annual_dataset_names(
        base_year, end_year, line=annual.LOCAL_LINE, years=chosen
    )
    assert names == {
        base_year: f"microcosm_us_{base_year}_acs_local",
        **{year: f"microcosm_us_{year}_acs_local" for year in chosen},
    }
    outside = data.draw(
        st.integers(base_year - 50, end_year + 50).filter(
            lambda year: not base_year < year <= end_year
        )
    )
    for bad in ([], [*chosen, chosen[0]], [*chosen, outside], [True]):
        with pytest.raises(ValueError, match="years must be"):
            annual.projection_years(base_year, end_year, bad)


def test_the_national_names_are_unchanged_by_the_local_line():
    assert annual.annual_dataset_names(2024, 2026) == {
        2024: "microcosm_us_2024",
        2025: "microcosm_us_2025",
        2026: "microcosm_us_2026",
    }
    assert annual.annual_dataset_names(2024, 2027, years=[2026]) == {
        2024: "microcosm_us_2024",
        2026: "microcosm_us_2026",
    }
    with pytest.raises(ValueError, match="base_dataset"):
        annual.annual_dataset_names(
            2024,
            2027,
            line=annual.LOCAL_LINE,
            base_dataset="microcosm_us_2025_acs_local",
        )


@settings(max_examples=200, deadline=None)
@given(
    ages=st.lists(st.integers(0, 120), min_size=1, max_size=50),
    bad=st.sampled_from([-1.0, 40.5, float("nan"), float("inf")]),
)
def test_age_cells_band_single_year_and_cps_coded_ages_alike(ages, bad):
    cells = annual.age_cells(pd.Series(ages, dtype="float64"))
    expected = [80 if 80 <= a <= 84 else min(a, 85) for a in ages]
    assert cells.tolist() == expected
    with pytest.raises(ValueError, match="whole nonnegative"):
        annual.age_cells(pd.Series([*ages, bad]))


@settings(max_examples=200, deadline=None)
@given(
    column=st.sampled_from(
        sorted(annual.GEOGRAPHY_COLUMNS)
        + ["congressional_district_geoid__2025", "age", "is_female", "is_disabled"]
        + ["detailed_occupation_recode", "industry_code"]
    ),
    others=st.lists(st.sampled_from(MONETARY), unique=True),
)
def test_a_factor_on_geography_or_a_person_attribute_is_refused(column, others):
    annual.refuse_protected_factors(others, what="the map")
    with pytest.raises(ValueError, match="carry unchanged"):
        annual.refuse_protected_factors([*others, column], what="the map")


@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_a_changed_carried_column_is_refused_and_a_factored_one_is_not(data):
    frame = local_frame(default_households())
    entity = data.draw(st.sampled_from(["person", "household"]))
    table = frame.table(entity)
    column = data.draw(st.sampled_from(list(table.columns)))
    row = data.draw(st.integers(0, len(table) - 1))
    projected = {name: frame.table(name).copy() for name in frame.entities}
    projected["household"]["household_weight"] = frame.weights_for("household").values
    annual.check_carried_columns(frame, projected, factored=())
    changed = projected[entity][column].copy()
    value = changed.iloc[row]
    changed.iloc[row] = (
        (not value)
        if isinstance(value, (bool, np.bool_))
        else value + 1
        if isinstance(value, (int, float, np.number))
        else f"{value}x"
    )
    projected[entity][column] = changed
    with pytest.raises(ValueError, match="differs from the base"):
        annual.check_carried_columns(frame, projected, factored=())
    annual.check_carried_columns(frame, projected, factored={column})


@settings(max_examples=100, deadline=None)
@given(
    weights=st.lists(st.floats(0.1, 1e4, allow_nan=False), min_size=5, max_size=5),
    scale=st.floats(0.2, 5.0, allow_nan=False),
)
def test_share_drift_is_zero_under_uniform_scaling_and_shares_sum_to_one(
    weights, scale
):
    households = local_frame(default_households()).table("household")
    base = np.array(weights)
    drift = annual.geography_share_drift(households, base, base * scale)
    assert set(drift) == {"state_fips", "county_fips_2023"}
    for summary in drift.values():
        assert summary["max_abs_share_change"] == pytest.approx(0.0, abs=1e-12)
        assert sum(v["projected_share"] for v in summary["values"].values()) == (
            pytest.approx(1.0)
        )
        assert summary["value_count"] == len(summary["values"])


# --- Refusals ---------------------------------------------------------------------


def test_a_geography_column_in_the_uprating_map_is_refused_before_writing(
    tmp_path, monkeypatch
):
    install_engine_stand_ins(monkeypatch)

    def with_geography(columns, years, *, system):
        totals, indices, mapping = uprating_map(columns, years, system=system)
        return totals, indices, {**mapping, "county_fips_2023": "cpi"}

    monkeypatch.setattr(annual, "uprating_series", with_geography)
    kwargs = local_build_kwargs(tmp_path)
    with pytest.raises(ValueError, match="county_fips_2023"):
        annual.build_annual_static_aging(**kwargs, years=[2026])
    assert not kwargs["output_dir"].exists()


def test_a_base_without_the_local_metadata_row_is_refused(tmp_path, monkeypatch):
    install_engine_stand_ins(monkeypatch)
    kwargs = local_build_kwargs(tmp_path)
    h5 = kwargs["base_artifact_root"] / LOCAL_FILE
    write_local_base(h5, local_frame(default_households()), artifact_kind="other")
    shutil.rmtree(kwargs["base_build_dir"].parent)
    write_local_build_folder(kwargs["base_artifact_root"], h5)
    with pytest.raises(ValueError, match="not a calibrated local-area artifact"):
        annual.build_annual_static_aging(**kwargs, years=[2026])
    assert not kwargs["output_dir"].exists()


def test_a_base_of_another_year_is_refused(tmp_path, monkeypatch):
    install_engine_stand_ins(monkeypatch)
    kwargs = local_build_kwargs(tmp_path)
    with pytest.raises(ValueError, match="_time_period must be exactly 2025"):
        annual.build_annual_static_aging(**{**kwargs, "base_year": 2025}, years=[2026])


@pytest.mark.parametrize(
    "extra",
    [
        {"base_h5": "x.h5"},
        {"base_sha256": "a" * 64},
        {"parent_release": "populace-us-2024-x"},
        {"base_dataset": LOCAL_DATASET},
    ],
)
def test_a_build_folder_and_explicit_base_flags_are_refused_together(
    tmp_path, monkeypatch, extra
):
    install_engine_stand_ins(monkeypatch)
    kwargs = local_build_kwargs(tmp_path)
    with pytest.raises(ValueError, match="do not also pass"):
        annual.build_annual_static_aging(**kwargs, **extra)


def test_a_local_base_needs_its_build_folder(tmp_path, monkeypatch):
    install_engine_stand_ins(monkeypatch)
    kwargs = local_build_kwargs(tmp_path)
    h5 = kwargs.pop("base_artifact_root") / LOCAL_FILE
    kwargs.pop("base_build_dir")
    with pytest.raises(ValueError, match="needs its build folder"):
        annual.build_annual_static_aging(
            **kwargs,
            base_h5=h5,
            base_sha256=sha256(h5),
            parent_release=LOCAL_BUILD,
            base_dataset=LOCAL_DATASET,
        )
    with pytest.raises(ValueError, match="name the base"):
        annual.build_annual_static_aging(**kwargs)


def test_an_h5_outside_the_folder_needs_the_artifact_root(tmp_path, monkeypatch):
    install_engine_stand_ins(monkeypatch)
    kwargs = local_build_kwargs(tmp_path)
    kwargs.pop("base_artifact_root")
    with pytest.raises(ValueError, match="--base-artifact-root"):
        annual.build_annual_static_aging(**kwargs)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (
            lambda m, b: m.update(release="microcosm-us-2024-evidence"),
            "is not a build of",
        ),
        (
            lambda m, b: b.update(build="populace-us-2024-other"),
            "does not name release",
        ),
        (
            lambda m, b: m.update(default_datasets={"national": LOCAL_DATASET}),
            "no default",
        ),
        (
            lambda m, b: m["artifacts"][LOCAL_DATASET].update(path="../x.h5"),
            "bare file name",
        ),
        (
            lambda m, b: m["artifacts"]["gate_summary"].update(kind="microdata"),
            "one microdata",
        ),
    ],
)
def test_a_malformed_base_build_folder_is_refused(tmp_path, change, match):
    kwargs = local_build_kwargs(tmp_path)
    folder = kwargs["base_build_dir"]
    release = json.loads((folder / "release_manifest.json").read_text())
    build = json.loads((folder / "build_manifest.json").read_text())
    change(release, build)
    (folder / "release_manifest.json").write_text(json.dumps(release))
    (folder / "build_manifest.json").write_text(json.dumps(build))
    with pytest.raises(ValueError, match=match):
        annual.read_base_build(folder, kwargs["base_artifact_root"])


def test_the_command_takes_a_build_folder_and_years():
    args = annual.command_parser().parse_args(
        [
            "--base-build-dir=package/releases/microcosm-us-2024-local",
            "--base-artifact-root=package",
            "--ssa-csv=ssa.csv",
            f"--ssa-sha256={'b' * 64}",
            "--model-source=model",
            f"--model-commit={'c' * 40}",
            "--model-version=2.2.1",
            "--output-dir=candidate",
            "--end-year=2027",
            "--years",
            "2026",
            "2027",
        ]
    )
    assert args.years == [2026, 2027]
    assert args.base_build_dir == "package/releases/microcosm-us-2024-local"
    assert (args.base_h5, args.base_sha256, args.parent_release) == (None, None, None)
