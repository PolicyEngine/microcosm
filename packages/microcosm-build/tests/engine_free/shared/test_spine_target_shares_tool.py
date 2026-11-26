"""tools/spine_target_shares.py reads saved inputs and needs no build (#1729).

Every number below is made up for the test.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from test_support.paths import paths_for


def _tool():
    path = paths_for("microcosm-build").repository / "tools" / "spine_target_shares.py"
    spec = importlib.util.spec_from_file_location("spine_target_shares_tool", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


#: Five households: two donor rows (heavy) and three survey rows.
_HOUSEHOLDS = pd.DataFrame(
    {
        "household_id": [1, 2, 3, 4, 5],
        "household_spine": ["donor", "donor", "acs", "acs", "acs"],
        "household_source_id": [10, 10, 1, 2, 3],
        # Numeric codes, as the lean frame stores them.
        "state_fips": [6, 36, 6, 6, 36],
        "congressional_district_geoid": [601, 3601, 601, 602, 3601],
    }
)
_DESIGN = np.asarray([27.0, 27.0, 1.0, 1.0, 1.0])
_ROLES = [
    {
        "name": "pop_state_06",
        "family": "census_population_ladder",
        "geography_level": "state",
        "state_fips": "06",
        "congressional_district_geoid": None,
        "role": "train",
    },
    {
        "name": "pop_cd_0601",
        "family": "census_population_ladder",
        "geography_level": "congressional_district",
        "state_fips": "06",
        "congressional_district_geoid": "0601",
        "role": "train",
    },
    {
        "name": "irs_soi.ca.capital_gains",
        "family": "irs_soi",
        "geography_level": "state",
        "state_fips": "06",
        "congressional_district_geoid": None,
        "role": "holdout",
    },
]
_MATRIX = sparse.csr_array(
    np.asarray(
        [
            [2.0, 0.0, 3.0, 1.0, 0.0],  # persons in California
            [2.0, 0.0, 3.0, 0.0, 0.0],  # persons in district 0601
            [900.0, 0.0, 0.0, 0.0, 0.0],  # an amount only the donor row carries
        ],
        dtype=np.float32,
    )
)


def test_the_us_adapter_scopes_each_target_to_its_own_area() -> None:
    inputs = _tool().us_acs_local_inputs(_HOUSEHOLDS, _ROLES)

    assert inputs["names"] == [role["name"] for role in _ROLES]
    assert inputs["target_scopes"] == [
        ("state", "06"),
        ("congressional_district", "0601"),
        ("state", "06"),
    ]
    # Household codes are padded the way the roles spell them.
    assert inputs["scope_codes"]["state"].tolist() == ["06", "36", "06", "06", "36"]
    assert inputs["scope_codes"]["congressional_district"].tolist() == [
        "0601",
        "3601",
        "0601",
        "0602",
        "3601",
    ]
    assert "county" not in inputs["scope_codes"]
    assert inputs["groups"][2] == "irs_soi/state"
    assert inputs["roles"] == ["train", "train", "holdout"]
    with pytest.raises(SystemExit, match="household_spine"):
        _tool().us_acs_local_inputs(_HOUSEHOLDS.drop(columns="household_spine"), _ROLES)


def test_the_report_reads_a_checkpoint_at_design_and_calibrated_weights(
    tmp_path, monkeypatch, capsys
) -> None:
    tool = _tool()
    # A stand-in for the release tool: the checkpoint loader and nothing else.
    release = ModuleType("build_us_acs_local_release")
    release.load_checkpoint_surface = lambda checkpoint_dir: (
        SimpleNamespace(table=lambda entity: _HOUSEHOLDS),
        _DESIGN,
        None,
        _ROLES,
        _MATRIX,
    )
    monkeypatch.setitem(sys.modules, "build_us_acs_local_release", release)
    np.savez(tmp_path / "weights_latest.npz", weights=[5.0, 27.0, 9.0, 8.0, 8.0])
    out = tmp_path / "report" / "shares.json"

    assert (
        tool.main(
            [
                "--us-acs-local-checkpoint",
                str(tmp_path),
                "--out",
                str(out),
                "--spine",
                "donor",
                "--top",
                "2",
            ]
        )
        == 0
    )

    report = json.loads(out.read_text())
    assert set(report) == {"design", "calibrated"}
    design = report["design"]
    assert design["spines"] == ["acs", "donor"]
    by_name = {row["name"]: row for row in design["targets"]}
    # California holds 27 of 29 on its donor row, which carries 54 of 58 persons.
    assert by_name["pop_state_06"]["mass_share"]["donor"] == pytest.approx(27 / 29)
    assert by_name["pop_state_06"]["share"]["donor"] == pytest.approx(54 / 58)
    # The amount is carried by the donor row alone, by one source household.
    capital = by_name["irs_soi.ca.capital_gains"]
    assert capital["share"] == {"acs": 0.0, "donor": 1.0}
    assert capital["carriers"] == {"acs": 0, "donor": 1}
    assert capital["role"] == "holdout"
    assert design["by_family_and_level"][0]["group"] == (
        "census_population_ladder/state"
    )
    # The calibrated weights moved mass off the California donor row.
    calibrated = {row["name"]: row for row in report["calibrated"]["targets"]}
    assert calibrated["pop_state_06"]["mass_share"]["donor"] == pytest.approx(5 / 22)
    printed = capsys.readouterr().out
    assert "[design] 3 targets" in printed
    assert "irs_soi.ca.capital_gains" in printed


def test_the_generic_inputs_need_only_a_matrix_weights_and_spines(
    tmp_path, capsys
) -> None:
    tool = _tool()
    sparse.save_npz(tmp_path / "matrix.npz", sparse.csr_matrix(_MATRIX))
    np.save(tmp_path / "weights.npy", _DESIGN)
    np.savez(
        tmp_path / "labels.npz", spines=_HOUSEHOLDS["household_spine"].to_numpy(str)
    )
    (tmp_path / "names.json").write_text(json.dumps(["a", "b", "c"]))
    out = tmp_path / "shares.json"

    assert (
        tool.main(
            [
                "--matrix",
                str(tmp_path / "matrix.npz"),
                "--weights",
                str(tmp_path / "weights.npy"),
                "--spines",
                str(tmp_path / "labels.npz"),
                "--names",
                str(tmp_path / "names.json"),
                "--minimum-carriers",
                "2",
                "--out",
                str(out),
            ]  # fmt: skip
        )
        == 0
    )
    report = json.loads(out.read_text())["weights"]
    assert report["minimum_carriers"] == 2
    # Each target has a spine with one carrier, so the floor withholds all.
    assert report["n_withheld"] == 3
    assert all("share" not in row for row in report["targets"])
    assert "3 withheld" in capsys.readouterr().out


def test_the_cli_refuses_a_mix_of_the_two_inputs(tmp_path) -> None:
    tool = _tool()
    for argv in (
        [],
        ["--matrix", "m.npz"],
        ["--us-acs-local-checkpoint", str(tmp_path), "--matrix", "m.npz"],
        ["--matrix", "m", "--weights", "w", "--spines", "s", "--minimum-carriers", "0"],
    ):
        with pytest.raises(SystemExit):
            tool._parse_args(argv)
