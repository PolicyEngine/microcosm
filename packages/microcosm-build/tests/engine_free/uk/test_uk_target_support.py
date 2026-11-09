"""The target-support sidecars and the per-target weight-stretch anatomy."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from microcosm.build.uk_runtime.target_support import (
    TARGET_SUPPORT_MANIFEST_FILE,
    TARGET_SUPPORT_MATRIX_FILE,
    TargetSupportError,
    TargetSupportSidecars,
    anatomy_for_targets,
    load_uk_target_support_sidecars,
    realised_max_weight_ratio,
    target_support_anatomy,
    verify_against_diagnostics,
    write_uk_target_support_sidecars,
)


def _sidecars() -> TargetSupportSidecars:
    # Six households; three compiled rows. Row "fares" is carried by four
    # households, two of which the solver stretched (x4 and x6); row "flat" is
    # carried by every household at one unit; row "signed" mixes signs.
    matrix = sparse.csr_array(
        np.array(
            [
                [10.0, 20.0, 0.0, 5.0, 0.0, 15.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [3.0, -2.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    )
    design = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    final = np.array([4.0, 1.0, 1.0, 6.0, 0.5, 1.0])
    return TargetSupportSidecars(
        matrix=matrix,
        names=("fares@2025", "flat@2025", "signed@2025"),
        target_vector=np.array([100.0, 12.0, 1.0]),
        design_weights=design,
        final_weights=final,
        household_ids=np.array([11, 12, 13, 14, 15, 16]),
    )


def test_anatomy_reports_stretched_mass_beside_the_frame_share() -> None:
    sidecars = _sidecars()
    report = target_support_anatomy(sidecars, "fares")
    # Contributions at final weights: 40, 20, 0, 30, 0, 15 -> 105.
    assert report["name"] == "fares@2025"
    assert report["final_estimate"] == pytest.approx(105.0)
    assert report["design_estimate"] == pytest.approx(50.0)
    assert report["final_relative_error"] == pytest.approx(0.05)
    assert report["design_relative_error"] == pytest.approx(-0.5)
    assert report["carrier_count"] == 4
    # Ranked by absolute contribution: h11 (40), h14 (30), h12 (20), h16 (15).
    assert report["top_1_share"] == pytest.approx(40 / 105)
    assert report["top_5_share"] == pytest.approx(1.0)
    assert [c["household_id"] for c in report["top_carriers"]] == [11, 14, 12, 16]
    # Stretched beyond 3x: h11 (x4) and h14 (x6) carry 70 of 105; beyond 5x
    # only h14 (30 of 105). Frame-wide, weight above 3x is 10 of 13.5.
    three = report["stretched"]["3x"]
    assert three["stretched_mass"] == pytest.approx(70 / 105)
    assert three["stretched_carriers"] == pytest.approx(0.5)
    assert three["frame_weight_share"] == pytest.approx(10 / 13.5)
    five = report["stretched"]["5x"]
    assert five["stretched_mass"] == pytest.approx(30 / 105)
    assert five["stretched_carriers"] == pytest.approx(0.25)
    assert five["frame_weight_share"] == pytest.approx(6 / 13.5)
    ratio = report["carrier_weight_ratio"]
    assert ratio["max"] == pytest.approx(6.0)
    assert ratio["median"] == pytest.approx(2.5)
    # A signed row ranks its dominant carrier by absolute contribution.
    signed = target_support_anatomy(sidecars, "signed@2025", top=2)
    assert signed["final_estimate"] == pytest.approx(3 * 4 - 2 * 1)
    assert [c["household_id"] for c in signed["top_carriers"]] == [11, 12]
    with pytest.raises(TargetSupportError, match="not a compiled row"):
        target_support_anatomy(sidecars, "missing")
    with pytest.raises(ValueError, match="positive"):
        target_support_anatomy(sidecars, "fares", stretch_thresholds=(0.0,))


def test_anatomy_for_targets_resolves_the_near_cap_fraction() -> None:
    sidecars = _sidecars()
    assert realised_max_weight_ratio(sidecars) == pytest.approx(6.0)
    reports = anatomy_for_targets(sidecars, ["fares", "flat"], top=3)
    assert [r["name"] for r in reports] == ["fares@2025", "flat@2025"]
    # Near the cap: carriers above 0.9 x 6 = 5.4, i.e. h14 alone of four.
    assert reports[0]["carrier_share_near_cap"] == pytest.approx(0.25)
    assert reports[1]["carrier_count"] == 6
    assert "top_3_share" in reports[0]


def test_sidecars_round_trip_and_refuse_tampering(tmp_path: Path) -> None:
    sidecars = _sidecars()
    household = pd.DataFrame({"household_id": sidecars.household_ids})
    result = SimpleNamespace(
        problem=SimpleNamespace(
            matrix=sidecars.matrix,
            names=sidecars.names,
            target_vector=sidecars.target_vector,
        ),
        initial_weights=SimpleNamespace(values=sidecars.design_weights),
        weights=sidecars.final_weights,
        weight_entity="household",
        frame=SimpleNamespace(table=lambda entity: household),
    )
    paths = write_uk_target_support_sidecars(result, tmp_path / "attempt")
    assert set(paths) == {"matrix", "vectors", "manifest"}
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["targets"] == 3 and manifest["records"] == 6
    assert manifest["household_ids"] is True
    loaded = load_uk_target_support_sidecars(tmp_path / "attempt")
    assert loaded.names == sidecars.names
    np.testing.assert_allclose(loaded.final_weights, sidecars.final_weights)
    np.testing.assert_allclose(loaded.matrix.toarray(), sidecars.matrix.toarray())
    assert loaded.household_ids is not None
    assert loaded.household_ids.tolist() == [11, 12, 13, 14, 15, 16]
    # A tampered matrix is refused by its digest; a missing file by name.
    sparse.save_npz(
        tmp_path / "attempt" / TARGET_SUPPORT_MATRIX_FILE,
        sparse.csr_array(np.eye(3, 6)),
    )
    with pytest.raises(TargetSupportError, match="sha256"):
        load_uk_target_support_sidecars(tmp_path / "attempt")
    (tmp_path / "attempt" / TARGET_SUPPORT_MANIFEST_FILE).unlink()
    with pytest.raises(TargetSupportError, match="missing"):
        load_uk_target_support_sidecars(tmp_path / "attempt")


def test_sidecars_refuse_misaligned_vectors() -> None:
    matrix = sparse.csr_array(np.ones((2, 3)))
    with pytest.raises(TargetSupportError, match="weight vectors disagree"):
        TargetSupportSidecars(
            matrix=matrix,
            names=("a", "b"),
            target_vector=np.array([1.0, 2.0]),
            design_weights=np.ones(3),
            final_weights=np.ones(2),
            household_ids=None,
        )
    with pytest.raises(TargetSupportError, match="target rows disagree"):
        TargetSupportSidecars(
            matrix=matrix,
            names=("a",),
            target_vector=np.array([1.0, 2.0]),
            design_weights=np.ones(3),
            final_weights=np.ones(3),
            household_ids=None,
        )


def test_verification_against_diagnostics_refuses_another_attempt() -> None:
    sidecars = _sidecars()
    reports = anatomy_for_targets(sidecars, ["fares", "flat"])
    diagnostics = {
        "targets": [
            {"name": "fares@2025", "final_estimate": 105.0},
            {"name": "flat", "final_estimate": 13.5},
        ]
    }
    verify_against_diagnostics(reports, diagnostics)
    with pytest.raises(TargetSupportError, match="differs from the diagnostics"):
        verify_against_diagnostics(
            reports, {"targets": [{"name": "fares@2025", "final_estimate": 90.0}]}
        )
    with pytest.raises(TargetSupportError, match="no final_estimate"):
        verify_against_diagnostics(reports, {"targets": []})


def test_cli_prints_and_writes_the_reports(tmp_path: Path, capsys) -> None:
    from tools.diagnose_uk_target_support import main

    sidecars = _sidecars()
    result = SimpleNamespace(
        problem=SimpleNamespace(
            matrix=sidecars.matrix,
            names=sidecars.names,
            target_vector=sidecars.target_vector,
        ),
        initial_weights=sidecars.design_weights,
        weights=sidecars.final_weights,
        weight_entity="household",
        frame=None,
    )
    attempt = tmp_path / "attempt"
    write_uk_target_support_sidecars(result, attempt)
    diagnostics = attempt / "calibration_diagnostics.json"
    diagnostics.write_text(
        json.dumps({"targets": [{"name": "fares@2025", "final_estimate": 105.0}]})
    )
    out = tmp_path / "anatomy.json"
    code = main(
        [
            "--attempt-dir",
            str(attempt),
            "--diagnostics",
            str(diagnostics),
            "--prefix",
            "fares",
            "--stretch",
            "2",
            "--json",
            str(out),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "fares@2025" in printed and ">2x" in printed
    written = json.loads(out.read_text())
    assert written[0]["stretched"]["2x"]["stretched_mass"] == pytest.approx(70 / 105)
