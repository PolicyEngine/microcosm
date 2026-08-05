"""Disclosure control and classification in the #612 Frame-linkage preflight.

The preflight's output is written to be posted on the tracking issue, which
makes it a publication under UK Data Service EUL CD137 clause 8 / CD171
§5.2.1: no unit-record values, nothing reported from one or two cases. Frame's
own exception messages embed real ids, so the tool must classify without ever
echoing them — these tests plant a sentinel id in violating tables and assert
it appears nowhere in any output.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]

SENTINEL_ID = 987654321


def _preflight_module():
    path = ROOT / "tools" / "preflight_uk_frame_linkage.py"
    spec = importlib.util.spec_from_file_location("preflight_uk_frame_linkage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREFLIGHT = _preflight_module()


def _clean_tables() -> dict[str, pd.DataFrame]:
    return {
        "person": pd.DataFrame(
            {
                "person_id": [1, 2, 3, 4],
                "person_benunit_id": [10, 10, 20, 20],
                "person_household_id": [100, 100, 200, 200],
                "age": [34.0, 3.0, 61.0, 59.0],
            }
        ),
        "benunit": pd.DataFrame({"benunit_id": [10, 20]}),
        "household": pd.DataFrame(
            {
                "household_id": [100, 200],
                "household_weight": [1000.0, 2000.0],
            }
        ),
    }


def _violating_tables() -> dict[str, pd.DataFrame]:
    """Every violation class at once, each below the SDC minimum, with a
    sentinel id on the orphaned benunit row (the value a leak would expose)."""

    return {
        "person": pd.DataFrame(
            {
                "person_id": [1, 2, 3, 4],
                "person_benunit_id": [10, 10, 20, 20],
                # Benunit 20 spans households 200 and 300 (split benunit);
                # household 300 does not exist (dangling membership).
                "person_household_id": [100, 100, 200, 300],
            }
        ),
        # Unsorted, duplicated, and one orphan row nobody references.
        "benunit": pd.DataFrame({"benunit_id": [20, 10, 10, SENTINEL_ID]}),
        # Unsorted ids and a negative weight.
        "household": pd.DataFrame(
            {
                "household_id": [200, 100],
                "household_weight": [1000.0, -5.0],
            }
        ),
    }


def _write_artifact(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    with pd.HDFStore(path, mode="w") as store:
        for name, table in tables.items():
            store.put(name, table)
        store.put("time_period", pd.Series(["2023"]))


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_json_loads(raw: str) -> dict[str, object]:
    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    return json.loads(raw, parse_constant=reject_nonstandard_constant)


def test_sdc_count_masks_small_nonzero_counts_only() -> None:
    assert PREFLIGHT.MINIMUM_SDC_COUNT == 3
    assert PREFLIGHT.sdc_count(0, minimum=10) == 0
    assert PREFLIGHT.sdc_count(3, minimum=10) == "< 10"
    assert PREFLIGHT.sdc_count(9, minimum=10) == "< 10"
    assert PREFLIGHT.sdc_count(10, minimum=10) == 10
    assert PREFLIGHT.sdc_count(25, minimum=10) == 25


@pytest.mark.parametrize("minimum", [1, 2])
def test_sdc_count_rejects_a_minimum_below_the_authoritative_floor(
    minimum: int,
) -> None:
    with pytest.raises(ValueError, match="at least 3"):
        PREFLIGHT.sdc_count(2, minimum=minimum)


def test_cli_rejects_an_sdc_minimum_below_the_floor_before_file_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_audit(*_args, **_kwargs):
        pytest.fail("an invalid SDC floor must be rejected before H5 access")

    monkeypatch.setattr(PREFLIGHT, "preflight_artifact", unexpected_audit)

    exit_code = PREFLIGHT.main(
        [str(tmp_path / "does-not-exist.h5"), "--sdc-minimum-count", "2"]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "at least 3" in captured.err


def test_weight_total_requires_the_same_minimum_nonzero_carriers() -> None:
    thin = pd.DataFrame({"household_weight": [4.0, 0.0, 0.0]})
    at_floor = pd.DataFrame({"household_weight": [1.0, 2.0, 3.0]})
    zero = pd.DataFrame({"household_weight": [0.0]})

    thin_report = PREFLIGHT.classify_household_weights(thin, minimum=3)
    floor_report = PREFLIGHT.classify_household_weights(at_floor, minimum=3)
    zero_report = PREFLIGHT.classify_household_weights(zero, minimum=3)

    assert thin_report["total_mass"] is None
    assert thin_report["total_mass_suppressed"] is True
    assert floor_report["total_mass"] == 6.0
    assert floor_report["total_mass_suppressed"] is False
    assert zero_report["total_mass"] == 0.0
    assert zero_report["total_mass_suppressed"] is False


def test_clean_artifact_constructs_a_frame(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    path = tmp_path / "clean.h5"
    _write_artifact(path, _clean_tables())

    report = PREFLIGHT.preflight_artifact(path, minimum=10)

    assert report["frame_constructed"] is True
    assert report["frame_construction_failure_reason"] is None
    assert report["preflight_passed"] is True
    assert report["time_period"] == "2023"
    assert report["household_weight_kind"] == "design"
    assert report["column_collisions"] == []
    for group in ("benunit", "household"):
        linkage = report["linkage"][group]
        assert linkage["ids_sorted_ascending"] is True
        assert linkage["orphaned_group_rows"] == 0
        assert linkage["dangling_memberships"] == 0
        assert linkage["id_duplicated"] == 0
    assert report["linkage"]["split_benunits"] == 0
    weights = report["household_weights"]
    assert weights["negative"] == 0
    assert weights["total_mass"] is None
    assert weights["total_mass_suppressed"] is True
    assert report["tables"]["person"]["rows"] == 4


def test_violations_are_classified_masked_and_never_leak_ids(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    path = tmp_path / "violating.h5"
    _write_artifact(path, _violating_tables())

    report = PREFLIGHT.preflight_artifact(path, minimum=10)

    assert report["frame_constructed"] is False
    assert report["frame_construction_failure_reason"]
    assert report["preflight_passed"] is False
    benunit = report["linkage"]["benunit"]
    assert benunit["ids_sorted_ascending"] is False
    assert benunit["id_duplicated"] == "< 10"
    assert benunit["orphaned_group_rows"] == "< 10"
    household = report["linkage"]["household"]
    assert household["ids_sorted_ascending"] is False
    assert household["dangling_memberships"] == "< 10"
    assert report["linkage"]["split_benunits"] == "< 10"
    assert report["household_weights"]["negative"] == "< 10"
    assert report["reserved_weight_column_collisions"] == []

    # The canary: the planted unit-record id appears nowhere in the report.
    rendered = json.dumps(report)
    assert str(SENTINEL_ID) not in rendered


def test_main_reports_both_artifacts_and_exits_nonzero_on_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    clean = tmp_path / "clean.h5"
    violating = tmp_path / "violating.h5"
    json_out = tmp_path / "report.json"
    _write_artifact(clean, _clean_tables())
    _write_artifact(violating, _violating_tables())

    exit_code = PREFLIGHT.main(
        [str(clean), str(violating), "--json-out", str(json_out)]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    report = json.loads(captured.out)
    assert report["sdc_minimum_count"] == 10
    constructed = {
        Path(artifact["path"]).name: artifact["frame_constructed"]
        for artifact in report["artifacts"]
    }
    assert constructed == {"clean.h5": True, "violating.h5": False}
    # The canary holds across every output channel: stdout, stderr, and the
    # JSON file all omit the planted unit-record id.
    assert str(SENTINEL_ID) not in captured.out
    assert str(SENTINEL_ID) not in captured.err
    assert str(SENTINEL_ID) not in json_out.read_text()


def test_main_exits_zero_when_every_artifact_constructs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    clean = tmp_path / "clean.h5"
    _write_artifact(clean, _clean_tables())

    assert PREFLIGHT.main([str(clean)]) == 0
    assert json.loads(capsys.readouterr().out)["artifacts"][0]["frame_constructed"]


@pytest.mark.parametrize("alias_kind", ["direct", "resolved", "symlink", "hardlink"])
def test_main_rejects_every_json_output_alias_before_auditing_any_input(
    alias_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    unrelated = tmp_path / "unrelated.h5"
    candidate = tmp_path / "candidate.h5"
    unrelated.write_bytes(b"unrelated artifact bytes")
    candidate.write_bytes(b"certified candidate bytes")
    if alias_kind == "direct":
        json_out = candidate
    elif alias_kind == "resolved":
        nested = tmp_path / "nested"
        nested.mkdir()
        json_out = nested / ".." / candidate.name
    elif alias_kind == "symlink":
        json_out = tmp_path / "candidate-symlink.json"
        json_out.symlink_to(candidate)
    else:
        json_out = tmp_path / "candidate-hardlink.json"
        json_out.hardlink_to(candidate)
    before = {path: _fingerprint(path) for path in (unrelated, candidate)}
    audited: list[Path] = []

    def record_unexpected_audit(path: Path, *, minimum: int) -> dict[str, object]:
        audited.append(path)
        return {
            "path": str(path),
            "frame_constructed": True,
            "preflight_passed": True,
        }

    monkeypatch.setattr(PREFLIGHT, "preflight_artifact", record_unexpected_audit)

    exit_code = PREFLIGHT.main(
        [str(unrelated), str(candidate), "--json-out", str(json_out)]
    )

    assert exit_code == 2
    assert audited == []
    assert {path: _fingerprint(path) for path in before} == before
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "must not alias any H5 input" in captured.err


def test_normal_run_preserves_every_h5_fingerprint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    json_out = tmp_path / "report.json"
    _write_artifact(first, _clean_tables())
    _write_artifact(second, _clean_tables())
    before = {path: _fingerprint(path) for path in (first, second)}

    exit_code = PREFLIGHT.main([str(first), str(second), "--json-out", str(json_out)])

    assert exit_code == 0
    assert {path: _fingerprint(path) for path in before} == before
    assert len(json.loads(capsys.readouterr().out)["artifacts"]) == 2
    assert json_out.exists()


def test_attrless_h5_has_the_documented_design_metadata_default(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "attrless.h5"
    _write_artifact(path, _clean_tables())
    with h5py.File(path, mode="r") as handle:
        assert "populace_household_weight_kind" not in handle.attrs
        assert "populace_mass_log_json" not in handle.attrs

    report = PREFLIGHT.preflight_artifact(path, minimum=10)

    assert report["audit_completed"] is True
    assert report["household_weight_kind"] == "design"


def test_null_membership_is_classified_and_has_a_sanitized_frame_reason(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    tables = _clean_tables()
    tables["person"].loc[0, "person_benunit_id"] = float("nan")
    path = tmp_path / "null-membership.h5"
    _write_artifact(path, tables)

    report = PREFLIGHT.preflight_artifact(path, minimum=10)

    benunit = report["linkage"]["benunit"]
    assert benunit["membership_na"] == "< 10"
    assert benunit["orphaned_group_rows"] == 0
    assert benunit["dangling_memberships"] == 0
    assert report["frame_constructed"] is False
    assert report["frame_construction_failure_reason"] == (
        "Frame rejects missing group memberships."
    )


def test_reserved_weight_column_collision_is_reported_separately(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    tables = _clean_tables()
    tables["person"]["person_weight"] = [1.0, 1.0, 1.0, 1.0]
    path = tmp_path / "reserved-column.h5"
    _write_artifact(path, tables)

    report = PREFLIGHT.preflight_artifact(path, minimum=10)

    assert report["column_collisions"] == []
    assert report["reserved_weight_column_collisions"] == ["person.person_weight"]
    assert report["frame_constructed"] is False
    assert report["frame_construction_failure_reason"] == (
        "Frame rejects reserved weight-column collisions."
    )


def test_malformed_weight_is_sanitized_and_later_artifact_is_still_audited(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    malformed = tmp_path / "malformed.h5"
    clean = tmp_path / "clean-after-malformed.h5"
    json_out = tmp_path / "batch-report.json"
    malformed_tables = _clean_tables()
    malformed_tables["household"]["household_weight"] = [
        "UNIT_SENTINEL",
        "2000.0",
    ]
    _write_artifact(malformed, malformed_tables)
    _write_artifact(clean, _clean_tables())

    exit_code = PREFLIGHT.main(
        [str(malformed), str(clean), "--json-out", str(json_out)]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    report = _strict_json_loads(captured.out)
    first, second = report["artifacts"]
    assert Path(first["path"]).name == malformed.name
    assert first["audit_completed"] is True
    assert first["household_weights"]["conversion_failures"] == "< 10"
    assert first["frame_constructed"] is False
    assert first["frame_construction_failure_reason"] == (
        "Frame rejects non-numeric household weights."
    )
    assert Path(second["path"]).name == clean.name
    assert second["audit_completed"] is True
    assert second["frame_constructed"] is True
    for channel in (captured.out, captured.err, json_out.read_text()):
        assert "UNIT_SENTINEL" not in channel


def test_non_finite_weight_emits_strict_json_without_exact_total(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    path = tmp_path / "non-finite.h5"
    json_out = tmp_path / "strict-report.json"
    tables = _clean_tables()
    tables["household"]["household_weight"] = [float("inf"), 2000.0]
    _write_artifact(path, tables)

    exit_code = PREFLIGHT.main([str(path), "--json-out", str(json_out)])
    captured = capsys.readouterr()

    assert exit_code == 1
    report = _strict_json_loads(captured.out)
    weights = report["artifacts"][0]["household_weights"]
    assert weights["non_finite"] == "< 10"
    assert weights["total_mass"] is None
    _strict_json_loads(json_out.read_text())
    assert "Infinity" not in captured.out
    assert "NaN" not in captured.out


def test_split_benunit_fails_preflight_even_when_frame_constructs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("tables")
    pytest.importorskip("h5py")
    tables = _clean_tables()
    # Benunit 10 spans two existing households; every ordinary Frame linkage
    # invariant still holds, isolating the nesting check shared with #610.
    tables["person"]["person_household_id"] = [100, 200, 200, 200]
    path = tmp_path / "split-benunit.h5"
    _write_artifact(path, tables)

    exit_code = PREFLIGHT.main([str(path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    artifact = _strict_json_loads(captured.out)["artifacts"][0]
    assert artifact["frame_constructed"] is True
    assert artifact["frame_construction_failure_reason"] is None
    assert artifact["linkage"]["split_benunits"] == "< 10"
    assert artifact["preflight_passed"] is False
    assert artifact["preflight_failure_reasons"] == [
        "Benunits span multiple households."
    ]
    assert "do not pass linkage preflight" in captured.err
