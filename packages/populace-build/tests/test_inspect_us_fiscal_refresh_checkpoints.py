import importlib.util
from pathlib import Path


def _load_inspector_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "inspect_us_fiscal_refresh_checkpoints.py"
    spec = importlib.util.spec_from_file_location(
        "inspect_us_fiscal_refresh_checkpoints", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str = "payload") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test__given_matching_cd_support_provenance__then_next_action_is_release_builder(
    tmp_path,
) -> None:
    inspector = _load_inspector_module()
    base_h5 = _write(tmp_path / "base.h5")
    ledger_facts = _write(tmp_path / "facts.jsonl")
    crosswalk = _write(tmp_path / "crosswalk.csv")
    out = tmp_path / "out"
    out.mkdir()
    expected_sha256 = inspector._sha256(crosswalk)

    payload = inspector.inspect_checkpoints(
        base_h5=base_h5,
        ledger_facts=ledger_facts,
        out=out,
        congressional_district_vintage_crosswalk=crosswalk,
        h5_attr_reader=lambda path: {
            "readable": True,
            "attrs": {
                inspector.CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR: (
                    expected_sha256
                ),
                inspector.CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR: (
                    inspector.CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE
                ),
            },
            "household_congressional_district_geoid": {
                "exists": True,
                "positive_unique_count": 436,
            },
            "error": None,
        },
    )

    assert payload["support_provenance"]["ready"] is True
    assert payload["recommended_next_action"] == "run_fiscal_refresh_release_builder"
    assert (
        "CD-vintage work can stay target-side"
        in payload["support_provenance"]["message"]
    )


def test__given_cd_lookup_without_provenance__then_next_action_is_stamp_attrs(
    tmp_path,
) -> None:
    inspector = _load_inspector_module()
    base_h5 = _write(tmp_path / "base.h5")
    ledger_facts = _write(tmp_path / "facts.jsonl")
    crosswalk = _write(tmp_path / "crosswalk.csv")

    payload = inspector.inspect_checkpoints(
        base_h5=base_h5,
        ledger_facts=ledger_facts,
        out=tmp_path / "out",
        congressional_district_vintage_crosswalk=crosswalk,
        h5_attr_reader=lambda path: {
            "readable": True,
            "attrs": {},
            "household_congressional_district_geoid": {
                "exists": True,
                "positive_unique_count": 436,
            },
            "error": None,
        },
    )

    assert payload["support_provenance"]["ready"] is False
    assert payload["recommended_next_action"] == "stamp_cd_provenance_attrs"
    assert "Stamp the provenance attrs" in payload["support_provenance"]["message"]


def test__given_missing_cd_lookup__then_next_action_is_support_h5(tmp_path) -> None:
    inspector = _load_inspector_module()
    base_h5 = _write(tmp_path / "base.h5")
    ledger_facts = _write(tmp_path / "facts.jsonl")
    crosswalk = _write(tmp_path / "crosswalk.csv")

    payload = inspector.inspect_checkpoints(
        base_h5=base_h5,
        ledger_facts=ledger_facts,
        out=tmp_path / "out",
        congressional_district_vintage_crosswalk=crosswalk,
        h5_attr_reader=lambda path: {
            "readable": True,
            "attrs": {},
            "household_congressional_district_geoid": {"exists": False},
            "error": None,
        },
    )

    assert payload["support_provenance"]["ready"] is False
    assert (
        payload["recommended_next_action"] == "build_support_h5_with_current_cd_lookup"
    )
    assert "Build the support H5 once" in payload["support_provenance"]["message"]


def test__given_h5_attrs_unreadable__then_next_action_names_us_extra(
    tmp_path,
) -> None:
    inspector = _load_inspector_module()
    base_h5 = _write(tmp_path / "base.h5")
    ledger_facts = _write(tmp_path / "facts.jsonl")
    crosswalk = _write(tmp_path / "crosswalk.csv")

    payload = inspector.inspect_checkpoints(
        base_h5=base_h5,
        ledger_facts=ledger_facts,
        out=tmp_path / "out",
        congressional_district_vintage_crosswalk=crosswalk,
        h5_attr_reader=lambda path: {
            "readable": False,
            "attrs": {},
            "read_error_kind": "missing_h5py",
            "error": "h5py is not installed",
        },
    )

    assert (
        payload["recommended_next_action"]
        == "rerun_preflight_with_populace_build_us_extra"
    )
    assert payload["support_provenance"]["message"] == "h5py is not installed"


def test__given_unreadable_h5__then_next_action_names_base_h5(tmp_path) -> None:
    inspector = _load_inspector_module()
    base_h5 = _write(tmp_path / "base.h5")
    ledger_facts = _write(tmp_path / "facts.jsonl")
    crosswalk = _write(tmp_path / "crosswalk.csv")

    payload = inspector.inspect_checkpoints(
        base_h5=base_h5,
        ledger_facts=ledger_facts,
        out=tmp_path / "out",
        congressional_district_vintage_crosswalk=crosswalk,
        h5_attr_reader=lambda path: {
            "readable": False,
            "attrs": {},
            "read_error_kind": "h5_read_error",
            "error": "Could not read H5 file",
        },
    )

    assert payload["recommended_next_action"] == "fix_unreadable_base_h5"
    assert payload["support_provenance"]["message"] == "Could not read H5 file"


def test__given_default_reader_gets_invalid_h5__then_payload_reports_read_error(
    tmp_path,
) -> None:
    inspector = _load_inspector_module()
    bad_h5 = _write(tmp_path / "not-really.h5", "not hdf5")

    result = inspector.read_h5_provenance(bad_h5)

    assert result["readable"] is False
    assert result["read_error_kind"] == "h5_read_error"
    assert "Could not read H5 file" in result["error"]


def test__given_missing_crosswalk__then_support_status_says_requested_missing(
    tmp_path,
) -> None:
    inspector = _load_inspector_module()
    base_h5 = _write(tmp_path / "base.h5")
    ledger_facts = _write(tmp_path / "facts.jsonl")

    payload = inspector.inspect_checkpoints(
        base_h5=base_h5,
        ledger_facts=ledger_facts,
        out=tmp_path / "out",
        congressional_district_vintage_crosswalk=tmp_path / "missing.csv",
    )

    assert (
        payload["recommended_next_action"]
        == "fix_missing_inputs:congressional_district_vintage_crosswalk"
    )
    assert (
        "requested but the crosswalk file is missing"
        in payload["support_provenance"]["message"]
    )


def test__given_crosswalk_directory__then_next_action_is_missing_input(
    tmp_path,
) -> None:
    inspector = _load_inspector_module()
    base_h5 = _write(tmp_path / "base.h5")
    ledger_facts = _write(tmp_path / "facts.jsonl")
    crosswalk_dir = tmp_path / "crosswalk-dir"
    crosswalk_dir.mkdir()

    payload = inspector.inspect_checkpoints(
        base_h5=base_h5,
        ledger_facts=ledger_facts,
        out=tmp_path / "out",
        congressional_district_vintage_crosswalk=crosswalk_dir,
    )

    assert (
        payload["recommended_next_action"]
        == "fix_missing_inputs:congressional_district_vintage_crosswalk"
    )
    assert (
        "requested but the crosswalk file is missing"
        in payload["support_provenance"]["message"]
    )


def test__given_existing_npz__then_checkpoint_says_summary_only(tmp_path) -> None:
    inspector = _load_inspector_module()
    base_h5 = _write(tmp_path / "base.h5")
    ledger_facts = _write(tmp_path / "facts.jsonl")
    artifact_root = tmp_path / "out" / "artifacts"
    _write(artifact_root / inspector.CALIBRATION_FILENAME, "npz")

    payload = inspector.inspect_checkpoints(
        base_h5=base_h5,
        ledger_facts=ledger_facts,
        out=tmp_path / "out",
    )

    calibration_npz = payload["checkpoints"]["calibration_npz"]
    assert calibration_npz["exists"] is True
    assert "not sufficient to recalibrate" in calibration_npz["note"]
