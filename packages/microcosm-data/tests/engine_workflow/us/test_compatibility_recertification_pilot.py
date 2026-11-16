"""One tiny MOCK H5 workflow with the real US constructor and batch scorer.

Test-local runtime, parent and bytecode proof stubs do not authenticate a
release. The real model handles only two complete synthetic households.
"""

from __future__ import annotations

import json
import math

import pandas as pd

from microcosm.data import compatibility_recertification as recert
from test_support.microcosm_data.compatibility_recertification_pilot import (
    mock_tables,
    write_mock_h5,
)
from tools import build_us_compatibility_recertification as scorer_tool
from tools import probe_us_compatibility_recertification as pilot


def test_tiny_mock_h5_retains_source_rows_through_real_native_scoring(
    tmp_path, monkeypatch
):
    import policyengine_us

    tables = mock_tables((2, 1, 3))
    artifact_root = tmp_path / "artifacts"
    release_dir = tmp_path / "MOCK_RELEASE_NOT_QUALIFIED"
    artifact_root.mkdir()
    release_dir.mkdir()
    h5 = artifact_root / "mock_synthetic.h5"
    write_mock_h5(h5, tables)
    initial_h5 = recert.sha256_file(h5)
    spm = {"geography_kind": "county"}
    (release_dir / "calibration_diagnostics.json").write_text(
        json.dumps({"build": {"post_export_scoring": {"spm": spm}}})
    )
    report = {
        "schema_version": 1,
        "dataset": {"filename": h5.name, "sha256": initial_h5},
        "parent": {
            "release_type": "calibration",
            "root_artifacts": {},
            "MOCK_METADATA_ONLY": True,
        },
        "compatibility": {"status": "pending"},
    }
    (release_dir / recert.REPORT_FILE).write_text(json.dumps(report))
    # These stubs exist only in this test. No real parent/runtime claim follows.
    monkeypatch.setattr(
        recert, "validate_recertification_candidate", lambda *_a, **_kw: report
    )
    monkeypatch.setattr(
        scorer_tool, "scoring_runtime", lambda _: {"MOCK_METADATA_ONLY": True}
    )
    monkeypatch.setattr(recert, "require_trusted_bytecode", lambda: None)
    monkeypatch.setattr(recert, "verify_loaded_modules", lambda: 0)
    monkeypatch.setattr(
        recert,
        "bytecode_isolation_status",
        lambda: {"isolated": True, "MOCK_METADATA_ONLY": True},
    )
    monkeypatch.setattr(
        scorer_tool, "_producer_identity", lambda: {"MOCK_METADATA_ONLY": True}
    )
    metric = {
        "metric_id": "mock_income_tax",
        "batching": recert.BATCHING_HOUSEHOLD,
        "entity": "tax_unit",
        "variables": ["income_tax"],
        "combine": "sum",
        "itemizers_only": False,
        "statistic": recert.STATISTIC_SUM,
        "source_variable": "income_tax",
    }
    state_metric = {
        **metric,
        "metric_id": "mock_state_excluded",
        "batching": recert.BATCHING_STATE,
    }
    monkeypatch.setattr(
        recert,
        "drift_metric_definitions",
        lambda _: {"metrics": [metric, state_metric]},
    )
    expected = {}
    person_mask = tables["person"].person_household_id.isin(
        tables["household"].household_id.iloc[:2]
    )
    expected["person"] = tables["person"].loc[person_mask].reset_index(drop=True)
    for entity, frame in tables.items():
        if entity != "person":
            ids = expected["person"][f"person_{entity}_id"]
            expected[entity] = frame.loc[
                frame[f"{entity}_id"].isin(set(ids))
            ].reset_index(drop=True)
    events = []
    native_constructor = policyengine_us.Microsimulation

    def constructor(**kwargs):
        events.append("constructor")
        assert kwargs["spm"] == spm
        assert kwargs["dataset"].years == [2024]
        for entity, frame in expected.items():
            pd.testing.assert_frame_equal(
                getattr(kwargs["dataset"].get_year(2024), entity).reset_index(
                    drop=True
                ),
                frame,
            )
        return native_constructor(**kwargs)

    monkeypatch.setattr(policyengine_us, "Microsimulation", constructor)
    native_guard = pilot._aggregate_guard

    def guard(simulation):
        events.append("guard")
        native_guard(simulation)

    monkeypatch.setattr(pilot, "_aggregate_guard", guard)
    native_total = scorer_tool._BatchScorer.total

    def total(self, metric):
        events.append("total")
        return native_total(self, metric)

    monkeypatch.setattr(scorer_tool._BatchScorer, "total", total)
    output = tmp_path / "engineering-only.json"
    result = pilot.probe_candidate(
        release_dir=release_dir,
        artifact_root=artifact_root,
        output=output,
        compatibility_wheels=(),
        max_households=2,
    )
    assert events == ["constructor", "guard", "guard", "guard", "total", "guard"]
    assert result["kind"] == pilot.PILOT_KIND
    assert result["population_estimate"] is False
    assert result["compatibility_certification"] == "not_attested"
    assert result["selection"]["actual_households"] == 2
    assert result["selection"]["raw_source_input_inventory"] == pilot._frame_inventory(
        expected
    )
    assert result["measurement"]["excluded_metric_ids"] == ["mock_state_excluded"]
    assert list(result["measurement"]["household_metrics"]) == ["mock_income_tax"]
    assert math.isfinite(
        result["measurement"]["household_metrics"]["mock_income_tax"]["value"]
    )
    assert result["runtime"]["packages"] == {"MOCK_METADATA_ONLY": True}
    assert recert.sha256_file(h5) == initial_h5
    assert json.loads(output.read_text()) == result
