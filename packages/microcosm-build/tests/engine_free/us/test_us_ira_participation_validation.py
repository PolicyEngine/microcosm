"""IRA validation uses aligned weights and the maintained batched scorer."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from test_support.paths import paths_for


def _tool():
    path = (
        paths_for("microcosm-build").repository
        / "tools"
        / "validate_us_ira_participation.py"
    )
    name = "test_ira_participation_validation_tool"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _result(values, weights):
    return SimpleNamespace(values=np.asarray(values), weights=np.asarray(weights))


def _scored_values(*, after=False):
    people = [10, 20, 30, 40, 50]
    tax_units = [10, 30, 50]
    return {
        ("traditional_ira_contributions", 2025, None): _result(
            [100, 0, 200, 0, 100 if after else 50], people
        ),
        ("roth_ira_contributions", 2025, None): _result([0, 10, 0, 0, 100], people),
        ("ira_active_participant", 2025, None): _result(
            [0, 1, 0 if after else 1, 1, 1], people
        ),
        ("traditional_ira_deduction", 2025, "tax_unit"): _result(
            [100, 200, 100] if after else [100, 100, 0], tax_units
        ),
        ("dependent_traditional_ira_deduction", 2025, "tax_unit"): _result(
            [0, 0, 100] if after else [0, 100, 0], tax_units
        ),
        ("tax_unit_is_filer", 2025, None): _result(
            [1, 1, 0] if after else [1, 0, 1], tax_units
        ),
    }


def _builder(log, *, change_values=None, stored_column=None):
    reads = []

    def load_frame(path, *, expected_sha256):
        reads.append(path)
        assert reads.count(path) == 1
        assert expected_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        return SimpleNamespace(
            entities=("person", "tax_unit"),
            table=lambda entity: pd.DataFrame(
                columns=[stored_column] if stored_column and entity == "person" else []
            ),
        )

    class Scorer:
        def __init__(self, path, *, maximum_microsim_batch_size, load_frame):
            self.path = path
            self.batch_size = maximum_microsim_batch_size
            self.dataset_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            load_frame(path, expected_sha256=self.dataset_sha256)
            self.results = _scored_values(after=path.name.startswith("after"))
            if change_values is not None:
                change_values(self.results)
            log.append((path.name, "open"))

        def open_consumer(self, name, plan):
            assert name == "ira_participation_validation"
            assert set(plan) == set(self.results)
            self.plan = plan
            scorer = self

            class Consumer:
                def simulate(self, reform):
                    assert reform is None
                    return self

                def calculate(self, variable, period, map_to=None):
                    return scorer.results[(variable, period, map_to)]

            return Consumer()

        def finish_consumer(self, consumer):
            return {
                "dataset_sha256": self.dataset_sha256,
                "maximum_batch_size": self.batch_size,
                "baseline_plan": [list(key) for key in self.plan],
            }

        def close(self):
            log.append((self.path.name, "close"))

    return SimpleNamespace(
        _HouseholdBatchedPostExportScorer=Scorer, _load_frame=load_frame
    )


def test_validation_separates_person_contributions_and_filing_return_deductions(
    tmp_path, monkeypatch
) -> None:
    tool = _tool()
    before = tmp_path / "before.h5"
    after = tmp_path / "after.h5"
    before.write_bytes(b"incumbent artifact")
    after.write_bytes(b"rebuilt artifact")
    log = []
    monkeypatch.setattr(tool, "_load_builder", lambda: _builder(log))
    monkeypatch.setattr(tool, "_read_base_period", lambda path: 2024)
    monkeypatch.setattr(
        tool,
        "_model_system",
        lambda: SimpleNamespace(variables=dict.fromkeys(tool.REQUIRED_VARIABLES)),
    )
    monkeypatch.setattr(
        tool,
        "_package_versions",
        lambda: {
            "policyengine-us": "reviewed-model",
            "policyengine-core": "reviewed-core",
        },
    )

    report = tool.validate_ira_participation(
        before_h5=before,
        after_h5=after,
        year=2025,
        maximum_microsim_batch_size=2,
    )

    assert log == [
        ("before.h5", "open"),
        ("before.h5", "close"),
        ("after.h5", "open"),
        ("after.h5", "close"),
    ]
    baseline = report["before"]
    assert baseline["gross_contributions"]["traditional"] == {
        "contributor_records": 3,
        "weighted_contributors": 90,
        "amount_dollars": 9500,
    }
    assert baseline["gross_contributions"]["roth"] == {
        "contributor_records": 2,
        "weighted_contributors": 70,
        "amount_dollars": 5200,
    }
    participation = baseline["active_participation"]
    assert participation["active_traditional_contributor_records"] == 2
    assert participation["weighted_active_traditional_contributors"] == 80
    assert participation["share_of_traditional_contributors_active"] == pytest.approx(
        8 / 9
    )
    deduction = baseline["model_derived_deduction"]
    assert deduction["grain"] == "tax_unit"
    assert deduction["weighted_positive_tax_units"] == 10
    assert (
        deduction["all_person_amount_dollars_including_dependent_own_returns"] == 4000
    )
    assert deduction["dependent_own_return_amount_dollars"] == 3000
    assert deduction["parent_tax_unit_amount_dollars"] == 1000
    assert deduction["dependent_own_return_counts_available"] is False
    assert deduction["positive_filing_tax_unit_records"] == 1
    assert deduction["weighted_deducting_returns"] == 10
    assert deduction["filing_tax_unit_amount_dollars"] == 1000
    assert report["after_minus_before"] == {
        "traditional_contribution_amount_dollars": 2500,
        "all_person_deduction_amount_dollars_including_dependent_own_returns": 8000,
        "filing_tax_unit_deduction_amount_dollars": 6000,
    }
    assert report["packages"]["policyengine-us"] == "reviewed-model"
    assert report["purpose"] == "diagnostic_validation_not_release_certification"
    for label, path in (("before", before), ("after", after)):
        artifact = report[label]["artifact"]
        assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert artifact["size_bytes"] == path.stat().st_size
        assert artifact["path"] == str(path.resolve())
        assert artifact["verified_base_period"] == 2024
        assert report[label]["post_export_scoring"]["maximum_batch_size"] == 2


def test_source_thousands_convert_to_dollars_and_cross_year_is_explicit(
    tmp_path,
) -> None:
    tool = _tool()
    path = tmp_path / "before.h5"
    path.write_bytes(b"artifact")
    metrics = tool._score_artifact(
        path, year=2025, batch_size=2, builder=_builder([]), verified_base_period=2024
    )
    benchmarks = {
        "gross_contributions": [
            {
                "year": 2022,
                "account_type": "traditional",
                "model_variable": "traditional_ira_contributions",
                "count_unit": "taxpayer",
                "number_of_taxpayers": 45,
                "amount_thousands": 19,
            }
        ],
        "model_derived_deductions": [
            {
                "year": 2025,
                "model_variable": "traditional_ira_deduction",
                "count_unit": "return",
                "number_of_returns": 5,
                "amount_thousands": 2,
            }
        ],
    }
    gross, deduction = tool._benchmark_comparisons(metrics, benchmarks, 2025)
    assert gross["benchmark_amount_dollars"] == 19_000
    assert gross["contributor_count_ratio"] == 2
    assert gross["amount_ratio"] == 0.5
    assert gross["same_tax_year"] is False
    assert deduction["benchmark_amount_dollars"] == 2000
    assert deduction["deducting_return_count_ratio"] == 2
    assert deduction["amount_ratio"] == 0.5
    assert deduction["same_tax_year"] is True


def test_missing_10032_variable_fails_before_opening_a_scorer(
    tmp_path, monkeypatch
) -> None:
    tool = _tool()
    path = tmp_path / "artifact.h5"
    path.write_bytes(b"artifact")
    variables = dict.fromkeys(tool.REQUIRED_VARIABLES)
    del variables["ira_active_participant"]
    monkeypatch.setattr(
        tool, "_model_system", lambda: SimpleNamespace(variables=variables)
    )
    monkeypatch.setattr(tool, "_read_base_period", lambda path: 2024)
    monkeypatch.setattr(
        tool, "_load_builder", lambda: pytest.fail("old model reached scorer")
    )
    with pytest.raises(RuntimeError, match=r"#10032.*ira_active_participant"):
        tool.validate_ira_participation(before_h5=path, after_h5=path, year=2025)


def test_misaligned_person_weights_refuse_a_joint_mask_and_close_scorer(
    tmp_path,
) -> None:
    tool = _tool()
    path = tmp_path / "before.h5"
    path.write_bytes(b"artifact")
    log = []

    def misalign(results):
        results[("ira_active_participant", 2025, None)].weights = np.ones(5)

    with pytest.raises(ValueError, match="entity weights differ"):
        tool._score_artifact(
            path,
            year=2025,
            batch_size=2,
            builder=_builder(log, change_values=misalign),
            verified_base_period=2024,
        )
    assert log == [("before.h5", "open"), ("before.h5", "close")]


def test_no_traditional_contributors_has_no_active_fraction(tmp_path) -> None:
    tool = _tool()
    path = tmp_path / "before.h5"
    path.write_bytes(b"artifact")

    def no_contributors(results):
        results[("traditional_ira_contributions", 2025, None)].values = np.zeros(5)

    metrics = tool._score_artifact(
        path,
        year=2025,
        batch_size=2,
        builder=_builder([], change_values=no_contributors),
        verified_base_period=2024,
    )
    assert metrics["active_participation"]["weighted_traditional_contributors"] == 0
    assert (
        metrics["active_participation"]["share_of_traditional_contributors_active"]
        is None
    )


def test_dependent_own_return_is_not_assigned_to_a_filing_parent(tmp_path) -> None:
    tool = _tool()
    path = tmp_path / "before.h5"
    path.write_bytes(b"artifact")

    def dependent_only(results):
        for variable in (
            "traditional_ira_deduction",
            "dependent_traditional_ira_deduction",
        ):
            results[(variable, 2025, "tax_unit")].values = np.asarray([100, 0, 0])

    metrics = tool._score_artifact(
        path,
        year=2025,
        batch_size=2,
        builder=_builder([], change_values=dependent_only),
        verified_base_period=2024,
    )["model_derived_deduction"]
    assert metrics["all_person_amount_dollars_including_dependent_own_returns"] == 1000
    assert metrics["dependent_own_return_amount_dollars"] == 1000
    assert metrics["weighted_deducting_returns"] == 0
    assert metrics["filing_tax_unit_amount_dollars"] == 0


def test_dependent_own_return_cannot_exceed_the_total_deduction(tmp_path) -> None:
    tool = _tool()
    path = tmp_path / "before.h5"
    path.write_bytes(b"artifact")

    def invalid_dependent_deduction(results):
        results[
            ("dependent_traditional_ira_deduction", 2025, "tax_unit")
        ].values = np.asarray([101, 0, 0])

    with pytest.raises(ValueError, match="Dependent own-return deductions exceed"):
        tool._score_artifact(
            path,
            year=2025,
            batch_size=2,
            builder=_builder([], change_values=invalid_dependent_deduction),
            verified_base_period=2024,
        )


@pytest.mark.parametrize(
    "stored_column",
    [
        "traditional_ira_deduction",
        "dependent_traditional_ira_deduction",
        "ira_active_participant",
        "traditional_ira_contributions",
        "roth_ira_contributions",
    ],
)
def test_stored_model_results_refuse_model_derived_validation(
    tmp_path, stored_column
) -> None:
    tool = _tool()
    path = tmp_path / "before.h5"
    path.write_bytes(b"artifact")
    log = []
    with pytest.raises(
        ValueError,
        match=f"stored model-result inputs are forbidden: person.{stored_column}",
    ):
        tool._score_artifact(
            path,
            year=2025,
            batch_size=2,
            builder=_builder(log, stored_column=stored_column),
            verified_base_period=2024,
        )
    assert not log  # Refusal precedes engine/scoring construction.


def test_already_aged_artifact_refuses_before_a_model_or_scorer_is_built(
    tmp_path, monkeypatch
) -> None:
    tool = _tool()
    path = tmp_path / "annual-2025.h5"
    path.write_bytes(b"artifact")
    monkeypatch.setattr(tool, "_read_base_period", lambda path: 2025)
    monkeypatch.setattr(
        tool, "_model_system", lambda: pytest.fail("aged inputs reached model")
    )
    with pytest.raises(
        ValueError, match=r"_time_period is 2025.*avoid aging annual inputs twice"
    ):
        tool.validate_ira_participation(before_h5=path, after_h5=path, year=2025)


@pytest.mark.parametrize(
    "metadata", [[2024], [2025], [], [2024, 2025], [2024.5], [True]]
)
def test_base_period_reader_reads_only_integer_scalar_metadata(
    tmp_path, monkeypatch, metadata
) -> None:
    tool = _tool()
    path = tmp_path / "artifact.h5"
    reads = []

    class Store:
        def __init__(self, requested_path, *, mode):
            assert requested_path == path
            assert mode == "r"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def keys(self):
            return ["/_time_period", "/person"]

        def get(self, key):
            reads.append(key)
            assert key == "_time_period"
            return pd.Series(metadata)

    monkeypatch.setattr(pd, "HDFStore", Store)
    if metadata in ([2024], [2025]):
        assert tool._read_base_period(path) == metadata[0]
    else:
        with pytest.raises(ValueError, match="_time_period must"):
            tool._read_base_period(path)
    assert reads == ["_time_period"]


@pytest.mark.parametrize("output_alias", ["before", "after", "hardlink"])
def test_output_path_cannot_overwrite_either_artifact(
    tmp_path, monkeypatch, output_alias
) -> None:
    tool = _tool()
    before = tmp_path / "before.h5"
    after = tmp_path / "after.h5"
    before.write_bytes(b"before original")
    after.write_bytes(b"after original")
    if output_alias == "hardlink":
        output = tmp_path / "report.json"
        output.hardlink_to(before)
    else:
        output = before if output_alias == "before" else after
    monkeypatch.setattr(
        tool,
        "validate_ira_participation",
        lambda **kwargs: pytest.fail("unsafe output reached simulation"),
    )
    with pytest.raises(SystemExit) as error:
        tool.main(
            [
                "--before-h5",
                str(before),
                "--after-h5",
                str(after),
                "--year",
                "2025",
                "--out",
                str(output),
            ]
        )
    assert error.value.code == 2
    assert before.read_bytes() == b"before original"
    assert after.read_bytes() == b"after original"


@pytest.mark.parametrize("year", [2022, 2023])
def test_cli_cannot_backcast_base_inputs_before_2024(
    tmp_path, monkeypatch, year
) -> None:
    tool = _tool()
    before = tmp_path / "before.h5"
    after = tmp_path / "after.h5"
    before.write_bytes(b"before original")
    after.write_bytes(b"after original")
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        tool, "_model_system", lambda: pytest.fail("backcast reached model")
    )
    monkeypatch.setattr(
        tool, "_load_builder", lambda: pytest.fail("backcast reached scorer")
    )
    monkeypatch.setattr(
        tool, "_read_base_period", lambda path: pytest.fail("backcast read artifact")
    )
    with pytest.raises(SystemExit) as error:
        tool.main(
            [
                "--before-h5",
                str(before),
                "--after-h5",
                str(after),
                "--year",
                str(year),
                "--out",
                str(output),
            ]
        )
    assert error.value.code == 2
    assert not output.exists()
