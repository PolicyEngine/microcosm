"""The post-staging pre-check of R2's reviewed-fill rules (microcosm#922).

``tools/check_us_acs_local_reviewed_fills.py`` grades a staging summary's
reviewed nulls with R2's own fill constants and finalize gate. These tests
use synthetic summaries and a stand-in engine, so no country engine is
imported.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from microcosm.build.us_runtime.esi_premiums import US_ESI_EMPLOYER_PREMIUM_COLUMN
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
_ACS = "acs_2024_1yr"


@pytest.fixture(scope="module")
def check():
    path = _TEST_PATHS.repository / "tools" / "check_us_acs_local_reviewed_fills.py"
    spec = importlib.util.spec_from_file_location("check_us_acs_fills", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def release(check):
    return check._release_tool()


def _null(column, spines, *, entity="person"):
    return {
        "entity": entity,
        "column": column,
        "dtype": "float",
        "missing_rows": sum(spines.values()),
        "rows": 100,
        "missing_rows_by_spine": spines,
    }


def _resolve(column):
    return 0, "int"


def test_stage_filled_columns_are_the_never_default_columns_r2_stages_fill(
    check, release
) -> None:
    filled = check.stage_filled_never_default(release)

    assert filled < set(release.NEVER_DEFAULT_FILLED)
    # R2's own reasons agree: a stage-filled column's reason says to run an
    # R2 stage first; every other one says to re-run staging.
    for key, reason in release._NEVER_DEFAULT_FILLED_REASONS.items():
        assert ("stage first" in reason) is (key in filled), key
        assert ("re-run staging" in reason) is (key not in filled), key


def test_staged_nulls_split_as_r2_fill_step_treats_them(check, release) -> None:
    stage_filled = check.stage_filled_never_default(release)
    stage_entity, stage_column = sorted(stage_filled)[0]
    staging_entity, staging_column = sorted(
        set(release.NEVER_DEFAULT_FILLED) - stage_filled
    )[0]
    nulls = [
        _null("strike_benefits", {_ACS: 7}),
        _null(stage_column, {_ACS: 3}, entity=stage_entity),
        _null(stage_column, {"asec_puf": 2}, entity=stage_entity),
        # An R2 stage never fills this column: staging must.
        _null(staging_column, {_ACS: 5}, entity=staging_entity),
        _null(US_ESI_EMPLOYER_PREMIUM_COLUMN, {_ACS: 4}),
    ]

    split = check.expected_fills(
        nulls,
        never_default_filled=release.NEVER_DEFAULT_FILLED,
        stage_filled=stage_filled,
        resolve_default=_resolve,
    )

    assert [fill["column"] for fill in split["fills"]] == ["strike_benefits"]
    assert split["fills"][0]["fill_value"] == "0"
    assert split["fills"][0]["missing_rows_by_spine"] == {_ACS: 7}
    assert [item["column"] for item in split["natively_filled"]] == [
        f"{stage_entity}.{stage_column}"
    ]
    assert [item["missing_rows_by_spine"] for item in split["natively_filled"]] == [
        {_ACS: 3}
    ]
    assert [
        (item["missing_rows_by_spine"], item["r2_stage_fills"])
        for item in split["denied"]
    ] == [({"asec_puf": 2}, True), ({_ACS: 5}, False)]
    assert [item["column"] for item in split["esi"]] == [
        f"person.{US_ESI_EMPLOYER_PREMIUM_COLUMN}"
    ]


@pytest.mark.parametrize(
    ("nulls", "version", "passed", "message"),
    [
        ([_null("strike_benefits", {_ACS: 7})], "2.2.1", True, None),
        (
            [_null("not_a_registered_input", {_ACS: 7})],
            "2.2.1",
            False,
            "person.not_a_registered_input",
        ),
        (
            [_null("strike_benefits", {_ACS: 7, "asec_puf": 1})],
            "2.2.1",
            False,
            "person.strike_benefits",
        ),
        ([_null("strike_benefits", {_ACS: 7})], "2.3.0", False, "2.3.0"),
    ],
)
def test_r2_gate_grades_the_expected_fills(
    check, release, nulls, version, passed, message
) -> None:
    split = check.expected_fills(
        nulls,
        never_default_filled=release.NEVER_DEFAULT_FILLED,
        stage_filled=check.stage_filled_never_default(release),
        resolve_default=_resolve,
    )

    report = check.grade(
        split,
        manifest_names=release.REVIEWED_FILL_MANIFESTS,
        never_default_filled=release.NEVER_DEFAULT_FILLED,
        installed_engine_version=version,
    )

    assert report["passed"] is passed
    if message is not None:
        assert any(message in refusal for refusal in report["refusals"])


def test_an_acs_only_null_that_staging_should_have_filled_is_refused(
    check, release
) -> None:
    stage_filled = check.stage_filled_never_default(release)
    staging_filled = sorted(set(release.NEVER_DEFAULT_FILLED) - stage_filled)
    assert staging_filled
    for entity, column in staging_filled:
        split = check.expected_fills(
            [_null(column, {_ACS: 3}, entity=entity)],
            never_default_filled=release.NEVER_DEFAULT_FILLED,
            stage_filled=stage_filled,
            resolve_default=_resolve,
        )

        report = check.grade(
            split,
            manifest_names=release.REVIEWED_FILL_MANIFESTS,
            never_default_filled=release.NEVER_DEFAULT_FILLED,
            installed_engine_version="2.2.1",
        )

        assert report["passed"] is False
        assert report["natively_filled"] == []
        assert any(
            refusal.startswith(f"{entity}.{column}: null on ['{_ACS}']")
            and "re-run staging" in refusal
            for refusal in report["refusals"]
        ), (entity, column)


def test_command_exits_2_on_an_unregistered_fill(
    check, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    variables = {
        "strike_benefits": SimpleNamespace(default_value=0),
        "not_a_registered_input": SimpleNamespace(default_value=0.0),
    }
    monkeypatch.setattr(
        check, "_tax_benefit_system", lambda: SimpleNamespace(variables=variables)
    )
    monkeypatch.setattr(check, "_engine_version", lambda _release: "2.2.1")
    summary = tmp_path / "staging.summary.json"
    out = tmp_path / "report.json"
    engine = {"policyengine_us": "2.2.1", "policyengine_core": "3.32.5"}
    summary.write_text(
        json.dumps(
            {
                "engine_versions": engine,
                "reviewed_engine_input_nulls": [_null("strike_benefits", {_ACS: 7})],
            }
        )
    )

    assert check.main(["--staging-summary", str(summary), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["expected_fills"] == 1

    summary.write_text(
        json.dumps(
            {
                "engine_versions": engine,
                "reviewed_engine_input_nulls": [
                    _null("strike_benefits", {_ACS: 7}),
                    _null("not_a_registered_input", {_ACS: 2}),
                ],
            }
        )
    )
    assert check.main(["--staging-summary", str(summary), "--out", str(out)]) == 2
    refusals = json.loads(out.read_text())["refusals"]
    assert any("not_a_registered_input" in refusal for refusal in refusals)


@pytest.mark.parametrize(
    "engine",
    [None, {"policyengine_us": "2.145.3", "policyengine_core": "3.36.1"}],
    ids=["no-record", "another-engine"],
)
def test_command_exits_2_on_a_staging_run_on_another_engine(
    check, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine
) -> None:
    """Materialize refuses a staging run whose engine is not the installed
    one before it loads the file, so the pre-check does too."""

    monkeypatch.setattr(
        check,
        "_tax_benefit_system",
        lambda: SimpleNamespace(
            variables={"strike_benefits": SimpleNamespace(default_value=0)}
        ),
    )
    monkeypatch.setattr(check, "_engine_version", lambda _release: "2.2.1")
    summary = tmp_path / "staging.summary.json"
    out = tmp_path / "report.json"
    payload = {"reviewed_engine_input_nulls": [_null("strike_benefits", {_ACS: 7})]}
    if engine is not None:
        payload["engine_versions"] = engine
    summary.write_text(json.dumps(payload))

    assert check.main(["--staging-summary", str(summary), "--out", str(out)]) == 2
    report = json.loads(out.read_text())
    assert report["staging_engine_versions"] == engine
    assert any(
        refusal.startswith("staging:") and "re-run staging on this engine" in refusal
        for refusal in report["refusals"]
    )
