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


def test_staged_nulls_split_as_r2_fill_step_treats_them(check, release) -> None:
    never_entity, never_column = sorted(release.NEVER_DEFAULT_FILLED)[0]
    nulls = [
        _null("strike_benefits", {_ACS: 7}),
        _null(never_column, {_ACS: 3}, entity=never_entity),
        _null(never_column, {"asec_puf": 2}, entity=never_entity),
        _null(US_ESI_EMPLOYER_PREMIUM_COLUMN, {_ACS: 4}),
    ]

    split = check.expected_fills(
        nulls,
        never_default_filled=release.NEVER_DEFAULT_FILLED,
        resolve_default=_resolve,
    )

    assert [fill["column"] for fill in split["fills"]] == ["strike_benefits"]
    assert split["fills"][0]["fill_value"] == "0"
    assert split["fills"][0]["missing_rows_by_spine"] == {_ACS: 7}
    assert [item["missing_rows_by_spine"] for item in split["natively_filled"]] == [
        {_ACS: 3}
    ]
    assert [item["missing_rows_by_spine"] for item in split["denied"]] == [
        {"asec_puf": 2}
    ]
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
    summary.write_text(
        json.dumps(
            {"reviewed_engine_input_nulls": [_null("strike_benefits", {_ACS: 7})]}
        )
    )

    assert check.main(["--staging-summary", str(summary), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["expected_fills"] == 1

    summary.write_text(
        json.dumps(
            {
                "reviewed_engine_input_nulls": [
                    _null("strike_benefits", {_ACS: 7}),
                    _null("not_a_registered_input", {_ACS: 2}),
                ]
            }
        )
    )
    assert check.main(["--staging-summary", str(summary), "--out", str(out)]) == 2
    refusals = json.loads(out.read_text())["refusals"]
    assert any("not_a_registered_input" in refusal for refusal in refusals)
