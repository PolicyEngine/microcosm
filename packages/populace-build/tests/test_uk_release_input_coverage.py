"""UK enhanced-FRS release input-column coverage, isolated from PE-UK."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from populace.build.uk_runtime import (
    UK_LOADER_INPUT_ALIASES,
    UKEffectiveMassCoveragePolicy,
    PolicyEngineUKCoverageEngine,
    UKReleaseInputColumn,
    UKReleaseInputCoverageManifest,
    assert_uk_release_input_coverage_manifest_current,
    load_efrs_parity_known_gaps,
    load_efrs_parity_reference,
    load_uk_release_input_coverage_manifest,
    uk_release_input_coverage_gate,
)
from populace.frame import EntitySchema, Frame, WeightKind, Weights

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _person_frame(columns: dict[str, np.ndarray]) -> Frame:
    n = len(next(iter(columns.values())))
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, n + 1, dtype="int64"),
            "person_benunit_id": np.ones(n, dtype="int64"),
            "person_household_id": np.ones(n, dtype="int64"),
            **{name: np.asarray(values) for name, values in columns.items()},
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.asarray([1], dtype="int64")})
    household = pd.DataFrame({"household_id": np.asarray([1], dtype="int64")})
    return Frame(
        {"person": person, "benunit": benunit, "household": household},
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(values=np.asarray([1000.0]), kind=WeightKind.DESIGN)},
    )


def _weighted_person_frame(
    columns: dict[str, np.ndarray],
    household_weights: np.ndarray,
) -> Frame:
    weights = np.asarray(household_weights, dtype=float)
    n = len(weights)
    ids = np.arange(1, n + 1, dtype="int64")
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_benunit_id": ids,
            "person_household_id": ids,
            **{name: np.asarray(values) for name, values in columns.items()},
        }
    )
    return Frame(
        {
            "person": person,
            "benunit": pd.DataFrame({"benunit_id": ids}),
            "household": pd.DataFrame({"household_id": ids}),
        },
        EntitySchema(group_entities=("benunit", "household")),
        {"household": Weights(values=weights, kind=WeightKind.DESIGN)},
    )
class _StubEngine:
    def __init__(
        self,
        defaults: dict[str, object],
        variables: set[str] | None = None,
    ) -> None:
        self._defaults = dict(defaults)
        self._variables = set(variables or defaults)

    def default_values(self, names) -> dict[str, object]:
        return {name: self._defaults[name] for name in names if name in self._defaults}

    def variables(self) -> list[str]:
        return sorted(self._variables)


def _manifest(
    columns: tuple[UKReleaseInputColumn, ...],
) -> UKReleaseInputCoverageManifest:
    return UKReleaseInputCoverageManifest(
        reference={"source": "test"},
        candidate_evidence={"source": "test"},
        columns=columns,
    )


_CONTRACT = _manifest(
    (
        UKReleaseInputColumn("employment_income", "required"),
        UKReleaseInputColumn("dividend_income", "required"),
        UKReleaseInputColumn(
            "property_income",
            "reviewed_exclusion",
            reason="not yet ported from enhanced FRS pipeline — pending review",
            tracking_note="Tracked in UK_COVERAGE_PROGRESS.md.",
        ),
    )
)
_DEFAULTS = {
    "employment_income": 0.0,
    "dividend_income": 0.0,
    "property_income": 0.0,
}


class TestUKReleaseInputCoverageGate:
    def test_full_required_set_with_signal_passes(self) -> None:
        frame = _person_frame(
            {
                "employment_income": np.asarray([0.0, 52_000.0, 12_000.0]),
                "dividend_income": np.asarray([0.0, 500.0, 0.0]),
            }
        )
        result = uk_release_input_coverage_gate(
            frame, _StubEngine(_DEFAULTS), manifest=_CONTRACT
        )
        assert result.passed
        assert result.failures == ()
        assert result.details["dormant_exclusions"] == ["property_income"]

    def test_missing_required_column_fails_and_names_efrs(self) -> None:
        frame = _person_frame({"employment_income": np.asarray([0.0, 52_000.0])})
        result = uk_release_input_coverage_gate(
            frame, _StubEngine(_DEFAULTS), manifest=_CONTRACT
        )
        assert not result.passed
        assert result.details["missing"] == ["dividend_income"]
        assert any(
            "dividend_income" in failure
            and "enhanced-FRS input column is absent" in failure
            for failure in result.failures
        )

    def test_default_only_required_column_fails(self) -> None:
        frame = _person_frame(
            {
                "employment_income": np.asarray([0.0, 52_000.0]),
                "dividend_income": np.asarray([0.0, 0.0]),
            }
        )
        result = uk_release_input_coverage_gate(
            frame, _StubEngine(_DEFAULTS), manifest=_CONTRACT
        )
        assert not result.passed
        assert result.details["degenerate_required"] == ["dividend_income"]
        assert any("every value equals" in failure for failure in result.failures)

    @pytest.mark.parametrize(
        "values",
        [
            np.asarray([np.nan, np.nan]),
            np.asarray(["", "  "], dtype=object),
            np.asarray(["", 0.0], dtype=object),
            np.asarray(["  ", 0.0], dtype=object),
            np.asarray([b"", 0.0], dtype=object),
        ],
    )
    def test_required_column_without_valid_observations_fails(self, values) -> None:
        frame = _person_frame(
            {
                "employment_income": np.asarray([0.0, 52_000.0]),
                "dividend_income": values,
            }
        )
        result = uk_release_input_coverage_gate(
            frame, _StubEngine(_DEFAULTS), manifest=_CONTRACT
        )
        assert not result.passed
        assert result.details["degenerate_required"] == ["dividend_income"]

    def test_stale_reviewed_exclusion_fails(self) -> None:
        frame = _person_frame(
            {
                "employment_income": np.asarray([0.0, 52_000.0]),
                "dividend_income": np.asarray([0.0, 500.0]),
                "property_income": np.asarray([0.0, 1_200.0]),
            }
        )
        result = uk_release_input_coverage_gate(
            frame, _StubEngine(_DEFAULTS), manifest=_CONTRACT
        )
        assert not result.passed
        assert result.details["stale_exclusions"] == ["property_income"]
        assert any(
            "Stale reviewed exclusions" in failure for failure in result.failures
        )

    def test_absent_or_default_only_exclusion_passes(self) -> None:
        frame = _person_frame(
            {
                "employment_income": np.asarray([0.0, 52_000.0]),
                "dividend_income": np.asarray([0.0, 500.0]),
                "property_income": np.asarray([0.0, 0.0]),
            }
        )
        result = uk_release_input_coverage_gate(
            frame, _StubEngine(_DEFAULTS), manifest=_CONTRACT
        )
        assert result.passed
        assert result.details["reviewed_exclusions"] == {
            "property_income": (
                "not yet ported from enhanced FRS pipeline — pending review"
            )
        }

    def test_signal_only_on_zero_weight_rows_fails_effective_mass(self) -> None:
        contract = _manifest((UKReleaseInputColumn("gift_aid", "required"),))
        frame = _weighted_person_frame(
            {"gift_aid": np.asarray([900.0, 0.0])},
            np.asarray([0.0, 1_000.0]),
        )

        result = uk_release_input_coverage_gate(
            frame,
            _StubEngine({"gift_aid": 0.0}),
            manifest=contract,
        )

        assert not result.passed
        assert result.details["insufficient_effective_mass"] == ["gift_aid"]
        diagnostic = result.details["effective_mass_by_column"]["gift_aid"]
        assert diagnostic["signal_rows"] == 1
        assert diagnostic["positive_mass_signal_rows"] == 0
        assert diagnostic["effective_signal_mass_share"] == 0.0
        assert any("zero-weight support" in failure for failure in result.failures)

    def test_positive_but_dust_mass_below_reviewed_floor_fails(self) -> None:
        contract = _manifest((UKReleaseInputColumn("gift_aid", "required"),))
        frame = _weighted_person_frame(
            {"gift_aid": np.asarray([900.0, 0.0])},
            np.asarray([0.5, 999_999.5]),
        )

        result = uk_release_input_coverage_gate(
            frame,
            _StubEngine({"gift_aid": 0.0}),
            manifest=contract,
        )

        assert not result.passed
        assert result.details["effective_mass_policy"][
            "minimum_nondefault_mass_share"
        ] == pytest.approx(1e-6)
        assert result.details["effective_mass_by_column"]["gift_aid"][
            "effective_signal_mass_share"
        ] == pytest.approx(5e-7)

    def test_signal_above_reviewed_effective_mass_floor_passes(self) -> None:
        contract = _manifest((UKReleaseInputColumn("gift_aid", "required"),))
        frame = _weighted_person_frame(
            {"gift_aid": np.asarray([900.0, 0.0])},
            np.asarray([2.0, 999_998.0]),
        )

        result = uk_release_input_coverage_gate(
            frame,
            _StubEngine({"gift_aid": 0.0}),
            manifest=contract,
        )

        assert result.passed
        assert result.details["effective_mass_by_column"]["gift_aid"][
            "effective_signal_mass_share"
        ] == pytest.approx(2e-6)

    def test_zero_mass_signal_does_not_stale_a_reviewed_exclusion(self) -> None:
        contract = _manifest(
            (
                UKReleaseInputColumn(
                    "gift_aid",
                    "reviewed_exclusion",
                    reason="not yet ported from enhanced FRS pipeline — pending review",
                    tracking_note="Tracked in UK_COVERAGE_PROGRESS.md.",
                ),
            )
        )
        frame = _weighted_person_frame(
            {"gift_aid": np.asarray([900.0, 0.0])},
            np.asarray([0.0, 1_000.0]),
        )

        result = uk_release_input_coverage_gate(
            frame,
            _StubEngine({"gift_aid": 0.0}),
            manifest=contract,
        )

        assert result.passed
        assert result.details["stale_exclusions"] == []


class TestUKManifest:
    def test_shipped_manifest_is_current(self) -> None:
        manifest = load_uk_release_input_coverage_manifest()
        assert_uk_release_input_coverage_manifest_current(
            engine=_StubEngine({}, set(manifest.declared_columns))
        )
        assert manifest.required_columns == frozenset(
            load_efrs_parity_reference().populated_layers
        )
        assert manifest.reviewed_exclusions == {}
        assert load_efrs_parity_known_gaps() == ()

    def test_manifest_refuses_empty_columns(self, tmp_path) -> None:
        bad = tmp_path / "empty.json"
        bad.write_text(
            json.dumps(
                {
                    "reference": {},
                    "candidate_evidence": {},
                    "effective_mass_coverage": {
                        "weight_source": "household_weight",
                        "minimum_nondefault_mass_share": 1e-6,
                        "reviewed_on": "2026-07-11",
                        "rationale": "reviewed test floor",
                    },
                    "columns": {},
                }
            )
        )
        with pytest.raises(ValueError, match="vacuous"):
            load_uk_release_input_coverage_manifest(str(bad))

    def test_effective_mass_policy_rejects_zero_floor(self) -> None:
        with pytest.raises(ValueError, match=r"in \(0, 1\]"):
            UKEffectiveMassCoveragePolicy(minimum_nondefault_mass_share=0.0)

    def test_reviewed_exclusion_requires_tracking_note(self) -> None:
        with pytest.raises(ValueError, match="tracking note"):
            UKReleaseInputColumn(
                "property_income",
                "reviewed_exclusion",
                reason="not yet ported from enhanced FRS pipeline — pending review",
            )

    def test_engine_graph_drift_is_rejected(self) -> None:
        manifest = load_uk_release_input_coverage_manifest()
        graph = set(manifest.declared_columns) - {"employment_income"}
        with pytest.raises(ValueError, match="employment_income"):
            assert_uk_release_input_coverage_manifest_current(
                engine=_StubEngine({}, graph)
            )

    def test_manifest_cannot_demote_a_column_absent_from_known_gaps(self) -> None:
        manifest = load_uk_release_input_coverage_manifest()
        columns = tuple(
            replace(
                column,
                status="reviewed_exclusion",
                reason="not yet ported from enhanced FRS pipeline — pending review",
                tracking_note="Tracked in UK_COVERAGE_PROGRESS.md.",
            )
            if column.name == "employment_income"
            else column
            for column in manifest.columns
        )
        demoted = replace(manifest, columns=columns)
        with pytest.raises(ValueError, match="employment_income.*remain required"):
            assert_uk_release_input_coverage_manifest_current(
                engine=_StubEngine({}, set(manifest.declared_columns)),
                manifest=demoted,
            )

    def test_loader_aliases_are_hard_covered(self) -> None:
        manifest = load_uk_release_input_coverage_manifest()
        assert set(UK_LOADER_INPUT_ALIASES) <= set(manifest.required_columns)

    def test_live_uk_adapter_recognises_loader_aliases(self) -> None:
        pytest.importorskip("policyengine_uk")
        engine = PolicyEngineUKCoverageEngine()
        assert set(UK_LOADER_INPUT_ALIASES) <= set(engine.variables())
        defaults = engine.default_values(UK_LOADER_INPUT_ALIASES)
        assert defaults == {name: 0 for name in UK_LOADER_INPUT_ALIASES}


def test_default_us_coverage_path_is_unchanged() -> None:
    from populace.build.us_runtime.release_input_coverage import (
        ReleaseInputColumn,
        ReleaseInputCoverageManifest,
        us_release_input_coverage_gate,
    )

    us_manifest = ReleaseInputCoverageManifest(
        reference={"source": "test"},
        columns=(ReleaseInputColumn("employment_income", "required"),),
    )
    frame = _person_frame({"dividend_income": np.asarray([0.0, 1.0])})
    result = us_release_input_coverage_gate(
        frame,
        _StubEngine({"employment_income": 0.0}),
        manifest=us_manifest,
    )
    assert not result.passed
    assert "required eCPS input column is absent" in result.failures[0]

    us_builder = (
        _REPO_ROOT / "tools" / "build_us_fiscal_refresh_release.py"
    ).read_text(encoding="utf-8")
    assert "us_release_input_coverage_gate" in us_builder
    assert "uk_release_input_coverage_gate" not in us_builder
