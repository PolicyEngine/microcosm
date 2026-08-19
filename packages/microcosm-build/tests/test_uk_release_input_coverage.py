"""UK enhanced-FRS release input-column coverage, isolated from PE-UK."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import (
    RESTORED_REFERENCE_EFRS_REQUIRED_INPUTS,
    UK_LOADER_INPUT_ALIASES,
    PolicyEngineUKCoverageEngine,
    UKEffectiveMassCoveragePolicy,
    UKReleaseInputColumn,
    UKReleaseInputCoverageManifest,
    assert_uk_release_input_coverage_build_stages,
    assert_uk_release_input_coverage_manifest_current,
    load_efrs_parity_known_gaps,
    load_efrs_parity_reference,
    load_uk_release_input_coverage_manifest,
    uk_release_input_coverage_gate,
)
from microcosm.frame import EntitySchema, Frame, MassChangeRecord, WeightKind, Weights

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
    *,
    weight_kind: WeightKind = WeightKind.DESIGN,
    mass_log: tuple[MassChangeRecord, ...] = (),
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
        {"household": Weights(values=weights, kind=weight_kind)},
        mass_log=mass_log,
    )


class _StubEngine:
    def __init__(
        self,
        defaults: dict[str, object],
        variables: set[str] | None = None,
        entities: dict[str, str] | None = None,
    ) -> None:
        self._defaults = dict(defaults)
        self._variables = set(variables or defaults)
        reference_entities = dict(load_efrs_parity_reference().input_entities)
        self._entities = {
            name: reference_entities.get(name, "person") for name in self._variables
        }
        self._entities.update(entities or {})

    def default_values(self, names) -> dict[str, object]:
        return {name: self._defaults[name] for name in names if name in self._defaults}

    def variables(self) -> list[str]:
        return sorted(self._variables)

    def variable_entities(self, names) -> dict[str, str]:
        return {name: self._entities[name] for name in names if name in self._entities}


def _manifest(
    columns: tuple[UKReleaseInputColumn, ...],
    *,
    family_coverage: dict[str, dict[str, object]] | None = None,
) -> UKReleaseInputCoverageManifest:
    return UKReleaseInputCoverageManifest(
        reference={"source": "test"},
        candidate_evidence={"source": "test"},
        columns=columns,
        family_coverage=family_coverage or {},
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


def _hmrc_family_coverage() -> dict[str, dict[str, object]]:
    return {
        "hmrc_spi_income": {
            "status": "required_at_build",
            "stage": "hmrc_spi_income",
            "effective_mass_requirements": {
                "gift_aid": {
                    "status": "distributional_required",
                    "minimum_nondefault_mass_share": 1e-6,
                    "support_channel_column": "person_support_channel",
                    "required_support_channel": "spi",
                    "mass_share_denominator": "all_person_effective_mass",
                }
            },
        }
    }


def _reviewed_gift_aid_exclusion() -> UKReleaseInputColumn:
    return UKReleaseInputColumn(
        "gift_aid",
        "reviewed_exclusion",
        reason="not yet ported from enhanced FRS pipeline — pending review",
        tracking_note="Tracked in UK_COVERAGE_PROGRESS.md.",
    )


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

    def test_formula_owned_persisted_override_is_a_hard_requirement(self) -> None:
        contract = _manifest(
            (UKReleaseInputColumn("state_pension_reported", "required"),)
        )
        engine = _StubEngine({"state_pension_reported": 0.0})

        absent = uk_release_input_coverage_gate(
            _person_frame({"age": np.asarray([40, 70])}),
            engine,
            manifest=contract,
        )
        default_only = uk_release_input_coverage_gate(
            _person_frame({"state_pension_reported": np.asarray([0.0, 0.0])}),
            engine,
            manifest=contract,
        )
        populated = uk_release_input_coverage_gate(
            _person_frame({"state_pension_reported": np.asarray([0.0, 12_000.0])}),
            engine,
            manifest=contract,
        )

        assert not absent.passed
        assert not default_only.passed
        assert populated.passed

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

    def test_required_column_on_wrong_entity_fails(self) -> None:
        frame = _person_frame({"dividend_income": np.asarray([500.0])})
        frame.table("household")["employment_income"] = np.asarray([52_000.0])

        result = uk_release_input_coverage_gate(
            frame,
            _StubEngine(_DEFAULTS),
            manifest=_CONTRACT,
        )

        assert not result.passed
        assert result.details["wrong_entity_columns"] == {
            "employment_income": {"actual": "household", "expected": "person"}
        }
        assert any(
            "same-named column on the wrong table" in failure
            for failure in result.failures
        )

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

    def test_explicit_person_weights_cannot_override_household_mass(self) -> None:
        ids = np.asarray([1, 2], dtype="int64")
        frame = Frame(
            {
                "person": pd.DataFrame(
                    {
                        "person_id": ids,
                        "person_benunit_id": ids,
                        "person_household_id": ids,
                        "gift_aid": [100.0, 0.0],
                    }
                ),
                "benunit": pd.DataFrame({"benunit_id": ids}),
                "household": pd.DataFrame({"household_id": ids}),
            },
            EntitySchema(group_entities=("benunit", "household")),
            {
                "person": Weights(
                    np.asarray([1_000.0, 0.0]),
                    WeightKind.IMPORTANCE,
                ),
                "household": Weights(
                    np.asarray([0.0, 1_000.0]),
                    WeightKind.CALIBRATED,
                ),
            },
        )

        result = uk_release_input_coverage_gate(
            frame,
            _StubEngine({"gift_aid": 0.0}),
            manifest=_manifest((UKReleaseInputColumn("gift_aid", "required"),)),
        )

        assert not result.passed
        assert (
            result.details["effective_mass_by_column"]["gift_aid"][
                "effective_signal_mass_share"
            ]
            == 0.0
        )

    def test_base_signal_cannot_satisfy_spi_distributional_family(self) -> None:
        contract = _manifest(
            (UKReleaseInputColumn("gift_aid", "required"),),
            family_coverage=_hmrc_family_coverage(),
        )
        frame = _weighted_person_frame(
            {
                "gift_aid": np.asarray([100.0, 0.0]),
                "person_support_channel": np.asarray(["frs", "spi"]),
            },
            np.asarray([1_000.0, 1_000.0]),
        )

        result = uk_release_input_coverage_gate(
            frame,
            _StubEngine({"gift_aid": 0.0}),
            manifest=contract,
        )

        assert not result.passed
        assert any("base-channel signal does not restore" in f for f in result.failures)

    def test_positive_spi_signal_satisfies_distributional_family(self) -> None:
        contract = _manifest(
            (UKReleaseInputColumn("gift_aid", "required"),),
            family_coverage=_hmrc_family_coverage(),
        )
        frame = _weighted_person_frame(
            {
                "gift_aid": np.asarray([0.0, 100.0]),
                "person_support_channel": np.asarray(["frs", "spi"]),
            },
            np.asarray([1_000.0, 1_000.0]),
        )

        result = uk_release_input_coverage_gate(
            frame,
            _StubEngine({"gift_aid": 0.0}),
            manifest=contract,
        )

        assert result.passed
        assert result.details["family_effective_mass"]["hmrc_spi_income"]["gift_aid"][
            "effective_signal_mass_share"
        ] == pytest.approx(0.5)

    def test_family_requires_reviewed_weight_kind_and_mass_record(self) -> None:
        family = _hmrc_family_coverage()
        family["hmrc_spi_income"].update(
            {
                "output_weight_kind": "calibrated",
                "required_mass_change_reason": "reviewed SPI allocation",
            }
        )
        contract = _manifest(
            (UKReleaseInputColumn("gift_aid", "required"),),
            family_coverage=family,
        )
        frame = _weighted_person_frame(
            {
                "gift_aid": np.asarray([0.0, 100.0]),
                "person_support_channel": np.asarray(["frs", "spi"]),
            },
            np.asarray([1_000.0, 1_000.0]),
        )

        result = uk_release_input_coverage_gate(
            frame,
            _StubEngine({"gift_aid": 0.0}),
            manifest=contract,
        )

        assert not result.passed
        assert any("expected reviewed kind" in failure for failure in result.failures)
        assert any("MassChangeRecord" in failure for failure in result.failures)

    def test_family_accepts_reviewed_calibrated_mass_state(self) -> None:
        family = _hmrc_family_coverage()
        family["hmrc_spi_income"].update(
            {
                "output_weight_kind": "calibrated",
                "required_mass_change_reason": "reviewed SPI allocation",
            }
        )
        contract = _manifest(
            (UKReleaseInputColumn("gift_aid", "required"),),
            family_coverage=family,
        )
        frame = _weighted_person_frame(
            {
                "gift_aid": np.asarray([0.0, 100.0]),
                "person_support_channel": np.asarray(["frs", "spi"]),
            },
            np.asarray([1_000.0, 1_000.0]),
            weight_kind=WeightKind.CALIBRATED,
            mass_log=(
                MassChangeRecord(
                    entity="household",
                    old_total=2_000.0,
                    new_total=2_000.0,
                    declared_factor=1.0,
                    reason="reviewed SPI allocation",
                ),
            ),
        )

        result = uk_release_input_coverage_gate(
            frame,
            _StubEngine({"gift_aid": 0.0}),
            manifest=contract,
        )

        assert result.passed
        assert (
            result.details["family_build_state"]["hmrc_spi_income"][
                "valid_mass_change_records"
            ]
            == 1
        )

    def test_integer_encoded_enum_default_is_not_signal(self) -> None:
        pytest.importorskip("policyengine_uk")
        contract = _manifest((UKReleaseInputColumn("gender", "required"),))
        frame = _weighted_person_frame(
            {"gender": np.asarray([0, 0], dtype=np.int16)},
            np.asarray([1.0, 1.0]),
        )

        result = uk_release_input_coverage_gate(
            frame,
            PolicyEngineUKCoverageEngine(),
            manifest=contract,
        )

        assert not result.passed
        assert result.details["degenerate_required"] == ["gender"]

    def test_zero_mass_signal_does_not_stale_a_reviewed_exclusion(self) -> None:
        contract = _manifest((_reviewed_gift_aid_exclusion(),))
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

    def test_deferred_family_does_not_enforce_future_distributional_gate(self) -> None:
        family = _hmrc_family_coverage()
        family["hmrc_spi_income"].update(
            {
                "status": "deferred_until_restored",
                "restoration_status": "blocked_pending_reviewed_frs_decomposition",
            }
        )
        contract = _manifest(
            (_reviewed_gift_aid_exclusion(),),
            family_coverage=family,
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
        assert result.details["family_effective_mass"] == {}
        assert result.details["family_build_state"] == {}


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
        assert manifest.required_build_stages == frozenset(
            {
                "hmrc_spi_income",
                "hmrc_cgt_gains",
                "was_wealth",
                "regional_property_uprating",
            }
        )
        assert RESTORED_REFERENCE_EFRS_REQUIRED_INPUTS == frozenset(
            {"charitable_investment_gifts", "gift_aid"}
        )

    def test_manifest_refuses_empty_columns(self, tmp_path) -> None:
        bad = tmp_path / "empty.json"
        bad.write_text(
            json.dumps(
                {
                    "reference": {},
                    "candidate_evidence": {"tier": "frs"},
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

    def test_distributional_family_requires_channel_denominator(self, tmp_path) -> None:
        source = (
            _REPO_ROOT
            / "packages"
            / "microcosm-build"
            / "src"
            / "microcosm"
            / "build"
            / "uk"
            / "release_input_coverage_manifest.json"
        )
        payload = json.loads(source.read_text(encoding="utf-8"))
        requirement = payload["family_coverage"]["hmrc_spi_income"][
            "effective_mass_requirements"
        ]["gift_aid"]
        requirement.pop("mass_share_denominator", None)
        bad = tmp_path / "missing_family_denominator.json"
        bad.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="mass_share_denominator"):
            load_uk_release_input_coverage_manifest(str(bad))

    @pytest.mark.parametrize("tier", [None, "public"])
    def test_candidate_tier_must_be_present_and_ratified(
        self,
        tmp_path: Path,
        tier: str | None,
    ) -> None:
        source = (
            _REPO_ROOT
            / "packages"
            / "microcosm-build"
            / "src"
            / "microcosm"
            / "build"
            / "uk"
            / "release_input_coverage_manifest.json"
        )
        payload = json.loads(source.read_text(encoding="utf-8"))
        if tier is None:
            payload["candidate_evidence"].pop("tier", None)
        else:
            payload["candidate_evidence"]["tier"] = tier
        bad = tmp_path / "bad_candidate_tier.json"
        bad.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="candidate_evidence.tier"):
            load_uk_release_input_coverage_manifest(str(bad))

    def test_deferred_family_requires_restoration_status(self, tmp_path) -> None:
        source = (
            _REPO_ROOT
            / "packages"
            / "microcosm-build"
            / "src"
            / "microcosm"
            / "build"
            / "uk"
            / "release_input_coverage_manifest.json"
        )
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["family_coverage"]["hmrc_spi_income"]["status"] = (
            "deferred_until_restored"
        )
        payload["family_coverage"]["hmrc_spi_income"].pop("restoration_status", None)
        bad = tmp_path / "deferred_without_blocker.json"
        bad.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ValueError, match="needs a restoration_status"):
            load_uk_release_input_coverage_manifest(str(bad))

    def test_required_family_stage_cannot_be_omitted(self) -> None:
        manifest = _manifest(
            (UKReleaseInputColumn("gift_aid", "required"),),
            family_coverage=_hmrc_family_coverage(),
        )

        with pytest.raises(ValueError, match="hmrc_spi_income"):
            assert_uk_release_input_coverage_build_stages((), manifest=manifest)

        result = assert_uk_release_input_coverage_build_stages(
            ("hmrc_spi_income",),
            manifest=manifest,
        )
        assert result is None

    def test_deferred_family_stage_is_not_required(self) -> None:
        family = _hmrc_family_coverage()
        family["hmrc_spi_income"].update(
            {
                "status": "deferred_until_restored",
                "restoration_status": "blocked_pending_reviewed_frs_decomposition",
            }
        )
        manifest = _manifest(
            (_reviewed_gift_aid_exclusion(),),
            family_coverage=family,
        )

        assert_uk_release_input_coverage_build_stages((), manifest=manifest)
        assert manifest.required_build_stages == frozenset()

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

    def test_engine_entity_drift_is_rejected(self) -> None:
        manifest = load_uk_release_input_coverage_manifest()
        graph = set(manifest.declared_columns)
        with pytest.raises(ValueError, match="owning entities disagree"):
            assert_uk_release_input_coverage_manifest_current(
                engine=_StubEngine(
                    {},
                    graph,
                    entities={"employment_income": "household"},
                )
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

    def test_manifest_pins_hmrc_source_contract_hash(self) -> None:
        manifest = load_uk_release_input_coverage_manifest()
        families = {
            name: dict(family) for name, family in manifest.family_coverage.items()
        }
        families["hmrc_spi_income"]["source_manifest_sha256"] = "0" * 64
        drifted = replace(manifest, family_coverage=families)

        with pytest.raises(ValueError, match="changed without regenerating"):
            assert_uk_release_input_coverage_manifest_current(
                engine=_StubEngine({}, set(manifest.declared_columns)),
                manifest=drifted,
            )

    def test_manifest_candidate_tier_must_match_hmrc_source_lineage(self) -> None:
        manifest = load_uk_release_input_coverage_manifest()
        families = {
            name: dict(family) for name, family in manifest.family_coverage.items()
        }
        families["hmrc_spi_income"]["base_candidate_tier"] = "cps-transfer"
        drifted = replace(manifest, family_coverage=families)

        with pytest.raises(ValueError, match="base_candidate_tier.*disagrees"):
            assert_uk_release_input_coverage_manifest_current(
                engine=_StubEngine({}, set(manifest.declared_columns)),
                manifest=drifted,
            )

    def test_deferred_family_columns_must_remain_reviewed_exclusions(self) -> None:
        manifest = load_uk_release_input_coverage_manifest()
        families = {
            name: dict(family) for name, family in manifest.family_coverage.items()
        }
        families["hmrc_spi_income"]["status"] = "deferred_until_restored"

        with pytest.raises(ValueError, match="deferred distributional requirement"):
            assert_uk_release_input_coverage_manifest_current(
                engine=_StubEngine({}, set(manifest.declared_columns)),
                manifest=replace(manifest, family_coverage=families),
            )

    def test_promoted_family_columns_must_be_required(self) -> None:
        manifest = load_uk_release_input_coverage_manifest()
        columns = tuple(
            replace(
                column,
                status="reviewed_exclusion",
                reason="not yet ported from enhanced FRS pipeline — pending review",
                tracking_note="Tracked in UK_COVERAGE_PROGRESS.md.",
            )
            if column.name == "gift_aid"
            else column
            for column in manifest.columns
        )

        with pytest.raises(ValueError, match="required_at_build"):
            assert_uk_release_input_coverage_manifest_current(
                engine=_StubEngine({}, set(manifest.declared_columns)),
                manifest=replace(manifest, columns=columns),
            )

    def test_loader_aliases_are_hard_covered(self) -> None:
        manifest = load_uk_release_input_coverage_manifest()
        assert set(UK_LOADER_INPUT_ALIASES) <= set(manifest.required_columns)

    def test_formula_owned_persisted_overrides_are_hard_covered(self) -> None:
        manifest = load_uk_release_input_coverage_manifest()
        reference = json.loads(
            (
                _REPO_ROOT
                / "packages"
                / "microcosm-build"
                / "src"
                / "microcosm"
                / "build"
                / "uk"
                / "efrs_parity_reference.json"
            ).read_text(encoding="utf-8")
        )
        overrides = set(
            reference["engine"]["formula_owned_persisted_overrides_included"]
        )
        assert len(overrides) == 13
        assert overrides <= set(manifest.required_columns)

    def test_live_uk_adapter_recognises_loader_aliases(self) -> None:
        pytest.importorskip("policyengine_uk")
        engine = PolicyEngineUKCoverageEngine()
        assert set(UK_LOADER_INPUT_ALIASES) <= set(engine.variables())
        defaults = engine.default_values(UK_LOADER_INPUT_ALIASES)
        assert defaults == {name: 0 for name in UK_LOADER_INPUT_ALIASES}
        assert engine.variable_entities(UK_LOADER_INPUT_ALIASES) == {
            name: "person" for name in UK_LOADER_INPUT_ALIASES
        }

    def test_live_uk_adapter_recognises_formula_owned_overrides(self) -> None:
        pytest.importorskip("policyengine_uk")
        engine = PolicyEngineUKCoverageEngine()
        names = ("state_pension_reported", "student_loan_repayments")
        assert set(names) <= set(engine.variables())
        assert engine.variable_entities(names) == {
            "state_pension_reported": "person",
            "student_loan_repayments": "person",
        }
        assert engine.default_values(names) == {
            "state_pension_reported": 0,
            "student_loan_repayments": 0,
        }


def test_default_us_coverage_path_is_unchanged() -> None:
    from microcosm.build.us_runtime.release_input_coverage import (
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
