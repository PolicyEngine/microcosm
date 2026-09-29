"""Tests split from packages/microcosm-build/tests/test_uk_release_input_coverage.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_release_input_coverage import *


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


class TestFamilyBuildStateFrame:
    """The family build-state half reads the spine frame when supplied."""

    def _contract(self):
        family = _hmrc_family_coverage()
        family["hmrc_spi_income"].update(
            {
                "output_weight_kind": "importance",
                "required_mass_change_reason": "reviewed SPI allocation",
            }
        )
        return _manifest(
            (UKReleaseInputColumn("gift_aid", "required"),),
            family_coverage=family,
        )

    def _frames(self):
        receipt = MassChangeRecord(
            entity="household",
            old_total=2_000.0,
            new_total=2_000.0,
            declared_factor=1.0,
            reason="reviewed SPI allocation",
        )
        columns = {
            "gift_aid": np.asarray([0.0, 100.0]),
            "person_support_channel": np.asarray(["frs", "spi"]),
        }
        spine = _weighted_person_frame(
            columns,
            np.asarray([1_000.0, 1_000.0]),
            weight_kind=WeightKind.IMPORTANCE,
            mass_log=(receipt,),
            time_period="2024",
        )
        release = _weighted_person_frame(
            columns,
            np.asarray([900.0, 1_100.0]),
            weight_kind=WeightKind.CALIBRATED,
            mass_log=(
                receipt,
                MassChangeRecord(
                    entity="household",
                    old_total=2_000.0,
                    new_total=2_000.0,
                    declared_factor=1.0,
                    reason="national calibration",
                ),
            ),
            time_period="2024",
        )
        return spine, release

    def test_gate_reads_build_state_from_the_spine_frame(self) -> None:
        spine, release = self._frames()
        engine = _StubEngine({"gift_aid": 0.0})

        on_release = uk_release_input_coverage_gate(
            release, engine, manifest=self._contract()
        )
        assert not on_release.passed
        assert any(
            "expected reviewed kind 'importance'" in f for f in on_release.failures
        )
        assert on_release.details["family_build_state_frame"] == "release"

        result = uk_release_input_coverage_gate(
            release, engine, manifest=self._contract(), build_state_frame=spine
        )
        assert result.passed, result.failures
        assert result.details["family_build_state_frame"] == "spine"
        state = result.details["family_build_state"]["hmrc_spi_income"]
        assert state["actual_output_weight_kind"] == "importance"
        assert state["valid_mass_change_records"] == 1
        # The coverage half still measured the release frame's weights.
        assert result.details["effective_mass_by_column"]["gift_aid"][
            "effective_signal_mass_share"
        ] == pytest.approx(1_100.0 / 2_000.0)

    def test_binding_hands_the_spine_frame_to_the_build_state_half(self) -> None:
        from microcosm.build.gate_battery import EvidenceContext
        from microcosm.build.uk_runtime.battery_bindings import (
            _evaluate_release_input_coverage,
        )

        spine, release = self._frames()
        artifacts = {
            "coverage_engine": _StubEngine({"gift_aid": 0.0}),
            "coverage_manifest": self._contract(),
            "spine_frame": spine,
        }
        result = _evaluate_release_input_coverage(
            EvidenceContext(frame=release, artifacts=artifacts), {}
        )
        assert result.passed, result.failures
        assert result.details["family_build_state_frame"] == "spine"
