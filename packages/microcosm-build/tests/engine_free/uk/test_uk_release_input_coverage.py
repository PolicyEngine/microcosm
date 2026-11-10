"""Tests split from packages/microcosm-build/tests/test_uk_release_input_coverage.py."""

# ruff: noqa: F403, F405
from importlib.resources import files

from microcosm.build.uk_runtime.release_input_coverage import (
    UK_FAMILY_MASS_CHANGE_SEMANTICS,
    _parse_family_coverage,
    _valid_mass_record,
    uk_effective_input_signal,
)
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
            release,
            engine,
            reference_surface=NO_REFERENCE_INPUTS,
            manifest=self._contract(),
        )
        assert not on_release.passed
        assert any(
            "expected reviewed kind 'importance'" in f for f in on_release.failures
        )
        assert on_release.details["family_build_state_frame"] == "release"

        result = uk_release_input_coverage_gate(
            release,
            engine,
            reference_surface=NO_REFERENCE_INPUTS,
            manifest=self._contract(),
            build_state_frame=spine,
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
            "reference_surface": NO_REFERENCE_INPUTS,
            "spine_frame": spine,
        }
        result = _evaluate_release_input_coverage(
            EvidenceContext(frame=release, artifacts=artifacts), {}
        )
        assert result.passed, result.failures
        assert result.details["family_build_state_frame"] == "spine"

    def test_binding_refuses_to_judge_the_calibrated_frame_without_build_state(
        self,
    ) -> None:
        """microcosm#1115 review: no spine build state means evidence absent,
        never a verdict read off the calibrated frame's weights."""
        from microcosm.build.gate_battery import EvidenceContext
        from microcosm.build.uk_runtime.battery_bindings import (
            UK_GATE_REGISTRY,
            _evaluate_release_input_coverage,
        )

        _, release = self._frames()
        artifacts = {
            "coverage_engine": _StubEngine({"gift_aid": 0.0}),
            "coverage_manifest": self._contract(),
            "reference_surface": NO_REFERENCE_INPUTS,
        }
        with pytest.raises(ValueError, match="spine build state"):
            _evaluate_release_input_coverage(
                EvidenceContext(frame=release, artifacts=artifacts), {}
            )
        binding = UK_GATE_REGISTRY["release_input_coverage"]
        assert "spine_frame" in binding.required_artifacts({})
        assert "spine_frame" not in binding.required_artifacts(
            {"check": "manifest_current"}
        )


class TestFamilyMassSemantics:
    """Every family conserves household mass (microcosm#1063 item 3)."""

    def _payload(self) -> dict[str, dict[str, object]]:
        resource = files("microcosm.build.uk").joinpath(
            "release_input_coverage_manifest.json"
        )
        return json.loads(resource.read_text(encoding="utf-8"))["family_coverage"]

    def test_parser_refuses_the_retired_mass_increasing_semantics(self) -> None:
        payload = self._payload()
        families = _parse_family_coverage(payload, resource="test")
        assert {family["mass_change_semantics"] for family in families.values()} == {
            UK_FAMILY_MASS_CHANGE_SEMANTICS
        }
        assert "spi_income_band_donors" in families
        payload["spi_income_band_donors"] = {
            **payload["spi_income_band_donors"],
            "mass_change_semantics": "mass_increasing_support",
        }
        with pytest.raises(ValueError, match="the only semantics is 'mass_conserving'"):
            _parse_family_coverage(payload, resource="test")

    def test_a_record_that_adds_mass_never_satisfies_a_family(self) -> None:
        conserving = MassChangeRecord(
            entity="household",
            old_total=2_000.0,
            new_total=2_000.0,
            declared_factor=1.0,
            reason="reviewed",
        )
        assert _valid_mass_record(conserving)
        # The pre-#1063 band-donor record: mass added, no declared factor.
        assert not _valid_mass_record(
            MassChangeRecord(
                entity="household",
                old_total=2_000.0,
                new_total=2_450.0,
                declared_factor=None,
                reason="reviewed",
            )
        )
        assert not _valid_mass_record({**conserving.__dict__, "entity": "person"})


class TestReferenceRequiredInputs:
    """microcosm#1095: the required inputs are the ones the reference carries."""

    _ENGINE_DEFAULTS = {
        "employment_income": 0.0,
        "dividend_income": 0.0,
        "property_income": 0.0,
    }

    def _frame(self, **columns):
        return _person_frame(
            {
                "employment_income": np.asarray([0.0, 52_000.0]),
                **{name: np.asarray(values) for name, values in columns.items()},
            }
        )

    def test_the_reference_supplies_the_required_inputs(self) -> None:
        result = uk_release_input_coverage_gate(
            self._frame(),
            _StubEngine(self._ENGINE_DEFAULTS),
            reference_surface=reference_surface("employment_income", "dividend_income"),
            manifest=_manifest(()),
        )
        assert not result.passed
        assert result.details["missing"] == ["dividend_income"]
        assert result.details["reference"] == reference_surface().label
        assert result.details["required_from_reference"] == 2
        assert result.details["required_by_manifest"] == []

    def test_a_reviewed_exclusion_beats_the_reference(self) -> None:
        contract = _manifest(
            (
                UKReleaseInputColumn(
                    "property_income",
                    "reviewed_exclusion",
                    reason="formula-owned in policyengine-uk",
                    tracking_note="microcosm#1106",
                ),
            )
        )
        result = uk_release_input_coverage_gate(
            self._frame(),
            _StubEngine(self._ENGINE_DEFAULTS),
            reference_surface=reference_surface("employment_income", "property_income"),
            manifest=contract,
        )
        assert result.passed, result.failures
        assert result.details["dormant_exclusions"] == ["property_income"]
        assert result.details["required_from_reference"] == 1

    def test_an_input_the_live_engine_does_not_load_is_not_required(self) -> None:
        result = uk_release_input_coverage_gate(
            self._frame(),
            _StubEngine({"employment_income": 0.0}),
            reference_surface=reference_surface("employment_income", "retired_input"),
            manifest=_manifest(()),
        )
        assert result.passed, result.failures

    def test_a_thin_reference_input_is_required_and_named_by_name_only(self) -> None:
        result = uk_release_input_coverage_gate(
            self._frame(dividend_income=[0.0, 500.0]),
            _StubEngine(self._ENGINE_DEFAULTS),
            reference_surface=reference_surface(
                "employment_income", thin=("dividend_income",)
            ),
            manifest=_manifest(()),
        )
        assert result.passed, result.failures
        assert result.details["required_thin_in_reference"] == ["dividend_income"]

    def test_a_reference_measured_at_another_floor_is_refused(self) -> None:
        with pytest.raises(ValueError, match="effective-mass floor"):
            uk_release_input_coverage_gate(
                self._frame(),
                _StubEngine(self._ENGINE_DEFAULTS),
                reference_surface=reference_surface("employment_income", floor=1e-3),
                manifest=_manifest(()),
            )

    def test_a_vacuous_required_set_is_refused(self) -> None:
        with pytest.raises(ValueError, match="vacuous"):
            uk_release_input_coverage_gate(
                self._frame(),
                _StubEngine(self._ENGINE_DEFAULTS),
                reference_surface=reference_surface(),
                manifest=_manifest(()),
            )


def test_effective_input_signal_measures_by_the_gates_own_rule() -> None:
    frame = _weighted_person_frame(
        {
            # Signal on positive mass: carried.
            "employment_income": np.asarray([0.0, 52_000.0, 0.0]),
            # Signal only on a zero-weight row: not carried.
            "dividend_income": np.asarray([500.0, 0.0, 0.0]),
            # Signal only on dust below the reviewed floor: not carried.
            "savings_interest_income": np.asarray([0.0, 0.0, 10.0]),
            # Every value the engine default: not carried.
            "property_income": np.asarray([0.0, 0.0, 0.0]),
            # Not a loader input: never measured.
            "not_an_input": np.asarray([1.0, 1.0, 1.0]),
        },
        np.asarray([0.0, 1_000.0, 1e-9]),
    )
    engine = _StubEngine(
        {
            "employment_income": 0.0,
            "dividend_income": 0.0,
            "savings_interest_income": 0.0,
            "property_income": 0.0,
        }
    )
    carried = uk_effective_input_signal(
        frame, engine, minimum_nondefault_mass_share=1e-6
    )
    assert set(carried) == {"employment_income"}
    entity, signal = carried["employment_income"]
    assert entity == "person"
    assert signal.tolist() == [False, True, False]


def test_figures_resting_on_fewer_than_three_source_households_are_withheld() -> None:
    """The UKDS floor of three (CD171): a required column one or two source
    households carry keeps its verdict, and its figures leave the report."""

    def gate(source_ids):
        frame = _weighted_person_frame(
            {
                # Signal on two rows: one source household, or two.
                "dividend_income": np.asarray([500.0, 500.0, 0.0, 0.0]),
                "employment_income": np.asarray([1.0, 2.0, 3.0, 4.0]),
            },
            np.asarray([1_000.0, 1_000.0, 1_000.0, 1_000.0]),
            source_ids=source_ids,
        )
        return uk_release_input_coverage_gate(
            frame,
            _StubEngine({"dividend_income": 0.0, "employment_income": 0.0}),
            reference_surface=reference_surface("dividend_income", "employment_income"),
            manifest=_manifest(()),
        )

    for source_ids in ([11, 11, 12, 13], [11, 12, 13, 14]):
        result = gate(source_ids)
        assert result.passed, result.failures
        by_column = result.details["effective_mass_by_column"]
        assert by_column["dividend_income"] == {
            "entity": "person",
            "source_households": "<3",
            "figures": "withheld",
        }
        assert by_column["employment_income"]["effective_signal_mass_share"] == 1.0

    # Every row a clone of one source household: even a column on every row
    # rests on one case, so its figures are withheld too.
    result = gate([11, 11, 11, 11])
    assert (
        "effective_signal_mass_share"
        not in (result.details["effective_mass_by_column"]["employment_income"])
    )
    # Three source households carry it: the figures are reported.
    spread = _weighted_person_frame(
        {"dividend_income": np.asarray([500.0, 500.0, 500.0, 0.0])},
        np.asarray([1_000.0] * 4),
        source_ids=[11, 12, 13, 14],
    )
    reported = uk_release_input_coverage_gate(
        spread,
        _StubEngine({"dividend_income": 0.0}),
        reference_surface=reference_surface("dividend_income"),
        manifest=_manifest(()),
    )
    assert reported.details["effective_mass_by_column"]["dividend_income"][
        "effective_signal_mass_share"
    ] == pytest.approx(0.75)
