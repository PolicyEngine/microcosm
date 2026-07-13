"""Take-up contract inventory tests (populace #312).

The inventory is only trustworthy if it is asserted against the installed
engine, so the load-bearing tests here (1) confirm the checked-in table matches
the pinned policyengine-us and (2) prove the assertion can actually fail when
the table drifts -- a guard that never fails is worthless (a
prove-it-can-find-something check).
"""

from __future__ import annotations

import copy
import json

import pytest

from populace.build.us_runtime.take_up_contract import (
    TAKE_UP_CONTRACT_ENGINE_FACT_KEYS,
    assert_take_up_contract_current,
    assert_take_up_treatments_consistent,
    count_calibrated_take_up_programs,
    load_take_up_contract,
    seeded_take_up_programs,
)
from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine

pytest.importorskip("policyengine_us")


class TestContractLoads:
    def test_loads_every_engine_take_up_variable(self) -> None:
        contract = load_take_up_contract()
        engine_names = set(PolicyEngineUSEngine().take_up_variables())
        table_names = {program.variable for program in contract.programs}
        assert table_names == engine_names

    def test_version_and_constraint_present(self) -> None:
        contract = load_take_up_contract()
        assert contract.version >= 1
        assert contract.country == "us"
        assert contract.asserted_constraint.startswith(">=")

    def test_every_program_has_a_valid_treatment(self) -> None:
        for program in load_take_up_contract().programs:
            assert program.populace_treatment in {
                "seed",
                "count_calibrated",
                "rate_unsourced",
                "model_simulated",
                "out_of_scope",
                "near_universal",
            }

    def test_ssi_uses_reporter_anchored_ssa_age_count_calibration(self) -> None:
        program = load_take_up_contract().program_map()["takes_up_ssi_if_eligible"]
        calibration = program.raw["calibration"]

        assert program in count_calibrated_take_up_programs()
        assert calibration == {
            "anchor": "SSI_VAL",
            "targets": ["ssa_ssi_federal_payment_recipients_by_age"],
            "target_table": "ssa_ssi_federal_payment_recipients_by_age",
            "target_source": (
                "https://www.ssa.gov/policy/docs/statcomps/"
                "ssi_monthly/2024-12/table01.html"
            ),
            "target_period": "2024-12",
            "target_measure": "Total with—Federal payment",
            "target_values": {
                "under_18": 1_001_922,
                "18_64": 3_905_779,
                "65_plus": 2_382_142,
            },
            "aggregate_target": 7_289_843,
            "age_bands": {
                "under_18": "age < 18",
                "18_64": "18 <= age < 65",
                "65_plus": "age >= 65",
            },
            "semantics": (
                "SSA SSI Monthly Statistics December 2024 Table 1 recipients "
                "in the Total with—Federal payment row, calibrated within "
                "uncapped_ssi > 0 by source-person identity; unreachable age "
                "bands saturate without assigning outside modeled eligibility"
            ),
        }
        assert program.raw["scope_owner"] == (
            "ssi_take_up source stage (eCPS exported-input coverage)"
        )

    def test_head_start_is_owned_by_measured_sipp_stage(self) -> None:
        program = load_take_up_contract().program_map()[
            "takes_up_head_start_if_eligible"
        ]

        assert program.populace_treatment == "out_of_scope"
        assert program.raw["scope_owner"] == (
            "sipp_head_start source stage (measured SIPP enrollment response)"
        )
        assert program.rate == {"status": "not_used_measured_source"}
        notes = program.raw["notes"]
        assert "EEDHEADST" in notes
        assert "direct December age-3--5" in notes
        assert "QRF" in notes
        assert "retired NIEER scalar" in notes

    def test_early_head_start_records_irreducible_source_unavailability(self) -> None:
        program = load_take_up_contract().program_map()[
            "takes_up_early_head_start_if_eligible"
        ]

        assert program.populace_treatment == "rate_unsourced"
        assert program.rate["status"] == "source_unavailable"
        followup = program.raw["followup"].lower()
        assert "locked individual-level source" in followup
        assert "infants" in followup
        assert "toddlers" in followup
        assert "pregnant" in followup
        notes = program.raw["notes"].lower()
        assert "ecps_parity_known_gaps.json" in notes
        assert "aggregate" in notes
        assert "synthesize" in notes


class TestEngineAssertion:
    def test_checked_in_table_matches_installed_engine(self) -> None:
        # The headline guard: the curated table's engine facts equal what the
        # pinned policyengine-us reports right now.
        assert_take_up_contract_current()

    def test_treatments_consistent_with_engine_classes(self) -> None:
        assert_take_up_treatments_consistent()

    def test_engine_facts_match_field_by_field(self) -> None:
        engine_contract = PolicyEngineUSEngine().take_up_contract()
        for program in load_take_up_contract().programs:
            facts = engine_contract[program.variable]
            for key in TAKE_UP_CONTRACT_ENGINE_FACT_KEYS:
                assert program.engine_facts()[key] == facts[key], (
                    f"{program.variable}.{key}"
                )


class TestAssertionCanFail:
    """Prove the guard can find drift; a guard that cannot fail is vacuous."""

    def _reload_with(self, monkeypatch, mutated: dict) -> None:
        load_take_up_contract.cache_clear()
        payload = json.dumps(mutated)

        class _FakePath:
            def read_text(self, *args, **kwargs):
                return payload

        monkeypatch.setattr(
            "populace.build.us_runtime.take_up_contract._contract_path",
            lambda: _FakePath(),
        )

    @pytest.fixture
    def base_table(self):
        raw = json.loads(
            __import__("importlib.resources", fromlist=["files"])
            .files("populace.build.us")
            .joinpath("take_up_contract.json")
            .read_text()
        )
        yield raw
        load_take_up_contract.cache_clear()

    def test_dropping_a_program_fails(self, monkeypatch, base_table) -> None:
        mutated = copy.deepcopy(base_table)
        mutated["programs"] = [
            p
            for p in mutated["programs"]
            if p["variable"] != "takes_up_snap_if_eligible"
        ]
        self._reload_with(monkeypatch, mutated)
        with pytest.raises(AssertionError, match="takes_up_snap_if_eligible"):
            assert_take_up_contract_current()

    def test_wrong_engine_class_fails(self, monkeypatch, base_table) -> None:
        mutated = copy.deepcopy(base_table)
        for program in mutated["programs"]:
            if program["variable"] == "takes_up_tanf_if_eligible":
                program["engine_class"] = "model_simulated"
        self._reload_with(monkeypatch, mutated)
        with pytest.raises(AssertionError, match="engine_class"):
            assert_take_up_contract_current()

    def test_wrong_default_fails(self, monkeypatch, base_table) -> None:
        mutated = copy.deepcopy(base_table)
        for program in mutated["programs"]:
            if program["variable"] == "takes_up_eitc":
                program["default"] = False
        self._reload_with(monkeypatch, mutated)
        with pytest.raises(AssertionError, match="default"):
            assert_take_up_contract_current()

    def test_phantom_program_fails(self, monkeypatch, base_table) -> None:
        mutated = copy.deepcopy(base_table)
        mutated["programs"].append(
            {
                "variable": "takes_up_nonexistent_if_eligible",
                "engine_class": "data_seeded",
                "entity": "person",
                "value_type": "bool",
                "default": True,
                "populace_treatment": "rate_unsourced",
            }
        )
        self._reload_with(monkeypatch, mutated)
        with pytest.raises(AssertionError, match="takes_up_nonexistent_if_eligible"):
            assert_take_up_contract_current()

    def test_seeding_a_model_simulated_flag_is_rejected_at_load(
        self, monkeypatch, base_table
    ) -> None:
        # A model_simulated flag marked seed must be caught (consistency guard).
        mutated = copy.deepcopy(base_table)
        for program in mutated["programs"]:
            if program["variable"] == "takes_up_tanf_if_eligible":
                program["engine_class"] = "model_simulated"
                program["populace_treatment"] = "seed"
        self._reload_with(monkeypatch, mutated)
        with pytest.raises(AssertionError, match="model_simulated"):
            assert_take_up_treatments_consistent()


class TestSeedProvenanceIsEnforced:
    """A seeded program must carry sourced provenance; the loader enforces it."""

    def _load_mutated(self, monkeypatch, mutated: dict):
        load_take_up_contract.cache_clear()
        payload = json.dumps(mutated)

        class _FakePath:
            def read_text(self, *args, **kwargs):
                return payload

        monkeypatch.setattr(
            "populace.build.us_runtime.take_up_contract._contract_path",
            lambda: _FakePath(),
        )

    @pytest.fixture
    def base_table(self):
        raw = json.loads(
            __import__("importlib.resources", fromlist=["files"])
            .files("populace.build.us")
            .joinpath("take_up_contract.json")
            .read_text()
        )
        yield raw
        load_take_up_contract.cache_clear()

    def test_seed_without_rate_source_is_refused(self, monkeypatch, base_table) -> None:
        mutated = copy.deepcopy(base_table)
        for program in mutated["programs"]:
            if program["variable"] == "takes_up_tanf_if_eligible":
                program["rate"] = {"value": 0.219}  # no source
        self._load_mutated(monkeypatch, mutated)
        with pytest.raises(ValueError, match="source"):
            load_take_up_contract()

    def test_seed_with_unsourced_status_is_refused(
        self, monkeypatch, base_table
    ) -> None:
        mutated = copy.deepcopy(base_table)
        for program in mutated["programs"]:
            if program["variable"] == "takes_up_tanf_if_eligible":
                program["rate"] = {
                    "value": 0.219,
                    "source": "https://example.com",
                    "status": "model_relative",
                }
        self._load_mutated(monkeypatch, mutated)
        with pytest.raises(ValueError, match="administrative-grade"):
            load_take_up_contract()

    def test_seeded_programs_all_have_administrative_provenance(self) -> None:
        # The real table: every seeded program carries a sourced rate.
        for program in seeded_take_up_programs():
            rate = program.rate
            assert rate.get("source"), program.variable
            assert str(rate.get("status", "")).startswith("sourced"), program.variable
