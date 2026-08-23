"""Take-up contract inventory tests (microcosm #312).

The inventory is only trustworthy if it is asserted against the installed
engine, so the load-bearing tests here (1) confirm the checked-in table matches
the pinned policyengine-us and (2) prove the assertion can actually fail when
the table drifts -- a guard that never fails is worthless (a
prove-it-can-find-something check).
"""

from __future__ import annotations

import copy
import hashlib
import json
from importlib.resources import files

import pytest

from microcosm.build.us_runtime.take_up_contract import (
    TAKE_UP_CONTRACT_ENGINE_FACT_KEYS,
    assert_take_up_contract_current,
    assert_take_up_treatments_consistent,
    count_calibrated_take_up_programs,
    load_legacy_take_up_contract_evidence,
    load_take_up_contract,
    seeded_take_up_programs,
    take_up_contract_identity,
)
from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

pytest.importorskip("policyengine_us")


class TestContractLoads:
    def test_runtime_view_is_compiled_through_country_spec(self) -> None:
        from microcosm.build.country_spec import (
            load_country_take_up_contract_projection,
        )

        projection = load_country_take_up_contract_projection("us")
        assert projection is not None
        contract = load_take_up_contract()
        evidence = load_legacy_take_up_contract_evidence()

        assert [dict(program.raw) for program in contract.programs] == list(
            projection["programs"]
        )
        assert take_up_contract_identity(contract) == take_up_contract_identity(
            evidence
        )

    def test_narrow_country_projection_refuses_a_stale_engine_lock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from microcosm.build.country_spec import (
            load_country_take_up_contract_projection,
        )
        from microcosm.build.spec_engine import engine_abi as engine_abi_module
        from microcosm.build.spec_engine.errors import SpecValidationError

        monkeypatch.setattr(
            engine_abi_module,
            "_installed_engine_version",
            lambda package: "0.0.0-stale-test",
        )
        with pytest.raises(
            SpecValidationError,
            match="engine version differs from the exact generated engine pin",
        ):
            load_country_take_up_contract_projection("us")

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

    def test_canonical_identity_covers_every_structured_field(self) -> None:
        contract = load_take_up_contract()

        assert take_up_contract_identity(contract) == {
            "version": contract.version,
            "country": contract.country,
            "resource_sha256": contract.resource_sha256,
            "asserted_constraint": contract.asserted_constraint,
            "inventory_built_against": contract.inventory_built_against,
            "programs": [dict(program.raw) for program in contract.programs],
        }

    def test_resource_digest_matches_the_complete_canonical_json(self) -> None:
        resource = json.loads(
            files("microcosm.build.us").joinpath("take_up_contract.json").read_text()
        )
        canonical = json.dumps(
            resource,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        expected_sha256 = hashlib.sha256(canonical).hexdigest()
        contract = load_take_up_contract()

        assert contract.resource_sha256 == expected_sha256
        assert take_up_contract_identity(contract)["resource_sha256"] == expected_sha256

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

    def test_ssi_uses_reporter_anchored_registry_band_targets(self) -> None:
        program = load_take_up_contract().program_map()["takes_up_ssi_if_eligible"]
        calibration = program.raw["calibration"]

        assert program in count_calibrated_take_up_programs()
        assert calibration["anchor"] == "SSI_VAL"
        assert calibration["targets"] == ["ssa_ssi_federal_payment_recipients_by_age"]
        assert calibration["target_source"] == (
            "https://www.ssa.gov/policy/docs/statcomps/ssi_monthly/2024-12/table01.html"
        )
        assert calibration["target_period"] == "2024-12"
        assert calibration["target_measure"] == "Total with—Federal payment"
        assert calibration["target_role"] == "ssa_ssi_age_band_recipients"
        assert calibration["age_bands"] == {
            "under_18": "age < 18",
            "18_64": "18 <= age < 65",
            "65_plus": "age >= 65",
        }
        # The SSA recipient counts live in the ledger and bind through the
        # calibration registry (populace#469/#470) — the contract must not
        # carry a hardcoded copy, and the semantics must describe seeded
        # Bernoulli priors, not flag count-matching.
        assert "target_values" not in calibration
        assert "aggregate_target" not in calibration
        semantics = calibration["semantics"]
        assert "populace#469" in semantics
        assert "never count-matches" in semantics
        assert "saturate" not in semantics
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
        load_legacy_take_up_contract_evidence.cache_clear()
        payload = json.dumps(mutated)

        class _FakePath:
            def read_text(self, *args, **kwargs):
                return payload

        monkeypatch.setattr(
            "microcosm.build.us_runtime.take_up_contract._contract_path",
            lambda: _FakePath(),
        )

    @pytest.fixture
    def base_table(self):
        raw = json.loads(
            __import__("importlib.resources", fromlist=["files"])
            .files("microcosm.build.us")
            .joinpath("take_up_contract.json")
            .read_text()
        )
        yield raw
        load_take_up_contract.cache_clear()
        load_legacy_take_up_contract_evidence.cache_clear()

    def test_dropping_a_program_fails(self, monkeypatch, base_table) -> None:
        mutated = copy.deepcopy(base_table)
        mutated["programs"] = [
            p
            for p in mutated["programs"]
            if p["variable"] != "takes_up_snap_if_eligible"
        ]
        self._reload_with(monkeypatch, mutated)
        with pytest.raises(AssertionError, match="takes_up_snap_if_eligible"):
            assert_take_up_contract_current(
                contract=load_legacy_take_up_contract_evidence()
            )

    @pytest.mark.parametrize(
        "resource_path",
        (
            pytest.param(("policy",), id="policy"),
            pytest.param(("doctrine", "engine_class"), id="doctrine"),
            pytest.param(
                ("asserted_engine", "package"),
                id="asserted_engine_package",
            ),
            pytest.param(
                ("asserted_engine", "note"),
                id="asserted_engine_note",
            ),
        ),
    )
    def test_resource_digest_binds_remaining_structured_fields(
        self,
        monkeypatch,
        base_table,
        resource_path: tuple[str, ...],
    ) -> None:
        load_take_up_contract.cache_clear()
        load_legacy_take_up_contract_evidence.cache_clear()
        baseline_identity = take_up_contract_identity(
            load_legacy_take_up_contract_evidence()
        )
        mutated = copy.deepcopy(base_table)
        parent = mutated
        for key in resource_path[:-1]:
            parent = parent[key]
        field = resource_path[-1]
        parent[field] = f"{parent[field]}-identity-mutation"
        self._reload_with(monkeypatch, mutated)

        changed_identity = take_up_contract_identity(
            load_legacy_take_up_contract_evidence()
        )

        assert (
            changed_identity["resource_sha256"] != baseline_identity["resource_sha256"]
        )
        assert {
            key: value
            for key, value in changed_identity.items()
            if key != "resource_sha256"
        } == {
            key: value
            for key, value in baseline_identity.items()
            if key != "resource_sha256"
        }

    def test_wrong_engine_class_fails(self, monkeypatch, base_table) -> None:
        mutated = copy.deepcopy(base_table)
        for program in mutated["programs"]:
            if program["variable"] == "takes_up_tanf_if_eligible":
                program["engine_class"] = "model_simulated"
        self._reload_with(monkeypatch, mutated)
        with pytest.raises(AssertionError, match="engine_class"):
            assert_take_up_contract_current(
                contract=load_legacy_take_up_contract_evidence()
            )

    def test_wrong_default_fails(self, monkeypatch, base_table) -> None:
        mutated = copy.deepcopy(base_table)
        for program in mutated["programs"]:
            if program["variable"] == "takes_up_eitc":
                program["default"] = False
        self._reload_with(monkeypatch, mutated)
        with pytest.raises(AssertionError, match="default"):
            assert_take_up_contract_current(
                contract=load_legacy_take_up_contract_evidence()
            )

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
            assert_take_up_contract_current(
                contract=load_legacy_take_up_contract_evidence()
            )

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
            assert_take_up_treatments_consistent(
                contract=load_legacy_take_up_contract_evidence()
            )


class TestSeedProvenanceIsEnforced:
    """A seeded program must carry sourced provenance; the loader enforces it."""

    def _load_mutated(self, monkeypatch, mutated: dict):
        load_take_up_contract.cache_clear()
        load_legacy_take_up_contract_evidence.cache_clear()
        payload = json.dumps(mutated)

        class _FakePath:
            def read_text(self, *args, **kwargs):
                return payload

        monkeypatch.setattr(
            "microcosm.build.us_runtime.take_up_contract._contract_path",
            lambda: _FakePath(),
        )

    @pytest.fixture
    def base_table(self):
        raw = json.loads(
            __import__("importlib.resources", fromlist=["files"])
            .files("microcosm.build.us")
            .joinpath("take_up_contract.json")
            .read_text()
        )
        yield raw
        load_take_up_contract.cache_clear()
        load_legacy_take_up_contract_evidence.cache_clear()

    def test_seed_without_rate_source_is_refused(self, monkeypatch, base_table) -> None:
        mutated = copy.deepcopy(base_table)
        for program in mutated["programs"]:
            if program["variable"] == "takes_up_tanf_if_eligible":
                program["rate"] = {"value": 0.219}  # no source
        self._load_mutated(monkeypatch, mutated)
        with pytest.raises(ValueError, match="source"):
            load_legacy_take_up_contract_evidence()

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
            load_legacy_take_up_contract_evidence()

    def test_seeded_programs_all_have_administrative_provenance(self) -> None:
        # The real table: every seeded program carries a sourced rate.
        for program in seeded_take_up_programs():
            rate = program.rate
            assert rate.get("source"), program.variable
            assert str(rate.get("status", "")).startswith("sourced"), program.variable
