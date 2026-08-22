"""Contract tests for the replacement head-to-head scorer.

The scorer is one common path for incumbent and candidate; these tests pin
the pieces that make the head-to-head honest without running the heavy
microsim materialization: the signature has no target-membership switches,
the scored-column contract cannot go silently missing on either side, the
terminal-battery receipt is observed rather than asserted, and the fixture
end-to-end run is deterministic byte-for-byte.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.calibrate import TargetRegistry
from microcosm.calibrate.registry import TargetSpec
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def _load_head_to_head_module():
    root = Path(__file__).resolve().parents[3]
    tools_path = str(root / "tools")
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    name = "score_us_release_head_to_head"
    if name in sys.modules:
        return sys.modules[name]
    path = root / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tiny_frame(
    *,
    measure_values: tuple[float, float],
    channels: tuple[str, str] | None = ("asec", "asec"),
) -> Frame:
    household = {
        "household_id": np.asarray([1, 2], dtype="int64"),
        "state_fips": np.asarray([6, 36], dtype="int64"),
        "m_income": np.asarray(measure_values, dtype="float64"),
        "n_flagged": np.asarray([1.0, 0.0], dtype="float64"),
    }
    person = {
        "person_id": np.asarray([1, 2], dtype="int64"),
        "person_household_id": np.asarray([1, 2], dtype="int64"),
        "person_tax_unit_id": np.asarray([1, 2], dtype="int64"),
        "person_spm_unit_id": np.asarray([1, 2], dtype="int64"),
        "person_family_id": np.asarray([1, 2], dtype="int64"),
        "person_marital_unit_id": np.asarray([1, 2], dtype="int64"),
    }
    tax_unit = {"tax_unit_id": np.asarray([1, 2], dtype="int64")}
    spm_unit = {"spm_unit_id": np.asarray([1, 2], dtype="int64")}
    if channels is not None:
        # The by-origin battery scopes its masks on the battery entities
        # (person, tax_unit, spm_unit), matching the observed live incumbent.
        for prefix, table in (
            ("person", person),
            ("tax_unit", tax_unit),
            ("spm_unit", spm_unit),
        ):
            table[f"{prefix}_support_channel"] = np.asarray(channels, dtype=object)
            table[f"{prefix}_support_clone_index"] = np.asarray([0, 0], dtype="int64")
    tables = {
        "person": pd.DataFrame(person),
        "household": pd.DataFrame(household),
        "tax_unit": pd.DataFrame(tax_unit),
        "spm_unit": pd.DataFrame(spm_unit),
        "family": pd.DataFrame({"family_id": np.asarray([1, 2], dtype="int64")}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": np.asarray([1, 2], dtype="int64")}
        ),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([10.0, 20.0], dtype="float64"),
                WeightKind.CALIBRATED,
            )
        },
    )


def _tiny_registry() -> TargetRegistry:
    return TargetRegistry(
        [
            TargetSpec(
                name="tiny_income_total",
                entity="household",
                value=500.0,
                measure="m_income",
                period=2024,
                source="fixture",
                family="fixture_family",
            ),
            TargetSpec(
                name="tiny_flagged_count",
                entity="household",
                value=12.0,
                measure="n_flagged",
                period=2024,
                source="fixture",
                family="fixture_family",
                metadata={"measure_mode": "indicator_sum"},
            ),
        ],
        country="us",
    )


def _fixture_yardstick(module) -> object:
    registry = _tiny_registry()
    release = module.release
    loss_weights = release._fiscal_target_loss_weights(registry)
    loss_basis = release._fiscal_target_loss_basis(registry, loss_weights)
    return module.FiscalYardstick(
        registry=registry,
        loss_weights=loss_weights,
        loss_basis=loss_basis,
        identity={
            "country": "us",
            "version": registry.version,
            "target_count": len(registry.specs),
            "ledger_facts": {"filename": "fixture.jsonl", "sha256": "0" * 64},
            "congressional_district_vintage_crosswalk": {
                "filename": "fixture.parquet",
                "sha256": "1" * 64,
            },
            "target_period": 2024,
            "age_targets": False,
            "allow_unaged_dollar_targets": True,
            "target_profile_coverage": {"passed": True, "failures": []},
            "environment": {
                "microcosm_commit": "fixture",
                "policyengine_us_version": "fixture",
            },
        },
    )


def _fixture_artifact(module, *, sha256: str, measure_values: tuple[float, float]):
    return module.LoadedArtifact(
        frame=_tiny_frame(measure_values=measure_values),
        identity={
            "kind": "h5",
            "filename": f"{sha256[:8]}.h5",
            "sha256": sha256,
            "size_bytes": 123,
        },
        loader={"kind": "microcosm_entity_h5", "weight_kind": "calibrated"},
        h5_path=Path(f"/nonexistent/{sha256[:8]}.h5"),
    )


def _patch_release_seams(module, monkeypatch) -> None:
    """Stub the four heavy release seams; everything downstream runs real."""

    release = module.release

    def _identity_repair(frame):
        return frame, {"mode": "fixture_noop"}

    def _stub_gate(*args, **kwargs):
        return SimpleNamespace(passed=True, failures=(), details={})

    def _stub_materialize(frame, specs, **kwargs):
        registry = TargetRegistry(list(specs), country="us")
        return (
            frame,
            registry,
            {
                "declared_targets": len(specs),
                "compiled_candidate_targets": len(specs),
                "dropped_target_names": [],
            },
        )

    def _stub_cd_probe(h5_path):
        return {
            module.CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR: None,
            module.CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR: None,
            "household_congressional_district_geoid": {
                "exists": True,
                "positive_unique_count": 2,
            },
        }

    monkeypatch.setattr(release, "_with_base_population_mass_repair", _identity_repair)
    monkeypatch.setattr(release, "_base_population_scale_gate", _stub_gate)
    monkeypatch.setattr(release, "_health_input_signal_gate", _stub_gate)
    monkeypatch.setattr(release, "_materialize_target_frame", _stub_materialize)
    monkeypatch.setattr(release, "_read_cd_vintage_support_provenance", _stub_cd_probe)


def test_head_to_head_signature_has_no_target_membership_switches() -> None:
    module = _load_head_to_head_module()

    assert set(inspect.signature(module.score_head_to_head).parameters) == {
        "incumbent",
        "candidate",
        "ledger_facts",
        "age_targets",
        "allow_unaged_dollar_targets",
        "congressional_district_vintage_crosswalk",
        "maximum_microsim_batch_size",
        "target_materialization_cache_dir",
        "candidate_manifest_sha256",
    }


def test_canonical_battery_contract_matches_production_registries() -> None:
    module = _load_head_to_head_module()

    contract = module._canonical_battery_contract()

    single = len(module.CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY)
    joint = len(module.CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY)
    assert single == 131
    assert joint == 1
    assert len(contract) == single + joint
    assert (
        "person/source_operator_immigration/"
        "joint[ssn_card_type,immigration_status_str][clone_0]" in contract
    )
    for row in contract.values():
        assert row["metric_legs"] == list(module._metric_legs(row["metric"]))


def test_observed_origin_battery_is_evidence_not_assertion() -> None:
    module = _load_head_to_head_module()

    no_columns = module._battery_payload_from_observed_origins(
        _tiny_frame(measure_values=(1.0, 2.0), channels=None)
    )
    asec_only = module._battery_payload_from_observed_origins(
        _tiny_frame(measure_values=(1.0, 2.0), channels=("asec", "asec"))
    )
    both_origins = module._battery_payload_from_observed_origins(
        _tiny_frame(measure_values=(1.0, 2.0), channels=("asec", "acs"))
    )

    assert no_columns["status"] == "inapplicable"
    assert "no support-channel" in no_columns["reason"]
    assert asec_only["status"] == "inapplicable"
    assert "no ACS-stacked origin rows" in asec_only["reason"]
    assert asec_only["observed_origins"]["total_acs_rows"] == 0
    assert asec_only["observed_origins"]["entities"]["person"]["asec_rows"] == 2
    assert both_origins["status"] == "inapplicable"
    assert "authenticated pool manifest" in both_origins["reason"]
    for payload in (no_columns, asec_only, both_origins):
        assert payload["comparison_count"] == 132
        assert all(
            row["status"] == "inapplicable" for row in payload["comparisons"].values()
        )


def test_scored_column_contract_refuses_silently_missing_columns() -> None:
    module = _load_head_to_head_module()
    registry = _tiny_registry()
    complete = _tiny_frame(measure_values=(1.0, 2.0))
    broken_tables = {
        entity: complete.table(entity).copy() for entity in complete.entities
    }
    broken_tables["household"] = broken_tables["household"].drop(columns=["n_flagged"])
    broken = Frame(
        broken_tables,
        US_SCHEMA,
        {"household": complete.weights_for("household")},
    )

    contract = module.scored_column_contract(
        complete, registry, artifact_name="incumbent"
    )

    assert ("household", "m_income", "measure") in contract
    assert ("household", "n_flagged", "measure") in contract
    with pytest.raises(ValueError, match="lacks scored column"):
        module.scored_column_contract(broken, registry, artifact_name="candidate")
    with pytest.raises(ValueError, match="differs from incumbent"):
        module._assert_identical_scored_contracts(
            {"incumbent": contract, "candidate": contract[:-1]}
        )


def test_fixture_end_to_end_is_deterministic_and_shares_one_path(
    monkeypatch, tmp_path
) -> None:
    module = _load_head_to_head_module()
    _patch_release_seams(module, monkeypatch)
    yardstick = _fixture_yardstick(module)
    incumbent = _fixture_artifact(
        module, sha256="a" * 64, measure_values=(100.0, 300.0)
    )
    candidate = _fixture_artifact(
        module, sha256="b" * 64, measure_values=(200.0, 290.0)
    )

    def _score_both() -> dict[str, object]:
        artifacts: dict[str, dict[str, object]] = {}
        contracts: dict[str, tuple] = {}
        for name, artifact in (("incumbent", incumbent), ("candidate", candidate)):
            payload, contract = module.score_loaded_artifact(
                artifact=artifact,
                artifact_name=name,
                yardstick=yardstick,
                maximum_microsim_batch_size=None,
                target_materialization_cache_dir=None,
            )
            artifacts[name] = payload
            contracts[name] = contract
        module._assert_identical_scored_contracts(contracts)
        return {
            "schema_version": module.SCHEMA_VERSION,
            "yardstick": {
                "fiscal_registry": dict(yardstick.identity),
                "fiscal_aggregate": {
                    "name": module.release.US_FISCAL_TARGET_LOSS_WEIGHTING,
                    "target_loss_cap": module.release.US_FISCAL_TARGET_LOSS_CAP,
                    "loss_basis": dict(yardstick.loss_basis),
                    "weighting_rule": "fixture",
                    "family_multipliers": None,
                    "code_citations": [],
                },
                "relative_error": {"rule": "fixture", "code_citation": "fixture"},
                "terminal_battery": {
                    "by_origin_only": True,
                    "single_column_comparison_count": 131,
                    "joint_comparison_count": 1,
                    "comparison_count": 132,
                    "metric_leg_count": 0,
                    "code_citations": [],
                },
                "code_citations": dict(module._CODE_CITATIONS),
            },
            "artifacts": {
                "incumbent": artifacts["incumbent"],
                "candidate": artifacts["candidate"],
            },
            "comparison": module._comparison_payload(
                artifacts["incumbent"], artifacts["candidate"]
            ),
        }

    payload_one = _score_both()
    payload_two = _score_both()

    assert payload_one == payload_two

    incumbent_fiscal = payload_one["artifacts"]["incumbent"]["fiscal"]
    rows = incumbent_fiscal["targets"]
    assert [row["name"] for row in rows] == [
        "tiny_income_total",
        "tiny_flagged_count",
    ]
    # Hand-computed: estimates are A @ w with w = (10, 20).
    # m_income: 100*10 + 300*20 = 7000 vs target 500 -> relative error 13.0,
    # capped scaled error 1.0. n_flagged: 1*10 + 0*20 = 10 vs target 12
    # -> relative error -1/6.
    assert rows[0]["actual"] == pytest.approx(7000.0)
    assert rows[0]["relative_error"] == pytest.approx(13.0)
    assert rows[0]["capped_scaled_absolute_error"] == pytest.approx(1.0)
    assert rows[1]["actual"] == pytest.approx(10.0)
    assert rows[1]["relative_error"] == pytest.approx(-2.0 / 12.0)
    contribution_sum = sum(row["weighted_loss_contribution"] for row in rows)
    assert contribution_sum == pytest.approx(incumbent_fiscal["weighted_loss"])

    comparison = payload_one["comparison"]
    counts = comparison["per_target_absolute_relative_error"]
    # Candidate: m_income actual 200*10 + 290*20 = 7800 (worse than 7000);
    # n_flagged identical inputs -> equal.
    assert counts["incumbent_lower_count"] == 1
    assert counts["equal_count"] == 1
    assert counts["candidate_lower_count"] == 0
    assert comparison["terminal_battery"]["head_to_head_comparable"] is False
    assert comparison["no_threshold_applied"] is True

    first = module.write_scorecard(payload_one, tmp_path / "one" / "scorecard")
    second = module.write_scorecard(payload_two, tmp_path / "two" / "scorecard")
    for path_one, path_two in zip(first, second, strict=True):
        assert path_one.read_bytes() == path_two.read_bytes()
    markdown = first[1].read_text()
    assert "US release replacement scorecard" in markdown
    assert "no ACS-stacked origin rows" in markdown


def test_dropped_targets_fail_loudly_before_scoring(monkeypatch) -> None:
    module = _load_head_to_head_module()
    _patch_release_seams(module, monkeypatch)

    def _dropping_materialize(frame, specs, **kwargs):
        kept = list(specs)[:-1]
        return (
            frame,
            TargetRegistry(kept, country="us"),
            {
                "declared_targets": len(specs),
                "compiled_candidate_targets": len(kept),
                "dropped_target_names": [list(specs)[-1].name],
            },
        )

    monkeypatch.setattr(
        module.release, "_materialize_target_frame", _dropping_materialize
    )

    with pytest.raises(ValueError, match="did not materialize the full fiscal"):
        module.score_loaded_artifact(
            artifact=_fixture_artifact(
                module, sha256="c" * 64, measure_values=(1.0, 2.0)
            ),
            artifact_name="incumbent",
            yardstick=_fixture_yardstick(module),
            maximum_microsim_batch_size=None,
            target_materialization_cache_dir=None,
        )


def test_live_incumbent_identity_annotation() -> None:
    module = _load_head_to_head_module()

    pin = module._POLICYENGINE_LIVE_US_INCUMBENT
    matched = module._live_incumbent_identity_if_matched(pin["sha256"])
    unmatched = module._live_incumbent_identity_if_matched("f" * 64)

    assert unmatched is None
    assert matched is not None
    resolved = matched["policyengine_package_resolved"]
    assert resolved["repo_id"] == "policyengine/populace-us"
    assert resolved["filename"] == "populace_us_2024.h5"
    assert resolved["default_dataset"] == "populace_us_2024"
    assert (
        resolved["revision"]
        == "populace-us-2024-buildp-sparse-rmloss100-cae8640-20260728T011454Z"
    )
