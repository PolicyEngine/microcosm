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

from microcosm.build.us_runtime.h5_io import write_nullable_us_h5
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


def _fixture_yardstick(module, registry=None) -> object:
    registry = registry if registry is not None else _tiny_registry()
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


def _tiny_frame_with_marketplace_columns(
    *,
    formula_owned: bool,
    interview_leaf: bool,
) -> Frame:
    frame = _tiny_frame(measure_values=(1.0, 2.0))
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    if formula_owned:
        tables["person"]["has_marketplace_health_coverage"] = np.asarray(
            [True, False], dtype=bool
        )
    if interview_leaf:
        tables["person"]["has_marketplace_health_coverage_at_interview"] = (
            np.asarray([True, False], dtype=bool)
        )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": frame.weights_for("household")},
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


def _score_loaded_as_incumbent(module, monkeypatch, loaded) -> dict[str, object]:
    _patch_release_seams(module, monkeypatch)
    monkeypatch.setattr(
        module,
        "compile_yardstick",
        lambda **kwargs: _fixture_yardstick(module),
    )
    monkeypatch.setattr(module, "load_artifact", lambda path, **kwargs: loaded)
    return module.score_head_to_head(
        incumbent=loaded.h5_path,
        candidate=None,
        ledger_facts=Path("/fixture/facts.jsonl"),
        congressional_district_vintage_crosswalk=Path("/fixture/crosswalk.parquet"),
        maximum_microsim_batch_size=1,
    )


def _complete_battery_comparisons(module) -> dict[str, dict[str, object]]:
    comparisons: dict[str, dict[str, object]] = {}
    for label, row in module._canonical_battery_contract().items():
        metric = row["metric"]
        if metric == "boolean_incidence":
            receipt = {
                "status": "tested",
                "metric": metric,
                "asec_incidence": 0.5,
                "acs_incidence": 0.5,
                "incidence_ratio_acs_over_asec": 1.0,
            }
        elif metric == "categorical_tvd":
            receipt = {
                "status": "tested",
                "metric": metric,
                "total_variation_distance": 0.0,
                "category_shares": {
                    "asec": {"fixture": 1.0},
                    "acs": {"fixture": 1.0},
                },
            }
        else:
            receipt = {
                "status": "tested",
                "metric": metric,
                "legs": {
                    sign: {
                        "asec_incidence": 0.5,
                        "acs_incidence": 0.5,
                        "incidence_ratio_acs_over_asec": 1.0,
                        "quantile_envelope_distance": 0.0,
                    }
                    for sign in ("positive", "negative")
                },
            }
        comparisons[label] = receipt
    return comparisons


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
        "candidate_manifest_sha256",
    }


def test_dense_candidate_streaming_plan_is_independent_of_total_pool_size() -> None:
    module = _load_head_to_head_module()
    dense_25pct_households = 918_350

    planned = module._streaming_target_column_payload_upper_bound_bytes(
        total_households=dense_25pct_households,
        household_slice_size=5_000,
        chunk_spec_count=module.MATERIALIZE_SCORE_CHUNK_SPECS,
    )
    one_dense_copy = (
        dense_25pct_households
        * module.MATERIALIZE_SCORE_CHUNK_SPECS
        * np.dtype(np.float64).itemsize
    )

    assert planned < 1024**3
    assert planned < module.MAX_RSS_BYTES
    assert one_dense_copy > module.MAX_RSS_BYTES


def test_canonical_battery_contract_matches_production_registries() -> None:
    module = _load_head_to_head_module()

    contract = module._canonical_battery_contract()

    single = len(module.CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY)
    joint = len(module.CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY)
    assert single == 134
    assert joint == 1
    assert len(contract) == single + joint
    assert sum(len(row["metric_legs"]) for row in contract.values()) == 372
    assert (
        sum(row["metric"] == "monetary_sign_separated" for row in contract.values())
        == 79
    )
    assert sum(row["metric"] == "boolean_incidence" for row in contract.values()) == 51
    assert sum(row["metric"] == "categorical_tvd" for row in contract.values()) == 5
    assert (
        "person/source_operator_immigration/"
        "joint[ssn_card_type,immigration_status_str][clone_0]" in contract
    )
    for row in contract.values():
        assert row["metric_legs"] == list(module._metric_legs(row["metric"]))


def test_observed_origin_battery_is_evidence_not_assertion(monkeypatch) -> None:
    module = _load_head_to_head_module()

    monkeypatch.setattr(
        module,
        "materialize_multispine_agreement_outputs",
        lambda frame: SimpleNamespace(
            frame=frame,
            receipt={"persisted_to_artifact": False},
        ),
    )
    monkeypatch.setattr(
        module,
        "by_origin_battery_artifact_evidence",
        lambda frame: SimpleNamespace(
            passed=True,
            failures=(),
            details={"comparisons": _complete_battery_comparisons(module)},
        ),
    )

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
    assert "empty ACS side" in asec_only["reason"]
    assert asec_only["observed_origins"]["total_acs_rows"] == 0
    assert asec_only["observed_origins"]["entities"]["person"]["asec_rows"] == 2
    assert both_origins["status"] == "computed_finished_h5"
    assert both_origins["production_receipt_authenticated"] is False
    assert both_origins["metric_leg_count"] == 372
    assert both_origins["scalar_leg_status_counts"] == {"computed": 372}
    for payload in (no_columns, asec_only):
        assert payload["comparison_count"] == 135
        assert all(
            row["status"] == "inapplicable" for row in payload["comparisons"].values()
        )


def test_pool_battery_receipt_refuses_a_silently_missing_scalar_leg() -> None:
    module = _load_head_to_head_module()
    comparisons = _complete_battery_comparisons(module)
    label = next(
        label
        for label, row in comparisons.items()
        if row["metric"] == "boolean_incidence"
    )
    del comparisons[label]["incidence_ratio_acs_over_asec"]
    terminal_gates = {
        "gates": {
            "us_by_origin_battery": {
                "passed": True,
                "failures": [],
                "details": {"comparisons": comparisons},
            }
        }
    }

    with pytest.raises(ValueError, match="omits computed leg"):
        module._battery_payload_from_pool_receipt(terminal_gates)


def test_origin_probe_uses_clone_zero_positive_weight_scope() -> None:
    module = _load_head_to_head_module()
    frame = _tiny_frame(measure_values=(1.0, 2.0), channels=("acs", "asec"))
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for entity in ("person", "tax_unit", "spm_unit"):
        tables[entity].loc[tables[entity].index[0], f"{entity}_support_clone_index"] = 1
    clone_scoped = Frame(
        tables,
        US_SCHEMA,
        {"household": frame.weights_for("household")},
    )
    zero_weight_scoped = Frame(
        {entity: frame.table(entity).copy() for entity in frame.entities},
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([0.0, 20.0], dtype="float64"),
                WeightKind.CALIBRATED,
            )
        },
    )

    observed_by_clone = module._observed_origin_receipt(clone_scoped)
    observed_by_weight = module._observed_origin_receipt(zero_weight_scoped)

    for observed in (observed_by_clone, observed_by_weight):
        assert observed["total_acs_rows"] == 0
        assert observed["total_asec_rows"] == 3
        for entity in ("person", "tax_unit", "spm_unit"):
            receipt = observed["entities"][entity]
            assert receipt["raw_origin_row_counts"] == {"acs": 1, "asec": 1}
            assert receipt["origin_row_counts"] == {"asec": 1}


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
        complete, registry.specs, artifact_name="incumbent"
    )

    assert ("household", "m_income", "measure") in contract
    assert ("household", "n_flagged", "measure") in contract
    with pytest.raises(ValueError, match="lacks scored column"):
        module.scored_column_contract(broken, registry.specs, artifact_name="candidate")
    with pytest.raises(ValueError, match="differs from incumbent"):
        module._assert_identical_scored_contracts(
            {"incumbent": contract, "candidate": contract[:-1]}
        )


def test_incumbent_and_candidate_h5_loaders_preserve_scored_contract(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    module = _load_head_to_head_module()
    registry = _tiny_registry()
    incumbent_path = tmp_path / "incumbent.h5"
    candidate_path = tmp_path / "candidate.h5"
    broken_path = tmp_path / "candidate_missing_measure.h5"
    incumbent_frame = _tiny_frame(measure_values=(1.0, 2.0))
    candidate_frame = _tiny_frame(measure_values=(3.0, 4.0))
    broken_tables = {
        entity: candidate_frame.table(entity).copy()
        for entity in candidate_frame.entities
    }
    broken_tables["household"] = broken_tables["household"].drop(columns=["n_flagged"])
    broken_frame = Frame(
        broken_tables,
        US_SCHEMA,
        {"household": candidate_frame.weights_for("household")},
    )
    for path, frame in (
        (incumbent_path, incumbent_frame),
        (candidate_path, candidate_frame),
        (broken_path, broken_frame),
    ):
        write_nullable_us_h5(
            frame,
            path,
            period=2024,
            artifact_kind="replacement_scorecard_fixture",
        )

    incumbent = module.load_artifact(incumbent_path)
    candidate = module.load_artifact(candidate_path)
    broken = module.load_artifact(broken_path)
    incumbent_contract = module.scored_column_contract(
        incumbent.frame,
        registry.specs,
        artifact_name="incumbent",
    )
    candidate_contract = module.scored_column_contract(
        candidate.frame,
        registry.specs,
        artifact_name="candidate",
    )

    module._assert_identical_scored_contracts(
        {"incumbent": incumbent_contract, "candidate": candidate_contract}
    )
    assert incumbent.loader["kind"] == candidate.loader["kind"]
    with pytest.raises(ValueError, match="lacks scored column"):
        module.scored_column_contract(
            broken.frame,
            registry.specs,
            artifact_name="candidate",
        )


def test_historical_formula_owned_h5_scores_with_drop_receipt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    module = _load_head_to_head_module()
    artifact_path = tmp_path / "formula_owned_with_leaf.h5"
    write_nullable_us_h5(
        _tiny_frame_with_marketplace_columns(
            formula_owned=True,
            interview_leaf=True,
        ),
        artifact_path,
        period=2024,
        artifact_kind="replacement_scorecard_fixture",
    )

    loaded = module.load_artifact(artifact_path)
    person_columns = set(loaded.frame.table("person").columns)
    assert "has_marketplace_health_coverage" not in person_columns
    assert "has_marketplace_health_coverage_at_interview" in person_columns

    payload = _score_loaded_as_incumbent(module, monkeypatch, loaded)
    receipt = payload["artifacts"]["incumbent"]["normalization_receipts"][
        "historical_formula_owned_columns"
    ]
    assert receipt == {
        "count": 1,
        "columns_by_entity": {
            "person": ["has_marketplace_health_coverage"],
        },
    }
    markdown = module.render_markdown(payload)
    assert "Dropped column count: **1**" in markdown
    assert "`person`: `has_marketplace_health_coverage`" in markdown


def test_historical_formula_owned_h5_refuses_missing_leaf(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    module = _load_head_to_head_module()
    artifact_path = tmp_path / "formula_owned_without_leaf.h5"
    write_nullable_us_h5(
        _tiny_frame_with_marketplace_columns(
            formula_owned=True,
            interview_leaf=False,
        ),
        artifact_path,
        period=2024,
        artifact_kind="replacement_scorecard_fixture",
    )

    with pytest.raises(ValueError) as exc_info:
        module.load_artifact(artifact_path)

    message = str(exc_info.value)
    assert "has_marketplace_health_coverage" in message
    assert "has_marketplace_health_coverage_at_interview" in message
    assert "required input leaves are absent" in message


def test_clean_historical_h5_scores_with_empty_drop_receipt(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    module = _load_head_to_head_module()
    artifact_path = tmp_path / "clean.h5"
    write_nullable_us_h5(
        _tiny_frame_with_marketplace_columns(
            formula_owned=False,
            interview_leaf=True,
        ),
        artifact_path,
        period=2024,
        artifact_kind="replacement_scorecard_fixture",
    )

    loaded = module.load_artifact(artifact_path)
    payload = _score_loaded_as_incumbent(module, monkeypatch, loaded)
    receipt = payload["artifacts"]["incumbent"]["normalization_receipts"][
        "historical_formula_owned_columns"
    ]
    assert receipt == {"count": 0, "columns_by_entity": {}}
    markdown = module.render_markdown(payload)
    assert "Dropped column count: **0**" in markdown
    assert "No formula-owned columns were present." in markdown


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
    incumbent_path = Path("/fixture/incumbent.h5")
    candidate_path = Path("/fixture/candidate.h5")
    artifacts = {incumbent_path: incumbent, candidate_path: candidate}
    monkeypatch.setattr(module, "compile_yardstick", lambda **kwargs: yardstick)
    monkeypatch.setattr(
        module,
        "load_artifact",
        lambda path, **kwargs: artifacts[path],
    )

    def _score_both() -> dict[str, object]:
        return module.score_head_to_head(
            incumbent=incumbent_path,
            candidate=candidate_path,
            ledger_facts=Path("/fixture/facts.jsonl"),
            congressional_district_vintage_crosswalk=Path("/fixture/crosswalk.parquet"),
            maximum_microsim_batch_size=1,
        )

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
    assert "empty ACS side" in markdown


def test_chunked_scoring_recombination_matches_one_shot(monkeypatch) -> None:
    """Chunked materialize-and-score must reproduce a one-shot score_targets
    + production attribution bitwise: same aggregate, same per-target rows."""

    from microcosm.calibrate import score_targets
    from microcosm.calibrate._target_loss_attribution import (
        assemble_target_loss_attribution,
    )

    module = _load_head_to_head_module()
    _patch_release_seams(module, monkeypatch)
    monkeypatch.setattr(module, "MATERIALIZE_SCORE_CHUNK_SPECS", 2)
    registry = TargetRegistry(
        [
            TargetSpec(
                name=name,
                entity="household",
                value=value,
                measure=measure,
                period=2024,
                source="fixture",
                family="fixture_family",
                signed=value < 0,
                metadata=(
                    {"measure_mode": "indicator_sum"} if measure == "n_flagged" else {}
                ),
            )
            for name, measure, value in (
                ("five_income_a", "m_income", 5_000.0),
                ("five_flagged_b", "n_flagged", 12.0),
                ("five_income_c", "m_income", 7_100.0),
                ("five_zero_d", "n_flagged", 0.0),
                ("five_income_e", "m_income", -250.0),
            )
        ],
        country="us",
    )
    yardstick = _fixture_yardstick(module, registry=registry)
    frame = _tiny_frame(measure_values=(100.0, 300.0))
    artifact = module.LoadedArtifact(
        frame=frame,
        identity={
            "kind": "h5",
            "filename": "five.h5",
            "sha256": "d" * 64,
            "size_bytes": 5,
        },
        loader={"kind": "microcosm_entity_h5", "weight_kind": "calibrated"},
        h5_path=Path("/nonexistent/five.h5"),
    )

    payload, _ = module.score_loaded_artifact(
        artifact=artifact,
        artifact_name="incumbent",
        yardstick=yardstick,
        maximum_microsim_batch_size=1,
    )

    one_shot = score_targets(
        frame,
        registry.to_target_set(),
        target_loss_weights=yardstick.loss_weights,
        target_loss_cap=module.release.US_FISCAL_TARGET_LOSS_CAP,
    )
    attribution = assemble_target_loss_attribution(one_shot)

    chunking = payload["normalization_receipts"]["materialize_score_chunking"]
    assert chunking["chunk_count"] == 3
    assert [chunk["spec_range"] for chunk in chunking["chunks"]] == [
        [0, 2],
        [2, 4],
        [4, 5],
    ]
    assert all(
        chunk["target_compilation"]["household_slices"] == 2
        for chunk in chunking["chunks"]
    )
    assert payload["fiscal"]["weighted_loss"] == float(one_shot.final_loss)
    assert payload["fiscal"]["fraction_within_10pct"] == one_shot.fraction_within_10pct
    rows = payload["fiscal"]["targets"]
    assert len(rows) == 5
    for row, diagnostic, attribution_row in zip(
        rows,
        one_shot.diagnostics,
        attribution.rows,
        strict=True,
    ):
        assert row["actual"] == float(diagnostic.final_estimate)
        assert row["target"] == float(diagnostic.target)
        assert row["relative_error"] == float(diagnostic.relative_error)
        assert row["target_loss_weight"] == attribution_row["target_loss_weight"]
        assert (
            row["target_loss_weight_share"]
            == attribution_row["target_loss_weight_share"]
        )
        assert row["target_loss_scale"] == attribution_row["target_loss_scale"]
        assert (
            row["capped_scaled_absolute_error"]
            == attribution_row["final_capped_scaled_error"]
        )
        assert (
            row["weighted_loss_contribution"]
            == attribution_row["final_loss_contribution"]
        )


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
        )


def test_artifact_path_keeps_h5_symlink_name(tmp_path) -> None:
    """A Hugging Face cache snapshot is an .h5-named symlink to an
    extensionless blob; the scorer must keep the snapshot name so the
    dataset loader's suffix validation and the filename identity both see
    the real artifact name."""

    module = _load_head_to_head_module()
    blob = tmp_path / "blobs" / ("a" * 8)
    blob.parent.mkdir()
    blob.write_bytes(b"not-really-h5")
    snapshot = tmp_path / "snapshots" / "populace_us_2024.h5"
    snapshot.parent.mkdir()
    snapshot.symlink_to(blob)

    kept = module._resolved_artifact_path(snapshot)

    assert kept.name == "populace_us_2024.h5"
    assert kept.suffix == ".h5"
    with pytest.raises(FileNotFoundError):
        module._resolved_artifact_path(tmp_path / "missing.h5")


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
