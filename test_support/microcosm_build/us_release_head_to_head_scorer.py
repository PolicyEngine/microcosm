"""Shared fixtures for the US release head-to-head scorer tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from microcosm.calibrate import TargetRegistry
from microcosm.calibrate.registry import TargetSpec
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


def _load_head_to_head_module():
    root = _TEST_PATHS.repository
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
        tables["person"]["has_marketplace_health_coverage_at_interview"] = np.asarray(
            [True, False], dtype=bool
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


__all__ = [name for name in globals() if not name.startswith("__")]
