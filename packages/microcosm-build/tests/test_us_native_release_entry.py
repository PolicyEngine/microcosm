"""Native release entry: option closure, consumer admission and refusal order.

Everything here is invented and engine-free. Forged owners must refuse at the
owner's own check before any other work. Where a later refusal, or the private
composition, needs to be reached, a stand-in replaces the owner check or the
helper is called directly. Those are ordering and wiring checks only. None is
evidence of a successful public native build, which needs a genuine issued
owner, a root-admitted consumer runtime and a closed native input profile.
"""

import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_us_fiscal_refresh_builder import _load_builder_module
from test_us_policyengine_h5_readback import _case, _logical_tables, _period

from microcosm.build.spm_input_contract import ROLE_INPUT, UNIVERSE_INPUT
from microcosm.build.us_runtime import graph_us_survey_enrichment as native_owner
from microcosm.build.us_runtime import native_survey_handoff as handoff
from microcosm.build.us_runtime import policyengine_h5_readback as readback
from microcosm.calibrate import (
    CalibrationHierarchy,
    HierarchyCategory,
    HierarchyGeography,
    HierarchyNode,
    TargetRegistry,
    TargetSpec,
)
from microcosm.frame import Frame, Weights
from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine
from microcosm.frame.rules import ExportContract

RELEASE_ID = "populace-us-2024-native-invented-20260923"
LEDGER_PIN = "a" * 64
SPM = {"geography_kind": "national"}
EFFECTIVE_SPM = {
    "forecast_content_sha256": "b" * 64,
    "scenario": "invented",
    "geography_kind": "national",
    "geography_id": None,
    "county_vintage": None,
    "as_of": None,
}


@pytest.fixture(scope="module")
def builder():
    return _load_builder_module()


@pytest.fixture(autouse=True)
def _no_country_engine(monkeypatch):
    """Every consumer here is invented; importing the country engine is a bug."""
    original = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if (
            name == "policyengine_us" or name.startswith("policyengine_us.")
        ) and not isinstance(sys.modules.get(name), SimpleNamespace):
            raise AssertionError("Actual country imports are forbidden")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


class InventedDataset:
    """Recorded only; never a PolicyEngine dataset."""


class InventedMicrosimulation:
    """Recorded only; never a PolicyEngine simulation."""


class InventedSystem:
    """Recorded only; never a PolicyEngine tax-benefit system."""


def _parent():
    """Six invented entity tables with complete SPM roles and canonical scope."""
    base, _, _ = _case("full")
    tables = {e: base.table(e).copy(deep=True) for e in base.entities}
    tables["household"]["state_fips"] = np.array([36, 36, 6], dtype=np.int64)
    weights = base.weights_for("household")
    return Frame(
        tables,
        base.schema,
        {"household": Weights(weights.values.copy(), weights.kind)},
        base.strata.copy(deep=True),
        metadata=base.metadata,
    )


def _declaration(frame, *, spm=None, period=2024, consumer_id_dtype="int64"):
    spm = SPM if spm is None else spm
    structural = {frame.schema.entity_id_column(e) for e in frame.entities} | {
        frame.schema.membership_column(g) for g in frame.schema.group_entities
    }
    leaves = sorted(
        {name for e in frame.entities for name in frame.table(e)} - structural
    )
    return handoff.NativeSurveyEngineProjectionSpec(
        period=period,
        consumer_identity="invented declaration; no consumer qualification",
        columns=tuple(
            (
                entity,
                tuple(
                    (name, str(frame.table(entity)[name].dtype))
                    for name in frame.table(entity)
                ),
            )
            for entity in frame.entities
        ),
        export_contract=ExportContract(
            required=("household_weight", *leaves),
            forbidden=(),
            optional=(),
            formula_owned_excluded=(),
            closed=True,
        ),
        enum_domains=(),
        consumer_id_dtype=consumer_id_dtype,
        spm_settings=tuple(sorted(spm.items())),
    )


def _argv(tmp_path, *extra, dense=True):
    return [
        "--out",
        str(tmp_path / "out"),
        "--ledger-facts",
        str(tmp_path / "facts.jsonl"),
        "--ledger-facts-sha256",
        LEDGER_PIN,
        "--release-id",
        RELEASE_ID,
        "--no-staging",
        "--no-target-materialization-cache",
        "--no-target-frame-checkpoint",
        "--epochs",
        "12",
        "--learning-rate",
        "0.03",
        "--max-weight-ratio",
        "4.0",
        "--seed",
        "17",
        "--l0-refit-lambda-share",
        "0.01",
        *(["--dense-default-dataset"] if dense else []),
        *extra,
    ]


def _engine(declaration, *, spm=None, defaults=None):
    return PolicyEngineUSEngine(
        contract=declaration.export_contract,
        defaults=defaults,
        spm=SPM if spm is None else spm,
    )


def _invented_consumer(builder, monkeypatch, *, effective=None, calls=None):
    constructors = builder._NativeReleaseConsumer(
        InventedDataset, InventedMicrosimulation, InventedSystem
    )

    def resolve(engine):
        if calls is not None:
            calls.append("constructors")
        return constructors

    def effective_spm(engine):
        if calls is not None:
            calls.append("effective_spm")
        return dict(EFFECTIVE_SPM if effective is None else effective)

    monkeypatch.setattr(builder, "_native_release_consumer_constructors", resolve)
    monkeypatch.setattr(builder, "_native_release_effective_spm", effective_spm)
    return constructors


def _manifest(builder, engine, constructors, effective=None):
    return {
        **builder._native_release_static_consumer_identity(engine, constructors),
        "effective_spm": dict(EFFECTIVE_SPM if effective is None else effective),
    }


def _forbid(monkeypatch, target, names, reached):
    for name in names:

        def forbidden(*args, _name=name, **kwargs):
            reached.append(_name)
            raise AssertionError(f"{_name} reached")

        monkeypatch.setattr(target, name, forbidden)


# Legacy base/pool/download/source-stage/writer/manifest paths and the native
# steps that must follow owner authentication.
_BUILDER_SENTINELS = (
    "main",
    "_main",
    "_parse_args",
    "_parse_native_release_args",
    "_admit_native_release_consumer",
    "_native_release_consumer_constructors",
    "_native_release_static_consumer_identity",
    "_native_release_effective_spm",
    "_download_base_h5",
    "_load_frame",
    "_load_base_pool_if_identified",
    "load_simulation_ready_us_multispine_pool",
    "_resolve_selection_source",
    "run_source_stage",
    "_compile_fiscal_release_target_registry",
    "_native_release_input_gate",
    "_load_or_materialize_target_frame",
    "_materialize_target_frame",
    "_calibrate_native_input_frame",
    "_calibrate_fiscal_support",
    "_run_prepared_native_fiscal_release",
    "write_calibration_diagnostics",
    "_write_release_calibration_diagnostics",
    "_build_manifests",
    "_copy_base_h5_for_local_audit",
    "_write_native_release_json",
    "StagingTelemetry",
)


class RecordingEngine(PolicyEngineUSEngine):
    """Any consumer method call means a forged owner got too far."""

    def __getattribute__(self, name):
        if name in {
            "write_dataset",
            "validate_input_representation",
            "export_contract",
            "_tax_benefit_system",
            "_import_policyengine_us",
            "_engine_computed_columns",
            "materialize",
        }:
            raise AssertionError(f"consumer {name} reached")
        return super().__getattribute__(name)


def _unissued_inputs(tmp_path):
    source = _parent()
    spec = _declaration(source)
    root = tmp_path / "descriptive-checkpoint"
    handoff._write_checkpoint(source, {"protocol": handoff.PROTOCOL}, root)
    checkpoint = handoff.load_native_survey_development_checkpoint(root)
    projection = handoff._project_native_survey_frame(source, spec)
    forged = native_owner.SurveyEnrichmentRun(
        None, None, None, None, None, None, (), b"invented"
    )
    checked_view = native_owner.CheckedSurveyEnrichmentRun(
        b"{}", "invented-digest", SimpleNamespace(frame=source)
    )
    return {
        "frame": source,
        "checkpoint": checkpoint,
        "checkpoint_report": checkpoint.report,
        "projection": projection,
        "checked_view": checked_view,
        "forged_run": forged,
        "object": object(),
    }


@pytest.mark.parametrize(
    "kind",
    [
        "frame",
        "checkpoint",
        "checkpoint_report",
        "projection",
        "checked_view",
        "forged_run",
        "object",
    ],
)
def test_public_entry_refuses_unissued_inputs_before_any_other_work(
    builder, monkeypatch, tmp_path, kind
):
    unsupported = _unissued_inputs(tmp_path)[kind]
    reached = []
    _forbid(monkeypatch, builder, _BUILDER_SENTINELS, reached)
    _forbid(monkeypatch, handoff, ("prepare_native_survey_engine_input",), reached)
    _forbid(monkeypatch, readback, ("write_verified_policyengine_h5_export",), reached)
    spec = _declaration(_parent())
    engine = RecordingEngine(contract=spec.export_contract, spm=SPM)
    with pytest.raises(ValueError, match="UNISSUED_RUN"):
        builder.build_native_survey_release(
            unsupported,
            argv=_argv(tmp_path, "--base-h5", str(tmp_path / "legacy.h5")),
            declaration=spec,
            engine=engine,
            consumer_manifest={"approved": True},
        )
    assert reached == []
    assert not (tmp_path / "out").exists()


def test_native_options_parse_only_the_consumed_surface(builder, tmp_path):
    options = builder._parse_native_release_args(_argv(tmp_path, dense=False))
    assert vars(options.solve) == {
        "exact_k": None,
        "dense_default_dataset": False,
        "epochs": 12,
        "learning_rate": 0.03,
        "max_weight_ratio": 4.0,
        "seed": 17,
        "l2_lambda": 0.0,
        "refit_l2_lambda": None,
        "l0_refit_lambda_share": 0.01,
    }
    assert options.args.release_id == RELEASE_ID
    assert options.args.no_staging is True
    assert options.args.target_family_loss_multipliers == {}


def test_every_consumed_option_exists_in_the_legacy_parser(builder, monkeypatch):
    import argparse

    seen = []
    original = argparse.ArgumentParser.parse_args

    def capture(self, *args, **kwargs):
        seen.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    builder._parse_args(["--out", "o", "--ledger-facts", "f", "--no-staging"])
    destinations = {action.dest for action in seen[0]._actions}
    # target_family_loss_multipliers is the parser's own post-parse normalization.
    assert builder._NATIVE_RELEASE_CONSUMED_OPTIONS - destinations == {
        "target_family_loss_multipliers"
    }
    assert set(builder._NATIVE_RELEASE_SOLVE_OPTIONS) <= destinations
    assert set(builder._NATIVE_RELEASE_REQUIRED_SWITCHES) <= destinations


@pytest.mark.parametrize(
    "extra",
    [
        ("--base-h5", "legacy.h5"),
        ("--warm-start-calibration-npz", "warm.npz"),
        ("--selection-source-h5", "support.h5"),
        ("--selection-source-manifest", "support.json"),
        ("--selection-mode", "informed_init"),
        ("--selection-mass-protection", "invented"),
        ("--input-mass-reference-h5", "reference.h5"),
        ("--export-input-mass-reference-h5", "reference.h5"),
        ("--incumbent-diagnostics", "incumbent.json"),
        ("--qrf-tail-concentration-exclusions", "tail.json"),
        ("--ssi-take-up-prior-weight-basis", "basis.json"),
        ("--asec-2023-weeks-unemployed-source", "weeks.csv"),
        ("--scf-summary-extract", "scf.dta"),
        ("--sipp-tip-donor", "tips.parquet"),
        ("--org-wages-donor", "org.parquet"),
        ("--checkpoint-root", "checkpoints"),
        ("--target-materialization-cache-dir", "cache"),
        ("--target-frame-checkpoint", "targets.h5"),
        ("--audit-export-targets",),
        ("--skip-reform-validation",),
        ("--skip-out-of-sample-reforms",),
        ("--skip-reform-coverage-smoke",),
        ("--allow-reform-coverage-smoke-failures",),
        ("--skip-demographics",),
        ("--allow-input-mass-drift",),
        ("--allow-ecps-parity-gaps",),
        ("--allow-input-coverage-gaps",),
        ("--allow-qrf-tail-concentration",),
        ("--allow-unaged-dollar-targets",),
        ("--no-age-targets",),
        ("--evidence-release",),
        ("--staging-dir", "staging"),
        ("--staging-run-id", "invented"),
    ],
)
def test_native_options_refuse_every_unconsumed_option(builder, tmp_path, extra):
    with pytest.raises(builder.NativeSurveyReleaseRefusalError) as refused:
        builder._parse_native_release_args(_argv(tmp_path, *extra))
    assert refused.value.code == "NATIVE_RELEASE_UNSUPPORTED_OPTIONS"
    # Diagnostics name the parser destination; --no-age-targets sets age_targets.
    expected = "--age-targets" if extra[0] == "--no-age-targets" else extra[0]
    assert refused.value.diagnostics["options"] == [expected]


@pytest.mark.parametrize(
    "extra",
    [
        ("--allow-gate-failed-base-pool",),
        ("--exact-k", "3"),
        ("--pool-manifest", "pool.json"),
        ("--refit-l2-lambda", "0.1"),
        ("--evidence-failure-owners", "owners.json"),
    ],
)
def test_native_options_refuse_invalid_legacy_combinations(builder, tmp_path, extra):
    # Dense solves reject a refit penalty in the shared parser already.
    with pytest.raises(builder.NativeSurveyReleaseRefusalError) as refused:
        builder._parse_native_release_args(_argv(tmp_path, *extra))
    assert refused.value.code in {
        "NATIVE_RELEASE_OPTIONS_INVALID",
        "NATIVE_RELEASE_UNSUPPORTED_OPTIONS",
    }


@pytest.mark.parametrize(
    "switch",
    [
        "--no-staging",
        "--no-target-materialization-cache",
        "--no-target-frame-checkpoint",
    ],
)
def test_native_options_require_disabled_staging_and_target_caches(
    builder, tmp_path, switch
):
    argv = [value for value in _argv(tmp_path) if value != switch]
    with pytest.raises(builder.NativeSurveyReleaseRefusalError) as refused:
        builder._parse_native_release_args(argv)
    if switch == "--no-staging":
        # The legacy parser keeps its own refusal of silent staging configuration.
        assert refused.value.code in {
            "NATIVE_RELEASE_OPTIONS_INVALID",
            "NATIVE_RELEASE_UNSUPPORTED_OPTIONS",
            "NATIVE_RELEASE_REQUIRED_OPTIONS",
        }
    else:
        assert refused.value.code == "NATIVE_RELEASE_REQUIRED_OPTIONS"
        assert refused.value.diagnostics == {"options": [switch]}


@pytest.mark.parametrize(
    "release_id",
    [
        None,
        "populace-us-",
        "populace-uk-2024-native",
        "populace-us-2024-evidence-native",
        "populace-us-../../escape",
        "populace-us-2024/native",
        "populace-us-2024 native",
    ],
)
def test_native_options_require_explicit_safe_release_id(builder, tmp_path, release_id):
    argv = _argv(tmp_path)
    index = argv.index("--release-id")
    del argv[index : index + 2]
    if release_id is not None:
        argv += ["--release-id", release_id]
    with pytest.raises(builder.NativeSurveyReleaseRefusalError) as refused:
        builder._parse_native_release_args(argv)
    assert refused.value.code == "NATIVE_RELEASE_ID"


@pytest.mark.parametrize("pin", [None, "A" * 64, "a" * 63, "not-a-digest"])
def test_native_options_require_ledger_facts_pin(builder, tmp_path, pin):
    argv = _argv(tmp_path)
    index = argv.index("--ledger-facts-sha256")
    del argv[index : index + 2]
    if pin is not None:
        argv += ["--ledger-facts-sha256", pin]
    with pytest.raises(builder.NativeSurveyReleaseRefusalError) as refused:
        builder._parse_native_release_args(argv)
    assert refused.value.code == "NATIVE_RELEASE_LEDGER_PIN"


@pytest.mark.parametrize("argv", ["--out o", b"--out", ("--out", 3), None])
def test_native_options_require_a_string_sequence(builder, argv):
    with pytest.raises(builder.NativeSurveyReleaseRefusalError) as refused:
        builder._parse_native_release_args(argv)
    assert refused.value.code == "NATIVE_RELEASE_ARGV"


def test_consumer_admission_binds_exact_identity(builder, monkeypatch):
    calls = []
    spec = _declaration(_parent())
    engine = _engine(spec)
    constructors = _invented_consumer(builder, monkeypatch, calls=calls)
    manifest = _manifest(builder, engine, constructors)
    admitted = builder._admit_native_release_consumer(engine, manifest, spec)
    assert admitted.constructors == constructors
    assert admitted.spm == SPM
    assert admitted.identity == json.loads(json.dumps(manifest))
    assert admitted.identity["explicit_spm"] == SPM
    assert admitted.identity["effective_spm"] == EFFECTIVE_SPM
    assert admitted.identity["constructors"]["dataset"]["qualname"] == "InventedDataset"
    assert admitted.identity["adapter"]["qualname"] == "PolicyEngineUSEngine"
    assert set(admitted.identity["distributions"]) == set(
        builder.NATIVE_RELEASE_CONSUMER_DISTRIBUTIONS
    )
    assert len(admitted.identity_sha256) == 64
    assert calls == ["constructors", "effective_spm"]


@pytest.mark.parametrize(
    "field",
    [
        "adapter",
        "export_contract",
        "explicit_spm",
        "constructors",
        "distributions",
        "protocol",
        "period",
        "extra",
    ],
)
def test_static_identity_mismatch_refuses_before_system_construction(
    builder, monkeypatch, field
):
    calls = []
    spec = _declaration(_parent())
    engine = _engine(spec)
    constructors = _invented_consumer(builder, monkeypatch, calls=calls)
    manifest = _manifest(builder, engine, constructors)
    if field == "extra":
        manifest["approved"] = True
    else:
        manifest[field] = {"invented": "different"}
    with pytest.raises(builder.NativeSurveyReleaseRefusalError) as refused:
        builder._admit_native_release_consumer(engine, manifest, spec)
    assert refused.value.code == "NATIVE_RELEASE_CONSUMER_IDENTITY"
    assert calls == ["constructors"]


def test_effective_spm_mismatch_refuses(builder, monkeypatch):
    spec = _declaration(_parent())
    engine = _engine(spec)
    constructors = _invented_consumer(
        builder, monkeypatch, effective={**EFFECTIVE_SPM, "scenario": "other"}
    )
    manifest = _manifest(builder, engine, constructors)
    with pytest.raises(builder.NativeSurveyReleaseRefusalError) as refused:
        builder._admit_native_release_consumer(engine, manifest, spec)
    assert refused.value.code == "NATIVE_RELEASE_CONSUMER_IDENTITY"


def test_unresolvable_effective_spm_refuses(builder, monkeypatch):
    spec = _declaration(_parent())
    engine = _engine(spec)
    constructors = _invented_consumer(builder, monkeypatch)
    manifest = _manifest(builder, engine, constructors)

    def unresolved(engine):
        raise ImportError("installed consumer has no SPM configuration report")

    monkeypatch.setattr(builder, "_native_release_effective_spm", unresolved)
    with pytest.raises(builder.NativeSurveyReleaseRefusalError) as refused:
        builder._admit_native_release_consumer(engine, manifest, spec)
    assert refused.value.code == "NATIVE_RELEASE_CONSUMER_SPM_UNRESOLVED"


@pytest.mark.parametrize(
    "change,code",
    [
        ("engine_type", "NATIVE_RELEASE_CONSUMER_TYPE"),
        ("engine_subclass", "NATIVE_RELEASE_CONSUMER_TYPE"),
        ("declaration_type", "NATIVE_RELEASE_DECLARATION"),
        ("period_string", "NATIVE_RELEASE_PERIOD"),
        ("period_other", "NATIVE_RELEASE_PERIOD"),
        ("id_dtype", "NATIVE_RELEASE_CONSUMER_ID_DTYPE"),
        ("manifest_type", "NATIVE_RELEASE_CONSUMER_MANIFEST"),
        ("manifest_json", "NATIVE_RELEASE_CONSUMER_MANIFEST"),
        ("defaults", "NATIVE_RELEASE_CONSUMER_DEFAULTS"),
        ("engine_spm", "NATIVE_RELEASE_CONSUMER_SPM"),
        ("implicit_spm", "NATIVE_RELEASE_CONSUMER_SPM"),
        ("contract", "NATIVE_RELEASE_CONSUMER_CONTRACT"),
        ("open_contract", "NATIVE_RELEASE_CONSUMER_CONTRACT"),
    ],
)
def test_consumer_admission_refuses_before_constructor_resolution(
    builder, monkeypatch, change, code
):
    calls = []
    parent = _parent()
    spec = _declaration(parent)
    engine = _engine(spec)
    manifest = {"invented": "identity"}
    _invented_consumer(builder, monkeypatch, calls=calls)
    if change == "engine_type":
        engine = SimpleNamespace(_defaults={}, _spm=SPM)
    elif change == "engine_subclass":
        engine = RecordingEngine(contract=spec.export_contract, spm=SPM)
    elif change == "declaration_type":
        spec = SimpleNamespace(**vars(spec)) if hasattr(spec, "__dict__") else {}
    elif change == "period_string":
        spec = _declaration(parent, period="2024")
    elif change == "period_other":
        spec = _declaration(parent, period=2025)
    elif change == "id_dtype":
        spec = _declaration(parent, consumer_id_dtype=None)
    elif change == "manifest_type":
        manifest = [("invented", "identity")]
    elif change == "manifest_json":
        manifest = {"invented": float("nan")}
    elif change == "defaults":
        engine = _engine(spec, defaults={"invented_default": 0})
    elif change == "engine_spm":
        engine = _engine(spec, spm={"geography_kind": "metro"})
    elif change == "implicit_spm":
        spec = _declaration(parent, spm={})
        engine = PolicyEngineUSEngine(contract=spec.export_contract)
    elif change in {"contract", "open_contract"}:
        from dataclasses import replace

        contract = spec.export_contract
        contract = (
            replace(contract, optional=("invented_extra",))
            if change == "contract"
            else replace(contract, closed=False)
        )
        engine = PolicyEngineUSEngine(contract=contract, spm=SPM)
        if change == "open_contract":
            spec = replace(spec, export_contract=contract)
    with pytest.raises(builder.NativeSurveyReleaseRefusalError) as refused:
        builder._admit_native_release_consumer(engine, manifest, spec)
    assert refused.value.code == code
    assert calls == []


def _gate_projection(frame):
    return handoff._project_native_survey_frame(frame, _declaration(frame))


def _registry(*, congressional_district=False):
    metadata = (
        {"congressional_district_geoid": "3601"} if congressional_district else {}
    )
    return TargetRegistry(
        (
            TargetSpec(
                name="invented.derived",
                entity="household",
                measure="derived_target",
                value=300.0,
                source="Invented tiny fixture",
                metadata=metadata,
                hierarchy=CalibrationHierarchy(
                    provider=HierarchyNode("invented", "Invented provider"),
                    category=HierarchyCategory(
                        "invented.derived", "Invented derived", "invented"
                    ),
                    geography=HierarchyGeography(
                        "0100000US", "United States", "national"
                    ),
                    dimensions=(),
                    target=HierarchyNode("invented.derived", "Invented derived"),
                ),
            ),
        ),
        country="us",
    )


def _complete_inventory(monkeypatch):
    # Invented roster: the maintained roster has hundreds of names this tiny
    # fixture does not carry. Only the other gate checks are exercised here.
    monkeypatch.setattr(handoff, "native_survey_input_inventory", lambda frame: [])


def test_input_gate_reports_declared_missing_names_and_counts_only(builder):
    projection = _gate_projection(_parent())
    gate = builder._native_release_input_gate(
        projection, target_specs=_registry().specs
    )
    assert gate["passed"] is False
    assert "NATIVE_RELEASE_INPUT_MISSING" in gate["failures"]
    assert gate["missing_count"] == len(gate["missing"]) > 0
    assert all(type(name) is str for _, name in gate["missing"])
    assert gate["source_signal_verified"] is False
    assert gate["applicability_verified"] is False
    assert gate["spm_status_counts"] == {"INCLUDED": 2, "OUTSIDE": 1, "UNRESOLVED": 1}
    text = json.dumps(gate)
    # Diagnostics never carry row identifiers or cell values.
    for private in ("9007199254740993", "36061", "3.25", "invented-a"):
        assert private not in text


def test_input_gate_passes_complete_profile_with_invented_roster(builder, monkeypatch):
    _complete_inventory(monkeypatch)
    gate = builder._native_release_input_gate(
        _gate_projection(_parent()), target_specs=_registry().specs
    )
    assert gate["passed"] is True and gate["failures"] == []


@pytest.mark.parametrize(
    "change,code",
    [
        ("state", "NATIVE_RELEASE_GEOGRAPHY_STATE"),
        ("district", "NATIVE_RELEASE_GEOGRAPHY_CONGRESSIONAL_DISTRICT"),
        ("missing_role", "NATIVE_RELEASE_EXPORT_PROFILE:H5_COMPLETE_BOOLEAN_SPM_ROLE"),
        ("nullable_count", "NATIVE_RELEASE_EXPORT_PROFILE:H5_UNSUPPORTED_DTYPE"),
        ("scope", "NATIVE_RELEASE_EXPORT_PROFILE:H5_CANONICAL_SPM_SCOPE"),
    ],
)
def test_input_gate_refuses_codec_role_scope_and_geography_gaps(
    builder, monkeypatch, change, code
):
    import pandas as pd

    _complete_inventory(monkeypatch)
    parent = _parent()
    tables = {e: parent.table(e).copy(deep=True) for e in parent.entities}
    specs = _registry().specs
    if change == "state":
        tables["household"] = tables["household"].drop(columns="state_fips")
    elif change == "district":
        specs = _registry(congressional_district=True).specs
    elif change == "missing_role":
        tables["person"].loc[0, ROLE_INPUT] = pd.NA
    elif change == "nullable_count":
        tables["person"]["invented_count"] = pd.array(
            [1, None, 3, 4, 5, 6], dtype="Int64"
        )
    else:
        tables["spm_unit"].loc[0, UNIVERSE_INPUT] = "UNKNOWN"
    frame = Frame(
        tables,
        parent.schema,
        {"household": parent.weights_for("household")},
        parent.strata,
        metadata=parent.metadata,
    )
    gate = builder._native_release_input_gate(
        _gate_projection(frame), target_specs=specs
    )
    assert gate["passed"] is False
    assert code in gate["failures"]


class InventedWriterEngine:
    """Leaf-only formula metadata and a byte writer; never HDF or a country engine."""

    def __init__(self, mutate=None):
        self.writes = []
        self.metadata_checks = 0
        self.mutate = mutate

    def _engine_computed_columns(self, tables, *, period):
        self.metadata_checks += 1
        return set()

    def write_dataset(self, bundle, path, *, period):
        self.writes.append((bundle, Path(path), period))
        Path(path).write_bytes(b"invented-control-flow-only-not-hdf")
        if self.mutate is not None:
            self.mutate()


def _materializer(calls, *, change_input=False, error=None):
    def materialize(base_frame, target_specs, **kwargs):
        calls.append({"base_frame": base_frame, "target_specs": target_specs, **kwargs})
        if error is not None:
            raise error
        tables = {e: base_frame.table(e).copy(deep=True) for e in base_frame.entities}
        tables["household"]["derived_target"] = np.array([20.0, 50.0, 100.0])
        if change_input:
            tables["person"].loc[
                tables["person"].index[0], "employment_income_before_lsr"
            ] = 7.0
        weights = base_frame.weights_for("household")
        target = Frame(
            tables,
            base_frame.schema,
            {"household": Weights(weights.values.copy(), weights.kind)},
            base_frame.strata.copy(deep=True),
            metadata=base_frame.metadata,
        )
        compilation = {
            "dropped_target_names": [],
            "target_frame_checkpoint": {"enabled": False, "status": "disabled"},
        }
        return target, _registry(), compilation

    return materialize


def _prepared_case(builder, monkeypatch, tmp_path, *, dense=True, fit_failures=()):
    parent = _parent()
    options = builder._parse_native_release_args(_argv(tmp_path, dense=dense))
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    calls = []
    monkeypatch.setattr(
        builder, "_load_or_materialize_target_frame", _materializer(calls)
    )
    if fit_failures is not None:
        # The invented registry lacks the maintained critical fiscal targets,
        # so the real fit gate would refuse it; a separate test covers that.
        monkeypatch.setattr(
            builder, "_release_gate_failures", lambda *a, **k: list(fit_failures)
        )
    engine = InventedWriterEngine()

    def read_h5(path):
        return _logical_tables(engine.writes[-1][0]), _period()

    monkeypatch.setattr(readback, "_read_h5", read_h5)
    constructors = builder._NativeReleaseConsumer(
        InventedDataset, InventedMicrosimulation, InventedSystem
    )
    kwargs = dict(
        target_specs=_registry().specs,
        options=options,
        engine=engine,
        constructors=constructors,
        spm=SPM,
        parent_reference="invented-native-parent-reference",
        calibration_specification=b'{"invented":true}',
        release_dir=release_dir,
    )
    return parent, kwargs, calls, engine


@pytest.mark.parametrize("dense", [True, False])
def test_private_composition_materializes_with_admitted_consumer_and_writes_verified_h5(
    builder, monkeypatch, tmp_path, dense
):
    reached = []
    _forbid(
        monkeypatch,
        builder,
        (
            "_download_base_h5",
            "_load_frame",
            "run_source_stage",
            "_build_manifests",
            "_write_release_calibration_diagnostics",
            "_copy_base_h5_for_local_audit",
            "StagingTelemetry",
            "_materialize_target_frame",
        ),
        reached,
    )
    parent, kwargs, calls, engine = _prepared_case(
        builder, monkeypatch, tmp_path, dense=dense
    )
    before = {e: parent.table(e).copy(deep=True) for e in parent.entities}
    initial = parent.weights_for("household").values.copy()
    prepared = builder._run_prepared_native_fiscal_release(parent, **kwargs)
    assert reached == []
    (call,) = calls
    assert call["base_frame"] is parent
    assert call["dataset_cls"] is InventedDataset
    assert call["microsimulation_cls"] is InventedMicrosimulation
    assert call["system_factory"] is InventedSystem
    assert call["formula_metadata"] is engine
    assert call["spm"] == SPM and call["spm"] is not kwargs["spm"]
    assert call["target_frame_checkpoint_path"] is None
    assert call["target_materialization_cache_dir"] is None
    assert (
        prepared.dataset_path
        == kwargs["release_dir"] / builder.NATIVE_RELEASE_DATASET_FILENAME
    )
    assert prepared.dataset_path.read_bytes() == b"invented-control-flow-only-not-hdf"
    assert prepared.h5_receipt.binding == prepared.attachment.binding
    assert prepared.h5_receipt.release_eligible is False
    assert prepared.fit_gate_failures == ()
    ((bundle, path, period),) = engine.writes
    assert bundle is not parent and period == 2024
    assert "derived_target" not in bundle.table("household")
    diagnostics = json.loads(prepared.diagnostics_path.read_text())
    assert diagnostics["build"]["release_eligible"] is False
    assert diagnostics["build"]["source_signal_gates_evaluated"] is False
    for entity, table in before.items():
        assert parent.table(entity).equals(table)
    np.testing.assert_array_equal(parent.weights_for("household").values, initial)
    assert sorted(p.name for p in kwargs["release_dir"].iterdir()) == sorted(
        [
            builder.NATIVE_RELEASE_DATASET_FILENAME,
            builder.NATIVE_RELEASE_DIAGNOSTICS_FILENAME,
        ]
    )


def test_real_fit_gate_refuses_invented_targets_before_any_h5(
    builder, monkeypatch, tmp_path
):
    parent, kwargs, _, engine = _prepared_case(
        builder, monkeypatch, tmp_path, fit_failures=None
    )
    with pytest.raises(builder.NativeSurveyReleaseRefusalError) as refused:
        builder._run_prepared_native_fiscal_release(parent, **kwargs)
    assert refused.value.code == "NATIVE_RELEASE_FIT_GATES"
    assert refused.value.diagnostics["failure_count"] > 0
    assert engine.writes == []
    assert not (
        kwargs["release_dir"] / builder.NATIVE_RELEASE_DATASET_FILENAME
    ).exists()
    diagnostics = json.loads(
        (
            kwargs["release_dir"] / builder.NATIVE_RELEASE_DIAGNOSTICS_FILENAME
        ).read_text()
    )
    assert diagnostics["build"]["fit_gate_failures"]


def test_changed_target_inputs_refuse_before_solve(builder, monkeypatch, tmp_path):
    parent, kwargs, _, engine = _prepared_case(builder, monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(
        builder,
        "_load_or_materialize_target_frame",
        _materializer(calls, change_input=True),
    )
    monkeypatch.setattr(
        builder,
        "_calibrate_fiscal_support",
        lambda *a, **k: pytest.fail("solve reached"),
    )
    with pytest.raises(ValueError, match="NATIVE_CALIBRATION_TARGET_INPUT"):
        builder._run_prepared_native_fiscal_release(parent, **kwargs)
    assert engine.writes == []


def test_materializer_errors_become_codes_without_private_examples(
    builder, monkeypatch, tmp_path
):
    parent, kwargs, _, engine = _prepared_case(builder, monkeypatch, tmp_path)
    monkeypatch.setattr(
        builder,
        "_load_or_materialize_target_frame",
        _materializer(
            [], error=ValueError("ambiguous ids examples: [9007199254740993]")
        ),
    )
    with pytest.raises(builder.NativeSurveyReleaseRefusalError) as refused:
        builder._run_prepared_native_fiscal_release(parent, **kwargs)
    assert str(refused.value) == "NATIVE_RELEASE_TARGET_MATERIALIZATION"
    assert refused.value.__cause__ is None and refused.value.__suppress_context__
    assert engine.writes == []


@pytest.mark.parametrize("problem", ["nonempty", "missing", "switch"])
def test_private_composition_requires_fresh_directory_and_native_options(
    builder, monkeypatch, tmp_path, problem
):
    parent, kwargs, calls, _ = _prepared_case(builder, monkeypatch, tmp_path)
    if problem == "nonempty":
        (kwargs["release_dir"] / "stale.json").write_text("{}")
        code = "NATIVE_RELEASE_DIRECTORY"
    elif problem == "missing":
        kwargs["release_dir"] = tmp_path / "absent"
        code = "NATIVE_RELEASE_DIRECTORY"
    else:
        kwargs["options"].args.no_target_frame_checkpoint = False
        code = "NATIVE_RELEASE_OPTIONS"
    with pytest.raises(builder.NativeSurveyReleaseRefusalError) as refused:
        builder._run_prepared_native_fiscal_release(parent, **kwargs)
    assert refused.value.code == code
    assert calls == []


def test_writer_mutation_of_parent_cannot_produce_a_receipt(
    builder, monkeypatch, tmp_path
):
    parent, kwargs, _, _ = _prepared_case(builder, monkeypatch, tmp_path)

    def mutate():
        parent.person.loc[parent.person.index[0], "employment_income_before_lsr"] = 11.0

    engine = InventedWriterEngine(mutate=mutate)
    monkeypatch.setattr(
        readback,
        "_read_h5",
        lambda path: (_logical_tables(engine.writes[-1][0]), _period()),
    )
    kwargs["engine"] = engine
    with pytest.raises(readback.PolicyEngineH5ReadbackError, match="INPUT_CHANGED"):
        builder._run_prepared_native_fiscal_release(parent, **kwargs)


def test_native_json_writer_is_create_only(builder, tmp_path):
    path = tmp_path / "manifest.json"
    digest = builder._write_native_release_json(path, {"release_eligible": False})
    assert json.loads(path.read_text()) == {"release_eligible": False}
    assert len(digest) == 64
    assert not (tmp_path / "manifest.json.tmp").exists()
    with pytest.raises(builder.NativeSurveyReleaseRefusalError) as refused:
        builder._write_native_release_json(path, {"release_eligible": True})
    assert refused.value.code == "NATIVE_RELEASE_OUTPUT_EXISTS"
    assert json.loads(path.read_text()) == {"release_eligible": False}
    with pytest.raises(ValueError):
        builder._write_native_release_json(tmp_path / "nan.json", {"x": float("nan")})
    assert not (tmp_path / "nan.json").exists()


def _final_case():
    source = _parent()
    spec = _declaration(source)
    projection = handoff._project_native_survey_frame(source, spec)
    population = SimpleNamespace(frame=source)
    owner = SimpleNamespace(digest="invented-owner", population=population)
    expected = (
        handoff._projection_spec_bytes(spec),
        handoff._projection_stamp(projection.frame),
        handoff._json(projection.report),
        "c" * 64,
        "c" * 64,
    )
    final = SimpleNamespace(digest="invented-owner", population=population)
    return final, owner, projection, spec, expected


@pytest.mark.parametrize(
    "change,code",
    [
        ("digest", "NATIVE_RELEASE_OWNER_CHANGED"),
        ("population", "NATIVE_RELEASE_OWNER_CHANGED"),
        ("source", "NATIVE_RELEASE_OWNER_CHANGED"),
        ("declaration", "NATIVE_RELEASE_DECLARATION_CHANGED"),
        ("projection_cell", "NATIVE_RELEASE_PROJECTION_CHANGED"),
        ("report", "NATIVE_RELEASE_PROJECTION_CHANGED"),
        ("consumer", "NATIVE_RELEASE_CONSUMER_CHANGED"),
    ],
)
def test_final_owner_io_cannot_change_retained_inputs(builder, change, code):
    from dataclasses import replace

    final, owner, projection, spec, expected = _final_case()
    builder._validate_native_release_after_owner_io(
        final, owner=owner, projection=projection, declaration=spec, expected=expected
    )
    if change == "digest":
        final.digest = "changed"
    elif change == "population":
        final.population = SimpleNamespace(frame=projection.source_frame)
    elif change == "source":
        final.population.frame = projection.frame
    elif change == "declaration":
        spec = replace(spec, consumer_identity="changed")
    elif change == "projection_cell":
        projection.frame.person.loc[projection.frame.person.index[0], ROLE_INPUT] = (
            False
        )
    elif change == "report":
        projection.report["release_eligible"] = True
    else:
        expected = (*expected[:4], "d" * 64)
    with pytest.raises(builder.NativeSurveyReleaseRefusalError) as refused:
        builder._validate_native_release_after_owner_io(
            final,
            owner=owner,
            projection=projection,
            declaration=spec,
            expected=expected,
        )
    assert refused.value.code == code


class StandInOwner:
    """Ordering/wiring stand-in for the owner check; never an issued owner."""

    def __init__(self, run, frame):
        self.run = run
        self.population = SimpleNamespace(frame=frame)
        self.calls = 0

    def check(self, run):
        if run is not self.run:
            raise ValueError("UNISSUED_RUN")
        self.calls += 1
        return SimpleNamespace(digest="d" * 64, population=self.population)


def _stand_in(builder, monkeypatch, tmp_path, *, events=None):
    source = _parent()
    run = object()
    owner = StandInOwner(run, source)
    events = [] if events is None else events

    def check(candidate):
        events.append("owner_check")
        return owner.check(candidate)

    monkeypatch.setattr(native_owner, "check_survey_enrichment_run", check)
    spec = _declaration(source)
    engine = _engine(spec)
    constructors = _invented_consumer(builder, monkeypatch)
    manifest = _manifest(builder, engine, constructors)

    def project(candidate, *, declaration, consumer):
        events.append("projection")
        assert candidate is run and consumer is engine
        result = handoff._project_native_survey_frame(source, declaration)
        result.report["consumer_representation_compatible"] = True
        result.report["owner_receipt_sha256"] = "d" * 64
        return result

    monkeypatch.setattr(handoff, "prepare_native_survey_engine_input", project)
    ledger = SimpleNamespace(provenance=lambda: {"facts_sha256": LEDGER_PIN})

    def compile_targets(args, *, congressional_district_vintage_crosswalk):
        events.append("targets")
        return ledger, _registry(), [], SimpleNamespace(passed=True)

    monkeypatch.setattr(
        builder, "_compile_fiscal_release_target_registry", compile_targets
    )
    return run, spec, engine, manifest, owner


def test_option_refusal_precedes_consumer_projection_and_output(
    builder, monkeypatch, tmp_path
):
    events = []
    run, spec, engine, manifest, _ = _stand_in(
        builder, monkeypatch, tmp_path, events=events
    )
    reached = []
    _forbid(monkeypatch, builder, ("_admit_native_release_consumer",), reached)
    with pytest.raises(builder.NativeSurveyReleaseRefusalError) as refused:
        builder.build_native_survey_release(
            run,
            argv=_argv(tmp_path, "--base-h5", "legacy.h5"),
            declaration=spec,
            engine=engine,
            consumer_manifest=manifest,
        )
    assert refused.value.code == "NATIVE_RELEASE_UNSUPPORTED_OPTIONS"
    assert events == ["owner_check"] and reached == []
    assert not (tmp_path / "out").exists()


def test_existing_release_directory_refuses_before_consumer(
    builder, monkeypatch, tmp_path
):
    run, spec, engine, manifest, _ = _stand_in(builder, monkeypatch, tmp_path)
    existing = tmp_path / "out" / builder.NATIVE_RELEASE_DIRECTORY / RELEASE_ID
    existing.mkdir(parents=True)
    (existing / "prior.json").write_text("{}")
    reached = []
    _forbid(monkeypatch, builder, ("_admit_native_release_consumer",), reached)
    with pytest.raises(builder.NativeSurveyReleaseRefusalError) as refused:
        builder.build_native_survey_release(
            run,
            argv=_argv(tmp_path),
            declaration=spec,
            engine=engine,
            consumer_manifest=manifest,
        )
    assert refused.value.code == "NATIVE_RELEASE_DIRECTORY_EXISTS"
    assert reached == [] and (existing / "prior.json").read_text() == "{}"


def test_consumer_mismatch_refuses_before_projection(builder, monkeypatch, tmp_path):
    events = []
    run, spec, engine, manifest, _ = _stand_in(
        builder, monkeypatch, tmp_path, events=events
    )
    manifest = {**manifest, "period": 2025}
    with pytest.raises(builder.NativeSurveyReleaseRefusalError) as refused:
        builder.build_native_survey_release(
            run,
            argv=_argv(tmp_path),
            declaration=spec,
            engine=engine,
            consumer_manifest=manifest,
        )
    assert refused.value.code == "NATIVE_RELEASE_CONSUMER_IDENTITY"
    assert events == ["owner_check"]
    assert not (tmp_path / "out").exists()


def test_input_gate_refuses_before_output_directory_and_model_construction(
    builder, monkeypatch, tmp_path
):
    events = []
    run, spec, engine, manifest, _ = _stand_in(
        builder, monkeypatch, tmp_path, events=events
    )
    reached = []
    _forbid(
        monkeypatch,
        builder,
        ("_load_or_materialize_target_frame", "_calibrate_native_input_frame"),
        reached,
    )
    with pytest.raises(builder.NativeSurveyReleaseRefusalError) as refused:
        builder.build_native_survey_release(
            run,
            argv=_argv(tmp_path),
            declaration=spec,
            engine=engine,
            consumer_manifest=manifest,
        )
    assert refused.value.code == "NATIVE_RELEASE_INPUT_GATE"
    assert refused.value.diagnostics["missing_count"] > 0
    assert events == ["owner_check", "projection", "targets"]
    assert reached == [] and not (tmp_path / "out").exists()


def test_public_entry_call_order_with_stand_in_owner_is_wiring_only(
    builder, monkeypatch, tmp_path
):
    """Wiring only: a stand-in owner and invented roster, never a positive build."""
    events = []
    run, spec, engine, manifest, owner = _stand_in(
        builder, monkeypatch, tmp_path, events=events
    )
    _complete_inventory(monkeypatch)
    writer = InventedWriterEngine()

    def write_dataset(self, bundle, path, period):
        events.append("h5_write")
        writer.write_dataset(bundle, path, period=period)

    monkeypatch.setattr(PolicyEngineUSEngine, "write_dataset", write_dataset)
    monkeypatch.setattr(
        PolicyEngineUSEngine,
        "_engine_computed_columns",
        lambda self, tables, *, period: set(),
    )
    monkeypatch.setattr(
        readback,
        "_read_h5",
        lambda path: (_logical_tables(writer.writes[-1][0]), _period()),
    )
    materialized = []
    monkeypatch.setattr(
        builder, "_load_or_materialize_target_frame", _materializer(materialized)
    )
    monkeypatch.setattr(builder, "_release_gate_failures", lambda *a, **k: [])
    original_json = builder._write_native_release_json

    def record_manifest(path, payload):
        events.append("manifest")
        return original_json(path, payload)

    monkeypatch.setattr(builder, "_write_native_release_json", record_manifest)
    result = builder.build_native_survey_release(
        run,
        argv=_argv(tmp_path),
        declaration=spec,
        engine=engine,
        consumer_manifest=manifest,
    )
    assert events == [
        "owner_check",
        "projection",
        "targets",
        "h5_write",
        "owner_check",
        "manifest",
    ]
    assert owner.calls == 2
    (call,) = materialized
    assert call["dataset_cls"] is InventedDataset
    assert call["spm"] == SPM
    assert result.release_eligible is False
    assert (
        result.release_dir
        == (tmp_path / "out" / builder.NATIVE_RELEASE_DIRECTORY / RELEASE_ID).resolve()
    )
    document = json.loads(result.manifest_path.read_text())
    assert document["release_eligible"] is False
    assert document["certified"] is False
    assert document["consumer"]["root_admitted"] is False
    assert document["outstanding_qualifications"] == list(
        builder.NATIVE_RELEASE_OUTSTANDING_QUALIFICATIONS
    )
    assert document["survey_poverty_role"].startswith("comparison_only")
    assert document["dataset"]["sha256"] == result.dataset_sha256
    assert document["owner"]["receipt_sha256"] == "d" * 64
    assert not (result.release_dir / "release_manifest.json").exists()


def test_final_owner_failure_leaves_no_manifest(builder, monkeypatch, tmp_path):
    """Wiring only: the stand-in owner fails its second check after the H5."""
    run, spec, engine, manifest, owner = _stand_in(builder, monkeypatch, tmp_path)
    _complete_inventory(monkeypatch)
    writer = InventedWriterEngine()
    monkeypatch.setattr(
        PolicyEngineUSEngine,
        "write_dataset",
        lambda self, bundle, path, period: writer.write_dataset(
            bundle, path, period=period
        ),
    )
    monkeypatch.setattr(
        PolicyEngineUSEngine,
        "_engine_computed_columns",
        lambda self, tables, *, period: set(),
    )
    monkeypatch.setattr(
        readback,
        "_read_h5",
        lambda path: (_logical_tables(writer.writes[-1][0]), _period()),
    )
    monkeypatch.setattr(builder, "_load_or_materialize_target_frame", _materializer([]))
    monkeypatch.setattr(builder, "_release_gate_failures", lambda *a, **k: [])
    original = owner.check

    def lose_owner(candidate):
        result = original(candidate)
        if owner.calls == 2:
            raise ValueError("RUN_CHANGED")
        return result

    monkeypatch.setattr(native_owner, "check_survey_enrichment_run", lose_owner)
    with pytest.raises(ValueError, match="RUN_CHANGED"):
        builder.build_native_survey_release(
            run,
            argv=_argv(tmp_path),
            declaration=spec,
            engine=engine,
            consumer_manifest=manifest,
        )
    release_dir = tmp_path / "out" / builder.NATIVE_RELEASE_DIRECTORY / RELEASE_ID
    assert (release_dir / builder.NATIVE_RELEASE_DATASET_FILENAME).exists()
    assert not (release_dir / builder.NATIVE_RELEASE_MANIFEST_FILENAME).exists()
