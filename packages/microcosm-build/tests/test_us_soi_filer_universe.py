"""Invented small supports exercising the installed US model, without reforms."""

import importlib.util
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import soi_return_universe as universe
from microcosm.calibrate import TargetSpec
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def _builder():
    path = (
        Path(__file__).resolve().parents[3] / "tools/build_us_fiscal_refresh_release.py"
    )
    spec = importlib.util.spec_from_file_location("soi_filer_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame():
    ids = np.arange(1, 5, dtype=np.int64)
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": ids,
                **{f"person_{entity}_id": ids for entity in US_SCHEMA.group_entities},
                "age": [30] * 4,
                "employment_income_before_lsr": [5000.0, 5000.0, 5000.0, 50000.0],
                "charitable_cash_donations": [0.0, 0.0, 0.0, 30000.0],
            }
        ),
        **{
            entity: pd.DataFrame({f"{entity}_id": ids})
            for entity in US_SCHEMA.group_entities
        },
    }
    # A nonfiler and a credit filer share the first household, but remain
    # separate tax units. Masking after collapse would retain both incomes.
    tables["person"]["person_household_id"] = [1, 1, 2, 3]
    tables["household"] = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "state_fips": [48, 48, 12],
            "congressional_district_geoid": [4801, 4802, 1201],
        }
    )
    tables["tax_unit"]["would_file_if_eligible_for_refundable_credit"] = [
        False,
        True,
        False,
        False,
    ]
    tables["tax_unit"]["would_file_taxes_voluntarily"] = [False, False, True, False]
    tables["tax_unit"]["takes_up_eitc"] = [True] * 4
    return Frame(
        tables, US_SCHEMA, {"household": Weights(np.ones(3), WeightKind.DESIGN)}
    )


def _target(name="count", source="count", **details):
    return TargetSpec(
        name=name,
        entity="household",
        measure=name,
        value=1.0,
        source="invented IRS-shaped engineering fixture",
        family="irs_soi",
        metadata={
            "variable": source,
            "source_variable": source,
            "agi_lower_bound": "-inf",
            "agi_upper_bound": "inf",
            "filing_status": "All",
            "soi_return_universe": "all_returns",
            "source_period": "2023",
            "target_period": "2024",
            **details,
        },
    )


def _targets():
    targets = []
    for region, geographic in (
        ("national", {}),
        ("state", {"state_fips": "48"}),
        ("cd", {"state_fips": "48", "congressional_district_geoid": "4801"}),
    ):
        for kind, source, extra in (
            ("count", "count", {}),
            ("amount", "adjusted_gross_income", {}),
            ("indicator", "adjusted_gross_income", {"measure_mode": "indicator_sum"}),
        ):
            targets.append(_target(f"{region}_{kind}", source, **geographic, **extra))
    # The explicit itemized universe must be sufficient without itemized_only.
    targets.extend(
        (
            _target("itemized_count", soi_return_universe="itemized_returns"),
            _target(
                "itemized_amount",
                "adjusted_gross_income",
                soi_return_universe="itemized_returns",
            ),
            _target("legacy_itemized_count", itemized_only="true"),
            _target(
                "restricted_count",
                agi_lower_bound="5000",
                agi_upper_bound="50000",
                filing_status="Single",
            ),
            _target("taxable_count", taxable_only="true"),
        )
    )
    return tuple(targets)


@pytest.fixture(scope="module")
def actual():
    import policyengine_us as peus

    builder = _builder()
    frame = _frame()
    builder._assert_no_formula_owned_columns(frame)
    simulation = peus.Microsimulation(dataset=builder._dataset_from_frame(frame))
    series = {
        name: simulation.calc(name, period=2024, map_to="tax_unit")
        for name in (
            "tax_unit_is_filer",
            "tax_unit_is_required_to_file",
            "tax_unit_itemizes",
            "adjusted_gross_income",
            "income_tax",
            "taxable_income",
        )
    }
    targets = _targets()
    result, registry, compilation = builder._materialize_target_frame(frame, targets)
    yield builder, frame, simulation, series, targets, result, registry, compilation
    builder.release_engine_simulation(simulation)


@pytest.mark.requires_us
def test_actual_filer_decisions_precede_household_collapse(actual):
    _, frame, simulation, series, targets, result, registry, compilation = actual
    assert series["tax_unit_is_filer"].tolist() == [False, True, True, True]
    assert series["tax_unit_is_required_to_file"].tolist() == [
        False,
        False,
        False,
        True,
    ]
    assert series["tax_unit_itemizes"].tolist() == [False, False, False, True]
    expected = {
        "national_count": [1.0, 1.0, 1.0],
        "national_amount": [5000.0, 5000.0, 50000.0],
        "national_indicator": [1.0, 1.0, 1.0],
        "state_count": [1.0, 1.0, 0.0],
        "state_amount": [5000.0, 5000.0, 0.0],
        "state_indicator": [1.0, 1.0, 0.0],
        "cd_count": [1.0, 0.0, 0.0],
        "cd_amount": [5000.0, 0.0, 0.0],
        "cd_indicator": [1.0, 0.0, 0.0],
        "itemized_count": [0.0, 0.0, 1.0],
        "itemized_amount": [0.0, 0.0, 50000.0],
        "legacy_itemized_count": [0.0, 0.0, 1.0],
        "restricted_count": [1.0, 1.0, 0.0],
        "taxable_count": [0.0, 0.0, 1.0],
    }
    for name, vector in expected.items():
        # Unweighted internal design columns, not manual population aggregation.
        assert result.table("household")[name].tolist() == vector
    assert len(registry) == len(targets)
    assert compilation["dropped_target_names"] == []
    for entity in frame.entities:
        if entity != "household":
            pd.testing.assert_frame_equal(frame.table(entity), result.table(entity))
    pd.testing.assert_frame_equal(
        frame.table("household"),
        result.table("household")[frame.table("household").columns],
    )
    assert result.weights_for("household") is frame.weights_for("household")
    # Actual MicroSeries operations retain the engine's internally mapped weights.
    filer, agi = series["tax_unit_is_filer"], series["adjusted_gross_income"]
    assert float(filer.sum()) == 3.0
    assert float((agi * filer).sum()) == 60000.0
    # Numeric geography must stay at household level: Core's default mapping
    # between group entities allocates numeric quantities across their members.
    household_agi = simulation.calc(
        "adjusted_gross_income", period=2024, map_to="household"
    )
    household_state = simulation.calc("state_fips", period=2024, map_to="household")
    assert float((household_agi * (household_state == 48)).sum()) == 15000.0


@pytest.mark.requires_us
def test_actual_receipt_survives_cold_and_warm_checkpoint(
    actual, tmp_path, monkeypatch
):
    builder, frame, _, _, targets, result, _, compilation = actual
    path = tmp_path / "target.h5"
    identity = {
        "materializer_version": builder.TARGET_FRAME_CHECKPOINT_MATERIALIZER_VERSION
    }
    # Exercise the normal cold orchestration/write path with the already
    # computed actual result; no second baseline simulation is necessary.
    monkeypatch.setattr(
        builder,
        "_materialize_target_frame",
        lambda *a, **k: (result, None, compilation),
    )
    cold = builder._load_or_materialize_target_frame(
        frame,
        targets,
        target_frame_checkpoint_path=path,
        target_frame_checkpoint_identity=identity,
    )
    receipt = cold[2]["soi_return_universe"]
    monkeypatch.setattr(
        builder,
        "_materialize_target_frame",
        lambda *a, **k: pytest.fail("warm path recalculated"),
    )
    warm = builder._load_or_materialize_target_frame(
        frame,
        targets,
        target_frame_checkpoint_path=path,
        target_frame_checkpoint_identity=identity,
    )
    assert warm[2]["soi_return_universe"] == receipt
    assert warm[2]["target_frame_checkpoint"]["status"] == "hit"
    assert receipt["contract"]["dependent_return_completeness"] == "not_established"
    assert (
        receipt["model"]["scope"]
        == "direct_filing_declarations_not_transitive_model_closure"
    )
    for name in universe.FILING_PROPENSITY_INPUTS:
        assert (
            receipt["filing_input_origins"][name]["origin"] == "supplied_frame_column"
        )
    pd.testing.assert_frame_equal(warm[0].table("household"), result.table("household"))
    assert (
        builder._read_target_frame_checkpoint(
            path, identity={"materializer_version": 12}, target_specs=targets
        )
        is None
    )
    with h5py.File(path, "r+") as h5:
        h5.attrs["compilation_json"] = "{}"
    with pytest.raises(ValueError, match="missing cached filer receipt"):
        builder._read_target_frame_checkpoint(
            path, identity=identity, target_specs=targets
        )


@pytest.mark.requires_us
def test_actual_default_origins_are_explicit_model_assumptions(actual):
    from policyengine_us import CountryTaxBenefitSystem, Microsimulation

    builder, frame, _, _, targets, *_ = actual
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["tax_unit"] = tables["tax_unit"].drop(
        columns=list(universe.FILING_PROPENSITY_INPUTS)
    )
    defaults = Frame(tables, US_SCHEMA, {"household": frame.weights_for("household")})
    sim = Microsimulation(dataset=builder._dataset_from_frame(defaults))
    contract = universe.soi_target_universe_contract(targets, period=2024)
    _, receipt = universe.materialize_soi_filer_universe(
        defaults, sim, CountryTaxBenefitSystem(), contract=contract
    )
    universe.validate_cached_soi_filer_receipt(receipt, defaults, contract=contract)
    assert {
        name: origin["value"]
        for name, origin in receipt["filing_input_origins"].items()
    } == {
        "would_file_if_eligible_for_refundable_credit": True,
        "would_file_taxes_voluntarily": False,
    }
    assert all(
        origin["interpretation"] == "model_assumption_not_observed_response"
        for origin in receipt["filing_input_origins"].values()
    )
    builder.release_engine_simulation(sim)


@pytest.mark.parametrize(
    "changes",
    [
        {"soi_return_universe": "unknown"},
        {
            "soi_return_universe": "returns_excluding_dependents",
            "ledger_layout_record_set_id": "all_returns",
        },
        {
            "soi_return_universe": "returns_excluding_dependents",
            "ledger_layout_record_set_id": "itemized_all_returns",
        },
        {"soi_return_universe": "itemized_returns", "itemized_only": "false"},
        {"ledger_layout_record_set_id": "all_returns_excluding_dependents"},
        {"ledger_layout_record_set_id": "itemized_all_returns"},
        {"ledger_domain": "individual_returns_with_itemized_deductions"},
        {"itemized_only": "1"},
        {"target_period": "2023"},
    ],
)
def test_unsupported_or_contradictory_universe_refuses_before_model(
    changes, monkeypatch
):
    builder = _builder()
    monkeypatch.setattr(
        builder,
        "_dataset_from_frame",
        lambda *a, **k: pytest.fail("constructed model dataset"),
    )
    with pytest.raises(ValueError, match="SOI return universe"):
        builder._materialize_target_frame(_frame(), (_target(**changes),))


def test_missing_universe_is_not_defaulted_and_interest_bridge_is_preserved():
    target = _target()
    missing = replace(
        target,
        metadata={
            k: v for k, v in target.metadata.items() if k != "soi_return_universe"
        },
    )
    with pytest.raises(ValueError, match="missing source universe"):
        universe.soi_target_universe_contract((missing,), period=2024)
    target = _target(soi_return_universe_bridge="returns_excluding_dependents")
    row = universe.soi_target_universe_contract((target,), period=2024)["targets"][0]
    assert row["source_universe"] == "all_returns"
    assert row["source_period"] == "2023"
    assert (
        row["target_value_universe_bridge"]
        == target.metadata["soi_return_universe_bridge"]
    )


@pytest.mark.parametrize("values", [[0, 1, 0, 1], [False, None, False, True]])
def test_invalid_propensity_input_refuses_without_boolean_coercion(values):
    frame = _frame()
    frame.table("tax_unit")[universe.FILING_PROPENSITY_INPUTS[0]] = values
    with pytest.raises(ValueError, match="nonnullable boolean"):
        universe._input_origin(frame, universe.FILING_PROPENSITY_INPUTS[0], True)


@pytest.mark.parametrize(
    "variable",
    [
        None,
        SimpleNamespace(
            entity=SimpleNamespace(key="person"),
            value_type=bool,
            definition_period="year",
        ),
        SimpleNamespace(
            entity=SimpleNamespace(key="tax_unit"),
            value_type=float,
            definition_period="year",
        ),
        SimpleNamespace(
            entity=SimpleNamespace(key="tax_unit"),
            value_type=bool,
            definition_period="month",
        ),
    ],
)
def test_missing_or_incompatible_engine_predicate_refuses(variable):
    with pytest.raises(ValueError, match="engine"):
        universe._bool_variable(
            SimpleNamespace(variables={"tax_unit_is_filer": variable}),
            "tax_unit_is_filer",
        )


@pytest.mark.parametrize(
    "vector",
    [
        np.array([0, 1]),
        np.array([True]),
        np.array([[True, False]]),
        np.array([None, True]),
    ],
)
def test_invalid_engine_output_shape_or_type_refuses(vector):
    with pytest.raises(ValueError, match="invalid boolean vector"):
        universe._bool_vector(vector, 2, "tax_unit_is_filer")


@pytest.mark.parametrize(
    "mutation",
    [
        "contract",
        "ids",
        "origin",
        "model",
        "core",
        "scope",
        "declaration",
        "path",
        "mask",
    ],
)
@pytest.mark.requires_us
def test_cached_receipt_refuses_changed_provenance(actual, mutation):
    _, _, _, _, targets, result, _, compilation = actual
    receipt = deepcopy(compilation["soi_return_universe"])
    if mutation == "contract":
        receipt["contract"]["model_period"] = 2023
    elif mutation == "ids":
        receipt["tax_unit_ids_sha256"] = "0" * 64
    elif mutation == "origin":
        receipt["filing_input_origins"][universe.FILING_PROPENSITY_INPUTS[0]][
            "sha256"
        ] = "0" * 64
    elif mutation == "model":
        receipt["model"]["version"] = "unreviewed"
    elif mutation == "core":
        receipt["model"]["core_version"] = "unreviewed"
    elif mutation == "scope":
        receipt["provenance_scope"] = "source_authenticated"
    elif mutation == "declaration":
        receipt["model"]["direct_variable_declarations"]["tax_unit_is_filer"][
            "sha256"
        ] = "0" * 64
    elif mutation == "path":
        receipt["model"]["direct_variable_declarations"]["tax_unit_is_filer"][
            "path"
        ] = "../../outside.py"
    else:
        receipt["actual_mask_sha256"]["tax_unit_is_filer"] = "bad"
    with pytest.raises(ValueError, match="SOI return universe"):
        universe.validate_cached_soi_filer_receipt(
            receipt,
            result,
            contract=universe.soi_target_universe_contract(targets, period=2024),
        )


@pytest.mark.requires_us
def test_formula_owned_filer_input_is_rejected(actual):
    builder, frame, *_ = actual
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["tax_unit"]["tax_unit_is_filer"] = [True] * 4
    injected = Frame(tables, US_SCHEMA, {"household": frame.weights_for("household")})
    with pytest.raises(ValueError, match="tax_unit_is_filer"):
        builder._assert_no_formula_owned_columns(injected)


def _table_4_3_ledger_facts(*, include_unsupported=True, include_return_count=False):
    """Invented Ledger facts with real source-layout declarations."""
    path = Path(__file__).with_name("test_us_fiscal_targets.py")
    spec = importlib.util.spec_from_file_location("soi_fact_fixture", path)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    facts = [
        *fixtures.packaged_reference_facts(),
        fixtures._dynamic_ledger_fact(
            source_record_id=(
                "irs_soi.ty2023.table_1_2.all_returns.all.wages_salaries_amount"
            ),
            source_name="irs_soi",
            measure_id="wages_salaries_amount",
            value=100.0,
            period_value=2023,
            layout_record_set_id="irs_soi.ty2023.table_1_2.all_returns",
            dimensions={"income_range": "all", "filing_status": "all"},
        ),
    ]
    if include_unsupported:
        measures = ["taxable_interest_amount"]
        if include_return_count:
            measures.append("taxable_interest_returns")
        for measure in measures:
            facts.append(
                fixtures._soi_taxable_interest_fact(
                    2023,
                    source_record_id=(
                        "irs_soi.ty2023.table_4_3.all_returns_excluding_dependents.all."
                        + measure
                    ),
                    value=100.0,
                    measure_id=measure,
                    layout_record_set_id=(
                        "irs_soi.ty2023.table_4_3.all_returns_excluding_dependents"
                    ),
                )
            )
    return facts


def _compiled_table_4_3_targets():
    """Use the real Ledger compiler on explicit invented Table 4.3 facts."""
    from microcosm.build.us_runtime import compile_us_fiscal_target_registry

    registry = compile_us_fiscal_target_registry(
        _table_4_3_ledger_facts(),
        target_period=2024,
        allow_unaged_dollar_targets=True,
    )
    # The public compiler requires reference facts for other families. Those
    # invented metadata fixtures are never passed to the model materializer.
    selected = tuple(spec for spec in registry.specs if spec.family == "irs_soi")
    assert len(selected) == 2
    return selected


@pytest.mark.parametrize("source_path", ["download", "base_h5", "pool"])
@pytest.mark.parametrize("include_unsupported", [True, False])
def test_soi_release_preflight_precedes_population_load(
    monkeypatch, tmp_path, source_path, include_unsupported
):
    _exercise_soi_release_preflight(
        monkeypatch, tmp_path, source_path, include_unsupported
    )


@pytest.mark.parametrize("include_unsupported", [True, False])
def test_soi_release_preflight_also_applies_to_evidence_release(
    monkeypatch, tmp_path, include_unsupported
):
    # Evidence mode converts terminal verdicts only; source preflights retain
    # the existing unconditional-refusal contract in both release modes.
    _exercise_soi_release_preflight(
        monkeypatch, tmp_path, "download", include_unsupported, evidence=True
    )


def _exercise_soi_release_preflight(
    monkeypatch, tmp_path, source_path, include_unsupported, *, evidence=False
):
    builder = _builder()
    facts = _table_4_3_ledger_facts(
        include_unsupported=include_unsupported, include_return_count=True
    )
    calls = []
    compiled = []
    original_compile = builder.compile_us_fiscal_target_registry

    def load_feed(path, **pins):
        assert path == tmp_path / "invented-feed"
        assert pins == {
            "expected_facts_sha256": "1" * 64,
            "expected_manifest_sha256": "2" * 64,
        }
        calls.append("feed")
        return SimpleNamespace(facts=facts)

    def compile_feed(actual_facts, **kwargs):
        assert actual_facts is facts
        result = original_compile(actual_facts, **kwargs)
        compiled.append(result)
        calls.append("compile")
        return result

    class PopulationStageReachedError(Exception):
        pass

    def population_stage(*args, **kwargs):
        calls.append("population")
        raise PopulationStageReachedError

    monkeypatch.setattr(builder, "_git_dirty", lambda: False)
    monkeypatch.setattr(builder, "load_ledger_consumer_artifact", load_feed)
    monkeypatch.setattr(builder, "compile_us_fiscal_target_registry", compile_feed)
    monkeypatch.setattr(builder, "_download_base_h5", population_stage)
    monkeypatch.setattr(builder, "_load_base_pool_if_identified", population_stage)
    monkeypatch.setattr(
        builder, "load_simulation_ready_us_multispine_pool", population_stage
    )
    monkeypatch.setattr(
        builder,
        "_load_verified_incumbent_diagnostics_payload",
        lambda *args, **kwargs: ({}, "3" * 64),
    )
    argv = [
        "--ledger-facts",
        str(tmp_path / "invented-feed"),
        "--ledger-facts-sha256",
        "1" * 64,
        "--ledger-manifest-sha256",
        "2" * 64,
        "--out",
        str(tmp_path / "output"),
        "--no-staging",
        "--no-age-targets",
        "--allow-unaged-dollar-targets",
    ]
    if source_path == "base_h5":
        argv += ["--base-h5", str(tmp_path / "unread.h5")]
    elif source_path == "pool":
        argv += [
            "--pool-manifest",
            str(tmp_path / "unread-pool.json"),
            "--pool-manifest-sha256",
            "4" * 64,
            "--pool-release-id",
            "invented-pool",
            "--exact-k",
            "N",
            "--exact-k-pi-hi",
            "0.5",
            "--seed",
            "0",
            "--incumbent-diagnostics",
            str(tmp_path / "invented-incumbent.json"),
            "--incumbent-diagnostics-sha256",
            "3" * 64,
            "--frozen-target-surface-sha256",
            "5" * 64,
        ]
    if evidence:
        argv += ["--evidence-release"]
    if include_unsupported:
        with pytest.raises(
            RuntimeError, match="SOI source-universe preflight"
        ) as error:
            builder._main(argv)
        assert calls == ["feed", "compile"]
        contract = universe.soi_target_universe_contract(compiled[0].specs, period=2024)
        assert len(contract["unsupported_targets"]) == 2
        for entry in contract["unsupported_targets"]:
            assert entry["name"] in str(error.value)
            assert entry["source_universe"] in str(error.value)
            assert entry["reason"] in str(error.value)
        # The low-level metadata diagnostic remains available unchanged.
        assert len(contract["targets"]) == 1
    else:
        with pytest.raises(PopulationStageReachedError):
            builder._main(argv)
        assert calls == ["feed", "compile", "population"]
        assert (
            universe.soi_target_universe_contract(compiled[0].specs, period=2024)[
                "unsupported_targets"
            ]
            == []
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.requires_us
def test_compiled_table_4_3_stays_visibly_unsupported_cold_and_warm(actual, tmp_path):
    builder, frame, *_ = actual
    # No exact interest control is removed or relabeled; unsupported source
    # universes remain available to diagnostics and release-coverage checks.
    targets = _targets() + _compiled_table_4_3_targets()
    unsupported = next(
        spec
        for spec in targets
        if spec.metadata["soi_return_universe"] == "returns_excluding_dependents"
    )
    path = tmp_path / "compiled.h5"
    kwargs = {
        "target_frame_checkpoint_path": path,
        "target_frame_checkpoint_identity": {"materializer_version": 13},
    }
    cold = builder._load_or_materialize_target_frame(frame, targets, **kwargs)
    warm = builder._load_or_materialize_target_frame(frame, targets, **kwargs)
    assert cold[2]["target_frame_checkpoint"]["status"] == "miss_written"
    assert warm[2]["target_frame_checkpoint"]["status"] == "hit"
    for result, registry, compilation in (cold, warm):
        assert len(registry) == len(targets) - 1
        assert unsupported.name not in {spec.name for spec in registry.specs}
        assert unsupported.measure not in result.table("household")
        assert result.table("household").national_amount.tolist() == [5000, 5000, 50000]
        supported = next(spec for spec in targets if "table_1_2" in spec.name)
        assert result.table("household")[supported.measure].tolist() == [
            5000,
            5000,
            50000,
        ]
        assert compilation["dropped_target_names"] == [unsupported.name]
        (entry,) = compilation["unsupported_target_entries"]
        assert entry["name"] == unsupported.name
        assert entry["source_universe"] == "returns_excluding_dependents"
        assert entry["reason"] == "dependent_return_input_contract_not_established"
        assert compilation["soi_return_universe"]["contract"][
            "unsupported_targets"
        ] == [entry]
        # Even a pre-existing same-named column cannot admit the unsupported row.
        result.table("household")[unsupported.measure] = 1.0
        recompiled, diagnostics = builder._compile_materialized_target_registry(
            result, targets
        )
        assert len(recompiled) == len(targets) - 1
        assert diagnostics["dropped_target_names"] == [unsupported.name]
        # Leave all ordinary gates active: unsupported is not successful calibration.
        outcome = SimpleNamespace(
            skipped=(), diagnostics=(), initial_loss=1.0, final_loss=0.5
        )
        failures = builder._release_gate_failures(outcome, compilation)
        assert "1 fiscal targets were not materialized." in failures
        parity = builder.us_release_target_parity_gate(registry)
        assert not parity.passed
        assert any("table_4_3" in failure for failure in parity.failures)


@pytest.mark.parametrize(
    "mutation",
    ["propensity", "presence", "tax_unit_order", "credit_take_up", "person_income"],
)
@pytest.mark.requires_us
def test_warm_checkpoint_misses_changed_live_inputs(
    actual, tmp_path, monkeypatch, mutation
):
    builder, original, _, _, targets, result, _, compilation = actual
    frame = _frame()
    path = tmp_path / "live.h5"
    identity = {"materializer_version": 13}
    monkeypatch.setattr(
        builder,
        "_materialize_target_frame",
        lambda *a, **k: (result, None, compilation),
    )
    builder._load_or_materialize_target_frame(
        original,
        targets,
        target_frame_checkpoint_path=path,
        target_frame_checkpoint_identity=identity,
    )
    if mutation == "propensity":
        frame.table("tax_unit").loc[0, "would_file_taxes_voluntarily"] = True
    elif mutation == "presence":
        frame.table("tax_unit").drop(
            columns="would_file_taxes_voluntarily", inplace=True
        )
    elif mutation == "tax_unit_order":
        table = frame.table("tax_unit")
        table.iloc[:] = table.iloc[::-1].to_numpy()
    elif mutation == "credit_take_up":
        frame.table("tax_unit").loc[1, "takes_up_eitc"] = False
    else:
        frame.table("person").loc[0, "employment_income_before_lsr"] = 50000.0
    assert (
        builder._read_target_frame_checkpoint(
            path, identity=identity, target_specs=targets, base_frame=frame
        )
        is None
    )

    # Public orchestration must then enter recomputation, without recalculating
    # a filing mask merely to decide whether the cache can be reused.
    class RecomputeReachedError(Exception):
        pass

    def recompute(*args, **kwargs):
        assert args[0] is frame
        raise RecomputeReachedError

    monkeypatch.setattr(builder, "_materialize_target_frame", recompute)
    with pytest.raises(RecomputeReachedError):
        builder._load_or_materialize_target_frame(
            frame,
            targets,
            target_frame_checkpoint_path=path,
            target_frame_checkpoint_identity=identity,
        )


@pytest.mark.requires_us
def test_live_propensity_change_recomputes_actual_filer_rows(actual, tmp_path):
    builder, frame, _, _, targets, result, _, compilation = actual
    path = tmp_path / "recompute.h5"
    identity = {"materializer_version": 13}
    builder._write_target_frame_checkpoint(
        path,
        frame=result,
        identity=identity,
        compilation={
            **compilation,
            "soi_live_input_frame_sha256": builder._late_frame_content_sha256(frame),
        },
    )
    changed = _frame()
    changed.table("tax_unit").loc[0, "would_file_taxes_voluntarily"] = True
    refreshed, _, receipt = builder._load_or_materialize_target_frame(
        changed,
        targets,
        target_frame_checkpoint_path=path,
        target_frame_checkpoint_identity=identity,
    )
    assert receipt["target_frame_checkpoint"]["status"] == "miss_written"
    assert refreshed.table("household").national_count.tolist() == [2, 1, 1]
    assert refreshed.table("household").national_amount.tolist() == [10000, 5000, 50000]
    assert (
        receipt["soi_return_universe"]["actual_mask_sha256"]
        != compilation["soi_return_universe"]["actual_mask_sha256"]
    )


@pytest.mark.requires_us
def test_missing_live_input_misses_but_malformed_stored_receipt_refuses(
    actual, tmp_path
):
    builder, frame, _, _, targets, result, _, compilation = actual
    path = tmp_path / "missing.h5"
    identity = {"materializer_version": 13}
    document = {
        **compilation,
        "soi_live_input_frame_sha256": builder._late_frame_content_sha256(frame),
    }
    builder._write_target_frame_checkpoint(
        path, frame=result, identity=identity, compilation=document
    )
    assert (
        builder._read_target_frame_checkpoint(
            path, identity=identity, target_specs=targets
        )
        is None
    )
    document["soi_live_input_frame_sha256"] = "bad"
    builder._write_target_frame_checkpoint(
        path, frame=result, identity=identity, compilation=document
    )
    with pytest.raises(ValueError, match="invalid cached live input digest"):
        builder._read_target_frame_checkpoint(
            path, identity=identity, target_specs=targets, base_frame=frame
        )
    document["soi_return_universe"] = {}
    builder._write_target_frame_checkpoint(
        path, frame=result, identity=identity, compilation=document
    )
    with pytest.raises(ValueError, match="missing cached filer receipt"):
        builder._read_target_frame_checkpoint(
            path, identity=identity, target_specs=targets
        )
