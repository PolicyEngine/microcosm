"""Development entry-point mechanics on invented captures and populations."""

import sys
from dataclasses import replace

import pytest
import test_us_national_age_counts as age_fixture
import test_us_survey_age_activation as activation_fixture

from microcosm.build.us_runtime import graph_survey_calibration as numeric
from microcosm.build.us_runtime import survey_age_activation as activation
from microcosm.build.us_runtime import survey_age_calibration as runner
from microcosm.calibrate.registry import TargetRegistry


def captured_targets(root):
    metadata, data = activation_fixture.invented_documents()
    header, row = data
    registry = age_fixture.fixture_registry()
    for spec in registry:
        row[header.index(spec.name + "E")] = str(int(spec.value))
    row[header.index("S0101_C01_001E")] = str(sum(int(s.value) for s in registry))
    declaration = activation_fixture.capture_fixture(root, metadata=metadata, data=data)
    return declaration, activation.activate_survey_age_targets(
        root, declaration=declaration
    )


def options():
    return {
        "base": "invented",
        "budget_node": "budget",
        "count_node": "counts",
        "epochs": 12,
        "learning_rate": 0.1,
    }


def test_original_node_stays_invented_only_and_development_pins_move_identity(tmp_path):
    declaration, registry = captured_targets(tmp_path)
    with pytest.raises(ValueError, match="INVENTED_AGE_REGISTRY_REQUIRED"):
        numeric.survey_age_calibration_node(registry, **options())
    binding = activation.activation_binding(declaration)
    node = numeric.survey_age_development_node(
        registry, activation_binding=binding, **options()
    )
    assert (
        numeric.calibration_activation_binding(node.params)["release_eligible"] is False
    )
    moved = replace(declaration, data_sha256="0" * 64)
    changed_registry = TargetRegistry(
        [
            replace(
                s,
                metadata={
                    **s.metadata,
                    "reference_sha256": activation.activation_digest(moved),
                },
            )
            for s in registry
        ],
        country="us",
    )
    other = numeric.survey_age_development_node(
        changed_registry,
        activation_binding=activation.activation_binding(moved),
        **options(),
    )
    assert node.normative() != other.normative()


@pytest.mark.parametrize(
    "value", [None, {}, "{", "null", '{"profile":"wrong"}', '{"a":1,"a":2}']
)
def test_malformed_activation_param_refuses(value):
    with pytest.raises(ValueError):
        numeric.calibration_activation_binding({"activation": value})


def test_noncanonical_activation_text_refuses(tmp_path):
    declaration, registry = captured_targets(tmp_path)
    node = numeric.survey_age_development_node(
        registry,
        activation_binding=activation.activation_binding(declaration),
        **options(),
    )
    with pytest.raises(ValueError, match="ACTIVATION_CANONICAL"):
        numeric.calibration_activation_binding(
            {"activation": " " + node.params["activation"]}
        )


@pytest.mark.parametrize(
    "change",
    [
        {"epochs": True},
        {"learning_rate": False},
        {"resume": "off"},
        {"activation": None},
    ],
)
def test_invalid_profile_refuses_before_source_or_target_access(
    tmp_path, monkeypatch, change
):
    calls = []
    monkeypatch.setattr(
        runner.age_activation,
        "activate_survey_age_targets",
        lambda *a, **k: calls.append("targets"),
    )
    monkeypatch.setattr(
        runner.source_graph,
        "run_authenticated_survey_population",
        lambda *a, **k: calls.append("survey"),
    )
    kwargs = {
        "age_source_dir": tmp_path / "targets",
        "activation": activation.SurveyAgeActivation("a" * 64, "b" * 64),
        "snapshot_root": tmp_path / "snapshots",
        "store_root": tmp_path / "store",
        "fraction": None,
        "seed_value": 0,
        "epochs": 12,
        "learning_rate": 0.1,
    }
    with pytest.raises(ValueError):
        runner.run_survey_age_development(tmp_path / "survey", **{**kwargs, **change})
    assert calls == []
    assert not tuple(tmp_path.iterdir())


def test_missing_or_changed_capture_refuses_before_survey(tmp_path, monkeypatch):
    declaration, _ = captured_targets(tmp_path / "age")
    calls = []
    monkeypatch.setattr(
        runner.source_graph,
        "run_authenticated_survey_population",
        lambda *a, **k: calls.append(True),
    )
    (tmp_path / "age" / "raw" / f"{declaration.data_sha256}.json").write_bytes(b"[]")
    with pytest.raises(ValueError, match="PIN"):
        runner.run_survey_age_development(
            tmp_path / "survey",
            age_source_dir=tmp_path / "age",
            activation=declaration,
            snapshot_root=tmp_path / "snapshots",
            store_root=tmp_path / "store",
            fraction=None,
            seed_value=0,
            epochs=12,
            learning_rate=0.1,
        )
    assert calls == []


def test_actual_invented_population_with_captured_age_profile_preserves_seven_nodes(
    tmp_path, monkeypatch
):
    from test_us_graph_survey_population import authenticated_arguments

    declaration, registry = captured_targets(tmp_path / "age")
    arguments = authenticated_arguments(tmp_path, monkeypatch)
    arguments["seed_value"] = arguments.pop("seed")
    result = runner.run_survey_age_development(
        **arguments,
        age_source_dir=tmp_path / "age",
        activation=declaration,
        epochs=12,
        learning_rate=0.1,
    )
    assert len(result.compiled.order) == 7
    assert result.release_eligible is False
    successor = result.successor.checked_view()
    runner.budgets._same_nonweight(successor.previous.frame, successor.current.frame)
    assert successor.current.frame.n("household") == 12
    receipt = result.manifest.node(numeric.CALIBRATION_NODE).receipt
    assert receipt["scope"] == "source_documented_survey_age_calibration_numbers_only"
    assert receipt["release_eligible"] is False
    assert result.diagnostics["verification"]["optimizer_rerun"] is False
    assert result.diagnostics["options"]["iterate_selection"] == "closing_state"
    assert result.diagnostics["options"]["iterate_selection_receipt"] == {}
    activation.verify_survey_age_targets(
        tmp_path / "age", declaration=declaration, registry=registry
    )


def test_late_target_mutation_during_store_work_refuses(tmp_path, monkeypatch):
    from test_us_graph_survey_population import authenticated_arguments

    declaration, _ = captured_targets(tmp_path / "age")
    arguments = authenticated_arguments(tmp_path, monkeypatch)
    arguments["seed_value"] = arguments.pop("seed")
    original = runner._load_diagnostics

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        (tmp_path / "age" / "raw" / f"{declaration.data_sha256}.json").write_bytes(
            b"[]"
        )
        return result

    monkeypatch.setattr(runner, "_load_diagnostics", changed)
    with pytest.raises(ValueError, match="PIN"):
        runner.run_survey_age_development(
            **arguments,
            age_source_dir=tmp_path / "age",
            activation=declaration,
            epochs=12,
            learning_rate=0.1,
        )


def test_successor_mutation_during_final_target_verification_refuses(
    tmp_path, monkeypatch
):
    from test_us_graph_survey_population import authenticated_arguments

    declaration, _ = captured_targets(tmp_path / "age")
    arguments = authenticated_arguments(tmp_path, monkeypatch)
    arguments["seed_value"] = arguments.pop("seed")
    issued = []
    changed = []
    admit = runner.budgets.admit_survey_weight_only_population.__code__
    original = activation.verify_survey_age_targets

    def observe_return(frame, event, value):
        if (
            event == "return"
            and frame.f_code is admit
            and type(value) is runner.budgets.SamplingOriginSuccessor
        ):
            issued.append(value)

    def mutate_after_target_check(*args, **kwargs):
        original(*args, **kwargs)
        assert len(issued) == 1
        object.__setattr__(
            issued[0], "payload", b"changed during final target verification"
        )
        changed.append(True)

    monkeypatch.setattr(
        activation, "verify_survey_age_targets", mutate_after_target_check
    )
    previous_profile = sys.getprofile()
    sys.setprofile(observe_return)
    try:
        with pytest.raises(
            runner.budgets.SurveyOriginBudgetError, match="UNISSUED_OR_CHANGED"
        ):
            runner.run_survey_age_development(
                **arguments,
                age_source_dir=tmp_path / "age",
                activation=declaration,
                epochs=12,
                learning_rate=0.1,
            )
    finally:
        sys.setprofile(previous_profile)
    assert changed == [True]
