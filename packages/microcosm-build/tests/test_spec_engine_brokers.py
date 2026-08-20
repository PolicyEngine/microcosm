"""Adversarial and golden tests for the F1 broker boundary."""

from __future__ import annotations

import copy
import datetime as datetime_module
import hashlib
import os
import time
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from microcosm.build.spec_engine.brokers import (
    AmbientAccessError,
    BrokerAccessError,
    BrokerAccessEvent,
    BrokerContractError,
    BrokerOwner,
    BrokerSession,
    DeclaredSource,
    DerivedSeedHandle,
    GeneratorLease,
    PhysicalOperation,
    RNGBehaviorIdentity,
    RNGInvocation,
    TorchGeneratorLease,
)
from microcosm.build.spec_engine.canonical import sha256_json
from microcosm.build.spec_engine.compiler_ir import CompiledSpecIR, compile_spec
from microcosm.build.spec_engine.loader import load_bundle
from microcosm.build.spec_engine.model import freeze_json, thaw_json
from microcosm.build.spec_engine.schemas import load_schema_registry


@pytest.fixture(scope="module")
def compiled_us() -> CompiledSpecIR:
    return compile_spec(load_bundle("us"))


def _fixture_run_provenance_wire() -> dict[str, object]:
    return {
        "identity_generation": 0,
        "source_grammar_receipt": None,
        "spec_binding": None,
        "authority_versions": {"fixture": 1},
        "code_inventory_digest": "a" * 64,
        "artifact_protocol_inventory": {"fixture": "v1"},
        "run_request": {"rung": "fixture"},
        "execution_receipt": {"backend": "fixture"},
    }


def _owner_session(
    compiled: CompiledSpecIR,
    kind: str,
    owner_id: str,
    *,
    run_inputs: dict[str, int] | None = None,
    rng_invocation_plan: dict[str, tuple[RNGInvocation, ...]] | None = None,
) -> BrokerSession:
    return BrokerSession.for_seed_owner(
        compiled.seed_stream_map,
        owner_kind=kind,
        owner_id=owner_id,
        run_provenance_identity=_fixture_run_provenance_wire(),
        run_inputs={} if run_inputs is None else run_inputs,
        rng_invocation_plan=rng_invocation_plan,
    )


def _source(path: Path, source_id: str = "fixture_source") -> DeclaredSource:
    payload = path.read_bytes()
    return DeclaredSource(
        id=source_id,
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
    )


def _non_rng_session(
    *,
    determinism: str = "deterministic",
    effects: tuple[str, ...] = ("none",),
    sources: tuple[DeclaredSource, ...] = (),
    environment: dict[str, str | None] | None = None,
    clocks: dict[str, float] | None = None,
    node_key: str = "a" * 64,
    require_byte_equivalence: bool = True,
) -> BrokerSession:
    return BrokerSession(
        owner=BrokerOwner("producer_node", "fixture_node"),
        determinism=determinism,
        effects=effects,
        protocol_id="legacy-v1",
        protocol_sha256="b" * 64,
        sources=sources,
        environment={} if environment is None else environment,
        clocks={} if clocks is None else clocks,
        run_provenance_identity=_fixture_run_provenance_wire(),
        node_key=node_key,
        require_byte_equivalence=require_byte_equivalence,
    )


def test_stream_token_binds_protocol_owner_site_stream_and_contract(
    compiled_us: CompiledSpecIR,
) -> None:
    session = _owner_session(
        compiled_us,
        "pipeline_operation",
        "assemble_stacked_spine",
        run_inputs={"sample_seed": 578},
    )
    with session.activate():
        token = session.rng.token("survey_sample_asec")
        wire = token.to_wire()
        assert wire["protocol_id"] == "legacy-v1"
        assert (
            wire["protocol_sha256"] == compiled_us.seed_stream_map.implementation_sha256
        )
        assert wire["owner"] == {
            "kind": "pipeline_operation",
            "id": "assemble_stacked_spine",
        }
        assert wire["site_id"] == "survey_sample_asec"
        assert wire["stream"] == "stream:sampling_asec"
        assert len(str(wire["contract_sha256"])) == 64
        assert wire["boundary_instance"] == 0
        assert wire["boundary_key"] == "default"
        assert wire["semantic_material"] == {}
        with pytest.raises(TypeError):
            token.semantic_material["mutate"] = 1  # type: ignore[index]
        with pytest.raises(BrokerAccessError, match="not granted"):
            session.rng.token("scf_household_source_selector")
    receipt = session.seal(status="aborted")
    receipt.validate()
    load_schema_registry().validate(
        receipt.to_wire(), "locks.schema.json#/$defs/broker_access_receipt"
    )
    assert receipt.to_wire()["run_provenance_identity"] == (
        _fixture_run_provenance_wire()
    )
    assert [event.disposition for event in receipt.events] == ["allowed", "refused"]


def test_legacy_v1_survey_sampling_matches_captured_constants_draw(
    compiled_us: CompiledSpecIR,
) -> None:
    session = _owner_session(
        compiled_us,
        "pipeline_operation",
        "assemble_stacked_spine",
        run_inputs={"sample_seed": 578},
    )
    with session.activate():
        token = session.rng.token("survey_sample_asec")
        with session.rng.generator(token) as generator:
            selected = np.sort(
                generator.choice(
                    np.arange(101, 141, dtype=np.int64), size=4, replace=False
                )
            )
    session.seal()
    # Captured from constants-mode frame_sampling.sample_frame_households.
    assert selected.tolist() == [109, 114, 131, 134]


def test_legacy_v1_scf_composite_seed_matches_captured_constants_draw(
    compiled_us: CompiledSpecIR,
) -> None:
    session = _owner_session(
        compiled_us,
        "source_stage",
        "scf_wealth",
        run_inputs={"build_model_seed": 23},
        rng_invocation_plan={
            "scf_household_source_selector": (
                RNGInvocation("period-2024", {"time_period": 2024}),
            )
        },
    )
    with session.activate():
        token = session.rng.token("scf_household_source_selector", "period-2024")
        with session.rng.generator(token) as generator:
            draws = generator.random(5)
    session.seal()
    # Captured from constants-mode financial_asset_source_is_scf.
    assert draws.tolist() == pytest.approx(
        [
            0.7925334903155368,
            0.29056553275599917,
            0.7176194059662899,
            0.5645121554854824,
            0.06925345806039351,
        ],
        rel=0.0,
        abs=0.0,
    )


def test_legacy_v1_qrf_child_streams_match_captured_constants_draw(
    compiled_us: CompiledSpecIR,
) -> None:
    session = _owner_session(
        compiled_us,
        "producer_node",
        "primary_puf_qrf",
        run_inputs={"build_model_seed": 19},
    )
    with session.activate():
        token = session.rng.token("primary_qrf_fit_draw")
        with session.rng.qrf_generators(token) as generators:
            fit = generators.fit.integers(0, 2**31 - 1, size=5)
            draw = generators.draw.random(8)
    session.seal()
    # Captured from constants-mode QRF SeedSequence(seed).spawn(2).
    assert fit.tolist() == [
        467277671,
        2116849096,
        946147772,
        118093912,
        547671721,
    ]
    assert draw.tolist() == pytest.approx(
        [
            0.638832981794515,
            0.20624401016621863,
            0.6313570369850963,
            0.2662559878583304,
            0.3514832003718673,
            0.2962560197448888,
            0.4502353625402643,
            0.9273504386472491,
        ],
        rel=0.0,
        abs=0.0,
    )


def test_sha_seed_material_is_ledger_ordered_and_chains_into_acs_qrf(
    compiled_us: CompiledSpecIR,
) -> None:
    optional_predictors = "education_level\0employment_income"
    session = _owner_session(
        compiled_us,
        "pipeline_operation",
        "gap_fill_stacked_spine",
        run_inputs={"build_model_seed": 19},
        rng_invocation_plan={
            "acs_transfer_family_seed": (
                RNGInvocation(
                    "person-income",
                    {"entity": "person", "family": "income"},
                ),
            ),
            "acs_transfer_pattern_seed": (
                RNGInvocation(
                    "person-income-pattern",
                    {
                        "entity": "person",
                        "family": "income",
                        "nul_joined_ordered_optional_predictors": optional_predictors,
                    },
                ),
            ),
            "acs_qrf_fit_draw": (
                RNGInvocation(
                    "person-income-pattern",
                    {
                        "derived_from": {
                            "site_id": "acs_transfer_pattern_seed",
                            "boundary_key": "person-income-pattern",
                        }
                    },
                ),
            ),
        },
    )
    expected_family_seed = int.from_bytes(
        hashlib.sha256(b"19\0person\0income").digest()[:4], "little"
    )
    expected_pattern_seed = int.from_bytes(
        hashlib.sha256(f"19\0person\0income\0{optional_predictors}".encode()).digest()[
            :4
        ],
        "little",
    )
    with session.activate():
        family_handle = session.rng.sha256_derived_seed(
            session.rng.token("acs_transfer_family_seed", "person-income")
        )
        pattern_handle = session.rng.sha256_derived_seed(
            session.rng.token("acs_transfer_pattern_seed", "person-income-pattern")
        )
        with session.rng.qrf_generators(
            session.rng.token("acs_qrf_fit_draw", "person-income-pattern")
        ) as generators:
            actual = generators.draw.random(5)
    receipt = session.seal()
    assert isinstance(family_handle, DerivedSeedHandle)
    assert isinstance(pattern_handle, DerivedSeedHandle)
    assert not hasattr(pattern_handle, "value")
    realized = [
        event.details["realized_seed"]
        for event in receipt.events
        if event.operation == "derived_seed"
    ]
    assert realized == [expected_family_seed, expected_pattern_seed]
    _, expected_draw_child = np.random.SeedSequence(expected_pattern_seed).spawn(2)
    expected = np.random.Generator(np.random.PCG64(expected_draw_child)).random(5)
    assert actual.tolist() == expected.tolist()


def test_sha_and_derived_qrf_refuse_caller_authored_seed_material(
    compiled_us: CompiledSpecIR,
) -> None:
    with pytest.raises(
        BrokerContractError, match="missing=.*entity|extra=.*components"
    ):
        _owner_session(
            compiled_us,
            "pipeline_operation",
            "gap_fill_stacked_spine",
            run_inputs={"build_model_seed": 19},
            rng_invocation_plan={
                "acs_transfer_family_seed": (
                    RNGInvocation(
                        "forged",
                        {"components": [999, "person", "income"]},
                    ),
                )
            },
        )

    session = _owner_session(
        compiled_us,
        "pipeline_operation",
        "gap_fill_stacked_spine",
        run_inputs={"build_model_seed": 19},
        rng_invocation_plan={
            "acs_transfer_pattern_seed": (
                RNGInvocation(
                    "pattern",
                    {
                        "entity": "person",
                        "family": "income",
                        "nul_joined_ordered_optional_predictors": "education_level",
                    },
                ),
            ),
            "acs_qrf_fit_draw": (
                RNGInvocation(
                    "pattern",
                    {
                        "derived_from": {
                            "site_id": "acs_transfer_pattern_seed",
                            "boundary_key": "pattern",
                        }
                    },
                ),
            ),
        },
    )
    with session.activate():
        with pytest.raises(BrokerAccessError, match="has not been produced"):
            session.rng.qrf_generators(session.rng.token("acs_qrf_fit_draw", "pattern"))
    receipt = session.seal(status="aborted")
    assert receipt.events[-1].reason_code == "derived_seed_not_produced"


def test_pandas_training_cap_sampling_is_broker_owned_and_byte_exact(
    compiled_us: CompiledSpecIR,
) -> None:
    frame = pd.DataFrame(
        {
            "row_id": np.arange(12, dtype=np.int64),
            "value": np.linspace(1.0, 12.0, 12),
        }
    )
    expected = frame.sample(n=4, random_state=17)
    session = _owner_session(
        compiled_us,
        "source_stage",
        "prior_year_income",
        run_inputs={"build_model_seed": 17},
        rng_invocation_plan={
            "prior_year_income_training_cap": (
                RNGInvocation("cap-4", {"stage_training_cap": 4}),
            )
        },
    )
    with session.activate():
        actual = session.rng.pandas_sample(
            session.rng.token("prior_year_income_training_cap", "cap-4"),
            frame,
            n=4,
        )
    receipt = session.seal()
    pd.testing.assert_frame_equal(actual, expected)
    assert receipt.events[-1].operation == "pandas_sample"
    assert not hasattr(session.rng, "integer_seed")


def test_sklearn_random_forest_fit_and_predict_stays_inside_broker(
    compiled_us: CompiledSpecIR,
) -> None:
    from sklearn.ensemble import RandomForestClassifier

    train_x = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 3,
        dtype=np.float64,
    )
    train_y = np.asarray([0, 1, 1, 2] * 3, dtype=np.int64)
    predict_x = np.asarray([[0.0, 0.0], [1.0, 1.0], [0.2, 0.8]], dtype=np.float64)
    params = {
        "n_estimators": 9,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "n_jobs": 1,
    }
    expected_model = RandomForestClassifier(**params, random_state=23)
    expected_model.fit(train_x, train_y)
    expected = expected_model.predict(predict_x)
    session = _owner_session(
        compiled_us,
        "source_stage",
        "vehicle_assets",
        run_inputs={"build_model_seed": 23},
    )
    with session.activate():
        actual = session.rng.random_forest_classifier_predict(
            session.rng.token("sipp_vehicle_count_random_forest_model"),
            train_x=train_x,
            train_y=train_y,
            predict_x=predict_x,
            params=params,
        )
    session.seal()
    assert isinstance(actual, np.ndarray)
    assert actual.tolist() == expected.tolist()
    assert not isinstance(actual, RandomForestClassifier)


def test_torch_reset_boundaries_are_broker_owned_and_match_manual_seed(
    compiled_us: CompiledSpecIR,
) -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(31)
    expected_first = torch.empty(5, dtype=torch.float32).uniform_(1e-8, 1.0 - 1e-8)
    expected_second = torch.empty(3, dtype=torch.float32).uniform_(1e-8, 1.0 - 1e-8)
    session = _owner_session(
        compiled_us,
        "pipeline_operation",
        "calibrate",
        run_inputs={"build_model_seed": 31},
        rng_invocation_plan={
            "torch_calibration_reseed": (
                RNGInvocation("calibrate-entry"),
                RNGInvocation("budget-evaluation-1"),
            )
        },
    )
    with session.activate():
        with session.rng.torch_generator(
            session.rng.token("torch_calibration_reseed", "calibrate-entry")
        ) as first_lease:
            assert isinstance(first_lease, TorchGeneratorLease)
            actual_first = first_lease.uniform(
                (5,), low=1e-8, high=1.0 - 1e-8, dtype=torch.float32
            )
            actual_second = first_lease.uniform(
                (3,), low=1e-8, high=1.0 - 1e-8, dtype=torch.float32
            )
        with session.rng.torch_generator(
            session.rng.token("torch_calibration_reseed", "budget-evaluation-1")
        ) as second_lease:
            reset_draw = second_lease.uniform(
                (5,), low=1e-8, high=1.0 - 1e-8, dtype=torch.float32
            )
    session.seal()
    assert actual_first.tolist() == expected_first.tolist()
    assert actual_second.tolist() == expected_second.tolist()
    assert reset_draw.tolist() == expected_first.tolist()
    assert first_lease.closed and second_lease.closed
    assert not hasattr(session.rng, "integer_seed")


def test_direct_torch_uniform_is_refused_by_ambient_guard() -> None:
    torch = pytest.importorskip("torch")
    session = _non_rng_session()
    with session.activate():
        with pytest.raises(AmbientAccessError, match="torch.Tensor.uniform_"):
            torch.zeros(2).uniform_()
    session.seal(status="aborted")


def test_legacy_v1_blake2b_stateless_uniform_matches_literal_vector(
    compiled_us: CompiledSpecIR,
) -> None:
    stable_keys = [10, 20, 30]
    session = _owner_session(
        compiled_us,
        "source_stage",
        "aca_marketplace_inputs",
        run_inputs={"build_model_seed": 0},
        rng_invocation_plan={
            "source_aca_assignment": (
                RNGInvocation(
                    "aca-output",
                    {
                        "output": "is_aca_eshi_eligible",
                        "stable_keys_sha256": sha256_json(
                            [str(key) for key in stable_keys]
                        ),
                    },
                ),
            )
        },
    )
    with session.activate():
        token = session.rng.token("source_aca_assignment", "aca-output")
        draws = session.rng.blake2b_uniforms(
            token,
            stable_keys=stable_keys,
        )
    session.seal()
    expected = [
        int.from_bytes(
            hashlib.blake2b(
                f"0:aca:is_aca_eshi_eligible:{key}".encode(), digest_size=8
            ).digest(),
            "big",
        )
        / 2**64
        for key in stable_keys
    ]
    assert draws.tolist() == expected


def test_blake2b_validates_stable_key_digest_from_invocation_plan(
    compiled_us: CompiledSpecIR,
) -> None:
    declared_keys = [10, 20, 30]
    session = _owner_session(
        compiled_us,
        "source_stage",
        "aca_marketplace_inputs",
        run_inputs={"build_model_seed": 0},
        rng_invocation_plan={
            "source_aca_assignment": (
                RNGInvocation(
                    "aca-output",
                    {
                        "output": "is_aca_eshi_eligible",
                        "stable_keys_sha256": sha256_json(
                            [str(key) for key in declared_keys]
                        ),
                    },
                ),
            )
        },
    )
    with session.activate():
        token = session.rng.token("source_aca_assignment", "aca-output")
        with pytest.raises(BrokerAccessError, match="stable-key digest"):
            session.rng.blake2b_uniforms(token, stable_keys=[10, 20, 31])
    receipt = session.seal(status="aborted")
    assert receipt.events[-1].reason_code == "stable_key_digest_mismatch"


def test_tokens_are_single_use_activation_scoped_and_tamper_evident(
    compiled_us: CompiledSpecIR,
) -> None:
    session = _owner_session(
        compiled_us,
        "pipeline_operation",
        "assemble_stacked_spine",
        run_inputs={"sample_seed": 578},
    )
    with session.activate():
        token = session.rng.token("survey_sample_asec")
        forged = replace(token, stream="sampling_acs")
        with pytest.raises(BrokerAccessError, match="rng_stream_mismatch"):
            session.rng.generator(forged)
        forged_material = replace(
            token,
            semantic_material=RNGInvocation("forged", {"value": 1}).semantic_material,
        )
        with pytest.raises(BrokerAccessError, match="rng_semantic_material_mismatch"):
            session.rng.generator(forged_material)
        with session.rng.generator(token) as generator:
            generator.random(1)
        with pytest.raises(BrokerAccessError, match="already_consumed"):
            session.rng.generator(token)
        with pytest.raises(BrokerAccessError, match="plan is exhausted"):
            session.rng.token("survey_sample_asec")
    session.seal(status="aborted")
    with pytest.raises(BrokerAccessError, match="sealed"):
        session.rng.token("survey_sample_asec")


def test_ordered_invocation_plan_allows_repeat_site_boundaries(
    compiled_us: CompiledSpecIR,
) -> None:
    session = _owner_session(
        compiled_us,
        "pipeline_operation",
        "assemble_stacked_spine",
        run_inputs={"sample_seed": 578},
        rng_invocation_plan={
            "survey_sample_asec": (
                RNGInvocation("first-batch"),
                RNGInvocation("second-batch"),
            )
        },
    )
    with session.activate():
        with pytest.raises(BrokerAccessError, match="expected boundary"):
            session.rng.token("survey_sample_asec", "second-batch")
        first = session.rng.token("survey_sample_asec", "first-batch")
        second = session.rng.token("survey_sample_asec", "second-batch")
        with pytest.raises(BrokerAccessError, match="consumption_order"):
            session.rng.generator(second)
        with session.rng.generator(first) as generator:
            first_draw = generator.random(1)
        with session.rng.generator(second) as generator:
            second_draw = generator.random(1)
        with pytest.raises(BrokerAccessError, match="plan is exhausted"):
            session.rng.token("survey_sample_asec", "third-batch")
    session.seal(status="aborted")
    assert first.boundary_instance == 0
    assert second.boundary_instance == 1
    assert first_draw.tolist() == second_draw.tolist()


def test_generator_lease_does_not_expose_raw_generator(
    compiled_us: CompiledSpecIR,
) -> None:
    session = _owner_session(
        compiled_us,
        "pipeline_operation",
        "assemble_stacked_spine",
        run_inputs={"sample_seed": 578},
    )
    with session.activate():
        token = session.rng.token("survey_sample_asec")
        with session.rng.generator(token) as generator:
            assert "generator" not in GeneratorLease.__slots__
            with pytest.raises(BrokerAccessError, match="internals"):
                _ = generator._generator  # type: ignore[attr-defined]  # noqa: SLF001
            with pytest.raises(BrokerAccessError, match="internals"):
                _ = generator._GeneratorLease__generator  # type: ignore[attr-defined]  # noqa: SLF001
            with pytest.raises(BrokerAccessError, match="safe broker allowlist"):
                _ = generator.bit_generator  # type: ignore[attr-defined]
            with pytest.raises(BrokerAccessError, match="safe broker allowlist"):
                _ = generator.spawn  # type: ignore[attr-defined]
            assert not hasattr(session._rng_leases, "_generator")  # noqa: SLF001
            handle = object.__getattribute__(generator, "_handle")
            serialized = session._rng_leases._states[handle]  # noqa: SLF001
            assert isinstance(serialized, bytes)
    session.seal(status="aborted")


def test_generator_lease_rejects_authority_return_and_seal_closes_lease(
    compiled_us: CompiledSpecIR,
) -> None:
    rogue_generator = np.random.default_rng(9)
    candidates = np.empty(1, dtype=object)
    candidates[0] = rogue_generator
    session = _owner_session(
        compiled_us,
        "pipeline_operation",
        "assemble_stacked_spine",
        run_inputs={"sample_seed": 578},
    )
    with session.activate():
        token = session.rng.token("survey_sample_asec")
        lease = session.rng.generator(token)
        with pytest.raises(BrokerAccessError, match="returned generator authority"):
            lease.choice(candidates)
        assert not lease.closed
    session.seal(status="aborted")
    assert lease.closed
    with pytest.raises(BrokerAccessError, match="closed"):
        lease.random(1)


def test_qrf_exact_state_restore_does_not_consume_ambient_entropy(
    compiled_us: CompiledSpecIR,
) -> None:
    first = _owner_session(
        compiled_us,
        "producer_node",
        "primary_puf_qrf",
        run_inputs={"build_model_seed": 19},
    )
    with first.activate():
        token = first.rng.token("primary_qrf_fit_draw")
        with first.rng.qrf_generators(token) as generators:
            generators.draw.random(7)
            state = generators.draw.bit_generator_state()
            expected = generators.draw.random(4)
    first.seal()

    second = _owner_session(
        compiled_us,
        "producer_node",
        "primary_puf_qrf",
        run_inputs={"build_model_seed": 19},
        rng_invocation_plan={
            "primary_qrf_fit_draw": (
                RNGInvocation("resume", {"restored_draw_state": state}),
            )
        },
    )
    with second.activate():
        token = second.rng.token("primary_qrf_fit_draw", "resume")
        with second.rng.qrf_generators(token) as generators:
            actual = generators.draw.random(4)
    second.seal()
    assert actual.tolist() == expected.tolist()


def test_declared_file_read_is_verified_and_raw_or_unknown_access_is_refused(
    tmp_path: Path,
) -> None:
    path = tmp_path / "declared.txt"
    path.write_bytes(b"brokered bytes\n")
    session = _non_rng_session(
        effects=("declared_source_read",), sources=(_source(path),)
    )
    with session.activate():
        assert session.files.read_bytes("fixture_source") == b"brokered bytes\n"
        with pytest.raises(BrokerAccessError, match="undeclared_source"):
            session.files.read_bytes("other")
        with pytest.raises(AmbientAccessError, match="ambient file"):
            path.read_bytes()
        with pytest.raises(AmbientAccessError, match="ambient file"):
            open(path, "rb")  # noqa: PTH123
    receipt = session.seal(status="aborted")
    assert [event.disposition for event in receipt.events] == [
        "allowed",
        "refused",
        "refused",
        "refused",
    ]


def test_declared_file_open_is_streamed_from_a_verified_handle(tmp_path: Path) -> None:
    path = tmp_path / "streamed.bin"
    path.write_bytes(b"0123456789")
    session = _non_rng_session(
        effects=("declared_source_read",), sources=(_source(path),)
    )
    with session.activate(), session.files.open_read("fixture_source") as stream:
        assert stream.read(4) == b"0123"
        assert stream.read() == b"456789"
    receipt = session.seal()
    assert receipt.status == "complete"
    assert receipt.events[0].operation == "open_read"


def test_declared_file_snapshots_support_h5py_zipfile_and_pandas(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "source.h5"
    with h5py.File(h5_path, mode="w") as h5:
        h5.create_dataset("values", data=np.asarray([1, 2, 3], dtype=np.int64))
    zip_path = tmp_path / "source.zip"
    with ZipFile(zip_path, mode="w") as archive:
        archive.writestr("rows.csv", "key,value\na,1\nb,2\n")
    csv_path = tmp_path / "source.csv"
    csv_path.write_text("key,value\nx,4\ny,5\n", encoding="utf-8")

    session = _non_rng_session(
        effects=("declared_source_read",),
        sources=(
            _source(h5_path, "h5"),
            _source(zip_path, "zip"),
            _source(csv_path, "csv"),
        ),
    )
    with session.activate(), ExitStack() as stack:
        snapshots = {
            source_id: stack.enter_context(session.files.open_snapshot(source_id))
            for source_id in ("csv", "h5", "zip")
        }
        with h5py.File(snapshots["h5"], mode="r") as h5:
            assert np.asarray(h5["values"]).tolist() == [1, 2, 3]
        with ZipFile(snapshots["zip"]) as archive:
            with archive.open("rows.csv") as member:
                assert pd.read_csv(member).to_dict("records") == [
                    {"key": "a", "value": 1},
                    {"key": "b", "value": 2},
                ]
        assert pd.read_csv(snapshots["csv"]).to_dict("records") == [
            {"key": "x", "value": 4},
            {"key": "y", "value": 5},
        ]
        with pytest.raises(AttributeError):
            snapshots["h5"].fileno()  # type: ignore[attr-defined]
    receipt = session.seal()
    assert [event.operation for event in receipt.events] == [
        "open_snapshot",
        "open_snapshot",
        "open_snapshot",
    ]


def test_file_snapshot_retains_descriptor_across_path_replacement_and_reopen_refuses(
    tmp_path: Path,
) -> None:
    path = tmp_path / "replaceable.bin"
    path.write_bytes(b"authenticated")
    declared = _source(path)
    original_open = open  # noqa: PTH123
    session = _non_rng_session(effects=("declared_source_read",), sources=(declared,))
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"replacement!!")
    with session.activate(), session.files.open_snapshot("fixture_source") as lease:
        os.replace(replacement, path)
        assert lease.read() == b"authenticated"
        with original_open(path, "rb") as reopened:
            assert reopened.read() == b"replacement!!"
    session.seal()

    reopened_session = _non_rng_session(
        effects=("declared_source_read",), sources=(declared,)
    )
    with reopened_session.activate():
        with pytest.raises(BrokerAccessError, match="verified identity"):
            with reopened_session.files.open_snapshot("fixture_source"):
                pass
    assert reopened_session.seal(status="aborted").status == "aborted"


def test_file_snapshot_close_refuses_in_place_drift_and_aborts_session(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mutable.bin"
    path.write_bytes(b"before")
    original_open = open  # noqa: PTH123
    session = _non_rng_session(
        effects=("declared_source_read",), sources=(_source(path),)
    )
    with session.activate():
        with pytest.raises(BrokerAccessError, match="changed while"):
            with session.files.open_snapshot("fixture_source") as lease:
                assert lease.read() == b"before"
                with original_open(path, "r+b") as mutable:
                    mutable.seek(0)
                    mutable.write(b"after!")
                    mutable.flush()
    receipt = session.seal(status="aborted")
    assert receipt.status == "aborted"
    assert receipt.events[-1].reason_code == "source_snapshot_drift"


def test_session_seal_closes_an_unexited_file_lease(tmp_path: Path) -> None:
    path = tmp_path / "dangling.bin"
    path.write_bytes(b"payload")
    session = _non_rng_session(
        effects=("declared_source_read",), sources=(_source(path),)
    )
    manager = session.files.open_read("fixture_source")
    with session.activate():
        lease = manager.__enter__()
        assert lease.read(1) == b"p"
    session.seal()
    assert lease.closed
    with pytest.raises(BrokerAccessError, match="closed"):
        lease.read()


def test_file_lease_refuses_raw_handle_escape_and_taints_session(
    tmp_path: Path,
) -> None:
    path = tmp_path / "opaque.bin"
    path.write_bytes(b"payload")
    session = _non_rng_session(
        effects=("declared_source_read",), sources=(_source(path),)
    )
    with session.activate(), session.files.open_read("fixture_source") as lease:
        with pytest.raises(BrokerAccessError, match="internals"):
            _ = lease._FileReadLease__stream  # type: ignore[attr-defined]  # noqa: SLF001
        assert lease.read() == b"payload"
    with pytest.raises(BrokerAccessError, match="recorded a refused access"):
        session.seal()
    assert session.receipt.status == "aborted"


@pytest.mark.parametrize("operation", ["read_text", "open_read"])
def test_invalid_declared_text_taints_the_session(
    tmp_path: Path, operation: str
) -> None:
    path = tmp_path / "invalid.bin"
    path.write_bytes(b"\xff")
    session = _non_rng_session(
        effects=("declared_source_read",), sources=(_source(path),)
    )
    with session.activate():
        with pytest.raises(BrokerAccessError, match="not valid"):
            if operation == "read_text":
                session.files.read_text("fixture_source")
            else:
                with session.files.open_read("fixture_source", binary=False):
                    pass
    receipt = session.seal(status="aborted")
    assert receipt.events[-1].reason_code == "source_text_decode_failed"


def test_file_broker_rejects_wrong_identity_and_symlink_binding(tmp_path: Path) -> None:
    path = tmp_path / "source.bin"
    path.write_bytes(b"before")
    declared = _source(path)
    path.write_bytes(b"after")
    session = _non_rng_session(effects=("declared_source_read",), sources=(declared,))
    with session.activate():
        with pytest.raises(BrokerAccessError, match="verified identity"):
            session.files.read_bytes("fixture_source")
    session.seal(status="aborted")

    target = tmp_path / "target.bin"
    target.write_bytes(b"target")
    link = tmp_path / "link.bin"
    link.symlink_to(target)
    with pytest.raises(BrokerContractError, match="symlink"):
        _source(link)


@pytest.mark.parametrize(
    "access",
    [
        lambda: os.environ["MICROCOSM_BROKER_SENTINEL"],
        lambda: os.environ.get("MICROCOSM_BROKER_SENTINEL"),
        lambda: os.getenv("MICROCOSM_BROKER_SENTINEL"),
        lambda: time.time(),
        lambda: time.monotonic(),
        lambda: time.perf_counter(),
        lambda: datetime_module.datetime.now(),
        lambda: os.stat("."),
        lambda: np.random.default_rng(),
        lambda: np.random.beta(1.0, 1.0),
    ],
)
def test_ambient_environment_clock_and_rng_access_is_refused_and_restored(
    monkeypatch: pytest.MonkeyPatch,
    access,
) -> None:
    monkeypatch.setenv("MICROCOSM_BROKER_SENTINEL", "secret-value")
    session = _non_rng_session()
    with session.activate():
        with pytest.raises(AmbientAccessError):
            access()
    receipt = session.seal(status="aborted")
    assert receipt.events[-1].broker == "ambient"
    assert receipt.events[-1].disposition == "refused"
    assert "secret-value" not in str(receipt.to_wire())
    assert os.environ["MICROCOSM_BROKER_SENTINEL"] == "secret-value"
    assert isinstance(time.time(), float)


def test_explicit_environment_and_clock_do_not_escape_reproducible_contract() -> None:
    session = _non_rng_session(
        effects=("declared_source_read",),
        environment={"WORKERS": "2"},
        clocks={"wall": 123.0},
    )
    with session.activate():
        with pytest.raises(BrokerAccessError, match="reproducible"):
            # A deterministic source-read grant authorizes files, not an
            # untyped environment dependency.
            session.environment.get("WORKERS")
        with pytest.raises(BrokerAccessError, match="reproducible"):
            session.clock.read("wall")
    session.seal(status="aborted")


def test_operational_receipts_do_not_change_spec_or_rng_behavior_identity(
    compiled_us: CompiledSpecIR,
) -> None:
    spec_sha256 = compiled_us.spec_binding.spec_sha256
    node_key = next(node.node_key for node in compiled_us.nodes)
    sessions = [
        _owner_session(
            compiled_us,
            "pipeline_operation",
            "assemble_stacked_spine",
            run_inputs={"sample_seed": 578},
        )
        for _ in range(2)
    ]
    behaviors = [session.rng.behavior_identity for session in sessions]
    for index, session in enumerate(sessions):
        with session.activate():
            token = session.rng.token("survey_sample_asec")
            with session.rng.generator(token) as generator:
                generator.random(index + 1)
                if index:
                    generator.random(1)
        session.seal()
    assert sessions[0].receipt.receipt_sha256 != sessions[1].receipt.receipt_sha256
    assert behaviors[0] == behaviors[1]
    assert all(isinstance(value, RNGBehaviorIdentity) for value in behaviors)
    assert compiled_us.spec_binding.spec_sha256 == spec_sha256
    assert next(node.node_key for node in compiled_us.nodes) == node_key
    assert "events" not in str(behaviors[0].to_wire())
    assert "receipt" not in str(behaviors[0].to_wire())


def test_rng_behavior_identity_binds_dynamic_time_period(
    compiled_us: CompiledSpecIR,
) -> None:
    site = next(
        site
        for site in compiled_us.seed_stream_map.sites
        if site.id == "scf_household_source_selector"
    )

    def session_for(period: int) -> BrokerSession:
        session = BrokerSession(
            owner=BrokerOwner("source_stage", "scf_wealth"),
            determinism="seeded",
            effects=("none",),
            protocol_id=compiled_us.seed_stream_map.protocol_id,
            protocol_sha256=compiled_us.seed_stream_map.implementation_sha256,
            seed_sites=(site,),
            run_inputs={"build_model_seed": 23},
            run_provenance_identity=_fixture_run_provenance_wire(),
            rng_invocation_plan={
                site.id: (RNGInvocation(f"period-{period}", {"time_period": period}),)
            },
        )
        return session

    first = session_for(2024)
    second = session_for(2025)
    assert first.rng.behavior_identity != second.rng.behavior_identity
    assert (
        first.rng.behavior_identity.identity_sha256
        != second.rng.behavior_identity.identity_sha256
    )
    with first.activate():
        token = first.rng.token(site.id, "period-2024")
        identity_site = first.rng.behavior_identity.to_wire()["sites"][0]
        assert isinstance(identity_site, dict)
        invocation = identity_site["invocations"][0]
        assert isinstance(invocation, dict)
        assert token.to_wire()["semantic_material"] == invocation["semantic_material"]
        with first.rng.generator(token) as generator:
            generator.random(1)
    first.seal()

    missing = BrokerSession(
        owner=BrokerOwner("source_stage", "scf_wealth"),
        determinism="seeded",
        effects=("none",),
        protocol_id=compiled_us.seed_stream_map.protocol_id,
        protocol_sha256=compiled_us.seed_stream_map.implementation_sha256,
        seed_sites=(site,),
        run_inputs={"build_model_seed": 23},
        run_provenance_identity=_fixture_run_provenance_wire(),
    )
    with missing.activate():
        with pytest.raises(BrokerAccessError, match="plan is exhausted"):
            missing.rng.token(site.id, "period-2024")
    missing.seal(status="aborted")


def test_receipt_digest_detects_tampering_without_becoming_a_node_key() -> None:
    session = _non_rng_session(node_key="c" * 64)
    with session.activate():
        with pytest.raises(AmbientAccessError):
            time.time()
    receipt = session.seal(status="aborted")
    forged = replace(receipt, status="complete")
    with pytest.raises(BrokerContractError, match="sequence|digest"):
        forged.validate()
    assert receipt.node_key == "c" * 64
    assert receipt.receipt_sha256 != receipt.node_key


def test_session_refuses_effect_or_owner_substitution(
    compiled_us: CompiledSpecIR,
) -> None:
    node = next(
        node
        for node in compiled_us.nodes
        if not node.seed_sites
        and thaw_json(node.capabilities).get("effects") == ["none"]
    )
    capabilities = thaw_json(node.capabilities)
    assert isinstance(capabilities, dict)
    session = BrokerSession.for_compiled_node(
        node, run_provenance_identity=_fixture_run_provenance_wire()
    )
    with pytest.raises(BrokerContractError, match="effects"):
        session.validate_executor_binding(
            node=node,
            determinism=str(capabilities["determinism"]),
            effects=("declared_source_read",),
            attempt=0,
            attempt_scope=None,
            require_byte_equivalence=True,
            run_provenance_identity=_fixture_run_provenance_wire(),
        )

    other = replace(node, id=f"{node.id}_other")
    with pytest.raises(BrokerContractError, match="owner"):
        session.validate_executor_binding(
            node=other,
            determinism=str(capabilities["determinism"]),
            effects=tuple(str(value) for value in capabilities["effects"]),
            attempt=0,
            attempt_scope=None,
            require_byte_equivalence=True,
            run_provenance_identity=_fixture_run_provenance_wire(),
        )


def test_session_authority_and_broker_facades_are_read_only() -> None:
    session = _non_rng_session()
    with pytest.raises(BrokerContractError, match="immutable"):
        session.effects = frozenset({"declared_source_read"})  # type: ignore[misc]
    with pytest.raises(BrokerContractError, match="immutable"):
        session._authority = replace(  # type: ignore[misc]  # noqa: SLF001
            session._authority,  # noqa: SLF001
            effects=frozenset({"declared_source_read"}),
        )
    with pytest.raises(BrokerContractError, match="immutable"):
        session.rng = object()  # type: ignore[misc]


def test_rng_normative_authority_is_recursively_immutable(
    compiled_us: CompiledSpecIR,
) -> None:
    session = _owner_session(
        compiled_us,
        "pipeline_operation",
        "assemble_stacked_spine",
        run_inputs={"sample_seed": 578},
    )
    with pytest.raises(BrokerContractError, match="immutable"):
        session.rng.protocol_sha256 = "f" * 64  # type: ignore[misc]
    with pytest.raises(TypeError):
        session.rng._contracts["survey_sample_asec"][  # noqa: SLF001
            "rng_family"
        ] = "forged"  # type: ignore[index]


def test_uninspectable_callable_shape_taints_the_session() -> None:
    class CallableKernel:
        def __call__(self) -> None:
            return None

    session = _non_rng_session()
    with pytest.raises(AmbientAccessError, match="directly inspectable"):
        session.validate_callable(CallableKernel(), role="kernel")
    receipt = session.seal(status="aborted")
    assert receipt.events[-1].reason_code == "uninspectable_callable_shape"


def test_prebound_ambient_scan_recurses_through_captured_mappings() -> None:
    box = {"clock": time.time}

    def kernel() -> float:
        return box["clock"]()

    session = _non_rng_session()
    with pytest.raises(AmbientAccessError, match="captures prohibited ambient"):
        session.validate_callable(kernel, role="kernel")
    receipt = session.seal(status="aborted")
    assert receipt.events[-1].reason_code == "prebound_ambient_access"


def test_prebound_scan_does_not_execute_hostile_container_protocols() -> None:
    touched = 0

    class HostileMapping(Mapping[str, object]):
        def __getitem__(self, key: str) -> object:
            raise KeyError(key)

        def __iter__(self) -> Iterator[str]:
            nonlocal touched
            touched += 1
            return iter(())

        def __len__(self) -> int:
            nonlocal touched
            touched += 1
            return 0

    hostile = HostileMapping()

    def kernel() -> object:
        return hostile

    session = _non_rng_session()
    with pytest.raises(AmbientAccessError, match="captures prohibited ambient"):
        session.validate_callable(kernel, role="kernel")
    assert touched == 0
    session.seal(status="aborted")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("broker", "unknown"),
        ("disposition", "ignored"),
        ("sequence", True),
        ("sequence", 0.5),
    ],
)
def test_broker_event_runtime_contract_refuses_unknown_values(
    field: str, value: object
) -> None:
    values = {
        "sequence": 0,
        "broker": "rng",
        "operation": "token",
        "resource": "fixture",
        "disposition": "allowed",
        "reason_code": "fixture",
    }
    values[field] = value
    with pytest.raises(BrokerContractError):
        BrokerAccessEvent(**values)  # type: ignore[arg-type]


def test_session_and_receipt_are_detached_from_caller_mappings() -> None:
    environment = {"TOKEN": "first"}
    session = _non_rng_session(
        determinism="nondeterministic",
        effects=("declared_source_read",),
        environment=environment,
        require_byte_equivalence=False,
    )
    environment["TOKEN"] = "mutated"
    with session.activate():
        assert session.environment.get("TOKEN") == "first"
    receipt = session.seal()
    wire = receipt.to_wire()
    copied = copy.deepcopy(wire)
    copied["status"] = "aborted"
    assert receipt.to_wire()["status"] == "complete"


def test_broker_runtime_contract_matches_provenance_and_integer_schema() -> None:
    invalid = _fixture_run_provenance_wire()
    invalid.update(
        {
            "identity_generation": 1,
            "source_grammar_receipt": {
                "schema_version": 1,
                "canonicalizer_version": 1,
                "migration_chain": [1],
            },
            "spec_binding": {
                "country": "us",
                "schema_id": "country_spec",
                "schema_version": 1,
                "canonicalizer_version": 1,
                "spec_sha256": "f" * 64,
                "attestation": "mirror-attested",
            },
        }
    )
    with pytest.raises(BrokerContractError, match="migration rows"):
        BrokerSession(
            owner=BrokerOwner("producer_node", "fixture_node"),
            determinism="deterministic",
            effects=("none",),
            protocol_id="legacy-v1",
            protocol_sha256="b" * 64,
            run_provenance_identity=invalid,
        )
    with pytest.raises(BrokerContractError, match="attempt"):
        BrokerSession(
            owner=BrokerOwner("producer_node", "fixture_node"),
            determinism="deterministic",
            effects=("none",),
            protocol_id="legacy-v1",
            protocol_sha256="b" * 64,
            run_provenance_identity=_fixture_run_provenance_wire(),
            attempt=0.5,  # type: ignore[arg-type]
        )


def test_broker_event_details_are_recursively_immutable() -> None:
    event = BrokerAccessEvent(
        sequence=0,
        broker="rng",
        operation="fixture",
        resource="fixture",
        disposition="allowed",
        reason_code="fixture",
        details={"nested": {"values": [1, 2]}},
    )
    nested = event.details["nested"]
    assert isinstance(nested, Mapping)
    with pytest.raises(TypeError):
        nested["values"] = ()  # type: ignore[index]


def test_pinned_physical_operation_runs_once_with_bound_input(
    compiled_us: CompiledSpecIR,
) -> None:
    node = next(
        node
        for node in compiled_us.nodes
        if thaw_json(node.capabilities).get("effects") == ["none"]
    )
    calls: list[str] = []
    input_binding_sha256 = sha256_json({"fixture": "physical-input"})

    def physical_operation() -> dict[str, int]:
        calls.append("called")
        return {"value": 42}

    session = BrokerSession.for_compiled_node(
        node,
        run_provenance_identity=_fixture_run_provenance_wire(),
        physical_operation=PhysicalOperation(
            function=physical_operation,
            implementation_sha256=node.kernel_implementation_sha256,
            input_binding_sha256=input_binding_sha256,
        ),
    )
    with pytest.raises(BrokerAccessError, match="active bound session"):
        session.kernel_view.run_physical_operation(
            input_binding_sha256=input_binding_sha256
        )
    with session.activate():
        assert session.kernel_view.run_physical_operation(
            input_binding_sha256=input_binding_sha256
        ) == {"value": 42}
        with pytest.raises(BrokerAccessError, match="only once"):
            session.kernel_view.run_physical_operation(
                input_binding_sha256=input_binding_sha256
            )
    assert calls == ["called"]
    receipt = session.seal(status="aborted")
    assert receipt.events[-1].reason_code == "physical_operation_repeat_invocation"


def test_physical_operation_requires_compiled_implementation_and_input_pins(
    compiled_us: CompiledSpecIR,
) -> None:
    node = next(
        node
        for node in compiled_us.nodes
        if thaw_json(node.capabilities).get("effects") == ["none"]
    )

    def physical_operation() -> None:
        return None

    input_binding_sha256 = sha256_json({"fixture": "physical-input"})
    with pytest.raises(BrokerContractError, match="implementation digest differs"):
        BrokerSession.for_compiled_node(
            node,
            run_provenance_identity=_fixture_run_provenance_wire(),
            physical_operation=PhysicalOperation(
                function=physical_operation,
                implementation_sha256="f" * 64,
                input_binding_sha256=input_binding_sha256,
            ),
        )

    session = BrokerSession.for_compiled_node(
        node,
        run_provenance_identity=_fixture_run_provenance_wire(),
        physical_operation=PhysicalOperation(
            function=physical_operation,
            implementation_sha256=node.kernel_implementation_sha256,
            input_binding_sha256=input_binding_sha256,
        ),
    )
    with session.activate():
        with pytest.raises(BrokerAccessError, match="differs from its contract"):
            session.kernel_view.run_physical_operation(input_binding_sha256="e" * 64)
    receipt = session.seal(status="aborted")
    assert receipt.events[-1].reason_code == "physical_input_binding_mismatch"


def test_physical_operation_seed_and_sink_scope_is_grant_bound(
    compiled_us: CompiledSpecIR, tmp_path: Path
) -> None:
    node = next(node for node in compiled_us.nodes if node.id == "primary_puf_qrf")
    input_binding_sha256 = sha256_json({"fixture": "seeded-sink-input"})
    target = tmp_path / "checkpoint.txt"

    def physical_operation() -> int:
        value = int(np.random.default_rng(7).integers(0, 100))
        target.write_text(str(value), encoding="utf-8")
        return value

    session = BrokerSession.for_compiled_node(
        node,
        run_provenance_identity=_fixture_run_provenance_wire(),
        physical_operation=PhysicalOperation(
            function=physical_operation,
            implementation_sha256=node.kernel_implementation_sha256,
            input_binding_sha256=input_binding_sha256,
            sink_roots=(tmp_path,),
        ),
    )
    with session.activate():
        result = session.kernel_view.run_physical_operation(
            input_binding_sha256=input_binding_sha256
        )
    receipt = session.seal()
    assert target.read_text(encoding="utf-8") == str(result)
    rng_events = [
        event
        for event in receipt.events
        if event.reason_code == "legacy_v1_physical_rng_grant"
    ]
    assert tuple(event.resource for event in rng_events) == tuple(
        site.id for site in node.seed_sites
    )
    assert any(
        event.reason_code == "legacy_v1_physical_sink_grant" for event in receipt.events
    )
    load_schema_registry().validate(
        receipt.to_wire(), "locks.schema.json#/$defs/broker_access_receipt"
    )


def test_physical_operation_source_effect_does_not_restore_ambient_reads(
    compiled_us: CompiledSpecIR, tmp_path: Path
) -> None:
    node = next(
        node
        for node in compiled_us.nodes
        if thaw_json(node.capabilities).get("determinism") == "deterministic"
        and thaw_json(node.capabilities).get("effects") == ["declared_source_read"]
    )
    source = tmp_path / "source.txt"
    source.write_text("snapshot me", encoding="utf-8")
    input_binding_sha256 = sha256_json({"fixture": "source-input"})

    def physical_operation() -> str:
        return source.read_text(encoding="utf-8")

    session = BrokerSession.for_compiled_node(
        node,
        run_provenance_identity=_fixture_run_provenance_wire(),
        physical_operation=PhysicalOperation(
            function=physical_operation,
            implementation_sha256=node.kernel_implementation_sha256,
            input_binding_sha256=input_binding_sha256,
        ),
    )
    with session.activate():
        with pytest.raises(AmbientAccessError, match="ambient file"):
            session.kernel_view.run_physical_operation(
                input_binding_sha256=input_binding_sha256
            )
    receipt = session.seal(status="aborted")
    assert receipt.status == "aborted"
    assert any(
        event.reason_code == "ambient_access_prohibited" for event in receipt.events
    )


def test_physical_operation_rejects_unsupported_authority_and_rng_return(
    compiled_us: CompiledSpecIR,
) -> None:
    pure_node = next(
        node
        for node in compiled_us.nodes
        if thaw_json(node.capabilities).get("effects") == ["none"]
    )

    def no_result() -> None:
        return None

    operation = PhysicalOperation(
        function=no_result,
        implementation_sha256=pure_node.kernel_implementation_sha256,
        input_binding_sha256="d" * 64,
    )
    capabilities = thaw_json(pure_node.capabilities)
    assert isinstance(capabilities, dict)
    capabilities["determinism"] = "nondeterministic"
    unsupported_node = replace(pure_node, capabilities=freeze_json(capabilities))
    with pytest.raises(BrokerContractError, match="does not support nondeterminism"):
        BrokerSession.for_compiled_node(
            unsupported_node,
            run_provenance_identity=_fixture_run_provenance_wire(),
            physical_operation=operation,
        )

    seeded_node = next(
        node
        for node in compiled_us.nodes
        if thaw_json(node.capabilities).get("determinism") == "seeded"
        and thaw_json(node.capabilities).get("effects") == ["declared_source_read"]
    )

    def return_generator() -> object:
        return np.random.default_rng(11)

    seeded_session = BrokerSession.for_compiled_node(
        seeded_node,
        run_provenance_identity=_fixture_run_provenance_wire(),
        physical_operation=PhysicalOperation(
            function=return_generator,
            implementation_sha256=seeded_node.kernel_implementation_sha256,
            input_binding_sha256="c" * 64,
        ),
    )
    with seeded_session.activate():
        with pytest.raises(BrokerAccessError, match="raw RNG authority"):
            seeded_session.kernel_view.run_physical_operation(
                input_binding_sha256="c" * 64
            )
    receipt = seeded_session.seal(status="aborted")
    assert any(
        event.reason_code == "physical_rng_authority_returned"
        for event in receipt.events
    )

    def seek_entropy() -> object:
        return np.random.SeedSequence()

    entropy_session = BrokerSession.for_compiled_node(
        seeded_node,
        run_provenance_identity=_fixture_run_provenance_wire(),
        physical_operation=PhysicalOperation(
            function=seek_entropy,
            implementation_sha256=seeded_node.kernel_implementation_sha256,
            input_binding_sha256="b" * 64,
        ),
    )
    with entropy_session.activate():
        with pytest.raises(BrokerAccessError, match="explicit seed material"):
            entropy_session.kernel_view.run_physical_operation(
                input_binding_sha256="b" * 64
            )
    entropy_receipt = entropy_session.seal(status="aborted")
    assert any(
        event.reason_code == "physical_rng_entropy_prohibited"
        for event in entropy_receipt.events
    )
