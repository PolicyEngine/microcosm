"""Single-authority closure gates for the typed US take-up contract."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator, Mapping
from functools import lru_cache
from pathlib import Path

import pytest

from microcosm.build.spec_engine import (
    CompiledSpecIR,
    SpecValidationError,
    compile_spec,
    load_bundle,
    load_schema_registry,
    project_legacy_take_up_contract,
    validate_take_up_semantics,
)
from microcosm.build.spec_engine.model import thaw_json
from microcosm.build.spec_engine.take_up_semantics import (
    project_legacy_take_up_identity,
)
from microcosm.build.us_runtime.take_up_contract import (
    load_legacy_take_up_contract_evidence,
    take_up_contract_identity,
)

pytest.importorskip(
    "policyengine_us",
    reason="live-engine oracle: the wheels gate's venv installs no engine",
    exc_type=ModuleNotFoundError,
)

ROOT = Path(__file__).resolve().parents[3]
US_ROOT = ROOT / "packages/microcosm-build/src/microcosm/build/us"


@lru_cache(maxsize=1)
def _compiled_us() -> CompiledSpecIR:
    return compile_spec(load_bundle("us"))


def _documents() -> tuple[dict[str, object], dict[str, object]]:
    resources = _compiled_us().resources_wire()
    return (
        copy.deepcopy(resources["take_up"]),
        copy.deepcopy(resources["sources"]),
    )


def _program(document: dict[str, object], program_id: str) -> dict[str, object]:
    programs = document["programs"]
    assert isinstance(programs, list)
    return next(row for row in programs if row["id"] == program_id)


def _steps(
    document: Mapping[str, object],
) -> Iterator[tuple[str, dict[str, object]]]:
    programs = document["programs"]
    assert isinstance(programs, list)
    for program_index, program in enumerate(programs):
        if "pipeline" in program:
            for step_index, step in enumerate(program["pipeline"]):
                yield f"take_up/programs/{program_index}/pipeline/{step_index}", step
        else:
            for segment_index, segment in enumerate(program["segments"]):
                for step_index, step in enumerate(segment["pipeline"]):
                    yield (
                        f"take_up/programs/{program_index}/segments/"
                        f"{segment_index}/pipeline/{step_index}",
                        step,
                    )


def _engine_abi_lock() -> dict[str, object]:
    lock = thaw_json(_compiled_us().generated_authorities["engine_abi_lock"])
    assert isinstance(lock, dict)
    return lock


def _project(
    take_up: Mapping[str, object],
    sources: Mapping[str, object],
    *,
    engine_abi_lock: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return project_legacy_take_up_contract(
        take_up,
        engine_abi_lock=(
            _engine_abi_lock() if engine_abi_lock is None else engine_abi_lock
        ),
        sources_document=sources,
    )


def test_generated_take_up_has_closed_coherent_semantics() -> None:
    take_up, sources = _documents()

    load_schema_registry().validate(take_up, "take_up.schema.json")
    validate_take_up_semantics(take_up, sources_document=sources)
    assert len(take_up["programs"]) == 13

    steps = list(_steps(take_up))
    assert len(steps) == 24
    assert sum("source_operation_ref" in step for _, step in steps) == 17


def test_source_backed_steps_are_thin_resolved_references() -> None:
    take_up, sources = _documents()
    stages = {stage["stage"]: stage["operations"] for stage in sources["stages"]}
    residual_fields = {
        "anchor_column",
        "calibration_review",
        "kernel",
        "kind",
        "rate_review",
        "source_operation_ref",
    }

    for pointer, step in _steps(take_up):
        reference = step.get("source_operation_ref")
        if reference is None:
            assert "operation_id" in step, pointer
            continue
        assert set(step) <= residual_fields, pointer
        operation = stages[reference["stage"]][reference["operation_index"]]
        assert operation["kind"] == reference["operation_id"], pointer

    for program in take_up["programs"]:
        assert "legacy_contract_fields" not in program
        assert "source_stage_ids" not in program


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("id", "duplicate program id"),
        ("variable", "program-to-variable mapping must be injective"),
    ],
)
def test_program_id_and_variable_mapping_is_unique(
    field: str,
    message: str,
) -> None:
    take_up, sources = _documents()
    programs = take_up["programs"]
    programs[1][field] = programs[0][field]

    with pytest.raises(
        SpecValidationError,
        match=rf"take_up/programs/1/{field}: {message}",
    ):
        validate_take_up_semantics(take_up, sources_document=sources)


def test_mixed_ownership_requires_segment_shape() -> None:
    take_up, sources = _documents()
    _program(take_up, "snap")["ownership"] = "mixed"

    with pytest.raises(
        SpecValidationError,
        match="mixed ownership requires segments",
    ):
        validate_take_up_semantics(take_up, sources_document=sources)


def test_mixed_segments_cannot_claim_the_same_scope_twice() -> None:
    take_up, sources = _documents()
    program = _program(take_up, "housing_assistance")
    segments = program["segments"]
    segments[1]["row_scope"] = copy.deepcopy(segments[0]["row_scope"])

    with pytest.raises(
        SpecValidationError,
        match=(
            "take_up/programs/11/segments/1/row_scope: duplicate mixed-ownership scope"
        ),
    ):
        validate_take_up_semantics(take_up, sources_document=sources)


def test_mixed_segments_must_represent_distinct_ownership_modes() -> None:
    take_up, sources = _documents()
    program = _program(take_up, "housing_assistance")
    segments = program["segments"]
    segments[1]["ownership"] = "measured"
    segments[1]["pipeline"][0]["kind"] = "measured_map"

    with pytest.raises(
        SpecValidationError,
        match="mixed ownership must cover at least two distinct ownership modes",
    ):
        validate_take_up_semantics(take_up, sources_document=sources)


def test_source_reference_stage_and_operation_must_resolve() -> None:
    take_up, sources = _documents()
    reference = _program(take_up, "snap")["pipeline"][0]["source_operation_ref"]
    reference["stage"] = "missing_stage"

    with pytest.raises(
        SpecValidationError,
        match=(
            "take_up/programs/0/pipeline/0/source_operation_ref/stage: "
            "dangling source stage 'missing_stage'"
        ),
    ):
        validate_take_up_semantics(take_up, sources_document=sources)

    take_up, sources = _documents()
    reference = _program(take_up, "snap")["pipeline"][0]["source_operation_ref"]
    reference["operation_id"] = "wrong_operation"
    with pytest.raises(
        SpecValidationError,
        match=(
            "take_up/programs/0/pipeline/0/source_operation_ref/operation_id: "
            "expected source operation kind 'derive_snap_take_up'"
        ),
    ):
        validate_take_up_semantics(take_up, sources_document=sources)


def test_final_owner_is_derived_from_the_last_source_backed_step() -> None:
    take_up, sources = _documents()
    _program(take_up, "snap")["final_owner_stage"] = "snap_take_up"

    with pytest.raises(
        SpecValidationError,
        match="take_up/programs/0/final_owner_stage: expected 'snap_state_take_up'",
    ):
        validate_take_up_semantics(take_up, sources_document=sources)


def test_ownership_rejects_an_incompatible_step_kind() -> None:
    take_up, sources = _documents()
    program = _program(take_up, "chip")
    program["pipeline"][0]["kind"] = "probability_seed"

    with pytest.raises(
        SpecValidationError,
        match="ownership 'engine' requires 'engine_default' steps",
    ):
        validate_take_up_semantics(take_up, sources_document=sources)


def test_typed_document_and_generated_abi_lock_reconstruct_frozen_evidence() -> None:
    take_up, sources = _documents()
    frozen_evidence = json.loads((US_ROOT / "take_up_contract.json").read_text())

    assert _project(take_up, sources) == frozen_evidence


def test_compiled_take_up_identity_is_byte_exact() -> None:
    take_up, sources = _documents()
    projected = _project(take_up, sources)

    assert project_legacy_take_up_identity(projected) == take_up_contract_identity(
        load_legacy_take_up_contract_evidence()
    )


def test_projection_refuses_bundle_to_abi_variable_drift() -> None:
    take_up, sources = _documents()
    mutated = _engine_abi_lock()
    mutated["programs"]["snap"]["variable"] = "takes_up_wrong_variable"

    with pytest.raises(
        SpecValidationError,
        match=(
            "take_up/programs/0/variable: bundle value "
            "'takes_up_snap_if_eligible' differs from engine_abi.lock"
        ),
    ):
        _project(take_up, sources, engine_abi_lock=mutated)


def test_every_normative_step_has_a_projection_pointer() -> None:
    take_up, sources = _documents()
    for pointer, _ in list(_steps(take_up)):
        mutated = copy.deepcopy(take_up)
        mutated_step = dict(_steps(mutated))[pointer]
        if "source_operation_ref" in mutated_step:
            del mutated_step["source_operation_ref"]["operation_id"]
            missing_pointer = f"{pointer}/source_operation_ref/operation_id"
        else:
            del mutated_step["operation_id"]
            missing_pointer = f"{pointer}/operation_id"
        with pytest.raises(SpecValidationError, match=missing_pointer):
            _project(mutated, sources)


def test_normative_owner_mutations_name_the_legacy_fields_they_change() -> None:
    take_up, sources = _documents()
    baseline = _project(take_up, sources)

    tanf = copy.deepcopy(take_up)
    _program(tanf, "tanf")["pipeline"][0]["rate"]["value"] = 0.218
    assert _project(tanf, sources)["programs"][1]["rate"]["value"] == 0.218
    assert baseline["programs"][1]["rate"]["value"] == 0.219

    aca = copy.deepcopy(take_up)
    _program(aca, "aca")["pipeline"][1]["rate_review"]["value"] = 0.673
    assert _project(aca, sources)["programs"][12]["rate"]["value"] == 0.673
    assert baseline["programs"][12]["rate"]["value"] == 0.672

    engine = copy.deepcopy(take_up)
    _program(engine, "chip")["pipeline"][0]["debt"]["rate_review"]["status"] = (
        "review_mutated"
    )
    assert _project(engine, sources)["programs"][4]["rate"]["status"] == (
        "review_mutated"
    )

    ssi = copy.deepcopy(take_up)
    _program(ssi, "ssi")["pipeline"][1]["target_table"] = "mutated_target"
    assert _project(ssi, sources)["programs"][7]["calibration"]["target_table"] == (
        "mutated_target"
    )


def test_closed_schema_rejects_the_retired_legacy_blob() -> None:
    take_up, _ = _documents()
    _program(take_up, "snap")["legacy_contract_fields"] = {"rate": {"value": 1}}

    with pytest.raises(SpecValidationError, match="legacy_contract_fields"):
        load_schema_registry().validate(take_up, "take_up.schema.json")
