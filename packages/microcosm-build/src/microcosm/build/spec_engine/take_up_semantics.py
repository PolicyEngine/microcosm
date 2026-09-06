"""Semantic closure and compatibility projection for typed take-up specs.

JSON Schema closes the shape of each pipeline step.  The relationships between
program ownership, segmented scopes, source stages, and the generated engine
ABI lock are cross-resource invariants, so they live in this pure front-end
module.  It deliberately imports neither ``CountrySpec`` nor the US runtime:
both use it, and keeping the dependency one-way prevents the generation tool
from recursing through the compatibility reader.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy

from .canonical import canonical_json_bytes, sha256_json
from .errors import SpecValidationError

_OWNERSHIP_KINDS = frozenset({"measured", "transferred", "modeled", "engine", "mixed"})
_STEP_KINDS = frozenset(
    {
        "assignment",
        "count_calibration",
        "delivery_gate",
        "engine_default",
        "imputed_transfer",
        "measured_map",
        "probability_seed",
    }
)
_MODELED_FINAL_STEP_KINDS = frozenset(
    {"assignment", "count_calibration", "delivery_gate", "probability_seed"}
)
_SYNTHETIC_FINAL_OWNER = {
    "engine": "policyengine_us_default",
    "modeled": "generic_take_up_seeder",
}
_DOCUMENTATION_FIELDS = frozenset(
    {"consumed_via", "engine_state_note", "followup", "notes", "scope_owner"}
)


def _mapping(value: object, *, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SpecValidationError(f"{location}: object required")
    return value


def _array(value: object, *, location: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise SpecValidationError(f"{location}: array required")
    return value


def _identifier(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise SpecValidationError(f"{location}: non-empty string required")
    return value


def _validate_owner_steps(
    *,
    ownership: str,
    steps: Sequence[Mapping[str, object]],
    location: str,
) -> None:
    kinds = [
        _identifier(step.get("kind"), location=f"{location}/{index}/kind")
        for index, step in enumerate(steps)
    ]
    for index, kind in enumerate(kinds):
        if kind not in _STEP_KINDS:
            raise SpecValidationError(
                f"{location}/{index}/kind: unsupported take-up step {kind!r}"
            )

    exact_kind = {
        "engine": "engine_default",
        "measured": "measured_map",
        "transferred": "imputed_transfer",
    }.get(ownership)
    if exact_kind is not None:
        for index, kind in enumerate(kinds):
            if kind != exact_kind:
                raise SpecValidationError(
                    f"{location}/{index}/kind: ownership {ownership!r} requires "
                    f"{exact_kind!r} steps; found {kind!r}"
                )
        return

    if ownership != "modeled":
        raise SpecValidationError(
            f"{location}: pipeline cannot use ownership {ownership!r}"
        )
    for index, kind in enumerate(kinds):
        if kind == "engine_default":
            raise SpecValidationError(
                f"{location}/{index}/kind: modeled ownership cannot contain an "
                "engine_default step"
            )
    if kinds[-1] not in _MODELED_FINAL_STEP_KINDS:
        raise SpecValidationError(
            f"{location}/{len(kinds) - 1}/kind: modeled ownership requires a "
            "modeled final step; found "
            f"{kinds[-1]!r}"
        )


def _validate_pipeline(
    *,
    pipeline_value: object,
    final_owner_value: object,
    ownership: str,
    location: str,
    source_stages: Mapping[str, Sequence[object]],
) -> tuple[str, ...]:
    pipeline = _array(pipeline_value, location=location)
    if not pipeline:
        raise SpecValidationError(f"{location}: pipeline must not be empty")
    steps = [
        _mapping(step, location=f"{location}/{index}")
        for index, step in enumerate(pipeline)
    ]
    _validate_owner_steps(ownership=ownership, steps=steps, location=location)

    referenced: list[str] = []
    for index, step in enumerate(steps):
        reference_value = step.get("source_operation_ref")
        if reference_value is None:
            continue
        step_location = f"{location}/{index}"
        reference = _mapping(
            reference_value, location=f"{step_location}/source_operation_ref"
        )
        stage_id = _identifier(
            reference.get("stage"),
            location=f"{step_location}/source_operation_ref/stage",
        )
        _effective_step(step, location=step_location, source_stages=source_stages)
        referenced.append(stage_id)

    final_owner = _identifier(
        final_owner_value,
        location=f"{location.rsplit('/', 1)[0]}/final_owner_stage",
    )
    if referenced:
        expected_owner = referenced[-1]
    else:
        expected_owner = _SYNTHETIC_FINAL_OWNER.get(ownership)
        if expected_owner is None:
            raise SpecValidationError(
                f"{location}: ownership {ownership!r} requires at least one "
                "source-backed step"
            )
    if final_owner != expected_owner:
        raise SpecValidationError(
            f"{location.rsplit('/', 1)[0]}/final_owner_stage: expected "
            f"{expected_owner!r} from the ordered pipeline; found "
            f"{final_owner!r}"
        )
    return tuple(referenced)


def validate_take_up_semantics(
    take_up: Mapping[str, object],
    *,
    sources_document: Mapping[str, object],
) -> None:
    """Validate take-up invariants that span programs and typed sources.

    The function is deterministic and side-effect free.  Errors use stable
    JSON-pointer-like locations so mutation tests identify the exact field
    whose relationship became invalid.
    """

    source_stages = _source_stage_map(sources_document)

    programs = _array(take_up.get("programs"), location="take_up/programs")
    seen_ids: dict[str, int] = {}
    seen_variables: dict[str, int] = {}
    for program_index, value in enumerate(programs):
        location = f"take_up/programs/{program_index}"
        program = _mapping(value, location=location)
        program_id = _identifier(program.get("id"), location=f"{location}/id")
        variable = _identifier(program.get("variable"), location=f"{location}/variable")
        if program_id in seen_ids:
            raise SpecValidationError(
                f"{location}/id: duplicate program id {program_id!r}; first "
                f"declared at take_up/programs/{seen_ids[program_id]}/id"
            )
        if variable in seen_variables:
            raise SpecValidationError(
                f"{location}/variable: program-to-variable mapping must be "
                f"injective; {variable!r} first declared at "
                f"take_up/programs/{seen_variables[variable]}/variable"
            )
        seen_ids[program_id] = program_index
        seen_variables[variable] = program_index

        ownership = _identifier(
            program.get("ownership"), location=f"{location}/ownership"
        )
        if ownership not in _OWNERSHIP_KINDS:
            raise SpecValidationError(
                f"{location}/ownership: unsupported ownership {ownership!r}"
            )
        if ownership == "mixed":
            if "pipeline" in program or "final_owner_stage" in program:
                raise SpecValidationError(
                    f"{location}: mixed ownership requires segments and forbids "
                    "program-level pipeline/final_owner_stage"
                )
            segments = _array(program.get("segments"), location=f"{location}/segments")
            if len(segments) < 2:
                raise SpecValidationError(
                    f"{location}/segments: mixed ownership requires at least "
                    "two scope segments"
                )
            first_scope: dict[bytes, int] = {}
            segment_owners: set[str] = set()
            for segment_index, segment_value in enumerate(segments):
                segment_location = f"{location}/segments/{segment_index}"
                segment = _mapping(segment_value, location=segment_location)
                if "row_scope" not in segment:
                    raise SpecValidationError(
                        f"{segment_location}/row_scope: scope required"
                    )
                try:
                    scope_key = canonical_json_bytes(segment["row_scope"])
                except (TypeError, ValueError) as error:
                    raise SpecValidationError(
                        f"{segment_location}/row_scope: canonical JSON scope required"
                    ) from error
                if scope_key in first_scope:
                    raise SpecValidationError(
                        f"{segment_location}/row_scope: duplicate mixed-ownership "
                        "scope; first declared at "
                        f"{location}/segments/{first_scope[scope_key]}/row_scope"
                    )
                first_scope[scope_key] = segment_index
                segment_owner = _identifier(
                    segment.get("ownership"),
                    location=f"{segment_location}/ownership",
                )
                if segment_owner == "mixed" or segment_owner not in _OWNERSHIP_KINDS:
                    raise SpecValidationError(
                        f"{segment_location}/ownership: a segment requires one "
                        "non-mixed ownership"
                    )
                segment_owners.add(segment_owner)
                _validate_pipeline(
                        pipeline_value=segment.get("pipeline"),
                        final_owner_value=segment.get("final_owner_stage"),
                        ownership=segment_owner,
                        location=f"{segment_location}/pipeline",
                        source_stages=source_stages,
                )
            if len(segment_owners) < 2:
                raise SpecValidationError(
                    f"{location}/segments: mixed ownership must cover at least "
                    "two distinct ownership modes"
                )
        else:
            if "segments" in program:
                raise SpecValidationError(
                    f"{location}: ownership {ownership!r} requires a program-level "
                    "pipeline/final_owner_stage and forbids segments"
                )
            _validate_pipeline(
                    pipeline_value=program.get("pipeline"),
                    final_owner_value=program.get("final_owner_stage"),
                    ownership=ownership,
                    location=f"{location}/pipeline",
                    source_stages=source_stages,
            )


def _program_steps(
    row: Mapping[str, object], *, location: str
) -> tuple[tuple[Mapping[str, object], str], ...]:
    """Return every typed step with its stable bundle pointer."""

    if "pipeline" in row:
        values = _array(row.get("pipeline"), location=f"{location}/pipeline")
        return tuple(
            (
                _mapping(value, location=f"{location}/pipeline/{index}"),
                f"{location}/pipeline/{index}",
            )
            for index, value in enumerate(values)
        )
    segments = _array(row.get("segments"), location=f"{location}/segments")
    result: list[tuple[Mapping[str, object], str]] = []
    for segment_index, segment_value in enumerate(segments):
        segment_location = f"{location}/segments/{segment_index}"
        segment = _mapping(segment_value, location=segment_location)
        for step_index, value in enumerate(
            _array(segment.get("pipeline"), location=f"{segment_location}/pipeline")
        ):
            step_location = f"{segment_location}/pipeline/{step_index}"
            result.append((_mapping(value, location=step_location), step_location))
    return tuple(result)


def _checked_step_kinds(
    steps: Sequence[tuple[Mapping[str, object], str]],
) -> tuple[str, ...]:
    kinds: list[str] = []
    for step, location in steps:
        kind = _identifier(step.get("kind"), location=f"{location}/kind")
        if kind not in _STEP_KINDS:
            raise SpecValidationError(
                f"{location}/kind: unsupported take-up step {kind!r}"
            )
        if "source_operation_ref" in step:
            reference = _mapping(
                step["source_operation_ref"],
                location=f"{location}/source_operation_ref",
            )
            _identifier(
                reference.get("operation_id"),
                location=f"{location}/source_operation_ref/operation_id",
            )
        else:
            _identifier(step.get("operation_id"), location=f"{location}/operation_id")
        _identifier(step.get("kernel"), location=f"{location}/kernel")
        kinds.append(kind)
    return tuple(kinds)


def _source_stage_map(
    sources_document: Mapping[str, object],
) -> dict[str, Sequence[object]]:
    stages = _array(sources_document.get("stages"), location="sources/stages")
    result: dict[str, Sequence[object]] = {}
    for index, value in enumerate(stages):
        location = f"sources/stages/{index}"
        stage = _mapping(value, location=location)
        stage_id = _identifier(stage.get("stage"), location=f"{location}/stage")
        if stage_id in result:
            raise SpecValidationError(f"{location}/stage: duplicate stage {stage_id!r}")
        result[stage_id] = _array(
            stage.get("operations"), location=f"{location}/operations"
        )
    return result


def _effective_step(
    step: Mapping[str, object],
    *,
    location: str,
    source_stages: Mapping[str, Sequence[object]],
) -> Mapping[str, object]:
    reference_value = step.get("source_operation_ref")
    if reference_value is None:
        return step
    reference = _mapping(
        reference_value, location=f"{location}/source_operation_ref"
    )
    stage_id = _identifier(
        reference.get("stage"), location=f"{location}/source_operation_ref/stage"
    )
    if stage_id not in source_stages:
        raise SpecValidationError(
            f"{location}/source_operation_ref/stage: dangling source stage {stage_id!r}"
        )
    operation_index = reference.get("operation_index")
    if not isinstance(operation_index, int) or isinstance(operation_index, bool):
        raise SpecValidationError(
            f"{location}/source_operation_ref/operation_index: integer required"
        )
    operations = source_stages[stage_id]
    if not 0 <= operation_index < len(operations):
        raise SpecValidationError(
            f"{location}/source_operation_ref/operation_index: {operation_index} "
            f"is out of range for source stage {stage_id!r}"
        )
    operation = _mapping(
        operations[operation_index],
        location=f"sources/stages/{stage_id}/operations/{operation_index}",
    )
    operation_id = _identifier(
        reference.get("operation_id"),
        location=f"{location}/source_operation_ref/operation_id",
    )
    if operation.get("kind") != operation_id:
        raise SpecValidationError(
            f"{location}/source_operation_ref/operation_id: expected source "
            f"operation kind {operation.get('kind')!r}; found {operation_id!r}"
        )
    return operation


def _derived_populace_treatment(
    row: Mapping[str, object],
    *,
    ownership: str,
    steps: Sequence[tuple[Mapping[str, object], str]],
    kinds: tuple[str, ...],
    location: str,
) -> str:
    if ownership == "engine":
        if kinds != ("engine_default",):
            raise SpecValidationError(
                f"{location}/pipeline: engine ownership requires exactly one "
                "engine_default step"
            )
        debt = _mapping(steps[0][0].get("debt"), location=f"{steps[0][1]}/debt")
        return _identifier(
            debt.get("populace_treatment"),
            location=f"{steps[0][1]}/debt/populace_treatment",
        )
    if ownership in {"measured", "transferred", "mixed"}:
        return "out_of_scope"
    if ownership != "modeled":
        raise SpecValidationError(
            f"{location}/ownership: unsupported ownership {ownership!r}"
        )
    if "dependence" in row:
        if kinds != ("probability_seed",):
            raise SpecValidationError(
                f"{location}/pipeline: dependent generic seed must contain one "
                "probability_seed step"
            )
        return "seed"
    if "delivery_gate" in kinds:
        return "count_calibrated"
    if kinds == ("assignment", "count_calibration"):
        return "count_calibrated"
    return "out_of_scope"


def _single_step_with(
    steps: Sequence[tuple[Mapping[str, object], str]],
    field: str,
    *,
    program_location: str,
) -> tuple[Mapping[str, object], str] | None:
    matches = [(step, location) for step, location in steps if field in step]
    if len(matches) > 1:
        raise SpecValidationError(
            f"{program_location}: {field} must have exactly one typed owner"
        )
    return matches[0] if matches else None


def _single_effective_step_with(
    steps: Sequence[tuple[Mapping[str, object], str]],
    field: str,
    *,
    program_location: str,
    source_stages: Mapping[str, Sequence[object]],
) -> tuple[Mapping[str, object], str, Mapping[str, object]] | None:
    matches = [
        (step, location, effective)
        for step, location in steps
        if field
        in (
            effective := _effective_step(
                step, location=location, source_stages=source_stages
            )
        )
    ]
    if len(matches) > 1:
        raise SpecValidationError(
            f"{program_location}: {field} must have exactly one typed owner"
        )
    return matches[0] if matches else None


def _project_reviewed_rate(
    *,
    ownership: str,
    treatment: str,
    steps: Sequence[tuple[Mapping[str, object], str]],
    location: str,
    source_stages: Mapping[str, Sequence[object]],
) -> dict[str, object]:
    if ownership in {"measured", "transferred", "mixed"}:
        return {"status": "not_used_measured_source"}
    if ownership == "engine":
        debt = _mapping(steps[0][0].get("debt"), location=f"{steps[0][1]}/debt")
        return deepcopy(
            dict(
                _mapping(
                    debt.get("rate_review"),
                    location=f"{steps[0][1]}/debt/rate_review",
                )
            )
        )
    if treatment == "seed":
        rate_owner = _single_effective_step_with(
            steps,
            "rate",
            program_location=location,
            source_stages=source_stages,
        )
        if rate_owner is None:
            raise SpecValidationError(f"{location}/pipeline: seeded rate required")
        return deepcopy(
            dict(
                _mapping(
                    rate_owner[2]["rate"], location=f"{rate_owner[1]}/rate"
                )
            )
        )

    rate_review = _single_step_with(
        steps, "rate_review", program_location=location
    )
    source_rate = _single_effective_step_with(
        steps,
        "take_up_rate",
        program_location=location,
        source_stages=source_stages,
    )
    if source_rate is not None:
        result = deepcopy(
            dict(
                _mapping(
                    source_rate[2]["take_up_rate"],
                    location=f"{source_rate[1]}/take_up_rate",
                )
            )
        )
        if rate_review is not None:
            for key, value in _mapping(
                rate_review[0]["rate_review"],
                location=f"{rate_review[1]}/rate_review",
            ).items():
                if key in result:
                    raise SpecValidationError(
                        f"{rate_review[1]}/rate_review/{key}: duplicates "
                        "source-stage rate authority"
                    )
                result[str(key)] = deepcopy(value)
        return result
    if rate_review is None:
        raise SpecValidationError(
            f"{location}/pipeline: reviewed rate has no typed owner"
        )
    return deepcopy(
        dict(
            _mapping(
                rate_review[0]["rate_review"],
                location=f"{rate_review[1]}/rate_review",
            )
        )
    )


def _project_reviewed_calibration(
    *,
    treatment: str,
    steps: Sequence[tuple[Mapping[str, object], str]],
    location: str,
    source_stages: Mapping[str, Sequence[object]],
) -> dict[str, object] | None:
    if treatment != "count_calibrated":
        if _single_step_with(
            steps, "calibration_review", program_location=location
        ) is not None:
            raise SpecValidationError(
                f"{location}/pipeline: calibration review requires "
                "count_calibrated treatment"
            )
        return None

    review_owner = _single_step_with(
        steps, "calibration_review", program_location=location
    )
    if review_owner is None:
        raise SpecValidationError(
            f"{location}/pipeline: calibration_review required"
        )
    review = deepcopy(
        dict(
            _mapping(
                review_owner[0]["calibration_review"],
                location=f"{review_owner[1]}/calibration_review",
            )
        )
    )
    delivery = next(
        ((step, pointer) for step, pointer in steps if step.get("kind") == "delivery_gate"),
        None,
    )
    if delivery is not None:
        seed = next(
            (
                (step, pointer)
                for step, pointer in steps
                if step.get("kind") == "probability_seed"
            ),
            None,
        )
        if seed is None:
            raise SpecValidationError(
                f"{location}/pipeline: delivery gate requires probability seed"
            )
        effective_seed = _effective_step(
            seed[0], location=seed[1], source_stages=source_stages
        )
        result = {
            "anchor": deepcopy(seed[0].get("anchor_column")),
            "targets": [deepcopy(delivery[0].get("target_table"))],
            "target_table": deepcopy(delivery[0].get("target_table")),
            "target_source": deepcopy(effective_seed.get("target_source")),
            "target_period": deepcopy(effective_seed.get("target_period")),
            "target_measure": deepcopy(effective_seed.get("target_measure")),
            "target_role": deepcopy(effective_seed.get("rate_target_role")),
            "age_bands": deepcopy(effective_seed.get("age_bands")),
        }
    else:
        calibration_steps = [
            (step, pointer)
            for step, pointer in steps
            if step.get("kind") == "count_calibration"
        ]
        if len(calibration_steps) != 1:
            raise SpecValidationError(
                f"{location}/pipeline: expected one count_calibration step"
            )
        step, pointer = calibration_steps[0]
        effective = _effective_step(
            step, location=pointer, source_stages=source_stages
        )
        result = {
            "anchor": deepcopy(effective.get("preserve_true_anchor")),
            "targets": deepcopy(effective.get("targets")),
        }
        if result["anchor"] is None or result["targets"] is None:
            raise SpecValidationError(
                f"{pointer}: calibration anchor and targets are required"
            )
    collision = sorted(set(result).intersection(review))
    if collision:
        raise SpecValidationError(
            f"{review_owner[1]}/calibration_review: duplicates typed fields "
            f"{collision!r}"
        )
    result.update(review)
    return result


def project_legacy_take_up_contract(
    typed_contract: Mapping[str, object],
    *,
    engine_abi_lock: Mapping[str, object],
    sources_document: Mapping[str, object],
) -> dict[str, object]:
    """Compile typed ownership/steps plus the reviewed ABI lock to legacy JSON.

    The frozen ``take_up_contract.json`` is comparison evidence only.  Program
    treatment, rate, seed method, and calibration blocks are derived from the
    typed pipeline; no generic compatibility-payload mapping is accepted.
    """

    metadata = _mapping(
        typed_contract.get("legacy_contract_metadata"),
        location="take_up/legacy_contract_metadata",
    )
    programs = _array(typed_contract.get("programs"), location="take_up/programs")
    lock_programs = _mapping(
        engine_abi_lock.get("programs"), location="engine_abi.lock/programs"
    )
    source_stages = _source_stage_map(sources_document)

    typed_ids: list[str] = []
    typed_variables: list[str] = []
    projected_programs: list[dict[str, object]] = []
    for index, value in enumerate(programs):
        location = f"take_up/programs/{index}"
        row = _mapping(value, location=location)
        program_id = _identifier(row.get("id"), location=f"{location}/id")
        variable = _identifier(row.get("variable"), location=f"{location}/variable")
        ownership = _identifier(
            row.get("ownership"), location=f"{location}/ownership"
        )
        steps = _program_steps(row, location=location)
        kinds = _checked_step_kinds(steps)
        treatment = _derived_populace_treatment(
            row,
            ownership=ownership,
            steps=steps,
            kinds=kinds,
            location=location,
        )
        rate = _project_reviewed_rate(
            ownership=ownership,
            treatment=treatment,
            steps=steps,
            location=location,
            source_stages=source_stages,
        )
        calibration = _project_reviewed_calibration(
            treatment=treatment,
            steps=steps,
            location=location,
            source_stages=source_stages,
        )
        documentation = _mapping(
            row.get("documentation"), location=f"{location}/documentation"
        )
        unknown_documentation = sorted(set(documentation) - _DOCUMENTATION_FIELDS)
        if unknown_documentation:
            raise SpecValidationError(
                f"{location}/documentation: unsupported fields "
                f"{unknown_documentation!r}"
            )

        if program_id not in lock_programs:
            raise SpecValidationError(
                f"{location}/id: {program_id!r} is absent from engine_abi.lock"
            )
        abi = _mapping(
            lock_programs[program_id],
            location=f"engine_abi.lock/programs/{program_id}",
        )
        locked_variable = _identifier(
            abi.get("variable"),
            location=f"engine_abi.lock/programs/{program_id}/variable",
        )
        if locked_variable != variable:
            raise SpecValidationError(
                f"{location}/variable: bundle value {variable!r} differs from "
                f"engine_abi.lock value {locked_variable!r}"
            )
        engine_facts: dict[str, object] = {}
        for key in ("entity", "value_type", "default", "engine_class"):
            if key not in abi:
                raise SpecValidationError(
                    f"engine_abi.lock/programs/{program_id}/{key}: value required"
                )
            engine_facts[key] = deepcopy(abi[key])
        projection = {
            "variable": variable,
            **engine_facts,
            "populace_treatment": treatment,
        }
        if treatment == "seed":
            projection["seed_method"] = (
                "calibrated_bernoulli_by_children"
                if _single_step_with(
                    steps, "rate_selector", program_location=location
                )
                is not None
                else "calibrated_bernoulli"
            )
        projection["rate"] = rate
        if calibration is not None:
            projection["calibration"] = calibration
        projection.update(deepcopy(dict(documentation)))
        projected_programs.append(projection)
        typed_ids.append(program_id)
        typed_variables.append(variable)

    duplicate_ids = sorted({item for item in typed_ids if typed_ids.count(item) > 1})
    duplicate_variables = sorted(
        {item for item in typed_variables if typed_variables.count(item) > 1}
    )
    if duplicate_ids:
        raise SpecValidationError(
            f"take_up/programs: duplicate program ids {duplicate_ids!r}"
        )
    if duplicate_variables:
        raise SpecValidationError(
            "take_up/programs: program-to-variable mapping must be injective; "
            f"duplicates={duplicate_variables!r}"
        )
    extra_lock_ids = sorted(set(lock_programs) - set(typed_ids))
    if extra_lock_ids:
        raise SpecValidationError(
            "take_up/programs: mapping is not total over engine_abi.lock; "
            f"missing program ids={extra_lock_ids!r}"
        )

    for key in ("policy", "asserted_engine", "doctrine"):
        if key not in metadata:
            raise SpecValidationError(
                f"take_up/legacy_contract_metadata/{key}: value required"
            )
    version = typed_contract.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise SpecValidationError("take_up/version: integer required")
    country = _identifier(typed_contract.get("country"), location="take_up/country")
    return {
        "version": version,
        "country": country,
        "policy": deepcopy(metadata["policy"]),
        "asserted_engine": deepcopy(metadata["asserted_engine"]),
        "doctrine": deepcopy(metadata["doctrine"]),
        "programs": projected_programs,
    }


def project_legacy_take_up_identity(
    legacy_contract: Mapping[str, object],
) -> dict[str, object]:
    """Project the exact constants-era identity from the compiled contract."""

    version = legacy_contract.get("version")
    country = legacy_contract.get("country")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise SpecValidationError(
            "take_up legacy projection/version: positive integer required"
        )
    if not isinstance(country, str) or not country:
        raise SpecValidationError(
            "take_up legacy projection/country: identifier required"
        )
    asserted = _mapping(
        legacy_contract.get("asserted_engine"),
        location="take_up legacy projection/asserted_engine",
    )
    programs = _array(
        legacy_contract.get("programs"),
        location="take_up legacy projection/programs",
    )
    if not all(isinstance(program, Mapping) for program in programs):
        raise SpecValidationError(
            "take_up legacy projection/programs: object rows required"
        )
    return {
        "version": version,
        "country": country,
        "resource_sha256": sha256_json(legacy_contract),
        "asserted_constraint": str(asserted.get("constraint", "")),
        "inventory_built_against": str(
            asserted.get("inventory_built_against", "")
        ),
        "programs": [deepcopy(dict(program)) for program in programs],
    }


__all__ = [
    "project_legacy_take_up_contract",
    "project_legacy_take_up_identity",
    "validate_take_up_semantics",
]
