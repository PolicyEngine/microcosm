"""Columns the US release tool writes between its repeated take-up passes.

``tools/build_us_fiscal_refresh_release.py`` computes person ``uncapped_ssi``
for December (``_ssi_person_uncapped_amount``) and person
``is_medicaid_eligible`` (``_medicaid_person_eligibility``) once on the
pre-calibration frame and again on the calibrated export frame. Each pass runs
policyengine-us over every household in batches. The later array equals the
earlier one, selected by person ID, only when no column written between the two
passes is an input of the variable, when the variable reads no weight and no
population aggregate, and when calibration keeps whole households.

These helpers name the stages ``_main`` runs between the two passes, in build
order, with the table each stage writes to and the columns it writes there
(each module's declared output constant; a read of the stage code found no
column outside those constants). The engine-contract test intersects them with
the engine's static dependency graph, and the engine-workflow test writes them
onto a small frame and compares the arrays the release tool's own passes return.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from types import ModuleType

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.medicaid_take_up import US_MEDICAID_TAKE_UP_VARIABLE
from microcosm.build.us_runtime.org_wages import US_ORG_WAGES_OUTPUT_COLUMNS
from microcosm.build.us_runtime.other_health_insurance import (
    US_OTHER_HEALTH_INSURANCE_STAGE_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.scf_auto_loans import US_SCF_AUTO_LOAN_OUTPUT_COLUMNS
from microcosm.build.us_runtime.scf_education_debt import (
    US_SCF_EDUCATION_DEBT_OUTPUT_COLUMNS,
)
from microcosm.build.us_runtime.sipp_tips import US_SIPP_TIPS_OUTPUT_COLUMNS
from microcosm.build.us_runtime.sipp_vehicles import US_SIPP_VEHICLE_OUTPUT_COLUMNS
from microcosm.build.us_runtime.snap_take_up import US_SNAP_TAKE_UP_OUTPUT_COLUMN
from microcosm.build.us_runtime.ssi_take_up import US_SSI_TAKE_UP_OUTPUT_COLUMNS
from microcosm.build.us_runtime.voluntary_filing import (
    US_VOLUNTARY_FILING_OUTPUT_COLUMNS,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

#: The two engine variables the release tool computes before and after
#: calibration. December SSI is read at ``f"{PERIOD}-12"``; Medicaid
#: eligibility at ``PERIOD``.
SSI_PASS_VARIABLE = "uncapped_ssi"
MEDICAID_PASS_VARIABLE = "is_medicaid_eligible"


@dataclass(frozen=True)
class StageOutputs:
    """One ``_main`` stage that reassigns ``base_frame`` between the passes."""

    name: str
    #: The function ``_main`` calls to build the new ``base_frame``.
    builder_call: str
    #: The frame table the stage writes to.
    entity: str
    columns: tuple[str, ...]


def stages_between_ssi_passes(builder: ModuleType) -> tuple[StageOutputs, ...]:
    """Stages from the first December SSI pass to the export frame, in order.

    The first SSI pass feeds the SSI take-up stage, so that stage's output is
    written after it. The SCF education-debt stage runs only with
    ``--scf-education-debt``.
    """

    return (
        StageOutputs(
            "ssi_take_up",
            "with_us_ssi_take_up",
            "person",
            tuple(US_SSI_TAKE_UP_OUTPUT_COLUMNS),
        ),
        StageOutputs(
            "scf_auto_loans",
            "with_us_scf_auto_loan_inputs",
            "household",
            tuple(US_SCF_AUTO_LOAN_OUTPUT_COLUMNS),
        ),
        StageOutputs(
            "scf_education_debt",
            "_with_scf_education_debt_stage",
            "person",
            tuple(US_SCF_EDUCATION_DEBT_OUTPUT_COLUMNS),
        ),
        StageOutputs(
            "sipp_vehicles",
            "with_us_sipp_vehicle_inputs",
            "household",
            tuple(US_SIPP_VEHICLE_OUTPUT_COLUMNS),
        ),
        StageOutputs(
            "voluntary_filing",
            "with_us_voluntary_filing_input",
            "tax_unit",
            tuple(US_VOLUNTARY_FILING_OUTPUT_COLUMNS),
        ),
        StageOutputs(
            "sipp_tips",
            "with_us_sipp_tip_inputs",
            "person",
            tuple(US_SIPP_TIPS_OUTPUT_COLUMNS),
        ),
        StageOutputs(
            "org_wages",
            "with_us_org_wages_inputs",
            "person",
            tuple(US_ORG_WAGES_OUTPUT_COLUMNS),
        ),
        StageOutputs(
            "aca_marketplace",
            "_with_aca_marketplace_source_outputs",
            "tax_unit",
            tuple(builder.US_ACA_SOURCE_OUTPUT_COLUMNS),
        ),
        *stages_between_medicaid_passes(),
    )


def stages_between_medicaid_passes() -> tuple[StageOutputs, ...]:
    """Stages from the first Medicaid eligibility pass to the export frame.

    The first Medicaid pass runs inside ``_with_medicaid_take_up_outputs`` and
    feeds the Medicaid take-up stage, so that stage's output is written after
    it.
    """

    return (
        StageOutputs(
            "medicaid_take_up",
            "_with_medicaid_take_up_outputs",
            "person",
            (US_MEDICAID_TAKE_UP_VARIABLE,),
        ),
        StageOutputs(
            "snap_state_take_up",
            "_with_snap_state_take_up_outputs",
            "spm_unit",
            (US_SNAP_TAKE_UP_OUTPUT_COLUMN,),
        ),
        StageOutputs(
            "other_health_insurance",
            "with_us_other_health_insurance_inputs",
            "person",
            tuple(US_OTHER_HEALTH_INSURANCE_STAGE_OUTPUT_COLUMNS),
        ),
    )


@dataclass(frozen=True)
class MainFrameAssignments:
    """Where ``_main`` runs the passes and reassigns its frames, by line."""

    first_ssi_pass_line: int
    second_ssi_pass_line: int
    #: ``(line, called function)`` for each ``base_frame = f(...)`` (or
    #: ``base_frame, ... = f(...)``) in ``_main``, in source order.
    base_frame_calls: tuple[tuple[int, str], ...]
    #: ``(line, called function)`` for each ``export_frame = f(...)``.
    export_frame_calls: tuple[tuple[int, str], ...]
    #: The variable each SSI pass reads its frame from, in source order.
    ssi_pass_frames: tuple[str, ...]


def _called_name(call: ast.Call) -> str:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ast.dump(function)


def _assigned_names(target: ast.expr) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, ast.Tuple):
        return tuple(
            element.id for element in target.elts if isinstance(element, ast.Name)
        )
    return ()


def _statements_outside_nested_scopes(node: ast.AST):
    """Every node below ``node`` except bodies of nested functions and lambdas."""

    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield child
        yield from _statements_outside_nested_scopes(child)


def main_frame_assignments(builder: ModuleType) -> MainFrameAssignments:
    """Read ``_main``'s source for the SSI passes and frame reassignments."""

    # ``_main`` is a module-level function, so its source parses as it stands.
    source_lines, first_line = inspect.getsourcelines(builder._main)
    tree = ast.parse("".join(source_lines))
    (main,) = (
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_main"
    )
    ssi_passes: list[tuple[int, str]] = []
    base_frame_calls: list[tuple[int, str]] = []
    export_frame_calls: list[tuple[int, str]] = []
    for node in _statements_outside_nested_scopes(main):
        if isinstance(node, ast.Call):
            # A December SSI pass is a direct call, or the reuse of the
            # recorded first pass on another frame.
            called = _called_name(node)
            frame_argument = None
            if called == "_ssi_person_uncapped_amount":
                frame_argument = node.args[0]
            elif (
                called == "_reuse_household_pass"
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "ssi_pass_record"
            ):
                frame_argument = node.args[1]
            if frame_argument is not None:
                ssi_passes.append(
                    (
                        first_line + node.lineno - 1,
                        frame_argument.id
                        if isinstance(frame_argument, ast.Name)
                        else ast.dump(frame_argument),
                    )
                )
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        names = {name for target in node.targets for name in _assigned_names(target)}
        called = _called_name(node.value)
        line = first_line + node.lineno - 1
        if "base_frame" in names:
            base_frame_calls.append((line, called))
        if "export_frame" in names:
            export_frame_calls.append((line, called))
    ssi_passes.sort()
    if len(ssi_passes) != 2:
        raise AssertionError(
            f"_main should run exactly two December SSI passes: {ssi_passes}."
        )
    return MainFrameAssignments(
        first_ssi_pass_line=ssi_passes[0][0],
        second_ssi_pass_line=ssi_passes[1][0],
        base_frame_calls=tuple(sorted(base_frame_calls)),
        export_frame_calls=tuple(sorted(export_frame_calls)),
        ssi_pass_frames=tuple(frame for _, frame in ssi_passes),
    )


# ---------------------------------------------------------------------------
# A small real-engine frame: seven households over five states
# ---------------------------------------------------------------------------

#: Person ID of the self-employed California adult whose Medicaid eligibility
#: the other-health stage's premium attribution changes.
SELF_EMPLOYED_PERSON_ID = 5
#: Person ID of the disabled Texas adult with a little employment income.
DISABLED_WORKER_PERSON_ID = 4

#: (person_id, household, tax_unit, spm_unit, family, marital_unit, age)
_PERSONS = (
    (1, 1, 10, 100, 1000, 10000, 70.0),
    (2, 2, 20, 200, 2000, 20000, 68.0),
    (3, 2, 20, 200, 2000, 20000, 72.0),
    (4, 3, 30, 300, 3000, 30000, 45.0),
    (5, 4, 40, 400, 4000, 40000, 30.0),
    (6, 5, 50, 500, 5000, 50000, 35.0),
    (7, 5, 50, 500, 5000, 50001, 6.0),
    (8, 6, 60, 600, 6000, 60000, 40.0),
    (9, 6, 60, 600, 6000, 60001, 10.0),
    (10, 7, 70, 700, 7000, 70000, 50.0),
)
#: (household_id, state_fips)
_HOUSEHOLDS = (
    (1, 6),
    (2, 36),
    (3, 48),
    (4, 6),
    (5, 6),
    (6, 12),
    (7, 24),
)


def take_up_pass_frame() -> Frame:
    """Seven households that reach SSI, Medicaid and SNAP in 2024.

    One aged single person and one aged couple with small Social Security
    benefits, a disabled adult and a disabled child with low earnings in the
    household, a self-employed adult just above California's Medicaid income
    limit, a parent and child with low earnings, and one higher-income adult.
    """

    person_ids = np.asarray([row[0] for row in _PERSONS], dtype="int64")
    person = pd.DataFrame(
        {
            "person_id": person_ids,
            "person_household_id": np.asarray([row[1] for row in _PERSONS], "int64"),
            "person_tax_unit_id": np.asarray([row[2] for row in _PERSONS], "int64"),
            "person_spm_unit_id": np.asarray([row[3] for row in _PERSONS], "int64"),
            "person_family_id": np.asarray([row[4] for row in _PERSONS], "int64"),
            "person_marital_unit_id": np.asarray([row[5] for row in _PERSONS], "int64"),
            "age": np.asarray([row[6] for row in _PERSONS]),
            "employment_income_before_lsr": np.asarray(
                [0.0, 0.0, 0.0, 3_000.0, 0.0, 20_000.0, 0.0, 30_000.0, 0.0, 90_000.0]
            ),
            "self_employment_income_before_lsr": np.asarray(
                [0.0, 0.0, 0.0, 0.0, 25_000.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ),
            "social_security_retirement": np.asarray(
                [6_000.0, 9_000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ),
            "meets_ssi_disability_criteria": np.isin(person_ids, (4, 9)),
        }
    )
    tables = {"person": person}
    for entity in ("tax_unit", "spm_unit", "family", "marital_unit"):
        ids = np.unique(person[f"person_{entity}_id"].to_numpy())
        tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    tables["household"] = pd.DataFrame(
        {
            "household_id": np.asarray([row[0] for row in _HOUSEHOLDS], "int64"),
            "state_fips": np.asarray([row[1] for row in _HOUSEHOLDS], "int64"),
        }
    )
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), WeightKind.DESIGN
            )
        },
    )


def _stage_output_values(
    column: str, row_count: int, variant: int, system
) -> np.ndarray:
    """Values for one stage output; variants 0 and 1 differ on every row."""

    rows = np.arange(row_count)
    variable = system.variables.get(column)
    value_type = float if variable is None else variable.value_type
    if value_type is bool:
        return (rows + variant) % 2 == 0
    if value_type is int:
        return (1 + (rows + 3 * variant) % 9).astype(np.int64)
    if value_type is float:
        scale = 1_000.0 if variant == 0 else 2_750.0
        return (rows + 1) * scale
    raise TypeError(f"No test values for {column!r} of type {value_type!r}.")


def with_stage_outputs(
    frame: Frame,
    stages: tuple[StageOutputs, ...],
    *,
    variant: int,
    system,
) -> Frame:
    """``frame`` with every column the stages write set to one value variant."""

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for stage in stages:
        table = tables[stage.entity]
        for column in stage.columns:
            table[column] = _stage_output_values(column, len(table), variant, system)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
    )


def with_person_columns(frame: Frame, person_id: int, **values) -> Frame:
    """``frame`` with person columns set for one person (zero or False elsewhere)."""

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    person = tables["person"]
    selected = person["person_id"].to_numpy() == person_id
    for column, value in values.items():
        if column in person:
            updated = person[column].to_numpy().copy()
        else:
            updated = np.zeros(len(person), dtype=type(value))
        updated[selected] = value
        person[column] = updated
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
    )


def by_person_id(frame: Frame, values: np.ndarray) -> pd.Series:
    """Person-aligned pass output indexed by ``person_id``."""

    return pd.Series(values, index=frame.table("person")["person_id"].to_numpy())
