"""ACA take-up and plan ratio for the ACS tax units of the ACS local lane.

The ACS transfer leaves both ACA inputs to the runtime, and the local lane
ran no ACA stage, so every ACS tax unit reached the engine pass with
``takes_up_aca_if_eligible`` and ``selected_marketplace_plan_benchmark_ratio``
missing, and the reviewed-null fill gave them the engine defaults, ``True``
and ``1.0`` (microcosm#1720). Every eligible ACS unit then received the full
premium tax credit. In the county proof of concept of 17 November 2026 the
pooled file started at 1.85 times the state SOI return count and 1.42 times
the amount, and the release solve removed the excess through the weights of
eligible ACS units instead of choosing takers among them (the microcosm#170
mechanism).

This stage assigns both inputs on ACS tax units with the donor's own method,
the fiscal lane's ``aca_marketplace_inputs`` manifest stage, against what
the donor rows leave of the stage's state counts. It fills only missing ACS
cells; donor values and any stored ACS value are kept.

- **Engine pre-pass.** Like the fiscal lane, the stage reads, for each tax
  unit, the weight of its people eligible for the credit, the credit before
  take-up, the benchmark (SLCSP) premium and the reported premium: the
  fiscal lane's ACA tax-unit table. The caller supplies it as a callable
  over a view of whole households (the builder owns the simulations and
  their household batching). The stage calls it on the donor households
  first, with their stored flags, then on the ACS households, where a
  missing take-up cell reads ``True`` and a missing ratio ``1.0``, the
  engine defaults. On policyengine-us 2.145.3 ``aca_ptc`` is gated on
  filing, and a unit that takes up a positive credit counts as filing for
  it (``files_return_for_aca_ptc``), so with the flag ``True`` it is the
  credit before take-up, the value the fiscal lane's stage reads. The pass
  runs after the SSI and Medicaid assignment: Medicaid eligibility bars the
  credit (``gov.aca.ineligible_coverage``) and reads SSI receipt.
- **Residual targets.** The donor H5's flags were drawn against the full
  state counts, and pool assembly then scaled every donor weight by
  ``1 - acs_share``. Each ACS target is the residual: the state count minus
  the donor rows' contribution, floored at zero. The contribution is summed
  as the stage sums the ACS rows: on the weight column of the operation that
  reads the table (eligible people for the CMS recipients, households for
  the SOI returns, credit times household weight for the SOI amounts), over
  the units that operation's variable and domain select, at this frame's
  pre-calibration weights. Where the ACS capacity is below a residual the
  shortfall is recorded; where the donor rows alone exceed a count the
  excess is recorded.
- **The stage** runs on the ACS tax units against the residual tables,
  through :func:`~microcosm.build.source_runtime.run_source_stage` with the
  US handlers: a draw at the CMS residual rate, a calibration to the CMS
  residual counts, then, where both SOI tables exist, a joint calibration
  that resets each state and fills its residual SOI return count with the
  units whose credit per return is closest to the residual average, the
  draw only breaking ties. So where SOI rows exist they decide the
  selection, as they do on the donor, and the CMS people the takers carry
  are reported (microcosm#168). The
  ratio is (premium + credit) / SLCSP for takers, clipped to 0.5-1.5, and
  1.0 otherwise; without a bronze table the stage stops after the clip, as
  the fiscal lane does.
- **Anchor.** Staging does not load the PUMS item that separates
  subsidized Marketplace coverage (``HIMRKS``, direct purchase with or
  without a premium subsidy). When that anchor is absent the fiscal lane
  preserves none, because observed Marketplace coverage is broader than
  subsidized coverage (``_aca_source_person_table``); this stage follows
  that rule, so every ACS taker is drawn. A stored ACS take-up cell
  (staging writes none) is kept: a stored ``True`` is a preserved anchor,
  and a unit stored ``False`` stays out of the stage.
- **Draws** come from seeded blake2b uniforms on the donor stage's stream
  (``aca:takes_up_aca_if_eligible``), keyed on
  ``acs_2024_1yr:SERIALNO:<lowest SPORDER of the unit>``, so a unit's draw
  depends only on the build seed and its own record.

The receipt records, per table and state, the count, the donor rows'
contribution, the residual, the ACS capacity and delivery, the pooled total
and any shortfall or excess, the people per return of each set of rows, and
a digest of the ACS assignment. :func:`acs_local_aca_take_up_signal_gate`
grades the flags and the receipt. The lane reports that verdict and never
stops on it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_runtime import SourceRuntimeConfig, run_source_stage
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.acs_transfer import ASEC_PUF_DONOR_SPINE
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "ACS_LOCAL_ACA_PLAN_RATIO_COLUMN",
    "ACS_LOCAL_ACA_RETURNS_RELATIVE_TOLERANCE",
    "ACS_LOCAL_ACA_STAGE",
    "ACS_LOCAL_ACA_TAKE_UP_COLUMN",
    "ACS_LOCAL_ACA_TAKE_UP_COLUMNS",
    "ACS_LOCAL_ACA_TAKE_UP_GATE_NAME",
    "ACS_LOCAL_ACA_TAKE_UP_ISSUE",
    "ACS_LOCAL_ACA_TAKE_UP_METHOD",
    "AcaTaxUnitTable",
    "acs_local_aca_take_up_assignment",
    "acs_local_aca_take_up_signal_gate",
    "with_acs_local_aca_take_up",
    "with_recorded_acs_local_aca_take_up",
]

ACS_LOCAL_ACA_TAKE_UP_ISSUE = "microcosm#1720"
ACS_LOCAL_ACA_TAKE_UP_GATE_NAME = "acs_local_aca_take_up_signal"
ACS_LOCAL_ACA_TAKE_UP_METHOD = (
    "donor_aca_marketplace_stage_on_acs_tax_units_at_donor_residual_targets"
)
#: The fiscal lane's ACA manifest stage, run here on the ACS tax units.
ACS_LOCAL_ACA_STAGE = "aca_marketplace_inputs"
ACS_LOCAL_ACA_TAKE_UP_COLUMN = "takes_up_aca_if_eligible"
ACS_LOCAL_ACA_PLAN_RATIO_COLUMN = "selected_marketplace_plan_benchmark_ratio"
#: The tax-unit inputs this stage owns on ACS rows: the manifest stage's
#: declared outputs, in its order.
ACS_LOCAL_ACA_TAKE_UP_COLUMNS: tuple[str, ...] = (
    ACS_LOCAL_ACA_TAKE_UP_COLUMN,
    ACS_LOCAL_ACA_PLAN_RATIO_COLUMN,
)
#: Grading only: where the ACS capacity exceeds a state's residual in the
#: binding table, the delivered ACS count should be within this share of the
#: residual (or within one unit's weight, the greedy fill's step). The lane
#: reports the verdict and never stops on it.
ACS_LOCAL_ACA_RETURNS_RELATIVE_TOLERANCE = 0.02

_TAKE_UP = ACS_LOCAL_ACA_TAKE_UP_COLUMN
_RATIO = ACS_LOCAL_ACA_PLAN_RATIO_COLUMN
#: The engine defaults a missing cell reads in the ACS pre-pass view.
_TAKE_UP_DEFAULT = True
_RATIO_DEFAULT = 1.0
_STATE = "state_fips"
_ID = "tax_unit_id"
_PERSON_TAX_UNIT = "person_tax_unit_id"
_SPORDER = "SPORDER"
_SERIALNO = "SERIALNO"
#: The deprecated alias the fiscal lane falls back on for observed coverage.
_COVERAGE_ALIAS = "has_marketplace_health_coverage"
_DRAW_KEY_FORMAT = f"{ACS_2024_1YR_SPINE}:SERIALNO:<lowest SPORDER of the unit>"
_WEIGHTS_BASIS = "acs_rows_pre_calibration_frame_household_weights"
_TARGET_BASIS = (
    "each state count minus the donor rows' contribution (stored flag and "
    "ratio x engine eligibility and credit on the donor households x "
    "pre-calibration frame weights, summed on the weight column of the stage "
    "operation that reads the table), floored at 0"
)
_ANCHOR_RULE = (
    "none: staging does not load the PUMS subsidy item HIMRKS, and the "
    "fiscal lane preserves no anchor when that item is absent "
    "(_aca_source_person_table: observed Marketplace coverage is broader "
    "than subsidized coverage); a stored ACS take-up True is kept as an "
    "anchor and a stored False stays out of the stage"
)
#: The columns of the fiscal lane's ACA tax-unit table the stage reads.
_ENGINE_TABLE_COLUMNS = (
    _ID,
    _STATE,
    "tax_unit_weight",
    "household_weight",
    "is_aca_ptc_eligible",
    "health_insurance_premiums_without_medicare_part_b",
    "assigned_aca_ptc",
    "weighted_assigned_aca_ptc",
    "slcsp",
    "aca_take_up_rate",
)
_ELIGIBLE = "is_aca_ptc_eligible"
_CREDIT = "assigned_aca_ptc"
_HOUSEHOLD_WEIGHT = "household_weight"
#: A receipt's residual must be its count minus the donor contribution; the
#: two are written from the same floats, so only a changed formula differs.
_RESIDUAL_RELATIVE_TOLERANCE = 1e-9
#: A saturated state assigns every unit in the domain: delivery equals
#: capacity up to floating-point summation order.
_SATURATED_RELATIVE_TOLERANCE = 1e-6
_STATE_NUMBERS = ("target", "donor", "residual", "acs_capacity", "acs_delivered")

#: The fiscal lane's ACA tax-unit table over a view of whole households,
#: given the target tables its take-up rate and person weights read.
AcaTaxUnitTable = Callable[[Frame, Mapping[str, pd.DataFrame]], pd.DataFrame]


@dataclass(frozen=True)
class _TableRole:
    """How the manifest stage counts one target table.

    ``variable`` is the operation's boolean (a column or ``column < x``),
    ``domain`` the units it may move, and ``weight`` the tax-unit column the
    table's counts are summed on.
    """

    operation: str
    variable: str
    weight: str
    domain: str | None


@dataclass(frozen=True)
class _StagePlan:
    """What the manifest stage reads, as this stage needs it."""

    stage: Any
    roles: Mapping[str, _TableRole]
    draw_column: str
    draw_stream: str
    aggregates: tuple[str, ...]
    anchor: str
    late_tables: frozenset[str]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _finite(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _stage_plan() -> _StagePlan:
    """The manifest stage and what its operations read."""

    from microcosm.build.us_runtime import (  # local: package init owns the manifest
        US_SOURCE_MANIFEST,
    )

    stage = US_SOURCE_MANIFEST.stage_map()[ACS_LOCAL_ACA_STAGE]
    if tuple(stage.outputs) != ACS_LOCAL_ACA_TAKE_UP_COLUMNS:
        raise ValueError(
            f"The {ACS_LOCAL_ACA_STAGE} stage declares outputs "
            f"{list(stage.outputs)}, not {list(ACS_LOCAL_ACA_TAKE_UP_COLUMNS)}."
        )
    roles: dict[str, _TableRole] = {}
    draws: set[str] = set()
    aggregates: tuple[str, ...] = ()
    anchor = ""
    stream = ""
    late: set[str] = set()
    clipped = False
    for operation in stage.operations:
        params = operation.parameters
        if "draw" in params:
            draws.add(str(params["draw"]))
        if operation.kind == "aggregate_person_to_tax_unit":
            aggregates = tuple(str(name) for name in params["aggregates"])
        elif operation.kind == "assign_binary_from_rate":
            anchor = str(params["reported_true_anchor"])
            # The handler's own stream when the draw column is absent.
            stream = f"aca:{params['output']}"
        elif operation.kind == "support_clip":
            clipped = True
        reads: list[tuple[str, object]] = []
        if operation.kind == "calibrate_binary_assignment":
            reads = [(str(name), params["weight"]) for name in params["targets"]]
        elif operation.kind == "calibrate_binary_assignment_joint_targets":
            reads = [
                (str(name), params["count_weight"]) for name in params["count_targets"]
            ] + [
                (str(name), params["amount_weight"])
                for name in params["amount_targets"]
            ]
        for name, weight in reads:
            roles.setdefault(
                name,
                _TableRole(
                    operation=operation.kind,
                    variable=str(params["variable"]),
                    weight=str(weight),
                    domain=params.get("domain"),
                ),
            )
            if clipped:
                late.add(name)
    if len(draws) != 1 or not anchor or not stream or not aggregates:
        raise ValueError(
            f"The {ACS_LOCAL_ACA_STAGE} stage no longer reads one draw column, a "
            "reported anchor and its person aggregates; review this stage."
        )
    return _StagePlan(
        stage=stage,
        roles=roles,
        draw_column=next(iter(draws)),
        draw_stream=stream,
        aggregates=aggregates,
        anchor=anchor,
        late_tables=frozenset(late),
    )


def _checked_tables(
    target_tables: Mapping[str, pd.DataFrame], plan: _StagePlan
) -> dict[str, pd.DataFrame]:
    """One finite, nonnegative count per zero-padded state in each table."""

    unread = sorted(set(target_tables) - set(plan.roles))
    if unread:
        raise ValueError(
            f"No operation of {ACS_LOCAL_ACA_STAGE} reads target table(s) "
            f"{unread}; their residuals cannot be summed."
        )
    if not target_tables:
        raise ValueError("ACS local ACA take-up requires its state target tables.")
    checked: dict[str, pd.DataFrame] = {}
    for name, table in target_tables.items():
        missing = [column for column in (_STATE, "target") if column not in table]
        if missing:
            raise ValueError(f"ACA target table {name} lacks column(s) {missing}.")
        states = table[_STATE].astype(str).str.zfill(2)
        values = pd.to_numeric(table["target"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if states.duplicated().any():
            raise ValueError(
                f"ACA target table {name} repeats state(s) "
                f"{sorted(states[states.duplicated()])}."
            )
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"ACA target table {name} must be finite and >= 0.")
        checked[name] = pd.DataFrame({_STATE: states.to_numpy(), "target": values})
    return checked


def _boolean_cells(values: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(flags, present, invalid)`` of a nullable boolean column."""

    present = values.notna().to_numpy(dtype=bool)
    valid = present & values.isin([0, 1]).to_numpy(dtype=bool)
    flags = np.zeros(len(values), dtype=bool)
    flags[valid] = values[valid].astype(bool).to_numpy(dtype=bool)
    return flags, present, present & ~valid


def _ratio_cells(values: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ratios, present, invalid)``; a valid ratio is finite and positive."""

    numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    present = values.notna().to_numpy(dtype=bool)
    valid = present & np.isfinite(numbers) & (numbers > 0)
    return np.where(valid, numbers, np.nan), present, present & ~valid


def _engine_table(
    function: AcaTaxUnitTable,
    view: Frame,
    tables: Mapping[str, pd.DataFrame],
    *,
    label: str,
) -> pd.DataFrame:
    """The callable's table, one row per view tax unit, in the view's order."""

    raw = function(view, tables)
    missing = [column for column in _ENGINE_TABLE_COLUMNS if column not in raw]
    if missing:
        raise ValueError(
            f"The {label} ACA engine pre-pass table lacks column(s) {missing}."
        )
    ids = view.table("tax_unit")[_ID].to_numpy()
    table = raw.loc[:, list(_ENGINE_TABLE_COLUMNS)].set_index(_ID)
    if not table.index.is_unique or len(table) != len(ids):
        raise ValueError(
            f"The {label} ACA engine pre-pass returned {len(table)} rows for "
            f"{len(ids)} tax units, or repeated a tax unit."
        )
    table = table.reindex(ids)
    numbers = table.drop(columns=[_STATE, _ELIGIBLE]).to_numpy(dtype=np.float64)
    if table[_STATE].isna().any() or not np.isfinite(numbers).all():
        raise ValueError(
            f"The {label} ACA engine pre-pass does not cover every tax unit with "
            "a state and finite values."
        )
    table = table.reset_index()
    table[_STATE] = table[_STATE].astype(str).str.zfill(2)
    table[_ELIGIBLE] = table[_ELIGIBLE].fillna(False).astype(bool)
    return table


def _selected(table: pd.DataFrame, role: _TableRole, *, label: str) -> np.ndarray:
    """The units ``role``'s operation counts, as the stage's handlers read it."""

    from microcosm.build.us_runtime.source_runtime import (
        _binary_calibration_value_state,
        _domain_mask,
    )

    values = _binary_calibration_value_state(table, role.variable, label=label)
    domain = _domain_mask(table, role.domain, label=label)
    return (values.values(table) & domain).to_numpy(dtype=bool)


def _domain(table: pd.DataFrame, role: _TableRole, *, label: str) -> np.ndarray:
    from microcosm.build.us_runtime.source_runtime import _domain_mask

    return _domain_mask(table, role.domain, label=label).to_numpy(dtype=bool)


def _state_sums(
    table: pd.DataFrame, selected: np.ndarray, column: str
) -> dict[str, float]:
    weights = pd.to_numeric(table[column], errors="coerce").fillna(0.0)
    sums = (
        pd.DataFrame(
            {_STATE: table[_STATE].to_numpy(), "w": np.where(selected, weights, 0.0)}
        )
        .groupby(_STATE, sort=True)["w"]
        .sum()
    )
    return {str(state): float(value) for state, value in sums.items()}


def _state_max(
    table: pd.DataFrame, selected: np.ndarray, column: str
) -> dict[str, float]:
    weights = pd.to_numeric(table[column], errors="coerce").fillna(0.0)
    frame = pd.DataFrame(
        {_STATE: table[_STATE].to_numpy(), "w": np.where(selected, weights, 0.0)}
    )
    return {
        str(state): float(value)
        for state, value in frame.groupby(_STATE, sort=True)["w"].max().items()
    }


def _with_stored(
    table: pd.DataFrame, take_up: np.ndarray, ratio: np.ndarray
) -> pd.DataFrame:
    result = table.copy()
    result[_TAKE_UP] = np.asarray(take_up, dtype=bool)
    result[_RATIO] = np.asarray(ratio, dtype=np.float64)
    return result


def _acs_draw_keys(frame: Frame, tax_unit_ids: np.ndarray) -> np.ndarray:
    """``acs_2024_1yr:SERIALNO:<lowest SPORDER>`` per ACS tax unit; unique."""

    person = frame.table("person")
    household = frame.table("household")
    # Three columns only: never a second copy of the ACS person rows.
    members = person.loc[
        person[_PERSON_TAX_UNIT].isin(tax_unit_ids),
        [_PERSON_TAX_UNIT, "person_household_id", _SPORDER],
    ]
    serial = members["person_household_id"].map(
        household.set_index("household_id")[_SERIALNO]
    )
    order = pd.to_numeric(members[_SPORDER], errors="coerce")
    if serial.isna().any() or order.isna().any():
        raise ValueError(
            "Every ACS person needs its household SERIALNO and its SPORDER to key "
            "the ACA take-up draws."
        )
    units = pd.DataFrame(
        {
            "unit": members[_PERSON_TAX_UNIT].to_numpy(),
            "serial": serial.astype(str).to_numpy(),
            "order": order.astype(np.int64).to_numpy(),
        }
    )
    if (units.groupby("unit")["serial"].nunique() > 1).any():
        raise ValueError("An ACS tax unit spans two ACS households.")
    first = units.sort_values(["unit", "order"]).groupby("unit", sort=False).first()
    keys = (
        f"{ACS_2024_1YR_SPINE}:" + first["serial"] + ":" + first["order"].astype(str)
    ).reindex(tax_unit_ids)
    if keys.isna().any():
        raise ValueError("An ACS tax unit has no person to key its ACA draw.")
    if keys.duplicated().any():
        raise ValueError(
            f"{int(keys.duplicated().sum())} ACS tax unit draw key(s) repeat; "
            f"{_DRAW_KEY_FORMAT} must identify one unit."
        )
    return keys.to_numpy(dtype=object)


def _with_tax_unit(frame: Frame, tax_unit: pd.DataFrame) -> Frame:
    return Frame(
        {
            entity: tax_unit if entity == "tax_unit" else frame.table(entity)
            for entity in frame.entities
        },
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _filled_flags(
    values: pd.Series, missing: np.ndarray, assigned: np.ndarray
) -> pd.Series:
    if not missing.any():
        return values
    filled = values.astype(object)
    filled.loc[missing] = assigned[missing]
    return filled.astype(bool) if filled.notna().all() else filled


def _filled_ratios(
    values: pd.Series, missing: np.ndarray, assigned: np.ndarray
) -> pd.Series:
    if not missing.any():
        return values
    filled = pd.to_numeric(values, errors="coerce").astype(np.float64)
    filled.loc[missing] = assigned[missing]
    return filled


def _assignment_sha256(
    tax_unit_ids: np.ndarray, take_up: np.ndarray, ratio: np.ndarray
) -> str:
    """Digest of the ACS tax units' two inputs, to prove two passes agree."""

    table = pd.DataFrame(
        {
            _ID: np.asarray(tax_unit_ids),
            _TAKE_UP: np.asarray(take_up, dtype=bool),
            _RATIO: np.asarray(ratio, dtype=np.float64),
        }
    )
    hashed = pd.util.hash_pandas_object(table, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _acs_masks(frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    """ACS persons and ACS tax units, by their origin tags."""

    person = frame.table("person")
    tax_unit = frame.table("tax_unit")
    tags = [spine_column(entity) for entity in ("person", "tax_unit")]
    for table, tag in zip((person, tax_unit), tags, strict=True):
        if tag not in table or table[tag].isna().any():
            raise ValueError(
                f"ACS local ACA take-up requires complete origin tags: {tag}."
            )
    return (
        person[tags[0]].eq(ACS_2024_1YR_SPINE).to_numpy(dtype=bool),
        tax_unit[tags[1]].eq(ACS_2024_1YR_SPINE).to_numpy(dtype=bool),
    )


def _rows(
    names: tuple[str, ...],
    tables: Mapping[str, pd.DataFrame],
    sums: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, list[dict[str, Any]]]:
    """Per table, one accounting row per target state."""

    result: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        rows: list[dict[str, Any]] = []
        table = tables[name]
        for state, target in zip(table[_STATE], table["target"], strict=True):
            target = float(target)
            donor = sums["donor"][name].get(state, 0.0)
            residual = max(target - donor, 0.0)
            capacity = sums["capacity"][name].get(state, 0.0)
            delivered = sums["delivered"][name].get(state, 0.0)
            pooled = donor + delivered
            rows.append(
                {
                    _STATE: str(state),
                    "target": target,
                    "donor": donor,
                    "donor_share_of_target": donor / target if target > 0 else None,
                    "residual": residual,
                    "acs_capacity": capacity,
                    "acs_delivered": delivered,
                    "acs_max_unit_weight": sums["max_unit"][name].get(state, 0.0),
                    "pooled": pooled,
                    "pooled_relative_error": (
                        (pooled - target) / target if target > 0 else None
                    ),
                    "capacity_shortfall": max(residual - capacity, 0.0),
                    "donor_excess": max(donor - target, 0.0),
                }
            )
        result[name] = rows
    return result


def _composition(
    plan: _StagePlan,
    tables: Mapping[str, pd.DataFrame],
    sums: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, Any] | None:
    """People per return, for each set of rows (microcosm#168)."""

    people = [
        name
        for name, role in plan.roles.items()
        if name in tables
        and role.operation == "calibrate_binary_assignment"
        and role.variable == _TAKE_UP
    ]
    returns = [
        name
        for name, role in plan.roles.items()
        if name in tables
        and role.operation == "calibrate_binary_assignment_joint_targets"
        and role.weight == _HOUSEHOLD_WEIGHT
    ]
    if not people or not returns:
        return None
    person_table, return_table = people[0], returns[0]

    def ratio(numerator: float, denominator: float) -> float | None:
        return numerator / denominator if denominator > 0 else None

    def total(kind: str, name: str) -> float:
        return float(sum(sums[kind][name].values()))

    return {
        "people_table": person_table,
        "returns_table": return_table,
        "target": ratio(
            float(tables[person_table]["target"].sum()),
            float(tables[return_table]["target"].sum()),
        ),
        "donor": ratio(total("donor", person_table), total("donor", return_table)),
        "acs": ratio(
            total("delivered", person_table), total("delivered", return_table)
        ),
        "pooled": ratio(
            total("donor", person_table) + total("delivered", person_table),
            total("donor", return_table) + total("delivered", return_table),
        ),
    }


def _binding_table(plan: _StagePlan, tables: Mapping[str, pd.DataFrame]) -> str:
    """The count table whose residuals the stage's last take-up step meets."""

    joint = [
        name
        for name, role in plan.roles.items()
        if role.operation == "calibrate_binary_assignment_joint_targets"
        and role.weight == _HOUSEHOLD_WEIGHT
        and name in tables
    ]
    amounts = [
        name
        for name, role in plan.roles.items()
        if role.operation == "calibrate_binary_assignment_joint_targets"
        and role.weight != _HOUSEHOLD_WEIGHT
        and name in tables
    ]
    if joint and amounts:
        return joint[0]
    first = [
        name
        for name, role in plan.roles.items()
        if role.operation == "calibrate_binary_assignment"
        and role.variable == _TAKE_UP
        and name in tables
    ]
    if not first:
        raise ValueError(
            f"No target table the {ACS_LOCAL_ACA_STAGE} stage's take-up "
            "calibration reads was given."
        )
    return first[0]


def with_acs_local_aca_take_up(
    frame: Frame,
    *,
    seed: int,
    target_tables: Mapping[str, pd.DataFrame],
    aca_tax_units: AcaTaxUnitTable,
    period: int = 2024,
) -> tuple[Frame, dict[str, Any]]:
    """Fill missing ACS-row ACA take-up and plan ratio; keep every other cell.

    Args:
        frame: The local lane's multispine US frame, after the SSI and
            Medicaid take-up stage. Its person and tax-unit tables must carry
            complete origin tags, its tax-unit table both ACA columns, its
            person table ``SPORDER`` and observed Marketplace coverage, and
            its household table ``SERIALNO`` and ``state_fips``. ACS persons
            must fill whole households.
        seed: The build seed for the draws.
        target_tables: The fiscal lane's ACA state tables
            (``state_fips``, ``target``) by name, as the ACS local release
            reads them from its compiled registry.
        aca_tax_units: Returns the fiscal lane's ACA tax-unit table for
            every tax unit of the view it is given, with the target tables
            its rate and person weights read: first the donor households
            (their stored flags, the full tables), then the ACS households
            (missing cells read ``True`` and ``1.0``, the residual tables).
        period: The year the stage runs for.

    Returns:
        The frame with filled ACS cells and a JSON-ready receipt, including
        each table's and state's donor contribution, residual, delivery,
        shortfall and excess, and a digest of the ACS assignment.

    Raises:
        ValueError: If the frame is not US-schema, origin tags or a required
            column are missing, an ACS household holds a non-ACS person, an
            ACS row lacks what its draw key reads, the targets are malformed,
            or the callable returns a table that does not cover the view.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("ACS local ACA take-up requires the US schema.")
    plan = _stage_plan()
    tables = _checked_tables(target_tables, plan)
    person = frame.table("person")
    tax_unit = frame.table("tax_unit")
    household = frame.table("household")
    absent = [
        f"tax_unit.{column}" for column in (_TAKE_UP, _RATIO) if column not in tax_unit
    ]
    absent += [
        f"person.{column}"
        for column in (_PERSON_TAX_UNIT, _SPORDER)
        if column not in person
    ]
    absent += [
        f"household.{column}"
        for column in (_SERIALNO, _STATE)
        if column not in household
    ]
    if absent:
        raise ValueError(
            f"ACS local ACA take-up requires column(s) {absent}; a missing "
            "take-up column would reach the engine as universal take-up."
        )
    acs_persons, acs_units = _acs_masks(frame)
    if not acs_units.any():
        raise ValueError("ACS local ACA take-up found no ACS tax units.")
    household_ids = person["person_household_id"]
    acs_households = pd.unique(household_ids[acs_persons])
    mixed = household_ids.isin(acs_households).to_numpy(dtype=bool) & ~acs_persons
    if mixed.any():
        raise ValueError(
            f"{int(mixed.sum())} non-ACS person(s) share a household with ACS "
            "persons; the engine pre-pass evaluates whole ACS households."
        )
    unit_ids = tax_unit[_ID].to_numpy()
    acs_ids = unit_ids[acs_units]
    member_units = person.loc[acs_persons, _PERSON_TAX_UNIT]
    if set(member_units) != set(acs_ids):
        raise ValueError(
            "The ACS-tagged tax units are not exactly the tax units of the ACS persons."
        )
    keys = _acs_draw_keys(frame, acs_ids)

    stored_flags, flag_present, flag_invalid = _boolean_cells(tax_unit[_TAKE_UP])
    stored_ratio, ratio_present, ratio_invalid = _ratio_cells(tax_unit[_RATIO])
    if (flag_invalid & acs_units).any() or (ratio_invalid & acs_units).any():
        raise ValueError(
            "Stored ACS ACA cells must be a boolean take-up flag and a finite "
            f"positive ratio; {int((flag_invalid & acs_units).sum())} flag and "
            f"{int((ratio_invalid & acs_units).sum())} ratio cell(s) are not."
        )

    # The donor rows' contribution sets each ACS residual target. A donor
    # cell the release did not store reads its engine default, as the
    # engine pass would read it; the receipt counts such cells.
    donor_units = ~acs_units
    donor_counts = {
        "tax_units": int(donor_units.sum()),
        "missing_take_up_cells": int((donor_units & ~flag_present).sum()),
        "invalid_take_up_cells": int((donor_units & flag_invalid).sum()),
        "missing_ratio_cells": int((donor_units & ~ratio_present).sum()),
        "invalid_ratio_cells": int((donor_units & ratio_invalid).sum()),
    }
    sums: dict[str, dict[str, dict[str, float]]] = {
        kind: {name: {} for name in tables}
        for kind in ("donor", "capacity", "delivered", "max_unit")
    }
    if donor_units.any():
        view = frame.select(~acs_persons)
        engine = _engine_table(aca_tax_units, view, tables, label="donor")
        del view
        donor_ids = engine[_ID].to_numpy()
        position = pd.Series(np.arange(len(unit_ids)), index=unit_ids)
        at = position.reindex(donor_ids).to_numpy()
        donor_flags = np.where(
            flag_present[at] & ~flag_invalid[at], stored_flags[at], _TAKE_UP_DEFAULT
        )
        donor_ratio = np.where(
            ratio_present[at] & ~ratio_invalid[at], stored_ratio[at], _RATIO_DEFAULT
        )
        evaluated = _with_stored(engine, donor_flags, donor_ratio)
        donor_counts["eligible_tax_units"] = int(
            (evaluated[_ELIGIBLE] & (evaluated[_CREDIT] > 0)).sum()
        )
        for name in tables:
            role = plan.roles[name]
            chosen = _selected(evaluated, role, label=f"donor {name}")
            sums["donor"][name] = _state_sums(evaluated, chosen, role.weight)
        del engine, evaluated
    residual_tables = {
        name: pd.DataFrame(
            {
                _STATE: table[_STATE].to_numpy(),
                "target": np.maximum(
                    table["target"].to_numpy(dtype=np.float64)
                    - np.asarray(
                        [sums["donor"][name].get(state, 0.0) for state in table[_STATE]]
                    ),
                    0.0,
                ),
            }
        )
        for name, table in tables.items()
    }

    # Engine pre-pass over the ACS households only. ``select`` copied every
    # table, so the view is this stage's own: its missing cells read the
    # engine defaults, as the reviewed fill gave them before this stage.
    view = frame.select(acs_persons)
    view_units = view.table("tax_unit")
    at = pd.Series(np.arange(len(unit_ids)), index=unit_ids).reindex(
        view_units[_ID].to_numpy()
    )
    at = at.to_numpy()
    view_units[_TAKE_UP] = np.where(
        flag_present[at], stored_flags[at], _TAKE_UP_DEFAULT
    )
    view_units[_RATIO] = np.where(ratio_present[at], stored_ratio[at], _RATIO_DEFAULT)
    engine = _engine_table(aca_tax_units, view, residual_tables, label="ACS")
    del view, view_units
    engine = engine.set_index(_ID).reindex(acs_ids).reset_index()
    acs_flag_present = flag_present[acs_units]
    acs_stored = stored_flags[acs_units]
    acs_ratio_present = ratio_present[acs_units]
    acs_stored_ratio = stored_ratio[acs_units]
    staged = ~acs_flag_present | acs_stored
    anchored = acs_flag_present & acs_stored

    stage_units = engine.loc[staged, list(_ENGINE_TABLE_COLUMNS)].reset_index(drop=True)
    stage_units[plan.draw_column] = stable_identity_uniforms(
        keys[staged], seed=int(seed), salt=plan.draw_stream
    )
    # The stage's person table: each ACS person's tax unit, its observed
    # Marketplace coverage (the fiscal lane falls back on the deprecated
    # alias, as here) and the anchor. Built column by column, so the ACS
    # person rows are never copied whole.
    units_of = person.loc[acs_persons, _PERSON_TAX_UNIT]
    coverage: dict[str, np.ndarray] = {}
    lacking: list[str] = []
    for column in plan.aggregates:
        if column == plan.anchor:
            continue
        source = column if column in person else _COVERAGE_ALIAS
        if source not in person:
            lacking.append(column)
            continue
        coverage[column] = (
            person.loc[acs_persons, source].fillna(False).astype(bool).to_numpy()
        )
    if lacking:
        raise ValueError(
            f"ACS local ACA take-up requires observed coverage {lacking} (or "
            f"{_COVERAGE_ALIAS}) on the ACS persons, as the fiscal lane does."
        )
    anchored_ids = set(acs_ids[anchored])
    person_table = pd.DataFrame(
        {
            _ID: units_of.to_numpy(),
            **coverage,
            plan.anchor: units_of.isin(anchored_ids).to_numpy(),
        }
    )
    del units_of, coverage
    person_table = person_table.loc[
        person_table[_ID].isin(set(stage_units[_ID]))
    ].reset_index(drop=True)
    stop_after = (
        None if any(name in tables for name in plan.late_tables) else "support_clip"
    )
    from microcosm.build.us_runtime import (  # local: package init owns the manifest
        us_source_operation_handlers,
    )

    output = run_source_stage(
        plan.stage,
        tables={
            "cps_person": person_table,
            "tax_unit": stage_units,
            **residual_tables,
        },
        operation_handlers=us_source_operation_handlers(),
        config=SourceRuntimeConfig(seed=int(seed), target_year=int(period)),
        stop_after=stop_after,
    )
    del person_table, stage_units
    aligned = output.set_index(_ID).reindex(engine.loc[staged, _ID].to_numpy())
    if aligned[list(ACS_LOCAL_ACA_TAKE_UP_COLUMNS)].isna().any().any():
        raise ValueError(
            f"The {ACS_LOCAL_ACA_STAGE} stage output does not cover every staged "
            "ACS tax unit."
        )
    take_up = np.where(acs_flag_present, acs_stored, False)
    ratio = np.where(acs_ratio_present, acs_stored_ratio, _RATIO_DEFAULT)
    staged_take_up = aligned[_TAKE_UP].astype(bool).to_numpy()
    staged_ratio = aligned[_RATIO].astype(np.float64).to_numpy()
    take_up[staged] = np.where(
        acs_flag_present[staged], take_up[staged], staged_take_up
    )
    ratio[staged] = np.where(acs_ratio_present[staged], ratio[staged], staged_ratio)
    del output, aligned

    evaluated = _with_stored(engine, take_up, ratio)
    for name in tables:
        role = plan.roles[name]
        chosen = _selected(evaluated, role, label=f"ACS {name}")
        domain = _domain(evaluated, role, label=f"ACS {name}") & staged
        sums["delivered"][name] = _state_sums(evaluated, chosen, role.weight)
        sums["capacity"][name] = _state_sums(evaluated, domain, role.weight)
        sums["max_unit"][name] = _state_max(evaluated, domain, role.weight)

    missing_take_up = np.zeros(len(tax_unit), dtype=bool)
    missing_take_up[acs_units] = ~acs_flag_present
    missing_ratio = np.zeros(len(tax_unit), dtype=bool)
    missing_ratio[acs_units] = ~acs_ratio_present
    result = frame
    if missing_take_up.any() or missing_ratio.any():
        full_take_up = np.zeros(len(tax_unit), dtype=bool)
        full_take_up[acs_units] = take_up
        full_ratio = np.full(len(tax_unit), _RATIO_DEFAULT, dtype=np.float64)
        full_ratio[acs_units] = ratio
        updated = tax_unit.copy()
        updated[_TAKE_UP] = _filled_flags(
            tax_unit[_TAKE_UP], missing_take_up, full_take_up
        )
        updated[_RATIO] = _filled_ratios(tax_unit[_RATIO], missing_ratio, full_ratio)
        result = _with_tax_unit(frame, updated)

    final = result.table("tax_unit").loc[acs_units, [_TAKE_UP, _RATIO]]
    final_flags, final_flag_present, _ = _boolean_cells(final[_TAKE_UP])
    final_ratio, final_ratio_present, _ = _ratio_cells(final[_RATIO])
    weights = engine[_HOUSEHOLD_WEIGHT].to_numpy(dtype=np.float64)
    eligible = engine[_ELIGIBLE].to_numpy(dtype=bool) & (
        engine[_CREDIT].to_numpy(dtype=np.float64) > 0
    )
    takers = take_up & eligible
    names = tuple(tables)
    binding = _binding_table(plan, tables)
    clipped = takers & np.isfinite(ratio)

    def weighted_share(selected: np.ndarray, within: np.ndarray) -> float | None:
        total = float(weights[within].sum())
        return float(weights[selected & within].sum()) / total if total > 0 else None

    receipt: dict[str, Any] = {
        "issue": ACS_LOCAL_ACA_TAKE_UP_ISSUE,
        "method": ACS_LOCAL_ACA_TAKE_UP_METHOD,
        "stage": ACS_LOCAL_ACA_STAGE,
        "stop_after": stop_after,
        "spine": ACS_2024_1YR_SPINE,
        "seed": int(seed),
        "draw_key": _DRAW_KEY_FORMAT,
        "draw_stream": plan.draw_stream,
        "columns": list(ACS_LOCAL_ACA_TAKE_UP_COLUMNS),
        "acs_tax_units": int(acs_units.sum()),
        "acs_households": int(len(acs_households)),
        "weights_basis": _WEIGHTS_BASIS,
        "target_basis": _TARGET_BASIS,
        "anchor": {
            "rule": _ANCHOR_RULE,
            "stored_true_tax_units": int(anchored.sum()),
        },
        "engine_prepass": {
            "rows": "donor_households_then_acs_households",
            "donor": donor_counts,
            "missing_take_up_cells_read_as": _TAKE_UP_DEFAULT,
            "missing_ratio_cells_read_as": _RATIO_DEFAULT,
            "potential_credit": (
                "aca_ptc with take-up read True: the credit before the filing "
                "gate (files_return_for_aca_ptc counts a unit taking up a "
                "positive credit as filing for it)"
            ),
            "acs_eligible_tax_units": int(eligible.sum()),
        },
        "binding_table": binding,
        "tables": {
            name: {
                "operation": plan.roles[name].operation,
                "variable": plan.roles[name].variable,
                "weight": plan.roles[name].weight,
                "domain": plan.roles[name].domain,
                "binds": name == binding,
                "states": rows,
            }
            for name, rows in _rows(names, tables, sums).items()
        },
        "composition": _composition(plan, tables, sums),
        "take_up": {
            "column": _TAKE_UP,
            "filled_rows": int((~acs_flag_present).sum()),
            "preserved_acs_rows": int(acs_flag_present.sum()),
            "stored_false_rows_kept": int((acs_flag_present & ~acs_stored).sum()),
            "unfilled_acs_rows": int((~final_flag_present).sum()),
            "taker_tax_units": int(takers.sum()),
            "weighted_take_up_share_of_eligible": weighted_share(takers, eligible),
            "weighted_take_up_share": weighted_share(
                take_up, np.ones(len(take_up), dtype=bool)
            ),
        },
        "plan_ratio": {
            "column": _RATIO,
            "filled_rows": int((~acs_ratio_present).sum()),
            "preserved_acs_rows": int(acs_ratio_present.sum()),
            "unfilled_acs_rows": int((~final_ratio_present).sum()),
            "taker_share_below_1": weighted_share(clipped & (ratio < 1.0), clipped),
            "taker_share_at_lower_clip": weighted_share(
                clipped & (ratio <= 0.5), clipped
            ),
            "taker_share_at_upper_clip": weighted_share(
                clipped & (ratio >= 1.5), clipped
            ),
        },
        "assigned_sha256": _assignment_sha256(acs_ids, final_flags, final_ratio),
    }
    return result, _json_ready(receipt)


def acs_local_aca_take_up_assignment(frame: Frame) -> pd.DataFrame:
    """The ACS tax units' ids and complete ACA inputs, in tax-unit order.

    Raises:
        ValueError: If origin tags are missing or an ACS cell is missing or
            invalid.
    """

    _, acs_units = _acs_masks(frame)
    rows = frame.table("tax_unit").loc[acs_units]
    for column in ACS_LOCAL_ACA_TAKE_UP_COLUMNS:
        if column not in rows:
            raise ValueError(f"The frame has no {column}.")
    flags, flag_present, flag_invalid = _boolean_cells(rows[_TAKE_UP])
    ratio, ratio_present, ratio_invalid = _ratio_cells(rows[_RATIO])
    if not (flag_present.all() and ratio_present.all()) or (
        flag_invalid.any() or ratio_invalid.any()
    ):
        raise ValueError(
            "The ACS ACA inputs are not complete on the ACS tax units; run the "
            "ACS local ACA take-up stage first."
        )
    return pd.DataFrame({_ID: rows[_ID].to_numpy(), _TAKE_UP: flags, _RATIO: ratio})


def with_recorded_acs_local_aca_take_up(
    frame: Frame, recorded: pd.DataFrame, *, assigned_sha256: str
) -> Frame:
    """Apply a recorded ACS assignment and prove it reproduces its digest.

    ``recorded`` is :func:`acs_local_aca_take_up_assignment` of the frame
    the stage ran on. It must cover exactly this frame's ACS tax units, in
    order; stored ACS cells must agree with it, missing ones take its
    values, and the resulting ACS assignment must hash to
    ``assigned_sha256``.

    Raises:
        ValueError: On any mismatch.
    """

    _, acs_units = _acs_masks(frame)
    tax_unit = frame.table("tax_unit")
    ids = tax_unit.loc[acs_units, _ID].to_numpy()
    required = (_ID, *ACS_LOCAL_ACA_TAKE_UP_COLUMNS)
    if (
        any(column not in recorded for column in required)
        or len(recorded) != len(ids)
        or not np.array_equal(recorded[_ID].to_numpy(), ids)
    ):
        raise ValueError(
            "The recorded ACS ACA take-up does not cover this frame's ACS tax "
            "units in order."
        )
    for column in ACS_LOCAL_ACA_TAKE_UP_COLUMNS:
        if column not in tax_unit:
            raise ValueError(f"The frame has no {column}.")
    flags = np.asarray(recorded[_TAKE_UP].to_numpy(), dtype=bool)
    ratio = np.asarray(recorded[_RATIO].to_numpy(), dtype=np.float64)
    stored_flags, flag_present, flag_invalid = _boolean_cells(
        tax_unit.loc[acs_units, _TAKE_UP]
    )
    stored_ratio, ratio_present, ratio_invalid = _ratio_cells(
        tax_unit.loc[acs_units, _RATIO]
    )
    if (
        flag_invalid.any()
        or ratio_invalid.any()
        or (stored_flags[flag_present] != flags[flag_present]).any()
        or (stored_ratio[ratio_present] != ratio[ratio_present]).any()
    ):
        raise ValueError("Stored ACS ACA cells disagree with the recorded assignment.")
    if _assignment_sha256(ids, flags, ratio) != assigned_sha256:
        raise ValueError("The recorded ACS ACA take-up does not reproduce its digest.")
    missing_take_up = np.zeros(len(tax_unit), dtype=bool)
    missing_take_up[acs_units] = ~flag_present
    missing_ratio = np.zeros(len(tax_unit), dtype=bool)
    missing_ratio[acs_units] = ~ratio_present
    full_flags = np.zeros(len(tax_unit), dtype=bool)
    full_flags[acs_units] = flags
    full_ratio = np.full(len(tax_unit), _RATIO_DEFAULT, dtype=np.float64)
    full_ratio[acs_units] = ratio
    updated = tax_unit.copy()
    updated[_TAKE_UP] = _filled_flags(tax_unit[_TAKE_UP], missing_take_up, full_flags)
    updated[_RATIO] = _filled_ratios(tax_unit[_RATIO], missing_ratio, full_ratio)
    return _with_tax_unit(frame, updated)


def _residual_failures(name: str, rows: object) -> list[str]:
    """Each state's residual must be its count minus the donor contribution."""

    if not isinstance(rows, list) or not rows:
        return [f"receipt: no state rows for {name}."]
    failures: list[str] = []
    for row in rows:
        numbers = (
            [_finite(row.get(key)) for key in _STATE_NUMBERS]
            if isinstance(row, Mapping)
            else [float("nan")]
        )
        state = row.get(_STATE) if isinstance(row, Mapping) else "?"
        if not all(np.isfinite(numbers)):
            failures.append(f"receipt: {name} state {state} has nonfinite values.")
            continue
        target, donor, residual = numbers[:3]
        expected = max(target - donor, 0.0)
        if abs(residual - expected) > _RESIDUAL_RELATIVE_TOLERANCE * max(target, 1.0):
            failures.append(
                f"receipt: {name} state {state} targets {residual:.0f}, not the "
                f"donor residual {expected:.0f}."
            )
    return failures


def _binding_grades(rows: list) -> tuple[list[str], list[dict[str, Any]]]:
    """Grade each state of the binding table; return failures and statuses."""

    tolerance = ACS_LOCAL_ACA_RETURNS_RELATIVE_TOLERANCE
    failures: list[str] = []
    graded: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        state = row.get(_STATE)
        residual = _finite(row.get("residual"))
        capacity = _finite(row.get("acs_capacity"))
        delivered = _finite(row.get("acs_delivered"))
        step = _finite(row.get("acs_max_unit_weight"))
        if not all(np.isfinite([residual, capacity, delivered])):
            continue
        error = (delivered - residual) / residual if residual > 0 else None
        if residual <= 0:
            status = "donor_meets_count"
        elif capacity <= 0:
            status = "no_acs_capacity"
        elif capacity <= residual:
            status = "saturated"
            if abs(delivered - capacity) > _SATURATED_RELATIVE_TOLERANCE * capacity:
                failures.append(
                    f"state {state}: ACS capacity {capacity:.0f} cannot reach the "
                    f"residual {residual:.0f}, so every eligible unit should "
                    f"take up, but delivery is {delivered:.0f}."
                )
        elif abs(delivered - residual) <= max(
            tolerance * residual, step if np.isfinite(step) else 0.0
        ):
            status = "within_tolerance"
        else:
            status = "outside_tolerance"
            failures.append(
                f"state {state}: delivered ACS count {delivered:.0f} misses the "
                f"donor residual {residual:.0f} by {error:+.1%}, beyond "
                f"{tolerance:.0%} and one unit's weight."
            )
        graded.append(
            {
                _STATE: state,
                "status": status,
                "relative_error": error,
                "capacity_shortfall": max(residual - capacity, 0.0),
            }
        )
    return failures, graded


def _receipt_failures(
    receipt: object, acs_units: int, details: dict[str, Any]
) -> list[str]:
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("issue") != ACS_LOCAL_ACA_TAKE_UP_ISSUE
        or receipt.get("method") != ACS_LOCAL_ACA_TAKE_UP_METHOD
    ):
        details["receipt"] = {"present": False}
        return [
            f"No acs_local_aca_take_up receipt ({ACS_LOCAL_ACA_TAKE_UP_ISSUE}); "
            "the ACS rows' ACA take-up cannot be shown to be assigned."
        ]
    failures: list[str] = []
    if type(receipt.get("seed")) is not int:
        failures.append("receipt: no integer seed.")
    if receipt.get("acs_tax_units") != acs_units:
        failures.append(
            f"receipt: records {receipt.get('acs_tax_units')!r} ACS tax unit(s) "
            f"but the frame has {acs_units}."
        )
    digest = receipt.get("assigned_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        failures.append("receipt: no assignment digest.")
    for key, column in (("take_up", _TAKE_UP), ("plan_ratio", _RATIO)):
        entry = receipt.get(key)
        if not isinstance(entry, Mapping) or entry.get("column") != column:
            failures.append(f"receipt: no {key} entry for {column}.")
        elif entry.get("unfilled_acs_rows") != 0:
            failures.append(
                f"receipt: {column} left {entry.get('unfilled_acs_rows')!r} ACS "
                "row(s) unfilled."
            )
    tables = receipt.get("tables")
    binding = receipt.get("binding_table")
    if not isinstance(tables, Mapping) or binding not in tables:
        failures.append("receipt: no table accounting with its binding table.")
        details["receipt"] = {"present": True, "failures": len(failures)}
        return failures
    for name, entry in tables.items():
        rows = entry.get("states") if isinstance(entry, Mapping) else None
        failures += _residual_failures(str(name), rows)
    binding_entry = tables[binding]
    rows = binding_entry.get("states") if isinstance(binding_entry, Mapping) else None
    binding_failures, graded = _binding_grades(rows if isinstance(rows, list) else [])
    failures += binding_failures
    details["binding_table"] = binding
    details["binding_states"] = graded
    details["composition"] = receipt.get("composition")
    details["receipt"] = {"present": True, "failures": len(failures)}
    return failures


def acs_local_aca_take_up_signal_gate(
    frame: Frame, *, receipt: Mapping[str, Any] | None
) -> GateResult:
    """Grade the ACS rows' ACA take-up and plan ratio. The lane only reports it.

    Fails when either column is absent, an ACS cell is missing or invalid
    (a take-up flag that is not boolean, a ratio that is not finite and
    positive), the ACS take-up is constant while some state of the binding
    table has a residual and ACS capacity, or ``receipt`` does not show the
    materialize assignment: no unfilled ACS unit, every table's residual
    equal to its count minus the donor contribution, and in the binding
    table (the SOI return count where the feed has it) each state's
    delivered ACS count within
    :data:`ACS_LOCAL_ACA_RETURNS_RELATIVE_TOLERANCE` of its residual, or of
    one unit's weight, where the ACS capacity exceeds it, and equal to the
    capacity where it does not. States the donor rows already meet, states
    without ACS capacity, shortfalls, excesses, the CMS people, the amounts
    and the people per return are reported, not failed.

    The ACS local release records this verdict in its receipt and in a
    diagnostic shadow entry and never stops on it (microcosm#1720).
    """

    failures: list[str] = []
    details: dict[str, Any] = {
        "columns": list(ACS_LOCAL_ACA_TAKE_UP_COLUMNS),
        "relative_tolerance": ACS_LOCAL_ACA_RETURNS_RELATIVE_TOLERANCE,
    }
    try:
        _, acs_units = _acs_masks(frame)
    except ValueError as exc:
        return GateResult(
            name=ACS_LOCAL_ACA_TAKE_UP_GATE_NAME,
            passed=False,
            failures=(str(exc),),
            details=details,
        )
    tax_unit = frame.table("tax_unit")
    tag = spine_column("tax_unit")
    if set(tax_unit[tag].unique()) - {ASEC_PUF_DONOR_SPINE, ACS_2024_1YR_SPINE}:
        failures.append("ACA take-up origin tags contain an unsupported spine.")
    absent = [
        column for column in ACS_LOCAL_ACA_TAKE_UP_COLUMNS if column not in tax_unit
    ]
    if absent:
        failures.append(
            f"Missing tax-unit column(s) {absent}; the engine default is "
            "universal take-up at the benchmark plan."
        )
    else:
        flags, flag_present, flag_invalid = _boolean_cells(tax_unit[_TAKE_UP])
        _, ratio_present, ratio_invalid = _ratio_cells(tax_unit[_RATIO])
        acs = {
            "missing_take_up_rows": int((acs_units & ~flag_present).sum()),
            "invalid_take_up_rows": int((acs_units & flag_invalid).sum()),
            "missing_ratio_rows": int((acs_units & ~ratio_present).sum()),
            "invalid_ratio_rows": int((acs_units & ratio_invalid).sum()),
            "unique_take_up_values": int(
                np.unique(flags[acs_units & flag_present & ~flag_invalid]).size
            ),
        }
        details["acs"] = acs
        for key, count in acs.items():
            if key != "unique_take_up_values" and count:
                failures.append(
                    f"{ACS_2024_1YR_SPINE}: {count} {key.replace('_', ' ')}."
                )
    failures += _receipt_failures(receipt, int(acs_units.sum()), details)
    open_states = [
        row
        for row in details.get("binding_states", [])
        if row["status"] in {"saturated", "within_tolerance", "outside_tolerance"}
    ]
    if open_states and details.get("acs", {}).get("unique_take_up_values", 2) < 2:
        failures.append(
            f"{ACS_2024_1YR_SPINE}: ACA take-up is constant although "
            f"{len(open_states)} state(s) have a residual the ACS rows can fill."
        )
    return GateResult(
        name=ACS_LOCAL_ACA_TAKE_UP_GATE_NAME,
        passed=not failures,
        failures=tuple(failures),
        details=_json_ready(details),
    )
