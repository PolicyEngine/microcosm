"""UK FRS benefit-unit stochastic take-up assignments."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import (
    assign_binary_from_rate,
    assign_binary_with_anchored_residual,
    clipped_normal_from_uniforms,
    stable_identity_uniforms,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.take_up_contract import (
    UKTakeUpContract,
    load_uk_take_up_contract,
)
from microcosm.frame import Frame
from microcosm.frame.rules import assert_rules_engine_country

FRS_TAKE_UP_OUTPUT_COLUMNS = (
    "would_claim_child_benefit",
    "child_benefit_opts_out",
    "would_claim_pc",
    "would_claim_uc",
    "would_claim_tfc",
    "would_claim_extended_childcare",
    "would_claim_universal_childcare",
    "would_claim_targeted_childcare",
    "would_claim_uc_childcare",
    "maximum_extended_childcare_hours_usage",
)
FRS_TAKE_UP_NONNEGATIVE_OUTPUT_COLUMNS = ("maximum_extended_childcare_hours_usage",)
UK_TAKE_UP_ANCHOR_AGGREGATES = {
    "child_benefit_reported_anchor": "child_benefit_reported",
    "pension_credit_reported_anchor": "pension_credit_reported",
    "universal_credit_reported_anchor": "universal_credit_reported",
}
UK_TAKE_UP_DECLARED_SEEDS = {output: 0 for output in FRS_TAKE_UP_OUTPUT_COLUMNS}
# Universal Credit is claimable only by a benefit unit with a working-age
# adult (policyengine-uk ``is_uc_eligible`` requires ``is_WA_adult``, which is
# ``is_adult & ~is_SP_age``). The draw's population is therefore those units;
# a unit with no working-age adult is never drawn (#882, microcosm#867). Each
# person's ``is_WA_adult`` is read from the engine at the build period
# (``uk_take_up_population``), so the population follows the engine's own
# State Pension age: by date of birth since policyengine-uk#1899, which from
# 2026-27 puts some 66-year-olds under it and others over, so no age bound can
# stand in for it. The stage manifest declares the engine read and the
# any-member aggregate, and the population on the ``would_claim_uc``
# operation; ``assert_take_up_stage_population_declaration`` refuses a manifest
# that stops saying so, so the declaration and the code cannot drift apart.
UK_UC_WORKING_AGE_ADULT = "is_WA_adult"
UK_TAKE_UP_ENGINE_PREDICTORS = (UK_UC_WORKING_AGE_ADULT,)
UK_UC_AGE_ELIGIBLE_AGGREGATE = "uc_age_eligible"
UK_UC_AGE_ELIGIBLE_METHOD = "any"
UK_UC_AGE_ELIGIBLE_SOURCE = UK_UC_WORKING_AGE_ADULT
UK_UC_TAKE_UP_OUTPUT = "would_claim_uc"
# The Universal Credit childcare element is claimed at one rate per family
# type (DWP publishes the single / couple split of the households receiving
# it); the couple rate applies where the benefit unit is a couple (#882).
UK_UC_CHILDCARE_RATE_KEYS = {
    "single": "uc_childcare_single",
    "couple": "uc_childcare_couple",
}
UK_TAKE_UP_SIGNAL_OUTPUTS = (
    ("benunit", "would_claim_child_benefit", "child_benefit"),
    ("benunit", "child_benefit_opts_out", "child_benefit_opts_out_rate"),
    ("benunit", "would_claim_pc", "pension_credit"),
    ("benunit", "would_claim_uc", "universal_credit"),
    ("benunit", "would_claim_tfc", "tax_free_childcare"),
    ("benunit", "would_claim_extended_childcare", "extended_childcare"),
    ("benunit", "would_claim_universal_childcare", "universal_childcare"),
    ("benunit", "would_claim_targeted_childcare", "targeted_childcare"),
    ("benunit", "would_claim_uc_childcare", "uc_childcare"),
    ("person", "would_claim_marriage_allowance", "marriage_allowance"),
    ("person", "would_claim_scp", "scp"),
    ("household", "household_owns_tv", "tv_ownership_rate"),
    ("household", "would_evade_tv_licence_fee", "tv_licence_evasion_rate"),
    (
        "household",
        "main_residential_property_purchased_is_first_home",
        "first_time_buyer_rate",
    ),
    ("household", "property_purchased", "property_purchase_rate"),
)


@dataclass(frozen=True, eq=False)
class UKTakeUpPopulation:
    """The engine's working-age adults at one build period.

    ``working_age_adult`` holds policyengine-uk's ``is_WA_adult`` for each row
    of the frame's person table, in row order, so a benefit unit is in the
    Universal Credit take-up population exactly when one of its members is
    flagged. ``source`` names the engine and version that computed it.
    """

    working_age_adult: np.ndarray = field(repr=False)
    period: str
    source: str

    def __post_init__(self) -> None:
        values = np.asarray(self.working_age_adult)
        if values.ndim != 1 or values.dtype.kind != "b":
            raise ValueError(
                f"{UK_UC_WORKING_AGE_ADULT} must be one boolean per person row; "
                f"got shape {values.shape} and dtype {values.dtype}"
            )
        object.__setattr__(self, "working_age_adult", values)

    def evidence(self) -> dict[str, object]:
        """JSON-safe provenance for gate details and stage receipts."""

        return {
            "variable": UK_UC_WORKING_AGE_ADULT,
            "period": self.period,
            "source": self.source,
            "working_age_adults": int(self.working_age_adult.sum()),
            "person_rows": int(self.working_age_adult.size),
        }


def uk_take_up_population(frame: Frame, engine: object) -> UKTakeUpPopulation:
    """Read every person's ``is_WA_adult`` from the engine at the frame's period.

    The engine computes it on the whole frame, the way a simulation of the
    released data does: whether a person is under State Pension age depends on
    their date of birth, which the engine places within each whole age and sex
    from the person ids and weights (policyengine-uk#1899).
    """

    assert_rules_engine_country(engine, "uk")
    period = uk_time_period(frame)
    materialized = engine.materialize(frame, UK_TAKE_UP_ENGINE_PREDICTORS, period)
    values = np.asarray(materialized[UK_UC_WORKING_AGE_ADULT])
    if values.shape != (frame.n("person"),):
        raise ValueError(
            f"{UK_UC_WORKING_AGE_ADULT} has shape {values.shape}; the frame has "
            f"{frame.n('person')} person row(s)"
        )
    return UKTakeUpPopulation(
        working_age_adult=values, period=period, source=_engine_source()
    )


def uk_engine_readable_frame(frame: Frame) -> Frame:
    """Fill by-design NaN on float columns so the engine can read the frame.

    The #717 SPI channel leaves hmrc_spi_* auxiliaries (e.g.
    ``other_investment_income``) NaN on FRS rows by design, and
    policyengine-uk refuses a dataset with NaN in any column. A person's
    working-age status reads only age, sex, ids and weights, so a copy with
    those floats filled at 0 gives the same answer. Missing values in a
    non-float column are left for the engine to refuse by name.
    """

    tables = {}
    for entity in ("person", "benunit", "household"):
        table = frame.table(entity).copy()
        for column in table.columns:
            if table[column].dtype.kind == "f" and table[column].isna().any():
                table[column] = table[column].fillna(0.0)
        tables[entity] = table
    return uk_national_frame(
        person=tables["person"],
        benunit=tables["benunit"],
        household=tables["household"],
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )


def _engine_source() -> str:
    try:
        version = metadata.version("policyengine-uk")
    except metadata.PackageNotFoundError:
        version = "not installed"
    return f"policyengine-uk {version} {UK_UC_WORKING_AGE_ADULT}"


def assert_take_up_stage_population_declaration(stage: SourceStageSpec) -> None:
    """Refuse a manifest whose declared UC population differs from the code's."""

    declared_engine_read = any(
        op.kind == "materialize_rules_engine_predictors"
        and tuple(op.parameters.get("predictors") or ()) == UK_TAKE_UP_ENGINE_PREDICTORS
        and op.parameters.get("consumed_only") is True
        for op in stage.operations
    )
    declared_aggregate = any(
        op.kind == "aggregate_person_to_benunit"
        and op.parameters.get("method") == UK_UC_AGE_ELIGIBLE_METHOD
        and dict(op.parameters.get("aggregates") or {})
        == {UK_UC_AGE_ELIGIBLE_AGGREGATE: UK_UC_AGE_ELIGIBLE_SOURCE}
        for op in stage.operations
    )
    declared_population = any(
        op.kind == "assign_binary_with_anchored_residual"
        and op.parameters.get("output") == UK_UC_TAKE_UP_OUTPUT
        and op.parameters.get("population") == UK_UC_AGE_ELIGIBLE_AGGREGATE
        for op in stage.operations
    )
    if not (declared_engine_read and declared_aggregate and declared_population):
        seen = [
            (op.kind, dict(op.parameters))
            for op in stage.operations
            if op.kind
            in ("materialize_rules_engine_predictors", "aggregate_person_to_benunit")
            or op.parameters.get("output") == UK_UC_TAKE_UP_OUTPUT
        ]
        raise ValueError(
            f"stage {stage.stage!r} must declare the consumed engine read of "
            f"{list(UK_TAKE_UP_ENGINE_PREDICTORS)!r}, the "
            f"{UK_UC_AGE_ELIGIBLE_AGGREGATE!r} aggregate ({UK_UC_AGE_ELIGIBLE_METHOD} "
            f"over {UK_UC_AGE_ELIGIBLE_SOURCE}) and "
            f"population={UK_UC_AGE_ELIGIBLE_AGGREGATE!r} on the "
            f"{UK_UC_TAKE_UP_OUTPUT!r} operation; the code draws Universal Credit "
            "take-up over that population and refuses a manifest that says "
            f"otherwise (declared: {seen!r})"
        )


@dataclass(frozen=True)
class UKFRSTakeUpStageTransform:
    """Whole-stage callable for UK benefit-unit take-up assignments."""

    contract: UKTakeUpContract
    stage: SourceStageSpec
    engine: object

    def __call__(self, frame: Frame) -> Frame:
        assert_take_up_stage_population_declaration(self.stage)
        population = uk_take_up_population(frame, self.engine)
        return add_frs_take_up(frame, contract=self.contract, population=population)

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return FRS_TAKE_UP_OUTPUT_COLUMNS


def add_frs_take_up(
    frame: Frame,
    *,
    contract: UKTakeUpContract,
    population: UKTakeUpPopulation,
) -> Frame:
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    anchors = aggregate_person_reported_to_benunit(person, benunit)
    uc_age_eligible = uc_age_eligible_benunits(person, benunit, population)
    derived = derive_frs_take_up(
        benunit, anchors=anchors, contract=contract, uc_age_eligible=uc_age_eligible
    )
    for column in FRS_TAKE_UP_OUTPUT_COLUMNS:
        benunit[column] = derived[column].to_numpy()
    result = uk_national_frame(
        person=person,
        benunit=benunit,
        household=frame.table("household").copy(),
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )
    validate_uk_national_frame(result)
    return result


def aggregate_person_reported_to_benunit(
    person: pd.DataFrame, benunit: pd.DataFrame
) -> pd.DataFrame:
    """Return anchor booleans by benefit unit for positive reported amounts."""

    grouped = pd.DataFrame(index=benunit["benunit_id"])
    for output, source in UK_TAKE_UP_ANCHOR_AGGREGATES.items():
        if source not in person.columns:
            raise KeyError(
                f"take-up anchor source column {source!r} is missing from the "
                "person table; anchored assignment refuses to run without it"
            )
        reporter_ids = person.loc[
            pd.to_numeric(person[source], errors="coerce").fillna(0) > 0,
            "person_benunit_id",
        ]
        grouped[output] = grouped.index.isin(reporter_ids)
    return grouped.reset_index(drop=True)


def uc_age_eligible_benunits(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    population: UKTakeUpPopulation,
) -> np.ndarray:
    """True where one of the benefit unit's members is a working-age adult."""

    working_age_adult = population.working_age_adult
    if working_age_adult.shape != (len(person),):
        raise ValueError(
            f"{UK_UC_WORKING_AGE_ADULT} has {working_age_adult.size} value(s) for "
            f"{len(person)} person row(s); the population must align with the "
            "person table"
        )
    eligible_ids = set(person["person_benunit_id"].to_numpy()[working_age_adult])
    return benunit["benunit_id"].isin(eligible_ids).to_numpy(dtype=bool)


def derive_frs_take_up(
    benunit: pd.DataFrame,
    *,
    anchors: pd.DataFrame,
    contract: UKTakeUpContract,
    uc_age_eligible: np.ndarray,
) -> pd.DataFrame:
    ids = benunit["benunit_id"].to_numpy()
    values = pd.DataFrame(index=benunit.index)
    population = np.asarray(uc_age_eligible, dtype=bool)
    if population.shape != ids.shape:
        raise ValueError("uc_age_eligible must align with the benefit-unit table")
    values["would_claim_child_benefit"] = assign_binary_with_anchored_residual(
        _draws(ids, "would_claim_child_benefit"),
        contract.rate("child_benefit"),
        anchors["child_benefit_reported_anchor"].to_numpy(dtype=bool),
    )
    values["child_benefit_opts_out"] = assign_binary_from_rate(
        _draws(ids, "child_benefit_opts_out"),
        contract.rate("child_benefit_opts_out_rate"),
    )
    values["would_claim_pc"] = assign_binary_with_anchored_residual(
        _draws(ids, "would_claim_pc"),
        contract.rate("pension_credit"),
        anchors["pension_credit_reported_anchor"].to_numpy(dtype=bool),
    )
    values["would_claim_uc"] = assign_binary_with_anchored_residual(
        _draws(ids, "would_claim_uc"),
        contract.rate("universal_credit"),
        anchors["universal_credit_reported_anchor"].to_numpy(dtype=bool),
        population=population,
    )
    for output, key in (
        ("would_claim_tfc", "tax_free_childcare"),
        ("would_claim_extended_childcare", "extended_childcare"),
        ("would_claim_universal_childcare", "universal_childcare"),
        ("would_claim_targeted_childcare", "targeted_childcare"),
    ):
        values[output] = assign_binary_from_rate(
            _draws(ids, output), contract.rate(key)
        )
    values["would_claim_uc_childcare"] = _draws(
        ids, "would_claim_uc_childcare"
    ) < uc_childcare_rates(benunit, contract)
    distribution = contract.continuous_entry("maximum_extended_childcare_hours_usage")
    values["maximum_extended_childcare_hours_usage"] = clipped_normal_from_uniforms(
        _draws(ids, "maximum_extended_childcare_hours_usage"),
        mean=float(distribution["mean"]),
        sd=float(distribution["sd"]),
        lower=float(distribution["lower"]),
        upper=float(distribution["upper"]),
    )
    return values


def uc_childcare_rates(benunit: pd.DataFrame, contract: UKTakeUpContract) -> np.ndarray:
    """Per-unit childcare-element take-up rate: the couple or single contract rate."""

    if "is_married" not in benunit.columns:
        raise KeyError(
            "benunit.is_married is missing; the Universal Credit childcare "
            "take-up rate is drawn by family type"
        )
    couple = benunit["is_married"].fillna(False).to_numpy(dtype=bool)
    return np.where(
        couple,
        contract.rate(UK_UC_CHILDCARE_RATE_KEYS["couple"]),
        contract.rate(UK_UC_CHILDCARE_RATE_KEYS["single"]),
    )


def _draws(ids: np.ndarray, output: str) -> np.ndarray:
    return stable_identity_uniforms(
        ids, seed=UK_TAKE_UP_DECLARED_SEEDS[output], salt=output
    )


def uk_take_up_signal_gate(
    frame: Frame,
    *,
    contract: UKTakeUpContract | None = None,
    maximum_share_deviation: float = 0.05,
    engine: object | None = None,
    population: UKTakeUpPopulation | None = None,
) -> GateResult:
    """Require non-constant UK stochastic flags near contract target shares.

    The Universal Credit share is measured over the units with a working-age
    adult, read from ``engine`` for this frame unless ``population`` is given.
    """

    resolved = contract if contract is not None else load_uk_take_up_contract()
    failures: list[str] = []
    details: dict[str, object] = {}
    uc_population: UKTakeUpPopulation | None = population
    for entity, output, key in UK_TAKE_UP_SIGNAL_OUTPUTS:
        table = frame.table(entity)
        if output not in table.columns:
            failures.append(f"{entity}.{output}: missing stochastic column.")
            continue
        values = np.asarray(table[output], dtype=bool)
        unique_count = int(pd.Series(values).nunique(dropna=False))
        weights = np.asarray(frame.resolve_weights(entity).values, dtype=np.float64)
        units = np.ones(values.shape, dtype=bool)
        if key == "universal_credit":
            # The contract rate is a share of the units that can claim: those
            # with a working-age adult, read from the engine for this frame at
            # its period. The gate measures the realized share on them.
            if uc_population is None:
                if engine is None:
                    raise ValueError(
                        "the take-up signal gate reads the Universal Credit "
                        "population from the rules engine; pass engine= or "
                        "population="
                    )
                uc_population = uk_take_up_population(
                    uk_engine_readable_frame(frame), engine
                )
            units = uc_age_eligible_benunits(
                frame.table("person"), table, uc_population
            )
            details["universal_credit_population"] = uc_population.evidence()
        if not units.any() or float(weights[units].sum()) <= 0.0:
            failures.append(
                f"{entity}.{output}: no unit in the draw's population carries "
                "weight; the take-up share cannot be measured."
            )
            continue
        share = float(np.average(values[units].astype(float), weights=weights[units]))
        target = _target_share(table, key, resolved, weights)
        details[f"{entity}.{output}"] = {
            "weighted_share": share,
            "target": target,
            "absolute_deviation": abs(share - target),
            "unique_count": unique_count,
            "population_units": int(units.sum()),
        }
        if unique_count < 2:
            failures.append(
                f"{entity}.{output}: constant column; stochastic take-up "
                "assignment must not collapse to a universal default."
            )
        if abs(share - target) > maximum_share_deviation:
            failures.append(
                f"{entity}.{output}: weighted share {share:.3f} differs from "
                f"contract target {target:.3f} by more than "
                f"{maximum_share_deviation:.3f}."
            )
    return GateResult(
        name="take_up_signal",
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )


def _target_share(
    table: pd.DataFrame,
    key: str,
    contract: UKTakeUpContract,
    weights: np.ndarray,
) -> float:
    if key == "uc_childcare":
        return float(np.average(uc_childcare_rates(table, contract), weights=weights))
    if key != "scp":
        return contract.rate(key)
    age = pd.to_numeric(table["age"], errors="coerce").fillna(0).to_numpy()
    targets = np.where(
        age < 6,
        contract.rate("scp_under_6"),
        contract.rate("scp_6_plus"),
    )
    return float(np.average(targets, weights=weights))
