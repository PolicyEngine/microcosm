"""UK FRS benefit-unit stochastic take-up assignments."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

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
# a unit with every adult at or over State Pension age is never drawn (#882,
# microcosm#867). The bounds come from the engine at the build instant through
# ``uk_take_up_population_policy``: State Pension age from the
# ``gov.dwp.state_pension.age`` parameters and the adult threshold from the
# engine's ``is_adult`` formula (a constant there, pinned by the lockstep test).
# The stage manifest declares the same population on the ``would_claim_uc``
# operation; ``assert_take_up_stage_population_declaration`` refuses a manifest
# that stops saying so, so the declaration and the code cannot drift apart.
UK_ENGINE_ADULT_AGE = 18
UK_UC_AGE_ELIGIBLE_AGGREGATE = "uc_age_eligible"
UK_UC_AGE_ELIGIBLE_METHOD = "any_adult_under_state_pension_age"
UK_UC_AGE_ELIGIBLE_SOURCE = "age"
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


@dataclass(frozen=True)
class UKTakeUpPopulationPolicy:
    """Engine bounds of the Universal Credit take-up population at one instant.

    ``adult_age`` is the engine's ``is_adult`` threshold and
    ``state_pension_age`` its ``gov.dwp.state_pension.age`` value, so a unit is
    in the population exactly when the engine's ``is_WA_adult`` is true for one
    of its members.
    """

    adult_age: int
    state_pension_age: int
    instant: str
    source: str

    def working_age(self, age: pd.Series | np.ndarray) -> np.ndarray:
        values = np.asarray(age, dtype=float)
        return (values >= self.adult_age) & (values < self.state_pension_age)


def uk_take_up_population_policy(build_period: int | str) -> UKTakeUpPopulationPolicy:
    """Read the working-age bounds from the engine at ``{year}-01-01``."""

    try:
        import policyengine_uk
        from policyengine_core.parameters import ParameterNode
    except ImportError as exc:
        raise ImportError(
            "UK take-up population bounds require `uv sync --all-packages --extra uk`."
        ) from exc

    parameters = ParameterNode(
        directory_path=str(Path(policyengine_uk.__file__).parent / "parameters")
    )
    instant = f"{int(build_period)}-01-01"
    ages = parameters.gov.dwp.state_pension.age
    male, female = float(ages.male(instant)), float(ages.female(instant))
    if male != female or not float(male).is_integer():
        raise ValueError(
            "the take-up population needs one State Pension age for every adult "
            f"at {instant}; the engine has male {male} and female {female}, so the "
            "stage must consume sex before it can form the population"
        )
    return UKTakeUpPopulationPolicy(
        adult_age=UK_ENGINE_ADULT_AGE,
        state_pension_age=int(male),
        instant=instant,
        source="policyengine-uk parameters " + metadata.version("policyengine-uk"),
    )


def assert_take_up_stage_population_declaration(stage: SourceStageSpec) -> None:
    """Refuse a manifest whose declared UC population differs from the code's."""

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
    if not (declared_aggregate and declared_population):
        raise ValueError(
            f"stage {stage.stage!r} must declare the {UK_UC_AGE_ELIGIBLE_AGGREGATE!r} "
            f"aggregate ({UK_UC_AGE_ELIGIBLE_METHOD} over "
            f"{UK_UC_AGE_ELIGIBLE_SOURCE}) and population={UK_UC_AGE_ELIGIBLE_AGGREGATE!r} "
            f"on the {UK_UC_TAKE_UP_OUTPUT!r} operation; the code draws Universal "
            "Credit take-up over that population and refuses a manifest that says "
            "otherwise"
        )


@dataclass(frozen=True)
class UKFRSTakeUpStageTransform:
    """Whole-stage callable for UK benefit-unit take-up assignments."""

    contract: UKTakeUpContract
    stage: SourceStageSpec
    population_policy: UKTakeUpPopulationPolicy | None = None

    def __call__(self, frame: Frame) -> Frame:
        assert_take_up_stage_population_declaration(self.stage)
        policy = self.population_policy or uk_take_up_population_policy(
            uk_time_period(frame)
        )
        return add_frs_take_up(frame, contract=self.contract, population_policy=policy)

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return FRS_TAKE_UP_OUTPUT_COLUMNS


def add_frs_take_up(
    frame: Frame,
    *,
    contract: UKTakeUpContract,
    population_policy: UKTakeUpPopulationPolicy,
) -> Frame:
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    anchors = aggregate_person_reported_to_benunit(person, benunit)
    uc_age_eligible = uc_age_eligible_benunits(person, benunit, population_policy)
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
    policy: UKTakeUpPopulationPolicy,
) -> np.ndarray:
    """True where the benefit unit has a working-age adult under ``policy``."""

    if "age" not in person.columns:
        raise KeyError(
            "person.age is missing; the Universal Credit take-up population "
            "cannot be formed without it"
        )
    age = pd.to_numeric(person["age"], errors="coerce").fillna(0)
    eligible_adult = policy.working_age(age)
    eligible_ids = set(person.loc[eligible_adult, "person_benunit_id"])
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
    population_policy: UKTakeUpPopulationPolicy | None = None,
) -> GateResult:
    """Require non-constant UK stochastic flags near contract target shares."""

    resolved = contract if contract is not None else load_uk_take_up_contract()
    failures: list[str] = []
    details: dict[str, object] = {}
    policy: UKTakeUpPopulationPolicy | None = population_policy
    for entity, output, key in UK_TAKE_UP_SIGNAL_OUTPUTS:
        table = frame.table(entity)
        if output not in table.columns:
            failures.append(f"{entity}.{output}: missing stochastic column.")
            continue
        values = np.asarray(table[output], dtype=bool)
        unique_count = int(pd.Series(values).nunique(dropna=False))
        weights = np.asarray(frame.resolve_weights(entity).values, dtype=np.float64)
        population = np.ones(values.shape, dtype=bool)
        if key == "universal_credit":
            # The contract rate is a share of the units that can claim: those
            # with a working-age adult under the engine's bounds at the build
            # instant. The gate measures the realized share on the same
            # population.
            if policy is None:
                policy = uk_take_up_population_policy(uk_time_period(frame))
            population = uc_age_eligible_benunits(frame.table("person"), table, policy)
            details["universal_credit_population_policy"] = {
                "adult_age": policy.adult_age,
                "state_pension_age": policy.state_pension_age,
                "instant": policy.instant,
                "source": policy.source,
            }
        if not population.any() or float(weights[population].sum()) <= 0.0:
            failures.append(
                f"{entity}.{output}: no unit in the draw's population carries "
                "weight; the take-up share cannot be measured."
            )
            continue
        share = float(
            np.average(values[population].astype(float), weights=weights[population])
        )
        target = _target_share(table, key, resolved, weights)
        details[f"{entity}.{output}"] = {
            "weighted_share": share,
            "target": target,
            "absolute_deviation": abs(share - target),
            "unique_count": unique_count,
            "population_units": int(population.sum()),
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
