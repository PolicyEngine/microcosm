"""UK ETB services and NHS allocation stage."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.gates import FitWeightRecord
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.frs_spine import read_pinned_tab
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.frame import Frame
from microcosm.frame.rules import assert_rules_engine_country

ETB_SERVICES_YEAR = 2024
ETB_SERVICES_WEEKS_IN_YEAR = 52
RAIL_FARE_INDEX_2023 = 1.110
NHS_BUDGET_2025_26 = 202_000_000_000.0
UK_ETB_SERVICES_PREDICTORS = (
    "is_adult",
    "is_child",
    "is_SP_age",
    "count_primary_education",
    "count_secondary_education",
    "count_further_education",
    "dla",
    "pip",
    "hbai_household_net_income",
)
# The education counts are not engine variables: the incumbent derives them
# from person current_education (etb.py:180-186). Only these materialize.
UK_ETB_SERVICES_ENGINE_VARIABLES = (
    "is_adult",
    "is_child",
    "is_SP_age",
    "dla",
    "pip",
    "hbai_household_net_income",
    "current_education",
)
UK_ETB_SERVICES_EDUCATION_COUNTS = {
    "count_primary_education": ("PRIMARY",),
    "count_secondary_education": ("LOWER_SECONDARY",),
    "count_further_education": ("UPPER_SECONDARY", "TERTIARY"),
}
UK_ETB_SERVICES_HOUSEHOLD_OUTPUT_COLUMNS = (
    "dfe_education_spending",
    "rail_subsidy_spending",
    "bus_subsidy_spending",
    "rail_usage",
)
UK_NHS_OUTPUT_COLUMNS = (
    "a_and_e_visits",
    "admitted_patient_visits",
    "outpatient_visits",
    "nhs_a_and_e_spending",
    "nhs_admitted_patient_spending",
    "nhs_outpatient_spending",
)
UK_ETB_SERVICES_OUTPUT_COLUMNS = (
    *UK_ETB_SERVICES_HOUSEHOLD_OUTPUT_COLUMNS,
    *UK_NHS_OUTPUT_COLUMNS,
)
UK_ETB_SERVICES_NONNEGATIVE_OUTPUT_COLUMNS = UK_ETB_SERVICES_OUTPUT_COLUMNS
UK_ETB_SERVICES_FIT_NAME = "uk_etb_2024_services"


@dataclass
class UKETBServicesStageTransform:
    stage: SourceStageSpec
    engine: object
    etb_tab_path: str | Path | None = None
    donor: pd.DataFrame | None = None
    nhs_table: pd.DataFrame | None = None
    last_fit_weight_records: tuple[FitWeightRecord, ...] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def fit_weight_records(self) -> tuple[FitWeightRecord, ...]:
        return (
            () if self.last_fit_weight_records is None else self.last_fit_weight_records
        )

    def __call__(self, frame: Frame) -> Frame:
        assert_rules_engine_country(self.engine, "uk")
        raw = (
            self.donor
            if self.donor is not None
            else read_pinned_tab(
                _require_path(self.etb_tab_path), self.stage.artifacts[0]
            )
        )
        donor = clean_etb_services_table(raw)
        predictors = recipient_predictors(frame, self.engine)
        draws, records = impute_etb_services(
            donor, predictors, seed=_qrf_seed(self.stage)
        )
        draws = support_clip_to_donor(draws, donor)
        draws["rail_usage"] = draws["rail_subsidy_spending"] / RAIL_FARE_INDEX_2023
        household = frame.table("household").copy()
        for column in UK_ETB_SERVICES_HOUSEHOLD_OUTPUT_COLUMNS:
            household[column] = draws[column].to_numpy()
        person = frame.table("person").copy()
        nhs = allocate_nhs_by_age_gender(
            person,
            household_weights=frame.weights_for("household").values,
            household=household,
            nhs_table=self.nhs_table,
        )
        for column in UK_NHS_OUTPUT_COLUMNS:
            person[column] = nhs[column].to_numpy()
        result = uk_national_frame(
            person=person,
            benunit=frame.table("benunit").copy(),
            household=household,
            time_period=uk_time_period(frame),
            weight_kind=uk_household_weight_kind(frame),
            household_weights=frame.weights_for("household").values,
            mass_log=frame.mass_log,
        )
        validate_uk_national_frame(result)
        self.last_fit_weight_records = records
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UK_ETB_SERVICES_OUTPUT_COLUMNS


def clean_etb_services_table(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.replace(r"^\s*$", np.nan, regex=True).copy()
    if "year" not in data:
        raise ValueError("ETB services donor is missing 'year'.")
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data = data[data["year"] == data["year"].max()].copy()
    required = [
        "adults",
        "childs",
        "disinc",
        "educ",
        "rail",
        "bussub",
        "hhold_adj_weight",
        "noretd",
        "primed",
        "secoed",
        "furted",
        "disliv",
        "pips",
    ]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(
            f"ETB services donor is missing required column(s): {missing}."
        )
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=required)
    train = pd.DataFrame()
    train["is_adult"] = data["adults"]
    train["is_child"] = data["childs"]
    train["hbai_household_net_income"] = data["disinc"] * ETB_SERVICES_WEEKS_IN_YEAR
    train["is_SP_age"] = data["noretd"]
    train["count_primary_education"] = data["primed"]
    train["count_secondary_education"] = data["secoed"]
    train["count_further_education"] = data["furted"]
    train["dla"] = data["disliv"]
    train["pip"] = data["pips"]
    train["weight"] = data["hhold_adj_weight"]
    train["dfe_education_spending"] = data["educ"] * ETB_SERVICES_WEEKS_IN_YEAR
    train["rail_subsidy_spending"] = data["rail"] * ETB_SERVICES_WEEKS_IN_YEAR
    train["bus_subsidy_spending"] = data["bussub"] * ETB_SERVICES_WEEKS_IN_YEAR
    return train


def household_grain_services_predictors(person_level: pd.DataFrame) -> pd.DataFrame:
    grouped = person_level.groupby("household_id", sort=False).sum(numeric_only=True)
    return grouped.loc[:, list(UK_ETB_SERVICES_PREDICTORS)]


def recipient_predictors(frame: Frame, engine: object) -> pd.DataFrame:
    """Materialize ETB services recipient predictors at household grain.

    The three education counts derive from person current_education (the
    incumbent's construction, etb.py:180-186) — they are not engine
    variables. Everything else materializes at its native entity and
    aggregates to household by person_household_id.
    """

    materialized = engine.materialize(
        frame, UK_ETB_SERVICES_ENGINE_VARIABLES, uk_time_period(frame)
    )
    household = frame.table("household")
    person = frame.table("person")
    group_keys = person["person_household_id"].to_numpy()
    household_ids = household["household_id"]

    def person_sum(values: np.ndarray) -> np.ndarray:
        summed = pd.Series(values.astype(float)).groupby(group_keys).sum()
        return summed.reindex(household_ids).fillna(0.0).to_numpy()

    result = pd.DataFrame(index=household.index)
    education = np.asarray(materialized["current_education"]).astype(str)
    for predictor in UK_ETB_SERVICES_PREDICTORS:
        if predictor in UK_ETB_SERVICES_EDUCATION_COUNTS:
            labels = UK_ETB_SERVICES_EDUCATION_COUNTS[predictor]
            result[predictor] = person_sum(np.isin(education, labels))
            continue
        values = np.asarray(materialized[predictor])
        entity = str(engine.variable_metadata(predictor).entity)
        if entity == "household":
            result[predictor] = values
        elif entity == "person":
            result[predictor] = person_sum(values)
        else:
            raise ValueError(f"unsupported ETB services predictor entity {entity!r}.")
    return result


def impute_etb_services(
    donor: pd.DataFrame, recipient: pd.DataFrame, *, seed: int, n_estimators: int = 100
) -> tuple[pd.DataFrame, tuple[FitWeightRecord, ...]]:
    from microcosm.fit import RegimeGatedQRF

    targets = UK_ETB_SERVICES_HOUSEHOLD_OUTPUT_COLUMNS[:3]
    model = RegimeGatedQRF(n_estimators=n_estimators, seed=seed)
    state = model.start_chain(
        donor,
        list(UK_ETB_SERVICES_PREDICTORS),
        list(targets),
        weights="weight",
    )
    raw = pd.DataFrame(index=recipient.index)
    records: list[FitWeightRecord] = []
    for target in targets:
        result = model.fit_draw_next(
            donor,
            recipient.loc[:, list(UK_ETB_SERVICES_PREDICTORS)],
            raw,
            state=state,
            weights="weight",
        )
        raw[target] = result.raw_draw
        records.append(
            FitWeightRecord(f"{UK_ETB_SERVICES_FIT_NAME}:{target}", result.weight_kind)
        )
        state = result.state
    return raw, tuple(records)


def support_clip_to_donor(draws: pd.DataFrame, donor: pd.DataFrame) -> pd.DataFrame:
    result = draws.copy()
    for column in UK_ETB_SERVICES_HOUSEHOLD_OUTPUT_COLUMNS[:3]:
        values = donor[column]
        finite = values[np.isfinite(values)]
        if finite.empty:
            continue
        result[column] = result[column].clip(float(finite.min()), float(finite.max()))
    return result


def donor_realized_ranges(donor: pd.DataFrame) -> dict[str, tuple[float, float]]:
    ranges = {}
    for column in UK_ETB_SERVICES_HOUSEHOLD_OUTPUT_COLUMNS[:3]:
        values = donor[column]
        finite = values[np.isfinite(values)]
        if not finite.empty:
            ranges[column] = (float(finite.min()), float(finite.max()))
    return ranges


def parse_nhs_age_bounds(age_group: str) -> tuple[int, int]:
    if age_group == "0 years":
        return 0, 1
    if age_group == "95 years or older":
        return 95, 120
    # Banded labels read "01-04 years": strip the unit suffix before
    # splitting (the incumbent slices the first five characters).
    stripped = age_group.removesuffix(" years").strip()
    if "-" in stripped:
        lo, hi = stripped.split("-", maxsplit=1)
        return int(lo.strip()), int(hi.strip()) + 1
    raise ValueError(f"unsupported NHS age group {age_group!r}")


def build_nhs_cell_table(
    raw: pd.DataFrame, person: pd.DataFrame, household: pd.DataFrame
) -> pd.DataFrame:
    nhs = raw.copy()
    bounds = nhs["Age group"].map(parse_nhs_age_bounds)
    nhs["Lower age"] = [lo for lo, _ in bounds]
    nhs["Upper age"] = [hi for _, hi in bounds]
    nhs["Gender"] = nhs["Gender"].str.upper()
    pivot = nhs.pivot_table(
        index=["Lower age", "Upper age", "Gender", "Service"],
        columns="Metric",
        values="Total",
        aggfunc="sum",
    ).reset_index()
    top = (
        pivot[pivot["Lower age"] >= 85]
        .groupby(["Gender", "Service"], as_index=False)[
            ["Activity Count", "Total Cost"]
        ]
        .sum()
    )
    top["Lower age"] = 85
    top["Upper age"] = 120
    pivot = pd.concat([pivot[pivot["Lower age"] < 85], top], ignore_index=True)
    counts = _weighted_person_counts(person, household)
    pivot["Total people"] = [
        counts((row["Lower age"], row["Upper age"]), row["Gender"])
        for _, row in pivot.iterrows()
    ]
    pivot["Per-person average units"] = pivot["Activity Count"] / pivot["Total people"]
    factor = NHS_BUDGET_2025_26 / pivot["Total Cost"].sum()
    pivot["Per-person average spending"] = (
        pivot["Total Cost"] / pivot["Total people"] * factor
    )
    return pivot


def allocate_nhs_by_age_gender(
    person: pd.DataFrame,
    *,
    household_weights: np.ndarray,
    household: pd.DataFrame,
    nhs_table: pd.DataFrame | None,
) -> pd.DataFrame:
    if nhs_table is None:
        path = (
            Path(__file__).resolve().parents[1]
            / "uk/nhs_consumption_by_age_gender.json"
        )
        nhs_table = pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))["rows"])
    cells = build_nhs_cell_table(nhs_table, person, household)
    output = pd.DataFrame(0.0, index=person.index, columns=UK_NHS_OUTPUT_COLUMNS)
    service_to_columns = {
        "A&E": ("a_and_e_visits", "nhs_a_and_e_spending"),
        "AE": ("a_and_e_visits", "nhs_a_and_e_spending"),
        "Admitted Patient": (
            "admitted_patient_visits",
            "nhs_admitted_patient_spending",
        ),
        "APC": (
            "admitted_patient_visits",
            "nhs_admitted_patient_spending",
        ),
        "Outpatient": ("outpatient_visits", "nhs_outpatient_spending"),
        "OP": ("outpatient_visits", "nhs_outpatient_spending"),
    }
    ages = pd.to_numeric(person["age"], errors="coerce").fillna(0)
    genders = person["gender"].map(_enum_name).str.upper()
    for _, row in cells.iterrows():
        visit_col, spending_col = service_to_columns[row["Service"]]
        mask = (
            (ages >= row["Lower age"])
            & (ages < row["Upper age"])
            & (genders == row["Gender"])
        )
        output.loc[mask, visit_col] = row["Per-person average units"]
        output.loc[mask, spending_col] = row["Per-person average spending"]
    return output


def _weighted_person_counts(person: pd.DataFrame, household: pd.DataFrame):
    weight_by_household = household.set_index("household_id")["household_weight"]
    person_weights = person["person_household_id"].map(weight_by_household).fillna(0.0)
    ages = pd.to_numeric(person["age"], errors="coerce").fillna(0)
    genders = person["gender"].map(_enum_name).str.upper()

    def count(age_bounds: tuple[int, int], gender: str) -> float:
        lo, hi = age_bounds
        mask = (ages >= lo) & (ages < hi) & (genders == gender)
        total = float(person_weights[mask].sum())
        return total if total > 0 else 1.0

    return count


def _qrf_seed(stage: SourceStageSpec) -> int:
    for operation in stage.operations:
        if operation.kind == "fit_weighted_qrf_chain":
            return int(operation.parameters.get("seed", 0))
    return 0


def _require_path(path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("ETB services stage requires a caller-supplied ETB tab path.")
    return Path(path).expanduser().resolve()


def _enum_name(value: object) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value)
