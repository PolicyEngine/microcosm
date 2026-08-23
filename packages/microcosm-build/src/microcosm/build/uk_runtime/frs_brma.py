"""UK FRS BRMA stochastic assignment from a committed count table."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import (
    sample_categorical_from_counts,
    stable_identity_uniforms,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.frame import Frame
from microcosm.frame.rules import assert_rules_engine_country

FRS_BRMA_OUTPUT_COLUMNS = ("brma",)
UK_BRMA_PREDICTORS = ("LHA_category",)
UK_BRMA_DECLARED_SEEDS = {"brma": 0}


@dataclass(frozen=True)
class UKFRSBRMAStageTransform:
    """Whole-stage callable for UK household BRMA assignment."""

    stage: SourceStageSpec
    engine: object
    count_resource: Mapping[str, Any] | None = None

    def __call__(self, frame: Frame) -> Frame:
        assert_rules_engine_country(self.engine, "uk")
        materialized = self.engine.materialize(
            frame, UK_BRMA_PREDICTORS, uk_time_period(frame)
        )
        resource = (
            self.count_resource
            if self.count_resource is not None
            else load_brma_count_resource()
        )
        return add_frs_brma(
            frame, lha_category=materialized["LHA_category"], count_resource=resource
        )

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return FRS_BRMA_OUTPUT_COLUMNS


def load_brma_count_resource() -> Mapping[str, Any]:
    return json.loads(
        files("microcosm.build.uk").joinpath("brma_rent_counts.json").read_text()
    )


def add_frs_brma(
    frame: Frame,
    *,
    lha_category: Sequence[object],
    count_resource: Mapping[str, Any],
) -> Frame:
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    household = frame.table("household").copy()
    if len(lha_category) != len(benunit):
        raise ValueError("LHA_category materialization must align to benunit rows.")
    benunit["LHA_category"] = [_enum_name(value) for value in lha_category]
    benunit["region"] = _benunit_regions(person, household, benunit)
    benunit["brma"] = assign_brma_by_cell(
        benunit,
        count_resource=count_resource,
        seed=UK_BRMA_DECLARED_SEEDS["brma"],
    )
    household["brma"] = collapse_benunit_brma_to_household(
        person,
        benunit[["benunit_id", "brma"]],
        household,
        seed=UK_BRMA_DECLARED_SEEDS["brma"],
    )
    benunit = benunit.drop(columns=["LHA_category", "region", "brma"])
    result = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )
    validate_uk_national_frame(result)
    return result


def assign_brma_by_cell(
    benunit: pd.DataFrame,
    *,
    count_resource: Mapping[str, Any],
    seed: int,
) -> np.ndarray:
    cells = count_resource["cells"]
    draws = stable_identity_uniforms(
        benunit["benunit_id"].to_numpy(), seed=seed, salt="brma"
    )
    assigned: list[str] = []
    for index, row in enumerate(benunit.itertuples(index=False)):
        region = _enum_name(row.region)
        category = _enum_name(row.LHA_category)
        try:
            counts = cells[region][category]
        except KeyError as exc:
            raise KeyError(
                f"missing BRMA count-table cell for region={region!r}, "
                f"LHA_category={category!r}."
            ) from exc
        assigned.append(
            str(
                sample_categorical_from_counts(np.array([draws[index]]), counts=counts)[
                    0
                ]
            )
        )
    return np.asarray(assigned, dtype=object)


def collapse_benunit_brma_to_household(
    person: pd.DataFrame,
    benunit_brma: pd.DataFrame,
    household: pd.DataFrame,
    *,
    seed: int,
) -> np.ndarray:
    mapping = _benunit_household_map(person)
    values = benunit_brma.copy()
    values["household_id"] = values["benunit_id"].map(mapping)
    household_ids = household["household_id"].to_numpy()
    draws = stable_identity_uniforms(
        household_ids, seed=seed, salt="brma:household_pick"
    )
    options_by_household = {
        household_id: group.sort_values("benunit_id")["brma"].astype(str).to_list()
        for household_id, group in values.groupby("household_id", sort=False)
    }
    assigned: list[str] = []
    for index, household_id in enumerate(household_ids):
        options = options_by_household.get(household_id)
        if not options:
            raise ValueError(f"household {household_id!r} has no benunit BRMA options.")
        pick = min(int(draws[index] * len(options)), len(options) - 1)
        assigned.append(options[pick])
    return np.asarray(assigned, dtype=object)


def _benunit_regions(
    person: pd.DataFrame, household: pd.DataFrame, benunit: pd.DataFrame
) -> np.ndarray:
    benunit_to_household = _benunit_household_map(person)
    household_regions = household.set_index("household_id")["region"]
    return (
        benunit["benunit_id"]
        .map(benunit_to_household)
        .map(household_regions)
        .map(_enum_name)
        .to_numpy()
    )


def _benunit_household_map(person: pd.DataFrame) -> pd.Series:
    placements = person[["person_benunit_id", "person_household_id"]].drop_duplicates()
    if placements["person_benunit_id"].duplicated().any():
        duplicated = placements.loc[
            placements["person_benunit_id"].duplicated(keep=False),
            "person_benunit_id",
        ].unique()
        raise ValueError(
            f"benunit id(s) split across households: {duplicated[:5].tolist()}."
        )
    return placements.set_index("person_benunit_id")["person_household_id"]


def _enum_name(value: object) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value)
