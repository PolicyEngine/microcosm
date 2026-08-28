"""Pooled immigration absolute-control mass regressions (microcosm #767)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.us_runtime import immigration as immigration_module
from microcosm.build.us_runtime import multispine_pool as multispine_pool_module
from microcosm.build.us_runtime.immigration import (
    US_IMMIGRATION_OUTPUT_COLUMNS,
    US_IMMIGRATION_STAGE_NAME,
    with_us_immigration_inputs,
)
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.spine_assembly import assemble_spines
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def _humanitarian_controls(*, ukraine: float) -> dict[str, object]:
    parole_origins = ("afghanistan", "ukraine", "nicaragua", "venezuela")
    tps_origins = (
        "venezuela",
        "el_salvador",
        "honduras",
        "nicaragua",
        "nepal",
        "other_designated",
    )
    return {
        "paroled_one_year": {
            origin: {
                "target": ukraine if origin == "ukraine" else 0.0,
                "source": f"https://example.com/parole/{origin}",
            }
            for origin in parole_origins
        },
        "refugee": {"target": 0.0, "source": "https://example.com/refugee"},
        "asylee": {"target": 0.0, "source": "https://example.com/asylee"},
        "deportation_withheld": {
            "target": 0.0,
            "source": "https://example.com/withheld",
        },
        "tps": {
            origin: {
                "target": 0.0,
                "source": f"https://example.com/tps/{origin}",
            }
            for origin in tps_origins
        },
    }


def _stage_spec() -> SourceStageSpec:
    return SourceStageSpec.from_mapping(
        {
            "stage": US_IMMIGRATION_STAGE_NAME,
            "survey": "test ASEC",
            "source": "https://example.com",
            "grain": "person",
            "operations": [
                {"kind": "read_table", "table": "person"},
                {
                    "kind": "derive_immigration_status",
                    "seed_from_build_config": True,
                    "time_period_from_build_config": True,
                    "undocumented_workers": {
                        "target": 4.0,
                        "source": "https://example.com/workers",
                    },
                    "undocumented_students": {
                        "target": 2.0,
                        "source": "https://example.com/students",
                    },
                    "undocumented_population_anchor": {
                        "value": 10.0,
                        "source": "https://example.com/population",
                    },
                    "humanitarian_status_stocks": _humanitarian_controls(ukraine=4.0),
                },
            ],
            "outputs": list(US_IMMIGRATION_OUTPUT_COLUMNS),
        }
    )


def _immigration_frame() -> Frame:
    rows: list[dict[str, object]] = []
    rows.extend({"PRCITSHP": 5, "WSAL_VAL": 10_000.0} for _ in range(4))
    rows.extend({"PRCITSHP": 5, "A_HSCOL": 2} for _ in range(3))
    rows.extend(
        {
            "PRCITSHP": 5,
            "PENATVTY": 164,
            "PEINUSYR": 28,
            "CAID": 1,
        }
        for _ in range(8)
    )
    baseline: dict[str, object] = {
        "PRCITSHP": 1,
        "PEINUSYR": 0,
        "PENATVTY": 57,
        "A_AGE": 40,
        "A_MARITL": 7,
        "A_SPOUSE": 0,
        "A_HSCOL": 0,
        "WSAL_VAL": 0.0,
        "SEMP_VAL": 0.0,
        "MCARE": 2,
        "CAID": 2,
        "IHSFLG": 2,
        "CHAMPVA": 2,
        "MIL": 2,
        "PEN_SC1": 0,
        "PEN_SC2": 0,
        "RESNSS1": 0,
        "RESNSS2": 0,
        "SS_YN": 2,
        "SSI_YN": 2,
        "PEIO1COW": 0,
        "A_MJOCC": 0,
        "PEAFEVER": 2,
        "SPM_CAPHOUSESUB": 0.0,
    }
    records = []
    for position, overrides in enumerate(rows, start=1):
        record = dict(baseline)
        record.update(overrides)
        record.update(
            {
                "person_id": position,
                "person_household_id": position,
                "person_tax_unit_id": position,
                "person_spm_unit_id": position,
                "person_family_id": position,
                "person_marital_unit_id": position,
            }
        )
        records.append(record)
    person = pd.DataFrame(records)
    ids = np.arange(1, len(person) + 1, dtype=np.int64)
    return Frame(
        {
            "person": person,
            **{
                entity: pd.DataFrame({f"{entity}_id": ids})
                for entity in US_SCHEMA.group_entities
            },
        },
        US_SCHEMA,
        {
            "household": Weights(
                np.ones(len(person), dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )


def _weighted_count(frame: Frame, mask: pd.Series) -> float:
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    return float(weights[mask.to_numpy(dtype=bool)].sum())


def test_stage_weight_scale_allocates_every_absolute_control_to_source_share(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(immigration_module, "us_immigration_stage_spec", _stage_spec)

    standalone = with_us_immigration_inputs(
        _immigration_frame(), seed=0, time_period=2024
    )
    pooled_projection = with_us_immigration_inputs(
        _immigration_frame(),
        seed=0,
        time_period=2024,
        person_weight_scale=2.0,
    )

    standalone_person = standalone.table("person")
    pooled_person = pooled_projection.table("person")
    assert (
        _weighted_count(
            standalone,
            standalone_person["immigration_status_str"].eq("PAROLED_ONE_YEAR"),
        )
        == 4.0
    )
    assert (
        _weighted_count(
            pooled_projection,
            pooled_person["immigration_status_str"].eq("PAROLED_ONE_YEAR"),
        )
        == 2.0
    )

    standalone_workers = standalone_person["WSAL_VAL"].gt(0)
    pooled_workers = pooled_person["WSAL_VAL"].gt(0)
    assert (
        _weighted_count(
            standalone,
            standalone_workers & standalone_person["ssn_card_type"].eq("NONE"),
        )
        == 4.0
    )
    assert (
        _weighted_count(
            pooled_projection,
            pooled_workers & pooled_person["ssn_card_type"].eq("NONE"),
        )
        == 2.0
    )

    standalone_students = standalone_person["A_HSCOL"].eq(2)
    pooled_students = pooled_person["A_HSCOL"].eq(2)
    assert (
        _weighted_count(
            standalone,
            standalone_students & standalone_person["ssn_card_type"].eq("NONE"),
        )
        == 2.0
    )
    assert (
        _weighted_count(
            pooled_projection,
            pooled_students & pooled_person["ssn_card_type"].eq("NONE"),
        )
        == 1.0
    )


def _spine(*, cps: bool, offset: int = 0) -> Frame:
    ids = np.asarray([1, 2], dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_tax_unit_id": ids,
            "person_spm_unit_id": ids,
            "person_family_id": ids,
            "person_marital_unit_id": ids,
            "age": [30.0, 50.0],
            **({"PERIDNUM": [f"p{offset + 1}", f"p{offset + 2}"]} if cps else {}),
        }
    )
    return Frame(
        {
            "person": person,
            **{
                entity: pd.DataFrame({f"{entity}_id": ids})
                for entity in US_SCHEMA.group_entities
            },
        },
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([2.0, 2.0]),
                WeightKind.DESIGN,
            )
        },
    )


def _replace_person(frame: Frame, person: pd.DataFrame) -> Frame:
    return Frame(
        {
            "person": person,
            **{
                entity: frame.table(entity)
                for entity in frame.entities
                if entity != "person"
            },
        },
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def test_postclone_chain_derives_and_receipts_inverse_cps_person_mass_share() -> None:
    assembled = assemble_spines(
        {"asec": _spine(cps=True), "acs": _spine(cps=False, offset=100)},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    cloned = clone_us_frame_for_puf_support(assembled)
    observed_scale: list[float] = []

    def immigration(
        available: Frame,
        *,
        person_weight_scale: float,
    ) -> Frame:
        observed_scale.append(person_weight_scale)
        person = available.table("person").copy()
        person["ssn_card_type"] = "CITIZEN"
        person["immigration_status_str"] = "CITIZEN"
        return _replace_person(available, person)

    completed = multispine_pool_module._run_source_operator_chain(
        cloned,
        phase="post_clone",
        operator_names=("with_us_immigration_inputs",),
        operators={"with_us_immigration_inputs": immigration},
    )

    person = cloned.table("person")
    weights = np.asarray(cloned.resolve_weights("person").values, dtype=np.float64)
    cps = person["PERIDNUM"].notna().to_numpy(dtype=bool)
    full_mass = float(weights.sum())
    cps_mass = float(weights[cps].sum())
    expected_scale = full_mass / cps_mass
    assert cps_mass / full_mass == pytest.approx(0.5)
    assert observed_scale == [pytest.approx(expected_scale)]

    scaling = completed.receipt["suboperators"][0]["kernel_receipt"][
        "person_design_weight_scaling"
    ]
    assert scaling == {
        "full_pool_person_design_weight_mass": pytest.approx(full_mass),
        "cps_projection_person_design_weight_mass": pytest.approx(cps_mass),
        "person_weight_scale": pytest.approx(expected_scale),
        "cps_person_mass_share": pytest.approx(cps_mass / full_mass),
    }
