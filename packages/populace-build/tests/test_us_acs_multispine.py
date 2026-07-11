from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from populace.build.gates import FitWeightRecord
from populace.build.us_runtime import acs_multispine
from populace.build.us_runtime.acs_pums import AcsPumsSource
from populace.build.us_runtime.acs_transfer import AcsImputedInput
from populace.build.us_runtime.base_pool import spine_column
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights


def _must_not_run(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("the disabled ACS path must do no downstream work")


def test__given_no_source__then_base_frame_is_an_untouched_identity(
    monkeypatch,
) -> None:
    base = object()
    monkeypatch.setattr(
        acs_multispine,
        "build_acs_pums_unit_frame",
        _must_not_run,
    )
    monkeypatch.setattr(acs_multispine, "map_acs_native_inputs", _must_not_run)
    monkeypatch.setattr(acs_multispine, "transfer_acs_inputs", _must_not_run)
    monkeypatch.setattr(acs_multispine, "with_optional_acs_spine", _must_not_run)

    result = acs_multispine.build_optional_acs_multispine(
        cast(Frame, base),
        source=None,
        chunksize=0,
        acs_share=2.0,
        seed=-1,
        n_estimators=0,
    )

    assert result.frame is base
    assert result.fit_records == ()
    assert result.provenance == {"enabled": False}
    json.dumps(result.provenance, allow_nan=False)


def test__given_source__then_stages_run_in_order_and_provenance_is_json_ready(
    monkeypatch,
    tmp_path,
) -> None:
    events: list[tuple[Any, ...]] = []
    base = object()
    raw_acs = object()
    mapped_acs = object()
    transferred_acs = object()
    pooled = object()
    source = AcsPumsSource(
        tmp_path / "csv_hus.zip",
        tmp_path / "csv_pus.zip",
    )
    target_families = {"person": {"tax_detail": ("taxable_interest_income",)}}
    fit_record = FitWeightRecord("acs_transfer:person:tax_detail", "design")
    imputed = AcsImputedInput(
        column="taxable_interest_income",
        entity="person",
        family="tax_detail",
        donor_spine="asec_puf",
        donor_channel="puf_support",
        predictors=("age", "is_female"),
        seed=91,
        weight_kind="design",
    )

    def fake_load(actual_source, *, chunksize):
        events.append(("load", actual_source, chunksize))
        return raw_acs, {
            "spine": "acs_2024_1yr",
            "person_rows": np.int64(2),
            "scratch_path": Path("inputs/acs"),
        }

    def fake_map(actual_raw):
        events.append(("map", actual_raw))
        return SimpleNamespace(
            frame=mapped_acs,
            native_inputs={
                "age": {
                    "entity": "person",
                    "source_columns": ("AGEP",),
                }
            },
        )

    def fake_transfer(actual_recipient, actual_donor, **kwargs):
        events.append(("transfer", actual_recipient, actual_donor, kwargs))
        return SimpleNamespace(
            frame=transferred_acs,
            imputed_inputs=(imputed,),
            fit_records=(fit_record,),
            deferred_inputs=("congressional_district_geoid",),
            resolved_donor_channel="puf_support",
        )

    def fake_pool(actual_base, actual_acs, *, acs_share):
        events.append(("pool", actual_base, actual_acs, acs_share))
        return pooled

    monkeypatch.setattr(acs_multispine, "build_acs_pums_unit_frame", fake_load)
    monkeypatch.setattr(acs_multispine, "map_acs_native_inputs", fake_map)
    monkeypatch.setattr(acs_multispine, "transfer_acs_inputs", fake_transfer)
    monkeypatch.setattr(acs_multispine, "with_optional_acs_spine", fake_pool)

    result = acs_multispine.build_optional_acs_multispine(
        cast(Frame, base),
        source,
        chunksize=25_000,
        acs_share=0.4,
        target_families=target_families,
        donor_spine="asec_puf",
        donor_channel="puf_support",
        seed=19,
        n_estimators=37,
    )

    assert events == [
        ("load", source, 25_000),
        ("map", raw_acs),
        (
            "transfer",
            mapped_acs,
            base,
            {
                "target_families": target_families,
                "donor_spine": "asec_puf",
                "donor_channel": "puf_support",
                "seed": 19,
                "n_estimators": 37,
                "max_targets_per_fit": 8,
            },
        ),
        ("pool", base, transferred_acs, 0.4),
    ]
    assert result.frame is pooled
    assert result.fit_records == (fit_record,)
    assert result.provenance == {
        "enabled": True,
        "acs_share": 0.4,
        "loader": {
            "spine": "acs_2024_1yr",
            "person_rows": 2,
            "scratch_path": "inputs/acs",
        },
        "native_inputs": {
            "age": {
                "entity": "person",
                "source_columns": ["AGEP"],
            }
        },
        "imputed_inputs": [
            {
                "column": "taxable_interest_income",
                "entity": "person",
                "family": "tax_detail",
                "donor_spine": "asec_puf",
                "donor_channel": "puf_support",
                "predictors": ["age", "is_female"],
                "seed": 91,
                "weight_kind": "design",
                "patterns": [],
                "unmodeled_recipient_rows": 0,
            }
        ],
        "deferred_inputs": ["congressional_district_geoid"],
        "fit_records": [
            {
                "fit_name": "acs_transfer:person:tax_detail",
                "weight_kind": "design",
            }
        ],
        "fit_configuration": {
            "donor_spine": "asec_puf",
            "requested_donor_channel": "puf_support",
            "resolved_donor_channel": "puf_support",
            "seed": 19,
            "n_estimators": 37,
            "max_targets_per_fit": 8,
        },
    }
    json.dumps(result.provenance, allow_nan=False)


def test__given_tiny_frames__then_mapping_transfer_and_pool_interoperate(
    monkeypatch,
    tmp_path,
) -> None:
    base = _base_donor_frame()
    raw_acs = _raw_acs_frame()
    source = AcsPumsSource(
        tmp_path / "csv_hus.zip",
        tmp_path / "csv_pus.zip",
    )

    monkeypatch.setattr(
        acs_multispine,
        "build_acs_pums_unit_frame",
        lambda actual_source, *, chunksize: (
            raw_acs,
            {
                "spine": "acs_2024_1yr",
                "vintage": actual_source.vintage,
                "chunksize": chunksize,
            },
        ),
    )

    result = acs_multispine.build_optional_acs_multispine(
        base,
        source,
        chunksize=2,
        acs_share=0.25,
        target_families={"person": {"tax_detail": ("qualified_dividend_income",)}},
        seed=4,
        n_estimators=2,
    )

    person = result.frame.table("person")
    acs_people = person[person[spine_column("person")] == "acs_2024_1yr"]
    assert acs_people["employment_income_before_lsr"].tolist() == pytest.approx(
        [55_000.0, 0.0]
    )
    assert acs_people["qualified_dividend_income"].notna().all()
    assert (
        result.frame.weights_for("household").total
        == base.weights_for("household").total
    )
    assert {item["column"] for item in result.provenance["imputed_inputs"]} == {
        "qualified_dividend_income"
    }
    assert result.fit_records[0].weight_kind == "design"


def _base_donor_frame() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_household_id": [1, 1, 2, 2],
            "person_tax_unit_id": [1, 1, 2, 2],
            "person_spm_unit_id": [1, 1, 2, 2],
            "person_family_id": [1, 1, 2, 2],
            "person_marital_unit_id": [1, 1, 2, 2],
            "age": [42.0, 38.0, 65.0, 12.0],
            "is_female": [False, True, True, False],
            "qualified_dividend_income": [100.0, 20.0, 2_000.0, 0.0],
        }
    )
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame({"household_id": [1, 2], "state_fips": [6, 36]}),
            "tax_unit": pd.DataFrame({"tax_unit_id": [1, 2]}),
            "spm_unit": pd.DataFrame({"spm_unit_id": [1, 2]}),
            "family": pd.DataFrame({"family_id": [1, 2]}),
            "marital_unit": pd.DataFrame({"marital_unit_id": [1, 2]}),
        },
        US_SCHEMA,
        {"household": Weights(np.asarray([100.0, 300.0]), WeightKind.DESIGN)},
        pd.Series("asec_puf", index=person.index, dtype=object),
    )


def _raw_acs_frame() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": [10, 11],
            "person_household_id": [10, 10],
            "person_tax_unit_id": [10, 10],
            "person_spm_unit_id": [10, 10],
            "person_family_id": [10, 10],
            "person_marital_unit_id": [10, 10],
            "AGEP": [40, 15],
            "SEX": [1, 2],
            "RELSHIPP": [20, 25],
            "ADJINC": [1_100_000, 1_100_000],
            "WAGP": [50_000.0, 0.0],
            "SEMP": [0.0, np.nan],
            "SSP": [0.0, 0.0],
            "SSIP": [0.0, 0.0],
            "RETP": [0.0, 0.0],
            "INTP": [100.0, 0.0],
        }
    )
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {
                    "household_id": [10],
                    "state_fips": [6],
                    "ADJHSG": [1_000_000],
                    "TEN": [3],
                    "RNTP": [1_000.0],
                    "GRNTP": [1_200.0],
                    "TAXAMT": [np.nan],
                }
            ),
            "tax_unit": pd.DataFrame({"tax_unit_id": [10]}),
            "spm_unit": pd.DataFrame({"spm_unit_id": [10]}),
            "family": pd.DataFrame({"family_id": [10]}),
            "marital_unit": pd.DataFrame({"marital_unit_id": [10]}),
        },
        US_SCHEMA,
        {"household": Weights(np.asarray([50.0]), WeightKind.DESIGN)},
        pd.Series("acs_2024_1yr", index=person.index, dtype=object),
    )
