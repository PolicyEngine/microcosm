"""PUF clone behavior on an already-assembled US spine pool."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.puf_support import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    clone_us_frame_for_puf_support,
    has_support_role_metadata,
    prepare_us_puf_tax_detail_chain_inputs,
    puf_tax_detail_clone_mask,
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_role_series,
    support_source_id_column,
)
from microcosm.build.us_runtime.support_provenance import spine_assembly_manifest
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def _assembled_spines() -> Frame:
    primary_ids = {
        "person": np.asarray([1, 101], dtype=np.int64),
        "household": np.asarray([1, 101], dtype=np.int64),
        "tax_unit": np.asarray([10, 110], dtype=np.int64),
        "spm_unit": np.asarray([100, 1_100], dtype=np.int64),
        "family": np.asarray([1_000, 11_000], dtype=np.int64),
        "marital_unit": np.asarray([10_000, 110_000], dtype=np.int64),
    }
    local_source_ids = {
        "person": np.asarray([1, 1], dtype=np.int64),
        "household": np.asarray([1, 1], dtype=np.int64),
        "tax_unit": np.asarray([10, 10], dtype=np.int64),
        "spm_unit": np.asarray([100, 100], dtype=np.int64),
        "family": np.asarray([1_000, 1_000], dtype=np.int64),
        "marital_unit": np.asarray([10_000, 10_000], dtype=np.int64),
    }
    channels = np.asarray(["asec", "acs"], dtype=object)

    tables: dict[str, pd.DataFrame] = {}
    for entity in US_SCHEMA.entities:
        primary = US_SCHEMA.entity_id_column(entity)
        tables[entity] = pd.DataFrame(
            {
                primary: primary_ids[entity],
                support_channel_column(entity): channels.copy(),
                support_source_id_column(entity): primary_ids[entity].copy(),
                spine_source_id_column(entity): local_source_ids[entity],
                support_clone_index_column(entity): np.zeros(2, dtype=np.int64),
            }
        )
    tables["person"] = tables["person"].assign(
        person_household_id=primary_ids["household"],
        person_tax_unit_id=primary_ids["tax_unit"],
        person_spm_unit_id=primary_ids["spm_unit"],
        person_family_id=primary_ids["family"],
        person_marital_unit_id=primary_ids["marital_unit"],
    )
    tables["tax_unit"]["filing_status_input"] = ["SINGLE", "JOINT"]
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([60.0, 40.0]),
                WeightKind.DESIGN,
            )
        },
        pd.Series(["asec_record", "acs_record"], name="stratum"),
        metadata=spine_assembly_manifest(
            tables,
            channels=("asec", "acs"),
        ),
    )


def test_puf_clone_preserves_source_spines_and_routes_by_clone_index() -> None:
    assembled = _assembled_spines()
    before = {
        entity: assembled.table(entity).copy(deep=True) for entity in assembled.entities
    }

    cloned = clone_us_frame_for_puf_support(assembled)

    for entity in assembled.entities:
        pd.testing.assert_frame_equal(assembled.table(entity), before[entity])
        table = cloned.table(entity)
        assert table[support_channel_column(entity)].tolist() == [
            "asec",
            "acs",
            "asec",
            "acs",
        ]
        assert table[support_clone_index_column(entity)].tolist() == [0, 0, 1, 1]
        assert table[support_source_id_column(entity)].tolist() == [
            *before[entity][support_source_id_column(entity)].tolist(),
            *before[entity][support_source_id_column(entity)].tolist(),
        ]
        assert table[spine_source_id_column(entity)].tolist() == [
            *before[entity][spine_source_id_column(entity)].tolist(),
            *before[entity][spine_source_id_column(entity)].tolist(),
        ]
        assert support_role_series(table, entity=entity).tolist() == [
            BASE_ASEC_SUPPORT_CHANNEL,
            BASE_ASEC_SUPPORT_CHANNEL,
            PUF_TAX_DETAIL_SUPPORT_CHANNEL,
            PUF_TAX_DETAIL_SUPPORT_CHANNEL,
        ]
        assert puf_tax_detail_clone_mask(table, entity=entity).tolist() == [
            False,
            False,
            True,
            True,
        ]

    assert cloned.weights_for("household").values.tolist() == [
        30.0,
        20.0,
        30.0,
        20.0,
    ]
    assert cloned.weights_for("household").total == 100.0


def test_puf_clone_rejects_support_channel_forged_through_mutable_table() -> None:
    assembled = _assembled_spines()
    assembled.table("person").loc[0, support_channel_column("person")] = "forged_source"

    with pytest.raises(ValueError, match="assembly manifest.*unknown channel"):
        clone_us_frame_for_puf_support(assembled)


def test_puf_qrf_preparation_accepts_all_source_spines() -> None:
    cloned = clone_us_frame_for_puf_support(_assembled_spines())
    donor = pd.DataFrame(
        {
            "puf_predictor_filing_status_code": [1.0, 2.0],
            "puf_predictor_tax_unit_person_count": [1.0, 1.0],
            "taxable_interest_income": [10.0, 20.0],
            "weight": [1.0, 1.0],
        }
    )

    inputs = prepare_us_puf_tax_detail_chain_inputs(
        cloned,
        donor,
        predictors=(
            "puf_predictor_filing_status_code",
            "puf_predictor_tax_unit_person_count",
        ),
        person_outputs=("taxable_interest_income",),
        tax_unit_outputs=(),
    )

    recipients = cloned.table("tax_unit").loc[inputs.recipient_features.index]
    assert recipients[support_channel_column("tax_unit")].tolist() == [
        "asec",
        "acs",
    ]
    assert recipients[support_clone_index_column("tax_unit")].tolist() == [1, 1]


def test_support_role_legacy_fallback_is_closed_to_known_roles() -> None:
    legacy = pd.DataFrame(
        {
            support_channel_column("person"): [
                BASE_ASEC_SUPPORT_CHANNEL,
                PUF_TAX_DETAIL_SUPPORT_CHANNEL,
            ]
        }
    )

    assert has_support_role_metadata(legacy, entity="person")
    assert support_role_series(legacy, entity="person").tolist() == [
        BASE_ASEC_SUPPORT_CHANNEL,
        PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    ]
    assert puf_tax_detail_clone_mask(legacy, entity="person").tolist() == [
        False,
        True,
    ]

    legacy.loc[1, support_channel_column("person")] = "acs"
    with pytest.raises(ValueError, match="exact ASEC/PUF"):
        support_role_series(legacy, entity="person")
    legacy.loc[1, support_channel_column("person")] = None
    with pytest.raises(ValueError, match="complete support provenance"):
        support_role_series(legacy, entity="person")
    assert not has_support_role_metadata(pd.DataFrame(), entity="person")
