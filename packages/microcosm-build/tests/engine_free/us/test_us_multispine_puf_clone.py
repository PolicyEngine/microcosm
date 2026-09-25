"""PUF clone behavior on an already-assembled US spine pool."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.puf_support import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    clone_us_frame_for_puf_support,
    has_assembled_support_metadata,
    has_support_role_metadata,
    prepare_us_puf_tax_detail_chain_inputs,
    puf_tax_detail_clone_mask,
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    support_role_series,
    support_source_id_column,
)
from microcosm.build.us_runtime.support_provenance import (
    require_assembled_support_provenance,
    spine_assembly_manifest,
    support_copy_rank_series,
    support_gate_source_channel_series,
    without_support_role_metadata,
)
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
        assert has_assembled_support_metadata(table, entity=entity)
        assert support_gate_source_channel_series(table, entity=entity).tolist() == [
            "asec",
            "acs",
            "asec",
            "acs",
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
    assert not has_assembled_support_metadata(legacy, entity="person")
    assert support_gate_source_channel_series(legacy, entity="person").tolist() == [
        BASE_ASEC_SUPPORT_CHANNEL,
        PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    ]
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


def test_support_copy_rank_uses_clone_index_on_every_frame_kind() -> None:
    # A historical PUF-support base with a capital-gains own-tail copy: two
    # PUF-role rows of source 7 that only the clone index tells apart.
    historical = pd.DataFrame(
        {
            support_source_id_column("person"): [7, 7, 7, 8, 8],
            support_channel_column("person"): [
                BASE_ASEC_SUPPORT_CHANNEL,
                PUF_TAX_DETAIL_SUPPORT_CHANNEL,
                PUF_TAX_DETAIL_SUPPORT_CHANNEL,
                BASE_ASEC_SUPPORT_CHANNEL,
                PUF_TAX_DETAIL_SUPPORT_CHANNEL,
            ],
            support_clone_index_column("person"): [0, 1, 2, 0, 1],
        },
        index=[10, 11, 12, 13, 14],
    )
    assert not has_assembled_support_metadata(historical, entity="person")
    assert support_role_series(historical, entity="person").tolist() == [
        BASE_ASEC_SUPPORT_CHANNEL,
        PUF_TAX_DETAIL_SUPPORT_CHANNEL,
        PUF_TAX_DETAIL_SUPPORT_CHANNEL,
        BASE_ASEC_SUPPORT_CHANNEL,
        PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    ]
    ranks = support_copy_rank_series(historical, entity="person")
    assert ranks.tolist() == [0, 1, 2, 0, 1]
    assert ranks.index.tolist() == [10, 11, 12, 13, 14]
    assert ranks.dtype == np.int64
    keyed = pd.DataFrame(
        {"source": historical[support_source_id_column("person")], "rank": ranks}
    )
    assert not keyed.duplicated().any()

    assembled = historical.assign(
        **{
            spine_source_id_column("person"): [1, 1, 1, 2, 2],
            support_channel_column("person"): ["acs"] * 5,
        }
    )
    assert support_copy_rank_series(assembled, entity="person").tolist() == [
        0,
        1,
        2,
        0,
        1,
    ]

    channel_only = historical.drop(columns=[support_clone_index_column("person")])
    assert support_copy_rank_series(channel_only, entity="person").tolist() == [
        0,
        1,
        1,
        0,
        1,
    ]

    # Rank inherits the role validator: a clone index must agree with its
    # historical channel, and metadata must be present.
    inconsistent = historical.copy()
    inconsistent.loc[12, support_clone_index_column("person")] = 0
    with pytest.raises(ValueError, match="inconsistent"):
        support_copy_rank_series(inconsistent, entity="person")
    with pytest.raises(ValueError, match="missing both"):
        support_copy_rank_series(pd.DataFrame({"x": [1]}), entity="person")


def _assembled_copies(*, channel: bool, clone_index: bool) -> pd.DataFrame:
    """Two assembled source people, each with a native and a donor copy."""

    table = pd.DataFrame(
        {
            support_source_id_column("tax_unit"): [7, 7, 8, 8],
            spine_source_id_column("tax_unit"): [70, 70, 80, 80],
        },
        index=[30, 31, 32, 33],
    )
    if channel:
        table[support_channel_column("tax_unit")] = ["asec"] * 4
    if clone_index:
        table[support_clone_index_column("tax_unit")] = [0, 1, 0, 1]
    return table


@pytest.mark.parametrize(
    ("channel", "clone_index", "missing"),
    [
        pytest.param(
            False,
            False,
            "'tax_unit_support_channel' and 'tax_unit_support_clone_index'",
            id="both_missing",
        ),
        pytest.param(
            True, False, "'tax_unit_support_clone_index'", id="clone_index_missing"
        ),
        pytest.param(False, True, "'tax_unit_support_channel'", id="channel_missing"),
    ],
)
def test_assembled_support_provenance_requires_channel_and_clone_index(
    channel: bool,
    clone_index: bool,
    missing: str,
) -> None:
    # Microcosm #992 gate findings: assembled tables that lost a provenance
    # column fell back to channel-only role ranks, (source, role) occurrence
    # pairing or the no-metadata one-row-per-source path, each of which hid a
    # support-copy disagreement. One owner-level validator refuses them all.
    # The role reader calls it first, so every role reader refuses, including
    # the prior-year and SSI summaries; the copy rank also calls it before its
    # role-rank fallback, and the Head Start and voluntary-filing consumers
    # call it at entry (pinned in the consumer tests).
    table = _assembled_copies(channel=channel, clone_index=clone_index)
    assert has_support_role_metadata(table, entity="tax_unit")
    message = (
        f"assembled support metadata requires {missing} alongside "
        "'tax_unit_spine_source_id': "
    )
    for accessor in (
        require_assembled_support_provenance,
        support_role_series,
        support_copy_rank_series,
        puf_tax_detail_clone_mask,
        support_gate_source_channel_series,
        without_support_role_metadata,
    ):
        with pytest.raises(ValueError) as refused:
            accessor(table, entity="tax_unit")
        assert str(refused.value).startswith(message), accessor.__name__


def test_assembled_support_provenance_accepts_complete_and_historical_tables() -> None:
    complete = _assembled_copies(channel=True, clone_index=True)
    assert require_assembled_support_provenance(complete, entity="tax_unit") is None
    assert support_copy_rank_series(complete, entity="tax_unit").tolist() == [
        0,
        1,
        0,
        1,
    ]
    roles = support_role_series(complete, entity="tax_unit")
    assert roles.tolist() == ["asec", "puf_tax_detail"] * 2
    channels = support_gate_source_channel_series(complete, entity="tax_unit")
    assert channels.tolist() == ["asec"] * 4
    # Without a raw spine ID the table is historical: clone-index, channel-only
    # and metadata-free layouts are all left to their own validators.
    for channel, clone_index in ((True, True), (True, False), (False, False)):
        historical = _assembled_copies(channel=channel, clone_index=clone_index).drop(
            columns=[spine_source_id_column("tax_unit")]
        )
        assert (
            require_assembled_support_provenance(historical, entity="tax_unit") is None
        )
    with pytest.raises(ValueError, match="entity must be a non-empty string"):
        require_assembled_support_provenance(complete, entity="")


def test_support_projection_removes_assembly_marker_without_mutating_input() -> None:
    table = _assembled_copies(channel=True, clone_index=True)
    original = table.copy(deep=True)
    projected = without_support_role_metadata(table, entity="tax_unit")
    assert not has_support_role_metadata(projected, entity="tax_unit")
    assert not has_assembled_support_metadata(projected, entity="tax_unit")
    pd.testing.assert_frame_equal(
        projected, table[[support_source_id_column("tax_unit")]]
    )
    pd.testing.assert_frame_equal(table, original)


_CLONE_INDEX_ACCESSORS = pytest.mark.parametrize(
    "accessor",
    [support_role_series, support_copy_rank_series, puf_tax_detail_clone_mask],
    ids=["role", "copy_rank", "puf_detail_mask"],
)
_SUPPORT_FRAME_KINDS = pytest.mark.parametrize(
    "assembled",
    [True, False],
    ids=["assembled", "historical_puf_support"],
)
_MALFORMED_CLONE_INDEX = (
    r"PUF support metadata column 'person_support_clone_index' must contain "
    r"nonnegative integers \(finite and representable as int64\)"
)


def _support_copies(clone_indices: Any, *, assembled: bool) -> pd.DataFrame:
    """Support copies of source 7: its native row, then PUF-role copies.

    Historical PUF-support frames carry the exact ASEC/PUF roles as channels;
    assembled frames carry a raw spine ID and a receipt-declared channel.
    """

    count = len(clone_indices)
    table = pd.DataFrame(
        {
            support_source_id_column("person"): [7] * count,
            support_channel_column("person"): [
                BASE_ASEC_SUPPORT_CHANNEL,
                *[PUF_TAX_DETAIL_SUPPORT_CHANNEL] * (count - 1),
            ],
            support_clone_index_column("person"): clone_indices,
        },
        index=[20 + position for position in range(count)],
    )
    if assembled:
        table[spine_source_id_column("person")] = [1] * count
        table[support_channel_column("person")] = ["acs"] * count
    return table


@_CLONE_INDEX_ACCESSORS
@_SUPPORT_FRAME_KINDS
@pytest.mark.parametrize(
    "clone_indices",
    [
        pytest.param([0, np.inf], id="inf"),
        pytest.param([0, -np.inf], id="negative_inf"),
        pytest.param([0, np.nan], id="nan"),
        pytest.param([0, 1.5], id="non_integer"),
        pytest.param([0, -1], id="negative"),
        # A finite negative float takes the float branch, not the integer
        # one, so it pins that branch's own nonnegativity check.
        pytest.param([0.0, -1.0], id="negative_float"),
        pytest.param([0, 2.0**63], id="float_past_int64"),
        pytest.param([0, "inf"], id="string_inf"),
        pytest.param(pd.array([0, pd.NA], dtype="Int64"), id="nullable_missing"),
        pytest.param(
            np.asarray([0, 2**63], dtype=np.uint64),
            id="unsigned_past_int64",
        ),
    ],
)
def test_clone_index_conversions_reject_malformed_provenance_before_casting(
    accessor: Callable[..., object],
    assembled: bool,
    clone_indices: Any,
) -> None:
    # Microcosm #992 gate finding: a raw NumPy cast maps +inf (and any float
    # past the int64 range) to an arbitrary int64 with only a RuntimeWarning,
    # so [0, inf] once passed Head Start canonical selection as a copy rank.
    # Escalating that warning proves the refusal happens before any cast.
    table = _support_copies(clone_indices, assembled=assembled)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(ValueError, match=_MALFORMED_CLONE_INDEX):
            accessor(table, entity="person")


@_SUPPORT_FRAME_KINDS
@pytest.mark.parametrize(
    ("clone_indices", "expected_ranks"),
    [
        pytest.param(np.asarray([0, 1, 2], dtype=np.int64), [0, 1, 2], id="int64"),
        pytest.param(np.asarray([0.0, 1.0, 2.0]), [0, 1, 2], id="float64"),
        pytest.param(pd.array([0, 1, 2], dtype="Int64"), [0, 1, 2], id="nullable"),
        pytest.param(np.asarray([0, 1, 2], dtype=np.uint64), [0, 1, 2], id="uint64"),
        pytest.param(["0", "1", "2"], [0, 1, 2], id="numeric_strings"),
    ],
)
def test_clone_index_conversions_leave_valid_encodings_unchanged(
    assembled: bool,
    clone_indices: Any,
    expected_ranks: list[int],
) -> None:
    table = _support_copies(clone_indices, assembled=assembled)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        roles = support_role_series(table, entity="person")
        ranks = support_copy_rank_series(table, entity="person")
        mask = puf_tax_detail_clone_mask(table, entity="person")

    assert roles.tolist() == [
        BASE_ASEC_SUPPORT_CHANNEL,
        PUF_TAX_DETAIL_SUPPORT_CHANNEL,
        PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    ]
    assert ranks.tolist() == expected_ranks
    assert ranks.dtype == np.int64
    assert ranks.index.tolist() == table.index.tolist()
    assert mask.tolist() == [False, True, False]


@pytest.mark.parametrize("clone_index", [3, 7, 2**53 + 1, 2**62])
@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_copy_rank_restricts_historical_domain_but_preserves_assembled_indices(
    clone_index: int,
    dtype: type,
) -> None:
    indices = np.asarray([0, 1, clone_index], dtype=dtype)
    historical = _support_copies(indices, assembled=False)
    with pytest.raises(ValueError, match="historical support clone indices.*0, 1, 2"):
        support_copy_rank_series(historical, entity="person")

    assembled = _support_copies(indices, assembled=True)
    ranks = support_copy_rank_series(assembled, entity="person")
    assert ranks.tolist() == indices.astype(np.int64).tolist()
    assert ranks.dtype == np.int64


@pytest.mark.parametrize("entity", US_SCHEMA.entities)
@pytest.mark.parametrize("source_state", ["missing", "nan", "none", "nullable"])
def test_assembled_provenance_requires_complete_source_ids(
    entity: str,
    source_state: str,
) -> None:
    table = _support_copies([0, 1, 2], assembled=True).rename(
        columns=lambda column: column.replace("person_", f"{entity}_", 1)
    )
    source = support_source_id_column(entity)
    if source_state == "missing":
        table = table.drop(columns=source)
    else:
        table[source] = {
            "nan": [np.nan, np.nan, 7],
            "none": [None, None, 7],
            "nullable": pd.array([pd.NA, pd.NA, 7], dtype="Int64"),
        }[source_state]
    for reader in (
        require_assembled_support_provenance,
        support_role_series,
        support_copy_rank_series,
        without_support_role_metadata,
    ):
        with pytest.raises(
            ValueError, match=f"assembled support metadata requires.*{source}"
        ):
            reader(table, entity=entity)


def test_assembled_copy_rank_preserves_int64_maximum() -> None:
    indices = np.asarray([0, 1, np.iinfo(np.int64).max], dtype=np.int64)
    table = _support_copies(indices, assembled=True)
    assert support_copy_rank_series(table, entity="person").tolist() == indices.tolist()
