"""Frozen-support / warm-start selection recovery (microcosm#328).

These tests exercise the identity join that recovers a frozen support onto a
freshly-rebuilt base, and the refusal contract that forbids fabricated or
truncated selections. All fixtures are tiny synthetic frames — no network, no
large H5.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.warm_start_selection import (
    DEFAULT_SELECTION_JOIN_KEY,
    SelectionSource,
    load_selection_source_from_h5,
    load_selection_source_from_manifest,
    select_frozen_support,
    write_selection_source_manifest,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

# A source ASEC household appears twice in the pool: clone 0 (asec channel) and
# clone 1 (puf_tax_detail channel). The identity that survives a base rebuild is
# (source_year, source_household_id, channel, clone) — never the assigned row id.


def _record(
    *,
    person_id: int,
    household_id: int,
    source_year: int,
    source_household_id: int,
    source_person_id: str,
    channel: str,
    clone: int,
) -> dict[str, object]:
    return {
        "person_id": person_id,
        "person_household_id": household_id,
        "person_tax_unit_id": household_id,
        "person_spm_unit_id": household_id,
        "person_family_id": household_id,
        "person_marital_unit_id": person_id,
        "source_year": source_year,
        "source_household_id": source_household_id,
        "source_person_id": source_person_id,
        "person_support_channel": channel,
        "person_support_clone_index": clone,
        "household_id": household_id,
        "household_support_channel": channel,
        "household_support_clone_index": clone,
    }


def _frame_from_records(records: list[dict[str, object]]) -> Frame:
    """Build a minimal US-schema frame, one person per household, from records.

    One person per household keeps the household identity unambiguous while still
    exercising the person->household expansion in the selector.
    """
    df = pd.DataFrame(records)
    person = pd.DataFrame(
        {
            "person_id": df["person_id"].to_numpy(dtype="int64"),
            "person_household_id": df["person_household_id"].to_numpy(dtype="int64"),
            "person_tax_unit_id": df["person_tax_unit_id"].to_numpy(dtype="int64"),
            "person_spm_unit_id": df["person_spm_unit_id"].to_numpy(dtype="int64"),
            "person_family_id": df["person_family_id"].to_numpy(dtype="int64"),
            "person_marital_unit_id": df["person_marital_unit_id"].to_numpy(
                dtype="int64"
            ),
            "source_year": df["source_year"].to_numpy(dtype="int64"),
            "source_household_id": df["source_household_id"].to_numpy(dtype="int64"),
            "source_person_id": df["source_person_id"].astype(str).to_numpy(),
            "person_support_channel": df["person_support_channel"]
            .astype(str)
            .to_numpy(),
            "person_support_clone_index": df["person_support_clone_index"].to_numpy(
                dtype="int64"
            ),
        }
    )
    household = pd.DataFrame(
        {
            "household_id": df["household_id"].to_numpy(dtype="int64"),
            "household_support_channel": df["household_support_channel"]
            .astype(str)
            .to_numpy(),
            "household_support_clone_index": df[
                "household_support_clone_index"
            ].to_numpy(dtype="int64"),
        }
    )
    n = len(df)
    return Frame(
        {
            "person": person,
            "household": household,
            "tax_unit": pd.DataFrame(
                {"tax_unit_id": df["person_tax_unit_id"].to_numpy(dtype="int64")}
            ),
            "spm_unit": pd.DataFrame(
                {"spm_unit_id": df["person_spm_unit_id"].to_numpy(dtype="int64")}
            ),
            "family": pd.DataFrame(
                {"family_id": df["person_family_id"].to_numpy(dtype="int64")}
            ),
            "marital_unit": pd.DataFrame(
                {
                    "marital_unit_id": df["person_marital_unit_id"].to_numpy(
                        dtype="int64"
                    )
                }
            ),
        },
        US_SCHEMA,
        {
            "household": Weights(
                np.full(n, 100.0, dtype="float64"), WeightKind.CALIBRATED
            )
        },
    )


def _pool_records() -> list[dict[str, object]]:
    """A 4-source-household × 2-clone base = 8 households across two years."""
    records: list[dict[str, object]] = []
    hid = 1
    pid = 1
    for source_year, source_hh in [(2024, 11), (2024, 22), (2023, 11), (2023, 44)]:
        for clone, channel in [(0, "asec"), (1, "puf_tax_detail")]:
            records.append(
                _record(
                    person_id=pid,
                    household_id=hid,
                    source_year=source_year,
                    source_household_id=source_hh,
                    source_person_id=f"{source_year}{source_hh:04d}0101",
                    channel=channel,
                    clone=clone,
                )
            )
            hid += 1
            pid += 1
    return records


def test__given_named_support__then_reduces_base_to_exactly_that_support() -> None:
    # Given a base pool and a source naming 3 of its 8 households (clone-aware)
    base = _frame_from_records(_pool_records())
    picks = [
        (2024, 11, "asec", 0),
        (2024, 22, "puf_tax_detail", 1),
        (2023, 44, "asec", 0),
    ]
    source = SelectionSource(
        join_key=DEFAULT_SELECTION_JOIN_KEY,
        identities=[list(p) for p in picks],
        provenance={"kind": "test"},
    )

    # When
    reduced, report = select_frozen_support(base, source)

    # Then exactly those households remain, and the report is clean
    assert reduced.n("household") == 3
    assert report.n_selected == 3
    assert report.n_unmapped == 0
    assert report.n_ambiguous == 0
    got = set(
        zip(
            reduced.table("household")["household_support_channel"].tolist(),
            reduced.table("household")["household_support_clone_index"].tolist(),
            strict=True,
        )
    )
    assert ("asec", 0) in got and ("puf_tax_detail", 1) in got


def _reassigned_ids(
    records: list[dict[str, object]], order: list[int]
) -> list[dict[str, object]]:
    """Re-emit the same source identities in a different arrangement.

    ``order`` permutes which source record lands at each ascending household id,
    so the assigned ``household_id``/``person_id`` differ from the natural build
    while the stable source identity of each record is unchanged. The Frame
    invariant (household ids sorted ascending) is preserved.
    """
    out: list[dict[str, object]] = []
    for row_position, source_index in enumerate(order, start=1):
        rec = dict(records[source_index])
        rec["person_id"] = row_position
        rec["household_id"] = row_position
        rec["person_household_id"] = row_position
        rec["person_tax_unit_id"] = row_position
        rec["person_spm_unit_id"] = row_position
        rec["person_family_id"] = row_position
        rec["person_marital_unit_id"] = row_position
        out.append(rec)
    return out


def test__given_reassigned_base_ids__then_recovers_the_same_support() -> None:
    # Given the same source records but assigned to DIFFERENT household ids
    records = _pool_records()
    base = _frame_from_records(records)
    reassigned = _frame_from_records(_reassigned_ids(records, [7, 5, 3, 1, 6, 4, 2, 0]))
    picks = [(2024, 11, "asec", 0), (2023, 11, "puf_tax_detail", 1)]
    source = SelectionSource(
        join_key=DEFAULT_SELECTION_JOIN_KEY,
        identities=[list(p) for p in picks],
        provenance={"kind": "test"},
    )

    # When
    from_ordered, _ = select_frozen_support(base, source)
    from_shuffled, _ = select_frozen_support(reassigned, source)

    # Then the identity join recovers the same support regardless of row order
    def support_ids(frame: Frame) -> set[tuple[int, int, str, int]]:
        p = frame.table("person")
        return set(
            zip(
                p["source_year"].tolist(),
                p["source_household_id"].tolist(),
                p["person_support_channel"].tolist(),
                p["person_support_clone_index"].tolist(),
                strict=True,
            )
        )

    assert support_ids(from_ordered) == support_ids(from_shuffled)
    assert from_ordered.n("household") == from_shuffled.n("household") == 2


def test__given_unmappable_identity__then_frozen_mode_refuses() -> None:
    # Given a source identity that does not exist in the base
    base = _frame_from_records(_pool_records())
    source = SelectionSource(
        join_key=DEFAULT_SELECTION_JOIN_KEY,
        identities=[[2024, 11, "asec", 0], [1999, 99, "asec", 0]],
        provenance={"kind": "test"},
    )

    # When / Then: it refuses loudly and returns nothing (no truncated selection)
    with pytest.raises(ValueError, match="unmapped"):
        select_frozen_support(base, source)


def test__given_key_missing_channel__then_refuses_non_unique_key() -> None:
    # Given a join key without the clone/channel component, a source household's
    # two clones collide, so the key is not unique in the base.
    base = _frame_from_records(_pool_records())
    source = SelectionSource(
        join_key=("source_year", "source_household_id"),
        identities=[[2024, 11], [2023, 44]],
        provenance={"kind": "test"},
    )

    # When / Then
    with pytest.raises(ValueError, match="unique"):
        select_frozen_support(base, source)


def test__given_ambiguous_source_row__then_refuses() -> None:
    # Given a base where two rows share the full identity key (a malformed base),
    # a matching source identity would be ambiguous.
    records = _pool_records()
    dup = dict(records[0])
    dup["person_id"] = 999
    dup["household_id"] = 999
    dup["person_household_id"] = 999
    dup["person_tax_unit_id"] = 999
    dup["person_spm_unit_id"] = 999
    dup["person_family_id"] = 999
    dup["person_marital_unit_id"] = 999
    base = _frame_from_records([*records, dup])
    source = SelectionSource(
        join_key=DEFAULT_SELECTION_JOIN_KEY,
        identities=[
            list(records[0][k] for k in ("source_year", "source_household_id"))
            + [
                records[0]["household_support_channel"],
                records[0]["household_support_clone_index"],
            ]
        ],
        provenance={"kind": "test"},
    )

    with pytest.raises(ValueError, match="unique|ambiguous"):
        select_frozen_support(base, source)


def test__given_source_h5_frame__then_loads_identities_from_it() -> None:
    # Given a "source" frame standing in for a published H5
    source_frame = _frame_from_records(
        [r for r in _pool_records() if r["source_year"] == 2024]
    )

    # When
    source = load_selection_source_from_h5(
        source_frame, join_key=DEFAULT_SELECTION_JOIN_KEY
    )

    # Then it captures one identity per source household
    assert source.n_identities == 4
    assert list(source.join_key) == list(DEFAULT_SELECTION_JOIN_KEY)


def test__given_manifest_roundtrip__then_identities_match_and_integrity_holds(
    tmp_path: Path,
) -> None:
    # Given a selection source distilled from a frame
    source_frame = _frame_from_records(_pool_records())
    source = load_selection_source_from_h5(
        source_frame,
        join_key=DEFAULT_SELECTION_JOIN_KEY,
        provenance={"kind": "h5", "path": "source.h5", "sha256": "deadbeef"},
    )
    path = tmp_path / "selection.json"
    write_selection_source_manifest(source, path)

    # When
    reloaded = load_selection_source_from_manifest(path)

    # Then the identities and key match
    assert reloaded.n_identities == source.n_identities
    assert list(reloaded.join_key) == list(source.join_key)
    reloaded_reduced, report = select_frozen_support(
        _frame_from_records(_pool_records()), reloaded
    )
    assert report.n_selected == source.n_identities

    # And a corrupted integrity hash is rejected
    payload = json.loads(path.read_text())
    payload["identities_sha256"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identities_sha256|integrity"):
        load_selection_source_from_manifest(path)


def test__given_informed_init_mode__then_drops_unmapped_without_raising() -> None:
    # Given an unmappable identity but informed_init mode
    base = _frame_from_records(_pool_records())
    source = SelectionSource(
        join_key=DEFAULT_SELECTION_JOIN_KEY,
        identities=[[2024, 11, "asec", 0], [1999, 99, "asec", 0]],
        provenance={"kind": "test"},
    )

    # When: informed_init tolerates unmapped records (they start cold)
    mask, report = source.base_selection_mask(base, mode="informed_init")

    # Then: one mapped, one dropped-and-recorded, no raise
    assert report.n_selected == 1
    assert report.n_unmapped == 1
    assert int(mask.sum()) == 1
