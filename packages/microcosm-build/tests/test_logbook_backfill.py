"""End-to-end guard for the checked-in #628 Logbook backfill."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from microcosm.build.logbook import (
    canonical_json_bytes,
    load_logbook_file,
    load_spool_rows,
)

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "logbook.jsonl"
SPOOL = ROOT / "logbook-spool"

# The archive is append-only: rows after the seed arrive one per real build
# attempt. The #628 backfill seed itself is immutable, so its fidelity pins
# apply to the archive's first 26 rows forever — never to the whole file.
SEED_ROW_COUNT = 26


def test_backfill_archive_and_operator_spool_match_end_to_end() -> None:
    rows = load_logbook_file(ARCHIVE)
    spool_rows = load_spool_rows(SPOOL)

    assert len(rows) >= SEED_ROW_COUNT
    assert rows == spool_rows
    seed = rows[:SEED_ROW_COUNT]
    assert Counter(row.disposition for row in seed) == {
        "published": 22,
        "certified": 1,
        "failed": 2,
        "superseded": 1,
    }
    assert {row.rung for row in seed} == {"f100"}
    assert [row.build_id for row in seed] == [
        row.build_id for row in sorted(seed, key=lambda row: (row.ts, row.build_id))
    ]
    # Post-seed rows extend the chain in order; every row names its parent.
    for parent, child in zip(rows, rows[1:], strict=False):
        assert child.prev_row_digest == parent.row_digest


def test_backfill_contains_all_hf_releases_and_three_run_era_rows() -> None:
    rows = load_logbook_file(ARCHIVE)
    release_rows = [row for row in rows if row.pipeline == "us-2024-release"]
    run_rows = [row for row in rows if row.pipeline == "us-pool-inc2"]

    assert len(release_rows) == 23
    assert len(run_rows) == 3
    assert all(row.artifact_location == row.build_id for row in release_rows)
    assert [row.build_id for row in run_rows] == [
        "populace-us-2024-pool-inc2-run5",
        "populace-us-2024-pool-inc2-run6",
        "populace-us-2024-pool-inc2-run7",
    ]
    certified = [row for row in release_rows if row.disposition == "certified"]
    assert [row.build_id for row in certified] == [
        "populace-us-2024-buildp-sparse-rmloss100-cae8640-20260728T011454Z"
    ]


def test_run_era_backfill_marks_inferences_and_prediction_links() -> None:
    by_build_id = {row.build_id: row for row in load_logbook_file(ARCHIVE)}
    run5, run6, run7 = (
        by_build_id[f"populace-us-2024-pool-inc2-run{n}"] for n in (5, 6, 7)
    )

    assert [run5.cost_usd, run6.cost_usd, run7.cost_usd] == [122, 65, 47.5]
    assert [run5.prediction_id, run6.prediction_id, run7.prediction_id] == [
        None,
        "p001",
        "p005",
    ]
    assert run6.gate_verdicts["terminal"]["related_prediction_ids"] == [
        "p001",
        "p002",
    ]
    assert run7.gate_verdicts["agreement"]["related_prediction_ids"] == [
        "p005",
        "p006",
        "p007",
        "p008",
    ]
    assert "inferred" in run5.gate_verdicts["terminal"]["timestamp_basis"]
    assert "inferred" in run6.gate_verdicts["terminal"]["wall_seconds_basis"]
    assert "midpoint" in run7.gate_verdicts["terminal"]["cost_basis"]

    for row in (run5, run6):
        terminal = row.gate_verdicts["terminal"]
        material = terminal["synthetic_identity_material"]
        expected = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
        assert row.identity_digest == expected

    assert run7.identity_digest == (
        "d5fdb55aaf8d37857b0ef8760526fa76cb0231c6c773efb12f15f022cdd8bdd6"
    )
