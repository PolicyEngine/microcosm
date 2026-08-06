import hashlib
import json
import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[3]
MIGRATION = (
    REPOSITORY_ROOT
    / "supabase"
    / "migrations"
    / "20260805000001_chronicle_predictions.sql"
)
SOURCE_RECORDS_DIGEST = (
    "1ff7529a7224a3c15a5bf14228edcc8a02f68ac26e6749720ba7b17952b5f9e3"
)


def _migration_records(sql: str) -> list[dict[str, object]]:
    match = re.search(
        r"\$chronicle_predictions\$\s*(\[.*\])\s*"
        r"\$chronicle_predictions\$::jsonb",
        sql,
        flags=re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def test_prediction_seed_preserves_all_harvested_records() -> None:
    sql = MIGRATION.read_text()
    seeded = _migration_records(sql)
    canonical_seed = json.dumps(
        seeded,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()

    assert hashlib.sha256(canonical_seed).hexdigest() == SOURCE_RECORDS_DIGEST
    assert len(seeded) == 17
    assert [record["id"] for record in seeded] == [
        f"p{number:03d}" for number in range(1, 18)
    ]
    assert len({record["id"] for record in seeded}) == 17


def test_prediction_seed_targets_chronicle_idempotently() -> None:
    sql = MIGRATION.read_text()

    assert "INSERT INTO chronicle.predictions" in sql
    assert "actual jsonb" in sql
    assert "ON CONFLICT (id) DO NOTHING" in sql
