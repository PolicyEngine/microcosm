import hashlib
import json
import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[3]
SCHEMA_MIGRATION = (
    REPOSITORY_ROOT / "supabase" / "migrations" / "20260805000000_logbook.sql"
)
MIGRATION = (
    REPOSITORY_ROOT
    / "supabase"
    / "migrations"
    / "20260805000001_logbook_predictions.sql"
)
SOURCE_RECORDS_DIGEST = (
    "7ec0ec8aa0944a654d45bc4fa6e07203c38b5b17d0880422751c67de7c6b9d7d"
)
PREDICTION_FIELDS = {
    "id",
    "ts",
    "claim",
    "type",
    "predicted",
    "p",
    "resolved",
    "resolves",
    "outcome",
    "actual",
    "note",
}


def _migration_records(sql: str) -> list[dict[str, object]]:
    match = re.search(
        r"\$logbook_predictions\$\s*(\[.*\])\s*"
        r"\$logbook_predictions\$::jsonb",
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
    assert len(seeded) == 21
    assert [record["id"] for record in seeded] == [
        *(f"p{number:03d}" for number in range(1, 18)),
        "p015-resolution",
        "p018",
        "p019",
        "p016-resolution",
    ]
    assert len({record["id"] for record in seeded}) == 21
    assert set().union(*(record.keys() for record in seeded)) == PREDICTION_FIELDS

    resolution_events = [record for record in seeded if "resolves" in record]
    assert [record["resolves"] for record in resolution_events] == ["p015", "p016"]
    assert all("claim" not in record for record in resolution_events)
    assert all("type" not in record for record in resolution_events)


def test_prediction_seed_targets_logbook_idempotently() -> None:
    sql = MIGRATION.read_text()

    assert "INSERT INTO logbook.predictions" in sql
    assert "actual jsonb" in sql
    assert "resolves text" in sql
    assert "ON CONFLICT (id) DO NOTHING" in sql


def test_prediction_schema_accepts_claims_and_resolution_events() -> None:
    sql = SCHEMA_MIGRATION.read_text()
    predictions = sql.split("CREATE TABLE logbook.predictions (", 1)[1].split(
        "\n);", 1
    )[0]

    for field in PREDICTION_FIELDS:
        assert re.search(rf"^    {field}\s", predictions, flags=re.MULTILINE), field
    assert "CONSTRAINT predictions_event_shape CHECK" in predictions
    assert "CONSTRAINT predictions_resolves_fk FOREIGN KEY (resolves)" in predictions
    assert "REFERENCES logbook.predictions (id)" in predictions
