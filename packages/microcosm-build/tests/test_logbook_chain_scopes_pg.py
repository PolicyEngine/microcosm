"""Exercise scoped Logbook chains against a real Postgres.

This integration test needs optional local packages that are not project
dependencies: ``pgserver`` for the embedded server and ``psycopg`` for typed
Postgres errors. CI and the default venv skip it. To run locally, either pip
install both into the project venv and run the file through pytest as usual, or
keep them in a separate harness venv and run this file standalone — it is
self-contained, but the shared conftest is not, so skip it:

```
<harness-venv>/bin/python -m pytest -q --noconftest -p no:cacheprovider \
    packages/microcosm-build/tests/test_logbook_chain_scopes_pg.py
```

The harness shims ``extensions.digest`` over the server's built-in ``sha256``
because the bundled Postgres lacks pgcrypto. Production runs the real pgcrypto
extension from the base Logbook migration.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

pgserver = pytest.importorskip("pgserver")
psycopg = pytest.importorskip("psycopg")

ROOT = Path(__file__).resolve().parents[3]
MIGRATIONS = ROOT / "supabase/migrations"
CHAIN_SCOPES_MIGRATION = MIGRATIONS / "20260818000000_logbook_chain_scopes.sql"
UK_LOCAL_SCOPE_MIGRATION = (
    MIGRATIONS / "20260829000000_logbook_uk_local_scope.sql"
)
ROWS = ROOT / "logbook/us.jsonl"
BASE_MIGRATIONS = [
    "20260805000000_logbook.sql",
    "20260805000001_logbook_predictions.sql",
    "20260813000000_logbook_f004_rung.sql",
    "20260815000000_logbook_f025_rung.sql",
]
PGCRYPTO_LINE = "CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA extensions;"
PGCRYPTO_RELOCATE = "        ALTER EXTENSION pgcrypto SET SCHEMA extensions;"
SHIM = """
CREATE SCHEMA IF NOT EXISTS extensions;
CREATE OR REPLACE FUNCTION extensions.digest(data bytea, algo text)
RETURNS bytea LANGUAGE sql IMMUTABLE AS $shim$
    SELECT CASE WHEN algo = 'sha256' THEN sha256(data) END;
$shim$;
"""
COLUMNS = [
    "build_id",
    "ts",
    "pipeline",
    "rung",
    "seed",
    "code_pin",
    "input_pins_digest",
    "identity_digest",
    "phases_reached",
    "gate_verdicts",
    "wall_seconds",
    "cost_usd",
    "artifact_location",
    "disposition",
    "prediction_id",
    "prev_row_digest",
    "row_digest",
]


def _connect(uri: str):
    return psycopg.connect(uri, autocommit=True)


def _apply_sql(connection, sql: str) -> None:
    with connection.cursor() as cursor:
        cursor.execute(sql)


def _apply_migrations(connection) -> None:
    _apply_sql(connection, SHIM)
    for name in BASE_MIGRATIONS:
        sql = (MIGRATIONS / name).read_text(encoding="utf-8")
        sql = sql.replace(PGCRYPTO_LINE, "-- (harness: pgcrypto shimmed above)")
        sql = sql.replace(PGCRYPTO_RELOCATE, "        NULL;  -- (harness shim)")
        _apply_sql(connection, sql)


def _archived_rows() -> list[dict]:
    return [json.loads(line) for line in ROWS.read_text().splitlines() if line]


def _build_row(build_id: str, *, pipeline: str, predecessor: str | None) -> dict:
    return {
        "build_id": build_id,
        "ts": "2026-08-18T12:00:00Z",
        "pipeline": pipeline,
        "rung": "f100",
        "seed": 42,
        "code_pin": "abc1234",
        "input_pins_digest": "1" * 64,
        "identity_digest": "2" * 64,
        "phases_reached": ["attempt_started"],
        "gate_verdicts": {
            "terminal": {
                "verdict": "failed",
                "receipt": "receipt://fixture.json",
            }
        },
        "wall_seconds": 1.0,
        "cost_usd": None,
        "artifact_location": None,
        "disposition": "failed",
        "prediction_id": None,
        "prev_row_digest": predecessor,
        "row_digest": None,
    }


def _insert(connection, row: dict) -> None:
    values = []
    for column in COLUMNS:
        value = row.get(column)
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        elif isinstance(value, float):
            # Numeric columns: send the exact decimal, never a float8 that
            # would acquire digits on the cast and change the row digest.
            value = repr(value)
        values.append(value)
    placeholders = ", ".join(["%s"] * len(COLUMNS))
    with connection.cursor() as cursor:
        cursor.execute(
            f"INSERT INTO logbook.builds ({', '.join(COLUMNS)}) "
            f"VALUES ({placeholders});",
            values,
        )


def _digest_of(connection, build_id: str) -> str:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT row_digest FROM logbook.builds WHERE build_id = %s;",
            (build_id,),
        )
        found = cursor.fetchone()
    assert found is not None, build_id
    return found[0]


def _count_builds(connection) -> int:
    with connection.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM logbook.builds;")
        found = cursor.fetchone()
    assert found is not None
    return found[0]


def _refuses(connection, row: dict, needle: str) -> None:
    with pytest.raises(psycopg.errors.Error, match=needle):
        _insert(connection, row)


def test_logbook_chain_scopes_migration_preserves_and_scopes_live_rows() -> None:
    server = pgserver.get_server(tempfile.mkdtemp())
    connection = _connect(server.get_uri())
    _apply_migrations(connection)
    archived = _archived_rows()
    archived_digests = {row["build_id"]: row["row_digest"] for row in archived}

    for row in archived:
        _insert(connection, row)
    assert _count_builds(connection) == 28

    _apply_sql(connection, CHAIN_SCOPES_MIGRATION.read_text(encoding="utf-8"))
    _apply_sql(connection, UK_LOCAL_SCOPE_MIGRATION.read_text(encoding="utf-8"))

    us_tail = _digest_of(connection, archived[-1]["build_id"])
    _insert(
        connection,
        _build_row("us-next", pipeline="us-stacked-pool", predecessor=us_tail),
    )
    us_next_tail = _digest_of(connection, "us-next")

    _refuses(
        connection,
        _build_row(
            "uk-frs-from-us",
            pipeline="uk-frs-staging",
            predecessor=us_next_tail,
        ),
        "scope uk/frs must have null prev_row_digest",
    )
    _insert(
        connection,
        _build_row(
            "uk-frs-genesis",
            pipeline="uk-frs-staging",
            predecessor=None,
        ),
    )
    # Vocabulary is closed-world and minimal: uk/local is ratified for #761,
    # while uk/firms still derives cleanly but remains unratified. Cross-scope
    # independence is already proven above: uk/frs opened while us had rows.
    _insert(
        connection,
        _build_row(
            "uk-local-genesis", pipeline="uk-local-rowwise", predecessor=None
        ),
    )
    _refuses(
        connection,
        _build_row("uk-firms-genesis", pipeline="uk-firms-staging", predecessor=None),
        "not in the ratified scope list",
    )

    uk_frs_genesis = _digest_of(connection, "uk-frs-genesis")
    _insert(
        connection,
        _build_row(
            "uk-frs-second",
            pipeline="uk-frs-staging",
            predecessor=uk_frs_genesis,
        ),
    )
    _refuses(
        connection,
        _build_row(
            "uk-frs-stale",
            pipeline="uk-frs-staging",
            predecessor=uk_frs_genesis,
        ),
        "current uk/frs tail is",
    )

    # A new-named US pipeline derives us/pool, an unratified scope: refused
    # both with a predecessor and at genesis — opening a scope is a reviewed
    # migration, never a side effect of a well-formed name.
    _refuses(
        connection,
        _build_row("us-pool-inc3-linked", pipeline="us-pool-inc3", predecessor=us_tail),
        "not in the ratified scope list",
    )
    _refuses(
        connection,
        _build_row("us-pool-inc3-genesis", pipeline="us-pool-inc3", predecessor=None),
        "not in the ratified scope list",
    )
    _insert(
        connection,
        _build_row(
            "us-after-new-scope",
            pipeline="us-stacked-pool",
            predecessor=us_next_tail,
        ),
    )

    _refuses(
        connection,
        _build_row("mystery", pipeline="mystery-pipeline", predecessor=None),
        "does not declare a chain scope",
    )

    changed = [
        build_id
        for build_id, digest in archived_digests.items()
        if _digest_of(connection, build_id) != digest
    ]
    assert changed == []
