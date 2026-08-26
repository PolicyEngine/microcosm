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
FAMILY_MIGRATION = MIGRATIONS / "20260830000000_logbook_family_model.sql"
ROWS = ROOT / "logbook/us.jsonl"
ROW_VERSION_FIXTURES = (
    ROOT / "packages/microcosm-build/tests/fixtures/logbook_row_versions.json"
)
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
VERSION_2_COLUMNS = [
    *COLUMNS,
    "row_format_version",
    "requested_k",
    "realized_k",
    "record_unit",
]


def _connect(uri: str):
    return psycopg.connect(uri, autocommit=True)


def _postgres_server():
    runtime = Path(tempfile.mkdtemp(prefix="microcosm-pgserver-runtime-"))
    server_class = pgserver.PostgresServer
    server_class.runtime_path = runtime
    server_class.lock_path = runtime / ".lockfile"
    server_class._lock = server_class.fasteners.InterProcessLock(server_class.lock_path)
    return pgserver.get_server(Path(tempfile.mkdtemp(prefix="microcosm-pgdata-")))


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


def _insert(connection, row: dict, *, versioned: bool = False) -> None:
    columns = VERSION_2_COLUMNS if versioned else COLUMNS
    values = []
    for column in columns:
        value = row.get(column)
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        elif isinstance(value, float):
            # Numeric columns: send the exact decimal, never a float8 that
            # would acquire digits on the cast and change the row digest.
            value = repr(value)
        values.append(value)
    placeholders = ", ".join(["%s"] * len(columns))
    with connection.cursor() as cursor:
        cursor.execute(
            f"INSERT INTO logbook.builds ({', '.join(columns)}) "
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


def _execute(connection, statement: str, values: tuple = ()) -> None:
    with connection.cursor() as cursor:
        cursor.execute(statement, values)


def _fetchone(connection, statement: str, values: tuple = ()) -> tuple:
    with connection.cursor() as cursor:
        cursor.execute(statement, values)
        found = cursor.fetchone()
    assert found is not None
    return found


def _fetchall(connection, statement: str, values: tuple = ()) -> list[tuple]:
    with connection.cursor() as cursor:
        cursor.execute(statement, values)
        return cursor.fetchall()


def test_logbook_chain_scopes_migration_preserves_and_scopes_live_rows() -> None:
    server = _postgres_server()
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


def test_family_model_migration_enforces_versioning_relationships_and_access() -> None:
    server = _postgres_server()
    connection = _connect(server.get_uri())
    _apply_migrations(connection)
    archived = _archived_rows()
    for row in archived:
        _insert(connection, row)
    _apply_sql(connection, CHAIN_SCOPES_MIGRATION.read_text(encoding="utf-8"))
    _apply_sql(connection, UK_LOCAL_SCOPE_MIGRATION.read_text(encoding="utf-8"))

    # Plain PostgreSQL does not provide Supabase's API roles. Create them
    # before applying this migration so its conditional public-view grants
    # can be exercised in the harness.
    _execute(connection, "CREATE ROLE anon NOLOGIN;")
    _apply_sql(connection, FAMILY_MIGRATION.read_text(encoding="utf-8"))

    assert [
        row[0]
        for row in _fetchall(
            connection,
            "SELECT row_format_version FROM logbook.builds ORDER BY ts, build_id;",
        )
    ] == [None] * len(archived)
    assert {
        row[0]: row[1]
        for row in _fetchall(
            connection,
            "SELECT build_id, row_digest FROM logbook.builds;",
        )
    } == {row["build_id"]: row["row_digest"] for row in archived}

    fixtures = json.loads(ROW_VERSION_FIXTURES.read_text(encoding="utf-8"))
    for fixture_name in ("legacy", "version_2", "version_2_exact_k"):
        fixture = fixtures[fixture_name]
        payload = json.dumps(fixture["row"])
        observed = _fetchone(
            connection,
            "SELECT logbook.expected_build_row_digest("
            "json_populate_record(NULL::logbook.builds, %s::json));",
            (payload,),
        )[0]
        assert observed == fixture["expected_row_digest"]

    us_tail = archived[-1]["row_digest"]
    us_first = _build_row(
        "family-us-20000-a",
        pipeline="us-stacked-pool",
        predecessor=us_tail,
    )
    us_first.update(
        row_format_version=2,
        requested_k=20_000,
        realized_k=20_000,
        record_unit="household",
    )
    _insert(connection, us_first, versioned=True)
    us_second = _build_row(
        "family-us-20000-b",
        pipeline="us-stacked-pool",
        predecessor=_digest_of(connection, us_first["build_id"]),
    )
    us_second.update(
        row_format_version=2,
        requested_k=20_000,
        realized_k=20_000,
        record_unit="household",
    )
    _insert(connection, us_second, versioned=True)
    us_mismatch = _build_row(
        "family-us-57240",
        pipeline="us-stacked-pool",
        predecessor=_digest_of(connection, us_second["build_id"]),
    )
    us_mismatch.update(
        row_format_version=2,
        requested_k=57_240,
        realized_k=57_240,
        record_unit="household",
    )
    _insert(connection, us_mismatch, versioned=True)

    legacy_null_rung = _build_row(
        "legacy-null-rung",
        pipeline="us-stacked-pool",
        predecessor=_digest_of(connection, us_mismatch["build_id"]),
    )
    legacy_null_rung["rung"] = None
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert(connection, legacy_null_rung)

    exact_k_null_rung = _build_row(
        "exact-k-null-rung",
        pipeline="us-stacked-pool",
        predecessor=_digest_of(connection, us_mismatch["build_id"]),
    )
    exact_k_null_rung.update(
        rung=None,
        row_format_version=2,
        requested_k=20_000,
        realized_k=20_000,
        record_unit="household",
    )
    _insert(connection, exact_k_null_rung, versioned=True)
    assert _fetchone(
        connection,
        "SELECT rung FROM logbook.builds WHERE build_id = %s;",
        (exact_k_null_rung["build_id"],),
    ) == (None,)

    different_unit = _build_row(
        "family-us-person-20000",
        pipeline="us-stacked-pool",
        predecessor=_digest_of(connection, exact_k_null_rung["build_id"]),
    )
    different_unit.update(
        row_format_version=2,
        requested_k=20_000,
        realized_k=20_000,
        record_unit="person",
    )
    _insert(connection, different_unit, versioned=True)

    uk_build = _build_row(
        "family-uk",
        pipeline="uk-frs-staging",
        predecessor=None,
    )
    uk_build.update(
        row_format_version=2,
        requested_k=10,
        realized_k=10,
        record_unit="household",
    )
    _insert(connection, uk_build, versioned=True)

    invalid_cardinality = _build_row(
        "invalid-cardinality",
        pipeline="us-stacked-pool",
        predecessor=_digest_of(connection, different_unit["build_id"]),
    )
    invalid_cardinality.update(
        row_format_version=2,
        requested_k=0,
        realized_k=None,
        record_unit="household",
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert(connection, invalid_cardinality, versioned=True)

    invalid_publication = dict(invalid_cardinality)
    invalid_publication.update(
        build_id="invalid-publication",
        requested_k=20_000,
        realized_k=19_999,
        disposition="published",
        artifact_location="hf://datasets/fixture/release",
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert(connection, invalid_publication, versioned=True)

    family_id = "12345678-1234-4234-9234-123456789abc"
    other_family_id = "22345678-1234-4234-9234-123456789abc"
    second_us_family_id = "62345678-1234-4234-9234-123456789abc"
    source_sha = "a" * 64
    family_upsert = (
        "INSERT INTO logbook.families "
        "(family_id, chain_scope, source_pool_sha256) VALUES (%s, %s, %s) "
        "ON CONFLICT (family_id) DO NOTHING;"
    )
    _execute(connection, family_upsert, (family_id, "us", source_sha))
    _execute(connection, family_upsert, (family_id, "us", source_sha))
    with pytest.raises(psycopg.errors.UniqueViolation, match="divergent content"):
        _execute(connection, family_upsert, (family_id, "uk/frs", "b" * 64))
    with pytest.raises(psycopg.errors.UniqueViolation, match="already belongs"):
        _execute(connection, family_upsert, (other_family_id, "us", source_sha))

    member_insert = (
        "INSERT INTO logbook.family_members (family_id, build_id) "
        "VALUES (%s, %s) ON CONFLICT (family_id, build_id) DO NOTHING;"
    )
    with pytest.raises(psycopg.errors.ForeignKeyViolation, match="does not exist"):
        _execute(
            connection,
            member_insert,
            ("32345678-1234-4234-9234-123456789abc", us_first["build_id"]),
        )
    with pytest.raises(psycopg.errors.ForeignKeyViolation, match="does not exist"):
        _execute(connection, member_insert, (family_id, "missing-build"))
    for build_id in (
        us_first["build_id"],
        us_second["build_id"],
        us_mismatch["build_id"],
        exact_k_null_rung["build_id"],
        different_unit["build_id"],
    ):
        _execute(connection, member_insert, (family_id, build_id))
    _execute(connection, member_insert, (family_id, us_first["build_id"]))
    with pytest.raises(psycopg.errors.CheckViolation, match="does not match"):
        _execute(connection, member_insert, (family_id, uk_build["build_id"]))
    _execute(connection, family_upsert, (other_family_id, "uk/frs", "b" * 64))
    _execute(
        connection,
        family_upsert,
        (second_us_family_id, "us", "c" * 64),
    )
    with pytest.raises(psycopg.errors.UniqueViolation):
        _execute(
            connection,
            member_insert,
            (second_us_family_id, us_first["build_id"]),
        )

    revocation_id = "32345678-1234-4234-9234-123456789abc"
    replacement_id = "42345678-1234-4234-9234-123456789abc"
    action_insert = (
        "INSERT INTO logbook.family_actions "
        "(action_id, family_id, build_id, action_type, related_build_id, "
        "recorded_at, actor, reason, evidence_location) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (action_id) DO NOTHING;"
    )
    revocation = (
        revocation_id,
        family_id,
        us_mismatch["build_id"],
        "revokes",
        None,
        "2026-08-21T12:00:00Z",
        "fixture",
        "Invalid output",
        None,
    )
    _execute(connection, action_insert, revocation)
    _execute(connection, action_insert, revocation)
    replacement = (
        replacement_id,
        family_id,
        us_second["build_id"],
        "supersedes",
        us_first["build_id"],
        "2026-08-21T12:01:00Z",
        "fixture",
        "Corrected output",
        None,
    )
    _execute(connection, action_insert, replacement)
    replacement_of_replacement = (
        "92345678-1234-4234-9234-123456789abc",
        family_id,
        exact_k_null_rung["build_id"],
        "supersedes",
        us_second["build_id"],
        "2026-08-21T12:01:15Z",
        "fixture",
        "Second corrected output",
        None,
    )
    _execute(connection, action_insert, replacement_of_replacement)
    with pytest.raises(psycopg.errors.UniqueViolation, match="divergent content"):
        _execute(
            connection,
            action_insert,
            (*replacement[:-2], "Different reason", replacement[-1]),
        )
    with pytest.raises(psycopg.errors.CheckViolation):
        _execute(
            connection,
            action_insert,
            (
                "72345678-1234-4234-9234-123456789abc",
                family_id,
                us_first["build_id"],
                "supersedes",
                us_first["build_id"],
                "2026-08-21T12:01:30Z",
                "fixture",
                "Self replacement",
                None,
            ),
        )
    with pytest.raises(psycopg.errors.UniqueViolation):
        _execute(
            connection,
            action_insert,
            (
                "82345678-1234-4234-9234-123456789abc",
                family_id,
                us_second["build_id"],
                "supersedes",
                us_first["build_id"],
                "2026-08-21T12:01:45Z",
                "fixture",
                "Conflicting direct replacement",
                None,
            ),
        )
    with pytest.raises(psycopg.errors.CheckViolation, match="matching requested_k"):
        _execute(
            connection,
            action_insert,
            (
                "52345678-1234-4234-9234-123456789abc",
                family_id,
                us_mismatch["build_id"],
                "supersedes",
                us_second["build_id"],
                "2026-08-21T12:02:00Z",
                "fixture",
                "Wrong size",
                None,
            ),
        )
    with pytest.raises(
        psycopg.errors.CheckViolation,
        match="matching requested_k and record_unit",
    ):
        _execute(
            connection,
            action_insert,
            (
                "a2345678-1234-4234-9234-123456789abc",
                family_id,
                different_unit["build_id"],
                "supersedes",
                exact_k_null_rung["build_id"],
                "2026-08-21T12:02:15Z",
                "fixture",
                "Wrong record unit",
                None,
            ),
        )

    public_build = _fetchone(
        connection,
        "SELECT requested_k, realized_k, record_unit "
        "FROM logbook.family_members_public WHERE build_id = %s;",
        (us_first["build_id"],),
    )
    assert public_build == (20_000, 20_000, "household")
    public_cardinalities = _fetchone(
        connection,
        "SELECT array_agg(requested_k ORDER BY requested_k, build_id) "
        "FROM logbook.family_members_public WHERE family_id = %s;",
        (family_id,),
    )[0]
    assert public_cardinalities == [20_000, 20_000, 20_000, 20_000, 57_240]
    status = _fetchone(
        connection,
        "SELECT revoked, superseded_by_build_id "
        "FROM logbook.family_member_status_public WHERE build_id = %s;",
        (us_first["build_id"],),
    )
    assert status == (False, us_second["build_id"])
    replacement_status = _fetchone(
        connection,
        "SELECT revoked, superseded_by_build_id "
        "FROM logbook.family_member_status_public WHERE build_id = %s;",
        (us_second["build_id"],),
    )
    assert replacement_status == (False, exact_k_null_rung["build_id"])
    revocation_status = _fetchone(
        connection,
        "SELECT revoked, superseded_by_build_id "
        "FROM logbook.family_member_status_public WHERE build_id = %s;",
        (us_mismatch["build_id"],),
    )
    assert revocation_status == (True, None)

    _execute(connection, "SET ROLE anon;")
    assert (
        _fetchone(connection, "SELECT count(*) FROM logbook.families_public;")[0] == 3
    )
    with pytest.raises(psycopg.errors.UndefinedColumn):
        _execute(connection, "SELECT cost_usd FROM logbook.family_members_public;")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        _execute(connection, "SELECT * FROM logbook.family_actions;")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        _execute(connection, "SELECT * FROM logbook.predictions;")
    _execute(connection, "RESET ROLE;")

    _execute(connection, "SET ROLE logbook_writer;")
    _execute(
        connection,
        family_upsert,
        ("92345678-1234-4234-9234-123456789abc", "us", "d" * 64),
    )
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        _execute(
            connection,
            "UPDATE logbook.families SET chain_scope = 'uk/frs' WHERE family_id = %s;",
            (family_id,),
        )
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        _execute(
            connection,
            "DELETE FROM logbook.family_members WHERE build_id = %s;",
            (us_first["build_id"],),
        )
    _execute(connection, "RESET ROLE;")
