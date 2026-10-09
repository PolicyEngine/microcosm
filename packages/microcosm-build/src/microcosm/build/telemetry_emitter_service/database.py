"""SQLAlchemy engine and session construction for the telemetry spool."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from sqlalchemy import URL, create_engine, event
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import Session, sessionmaker

from microcosm.build.telemetry_emitter_service.constants import (
    DATABASE_TIMEOUT_SECONDS,
)


def sqlite_database_url(path: Path | str) -> URL:
    """Return a SQLAlchemy URL for a local SQLite spool path."""

    return URL.create("sqlite+pysqlite", database=str(Path(path)))


def create_spool_engine(
    path: Path | str,
    *,
    immediate_transactions: bool = False,
    busy_timeout_seconds: Callable[[], float] | None = None,
) -> Engine:
    """Create the SQLAlchemy engine used by one emitter service process.

    ``busy_timeout_seconds`` is read each time a connection is checked out and
    caps how long one statement waits in SQLite's busy handler for another
    process's lock; by default that is ``DATABASE_TIMEOUT_SECONDS``.

    By default Python's sqlite3 module opens a transaction only before a data
    change, never before DDL. With ``immediate_transactions`` every SQLAlchemy
    transaction is instead one ``BEGIN IMMEDIATE`` transaction: it takes the
    write lock at the start, waiting in SQLite's busy handler, and holds it
    until commit. This is SQLAlchemy's documented pysqlite recipe for
    transactional DDL.
    """

    engine = create_engine(
        sqlite_database_url(path),
        connect_args={"timeout": DATABASE_TIMEOUT_SECONDS},
    )
    if busy_timeout_seconds is not None:

        def apply_busy_timeout(dbapi_connection, connection_record, proxy) -> None:
            milliseconds = max(0, round(busy_timeout_seconds() * 1000))
            dbapi_connection.execute(f"PRAGMA busy_timeout = {milliseconds}")

        event.listen(engine, "checkout", apply_busy_timeout)
    if immediate_transactions:
        event.listen(engine, "connect", _disable_driver_transactions)
        event.listen(engine, "begin", _begin_immediate)
    return engine


def _disable_driver_transactions(dbapi_connection, connection_record) -> None:
    dbapi_connection.isolation_level = None


def _begin_immediate(connection: Connection) -> None:
    connection.exec_driver_sql("BEGIN IMMEDIATE")


def create_spool_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create short-lived ORM sessions bound to the spool engine."""

    return sessionmaker(engine, expire_on_commit=False)
