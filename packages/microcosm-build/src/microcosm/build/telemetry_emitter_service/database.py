"""SQLAlchemy engine and session construction for the telemetry spool."""

from __future__ import annotations

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
) -> Engine:
    """Create the SQLAlchemy engine used by one emitter service process.

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
