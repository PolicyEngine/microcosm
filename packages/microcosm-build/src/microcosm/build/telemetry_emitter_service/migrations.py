"""Programmatic Alembic migration entry points for the telemetry spool.

Every checkout and worktree on a host shares one spool, so checkouts at
different microcosm versions open the same file, and an older one can find it
migrated past its own head. Every revision is therefore additive: it may add
tables, nullable or defaulted columns and non-unique indexes, and leaves every
existing table, column, index, constraint, trigger and row as it was, so older
code reads and writes the migrated spool as before. A test checks every
packaged revision (``test_telemetry_spool_versions.py``). A change that older
code cannot use does not belong in this history; it needs a new spool file.

Whenever this module moves a spool to a new revision it also records the
spool's lineage: every revision in the history of the checkout that moved it.
A checkout that does not know a spool's revision uses the spool as it is when
that lineage includes its own head, and refuses it otherwise, as when a branch
with a different migration stamped it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Column, MetaData, Table, Text, delete, insert, inspect, select
from sqlalchemy.engine import Connection, Engine

from microcosm.build.telemetry_emitter_service.constants import (
    DATABASE_TIMEOUT_SECONDS,
    SPOOL_LINEAGE_TABLE,
    UNKNOWN_SPOOL_REVISION_ERROR,
)
from microcosm.build.telemetry_emitter_service.database import create_spool_engine

_MIGRATION_TARGET = "head"

_LINEAGE = Table(
    SPOOL_LINEAGE_TABLE,
    MetaData(),
    Column("revision", Text, primary_key=True),
)


class SpoolRevisionState(StrEnum):
    """Where a spool's revision lies relative to one checkout's history."""

    AT_HEAD = "at_head"
    BEHIND = "behind"
    AHEAD = "ahead"
    INCOMPATIBLE = "incompatible"


class IncompatibleSpoolRevisionError(RuntimeError):
    """The spool's revision neither precedes nor descends from this head."""


@dataclass(frozen=True)
class MigrationHistory:
    """One checkout's spool migrations: its head and every revision up to it."""

    head: str | None
    revisions: frozenset[str]


@contextmanager
def alembic_config(
    *,
    connection: Connection | None = None,
    script_location: Traversable | None = None,
) -> Iterator[Config]:
    """Yield Alembic configuration with migrations on the filesystem.

    ``script_location`` is a migration directory laid out like the packaged
    one, which it defaults to; tests pass other checkouts' histories.
    """

    if script_location is None:
        script_location = files(__package__).joinpath("alembic")
    with as_file(script_location) as migration_directory:
        config = Config()
        config.set_main_option("script_location", str(migration_directory))
        if connection is not None:
            config.attributes["connection"] = connection
        yield config


def classify_spool_revision(
    revision: str | None,
    lineage: frozenset[str] | None,
    history: MigrationHistory,
) -> SpoolRevisionState:
    """Return where a spool stamped ``revision`` lies relative to ``history``.

    ``lineage`` is the history the spool records, or ``None`` if it records
    none. A revision missing from this checkout's history is usable as it is
    only when that lineage holds both it and this checkout's head: a checkout
    descended from this one then migrated the spool, additively.
    """

    if revision == history.head:
        return SpoolRevisionState.AT_HEAD
    if revision is None or revision in history.revisions:
        return SpoolRevisionState.BEHIND
    if lineage is not None and {revision, history.head} <= lineage:
        return SpoolRevisionState.AHEAD
    return SpoolRevisionState.INCOMPATIBLE


def upgrade_spool_database(
    engine: Engine,
    *,
    busy_timeout_seconds: float = DATABASE_TIMEOUT_SECONDS,
    script_location: Traversable | None = None,
) -> SpoolRevisionState:
    """Bring a spool behind this checkout's head up to it, one process at a time.

    Returns where the spool's revision lay before the call. A spool at head, or
    past it on a line descended from it, is only read, so opening it takes no
    write lock. Neither does refusing any other revision this checkout does not
    know, which raises ``IncompatibleSpoolRevisionError``. Otherwise the whole
    upgrade, DDL, version stamp and lineage, runs in one ``BEGIN IMMEDIATE``
    transaction whose statements each wait at most ``busy_timeout_seconds`` for
    another process's lock. Without that transaction two services opening a new
    spool at once interleave their DDL, and one fails with "table already
    exists". The revision is read again once the lock is held, so a service
    that waited for it finds the spool where the last migrator left it, and
    changes nothing unless it is still behind.
    """

    history = migration_history(script_location)
    with engine.connect() as connection:
        state = _spool_revision_state(connection, history)
    if state is not SpoolRevisionState.BEHIND:
        return state
    migration_engine = create_spool_engine(
        engine.url.database,
        immediate_transactions=True,
        busy_timeout_seconds=lambda: busy_timeout_seconds,
    )
    try:
        with migration_engine.begin() as connection:
            state = _spool_revision_state(connection, history)
            if state is SpoolRevisionState.BEHIND:
                with alembic_config(
                    connection=connection,
                    script_location=script_location,
                ) as config:
                    command.upgrade(config, _MIGRATION_TARGET)
                _record_lineage(connection, history)
    finally:
        migration_engine.dispose()
    return state


def _spool_revision_state(
    connection: Connection,
    history: MigrationHistory,
) -> SpoolRevisionState:
    revision = MigrationContext.configure(connection).get_current_revision()
    lineage = None
    if revision is not None and revision not in history.revisions:
        lineage = _recorded_lineage(connection)
    state = classify_spool_revision(revision, lineage, history)
    if state is SpoolRevisionState.INCOMPATIBLE:
        raise IncompatibleSpoolRevisionError(
            UNKNOWN_SPOOL_REVISION_ERROR.format(revision=revision, head=history.head)
        )
    return state


def _recorded_lineage(connection: Connection) -> frozenset[str] | None:
    if not inspect(connection).has_table(SPOOL_LINEAGE_TABLE):
        return None
    return frozenset(connection.scalars(select(_LINEAGE.c.revision)))


def _record_lineage(connection: Connection, history: MigrationHistory) -> None:
    _LINEAGE.create(connection, checkfirst=True)
    connection.execute(delete(_LINEAGE))
    connection.execute(
        insert(_LINEAGE),
        [{"revision": revision} for revision in sorted(history.revisions)],
    )


def current_database_revision(path: Path | str) -> str | None:
    """Return the Alembic revision recorded by one spool database."""

    engine = create_spool_engine(path)
    try:
        with engine.connect() as connection:
            return MigrationContext.configure(connection).get_current_revision()
    finally:
        engine.dispose()


def recorded_spool_lineage(path: Path | str) -> frozenset[str] | None:
    """Return the lineage one spool database records, or ``None`` if none."""

    engine = create_spool_engine(path)
    try:
        with engine.connect() as connection:
            return _recorded_lineage(connection)
    finally:
        engine.dispose()


def migration_history(script_location: Traversable | None = None) -> MigrationHistory:
    """Return a checkout's migration history, by default the packaged one."""

    with alembic_config(script_location=script_location) as config:
        scripts = ScriptDirectory.from_config(config)
        return MigrationHistory(
            head=scripts.get_current_head(),
            revisions=frozenset(script.revision for script in scripts.walk_revisions()),
        )


def migration_head_revision(
    script_location: Traversable | None = None,
) -> str | None:
    """Return the single current head from a checkout's migration history."""

    return migration_history(script_location).head


def migration_revisions(
    script_location: Traversable | None = None,
) -> frozenset[str]:
    """Return every revision in a checkout's migration history."""

    return migration_history(script_location).revisions
