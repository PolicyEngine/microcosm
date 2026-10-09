"""Programmatic Alembic migration entry points for the telemetry spool."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.engine import Connection, Engine

from microcosm.build.telemetry_emitter_service.database import create_spool_engine

_MIGRATION_TARGET = "head"


@contextmanager
def alembic_config(
    *,
    connection: Connection | None = None,
) -> Iterator[Config]:
    """Yield Alembic configuration with migrations on the filesystem."""

    migration_resources = files(__package__).joinpath("alembic")
    with as_file(migration_resources) as migration_directory:
        config = Config()
        config.set_main_option("script_location", str(migration_directory))
        if connection is not None:
            config.attributes["connection"] = connection
        yield config


def upgrade_spool_database(engine: Engine) -> None:
    """Bring a spool to the packaged head, one migrating process at a time.

    A spool already at head is only read, so opening it takes no write lock.
    Otherwise the whole upgrade, DDL and version stamp, runs in one
    ``BEGIN IMMEDIATE`` transaction. Without it two services opening a new
    spool at once interleave their DDL, and one fails with "table already
    exists". A service that waits for that lock finds the spool at head when
    it gets it, and Alembic then changes nothing.
    """

    with engine.connect() as connection:
        current = MigrationContext.configure(connection).get_current_revision()
    if current == migration_head_revision():
        return
    migration_engine = create_spool_engine(
        engine.url.database,
        immediate_transactions=True,
    )
    try:
        with migration_engine.begin() as connection:
            with alembic_config(connection=connection) as config:
                command.upgrade(config, _MIGRATION_TARGET)
    finally:
        migration_engine.dispose()


def current_database_revision(path: Path | str) -> str | None:
    """Return the Alembic revision recorded by one spool database."""

    engine = create_spool_engine(path)
    try:
        with engine.connect() as connection:
            return MigrationContext.configure(connection).get_current_revision()
    finally:
        engine.dispose()


def migration_head_revision() -> str | None:
    """Return the single current head from the packaged migration history."""

    with alembic_config() as config:
        return ScriptDirectory.from_config(config).get_current_head()
