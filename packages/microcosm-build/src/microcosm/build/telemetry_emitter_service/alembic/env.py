"""Alembic environment for the local telemetry spool."""

from alembic import context
from sqlalchemy.engine import Connection

from microcosm.build.telemetry_emitter_service.constants import SPOOL_LINEAGE_TABLE
from microcosm.build.telemetry_emitter_service.models import SpoolModel

_MISSING_CONNECTION_ERROR = (
    "telemetry spool migrations require a programmatically supplied connection"
)


def _include_name(name: str | None, type_: str, parent_names: object) -> bool:
    # The migration runner keeps the lineage table, as Alembic keeps
    # alembic_version, so the models do not describe it.
    return not (type_ == "table" and name == SPOOL_LINEAGE_TABLE)


def run_migrations() -> None:
    """Run migrations on the connection supplied by the emitter service."""

    connection = context.config.attributes.get("connection")
    if not isinstance(connection, Connection):
        raise RuntimeError(_MISSING_CONNECTION_ERROR)
    context.configure(
        connection=connection,
        target_metadata=SpoolModel.metadata,
        render_as_batch=True,
        include_name=_include_name,
    )
    with context.begin_transaction():
        context.run_migrations()


run_migrations()
