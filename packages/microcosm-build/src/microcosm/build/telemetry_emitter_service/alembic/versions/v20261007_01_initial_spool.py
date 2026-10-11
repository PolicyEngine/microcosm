"""Create or adopt the telemetry spool schema.

Revision ID: 20261007_01
Revises:
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20261007_01"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_RUNS_TABLE = "telemetry_runs"
_EVENTS_TABLE = "telemetry_events"
_EVENTS_INDEX = "telemetry_events_run_sequence"
_PENDING_UPLOAD_STATE = "pending"
_LOCAL_ONLY_UPLOAD_STATE = "local_only"
_PRE_ELIGIBILITY_REASON = "created_before_upload_eligibility"


def _create_runs_table() -> None:
    op.create_table(
        _RUNS_TABLE,
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("producer_id", sa.Text(), nullable=False),
        sa.Column("registration_json", sa.Text(), nullable=False),
        sa.Column(
            "next_sequence",
            sa.Integer(),
            nullable=False,
            server_default="1",
        ),
        sa.Column(
            "upload_state",
            sa.Text(),
            nullable=False,
            server_default=_PENDING_UPLOAD_STATE,
        ),
        sa.Column("local_only_reason", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("run_id", "producer_id"),
    )


def _adopt_runs_table(inspector: sa.Inspector) -> None:
    column_names = {column["name"] for column in inspector.get_columns(_RUNS_TABLE)}
    added_upload_state = "upload_state" not in column_names
    if added_upload_state:
        op.add_column(
            _RUNS_TABLE,
            sa.Column(
                "upload_state",
                sa.Text(),
                nullable=False,
                server_default=_PENDING_UPLOAD_STATE,
            ),
        )
    added_local_only_reason = "local_only_reason" not in column_names
    if added_local_only_reason:
        op.add_column(
            _RUNS_TABLE,
            sa.Column("local_only_reason", sa.Text(), nullable=True),
        )

    runs = sa.table(
        _RUNS_TABLE,
        sa.column("upload_state", sa.Text()),
        sa.column("local_only_reason", sa.Text()),
    )
    if added_upload_state:
        op.execute(sa.update(runs).values(upload_state=_LOCAL_ONLY_UPLOAD_STATE))
    if added_upload_state or added_local_only_reason:
        op.execute(
            sa.update(runs)
            .where(runs.c.upload_state == _LOCAL_ONLY_UPLOAD_STATE)
            .values(local_only_reason=_PRE_ELIGIBILITY_REASON)
        )


def _create_events_table() -> None:
    op.create_table(
        _EVENTS_TABLE,
        sa.Column("event_id", sa.Text(), nullable=False),
        sa.Column("run_id", sa.Text(), nullable=False),
        sa.Column("producer_id", sa.Text(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id", "producer_id"],
            [f"{_RUNS_TABLE}.run_id", f"{_RUNS_TABLE}.producer_id"],
            name="fk_telemetry_events_run",
        ),
        sa.PrimaryKeyConstraint("event_id"),
        sa.UniqueConstraint(
            "run_id",
            "producer_id",
            "sequence",
            name="uq_telemetry_events_run_producer_sequence",
        ),
    )


def upgrade() -> None:
    """Create the current schema or adopt a pre-Alembic spool."""

    inspector = sa.inspect(op.get_bind())
    table_names = set(inspector.get_table_names())
    if _RUNS_TABLE not in table_names:
        _create_runs_table()
    else:
        _adopt_runs_table(inspector)

    inspector = sa.inspect(op.get_bind())
    table_names = set(inspector.get_table_names())
    if _EVENTS_TABLE not in table_names:
        _create_events_table()

    inspector = sa.inspect(op.get_bind())
    index_names = {index["name"] for index in inspector.get_indexes(_EVENTS_TABLE)}
    if _EVENTS_INDEX not in index_names:
        op.create_index(
            _EVENTS_INDEX,
            _EVENTS_TABLE,
            ["run_id", "producer_id", "sequence"],
        )


def downgrade() -> None:
    """Remove the local spool schema."""

    op.drop_index(_EVENTS_INDEX, table_name=_EVENTS_TABLE)
    op.drop_table(_EVENTS_TABLE)
    op.drop_table(_RUNS_TABLE)
