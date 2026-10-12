"""Shared helpers for UK CGT projection tests."""

from microcosm.build.country_spec import load_country_spec

GATE_ID = "uk_cgt_projection_entrants"


def manifest_entry():
    """Return the declared UK CGT projection validation entry."""

    return next(
        gate for gate in load_country_spec("uk").gates.gates if gate.id == GATE_ID
    )
