"""Tests split from packages/microcosm-build/tests/test_us_stored_input_register.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_stored_input_register import *


def test_the_modeled_stored_tables_are_what_the_writer_stores(builder, tmp_path):
    """Differential: the gate's model of the export against the H5 the real
    writer produces from the same frame, register columns and a stale input
    included; the gate and the written bytes reach the same verdict."""

    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

    frame = _frame(
        would_claim_wic=[True, False],
        person_source_id=[11, 12],
        person_support_channel=["asec", "asec"],
        source_year=[2024, 2024],
        tax_unit_role_input=["HEAD", "HEAD"],
    )
    path = tmp_path / "populace_us_2024.h5"
    PolicyEngineUSEngine().write_dataset(frame, path, period=builder.PERIOD)

    written = h5_stored_tables(path)
    modeled = builder._export_stored_tables(frame)
    assert {table: set(columns) for table, columns in written.items()} == {
        table: set(columns) for table, columns in modeled.items()
    }

    failures, details = builder._stored_input_gate_failures(frame, stage="export frame")
    assert details["refused"] == ["would_claim_wic"]
    assert len(failures) == 1
    assert builder._written_stored_input_verdict_mismatch(path, details) is None


def test_the_inputs_the_acs_native_reasons_name_are_engine_inputs():
    """What the ACS-native and alias reasons call engine inputs are inputs of
    the installed engine, not formulas, and the acs_ amounts themselves are
    not variables."""

    from policyengine_us.system import system

    named_inputs = {
        component
        for feature in acs_transfer._RECIPIENT_COMBINED_SOURCES
        for component in acs_transfer._DONOR_COMBINED_COMPONENTS[feature]
    } | {"pre_subsidy_rent", "real_estate_taxes", "puma"}
    for name in sorted(named_inputs):
        assert name in system.variables, name
        assert not system.variables[name].formulas, name
    assert not _ACS_NATIVE_AMOUNTS & set(system.variables)


def test_the_writer_refuses_the_engine_names_outside_the_convention(builder, tmp_path):
    """policyengine-us's only variables outside the lowercase convention are
    state-code formulas; the writer refuses to store a formula-owned column,
    so an uppercase stored column can never be one of them."""

    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

    frame = _frame()
    frame = Frame(
        {
            **{entity: frame.table(entity) for entity in frame.entities},
            "household": frame.table("household").assign(CA=[True, False]),
        },
        US_SCHEMA,
        {"household": frame.weights_for("household")},
    )

    with pytest.raises(
        ValueError, match="formula-owned column\\(s\\) present: \\['CA'\\]"
    ):
        PolicyEngineUSEngine().write_dataset(
            frame, tmp_path / "populace_us_2024.h5", period=builder.PERIOD
        )
