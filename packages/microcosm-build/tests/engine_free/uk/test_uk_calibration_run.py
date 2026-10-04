"""Tests using shared support from ``test_support.microcosm_build.uk_calibration_run``."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_calibration_run import *


def test_strict_checkpoint_binds_contents_and_retains_gate_payload(tmp_path):
    frame = _frame()
    path, gate_path, _ = _bound_checkpoint(tmp_path, frame)
    sidecar = calibration_run.load_bound_spine_checkpoint(path, frame)
    provenance = calibration_run.strict_spine_provenance_from_sidecar(path, sidecar)
    assert provenance["fit_weight_records"] == sidecar["fit_weight_records"]
    assert provenance["spine_gate_report"]["payload"] == json.loads(
        gate_path.read_bytes()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_identity",
        "wrong_identity",
        "bypass",
        "gate_bytes",
        "missing_gate_binding",
        "gate_roster",
        "gate_policy",
    ],
)
def test_strict_checkpoint_rejects_unbound_or_changed_evidence(tmp_path, mutation):
    frame = _frame()
    path, gate_path, sidecar = _bound_checkpoint(tmp_path, frame)
    if mutation == "missing_identity":
        sidecar.pop("uk_frame_content_identity")
    elif mutation == "wrong_identity":
        sidecar["uk_frame_content_identity"] = "f" * 64
    elif mutation == "bypass":
        sidecar["spine_gate_bypass"] = {"reviewed": True, "reason": "historical"}
    elif mutation == "gate_bytes":
        gate_path.write_text(gate_path.read_text() + "\n")
    elif mutation == "missing_gate_binding":
        sidecar.pop("spine_gate_report")
    elif mutation == "gate_policy":
        report = json.loads(gate_path.read_bytes())
        report["policy_sha256"] = "f" * 64
        gate_path.write_text(json.dumps(report))
        sidecar["spine_gate_report"]["sha256"] = hashlib.sha256(
            gate_path.read_bytes()
        ).hexdigest()
    else:
        report = json.loads(gate_path.read_bytes())
        report["gates"].pop(next(iter(report["gates"])))
        gate_path.write_text(json.dumps(report))
        sidecar["spine_gate_report"]["sha256"] = hashlib.sha256(
            gate_path.read_bytes()
        ).hexdigest()
    path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError):
        calibration_run.load_bound_spine_checkpoint(path, frame)


def test_strict_checkpoint_accepts_explicit_declared_gate_path(tmp_path):
    frame = _frame()
    path, gate_path, _ = _bound_checkpoint(tmp_path, frame)
    moved = gate_path.rename(tmp_path / "declared-gates.json")
    sidecar = calibration_run.load_bound_spine_checkpoint(
        path, frame, gate_report_path=moved
    )
    provenance = calibration_run.strict_spine_provenance_from_sidecar(
        path, sidecar, gate_report_path=moved
    )
    assert provenance["spine_gate_report"]["path"] == str(moved)


def test_gate_scope_classifies_every_uk_gate():
    all_ids = {entry.id for entry in load_country_spec("uk").gates.gates}
    assert (
        set(UK_CALIBRATION_GATE_SCOPE) | set(UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS)
        == all_ids
    )
    assert set(UK_CALIBRATION_GATE_SCOPE).isdisjoint(
        UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS
    )
    assert all(UK_CALIBRATION_GATE_SCOPE_EXCLUSIONS.values())


def test_import_hygiene_does_not_load_national_build_in_fresh_subprocess():
    source = Path(calibration_run.__file__).read_text(encoding="utf-8")
    legacy_module = ".".join(("microcosm", "build", "uk_runtime", "national_build"))
    assert legacy_module not in source
    assert " ".join(("from", legacy_module, "import")) not in source


def test_aggregate_admin_measurement_convention_and_refusals():
    frame = _frame()
    manifest = _manifest_with_small_anchor()

    totals, receipt = calibration_run.uk_aggregate_admin_totals(frame, manifest)

    # Small anchors measure as the weighted mean over carriers; the NHS total
    # measures as the person total under mapped household weights: 4 persons
    # x 50.0 x weight 10.0.
    assert totals["electricity_mean_spending"] == pytest.approx(1.0)
    assert totals["nhs_spending_total"] == pytest.approx(2000.0)
    assert "need_electricity_mean_spending" not in totals
    by_anchor = {row["anchor"]: row for row in receipt}
    assert by_anchor["nhs_spending_total"]["entity"] == "person"
    assert (
        by_anchor["electricity_mean_spending"]["statistic_convention"]
        == "assessed_by_anchor_magnitude"
    )

    stripped = _frame()
    stripped.table("household").drop(columns=["electricity_consumption"], inplace=True)
    with pytest.raises(ValueError, match="household.electricity_consumption"):
        calibration_run.uk_aggregate_admin_totals(stripped, manifest)


def test_nhs_anchor_composes_from_the_columns_the_spine_actually_carries():
    """The anchor is published as one total; the spine carries it in three parts.

    Composing is the translation from the published concept to ours, and the
    receipt has to say so — the anchor measured a silent zero for as long as it
    named a column no stage produces.
    """

    frame = _frame()
    person = frame.table("person")
    person.drop(columns=["nhs_spending"], inplace=True)
    person["nhs_a_and_e_spending"] = [20.0, 20.0, 20.0, 20.0]
    person["nhs_admitted_patient_spending"] = [25.0, 25.0, 25.0, 25.0]
    person["nhs_outpatient_spending"] = [5.0, 5.0, 5.0, 5.0]
    manifest = calibration_run._calibration_gate_manifest()

    totals, receipt = calibration_run.uk_aggregate_admin_totals(frame, manifest)

    # Same 4 persons x 50.0 x weight 10.0 as the single-column fixture.
    assert totals["nhs_spending_total"] == pytest.approx(2000.0)
    by_anchor = {row["anchor"]: row for row in receipt}
    assert by_anchor["nhs_spending_total"]["composed_from"] == list(
        UK_NHS_SPENDING_COMPONENT_COLUMNS
    )


def test_partly_carried_derived_anchor_refuses_and_names_the_missing_part():
    frame = _frame()
    person = frame.table("person")
    person.drop(columns=["nhs_spending"], inplace=True)
    person["nhs_a_and_e_spending"] = [20.0, 20.0, 20.0, 20.0]
    manifest = calibration_run._calibration_gate_manifest()

    with pytest.raises(ValueError, match="nhs_admitted_patient_spending"):
        calibration_run.uk_aggregate_admin_totals(frame, manifest)


def test_seam_pipeline_derives_a_ratified_logbook_scope():
    """The seam appends to the FRS line's chain, not a new unratified one."""

    logbook_tool = _load_logbook_tool()

    scope = logbook_tool._chain_scope(calibration_run._PIPELINE)

    assert scope == "uk/frs"
    assert scope in logbook_tool.DECLARED_SCOPES


def test_the_uk_block_delegates_rather_than_reassembling_the_shared_one(tmp_path):
    """Every field of the shared provenance block reaches the UK block.

    Pinned as a property, not as a list: the failure being prevented is a
    field the loader learns and this seam silently drops, and a test that
    enumerated today's fields would not catch tomorrow's.
    """
    artifact = load_ledger_consumer_artifact(_mixed_epoch_artifact_dir(tmp_path))
    shared = artifact.provenance()

    block = calibration_run._ledger_provenance(artifact)

    for field, value in shared.items():
        if field in {"path_name", "schema_version", "profiles"}:
            # Not identity of the feed: the directory name is local, and the
            # manifest id is carried in the narrower manifest sub-block.
            continue
        assert block[field] == value, field
    assert block["manifest"]["schema_version"] == shared["schema_version"]


# --- The contracts the graph's national path still stands on --------------


def _spec(name: str) -> TargetSpec:
    return TargetSpec(
        name=name,
        entity="benunit",
        measure=f"test/{name}",
        value=1.0,
        source="test",
        family="test",
        metadata={"contract_target_id": name},
    )


def test_band_edge_register_must_reconstitute_the_compiled_roster():
    """The reconciliation always runs: an empty receipt claims nothing was
    pruned, so the two rosters must then be name-identical; a receipt that
    disagrees with the pruned names refuses. The graph's national target
    node and the dense target compiler both stand on this check."""
    compiled = TargetRegistry([_spec("a"), _spec("b")], country="uk")
    pruned = TargetRegistry([_spec("a")], country="uk")
    calibration_run._validate_band_edge_registry(
        register_registry=pruned,
        band_edge_registry=compiled,
        exclusion_receipt={"b": {"reason": "test"}},
    )
    calibration_run._validate_band_edge_registry(
        register_registry=compiled, band_edge_registry=compiled, exclusion_receipt={}
    )
    with pytest.raises(ValueError, match="must match the measure exclusion receipt"):
        calibration_run._validate_band_edge_registry(
            register_registry=pruned, band_edge_registry=compiled, exclusion_receipt={}
        )
    with pytest.raises(ValueError, match="must include every materialized"):
        calibration_run._validate_band_edge_registry(
            register_registry=compiled,
            band_edge_registry=pruned,
            exclusion_receipt={},
        )


def test_attempt_ids_carry_the_seam_prefix_and_are_unique():
    """The driver's national role mints the seam's attempt id before telemetry
    opens (microcosm#823); two mints of one instant differ."""
    from datetime import UTC, datetime

    instant = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)
    first = calibration_run.new_uk_calibration_attempt_id(timestamp=instant)
    second = calibration_run.new_uk_calibration_attempt_id(timestamp=instant)
    prefix = calibration_run.UK_CALIBRATION_ATTEMPT_ID_PREFIX + "20260929T120000Z-"
    assert first.startswith(prefix) and second.startswith(prefix)
    assert first != second
