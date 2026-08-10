"""US PUF capital-gains own-tail transfer tests."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.puf_capital_gains_tail as tail_module
import microcosm.build.us_runtime.puf_interest_components as interest_module
from microcosm.build.us_runtime.capital_gain_distributions import (
    load_capital_gain_distribution_shares,
)
from microcosm.build.us_runtime.puf_aggregate_records import (
    load_default_puf_aggregate_disaggregation_spec,
)
from microcosm.build.us_runtime.puf_capital_gains_tail import (
    PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_FILING_STATUS_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_DONOR_SYNTHETIC_COLUMN,
    PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS,
    PUF_CAPITAL_GAINS_TAIL_POSITIVE_MASS_FIVE_X_TARGET,
    PUF_CAPITAL_GAINS_TAIL_QUANTILE,
    PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN,
    assert_puf_capital_gains_tail_survives_selection,
    puf_capital_gains_tail_concentration_gate,
    select_puf_capital_gains_tail_donors,
    transfer_puf_capital_gains_tail,
    write_puf_capital_gains_tail_manifest,
)
from microcosm.build.us_runtime.puf_support import (
    PUF_DONOR_SOURCE_ADJUSTED_GROSS_INCOME_COLUMN,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    clone_us_frame_for_puf_support,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.build.us_runtime.support_provenance import (
    spine_assembly_manifest,
    spine_source_id_column,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def _expanded_recipient_frame() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3, 4], dtype="int64"),
            "person_household_id": np.asarray([1, 2, 3, 4], dtype="int64"),
            "person_tax_unit_id": np.asarray([10, 20, 30, 40], dtype="int64"),
            "person_spm_unit_id": np.asarray([100, 200, 300, 400], dtype="int64"),
            "person_family_id": np.asarray(
                [1_000, 2_000, 3_000, 4_000],
                dtype="int64",
            ),
            "person_marital_unit_id": np.asarray(
                [10_000, 20_000, 30_000, 40_000],
                dtype="int64",
            ),
            "employment_income_before_lsr": [50_000.0, 80_000.0, 5_000.0, 9_000.0],
            "self_employment_income_before_lsr": [0.0, 2_000.0, 0.0, 1_000.0],
            "taxable_interest_income": [10.0, 20.0, 30.0, 40.0],
            "qualified_dividend_income": [0.0, 0.0, 0.0, 0.0],
            "non_qualified_dividend_income": [0.0, 0.0, 0.0, 0.0],
            "short_term_capital_gains": [100.0, -200.0, 300.0, -400.0],
            "long_term_capital_gains_before_response": [
                1_000.0,
                2_000.0,
                3_000.0,
                4_000.0,
            ],
            "long_term_capital_gains_on_collectibles": [1.0, 2.0, 3.0, 4.0],
            "non_sch_d_capital_gains": [5.0, 6.0, 7.0, 8.0],
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": np.asarray([1, 2, 3, 4], dtype="int64"),
                "state_fips": [6, 36, 48, 12],
            }
        ),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": np.asarray([10, 20, 30, 40], dtype="int64"),
                "filing_status_input": ["JOINT", "SINGLE", "JOINT", "SINGLE"],
                "adjusted_gross_income": [50_000.0, 82_000.0, 5_000.0, 10_000.0],
                "income_tax": [4_000.0, 8_000.0, 0.0, 100.0],
                "unrecaptured_section_1250_gain": [10.0, 20.0, 30.0, 40.0],
            }
        ),
        "spm_unit": pd.DataFrame(
            {"spm_unit_id": np.asarray([100, 200, 300, 400], dtype="int64")}
        ),
        "family": pd.DataFrame(
            {"family_id": np.asarray([1_000, 2_000, 3_000, 4_000], dtype="int64")}
        ),
        "marital_unit": pd.DataFrame(
            {
                "marital_unit_id": np.asarray(
                    [10_000, 20_000, 30_000, 40_000],
                    dtype="int64",
                )
            }
        ),
    }
    base = Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([100.0, 300.0, 200.0, 400.0]),
                WeightKind.DESIGN,
            )
        },
        pd.Series(["a", "b", "c", "d"], name="stratum"),
    )
    return clone_us_frame_for_puf_support(base)


def test_tail_execution_identity_binds_resolved_spec_and_soi_asset(
    tmp_path: Path,
) -> None:
    baseline = tail_module.puf_capital_gains_tail_execution_inputs_identity()
    buckets = baseline["aggregate_disaggregation_spec"]["buckets"]
    assert [bucket["recid"] for bucket in buckets] == sorted(
        bucket["recid"] for bucket in buckets
    )

    source = interest_module.files("microcosm.build.us").joinpath(
        interest_module._SOURCE_ASSET
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["agi_bands"][0]["total_interest_paid_amount"] += 1
    payload["agi_bands"][0]["investment_interest_amount"] += 1
    changed_asset = tmp_path / interest_module._SOURCE_ASSET
    changed_asset.write_text(json.dumps(payload), encoding="utf-8")
    changed = interest_module.puf_e19200_interest_components_asset_identity(
        changed_asset
    )

    soi = baseline["soi_e19200_agi_bands"]
    assert soi["asset_sha256"] != changed["asset_sha256"]
    assert soi["agi_bands"][0] != changed["agi_bands"][0]


def _donor() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tax_unit_id": [10, 20, 1_000_001],
            "weight": [996.0, 3.0, 1.0],
            "filing_status_code": [1.0, 2.0, 1.0],
            PUF_DONOR_SOURCE_ADJUSTED_GROSS_INCOME_COLUMN: [
                100_000.0,
                5_000_000.0,
                10_000_000.0,
            ],
            "short_term_capital_gains": [0.0, -10_000_000.0, 5_000_000.0],
            "long_term_capital_gains_before_response": [
                100_000.0,
                100_000_000.0,
                75_000_000.0,
            ],
            "long_term_capital_gains_on_collectibles": [
                0.0,
                2_000_000.0,
                1_000_000.0,
            ],
            "non_sch_d_capital_gains": [0.0, 3_000_000.0, 0.0],
            "unrecaptured_section_1250_gain": [
                0.0,
                4_000_000.0,
                250_000.0,
            ],
        }
    )


def _assembled_native_recipient_frame() -> Frame:
    expanded = _expanded_recipient_frame()
    native = expanded.select(
        expanded.table("person")[support_clone_index_column("person")].eq(0).to_numpy()
    )
    tables = {entity: native.table(entity).copy() for entity in native.entities}
    for entity, table in tables.items():
        table[spine_source_id_column(entity)] = table[
            support_source_id_column(entity)
        ].to_numpy()
    return Frame(
        tables,
        native.schema,
        {
            "household": Weights(
                native.weights_for("household").values * 2.0,
                native.weights_for("household").kind,
            )
        },
        native.strata,
        mass_log=native.mass_log,
        metadata=spine_assembly_manifest(tables, channels=("asec", "acs")),
    )


def _partially_attached_recipient_frame() -> Frame:
    return clone_us_frame_for_puf_support(
        _assembled_native_recipient_frame(),
        clone_attachment_fraction=0.5,
        clone_attachment_seed=1,
    )


def _replace_entity_table(frame: Frame, entity: str, table: pd.DataFrame) -> Frame:
    tables = {name: frame.table(name).copy() for name in frame.entities}
    tables[entity] = table
    return Frame(
        tables,
        frame.schema,
        {name: frame.weights_for(name) for name in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _frame_digest(frame: Frame) -> str:
    """Hash every frame byte-bearing surface for pre-fix parity checks."""

    payload = (
        [(entity, frame.table(entity)) for entity in frame.entities],
        [
            (
                entity,
                frame.weights_for(entity).values,
                frame.weights_for(entity).kind.value,
            )
            for entity in frame.weighted_entities
        ],
        frame.strata,
        frame.mass_log,
    )
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()


def _pre_652_all_adequate_reference_frame() -> Frame:
    """Run the pre-support-filter allocation path in the live test runtime."""

    frame = _expanded_recipient_frame()
    donor = _donor()
    tail, _selection = select_puf_capital_gains_tail_donors(donor)
    normalization = frame.weights_for("household").total / float(donor["weight"].sum())
    assigned_weights = tail["weight"].to_numpy(dtype=np.float64) * normalization
    candidates = tail_module._recipient_candidates(
        frame,
        maximum_transfer_weight=float(assigned_weights.max()),
        seed=567,
    )
    assignments = tail_module._assign_tail_donors(
        tail,
        assigned_weights=assigned_weights,
        candidates=candidates,
    )
    reference, _clone_receipt = tail_module._clone_and_transfer(frame, assignments)
    return reference


def _load_support_builder_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_us_puf_support_base.py"
    spec = importlib.util.spec_from_file_location("build_us_puf_support_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tail_transfer_matches_full_and_partial_clone_attachments() -> None:
    assembled = _assembled_native_recipient_frame()
    full = clone_us_frame_for_puf_support(
        assembled,
        clone_attachment_fraction=1.0,
        clone_attachment_seed=1,
    )
    partial = _partially_attached_recipient_frame()

    assert tail_module._support_clone_multiplier(full) == 100_000
    assert tail_module._support_clone_multiplier(partial) == 100_000
    household = partial.table("household")
    clone_index = household[support_clone_index_column("household")]
    attached_sources = household.loc[
        clone_index.eq(1),
        support_source_id_column("household"),
    ].astype("int64")
    assert attached_sources.tolist() == [2, 3]

    transferred, manifest = transfer_puf_capital_gains_tail(
        partial,
        _donor(),
        seed=567,
    )

    assert manifest["record_count"] == 2
    assert manifest["clone"]["id_multiplier"] == 100_000
    partial.weights_for("household").assert_mass_conserved(
        transferred.weights_for("household")
    )


def test_support_clone_multiplier_rejects_unmatched_detail_source_id() -> None:
    partial = _partially_attached_recipient_frame()
    person = partial.table("person").copy()
    detail = person[support_clone_index_column("person")].eq(1)
    person.loc[person.index[detail][0], support_source_id_column("person")] = 999_999
    corrupted = _replace_entity_table(partial, "person", person)

    with pytest.raises(
        ValueError,
        match=(
            "PUF support person detail clone source IDs have no native match: "
            r"\[999999\]"
        ),
    ):
        tail_module._support_clone_multiplier(corrupted)


def test_support_clone_multiplier_rejects_duplicate_detail_source_id() -> None:
    partial = _partially_attached_recipient_frame()
    person = partial.table("person").copy()
    detail_positions = person.index[
        person[support_clone_index_column("person")].eq(1)
    ].tolist()
    source_id = support_source_id_column("person")
    person.loc[detail_positions[1], source_id] = person.loc[
        detail_positions[0], source_id
    ]
    corrupted = _replace_entity_table(partial, "person", person)

    with pytest.raises(
        ValueError,
        match="PUF support person detail clone source IDs are not unique",
    ):
        tail_module._support_clone_multiplier(corrupted)


def test_tail_selection_consumes_synthetic_eligibility() -> None:
    assert PUF_CAPITAL_GAINS_TAIL_QUANTILE == 0.995
    default_spec = load_default_puf_aggregate_disaggregation_spec()
    assert default_spec.synthetic_tail_support_eligible

    eligible, receipt = select_puf_capital_gains_tail_donors(
        _donor(),
        spec=default_spec,
    )
    assert eligible["tax_unit_id"].tolist() == [20, 1_000_001]
    assert receipt["comparison"] == "strictly_greater_than"
    assert receipt["realized_boundary"] == 100_000.0
    assert receipt["next_reference_quantile"] == 0.999
    assert receipt["next_reference_boundary"] == 90_000_000.0
    assert receipt["recipient_topcode"] == 1_999_998.0
    assert receipt["tail_record_count"] == 2
    assert receipt["synthetic_tail_record_count"] == 1

    regular_only, regular_receipt = select_puf_capital_gains_tail_donors(
        _donor(),
        spec=dataclasses.replace(
            default_spec,
            synthetic_tail_support_eligible=False,
        ),
    )
    assert regular_only["tax_unit_id"].tolist() == [20]
    assert not regular_receipt["synthetic_tail_support_eligible"]
    assert regular_receipt["synthetic_tail_record_count"] == 0


def test_tail_transfer_splits_weights_and_copies_joint_vectors(
    tmp_path: Path,
) -> None:
    frame = _expanded_recipient_frame()
    donor = _donor()
    before_household_weights = frame.weights_for("household")
    before_employment_mass = float(
        np.dot(
            frame.table("person")["employment_income_before_lsr"],
            frame.resolve_weights("person").values,
        )
    )
    before_tax_unit_masses = {
        column: float(
            np.dot(
                frame.table("tax_unit")[column],
                frame.resolve_weights("tax_unit").values,
            )
        )
        for column in ("adjusted_gross_income", "income_tax")
    }
    before_state_weights = (
        pd.Series(
            frame.weights_for("household").values,
            index=frame.table("household")["state_fips"],
        )
        .groupby(level=0)
        .sum()
    )

    transferred, manifest = transfer_puf_capital_gains_tail(
        frame,
        donor,
        seed=567,
    )
    repeated, repeated_manifest = transfer_puf_capital_gains_tail(
        frame,
        donor,
        seed=567,
    )

    assert manifest["manifest_sha256"] == repeated_manifest["manifest_sha256"]
    assert manifest["assignment_sha256"] == repeated_manifest["assignment_sha256"]
    pd.testing.assert_frame_equal(
        transferred.table("person"),
        repeated.table("person"),
    )
    np.testing.assert_array_equal(
        transferred.weights_for("household").values,
        repeated.weights_for("household").values,
    )

    before_household_weights.assert_mass_conserved(transferred.weights_for("household"))
    assert transferred.n("household") == frame.n("household") + 2
    assert transferred.n("person") == frame.n("person") + 2
    for group in US_SCHEMA.group_entities:
        assert transferred.n(group) == frame.n(group) + 2
    assert manifest["weight_domain"]["design_weight_normalization"] == 1.0
    assert manifest["weight_domain"]["assigned_tail_weight"] == 4.0
    assert manifest["clone"]["household_weight_difference"] == 0.0
    for group in US_SCHEMA.group_entities:
        assert manifest["clone"]["effective_group_weight_differences"][group] == 0.0
    assert manifest["carrier_reconciliation"]["passed"]
    assert manifest["carrier_reconciliation"]["observed_tail_tax_unit_count"] == 2
    assert set(
        manifest["carrier_reconciliation"]["maximum_absolute_differences"].values()
    ) == {0.0}
    assert manifest["joint_vector_policy"]["amount_scale"] == 1.0
    assert not manifest["joint_vector_policy"]["legs_scaled_independently"]

    after_employment_mass = float(
        np.dot(
            transferred.table("person")["employment_income_before_lsr"],
            transferred.resolve_weights("person").values,
        )
    )
    assert after_employment_mass == before_employment_mass
    for column, before_mass in before_tax_unit_masses.items():
        after_mass = float(
            np.dot(
                transferred.table("tax_unit")[column],
                transferred.resolve_weights("tax_unit").values,
            )
        )
        assert after_mass == before_mass
    after_state_weights = (
        pd.Series(
            transferred.weights_for("household").values,
            index=transferred.table("household")["state_fips"],
        )
        .groupby(level=0)
        .sum()
    )
    pd.testing.assert_series_equal(after_state_weights, before_state_weights)

    person = transferred.table("person")
    tax_unit = transferred.table("tax_unit")
    household = transferred.table("household")
    for record in manifest["records"]:
        tail_tax_unit_id = record["tail_tax_unit_id"]
        tail_household_id = record["tail_household_id"]
        tail_people = person.loc[person["person_tax_unit_id"] == tail_tax_unit_id]
        assert len(tail_people) == 1
        for column in PUF_CAPITAL_GAINS_TAIL_PERSON_COLUMNS:
            assert tail_people[column].sum() == record["joint_vector"][column]
        tail_tax_unit = tax_unit.loc[tax_unit["tax_unit_id"] == tail_tax_unit_id]
        assert len(tail_tax_unit) == 1
        assert bool(tail_tax_unit[PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN].iloc[0])
        assert (
            tail_tax_unit[PUF_CAPITAL_GAINS_TAIL_DONOR_SOURCE_ID_COLUMN].iloc[0]
            == record["donor_source_id"]
        )
        assert (
            bool(tail_tax_unit[PUF_CAPITAL_GAINS_TAIL_DONOR_SYNTHETIC_COLUMN].iloc[0])
            == record["donor_is_synthetic"]
        )
        assert (
            tail_tax_unit[PUF_CAPITAL_GAINS_TAIL_DONOR_FILING_STATUS_COLUMN].iloc[0]
            == record["donor_filing_status_code"]
        )
        assert (
            tail_tax_unit[PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN].iloc[0]
            == record["donor_agi_band_index"]
        )
        assert (
            tail_tax_unit[PUF_CAPITAL_GAINS_TAIL_TRANSFER_WEIGHT_COLUMN].iloc[0]
            == record["assigned_weight"]
        )
        assert (
            tail_tax_unit["unrecaptured_section_1250_gain"].iloc[0]
            == record["joint_vector"]["unrecaptured_section_1250_gain"]
        )
        assert (
            tail_tax_unit["filing_status_input"].iloc[0]
            == record["donor_filing_status"]
        )
        tail_household = household.loc[household["household_id"] == tail_household_id]
        assert len(tail_household) == 1
        assert (
            tail_household[support_channel_column("household")].iloc[0]
            == PUF_TAX_DETAIL_SUPPORT_CHANNEL
        )
        assert tail_household[support_clone_index_column("household")].iloc[0] == 2
        assert record["recipient_household_weight_before"] == (
            record["recipient_household_weight_after"] + record["assigned_weight"]
        )

    for column in (
        "short_term_capital_gains",
        "long_term_capital_gains_before_response",
    ):
        reconciliation = manifest["signed_leg_reconciliation"][column]
        assert reconciliation["difference"] == 0.0
        assert (
            reconciliation["expected_frame_weighted_signed_mass"]
            == (reconciliation["transferred_frame_weighted_signed_mass"])
        )
    assert manifest["tail_concentration_gate"]["passed"]
    assert manifest["frame_after_stage_concentration_gate"]["passed"]
    after_receipt = manifest["tail_distribution_receipts"]["frame_after_stage"]
    assert after_receipt["positive_mass_five_x_target"] == (
        PUF_CAPITAL_GAINS_TAIL_POSITIVE_MASS_FIVE_X_TARGET
    )
    assert after_receipt["positive_mass_five_x_headroom"] == (
        after_receipt["positive_mass_five_x_ceiling"]
        - PUF_CAPITAL_GAINS_TAIL_POSITIVE_MASS_FIVE_X_TARGET
    )
    assert after_receipt["positive_mass_five_x_target_exceeded"] == (
        after_receipt["positive_mass_five_x_ceiling"]
        > PUF_CAPITAL_GAINS_TAIL_POSITIVE_MASS_FIVE_X_TARGET
    )
    assert (
        manifest["tail_distribution_receipts"]["donor"]["conditional_positive_mean"]
        == manifest["tail_distribution_receipts"]["frame_transferred"][
            "conditional_positive_mean"
        ]
    )

    manifest_path = tmp_path / "tail.manifest.json"
    file_sha = write_puf_capital_gains_tail_manifest(manifest_path, manifest)
    assert file_sha == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    first_bytes = manifest_path.read_bytes()
    repeated_sha = write_puf_capital_gains_tail_manifest(
        manifest_path,
        repeated_manifest,
    )
    assert repeated_sha == file_sha
    assert manifest_path.read_bytes() == first_bytes

    tampered = json.loads(first_bytes)
    tampered["records"][0]["tail_person_id"] += 1
    tampered_payload = dict(tampered)
    tampered_payload.pop("manifest_sha256")
    tampered["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            tampered_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="assignment SHA mismatch"):
        write_puf_capital_gains_tail_manifest(manifest_path, tampered)


def test_thin_filing_status_is_named_counted_and_not_attached() -> None:
    """A thin status is skipped whole while an adequate peer still attaches."""

    frame = _expanded_recipient_frame()
    donor = _donor()
    donor.loc[donor["tax_unit_id"].eq(20), "filing_status_code"] = 3.0

    transferred, manifest = transfer_puf_capital_gains_tail(
        frame,
        donor,
        seed=567,
    )

    support = manifest["recipient_support"]
    by_status = {receipt["filing_status"]: receipt for receipt in support["strata"]}
    assert support["insufficient_support_stratum_count"] == 1
    assert support["insufficient_support_strata"] == ["SEPARATE"]
    assert by_status["SEPARATE"] == {
        "filing_status_code": 3,
        "filing_status": "SEPARATE",
        "status": "insufficient_support",
        "observed_count": 0,
        "required_minimum": 1,
        "attached_donor_count": 0,
        "skipped_donor_count": 1,
    }
    assert by_status["SINGLE"]["status"] == "attached"
    assert by_status["SINGLE"]["observed_count"] == 2
    assert by_status["SINGLE"]["required_minimum"] == 1
    assert by_status["SURVIVING_SPOUSE"]["status"] == "not_applicable"
    assert manifest["record_count"] == 1
    assert {record["donor_filing_status"] for record in manifest["records"]} == {
        "SINGLE"
    }
    assert transferred.n("household") == frame.n("household") + 1


def test_adequate_strata_match_pre_fix_frame_bytes() -> None:
    """All-adequate fixtures preserve the exact pre-#652 allocation bytes."""

    transferred, manifest = transfer_puf_capital_gains_tail(
        _expanded_recipient_frame(),
        _donor(),
        seed=567,
    )

    # Pandas' pickle bytes vary across supported runtime versions, so compare
    # against the exact pre-#652 path under the same runtime instead of blessing
    # one environment's pickle digest.
    assert _frame_digest(transferred) == _frame_digest(
        _pre_652_all_adequate_reference_frame()
    )
    assert manifest["assignment_sha256"] == (
        "1b2262da65fa851e0a990ca9f04dee661de0145724f82aef679557bc92418937"
    )
    assert manifest["recipient_support"]["insufficient_support_strata"] == []


def test_tail_support_receipt_tampering_fails_closed() -> None:
    """Rehashing only the envelope cannot launder a changed support count."""

    _transferred, manifest = transfer_puf_capital_gains_tail(
        _expanded_recipient_frame(),
        _donor(),
        seed=567,
    )
    tampered = json.loads(json.dumps(manifest))
    tampered["recipient_support"]["strata"][0]["observed_count"] += 1
    tampered.pop("manifest_sha256")
    tampered["manifest_sha256"] = tail_module._canonical_sha256(tampered)

    with pytest.raises(ValueError, match="recipient-support SHA mismatch"):
        tail_module.validate_puf_capital_gains_tail_manifest(tampered)


def test_tail_transfer_rejects_group_membership_crossing_households() -> None:
    frame = _expanded_recipient_frame()
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    person = tables["person"]
    puf = person[support_channel_column("person")].eq(PUF_TAX_DETAIL_SUPPORT_CHANNEL)
    shared_family_id = int(person.loc[puf, "person_family_id"].iloc[0])
    person.loc[puf, "person_family_id"] = shared_family_id
    referenced_family_ids = set(person["person_family_id"].astype("int64"))
    tables["family"] = tables["family"].loc[
        tables["family"]["family_id"].isin(referenced_family_ids)
    ]
    invalid = Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        frame.strata,
    )

    with pytest.raises(ValueError, match="family membership crosses"):
        transfer_puf_capital_gains_tail(invalid, _donor(), seed=567)


def test_transfer_reconciliation_reads_materialized_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_clone = tail_module._clone_and_transfer

    def corrupt_clone(frame, assignments):
        transferred, receipt = real_clone(frame, assignments)
        person = transferred.table("person")
        tail = person[support_clone_index_column("person")].eq(2)
        position = person.index[tail][0]
        person.loc[position, "short_term_capital_gains"] += 1.0
        return transferred, receipt

    monkeypatch.setattr(tail_module, "_clone_and_transfer", corrupt_clone)

    with pytest.raises(
        AssertionError,
        match="materialized carrier changed short_term_capital_gains",
    ):
        transfer_puf_capital_gains_tail(
            _expanded_recipient_frame(),
            _donor(),
            seed=567,
        )


def test_transfer_reconciliation_reads_materialized_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_clone = tail_module._clone_and_transfer

    def corrupt_clone(frame, assignments):
        transferred, receipt = real_clone(frame, assignments)
        tax_unit = transferred.table("tax_unit")
        applied = tax_unit[PUF_CAPITAL_GAINS_TAIL_APPLIED_COLUMN].astype(bool)
        position = tax_unit.index[applied][0]
        tax_unit.loc[position, PUF_CAPITAL_GAINS_TAIL_DONOR_AGI_BAND_COLUMN] += 1
        return transferred, receipt

    monkeypatch.setattr(tail_module, "_clone_and_transfer", corrupt_clone)

    with pytest.raises(
        AssertionError,
        match="materialized carrier changed lineage field donor_agi_band_index",
    ):
        transfer_puf_capital_gains_tail(
            _expanded_recipient_frame(),
            _donor(),
            seed=567,
        )


def test_frozen_selection_must_retain_every_tail_donor() -> None:
    transferred, manifest = transfer_puf_capital_gains_tail(
        _expanded_recipient_frame(),
        _donor(),
        seed=567,
    )
    receipt = assert_puf_capital_gains_tail_survives_selection(
        transferred,
        transferred,
    )
    assert receipt["passed"]
    assert receipt["status"] == "retained"
    assert receipt["base_tail_record_count"] == manifest["record_count"]

    person = transferred.table("person")
    without_tail = transferred.select(
        ~person[support_clone_index_column("person")].eq(2).to_numpy()
    )
    with pytest.raises(
        ValueError,
        match="Regenerate the selection-source manifest",
    ):
        assert_puf_capital_gains_tail_survives_selection(
            transferred,
            without_tail,
        )


def test_tail_vectors_feed_schedule_d_stage_without_changing_source_legs() -> None:
    transferred, manifest = transfer_puf_capital_gains_tail(
        _expanded_recipient_frame(),
        _donor(),
        seed=567,
    )
    source_columns = (
        "short_term_capital_gains",
        "long_term_capital_gains_before_response",
        "non_sch_d_capital_gains",
    )
    before = transferred.table("person")[list(source_columns)].copy()
    builder = _load_support_builder_module()

    distributed, _ = builder._capital_gain_distributions_stage(
        SimpleNamespace(seed=567, target_year=2024),
        transferred,
    )

    pd.testing.assert_frame_equal(
        distributed.table("person")[list(source_columns)],
        before,
    )
    schedule_d_by_tax_unit = (
        distributed.table("person")
        .groupby("person_tax_unit_id", sort=False)[
            "schedule_d_capital_gain_distributions"
        ]
        .sum()
    )
    share = load_capital_gain_distribution_shares().schedule_d_cgd_share_of_lt_net_gains
    for record in manifest["records"]:
        vector = record["joint_vector"]
        actual = schedule_d_by_tax_unit.loc[record["tail_tax_unit_id"]]
        if vector["non_sch_d_capital_gains"] > 0.0:
            assert actual == 0.0
        else:
            assert actual == pytest.approx(
                vector["long_term_capital_gains_before_response"] * share
            )


def test_tail_stratum_passes_existing_weighted_top_100_gate() -> None:
    tail_count = 600
    combined = np.linspace(2_100_000.0, 10_000_000.0, tail_count)
    short_term = np.where(
        np.arange(tail_count) % 2 == 0,
        -0.1 * combined,
        0.1 * combined,
    )
    donor = pd.DataFrame(
        {
            "tax_unit_id": np.arange(1, tail_count + 2, dtype=np.int64),
            "weight": np.concatenate(
                [np.asarray([996.0]), np.full(tail_count, 4.0 / tail_count)]
            ),
            "filing_status_code": np.ones(tail_count + 1),
            PUF_DONOR_SOURCE_ADJUSTED_GROSS_INCOME_COLUMN: np.concatenate(
                [np.asarray([100_000.0]), combined]
            ),
            "short_term_capital_gains": np.concatenate([np.asarray([0.0]), short_term]),
            "long_term_capital_gains_before_response": np.concatenate(
                [np.asarray([100_000.0]), combined - short_term]
            ),
            "long_term_capital_gains_on_collectibles": np.concatenate(
                [np.asarray([0.0]), 0.01 * combined]
            ),
            "non_sch_d_capital_gains": np.zeros(tail_count + 1),
            "unrecaptured_section_1250_gain": np.concatenate(
                [np.asarray([0.0]), 0.005 * combined]
            ),
        }
    )

    tail, receipt = select_puf_capital_gains_tail_donors(donor)
    assert len(tail) == tail_count
    assert receipt["realized_boundary"] == 100_000.0
    gate = puf_capital_gains_tail_concentration_gate(tail)
    assert gate.passed
    assert (
        gate.details["carrier_counts"]["short_term_plus_long_term_capital_gains"]
        == tail_count
    )
    assert gate.details["top_share"]["short_term_plus_long_term_capital_gains"] < 0.75


def test_joint_vectors_arrive_verbatim_from_selected_donors() -> None:
    """microcosm#570 review, Critical: the candidate merge previously
    replaced donor joint-vector values held at tax-unit grain (notably
    unrecaptured_section_1250_gain) with the RECIPIENT's existing values —
    99.7% of intended donor mass lost, and reconciliation self-confirmed
    because it derives expectations from assignments. Every joint-vector
    column must arrive verbatim from the SELECTED donor, per-column, into
    assignments, the manifest, and the materialized frame."""
    from microcosm.build.us_runtime.puf_capital_gains_tail import (
        transfer_puf_capital_gains_tail,
    )

    frame = _expanded_recipient_frame()
    donor = _donor()
    transferred, manifest = transfer_puf_capital_gains_tail(frame, donor, seed=7)

    donor_by_id = donor.set_index("tax_unit_id")
    joint_columns = (
        "short_term_capital_gains",
        "long_term_capital_gains_before_response",
        "long_term_capital_gains_on_collectibles",
        "non_sch_d_capital_gains",
        "unrecaptured_section_1250_gain",
    )
    records = manifest["records"]
    assert records
    for record in records:
        source = donor_by_id.loc[record["donor_source_id"]]
        for column in joint_columns:
            assert float(record["joint_vector"][column]) == float(source[column]), (
                f"{column} did not arrive verbatim for donor "
                f"{record['donor_source_id']}"
            )
    # And the materialized frame carries the same values on the tail clones.
    tax_unit = transferred.table("tax_unit")
    by_tail = {record["tail_tax_unit_id"]: record for record in records}
    clone_mask = tax_unit["tax_unit_id"].isin(list(by_tail))
    assert int(clone_mask.sum()) == len(records)
    for _, clone in tax_unit.loc[clone_mask].iterrows():
        record = by_tail[int(clone["tax_unit_id"])]
        for column in (
            "long_term_capital_gains_on_collectibles",
            "non_sch_d_capital_gains",
            "unrecaptured_section_1250_gain",
        ):
            if column in tax_unit.columns:
                assert float(clone[column]) == float(record["joint_vector"][column])


def test_frame_gate_scopes_to_stage_attributable_worsening() -> None:
    """microcosm#571 rounds 1-2: the comparator reads RAW pre/post
    measurements under the production gate's finite/positive-mass mask
    (only the min-carriers floor omitted), derives over-threshold
    membership from the PRODUCTION gate, tolerates ULP noise, and fails
    only over-threshold columns the stage strictly worsened. The replay
    case pins the exact Base-P3 geometry: pre 100%/97 carriers (thin —
    the production gate omits it), post ~83.9%/1,135 carriers (over
    threshold, gate-flagged), NOT worsened -> passes."""
    from microcosm.build.gates import tail_concentration_gate
    from microcosm.build.us_runtime.puf_capital_gains_tail import (
        _WORSENING_SHARE_TOLERANCE,
        _raw_top_share_receipts,
        _stage_attributable_concentration_failures,
    )

    n = 3_000
    weights = np.ones(n)
    pre_collect = np.zeros(n)
    pre_collect[:97] = 1_000_000.0
    # Post: 100 records at $1M + 1,035 records sized so the top-100 share
    # lands at the replay's 0.839: rest = top * (1/share - 1).
    post_collect = np.zeros(n)
    post_collect[:100] = 1_000_000.0
    rest_total = 100 * 1_000_000.0 * (1.0 / 0.839 - 1.0)
    post_collect[100:1_135] = rest_total / 1_035
    # A NaN row must not poison the measurement (round-2 High).
    pre_collect[n - 1] = np.nan
    post_collect[n - 1] = np.nan

    pre_receipts = _raw_top_share_receipts({"collect": pre_collect}, weights)
    post_receipts = _raw_top_share_receipts({"collect": post_collect}, weights)
    assert pre_receipts["collect"]["top_share"] == pytest.approx(1.0)
    assert pre_receipts["collect"]["carriers"] == 97
    assert post_receipts["collect"]["top_share"] == pytest.approx(0.839, abs=1e-9)
    assert post_receipts["collect"]["carriers"] == 1_135
    assert pre_receipts["collect"]["distinct_values"] == 1
    assert post_receipts["collect"]["distinct_values"] == 2

    # Membership through the PRODUCTION gate: pre is thin (omitted), post
    # is checked and fails the absolute threshold.
    pre_gate = tail_concentration_gate(
        {"collect": pre_collect[np.isfinite(pre_collect)]},
        {"collect": weights[np.isfinite(pre_collect)]},
    )
    assert "collect" not in pre_gate.details["top_share"]
    post_gate = tail_concentration_gate(
        {"collect": post_collect[np.isfinite(post_collect)]},
        {"collect": weights[np.isfinite(post_collect)]},
    )
    assert not post_gate.passed

    failures, receipts = _stage_attributable_concentration_failures(
        pre_receipts, post_receipts, post_gate
    )
    assert failures == []
    assert receipts["collect"]["over_threshold"] is True
    assert receipts["collect"]["stage_worsened_share"] is False
    assert receipts["collect"]["pre_stage_carriers"] == 97
    assert receipts["collect"]["post_stage_carriers"] == 1_135

    # A stage-worsened over-threshold column still fails, derived through
    # the production gate as well.
    worsened_pre = np.zeros(n)
    worsened_pre[:700] = np.linspace(1.0, 700.0, 700)
    worsened_post = np.zeros(n)
    worsened_post[:600] = 1.0
    worsened_post[:100] = 1_000_000.0
    worsened_gate = tail_concentration_gate({"w": worsened_post}, {"w": weights})
    assert not worsened_gate.passed
    worsened_failures, worsened_receipts = _stage_attributable_concentration_failures(
        _raw_top_share_receipts({"w": worsened_pre}, weights),
        _raw_top_share_receipts({"w": worsened_post}, weights),
        worsened_gate,
    )
    assert len(worsened_failures) == 1 and worsened_failures[0].startswith("w:")
    assert worsened_receipts["w"]["stage_worsened_share"] is True

    # ULP-scale movement is numerical noise, not a worsening — asserted
    # against an OVER-THRESHOLD gate result so the tolerance is load-bearing
    # (review round 3: an all-zero gate made this vacuous).
    noisy = np.zeros(n)
    noisy[:600] = 1.0
    noisy[:100] = 1_000_000.0
    noisy_gate = tail_concentration_gate({"n": noisy}, {"n": weights})
    assert not noisy_gate.passed
    noise_failures, noise_receipts = _stage_attributable_concentration_failures(
        {"n": {"top_share": 0.84, "carriers": 900, "distinct_values": 900}},
        {
            "n": {
                "top_share": 0.84 + _WORSENING_SHARE_TOLERANCE / 2,
                "carriers": 900,
                "distinct_values": 900,
            }
        },
        noisy_gate,
    )
    assert noise_failures == []
    assert noise_receipts["n"]["over_threshold"] is True
    assert noise_receipts["n"]["stage_worsened_share"] is False


def test_undeclared_candidate_overlap_fails_loud() -> None:
    """microcosm#570 hardening: a donor/candidate column collision outside
    the declared partition (joint vector = donor-owned; donor tax-unit
    outputs = recipient-owned) must raise instead of silently replacing
    donor payload."""
    import microcosm.build.us_runtime.puf_capital_gains_tail as mod

    frame = _expanded_recipient_frame()
    donor = _donor()
    tail, _ = mod.select_puf_capital_gains_tail_donors(donor)
    weights = tail["weight"].to_numpy(dtype=np.float64)
    candidates = mod._recipient_candidates(
        frame,
        maximum_transfer_weight=float(weights.max()),
        seed=7,
    )
    poisoned = candidates.copy()
    poisoned["filing_status_code"] = 1.0  # collides with the donor's column
    with pytest.raises(ValueError, match="undeclared donor/candidate"):
        mod._assign_tail_donors(
            tail,
            assigned_weights=weights,
            candidates=poisoned,
        )


def test_donor_key_bijection_is_asserted(monkeypatch) -> None:
    """microcosm#570 hardening: every selected donor must be consumed exactly
    once; a dropped assignment fails at construction."""
    import microcosm.build.us_runtime.puf_capital_gains_tail as mod

    frame = _expanded_recipient_frame()
    donor = _donor()
    real_assign = mod._assign_tail_donors

    def dropping_assign(*args, **kwargs):
        return real_assign(*args, **kwargs).iloc[:-1]

    monkeypatch.setattr(mod, "_assign_tail_donors", dropping_assign)
    with pytest.raises(ValueError, match="bijection"):
        mod.transfer_puf_capital_gains_tail(frame, donor, seed=7)
