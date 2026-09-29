"""The CGT asset-type stage: residential flag, main asset type, receipts."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime import cgt_asset_type
from microcosm.build.uk_runtime.cgt_asset_type import (
    CGT_ASSET_TYPE_COLUMN,
    CGT_ASSET_TYPE_DOMAIN,
    CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES,
    CGT_ASSET_TYPE_NONE,
    CGT_ASSET_TYPE_RESIDENTIAL,
    CGT_ASSET_TYPE_SUB_AEA,
    CGT_RESIDENTIAL_GAINS_COLUMN,
    HMRC_CGT_ASSET_TYPE_RECORD_SETS,
    HMRC_CGT_ASSET_TYPE_RESOURCE,
    UK_CGT_ASSET_TYPE_MASS_CONSERVATION_REASON,
    UK_CGT_ASSET_TYPE_STAGE_NAME,
    HMRCCGTAssetTypeFacts,
    assign_uk_cgt_asset_types,
    cgt_asset_type_operation_parameters,
    fit_type_weights,
    load_hmrc_cgt_asset_type_facts,
    solve_residential_logistic,
    uk_cgt_asset_type_stage_transform,
)
from microcosm.build.uk_runtime.cgt_imputation import (
    UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS,
    UKCGTPolicyParameters,
)
from microcosm.build.uk_runtime.content_identity import uk_frame_content_identity
from microcosm.build.uk_runtime.ledger_fact_vendoring import load_vendored_resource
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import Frame

PARAMETERS = UKCGTPolicyParameters(
    personal_allowance=12_570.0,
    personal_allowance_taper_threshold=100_000.0,
    personal_allowance_taper_rate=0.5,
    annual_exempt_amount=3_000.0,
    instant="2024-06-01",
    source="test",
)


def _frame(gains, *, weights=None, time_period: str = "2024") -> Frame:
    rows = len(gains)
    person = pd.DataFrame(
        {
            "person_id": np.arange(rows, dtype="int64"),
            "person_household_id": np.arange(rows, dtype="int64"),
            "person_benunit_id": np.arange(rows, dtype="int64"),
            "capital_gains": np.asarray(gains, dtype=float),
            "age": np.full(rows, 45, dtype="int64"),
        }
    )
    for column in UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS:
        person[column] = 0.0
    household = pd.DataFrame(
        {
            "household_id": np.arange(rows, dtype="int64"),
            "household_weight": (
                np.full(rows, 60.0)
                if weights is None
                else np.asarray(weights, dtype=float)
            ),
            "region": np.full(rows, "LONDON", dtype=object),
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.arange(rows, dtype="int64")})
    return uk_national_frame(
        person=person, benunit=benunit, household=household, time_period=time_period
    )


def _synthetic_gains(rows: int = 20_000, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gains = np.where(rng.random(rows) < 0.8, np.exp(rng.normal(10.5, 1.8, rows)), 0.0)
    gains[:50] = -1_000.0
    return gains


class TestVendoredFacts:
    def test_committed_resource_types_and_restates_table_8a(self) -> None:
        facts = load_hmrc_cgt_asset_type_facts()

        assert facts.table8a_taxpayers_total == 205_000.0
        assert facts.table8a_gains_total == 12_914_000_000.0
        assert facts.table8b_individuals_taxpayers == 171_000.0
        assert facts.table8b_all_taxpayers == 173_000.0
        assert facts.residential_taxpayers_individuals_basis == pytest.approx(
            205_000.0 * 171_000.0 / 173_000.0
        )
        assert facts.residential_gains_individuals_basis == pytest.approx(
            12_914_000_000.0 * 10_337_000_000.0 / 10_904_000_000.0
        )
        assert {row.asset_type for row in facts.table7_types} == {
            CGT_ASSET_TYPE_RESIDENTIAL,
            *CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES,
        }
        shares = facts.non_residential_gains_shares()
        assert sum(shares.values()) == pytest.approx(1.0)
        assert shares["unlisted_shares"] > 0.5
        assert facts.table7_type("listed_shares").mean_gain_per_disposal < 10_000.0
        assert facts.table7_type("unlisted_shares").mean_gain_per_disposal > 100_000.0

    def test_every_declared_record_set_is_present(self) -> None:
        payload = load_vendored_resource(HMRC_CGT_ASSET_TYPE_RESOURCE)
        record_sets = {row["layout"]["record_set_id"] for row in payload["rows"]}

        assert record_sets == set(HMRC_CGT_ASSET_TYPE_RECORD_SETS)
        assert payload["consumers"] == ["uk_runtime.cgt_asset_type"]

    def test_refuses_a_resource_from_another_feed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = copy.deepcopy(load_vendored_resource(HMRC_CGT_ASSET_TYPE_RESOURCE))
        payload["source_fact_feed"]["facts_sha256"] = "0" * 64
        monkeypatch.setattr(
            cgt_asset_type, "load_vendored_resource", lambda _name: payload
        )

        with pytest.raises(ValueError, match="differs from the committed UK pin"):
            load_hmrc_cgt_asset_type_facts()

    def test_refuses_rows_from_another_workbook(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = copy.deepcopy(load_vendored_resource(HMRC_CGT_ASSET_TYPE_RESOURCE))
        for row in payload["rows"]:
            if row["layout"]["record_set_id"].endswith("table8a.ty2024"):
                row["source"]["source_sha256"] = "f" * 64
        monkeypatch.setattr(
            cgt_asset_type, "load_vendored_resource", lambda _name: payload
        )

        with pytest.raises(ValueError, match="other than the pinned"):
            load_hmrc_cgt_asset_type_facts()

    def test_refuses_a_drifted_asset_type_roster(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = copy.deepcopy(load_vendored_resource(HMRC_CGT_ASSET_TYPE_RESOURCE))
        for row in payload["rows"]:
            if row["dimensions"].get("cgt_asset_type") == "listed_shares":
                row["dimensions"]["cgt_asset_type"] = "quoted_shares"
        monkeypatch.setattr(
            cgt_asset_type, "load_vendored_resource", lambda _name: payload
        )

        with pytest.raises(ValueError, match="asset-type roster drifted"):
            load_hmrc_cgt_asset_type_facts()


class TestResidentialSolve:
    def test_meets_both_targets_in_expectation(self) -> None:
        gains = np.exp(np.random.default_rng(3).normal(10.5, 1.8, 5_000))
        weights = np.full(gains.size, 60.0)
        count_target = 0.35 * weights.sum()
        gains_target = 0.12 * (weights * gains).sum()

        a, b, centre = solve_residential_logistic(
            gains, weights, count_target=count_target, gains_target=gains_target
        )
        probabilities = 1.0 / (1.0 + np.exp(-(a + b * (np.log(gains) - centre))))

        assert (weights * probabilities).sum() == pytest.approx(count_target, rel=1e-6)
        assert (weights * gains * probabilities).sum() == pytest.approx(
            gains_target, rel=1e-6
        )
        # Residential gainers are small relative to the whole: negative slope.
        assert b < 0

    def test_refuses_unattainable_targets(self) -> None:
        gains = np.exp(np.random.default_rng(3).normal(10.5, 1.0, 500))
        weights = np.full(gains.size, 10.0)

        with pytest.raises(ValueError, match="not below the liable taxpayer mass"):
            solve_residential_logistic(
                gains, weights, count_target=weights.sum(), gains_target=1.0
            )
        with pytest.raises(ValueError, match="not below the liable gains mass"):
            solve_residential_logistic(
                gains,
                weights,
                count_target=1.0,
                gains_target=(weights * gains).sum(),
            )
        # A mean far above what the largest gains can carry at this count.
        with pytest.raises(ValueError, match="outside the range"):
            solve_residential_logistic(
                gains,
                weights,
                count_target=0.5 * weights.sum(),
                gains_target=0.999 * (weights * gains).sum(),
            )


class TestTypeWeights:
    def test_reproduces_the_share_targets(self) -> None:
        facts = load_hmrc_cgt_asset_type_facts()
        gains = np.exp(np.random.default_rng(5).normal(11.0, 1.5, 4_000))
        weights = np.full(gains.size, 30.0)
        medians = {
            asset_type: facts.table7_type(asset_type).mean_gain_per_disposal
            for asset_type in CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES
        }
        targets = facts.non_residential_gains_shares()

        type_weights, probabilities, iterations = fit_type_weights(
            gains, weights, medians=medians, share_targets=targets
        )

        assert probabilities.shape == (gains.size, 5)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        achieved = ((weights * gains)[:, None] * probabilities).sum(axis=0) / (
            weights * gains
        ).sum()
        for index, asset_type in enumerate(CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES):
            assert achieved[index] == pytest.approx(targets[asset_type], abs=1e-5)
        assert type_weights.sum() == pytest.approx(1.0)
        assert iterations < 100


class TestAssignment:
    def test_flags_and_types_every_liable_gainer_and_nobody_else(self) -> None:
        facts = load_hmrc_cgt_asset_type_facts()
        gains = _synthetic_gains()
        frame = _frame(gains)

        result, summary = assign_uk_cgt_asset_types(frame, facts, PARAMETERS)

        person = result.table("person")
        types = person[CGT_ASSET_TYPE_COLUMN].to_numpy()
        residential = person[CGT_RESIDENTIAL_GAINS_COLUMN].to_numpy()
        liable = gains > PARAMETERS.annual_exempt_amount
        assert set(types) <= set(CGT_ASSET_TYPE_DOMAIN)
        assert (types[gains <= 0] == CGT_ASSET_TYPE_NONE).all()
        assert (types[(gains > 0) & ~liable] == CGT_ASSET_TYPE_SUB_AEA).all()
        assert not np.isin(
            types[liable], [CGT_ASSET_TYPE_NONE, CGT_ASSET_TYPE_SUB_AEA]
        ).any()
        flagged = types == CGT_ASSET_TYPE_RESIDENTIAL
        assert (residential[flagged] == gains[flagged]).all()
        assert (residential[~flagged] == 0.0).all()
        # Untouched columns and weights, one conservation receipt.
        assert (person["capital_gains"].to_numpy() == gains).all()
        assert result.mass_log[-1].reason == UK_CGT_ASSET_TYPE_MASS_CONSERVATION_REASON
        assert result.mass_log[-1].old_total == result.mass_log[-1].new_total

        evidence = summary.evidence()
        assert evidence["stage"] == UK_CGT_ASSET_TYPE_STAGE_NAME
        residential_receipt = evidence["residential"]
        assert residential_receipt["expected_count"] == pytest.approx(
            facts.residential_taxpayers_individuals_basis, rel=1e-6
        )
        assert residential_receipt["expected_gains"] == pytest.approx(
            facts.residential_gains_individuals_basis, rel=1e-6
        )
        # Systematic sampling lands within one person of the expected count.
        assert (
            abs(
                residential_receipt["achieved_count"]
                - residential_receipt["expected_count"]
            )
            <= 60.0
        )
        assert residential_receipt["achieved_gains"] == pytest.approx(
            residential_receipt["expected_gains"], rel=0.05
        )
        assert residential_receipt["logistic_slope"] < 0
        shares = evidence["asset_type"]["achieved_gains_share"]
        assert sum(shares.values()) == pytest.approx(1.0)
        assert shares["unlisted_shares"] > 0.4
        assert evidence["value_counts"][CGT_ASSET_TYPE_RESIDENTIAL] == int(
            flagged.sum()
        )
        assert len(evidence["composition_by_band"]) == 10
        assert evidence["facts"]["resource_sha256"] == facts.resource_sha256
        assert evidence["seeds"] == {"residential_flag": 553, "asset_type": 554}

    def test_is_deterministic_and_seed_sensitive(self) -> None:
        facts = load_hmrc_cgt_asset_type_facts()
        frame = _frame(_synthetic_gains())

        first, _ = assign_uk_cgt_asset_types(frame, facts, PARAMETERS)
        second, _ = assign_uk_cgt_asset_types(frame, facts, PARAMETERS)
        other, _ = assign_uk_cgt_asset_types(
            frame, facts, PARAMETERS, asset_type_seed=999
        )

        assert uk_frame_content_identity(first) == uk_frame_content_identity(second)
        assert uk_frame_content_identity(first) != uk_frame_content_identity(other)

    def test_refuses_a_frame_that_cannot_hold_the_targets(self) -> None:
        facts = load_hmrc_cgt_asset_type_facts()
        frame = _frame([10_000.0, 20_000.0, 50_000.0], weights=[1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match="not below the liable taxpayer mass"):
            assign_uk_cgt_asset_types(frame, facts, PARAMETERS)

    def test_refuses_a_frame_already_classified(self) -> None:
        facts = load_hmrc_cgt_asset_type_facts()
        frame = _frame(_synthetic_gains())
        classified, _ = assign_uk_cgt_asset_types(frame, facts, PARAMETERS)

        with pytest.raises(ValueError, match="already carries"):
            assign_uk_cgt_asset_types(classified, facts, PARAMETERS)


class TestStageContract:
    def test_manifest_operations_restate_the_reviewed_parameters(self) -> None:
        spec = load_country_spec("uk")
        assert spec.sources is not None
        stage = spec.sources.stage_map()[UK_CGT_ASSET_TYPE_STAGE_NAME]
        expected = cgt_asset_type_operation_parameters()

        assert [operation.kind for operation in stage.operations] == list(expected)
        for operation in stage.operations:
            assert dict(operation.parameters) == expected[operation.kind]
        assert tuple(stage.outputs) == (
            CGT_ASSET_TYPE_COLUMN,
            CGT_RESIDENTIAL_GAINS_COLUMN,
        )
        roles = {artifact["role"]: artifact for artifact in stage.artifacts}
        assert roles["cgt_asset_type_facts"]["resource"] == HMRC_CGT_ASSET_TYPE_RESOURCE
        assert roles["cgt_asset_type_facts"]["runtime_sha256_required"] is True

    def test_transform_refuses_a_drifted_declaration(self) -> None:
        from dataclasses import replace

        spec = load_country_spec("uk")
        assert spec.sources is not None
        stage = spec.sources.stage_map()[UK_CGT_ASSET_TYPE_STAGE_NAME]
        drifted = replace(stage, operations=tuple(reversed(stage.operations)))

        with pytest.raises(ValueError, match="operation order drifted"):
            uk_cgt_asset_type_stage_transform(drifted)

    def test_transform_runs_from_the_resource_and_the_seam(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = load_country_spec("uk")
        assert spec.sources is not None
        stage = spec.sources.stage_map()[UK_CGT_ASSET_TYPE_STAGE_NAME]
        facts = load_hmrc_cgt_asset_type_facts()
        frame = _frame(_synthetic_gains(6_000))
        resolved: list[str] = []

        def load_facts():
            resolved.append("facts")
            return facts

        def load_parameters(period):
            assert period == "2024"
            resolved.append("parameters")
            return PARAMETERS

        monkeypatch.setattr(
            cgt_asset_type, "load_hmrc_cgt_asset_type_facts", load_facts
        )
        monkeypatch.setattr(cgt_asset_type, "uk_cgt_policy_parameters", load_parameters)
        from_resource = uk_cgt_asset_type_stage_transform(stage)
        result_a = from_resource(frame)
        assert resolved == ["facts", "parameters"]

        seam = uk_cgt_asset_type_stage_transform(
            stage, facts=facts, parameters=PARAMETERS
        )
        result_b = seam(frame)

        assert uk_frame_content_identity(result_a) == uk_frame_content_identity(
            result_b
        )
        assert from_resource.checkpoint_metadata() == seam.checkpoint_metadata()
        assert from_resource.checkpoint_metadata()["evidence"]["stage"] == (
            UK_CGT_ASSET_TYPE_STAGE_NAME
        )


def test_synthetic_facts_type_check_without_the_feed() -> None:
    """A synthetic facts object drives the stage without the vendored rows."""
    from microcosm.build.uk_runtime.cgt_asset_type import HMRCCGTTable7Type

    rows = tuple(
        HMRCCGTTable7Type(
            asset_type=name, category="financial", disposals=d, proceeds=3 * g, gains=g
        )
        for name, d, g in (
            ("listed_shares", 1_000_000.0, 5.0e9),
            ("unlisted_shares", 300_000.0, 33.0e9),
            ("other_financial_assets", 1_000_000.0, 16.0e9),
            ("agricultural_commercial_industrial_land_buildings", 13_000.0, 2.0e9),
            (CGT_ASSET_TYPE_RESIDENTIAL, 170_000.0, 10.0e9),
            ("other_non_financial_assets", 20_000.0, 3.0e9),
        )
    )
    facts = HMRCCGTAssetTypeFacts(
        table8a_taxpayers_total=2_000.0,
        table8a_gains_total=60_000_000.0,
        table8a_disposals_total=2_200.0,
        table8a_tax_total=12_000_000.0,
        table8b_individuals_taxpayers=95.0,
        table8b_individuals_gains=95.0,
        table8b_all_taxpayers=100.0,
        table8b_all_gains=100.0,
        table7_types=rows,
        table7_total_gains=sum(row.gains for row in rows),
        table7_total_disposals=sum(row.disposals for row in rows),
        resource="synthetic.json",
        resource_sha256="synthetic",
        source_commit="synthetic",
    )
    frame = _frame(_synthetic_gains(3_000), weights=np.full(3_000, 5.0))

    _, summary = assign_uk_cgt_asset_types(frame, facts, PARAMETERS)

    assert summary.evidence()["residential"]["count_target_individuals_basis"] == (
        pytest.approx(1_900.0)
    )
