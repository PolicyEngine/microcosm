"""The HMRC Table 3 joint distribution, typed from the vendored feed rows."""

from __future__ import annotations

import copy

import pytest

from microcosm.build.uk_runtime import hmrc_capital_gains
from microcosm.build.uk_runtime.hmrc_capital_gains import (
    HMRC_CGT_BUILD_PERIOD,
    HMRC_CGT_CONDITIONING_RECORD_SETS,
    HMRC_CGT_CONDITIONING_RESOURCE,
    HMRC_CGT_GAIN_BAND_LOWER_BOUNDS,
    HMRC_CGT_INCOME_BAND_LOWER_BOUNDS,
    HMRC_CGT_JOINT_RECORD_SET_PREFIX,
    HMRC_CGT_JOINT_SOURCE_FILE,
    HMRC_CGT_JOINT_SOURCE_SHA256,
    HMRC_CGT_SOURCE_VINTAGE,
    HMRC_CGT_TOTAL_GAINS_GBP,
    HMRC_CGT_TOTAL_INDIVIDUALS,
    load_hmrc_cgt_conditioning_facts,
    load_hmrc_cgt_joint_distribution,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import load_vendored_resource


def _payload() -> dict:
    return copy.deepcopy(load_vendored_resource(HMRC_CGT_CONDITIONING_RESOURCE))


def _joint_rows(payload: dict) -> list[dict]:
    return [
        row
        for row in payload["rows"]
        if row["layout"]["record_set_id"].startswith(HMRC_CGT_JOINT_RECORD_SET_PREFIX)
    ]


def _load_from(monkeypatch: pytest.MonkeyPatch, payload: dict):
    monkeypatch.setattr(
        hmrc_capital_gains, "load_vendored_resource", lambda _name: payload
    )
    return load_hmrc_cgt_joint_distribution()


class TestCommittedResource:
    def test_types_every_published_cell(self) -> None:
        distribution = load_hmrc_cgt_joint_distribution()

        assert len(distribution.cells) == len(HMRC_CGT_GAIN_BAND_LOWER_BOUNDS) * len(
            HMRC_CGT_INCOME_BAND_LOWER_BOUNDS
        )
        assert {cell.gain_lower_bound for cell in distribution.cells} == set(
            HMRC_CGT_GAIN_BAND_LOWER_BOUNDS
        )
        assert {cell.income_lower_bound for cell in distribution.cells} == set(
            HMRC_CGT_INCOME_BAND_LOWER_BOUNDS
        )
        assert [total.gain_lower_bound for total in distribution.band_totals] == list(
            HMRC_CGT_GAIN_BAND_LOWER_BOUNDS
        )
        assert [
            total.income_lower_bound for total in distribution.income_totals
        ] == list(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS)

    def test_values_are_people_and_pounds(self) -> None:
        """The feed carries counts in people and amounts in pounds."""
        distribution = load_hmrc_cgt_joint_distribution()

        cell = distribution.cell(gain_lower_bound=0, income_lower_bound=0)
        assert cell.individuals == 71_000.0
        assert cell.gains == 414_000_000.0
        assert distribution.total_individuals == HMRC_CGT_TOTAL_INDIVIDUALS
        assert distribution.total_gains == HMRC_CGT_TOTAL_GAINS_GBP

    def test_suppressed_cells_are_none_not_zero(self) -> None:
        """A count too small to publish is absent from the feed, not zero."""
        distribution = load_hmrc_cgt_joint_distribution()

        top = distribution.cell(gain_lower_bound=5_000_000, income_lower_bound=0)
        assert top.individuals_suppressed
        assert not top.gains_suppressed
        assert top.gains == 2_088_000_000.0
        assert distribution.band_total(5_000_000).individuals == 3_000.0
        assert distribution.unpublished_gains >= 0.0

    def test_provenance_names_the_resource_and_the_publisher_workbook(self) -> None:
        distribution = load_hmrc_cgt_joint_distribution()
        conditioning = load_hmrc_cgt_conditioning_facts()

        source = distribution.source
        assert source.resource == HMRC_CGT_CONDITIONING_RESOURCE
        assert source.resource_sha256 == conditioning.resource_sha256
        assert source.source_commit == conditioning.source_commit
        assert source.record_set_prefix == HMRC_CGT_JOINT_RECORD_SET_PREFIX
        assert source.source_file == HMRC_CGT_JOINT_SOURCE_FILE
        assert source.source_sha256 == HMRC_CGT_JOINT_SOURCE_SHA256
        assert source.source_vintage == HMRC_CGT_SOURCE_VINTAGE
        assert source.build_period == HMRC_CGT_BUILD_PERIOD

    def test_joint_and_conditioning_rows_describe_one_universe(self) -> None:
        """Table 3's all-gains row is the Table 1 individuals observation."""
        distribution = load_hmrc_cgt_joint_distribution()
        conditioning = load_hmrc_cgt_conditioning_facts()

        assert (
            distribution.total_individuals == conditioning.table1.individuals_taxpayers
        )
        assert distribution.total_gains == conditioning.table1.individuals_gains
        folded = conditioning.size_bands_aggregated(HMRC_CGT_GAIN_BAND_LOWER_BOUNDS)
        for total in distribution.band_totals:
            people, gains = folded[total.gain_lower_bound]
            assert total.individuals == people
            assert abs(total.gains - gains) <= 1_000_000.0

    def test_every_record_set_the_manifests_declare_is_present(self) -> None:
        payload = _payload()
        record_sets = {row["layout"]["record_set_id"] for row in payload["rows"]}
        for declared in HMRC_CGT_CONDITIONING_RECORD_SETS:
            if declared.endswith("."):
                assert any(name.startswith(declared) for name in record_sets), declared
            else:
                assert declared in record_sets


class TestRefusals:
    def test_refuses_a_resource_from_another_feed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _payload()
        payload["source_fact_feed"]["facts_sha256"] = "0" * 64

        with pytest.raises(ValueError, match="differs from the committed UK pin"):
            _load_from(monkeypatch, payload)

    def test_refuses_rows_from_another_publisher_workbook(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _payload()
        for row in _joint_rows(payload):
            row["source"]["source_sha256"] = "f" * 64

        with pytest.raises(ValueError, match="not the pinned"):
            _load_from(monkeypatch, payload)

    def test_refuses_a_drifted_income_band_roster(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _payload()
        for row in _joint_rows(payload):
            band = row["dimensions"].get("cgt_taxable_income_band")
            if band == "income_125140_to_199999":
                row["dimensions"]["cgt_taxable_income_band"] = "income_150000_to_199999"
                row["measure_id"] = row["measure_id"].replace("125140", "150000")

        with pytest.raises(ValueError, match="income-band roster drifted"):
            _load_from(monkeypatch, payload)

    def test_refuses_a_missing_gain_band(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = _payload()
        payload["rows"] = [
            row
            for row in payload["rows"]
            if row["layout"]["record_set_id"]
            != f"{HMRC_CGT_JOINT_RECORD_SET_PREFIX}gain_25000_to_49999"
        ]
        payload["row_count"] = len(payload["rows"])

        with pytest.raises(ValueError, match="gain-band roster drifted"):
            _load_from(monkeypatch, payload)

    def test_refuses_a_drifted_published_total(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _payload()
        for row in _joint_rows(payload):
            if row["measure_id"] == "taxpayers_all_incomes" and not row["dimensions"]:
                row["value"] = 552_000

        with pytest.raises(ValueError, match="published totals drifted"):
            _load_from(monkeypatch, payload)

    def test_refuses_cells_summing_past_the_published_total(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _payload()
        for row in _joint_rows(payload):
            if row["measure_id"] == "gains_income_0_to_37699" and row["dimensions"].get(
                "cgt_gain_band"
            ):
                row["value"] = row["value"] * 10

        with pytest.raises(ValueError, match="above the published total"):
            _load_from(monkeypatch, payload)

    def test_refuses_a_measure_that_disagrees_with_its_band(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _payload()
        row = next(
            row
            for row in _joint_rows(payload)
            if row["measure_id"] == "taxpayers_income_0_to_37699"
            and row["dimensions"].get("cgt_gain_band") == "gain_0_to_9999"
        )
        row["measure_id"] = "taxpayers_income_37700_to_49999"

        with pytest.raises(ValueError, match="disagrees with its income band"):
            _load_from(monkeypatch, payload)


def test_conditioning_loader_refuses_a_size_band_that_disagrees_with_table_3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload()
    for row in payload["rows"]:
        if (
            row["layout"]["record_set_id"].startswith(
                "hmrc.cgt_size_of_gain_2026.table2_1a.ty2024."
            )
            and row["dimensions"].get("cgt_gain_band") == "gain_25000_to_49999"
            and row["measure_id"] == "taxpayers_individuals"
        ):
            row["value"] = row["value"] + 20_000
    monkeypatch.setattr(
        hmrc_capital_gains, "load_vendored_resource", lambda _name: payload
    )

    with pytest.raises(ValueError, match="disagree on the band from 25000"):
        load_hmrc_cgt_conditioning_facts()


def test_diagnostic_read_can_skip_the_feed_identity_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload()
    payload["source_fact_feed"]["source_commit"] = "0" * 40
    monkeypatch.setattr(
        hmrc_capital_gains, "load_vendored_resource", lambda _name: payload
    )

    distribution = load_hmrc_cgt_joint_distribution(verify_feed_identity=False)

    assert distribution.source.source_commit == "0" * 40
