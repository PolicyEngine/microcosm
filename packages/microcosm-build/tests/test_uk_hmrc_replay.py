"""Contract tests for aggregate-only UK HMRC replay reporting."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from microcosm.build.uk_runtime.hmrc_income import (
    HMRC_SPI_BUILD_PERIOD,
    HMRC_SPI_INCOME_BAND_LOWER_BOUNDS,
    HMRC_SPI_INCOME_COMPONENTS,
    HMRC_SPI_TARGET_RECORD_COUNT,
    HMRCIncomeBandTargetRecord,
    HMRCIncomeSourceProvenance,
    HMRCIncomeTargetSet,
    materialize_hmrc_spi_income_band_targets,
)
from microcosm.build.uk_runtime.hmrc_replay import (
    CANONICAL_HMRC_FACT_FENCES,
    FULL_FRS_TI_BAND_FENCE_ID,
    HMRCReplayDiagnosticAggregate,
    HMRCReplayFact,
    HMRCReplayReport,
    build_conservative_hmrc_replay_report,
    classify_hmrc_replay_targets,
    write_hmrc_replay_report,
)

_PINNED_ODS_PATH = (
    Path(__file__).resolve().parents[3]
    / "inputs"
    / "hmrc"
    / "Collated_Tables_3_1_to_3_11_2324.ods"
)
_EXPECTED_FENCE_IDS = {
    "frs_epb_source_absent",
    "frs_exps_source_absent",
    "frs_taxterm_source_absent",
    "frs_mothinc_source_absent",
    "frs_otherinc_source_absent",
    "frs_ossben_identifiable_subset",
    "frs_srp_regular_code5_subset",
    FULL_FRS_TI_BAND_FENCE_ID,
}


def _complete_source_targets(tmp_path: Path) -> HMRCIncomeTargetSet:
    source = HMRCIncomeSourceProvenance(
        local_path=tmp_path / "hmrc.ods",
        sha256="a" * 64,
        publication_url="https://example.test/publication",
        ods_url="https://example.test/hmrc.ods",
        source_vintage="2023-24",
        source_tax_year="2023-24",
        source_tax_year_start=2023,
        build_period=HMRC_SPI_BUILD_PERIOD,
        table_names=("Table_3_6", "Table_3_7"),
        size_bytes=166_693,
        mime_type="application/vnd.oasis.opendocument.spreadsheet",
    )
    targets: list[HMRCIncomeBandTargetRecord] = []
    upper_bounds = (*HMRC_SPI_INCOME_BAND_LOWER_BOUNDS[1:], None)
    index = 0
    for lower_bound, upper_bound in zip(
        HMRC_SPI_INCOME_BAND_LOWER_BOUNDS,
        upper_bounds,
        strict=True,
    ):
        for component in HMRC_SPI_INCOME_COMPONENTS:
            for measure, unit in (("count", "people"), ("amount", "GBP")):
                index += 1
                targets.append(
                    HMRCIncomeBandTargetRecord(
                        name=(
                            f"hmrc/{component}_{measure}_income_band_"
                            f"{lower_bound}_to_{upper_bound or 'inf'}"
                        ),
                        component=component,
                        measure=measure,
                        unit=unit,
                        value=float(index * 1_000),
                        period=HMRC_SPI_BUILD_PERIOD,
                        total_income_lower_bound=lower_bound,
                        total_income_upper_bound=upper_bound,
                    )
                )
    return HMRCIncomeTargetSet(source=source, targets=tuple(targets))


def _evidence() -> dict[str, dict[str, object]]:
    return {
        "source_evidence": {
            "base_candidate": {"sha256": "b" * 64, "size_bytes": 1_315_880_118},
            "spi_donor": {"sha256": "c" * 64, "size_bytes": 141_323_762},
            "hmrc_surface": {"sha256": "d" * 64, "size_bytes": 166_693},
        },
        "build_evidence": {
            "period": "2023",
            "seed": 42,
            "spi_prior_mass_share": 0.5,
        },
        "qrf_evidence": {
            "stage1": {"fit_weight_kind": "design", "sample_size": 100_000},
            "post_draw_identity": {"maximum_absolute_error_gbp": 0.0},
        },
        "effective_mass_evidence": {
            "minimum_nondefault_mass_share": 0.000001,
            "columns": {
                "gift_aid": {"share": 0.001, "passed": True},
                "charitable_investment_gifts": {
                    "share": 0.000002,
                    "passed": True,
                },
            },
        },
    }


def _report(tmp_path: Path, **kwargs) -> HMRCReplayReport:
    evidence = _evidence()
    evidence.update(kwargs)
    return build_conservative_hmrc_replay_report(
        _complete_source_targets(tmp_path),
        **evidence,
    )


def test_canonical_fences_preserve_all_constituent_findings() -> None:
    fences = {fence.fence_id: fence for fence in CANONICAL_HMRC_FACT_FENCES}

    assert set(fences) == _EXPECTED_FENCE_IDS
    assert set(fences[FULL_FRS_TI_BAND_FENCE_ID].dependent_fence_ids) == (
        _EXPECTED_FENCE_IDS - {FULL_FRS_TI_BAND_FENCE_ID}
    )
    assert "receipt flags" in fences["frs_epb_source_absent"].finding
    assert "5.1302528%" in fences["frs_exps_source_absent"].mass_implication
    assert "gross redundancy pay" in fences["frs_taxterm_source_absent"].finding
    assert "1.4650566%" in fences["frs_otherinc_source_absent"].mass_implication
    assert "identifiable subset" in fences["frs_ossben_identifiable_subset"].finding
    assert (
        "regular code-5 State Pension"
        in fences["frs_srp_regular_code5_subset"].mass_implication
    )


def test_complete_surface_is_0_exact_0_directional_208_fenced(tmp_path) -> None:
    report = _report(tmp_path)

    assert len(report.facts) == HMRC_SPI_TARGET_RECORD_COUNT == 208
    assert {fact.classification for fact in report.facts} == {"excluded"}
    assert {fact.outcome for fact in report.facts} == {"excluded_with_fence"}
    assert {fact.fence_ids for fact in report.facts} == {(FULL_FRS_TI_BAND_FENCE_ID,)}
    assert {fact.blocked_dependencies for fact in report.facts} == {
        ("hmrc_spi_assessable_income",)
    }
    assert all(
        fact.estimate is None
        and fact.delta is None
        and fact.relative_delta is None
        and fact.operator is None
        for fact in report.facts
    )
    assert report.summary == {
        "status": "reviewed_exclusions_only",
        "total_facts": 208,
        "exact_pass": 0,
        "exact_fail": 0,
        "directional_pass": 0,
        "directional_fail": 0,
        "excluded_with_fence": 208,
        "comparison_coverage_count": 0,
        "comparison_coverage_share": 0.0,
        "release_blocking_comparison_failures": 0,
        "all_facts_adjudicated": True,
        "all_exclusions_fenced": True,
    }


def test_payload_is_json_safe_and_contains_only_aggregate_evidence(tmp_path) -> None:
    payload = _report(tmp_path).to_payload()
    rendered = json.dumps(payload, allow_nan=False, sort_keys=True)
    round_trip = json.loads(rendered)

    assert round_trip["report_kind"] == "uk_hmrc_income_208_fact_replay"
    assert round_trip["source_evidence"] == _evidence()["source_evidence"]
    assert len(round_trip["facts"]) == 208
    assert all(row["estimate"] is None for row in round_trip["facts"])
    assert "person_id" not in rendered
    assert "household_id" not in rendered
    assert "benunit_id" not in rendered


def test_optional_diagnostics_are_explicitly_non_comparable(tmp_path) -> None:
    diagnostic = HMRCReplayDiagnosticAggregate(
        name="spi_gift_aid_weighted_total",
        scope="spi_support_channel",
        metric="weighted_total",
        value=123_456.0,
        unit="GBP",
        non_comparability_reason=(
            "SPI support occupies a reviewed prior-mass channel and is not a "
            "published HMRC fact estimate."
        ),
        metadata={"weight_kind": "importance"},
    )
    report = _report(tmp_path, diagnostic_aggregates=(diagnostic,))

    row = report.to_payload()["diagnostic_aggregates"][0]
    assert row["comparable_to_hmrc"] is False
    assert row["scope"] == "spi_support_channel"
    assert row["non_comparability_reason"]


def test_diagnostic_cannot_claim_hmrc_comparability() -> None:
    with pytest.raises(ValueError, match="explicitly non-comparable"):
        HMRCReplayDiagnosticAggregate(
            name="unsafe",
            scope="frs",
            metric="amount",
            value=1.0,
            unit="GBP",
            non_comparability_reason="Known partial measure.",
            comparable_to_hmrc=True,
            metadata={"weight_kind": "importance"},
        )


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        ({"fence_ids": ()}, "require both a fence"),
        ({"blocked_dependencies": ()}, "require both a fence"),
        ({"estimate": 1.0}, "must not carry an estimate"),
        ({"operator": "less_than_or_equal"}, "must not carry an estimate"),
        ({"outcome": "exact_pass"}, "invalid for classification"),
    ),
)
def test_excluded_fact_invariants_fail_closed(
    tmp_path,
    replacement,
    message,
) -> None:
    fact = classify_hmrc_replay_targets(_complete_source_targets(tmp_path))[0]

    with pytest.raises(ValueError, match=message):
        replace(fact, **replacement)


def test_future_exact_and_directional_fact_invariants() -> None:
    exact = HMRCReplayFact(
        target_name="exact",
        component="employment_income",
        measure="amount",
        unit="GBP",
        period="2023",
        total_income_lower_bound=12_570,
        total_income_upper_bound=15_000,
        published_value=100.0,
        classification="exact",
        outcome="exact_pass",
        estimate=95.0,
        delta=-5.0,
        relative_delta=-0.05,
        operator="absolute_relative_error_less_than_or_equal",
        comparison_limit=0.05,
    )
    directional = replace(
        exact,
        target_name="directional",
        classification="directional",
        outcome="directional_pass",
        operator="less_than_or_equal",
        comparison_limit=None,
        fence_ids=("frs_srp_regular_code5_subset",),
        blocked_dependencies=("hmrc_spi_state_pension_income",),
    )

    assert exact.classification == "exact"
    assert directional.classification == "directional"
    with pytest.raises(ValueError, match="delta must equal"):
        replace(exact, delta=-4.0)
    with pytest.raises(ValueError, match="operator='less_than_or_equal'"):
        replace(directional, operator="greater_than_or_equal")
    with pytest.raises(ValueError, match="outcome disagrees"):
        replace(directional, outcome="directional_fail")


def test_incomplete_or_duplicate_source_surface_is_rejected(tmp_path) -> None:
    source = _complete_source_targets(tmp_path)

    with pytest.raises(ValueError, match="complete 208-fact"):
        classify_hmrc_replay_targets(
            HMRCIncomeTargetSet(source=source.source, targets=source.targets[:-1])
        )

    duplicate = replace(source.targets[-1], name=source.targets[0].name)
    with pytest.raises(ValueError, match="names must be unique"):
        classify_hmrc_replay_targets(
            HMRCIncomeTargetSet(
                source=source.source,
                targets=(*source.targets[:-1], duplicate),
            )
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "source_evidence",
            {"person_id": [1, 2]},
            "row-level data",
        ),
        (
            "qrf_evidence",
            {"fits": [{"name": "stage1"}]},
            "sequence of records",
        ),
        (
            "effective_mass_evidence",
            {"share": float("nan")},
            "NaN or Infinity",
        ),
        (
            "build_evidence",
            {},
            "non-empty aggregate evidence mapping",
        ),
    ),
)
def test_evidence_rejects_row_data_and_non_json_values(
    tmp_path,
    field,
    value,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        _report(tmp_path, **{field: value})


def test_report_rejects_missing_canonical_fence(tmp_path) -> None:
    report = _report(tmp_path)

    with pytest.raises(ValueError, match="complete canonical fence set"):
        replace(report, fences=report.fences[:-1])

    drifted = replace(report.fences[0], finding="weakened")
    with pytest.raises(ValueError, match="differs from the canonical"):
        replace(report, fences=(drifted, *report.fences[1:]))


def test_atomic_writer_uses_exact_caller_json_path(tmp_path) -> None:
    report = _report(tmp_path)
    output = tmp_path / "nested" / "hmrc_replay.json"

    written = write_hmrc_replay_report(report, output)
    first = json.loads(written.read_text(encoding="utf-8"))
    written_again = write_hmrc_replay_report(report, output)

    assert written == output.resolve()
    assert written_again == written
    assert first == report.to_payload()
    assert json.loads(written.read_text(encoding="utf-8")) == first
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_conservative_report_accepts_spine_mode_report_kind(tmp_path) -> None:
    report = _report(tmp_path, report_kind="uk_hmrc_spi_income_spine_208_fact_replay")

    assert report.to_payload()["report_kind"] == (
        "uk_hmrc_spi_income_spine_208_fact_replay"
    )


def test_writer_rejects_non_json_and_symbolic_link_paths(tmp_path) -> None:
    report = _report(tmp_path)

    with pytest.raises(ValueError, match="end with '.json'"):
        write_hmrc_replay_report(report, tmp_path / "report.txt")

    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        write_hmrc_replay_report(report, link)


@pytest.mark.skipif(
    not _PINNED_ODS_PATH.is_file(),
    reason="reviewed HMRC ODS is an optional local input",
)
def test_real_pinned_ods_classifies_all_208_without_candidate_estimates() -> None:
    targets = materialize_hmrc_spi_income_band_targets(
        _PINNED_ODS_PATH,
        build_period=HMRC_SPI_BUILD_PERIOD,
    )

    facts = classify_hmrc_replay_targets(targets)

    assert len(facts) == 208
    assert {fact.outcome for fact in facts} == {"excluded_with_fence"}
    assert all(fact.estimate is None and fact.delta is None for fact in facts)
