"""UK firm employees at or below the secondary threshold, from ASHE percentiles."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.firm_pay_distribution import (
    UK_FIRM_PAY_DISTRIBUTION_FILE,
    UK_FIRM_PAY_DISTRIBUTION_SOURCE_COLUMNS,
    assign_uk_firm_pay_below_secondary_threshold,
    read_uk_firm_pay_distribution_source,
    uk_firm_pay_below_threshold_by_section,
)
from microcosm.build.uk_runtime.firm_paybill import read_uk_firm_paybill_source
from test_support.paths import paths_for

PATHS = paths_for("microcosm-build")
FIXTURES = PATHS.tests / "fixtures" / "uk"
SOURCE_DIR = FIXTURES / "firm_pay_distribution_2024"
#: policyengine-uk gov.hmrc.national_insurance.class_1.employer_scale
#: bracket 1 threshold for 2024-25: 175 pounds a week x 52 weeks.
THRESHOLD = 175 * 52


@pytest.fixture(scope="module")
def source():
    return read_uk_firm_pay_distribution_source(SOURCE_DIR)


def test_fixture_matches_provenance_and_paybill_table(source):
    csv_path = SOURCE_DIR / UK_FIRM_PAY_DISTRIBUTION_FILE
    record = json.loads(
        (SOURCE_DIR / (UK_FIRM_PAY_DISTRIBUTION_FILE + ".provenance.json")).read_text()
    )
    assert record["accessed"] == "2026-11-22"
    assert (
        record["files"][UK_FIRM_PAY_DISTRIBUTION_FILE]["sha256"]
        == hashlib.sha256(csv_path.read_bytes()).hexdigest()
        == source.sha256
    )
    ons = record["sources"]["ons_ashe_table_4"]
    paybill_record = json.loads(
        (FIXTURES / "firm_paybill_2024" / ".provenance.json").read_text()
    )
    assert ons["sha256"] == paybill_record["source"]["zip_sha256"]
    assert ons["url"] == paybill_record["source"]["download_url"]
    for key in ("landing_page", "edition", "sha256", "url"):
        assert ons[key]
    assert record["method"]

    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    assert tuple(frame.columns) == UK_FIRM_PAY_DISTRIBUTION_SOURCE_COLUMNS
    paybill = read_uk_firm_paybill_source(FIXTURES / "firm_paybill_2024")
    means = frame.set_index("sic_section")["mean"]
    for section, mean in paybill.section_mean.items():
        assert float(means[section]) == mean
    assert float(means["ALL"]) == paybill.all_employees_mean
    for section, marker in paybill.unpublished.items():
        assert means[section] == marker
    # ONS markers are kept, never written as 0 or blank.
    assert frame.set_index("sic_section").loc["A", "p10"] == "x"
    assert not (frame == "").to_numpy().any()
    assert not source.section_percentiles["U"]


def test_share_and_mean_follow_the_piecewise_line(source):
    table = uk_firm_pay_below_threshold_by_section(
        source, threshold_gbp=THRESHOLD
    ).set_index("sic_section")
    # ALL: p10 = 11412.1 is above 9,100, so the line is 0 -> p10.
    p10 = source.section_percentiles["ALL"][10]
    assert table.loc["ALL", "share_at_or_below"] == pytest.approx(0.1 * THRESHOLD / p10)
    assert table.loc["ALL", "mean_pay_at_or_below_gbp"] == pytest.approx(THRESHOLD / 2)
    # I: p20 = 8180.7 < 9,100 <= p25 = 9610.5.
    curve = source.section_percentiles["I"]
    share = 0.20 + 0.05 * (THRESHOLD - curve[20]) / (curve[25] - curve[20])
    area = (
        0.10 * curve[10] / 2
        + 0.10 * (curve[10] + curve[20]) / 2
        + (share - 0.20) * (curve[20] + THRESHOLD) / 2
    )
    assert table.loc["I", "share_at_or_below"] == pytest.approx(share)
    assert table.loc["I", "mean_pay_at_or_below_gbp"] == pytest.approx(area / share)
    # A has no p10: the line runs 0 -> p20.
    assert table.loc["A", "share_at_or_below"] == pytest.approx(
        0.2 * THRESHOLD / source.section_percentiles["A"][20]
    )
    # U and UNKNOWN take the all-employees row.
    for key in ("U", "UNKNOWN"):
        assert table.loc[key, "level"] == "all_employees"
        assert (
            table.loc[key, "share_at_or_below"] == table.loc["ALL", "share_at_or_below"]
        )
    assert (table["share_at_or_below"].between(0, 1)).all()
    assert (table["mean_pay_at_or_below_gbp"] <= THRESHOLD).all()
    zero = uk_firm_pay_below_threshold_by_section(source, threshold_gbp=0)
    assert (zero["share_at_or_below"] == 0).all()
    with pytest.raises(ValueError, match="above the highest"):
        uk_firm_pay_below_threshold_by_section(source, threshold_gbp=1e6)


def test_assignment_is_integer_deterministic_and_bounded(source):
    rng = np.random.default_rng(0)
    n = 20_000
    sections = rng.choice(["I", "G", "K", "U", "UNKNOWN", "A"], size=n)
    employees = rng.integers(0, 40, size=n)
    table = pd.DataFrame(
        {
            "firm_employees": employees,
            "firm_sic_section": sections,
            "firm_paybill": employees * 3_000.0,
        }
    )
    out = assign_uk_firm_pay_below_secondary_threshold(
        table, source, threshold_gbp=THRESHOLD
    )
    count = out["firm_employees_below_secondary_threshold"]
    paybill = out["firm_paybill_below_secondary_threshold"]
    assert count.dtype == np.int64 and paybill.dtype == np.float64
    assert (count >= 0).all() and (count <= employees).all()
    assert (paybill >= 0).all() and (paybill <= table["firm_paybill"]).all()
    assert (paybill <= count * THRESHOLD).all()
    again = assign_uk_firm_pay_below_secondary_threshold(
        table, source, threshold_gbp=THRESHOLD
    )
    assert np.array_equal(count, again["firm_employees_below_secondary_threshold"])

    shares = (
        uk_firm_pay_below_threshold_by_section(source, threshold_gbp=THRESHOLD)
        .set_index("sic_section")["share_at_or_below"]
        .reindex(sections)
        .to_numpy()
    )
    expected = employees * shares
    assert np.all(np.abs(count - expected) < 1)
    assert count.sum() == pytest.approx(expected.sum(), rel=0.01)

    # Without firm_paybill the paybill is count x section mean below.
    uncapped = assign_uk_firm_pay_below_secondary_threshold(
        table.drop(columns="firm_paybill"), source, threshold_gbp=THRESHOLD
    )
    assert (uncapped["firm_paybill_below_secondary_threshold"] >= paybill).all()

    # Generator rows: employment and sic_code (division 56 is section I).
    firms = pd.DataFrame({"employment": [100, 0], "sic_code": ["00056", "00056"]})
    generated = assign_uk_firm_pay_below_secondary_threshold(
        firms, source, threshold_gbp=THRESHOLD
    )
    share_i = (
        uk_firm_pay_below_threshold_by_section(source, threshold_gbp=THRESHOLD)
        .set_index("sic_section")
        .loc["I", "share_at_or_below"]
    )
    assert (
        abs(generated["firm_employees_below_secondary_threshold"][0] - 100 * share_i)
        < 1
    )
    assert generated["firm_employees_below_secondary_threshold"][1] == 0
    with pytest.raises(ValueError, match="needs firm_employees"):
        assign_uk_firm_pay_below_secondary_threshold(
            pd.DataFrame({"x": [1]}), source, threshold_gbp=THRESHOLD
        )


def test_reader_rejects_bad_tables(tmp_path):
    good = (SOURCE_DIR / UK_FIRM_PAY_DISTRIBUTION_FILE).read_text()
    bad = tmp_path / UK_FIRM_PAY_DISTRIBUTION_FILE
    bad.write_text(good.replace(",p90\n", ",p95\n"))
    with pytest.raises(ValueError, match="columns"):
        read_uk_firm_pay_distribution_source(tmp_path)
    lines = good.splitlines()
    bad.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="once each"):
        read_uk_firm_pay_distribution_source(tmp_path)
