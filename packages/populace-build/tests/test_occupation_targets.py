from __future__ import annotations

import pytest

from populace.build.uk_runtime import (
    aps_age_band_employment_targets,
    aps_occupation_employment_targets,
    ashe_occupation_employment_targets,
)

ASHE_FIXTURE = """soc_code,soc_title,employment_jobs,median_annual_pay
1111,Chief executives and senior officials,133000,89835
1112,Elected officers and representatives,,
2211,Generalist medical practitioners,110000,76512
221,Health professionals aggregate,505000,52000
2211,Generalist medical practitioners,999,1
6135,Care workers and home carers,714000,23456
"""

APS_OCC_FIXTURE = """soc_code,soc_title,employment,period,geography
11,Corporate Managers and Directors,2619900,Jan 2025-Dec 2025,United Kingdom
12,Other Managers and Proprietors,1189900,Jan 2025-Dec 2025,United Kingdom
92,Elementary Administration and Service Occupations,not_a_number,Jan 2025-Dec 2025,United Kingdom
"""

APS_AGE_FIXTURE = """age_band,employment,period,geography
16+,33321600,Jan 2025-Dec 2025,United Kingdom
16-19,910600,Jan 2025-Dec 2025,United Kingdom
65+,1590500,Jan 2025-Dec 2025,United Kingdom
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


class TestAsheOccupationEmploymentTargets:
    def test_builds_targets_matching_fixture_values(self, tmp_path):
        build = ashe_occupation_employment_targets(
            _write(tmp_path, "ashe.csv", ASHE_FIXTURE)
        )
        by_name = {target.name: target for target in build.target_set}
        assert len(build.target_set) == 3
        assert by_name["ons/ashe14/occupation/1111/employment_jobs"].value == 133000
        assert by_name["ons/ashe14/occupation/2211/employment_jobs"].value == 110000
        assert by_name["ons/ashe14/occupation/6135/employment_jobs"].value == 714000
        for target in build.target_set:
            assert target.entity == "person"
            assert target.period == 2025
            assert target.measure.startswith("employee_jobs/occupation/")
            assert "ASHE" in target.source
            assert "2025 provisional" in target.source

    def test_bad_rows_are_skipped_and_reported_not_silently_dropped(self, tmp_path):
        build = ashe_occupation_employment_targets(
            _write(tmp_path, "ashe.csv", ASHE_FIXTURE)
        )
        reasons = {entry.row_number: entry.reason for entry in build.skipped}
        # Row 2: suppressed (blank) job count; row 4: 3-digit aggregate code;
        # row 5: duplicate unit group.
        assert set(reasons) == {2, 4, 5}
        assert "employment_jobs" in reasons[2]
        assert "four-digit" in reasons[4]
        assert "duplicate" in reasons[5]
        # The offending rows travel with the report.
        skipped_codes = {entry.row["soc_code"] for entry in build.skipped}
        assert skipped_codes == {"1112", "221", "2211"}

    def test_missing_required_column_raises(self, tmp_path):
        path = _write(
            tmp_path, "bad.csv", "soc_code,soc_title\n1111,Chief executives\n"
        )
        with pytest.raises(ValueError, match="employment_jobs"):
            ashe_occupation_employment_targets(path)

    def test_all_rows_skipped_raises_rather_than_empty_set(self, tmp_path):
        path = _write(
            tmp_path,
            "empty.csv",
            "soc_code,soc_title,employment_jobs,median_annual_pay\n"
            "1111,Chief executives,,\n",
        )
        with pytest.raises(ValueError, match="every row was skipped"):
            ashe_occupation_employment_targets(path)


class TestApsOccupationEmploymentTargets:
    def test_builds_targets_matching_fixture_values(self, tmp_path):
        build = aps_occupation_employment_targets(
            _write(tmp_path, "aps_occ.csv", APS_OCC_FIXTURE)
        )
        by_name = {target.name: target for target in build.target_set}
        assert len(build.target_set) == 2
        target = by_name["ons/aps/occupation_submajor/11/employment"]
        assert target.value == 2619900
        assert target.entity == "person"
        assert target.measure == "employment/occupation_submajor/11"
        assert target.period == "Jan 2025-Dec 2025"
        assert "NM_17_1" in target.source
        assert "United Kingdom" in target.source
        assert by_name["ons/aps/occupation_submajor/12/employment"].value == 1189900

    def test_bad_employment_is_skipped_and_reported(self, tmp_path):
        build = aps_occupation_employment_targets(
            _write(tmp_path, "aps_occ.csv", APS_OCC_FIXTURE)
        )
        assert len(build.skipped) == 1
        (entry,) = build.skipped
        assert entry.row_number == 3
        assert "not_a_number" in entry.reason
        assert entry.row["soc_code"] == "92"


class TestApsAgeBandEmploymentTargets:
    def test_total_band_is_excluded_by_default_and_reported(self, tmp_path):
        build = aps_age_band_employment_targets(
            _write(tmp_path, "aps_age.csv", APS_AGE_FIXTURE)
        )
        names = [target.name for target in build.target_set]
        assert names == [
            "ons/aps/employment/age/16_19",
            "ons/aps/employment/age/65plus",
        ]
        by_name = {target.name: target for target in build.target_set}
        assert by_name["ons/aps/employment/age/16_19"].value == 910600
        assert by_name["ons/aps/employment/age/65plus"].value == 1590500
        assert by_name["ons/aps/employment/age/65plus"].measure == (
            "employment/age/65plus"
        )
        (entry,) = build.skipped
        assert entry.row["age_band"] == "16+"
        assert "roll-up" in entry.reason

    def test_include_total_keeps_the_roll_up(self, tmp_path):
        build = aps_age_band_employment_targets(
            _write(tmp_path, "aps_age.csv", APS_AGE_FIXTURE),
            include_total=True,
        )
        by_name = {target.name: target for target in build.target_set}
        assert by_name["ons/aps/employment/age/16plus"].value == 33321600
        assert build.skipped == ()
