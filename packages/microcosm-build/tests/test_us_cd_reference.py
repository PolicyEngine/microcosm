"""Public CD reference inventory preserves source uncertainty and geography."""

import csv
import hashlib
import io
import json
import zipfile

import pytest

from microcosm.build.us_runtime.cd_reference import (
    acs_table_inventory,
    build_bundle,
    classify_acs_value,
    crosswalk_alignment,
    irs_csv_inventory,
    xlsx_disclosure_inventory,
)
from microcosm.build.us_runtime.cd_reference_sources import census_requests


@pytest.mark.parametrize(
    ("raw", "annotation", "expected"),
    [
        ("0", None, "numeric"),
        (None, None, "missing"),
        ("-666666666", None, "insufficient_sample"),
        ("-999999999", "N", "insufficient_geographic_sample"),
        ("-888888888", "(X)", "not_applicable_or_available"),
        ("-222222222", "**", "moe_insufficient_sample"),
        ("-333333333", "***", "moe_open_interval"),
        ("-555555555", "*****", "controlled_estimate"),
        ("250000", "250,000+", "open_interval"),
        ("12", "unknown", "unknown_annotation"),
    ],
)
def test_annotation_classes_keep_original_values(raw, annotation, expected):
    result = classify_acs_value(raw, annotation)
    assert result["raw"] == raw
    assert result["annotation"] == annotation
    assert result["class"] == expected
    if expected != "numeric":
        assert result["numeric_value"] is None


def test_annotation_takes_precedence_without_silencing_conflict():
    result = classify_acs_value("-666666666", "(X)")
    assert result["class"] == "not_applicable_or_available"
    assert result["numeric_value"] is None
    assert result["annotation_conflict"] is True


def _acs(table="B19001"):
    variable = f"{table}_001" if table != "S0101" else "S0101_C01_001"
    metadata = {"variables": {}}
    for suffix in ("E", "M", "EA", "MA"):
        metadata["variables"][variable + suffix] = {
            "label": "Total",
            "concept": "Household income" if table == "B19001" else "Age and sex",
            "predicateType": "string" if suffix.endswith("A") else "int",
        }
    rows = [
        [
            "NAME",
            variable + "E",
            variable + "M",
            variable + "EA",
            variable + "MA",
            "state",
            "congressional district",
        ]
    ]
    for state, district in (("01", "01"), ("11", "98"), ("72", "98")):
        rows.append(
            [
                f"District (119th Congress), {state}",
                "100",
                "12",
                None,
                None,
                state,
                district,
            ]
        )
    return metadata, rows


def test_acs_keeps_dc_pr_exclusion_moe_and_same_source_holdout():
    metadata, rows = _acs()
    result = acs_table_inventory("B19001", metadata, rows)
    assert result["geography_ids"] == ["5001900US0101", "5001900US1100"]
    assert result["excluded_geographies"][0]["state_fips"] == "72"
    assert result["excluded_geographies"][0]["reason"] == "outside_us_50_states_dc"
    assert result["reservation"]["status"] == "reserved_same_source_holdout"
    assert result["reservation"]["covers_derived_versions"] is True
    assert result["reservation"]["source_independent"] is False
    assert result["cells"][0]["moe"]["raw"] == "12"
    assert result["confidence_level"] == 0.9
    assert "variance" not in result["cells"][0]


def test_age_tables_keep_distinct_grouping_contracts():
    subject = acs_table_inventory("S0101", *_acs("S0101"))
    detailed = acs_table_inventory("B01001", *_acs("B01001"))
    assert subject["grouping"] != detailed["grouping"]
    assert subject["reservation"]["status"] == "inventory_only"


def test_acs_refuses_duplicate_geography_and_missing_annotation_column():
    metadata, rows = _acs()
    with pytest.raises(ValueError, match="duplicate geography"):
        acs_table_inventory("B19001", metadata, [*rows, rows[1]])
    incomplete = [
        [value for index, value in enumerate(row) if index != 4] for row in rows
    ]
    with pytest.raises(ValueError, match="missing.*MA"):
        acs_table_inventory("B19001", metadata, incomplete)


def test_target_mixture_uses_incoming_pair_population_not_outgoing_weights():
    rows = [
        {
            "source_geography_id": "5001700US0101",
            "target_geography_id": "5001900US0101",
            "pair_population": "80",
            "weight": "0.8",
        },
        {
            "source_geography_id": "5001700US0101",
            "target_geography_id": "5001900US0102",
            "pair_population": "20",
            "weight": "0.2",
        },
        {
            "source_geography_id": "5001700US0102",
            "target_geography_id": "5001900US0101",
            "pair_population": "20",
            "weight": "1",
        },
    ]
    result = crosswalk_alignment(rows)
    first = result["targets"][0]
    assert first["target_geography_id"] == "5001900US0101"
    assert first["max_incoming_share"] == pytest.approx(0.8)
    assert first["non_dominant_incoming_share"] == pytest.approx(0.2)
    assert first["known_allocation_error"] is None
    assert sum(
        item["incoming_share"] for item in first["contributions"]
    ) == pytest.approx(1)
    assert result["targets"][1]["max_incoming_share"] == 1
    rows[0]["pair_population"] = "-1"
    with pytest.raises(ValueError, match="positive"):
        crosswalk_alignment(rows)


def test_irs_preserves_national_state_and_at_large_proxy_distinctions():
    payload = b"STATEFIPS,STATE,CONG_DISTRICT,agi_stub,N1,A00100\n00,US,00,0,100,200\n01,AL,00,0,50,100\n01,AL,01,0,20,40\n11,DC,00,0,5,10\n"
    result = irs_csv_inventory(payload)
    scopes = [row["published_geography_level"] for row in result["rows"]]
    assert scopes == ["national", "state", "congressional_district", "state"]
    assert result["rows"][1]["at_large_proxy_geography_id"] is None
    assert result["rows"][3]["at_large_proxy_geography_id"] == "5001700US1100"
    assert (
        result["uncertainty"]["variance_status"]
        == "not_published_needs_method_resolution"
    )
    assert result["disclosure"]["numeric_zero_proves_unsuppressed"] is False
    assert result["units"]["rounding_increment"] is None


def test_irs_duplicate_source_row_is_refused():
    payload = b"STATEFIPS,STATE,CONG_DISTRICT,agi_stub,N1\n11,DC,00,0,5\n11,DC,00,0,6\n"
    with pytest.raises(ValueError, match="duplicate"):
        irs_csv_inventory(payload)


def _workbook():
    output = io.BytesIO()
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            "xl/sharedStrings.xml",
            f"<sst {ns}><si><t>**</t></si><si><t>** - Not shown</t></si></sst>",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet {ns}><sheetData><row><c r="B2" t="s"><v>0</v></c><c r="B4" t="s"><v>1</v></c></row></sheetData></worksheet>',
        )
    return output.getvalue()


def test_xlsx_suppression_markers_remain_cell_located_not_joined_to_csv():
    result = xlsx_disclosure_inventory(_workbook())
    assert result["csv_cell_mapping"] == "not_performed"
    assert [(row["cell"], row["class"]) for row in result["markers"]] == [
        ("B2", "suppression_or_combination_marker"),
        ("B4", "disclosure_note"),
    ]


def _bundle_inputs(tmp_path):
    authority = tmp_path / "authority"
    authority.mkdir()
    blobs = {
        "22incd.csv": b"STATEFIPS,STATE,CONG_DISTRICT,agi_stub,N1\n01,AL,01,0,100\n11,DC,00,0,10\n",
        "22incdall.xlsx": _workbook(),
    }
    index = []
    for name, payload in blobs.items():
        (authority / name).write_bytes(payload)
        index.append(
            {
                "filename": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "url": "https://www.irs.gov/pub/irs-soi/" + name,
            }
        )
    (authority / "source-index.json").write_text(json.dumps(index))
    guide = io.BytesIO()
    with zipfile.ZipFile(guide, "w") as archive:
        archive.writestr(
            "word/document.xml", "<document><p>Population and sample</p></document>"
        )
    (authority / "22incddocguide.docx").write_bytes(guide.getvalue())
    (authority / "22incddocguide.sha256").write_text(
        hashlib.sha256(guide.getvalue()).hexdigest()
    )
    crosswalk = tmp_path / "crosswalk.csv"
    with crosswalk.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["source_geography_id", "target_geography_id", "pair_population", "weight"]
        )
        writer.writerow(["5001700US0101", "5001900US0101", "100", "1"])
        writer.writerow(["5001700US1100", "5001900US1100", "10", "1"])
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "crosswalk_sha256": hashlib.sha256(crosswalk.read_bytes()).hexdigest(),
                "source_geography_vintage": "117th_congress",
                "target_geography_vintage": "119th_congress",
                "block_vintage": "2020_tabulation_blocks",
            }
        )
    )
    calls = []

    def fetch(url, _key):
        request = next(item for item in census_requests() if item["url"] == url)
        calls.append(url)
        metadata, rows = _acs(request["table"])
        return json.dumps(metadata if request["kind"] == "metadata" else rows).encode()

    return authority, crosswalk, provenance, calls, fetch


def test_bundle_replays_pinned_sources_and_retains_separate_lineage(tmp_path):
    authority, crosswalk, provenance, calls, fetch = _bundle_inputs(tmp_path)
    output = tmp_path / "bundle"
    first = build_bundle(output, authority, crosswalk, provenance, fetch=fetch)
    before = (output / "reference-inventory.json").read_bytes()
    second = build_bundle(output, authority, crosswalk, provenance, offline=True)
    assert len(calls) == 8
    assert first == second
    assert before == (output / "reference-inventory.json").read_bytes()
    assert first["candidate_evaluation"] == "not_performed"
    assert first["target_activation"] is False
    assert first["reconciliation"]["irs_missing_crosswalk_source_ids"] == []
    assert first["lineage_products"]["crosswalked_values"] == "not_generated"
    assert first["lineage_products"]["state_rescaled_values"] == "not_generated"
    for table in ("B19001", "B25003"):
        assert first["acs"][table]["reservation"]["targets_allowed"] is False
        assert first["acs"][table]["reservation"]["tuning_allowed"] is False


@pytest.mark.parametrize(
    "corruption", ["irs_digest", "crosswalk_digest", "crosswalk_vintage"]
)
def test_bundle_refuses_corrupt_source_identity(tmp_path, corruption):
    authority, crosswalk, provenance, _calls, fetch = _bundle_inputs(tmp_path)
    if corruption == "irs_digest":
        (authority / "22incd.csv").write_bytes(b"changed")
    elif corruption == "crosswalk_digest":
        crosswalk.write_bytes(b"changed")
    else:
        metadata = json.loads(provenance.read_text())
        metadata["target_geography_vintage"] = "118th_congress"
        provenance.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="digest|vintage"):
        build_bundle(tmp_path / "bundle", authority, crosswalk, provenance, fetch=fetch)
    assert not (tmp_path / "bundle" / "reference-inventory.json").exists()
