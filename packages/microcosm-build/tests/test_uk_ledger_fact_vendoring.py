"""Vendored Chronicle facts are pinned copies, regenerable from the feed."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from importlib import resources as importlib_resources
from pathlib import Path

import pytest

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.uk_runtime.chronicle_feed import (
    load_uk_chronicle_feed,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import (
    VENDOR_SELECTIONS_RESOURCE,
    VENDORED_RESOURCE_KIND,
    feed_identity,
    load_vendor_selections,
    load_vendored_resource,
    rows_matching,
    validate_vendor_selections,
    vendor_resource,
    vendor_row,
)

ROOT = Path(__file__).resolve().parents[3]
STABLE_UK_FACT_FEED_NAME = ".codex-work/consumer_facts_uk.jsonl"


def _package_dir() -> Path:
    return Path(
        str(
            importlib_resources.files("microcosm.build.uk").joinpath(
                "country_package.json"
            )
        )
    ).parent


def _country_package_paths() -> set[str]:
    package = json.loads((_package_dir() / "country_package.json").read_text())
    return {row["path"] for row in package["resources"]}


def _pinned_feed(tmp_path: Path):
    configured = os.environ.get("CHRONICLE_UK_FACTS")
    feed = Path(configured) if configured else ROOT / STABLE_UK_FACT_FEED_NAME
    if not feed.exists():
        pytest.skip("pinned UK Chronicle national consumer feed is not present")
    if feed.is_dir():
        artifact_path = feed
    else:
        manifest = (
            ROOT / ".codex-work/consumer_facts_uk_manifest.json"
            if not configured
            else feed.with_name("manifest.json")
        )
        if not manifest.is_file():
            pytest.skip("pinned UK Chronicle consumer manifest is not present")
        artifact_path = tmp_path / "consumer-artifact"
        artifact_path.mkdir()
        (artifact_path / "consumer_facts.jsonl").symlink_to(feed.resolve())
        (artifact_path / "manifest.json").symlink_to(manifest.resolve())
    pin = load_uk_chronicle_feed()
    return artifact_path, load_ledger_consumer_artifact(
        artifact_path,
        expected_facts_sha256=pin.facts_sha256,
        expected_manifest_sha256=pin.manifest_sha256,
    )


def test_selection_register_is_declared_and_well_formed() -> None:
    register = load_vendor_selections()
    assert VENDOR_SELECTIONS_RESOURCE in _country_package_paths()
    names = [entry["resource"] for entry in register["resources"]]
    assert len(names) == len(set(names))
    for entry in register["resources"]:
        assert entry["resource"] in _country_package_paths(), entry["resource"]


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema_version": 2},
        {"kind": "something_else"},
        {"resources": []},
    ],
)
def test_selection_register_refuses_bad_shapes(mutation) -> None:
    register = json.loads(json.dumps(load_vendor_selections()))
    register.update(mutation)
    with pytest.raises(ValueError):
        validate_vendor_selections(register)


def test_register_consumers_are_modules_that_read_the_resource_today() -> None:
    """`consumers` is a statement of fact, `planned_consumers` a statement of intent.

    Every present consumer must be an importable microcosm.build module whose
    source names the resource file; every resource without a present reader
    must name at least one planned reader (the #890 spine half).
    """

    import importlib
    import inspect

    register = load_vendor_selections()
    present = 0
    for entry in register["resources"]:
        assert entry["consumers"] or entry["planned_consumers"], entry["resource"]
        for consumer in entry["consumers"]:
            module = importlib.import_module(f"microcosm.build.{consumer}")
            assert entry["resource"] in inspect.getsource(module), (
                entry["resource"],
                consumer,
            )
            present += 1
    assert present >= 1


def test_selection_register_refuses_unknown_selector_fields() -> None:
    register = json.loads(json.dumps(load_vendor_selections()))
    register["resources"][0]["selections"][0]["selector"]["not_a_field"] = "x"
    with pytest.raises(ValueError, match="unsupported selector field"):
        validate_vendor_selections(register)


def test_committed_vendored_resources_carry_the_national_pin_and_declared_counts() -> (
    None
):
    register = load_vendor_selections()
    pin = load_uk_chronicle_feed()
    for entry in register["resources"]:
        payload = load_vendored_resource(entry["resource"])
        assert payload["kind"] == VENDORED_RESOURCE_KIND
        assert payload["source_fact_feed"] == feed_identity(pin)
        assert payload["consumers"] == entry["consumers"]
        assert payload["planned_consumers"] == entry["planned_consumers"]
        by_label = {row["label"]: row for row in payload["selections"]}
        assert set(by_label) == {row["label"] for row in entry["selections"]}
        for selection in entry["selections"]:
            assert (
                by_label[selection["label"]]["row_count"]
                == selection["expected_row_count"]
            )
            assert by_label[selection["label"]]["selector"] == selection["selector"]
        keys = [row["aggregate_fact_key"] for row in payload["rows"]]
        assert len(keys) == len(set(keys))
        assert payload["row_count"] == len(keys)
        assert all(isinstance(row["value"], (int, float)) for row in payload["rows"])


def test_vendor_row_copies_keys_provenance_and_value_verbatim() -> None:
    fact = {
        "aggregate_fact_key": "ledger.aggregate_fact.v2:abc",
        "semantic_fact_key": "ledger.semantic_fact.v2:def",
        "aggregation": {"method": "sum"},
        "dimensions": {"b": 2, "a": "x"},
        "entity": {"name": "household", "role": "r"},
        "geography": {"id": "E92000001", "level": "country", "vintage": "current"},
        "layout": {
            "record_set_id": "rs",
            "groupby_dimension": "g",
            "groupby_value_id": "v",
            "table_record_kind": "total",
        },
        "lineage": {"source_record_id": "src.rec"},
        "observed_measure": {
            "source_concept": "x.y",
            "source_measure_id": "m",
            "unit": "gbp",
        },
        "period": {"type": "fiscal_year", "value": 2024},
        "period_coverage": {
            "start_date": "2024-04-01",
            "end_date": "2025-03-31",
            "basis": "fiscal",
        },
        "source": {
            "source_name": "x",
            "source_table": "t",
            "source_file": "f.ods",
            "source_sha256": "0" * 64,
            "vintage": "v1",
            "raw_r2_key": "raw/x/x-pkg/2025/0/f.ods",
        },
        "value": 12.5,
    }
    row = vendor_row(fact)
    assert row["aggregate_fact_key"] == "ledger_aggregate_fact_v2_abc"
    assert row["semantic_fact_key"] == "ledger_semantic_fact_v2_def"
    assert row["value"] == 12.5
    assert row["concept"] == "x.y"
    assert list(row["dimensions"]) == ["a", "b"]
    assert row["source"]["source_sha256"] == "0" * 64
    assert row["period"] == {"type": "fiscal_year", "value": 2024}
    with pytest.raises(ValueError, match="non-numeric"):
        vendor_row({**fact, "value": "12"})


def test_vendor_resource_refuses_count_drift() -> None:
    fact = {
        "aggregate_fact_key": "ledger.aggregate_fact.v2:one",
        "semantic_fact_key": "ledger.semantic_fact.v2:one",
        "observed_measure": {"source_concept": "x.y", "source_name": "x"},
        "geography": {"id": "K02000001", "level": "country"},
        "period": {"type": "calendar_year", "value": 2024},
        "layout": {},
        "value": 1.0,
    }
    entry = {
        "resource": "probe.json",
        "purpose": "p",
        "consumers": [],
        "planned_consumers": ["c"],
        "selections": [
            {
                "label": "s",
                "selector": {"source_concept": "x.y"},
                "expected_row_count": 2,
            }
        ],
    }
    pin = load_uk_chronicle_feed()
    with pytest.raises(ValueError, match="matched 1 facts, expected 2"):
        vendor_resource(entry, [fact], pin=pin)
    relaxed = vendor_resource(entry, [fact], pin=pin, strict_counts=False)
    assert relaxed.row_count == 1
    assert relaxed.payload["selections"][0]["row_count"] == 1


def test_rows_matching_filters_by_alias_and_membership() -> None:
    payload = load_vendored_resource("dft_bus_value_anchors.json")
    london = rows_matching(
        payload,
        concept="dft.local_bus_passenger_fare_receipts",
        geography_id="E12000007",
        period_value=2025,
    )
    assert len(london) == 1
    assert london[0]["value"] == pytest.approx(1_347_434_943.01459)
    areas = rows_matching(
        payload,
        concept="dft.local_bus_total_estimated_net_support",
        period_value=[2025],
    )
    assert {row["geography"]["id"] for row in areas} == {
        "E92000001",
        "E12000007",
        "dft:england_outside_london",
    }


def test_committed_vendored_resources_regenerate_from_pinned_feed(
    tmp_path: Path,
) -> None:
    artifact_path, artifact = _pinned_feed(tmp_path)
    generated = tmp_path / "generated"
    generated.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "vendor_uk_ledger_facts.py"),
            "--ledger-facts",
            str(artifact_path),
            "--ledger-facts-sha256",
            artifact.facts_sha256,
            "--output-dir",
            str(generated),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "wrote" in completed.stdout
    for entry in load_vendor_selections()["resources"]:
        committed = (_package_dir() / entry["resource"]).read_bytes()
        regenerated = (generated / entry["resource"]).read_bytes()
        assert (
            hashlib.sha256(committed).hexdigest()
            == hashlib.sha256(regenerated).hexdigest()
        ), entry["resource"]
