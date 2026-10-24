from __future__ import annotations

import hashlib
import json
from importlib.resources import files

import pytest


def test_national_feed_records_the_complete_merged_source_artifact():
    from microcosm.build.uk_runtime.chronicle_feed import (
        load_uk_chronicle_feed,
    )

    pin = load_uk_chronicle_feed()
    resource = files("microcosm.build.uk").joinpath("chronicle_feed.json")
    raw = resource.read_bytes()
    assert pin.source_commit == "200bf6d559cf85bc9c2c5ee4df8c50ba0f5d09d1"
    assert pin.source_repo == "PolicyEngine/chronicle"
    assert pin.fact_row_count == 1502291
    assert pin.facts_sha256 == (
        "b6e69c90ef0d65674d8c9ce6285113aa5ce36e6d25f6f662d106bdd5c2c27dbc"
    )
    assert pin.manifest_sha256 == (
        "7d9c75fdcf9b638848979fa51eff5e51d4caabbef263ed6c2a8a0b9a3305b460"
    )
    assert pin.artifact_schema_version == "policyengine_ledger.consumer_artifact.v2"
    assert pin.consumer_fact_schema_versions == ("chronicle.consumer_fact.v4",)
    assert pin.consumer_fact_schema_sha256 == (
        "d953306d6f01fbc64db4372d0262d3012ee590269f8a25b276b18dca0626c6a0"
    )
    assert pin.resource_sha256 == hashlib.sha256(raw).hexdigest()
    assert pin.resource_size_bytes == len(raw)
    assert pin.to_dict()["source_commit"] == pin.source_commit


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("facts_sha256", "not-a-digest"),
        ("manifest_sha256", "A" * 64),
        ("consumer_fact_schema_sha256", "a" * 63),
        ("source_commit", "ec7169b5"),
        ("fact_row_count", True),
        ("country", "us"),
    ],
)
def test_national_feed_rejects_malformed_identity(
    monkeypatch, tmp_path, field, bad_value
):
    from microcosm.build.uk_runtime import chronicle_feed

    raw = json.loads(chronicle_feed._feed_path().read_text())
    raw[field] = bad_value
    path = tmp_path / "pin.json"
    path.write_text(json.dumps(raw))
    monkeypatch.setattr(chronicle_feed, "_feed_path", lambda: path)

    with pytest.raises(ValueError, match=field):
        chronicle_feed.load_uk_chronicle_feed()


def test_committed_feed_pin_check_refuses_a_foreign_artifact():
    from microcosm.build.uk_runtime.chronicle_feed import (
        UKChronicleFeedPinError,
        load_uk_chronicle_feed,
        require_committed_uk_chronicle_feed_pin,
    )

    pin = load_uk_chronicle_feed()
    assert (
        require_committed_uk_chronicle_feed_pin(
            pin.facts_sha256,
            manifest_sha256=pin.manifest_sha256,
            allow_unpinned_feed=False,
            pin=pin,
        )
        is pin
    )
    with pytest.raises(UKChronicleFeedPinError, match="facts: loaded 0") as excinfo:
        require_committed_uk_chronicle_feed_pin(
            "0" * 64,
            manifest_sha256=pin.manifest_sha256,
            allow_unpinned_feed=False,
            pin=pin,
        )
    assert "manifest:" not in str(excinfo.value)
    with pytest.raises(UKChronicleFeedPinError, match="manifest: loaded None"):
        require_committed_uk_chronicle_feed_pin(
            pin.facts_sha256, manifest_sha256=None, allow_unpinned_feed=False, pin=pin
        )
    # A reviewed diagnostic run may override; the caller records the override.
    assert (
        require_committed_uk_chronicle_feed_pin(
            "0" * 64, manifest_sha256=None, allow_unpinned_feed=True, pin=pin
        )
        is pin
    )


def test_national_and_local_census_read_the_one_chronicle_pin():
    """The national and local surfaces share uk/chronicle_feed.json (#890)."""

    from microcosm.build.uk_runtime.chronicle_feed import (
        load_uk_chronicle_feed,
        require_committed_uk_chronicle_feed_pin,
    )
    from microcosm.build.uk_runtime.local_target_census import _LEDGER_FACT_FEED_PIN

    pin = load_uk_chronicle_feed()
    require_committed_uk_chronicle_feed_pin(
        pin.facts_sha256, manifest_sha256=pin.manifest_sha256, allow_unpinned_feed=False
    )
    assert _LEDGER_FACT_FEED_PIN["facts_sha256"] == pin.facts_sha256
    assert _LEDGER_FACT_FEED_PIN["manifest_sha256"] == pin.manifest_sha256
    assert _LEDGER_FACT_FEED_PIN["source_commit"] == pin.source_commit
    assert _LEDGER_FACT_FEED_PIN["fact_row_count"] == pin.fact_row_count
