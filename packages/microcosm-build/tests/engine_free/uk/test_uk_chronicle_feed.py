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
    assert pin.source_commit == "80a3482905843d447f8811b500a56dbf3b5f47fa"
    assert pin.source_repo == "PolicyEngine/chronicle"
    assert pin.fact_row_count == 2030062
    assert pin.facts_sha256 == (
        "cb3342c3f59ff961a0ae369ca489293ddd6d021fdeb5cfafb89f6d125329bdfe"
    )
    assert pin.manifest_sha256 == (
        "d2323e7bddb30043fafecc2249fca0ee1506dae8ff20bc5f4db8575dc2ec1b73"
    )
    assert pin.artifact_schema_version == "policyengine_ledger.consumer_artifact.v2"
    assert pin.consumer_fact_schema_versions == ("chronicle.consumer_fact.v5",)
    assert pin.consumer_fact_schema_sha256 == (
        "acd33a091cd8d192be5d264e3e94c13e00afb6fd1972e57d4576ac55a54f6970"
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


@pytest.mark.parametrize(
    "versions",
    [
        ["chronicle.consumer_fact.v6"],
        ["chronicle.consumer_fact.v4", "chronicle.consumer_fact.v6"],
    ],
    ids=["undeclared_only", "declared_and_undeclared"],
)
def test_national_feed_rejects_an_undeclared_row_contract(
    monkeypatch, tmp_path, versions
):
    """A pin naming a row contract chronicle_epoch does not declare is refused."""
    from microcosm.build.uk_runtime import chronicle_feed

    raw = json.loads(chronicle_feed._feed_path().read_text())
    raw["consumer_fact_schema_versions"] = versions
    path = tmp_path / "pin.json"
    path.write_text(json.dumps(raw))
    monkeypatch.setattr(chronicle_feed, "_feed_path", lambda: path)

    with pytest.raises(ValueError) as refusal:
        chronicle_feed.load_uk_chronicle_feed()

    message = str(refusal.value)
    assert "['chronicle.consumer_fact.v6']" in message
    assert "microcosm.build.chronicle_epoch" in message


@pytest.mark.parametrize(
    ("tool", "check"),
    [
        ("tools/vendor_uk_ledger_facts.py", "require_uk_chronicle_feed_row_contracts("),
        ("tools/certify_uk_release_cut.py", "require_uk_chronicle_feed_row_contracts("),
        (
            "tools/generate_uk_target_references.py",
            "require_uk_chronicle_feed_subset_row_contracts(",
        ),
        (
            "tools/generate_uk_local_target_references.py",
            "require_uk_chronicle_feed_subset_row_contracts(",
        ),
        (
            "tools/build_uk_ledger_compile_parity_signed_differences.py",
            "require_uk_chronicle_feed_subset_row_contracts(",
        ),
        (
            "tools/build_uk_desnz_constituency_energy_facts.py",
            "require_uk_chronicle_feed_subset_row_contracts(",
        ),
    ],
)
def test_uk_feed_tools_check_the_row_contracts_of_the_rows_they_read(tool, check):
    """Each UK tool that reads the pinned feed's rows holds their row contracts.

    The full-artifact tools hold the rows to the pin's list when the facts are
    the committed feed; the tools that read a subset of it, or only some of
    its rows, require each row's contract to be one the pin lists. Both refuse
    a row contract ``microcosm.build.chronicle_epoch`` does not declare.
    """
    from test_support.paths import paths_for

    source = (paths_for("microcosm-build").repository / tool).read_text()

    assert check in source


@pytest.mark.parametrize(
    ("rows", "pinned", "refused"),
    [
        (
            ("chronicle.consumer_fact.v5",),
            ("chronicle.consumer_fact.v4",),
            True,
        ),
        (
            ("chronicle.consumer_fact.v4", "chronicle.consumer_fact.v5"),
            ("chronicle.consumer_fact.v4",),
            True,
        ),
        (
            ("chronicle.consumer_fact.v4",),
            ("chronicle.consumer_fact.v4", "chronicle.consumer_fact.v5"),
            False,
        ),
        ((), ("chronicle.consumer_fact.v4",), False),
    ],
    ids=["declared_not_on_the_list", "one_not_on_the_list", "strict_subset", "none"],
)
def test_subset_check_requires_each_row_contract_on_the_pin(rows, pinned, refused):
    """A subset of the pinned feed may carry fewer row contracts, never others."""
    from microcosm.build.ledger_artifact import ChronicleFeedPinMismatchError
    from microcosm.build.uk_runtime.chronicle_feed import (
        require_uk_chronicle_feed_subset_row_contracts,
    )

    if refused:
        with pytest.raises(ChronicleFeedPinMismatchError):
            require_uk_chronicle_feed_subset_row_contracts(rows, pinned_versions=pinned)
    else:
        require_uk_chronicle_feed_subset_row_contracts(rows, pinned_versions=pinned)


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
