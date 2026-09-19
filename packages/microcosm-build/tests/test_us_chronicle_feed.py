"""US Chronicle feed pin, scope file and feed builder.

``us/chronicle_feed.json`` pins the feed the US target registry compiles
from; ``us/chronicle_feed_scope.json`` names the Chronicle package and build
year for every (record set, period) pair; ``tools/build_us_chronicle_feed.py``
rebuilds the feed from those two. These tests pin the declaration, gate the
parity resources against it, check the scope file's shape, and drive the
builder against a fake Chronicle command.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import textwrap
from importlib.resources import files
from pathlib import Path

import pytest

from microcosm.build.us_runtime import chronicle_feed
from microcosm.build.us_runtime.chronicle_feed import (
    load_us_chronicle_feed,
    us_chronicle_feed_scope_sha256,
)

_ROOT = Path(__file__).resolve().parents[3]
_CHRONICLE_COMMIT = "b571381fcd875393ea0dabc326558cfa2ca8e8fa"
_FACTS_SHA256 = "4d1dba8c1b6274877bf184fa6de5d99b13fc61f34709ccab1487db2b5c64a79f"
_MANIFEST_SHA256 = "38ec5bf1efe5a0bd017ec5279065e2ea7645b37da237197f03ae2fbca28cadac"
_SCHEMA_SHA256 = "bdb51e2a8115634633ba7448c4005930fd9c0bfbade5e1b079b6bc24da485d3d"


def _load_tool(name: str):
    path = _ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _us_resource(name: str) -> dict:
    return json.loads(files("microcosm.build.us").joinpath(name).read_text())


def test_pin_records_the_rebuilt_consumer_artifact() -> None:
    pin = load_us_chronicle_feed()
    raw = files("microcosm.build.us").joinpath("chronicle_feed.json").read_bytes()
    assert pin.source_repo == "PolicyEngine/chronicle"
    assert pin.source_commit == _CHRONICLE_COMMIT
    assert pin.scope == "us_fiscal_targets"
    assert pin.fact_row_count == 39158
    assert pin.facts_sha256 == _FACTS_SHA256
    assert pin.consumer_fact_schema_versions == ("chronicle.consumer_fact.v3",)
    assert pin.consumer_fact_schema_sha256 == _SCHEMA_SHA256
    assert not pin.is_bare_feed
    assert pin.manifest_sha256 == _MANIFEST_SHA256
    assert pin.artifact_schema_version == "policyengine_ledger.consumer_artifact.v2"
    assert pin.scope_sha256 == us_chronicle_feed_scope_sha256()
    assert pin.resource_sha256 == hashlib.sha256(raw).hexdigest()
    assert pin.resource_size_bytes == len(raw)
    assert pin.to_dict()["consumer_fact_schema_versions"] == [
        "chronicle.consumer_fact.v3"
    ]


def test_pin_still_accepts_a_bare_feed_declaration(monkeypatch, tmp_path) -> None:
    raw = json.loads(chronicle_feed._feed_path().read_text())
    raw.update(manifest_sha256=None, artifact_schema_version=None)
    path = tmp_path / "bare-feed.json"
    path.write_text(json.dumps(raw))
    monkeypatch.setattr(chronicle_feed, "_feed_path", lambda: path)
    assert load_us_chronicle_feed().is_bare_feed


def test_real_artifact_matches_the_declared_fact_and_manifest_pins() -> None:
    from microcosm.build.ledger_artifact import load_ledger_consumer_artifact

    artifact_path = _load_tool(
        "build_us_target_parity_manifest"
    ).DEFAULT_FEED_PATH.parent
    if not (artifact_path / "manifest.json").is_file():
        pytest.skip(f"pinned artifact not present at {artifact_path}")
    pin = load_us_chronicle_feed()
    artifact = load_ledger_consumer_artifact(
        artifact_path,
        expected_facts_sha256=pin.facts_sha256,
        expected_manifest_sha256=pin.manifest_sha256,
    )
    assert artifact.fact_row_count == pin.fact_row_count
    assert artifact.schema_version == pin.artifact_schema_version
    assert artifact.fact_schema_versions == pin.consumer_fact_schema_versions
    assert artifact.manifest["consumer_fact_schema_sha256"] == (
        pin.consumer_fact_schema_sha256
    )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("facts_sha256", "not-a-digest", "facts_sha256"),
        ("consumer_fact_schema_sha256", "a" * 63, "consumer_fact_schema_sha256"),
        ("source_commit", "c5e5bf8", "source_commit"),
        ("fact_row_count", True, "fact_row_count"),
        ("country", "uk", "country"),
        ("scope", "uk_calibration", "scope"),
        ("scope_sha256", "0" * 64, "scope_sha256 does not match"),
        ("manifest_sha256", "A" * 64, "manifest_sha256"),
        ("manifest_sha256", None, "both be set"),
        ("artifact_schema_version", None, "both"),
        ("build", "  ", "build"),
        ("consumer_fact_schema_versions", [], "consumer_fact_schema_versions"),
    ],
)
def test_pin_rejects_malformed_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    bad_value: object,
    message: str,
) -> None:
    raw = json.loads(chronicle_feed._feed_path().read_text())
    raw[field] = bad_value
    path = tmp_path / "pin.json"
    path.write_text(json.dumps(raw))
    monkeypatch.setattr(chronicle_feed, "_feed_path", lambda: path)
    with pytest.raises(ValueError, match=message):
        chronicle_feed.load_us_chronicle_feed()


def test_pin_rejects_an_unsupported_artifact_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = json.loads(chronicle_feed._feed_path().read_text())
    raw["manifest_sha256"] = "a" * 64
    raw["artifact_schema_version"] = "not.a.consumer.artifact"
    path = tmp_path / "pin.json"
    path.write_text(json.dumps(raw))
    monkeypatch.setattr(chronicle_feed, "_feed_path", lambda: path)
    with pytest.raises(ValueError, match="artifact_schema_version is unsupported"):
        chronicle_feed.load_us_chronicle_feed()


def test_parity_resources_and_generator_restate_the_pin() -> None:
    pin = load_us_chronicle_feed()
    manifest = _us_resource("target_parity_manifest.json")
    feed_families = _us_resource("target_parity_feed_families.json")
    generator = _load_tool("build_us_target_parity_manifest")
    assert feed_families["feed_sha256"] == pin.facts_sha256
    assert manifest["reference"]["feed_sha256"] == pin.facts_sha256
    assert generator.EXPECTED_FEED_SHA256 == pin.facts_sha256
    assert feed_families["feed"] == generator.DEFAULT_FEED_NAME
    assert manifest["reference"]["feed"] == generator.DEFAULT_FEED_NAME
    assert generator.DEFAULT_FEED_PATH.name == generator.DEFAULT_FEED_NAME
    assert manifest["reference"]["compiled_families"] == "32"
    assert manifest["reference"]["reviewed_exclusions"] == "52"


def test_scope_names_every_pinned_pair_once() -> None:
    builder = _load_tool("build_us_chronicle_feed")
    scope_path = builder.default_scope_path()
    scope = builder.load_scope(scope_path)
    raw = scope["raw"]
    assert scope["commit"] == load_us_chronicle_feed().source_commit
    assert len(scope["pairs"]) == 586 == raw["pair_count"]
    assert sum(len(p) for p in scope["runs"].values()) == 62
    assert set(scope["runs"]) == {
        2020,
        2021,
        2022,
        2023,
        2024,
        2025,
        2026,
        2027,
        2028,
        2029,
    }
    listed = [
        (e["record_set_id"], e["period_type"], e["period_value"]) for e in raw["pairs"]
    ]
    assert listed == sorted(listed), "pairs are written in sorted order"
    assert all(
        package.startswith("packages/") for package, _ in scope["pairs"].values()
    )
    assert {e["period_type"] for e in raw["pairs"]} == {
        "calendar_year",
        "fiscal_year",
        "month",
        "tax_year",
    }
    assert raw["rule"].strip()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda s: s["pairs"].append(dict(s["pairs"][0])), "listed twice"),
        (lambda s: s["runs"].pop(next(iter(s["runs"]))), "runs listing disagrees"),
        (lambda s: s.update(pair_count=1), "pair_count disagrees"),
        (lambda s: s.update(source_commit="abc"), "40-hex"),
        (lambda s: s["pairs"][0].update(package="irs_soi/table_1_4"), "packages/ path"),
    ],
    ids=["duplicate-pair", "runs-mismatch", "pair-count", "commit", "package-path"],
)
def test_load_scope_refuses_an_inconsistent_file(
    tmp_path: Path, mutate, message: str
) -> None:
    builder = _load_tool("build_us_chronicle_feed")
    scope = json.loads(builder.default_scope_path().read_text())
    mutate(scope)
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(scope))
    with pytest.raises(ValueError, match=message):
        builder.load_scope(path)


_FAKE_CHRONICLE = textwrap.dedent(
    '''
    """Stand-in for the chronicle CLI: bundles come from a fixture file."""
    import hashlib, json, os, shutil, sys
    from pathlib import Path

    fixture = json.loads(Path(os.environ["FAKE_CHRONICLE_FIXTURE"]).read_text())
    args = sys.argv[1:]

    def option(name, multi=False):
        values = [args[i + 1] for i, a in enumerate(args) if a == name]
        return values if multi else values[0]

    if args[0] == "build-bundle":
        year = option("--year")
        out = Path(option("--out"))
        if out.exists():
            shutil.rmtree(out)
        entries = []
        for package in option("--source", multi=True):
            target = out / "sources" / package.replace("/", "-")
            target.mkdir(parents=True)
            rows = fixture["bundles"].get(f"{package}@{year}", [])
            facts = target / "consumer_facts.jsonl"
            facts.write_text("".join(json.dumps(r, sort_keys=True) + "\\n" for r in rows))
            entries.append(
                {
                    "source": package,
                    "valid": package not in fixture.get("invalid", []),
                    "outputs": {"consumer_facts": str(facts)},
                }
            )
        (out / "source_packages.json").write_text(
            json.dumps({"year": int(year), "source_packages": entries})
        )
    elif args[0] == "build-consumer-artifact":
        out = Path(option("--out"))
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        payload = Path(option("--facts")).read_bytes()
        (out / "consumer_facts.jsonl").write_bytes(payload)
        (out / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "policyengine_ledger.consumer_artifact.v2",
                    "facts_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        )
    else:
        raise SystemExit(f"unknown command {args[0]}")
    '''
)


def _row(key: str, record_set: str, period_type: str, period_value, value) -> dict:
    return {
        "aggregate_fact_key": key,
        "layout": {"record_set_id": record_set},
        "period": {"type": period_type, "value": period_value},
        "value": value,
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def chronicle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """A clean git checkout, a fake chronicle command and a two-package fixture."""

    root = tmp_path / "chronicle"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    (root / "README").write_text("fixture\n")
    _git(root, "add", "README")
    _git(root, "commit", "-q", "-m", "fixture")
    commit = _git(root, "rev-parse", "HEAD")

    fixture = {
        "bundles": {
            "packages/a/one@2023": [
                _row("k2", "rs.a", "tax_year", 2023, 10),
                # Out of scope: a record set the scope does not keep.
                _row("k9", "rs.z", "tax_year", 2023, 99),
                # Scoped to packages/b/two@2024, so this copy must be ignored.
                _row("k1", "rs.b", "calendar_year", 2024, 777),
            ],
            "packages/b/two@2024": [_row("k1", "rs.b", "calendar_year", 2024, 20)],
        }
    }
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture))
    monkeypatch.setenv("FAKE_CHRONICLE_FIXTURE", str(fixture_path))
    fake = tmp_path / "fake_chronicle.py"
    fake.write_text(_FAKE_CHRONICLE)

    scope = {
        "version": 1,
        "country": "us",
        "source_repo": "PolicyEngine/chronicle",
        "source_commit": commit,
        "rule": "fixture",
        "pair_count": 2,
        "runs": {"2023": ["packages/a/one"], "2024": ["packages/b/two"]},
        "pairs": [
            {
                "record_set_id": "rs.a",
                "period_type": "tax_year",
                "period_value": "2023",
                "package_id": "a-one",
                "package": "packages/a/one",
                "build_year": 2023,
            },
            {
                "record_set_id": "rs.b",
                "period_type": "calendar_year",
                "period_value": "2024",
                "package_id": "b-two",
                "package": "packages/b/two",
                "build_year": 2024,
            },
        ],
    }
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(json.dumps(scope))
    return {
        "root": root,
        "commit": commit,
        "fixture": fixture,
        "fixture_path": fixture_path,
        "scope": scope,
        "scope_path": scope_path,
        "command": f"{sys.executable} {fake}",
    }


def _build(chronicle: dict, out: Path, *extra: str) -> dict:
    builder = _load_tool("build_us_chronicle_feed")
    assert (
        builder.main(
            [
                "--chronicle-root",
                str(chronicle["root"]),
                "--out",
                str(out),
                "--scope",
                str(chronicle["scope_path"]),
                "--chronicle-command",
                chronicle["command"],
                *extra,
            ]
        )
        == 0
    )
    return json.loads((out / "receipt.json").read_text())


def test_builder_keeps_each_scoped_pair_from_its_own_run_and_is_deterministic(
    chronicle: dict, tmp_path: Path
) -> None:
    first = _build(chronicle, tmp_path / "run-1")
    lines = (tmp_path / "run-1" / "consumer_facts.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert [r["aggregate_fact_key"] for r in rows] == ["k1", "k2"]
    assert [r["value"] for r in rows] == [20, 10]
    assert first["row_count"] == 2
    assert first["pair_count"] == 2
    assert first["source_commit"] == chronicle["commit"]
    assert [run["year"] for run in first["runs"]] == [2023, 2024]
    assert "--source packages/a/one" in first["runs"][0]["command"]
    assert (
        first["facts_sha256"]
        == hashlib.sha256(
            (tmp_path / "run-1" / "consumer_facts.jsonl").read_bytes()
        ).hexdigest()
    )
    assert first["artifact"]["facts_sha256"] == first["facts_sha256"]
    assert (
        first["artifact"]["artifact_schema_version"]
        == "policyengine_ledger.consumer_artifact.v2"
    )
    assert len(first["artifact"]["manifest_sha256"]) == 64

    second = _build(chronicle, tmp_path / "run-2", "--skip-artifact")
    assert second["facts_sha256"] == first["facts_sha256"]
    assert second["artifact"] is None
    assert (tmp_path / "run-2" / "consumer_facts.jsonl").read_bytes() == (
        tmp_path / "run-1" / "consumer_facts.jsonl"
    ).read_bytes()


def test_builder_refuses_to_overwrite_without_replace(
    chronicle: dict, tmp_path: Path
) -> None:
    _build(chronicle, tmp_path / "run", "--skip-artifact")
    with pytest.raises(SystemExit, match="pass --replace"):
        _build(chronicle, tmp_path / "run", "--skip-artifact")
    _build(chronicle, tmp_path / "run", "--skip-artifact", "--replace")


def test_builder_refuses_a_pair_no_run_emits(chronicle: dict, tmp_path: Path) -> None:
    chronicle["fixture"]["bundles"]["packages/b/two@2024"] = []
    chronicle["fixture_path"].write_text(json.dumps(chronicle["fixture"]))
    with pytest.raises(SystemExit, match=r"1 scoped pairs produced no row"):
        _build(chronicle, tmp_path / "run", "--skip-artifact")


def test_builder_refuses_conflicting_bytes_under_one_fact_key(
    chronicle: dict, tmp_path: Path
) -> None:
    chronicle["fixture"]["bundles"]["packages/a/one@2023"].append(
        _row("k2", "rs.a", "tax_year", 2023, 11)
    )
    chronicle["fixture_path"].write_text(json.dumps(chronicle["fixture"]))
    with pytest.raises(SystemExit, match=r"aggregate_fact_key k2 appears twice"):
        _build(chronicle, tmp_path / "run", "--skip-artifact")


def test_builder_refuses_a_package_chronicle_marked_invalid(
    chronicle: dict, tmp_path: Path
) -> None:
    chronicle["fixture"]["invalid"] = ["packages/a/one"]
    chronicle["fixture_path"].write_text(json.dumps(chronicle["fixture"]))
    with pytest.raises(SystemExit, match=r"marked 'packages/a/one' invalid"):
        _build(chronicle, tmp_path / "run", "--skip-artifact")


def test_builder_refuses_the_wrong_commit_or_a_dirty_tree(
    chronicle: dict, tmp_path: Path
) -> None:
    chronicle["scope"]["source_commit"] = "0" * 40
    chronicle["scope_path"].write_text(json.dumps(chronicle["scope"]))
    with pytest.raises(SystemExit, match="not the scope's commit"):
        _build(chronicle, tmp_path / "run", "--skip-artifact")

    chronicle["scope"]["source_commit"] = chronicle["commit"]
    chronicle["scope_path"].write_text(json.dumps(chronicle["scope"]))
    (chronicle["root"] / "stray.txt").write_text("uncommitted\n")
    with pytest.raises(SystemExit, match="uncommitted changes"):
        _build(chronicle, tmp_path / "run", "--skip-artifact")
