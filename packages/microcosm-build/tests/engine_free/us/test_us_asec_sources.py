"""Pinned CPS ASEC source coordinates and the base builder's digest check.

``microcosm.build.us_runtime.asec_sources`` owns the immutable coordinates of
the four processed ASEC inputs and the default pool (the newest three).
``tools/build_us_puf_support_base.py --asec-h5-sha256`` refuses an input whose
bytes differ from its declared digest before any stage runs, and
``tools/fetch_us_asec_sources.py`` resolves the pinned files and prints those
arguments. These tests pin all four, hold the income-2022..2024 coordinates to
their 2026-09-18 values so historical builds stay byte-reproducible, and bind
the module's digests to the hermetic-input evidence already recorded in
``ecps_parity_known_gaps.json`` wherever both name a year.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from microcosm.build.us_runtime import asec_sources
from microcosm.build.us_runtime.asec_sources import (
    ASEC_DEFAULT_POOL_INCOME_YEARS,
    ASEC_DEFAULT_POOL_SIZE,
    ASEC_SOURCE_ARTIFACTS,
    ASEC_SOURCE_REPOSITORY_ID,
    ASEC_SOURCE_REPOSITORY_TYPE,
    ASEC_SOURCE_REVISION,
    ASEC_SOURCE_REVISION_2025,
    AsecSourceArtifact,
    asec_source_artifact,
    fetch_asec_source,
    newest_pinned_income_years,
)
from microcosm.build.us_runtime.education_assistance_source import (
    ASEC_EDUCATION_ASSISTANCE_ARCHIVES,
)
from microcosm.build.us_runtime.parity_reference import (
    ECPS_PARITY_KNOWN_GAPS_RESOURCE,
)
from microcosm.build.us_runtime.public_assistance_type_source import (
    ASEC_PUBLIC_ASSISTANCE_TYPE_AUDIT_PINS,
)
from microcosm.build.us_runtime.spm_role_source import ASEC_SPM_ROLE_SOURCES
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

_ROOT = _TEST_PATHS.repository
_FIXTURE_BYTES = b"asec-fixture-bytes-2022"
_FIXTURE_SHA256 = hashlib.sha256(_FIXTURE_BYTES).hexdigest()
_WRONG_SHA256 = "0" * 64
_ASEC_FILENAME = re.compile(r"census_cps_(\d{4})\.h5")


def _load_tool_module(name: str):
    path = _ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fixture_artifact(year: int = 2022) -> AsecSourceArtifact:
    return AsecSourceArtifact(
        income_year=year,
        filename=f"census_cps_{year}.h5",
        sha256=_FIXTURE_SHA256,
        size_bytes=len(_FIXTURE_BYTES),
    )


@pytest.fixture
def fixture_pins(monkeypatch: pytest.MonkeyPatch) -> AsecSourceArtifact:
    artifact = _fixture_artifact()
    monkeypatch.setattr(
        asec_sources,
        "ASEC_SOURCE_ARTIFACTS",
        MappingProxyType({2022: artifact}),
    )
    return artifact


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


def _forbid_download(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    def refuse(**kwargs):
        raise AssertionError("hf_hub_download must not be called")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", refuse)


def _fake_download(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    tmp_path: Path,
) -> dict[str, object]:
    import huggingface_hub

    calls: dict[str, object] = {}

    def fake(**kwargs):
        calls.update(kwargs)
        target = tmp_path / "hub" / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    return calls


def _hermetic_asec_digests() -> dict[int, set[str]]:
    resource = (
        _ROOT
        / "packages"
        / "microcosm-build"
        / "src"
        / "microcosm"
        / "build"
        / "us"
        / ECPS_PARITY_KNOWN_GAPS_RESOURCE
    )
    found: dict[int, set[str]] = {}

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"hermetic_inputs", "hermetic_asec_inputs"}:
                    for entry in value:
                        match = _ASEC_FILENAME.fullmatch(entry["filename"])
                        if match is not None:
                            year = int(match.group(1))
                            found.setdefault(year, set()).add(entry["sha256"])
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(json.loads(resource.read_text(encoding="utf-8")))
    return found


def test_pins_agree_with_the_hermetic_input_evidence() -> None:
    # The evidence records the historical Build J inputs (income 2022-2024);
    # every year it names must carry exactly the pinned digest.
    evidence = _hermetic_asec_digests()
    assert set(evidence) == {2022, 2023, 2024}
    assert set(evidence) <= set(ASEC_SOURCE_ARTIFACTS)
    for year, digests in evidence.items():
        assert digests == {ASEC_SOURCE_ARTIFACTS[year].sha256}, year


def test_coordinates_are_well_formed_and_immutable() -> None:
    assert ASEC_SOURCE_REPOSITORY_ID == "policyengine/microcosm-us-sources"
    assert ASEC_SOURCE_REPOSITORY_TYPE == "dataset"
    assert set(ASEC_SOURCE_ARTIFACTS) == {2022, 2023, 2024, 2025}
    for year, artifact in ASEC_SOURCE_ARTIFACTS.items():
        assert artifact.income_year == year
        assert artifact.filename == f"census_cps_{year}.h5"
        assert re.fullmatch(r"[0-9a-f]{64}", artifact.sha256)
        assert re.fullmatch(r"[0-9a-f]{40}", artifact.revision)
        assert artifact.size_bytes > 0
        assert artifact.url == (
            "https://huggingface.co/datasets/policyengine/microcosm-us-sources"
            f"/resolve/{artifact.revision}/census_cps_{year}.h5"
        )
    assert len({artifact.sha256 for artifact in ASEC_SOURCE_ARTIFACTS.values()}) == (
        len(ASEC_SOURCE_ARTIFACTS)
    )
    with pytest.raises(TypeError):
        ASEC_SOURCE_ARTIFACTS[2026] = _fixture_artifact(2026)  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        ASEC_SOURCE_ARTIFACTS[2022].sha256 = _WRONG_SHA256  # type: ignore[misc]


def test_historical_coordinates_are_unchanged() -> None:
    # A historical income-2022..2024 build resolves exactly what it resolved
    # when the files were first mirrored: same revision, URL, digest and size.
    # Adding a year uploads a new revision and never moves these.
    assert ASEC_SOURCE_REVISION == "78acc83ea8b099a97cb0d658bbed91ea75aae8b0"
    for year in (2022, 2023, 2024):
        assert ASEC_SOURCE_ARTIFACTS[year].url == (
            "https://huggingface.co/datasets/policyengine/microcosm-us-sources"
            f"/resolve/78acc83ea8b099a97cb0d658bbed91ea75aae8b0/census_cps_{year}.h5"
        )
    assert {
        year: (artifact.revision, artifact.sha256, artifact.size_bytes)
        for year, artifact in ASEC_SOURCE_ARTIFACTS.items()
        if year <= 2024
    } == {
        2022: (
            ASEC_SOURCE_REVISION,
            "7ccca976284bb47815d84460cc4f75a0a65d26d7754ab0a0f417de351b3d474e",
            301_129_278,
        ),
        2023: (
            ASEC_SOURCE_REVISION,
            "cb57817327799f42b741caed5f9be94d04021c2e6809c1ad7bd0686da5428d88",
            299_036_610,
        ),
        2024: (
            ASEC_SOURCE_REVISION,
            "ec36604cb735a660b51b0b2f90be27d803b5878f3464fb30d0eacead59c1260d",
            323_994_739,
        ),
    }


def test_income_year_2025_is_pinned_at_its_own_upload() -> None:
    artifact = asec_source_artifact(2025)
    assert artifact.revision == ASEC_SOURCE_REVISION_2025
    assert ASEC_SOURCE_REVISION_2025 != ASEC_SOURCE_REVISION
    assert artifact.sha256 == (
        "4c5a32188b6acfbcfbeb9f5d719d3847873887b72ebdbb19bc402c16e0a64e58"
    )
    assert artifact.size_bytes == 304_753_967


def test_default_pool_is_the_newest_three_pinned_years() -> None:
    assert ASEC_DEFAULT_POOL_SIZE == 3
    assert ASEC_DEFAULT_POOL_INCOME_YEARS == (2023, 2024, 2025)
    assert ASEC_DEFAULT_POOL_INCOME_YEARS == tuple(
        sorted(ASEC_SOURCE_ARTIFACTS)[-ASEC_DEFAULT_POOL_SIZE:]
    )
    # 2022 leaves the default but stays pinned and resolvable.
    assert 2022 not in ASEC_DEFAULT_POOL_INCOME_YEARS
    assert asec_source_artifact(2022).filename == "census_cps_2022.h5"


@given(
    years=st.sets(st.integers(min_value=1990, max_value=2100), min_size=1, max_size=12),
    size=st.integers(min_value=-2, max_value=14),
)
def test_newest_pinned_income_years_properties(years: set[int], size: int) -> None:
    artifacts = {year: _fixture_artifact(year) for year in years}
    if size < 1 or size > len(years):
        with pytest.raises(ValueError):
            newest_pinned_income_years(size, artifacts)
        return
    pool = newest_pinned_income_years(size, artifacts)
    assert len(pool) == size
    assert list(pool) == sorted(pool)
    assert set(pool) <= years
    # Every pinned year outside the pool is older than every pooled year.
    assert all(other < min(pool) for other in years - set(pool))
    # Pinning a newer year shifts the pool by exactly one year.
    newer = max(years) + 1
    shifted = newest_pinned_income_years(
        size, {**artifacts, newer: _fixture_artifact(newer)}
    )
    assert shifted == (*pool[1:], newer)


def test_every_pinned_year_is_pinned_in_every_year_keyed_registry() -> None:
    # A pooled year needs its processed H5 and its Census survey archive
    # (education and PAW_TYP sidecars, SPM role, #720 person columns). If any
    # registry lacked a pinned year, a build of that year would refuse
    # mid-construction instead of here.
    pinned = set(ASEC_SOURCE_ARTIFACTS)
    assert set(ASEC_EDUCATION_ASSISTANCE_ARCHIVES) == pinned
    assert set(ASEC_SPM_ROLE_SOURCES) == pinned
    assert set(ASEC_PUBLIC_ASSISTANCE_TYPE_AUDIT_PINS) == pinned
    assert set(ASEC_DEFAULT_POOL_INCOME_YEARS) <= pinned
    for year in pinned:
        archive = ASEC_EDUCATION_ASSISTANCE_ARCHIVES[year]
        assert archive.survey_year == year + 1
        assert archive.member == f"pppub{(year + 1) % 100:02d}.csv"
        assert archive.zip_url == (
            "https://www2.census.gov/programs-surveys/cps/datasets/"
            f"{year + 1}/march/asecpub{(year + 1) % 100:02d}csv.zip"
        )
        audit = ASEC_PUBLIC_ASSISTANCE_TYPE_AUDIT_PINS[year]
        assert audit.rows == archive.rows
        assert sum(audit.paw_type_counts) == audit.rows
        assert audit.paw_positive_tanf_rows <= audit.paw_positive_rows
        assert ASEC_SPM_ROLE_SOURCES[year].persons == archive.rows


def test_unknown_year_refuses_before_any_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_download(monkeypatch)
    with pytest.raises(ValueError, match=r"No pinned ASEC source for income year 2019"):
        asec_source_artifact(2019)
    with pytest.raises(
        ValueError, match=r"pinned years are \[2022, 2023, 2024, 2025\]"
    ):
        fetch_asec_source(2019)


def test_fetch_returns_a_matching_local_path_without_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fixture_pins: AsecSourceArtifact,
) -> None:
    _forbid_download(monkeypatch)
    local = tmp_path / "anywhere.h5"
    local.write_bytes(_FIXTURE_BYTES)
    assert fetch_asec_source(2022, local_path=local) == local


def test_fetch_ignores_a_local_path_whose_bytes_differ(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fixture_pins: AsecSourceArtifact,
) -> None:
    calls = _fake_download(monkeypatch, _FIXTURE_BYTES, tmp_path)
    local = tmp_path / "stale.h5"
    # Same byte length, different bytes: only the digest can tell them apart.
    local.write_bytes(b"x" * len(_FIXTURE_BYTES))
    resolved = fetch_asec_source(2022, local_path=local)
    assert resolved == tmp_path / "hub" / "census_cps_2022.h5"
    assert calls == {
        "repo_id": ASEC_SOURCE_REPOSITORY_ID,
        "filename": "census_cps_2022.h5",
        "repo_type": ASEC_SOURCE_REPOSITORY_TYPE,
        "revision": ASEC_SOURCE_REVISION,
        "cache_dir": None,
    }


def test_fetch_prefers_the_microcosm_cache_over_the_hub(
    monkeypatch: pytest.MonkeyPatch,
    home: Path,
    fixture_pins: AsecSourceArtifact,
) -> None:
    _forbid_download(monkeypatch)
    cached = home / ".cache" / "microcosm" / "asec" / "census_cps_2022.h5"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(_FIXTURE_BYTES)
    assert fetch_asec_source(2022) == cached


def test_fetch_skips_a_cached_file_of_the_wrong_size(
    monkeypatch: pytest.MonkeyPatch,
    home: Path,
    fixture_pins: AsecSourceArtifact,
) -> None:
    cached = home / ".cache" / "microcosm" / "asec" / "census_cps_2022.h5"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(_FIXTURE_BYTES + b"!")
    calls = _fake_download(monkeypatch, _FIXTURE_BYTES, home)
    assert fetch_asec_source(2022) == home / "hub" / "census_cps_2022.h5"
    assert calls["revision"] == ASEC_SOURCE_REVISION


def test_fetch_downloads_each_year_at_its_own_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    newer = AsecSourceArtifact(
        income_year=2025,
        filename="census_cps_2025.h5",
        sha256=_FIXTURE_SHA256,
        size_bytes=len(_FIXTURE_BYTES),
        revision=ASEC_SOURCE_REVISION_2025,
    )
    monkeypatch.setattr(
        asec_sources,
        "ASEC_SOURCE_ARTIFACTS",
        MappingProxyType({2022: _fixture_artifact(), 2025: newer}),
    )
    calls = _fake_download(monkeypatch, _FIXTURE_BYTES, tmp_path)
    fetch_asec_source(2025, tmp_path / "hf-cache")
    assert calls["revision"] == ASEC_SOURCE_REVISION_2025
    fetch_asec_source(2022, tmp_path / "hf-cache")
    assert calls["revision"] == ASEC_SOURCE_REVISION


def test_fetch_with_a_cache_dir_goes_to_the_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
    home: Path,
    fixture_pins: AsecSourceArtifact,
) -> None:
    cached = home / ".cache" / "microcosm" / "asec" / "census_cps_2022.h5"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(_FIXTURE_BYTES)
    calls = _fake_download(monkeypatch, _FIXTURE_BYTES, home)
    resolved = fetch_asec_source(2022, home / "hf-cache")
    assert resolved == home / "hub" / "census_cps_2022.h5"
    assert calls["cache_dir"] == str(home / "hf-cache")


def test_fetch_refuses_a_download_whose_bytes_differ(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fixture_pins: AsecSourceArtifact,
) -> None:
    _fake_download(monkeypatch, b"not the pinned bytes", tmp_path)
    with pytest.raises(ValueError, match=r"failed immutable-source verification"):
        fetch_asec_source(2022, tmp_path / "hf-cache")


def _args(asec_h5: list[str] | None, pins: list[str] | None) -> SimpleNamespace:
    return SimpleNamespace(asec_h5=asec_h5, asec_h5_sha256=pins)


def test_verify_is_a_no_op_without_pins() -> None:
    builder = _load_tool_module("build_us_puf_support_base")
    assert builder._verify_asec_source_digests(_args(None, None)) is None
    assert builder._verify_asec_source_digests(_args(["2022=missing.h5"], [])) is None


@pytest.mark.parametrize(
    ("asec_h5", "pins", "message"),
    [
        (None, [f"2022={_FIXTURE_SHA256}"], r"requires --asec-h5"),
        (["2022=a.h5"], ["2022=abc"], r"64 hexadecimal"),
        (["2022=a.h5"], ["2022=" + "g" * 64], r"64 hexadecimal"),
        (
            ["2022=a.h5"],
            [f"2022={_FIXTURE_SHA256}", f"2022={_FIXTURE_SHA256}"],
            r"names year 2022 twice",
        ),
        (
            ["2022=a.h5"],
            [f"2023={_FIXTURE_SHA256}"],
            r"names year 2023, which no --asec-h5",
        ),
        (
            ["2022=a.h5"],
            [f"2022={_WRONG_SHA256}"],
            r"ASEC 2022 CLI pin differs from the canonical pin",
        ),
        (["2022=a.h5"], ["2022abc"], r"must be YEAR=SHA256"),
        (["2022=a.h5"], [f"twenty22={_FIXTURE_SHA256}"], r"year must be an integer"),
        (["2022"], [f"2022={_FIXTURE_SHA256}"], r"ASEC source must be YEAR=PATH"),
        (["y=a.h5"], [f"2022={_FIXTURE_SHA256}"], r"--asec-h5 'y=a.h5': the year"),
        (
            ["2018=a.h5", "2019=b.h5"],
            [f"2018={_FIXTURE_SHA256}"],
            r"pins \[2018\] but --asec-h5 also names \[2019\]",
        ),
        (
            ["2019=missing-a.h5", "2022=missing-b.h5"],
            [f"2019={_FIXTURE_SHA256}", f"2022={_WRONG_SHA256}"],
            r"ASEC 2022 CLI pin differs from the canonical pin",
        ),
        (
            ["2019=a.h5", "2019=b.h5"],
            [f"2019={_FIXTURE_SHA256}"],
            r"Duplicate --asec-h5 mapping for year 2019",
        ),
    ],
    ids=[
        "no-source",
        "short-hex",
        "non-hex",
        "duplicate-year",
        "unmapped-year",
        "not-canonical",
        "pin-without-equals",
        "pin-year-not-integer",
        "source-without-equals",
        "source-year-not-integer",
        "partial-pinning",
        "canonical-before-any-hash",
        "duplicate-source-year",
    ],
)
def test_verify_refuses_malformed_or_divergent_pins(
    asec_h5: list[str] | None,
    pins: list[str],
    message: str,
) -> None:
    # No case names a file that exists: every refusal here fires before any
    # byte is read, the canonical comparison included — also with two years,
    # where the 2019 file would be hashed first if hashing were interleaved.
    builder = _load_tool_module("build_us_puf_support_base")
    with pytest.raises(SystemExit, match=message):
        builder._verify_asec_source_digests(_args(asec_h5, pins))


def test_verify_refuses_a_pinned_file_that_does_not_exist(tmp_path: Path) -> None:
    builder = _load_tool_module("build_us_puf_support_base")
    missing = tmp_path / "census_cps_2019.h5"
    with pytest.raises(
        SystemExit, match=rf"ASEC 2019 source {re.escape(str(missing))} does not exist"
    ):
        builder._verify_asec_source_digests(
            _args([f"2019={missing}"], [f"2019={_FIXTURE_SHA256}"])
        )


def test_verify_refuses_bytes_that_differ_from_the_declared_digest(
    tmp_path: Path,
) -> None:
    builder = _load_tool_module("build_us_puf_support_base")
    source = tmp_path / "census_cps_2019.h5"
    source.write_bytes(_FIXTURE_BYTES)
    expected = (
        rf"ASEC 2019 source {re.escape(str(source))} failed digest verification: "
        rf"expected {_WRONG_SHA256}, got {_FIXTURE_SHA256}"
    )
    with pytest.raises(SystemExit, match=expected):
        builder._verify_asec_source_digests(
            _args([f"2019={source}"], [f"2019={_WRONG_SHA256}"])
        )


def test_verify_accepts_matching_bytes_for_pinned_and_unpinned_years(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    builder = _load_tool_module("build_us_puf_support_base")
    monkeypatch.setattr(
        builder,
        "ASEC_SOURCE_ARTIFACTS",
        MappingProxyType({2022: _fixture_artifact()}),
    )
    pinned = tmp_path / "census_cps_2022.h5"
    unpinned = tmp_path / "census_cps_2019.h5"
    pinned.write_bytes(_FIXTURE_BYTES)
    unpinned.write_bytes(_FIXTURE_BYTES)
    assert (
        builder._verify_asec_source_digests(
            _args(
                [f"2022={pinned}", f"2019={unpinned}"],
                [f"2022={_FIXTURE_SHA256.upper()}", f"2019={_FIXTURE_SHA256}"],
            )
        )
        is None
    )


def _receipt(**digests: str) -> dict:
    return {
        "kind": "pooled_asec",
        "sources": [
            {"year": int(year), "sha256": sha} for year, sha in digests.items()
        ],
    }


def test_receipt_digests_must_equal_the_locked_pins() -> None:
    builder = _load_tool_module("build_us_puf_support_base")
    pinned = _args(["2019=a.h5"], [f"2019={_FIXTURE_SHA256}"])
    assert (
        builder._require_receipt_digests_match_pins(
            pinned, _receipt(**{"2019": _FIXTURE_SHA256})
        )
        is None
    )
    assert (
        builder._require_receipt_digests_match_pins(
            _args(["2019=a.h5"], None), _receipt(**{"2019": _WRONG_SHA256})
        )
        is None
    ), "no pins, nothing to compare"
    with pytest.raises(
        SystemExit,
        match=rf"ASEC 2019 source construction read bytes with sha256 {_WRONG_SHA256}",
    ):
        builder._require_receipt_digests_match_pins(
            pinned, _receipt(**{"2019": _WRONG_SHA256})
        )
    with pytest.raises(SystemExit, match=r"records no digest for it"):
        builder._require_receipt_digests_match_pins(
            pinned, _receipt(**{"2018": _FIXTURE_SHA256})
        )


def test_completed_stage_reentry_reads_the_receipt_under_base_source() -> None:
    builder = _load_tool_module("build_us_puf_support_base")
    pinned = _args(["2019=a.h5"], [f"2019={_FIXTURE_SHA256}"])
    # The shape _source_construction_stage returns and StageRuntime stores.
    stage_metadata = {
        "base_source": _receipt(**{"2019": _FIXTURE_SHA256}),
        "base_rows": 3,
        "base_household_weight_total": 1.0,
        "weeks_unemployed_source_path": None,
    }
    assert (
        builder._require_completed_source_receipt_matches_pins(pinned, stage_metadata)
        is None
    )
    stage_metadata["base_source"] = _receipt(**{"2019": _WRONG_SHA256})
    with pytest.raises(SystemExit, match=r"read bytes with sha256"):
        builder._require_completed_source_receipt_matches_pins(pinned, stage_metadata)
    with pytest.raises(SystemExit, match=r"records no digest for it"):
        builder._require_completed_source_receipt_matches_pins(pinned, {"base_rows": 3})
    assert (
        builder._require_completed_source_receipt_matches_pins(
            _args(["2019=a.h5"], None), {"base_rows": 3}
        )
        is None
    ), "no pins, nothing to compare"


def _parse_asec_build_args(builder, tmp_path: Path, extra: list[str]):
    return builder._parse_args(
        [
            "--asec-h5",
            f"2022={tmp_path / 'census_cps_2022.h5'}",
            "--puf-h5",
            str(tmp_path / "puf.h5"),
            "--out",
            str(tmp_path / "out"),
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--without-block-ladder",
            *extra,
        ]
    )


def test_pins_are_forwarded_to_stage_children_and_locked_in_the_run_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    builder = _load_tool_module("build_us_puf_support_base")
    monkeypatch.setattr(
        builder,
        "_builder_code_identity",
        lambda: {"source_sha256": "builder"},
    )
    monkeypatch.setattr(
        builder,
        "ASEC_SOURCE_ARTIFACTS",
        MappingProxyType({2022: _fixture_artifact()}),
    )
    # Upper case with surrounding whitespace: forwarded verbatim to the stage
    # children, locked normalized so an equivalent resume compares equal.
    pin = f"2022= {_FIXTURE_SHA256.upper()} "
    args = _parse_asec_build_args(builder, tmp_path, ["--asec-h5-sha256", pin])
    command = builder._stage_cli_args(args, "source_construction")
    assert command[command.index("--asec-h5-sha256") + 1] == pin
    assert builder._stage_run_config(args)["asec_h5_sha256"] == {
        "2022": _FIXTURE_SHA256
    }

    unpinned = _parse_asec_build_args(builder, tmp_path, [])
    assert "--asec-h5-sha256" not in builder._stage_cli_args(
        unpinned, "source_construction"
    )
    assert builder._stage_run_config(unpinned)["asec_h5_sha256"] is None


def test_main_refuses_a_wrong_digest_before_dispatching_any_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    builder = _load_tool_module("build_us_puf_support_base")
    source = tmp_path / "census_cps_2019.h5"
    source.write_bytes(_FIXTURE_BYTES)

    def never(*args, **kwargs):
        raise AssertionError("no stage may run after a refused digest")

    for name in ("_run_all", "_run_staged_all", "_run_configured_stage"):
        monkeypatch.setattr(builder, name, never)
    with pytest.raises(
        SystemExit, match=r"ASEC 2019 source .* failed digest verification"
    ):
        builder.main(
            [
                "--asec-h5",
                f"2019={source}",
                "--asec-h5-sha256",
                f"2019={_WRONG_SHA256}",
                "--puf-h5",
                str(tmp_path / "puf.h5"),
                "--out",
                str(tmp_path / "out"),
                "--without-block-ladder",
            ]
        )


def test_fetch_tool_prints_builder_arguments_for_the_default_pool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_tool_module("fetch_us_asec_sources")
    seen: list[tuple[int, object]] = []

    def fake_fetch(year: int, cache_dir=None) -> Path:
        seen.append((year, cache_dir))
        return tmp_path / f"census_cps_{year}.h5"

    monkeypatch.setattr(tool, "fetch_asec_source", fake_fetch)
    assert tool.main([]) == 0
    assert seen == [(2023, None), (2024, None), (2025, None)]
    assert capsys.readouterr().out.splitlines() == [
        f"--asec-h5 {year}={tmp_path / f'census_cps_{year}.h5'} "
        f"--asec-h5-sha256 {year}={ASEC_SOURCE_ARTIFACTS[year].sha256}"
        for year in ASEC_DEFAULT_POOL_INCOME_YEARS
    ]


def test_fetch_tool_resolves_every_pinned_year_on_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_tool_module("fetch_us_asec_sources")
    seen: list[int] = []

    def fake_fetch(year: int, cache_dir=None) -> Path:
        seen.append(year)
        return tmp_path / f"census_cps_{year}.h5"

    monkeypatch.setattr(tool, "fetch_asec_source", fake_fetch)
    assert tool.main(["--all-pinned"]) == 0
    assert seen == [2022, 2023, 2024, 2025]
    assert tool.main(["2022", "2023", "2024"]) == 0
    assert seen[4:] == [2022, 2023, 2024]
    with pytest.raises(SystemExit):
        tool.main(["--all-pinned", "2022"])


def test_fetch_tool_takes_years_and_a_cache_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _load_tool_module("fetch_us_asec_sources")
    seen: list[tuple[int, object]] = []

    def fake_fetch(year: int, cache_dir=None) -> Path:
        seen.append((year, cache_dir))
        return tmp_path / f"census_cps_{year}.h5"

    monkeypatch.setattr(tool, "fetch_asec_source", fake_fetch)
    assert tool.main(["2023", "--cache-dir", str(tmp_path / "hf")]) == 0
    assert seen == [(2023, tmp_path / "hf")]
    assert capsys.readouterr().out.splitlines() == [
        f"--asec-h5 2023={tmp_path / 'census_cps_2023.h5'} "
        f"--asec-h5-sha256 2023={ASEC_SOURCE_ARTIFACTS[2023].sha256}"
    ]


def test_fetch_tool_refuses_an_unpinned_year(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_download(monkeypatch)
    tool = _load_tool_module("fetch_us_asec_sources")
    with pytest.raises(SystemExit, match=r"No pinned ASEC source for income year 2019"):
        tool.main(["2019"])


def test_year_keyed_loaders_default_to_the_default_pool() -> None:
    # A bare call loads the default pool, never every registered year: with
    # 2022 still pinned, "every year" would silently mean a four-year pool.
    import inspect

    from microcosm.build.us_runtime.education_assistance_source import (
        load_asec_education_assistance_sources,
    )
    from microcosm.build.us_runtime.public_assistance_type_source import (
        load_asec_public_assistance_type_sources,
    )
    from microcosm.build.us_runtime.spm_independence_role import (
        resolve_asec_spm_role_source_paths,
    )

    for loader in (
        load_asec_education_assistance_sources,
        load_asec_public_assistance_type_sources,
        resolve_asec_spm_role_source_paths,
    ):
        default = inspect.signature(loader).parameters["income_years"].default
        assert default == ASEC_DEFAULT_POOL_INCOME_YEARS, loader.__name__
