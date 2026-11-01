"""The one-command UK cut runner: pins, fetching, argv and resumability."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
import json
import re
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tools import run_uk_cut as cut

REPO = Path(cut.__file__).resolve().parents[1]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# Pins: the runner accepts exactly what the build checks                        #
# --------------------------------------------------------------------------- #


def test_every_declared_input_resolves_to_a_committed_pin():
    sources = {source.key: source for source in cut.load_input_sources()}
    assert set(sources) == {
        "frs_2024_25",
        "spi_2022_23",
        "was_round_8",
        "etb_1977_2024",
        "nts_2002_2024",
        "lcfs_2024_25",
        "hmrc_personal_incomes_ods",
        "chronicle_uk",
        "input_mass_reference",
        "incumbent_efrs",
    }
    for source in sources.values():
        assert source.files
        for pin in source.files:
            assert len(pin.sha256) == 64 and int(pin.sha256, 16) >= 0


def test_every_hosted_archive_is_pinned_to_a_revision_and_digest():
    """Each zip is read at a fixed commit and checked whole before any member."""

    for source in cut.load_input_sources():
        if source.kind != "hf_zip_members":
            continue
        assert re.fullmatch(r"[0-9a-f]{40}", source.revision or ""), source.key
        assert re.fullmatch(r"[0-9a-f]{64}", source.archive_sha256 or ""), source.key


def test_the_runner_pins_are_the_build_pins():
    """Lockstep with the files the build itself reads its pins from."""

    sources = {source.key: source for source in cut.load_input_sources()}
    feed = json.loads((cut.UK_DIR / "chronicle_feed.json").read_text())
    chronicle = {pin.name: pin.sha256 for pin in sources["chronicle_uk"].files}
    assert chronicle == {
        "manifest.json": feed["manifest_sha256"],
        "consumer_facts.jsonl": feed["facts_sha256"],
    }
    assert (
        sources["chronicle_uk"].path_in_repo == f"chronicle-uk/{feed['source_commit']}"
    )
    frs = json.loads((cut.UK_DIR / "frs_release.json").read_text())["acquisition"]
    assert sources["frs_2024_25"].revision == frs["huggingface_revision"]
    assert sources["frs_2024_25"].archive_sha256 == frs["zip_sha256"]
    assert sources["frs_2024_25"].filename == frs["zip_filename"]
    identity = cut._gate_reference_identity(cut.UK_DIR)
    incumbent = sources["incumbent_efrs"]
    assert incumbent.revision == identity["revision"]
    assert incumbent.files[0].sha256 == identity["sha256"]
    stage_pins = cut.committed_source_pins()
    for key in ("frs_2024_25", "spi_2022_23", "was_round_8", "etb_1977_2024"):
        for pin in sources[key].files:
            assert pin.sha256 == stage_pins[pin.name]["sha256"]
            assert pin.size_bytes == stage_pins[pin.name]["size_bytes"]


def test_the_chronicle_path_follows_the_feed_pin(tmp_path):
    """A re-pin of chronicle_feed.json needs no edit to the inputs manifest."""

    for name in ("source_stages.json", "frs_release.json", "gates.json"):
        (tmp_path / name).write_bytes((cut.UK_DIR / name).read_bytes())
    feed = json.loads((cut.UK_DIR / "chronicle_feed.json").read_text())
    feed["source_commit"] = "f" * 40
    (tmp_path / "chronicle_feed.json").write_text(json.dumps(feed))
    sources = {s.key: s for s in cut.load_input_sources(cut.CUT_INPUTS, tmp_path)}
    assert sources["chronicle_uk"].path_in_repo == "chronicle-uk/" + "f" * 40
    raw = next(
        s
        for s in json.loads(cut.CUT_INPUTS.read_text())["sources"]
        if s["key"] == "chronicle_uk"
    )
    assert raw["revision"] is None
    assert not re.search(r"[0-9a-f]{40}", json.dumps(raw))


def test_an_unknown_pin_selector_is_refused(tmp_path):
    manifest = json.loads(cut.CUT_INPUTS.read_text())
    manifest["sources"][0]["revison_pin"] = "frs_release"
    path = tmp_path / "uk_cut_inputs.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(cut.CutError, match="unknown pin selector"):
        cut.load_input_sources(path)


def test_the_hosted_mass_reference_is_the_one_the_gate_binds():
    from microcosm.build.uk_runtime.weighted_integrity import (
        UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256,
    )

    manifest = json.loads(cut.CUT_INPUTS.read_text())
    pin = manifest["file_pins"]["uk_input_mass_reference.json"]
    assert pin["canonical_evidence_sha256"] == UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256
    source = next(s for s in manifest["sources"] if s["key"] == "input_mass_reference")
    assert source["path_in_repo"].endswith(UK_INPUT_MASS_REFERENCE_EVIDENCE_SHA256)


def test_every_spine_tab_flag_has_a_pinned_input():
    """The spine's licensed tabs are all fetched, under the names it reads."""

    paths = cut.CutPaths(Path("/w"), "t", "0" * 40)
    argv = cut.spine_argv(paths, python="py", staging_local_only=True)
    sources = {s.key: s for s in cut.load_input_sources()}
    fetched = {
        str(paths.input_file(s.directory, pin.name))
        for s in sources.values()
        for pin in s.files
    }
    values = [a for a in argv[2:] if a.startswith("/w/inputs/")]
    assert values
    for value in values:
        assert value in fetched or value == str(paths.inputs / "frs_2024_25")


# --------------------------------------------------------------------------- #
# Every argv is accepted by the tool's own parser                               #
# --------------------------------------------------------------------------- #


def _paths(tmp_path: Path) -> cut.CutPaths:
    return cut.CutPaths(tmp_path, "test", "a" * 40)


def test_spine_argv_parses():
    from microcosm.build.uk_runtime import spine_build

    paths = cut.CutPaths(Path("/w"), "t", "a" * 40)
    args = spine_build._parse_args(
        cut.spine_argv(paths, python="py", staging_local_only=True)[2:]
    )
    assert args.release_candidate is True
    assert str(args.spine_h5) == str(paths.spine_h5)


@pytest.mark.parametrize("dry_run", [True, False])
def test_national_argv_parses(tmp_path, dry_run):
    from microcosm.build.uk_runtime import full_build_cli

    sources = {s.key: s for s in cut.load_input_sources()}
    argv = cut.national_argv(
        _paths(tmp_path),
        python="py",
        spine_sha256="b" * 64,
        sources=sources,
        dry_run=dry_run,
        logbook_prev_row_digest="c" * 64,
    )
    args = full_build_cli.parse_args(argv[2:])
    assert args.release_role == "national"
    assert args.dry_run is dry_run
    assert args.input_sha256 == "b" * 64


@pytest.mark.parametrize("dry_run", [True, False])
def test_the_national_build_never_publishes_to_the_orrery(tmp_path, dry_run):
    """With remote staging, the build would otherwise submit its graph."""

    from microcosm.build.uk_runtime import full_build_cli
    from microcosm.build.uk_runtime.orrery_publication import publication_enabled

    sources = {s.key: s for s in cut.load_input_sources()}
    argv = cut.national_argv(
        _paths(tmp_path),
        python="py",
        spine_sha256="b" * 64,
        sources=sources,
        dry_run=dry_run,
    )
    assert "--publish-orrery" not in argv
    assert publication_enabled(full_build_cli.parse_args(argv[2:])) is False


def _tool(name: str):
    import importlib

    return importlib.import_module(f"tools.{name}")


def test_certify_and_assemble_argv_parse(tmp_path):
    sources = {s.key: s for s in cut.load_input_sources()}
    paths = _paths(tmp_path)
    certify = _tool("certify_uk_release_cut")._parse_args(
        cut.certify_argv(
            paths,
            python="py",
            candidate_sha256="d" * 64,
            spine_sha256="e" * 64,
            sources=sources,
        )[2:]
    )
    assert certify.release_id == cut.RELEASE_ID
    assemble = _tool("assemble_uk_release_dir")._parse_args(
        cut.assemble_argv(paths, python="py")[2:]
    )
    assert str(assemble.out_dir) == "releases"


def test_score_argv_names_only_flags_the_scorer_declares(tmp_path, capsys):
    sources = {s.key: s for s in cut.load_input_sources()}
    argv = cut.score_argv(
        _paths(tmp_path), python="py", candidate_sha256="f" * 64, sources=sources
    )
    with pytest.raises(SystemExit):
        _tool("score_uk_national_candidate").main(["--help"])
    help_text = capsys.readouterr().out
    for flag in (a for a in argv[2:] if a.startswith("--")):
        assert flag in help_text


@given(tag=st.text(min_size=1, max_size=80))
def test_publication_is_always_inspection_only(tag):
    """Whatever the tag, the publisher is told not to move main or a pointer."""

    paths = cut.CutPaths(Path("/w"), "t", "a" * 40)
    for preflight in (True, False):
        argv = cut.publish_argv(paths, python="py", tag=tag, preflight_only=preflight)
        assert "--no-latest" in argv and "--tag-only" in argv
        assert "--promote-line" not in argv
        assert argv[argv.index("--repo-id") + 1] == cut.PUBLISH_REPO
        assert argv[argv.index("--tag-name") + 1] == tag


def test_publication_argv_uses_only_declared_flags(capsys):
    from microcosm.data import publish_cli

    paths = cut.CutPaths(Path("/w"), "t", "a" * 40)
    argv = cut.publish_argv(paths, python="py", tag="x", preflight_only=True)
    assert argv[1:3] == ["-m", "microcosm.data.publish_cli"]
    assert argv[3] == str(paths.release_dir)
    with pytest.raises(SystemExit):
        publish_cli.main(["--help"])
    help_text = capsys.readouterr().out
    for flag in (a for a in argv[4:] if a.startswith("--")):
        assert flag in help_text


# --------------------------------------------------------------------------- #
# Fetching: verified or absent, and idempotent                                  #
# --------------------------------------------------------------------------- #


class FakeHub:
    """Serves files and zips from memory and counts downloads."""

    def __init__(self, root: Path, files: dict[str, bytes]) -> None:
        self.root = root
        self.files = files
        self.downloads = 0

    def hf_file(self, *, repo_id, repo_type, filename, revision, local_dir=None):
        self.downloads += 1
        if filename not in self.files:
            raise FileNotFoundError(f"404 {filename}")
        path = (local_dir or self.root / "hub") / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[filename])
        return path

    def url(self, url, dest):
        self.downloads += 1
        dest.write_bytes(self.files[url])
        return dest


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(f"UKDA-0000-tab/tab/{name}", data)
    return buffer.getvalue()


def _zip_source(members: dict[str, bytes], *, pins=None) -> cut.InputSource:
    pins = pins or {name: _sha(data) for name, data in members.items()}
    return cut.InputSource(
        key="survey",
        kind="hf_zip_members",
        directory="survey",
        files=tuple(cut.PinnedFile(name, pins[name]) for name in members),
        repo_id="org/repo",
        repo_type="model",
        revision="r",
        filename="survey.zip",
    )


member_maps = st.dictionaries(
    st.from_regex(r"[a-z]{1,8}\.tab", fullmatch=True),
    st.binary(max_size=256),
    min_size=1,
    max_size=4,
)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(members=member_maps)
def test_a_matching_archive_installs_once(tmp_path_factory, members):
    root = tmp_path_factory.mktemp("fetch")
    hub = FakeHub(root, {"survey.zip": _zip(members)})
    source = _zip_source(members)
    first = cut.fetch_inputs([source], root / "inputs", downloader=hub)
    assert [r.status for r in first] == ["fetched"]
    for name, data in members.items():
        assert (root / "inputs/survey" / name).read_bytes() == data
    downloads = hub.downloads
    second = cut.fetch_inputs([source], root / "inputs", downloader=hub)
    assert [r.status for r in second] == ["present"]
    assert hub.downloads == downloads


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(members=member_maps, data=st.data())
def test_a_file_that_misses_its_pin_is_never_installed(tmp_path_factory, members, data):
    root = tmp_path_factory.mktemp("fetch")
    victim = data.draw(st.sampled_from(sorted(members)))
    pins = {name: _sha(value) for name, value in members.items()}
    pins[victim] = _sha(members[victim] + b"tampered")
    hub = FakeHub(root, {"survey.zip": _zip(members)})
    result = cut.fetch_inputs(
        [_zip_source(members, pins=pins)], root / "inputs", downloader=hub
    )
    assert result[0].status == "refused"
    assert not (root / "inputs/survey" / victim).exists()
    assert not list((root / "inputs").rglob("*.partial"))


def test_a_corrupted_installed_file_is_replaced_not_trusted(tmp_path):
    members = {"a.tab": b"alpha", "b.tab": b"beta"}
    hub = FakeHub(tmp_path, {"survey.zip": _zip(members)})
    source = _zip_source(members)
    cut.fetch_inputs([source], tmp_path / "inputs", downloader=hub)
    installed = tmp_path / "inputs/survey/a.tab"
    installed.write_bytes(b"ALPHA")  # same size, different bytes and mtime
    result = cut.fetch_inputs([source], tmp_path / "inputs", downloader=hub)
    assert result[0].status == "fetched"
    assert installed.read_bytes() == b"alpha"


def test_a_missing_upload_is_reported_with_its_note(tmp_path):
    source = cut.InputSource(
        key="nts",
        kind="hf_zip_members",
        directory="nts",
        files=(cut.PinnedFile("trip.tab", "0" * 64),),
        repo_id="org/repo",
        repo_type="model",
        filename="nts.zip",
        upload_note="upload nts.zip",
    )
    result = cut.fetch_inputs(
        [source], tmp_path / "inputs", downloader=FakeHub(tmp_path, {})
    )
    assert result[0].status == "missing"
    assert "upload nts.zip" in result[0].detail


def test_the_archive_digest_is_checked_when_pinned(tmp_path):
    members = {"a.tab": b"alpha"}
    source = _zip_source(members)
    pinned = cut.InputSource(**{**source.__dict__, "archive_sha256": "0" * 64})
    hub = FakeHub(tmp_path, {"survey.zip": _zip(members)})
    result = cut.fetch_inputs([pinned], tmp_path / "inputs", downloader=hub)
    assert result[0].status == "refused"
    assert "survey.zip hashes to" in result[0].detail


# --------------------------------------------------------------------------- #
# Resumability: each step runs once, and nothing is overwritten                 #
# --------------------------------------------------------------------------- #


BUILD_ROW = "d" * 64


class FakeBuild:
    """Stands in for the five tools: writes what each step writes."""

    def __init__(
        self,
        paths: cut.CutPaths,
        fail: set[str] | None = None,
        signer: str | None = None,
    ) -> None:
        self.paths = paths
        self.calls: list[str] = []
        self.argv: dict[str, list[str]] = {}
        self.env: dict[str, dict[str, str]] = {}
        self.fail = fail or set()
        self.signer = signer

    def _report(self, **payload) -> str:
        """A gate report, signed by ``signer``'s fingerprint when one is set."""

        if self.signer:
            payload["attestation"] = {"signing_key_sha256": self.signer}
        return json.dumps(payload)

    def __call__(
        self, argv: list[str], cwd: Path, log_path: Path, env: dict[str, str]
    ) -> int:
        tool = Path(argv[1]).name if argv[1] != "-m" else argv[2]
        step = {
            "build_uk_frs_spine.py": "spine",
            "build_uk_full.py": "national-dry-run"
            if "--dry-run" in argv
            else "national",
            "score_uk_national_candidate.py": "score",
            "certify_uk_release_cut.py": "certify",
            "assemble_uk_release_dir.py": "assemble",
            "microcosm.data.publish_cli": "publish",
        }[tool]
        self.calls.append(step)
        self.argv[step] = argv
        self.env[step] = env
        if step in self.fail:
            return 1
        getattr(self, step.replace("-", "_"))(log_path)
        return 0

    def spine(self, log_path):
        h5 = self.paths.spine_h5
        h5.write_bytes(b"spine")
        h5.with_name(h5.name + ".sha256").write_text(f"{_sha(b'spine')}  {h5.name}\n")
        h5.with_suffix(".build.json").write_text("{}")
        h5.with_suffix(".spine_gates.json").write_text(self._report(shippable=True))

    def national_dry_run(self, log_path):
        log_path.write_text('{"dry_run": true}')

    def national(self, log_path):
        run = self.paths.run_dir
        run.mkdir(parents=True)
        self.paths.candidate_h5.write_bytes(b"candidate")
        for name in (
            "calibration_diagnostics.json",
            "national_target_registry.json",
            "national_contract_registry.json",
        ):
            (run / name).write_text("{}")
        (run / f"{cut.CANDIDATE_NAME}.terminal_gates.json").write_text(self._report())
        (run / "build_record.json").write_text(
            json.dumps({"artifacts": {"staging_h5": {"sha256": _sha(b"candidate")}}})
        )
        (run / "logbook-spool").mkdir()
        (run / "logbook-spool" / f"{BUILD_ROW}.json").write_text("{}")

    def score(self, log_path):
        (self.paths.run_dir / "score_vs_incumbent.json").write_text(
            json.dumps(
                {
                    "artifacts": {"candidate": {"sha256": _sha(b"candidate")}},
                    "evaluation": {"verdict": "passed"},
                }
            )
        )

    def certify(self, log_path):
        run = self.paths.run_dir
        (run / f"{cut.CANDIDATE_NAME}.release_cut_gates.json").write_text(
            self._report(shippable=True)
        )
        (run / f"{cut.CANDIDATE_NAME}.release_certification.json").write_text(
            self._report(shippable=True, candidate={"sha256": _sha(b"candidate")})
        )

    def assemble(self, log_path):
        self.paths.release_dir.mkdir(parents=True)
        (self.paths.release_dir / "release_manifest.json").write_text("{}")
        log_path.write_text(
            "assembling...\n"
            + json.dumps({"cut_tag": "microcosm-uk-2024-25-national-x", "n": 1})
        )

    def publish(self, log_path):
        pass


def _runner(tmp_path, fake=None, signer=None):
    paths = _paths(tmp_path)
    fake = fake or FakeBuild(paths, signer=signer)
    sources = {s.key: s for s in cut.load_input_sources()}
    runner = cut.Runner(
        paths=paths, sources=sources, python="py", execute=fake, log=lambda _: None
    )
    return runner, fake


def _all(runner):
    cut.run_spine(runner, staging_local_only=True)
    cut.run_national(runner, retry=False, logbook_prev_row_digest=None)
    cut.run_score(runner)
    cut.run_certify(runner)
    return cut.run_assemble(runner)


def test_certification_chains_on_the_national_build_row(tmp_path):
    runner, fake = _runner(tmp_path)
    cut.run_spine(runner, staging_local_only=True)
    cut.run_national(runner, retry=False, logbook_prev_row_digest="a" * 64)
    cut.run_score(runner)
    cut.run_certify(runner)
    national, certify = fake.argv["national"], fake.argv["certify"]
    assert national[national.index("--logbook-prev-row-digest") + 1] == "a" * 64
    assert certify[certify.index("--logbook-prev-row-digest") + 1] == BUILD_ROW


@pytest.mark.parametrize("rows", [0, 2])
def test_certification_needs_exactly_one_build_row(tmp_path, rows):
    runner, fake = _runner(tmp_path)
    cut.run_spine(runner, staging_local_only=True)
    cut.run_national(runner, retry=False, logbook_prev_row_digest=None)
    cut.run_score(runner)
    spool = runner.paths.run_dir / "logbook-spool"
    for row in spool.glob("*.json"):
        row.unlink()
    for index in range(rows):
        (spool / f"{index:064x}.json").write_text("{}")
    with pytest.raises(cut.CutError, match="Logbook rows"):
        cut.run_certify(runner)
    assert "certify" not in fake.calls


def test_a_complete_run_executes_each_step_once_and_resumes_to_nothing(tmp_path):
    runner, fake = _runner(tmp_path)
    summary = _all(runner)
    assert summary["tag"] == "microcosm-uk-2024-25-national-x"
    assert fake.calls == [
        "spine",
        "national-dry-run",
        "national",
        "score",
        "certify",
        "assemble",
    ]
    _all(runner)  # every done-check now passes
    assert len(fake.calls) == 6
    state = json.loads(runner.paths.state.read_text())
    assert {step: s["status"] for step, s in state["steps"].items()} == {
        step: "done"
        for step in (
            "spine",
            "national-dry-run",
            "national",
            "score",
            "certify",
            "assemble",
        )
    }


@pytest.mark.parametrize(
    "failing", ["spine", "national", "score", "certify", "assemble"]
)
def test_a_failed_step_stops_the_run_and_a_rerun_resumes_there(tmp_path, failing):
    runner, fake = _runner(tmp_path)
    fake.fail = {failing}
    with pytest.raises(cut.CutError, match=f"{failing} exited 1"):
        _all(runner)
    assert json.loads(runner.paths.state.read_text())["steps"][failing]["status"] == (
        "failed"
    )
    before = list(fake.calls)
    fake.fail = set()
    if failing == "national":
        # The tool refuses an occupied --out; the runner moves the failed
        # attempt aside only when asked.
        runner.paths.run_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(cut.CutError, match="--retry-national"):
            _all(runner)
        cut.run_national(runner, retry=True, logbook_prev_row_digest=None)
    _all(runner)
    assert fake.calls.count(failing) == before.count(failing) + 1
    # Steps that finished before the failure are not repeated.
    for done in fake.calls[: before.index(failing)]:
        assert fake.calls.count(done) == 1


def test_an_incomplete_spine_is_never_rebuilt_over(tmp_path):
    runner, fake = _runner(tmp_path)
    runner.paths.spine_dir.mkdir(parents=True)
    runner.paths.spine_h5.write_bytes(b"half a spine")
    with pytest.raises(cut.CutError, match="--retry-spine"):
        cut.run_spine(runner, staging_local_only=True)
    assert fake.calls == []
    assert runner.paths.spine_h5.read_bytes() == b"half a spine"


def test_retrying_the_national_run_keeps_the_failed_attempt(tmp_path):
    runner, fake = _runner(tmp_path)
    cut.run_spine(runner, staging_local_only=True)
    runner.paths.run_dir.mkdir(parents=True)
    (runner.paths.run_dir / "evidence.json").write_text("kept")
    cut.run_national(runner, retry=True, logbook_prev_row_digest=None)
    aside = list(runner.paths.cut.glob("national.failed-*"))
    assert len(aside) == 1 and (aside[0] / "evidence.json").read_text() == "kept"


def test_a_losing_candidate_is_not_certified(tmp_path):
    runner, fake = _runner(tmp_path)

    def failing_score(log_path):
        (runner.paths.run_dir / "score_vs_incumbent.json").write_text(
            json.dumps(
                {
                    "artifacts": {"candidate": {"sha256": _sha(b"candidate")}},
                    "evaluation": {"verdict": "failed"},
                }
            )
        )

    fake.score = failing_score
    cut.run_spine(runner, staging_local_only=True)
    cut.run_national(runner, retry=False, logbook_prev_row_digest=None)
    with pytest.raises(cut.CutError, match="does not pass the incumbent score"):
        cut.run_score(runner)
    assert "certify" not in fake.calls


@given(
    prefix=st.lists(st.text(alphabet=st.characters(blacklist_characters="{}"))),
    tag=st.from_regex(r"[a-z0-9-]{1,40}", fullmatch=True),
)
def test_the_assembly_summary_is_found_after_any_log_noise(prefix, tag):
    text = "\n".join(prefix) + "\n" + json.dumps({"cut_tag": tag, "nested": {"a": 1}})
    assert cut._last_json_object(text)["cut_tag"] == tag


def test_status_reads_without_running_anything(tmp_path):
    paths = _paths(tmp_path)
    out = io.StringIO()
    with redirect_stdout(out):
        cut.status(paths)
    assert "todo spine" in out.getvalue()


def test_a_hosted_file_is_moved_into_place_not_copied(tmp_path):
    """A multi-gigabyte input must not occupy the disk twice."""

    data = b"facts"
    source = cut.InputSource(
        key="chronicle",
        kind="hf_files",
        directory="chronicle",
        files=(cut.PinnedFile("consumer_facts.jsonl", _sha(data)),),
        repo_id="org/inputs",
        repo_type="dataset",
        path_in_repo="chronicle-uk/abc",
    )
    hub = FakeHub(tmp_path, {"chronicle-uk/abc/consumer_facts.jsonl": data})
    result = cut.fetch_inputs([source], tmp_path / "inputs", downloader=hub)
    assert result[0].status == "fetched"
    installed = tmp_path / "inputs/chronicle/consumer_facts.jsonl"
    assert installed.read_bytes() == data
    leftovers = [
        p
        for p in (tmp_path / "inputs").rglob("*")
        if p.is_file() and p != installed and p.name != ".verified.json"
    ]
    assert leftovers == []


def test_a_failed_dry_run_is_rerun(tmp_path):
    runner, fake = _runner(tmp_path)
    cut.run_spine(runner, staging_local_only=True)
    fake.fail = {"national-dry-run"}
    with pytest.raises(cut.CutError, match="national-dry-run exited 1"):
        cut.run_national(runner, retry=False, logbook_prev_row_digest=None)
    fake.fail = set()
    cut.run_national(runner, retry=False, logbook_prev_row_digest=None)
    assert fake.calls.count("national-dry-run") == 2
    assert fake.calls.count("national") == 1


def test_a_retried_spine_keeps_the_signed_failure_report(tmp_path):
    """A gate-blocked spine writes its signed report before any H5; the
    builder deletes sidecars on start, so a retry must move the attempt."""

    runner, fake = _runner(tmp_path)
    paths = runner.paths
    paths.spine_dir.mkdir(parents=True)
    report = paths.spine_h5.with_suffix(".spine_gates.json")
    report.write_text('{"shippable": false, "attestation": {"signature": "s"}}')
    with pytest.raises(cut.CutError, match="--retry-spine"):
        cut.run_spine(runner, staging_local_only=True)
    assert fake.calls == [] and report.exists()
    cut.run_spine(runner, staging_local_only=True, retry=True)
    kept = list(paths.cut.glob("spine.failed-*/" + report.name))
    assert len(kept) == 1 and '"shippable": false' in kept[0].read_text()
    assert cut.spine_done(paths)[0]


def test_retried_score_and_certification_keep_the_earlier_evidence(tmp_path):
    runner, fake = _runner(tmp_path)
    cut.run_spine(runner, staging_local_only=True)
    cut.run_national(runner, retry=False, logbook_prev_row_digest=None)
    receipt = runner.paths.run_dir / "score_vs_incumbent.json"
    receipt.write_text('{"evaluation": {"verdict": "failed"}}')
    cut.run_score(runner)
    kept = list(runner.paths.run_dir.glob("score_vs_incumbent.json.superseded-*"))
    assert len(kept) == 1 and "failed" in kept[0].read_text()
    certification = (
        runner.paths.run_dir / f"{cut.CANDIDATE_NAME}.release_certification.json"
    )
    certification.write_text('{"shippable": false}')
    cut.run_certify(runner)
    kept = list(runner.paths.run_dir.glob(f"{certification.name}.superseded-*"))
    assert len(kept) == 1 and "false" in kept[0].read_text()
    assert cut.certify_done(runner.paths)[0]


def test_a_relative_work_dir_is_made_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = cut._parse(["status", "--label", "x", "--work-dir", "work",
                       "--previous-h5", "prev.h5"])  # fmt: skip
    assert args.work_dir == tmp_path.resolve() / "work"
    assert args.previous_h5 == tmp_path.resolve() / "prev.h5"


def test_a_stale_stage1_comparison_is_redone_and_kept(tmp_path):
    runner, fake = _runner(tmp_path)
    cut.run_spine(runner, staging_local_only=True)
    cut.run_national(runner, retry=False, logbook_prev_row_digest=None)
    previous = tmp_path / "previous.h5"
    previous.write_bytes(b"previous")
    output = runner.paths.diagnostics / "stage1_comparison.json"

    def stage1(argv, cwd, log_path, env):
        fake.calls.append("stage1")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "artifacts": {
                        "candidate": {"sha256": _sha(b"candidate")},
                        "previous": {"sha256": _sha(previous.read_bytes())},
                    }
                }
            )
        )
        return 0

    runner.execute = stage1
    cut.run_stage1(runner, previous)
    cut.run_stage1(runner, previous)  # current: not rerun
    assert fake.calls.count("stage1") == 1
    previous.write_bytes(b"another promoted cut")
    cut.run_stage1(runner, previous)
    assert fake.calls.count("stage1") == 2
    assert len(list(output.parent.glob("stage1_comparison.json.superseded-*"))) == 1


def test_the_cache_never_trusts_bytes_it_did_not_hash(tmp_path):
    """Equal-length replacement with the mtime restored is still re-hashed."""

    import os

    data = b"original-bytes"
    path = tmp_path / "input.tab"
    path.write_bytes(data)
    pin = cut.PinnedFile("input.tab", _sha(data), len(data))
    cache = cut.VerifiedFiles(tmp_path / ".verified.json")
    assert cache.matches(path, pin)
    stat = path.stat()
    path.write_bytes(b"tampered-bytes")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_mtime_ns == stat.st_mtime_ns
    assert not cache.matches(path, pin)


# --------------------------------------------------------------------------- #
# Signing key: a cut resumes only under the key that signed it                  #
# --------------------------------------------------------------------------- #

KEY_A, KEY_B = b"\xa1" * 32, b"\xb2" * 32


def _fp(key: bytes) -> str:
    return hashlib.sha256(key).hexdigest()


def _env(key: bytes) -> dict[str, str]:
    from microcosm.data.gate_signing_keys import UK_GATE_SIGNING_KEY_ENV

    return {UK_GATE_SIGNING_KEY_ENV: base64.b64encode(key).decode()}


@pytest.fixture
def two_builders(monkeypatch):
    """Two registered, active builders' keys, each with its own label."""

    from microcosm.data import gate_signing_keys as registry

    table = {
        _fp(key): registry.GateSigningKey(
            key_sha256=_fp(key),
            label=label,
            status="active",
            custodian=f"{label}'s holder",
            held_in="the test",
            registered="2026-10-26",
            evidence="test",
        )
        for key, label in ((KEY_A, "builder-a"), (KEY_B, "builder-b"))
    }
    monkeypatch.setattr(
        registry, "_registry", lambda country: table if country == "uk" else None
    )
    return table


def test_a_resumed_cut_needs_the_key_that_signed_it(tmp_path, two_builders):
    runner, _ = _runner(tmp_path, signer=_fp(KEY_A))
    cut.run_spine(runner, staging_local_only=True)
    cut.run_national(runner, retry=False, logbook_prev_row_digest=None)
    other = cut.signing_key_check(paths=runner.paths, environ=_env(KEY_B))
    assert not other.ok
    assert "builder-a" in other.detail and _fp(KEY_A) in other.detail
    assert cut.signing_key_check(paths=runner.paths, environ=_env(KEY_A)).ok


def test_an_unfinished_attempt_does_not_bind_the_key(tmp_path, two_builders):
    runner, _ = _runner(tmp_path, signer=_fp(KEY_A))
    cut.run_spine(runner, staging_local_only=True)
    run = runner.paths.run_dir
    run.mkdir(parents=True)
    (run / f"{cut.CANDIDATE_NAME}.terminal_gates.json").write_text(
        json.dumps({"attestation": {"signing_key_sha256": _fp(KEY_B)}})
    )
    # No build record: --retry-national moves this attempt aside before it runs.
    assert cut.signing_key_check(paths=runner.paths, environ=_env(KEY_A)).ok


def test_a_cut_signed_by_two_keys_is_refused(tmp_path, two_builders):
    runner, fake = _runner(tmp_path, signer=_fp(KEY_A))
    cut.run_spine(runner, staging_local_only=True)
    fake.signer = _fp(KEY_B)
    cut.run_national(runner, retry=False, logbook_prev_row_digest=None)
    check = cut.signing_key_check(paths=runner.paths, environ=_env(KEY_A))
    assert not check.ok and "more than one key" in check.detail


def test_a_signer_retired_after_certification_still_finishes(tmp_path, two_builders):
    runner, _ = _runner(tmp_path, signer=_fp(KEY_A))
    cut.run_spine(runner, staging_local_only=True)
    cut.run_national(runner, retry=False, logbook_prev_row_digest=None)
    cut.run_score(runner)
    two_builders[_fp(KEY_A)] = dataclasses.replace(
        two_builders[_fp(KEY_A)], status="retired", retired_on="2026-10-26"
    )
    # The certification still has to be signed, which a retired key may not do.
    assert not cut.signing_key_check(paths=runner.paths, environ=_env(KEY_A)).ok
    cut.run_certify(runner)
    # Assembly and publication only verify.
    assert cut.signing_key_check(paths=runner.paths, environ=_env(KEY_A)).ok


def test_the_receipt_names_the_cut_signer(tmp_path, two_builders, monkeypatch):
    runner, _ = _runner(tmp_path, signer=_fp(KEY_A))
    _all(runner)
    for name, value in _env(KEY_B).items():
        monkeypatch.setenv(name, value)
    cut.write_receipt(runner, {"tag": "t", "published": False})
    receipt = json.loads(runner.paths.receipt.read_text())
    assert receipt["signing_key_sha256"] == _fp(KEY_A)


def test_the_signed_reports_are_the_ones_assembly_reads(tmp_path):
    runner, _ = _runner(tmp_path)
    _all(runner)
    argv = cut.assemble_argv(runner.paths, python="py")

    def value(flag: str) -> Path:
        return Path(argv[argv.index(flag) + 1])

    assert set(cut.cut_signed_reports(runner.paths)) == {
        value("--spine-h5").with_suffix(".spine_gates.json"),
        value("--seam-gate-report"),
        value("--release-cut-gate-json"),
        value("--certification-json"),
    }


@settings(max_examples=60, deadline=None)
@given(signers=st.lists(st.sampled_from([None, "a", "b"]), min_size=1, max_size=4))
def test_reports_bind_one_key_or_none(tmp_path_factory, signers):
    root = tmp_path_factory.mktemp("reports")
    fingerprints = {"a": _fp(KEY_A), "b": _fp(KEY_B)}
    reports = []
    for index, signer in enumerate(signers):
        attestation = (
            {} if signer is None else {"signing_key_sha256": fingerprints[signer]}
        )
        report = root / f"report{index}.json"
        report.write_text(json.dumps({"attestation": attestation}))
        reports.append(report)
    named = {signer for signer in signers if signer}
    if len(named) > 1:
        with pytest.raises(cut.CutError, match="more than one key"):
            cut.reports_signing_key(reports)
    else:
        expected = fingerprints[named.pop()] if named else None
        assert cut.reports_signing_key(reports) == expected


# --------------------------------------------------------------------------- #
# Host checks: nothing specific to one machine unless the operator asks         #
# --------------------------------------------------------------------------- #


def test_no_reservation_is_checked_unless_configured(tmp_path):
    names = [check.name for check in cut.host_checks(tmp_path, min_free_gb=0)]
    assert "host reservation" not in names


def test_a_configured_reservation_tool_that_is_missing_fails(tmp_path):
    checks = cut.host_checks(
        tmp_path,
        min_free_gb=0,
        reservation_tool=tmp_path / "missing.py",
        reservation_caller="builder",
    )
    reservation = next(check for check in checks if check.name == "host reservation")
    assert not reservation.ok and reservation.blocking


@pytest.mark.parametrize("caller, allowed", [("builder", True), ("other", False)])
def test_the_reservation_tool_is_asked_about_the_caller(tmp_path, caller, allowed):
    tool = tmp_path / "reservation.py"
    tool.write_text(
        "import sys\n"
        "raise SystemExit(0 if sys.argv[1:] == ['allowed', 'builder'] else 1)\n"
    )
    checks = cut.host_checks(
        tmp_path, min_free_gb=0, reservation_tool=tool, reservation_caller=caller
    )
    reservation = next(check for check in checks if check.name == "host reservation")
    assert reservation.ok is allowed


@pytest.mark.parametrize(
    "flags", [["--reservation-tool", "r.py"], ["--reservation-caller", "builder"]]
)
def test_a_reservation_needs_both_its_tool_and_its_caller(flags):
    with pytest.raises(SystemExit):
        cut._parse(["preflight", "--label", "t", *flags])


def test_the_signing_key_reaches_only_the_steps_that_sign_or_verify(
    tmp_path, monkeypatch
):
    from microcosm.data.gate_signing_keys import UK_GATE_SIGNING_KEY_ENV

    monkeypatch.setenv(UK_GATE_SIGNING_KEY_ENV, "key-under-test")
    monkeypatch.setenv("HF_TOKEN", "hf-under-test")
    runner, fake = _runner(tmp_path)
    cut.run_spine(runner, staging_local_only=True)
    cut.run_national(runner, retry=False, logbook_prev_row_digest=None)
    # Make the score step run, so its environment is seen too.
    (runner.paths.run_dir / "score_vs_incumbent.json").unlink(missing_ok=True)
    cut.run_score(runner)
    cut.run_certify(runner)
    cut.run_assemble(runner)
    with_key = {
        step for step, env in fake.env.items() if UK_GATE_SIGNING_KEY_ENV in env
    }
    assert with_key == {"spine", "national", "certify", "assemble"}
    assert {"national-dry-run", "score"} <= set(fake.env) - with_key
    assert all(env["HF_TOKEN"] == "hf-under-test" for env in fake.env.values())


def test_the_preflight_test_runs_never_get_the_key(monkeypatch):
    from microcosm.data.gate_signing_keys import UK_GATE_SIGNING_KEY_ENV

    monkeypatch.setenv(UK_GATE_SIGNING_KEY_ENV, "key-under-test")
    seen = []
    monkeypatch.setattr(
        cut.subprocess,
        "run",
        lambda argv, **kw: (
            seen.append(kw["env"]) or cut.subprocess.CompletedProcess(argv, 0, "", "")
        ),
    )
    cut.contract_pin_check("py")
    cut.surfaces_check("py", Path("/facts"))
    assert len(seen) == 2
    assert all(UK_GATE_SIGNING_KEY_ENV not in env for env in seen)
    assert seen[1]["CHRONICLE_UK_FACTS"] == "/facts"
