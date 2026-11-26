"""The policyengine.py certification handoff: routing, guards and rewrites."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import subprocess
from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tools import certify_uk_cut_in_policyengine_py as handoff

NATIONAL = "uk-frs-calibration-attempt-20261021T195620Z-a858c100"
DENSE = "uk-local-candidate-f100-s42-20261019T000432Z-ee5c2dbe"
NEW_DENSE = "uk-local-candidate-f100-s42-20261026T010203Z-0badc0de"
NATIONAL_COMMIT = "4a3b77ac" * 5
DENSE_COMMIT = "9da09946" * 5
NEW_DENSE_COMMIT = "0badc0de" * 5
#: A national build as it was named while UK builds were tagged.
NATIONAL_TAG = "microcosm-uk-2024-25-national-20261021T195620Z-a858c100"


def _build(release, build, commit, model="2.134.4", claims=None):
    return handoff.PublishedBuild(
        release=release,
        build=build,
        commit=commit,
        model=model,
        core="3.33.3",
        claims=(f"=={model}",) if claims is None else claims,
    )


def test_a_build_is_named_by_its_name_and_optionally_its_commit():
    args = handoff.parse(
        ["--national-build", NATIONAL, "--drop-regional-release", "--pepy-repo", "p"]
    )
    assert (args.national_build, args.national_data_commit) == (NATIONAL, None)
    args = handoff.parse(
        ["--national-build", NATIONAL, "--national-data-commit", NATIONAL_COMMIT,
         "--dense-build", DENSE, "--dense-data-commit", DENSE_COMMIT,
         "--pepy-repo", "p"]
    )  # fmt: skip
    assert (args.national_data_commit, args.dense_build, args.dense_data_commit) == (
        NATIONAL_COMMIT,
        DENSE,
        DENSE_COMMIT,
    )


@pytest.mark.parametrize(
    ("national", "message"),
    [
        # policyengine.py refuses every tagged manifest from a repository
        # with releases.json, so a tag no longer names a build.
        (["--national-tag", NATIONAL_TAG], "required: --national-build"),
        (["--national-build", "../" + NATIONAL], "is not a build name"),
        (["--national-build", "releases/x"], "is not a build name"),
        # A branch or a short commit moves or is ambiguous; a pin is a commit.
        (
            ["--national-build", NATIONAL, "--national-data-commit", "main"],
            "--national-data-commit must be a full 40-character commit",
        ),
        (
            [
                "--national-build",
                NATIONAL,
                "--national-data-commit",
                NATIONAL_COMMIT[:9],
            ],
            "--national-data-commit must be a full 40-character commit",
        ),  # fmt: skip
        (
            ["--national-build", NATIONAL, "--dense-data-commit", DENSE_COMMIT],
            "--dense-data-commit goes with --dense-build",
        ),
    ],
)
def test_only_a_safe_build_name_and_a_full_commit_are_taken(national, message, capsys):
    with pytest.raises(SystemExit):
        handoff.parse([*national, "--drop-regional-release", "--pepy-repo", "p"])
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("model", "claims", "version", "covered"),
    [
        ("2.134.4", ("==2.134.4",), "2.134.4", True),
        ("2.132.6", ("==2.132.6",), "2.134.4", False),
        ("2.132.6", ("==2.132.6", "==2.134.4"), "2.134.4", True),
    ],
)
def test_coverage_is_built_with_or_an_exact_claim(model, claims, version, covered):
    build = _build(handoff.DENSE_RELEASE, DENSE, DENSE_COMMIT, model, claims)
    assert handoff.covers(build, version) is covered


def _args(**kwargs):
    base = dict(
        dense_build=None,
        keep_regional_release=False,
        area_coded_national_uri=None,
        drop_regional_release=False,
    )
    return argparse.Namespace(**{**base, **kwargs})


def test_a_build_is_certified_from_its_folder_at_its_commit():
    assert handoff.manifest_uri(
        handoff.NATIONAL_RELEASE, NATIONAL, NATIONAL_COMMIT
    ) == (
        f"hf://dataset/policyengine/populace-uk-private@{NATIONAL_COMMIT}"
        f"/releases/microcosm-uk-2024-25-national/{NATIONAL}/release_manifest.json"
    )
    build = _build(handoff.NATIONAL_RELEASE, NATIONAL, NATIONAL_COMMIT)
    assert build.manifest_uri == handoff.manifest_uri(
        handoff.NATIONAL_RELEASE, NATIONAL, NATIONAL_COMMIT
    )
    assert build.pin == handoff.BuildPin(
        handoff.NATIONAL_RELEASE, NATIONAL, NATIONAL_COMMIT
    )


def test_each_route_produces_its_certify_data_flags():
    record = {
        "regional_source_manifest_uri": handoff.manifest_uri(
            handoff.DENSE_RELEASE, DENSE, DENSE_COMMIT
        )
    }
    new = _build(handoff.DENSE_RELEASE, NEW_DENSE, NEW_DENSE_COMMIT)
    dense = handoff.regional_flags(_args(dense_build=NEW_DENSE), record, new)
    assert dense == ["--regional-manifest-uri", new.manifest_uri]
    kept = handoff.regional_flags(_args(keep_regional_release=True), record, None)
    assert kept == ["--regional-manifest-uri", record["regional_source_manifest_uri"]]
    area = handoff.regional_flags(
        _args(area_coded_national_uri="hf://model/x@1/r"), {}, None
    )
    assert area[-2:] == ["--regional-release-kind", "area_coded_national"]
    dropped = handoff.regional_flags(_args(drop_regional_release=True), record, None)
    assert dropped == ["--drop-regional-release"]
    with pytest.raises(handoff.CertifyError, match="no regional release"):
        handoff.regional_flags(_args(keep_regional_release=True), {}, None)


def test_only_a_folder_manifest_in_the_build_repository_names_a_build():
    uri = handoff.manifest_uri(handoff.DENSE_RELEASE, DENSE, DENSE_COMMIT)
    assert handoff.folder_reference(uri) == handoff.BuildPin(
        handoff.DENSE_RELEASE, DENSE, DENSE_COMMIT
    )
    tagged = (
        "hf://dataset/policyengine/populace-uk-private@microcosm-uk-2024-25-dense-"
        "20261019T000432Z-ee5c2dbe/releases/microcosm-uk-2024-25-dense/"
        "release_manifest.json"
    )
    for other in (
        tagged,
        uri.replace("populace-uk-private", "another-repository"),
        uri.replace("release_manifest.json", "build_manifest.json"),
        "hf://model/x@1/r",
        "",
        None,
    ):
        assert handoff.folder_reference(other) is None


def test_routes_are_mutually_exclusive_and_required():
    repo = ["--pepy-repo", "pepy"]
    with pytest.raises(SystemExit):
        handoff.parse(["--national-build", NATIONAL, *repo])
    with pytest.raises(SystemExit):
        handoff.parse(
            [
                "--national-build",
                NATIONAL,
                "--dense-build",
                DENSE,
                "--drop-regional-release",
                *repo,
            ]
        )
    with pytest.raises(SystemExit):
        handoff.parse(
            [
                "--national-build",
                NATIONAL,
                "--drop-regional-release",
                "--open-pr",
                *repo,
            ]
        )
    # The policyengine.py clone is the operator's to name; no machine's layout.
    with pytest.raises(SystemExit):
        handoff.parse(["--national-build", NATIONAL, "--drop-regional-release"])


def test_the_regional_build_and_commit_are_rewritten_only_as_literals(tmp_path):
    for relative in handoff.CONSTANT_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f'A = "{DENSE}"\nB = "releases/r/{DENSE}/x.h5"\n'
            f'R = "{DENSE_COMMIT}"\nU = "x.h5@{DENSE_COMMIT}"\n'
            f"# {DENSE} at {DENSE_COMMIT} in a comment stays\nC = 'x{DENSE}'\n"
        )
    old = handoff.BuildPin(handoff.DENSE_RELEASE, DENSE, DENSE_COMMIT)
    new = handoff.BuildPin(handoff.DENSE_RELEASE, NEW_DENSE, NEW_DENSE_COMMIT)
    changed = handoff.rewrite_regional_constants(tmp_path, old, new)
    assert sorted(changed) == sorted(handoff.CONSTANT_FILES)
    text = (tmp_path / handoff.CONSTANT_FILES[0]).read_text()
    assert f'A = "{NEW_DENSE}"' in text
    assert f'"releases/r/{NEW_DENSE}/x.h5"' in text
    assert f'R = "{NEW_DENSE_COMMIT}"' in text
    assert f'U = "x.h5@{NEW_DENSE_COMMIT}"' in text
    assert f"# {DENSE} at {DENSE_COMMIT} in a comment stays" in text
    assert handoff.rewrite_regional_constants(tmp_path, new, new) == []
    assert handoff.rewrite_regional_constants(tmp_path, None, new) == []
    assert handoff.rewrite_regional_constants(tmp_path, new, None) == []


@given(
    old=st.from_regex(r"[a-z0-9.\-]{3,20}", fullmatch=True),
    new=st.from_regex(r"[a-z0-9.\-]{3,20}", fullmatch=True),
)
@settings(deadline=None)  # This property performs file I/O on a shared build host.
def test_the_rewrite_is_a_bijection_on_literals(tmp_path_factory, old, new):
    """Rewriting old->new then new->old restores a file of quoted literals."""

    if old == new or old in new or new in old:
        return
    root = tmp_path_factory.mktemp("constants")
    original = f'X = "{old}"\nY = "@{old}/"\n'
    for relative in handoff.CONSTANT_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(original)
    commit = "c" * 40
    before = handoff.BuildPin(handoff.DENSE_RELEASE, old, commit)
    after = handoff.BuildPin(handoff.DENSE_RELEASE, new, commit)
    handoff.rewrite_regional_constants(root, before, after)
    handoff.rewrite_regional_constants(root, after, before)
    for relative in handoff.CONSTANT_FILES:
        assert (root / relative).read_text() == original


def test_dropped_overlays_are_reported():
    old = {"dataset_overlays": {"uk": {"frs_2024_25": {}, "enhanced_frs_2024_25": {}}}}
    new = {"dataset_overlays": {"uk": {"frs_2024_25": {}}}}
    assert handoff.overlay_diff(old, new) == ["enhanced_frs_2024_25"]
    assert handoff.overlay_diff({}, {}) == []


def test_the_pr_body_leaves_every_judgment_to_a_person():
    body = handoff.pr_body(
        {
            "national_build": NATIONAL,
            "national_commit": NATIONAL_COMMIT,
            "model": "2.134.4",
            "core": "3.33.3",
            "regional_summary": "`--drop-regional-release`",
            "basis": "built_with_model_package",
            "microcosm_commit": "a" * 40,
            "commands": ["python scripts/bundle.py certify-data ..."],
            "dropped_overlays": ["enhanced_frs_2024_25"],
            "warnings": ["regional release: certifying on a claim"],
            "tests_failed": ["tests/test_models.py::test_x"],
            "uc_floor_passed": False,
        }
    )
    for needed in (
        f"Microcosm UK national build `{NATIONAL}`",
        f"`{NATIONAL_COMMIT}`",
        "UC floor acceptance",
        "enhanced_frs_2024_25",
        "Certification warnings",
        "tests/test_models.py::test_x",
        "One writer",
        "- [ ]",
    ):
        assert needed in body
    assert NATIONAL_TAG not in body and "cut" not in body.split("\n", 1)[0]


def test_the_pr_body_asks_for_a_check_when_old_builds_shared_a_commit():
    """policyengine.py's rewrite moves every quoted copy of the old national
    commit, so a local-area constant that shared it moves too."""

    context = {
        "national_build": NATIONAL,
        "national_commit": NATIONAL_COMMIT,
        "model": "2.134.4",
        "core": "3.33.3",
        "regional_summary": "`--keep-regional-release`",
        "microcosm_commit": "a" * 40,
        "commands": [],
    }
    assert "shared commit" not in handoff.pr_body(context)
    body = handoff.pr_body({**context, "shared_old_commit": DENSE_COMMIT})
    assert f"shared commit `{DENSE_COMMIT}`" in body
    assert "- [ ] **Local-area revision constants.**" in body


MISMATCH = [
    "--accept-regional-model-mismatch",
    "area receipts are uncertified (#685)",
    "--audited-by",
    "A Reviewer",
    "--regional-limitation",
    "VAT predates the factor fix",
]


def _dense_route(
    monkeypatch, tmp_path, *, audit_support=None, regional_uri=None, printed=None
):
    """A national build on 2.134.4 and a local-area build on 2.132.6, offline.

    Returns the builds read (release, build, commit) and the local-area
    builds whose microdata was downloaded.
    """

    builds = {
        NATIONAL: _build(handoff.NATIONAL_RELEASE, NATIONAL, NATIONAL_COMMIT),
        DENSE: _build(handoff.DENSE_RELEASE, DENSE, DENSE_COMMIT, "2.132.6"),
    }
    reads, downloads = [], []

    def read(release, build, *, commit=None, **kwargs):
        reads.append((release, build, commit))
        found = builds[build]
        return found if commit is None else _replace(found, commit=commit)

    monkeypatch.setattr(handoff, "read_published_build", read)
    monkeypatch.setattr(handoff, "writer_check", lambda build, repo: [])
    monkeypatch.setattr(
        handoff, "regional_audit_support", lambda repo, base: audit_support
    )
    monkeypatch.setattr(
        handoff,
        "fetch_regional_microdata",
        lambda build, **kw: downloads.append(build.build) or {},
    )
    monkeypatch.setattr(
        handoff.subprocess,
        "run",
        lambda *a, **k: handoff.subprocess.CompletedProcess(a, 0, "", ""),
    )
    if regional_uri is None:
        regional_uri = handoff.manifest_uri(handoff.DENSE_RELEASE, DENSE, DENSE_COMMIT)
    bundle = json.dumps(
        {"data_releases": {"uk": {"regional_source_manifest_uri": regional_uri}}}
    )

    def check_output(argv, **kwargs):
        if argv[:2] == ["git", "show"]:
            return bundle
        if argv == ["git", "rev-parse", "HEAD"]:
            return "a" * 40
        return "b" * 40

    monkeypatch.setattr(handoff.subprocess, "check_output", check_output)
    return reads, downloads


def _replace(build, **changes):
    return dataclasses.replace(build, **changes)


def test_a_dry_run_refuses_an_uncovered_dense_build_before_writing(
    monkeypatch, tmp_path
):
    reads, downloads = _dense_route(monkeypatch, tmp_path)
    base = ["--national-build", NATIONAL, "--keep-regional-release", "--dry-run",
            "--pepy-repo", str(tmp_path)]  # fmt: skip
    with pytest.raises(
        handoff.CertifyError, match="built with policyengine-uk 2.132.6"
    ):
        handoff.certify(handoff.parse(base))
    # An accepted mismatch plans the audit; a dry run downloads nothing.
    assert handoff.certify(handoff.parse([*base, *MISMATCH])) == 0
    assert downloads == []
    another = handoff.parse(
        ["--national-build", NATIONAL, "--dense-build", DENSE, "--dry-run",
         "--pepy-repo", str(tmp_path)]
    )  # fmt: skip
    monkeypatch.setattr(handoff, "writer_check", lambda build, repo: ["#700 open"])
    with pytest.raises(handoff.CertifyError, match="already being certified"):
        handoff.certify(another)


def test_a_kept_local_area_build_is_read_at_the_bundles_commit(
    monkeypatch, tmp_path, capsys
):
    """The bundle pins its local-area build by folder and commit; the route
    reads that build there, so its model check and audit still run."""

    reads, _ = _dense_route(monkeypatch, tmp_path)
    args = handoff.parse(
        ["--national-build", NATIONAL, "--keep-regional-release", "--dry-run",
         "--pepy-repo", str(tmp_path), *MISMATCH]
    )  # fmt: skip
    assert handoff.certify(args) == 0
    assert reads == [
        (handoff.NATIONAL_RELEASE, NATIONAL, None),
        (handoff.DENSE_RELEASE, DENSE, DENSE_COMMIT),
    ]
    output = capsys.readouterr().out
    plan = json.loads(output[output.find("{") :])
    assert plan["regional"].endswith("(regional built on 2.132.6; mismatch accepted)")


def test_a_named_build_and_its_commit_reach_certify_data(monkeypatch, tmp_path, capsys):
    reads, _ = _dense_route(monkeypatch, tmp_path)
    args = handoff.parse(
        ["--national-build", NATIONAL, "--national-data-commit", "1" * 40,
         "--dense-build", DENSE, "--dense-data-commit", "2" * 40,
         "--accept-regional-model-mismatch", "recorded", "--audited-by", "A",
         "--regional-limitation", "L", "--dry-run", "--pepy-repo", str(tmp_path)]
    )  # fmt: skip
    assert handoff.certify(args) == 0
    assert reads == [
        (handoff.NATIONAL_RELEASE, NATIONAL, "1" * 40),
        (handoff.DENSE_RELEASE, DENSE, "2" * 40),
    ]
    output = capsys.readouterr().out
    plan = json.loads(output[output.find("{") :])
    assert plan["national_build"] == NATIONAL
    assert plan["national_manifest_uri"] == handoff.manifest_uri(
        handoff.NATIONAL_RELEASE, NATIONAL, "1" * 40
    )
    assert (
        handoff.manifest_uri(handoff.DENSE_RELEASE, DENSE, "2" * 40)
        in (plan["regional"])
    )
    assert plan["branch"] == "certify-uk-a858c100"


def test_a_kept_tagged_regional_release_is_refused_before_it_is_read(
    monkeypatch, tmp_path
):
    """policyengine.py refuses a tagged manifest from a repository with
    releases.json, so keeping one could never certify."""

    tagged = (
        "hf://dataset/policyengine/populace-uk-private@microcosm-uk-2024-25-dense-"
        "20261019T000432Z-ee5c2dbe/releases/microcosm-uk-2024-25-dense/"
        "release_manifest.json"
    )
    reads, _ = _dense_route(monkeypatch, tmp_path, regional_uri=tagged)
    args = handoff.parse(
        ["--national-build", NATIONAL, "--keep-regional-release", "--dry-run",
         "--pepy-repo", str(tmp_path)]
    )  # fmt: skip
    with pytest.raises(handoff.CertifyError, match="not a build in its own folder"):
        handoff.certify(args)
    assert [read[1] for read in reads] == [NATIONAL]


def test_a_base_without_the_audit_refuses_before_any_download(monkeypatch, tmp_path):
    _, downloads = _dense_route(
        monkeypatch, tmp_path, audit_support=f"it has no {handoff.AUDIT_SCRIPT}"
    )
    args = handoff.parse(
        ["--national-build", NATIONAL, "--keep-regional-release",
         "--pepy-repo", str(tmp_path), "--skip-tests", *MISMATCH]
    )  # fmt: skip
    with pytest.raises(handoff.CertifyError, match="policyengine.py#699"):
        handoff.certify(args)
    assert downloads == []


@pytest.mark.parametrize(
    "drop",
    [
        ["--audited-by", "A Reviewer"],
        ["--regional-limitation", "VAT predates the factor fix"],
    ],
)
def test_an_accepted_mismatch_needs_an_auditor_and_a_limitation(drop):
    flags = list(MISMATCH)
    index = flags.index(drop[0])
    del flags[index : index + 2]
    with pytest.raises(SystemExit):
        handoff.parse(
            ["--national-build", NATIONAL, "--keep-regional-release",
             "--pepy-repo", "p", *flags]
        )  # fmt: skip
    with pytest.raises(SystemExit):
        handoff.parse(
            ["--national-build", NATIONAL, "--keep-regional-release",
             "--pepy-repo", "p", *drop]
        )  # fmt: skip


@pytest.mark.parametrize("flag", MISMATCH[0::2])
@pytest.mark.parametrize("text", ["   ", "\u200b"])
def test_a_mismatch_needs_readable_text_in_every_field(flag, text):
    """policyengine.py refuses these later; refuse them before any work."""

    flags = list(MISMATCH)
    flags[flags.index(flag) + 1] = text
    with pytest.raises(SystemExit):
        handoff.parse(
            ["--national-build", NATIONAL, "--keep-regional-release",
             "--pepy-repo", "p", *flags]
        )  # fmt: skip
    assert handoff.readable("ok") and not handoff.readable(text)


def test_the_build_folder_support_check_reads_the_base(monkeypatch, tmp_path):
    answers, seen = {"grep": 0}, []

    def run(argv, **kwargs):
        assert kwargs["cwd"] == tmp_path
        seen.append(argv)
        return handoff.subprocess.CompletedProcess(argv, answers["grep"], b"", b"")

    monkeypatch.setattr(handoff.subprocess, "run", run)
    assert handoff.build_folder_support(tmp_path, "c" * 40) is None
    assert seen == [
        ["git", "grep", "-q", "-e", "def read_build_folder", "c" * 40, "--",
         "src/policyengine/provenance/manifest.py"]
    ]  # fmt: skip
    answers["grep"] = 1
    assert "read_build_folder" in handoff.build_folder_support(tmp_path, "c" * 40)


def test_a_base_that_cannot_read_build_folders_refuses_before_any_read(
    monkeypatch, tmp_path
):
    """A .py base without #683 certifies only tags, which the build repository
    no longer has, and an accepted mismatch would download every regional
    file and set up a worktree before refusing."""

    reads, downloads = _dense_route(monkeypatch, tmp_path)

    def run(argv, **kwargs):
        code = 1 if "def read_build_folder" in argv else 0
        return subprocess.CompletedProcess(argv, code, "", "")

    monkeypatch.setattr(handoff.subprocess, "run", run)
    args = handoff.parse(
        ["--national-build", NATIONAL, "--dense-build", DENSE, "--dry-run",
         "--pepy-repo", str(tmp_path), *MISMATCH]
    )  # fmt: skip
    with pytest.raises(handoff.CertifyError, match="policyengine.py#683"):
        handoff.certify(args)
    assert (reads, downloads) == ([], [])


def test_the_audit_support_check_reads_the_base(monkeypatch, tmp_path):
    answers = {"cat-file": 0, "grep": 0}

    def run(argv, **kwargs):
        assert argv[0] == "git" and kwargs["cwd"] == tmp_path
        return handoff.subprocess.CompletedProcess(argv, answers[argv[1]], b"", b"")

    monkeypatch.setattr(handoff.subprocess, "run", run)
    assert handoff.regional_audit_support(tmp_path, "c" * 40) is None
    answers["grep"] = 1
    assert "--regional-input-surface-evidence" in handoff.regional_audit_support(
        tmp_path, "c" * 40
    )
    answers["cat-file"] = 128
    assert handoff.AUDIT_SCRIPT in handoff.regional_audit_support(tmp_path, "c" * 40)


class FakeHub:
    """The Hub calls the tool makes, answered from ``files``.

    ``files`` maps ``(repo_id, path, revision)`` to bytes, ``last_commits``
    maps a path to its last commit on ``main``, and ``history`` lists the
    commits in ``main``'s history. ``failures`` maps a file name, or
    ``"paths-info"``, to the exception the Hub raises for it.
    """

    def __init__(self, root, files, last_commits=None, history=()):
        self.root, self.files = root, files
        self.last_commits = last_commits or {}
        self.history = list(history)
        self.failures = {}
        self.downloads, self.listings = [], []

    def hf_hub_download(self, repo_id, filename, *, repo_type=None, revision=None,
                        **options):  # fmt: skip
        assert repo_type == "dataset"
        self.downloads.append((repo_id, filename, revision, options))
        if filename in self.failures:
            raise self.failures[filename]
        key = (repo_id, filename, revision)
        if key not in self.files:
            raise FileNotFoundError(f"{repo_id} has no {filename} at {revision}")
        path = self.root / hashlib.sha256(repr(key).encode()).hexdigest()
        path.write_bytes(self.files[key])
        return str(path)

    def get_paths_info(self, repo_id, paths, *, expand=False, revision=None,
                       repo_type=None):  # fmt: skip
        self.listings.append((repo_id, list(paths), expand, revision, repo_type))
        if "paths-info" in self.failures:
            raise self.failures["paths-info"]
        return [
            SimpleNamespace(path=path, last_commit=SimpleNamespace(oid=commit))
            for path, commit in self.last_commits.items()
            if path in paths
        ]

    def list_repo_commits(self, repo_id, *, repo_type=None, revision=None):
        assert (repo_id, repo_type, revision) == (REPO, "dataset", "main")
        if "commits" in self.failures:
            raise self.failures["commits"]
        return [SimpleNamespace(commit_id=commit) for commit in self.history]


REPO = "policyengine/populace-uk-private"


def _folder(release, build):
    return f"releases/{release}/{build}"


def _releases_record(withdrawn=()):
    def release(role, build):
        return {
            "role": role,
            "current": None,
            "builds": {
                build: {
                    "published_at": "2026-11-09T15:41:23+00:00",
                    "withdrawn": build in withdrawn,
                }
            },
        }

    return json.dumps(
        {
            "schema_version": 1,
            "default": handoff.NATIONAL_RELEASE,
            "releases": {
                handoff.NATIONAL_RELEASE: release("national", NATIONAL),
                handoff.DENSE_RELEASE: release("local_area", DENSE),
            },
        }
    ).encode()


def _folder_manifest(release, build, **changes):
    manifest = {
        "release": release,
        "build": build,
        "built_at": "2026-11-09T15:00:00+00:00",
        "built_with_model_package": {"name": "policyengine-uk", "version": "2.179.2"},
        "built_with_core_package": {"name": "policyengine-core", "version": "3.34.10"},
        "compatible_model_packages": [
            {"name": "policyengine-uk", "specifier": "==2.179.2"}
        ],
        "artifacts": {},
    }
    return json.dumps({**manifest, **changes}).encode()


def _national_hub(
    tmp_path, *, withdrawn=(), manifest=None, at=NATIONAL_COMMIT, history=None
):
    folder = _folder(handoff.NATIONAL_RELEASE, NATIONAL)
    files = {
        (REPO, "releases.json", "main"): _releases_record(withdrawn),
        (REPO, f"{folder}/release_manifest.json", at): manifest
        or _folder_manifest(handoff.NATIONAL_RELEASE, NATIONAL),
    }
    return FakeHub(
        tmp_path,
        files,
        {f"{folder}/release_manifest.json": NATIONAL_COMMIT},
        history=[NATIONAL_COMMIT, at] if history is None else history,
    )


def test_a_listed_build_is_read_at_the_commit_that_published_its_folder(tmp_path):
    hub = _national_hub(tmp_path)
    build = handoff.read_published_build(handoff.NATIONAL_RELEASE, NATIONAL, api=hub)
    assert build == handoff.PublishedBuild(
        release=handoff.NATIONAL_RELEASE,
        build=NATIONAL,
        commit=NATIONAL_COMMIT,
        model="2.179.2",
        core="3.34.10",
        claims=("==2.179.2",),
    )
    folder = _folder(handoff.NATIONAL_RELEASE, NATIONAL)
    assert hub.listings == [
        (REPO, [f"{folder}/release_manifest.json"], True, "main", "dataset")
    ]
    record, manifest = hub.downloads
    # A record cached before a withdrawal must not offer the build.
    assert record[:3] == (REPO, "releases.json", "main")
    assert record[3].get("force_download") is True
    assert manifest[:3] == (REPO, f"{folder}/release_manifest.json", NATIONAL_COMMIT)


def test_an_operators_commit_replaces_the_publishing_commit(tmp_path):
    other = "5" * 40
    hub = _national_hub(tmp_path, at=other)
    build = handoff.read_published_build(
        handoff.NATIONAL_RELEASE, NATIONAL, commit=other, api=hub
    )
    assert build.commit == other
    assert hub.listings == []


def test_an_operators_commit_must_be_in_mains_history(tmp_path):
    """A commit on another ref, such as a Hub pull request's, may hold an
    edited copy of the folder that main never held."""

    other = "5" * 40
    hub = _national_hub(tmp_path, at=other, history=[NATIONAL_COMMIT])
    with pytest.raises(handoff.CertifyError, match="is not in main's history"):
        handoff.read_published_build(
            handoff.NATIONAL_RELEASE, NATIONAL, commit=other, api=hub
        )
    assert [download[1] for download in hub.downloads] == ["releases.json"]


@pytest.mark.parametrize(
    ("failing", "message"),
    [
        ("releases.json", "could not read releases.json"),
        ("paths-info", "could not ask the Hub which commit published"),
    ],
)
def test_a_hub_failure_is_a_refusal(tmp_path, failing, message):
    hub = _national_hub(tmp_path)
    hub.failures[failing] = RuntimeError("503 Service Unavailable")
    with pytest.raises(handoff.CertifyError, match=message) as refusal:
        handoff.read_published_build(handoff.NATIONAL_RELEASE, NATIONAL, api=hub)
    assert "503 Service Unavailable" in str(refusal.value)


def test_a_failing_history_listing_is_a_refusal(tmp_path):
    other = "5" * 40
    hub = _national_hub(tmp_path, at=other)
    hub.failures["commits"] = RuntimeError("503 Service Unavailable")
    with pytest.raises(handoff.CertifyError, match="could not list main's history"):
        handoff.read_published_build(
            handoff.NATIONAL_RELEASE, NATIONAL, commit=other, api=hub
        )


def test_an_unreadable_releases_record_is_named_in_the_refusal(tmp_path):
    hub = _national_hub(tmp_path)
    hub.files[(REPO, "releases.json", "main")] = b"not json"
    with pytest.raises(
        handoff.CertifyError,
        match="releases.json at main in policyengine/populace-uk-private: it is "
        "not a readable JSON document",
    ):
        handoff.read_published_build(handoff.NATIONAL_RELEASE, NATIONAL, api=hub)


def _bundle_at(worktree, revision, regional_uri):
    path = worktree / "src/policyengine/data/bundle/manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "data_package": {"release_manifest_revision": revision},
        "regional_source_manifest_uri": regional_uri,
    }
    path.write_text(json.dumps({"data_releases": {"uk": record}}))
    return "src/policyengine/data/bundle/manifest.json"


def test_a_resumed_worktree_must_certify_the_same_commits(tmp_path):
    """A resume at another commit, or with another regional route, would keep
    the first attempt's constants while certify-data writes this run's pins
    into the bundle."""

    national = _build(handoff.NATIONAL_RELEASE, NATIONAL, NATIONAL_COMMIT)
    dense = _build(handoff.DENSE_RELEASE, DENSE, DENSE_COMMIT)
    relative = _bundle_at(tmp_path, NATIONAL_COMMIT, dense.manifest_uri)
    handoff.check_resumed_pins(tmp_path, [relative], national, dense.manifest_uri)
    # A worktree whose bundle is not yet modified holds nothing to compare.
    moved = _replace(national, commit="5" * 40)
    handoff.check_resumed_pins(tmp_path, [], moved, dense.manifest_uri)
    with pytest.raises(handoff.CertifyError, match="national build at"):
        handoff.check_resumed_pins(tmp_path, [relative], moved, dense.manifest_uri)
    other = _replace(dense, commit="6" * 40)
    # Another local-area build, an area-coded release, or no routing at all.
    for regional in (
        other.manifest_uri,
        "hf://dataset/policyengine/another@1/releases/r/release_manifest.json",
        None,
    ):
        with pytest.raises(handoff.CertifyError, match="routes regions to"):
            handoff.check_resumed_pins(tmp_path, [relative], national, regional)
    relative = _bundle_at(tmp_path, NATIONAL_COMMIT, None)
    handoff.check_resumed_pins(tmp_path, [relative], national, None)
    with pytest.raises(handoff.CertifyError, match="routes regions to None"):
        handoff.check_resumed_pins(tmp_path, [relative], national, dense.manifest_uri)


@pytest.mark.parametrize(
    ("hub_changes", "build", "message"),
    [
        ({"withdrawn": (NATIONAL,)}, NATIONAL, "has been withdrawn"),
        ({}, NATIONAL + "-x", "does not list build"),
        (
            {"manifest": _folder_manifest(handoff.NATIONAL_RELEASE, DENSE)},
            NATIONAL,
            f"names release 'microcosm-uk-2024-25-national' and build '{DENSE}'",
        ),
        (
            {
                "manifest": _folder_manifest(
                    handoff.NATIONAL_RELEASE, NATIONAL, built_with_core_package={}
                )
            },
            NATIONAL,
            "no built_with_core_package version",
        ),
        (
            {"manifest": json.dumps({"build": {"cut_tag": NATIONAL_TAG}}).encode()},
            NATIONAL,
            "names release None",
        ),
    ],
)
def test_a_build_policyengine_py_would_refuse_is_refused_first(
    tmp_path, hub_changes, build, message
):
    hub = _national_hub(tmp_path, **hub_changes)
    with pytest.raises(handoff.CertifyError, match=re.escape(message)):
        handoff.read_published_build(handoff.NATIONAL_RELEASE, build, api=hub)


def test_a_build_with_no_folder_on_main_is_refused(tmp_path):
    hub = _national_hub(tmp_path)
    hub.last_commits = {}
    with pytest.raises(handoff.CertifyError, match="has no releases/"):
        handoff.read_published_build(handoff.NATIONAL_RELEASE, NATIONAL, api=hub)
    hub = _national_hub(tmp_path, at="6" * 40, history=["7" * 40])
    with pytest.raises(handoff.CertifyError, match="could not read"):
        handoff.read_published_build(
            handoff.NATIONAL_RELEASE, NATIONAL, commit="7" * 40, api=hub
        )


def _dense_hub(tmp_path, artifacts, data):
    folder = _folder(handoff.DENSE_RELEASE, DENSE)
    manifest = _folder_manifest(handoff.DENSE_RELEASE, DENSE, artifacts=artifacts)
    return FakeHub(
        tmp_path,
        {(REPO, f"{folder}/release_manifest.json", DENSE_COMMIT): manifest, **data},
    )


def test_every_regional_microdata_file_is_fetched_and_checked(tmp_path):
    folder = _folder(handoff.DENSE_RELEASE, DENSE)
    digest = {name: hashlib.sha256(name.encode()).hexdigest()
              for name in ("dense", "other", "far")}  # fmt: skip
    artifacts = {
        # The build's own files are bare names, alone or after the folder,
        # with no revision: they are read in the folder at the build's commit.
        "microcosm_uk_2024_25_dense": {
            "kind": "microdata", "path": "dense.h5", "repo_id": REPO,
            "role": "dataset", "sha256": digest["dense"],
        },
        "second": {
            "kind": "microdata", "path": f"{folder}/other.H5", "repo_id": REPO,
            "sha256": digest["other"],
        },
        # A file in another repository keeps its own path and revision.
        "elsewhere": {
            "kind": "microdata", "path": "x/far.h5",
            "repo_id": "policyengine/another", "revision": "r9",
            "sha256": digest["far"],
        },
        "table": {"kind": "microdata", "path": "notes.csv", "repo_id": REPO,
                  "sha256": "0" * 64},
        "receipt": {"kind": "evidence", "path": "dense.h5", "repo_id": REPO,
                    "sha256": "0" * 64},
    }  # fmt: skip
    data = {
        (REPO, f"{folder}/dense.h5", DENSE_COMMIT): b"dense",
        (REPO, f"{folder}/other.H5", DENSE_COMMIT): b"other",
        ("policyengine/another", "x/far.h5", "r9"): b"far",
    }
    hub = _dense_hub(tmp_path, artifacts, data)
    dense = _build(handoff.DENSE_RELEASE, DENSE, DENSE_COMMIT)
    found = handoff.fetch_regional_microdata(
        dense, timings=tmp_path / "t.jsonl", api=hub
    )
    assert sorted(found) == ["elsewhere", "microcosm_uk_2024_25_dense", "second"]
    assert {key: found[key].read_bytes() for key in found} == {
        "microcosm_uk_2024_25_dense": b"dense",
        "second": b"other",
        "elsewhere": b"far",
    }
    assert [download[:3] for download in hub.downloads] == [
        (REPO, f"{folder}/release_manifest.json", DENSE_COMMIT),
        ("policyengine/another", "x/far.h5", "r9"),
        (REPO, f"{folder}/dense.h5", DENSE_COMMIT),
        (REPO, f"{folder}/other.H5", DENSE_COMMIT),
    ]
    hub.files[(REPO, f"{folder}/other.H5", DENSE_COMMIT)] = b"changed"
    with pytest.raises(handoff.CertifyError, match="not the manifest's"):
        handoff.fetch_regional_microdata(dense, timings=tmp_path / "t.jsonl", api=hub)


def test_a_failing_regional_download_is_a_refusal(tmp_path):
    folder = _folder(handoff.DENSE_RELEASE, DENSE)
    entry = {"kind": "microdata", "path": "dense.h5", "repo_id": REPO,
             "sha256": "0" * 64}  # fmt: skip
    hub = _dense_hub(tmp_path, {"microcosm_uk_2024_25_dense": entry}, {})
    hub.failures[f"{folder}/dense.h5"] = RuntimeError("503 Service Unavailable")
    dense = _build(handoff.DENSE_RELEASE, DENSE, DENSE_COMMIT)
    with pytest.raises(handoff.CertifyError, match="could not download") as refusal:
        handoff.fetch_regional_microdata(dense, timings=tmp_path / "t.jsonl", api=hub)
    assert f"{folder}/dense.h5 at {DENSE_COMMIT}" in str(refusal.value)
    hub = _dense_hub(tmp_path, {}, {})
    hub.failures[f"{folder}/release_manifest.json"] = RuntimeError("timed out")
    with pytest.raises(handoff.CertifyError, match="could not download"):
        handoff.fetch_regional_microdata(dense, timings=tmp_path / "t.jsonl", api=hub)


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        ({"path": "sub/dense.h5", "repo_id": REPO}, "not a file directly inside"),
        (
            {
                "path": f"releases/microcosm-uk-2024-25-dense/{DENSE}/../x/dense.h5",
                "repo_id": REPO,
            },
            "not a file directly inside",
        ),  # fmt: skip
        ({"path": "far.h5", "repo_id": "policyengine/another"}, "names no revision"),
        ({"path": "dense.h5", "repo_id": REPO, "sha256": None}, "no sha256"),
    ],
)
def test_a_regional_file_outside_its_pin_is_refused(tmp_path, artifact, message):
    entry = {"kind": "microdata", "sha256": "0" * 64, **artifact}
    hub = _dense_hub(tmp_path, {"microcosm_uk_2024_25_dense": entry}, {})
    dense = _build(handoff.DENSE_RELEASE, DENSE, DENSE_COMMIT)
    with pytest.raises(handoff.CertifyError, match=message):
        handoff.fetch_regional_microdata(dense, timings=tmp_path / "t.jsonl", api=hub)
    assert len(hub.downloads) == 1


def test_the_audit_argv_passes_every_file_and_limitation(tmp_path):
    argv = handoff.regional_audit_argv(
        "py",
        manifest_uri="hf://x",
        artifacts={"b": tmp_path / "b.h5", "a": tmp_path / "a.h5"},
        model="2.147.2",
        audited_by="A Reviewer",
        limitations=["one", "two"],
        output=tmp_path / "evidence.json",
    )
    assert argv[:4] == ["py", handoff.AUDIT_SCRIPT, "--regional-manifest-uri", "hf://x"]
    assert argv.count("--artifact") == 2 and argv.count("--limitation") == 2
    assert argv[argv.index("--artifact") + 1] == f"a={tmp_path / 'a.h5'}"
    assert argv[-2:] == ["--output", str(tmp_path / "evidence.json")]
    help_text = "usage: --regional-manifest-uri --artifact --model-version --output"
    assert handoff.undeclared_flags(help_text, argv) == ["--audited-by", "--limitation"]


GOOD_EVIDENCE = {
    "schema_version": 2,
    "model_version": "2.147.2",
    "build_model_version": "2.132.6",
    "dropped_columns": [],
    "defaulted_inputs": ["local_bus_uncapped_single_fare"],
    "never_modelled_columns": ["constituency_code_oa"],
    "audited_by": "A Reviewer",
    "limitations": ["VAT predates the factor fix"],
}


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"schema_version": 1}, "target model only"),
        ({"model_version": "2.151.0"}, "model_version"),
        ({"build_model_version": "2.144.0"}, "build_model_version"),
        ({"dropped_columns": ["age"]}, "would lose them: age"),
    ],
)
def test_only_build_relative_drop_free_evidence_is_used(tmp_path, changes, message):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps({**GOOD_EVIDENCE, **changes}))
    with pytest.raises(handoff.CertifyError, match=message):
        handoff.read_regional_evidence(path, model="2.147.2", build_model="2.132.6")


def test_build_relative_evidence_is_read(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(GOOD_EVIDENCE))
    evidence = handoff.read_regional_evidence(
        path, model="2.147.2", build_model="2.132.6"
    )
    assert evidence["defaulted_inputs"] == ["local_bus_uncapped_single_fare"]
    with pytest.raises(handoff.CertifyError, match="no readable"):
        handoff.read_regional_evidence(
            tmp_path / "missing.json", model="2.147.2", build_model="2.132.6"
        )


def test_the_pr_body_lists_the_audited_area_input_surface():
    body = handoff.pr_body(
        {
            "national_build": NATIONAL,
            "national_commit": NATIONAL_COMMIT,
            "model": "2.147.2",
            "core": "3.33.3",
            "regional_summary": "`--keep-regional-release`",
            "microcosm_commit": "a" * 40,
            "commands": [],
            "regional_evidence": {
                "build_model_version": "2.132.6",
                "audited_by": "A Reviewer",
                "dropped_columns": [],
                "defaulted_inputs": ["local_bus_uncapped_single_fare"],
                "never_modelled_columns": ["constituency_code_oa", "la_code_oa"],
                "limitations": ["VAT predates the factor fix"],
            },
        }
    )
    assert "built on 2.132.6, audited by A Reviewer" in body
    assert "local_bus_uncapped_single_fare" in body
    assert "2 stored columns neither model defines" in body
    assert "VAT predates the factor fix" in body


def test_the_resource_limits_are_the_operators_to_set(monkeypatch, tmp_path):
    """No load limit unless the operator sets one; the disk floor is a default."""

    monkeypatch.setattr(handoff.os, "getloadavg", lambda: (12.0, 0.0, 0.0))
    args = handoff.parse(
        ["--national-build", NATIONAL, "--drop-regional-release", "--pepy-repo", "p"]
    )
    assert args.max_load1 is None
    assert handoff.resource_guard(tmp_path, min_free_bytes=0)["load1"] == 12.0
    assert handoff.resource_guard(tmp_path, max_load1=20, min_free_bytes=0)
    with pytest.raises(handoff.CertifyError, match="maximum 8"):
        handoff.resource_guard(tmp_path, max_load1=8, min_free_bytes=0)
    shell = handoff.Shell(tmp_path, tmp_path / "run.log", max_load1=8, min_free_bytes=0)
    spawned = []
    monkeypatch.setattr(handoff, "run_logged", lambda *a, **k: spawned.append(a))
    with pytest.raises(handoff.CertifyError, match="maximum 8"):
        shell(["heavy"])
    assert not spawned


def _writer_listing(monkeypatch, *, prs=(), heads=(), ls_remote_code=0):
    """Fake ``gh pr list`` and ``git ls-remote`` for the one-writer check."""

    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[:3] == ["gh", "pr", "list"]:
            return subprocess.CompletedProcess(argv, 0, json.dumps(list(prs)), "")
        lines = "".join(f"{'0' * 40}\trefs/heads/{head}\n" for head in heads)
        return subprocess.CompletedProcess(argv, ls_remote_code, lines, "no access")

    monkeypatch.setattr(handoff.subprocess, "run", run)
    return calls


@pytest.mark.parametrize(
    "head",
    [
        "certify-uk-a858c100",
        "certify-uk-a858c100-national",
        "hub/certify-uk-a858c100",
    ],
)
def test_any_branch_naming_the_cut_is_another_writer(monkeypatch, tmp_path, head):
    _writer_listing(monkeypatch, heads=["main", head])
    assert handoff.writer_check(NATIONAL, tmp_path) == [
        f"branch {head} exists on origin"
    ]


def test_an_open_pull_request_naming_the_cut_is_another_writer(monkeypatch, tmp_path):
    prs = [
        {"number": 1, "title": "x", "body": "", "headRefName": "certify-uk-a858c100-b"},
        {"number": 2, "title": f"Certify {NATIONAL}", "body": "", "headRefName": "y"},
        {"number": 3, "title": "z", "body": f"uses {NATIONAL}", "headRefName": "z"},
        {
            "number": 4,
            "title": "other",
            "body": "",
            "headRefName": "certify-uk-ffff0000",
        },
    ]
    calls = _writer_listing(monkeypatch, prs=prs)
    found = handoff.writer_check(NATIONAL, tmp_path)
    assert [line.split()[0] for line in found] == ["#1", "#2", "#3"]
    listing = calls[0]
    assert listing[listing.index("--limit") + 1] == "1000"


def test_another_cuts_branch_is_not_a_writer(monkeypatch, tmp_path):
    _writer_listing(monkeypatch, heads=["certify-uk-ffff0000", "main"])
    assert handoff.writer_check(NATIONAL, tmp_path) == []


def test_an_unreadable_branch_listing_refuses(monkeypatch, tmp_path):
    _writer_listing(monkeypatch, ls_remote_code=128)
    with pytest.raises(handoff.CertifyError, match="could not list policyengine.py"):
        handoff.writer_check(NATIONAL, tmp_path)


GOOD_UC_RECEIPT = {
    "check": "uk_uc_floor",
    "passed": True,
    "measures": [
        {"name": "benefit_units", "ratio": 0.92, "passed": True},
        {"name": "spend", "ratio": 0.96, "passed": True},
    ],
}


def _with(**changes):
    receipt = json.loads(json.dumps(GOOD_UC_RECEIPT))
    receipt.update(changes)
    return receipt


def _with_measure(index, **changes):
    receipt = _with()
    receipt["measures"][index].update(changes)
    return receipt


@pytest.mark.parametrize(
    "receipt",
    [
        [GOOD_UC_RECEIPT],
        "uk_uc_floor",
        None,
        {key: value for key, value in GOOD_UC_RECEIPT.items() if key != "passed"},
        _with(passed="true"),
        _with(check="other"),
        _with(measures={"benefit_units": 1}),
        _with(measures=[["benefit_units", 1.0, True]]),
        _with(measures=[*GOOD_UC_RECEIPT["measures"], {"ratio": 1.0, "passed": True}]),
        _with(measures=[*GOOD_UC_RECEIPT["measures"], GOOD_UC_RECEIPT["measures"][0]]),
        _with(measures=GOOD_UC_RECEIPT["measures"][:1]),
        _with_measure(0, passed=None),
        _with_measure(0, ratio=True),
        _with_measure(0, ratio="0.9"),
        _with_measure(0, ratio=10**400),
        _with_measure(0, ratio=float("nan")),
        _with_measure(0, passed=False),
    ],
)
def test_an_incomplete_uc_floor_receipt_fails_operationally(receipt):
    with pytest.raises(handoff.CertifyError, match="failed operationally"):
        handoff.uc_floor_summary(receipt)


def test_a_complete_uc_floor_receipt_summarizes_its_measures():
    passed, summary = handoff.uc_floor_summary(GOOD_UC_RECEIPT)
    assert passed is True
    assert summary == (
        "passed; measured/benchmark ratios: benefit_units=0.920000, spend=0.960000"
    )


_json = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=True)
    | st.text(max_size=12)
    | st.sampled_from(["uk_uc_floor", "benefit_units", "spend"]),
    lambda children: (
        st.lists(children, max_size=4)
        | st.dictionaries(
            st.sampled_from(["check", "passed", "measures", "name", "ratio"]),
            children,
            max_size=5,
        )
    ),
    max_leaves=20,
)


@settings(max_examples=300, deadline=None)
@given(receipt=_json)
def test_any_json_receipt_summarizes_or_fails_operationally(receipt):
    try:
        passed, summary = handoff.uc_floor_summary(receipt)
    except handoff.CertifyError as error:
        assert "failed operationally" in str(error)
    else:
        assert isinstance(passed, bool) and summary


def test_the_certification_commit_credits_only_named_co_authors():
    assert handoff.commit_message(NATIONAL) == f"Certify Microcosm UK build {NATIONAL}"
    message = handoff.commit_message(
        NATIONAL, co_authors=["A Person <a@example.org>", "B <b@example.org>"]
    )
    assert message.splitlines()[2:] == [
        "Co-Authored-By: A Person <a@example.org>",
        "Co-Authored-By: B <b@example.org>",
    ]


@pytest.mark.parametrize(
    "author", ["no email", "Name <a@b>\nCo-Authored-By: x <y@z>", "<a@b.org>"]
)
def test_a_co_author_must_be_one_name_and_email(author):
    with pytest.raises(SystemExit):
        handoff.parse(
            [
                "--national-build",
                NATIONAL,
                "--drop-regional-release",
                "--pepy-repo",
                "p",
                "--co-author",
                author,
            ]
        )


def test_the_pr_body_footer_is_the_operators_choice():
    context = {
        "national_build": NATIONAL,
        "national_commit": NATIONAL_COMMIT,
        "model": "2.162.3",
        "core": "3.33.3",
        "regional_summary": "`--drop-regional-release`",
        "microcosm_commit": "a" * 40,
        "commands": [],
    }
    assert "Claude" not in handoff.pr_body(context)
    footer = "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
    assert handoff.pr_body({**context, "footer": footer}).rstrip().endswith(footer)
