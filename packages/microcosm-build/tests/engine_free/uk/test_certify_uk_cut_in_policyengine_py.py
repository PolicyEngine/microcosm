"""The policyengine.py certification handoff: routing, guards and rewrites."""

from __future__ import annotations

import argparse
import json
import subprocess

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tools import certify_uk_cut_in_policyengine_py as handoff

NATIONAL = "microcosm-uk-2024-25-national-20261021T195620Z-a858c100"
DENSE = "microcosm-uk-2024-25-dense-20261019T000432Z-ee5c2dbe"
NEW_DENSE = "microcosm-uk-2024-25-dense-20261026T010203Z-0badc0de"


def test_only_cut_tags_are_certified():
    assert handoff.check_tag(NATIONAL, "national") == NATIONAL
    for bad in ("main", NATIONAL + "x", DENSE, "microcosm-uk-2024-25-national"):
        with pytest.raises(handoff.CertifyError, match="never @main"):
            handoff.check_tag(bad, "national")


@pytest.mark.parametrize(
    ("model", "claims", "version", "covered"),
    [
        ("2.134.4", ("==2.134.4",), "2.134.4", True),
        ("2.132.6", ("==2.132.6",), "2.134.4", False),
        ("2.132.6", ("==2.132.6", "==2.134.4"), "2.134.4", True),
    ],
)
def test_coverage_is_built_with_or_an_exact_claim(model, claims, version, covered):
    cut = handoff.CutVersions(tag=DENSE, model=model, core="3.33.3", claims=claims)
    assert handoff.covers(cut, version) is covered


def _args(**kwargs):
    base = dict(
        dense_tag=None,
        keep_regional_release=False,
        area_coded_national_uri=None,
        drop_regional_release=False,
    )
    return argparse.Namespace(**{**base, **kwargs})


def test_each_route_produces_its_certify_data_flags():
    record = {"regional_source_manifest_uri": handoff.manifest_uri(DENSE, "x")}
    dense = handoff.regional_flags(_args(dense_tag=NEW_DENSE), record)
    assert dense[0] == "--regional-manifest-uri" and f"@{NEW_DENSE}/" in dense[1]
    assert handoff.regional_tag(dense) == NEW_DENSE
    kept = handoff.regional_flags(_args(keep_regional_release=True), record)
    assert kept == ["--regional-manifest-uri", record["regional_source_manifest_uri"]]
    area = handoff.regional_flags(_args(area_coded_national_uri="hf://model/x@1/r"), {})
    assert area[-2:] == ["--regional-release-kind", "area_coded_national"]
    assert handoff.regional_tag(area) is None
    dropped = handoff.regional_flags(_args(drop_regional_release=True), record)
    assert dropped == ["--drop-regional-release"]
    with pytest.raises(handoff.CertifyError, match="no regional release"):
        handoff.regional_flags(_args(keep_regional_release=True), {})


def test_routes_are_mutually_exclusive_and_required():
    repo = ["--pepy-repo", "pepy"]
    with pytest.raises(SystemExit):
        handoff.parse(["--national-tag", NATIONAL, *repo])
    with pytest.raises(SystemExit):
        handoff.parse(
            [
                "--national-tag",
                NATIONAL,
                "--dense-tag",
                DENSE,
                "--drop-regional-release",
                *repo,
            ]
        )
    with pytest.raises(SystemExit):
        handoff.parse(
            ["--national-tag", NATIONAL, "--drop-regional-release", "--open-pr", *repo]
        )
    # The policyengine.py clone is the operator's to name; no machine's layout.
    with pytest.raises(SystemExit):
        handoff.parse(["--national-tag", NATIONAL, "--drop-regional-release"])


def test_the_regional_tag_is_rewritten_only_as_a_literal(tmp_path):
    for relative in handoff.CONSTANT_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f'A = "{DENSE}"\nB = "hf://x@{DENSE}/releases/r.json"\n'
            f"# {DENSE} in a comment stays\nC = 'x{DENSE}'\n"
        )
    changed = handoff.rewrite_regional_constants(tmp_path, DENSE, NEW_DENSE)
    assert sorted(changed) == sorted(handoff.CONSTANT_FILES)
    text = (tmp_path / handoff.CONSTANT_FILES[0]).read_text()
    assert f'A = "{NEW_DENSE}"' in text
    assert f"@{NEW_DENSE}/releases" in text
    assert f"# {DENSE} in a comment stays" in text
    assert handoff.rewrite_regional_constants(tmp_path, DENSE, DENSE) == []


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
    handoff.rewrite_regional_constants(root, old, new)
    handoff.rewrite_regional_constants(root, new, old)
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
            "national_tag": NATIONAL,
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
        "UC floor acceptance",
        "enhanced_frs_2024_25",
        "Certification warnings",
        "tests/test_models.py::test_x",
        "One writer",
        "- [ ]",
    ):
        assert needed in body


MISMATCH = [
    "--accept-regional-model-mismatch",
    "area receipts are uncertified (#685)",
    "--audited-by",
    "A Reviewer",
    "--regional-limitation",
    "VAT predates the factor fix",
]


def _dense_route(monkeypatch, tmp_path, *, audit_support=None):
    """A national cut on 2.134.4 and a regional cut on 2.132.6, offline."""

    cuts = {
        NATIONAL: handoff.CutVersions(NATIONAL, "2.134.4", "3.33.3", ("==2.134.4",)),
        DENSE: handoff.CutVersions(DENSE, "2.132.6", "3.33.3", ("==2.132.6",)),
    }
    monkeypatch.setattr(handoff, "read_cut", lambda tag, release: cuts[tag])
    monkeypatch.setattr(handoff, "writer_check", lambda tag, repo: [])
    monkeypatch.setattr(
        handoff, "regional_audit_support", lambda repo, base: audit_support
    )
    downloads = []
    monkeypatch.setattr(
        handoff,
        "fetch_regional_microdata",
        lambda tag, release, **kw: downloads.append(tag) or {},
    )
    monkeypatch.setattr(
        handoff.subprocess,
        "run",
        lambda *a, **k: handoff.subprocess.CompletedProcess(a, 0, "", ""),
    )
    bundle = (
        '{"data_releases": {"uk": {"regional_source_manifest_uri": '
        f'"{handoff.manifest_uri(DENSE, handoff.DENSE_RELEASE)}"}}}}}}'
    )

    def check_output(argv, **kwargs):
        if argv[:2] == ["git", "show"]:
            return bundle
        if argv == ["git", "rev-parse", "HEAD"]:
            return "a" * 40
        return "b" * 40

    monkeypatch.setattr(handoff.subprocess, "check_output", check_output)
    return downloads


def test_a_dry_run_refuses_an_uncovered_dense_cut_before_writing(monkeypatch, tmp_path):
    downloads = _dense_route(monkeypatch, tmp_path)
    base = ["--national-tag", NATIONAL, "--keep-regional-release", "--dry-run",
            "--pepy-repo", str(tmp_path)]  # fmt: skip
    with pytest.raises(
        handoff.CertifyError, match="built with policyengine-uk 2.132.6"
    ):
        handoff.certify(handoff.parse(base))
    # An accepted mismatch plans the audit; a dry run downloads nothing.
    assert handoff.certify(handoff.parse([*base, *MISMATCH])) == 0
    assert downloads == []
    another = handoff.parse(
        ["--national-tag", NATIONAL, "--dense-tag", DENSE, "--dry-run",
         "--pepy-repo", str(tmp_path)]
    )  # fmt: skip
    monkeypatch.setattr(handoff, "writer_check", lambda tag, repo: ["#700 open"])
    with pytest.raises(handoff.CertifyError, match="already being certified"):
        handoff.certify(another)


def test_a_base_without_the_audit_refuses_before_any_download(monkeypatch, tmp_path):
    downloads = _dense_route(
        monkeypatch, tmp_path, audit_support=f"it has no {handoff.AUDIT_SCRIPT}"
    )
    args = handoff.parse(
        ["--national-tag", NATIONAL, "--keep-regional-release",
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
            ["--national-tag", NATIONAL, "--keep-regional-release",
             "--pepy-repo", "p", *flags]
        )  # fmt: skip
    with pytest.raises(SystemExit):
        handoff.parse(
            ["--national-tag", NATIONAL, "--keep-regional-release",
             "--pepy-repo", "p", *drop]
        )  # fmt: skip


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


def test_every_regional_microdata_file_is_fetched_and_checked(monkeypatch, tmp_path):
    import hashlib
    import sys
    import types

    files = {"dense.h5": b"dense", "other.H5": b"other", "notes.csv": b"csv"}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    manifest = {
        "artifacts": {
            "microcosm_uk_2024_25_dense": {
                "kind": "microdata", "path": "dense.h5", "revision": "r1",
                "sha256": hashlib.sha256(b"dense").hexdigest(),
            },
            "second": {
                "kind": "microdata", "path": "other.H5", "revision": "r2",
                "sha256": hashlib.sha256(b"other").hexdigest(),
            },
            "table": {"kind": "microdata", "path": "notes.csv", "revision": "r3",
                      "sha256": "0" * 64},
            "receipt": {"kind": "evidence", "path": "dense.h5", "revision": "r4",
                        "sha256": "0" * 64},
        }
    }  # fmt: skip
    (tmp_path / "release_manifest.json").write_text(json.dumps(manifest))

    def download(repo, path, **kwargs):
        name = path.rsplit("/", 1)[-1]
        return str(
            tmp_path / ("release_manifest.json" if name.endswith(".json") else name)
        )

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=download)
    )
    found = handoff.fetch_regional_microdata(
        DENSE, handoff.DENSE_RELEASE, timings=tmp_path / "t.jsonl"
    )
    assert found == {
        "microcosm_uk_2024_25_dense": tmp_path / "dense.h5",
        "second": tmp_path / "other.H5",
    }
    (tmp_path / "other.H5").write_bytes(b"changed")
    with pytest.raises(handoff.CertifyError, match="not the manifest's"):
        handoff.fetch_regional_microdata(
            DENSE, handoff.DENSE_RELEASE, timings=tmp_path / "t.jsonl"
        )


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
            "national_tag": NATIONAL,
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
        ["--national-tag", NATIONAL, "--drop-regional-release", "--pepy-repo", "p"]
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
    assert handoff.commit_message(NATIONAL) == f"Certify Microcosm UK cut {NATIONAL}"
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
                "--national-tag",
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
        "national_tag": NATIONAL,
        "model": "2.162.3",
        "core": "3.33.3",
        "regional_summary": "`--drop-regional-release`",
        "microcosm_commit": "a" * 40,
        "commands": [],
    }
    assert "Claude" not in handoff.pr_body(context)
    footer = "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
    assert handoff.pr_body({**context, "footer": footer}).rstrip().endswith(footer)
