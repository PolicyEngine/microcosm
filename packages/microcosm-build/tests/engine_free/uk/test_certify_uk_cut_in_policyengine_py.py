"""The policyengine.py certification handoff: routing, guards and rewrites."""

from __future__ import annotations

import argparse

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
            "uc_floor_receipt": {
                "check": "uk_uc_floor",
                "passed": False,
                "measures": [
                    {"name": "benefit_units", "ratio": 0.8, "passed": False},
                    {"name": "spend", "ratio": 0.95, "passed": True},
                ],
            },
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


def test_a_dry_run_refuses_an_uncovered_dense_cut_before_writing(monkeypatch, tmp_path):
    cuts = {
        NATIONAL: handoff.CutVersions(NATIONAL, "2.134.4", "3.33.3", ("==2.134.4",)),
        DENSE: handoff.CutVersions(DENSE, "2.132.6", "3.33.3", ("==2.132.6",)),
    }
    monkeypatch.setattr(handoff, "read_cut", lambda tag, release: cuts[tag])
    monkeypatch.setattr(handoff, "writer_check", lambda tag, repo: [])
    surfaces = {
        "2.132.6": {"all": ["a", "b", "f"], "inputs": ["a", "b"]},
        "2.134.4": {"all": ["a", "b", "f", "n"], "inputs": ["a", "b", "f", "n"]},
    }

    def model_surface(version, core_version=None, **kwargs):
        assert core_version == "3.33.3"
        assert kwargs["resource_root"] == tmp_path
        assert kwargs["timings"].suffix == ".jsonl"
        return surfaces[version]

    monkeypatch.setattr(handoff, "model_surface", model_surface)
    stored = {"a", "b", "f", "x"}  # x: a support column neither model defines
    monkeypatch.setattr(
        handoff, "regional_stored_columns", lambda tag, release, **kw: stored
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
    base_commit = "b" * 40

    def check_output(argv, **kwargs):
        if argv[:2] == ["git", "show"]:
            assert (
                argv[2] == f"{base_commit}:src/policyengine/data/bundle/manifest.json"
            )
            return bundle
        if argv == ["git", "rev-parse", "HEAD"]:
            return "a" * 40
        assert argv == ["git", "rev-parse", "--verify", "origin/main^{commit}"]
        return base_commit

    monkeypatch.setattr(handoff.subprocess, "check_output", check_output)
    args = handoff.parse(
        ["--national-tag", NATIONAL, "--keep-regional-release", "--dry-run",
         "--pepy-repo", str(tmp_path)]
    )  # fmt: skip
    with pytest.raises(
        handoff.CertifyError, match="built with policyengine-uk 2.132.6"
    ):
        handoff.certify(args)
    args.accept_regional_model_mismatch = "area receipts are uncertified (#685)"
    assert handoff.certify(args) == 0
    # A stored column the newer model no longer defines would be lost in area
    # runs, even one the build model computed (f is a formula there).
    surfaces["2.134.4"] = {"all": ["b", "n"], "inputs": ["b", "n"]}
    with pytest.raises(handoff.CertifyError, match="area runs would lose them: a, f"):
        handoff.certify(args)
    args.allow_dropped_columns = "a is retired upstream; reviewed"
    assert handoff.certify(args) == 0
    another = handoff.parse(
        ["--national-tag", NATIONAL, "--dense-tag", DENSE, "--dry-run",
         "--pepy-repo", str(tmp_path)]
    )  # fmt: skip
    monkeypatch.setattr(handoff, "writer_check", lambda tag, repo: ["#700 open"])
    with pytest.raises(handoff.CertifyError, match="already being certified"):
        handoff.certify(another)


names = st.sets(st.from_regex(r"[a-z]{1,6}", fullmatch=True), max_size=12)


@given(
    built_all=names,
    built_inputs=names,
    cert_all=names,
    cert_inputs=names,
    stored=names,
)
def test_the_input_surface_change_is_exact(
    built_all, built_inputs, cert_all, cert_inputs, stored
):
    """Measured on the file's own columns: dropped are stored columns the
    build model defined and the certified model does not; defaulted are the
    certified model's inputs the file does not store that the build model did
    not take as inputs. Columns neither model knows, inputs absent under both
    models, and stored formula variables that still exist are no change."""

    built_all |= built_inputs
    cert_all |= cert_inputs
    built = {"all": sorted(built_all), "inputs": sorted(built_inputs)}
    certified = {"all": sorted(cert_all), "inputs": sorted(cert_inputs)}
    change = handoff.input_surface_change(built, certified, stored)
    dropped, defaulted = set(change["dropped"]), set(change["defaulted"])
    assert dropped == (stored & built_all) - cert_all
    assert defaulted == (cert_inputs - stored) - built_inputs
    assert dropped <= stored and not dropped & cert_all
    assert not defaulted & stored and not defaulted & built_inputs
    assert not (stored - built_all - cert_all) & (dropped | defaulted)
    assert not (stored & cert_all) & dropped
    assert change == handoff.input_surface_change(built, certified, set(stored))


def test_an_unchanged_model_changes_nothing():
    surface = {"all": ["a", "b"], "inputs": ["a"]}
    for stored in (set(), {"a"}, {"a", "b", "support"}):
        assert handoff.input_surface_change(surface, surface, stored) == {
            "dropped": [],
            "defaulted": [],
        }


def test_a_stored_formula_variable_the_new_model_removes_is_dropped():
    """The reviewer's case: Microcosm stores formula variables too."""

    built = {"all": ["is_uc_claimant", "age"], "inputs": ["age"]}
    certified = {"all": ["age"], "inputs": ["age"]}
    change = handoff.input_surface_change(built, certified, {"is_uc_claimant", "age"})
    assert change == {"dropped": ["is_uc_claimant"], "defaulted": []}


def test_a_stored_variable_that_became_an_input_is_not_defaulted():
    built = {"all": ["fare"], "inputs": []}
    certified = {"all": ["fare"], "inputs": ["fare"]}
    assert handoff.input_surface_change(built, certified, {"fare"})["defaulted"] == []
    assert handoff.input_surface_change(built, certified, set())["defaulted"] == [
        "fare"
    ]


def _table_h5(path, entities):
    import h5py
    import numpy as np

    with h5py.File(path, "w") as store:
        for entity, columns in entities.items():
            dtype = np.dtype([("index", "i8"), *((c, "f8") for c in columns)])
            group = store.create_group(entity)
            group.create_dataset("table", data=np.zeros(2, dtype=dtype))
            index = group.create_group("_i_table")
            for column in columns:
                index.create_group(column)


def test_the_column_inventory_reads_microcosm_table_files(tmp_path):
    path = tmp_path / "dense.h5"
    _table_h5(
        path,
        {
            "person": ["age", "is_uc_claimant"],
            "benunit": ["benunit_id"],
            "household": ["household_weight", "constituency_code_oa"],
        },
    )
    assert handoff.h5_stored_columns(path) == {
        "age",
        "is_uc_claimant",
        "benunit_id",
        "household_weight",
        "constituency_code_oa",
    }


def test_the_column_inventory_reads_flat_files_and_refuses_others(tmp_path):
    import h5py

    flat = tmp_path / "flat.h5"
    with h5py.File(flat, "w") as store:
        store.create_dataset("age", data=[1, 2])
        store.create_group("employment_income").create_dataset("2024", data=[0.0])
    assert handoff.h5_stored_columns(flat) == {"age", "employment_income"}
    mixed = tmp_path / "mixed.h5"
    _table_h5(mixed, {"person": ["age"]})
    with h5py.File(mixed, "a") as store:
        store.create_dataset("stray", data=[1])
    with pytest.raises(handoff.CertifyError, match="not a recognised"):
        handoff.h5_stored_columns(mixed)
    fixed = tmp_path / "fixed.h5"
    _table_h5(fixed, {"person": ["values_block_0"]})
    with pytest.raises(handoff.CertifyError, match="fixed-format"):
        handoff.h5_stored_columns(fixed)


def test_the_pr_body_lists_the_area_input_surface():
    body = handoff.pr_body(
        {
            "national_tag": NATIONAL,
            "model": "2.146.4",
            "core": "3.33.3",
            "regional_summary": "`--keep-regional-release`",
            "microcosm_commit": "a" * 40,
            "commands": [],
            "input_surface": {
                "dropped": [],
                "defaulted": ["local_bus_uncapped_single_fare"],
            },
            "regional_model": "2.132.6",
        }
    )
    assert "built on 2.132.6" in body
    assert "local_bus_uncapped_single_fare" in body


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
