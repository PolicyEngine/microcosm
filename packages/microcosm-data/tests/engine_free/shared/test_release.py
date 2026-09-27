"""Tests split from packages/microcosm-data/tests/test_release.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_data.release import *


def test_annual_cut_passes_real_release_contract(
    release_dir, artifact_root, annual_release
):
    prepared = release_module.prepare_release(
        release_dir,
        artifact_root=artifact_root,
        tag_name=annual_release,
        update_latest=False,
    )
    assert prepared.tag == annual_release
    assert {"annual_manifest.json", "annual_acceptance.json"}.issubset(
        prepared.filenames
    )
    assert {"annual_2025.h5", "populace_us_2024.h5"}.issubset(prepared.root_artifacts)

def test_annual_cut_uploads_exact_artifacts_without_latest(
    hub, release_dir, artifact_root, annual_release
):
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        tag_name=annual_release,
        update_latest=False,
    )
    uploads = dict(hub.uploads)
    assert uploads["annual_2025.h5"] == (artifact_root / "annual_2025.h5").read_bytes()
    assert (
        uploads["populace_us_2024.h5"]
        == (artifact_root / "populace_us_2024.h5").read_bytes()
    )
    for name in (
        "annual_manifest.json",
        "annual_acceptance.json",
        "projection_2025.json",
        "release_manifest.json",
    ):
        assert (
            uploads[f"releases/{release_dir.name}/{name}"]
            == (release_dir / name).read_bytes()
        )
    assert hub.tags == [{"tag": annual_release, "revision": "commit-1"}]
    assert LATEST_POINTER_PATH not in uploads
    assert LATEST_EVIDENCE_POINTER_PATH not in uploads

@pytest.mark.parametrize(
    "option", ["latest", "wrong_tag", "absent_annual", "invalid_base"]
)
def test_annual_cut_preserves_publisher_and_base_guards(
    release_dir, artifact_root, annual_release, option
):
    path = release_dir / "release_manifest.json"
    manifest = json.loads(path.read_text())
    if option == "absent_annual":
        del manifest["metadata"]
        path.write_text(json.dumps(manifest))
    elif option == "invalid_base":
        (release_dir / "calibration_diagnostics.json").write_text("{}")
    with pytest.raises((ReleaseContractError, ValueError)):
        release_module.prepare_release(
            release_dir,
            artifact_root=artifact_root,
            tag_name=release_dir.name if option == "wrong_tag" else annual_release,
            update_latest=option == "latest",
        )

def test_pointer_payload_names_every_contract_file() -> None:
    payload = latest_pointer_payload(RELEASE_ID, updated_at="2026-06-11T13:53:15+00:00")
    assert payload["schema_version"] == LATEST_POINTER_SCHEMA_VERSION
    assert payload["release_id"] == RELEASE_ID
    assert set(payload["paths"]) == {
        name.removesuffix(".json") for name in required_release_files(RELEASE_ID)
    }
    assert (
        payload["paths"]["build_manifest"]
        == f"releases/{RELEASE_ID}/build_manifest.json"
    )
    assert (
        payload["paths"]["us_source_coverage"]
        == f"releases/{RELEASE_ID}/{US_SOURCE_COVERAGE_DIAGNOSTICS_FILE}"
    )

@pytest.mark.parametrize(
    ("release_id", "line", "revision"),
    [
        (UK_NATIONAL_RELEASE_ID, "national", UK_NATIONAL_CUT_TAG),
        (UK_LOCAL_RELEASE_ID, "local-k55000", UK_LOCAL_RELEASE_ID),
    ],
)
def test_line_pointer_payload_round_trips_for_each_line_path(
    hub: FakeHub, release_id: str, line: str, revision: str
) -> None:
    payload = line_pointer_payload(
        release_id,
        line=line,
        revision=revision,
        updated_at="2026-09-20T12:00:00+00:00",
    )
    pointer_path = line_pointer_path(line)
    hub.seed_main_file(pointer_path, json.dumps(payload).encode())

    pointer = latest_line_release("policyengine/populace-us", line=line, api=hub)

    assert pointer.release_id == release_id
    assert pointer.revision == revision
    assert pointer.line == line
    assert pointer.updated_at == "2026-09-20T12:00:00+00:00"
    assert payload["schema_version"] == LATEST_POINTER_SCHEMA_VERSION
    assert pointer_path == f"latest-{line}.json"

@pytest.mark.parametrize(
    "line",
    ["", "National", "local-k0", "local-k01", "national/other"],
)
def test_line_pointer_path_refuses_invalid_lines(line: str) -> None:
    with pytest.raises(ValueError, match="invalid release line"):
        line_pointer_path(line)

@pytest.mark.parametrize(
    "revision",
    [
        "",
        UK_NATIONAL_RELEASE_ID + "-hotfix",
        UK_NATIONAL_RELEASE_ID + "-20260920t120000Z-deadbeef",
        UK_NATIONAL_RELEASE_ID + "-20260920T120000Z-DEADBEEF",
    ],
)
def test_line_pointer_payload_refuses_revisions_outside_the_cut_family(
    revision: str,
) -> None:
    with pytest.raises(ValueError, match="per-cut tag"):
        line_pointer_payload(
            UK_NATIONAL_RELEASE_ID,
            line="national",
            revision=revision,
        )

def test_line_pointer_payload_refuses_a_line_id_mismatch() -> None:
    with pytest.raises(ValueError, match="belongs to line 'national'"):
        line_pointer_payload(
            UK_NATIONAL_RELEASE_ID,
            line="local-k55000",
            revision=UK_NATIONAL_CUT_TAG,
        )

@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"update_latest": False}, "requires update_latest=True"),
        ({"tag_only": True}, "cannot use tag_only=True"),
        ({"evidence": True}, "cannot publish at the evidence tier"),
    ],
)
def test_prepare_release_refuses_incompatible_line_modes(
    release_dir: Path, monkeypatch, options: dict, message: str
) -> None:
    monkeypatch.setattr(
        release_module, "validate_release_dir", lambda _path, **_kwargs: None
    )
    monkeypatch.setattr(
        release_module,
        "validate_evidence_release_dir",
        lambda _path, **_kwargs: None,
    )

    with pytest.raises(ValueError, match=message):
        release_module.prepare_release(release_dir, line="national", **options)

def test_prepare_release_refuses_a_line_id_mismatch(
    release_dir: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        release_module, "validate_release_dir", lambda _path, **_kwargs: None
    )

    with pytest.raises(ValueError, match="belongs to line None"):
        release_module.prepare_release(release_dir, line="national")

def test_national_cut_promotes_only_its_line_pointer(
    release_dir: Path, artifact_root: Path, tmp_path: Path, monkeypatch
) -> None:
    national = _as_uk_national_line_release(release_dir)
    monkeypatch.setattr(
        release_module, "validate_release_dir", lambda _path, **_kwargs: None
    )
    alerts: list = []
    monkeypatch.setattr(
        release_module,
        "notify_release",
        lambda repo_id, release_id, updated_at, **kwargs: alerts.append(
            (repo_id, release_id, updated_at, kwargs)
        ),
    )
    uk_hub = FakeHub("policyengine/populace-uk-private")
    uk_hub._download_dir = tmp_path / "uk-hub-cache"
    june_pointer = latest_pointer_payload(
        JUNE_UK_RELEASE_ID, updated_at="2026-06-19T02:38:00+00:00"
    )
    june_pointer_bytes = json.dumps(june_pointer, indent=1).encode()
    uk_hub.seed_main_file(LATEST_POINTER_PATH, june_pointer_bytes)

    published = publish_release(
        national,
        "policyengine/populace-uk-private",
        api=uk_hub,
        artifact_root=artifact_root,
        tag_name=UK_NATIONAL_CUT_TAG,
        line="national",
        updated_at="2026-09-20T12:00:00+00:00",
    )

    pointer_path = line_pointer_path("national")
    main_files = uk_hub._commits[uk_hub._refs["main"]]
    assert main_files[LATEST_POINTER_PATH] == june_pointer_bytes
    assert {
        path
        for path in main_files
        if path.startswith("latest") and path.endswith(".json")
    } == {LATEST_POINTER_PATH, pointer_path}
    assert uk_hub.events[-1][1]["paths"][-1] == pointer_path
    assert uk_hub.uploads[-1][0] == pointer_path
    assert uk_hub.tags == [{"tag": UK_NATIONAL_CUT_TAG, "revision": "commit-1"}]
    assert published["revision"] == UK_NATIONAL_CUT_TAG
    assert published["line"] == "national"
    assert (
        latest_line_release(
            "policyengine/populace-uk-private", line="national", api=uk_hub
        ).revision
        == UK_NATIONAL_CUT_TAG
    )
    assert alerts == [
        (
            "policyengine/populace-uk-private",
            UK_NATIONAL_RELEASE_ID,
            "2026-09-20T12:00:00+00:00",
            {
                "warn_if_unset": True,
                "line": "national",
                "revision": UK_NATIONAL_CUT_TAG,
            },
        )
    ]

def test_local_k_role_moves_only_its_own_line_pointer(tmp_path: Path) -> None:
    from .test_local_area_contract import _write_local_bundle

    release_dir = _write_local_bundle(tmp_path, release_id=UK_LOCAL_RELEASE_ID)
    uk_hub = FakeHub("policyengine/populace-uk-private")
    uk_hub._download_dir = tmp_path / "uk-local-hub-cache"

    published = publish_release(
        release_dir,
        "policyengine/populace-uk-private",
        api=uk_hub,
        artifact_root=tmp_path,
        line="local-k55000",
        notify=False,
    )

    pointer_path = line_pointer_path("local-k55000")
    assert published["revision"] == UK_LOCAL_RELEASE_ID
    assert uk_hub.tags == [{"tag": UK_LOCAL_RELEASE_ID, "revision": "commit-1"}]
    assert uk_hub.events[-1][1]["paths"][-1] == pointer_path
    assert all(path != LATEST_POINTER_PATH for path, _ in uk_hub.uploads)
    assert all(path != LATEST_EVIDENCE_POINTER_PATH for path, _ in uk_hub.uploads)

def test_line_promotion_alert_names_the_line_and_revision() -> None:
    from microcosm.data.slack import notify_release

    sent: list = []
    assert notify_release(
        "policyengine/populace-uk-private",
        UK_NATIONAL_RELEASE_ID,
        webhook="https://hooks.slack.test/uk",
        post=lambda url, payload: sent.append((url, payload)),
        line="national",
        revision=UK_NATIONAL_CUT_TAG,
    )
    headline = sent[0][1]["blocks"][0]["text"]["text"]
    assert "line promoted" in headline
    assert "national" in headline
    assert UK_NATIONAL_CUT_TAG in headline

def test_publish_release_announces_after_pointer(
    hub: FakeHub, release_dir: Path, artifact_root: Path, monkeypatch
) -> None:
    """The alert fires from publish_release itself — every publish path, not
    just the CLI — with the released id and timestamp."""
    calls: list = []
    monkeypatch.setattr(
        "microcosm.data.release.notify_release",
        lambda repo_id, release_id, updated_at, **kw: calls.append(
            (repo_id, release_id, updated_at, kw)
        ),
    )
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-06-11T13:53:15+00:00",
    )
    assert calls == [
        (
            "policyengine/populace-us",
            RELEASE_ID,
            "2026-06-11T13:53:15+00:00",
            {"warn_if_unset": True},
        )
    ]

def test_publish_release_notify_false_skips_alert(
    hub: FakeHub, release_dir: Path, artifact_root: Path, monkeypatch
) -> None:
    calls: list = []
    monkeypatch.setattr(
        "microcosm.data.release.notify_release", lambda *a, **k: calls.append(a)
    )
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        notify=False,
    )
    assert calls == []

def test_publish_uploads_pointer_last(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-06-11T13:53:15+00:00",
    )
    final_event, final_commit = hub.events[-1]
    assert final_event == "create_commit"
    assert final_commit["revision"] == "main"
    assert final_commit["paths"][-1] == LATEST_POINTER_PATH
    for filename in required_release_files(RELEASE_ID):
        assert f"releases/{RELEASE_ID}/{filename}" in final_commit["paths"][:-1]

def test_publish_no_latest_never_touches_pointer(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-06-11T13:53:15+00:00",
        update_latest=False,
    )
    # Immutable branch + tag flow is unchanged; the final main commit exists
    # (release copies + root artifacts) but carries NO pointer operation.
    assert [event for event, _ in hub.events] == [
        "create_branch",
        "create_commit",
        "create_tag",
        "delete_branch",
        "create_commit",
    ]
    final_event, final_commit = hub.events[-1]
    assert final_event == "create_commit"
    assert final_commit["revision"] == "main"
    assert LATEST_POINTER_PATH not in final_commit["paths"]
    assert final_commit["message"] == f"Publish non-default release {RELEASE_ID}"
    assert {
        "populace_us_2024.h5",
        "populace_us_2024_calibration.npz",
    }.issubset(final_commit["paths"])

def test_exact_k_tag_only_publish_never_mutates_main(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    main_before = hub._refs["main"]

    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-06-11T13:53:15+00:00",
        update_latest=False,
        tag_only=True,
    )

    assert [event for event, _ in hub.events] == [
        "create_branch",
        "create_commit",
        "create_tag",
        "delete_branch",
    ]
    assert hub._refs["main"] == main_before

    canonical_root_paths = {
        "populace_us_2024.h5",
        "populace_us_2024_calibration.npz",
    }
    main_commit_paths = {
        path
        for event, receipt in hub.events
        if event == "create_commit" and receipt["revision"] == "main"
        for path in receipt["paths"]
    }
    canonical_root_mutation_present = bool(canonical_root_paths & main_commit_paths)
    assert canonical_root_mutation_present is False
    assert canonical_root_paths.isdisjoint(main_commit_paths)

    immutable = hub.events[1][1]
    assert canonical_root_paths.issubset(immutable["paths"])
    assert {
        f"releases/{RELEASE_ID}/{filename}"
        for filename in required_release_files(RELEASE_ID)
    }.issubset(immutable["paths"])
    assert hub.tags == [{"tag": RELEASE_ID, "revision": immutable["commit"]}]

    tagged_manifest = Path(
        hub.hf_hub_download(
            repo_id="policyengine/populace-us",
            filename=f"releases/{RELEASE_ID}/release_manifest.json",
            repo_type="dataset",
            revision=RELEASE_ID,
        )
    )
    tagged_h5 = Path(
        hub.hf_hub_download(
            repo_id="policyengine/populace-us",
            filename="populace_us_2024.h5",
            repo_type="dataset",
            revision=RELEASE_ID,
        )
    )
    assert (
        tagged_manifest.read_bytes()
        == (release_dir / "release_manifest.json").read_bytes()
    )
    assert (
        tagged_h5.read_bytes() == (artifact_root / "populace_us_2024.h5").read_bytes()
    )
    with pytest.raises(FileNotFoundError, match="populace_us_2024.h5@main"):
        hub.hf_hub_download(
            repo_id="policyengine/populace-us",
            filename="populace_us_2024.h5",
            repo_type="dataset",
        )

@pytest.mark.parametrize("tag_only", [True, False], ids=["tag-only", "no-latest"])
def test_us_default_flip_after_a_non_default_publish_reuses_the_release_tag(
    hub: FakeHub, release_dir: Path, artifact_root: Path, tag_only: bool, monkeypatch
) -> None:
    """microcosm#450: publish immutable now, make it the default later.

    The first call cuts the release-id tag without touching ``latest.json``
    (``--tag-only`` or ``--no-latest``). The flip is the plain default
    publish of the same release directory: it recognises the tag that
    already describes this release, writes no second immutable revision
    and no second tag, and lands only the main commit carrying the pointer.
    Before #966 the flip died on ``create_tag`` with a 409.
    """
    monkeypatch.setattr(release_module, "notify_release", lambda *a, **k: None)
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-06-11T13:53:15+00:00",
        update_latest=False,
        tag_only=tag_only,
    )
    tags_after_cut = list(hub.tags)
    assert [tag["tag"] for tag in tags_after_cut] == [RELEASE_ID]
    tagged_revision = hub._refs[RELEASE_ID]
    assert LATEST_POINTER_PATH not in hub._commits[hub._refs["main"]]
    main_before_flip = hub._refs["main"]
    events_after_cut = len(hub.events)

    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-06-12T09:00:00+00:00",
    )

    flip_events = hub.events[events_after_cut:]
    assert [event for event, _ in flip_events] == ["create_commit"]
    flip = flip_events[0][1]
    assert flip["revision"] == "main"
    assert flip["parent_commit"] == main_before_flip
    assert LATEST_POINTER_PATH in flip["paths"]
    assert hub.tags == tags_after_cut
    assert hub._refs[RELEASE_ID] == tagged_revision
    pointer = latest_release("policyengine/populace-us", api=hub)
    assert pointer.release_id == RELEASE_ID

def test_us_default_flip_refuses_a_release_tag_that_describes_another_cut(
    hub: FakeHub, release_dir: Path, artifact_root: Path, monkeypatch
) -> None:
    """microcosm#450: the flip reuses a tag only when it is this release.

    A local release directory whose manifest differs from the tagged one
    (a rebuilt cut under the same release id) must refuse before any main
    commit, so the default pointer never names bytes nobody published.
    """
    monkeypatch.setattr(release_module, "notify_release", lambda *a, **k: None)
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-06-11T13:53:15+00:00",
        update_latest=False,
        tag_only=True,
    )
    main_before_flip = hub._refs["main"]
    events_after_cut = len(hub.events)
    manifest_path = release_dir / "release_manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="describes another"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
            updated_at="2026-06-12T09:00:00+00:00",
        )

    assert hub.events[events_after_cut:] == []
    assert hub._refs["main"] == main_before_flip

@pytest.mark.parametrize(
    ("publish_options", "message"),
    [
        ({"update_latest": True}, "requires update_latest=False"),
        (
            {"update_latest": False, "create_tag": False},
            "requires create_tag=True",
        ),
    ],
)
def test_tag_only_rejects_unsafe_modes_before_remote_mutation(
    hub: FakeHub,
    release_dir: Path,
    artifact_root: Path,
    publish_options: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
            tag_only=True,
            **publish_options,
        )

    assert hub.events == []
    assert hub.uploads == []

def test_publish_commits_immutable_release_before_root_and_pointer(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-06-11T13:53:15+00:00",
    )
    assert [event for event, _ in hub.events] == [
        "create_branch",
        "create_commit",
        "create_tag",
        "delete_branch",
        "create_commit",
    ]

    staging_branch = f"release-staging/{RELEASE_ID}"
    immutable = hub.events[1][1]
    assert immutable["revision"] == staging_branch
    assert set(immutable["paths"]) == {
        "populace_us_2024.h5",
        "populace_us_2024_calibration.npz",
        *{
            f"releases/{RELEASE_ID}/{filename}"
            for filename in required_release_files(RELEASE_ID)
        },
        # The gate verdicts are manifest artifacts, so the immutable release
        # commit carries them without --extra-file.
        *{
            f"releases/{RELEASE_ID}/{filename}"
            for filename in GATE_EVIDENCE_FILES.values()
        },
    }
    assert LATEST_POINTER_PATH not in immutable["paths"]

    tag = hub.events[2][1]
    assert tag == {"tag": RELEASE_ID, "revision": immutable["commit"]}
    assert hub.events[3] == ("delete_branch", {"branch": staging_branch})

    convenience = hub.events[4][1]
    assert convenience["revision"] == "main"
    assert convenience["parent_commit"] == "commit-0"
    assert convenience["paths"][-3:] == [
        "populace_us_2024.h5",
        "populace_us_2024_calibration.npz",
        LATEST_POINTER_PATH,
    ]
    assert hub.tags == [{"tag": RELEASE_ID, "revision": immutable["commit"]}]

def test_failed_main_commit_leaves_root_and_pointer_unchanged(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    old_pointer = b'{"release_id": "old-release"}'
    old_artifact = b"old certified root artifact"
    hub.seed_main_file(LATEST_POINTER_PATH, old_pointer)
    hub.seed_main_file("populace_us_2024.h5", old_artifact)
    hub.fail_main_commit = True

    with pytest.raises(RuntimeError, match="injected main commit failure"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
        )

    main_commit = hub._commits[hub._refs["main"]]
    assert main_commit[LATEST_POINTER_PATH] == old_pointer
    assert main_commit["populace_us_2024.h5"] == old_artifact
    assert hub.tags == [{"tag": RELEASE_ID, "revision": "commit-1"}]
    assert hub.events[-1][0] == "create_commit_failed"

def test_non_atomic_backend_is_refused_before_remote_mutation(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    with pytest.raises(TypeError, match="immutable-first and atomic"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=NonAtomicHub(hub),
            artifact_root=artifact_root,
        )

    assert hub.events == []
    assert hub.uploads == []

def test_publish_uploads_manifest_release_diagnostics_from_release_dir(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    (release_dir / "reform_validation.json").write_text('{"schema_version": 1}')
    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["reform_validation"] = {
        "kind": "diagnostics",
        "path": "reform_validation.json",
        "repo_id": "policyengine/populace-us",
        "revision": RELEASE_ID,
        "sha256": _sha256(release_dir / "reform_validation.json"),
    }
    manifest_path.write_text(json.dumps(manifest))

    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-06-11T13:53:15+00:00",
    )

    uploaded_paths = [path for path, _ in hub.uploads]
    release_path = f"releases/{RELEASE_ID}/reform_validation.json"
    assert "reform_validation.json" not in uploaded_paths
    assert release_path in uploaded_paths
    assert uploaded_paths.index(release_path) < uploaded_paths.index(
        LATEST_POINTER_PATH
    )

def test_publish_uploads_ssi_take_up_diagnostics_without_extra_files(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    diagnostics_path = release_dir / "us_ssi_take_up.json"
    diagnostics_path.write_text('{"schema_version": 1}')
    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["us_ssi_take_up"] = {
        "kind": "diagnostics",
        "path": diagnostics_path.name,
        "repo_id": "policyengine/populace-us",
        "revision": RELEASE_ID,
        "sha256": _sha256(diagnostics_path),
    }
    manifest_path.write_text(json.dumps(manifest))

    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-06-11T13:53:15+00:00",
    )

    uploaded_paths = [path for path, _ in hub.uploads]
    release_path = f"releases/{RELEASE_ID}/{diagnostics_path.name}"
    assert diagnostics_path.name not in uploaded_paths
    assert release_path in uploaded_paths
    assert uploaded_paths.index(release_path) < uploaded_paths.index(
        LATEST_POINTER_PATH
    )

def test_publish_requires_artifact_root_for_root_artifacts(
    hub: FakeHub, release_dir: Path
) -> None:
    with pytest.raises(ValueError, match="pass artifact_root"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
        )
    assert hub.uploads == []

def test_missing_root_artifact_uploads_nothing(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    (artifact_root / "populace_us_2024_calibration.npz").unlink()
    with pytest.raises(FileNotFoundError, match="populace_us_2024_calibration"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
        )
    assert hub.uploads == []

def test_root_artifact_hash_mismatch_uploads_nothing(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    (artifact_root / "populace_us_2024.h5").write_bytes(b"wrong payload")
    with pytest.raises(ValueError, match="release artifact 'populace_us_2024.h5'"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
        )
    assert hub.uploads == []

def test_release_tag_is_created_before_pointer(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        create_tag=True,
        updated_at="2026-06-11T13:53:15+00:00",
    )
    assert hub.tags == [{"tag": RELEASE_ID, "revision": "commit-1"}]
    assert [event for event, _ in hub.events][-3:] == [
        "create_tag",
        "delete_branch",
        "create_commit",
    ]
    assert hub.uploads[-1][0] == LATEST_POINTER_PATH

def test_release_id_artifact_revision_requires_release_tag(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    with pytest.raises(ValueError, match="pins artifacts to revisions.*must create"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
            create_tag=False,
        )
    assert hub.uploads == []
    assert hub.tags == []

def test_release_id_artifact_revision_rejects_tag_name_override(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    with pytest.raises(ValueError, match="tag_name must match.*artifact revision"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
            tag_name="different-tag",
        )
    assert hub.uploads == []
    assert hub.tags == []

def test_per_cut_artifact_revision_publishes_matching_tag_without_latest(
    hub: FakeHub,
    release_dir: Path,
    artifact_root: Path,
    monkeypatch,
) -> None:
    cut_tag = RELEASE_ID + "-20260828T101112Z-1a2b3c4d"
    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for artifact in manifest["artifacts"].values():
        artifact["revision"] = cut_tag
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        release_module, "validate_release_dir", lambda _path, **_kwargs: None
    )

    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        tag_name=cut_tag,
        update_latest=False,
    )

    assert hub.tags == [{"tag": cut_tag, "revision": "commit-1"}]
    assert all(path != LATEST_POINTER_PATH for path, _payload in hub.uploads)

def test_per_cut_artifact_revision_refuses_dangling_default_tag(
    hub: FakeHub,
    release_dir: Path,
    artifact_root: Path,
    monkeypatch,
) -> None:
    cut_tag = RELEASE_ID + "-20260828T101112Z-1a2b3c4d"
    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for artifact in manifest["artifacts"].values():
        artifact["revision"] = cut_tag
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        release_module, "validate_release_dir", lambda _path, **_kwargs: None
    )

    with pytest.raises(ValueError, match="uniform per-cut artifact revision"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
        )
    assert hub.uploads == []
    assert hub.tags == []

def test_unreadable_artifact_revisions_refuse_instead_of_vanishing(
    hub: FakeHub,
    release_dir: Path,
    artifact_root: Path,
    monkeypatch,
) -> None:
    # A non-string revision must not shrink the pin set into vacuous guards:
    # the empty set may only ever mean "no artifacts declared", never
    # "artifacts whose pins could not be read".
    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for artifact in manifest["artifacts"].values():
        artifact["revision"] = 123
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        release_module, "validate_release_dir", lambda _path, **_kwargs: None
    )

    with pytest.raises(ValueError, match="missing or non-string revisions"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
        )
    assert hub.uploads == []
    assert hub.tags == []

def test_empty_artifact_revisions_refuse_publication(
    hub: FakeHub,
    release_dir: Path,
    artifact_root: Path,
    monkeypatch,
) -> None:
    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"] = {}
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        release_module, "validate_release_dir", lambda _path, **_kwargs: None
    )

    with pytest.raises(ValueError, match="declares no artifact revisions"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
        )
    assert hub.uploads == []
    assert hub.tags == []

def test_per_cut_tag_refuses_latest_promotion(
    hub: FakeHub,
    release_dir: Path,
    artifact_root: Path,
    monkeypatch,
) -> None:
    # The pointer may only ever name a release whose tag IS the release id:
    # a per-cut inspect tag with the default update_latest must refuse before
    # any remote mutation, not promote.
    cut_tag = RELEASE_ID + "-20260828T101112Z-1a2b3c4d"
    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for artifact in manifest["artifacts"].values():
        artifact["revision"] = cut_tag
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        release_module, "validate_release_dir", lambda _path, **_kwargs: None
    )

    with pytest.raises(ValueError, match="inspect-only"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
            tag_name=cut_tag,
        )
    assert hub.uploads == []
    assert hub.tags == []
    assert hub.events == []

def test_invalid_release_uploads_nothing(hub: FakeHub, release_dir: Path) -> None:
    (release_dir / "build_manifest.json").unlink()
    with pytest.raises(ReleaseContractError):
        publish_release(release_dir, "policyengine/populace-us", api=hub)
    assert hub.uploads == []

def test_invalid_calibration_diagnostics_uploads_nothing(
    hub: FakeHub, release_dir: Path
) -> None:
    (release_dir / "calibration_diagnostics.json").write_text("{}")
    with pytest.raises(ReleaseContractError, match="calibration_diagnostics"):
        publish_release(release_dir, "policyengine/populace-us", api=hub)
    assert hub.uploads == []

def test_nonstandard_nan_calibration_diagnostics_uploads_nothing(
    hub: FakeHub, release_dir: Path
) -> None:
    (release_dir / "calibration_diagnostics.json").write_text(
        '{"schema_version": 4, "targets": [], "loss_trajectory": [NaN], '
        '"skipped": [], "options": {}}'
    )
    with pytest.raises(ReleaseContractError, match="calibration_diagnostics"):
        publish_release(release_dir, "policyengine/populace-us", api=hub)
    assert hub.uploads == []

def test_extra_files_ride_along_before_the_pointer(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        extra_files=("calibration_diagnostics.json",),
    )
    uploaded_paths = [path for path, _ in hub.uploads]
    extra = f"releases/{RELEASE_ID}/calibration_diagnostics.json"
    assert extra in uploaded_paths
    assert uploaded_paths.index(extra) < uploaded_paths.index(LATEST_POINTER_PATH)

def test_missing_extra_file_fails_loudly(hub: FakeHub, release_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="support_audit"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            extra_files=("support_audit.json",),
        )
    assert hub.uploads == []

def test_publish_then_resolve_round_trips(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    published = publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-06-11T13:53:15+00:00",
    )
    pointer = latest_release("policyengine/populace-us", api=hub)
    assert pointer.release_id == RELEASE_ID
    assert pointer.revision == RELEASE_ID
    assert pointer.line is None
    assert pointer.updated_at == "2026-06-11T13:53:15+00:00"
    assert pointer.paths == published["paths"]

@pytest.mark.parametrize("field", ["line", "revision"])
def test_latest_release_refuses_line_pointer_fields(hub: FakeHub, field: str) -> None:
    payload = latest_pointer_payload(RELEASE_ID)
    payload[field] = None
    hub.seed_main_file(LATEST_POINTER_PATH, json.dumps(payload).encode())

    with pytest.raises(ValueError, match=field):
        latest_release("policyengine/populace-us", api=hub)

@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("line", "local-k55000", "declares line"),
        ("revision", None, "revision"),
        ("revision", "", "revision"),
        ("revision", UK_NATIONAL_RELEASE_ID + "-hotfix", "revision"),
        ("tier", "certified", "tier"),
    ],
)
def test_latest_line_release_refuses_foreign_pointer_fields(
    hub: FakeHub, field: str, value: object, message: str
) -> None:
    payload = line_pointer_payload(
        UK_NATIONAL_RELEASE_ID,
        line="national",
        revision=UK_NATIONAL_CUT_TAG,
    )
    payload[field] = value
    hub.seed_main_file(line_pointer_path("national"), json.dumps(payload).encode())

    with pytest.raises(ValueError, match=message):
        latest_line_release("policyengine/populace-us", line="national", api=hub)

def test_latest_line_release_refuses_a_release_from_another_line(
    hub: FakeHub,
) -> None:
    payload = latest_pointer_payload(UK_LOCAL_RELEASE_ID)
    payload.update(line="national", revision=UK_LOCAL_RELEASE_ID)
    hub.seed_main_file(line_pointer_path("national"), json.dumps(payload).encode())

    with pytest.raises(ValueError, match="belongs to line 'local-k55000'"):
        latest_line_release("policyengine/populace-us", line="national", api=hub)

def test_future_pointer_schema_is_refused(hub: FakeHub) -> None:
    hub.seed_main_file(
        LATEST_POINTER_PATH,
        json.dumps({"schema_version": LATEST_POINTER_SCHEMA_VERSION + 1}).encode(),
    )
    with pytest.raises(ValueError, match="Upgrade microcosm-data"):
        latest_release("policyengine/populace-us", api=hub)

def test_pointer_without_release_id_is_refused(hub: FakeHub) -> None:
    hub.seed_main_file(
        LATEST_POINTER_PATH,
        json.dumps({"schema_version": LATEST_POINTER_SCHEMA_VERSION}).encode(),
    )
    with pytest.raises(ValueError, match="release_id"):
        latest_release("policyengine/populace-us", api=hub)

def test_pointer_without_contract_paths_is_refused(hub: FakeHub) -> None:
    hub.seed_main_file(
        LATEST_POINTER_PATH,
        json.dumps(
            {
                "schema_version": LATEST_POINTER_SCHEMA_VERSION,
                "release_id": RELEASE_ID,
                "paths": {"build_manifest": "releases/x/build_manifest.json"},
            }
        ).encode(),
    )
    with pytest.raises(ValueError, match="paths"):
        latest_release("policyengine/populace-us", api=hub)

def test_pointer_with_swapped_contract_path_is_refused(hub: FakeHub) -> None:
    payload = latest_pointer_payload(RELEASE_ID)
    payload["paths"]["build_manifest"] = (
        f"releases/{RELEASE_ID}/calibration_diagnostics.json"
    )
    hub.seed_main_file(LATEST_POINTER_PATH, json.dumps(payload).encode())

    with pytest.raises(ValueError, match="malformed=\\['build_manifest'\\]"):
        latest_release("policyengine/populace-us", api=hub)

def test_evidence_pointer_payload_mirrors_the_certified_payload() -> None:
    updated_at = "2026-07-22T13:53:15+00:00"
    certified_shape = latest_pointer_payload(EVIDENCE_RELEASE_ID, updated_at=updated_at)
    payload = latest_evidence_pointer_payload(
        EVIDENCE_RELEASE_ID, updated_at=updated_at
    )
    assert payload == {**certified_shape, "tier": "evidence"}

def test_publish_evidence_release_never_touches_the_certified_pointer(
    hub: FakeHub, evidence_release_dir: Path, artifact_root: Path
) -> None:
    payload = publish_release(
        evidence_release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-07-22T13:53:15+00:00",
        evidence=True,
    )
    assert payload["tier"] == "evidence"
    assert payload["release_id"] == EVIDENCE_RELEASE_ID
    # Immutable tag flow unchanged: the tag is the evidence release id.
    assert [tag["tag"] for tag in hub.tags] == [EVIDENCE_RELEASE_ID]
    # The evidence pointer lands last, in the final main commit.
    final_event, final_commit = hub.events[-1]
    assert final_event == "create_commit"
    assert final_commit["paths"][-1] == LATEST_EVIDENCE_POINTER_PATH
    # The certified pointer is never written, anywhere in the flow.
    assert all(path != LATEST_POINTER_PATH for path, _ in hub.uploads)
    published = json.loads(dict(hub.uploads)[LATEST_EVIDENCE_POINTER_PATH])
    assert published["tier"] == "evidence"
    assert published["release_id"] == EVIDENCE_RELEASE_ID

def test_publish_evidence_release_refuses_a_certified_release_dir(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    """--evidence on a certified-shape release: refused, nothing uploaded.
    The tier must be declared by the artifact, not chosen at publish time."""
    with pytest.raises(ReleaseContractError):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
            evidence=True,
        )
    assert hub.uploads == []

def test_certified_publish_refuses_an_evidence_release_dir(
    hub: FakeHub, evidence_release_dir: Path, artifact_root: Path
) -> None:
    with pytest.raises(ReleaseContractError):
        publish_release(
            evidence_release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
        )
    assert hub.uploads == []

def test_publish_evidence_no_latest_skips_the_evidence_pointer(
    hub: FakeHub, evidence_release_dir: Path, artifact_root: Path
) -> None:
    publish_release(
        evidence_release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        update_latest=False,
        evidence=True,
    )
    final_event, final_commit = hub.events[-1]
    assert final_event == "create_commit"
    assert LATEST_EVIDENCE_POINTER_PATH not in final_commit["paths"]
    assert all(path != LATEST_POINTER_PATH for path, _ in hub.uploads)
    assert (
        final_commit["message"]
        == f"Publish non-default evidence release {EVIDENCE_RELEASE_ID}"
    )

def test_publish_evidence_release_announces_the_tier(
    hub: FakeHub, evidence_release_dir: Path, artifact_root: Path, monkeypatch
) -> None:
    calls: list = []
    monkeypatch.setattr(
        "microcosm.data.release.notify_release",
        lambda repo_id, release_id, updated_at, **kw: calls.append(
            (repo_id, release_id, updated_at, kw)
        ),
    )
    publish_release(
        evidence_release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-07-22T13:53:15+00:00",
        evidence=True,
    )
    assert calls == [
        (
            "policyengine/populace-us",
            EVIDENCE_RELEASE_ID,
            "2026-07-22T13:53:15+00:00",
            {"warn_if_unset": True, "tier": "evidence"},
        )
    ]

def test_publish_then_latest_evidence_release_round_trips(
    hub: FakeHub, evidence_release_dir: Path, artifact_root: Path
) -> None:
    publish_release(
        evidence_release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-07-22T13:53:15+00:00",
        evidence=True,
    )
    pointer = latest_evidence_release("policyengine/populace-us", api=hub)
    assert pointer.release_id == EVIDENCE_RELEASE_ID
    assert pointer.revision == EVIDENCE_RELEASE_ID
    assert pointer.line is None
    assert pointer.tier == "evidence"
    assert pointer.updated_at == "2026-07-22T13:53:15+00:00"
    assert (
        pointer.paths["build_manifest"]
        == f"releases/{EVIDENCE_RELEASE_ID}/build_manifest.json"
    )

def test_latest_release_refuses_an_evidence_tier_pointer(hub: FakeHub) -> None:
    """Tier defense on the certified consumer: if an evidence payload ever
    lands in latest.json, readers refuse it rather than certify it."""
    payload = latest_evidence_pointer_payload(EVIDENCE_RELEASE_ID)
    hub.seed_main_file(LATEST_POINTER_PATH, json.dumps(payload).encode())

    with pytest.raises(ValueError, match="tier"):
        latest_release("policyengine/populace-us", api=hub)

def test_latest_evidence_release_requires_the_evidence_tier(hub: FakeHub) -> None:
    payload = latest_pointer_payload(EVIDENCE_RELEASE_ID)
    hub.seed_main_file(LATEST_EVIDENCE_POINTER_PATH, json.dumps(payload).encode())

    with pytest.raises(ValueError, match="tier"):
        latest_evidence_release("policyengine/populace-us", api=hub)

def test_latest_evidence_release_requires_the_id_segment(hub: FakeHub) -> None:
    payload = latest_evidence_pointer_payload(RELEASE_ID)
    hub.seed_main_file(LATEST_EVIDENCE_POINTER_PATH, json.dumps(payload).encode())

    with pytest.raises(ValueError, match="-evidence-"):
        latest_evidence_release("policyengine/populace-us", api=hub)

def test_certified_latest_pointer_keeps_its_certified_shape(hub: FakeHub) -> None:
    """The certified pointer payload gains no tier field — its bytes are the
    pre-#506 shape, so existing consumers see no drift."""
    payload = latest_pointer_payload(RELEASE_ID, updated_at="2026-06-11T00:00:00+00:00")
    assert "tier" not in payload
    hub.seed_main_file(LATEST_POINTER_PATH, json.dumps(payload).encode())
    pointer = latest_release("policyengine/populace-us", api=hub)
    assert pointer.tier == "certified"

def test_certified_publish_never_touches_the_evidence_pointer(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    """The converse of the evidence-pointer isolation: a certified publish
    moves latest.json only, never latest-evidence.json."""
    publish_release(
        release_dir,
        "policyengine/populace-us",
        api=hub,
        artifact_root=artifact_root,
        updated_at="2026-06-11T13:53:15+00:00",
    )
    assert any(path == LATEST_POINTER_PATH for path, _ in hub.uploads)
    assert all(path != LATEST_EVIDENCE_POINTER_PATH for path, _ in hub.uploads)

def test_latest_release_refuses_any_tier_field(hub: FakeHub) -> None:
    """No certified producer writes a tier field; even 'certified' or null is
    foreign and refused rather than consumed as the default."""
    for tier_value in ("certified", None):
        payload = latest_pointer_payload(RELEASE_ID)
        payload["tier"] = tier_value
        hub.seed_main_file(LATEST_POINTER_PATH, json.dumps(payload).encode())
        with pytest.raises(ValueError, match="tier"):
            latest_release("policyengine/populace-us", api=hub)

def test_publish_refuses_root_artifacts_at_pointer_paths(
    hub: FakeHub, release_dir: Path, evidence_release_dir: Path, artifact_root: Path
) -> None:
    """A manifest-declared root artifact must not be able to smuggle a
    pointer write past the tier's pointer selection — in either direction."""
    _declare_root_artifact(
        evidence_release_dir, key="smuggled_pointer", path=LATEST_POINTER_PATH
    )
    with pytest.raises(ValueError, match="reserved"):
        publish_release(
            evidence_release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
            update_latest=False,
            evidence=True,
        )
    assert hub.uploads == []

    _declare_root_artifact(
        release_dir, key="smuggled_pointer", path=LATEST_EVIDENCE_POINTER_PATH
    )
    with pytest.raises(ValueError, match="reserved"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
        )
    assert hub.uploads == []

@pytest.mark.parametrize(
    "pointer_path",
    [
        "latest-national.json",
        "latest-local-k55000.json",
        "latest-future-line-123.json",
    ],
)
def test_publish_refuses_root_artifacts_at_every_reserved_line_pointer_path(
    hub: FakeHub,
    release_dir: Path,
    artifact_root: Path,
    pointer_path: str,
) -> None:
    _declare_root_artifact(release_dir, key="smuggled_line_pointer", path=pointer_path)

    with pytest.raises(ValueError, match="reserved pointer path"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
        )

    assert hub.uploads == []

def test_publish_refuses_unclean_root_artifact_paths(
    hub: FakeHub, evidence_release_dir: Path, artifact_root: Path
) -> None:
    """'./latest.json' must not dodge the reserved-path comparison and get
    canonicalized to the pointer by the Hub afterwards (sol round-2)."""
    _declare_root_artifact(
        evidence_release_dir, key="smuggled_pointer", path=f"./{LATEST_POINTER_PATH}"
    )
    with pytest.raises(ValueError, match="clean relative POSIX path"):
        publish_release(
            evidence_release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
            update_latest=False,
            evidence=True,
        )
    assert hub.uploads == []

def test_publish_refuses_path_components_in_extra_files(
    hub: FakeHub, release_dir: Path, artifact_root: Path
) -> None:
    """extra_files land under releases/<id>/ — a traversal name could escape
    that prefix once the service canonicalizes the path."""
    outside = release_dir.parent / "escape.json"
    outside.write_text("{}")
    with pytest.raises(ValueError, match="bare file name"):
        publish_release(
            release_dir,
            "policyengine/populace-us",
            api=hub,
            artifact_root=artifact_root,
            extra_files=("../escape.json",),
        )
    assert hub.uploads == []

@pytest.mark.parametrize(
    ("condition", "error_type", "message"),
    [
        ("extra-file-traversal", ValueError, "bare file name"),
        ("missing-extra-file", FileNotFoundError, "extra release file"),
        ("tag-mismatch", ValueError, "tag_name must match"),
        ("no-create-tag", ValueError, "must create the matching"),
        ("missing-artifact-root", ValueError, "pass artifact_root"),
        ("missing-root-artifact", FileNotFoundError, "release artifact"),
        ("root-artifact-hash-mismatch", ValueError, "has sha256"),
        ("reserved-pointer-path", ValueError, "reserved pointer path"),
        ("unclean-root-path", ValueError, "clean relative POSIX path"),
        ("gate-evidence-hash-mismatch", ReleaseContractError, "declares sha256"),
        ("missing-gate-evidence", FileNotFoundError, "qrf_tail_concentration"),
    ],
)
def test_cli_preflight_matches_publisher_guards_before_hub_activity(
    hub: FakeHub,
    release_dir: Path,
    artifact_root: Path,
    monkeypatch,
    condition: str,
    error_type: type[Exception],
    message: str,
) -> None:
    """Real contracts used to pass CLI preflight despite these publisher errors."""
    kwargs = {"artifact_root": artifact_root}
    cli_args = [str(release_dir), "--artifact-root", str(artifact_root)]
    if condition == "extra-file-traversal":
        (release_dir.parent / "escape.json").write_text("{}")
        kwargs["extra_files"] = ("../escape.json",)
        cli_args.extend(["--extra-file", "../escape.json"])
    elif condition == "missing-extra-file":
        kwargs["extra_files"] = ("missing.json",)
        cli_args.extend(["--extra-file", "missing.json"])
    elif condition == "tag-mismatch":
        kwargs["tag_name"] = "different-tag"
        cli_args.extend(["--tag-name", "different-tag"])
    elif condition == "no-create-tag":
        kwargs["create_tag"] = False
        cli_args.append("--no-create-tag")
    elif condition == "missing-artifact-root":
        kwargs = {}
        cli_args = [str(release_dir)]
    elif condition == "missing-root-artifact":
        (artifact_root / "populace_us_2024_calibration.npz").unlink()
    elif condition == "root-artifact-hash-mismatch":
        (artifact_root / "populace_us_2024.h5").write_bytes(b"wrong payload")
    elif condition == "reserved-pointer-path":
        _declare_root_artifact(
            release_dir, key="smuggled_pointer", path=LATEST_EVIDENCE_POINTER_PATH
        )
    elif condition == "unclean-root-path":
        _declare_root_artifact(
            release_dir, key="smuggled_pointer", path=f"./{LATEST_POINTER_PATH}"
        )
    elif condition == "gate-evidence-hash-mismatch":
        # A verdict edited after the manifest bound it (route A PR-3).
        (release_dir / "qrf_tail_concentration.json").write_text("{}")
    elif condition == "missing-gate-evidence":
        # A bound verdict that is not on disk cannot ship: the publisher
        # looks for it as a root artifact and refuses.
        (release_dir / "qrf_tail_concentration.json").unlink()
    else:
        raise AssertionError(f"unhandled condition: {condition}")

    def unexpected_hub_activity(*args, **kwargs):
        pytest.fail("invalid local publication must fail before any Hub activity")

    monkeypatch.setattr(release_module, "_hf_api", unexpected_hub_activity)
    # FakeHub logs writes; forbid its read methods too, so an empty event log
    # proves these failures occurred before any supplied-client activity.
    monkeypatch.setattr(hub, "repo_info", unexpected_hub_activity)
    monkeypatch.setattr(hub, "hf_hub_download", unexpected_hub_activity)

    with pytest.raises(error_type, match=message) as publication_error:
        publish_release(release_dir, "policyengine/populace-us", **kwargs)
    with pytest.raises(error_type) as supplied_hub_error:
        publish_release(release_dir, "policyengine/populace-us", api=hub, **kwargs)
    with pytest.raises(error_type) as cli_publication_error:
        publish_cli.main(cli_args)
    with pytest.raises(error_type) as preflight_error:
        publish_cli.main([*cli_args, "--preflight-only"])

    for error in (supplied_hub_error, cli_publication_error, preflight_error):
        assert type(error.value) is type(publication_error.value)
        assert str(error.value) == str(publication_error.value)
    assert hub.events == []
    assert hub.uploads == []
    assert hub.tags == []

def test_cli_valid_preflight_performs_no_publication_or_notification(
    release_dir: Path, artifact_root: Path, monkeypatch, capsys
) -> None:
    def unexpected_side_effect(*args, **kwargs):
        pytest.fail("preflight must not construct a Hub client, publish, or notify")

    monkeypatch.setattr(release_module, "_hf_api", unexpected_side_effect)
    monkeypatch.setattr(release_module, "notify_release", unexpected_side_effect)
    monkeypatch.setattr(publish_cli, "publish_release", unexpected_side_effect)

    assert (
        publish_cli.main(
            [
                str(release_dir),
                "--artifact-root",
                str(artifact_root),
                "--preflight-only",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {"valid": True, "published": False}

def test_preflight_prepares_the_bound_gate_evidence_for_upload(
    release_dir: Path, artifact_root: Path
) -> None:
    """Route A remediation PR-3: prepare_release uploads only contract files,
    manifest artifacts and extra files, so the gate verdicts ship only because
    the manifest binds them. Preflight must list each one as a release-dir
    upload, never as a root artifact."""
    prepared = release_module.prepare_release(release_dir, artifact_root=artifact_root)

    assert set(GATE_EVIDENCE_FILES.values()) <= set(prepared.filenames)
    assert not set(GATE_EVIDENCE_FILES.values()) & set(prepared.root_artifacts)

@pytest.mark.parametrize("evidence", [False, True])
@pytest.mark.parametrize("release_type", [None, "calibration"])
@pytest.mark.parametrize("argument", ["parent_h5", "compatibility_wheels", "both"])
def test_non_enrichment_publisher_refuses_enrichment_only_arguments(
    hub, release_dir, artifact_root, monkeypatch, release_type, argument, evidence
):
    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if release_type is not None:
        manifest["release_type"] = release_type
        manifest_path.write_text(json.dumps(manifest))
    kwargs = {"artifact_root": artifact_root, "evidence": evidence}
    cli_args = [str(release_dir), "--artifact-root", str(artifact_root)]
    if evidence:
        cli_args.append("--evidence")
    if argument in ("parent_h5", "both"):
        kwargs["parent_h5"] = artifact_root / "populace_us_2024.h5"
        cli_args.extend(["--parent-h5", str(kwargs["parent_h5"])])
    if argument in ("compatibility_wheels", "both"):
        kwargs["compatibility_wheels"] = (artifact_root / "unrelated.whl",)
        cli_args.extend(
            ["--compatibility-wheel", str(kwargs["compatibility_wheels"][0])]
        )

    def no_hub(*args, **kwargs):
        pytest.fail("non-enrichment arguments must fail before Hub activity")

    monkeypatch.setattr(release_module, "_hf_api", no_hub)
    monkeypatch.setattr(hub, "repo_info", no_hub)
    monkeypatch.setattr(hub, "hf_hub_download", no_hub)
    message = "require a source_enrichment release"
    with pytest.raises(ValueError, match=message):
        release_module.prepare_release(release_dir, **kwargs)
    with pytest.raises(ValueError, match=message):
        publish_release(release_dir, "policyengine/populace-us", api=hub, **kwargs)
    with pytest.raises(ValueError, match=message):
        publish_cli.main(cli_args)
    with pytest.raises(ValueError, match=message):
        publish_cli.main([*cli_args, "--preflight-only"])
    assert hub.events == []
    assert hub.uploads == []

def test_inspect_then_promote_reuses_the_immutable_tag(
    release_dir: Path, artifact_root: Path, tmp_path: Path, monkeypatch
) -> None:
    """The documented two-step sequence: publish for inspection under the cut
    tag, then promote the same cut. The second call recognises the existing
    tag that describes this release, writes no second immutable revision
    and only the main commit carrying the pointer (review of #966)."""
    national = _as_uk_national_line_release(release_dir)
    monkeypatch.setattr(
        release_module, "validate_release_dir", lambda _path, **_kwargs: None
    )
    monkeypatch.setattr(release_module, "notify_release", lambda *a, **k: None)
    hub = _uk_hub(tmp_path)
    june_pointer_bytes = hub._commits[hub._refs["main"]][LATEST_POINTER_PATH]

    publish_release(
        national,
        "policyengine/populace-uk-private",
        api=hub,
        artifact_root=artifact_root,
        tag_name=UK_NATIONAL_CUT_TAG,
        update_latest=False,
    )
    assert line_pointer_path("national") not in hub._commits[hub._refs["main"]]
    events_after_inspect = len(hub.events)

    promoted = publish_release(
        national,
        "policyengine/populace-uk-private",
        api=hub,
        artifact_root=artifact_root,
        tag_name=UK_NATIONAL_CUT_TAG,
        line="national",
        updated_at="2026-09-23T12:00:00+00:00",
    )

    kinds = [event for event, _ in hub.events[events_after_inspect:]]
    assert kinds == ["create_commit"], kinds
    assert hub.tags == [{"tag": UK_NATIONAL_CUT_TAG, "revision": "commit-1"}]
    main_files = hub._commits[hub._refs["main"]]
    assert main_files[LATEST_POINTER_PATH] == june_pointer_bytes
    assert json.loads(main_files[line_pointer_path("national")])["revision"] == (
        UK_NATIONAL_CUT_TAG
    )
    assert promoted["revision"] == UK_NATIONAL_CUT_TAG
    assert (
        latest_line_release(
            "policyengine/populace-uk-private", line="national", api=hub
        ).revision
        == UK_NATIONAL_CUT_TAG
    )

def test_retry_after_a_failed_pointer_commit_reuses_the_tag(
    release_dir: Path, artifact_root: Path, tmp_path: Path, monkeypatch
) -> None:
    """A failure between tag creation and the pointer commit leaves the tag;
    the retry stands on it instead of dying on create_tag."""
    national = _as_uk_national_line_release(release_dir)
    monkeypatch.setattr(
        release_module, "validate_release_dir", lambda _path, **_kwargs: None
    )
    monkeypatch.setattr(release_module, "notify_release", lambda *a, **k: None)
    hub = _uk_hub(tmp_path)
    hub.fail_main_commit = True
    with pytest.raises(RuntimeError, match="injected main commit failure"):
        publish_release(
            national,
            "policyengine/populace-uk-private",
            api=hub,
            artifact_root=artifact_root,
            tag_name=UK_NATIONAL_CUT_TAG,
            line="national",
        )
    assert hub.tags == [{"tag": UK_NATIONAL_CUT_TAG, "revision": "commit-1"}]
    assert line_pointer_path("national") not in hub._commits[hub._refs["main"]]

    hub.fail_main_commit = False
    publish_release(
        national,
        "policyengine/populace-uk-private",
        api=hub,
        artifact_root=artifact_root,
        tag_name=UK_NATIONAL_CUT_TAG,
        line="national",
    )

    assert hub.tags == [{"tag": UK_NATIONAL_CUT_TAG, "revision": "commit-1"}]
    assert line_pointer_path("national") in hub._commits[hub._refs["main"]]

def test_promotion_refuses_a_tag_that_describes_another_release(
    release_dir: Path, artifact_root: Path, tmp_path: Path, monkeypatch
) -> None:
    national = _as_uk_national_line_release(release_dir)
    monkeypatch.setattr(
        release_module, "validate_release_dir", lambda _path, **_kwargs: None
    )
    monkeypatch.setattr(release_module, "notify_release", lambda *a, **k: None)
    hub = _uk_hub(tmp_path)
    # A tag already pointing at a revision with no manifest for this release.
    hub._refs[UK_NATIONAL_CUT_TAG] = hub._refs["main"]
    with pytest.raises(ValueError, match="does not describe this release"):
        publish_release(
            national,
            "policyengine/populace-uk-private",
            api=hub,
            artifact_root=artifact_root,
            tag_name=UK_NATIONAL_CUT_TAG,
            line="national",
        )
    # A tag whose manifest is another cut's.
    hub._commits["other"] = {
        f"releases/{UK_NATIONAL_RELEASE_ID}/release_manifest.json": b"{}"
    }
    hub._refs[UK_NATIONAL_CUT_TAG] = "other"
    with pytest.raises(ValueError, match="describes another release"):
        publish_release(
            national,
            "policyengine/populace-uk-private",
            api=hub,
            artifact_root=artifact_root,
            tag_name=UK_NATIONAL_CUT_TAG,
            line="national",
        )
    assert line_pointer_path("national") not in hub._commits[hub._refs["main"]]

def test_line_pointer_path_refuses_the_evidence_line() -> None:
    with pytest.raises(ValueError, match="collides with the evidence-tier pointer"):
        line_pointer_path("evidence")

def test_line_promotion_refuses_a_release_whose_role_is_not_the_lines(
    release_dir: Path, artifact_root: Path, monkeypatch
) -> None:
    """The national line carries national-default releases only; a
    local-area manifest cannot ride the national pointer on its id."""
    national = _as_uk_national_line_release(release_dir)
    manifest_path = national / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_role"] = "non_default_local_area"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        release_module, "validate_release_dir", lambda _path, **_kwargs: None
    )
    with pytest.raises(ValueError, match="publishes only 'national_default'"):
        release_module.prepare_release(
            national,
            artifact_root=artifact_root,
            tag_name=UK_NATIONAL_CUT_TAG,
            line="national",
        )

def test_transient_errors_while_checking_an_existing_tag_propagate(
    release_dir: Path, artifact_root: Path, tmp_path: Path, monkeypatch
) -> None:
    """Only "absent" answers are swallowed: a transient failure looking up
    the tag or its manifest propagates, so a good cut is never orphaned
    and the create path is never re-entered with the tag in place."""
    national = _as_uk_national_line_release(release_dir)
    monkeypatch.setattr(
        release_module, "validate_release_dir", lambda _path, **_kwargs: None
    )
    monkeypatch.setattr(release_module, "notify_release", lambda *a, **k: None)
    hub = _uk_hub(tmp_path)
    original_repo_info = hub.repo_info

    def flaky_repo_info(*, repo_id, repo_type, revision=None):
        if revision == UK_NATIONAL_CUT_TAG:
            raise RuntimeError("injected transient Hub failure")
        return original_repo_info(
            repo_id=repo_id, repo_type=repo_type, revision=revision
        )

    monkeypatch.setattr(hub, "repo_info", flaky_repo_info)
    with pytest.raises(RuntimeError, match="injected transient Hub failure"):
        publish_release(
            national,
            "policyengine/populace-uk-private",
            api=hub,
            artifact_root=artifact_root,
            tag_name=UK_NATIONAL_CUT_TAG,
            line="national",
        )
    assert hub.tags == []
    assert not [event for event, _ in hub.events if event == "create_branch"]

    monkeypatch.setattr(hub, "repo_info", original_repo_info)
    hub._refs[UK_NATIONAL_CUT_TAG] = hub._refs["main"]

    def flaky_download(*, repo_id, filename, repo_type, revision=None):
        raise RuntimeError("injected transient download failure")

    monkeypatch.setattr(hub, "hf_hub_download", flaky_download)
    with pytest.raises(RuntimeError, match="injected transient download failure"):
        publish_release(
            national,
            "policyengine/populace-uk-private",
            api=hub,
            artifact_root=artifact_root,
            tag_name=UK_NATIONAL_CUT_TAG,
            line="national",
        )
    assert line_pointer_path("national") not in hub._commits[hub._refs["main"]]
