"""Every release publishes into build folders, and no tagged publisher is left.

``microcosm-publish-release`` (``microcosm.data.publish_cli``) committed a
release's files to Hugging Face directly, tagged them and wrote a
``latest*.json`` pointer file. Once the US, UK and BE release repositories all
recorded their builds in ``releases.json``, it refused every one of them, and
``microcosm-publish-release-build`` publishes each build through the emitter
instead. These tests check that the tagged publisher is gone and that the
checks both commands shared stay with the build-folder publication.
"""

from __future__ import annotations

import importlib.util
import json
import re
import tomllib

import microcosm.data
from microcosm.data import release as release_module
from test_support.paths import REPOSITORY_ROOT

#: The tagged publisher's public names, none of which is left.
TAGGED_PUBLISHER_NAMES = ("publish_release", "prepare_release", "PreparedRelease")

#: How code ran the tagged publisher: its module, or its command (but not
#: ``microcosm-publish-release-build``).
TAGGED_PUBLISHER_REFERENCE = re.compile(
    r"microcosm\.data\.publish_cli|microcosm-publish-release(?!-build)"
)


def test_microcosm_data_installs_no_tagged_publish_command() -> None:
    pyproject = REPOSITORY_ROOT / "packages/microcosm-data/pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text())["project"].get("scripts", {})

    assert "microcosm-publish-release" not in scripts
    assert not [
        name
        for name, target in scripts.items()
        if target.startswith("microcosm.data.publish_cli")
    ]


def test_the_tagged_publisher_and_its_slack_alert_are_gone() -> None:
    assert importlib.util.find_spec("microcosm.data.publish_cli") is None
    assert importlib.util.find_spec("microcosm.data.slack") is None


def test_no_module_exports_the_tagged_publisher() -> None:
    for name in TAGGED_PUBLISHER_NAMES:
        assert not hasattr(microcosm.data, name), name
        assert name not in microcosm.data.__all__, name
        assert not hasattr(release_module, name), name
        assert name not in release_module.__all__, name


def test_the_shared_publication_checks_live_beside_build_folder_preparation(
    tmp_path,
) -> None:
    publication_refusal = getattr(release_module, "publication_refusal", None)
    assert callable(publication_refusal)
    assert "publication_refusal" in release_module.__all__
    (tmp_path / "build_manifest.json").write_text(json.dumps({"non_release": True}))

    refusal = publication_refusal(tmp_path, allow_missing_staging=True)

    assert refusal is not None
    assert "non-release smoke output" in refusal


def test_no_tool_or_package_runs_the_tagged_publisher() -> None:
    assert not (REPOSITORY_ROOT / "tools/publish_release.sh").exists()
    searched = [
        *sorted((REPOSITORY_ROOT / "tools").rglob("*")),
        *sorted((REPOSITORY_ROOT / "packages").glob("*/src/**/*")),
    ]
    references = [
        f"{source.relative_to(REPOSITORY_ROOT)}:{number}"
        for source in searched
        if source.is_file() and source.suffix in {".py", ".sh", ".example", ".toml"}
        for number, text in enumerate(
            source.read_text(errors="replace").splitlines(), start=1
        )
        if TAGGED_PUBLISHER_REFERENCE.search(text)
    ]

    assert references == []
