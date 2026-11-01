"""Repo-root pytest configuration.

CI invokes pytest as a console script, which does not place the working
directory on ``sys.path``. Tests that exercise the F0 migration tooling
import the ``tools.us_bundle_generation`` package from the repository
root, so the root joins the path here explicitly rather than by the
accident of ``python -m pytest``.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools.ci_test_plan import (  # noqa: E402
    TEST_GROUPS,
    TestGroup,
    group_name_for_collection_path,
)


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="collect tests under tests/integration/",
    )


def _test_group(path: Path) -> TestGroup | None:
    """Return registry metadata for the group containing a collection path."""

    group = group_name_for_collection_path(path)
    return TEST_GROUPS.get(group) if group is not None else None


def pytest_ignore_collect(collection_path: Path, config) -> bool | None:
    """Exclude unavailable test environments before importing their modules."""

    group = _test_group(Path(collection_path))
    if group is None:
        return None
    if group.integration and not config.getoption("--run-integration"):
        return True
    if (
        group.engine_module is not None
        and importlib.util.find_spec(group.engine_module) is None
    ):
        return True
    return None


def pytest_collection_modifyitems(items) -> None:
    """Expose directory-derived country requirements as pytest markers."""

    for item in items:
        group = _test_group(Path(item.path))
        if group is None:
            continue
        if group.integration:
            item.add_marker("integration")
        if group.engine_module is not None:
            item.add_marker(f"requires_{group.country}")


@pytest.fixture(autouse=True)
def fake_telemetry_emitters(monkeypatch, tmp_path):
    """Default every workspace test to in-memory telemetry, never production."""
    from huggingface_hub import constants as hf_constants
    from microcosm_emitter.host import launcher

    from microcosm.build import telemetry_emitter
    from microcosm.build.uk_runtime import orrery_publication
    from test_support.microcosm_build.telemetry import (
        FakeTelemetryEmitter,
        fake_publication_inventory,
        fake_publish_graph,
    )

    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    isolated_hf = tmp_path / "telemetry-hf"
    token_path = isolated_hf / "token"
    monkeypatch.setenv("HF_HOME", str(isolated_hf))
    monkeypatch.setenv("HF_TOKEN_PATH", str(token_path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "telemetry-cache"))
    # HF constants may have been imported before this fixture ran.
    monkeypatch.setattr(hf_constants, "HF_HOME", str(isolated_hf))
    monkeypatch.setattr(hf_constants, "HF_TOKEN_PATH", str(token_path))

    emitters = []

    def start(_cls, **kwargs):
        fields = telemetry_emitter.TelemetryRun.__dataclass_fields__
        run = telemetry_emitter.TelemetryRun(
            **{name: value for name, value in kwargs.items() if name in fields}
        )
        emitter = FakeTelemetryEmitter(
            run, close_service_on_finish=kwargs.get("close_service_on_finish", True)
        )
        emitters.append(emitter)
        return emitter

    monkeypatch.setattr(
        telemetry_emitter.LocalTelemetryEmitter, "start", classmethod(start)
    )
    monkeypatch.setattr(
        orrery_publication, "publication_inventory", fake_publication_inventory
    )
    monkeypatch.setattr(orrery_publication, "publish_graph", fake_publish_graph)

    def unexpected_start(*args, **options):
        pytest.fail("A build test tried to start a real emitter service")

    monkeypatch.setattr(launcher, "start_services", unexpected_start)
    # Reset any cached handle left by a previously loaded build entrypoint.
    for module in tuple(sys.modules.values()):
        if isinstance(
            getattr(module, "_ACTIVE_EMITTER", None),
            telemetry_emitter.LocalTelemetryEmitter,
        ):
            monkeypatch.setattr(module, "_ACTIVE_EMITTER", None)

    yield emitters
    for emitter in emitters:
        emitter.close()


@pytest.fixture(autouse=True)
def _admit_test_gate_signing_keys(request, monkeypatch):
    """Let tests sign with their own UK gate keys.

    Production verifies only keys registered in
    ``microcosm.data.gate_signing_keys``. Tests sign with throwaway keys, so
    every unregistered key counts as an active test key here, while the
    committed entries keep their real status. A test marked
    ``committed_gate_key_registry`` sees the committed registry unchanged.
    """

    if request.node.get_closest_marker("committed_gate_key_registry"):
        return
    try:
        from microcosm.data import gate_signing_keys
    except ImportError:
        return
    committed = gate_signing_keys._registry

    class _AdmitTestKeys(dict):
        def __missing__(self, key_sha256):
            return gate_signing_keys.GateSigningKey(
                key_sha256=key_sha256,
                label="test-key",
                status="active",
                custodian="the test suite",
                held_in="the test",
                registered="test",
                evidence="admitted by the root conftest",
            )

        def get(self, key_sha256, default=None):
            return self[key_sha256]

    def registry(country):
        entries = committed(country)
        return None if entries is None else _AdmitTestKeys(entries)

    monkeypatch.setattr(gate_signing_keys, "_registry", registry)
