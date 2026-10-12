"""The emitter service's readiness path stays off the modelling stack.

``LocalTelemetryEmitter.start`` waits for the service to answer a ping before
the build proceeds, and abandons hosted telemetry if it does not. Everything
the service imports before that answer is startup latency, so the probe below
does the service's pre-readiness work in a fresh interpreter and checks which
modules it loaded. The check is on the module set, not wall time, so it holds
on a loaded host.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from microcosm.build.telemetry_emitter_service import collector as collector_module

#: Modules the service never needs: the modelling stack the build package's
#: exports pull in, the other shards, and the Hugging Face client (needed only
#: by the delivery worker after the service answers, and only without an
#: environment token).
_FORBIDDEN = (
    "torch",
    "scipy",
    "pandas",
    "numpy",
    "pyarrow",
    "microcosm.frame",
    "microcosm.calibrate",
    "microcosm.fit",
    "microcosm.graph",
    "microcosm.diagnostics",
    "microcosm.data",
    "microcosm.build.country_spec",
    "microcosm.build.gates",
    "microcosm.build.ledger_targets",
    "huggingface_hub",
    "httpx",
)

# Same imports as ``python -m microcosm.build.telemetry_emitter_service`` (the
# package and ``main``), then the work ``main`` and ``EmitterService.run`` do
# before binding the socket, then the token lookup with an environment token.
_READINESS_PROBE = r"""
import json
import os
import sys

import microcosm.build.telemetry_emitter_service.main
from microcosm.build.telemetry_emitter_service import collector
from microcosm.build.telemetry_emitter_service.collector import CollectorDelivery
from microcosm.build.telemetry_emitter_service.resources import ProcessTreeSampler
from microcosm.build.telemetry_emitter_service.spool import EventSpool

spool = EventSpool(sys.argv[1])
CollectorDelivery(spool, development_collector_url="http://127.0.0.1:1")
ProcessTreeSampler(os.getpid())
spool.register({"run_id": "probe", "producer_id": "probe"})
print(json.dumps({"token": collector._huggingface_token(), "modules": sorted(sys.modules)}))
"""


def test_service_readiness_path_loads_no_modelling_stack(tmp_path) -> None:
    environment = {**os.environ, "HF_TOKEN": "hf-environment-token"}
    completed = subprocess.run(
        [sys.executable, "-c", _READINESS_PROBE, str(tmp_path / "events.sqlite3")],
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    probe = json.loads(completed.stdout.splitlines()[-1])
    assert probe["token"] == "hf-environment-token"
    leaked = sorted(
        name
        for name in probe["modules"]
        if any(name == root or name.startswith(f"{root}.") for root in _FORBIDDEN)
    )
    assert not leaked


def test_token_lookup_falls_back_to_the_cached_hugging_face_login() -> None:
    """Without an environment token the deferred import reads the cached login.

    The autouse telemetry fixture clears the token variables and points the
    Hugging Face token path into this test's temporary directory.
    """
    from huggingface_hub import constants

    token_path = constants.HF_TOKEN_PATH
    os.makedirs(os.path.dirname(token_path), exist_ok=True)
    with open(token_path, "w") as handle:
        handle.write("hf-cached-login\n")

    assert collector_module._huggingface_token() == "hf-cached-login"


def test_token_lookup_is_empty_without_any_credential() -> None:
    assert not collector_module._huggingface_token()
