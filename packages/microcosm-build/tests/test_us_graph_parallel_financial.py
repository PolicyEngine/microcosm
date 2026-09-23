"""Amendment 26 on the native nineteen-node financial graph: workers move no byte.

The same invented, source-issued fixture as
``test_us_graph_atomic_survey_financial.py`` is run cold into two fresh stores,
once sequentially and once with four kernel workers. The runner and its nested
nine-node population run take no worker argument -- their modules' bytes are
bound into the US implementation hashes -- so both runs select their worker
count through ``MICROCOSM_GRAPH_MAX_WORKERS``, the switch amendment 26 provides
for exactly this.

Both graphs' manifest keys, content-addressed projections and every node's
content receipt must be equal, and so must every relative path and SHA-256 in
the two store object trees. The parallel run must also have called native
kernels on worker threads, with more than one running at a time, or the
comparison proves nothing.
"""

import hashlib
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_us_current_asec_demographics import _demographic_arguments
from test_us_graph_atomic_survey_population import _support_payload

import microcosm.graph.scheduling as scheduling
from microcosm.build.us_runtime import graph_atomic_survey_financial as runner
from microcosm.graph.canonical import canonical_json


class _KernelThreads:
    """Records which thread starts each kernel call, and the peak overlap.

    Observed through ``sys.monitoring`` on ``invoke_kernel``'s code object,
    because the native runtime refuses a replaced module attribute: its
    loaded-producer drift check compares every imported alias with its origin.
    Nothing here changes what runs.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.threads = []

    def _start(self, code, offset):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.threads.append(threading.current_thread().name)

    def _stop(self, code, offset, value):
        with self._lock:
            self.active -= 1

    @contextmanager
    def watching(self):
        monitoring = sys.monitoring
        tool = next(
            (tool for tool in (3, 4) if monitoring.get_tool(tool) is None), None
        )
        if tool is None:
            pytest.skip("no free sys.monitoring tool id")
        events = monitoring.events
        code = scheduling.invoke_kernel.__code__
        monitoring.use_tool_id(tool, "microcosm-graph-parallel-probe")
        try:
            monitoring.register_callback(tool, events.PY_START, self._start)
            monitoring.register_callback(tool, events.PY_RETURN, self._stop)
            # invoke_kernel catches every Exception, so it returns normally
            # except under a BaseException, which fails this test anyway.
            monitoring.set_local_events(tool, code, events.PY_START | events.PY_RETURN)
            yield self
        finally:
            monitoring.set_local_events(tool, code, 0)
            for event in (events.PY_START, events.PY_RETURN):
                monitoring.register_callback(tool, event, None)
            monitoring.free_tool_id(tool)


def _objects(store_root: Path) -> dict[str, str]:
    root = store_root / "objects"
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _identity(manifest) -> tuple:
    return (
        manifest.key,
        canonical_json(manifest.content_addressed),
        {
            node_id: canonical_json(receipt._content_payload())
            for node_id, receipt in manifest.nodes.items()
        },
        {node_id: receipt.hit for node_id, receipt in manifest.nodes.items()},
    )


def _runs(root: Path, worker_counts: tuple[str, ...]) -> dict[str, SimpleNamespace]:
    """One invented source tree, then one cold run per worker count.

    The invented fixture is not byte-reproducible across two builds of it, so
    every run reads the same built sources; each gets its own fresh store and
    snapshot directory.
    """

    runs = {}
    with pytest.MonkeyPatch.context() as patch:
        arguments = _demographic_arguments(root, patch, unknown=False, zero=False)
        payload, source_ids = _support_payload()
        support_path = root / "invented-block-support.npz"
        support_path.write_bytes(payload)
        config = runner.reconstruction.AtomicSurveyReconstruction(
            support_path=str(support_path),
            support_sha256=hashlib.sha256(payload).hexdigest(),
            source_ids=tuple(sorted(source_ids.items())),
            seed=17,
        )
        for workers in worker_counts:
            patch.setenv(scheduling.MAX_WORKERS_ENVIRONMENT, workers)
            store_root = root / f"store-w{workers}"
            call = {
                **arguments,
                "snapshot_root": root / f"snapshots-w{workers}",
                "store_root": store_root,
                "geography_config": config,
                "demographic_conditioning": True,
                "n_estimators": 2,
                "return_values": True,
                "resume": "auto",
            }
            with _KernelThreads().watching() as probe:
                run = runner.run_atomic_survey_financial(**call)
            runs[workers] = SimpleNamespace(
                financial=_identity(run.manifest),
                prefix=_identity(run.prefix.manifest),
                order=tuple(run.compiled.order),
                objects=_objects(store_root),
                probe=probe,
            )
    return runs


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    return _runs(tmp_path_factory.mktemp("financial"), ("1", "4"))


def test_four_workers_reproduce_the_sequential_nineteen_node_run(runs):
    sequential, parallel = runs["1"], runs["4"]
    assert len(sequential.order) == 19
    assert parallel.order == sequential.order
    assert parallel.prefix == sequential.prefix
    assert parallel.financial == sequential.financial
    assert sequential.objects, "the sequential run published nothing"
    assert parallel.objects == sequential.objects


def test_the_parallel_run_called_native_kernels_on_overlapping_workers(runs):
    sequential, parallel = runs["1"], runs["4"]
    main = threading.main_thread().name
    assert set(sequential.probe.threads) == {main}
    assert sequential.probe.peak == 1
    assert parallel.probe.threads
    assert all(
        name.startswith(scheduling.THREAD_NAME_PREFIX)
        for name in parallel.probe.threads
    )
    assert len(parallel.probe.threads) == len(sequential.probe.threads)
    assert parallel.probe.peak >= 2
