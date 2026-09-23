"""Census of what the 19-node native financial run holds, on invented inputs.

This is a measurement harness, not a contract test. It is not collected by the
default suite (``testpaths = ["packages"]``). Run it explicitly:

    uv run pytest -p conftest -s \
        experiments/native-memory-model-20260923/test_retention_census.py

It reuses the invented, source-issued fixture of
``test_us_graph_atomic_survey_financial.known_financial_run`` (same call:
demographic conditioning, two estimators, no property/status/roles) and runs
the real ``run_atomic_survey_financial`` once cold, with both retention
profiles. Five hooks record *structure*, never values:

1. every observer snapshot's per-entity row count, column count, dtype tokens
   and physical bytes (the per-node shape table);
2. at the call of ``atomic._states`` -- after the independent expected loop,
   when the detached observations, expected/current replay pools, base
   expectations, geography reconstruction, prefix, manifest and loaded
   artifacts are all simultaneously live -- a census of the runner's locals:
   unique physical buffers reachable from each local, deduplicated across
   locals in a fixed attribution order;
3. the same census over the returned run object;
4. inside the nine-node prefix runner, at the return of its second ("fresh")
   geography reconstruction, when its expected pool, its detached
   observations and both reconstructions are live together;
5. in the financial runner just after its 19-node ``run_graph`` returns.

Buffers are deduplicated by physical identity (NumPy root buffer address and
size, Arrow buffer address and size, Python object identity for ``str``/object
cells and ``bytes``). Every DataFrame buffer is also classified by the entity
table it belongs to and whether that table sits at the stacked-source grain,
the cloned grain, or neither, so the bytes can be scaled per row to other
fractions. Invented row counts are tiny; only per-row bytes, column counts,
dtype mixes and object multiplicities are meant to transfer to real scale.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import types
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_current_asec_demographics import _demographic_arguments
from test_us_graph_atomic_survey_population import _support_payload

from microcosm.build.us_runtime import graph_atomic_survey_financial as runner

OUT = Path(
    os.environ.get(
        "MEMORY_CENSUS_OUT", Path(__file__).with_name("retention-census.json")
    )
)

try:  # Arrow is optional in this workspace; graph strings use Python storage.
    import pyarrow as pa
except ImportError:  # pragma: no cover
    pa = None


# --------------------------------------------------------------------------
# Physical buffer identities
# --------------------------------------------------------------------------


def _ndarray_buffers(array):
    root = array
    while isinstance(root.base, np.ndarray):
        root = root.base
    if root.dtype == object:
        # Pointer array plus every distinct referent.
        yield ("np", root.__array_interface__["data"][0], root.nbytes), root.nbytes
        for cell in root.ravel():
            if cell is None or cell is pd.NA or isinstance(cell, bool):
                continue
            if isinstance(cell, float) and cell != cell:
                continue
            yield ("obj", id(cell)), sys.getsizeof(cell)
        return
    base = root.base
    if isinstance(base, (bytes, bytearray, memoryview)):
        size = len(base) if not isinstance(base, memoryview) else base.nbytes
        yield ("pybuf", id(base)), size
        return
    yield ("np", root.__array_interface__["data"][0], root.nbytes), root.nbytes


def _array_buffers(values):
    if isinstance(values, np.ndarray):
        yield from _ndarray_buffers(values)
        return
    if isinstance(values, pd.Index):
        if isinstance(values, pd.RangeIndex):
            return
        yield from _array_buffers(values.array)
        return
    if isinstance(values, pd.Series):
        yield from _array_buffers(values.array)
        return
    if hasattr(values, "_data") and hasattr(values, "_mask"):  # masked
        yield from _ndarray_buffers(values._data)
        yield from _ndarray_buffers(values._mask)
        return
    if pa is not None and hasattr(values, "_pa_array"):
        chunked = values._pa_array
        for chunk in chunked.chunks:
            for buffer in chunk.buffers():
                if buffer is not None:
                    yield ("arrow", buffer.address, buffer.size), buffer.size
        return
    if hasattr(values, "_ndarray"):  # NumpyExtensionArray, StringArray, dt
        yield from _ndarray_buffers(values._ndarray)
        return
    if hasattr(values, "codes") and hasattr(values, "categories"):
        yield from _ndarray_buffers(np.asarray(values.codes))
        yield from _array_buffers(values.categories)
        return
    yield from _ndarray_buffers(np.asarray(values))


def _dtype_token(series):
    dtype = series.dtype
    if isinstance(dtype, pd.StringDtype):
        return f"string[{dtype.storage}]"
    return str(dtype)


def _frame_tables(frame):
    for entity in frame.entities:
        yield "entity", entity, frame.table(entity)
    for name in frame.links:
        yield "link", name, frame.link(name)


# --------------------------------------------------------------------------
# Census over arbitrary object graphs
# --------------------------------------------------------------------------

_SKIP = (
    types.ModuleType,
    type,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    types.CodeType,
    types.FrameType,
    str,
    int,
    float,
    complex,
    bool,
    type(None),
    np.dtype,
)


class Census:
    """Unique physical buffers reachable from named roots, first owner wins."""

    def __init__(self, grains):
        self.grains = grains  # {(entity, rows): grain}
        self.seen = set()
        self.visited = set()
        self.rows = []  # one row per (root, table class)

    def _table(self, root, kind, name, table, owner_frame_id):
        rows = len(table)
        grain = self.grains.get((name, rows), "other")
        bytes_by_dtype = defaultdict(int)
        for column in table.columns:
            series = table[column]
            token = _dtype_token(series)
            for key, size in _array_buffers(series.array):
                if key in self.seen:
                    continue
                self.seen.add(key)
                bytes_by_dtype[token] += size
        for key, size in _array_buffers(table.index):
            if key not in self.seen:
                self.seen.add(key)
                bytes_by_dtype["index"] += size
        total = sum(bytes_by_dtype.values())
        self.rows.append(
            {
                "root": root,
                "kind": kind,
                "table": name,
                "rows": rows,
                "columns": len(table.columns),
                "grain": grain,
                "new_bytes": total,
                "new_bytes_by_dtype": dict(bytes_by_dtype),
                "frame": owner_frame_id,
            }
        )

    def _leaf_bytes(self, root, label, key, size):
        if key in self.seen:
            return
        self.seen.add(key)
        self.rows.append(
            {
                "root": root,
                "kind": label,
                "table": None,
                "rows": None,
                "columns": None,
                "grain": "unscaled",
                "new_bytes": size,
                "new_bytes_by_dtype": {label: size},
                "frame": None,
            }
        )

    def walk(self, root, obj, limit=3_000_000):
        stack = [obj]
        steps = 0
        while stack:
            item = stack.pop()
            steps += 1
            if steps > limit:
                raise RuntimeError(f"census walk exceeded {limit} objects at {root}")
            if isinstance(item, _SKIP):
                continue
            ident = id(item)
            if ident in self.visited:
                continue
            self.visited.add(ident)
            if isinstance(item, (bytes, bytearray)):
                self._leaf_bytes(root, "bytes", ("obj", ident), sys.getsizeof(item))
                continue
            if isinstance(item, memoryview):
                continue
            if type(item).__name__ == "Frame" and hasattr(item, "_tables"):
                for kind, name, table in _frame_tables(item):
                    self.visited.add(id(table))
                    self._table(root, kind, name, table, ident)
                weights = [item.weights_for(e).values for e in item.weighted_entities]
                for values in weights:
                    for key, size in _array_buffers(np.asarray(values)):
                        if key not in self.seen:
                            self.seen.add(key)
                            self.rows.append(
                                {
                                    "root": root,
                                    "kind": "weights",
                                    "table": None,
                                    "rows": len(values),
                                    "columns": 1,
                                    "grain": "weights",
                                    "new_bytes": size,
                                    "new_bytes_by_dtype": {"float64": size},
                                    "frame": ident,
                                }
                            )
                strata = item.strata
                if strata is not None:
                    stack.append(strata)
                stack.append(item._metadata)
                stack.append(item._mass_log)
                continue
            if isinstance(item, pd.DataFrame):
                self._table(root, "dataframe", None, item, None)
                continue
            if isinstance(item, (pd.Series, pd.Index)):
                total = 0
                for key, size in _array_buffers(item):
                    if key not in self.seen:
                        self.seen.add(key)
                        total += size
                if total:
                    self.rows.append(
                        {
                            "root": root,
                            "kind": "series",
                            "table": None,
                            "rows": len(item),
                            "columns": 1,
                            "grain": "series",
                            "new_bytes": total,
                            "new_bytes_by_dtype": {_dtype_token(item): total},
                            "frame": None,
                        }
                    )
                continue
            if isinstance(item, np.ndarray):
                total = 0
                for key, size in _ndarray_buffers(item):
                    if key not in self.seen:
                        self.seen.add(key)
                        total += size
                if total:
                    self.rows.append(
                        {
                            "root": root,
                            "kind": "ndarray",
                            "table": None,
                            "rows": int(item.shape[0]) if item.ndim else 1,
                            "columns": 1,
                            "grain": "ndarray",
                            "new_bytes": total,
                            "new_bytes_by_dtype": {str(item.dtype): total},
                            "frame": None,
                        }
                    )
                continue
            if isinstance(item, (dict, types.MappingProxyType)):
                stack.extend(item.keys())
                stack.extend(item.values())
                continue
            if isinstance(item, (list, tuple, set, frozenset)):
                stack.extend(item)
                continue
            if isinstance(item, types.CellType):
                try:
                    stack.append(item.cell_contents)
                except ValueError:
                    pass
                continue
            state = getattr(item, "__dict__", None)
            if isinstance(state, dict):
                stack.extend(state.values())
            for klass in type(item).__mro__:
                for slot in getattr(klass, "__slots__", ()) or ():
                    if isinstance(slot, str) and hasattr(item, slot):
                        try:
                            stack.append(getattr(item, slot))
                        except Exception:  # noqa: BLE001 - descriptor quirks
                            pass

    def summary(self):
        by_root = defaultdict(lambda: defaultdict(int))
        for row in self.rows:
            by_root[row["root"]][row["grain"]] += row["new_bytes"]
        return {root: dict(grains) for root, grains in by_root.items()}


# --------------------------------------------------------------------------
# Snapshot shapes
# --------------------------------------------------------------------------


def _snapshot_shape(node_id, population):
    frame = population.frame
    tables = {}
    for kind, name, table in _frame_tables(frame):
        seen, physical = set(), 0
        by_dtype = defaultdict(lambda: [0, 0])
        for column in table.columns:
            series = table[column]
            token = _dtype_token(series)
            size = 0
            for key, nbytes in _array_buffers(series.array):
                if key not in seen:
                    seen.add(key)
                    size += nbytes
            physical += size
            by_dtype[token][0] += 1
            by_dtype[token][1] += size
        tables[name] = {
            "kind": kind,
            "rows": len(table),
            "columns": len(table.columns),
            "physical_bytes": physical,
            "dtypes": {
                k: {"columns": v[0], "bytes": v[1]} for k, v in by_dtype.items()
            },
        }
    return {"node": node_id, "version": population.version, "tables": tables}


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------

_LOCAL_ORDER = (
    "prefix",
    "entry",
    "retained",
    "qualified",
    "geography",
    "base_expected",
    "donor",
    "donor_columns",
    "manifest",
    "observed",
    "witnessed",
    "expected",
    "current",
    "tax_populations",
    "loaded",
    "prefix_artifacts",
    "projection_bytes",
    "matrix_bytes",
    "raw",
    "applications",
    "drawn",
    "receipts",
    "store",
    "kernels",
    "compiled",
    "sources",
)

# Inside the nine-node prefix runner (graph_atomic_survey_population), at the
# return of its second, "fresh" geography reconstruction: its expected pool,
# its detached observations, both reconstructions and its manifest are live.
_PREFIX_ORDER = (
    "prefix",
    "preparation_entry",
    "expected",
    "observed",
    "geography",
    "fresh",
    "manifest",
    "result",
    "states",
    "prefix_artifacts",
    "view",
    "store",
    "kernels",
    "compiled",
    "sources",
)


def _census_locals(record, key, local, grains, order):
    gc.collect()
    census = Census(grains)
    for name in order:
        if name in local:
            census.walk(name, local[name])
    census.walk("other_locals", [v for k, v in local.items() if k not in order])
    census.walk("module_state", _module_state())
    census.walk("gc_unreached", _gc_frames(census))
    record[key] = {
        "locals": sorted(local),
        "summary": census.summary(),
        "rows": census.rows,
    }


def _run(root, patch, retention):
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
    call = {
        **arguments,
        "store_root": root / "store",
        "geography_config": config,
        "demographic_conditioning": True,
        "n_estimators": 2,
        "return_values": True,
        "_population_retention": retention,
    }
    # Closures hold only a SimpleNamespace: the runner's live-implementation
    # fence seals closure contents, and a growing dict would move that seal.
    holder = SimpleNamespace(record={"retention": retention, "snapshots": []})
    original_run_graph = runner.run_graph
    original_states = runner.atomic._states
    original_reconstruct = runner.reconstruction.reconstruct_atomic_survey_geography

    def run_graph(compiled, **kwargs):
        observer = kwargs.get("_population_observer")
        if observer is None:
            return original_run_graph(compiled, **kwargs)
        graph_order = tuple(compiled.order)

        def observe(node_id, population):
            shape = _snapshot_shape(node_id, population)
            shape["graph_nodes"] = len(graph_order)
            holder.record["snapshots"].append(shape)
            observer(node_id, population)

        manifest = original_run_graph(
            compiled, **{**kwargs, "_population_observer": observe}
        )
        caller = sys._getframe(1)
        if (
            caller.f_code.co_name == "run_atomic_survey_financial"
            and "post_graph_census" not in holder.record
        ):
            # Just after the 19-node run_graph returns: every detached
            # observation, the base expectations, the financial geography
            # reconstruction, the prefix and the manifest are live together.
            local = dict(caller.f_locals)
            local["manifest"] = manifest
            prefix = local["prefix"]
            grains, _ = _grains_from(
                prefix.manifest.population(runner.survey.CREATE_NODE),
                prefix.clone_population.frame,
            )
            _census_locals(
                holder.record, "post_graph_census", local, grains, _LOCAL_ORDER
            )
            del local, prefix
        return manifest

    def reconstruct(*args, **kwargs):
        result = original_reconstruct(*args, **kwargs)
        caller = sys._getframe(1)
        if (
            caller.f_code.co_name == "run_atomic_survey_population"
            and "geography" in caller.f_locals
            and "prefix_peak_census" not in holder.record
        ):
            local = dict(caller.f_locals)
            local["fresh"] = result
            terminal = local["geography"].nodes[-1].id
            grains, _ = _grains_from(
                local["expected"][runner.survey.CREATE_NODE].frame,
                local["observed"][terminal].frame,
            )
            _census_locals(
                holder.record, "prefix_peak_census", local, grains, _PREFIX_ORDER
            )
            del local
        return result

    def states(*args, **kwargs):
        caller = sys._getframe(1)
        if (
            caller.f_code.co_name == "run_atomic_survey_financial"
            and "states_census" not in holder.record
        ):
            local = dict(caller.f_locals)
            holder.record["states_locals"] = sorted(local)
            prefix = local["prefix"]
            grains, counts = _grains_from(
                prefix.manifest.population(runner.survey.CREATE_NODE),
                prefix.clone_population.frame,
            )
            holder.record["fixture_counts"] = counts
            gc.collect()
            census = Census(grains)
            for name in _LOCAL_ORDER:
                if name in local:
                    census.walk(name, local[name])
            census.walk(
                "other_locals",
                [v for k, v in local.items() if k not in _LOCAL_ORDER],
            )
            census.walk("module_state", _module_state())
            census.walk("gc_unreached", _gc_frames(census))
            holder.record["states_census"] = {
                "summary": census.summary(),
                "rows": census.rows,
            }
            del local, prefix, census
        return original_states(*args, **kwargs)

    patch.setattr(runner, "run_graph", run_graph)
    patch.setattr(runner.atomic, "_states", states)
    patch.setattr(
        runner.reconstruction, "reconstruct_atomic_survey_geography", reconstruct
    )
    result = runner.run_atomic_survey_financial(**call, resume="auto")
    return holder.record, result


def _module_state():
    """Every module-level container of the loaded microcosm modules."""
    state = []
    for name, module in list(sys.modules.items()):
        if not name.startswith("microcosm") or module is None:
            continue
        for value in vars(module).values():
            if isinstance(value, _SKIP):
                continue
            state.append(value)
    return state


def _gc_frames(census):
    """Live tables and frames the named roots did not reach."""
    found = []
    for item in gc.get_objects():
        if id(item) in census.visited:
            continue
        if isinstance(item, pd.DataFrame) or (
            type(item).__name__ == "Frame" and hasattr(item, "_tables")
        ):
            found.append(item)
    return found


def _grains_from(create, clone):
    grains = {}
    for entity in create.entities:
        grains[(entity, create.n(entity))] = "stacked"
    for entity in clone.entities:
        grains.setdefault((entity, clone.n(entity)), "cloned")
    return grains, {
        "stacked": {e: create.n(e) for e in create.entities},
        "cloned": {e: clone.n(e) for e in clone.entities},
    }


@pytest.mark.parametrize("retention", ["all", "compact"])
def test_retention_census(tmp_path, retention):
    with pytest.MonkeyPatch.context() as patch:
        record, result = _run(tmp_path, patch, retention)
    grains, _ = _grains_from(
        result.prefix.manifest.population(runner.survey.CREATE_NODE),
        result.prefix.clone_population.frame,
    )
    gc.collect()
    returned = Census(grains)
    returned.walk("returned_run", result)
    returned.walk("module_state", _module_state())
    record["returned_census"] = {"summary": returned.summary(), "rows": returned.rows}
    record["python"] = sys.version.split()[0]
    record["numpy"] = np.__version__
    record["pandas"] = pd.__version__
    existing = json.loads(OUT.read_text()) if OUT.exists() else {}
    existing[retention] = record
    OUT.write_text(json.dumps(existing, indent=1, sort_keys=True, default=str))
    assert record["snapshots"], "observer recorded nothing"
    assert "prefix_peak_census" in record, "prefix fresh reconstruction not reached"
    assert "post_graph_census" in record, "financial run_graph return not reached"
