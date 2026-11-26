#!/usr/bin/env python3
"""Probe at most 32 complete source households without certifying a release.

This engineering tool preserves selected source rows and their weights. It
does not draw a statistical sample, write a dataset, run a state pass, or
produce a compatibility_recertification_scores document.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import struct
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

# Match the ordinary recertifier: isolate bytecode before Microcosm imports.
if __name__ == "__main__" and sys.pycache_prefix is None:
    sys.pycache_prefix = tempfile.mkdtemp(prefix="recertification-pilot-pycache-")

# Direct scripts put tools/ on sys.path; resolve sibling tools from this checkout.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microcosm.data import compatibility_recertification as recert  # noqa: E402

MAX_HOUSEHOLDS = 32
PILOT_KIND = "compatibility_recertification_engineering_pilot"
PILOT_SCHEMA_VERSION = 1
_CHUNK_BYTES = 8 * 2**20


class EngineeringPilotError(ValueError):
    """The bounded engineering contract cannot hold for these inputs."""


def _validate_limit(max_households: int) -> None:
    if type(max_households) is not int or not 1 <= max_households <= MAX_HOUSEHOLDS:
        raise EngineeringPilotError("max_households must be an integer from 1 to 32")


def _ids(values, label: str):
    import numpy as np

    result = np.asarray(values)
    if result.ndim != 1 or result.dtype.kind not in "iu":
        raise EngineeringPilotError(f"{label} must retain an integer ID dtype")
    if len(np.unique(result)) != len(result):
        raise EngineeringPilotError(f"{label} contains duplicate IDs")
    return result


def _select_rows(household_ids, memberships, group_ids, *, max_households: int):
    """Return source-position masks, retaining every selected group's members."""
    import numpy as np

    from tools import sample_us_export_households as sampler

    _validate_limit(max_households)
    households = _ids(household_ids, "household_id")
    if not len(households):
        raise EngineeringPilotError("the source has no households")
    _ids(memberships["person_id"].to_numpy(), "person_id")
    selected = households[:max_households].copy()
    person_household = memberships["person_household_id"].to_numpy()
    if person_household.dtype.kind not in "iu":
        raise EngineeringPilotError("person_household_id must retain an integer dtype")
    if not np.isin(person_household, households).all():
        raise EngineeringPilotError("person memberships reference a missing household")
    person_mask = np.isin(person_household, selected)
    if not np.isin(selected, person_household[person_mask]).all():
        raise EngineeringPilotError("a selected household has no person members")
    try:
        sampler.assert_units_nest_in_households(memberships.loc[person_mask])
    except ValueError:
        raise EngineeringPilotError(
            "selected group units do not nest within whole households"
        ) from None
    for entity in sampler.US_GROUP_ENTITIES:
        entity_ids = _ids(group_ids[entity], f"{entity}_id")
        members = memberships[f"person_{entity}_id"].to_numpy()
        if members.dtype.kind not in "iu":
            raise EngineeringPilotError(f"person_{entity}_id has a noninteger dtype")
        if not np.isin(members, entity_ids).all():
            raise EngineeringPilotError(
                f"person memberships reference a missing {entity}"
            )
        wanted = np.unique(members[person_mask])
        if not np.isin(wanted, entity_ids).all():
            raise EngineeringPilotError(
                f"selected {entity} memberships lack a group row"
            )
        if entity != "household" and np.isin(members[~person_mask], wanted).any():
            raise EngineeringPilotError(
                f"a selected {entity} reaches another household"
            )
    masks = sampler.entity_row_masks(memberships, group_ids, selected)
    if any(not mask.any() for mask in masks.values()):
        raise EngineeringPilotError("selected households lack a required entity table")
    return selected, masks


def _field(digest, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _scalar_bytes(value) -> bytes:
    """Distinguish stored scalar types; never hash object-array pointers."""
    import numpy as np
    import pandas as pd

    if value is None:
        return b"none"
    if value is pd.NA:
        return b"pandas.NA"
    if value is pd.NaT:
        return b"pandas.NaT"
    if isinstance(value, np.generic):
        return b"numpy:" + value.dtype.str.encode() + b":" + value.tobytes()
    if isinstance(value, bytes):
        return b"bytes:" + value
    if isinstance(value, str):
        return b"str:" + value.encode("utf-8")
    if isinstance(value, bool):
        return b"bool:1" if value else b"bool:0"
    if isinstance(value, int):
        return b"int:" + str(value).encode("ascii")
    if isinstance(value, float):
        return b"float64:" + struct.pack("!d", value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return type(value).__name__.encode() + b":" + value.isoformat().encode()
    raise EngineeringPilotError(
        f"unsupported stored scalar type {type(value).__name__}"
    )


def _array_digest(values) -> str:
    import numpy as np

    array = np.asarray(values)
    digest = hashlib.sha256()
    _field(digest, array.dtype.str.encode())
    _field(digest, json.dumps(array.shape).encode())
    if array.dtype.hasobject:
        for value in array.flat:
            _field(digest, _scalar_bytes(value))
    else:
        _field(digest, np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _frame_inventory(tables: Mapping) -> dict:
    """Hash column order, dtypes and values, including categorical metadata."""
    import pandas as pd

    inventory = {}
    for entity, frame in tables.items():
        if not frame.columns.is_unique or any(not isinstance(c, str) for c in frame):
            raise EngineeringPilotError(f"{entity} has duplicate or nontext columns")
        columns = []
        for column in frame:
            series = frame[column]
            metadata = {"name": column, "dtype": repr(series.dtype)}
            if isinstance(series.dtype, pd.CategoricalDtype):
                metadata["categories_sha256"] = _array_digest(series.cat.categories)
                metadata["ordered"] = series.cat.ordered
            columns.append(
                {**metadata, "values_sha256": _array_digest(series.to_numpy())}
            )
        inventory[entity] = {"rows": len(frame), "columns": columns}
    return inventory


def _read_selected_tables(h5: Path, *, max_households: int, expected_period: int):
    """Scan compact memberships, then retain only complete selected source rows."""
    import pandas as pd

    from tools import sample_us_export_households as sampler

    with pd.HDFStore(str(h5), mode="r") as store:
        keys = {key.strip("/") for key in store.keys()}
        if set(sampler.US_ENTITIES) - keys or sampler.TIME_PERIOD_KEY not in keys:
            raise EngineeringPilotError(
                "source lacks entity tables or explicit time period"
            )
        period = store[sampler.TIME_PERIOD_KEY]
        if len(period) != 1 or type(period.iloc[0]) is bool:
            raise EngineeringPilotError("source time period must contain one integer")
        import numpy as np

        if not isinstance(period.iloc[0], (int, np.integer)):
            raise EngineeringPilotError(
                "source time period must retain an integer dtype"
            )
        source_year = int(period.iloc[0])
        if source_year != expected_period:
            raise EngineeringPilotError("source year differs from the scoring period")
        memberships = sampler.read_table_rows(
            store,
            "person",
            columns=("person_id", *sampler.PERSON_MEMBERSHIP_COLUMNS),
            chunk_bytes=_CHUNK_BYTES,
        )
        group_ids = {
            entity: sampler.read_table_rows(
                store,
                entity,
                columns=(sampler.id_column(entity),),
                chunk_bytes=_CHUNK_BYTES,
            )[sampler.id_column(entity)].to_numpy()
            for entity in sampler.US_GROUP_ENTITIES
        }
        selected, masks = _select_rows(
            group_ids["household"],
            memberships,
            group_ids,
            max_households=max_households,
        )
        tables = {
            entity: sampler.read_table_rows(
                store, entity, row_mask=masks[entity], chunk_bytes=_CHUNK_BYTES
            )
            for entity in sampler.US_ENTITIES
        }
    if not tables["household"]["household_id"].to_numpy().tolist() == selected.tolist():
        raise EngineeringPilotError("selection changed source household order")
    return tables, source_year


def _reviewed_columns(changes: Mapping, raw_verdict: Mapping) -> dict:
    """Accept only the existing reviewed-change authority; retain the raw verdict."""
    failures = recert.reviewed_column_change_failures(changes)
    if failures:
        raise EngineeringPilotError(" ".join(failures))
    # A raw verdict may refuse explicitly reviewed retired columns. It stays
    # false here: neither this receipt nor the reviewed-change check relabels it.
    return {
        "reviewed_changes_accepted": True,
        "changes": changes,
        "raw_verdict": raw_verdict,
    }


def _aggregate_guard(simulation) -> None:
    known = recert.engine_known_periods(
        simulation,
        recert.POPULATION_AGGREGATE_VARIABLES,
        walk_branches=True,
        skip_unknown=True,
    )
    if known:
        names = sorted({name for name, _ in known})
        raise EngineeringPilotError(
            f"incomplete-support engine computed population/state aggregates: {names}"
        )


def _check_output(output: Path, release_dir: Path, artifact_root: Path) -> None:
    if os.path.lexists(output):
        raise FileExistsError(f"{output} exists; refusing to overwrite it")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise EngineeringPilotError(
            "output parent must be an existing regular directory"
        )
    resolved = output.resolve()
    for root in (release_dir.resolve(), artifact_root.resolve()):
        if resolved.is_relative_to(root):
            raise EngineeringPilotError(
                "output must lie outside the input release/artifacts"
            )


def _input_files(release_dir: Path, artifact_root: Path, report: Mapping) -> dict:
    """Bind carried metadata and every declared parent artifact without rewriting it."""
    paths = {f"release/{path.name}": path for path in release_dir.iterdir()}
    for name in report["parent"]["root_artifacts"]:
        child = recert.child_artifact_path(report["parent"], release_dir.name, name)
        paths[f"artifacts/{child}"] = artifact_root / child
    result = {}
    for name, path in sorted(paths.items()):
        if path.is_symlink() or not path.is_file():
            raise EngineeringPilotError(
                "input evidence must use regular nonsymlink files"
            )
        result[name] = recert.sha256_file(path)
    return result


def _write_receipt(output: Path, document: Mapping) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(output, flags, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except BaseException:
        # Do not remove an independently replaced path while handling a failure.
        # An incomplete exclusive output remains evidence and cannot be reused.
        raise


def probe_candidate(
    *,
    release_dir: Path,
    artifact_root: Path,
    output: Path,
    compatibility_wheels: Sequence[Path],
    max_households: int = MAX_HOUSEHOLDS,
) -> dict:
    """Run one native source-preserving household probe; grant no compatibility."""
    _validate_limit(max_households)
    release_dir, artifact_root, output = map(Path, (release_dir, artifact_root, output))
    _check_output(output, release_dir, artifact_root)
    started = time.perf_counter()
    from tools import build_us_compatibility_recertification as scorer_tool

    runtime = scorer_tool.scoring_runtime(tuple(compatibility_wheels))
    recert.require_trusted_bytecode()
    report = recert.validate_recertification_candidate(
        release_dir, artifact_root=artifact_root, require_compatibility=False
    )
    if (
        type(report.get("schema_version")) is not int
        or report.get("schema_version") != 1
        or report.get("parent", {}).get("release_type") != "calibration"
        or report.get("compatibility", {}).get("status") != "pending"
    ):
        raise EngineeringPilotError(
            "pilot requires a pending schema1 calibration parent"
        )
    input_files = _input_files(release_dir, artifact_root, report)
    pilot_source_sha256 = recert.sha256_file(__file__)
    filename = report["dataset"]["filename"]
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise EngineeringPilotError("source filename must be a bare filename")
    h5 = artifact_root / filename
    if h5.is_symlink() or not h5.is_file():
        raise EngineeringPilotError("source H5 must be a regular nonsymlink file")
    source_sha256 = recert.sha256_file(h5)
    if source_sha256 != report["dataset"]["sha256"]:
        raise EngineeringPilotError("source H5 differs from the admitted candidate")
    diagnostics_path = release_dir / "calibration_diagnostics.json"
    diagnostics_sha256 = recert.sha256_file(diagnostics_path)
    diagnostics = json.loads(diagnostics_path.read_text())
    spm = ((diagnostics.get("build") or {}).get("post_export_scoring") or {}).get("spm")
    if not isinstance(spm, dict):
        raise EngineeringPilotError("parent diagnostics lack their SPM selection")
    definitions = recert.drift_metric_definitions(diagnostics)
    metrics = [
        m for m in definitions["metrics"] if m["batching"] == recert.BATCHING_HOUSEHOLD
    ]
    if not metrics or len({m["metric_id"] for m in metrics}) != len(metrics):
        raise EngineeringPilotError("household metric set is empty or repeats an ID")
    excluded = [m["metric_id"] for m in definitions["metrics"] if m not in metrics]

    from microcosm.data import stored_inputs

    stored_tables = stored_inputs.h5_stored_tables(h5)
    if set(stored_tables) != set(recert._ENTITIES):
        raise EngineeringPilotError("source must contain exactly the six entity tables")
    engine = stored_inputs.installed_us_engine()
    changes = recert.stored_column_changes(
        stored_tables,
        engine_variables=engine.variables,
        stored_input_register=stored_inputs.US_STORED_NON_VARIABLE_COLUMNS,
    )
    columns = _reviewed_columns(changes, stored_inputs.h5_verdict(h5, engine=engine))
    selected, year = _read_selected_tables(
        h5, max_households=max_households, expected_period=recert.SCORING_PERIOD
    )
    source_inputs = _frame_inventory(selected)
    if set(selected) != set(recert._ENTITIES):
        raise EngineeringPilotError("selection must retain the six source entities")

    from policyengine_us import Microsimulation
    from policyengine_us.data import USSingleYearDataset, extend_single_year_dataset

    dataset = extend_single_year_dataset(
        USSingleYearDataset(**selected, time_period=year), end_year=year
    )
    if dataset.years != [year]:
        raise EngineeringPilotError("source-year container generated another period")
    source_year_dataset = dataset.get_year(year)
    dataset_tables = {
        entity: getattr(source_year_dataset, entity) for entity in recert._ENTITIES
    }
    if _frame_inventory(dataset_tables) != source_inputs:
        raise EngineeringPilotError("one-year dataset changed selected source inputs")
    frames = scorer_tool._BatchFrames(source_year_dataset, withheld=[])
    ids = selected["household"]["household_id"].to_numpy()
    batch = frames.batch(ids)
    if _frame_inventory(batch) != source_inputs:
        raise EngineeringPilotError("batch construction changed source rows or dtypes")
    if not 1 <= len(ids) <= max_households <= MAX_HOUSEHOLDS:
        raise EngineeringPilotError("selection exceeded the household bound")
    simulation = Microsimulation(dataset=dataset, spm=dict(spm))
    scorer = None
    totals = {}
    try:
        _aggregate_guard(simulation)
        scorer = scorer_tool._BatchScorer(simulation, batch)
        _aggregate_guard(simulation)
        for metric in metrics:
            _aggregate_guard(simulation)
            value = scorer.total(metric)
            _aggregate_guard(simulation)
            if not math.isfinite(value):
                raise EngineeringPilotError("a household metric is not finite")
            totals[metric["metric_id"]] = {"value": value}
        if any(
            _frame_inventory(tables) != source_inputs
            for tables in (selected, dataset_tables, batch)
        ):
            raise EngineeringPilotError(
                "native construction/scoring changed source inputs"
            )
    finally:
        del scorer
        del simulation
        gc.collect()
    recert.verify_loaded_modules()
    bytecode = recert.bytecode_isolation_status()
    if bytecode.get("isolated") is not True:
        raise EngineeringPilotError("native modules ran from untrusted bytecode")
    if recert.sha256_file(h5) != source_sha256:
        raise EngineeringPilotError("source H5 changed during the pilot")
    if recert.sha256_file(diagnostics_path) != diagnostics_sha256:
        raise EngineeringPilotError("parent diagnostics changed during the pilot")
    if _input_files(release_dir, artifact_root, report) != input_files:
        raise EngineeringPilotError("carried input evidence changed during the pilot")
    if recert.sha256_file(__file__) != pilot_source_sha256:
        raise EngineeringPilotError("pilot source changed during execution")
    producer = scorer_tool._producer_identity()
    document = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "kind": PILOT_KIND,
        "scope": "bounded complete-household constructor/scorer engineering only",
        "population_estimate": False,
        "compatibility_certification": "not_attested",
        "source": {
            "dataset": report["dataset"],
            "parent": report["parent"],
            "diagnostics_sha256": diagnostics_sha256,
            "input_files_sha256": input_files,
        },
        "selection": {
            "method": "first source households in stored row order; complete membership closure",
            "requested_max_households": max_households,
            "actual_households": len(ids),
            "household_id_order_sha256": _array_digest(ids),
            "raw_source_input_inventory": source_inputs,
            "weights": "original source household weights, without rescaling",
            "source_period": year,
        },
        "runtime": {"python": platform.python_version(), "packages": runtime},
        "source_columns": columns,
        "measurement": {
            "method": "one native whole-household engineering probe; no state pass",
            "spm": spm,
            "metric_definitions_sha256": recert.canonical_sha256(definitions),
            "excluded_metric_ids": excluded,
            "household_metrics": totals,
        },
        "bytecode": {"isolated": True},
        "producer": producer,
        "pilot_source_sha256": pilot_source_sha256,
        "elapsed_seconds": time.perf_counter() - started,
        "untested": [
            "full population/state passes",
            "baseline/withheld comparison and drift",
            "certification/claim/publication",
            "wrapper/bundle admission",
            "historical-law/scientific adapter and study adoption",
        ],
    }
    _write_receipt(output, document)
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-households", type=int, default=MAX_HOUSEHOLDS)
    parser.add_argument(
        "--compatibility-wheel", type=Path, action="append", required=True
    )
    args = parser.parse_args(argv)
    try:
        document = probe_candidate(
            release_dir=args.release_dir,
            artifact_root=args.artifact_root,
            output=args.output,
            compatibility_wheels=tuple(args.compatibility_wheel),
            max_households=args.max_households,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Engineering pilot refused: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "kind": document["kind"],
                "households": document["selection"]["actual_households"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
