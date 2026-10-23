#!/usr/bin/env python3
"""Build, and score, a compatibility recertification of a published US release.

Three local steps; none publishes, stages or calibrates:

- ``fetch-parent``: download a published parent's ``releases/<id>/`` files and
  root artifacts at its immutable tag, and check every pinned digest.
- ``build``: package a pending ``compatibility_recertification`` candidate. The
  parent's H5 and every other artifact are hard-linked (copied only across
  devices) or copied byte for byte; nothing is derived. The contract runs
  before the directory is exposed, atomically.
- ``score``: score the candidate's H5 under the installed policyengine-us,
  in household batches, for every metric
  ``microcosm.data.compatibility_recertification.drift_metric_definitions``
  defines. Run it once in an environment with the parent's built-with engine
  and once (``--variant as_stored`` and ``--variant without_newly_read_inputs``)
  in the tested environment; certification combines the three documents.

Scoring runs baseline microsimulations over the whole population. Check
``reservation.py allowed`` first. See ``docs/us-compatibility-recertification.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import gc
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path

from microcosm.data import compatibility_recertification as recert
from microcosm.data import contract, source_enrichment, stored_inputs

_PRODUCER_FILES = recert.PRODUCER_SOURCE_FILES
_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_BATCH_HOUSEHOLDS = 2_000


def _producer_identity() -> dict:
    """The clean commit and source digests publication authenticates."""

    import h5py

    loaded = dict(
        zip(
            _PRODUCER_FILES,
            (
                __file__,
                recert.__file__,
                source_enrichment.__file__,
                stored_inputs.__file__,
                contract.__file__,
            ),
            strict=True,
        )
    )
    for filename, actual in loaded.items():
        if Path(actual).resolve() != (_ROOT / filename).resolve():
            raise ValueError(
                f"Producer must execute this checkout's {filename}; install this "
                "workspace's microcosm-data (editable) in the scoring environment"
            )

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=_ROOT, text=True).strip()

    return {
        "repository": "https://github.com/PolicyEngine/microcosm",
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "runtime": {
            "python": platform.python_version(),
            "hdf5": h5py.version.hdf5_version,
            **{
                package: metadata.version(package)
                for package in ("numpy", "h5py", "microcosm-data")
            },
        },
        "source_files_sha256": {
            filename: recert.sha256_file(_ROOT / filename)
            for filename in _PRODUCER_FILES
        },
    }


# ---------------------------------------------------------------------------
# fetch-parent
# ---------------------------------------------------------------------------


def fetch_parent(
    *, release_id: str, output_dir: Path, repo_id: str = recert.REPO_ID
) -> dict:
    """Download a published parent at its tag into ``output_dir``; verify it."""

    from huggingface_hub import HfApi, hf_hub_download

    if output_dir.exists():
        raise FileExistsError(f"{output_dir} exists; choose a new directory")
    release_dir = output_dir / "releases" / release_id
    artifact_root = output_dir / "artifacts"
    release_dir.mkdir(parents=True)
    artifact_root.mkdir()

    def download(path_in_repo: str, destination: Path) -> None:
        local = hf_hub_download(
            repo_id,
            path_in_repo,
            repo_type="dataset",
            revision=release_id,
            local_dir=destination,
        )
        if not Path(local).is_file():
            raise FileNotFoundError(path_in_repo)

    prefix = f"releases/{release_id}/"
    for name in ("release_manifest.json", "build_manifest.json"):
        download(prefix + name, output_dir)
    manifest = json.loads((release_dir / "release_manifest.json").read_text())
    # A release file is one the Hub serves under releases/<id>/ at the tag; the
    # rest are root artifacts, as the publisher uploaded them.
    published = {
        Path(item.path).name
        for item in HfApi().list_repo_tree(
            repo_id, repo_type="dataset", revision=release_id, path_in_repo=prefix
        )
    }
    for entry in manifest.get("artifacts", {}).values():
        path = entry["path"]
        if Path(path).name == path and path in published:
            download(prefix + path, output_dir)
        else:
            download(path, artifact_root)
    return recert.describe_parent(release_dir, artifact_root)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def _place(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.ENOTSUP}:
            raise
        shutil.copyfile(source, destination)
    destination.chmod(0o400)


def _expose_without_overwrite(staging: Path, output_dir: Path) -> None:
    """Reserve ``output_dir`` with an exclusive mkdir, then rename onto it."""

    occupied = "Output appeared during the build; refusing to overwrite"
    try:
        os.mkdir(output_dir, 0o700)
    except FileExistsError:
        raise FileExistsError(occupied) from None
    try:
        os.rename(staging, output_dir)
    except OSError as exc:
        with contextlib.suppress(OSError):
            os.rmdir(output_dir)
        if exc.errno in {errno.EEXIST, errno.ENOTEMPTY, errno.ENOTDIR}:
            raise FileExistsError(occupied) from exc
        raise


def build_candidate(
    *,
    parent_release_dir: Path,
    parent_artifact_root: Path,
    output_dir: Path,
    release_id: str,
) -> dict:
    """Package a pending recertification of the parent; return its report."""

    parent = recert.describe_parent(parent_release_dir, parent_artifact_root)
    if (
        not release_id.startswith(recert.US_RELEASE_PREFIX)
        or Path(release_id).name != release_id
        or release_id == parent["build_id"]
        or "\\" in release_id
    ):
        raise ValueError("Choose a new bare US release id for the recertification")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Output directory already exists; choose a new candidate")
    producer = _producer_identity()
    parent_manifest = json.loads(
        (parent_release_dir / "release_manifest.json").read_text()
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".recertification-", dir=output_dir.parent))
    try:
        os.chmod(staging, 0o700)
        release_dir = staging / "releases" / release_id
        release_dir.mkdir(parents=True, mode=0o700)
        artifact_root = staging / "artifacts"
        artifact_root.mkdir(mode=0o700)
        for path in parent["root_artifacts"]:
            destination = artifact_root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            _place(parent_artifact_root / path, destination)
        for name in parent["release_files"]:
            shutil.copyfile(parent_release_dir / name, release_dir / name)
        shutil.copyfile(
            parent_release_dir / "release_manifest.json",
            release_dir / recert.PARENT_RELEASE_MANIFEST_FILE,
        )
        shutil.copyfile(
            parent_release_dir / "build_manifest.json",
            release_dir / recert.PARENT_BUILD_MANIFEST_FILE,
        )
        report = {
            "schema_version": recert.REPORT_SCHEMA_VERSION,
            "release_type": recert.RECERTIFICATION_RELEASE_TYPE,
            "parent": parent,
            "dataset": parent["dataset"],
            "transformation": "none",
            "parent_contract": recert.replay_parent_contract(
                release_dir, parent, artifact_root=artifact_root
            ),
            "compatibility": {"status": "pending"},
        }
        if report["parent_contract"]["other_failures"]:
            raise ValueError(
                "The parent fails the release contract on its own bytes: "
                + " ".join(report["parent_contract"]["other_failures"])
            )
        recert.write_json(release_dir / recert.REPORT_FILE, report)
        recert.write_json(
            release_dir / "build_manifest.json",
            {
                "build_id": release_id,
                "release_type": recert.RECERTIFICATION_RELEASE_TYPE,
                "recertified_from": parent["build_id"],
                "code": producer,
                "dataset": parent["dataset"],
                "calibration": {
                    "mode": "inherited",
                    "parent_build_id": parent["build_id"],
                    "diagnostics_sha256": parent["release_files"][
                        "calibration_diagnostics.json"
                    ],
                },
                "staging": {
                    "enabled": False,
                    "reason": "local recertification candidate; no calibration",
                },
            },
        )
        parent_build = parent_manifest["build"]
        manifest = {
            key: parent_manifest[key]
            for key in (
                "schema_version",
                "dataset_role",
                "default_datasets",
                "calibration_diagnostics",
                "compatible_model_packages",
                "compatible_core_packages",
            )
            if key in parent_manifest
        }
        manifest["release_type"] = recert.RECERTIFICATION_RELEASE_TYPE
        manifest["data_package"] = {
            "name": "microcosm-data",
            "version": metadata.version("microcosm-data"),
        }
        manifest["build"] = {
            "build_id": release_id,
            "recertified_from": parent["build_id"],
            **{
                key: parent_build[key]
                for key in (
                    "built_at",
                    "built_with_core_package",
                    "built_with_model_package",
                )
                if key in parent_build
            },
        }
        artifacts = {}
        for key, entry in parent_manifest["artifacts"].items():
            artifacts[key] = {**entry, "revision": release_id}
        for key, name in recert.CHILD_ARTIFACT_KEYS.items():
            artifacts[key] = {
                "kind": "evidence",
                "path": name,
                "repo_id": recert.REPO_ID,
                "revision": release_id,
                "sha256": recert.sha256_file(release_dir / name),
            }
        manifest["artifacts"] = artifacts
        recert.write_json(release_dir / "release_manifest.json", manifest)
        for path in release_dir.iterdir():
            path.chmod(0o400)
        recert.validate_recertification_candidate(
            release_dir, artifact_root=artifact_root
        )
        if (
            _producer_identity()["source_files_sha256"]
            != producer["source_files_sha256"]
        ):
            raise ValueError(
                "Producer source files changed while building the candidate"
            )
        _expose_without_overwrite(staging, output_dir)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def _household_batches(household_ids, batch_households: int):
    for start in range(0, len(household_ids), batch_households):
        yield household_ids[start : start + batch_households]


def _state_batches(household_ids, state_fips):
    for state in sorted(set(state_fips.tolist())):
        yield household_ids[state_fips == state]


class _BatchFrames:
    """The country loader's tables, sliced to whole households."""

    def __init__(self, data, *, withheld: Sequence[str]):
        self.tables = {}
        for entity in recert._ENTITIES:
            frame = getattr(data, entity)
            drop = [column for column in withheld if column in frame.columns]
            self.tables[entity] = frame.drop(columns=drop) if drop else frame
        person = self.tables["person"]
        for group in recert._ENTITIES[1:]:
            column = f"person_{group}_id"
            nested = person.groupby(column)["person_household_id"].nunique()
            if (nested != 1).any():
                raise ValueError(f"{group} units must nest in one household")

    def batch(self, household_ids):
        person = self.tables["person"]
        mask = person["person_household_id"].isin(household_ids)
        people = person.loc[mask]
        tables = {"person": people.reset_index(drop=True)}
        for group in recert._ENTITIES[1:]:
            frame = self.tables[group]
            ids = people[f"person_{group}_id"].unique()
            tables[group] = frame.loc[frame[f"{group}_id"].isin(ids)].reset_index(
                drop=True
            )
        return tables


def _engine_known_periods(simulation, variables) -> set[tuple[str, str]]:
    known = set()
    for variable in variables:
        if variable not in simulation.tax_benefit_system.variables:
            continue
        known.update(
            (variable, str(period))
            for period in simulation.get_holder(variable).get_known_periods()
        )
    return known


def _values(simulation, variable):
    import numpy as np

    result = simulation.calculate(variable, recert.SCORING_PERIOD)
    values = getattr(result, "values", result)
    if hasattr(values, "decode_to_str"):
        values = values.decode_to_str()
    return np.asarray(values)


class _BatchScorer:
    """Scores one batch engine for a set of metric definitions."""

    def __init__(self, simulation, tables):
        import numpy as np

        self.simulation = simulation
        self.tables = tables
        household = tables["household"]
        weight = dict(
            zip(
                household["household_id"].to_numpy(),
                household["household_weight"].to_numpy(),
                strict=True,
            )
        )
        person = tables["person"]
        self.person_weight = np.asarray(
            [weight[h] for h in person["person_household_id"].to_numpy()],
            dtype=np.float64,
        )
        self.group_weight = {
            "household": household["household_weight"].to_numpy(np.float64)
        }
        self.person_group_position = {}
        for group in recert._ENTITIES[1:]:
            ids = tables[group][f"{group}_id"].to_numpy()
            engine_ids = _values(simulation, f"{group}_id")
            if not np.array_equal(engine_ids.astype(ids.dtype), ids):
                raise ValueError(f"engine {group} order differs from the batch table")
            position = {value: index for index, value in enumerate(ids)}
            self.person_group_position[group] = np.asarray(
                [position[value] for value in person[f"person_{group}_id"].to_numpy()]
            )
            if group != "household":
                first = np.full(len(ids), -1)
                first[self.person_group_position[group][::-1]] = np.arange(len(person))[
                    ::-1
                ]
                self.group_weight[group] = self.person_weight[first]
        engine_people = _values(simulation, "person_id")
        if not np.array_equal(
            engine_people.astype(person["person_id"].dtype),
            person["person_id"].to_numpy(),
        ):
            raise ValueError("engine person order differs from the batch table")
        self.entity_of = {
            name: variable.entity.key
            for name, variable in simulation.tax_benefit_system.variables.items()
        }

    def _at(self, variable, entity):
        import numpy as np

        values = _values(self.simulation, variable).astype(np.float64)
        source = self.entity_of[variable]
        target = "household" if entity == "household_members" else entity
        if source == target:
            return values
        if source != "person":
            raise ValueError(f"{variable} is a {source} variable, not one of {target}")
        out = np.zeros(len(self.group_weight[target]), dtype=np.float64)
        np.add.at(out, self.person_group_position[target], values)
        return out

    def total(self, metric) -> float:
        import numpy as np

        entity = metric["entity"]
        arrays = [self._at(variable, entity) for variable in metric["variables"]]
        if metric["combine"] == "nonrefundable_ctc":
            ctc, limit = arrays
            values = np.maximum(np.minimum(ctc, limit), 0.0)
        elif metric["combine"] == "part_b_irmaa":
            gross, base, enrolled = arrays
            values = (gross - base) * enrolled
        else:
            values = np.sum(arrays, axis=0)
        if metric["itemizers_only"]:
            values = values * self._at("tax_unit_itemizes", "tax_unit")
        statistic = metric["statistic"]
        if statistic != recert.STATISTIC_SUM:
            values = recert.signed_component(values, metric["source_variable"] or "")
            if statistic == recert.STATISTIC_POSITIVE_COUNT:
                values = (values > 0).astype(np.float64)
        target = "household" if entity == "household_members" else entity
        weights = (
            self.person_weight if target == "person" else self.group_weight[target]
        )
        return float(np.dot(weights, values))

    def filing_status(self, counts):
        import numpy as np

        engine = _values(self.simulation, "filing_status").astype(str)
        table = self.tables["tax_unit"]
        stated = (
            table["filing_status_input"].astype(str).to_numpy()
            if "filing_status_input" in table.columns
            else np.full(len(engine), "UNSTATED")
        )
        weights = self.group_weight["tax_unit"]
        for s, e, w in zip(stated, engine, weights, strict=True):
            cell = counts.setdefault(s, {}).setdefault(
                e, {"tax_units": 0, "weighted": 0.0}
            )
            cell["tax_units"] += 1
            cell["weighted"] += float(w)

    def roles(self, counts):
        import numpy as np

        person = self.tables["person"]
        if "tax_unit_role_input" not in person.columns:
            return
        head = _values(self.simulation, "is_tax_unit_head").astype(bool)
        spouse = _values(self.simulation, "is_tax_unit_spouse").astype(bool)
        engine = np.where(head, "HEAD", np.where(spouse, "SPOUSE", "DEPENDENT"))
        stored = person["tax_unit_role_input"].astype(str).to_numpy()
        differs = engine != stored
        counts["persons"] += len(stored)
        counts["persons_differing"] += int(differs.sum())
        counts["weighted_persons_differing"] += float(self.person_weight[differs].sum())
        units = np.unique(self.person_group_position["tax_unit"][differs])
        counts["tax_units_differing"] += len(units)
        counts["weighted_tax_units_differing"] += float(
            self.group_weight["tax_unit"][units].sum()
        )


def score_candidate(
    *,
    release_dir: Path,
    artifact_root: Path,
    output: Path,
    variant: str,
    batch_households: int = _DEFAULT_BATCH_HOUSEHOLDS,
    microsimulation_cls=None,
) -> dict:
    """Score the candidate's H5 under the installed engine; write the document."""

    import numpy as np
    from policyengine_us.data import USSingleYearDataset

    if microsimulation_cls is None:
        from policyengine_us import Microsimulation

        microsimulation_cls = Microsimulation
    if output.exists():
        raise FileExistsError(f"{output} exists")
    report = json.loads((release_dir / recert.REPORT_FILE).read_text())
    dataset = report["dataset"]
    h5 = artifact_root / dataset["filename"]
    if recert.sha256_file(h5) != dataset["sha256"]:
        raise ValueError("The candidate H5 differs from its report")
    diagnostics = json.loads((release_dir / "calibration_diagnostics.json").read_text())
    spm = ((diagnostics.get("build") or {}).get("post_export_scoring") or {}).get("spm")
    if not isinstance(spm, dict):
        raise ValueError(
            "The parent's diagnostics record no post-export SPM selection to score under"
        )
    definitions = recert.drift_metric_definitions(diagnostics)
    tables = stored_inputs.h5_stored_tables(h5)
    engine = stored_inputs.installed_us_engine()
    changes = recert.stored_column_changes(
        tables,
        engine_variables=engine.variables,
        stored_input_register=stored_inputs.US_STORED_NON_VARIABLE_COLUMNS,
    )
    if variant == recert.VARIANT_AS_STORED:
        withheld: list[str] = []
    elif variant == recert.VARIANT_WITHOUT_NEWLY_READ:
        withheld = sorted(item["column"] for item in changes[recert.CHANGE_READ])
        if not withheld:
            raise ValueError(
                "This engine newly reads no stored column; nothing to withhold"
            )
    else:
        raise ValueError(f"unknown variant {variant!r}")
    verdict = stored_inputs.h5_verdict(h5, engine=engine)
    producer = _producer_identity()
    frames = _BatchFrames(USSingleYearDataset(file_path=str(h5)), withheld=withheld)
    household = frames.tables["household"]
    household_ids = household["household_id"].to_numpy()
    metrics = {metric["metric_id"]: metric for metric in definitions["metrics"]}
    totals = dict.fromkeys(metrics, 0.0)
    filing_status: dict = {}
    roles = dict.fromkeys(
        (
            "persons",
            "persons_differing",
            "weighted_persons_differing",
            "tax_units_differing",
            "weighted_tax_units_differing",
        ),
        0,
    )
    passes = []
    for batching, batches, allowed in (
        (
            recert.BATCHING_HOUSEHOLD,
            _household_batches(household_ids, batch_households),
            (),
        ),
        (
            recert.BATCHING_STATE,
            _state_batches(household_ids, household["state_fips"].to_numpy()),
            recert.STATE_AGGREGATE_VARIABLES,
        ),
    ):
        selected = [m for m in metrics.values() if m["batching"] == batching]
        count = 0
        for ids in batches:
            batch = frames.batch(ids)
            simulation = microsimulation_cls(
                dataset=USSingleYearDataset(**batch, time_period=recert.SCORING_PERIOD),
                spm=dict(spm),
            )
            known = _engine_known_periods(
                simulation, recert.POPULATION_AGGREGATE_VARIABLES
            )
            scorer = _BatchScorer(simulation, batch)
            for metric in selected:
                totals[metric["metric_id"]] += scorer.total(metric)
            if batching == recert.BATCHING_HOUSEHOLD:
                scorer.filing_status(filing_status)
                scorer.roles(roles)
            computed = (
                _engine_known_periods(simulation, recert.POPULATION_AGGREGATE_VARIABLES)
                - known
            )
            refused = sorted(v for v, _ in computed if v not in allowed)
            if refused:
                raise RuntimeError(
                    f"A {batching} batch engine computed {refused}, which aggregate "
                    "over more than one batch can hold; this metric set cannot be "
                    "scored in batches (microcosm#956)."
                )
            count += 1
            del simulation, scorer, batch
            gc.collect()
        passes.append(
            {"batching": batching, "batches": count, "metrics": len(selected)}
        )
    packages = source_enrichment._runtime_package_identities((), require_wheels=False)
    document = {
        "schema_version": recert.SCORES_SCHEMA_VERSION,
        "kind": recert.SCORES_KIND,
        "dataset": dict(dataset),
        "variant": {"id": variant, "withheld_columns": withheld},
        "metric_definitions_sha256": recert.canonical_sha256(definitions),
        "runtime": {
            "python": platform.python_version(),
            "packages": {
                name: {
                    "version": identity["version"],
                    "source_sha256": identity["source_sha256"],
                }
                for name, identity in packages.items()
            },
        },
        "engine": {
            "label": engine.label,
            "stored_input_verdict": {
                key: verdict[key]
                for key in (
                    "passed",
                    "refused",
                    "registered_non_variables",
                    "register_sha256",
                )
            },
            "column_changes": changes,
        },
        "scoring": {
            "method": "household_batched_country_loader",
            "period": recert.SCORING_PERIOD,
            "spm": spm,
            "batch_households": batch_households,
            "n_households": int(len(household_ids)),
            "passes": passes,
            "weights": "household_weight of each unit's household, from the H5",
        },
        "metrics": {metric_id: {"value": value} for metric_id, value in totals.items()},
        "filing_status": filing_status,
        "tax_unit_roles": roles,
        "code": producer,
    }
    if not all(np.isfinite(value) for value in totals.values()):
        raise ValueError("a scored total is not finite")
    with output.open("x") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-parent")
    fetch.add_argument("--release-id", required=True)
    fetch.add_argument("--output-dir", type=Path, required=True)
    fetch.add_argument("--repo-id", default=recert.REPO_ID)
    build = commands.add_parser("build")
    build.add_argument("--parent-release-dir", type=Path, required=True)
    build.add_argument("--parent-artifact-root", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--release-id", required=True)
    score = commands.add_parser("score")
    score.add_argument("--release-dir", type=Path, required=True)
    score.add_argument("--artifact-root", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument(
        "--variant",
        choices=(recert.VARIANT_AS_STORED, recert.VARIANT_WITHOUT_NEWLY_READ),
        default=recert.VARIANT_AS_STORED,
    )
    score.add_argument(
        "--batch-households", type=int, default=_DEFAULT_BATCH_HOUSEHOLDS
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch-parent":
            result = fetch_parent(
                release_id=args.release_id,
                output_dir=args.output_dir,
                repo_id=args.repo_id,
            )
            print(
                json.dumps({"parent": result["build_id"], "dataset": result["dataset"]})
            )
        elif args.command == "build":
            report = build_candidate(
                parent_release_dir=args.parent_release_dir,
                parent_artifact_root=args.parent_artifact_root,
                output_dir=args.output_dir,
                release_id=args.release_id,
            )
            print(
                json.dumps(
                    {
                        "dataset": report["dataset"],
                        "parent_contract": report["parent_contract"]["status"],
                        "compatibility": report["compatibility"],
                    },
                    indent=2,
                )
            )
        else:
            document = score_candidate(
                release_dir=args.release_dir,
                artifact_root=args.artifact_root,
                output=args.output,
                variant=args.variant,
                batch_households=args.batch_households,
            )
            print(
                json.dumps(
                    {"engine": document["engine"]["label"], "variant": args.variant}
                )
            )
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Compatibility recertification refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
