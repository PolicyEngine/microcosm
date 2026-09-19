"""Explicitly admitted, hash-pinned ACS structural pilot; no model execution.

Prepare only until the source matrix and independent review are accepted. The
caller supplies an independently frozen pins document and its expected hash.
Only aggregate evidence is written; failures never print rows or exception text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path(
    "/Users/maxghenis/PolicyEngine/_recovered/scratch-backup/893/codex-dense-spm-hours-20260914"
)
CANONICAL = Path(
    "/Users/maxghenis/spm-rebuild-20260908/worktrees/spm-canonical-units-20260914"
)
CANONICAL_HEAD = "bcf45768003bb79addfafb0e6d9c7d2d5e547d9d"
CANONICAL_SHA = "ce0d328d856ca81862b4e80947b6f6269a89bbd842cbdbb1569411b319da5a33"
MANIFEST_SHA = "33c65532972eb7a2ec045768b9510ef4c732a3a42862b0e41ca4792db993b4ee"
RSS_LIMIT = 3 * 1024**3
PROJECTION = (
    "person_id",
    "person_household_id",
    "old_spm_unit_id",
    "proposed_spm_unit_id",
)
ROLE_MAPPING = {"source_observed": "observed_relationship_rule"}
SPM_FIELDS = (
    "spm_unit_id",
    "spm_unit_pre_subsidy_childcare_expenses",
    "receives_housing_assistance",
    "takes_up_housing_assistance_if_eligible",
    "spm_unit_tenure_type",
    "spm_unit_energy_subsidy",
    "spm_unit_source_id",
    "spm_unit_support_channel",
    "spm_unit_support_clone_index",
    "takes_up_tanf_if_eligible",
    "takes_up_snap_if_eligible",
    "spm_unit_spine",
)
REGROUP_TABLES = (
    "spm_units",
    "crosswalk",
    "childcare_ledger",
    "field_provenance",
    "exceptions",
)
TENURE_POLICIES = ("preserve_parent_tenure_v1", "acs_ten4_no_mortgage_v1")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError(reason)


def record_hash(records: list[dict], columns: tuple[str, ...]) -> str:
    selected = [{key: row[key] for key in columns} for row in records]
    selected.sort(key=lambda row: row["person_id"])
    payload = json.dumps(
        selected, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def peak_rss() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def write_json(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(args: argparse.Namespace, output: Path) -> dict:
    started = time.monotonic()
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(180)
    stop = threading.Event()

    def watch_rss() -> None:
        while not stop.wait(0.05):
            if peak_rss() > RSS_LIMIT:
                try:
                    write_json(
                        output / "FAILURE.json",
                        {"status": "fail", "reason": "rss_budget_exceeded"},
                    )
                finally:
                    os._exit(137)

    threading.Thread(target=watch_rss, daemon=True).start()
    try:
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            require(os.environ.get(name) == "1", "numerical_threads_not_one")
        pins_path = args.pins.resolve()
        require(sha(pins_path) == args.expect_pins_sha256, "pins_document_mismatch")
        pins = json.loads(pins_path.read_text())
        require(
            pins["scope"] == "acs_spm_source_512a_prepared_only", "pins_scope_mismatch"
        )
        input_paths = {
            name: Path(entry["path"]).resolve()
            for name, entry in pins["inputs"].items()
        }
        require(
            all(path.suffix in {".json", ".parquet"} for path in input_paths.values()),
            "unapproved_input_format",
        )
        for name, path in input_paths.items():
            require(sha(path) == pins["inputs"][name]["sha256"], "input_pin_mismatch")
        require(
            not pins.get("source_freeze_pending", True), "source_freeze_not_complete"
        )
        required_sources = {
            "packages/microcosm-build/src/microcosm/build/acs_spm_partition.py",
            "packages/microcosm-build/src/microcosm/build/acs_spm_regroup.py",
            "packages/microcosm-build/src/microcosm/build/acs_spm_source_receipt.py",
            "experiments/acs_spm_source_512a.py",
        }
        require(
            required_sources.issubset(pins["source_files"]), "source_pins_incomplete"
        )
        for relative, expected in pins["source_files"].items():
            require(sha(ROOT / relative) == expected, "helper_source_mismatch")
        require(
            sha(CANONICAL / "spm_calculator/units.py") == CANONICAL_SHA,
            "canonical_source_mismatch",
        )
        canonical_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=CANONICAL, text=True
        ).strip()
        require(canonical_head == CANONICAL_HEAD, "canonical_head_mismatch")
        require(
            not subprocess.check_output(
                ["git", "status", "--porcelain=v1"], cwd=CANONICAL, text=True
            ).strip(),
            "canonical_worktree_dirty",
        )
        manifest = json.loads(input_paths["primitive_manifest"].read_text())
        require(
            sha(input_paths["primitive_manifest"]) == MANIFEST_SHA,
            "manifest_pin_mismatch",
        )
        require(
            set(manifest["entities"])
            == {
                "person",
                "household",
                "tax_unit",
                "spm_unit",
                "family",
                "marital_unit",
            },
            "entity_set_mismatch",
        )
        for entity, entry in manifest["entities"].items():
            require(
                input_paths[entity]
                == (input_paths["primitive_manifest"].parent / entry["path"]).resolve(),
                "primitive_path_mismatch",
            )
            require(
                pins["inputs"][entity]["sha256"] == entry["sha256"],
                "primitive_hash_mismatch",
            )
        registry_report = json.loads(input_paths["registry_report"].read_text())
        partition_report = json.loads(input_paths["partition_report"].read_text())
        golden_regroup_report = json.loads(input_paths["regroup_report"].read_text())
        require(
            manifest["provenance"]["preparation_sha256"]
            == partition_report["input_preparation_sha256"],
            "pilot_preparation_mismatch",
        )
        require(
            manifest["provenance"]["parent_file_sha256"]
            == registry_report["parent_sha256"]
            == golden_regroup_report["parent_sha256"],
            "pilot_parent_mismatch",
        )
        # Compute original projections BEFORE calling any new helper. Whole-file
        # hashes remain the old registry's inputs, not hashes of renamed records.
        goldens = {}
        golden_evidence = {}
        for sensitivity in (False, True):
            name = "partner_true" if sensitivity else "partner_false"
            path = input_paths[name]
            require(
                sha(path) == registry_report["input_membership_sha256"][path.name],
                "registry_golden_mismatch",
            )
            require(
                sha(path) == partition_report["policies"][name]["membership_sha256"],
                "partition_golden_mismatch",
            )
            original = json.loads(path.read_text())
            migrated = [
                {
                    **row,
                    "role_source": ROLE_MAPPING.get(
                        row["role_source"], row["role_source"]
                    ),
                }
                for row in original
            ]
            goldens[sensitivity] = migrated
            golden_evidence[name] = {
                "original_file_sha256": sha(path),
                "membership_projection_sha256": record_hash(original, PROJECTION),
                "old_role_label_sha256": record_hash(
                    original, ("person_id", "role_source")
                ),
                "migrated_role_label_sha256": record_hash(
                    migrated, ("person_id", "role_source")
                ),
            }
        require(
            golden_evidence["partner_true"]["membership_projection_sha256"]
            == golden_evidence["partner_false"]["membership_projection_sha256"],
            "golden_sensitivity_membership_mismatch",
        )
        require(
            sha(input_paths["registry_membership"])
            == registry_report["membership_sha256"],
            "registry_membership_mismatch",
        )
        forbidden = (
            "policyengine_us",
            "policyengine_uk",
            "policyengine",
            "microcosm.build.us_runtime",
        )
        attempts = {"country_import": 0, "network": 0, "unapproved_population_read": 0}
        allowed_population = {str(input_paths[name]) for name in manifest["entities"]}

        def audit(event, values):
            if event == "import" and any(
                values[0] == prefix or values[0].startswith(prefix + ".")
                for prefix in forbidden
            ):
                attempts["country_import"] += 1
                raise RuntimeError("country_import_prohibited")
            if event in {
                "socket.connect",
                "socket.connect_ex",
                "socket.getaddrinfo",
                "socket.sendto",
            }:
                attempts["network"] += 1
                raise RuntimeError("network_prohibited")
            if event == "open" and isinstance(values[0], (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(values[0])).resolve()
                if (
                    path.suffix.lower()
                    in {
                        ".h5",
                        ".hdf",
                        ".hdf5",
                        ".npy",
                        ".npz",
                        ".parquet",
                        ".csv",
                        ".dta",
                        ".sas7bdat",
                    }
                    and str(path) not in allowed_population
                ):
                    attempts["unapproved_population_read"] += 1
                    raise RuntimeError("unapproved_population_read")

        sys.addaudithook(audit)
        sys.path[:0] = [
            str(CANONICAL),
            *[str(path) for path in sorted((ROOT / "packages").glob("*/src"))],
        ]
        import pandas as pd

        from microcosm.build.acs_spm_partition import (
            ACS_SPM_DEVELOPMENT_POLICY,
            reconstruct_acs_spm_partition,
        )
        from microcosm.build.acs_spm_regroup import (
            AcsSpmLegacyDefaults,
            regroup_acs_spm_units,
        )
        from microcosm.build.acs_spm_source_receipt import build_acs_spm_source_receipt

        tables = {
            entity: pd.read_parquet(input_paths[entity])
            for entity in manifest["entities"]
        }
        for entity, table in tables.items():
            entry = manifest["entities"][entity]
            require(
                len(table) == entry["row_count"]
                and set(table.columns) == set(entry["columns"]),
                "primitive_shape_mismatch",
            )
        originals = {entity: table.copy(deep=True) for entity, table in tables.items()}
        persons = tables["person"].copy(deep=True)
        households = tables["household"].copy(deep=True)
        units = tables["spm_unit"].loc[:, list(SPM_FIELDS)].copy(deep=True)
        require(
            persons.SPORDER.notna().all()
            and persons.SPORDER.eq(persons.SPORDER.round()).all(),
            "invalid_sporder",
        )
        persons["SPORDER"] = persons.SPORDER.astype("int64")
        require("TYPEHUGQ" not in persons, "unexpected_person_typehugq")
        require(households.household_id.is_unique, "duplicate_household_ids")
        persons["TYPEHUGQ"] = persons.person_household_id.map(
            households.set_index("household_id").TYPEHUGQ
        )
        counts = dict(
            zip(
                households.household_id.astype(int),
                households.NP.astype(int),
                strict=True,
            )
        )
        require(len(persons) == 1183 and len(households) == 512, "pilot_size_mismatch")
        membership = pd.DataFrame(
            json.loads(input_paths["registry_membership"].read_text())
        )
        defaults_report = json.loads(input_paths["defaults_receipt"].read_text())
        defaults = {
            field["name"]: field["legacy_default"]
            for field in defaults_report["fields"]
        }
        null_report = json.loads(input_paths["null_register"].read_text())
        nulls = null_report["reviewed_limitations"][1][
            "engine_input_nulls_excluding_group_quarters_housing"
        ]
        for field in defaults:
            evidence = [row for row in nulls if row["column"] == field]
            require(
                len(evidence) == 1
                and evidence[0]["missing_rows_by_spine"]["acs_2024_1yr"] == 1531614,
                "legacy_null_evidence_mismatch",
            )
        declared = AcsSpmLegacyDefaults(
            "acs_2024_1yr",
            manifest["provenance"]["parent_file_sha256"],
            pins["inputs"]["null_register"]["sha256"],
            defaults,
        )
        normalized_originals = [
            table.copy(deep=True) for table in (persons, units, households, membership)
        ]
        partitions = {}
        for sensitivity in (False, True):
            result = reconstruct_acs_spm_partition(
                persons,
                units,
                household_person_counts=counts,
                policy=ACS_SPM_DEVELOPMENT_POLICY,
                minor_partner_role=sensitivity,
            )
            result.require_resolved()
            golden = pd.DataFrame(goldens[sensitivity])
            selected = result.membership.loc[:, golden.columns].reset_index(drop=True)
            pd.testing.assert_frame_equal(
                selected, golden, check_dtype=False, check_exact=True
            )
            name = "partner_true" if sensitivity else "partner_false"
            records = json.loads(selected.to_json(orient="records"))
            require(
                record_hash(records, PROJECTION)
                == golden_evidence[name]["membership_projection_sha256"],
                "membership_projection_mismatch",
            )
            require(
                record_hash(records, ("person_id", "role_source"))
                == golden_evidence[name]["migrated_role_label_sha256"],
                "role_label_migration_mismatch",
            )
            old_provenance = {
                key: value
                for key, value in partition_report["policies"][name].items()
                if key not in {"membership_sha256", "regrouping"}
            }
            new_provenance = {
                key: value
                for key, value in result.provenance.items()
                if key != "assembler"
            }
            require(
                new_provenance == old_provenance,
                "unexplained_partition_provenance_change",
            )
            require(
                result.provenance["source_unresolved_relationship_households"] == 8,
                "uncertainty_count_mismatch",
            )
            partitions[sensitivity] = result
        receipts = {}
        for policy in TENURE_POLICIES:
            regroup = regroup_acs_spm_units(
                persons,
                units,
                households,
                membership,
                legacy_defaults=declared,
                tenure_policy=policy,
                older_care_evidence=None,
            )
            for table_name in REGROUP_TABLES:
                actual = getattr(regroup, table_name).reset_index(drop=True)
                golden_path = input_paths[f"{policy}/{table_name}"]
                require(
                    sha(golden_path)
                    == golden_regroup_report["policies"][policy]["artifacts"][
                        table_name
                    ]["sha256"],
                    "regroup_golden_pin_mismatch",
                )
                golden = pd.DataFrame(
                    json.loads(golden_path.read_text()), columns=actual.columns
                )
                pd.testing.assert_frame_equal(
                    actual, golden, check_dtype=False, check_exact=True
                )
            require(
                len(regroup.spm_units) == 542
                and regroup.metadata["group_quarters_units_preserved"] == 62,
                "regroup_count_mismatch",
            )
            for key, value in regroup.metadata.items():
                require(
                    value == golden_regroup_report["policies"][policy][key],
                    "unexplained_regroup_provenance_change",
                )
            receipts[policy] = build_acs_spm_source_receipt(
                persons,
                units,
                households,
                partitions=partitions,
                membership=membership,
                regroup=regroup,
                legacy_defaults=declared,
                tenure_policy=policy,
                source_references={
                    name: entry["sha256"] for name, entry in pins["inputs"].items()
                },
            )
        for actual, original in zip(
            (persons, units, households, membership), normalized_originals, strict=True
        ):
            pd.testing.assert_frame_equal(actual, original, check_exact=True)
        for entity, table in tables.items():
            pd.testing.assert_frame_equal(table, originals[entity], check_exact=True)
        for name, path in input_paths.items():
            require(sha(path) == pins["inputs"][name]["sha256"], "input_bytes_changed")
        for relative, expected in pins["source_files"].items():
            require(sha(ROOT / relative) == expected, "source_bytes_changed")
        require(not any(attempts.values()), "forbidden_operation_attempted")
        require(
            not any(
                name == prefix or name.startswith(prefix + ".")
                for name in sys.modules
                for prefix in forbidden
            ),
            "forbidden_module_loaded",
        )
        return {
            "status": "pass",
            "scope": "development_source_only",
            "pins_sha256": args.expect_pins_sha256,
            "source_files": pins["source_files"],
            "source_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "canonical_head": canonical_head,
            "canonical_file_sha256": CANONICAL_SHA,
            "authenticated_input_sha256": {
                name: entry["sha256"] for name, entry in pins["inputs"].items()
            },
            "golden_comparisons": golden_evidence,
            "role_label_mapping": ROLE_MAPPING,
            "enumerated_provenance_changes": [
                "Added actual assembler probe identity to partition provenance",
                "Renamed only role_source source_observed to observed_relationship_rule",
            ],
            "normalization": [
                "SPORDER exact numeric to int64",
                "TYPEHUGQ mapped from household by household_id",
                "Original twelve SPM fields only; categoricals already decoded",
            ],
            "receipts": receipts,
            "primitive_tables_and_source_bytes_unchanged": True,
            "guard_attempts": attempts,
            "peak_rss_bytes": peak_rss(),
            "limits": {
                "cpu_seconds": 120,
                "wall_seconds": 180,
                "rss_bytes": RSS_LIMIT,
                "rss_enforcement": "50ms process peak-RSS watchdog; exit 137",
                "numerical_threads": 1,
            },
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    finally:
        stop.set()
        signal.alarm(0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admit-512a", action="store_true", required=True)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--expect-pins-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(json.dumps({"status": "refused", "error_type": type(error).__name__}))
        return 2
    try:
        report = run(args, output)
    except Exception as error:
        report = {
            "status": "fail",
            "scope": "development_source_only",
            "error_type": type(error).__name__,
        }
    try:
        write_json(output / "REPORT.json", report)
    except OSError as error:
        print(json.dumps({"status": "fail", "error_type": type(error).__name__}))
        return 2
    print(
        json.dumps({"status": report["status"], "report": str(output / "REPORT.json")})
    )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
