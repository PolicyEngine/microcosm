"""Exclusive attribution of the 1/1000 native run's outside-graph CPU.

Reads the completed aggregate report of the 2026-09-19 native attribution run
at 9af56aa8a (``completed-aggregate-report.json``, SHA-256
4e7be8adfe472a684cd8ff86a7139cffb42e8a3699d8072d02775fdd9554976d) and
partitions every sampled stack row in the ``outside_graph_executor_calls``
bucket twice, each partition exclusive and exhaustive:

* by **issuer** -- among the known owner entry points on the stack, the one
  listed first in ``ISSUERS`` (specific owners before their enclosing
  preparation), i.e. which source owner paid; and
* by **operation** -- the first known operation class met walking from the
  leaf toward the root, i.e. what the CPU was spent doing.

It also reports the cross-tabulation and, for each memo-relevant pure
function, its inclusive CPU and call-site count. Values are the harness's
statistical process-CPU estimates charged to the sampled main-thread stack
(0.25 s interval); they are not exact function CPU. Only stdlib is used and
no source, payload or record is read.

    python attribute_outside_graph.py REPORT.json [OUT.json]
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import sys
from pathlib import Path

EXPECTED_REPORT_SHA256 = (
    "4e7be8adfe472a684cd8ff86a7139cffb42e8a3699d8072d02775fdd9554976d"
)

# (label, module basename, function names). Precedence: earliest entry wins.
ISSUERS = (
    (
        "acs_source_catalogue",
        "acs_population_catalogue.py",
        {"issue_acs_source_catalogue"},
    ),
    (
        "acs_native_coverage",
        "acs_native_coverage_binding.py",
        {"issue_acs_native_coverage"},
    ),
    (
        "asec_source_catalogue",
        "asec_population_catalogue.py",
        {"issue_asec_source_catalogue"},
    ),
    (
        "asec_native_population",
        "asec_2024_native_population.py",
        {"load_authenticated_asec_2024_native_population"},
    ),
    (
        "catalogue_selection",
        "survey_catalogue_selection.py",
        {"plan_catalogue_selection"},
    ),
    (
        "predictor_requalification",
        "graph_current_survey_predictors.py",
        {"verify_materialized_current_survey_predictors"},
    ),
    (
        "predictor_requalification",
        "current_survey_predictors.py",
        {"qualify_current_survey_predictors"},
    ),
    (
        "atomic_geography_requalification",
        "survey_atomic_geography.py",
        {"reconstruct_atomic_survey_geography"},
    ),
    (
        "verification_epoch_close",
        "survey_population_preparation.py",
        {"verification_epoch"},
    ),
    (
        "preparation_assembly_and_validation",
        "survey_population_preparation.py",
        {"prepare_authenticated_survey_population"},
    ),
    (
        "preparation_borrow_checks",
        "graph_survey_population.py",
        {"_checked_preparation"},
    ),
    ("run_issuance_seals", "graph_atomic_survey_financial.py", {"_issue_run"}),
)

ZIP_FUNCTIONS = {"_read1", "_read2", "read", "read1", "peek", "readline"}
# (label, predicate over (basename, function)). Leaf-first: first match wins.
OPERATIONS = (
    (
        "zip_member_decompression",
        lambda b, f: b == "__init__.py" and f in ZIP_FUNCTIONS,
    ),
    (
        "live_frame_seal_recomputation",
        lambda b, f: (
            f
            in {
                "_series_digest",
                "_frame_signature",
                "_frame_identity",
                "_frame_witness",
                "frame_content_sha256",
                "_population_stamp",
                "_float_cells",
                "_cells_blob",
                "_index_blob",
            }
        ),
    ),
    (
        "custody_copy_hash_and_byte_guards",
        lambda b, f: (
            f
            in {
                "_capture",
                "_copy",
                "_persisted_sha",
                "_file_identity",
                "feed",
                "_fast_record",
                "_slow_record",
                "_sha",
                "_snapshot",
                "readinto",
            }
        ),
    ),
    (
        "csv_record_parsing",
        lambda b, f: (
            f
            in {
                "_literal_csv_records",
                "_csv_record",
                "_decode_record",
                "lines",
                "_records",
                "_fence",
            }
            or b in {"c_parser_wrapper.py", "readers.py"}
        ),
    ),
    (
        "json_encoding_and_byte_budgets",
        lambda b, f: (
            b in {"encoder.py", "canonical.py", "decoder.py"}
            or f
            in {"_json_chunks", "_json_size", "visit", "charge", "_json", "_charge"}
        ),
    ),
    (
        "pandas_and_numpy_frame_operations",
        lambda b, f: (
            b
            in {
                "base.py",
                "frame.py",
                "generic.py",
                "construction.py",
                "array.py",
                "managers.py",
                "series.py",
                "string_arrow.py",
                "_arraysetops_impl.py",
                "common.py",
                "indexing.py",
                "algorithms.py",
                "categorical.py",
                "missing.py",
                "fromnumeric.py",
                "string_.py",
                "masked.py",
                "blocks.py",
                "numeric.py",
                "ops.py",
                "compute.py",
                "_arrow_string_mixins.py",
                "grouper.py",
                "range.py",
                "multi.py",
                "nanops.py",
            }
        ),
    ),
)

# Deterministic pure functions of (source bytes, arguments, code) that the
# memo lane evaluates, keyed (basename, function).
MEMO_CANDIDATES = (
    ("acs_housing_universe_source.py", "_archive"),
    ("acs_housing_universe_source.py", "_reconstruct"),
    ("acs_housing_universe_source.py", "_select"),
    ("acs_person_coverage_authentication.py", "_inventory"),
    ("acs_population_catalogue.py", "_collect"),
    ("acs_person_coverage_columns.py", "_scan_acs_person_coverage"),
    ("acs_person_coverage_columns.py", "read_acs_person_coverage_columns"),
    ("acs_pums.py", "load_acs_pums_tables"),
    ("asec_coverage_authentication.py", "_reconstruct"),
    ("asec_coverage_authentication.py", "_capture"),
    ("asec_person_income_source.py", "_verify"),
    ("asec_demographic_source.py", "_reconstruct"),
    ("asec_current_money_source.py", "_series_digest"),
)


def _frames(row):
    """(basename, function) from leaf to root."""
    return [(Path(path).name, name) for path, name in row["stack"]]


def issuer(row):
    frames = set(_frames(row))
    for label, module, names in ISSUERS:
        if any((module, name) in frames for name in names):
            return label
    return "other_runner_work"


def operation(row):
    for basename, function in _frames(row):
        for label, predicate in OPERATIONS:
            if predicate(basename, function):
                return label
    return "other_python_validation_and_assembly"


def partition(rows):
    by_issuer = collections.Counter()
    by_operation = collections.Counter()
    cross = collections.defaultdict(collections.Counter)
    inclusive = collections.Counter()
    sites = collections.defaultdict(set)
    total = 0.0
    for row in rows:
        if row["bucket"] != "outside_graph_executor_calls":
            continue
        cpu = row["process_cpu_seconds"]
        total += cpu
        who, what = issuer(row), operation(row)
        by_issuer[who] += cpu
        by_operation[what] += cpu
        cross[who][what] += cpu
        frames = _frames(row)
        seen = set()
        for index, frame in enumerate(frames):
            if frame in MEMO_CANDIDATES and frame not in seen:
                # Count each candidate once per row even under recursion.
                seen.add(frame)
                inclusive[frame] += cpu
                caller = frames[index + 1] if index + 1 < len(frames) else None
                sites[frame].add((who, caller))
    for counter in (by_issuer, by_operation):
        assert math.isclose(sum(counter.values()), total, abs_tol=1e-6)
    return {
        "outside_graph_cpu_seconds": round(total, 6),
        "by_issuer": _rounded(by_issuer),
        "by_operation": _rounded(by_operation),
        "issuer_by_operation": {k: _rounded(v) for k, v in sorted(cross.items())},
        "memo_candidates_inclusive": {
            f"{basename}:{function}": {
                "inclusive_cpu_seconds": round(inclusive[(basename, function)], 6),
                "distinct_issuer_call_sites": len(sites[(basename, function)]),
            }
            for basename, function in MEMO_CANDIDATES
        },
    }


def _rounded(counter):
    return {
        key: round(value, 6)
        for key, value in sorted(counter.items(), key=lambda item: -item[1])
    }


def main(argv):
    report_path = Path(argv[1])
    raw = report_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    report = json.loads(raw)
    result = {
        "schema": "microcosm.native-source-auth-memo.attribution.v1",
        "report_sha256": digest,
        "report_is_expected_9af56aa8a_run": digest == EXPECTED_REPORT_SHA256,
        "native_commit": report["native_commit"],
        "method": (
            "Exclusive partitions of sampled outside-graph stack rows. Issuer: "
            "highest-precedence owner entry on the stack. Operation: first "
            "listed operation class from the stack leaf. Statistical process-"
            "CPU estimates at a 0.25 s sampling interval; not exact function "
            "CPU. Inclusive candidate values overlap and must not be summed."
        ),
        "phases": {},
    }
    for phase in ("cold", "replay"):
        completed = report["phase_accounting"]["completed"][phase]
        entry = partition(completed["sampled_stacks"])
        entry["phase_process_cpu_seconds"] = round(completed["process_cpu_seconds"], 6)
        entry["outside_graph_share_of_phase_cpu"] = round(
            entry["outside_graph_cpu_seconds"] / completed["process_cpu_seconds"], 6
        )
        result["phases"][phase] = entry
    text = json.dumps(result, indent=2, sort_keys=False) + "\n"
    if len(argv) > 2:
        Path(argv[2]).write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
