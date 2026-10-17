"""The dense sign-off drafting tool (microcosm#1301) on synthetic run directories.

Every run directory here is written by the production code a dense build runs:
the real target-fit and area-support gate functions produce the details, the
full gate manifest and enforcement classify them, and the dense replay writes
and signs the local gate report. The rows, support figures and registers are
synthetic, and the area codes are placeholders no area carries (``E06999998``,
``E14999999`` and the like), so no synthetic figure sits beside a real area.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.gate_battery import (
    GateOutcome,
    GatePhaseReport,
    GateStatus,
    gate_phase_report_payload,
)
from microcosm.build.gates import (
    GateResult,
    area_support_below_floor,
    area_support_gate,
)
from microcosm.build.stage_evidence import encode_stage_evidence
from microcosm.build.uk_runtime.full_gates import (
    classify_full_gate_outcomes,
    uk_full_gate_manifest,
    uk_full_gate_scope_receipt,
)
from microcosm.build.uk_runtime.graph_national import replay_uk_dense_gate_battery
from microcosm.build.uk_runtime.terminal_gates import uk_target_fit_gate
from microcosm.build.uk_runtime.weighted_integrity import coerce_reviewed_exclusions
from test_support.git_repositories import NO_BACKGROUND_MAINTENANCE
from test_support.microcosm_build.uk_full_build_cli import (
    SELECTION,
    SIGNING_KEY_ENV,
    STEM,
    TEST_SIGNING_KEY,
)
from test_support.paths import paths_for

REPOSITORY = paths_for("microcosm-build").repository


def _load_tool():
    """``tools/draft_uk_dense_signoff.py`` as a private module.

    It is registered in ``sys.modules`` before it runs: ``dataclasses``
    resolves its string annotations through ``sys.modules[cls.__module__]``.
    """

    name = "draft_uk_dense_signoff_under_test"
    path = REPOSITORY / "tools" / "draft_uk_dense_signoff.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()
#: The unpatched seams, bound before the autouse fixture replaces them.
REAL_REGISTER_WIRING = TOOL.register_wiring
REAL_REVIEW_REGISTERS = TOOL._review_registers

BUILD_CLOCK = date(2026, 10, 20)
APPROVED_ON = "2026-10-22"
GATE_CLOCK = date(2026, 10, 22)
ATTEMPT_1 = "20261020T120000Z-0a1b2c3d"
ATTEMPT_2 = "20261022T120000Z-4e5f6a7b"
SOLVE = {
    "problem": "problem-key",
    "solution": "solution-key",
    "result": "result-key",
    "selection": "selection-key",
    "holdout": "holdout-key",
}
#: Planted figures: they may reach the private review file and nothing else.
PLANTED_ERROR = 0.3137731
PLANTED_ESS = 37.4321
PLANTED_MAX_WEIGHT = 98765.4321
PLANTED = ("3137731", "37.4321", "98765")

NATIONAL = {
    "synthetic.national_a@2025": PLANTED_ERROR,
    "synthetic.national_b@2025": 0.2550213,
    "synthetic.national_c@2025": 0.0213,
    "synthetic.national_d@2025": -0.2461,
    "synthetic.national_e@2025": 0.1042,
}
LOCAL = (
    ("census_households", "E14999998", "households", -0.4021),
    ("census_households", "E14999999", "households", 0.0517),
    ("synthetic_income", "E06999998", "count", 0.1233),
)
#: (level, code, positive rows, distinct sources, ESS)
SUPPORT = (
    ("constituency", "E14999998", 120, 110, 95.5),
    ("constituency", "E14999999", 130, 125, 101.25),
    ("local_authority", "E06999998", 40, 38, PLANTED_ESS),
    ("local_authority", "E06999999", 60, 55, 55.5),
)


def _entry(approved_on: str, expires_on: str, *, lines=("local",), reason=None):
    payload = {
        "reason": reason or "Synthetic reviewed entry for the sign-off tests.",
        "approved_by": "test-reviewer",
        "adjudication": "microcosm#1301 (synthetic)",
        "approved_on": approved_on,
        "expires_on": expires_on,
    }
    if lines is not None:
        payload["release_lines"] = list(lines)
    return payload


def _measure_exclusion(expires_on: str) -> dict:
    """A measure exclusion as the target compilation receipts it (synthetic)."""

    return {
        "reason": "Synthetic measure exclusion for the sign-off tests.",
        "tracking": "microcosm#1301",
        "approved_by": "test-reviewer",
        "adjudication": "microcosm#1301 (synthetic)",
        "approved_on": "2026-09-01",
        "expires_on": expires_on,
        "evaluated_on": BUILD_CLOCK.isoformat(),
    }


#: The measure exclusions build 1's solve ran under, expiring well after the
#: drafted entries unless a test says otherwise.
MEASURE_EXCLUSIONS = {"synthetic.measure_a@2025": _measure_exclusion("2027-01-31")}


def degenerate_register(entries: dict | None = None) -> dict:
    """A degenerate register; by default one local entry expiring late."""

    return {
        "schema_version": 2,
        "description": "Synthetic degenerate register.",
        "exclusions": {
            "household.synthetic_constant": _entry("2026-09-01", "2027-03-01")
        }
        if entries is None
        else entries,
    }


def build_1_registers() -> dict:
    """The committed registers build 1 stood on."""

    return {
        "uk_target_fit": {
            "schema_version": 2,
            "description": "Synthetic national target-fit register.",
            "exclusions": {
                # Back inside the bound on build 1: build 2 fails it as stale.
                "synthetic.national_e@2025": _entry("2026-10-01", "2026-10-29"),
            },
        },
        "uk_local_target_fit": {
            "schema_version": 2,
            "description": "Synthetic local target-fit register.",
            "exclusions": {},
        },
        "uk_local_area_support": {
            "schema_version": 3,
            "description": "Synthetic local-area support register.",
            "exclusions": {
                # Every floor met on build 1: stale.
                "local_authority/E06999999": _entry(
                    "2026-09-02", "2026-12-02", lines=None
                ),
            },
            "bound_despite_support_floor": {
                "census_households": _entry("2026-09-10", "2026-12-10", lines=None),
            },
        },
    }


INDENTS = {"uk_target_fit": 1, "uk_local_target_fit": 1, "uk_local_area_support": 2}
SPECS = {spec.gate_id: spec for spec in TOOL.REGISTERS}

PIN_SOURCE = '''"""Synthetic register pins."""

from microcosm.build.uk_runtime.terminal_gates import (
    uk_default_local_target_fit_reviewed_exclusions,
    uk_default_target_fit_reviewed_exclusions,
)


def test_committed_target_fit_register_retains_only_live_deferrals() -> None:
    register = uk_default_target_fit_reviewed_exclusions()
    # The history of the register's deferrals.
    assert set(register) == {
        "synthetic.national_e@2025",
    }
    assert {record.release_lines for record in register.values()} == {("local",)}


def test_committed_local_target_fit_register_is_empty() -> None:
    assert uk_default_local_target_fit_reviewed_exclusions() == {}
'''

AREA_PIN_SOURCE = '''"""Synthetic area-register pins."""

from microcosm.build.uk_runtime.weighted_integrity import (
    load_uk_local_area_support_exclusion_register,
)


def test_local_area_support_register_loads_both_schema3_blocks() -> None:
    loaded = load_uk_local_area_support_exclusion_register(None)
    assert set(loaded["exclusions"]) == {
        "local_authority/E06999999",
    }
    assert set(loaded["bound_despite_support_floor"]) == {"census_households"}
'''


def generator_source(deferred=()) -> str:
    """A local reference generator declaring its own deferred areas."""

    return (
        '"""Synthetic local reference generator."""\n\n'
        f"{TOOL.DEFERRED_AREAS_CONSTANT}: tuple[str, ...] = {tuple(deferred)!r}\n"
    )


def _dump(payload: dict, indent: int) -> str:
    return json.dumps(payload, indent=indent, ensure_ascii=False) + "\n"


def make_repo(
    root: Path, registers: dict, *, deferred=(), degenerate: dict | None = None
) -> Path:
    for gate_id, payload in registers.items():
        path = root / SPECS[gate_id].path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_dump(payload, INDENTS[gate_id]), encoding="utf-8")
    for relative, text in (
        (TOOL.TERMINAL_GATES_TEST, PIN_SOURCE),
        (TOOL.WEIGHTED_INTEGRITY_TEST, AREA_PIN_SOURCE),
        (TOOL.LOCAL_REFERENCE_GENERATOR, generator_source(deferred)),
        (TOOL.DEGENERATE_REGISTER, _dump(degenerate or degenerate_register(), 1)),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def read_registers(root: Path) -> dict:
    return {
        gate_id: json.loads((root / spec.path).read_text(encoding="utf-8"))
        for gate_id, spec in SPECS.items()
    }


def _floors() -> dict:
    (entry,) = (
        entry
        for entry in load_country_spec("uk").gates.gates
        if entry.id == "uk_local_area_support"
    )
    return {
        "minimum_rows": int(entry.parameters["minimum_rows"]),
        "minimum_effective_sample_size": float(
            entry.parameters["minimum_effective_sample_size"]
        ),
        "minimum_distinct_sources": int(entry.parameters["minimum_distinct_sources"]),
    }


def _support_frame(support) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "area_code": code,
                "assigned_households": rows + 5,
                "nonzero_households": rows,
                "nonzero_source_households": sources,
                "weight_sum": 1000.0,
                "max_weight": PLANTED_MAX_WEIGHT,
                "effective_sample_size": ess,
                "geography_level": level,
            }
            for level, code, rows, sources, ess in support
        ]
    )
    frame["support_below_floor"] = area_support_below_floor(frame, **_floors())
    return frame


def _target_rows(national, local) -> list[dict]:
    rows = [
        {
            "name": name,
            "target_name": name,
            "target": 100.0,
            "estimate": 100.0 * (1.0 + error),
            "relative_error": error,
            "abs_relative_error": abs(error),
            "family": "synthetic_national",
            "geography_level": "country",
            "area_type": "country",
            "area_code": "",
            "metric": name,
        }
        for name, error in national.items()
    ]
    for family, area, metric, error in local:
        name = f"synthetic.local.{family}.{area}@2025"
        level = "constituency" if area.startswith("E14") else "local_authority"
        rows.append(
            {
                "name": name,
                "target_name": name,
                "target": 50.0,
                "estimate": 50.0 * (1.0 + error),
                "relative_error": error,
                "abs_relative_error": abs(error),
                "family": family,
                "geography_level": level,
                "area_type": level,
                "area_code": area,
                "metric": metric,
            }
        )
    return rows


def _fit(errors: dict, block: dict, clock: date) -> GateResult:
    records = coerce_reviewed_exclusions(block, label="fixture")
    in_line = {key: r for key, r in records.items() if r.applies_on("local")}
    result = uk_target_fit_gate(errors, reviewed_exclusions=in_line, now=clock)
    out = sorted(set(records) - set(in_line))
    if out:
        result = replace(
            result,
            details={
                **result.details,
                "release_line": "local",
                "out_of_line_exclusions": out,
            },
        )
    return result


def _area(frame: pd.DataFrame, register: dict, clock: date) -> GateResult:
    records = {
        key: record
        for key, record in coerce_reviewed_exclusions(
            register["exclusions"], label="fixture"
        ).items()
        if record.applies_on("local")
    }
    bound = coerce_reviewed_exclusions(
        register["bound_despite_support_floor"], label="fixture"
    )
    invalid = sorted(
        f"{block}/{key}"
        for block, entries in (
            ("exclusions", records),
            ("bound_despite_support_floor", bound),
        )
        for key, record in entries.items()
        if record.expired(clock) or record.premature(clock)
    )
    if invalid:
        return GateResult(
            name="area_support",
            passed=False,
            failures=tuple(f"{key}: outside its window." for key in invalid),
            details={"invalid_reviewed_exclusions": invalid},
        )
    levels = ["constituency", "local_authority"]
    return area_support_gate(
        frame,
        area_roster={
            level: frame.loc[frame["geography_level"] == level, "area_code"].tolist()
            for level in levels
        },
        geography_levels=levels,
        reviewed_exclusions={k: r.policy_payload() for k, r in records.items()},
        **_floors(),
    )


def write_run(
    path: Path,
    *,
    registers: dict,
    attempt_id: str = ATTEMPT_1,
    clock: date = BUILD_CLOCK,
    national: dict | None = None,
    local=LOCAL,
    support=SUPPORT,
    failed: dict | None = None,
    release_candidate: bool = True,
    sample_fraction: float = 1.0,
    artifacts: dict | None = None,
    measure_exclusions: dict | None = None,
) -> Path:
    """A dense run directory as a blocked or passing build leaves it.

    Like a build its gates stopped before the export, it holds the target
    compilation's surface but no candidate manifest; the surface here carries
    only the measure-exclusion receipt.
    """

    national = dict(NATIONAL if national is None else national)
    targets = _target_rows(national, local)
    support_frame = _support_frame(support)
    results = {
        "uk_target_fit": _fit(
            national, registers["uk_target_fit"]["exclusions"], clock
        ),
        "uk_local_target_fit": _fit(
            {f"{f}/{a}/{m}": e for f, a, m, e in local},
            registers["uk_local_target_fit"]["exclusions"],
            clock,
        ),
        "uk_local_area_support": _area(
            support_frame, registers["uk_local_area_support"], clock
        ),
    }
    for gate_id, details in (failed or {}).items():
        results[gate_id] = GateResult(
            name=gate_id,
            passed=False,
            failures=("synthetic failure",),
            details=details,
        )
    gates = uk_full_gate_manifest(SELECTION)
    outcomes = []
    for entry in gates.gates:
        if entry.phase != "terminal":
            continue
        if entry.not_applicable is not None:
            outcomes.append(
                GateOutcome(
                    entry, GateStatus.NOT_APPLICABLE, reason=entry.not_applicable
                )
            )
            continue
        result = results.get(entry.id) or GateResult(
            name=entry.gate, passed=True, details={}
        )
        status = GateStatus.PASSED if result.passed else GateStatus.FAILED
        outcomes.append(GateOutcome(entry, status, result))
    report = GatePhaseReport("terminal", tuple(outcomes))
    document = {
        "schema_version": 1,
        "kind": "uk_full_gate_report",
        "selection_receipt": SELECTION,
        "sample_fraction": sample_fraction,
        "release_candidate": release_candidate,
        "scope": uk_full_gate_scope_receipt(SELECTION),
        "report": gate_phase_report_payload(report, gates=gates),
        "upstream_phase_reports": {},
        "target_diagnostics": targets,
        "area_support": support_frame.to_dict(orient="records"),
        "artifacts": dict(SOLVE if artifacts is None else artifacts),
        "enforcement": classify_full_gate_outcomes(
            report,
            sample_fraction=sample_fraction,
            release_candidate=release_candidate,
        ),
    }
    path.mkdir(parents=True)
    payload = encode_stage_evidence(document)
    (path / TOOL.FULL_GATE_REPORT_FILENAME).write_bytes(payload)
    replay_uk_dense_gate_battery(
        json.loads(payload),
        report_path=path / f"{STEM}.local_gates.json",
        release_id=attempt_id,
        release_candidate=release_candidate,
    )
    pd.DataFrame(targets).to_csv(path / f"{STEM}.targets.csv", index=False)
    support_frame.to_csv(path / f"{STEM}.area_support.csv", index=False)
    (path / TOOL.TARGET_SURFACE_FILENAME).write_text(
        json.dumps(
            {
                "measure_exclusions": MEASURE_EXCLUSIONS
                if measure_exclusions is None
                else measure_exclusions
            }
        )
    )
    spool = path / "logbook-spool"
    spool.mkdir()
    (spool / "row.json").write_text(json.dumps({"build_id": attempt_id}))
    return path


@pytest.fixture(autouse=True)
def _synthetic_environment(monkeypatch):
    """Sign with the test key, wire the local target-fit register, keep the
    real registers' windows out of the national review invariant (its own
    tests below exercise it) and verify on build 2's gate clock. The signed
    reports carry the live dense digests, so --verify checks them against the
    contract's committed pin."""

    monkeypatch.setenv(SIGNING_KEY_ENV, TEST_SIGNING_KEY)
    monkeypatch.setattr(
        TOOL, "register_wiring", lambda gates=None: dict.fromkeys(SPECS)
    )
    monkeypatch.setattr(TOOL, "_review_registers", list)
    monkeypatch.setattr(TOOL, "_today", lambda: GATE_CLOCK)


@pytest.fixture
def repo(tmp_path) -> Path:
    return make_repo(tmp_path / "repo", build_1_registers())


@pytest.fixture
def build_1(tmp_path) -> Path:
    return write_run(tmp_path / "build-1", registers=build_1_registers())


def draft(build: Path, repo: Path, out: Path, *extra: str) -> int:
    return TOOL.main(
        [
            "--run-dir",
            str(build),
            "--approved-by",
            "juaristi22",
            "--approved-on",
            APPROVED_ON,
            "--repo-root",
            str(repo),
            "--out",
            str(out),
            *extra,
        ]
    )


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *NO_BACKGROUND_MAINTENANCE, *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def apply_patch(repo: Path, patch: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    git(repo, "init", "--quiet")
    git(repo, "apply", "--check", str(patch))
    git(repo, "apply", str(patch))


def test_draft_signs_every_signable_failure_and_retires_stale_entries(
    build_1, repo, tmp_path, capsys
):
    out = tmp_path / "draft"
    assert draft(build_1, repo, out) == 0
    receipt = json.loads((out / "signoff_draft.json").read_text())

    assert receipt["signable"] is True
    assert receipt["build_1"]["attempt_id"] == ATTEMPT_1
    national = receipt["registers"]["uk_target_fit"]
    assert national["add"] == [
        "synthetic.national_a@2025",
        "synthetic.national_b@2025",
    ]
    assert national["remove"] == ["synthetic.national_e@2025"]
    assert national["stale_risk_count"] == 1
    assert national["near_bound_count"] == 1
    local = receipt["registers"]["uk_local_target_fit"]
    assert local["add"] == ["census_households/E14999998/households"]
    area = receipt["registers"]["uk_local_area_support"]
    assert area["add"] == ["local_authority/E06999998"]
    assert area["remove"] == ["local_authority/E06999999"]
    assert receipt["predicted_build_2"] == {
        gate: {"passes": True, "failure_count": 0} for gate in SPECS
    }
    assert receipt["gate_clock_range"] == [APPROVED_ON, "2026-11-19"]
    # The drafted fit entries expire first; build 1's measure exclusion and
    # the degenerate entry outlast them.
    assert receipt["assembly_deadline"] == {
        "date": "2026-11-19",
        "set_by": [
            "uk_local_target_fit:census_households/E14999998/households",
            "uk_target_fit:synthetic.national_a@2025",
            "uk_target_fit:synthetic.national_b@2025",
        ],
    }
    assert receipt["build_1"]["measure_exclusions_from"] == (
        TOOL.TARGET_SURFACE_FILENAME
    )
    assert receipt["pins"] == {
        "files": [
            TOOL.TERMINAL_GATES_TEST.as_posix(),
            TOOL.WEIGHTED_INTEGRITY_TEST.as_posix(),
        ],
        "manual": [],
    }
    assert (
        "**Assembly deadline: 2026-11-19**, when "
        "`uk_local_target_fit:census_households/E14999998/households`, "
        "`uk_target_fit:synthetic.national_a@2025` and "
        "`uk_target_fit:synthetic.national_b@2025` expire: "
    ) in (out / "PR_BODY.md").read_text()
    # A sign-off with many cells names three and counts the rest.
    assert TOOL._named(["a", "b", "c", "d", "e"]) == "`a`, `b`, `c` and 2 more"

    apply_patch(repo, out / "signoff.patch")
    after = read_registers(repo)
    assert set(after["uk_target_fit"]["exclusions"]) == {
        "synthetic.national_a@2025",
        "synthetic.national_b@2025",
    }
    for gate_id in SPECS:
        for entry in after[gate_id]["exclusions"].values():
            assert entry["release_lines"] == ["local"]
            assert entry["approved_on"] == APPROVED_ON
            assert ATTEMPT_1 in entry["reason"]
            assert "%" not in entry["reason"]
    assert (
        after["uk_target_fit"]["exclusions"]["synthetic.national_a@2025"]["expires_on"]
        == "2026-11-19"
    )
    assert (
        after["uk_local_area_support"]["exclusions"]["local_authority/E06999998"][
            "expires_on"
        ]
        == "2027-01-22"
    )
    # The block the generator's bound families track is never touched.
    assert (
        after["uk_local_area_support"]["bound_despite_support_floor"]
        == build_1_registers()["uk_local_area_support"]["bound_despite_support_floor"]
    )
    pins = (repo / TOOL.TERMINAL_GATES_TEST).read_text()
    ast.parse(pins)
    assert '"synthetic.national_a@2025",' in pins
    assert '"census_households/E14999998/households",' in pins
    assert pins.count("record.release_lines") == 2
    area_pins = (repo / TOOL.WEIGHTED_INTEGRITY_TEST).read_text()
    ast.parse(area_pins)
    assert (
        '    assert set(loaded["exclusions"]) == {\n'
        '        "local_authority/E06999998",\n    }\n'
        '    assert {record.release_lines for record in loaded["exclusions"].values()}'
    ) in area_pins
    assert "local_authority/E06999999" not in area_pins
    assert (
        '    assert set(loaded["bound_despite_support_floor"]) == {"census_households"}'
        in area_pins
    )
    assert list((repo / "changelog.d").glob("1301-uk-dense-signoff-*.changed.md"))


def test_public_outputs_carry_no_figures(build_1, repo, tmp_path, capsys):
    out = tmp_path / "draft"
    assert draft(build_1, repo, out) == 0
    public = [
        (out / "signoff.patch").read_text(),
        (out / "signoff_draft.json").read_text(),
        (out / "PR_BODY.md").read_text(),
        capsys.readouterr().out,
    ]
    for text in public:
        for figure in PLANTED:
            assert figure not in text
    private = out / "PRIVATE_review.csv"
    assert private.stat().st_mode & 0o777 == 0o600
    content = private.read_text()
    assert content.startswith("# PRIVATE")
    assert "3137731" in content and "37.4321" in content
    # The single-record maximum weight is never read at all.
    assert "98765" not in content


def test_local_fit_bands_reach_only_the_private_file_and_the_terminal(tmp_path, capsys):
    # A local cell signed just beyond the bound (stale risk) and one just
    # inside it (near bound): with its key in the patch, a public count of
    # the first would bracket a named local estimate.
    local = (
        ("census_households", "E14999998", "households", -0.2521),
        ("census_households", "E14999999", "households", 0.2451),
        ("synthetic_income", "E06999998", "count", 0.1233),
    )
    registers = build_1_registers()
    repo = make_repo(tmp_path / "repo", registers)
    build = write_run(tmp_path / "build-1", registers=registers, local=local)
    out = tmp_path / "draft"
    assert draft(build, repo, out) == 0

    receipt = json.loads((out / "signoff_draft.json").read_text())
    for gate_id in ("uk_local_target_fit", "uk_local_area_support"):
        summary = receipt["registers"][gate_id]
        assert "stale_risk_count" not in summary, gate_id
        assert "near_bound_count" not in summary, gate_id
    national = receipt["registers"]["uk_target_fit"]
    assert (national["stale_risk_count"], national["near_bound_count"]) == (1, 1)

    body = (out / "PR_BODY.md").read_text()
    assert (
        "| Register | Added | Renewed | Kept | Retired | Rescoped | For review |"
        in body
    )
    assert "Stale-risk |" not in body and "Near-bound |" not in body
    assert "National target-fit rows: 1 stale-risk, 1 near-bound." in body
    assert "Local target-fit cells" in body
    disclosure = body.strip().splitlines()[-1]
    assert disclosure.startswith("Disclosure: this body carries ")
    assert "the national register's stale-risk and near-bound counts" in disclosure
    assert "the local registers' stale-risk and near-bound counts" in disclosure

    printed = json.loads(capsys.readouterr().out)
    assert printed["fit_band_counts"] == {
        "uk_target_fit": {"stale_risk": 1, "near_bound": 1},
        "uk_local_target_fit": {"stale_risk": 1, "near_bound": 1},
    }
    private = pd.read_csv(out / "PRIVATE_review.csv", skiprows=1, dtype=str)
    cells = private[private["gate"] == "uk_local_target_fit"].set_index("key")
    assert cells.loc["census_households/E14999998/households", "stale_risk"] == "True"
    assert cells.loc["census_households/E14999999/households", "near_bound"] == "True"


def test_a_non_signable_failure_writes_no_patch(repo, tmp_path):
    build = write_run(
        tmp_path / "build-1",
        registers=build_1_registers(),
        failed={"uk_local_per_family_fit": {}},
    )
    out = tmp_path / "draft"
    assert draft(build, repo, out) == 1
    receipt = json.loads((out / "signoff_draft.json").read_text())
    assert receipt["signable"] is False
    assert "uk_local_per_family_fit" in receipt["not_signable"]
    assert not (out / "signoff.patch").exists()
    assert "third build" in (out / "PR_BODY.md").read_text()


def test_input_mass_failures_are_left_for_hand_signing(repo, tmp_path):
    build = write_run(
        tmp_path / "build-1",
        registers=build_1_registers(),
        failed={"uk_input_mass_parity": {}},
    )
    out = tmp_path / "draft"
    assert draft(build, repo, out) == 0
    receipt = json.loads((out / "signoff_draft.json").read_text())
    assert receipt["hand_sign"] == ["uk_input_mass_parity"]
    assert "uk_input_mass_parity" in (out / "PR_BODY.md").read_text()


def test_a_local_register_the_checkout_does_not_wire_is_unsignable(
    build_1, repo, tmp_path, monkeypatch
):
    gates = load_country_spec("uk").gates
    unwired = replace(
        gates,
        gates=tuple(
            replace(
                entry,
                parameters={
                    k: v
                    for k, v in entry.parameters.items()
                    if k != "reviewed_exclusions_resource"
                },
            )
            if entry.id == "uk_local_target_fit"
            else entry
            for entry in gates.gates
        ),
    )
    wiring = REAL_REGISTER_WIRING(unwired)
    assert wiring["uk_target_fit"] is None
    assert wiring["uk_local_area_support"] is None
    assert (
        "wires no local_target_fit_reviewed_exclusions.json"
        in (wiring["uk_local_target_fit"])
    )
    monkeypatch.setattr(TOOL, "register_wiring", lambda gates=None: wiring)
    out = tmp_path / "draft"
    assert draft(build_1, repo, out) == 1
    receipt = json.loads((out / "signoff_draft.json").read_text())
    assert receipt["not_signable"] == {
        "uk_local_target_fit": wiring["uk_local_target_fit"]
    }
    assert "uk_local_target_fit" not in receipt["registers"]
    assert not (out / "signoff.patch").exists()


def test_a_stale_entry_the_generator_defers_is_held_not_retired(build_1, tmp_path):
    # Its cells are deferred on the generator's own constants, lockstep-tested
    # against the register: retiring the entry moves the solve.
    repo = make_repo(tmp_path / "repo", build_1_registers(), deferred=("E06999999",))
    out = tmp_path / "draft"
    assert draft(build_1, repo, out) == 1
    receipt = json.loads((out / "signoff_draft.json").read_text())
    area = receipt["registers"]["uk_local_area_support"]
    assert area["held"] == ["local_authority/E06999999"]
    assert area["remove"] == []
    reason = receipt["not_signable"]["uk_local_area_support"]
    assert "local_authority/E06999999 is held" in reason
    assert "SUPPORT_FLOOR_DEFERRED_AREA_IDS" in reason
    assert not (out / "signoff.patch").exists()


def test_an_expiring_entry_the_generator_defers_is_left_to_hand_renewal(tmp_path):
    registers = build_1_registers()
    # In force for build 1, expired at the gate clock.
    registers["uk_local_area_support"]["exclusions"]["local_authority/E06999998"] = (
        _entry("2026-09-01", "2026-10-21", lines=None)
    )
    repo = make_repo(tmp_path / "repo", registers, deferred=("E06999998",))
    build = write_run(tmp_path / "build-1", registers=registers)
    out = tmp_path / "draft"
    assert draft(build, repo, out) == 1
    receipt = json.loads((out / "signoff_draft.json").read_text())
    area = receipt["registers"]["uk_local_area_support"]
    assert area["held"] == ["local_authority/E06999998"]
    assert area["renew"] == []
    assert "renew the entry by hand" in receipt["not_signable"]["uk_local_area_support"]


def test_a_generator_without_its_own_deferrals_makes_areas_unsignable(
    build_1, repo, tmp_path
):
    (repo / TOOL.LOCAL_REFERENCE_GENERATOR).write_text('"""No constants."""\n')
    assert TOOL.generator_deferred_areas(repo) is None
    out = tmp_path / "draft"
    assert draft(build_1, repo, out) == 1
    receipt = json.loads((out / "signoff_draft.json").read_text())
    assert (
        "declares no SUPPORT_FLOOR_DEFERRED_AREA_IDS"
        in (receipt["not_signable"]["uk_local_area_support"])
    )
    assert "uk_local_area_support" not in receipt["registers"]


#: Reviewer text that states a figure (synthetic): each must be refused.
FIGURE_PROBES = (
    "7 rows against the 50-row floor",
    "ESS=3.2",
    "sources 4 < 50",
    "12 percent above its target",
    "household 4521 weight 312.5",
    "a share of twelve %",
    "ESS-3.2#1301",
    "reviewed 2026-02-30",
    # A bare #N is no issue reference: it can number anything.
    "household #4521 carries weight #312",
    "record #1234 decides the cell",
    "#7 rows against the #50-row floor",
    # Nor is a word#N of a repository the allowlist does not name, nor one
    # that runs into another word.
    "rows#7 against floor#50",
    "household-microcosm#4521",
    "microcosm#1301#7",
    # A figure shaped like a date, outside 2000-2099.
    "household 1234-05-06",
    # Percent signs in their Unicode forms.
    "#25\uff05",
    "a share of twelve \ufe6a",
    "a share of twelve \u066a",
    "a share of twelve \u2030",
    # Digits in a URL that is not a PolicyEngine issue or pull request.
    "https://example.org/households/4521",
    "https://github.com/PolicyEngine/microcosm/pull/1307/commits/4521abc",
)


@pytest.mark.parametrize("text", FIGURE_PROBES)
def test_free_text_stating_a_figure_is_refused(text):
    with pytest.raises(TOOL.InputError, match="--adjudication"):
        TOOL.figure_free(text, "--adjudication")


@pytest.mark.parametrize(
    "text",
    [
        "microcosm#1301 ruling of 2026-10-16",
        "PolicyEngine/policyengine-uk#2459 and microcosm#1301, reviewed 2026-10-16.",
        "PolicyEngine/policyengine.py#553 (see microcosm#1301)",
        "Synthetic reviewer note without figures.",
        "chronicle#359 and uk-data#521, after microcosm#1300",
        "microcosm#1301#issuecomment-6232359992",
        "PolicyEngine/microcosm#1301#issuecomment-6232359992 (the ruling)",
        "https://github.com/PolicyEngine/microcosm/issues/1301#issuecomment-6232359992",
        "the plan (https://github.com/PolicyEngine/microcosm/pull/1307), 2026-10-17",
    ],
)
def test_free_text_with_only_issue_references_and_dates_is_accepted(text):
    assert TOOL.figure_free(text, "--adjudication") == text


@pytest.mark.parametrize(
    "text",
    [
        # Cite the comment by its anchor instead (microcosm#1301#issuecomment-N).
        "microcosm#1301 (issue comment 6232359992)",
        "K=15",
        "150,000 households",
        "plan arm c9",
        "rehearsal R5",
        "the 2025 uprating",
    ],
)
def test_a_number_outside_a_reference_or_date_is_refused(text):
    """Legitimate text whose number stands outside an issue reference, a
    PolicyEngine URL or a date is refused too: the rule cannot tell a plan
    code or a count from a figure. The manual scan stays the backstop."""
    with pytest.raises(TOOL.InputError, match="a bare #N is refused"):
        TOOL.figure_free(text, "--adjudication")


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--class-note", "local_authority/*=Synthetic: 7 rows against the floor."),
        ("--area-name", "E06999998=Synthetic household 4521 weight 312.5"),
        ("--adjudication", "microcosm#1301: ESS=3.2 on build 1"),
        ("--adjudication", "Synthetic: 12% above its target"),
    ],
)
def test_each_free_text_argument_is_checked_before_the_build_is_read(
    repo, tmp_path, capsys, flag, value
):
    # No run directory exists: the refusal names the flag, not the build.
    out = tmp_path / "draft"
    assert draft(tmp_path / "no-build", repo, out, flag, value) == 2
    assert flag in capsys.readouterr().err
    assert not out.exists()


def test_an_adjudication_citing_issues_and_dates_is_drafted(build_1, repo, tmp_path):
    adjudication = "microcosm#1301 ruling of 2026-10-16"
    note = "census_households/*=Reviewed under PolicyEngine/microcosm#1301."
    out = tmp_path / "draft"
    assert (
        draft(build_1, repo, out, "--adjudication", adjudication, "--class-note", note)
        == 0
    )
    apply_patch(repo, out / "signoff.patch")
    after = read_registers(repo)
    for gate_id in SPECS:
        for entry in after[gate_id]["exclusions"].values():
            assert entry["adjudication"] == adjudication
    cell = after["uk_local_target_fit"]["exclusions"][
        "census_households/E14999998/households"
    ]
    assert cell["reason"].endswith(" Reviewed under PolicyEngine/microcosm#1301.")


def test_the_drafted_entries_still_get_the_release_disclosure_walk():
    # The signed local report build 2 ships carries every reason its gates
    # stand on, so the drafted entries get the release's disclosure walk too.
    spec = SPECS["uk_local_area_support"]
    drafted = TOOL.RegisterDraft(spec=spec, before={})
    drafted.entries["local_authority/E06999998"] = _entry(
        APPROVED_ON, "2026-11-19", reason="Synthetic reason stating rows 7 < 50."
    )
    (failure,) = TOOL.drafted_disclosure_failures({spec.gate_id: drafted})
    assert "measured support" in failure


@pytest.mark.parametrize(
    "login",
    [
        "juaristi22 (rows 7 of 50)",
        "Synthetic Reviewer",
        "reviewer@example.org",
        "-reviewer",
        " juaristi22",
        "r" * 40,
        # Login-shaped figures: a leading digit, or more than one digit run.
        "4521",
        "7-rows",
        "rows-7-of-50",
        "household4521-weight312",
    ],
)
def test_approved_by_must_be_a_github_login(repo, tmp_path, capsys, login):
    # No run directory exists: the refusal names the flag, not the build.
    out = tmp_path / "draft"
    assert draft(tmp_path / "no-build", repo, out, f"--approved-by={login}") == 2
    assert "--approved-by must be a GitHub login" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize(
    "login",
    ["juaristi22", "MaxGhenis", "vahid-ahmadi", "synthetic-reviewer", "r", "r" * 39],
)
def test_a_github_login_is_accepted(login):
    assert TOOL.github_login(login, "--approved-by") == login


def test_a_signable_gate_that_failed_closed_is_refused(repo, tmp_path, capsys):
    build = write_run(
        tmp_path / "build-1",
        registers=build_1_registers(),
        failed={
            "uk_local_target_fit": {
                "evaluation_error": {"type": "ValueError", "message": "synthetic"}
            }
        },
    )
    assert draft(build, repo, tmp_path / "draft") == 2
    assert "failed closed" in capsys.readouterr().err


def _tamper_targets(build: Path) -> None:
    path = build / f"{STEM}.targets.csv"
    frame = pd.read_csv(path, keep_default_na=False)
    frame.loc[0, "relative_error"] = 0.0
    frame.to_csv(path, index=False)


def _tamper_logbook(build: Path) -> None:
    (build / "logbook-spool" / "row.json").write_text(
        json.dumps({"build_id": ATTEMPT_2})
    )


def _move_manifest(build: Path) -> None:
    path = build / TOOL.FULL_GATE_REPORT_FILENAME
    document = json.loads(path.read_bytes())
    document["report"]["gates_manifest_sha256"] = "0" * 64
    path.write_bytes(encode_stage_evidence(document))


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (_tamper_targets, "targets.csv"),
        (_tamper_logbook, "Logbook attempt id"),
        (_move_manifest, "does not decode"),
    ],
)
def test_inconsistent_inputs_are_refused(
    build_1, repo, tmp_path, capsys, tamper, message
):
    tamper(build_1)
    assert draft(build_1, repo, tmp_path / "draft") == 2
    assert message in capsys.readouterr().err


def test_an_unbound_report_is_refused(repo, tmp_path, capsys):
    build = write_run(
        tmp_path / "build-1",
        registers=build_1_registers(),
        attempt_id=TOOL.UK_DENSE_UNBOUND_RELEASE_ID,
    )
    assert draft(build, repo, tmp_path / "draft") == 2
    assert "no Logbook attempt" in capsys.readouterr().err


def test_a_development_build_is_refused_or_dry_run_only(repo, tmp_path, capsys):
    build = write_run(
        tmp_path / "build-1",
        registers=build_1_registers(),
        release_candidate=False,
    )
    assert draft(build, repo, tmp_path / "draft") == 2
    assert "--allow-dev-build" in capsys.readouterr().err
    out = tmp_path / "dry-run"
    assert draft(build, repo, out, "--allow-dev-build") == 1
    receipt = json.loads((out / "signoff_draft.json").read_text())
    assert receipt["signable"] is False and receipt["dev_build"] is True
    assert receipt["registers"]["uk_target_fit"]["add"]
    assert not (out / "signoff.patch").exists()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--gate-clock", "2026-10-21"), "precedes --approved-on"),
        (("--expires-on", "2027-02-01"), "longer than"),
        (("--expires-on", "2026-10-22"), "after the gate clock"),
        (("--approved-on", "2026-10-18", "--gate-clock", "2026-10-19"), "solve clock"),
    ],
)
def test_window_refusals(build_1, repo, tmp_path, capsys, extra, message):
    assert draft(build_1, repo, tmp_path / "draft", *extra) == 2
    assert message in capsys.readouterr().err


def test_the_national_graph_review_invariant_is_kept(
    build_1, repo, tmp_path, capsys, monkeypatch
):
    monkeypatch.setattr(
        TOOL,
        "_review_registers",
        lambda: [[{"approved_on": "2026-10-01", "expires_on": "2026-10-22"}]],
    )
    assert draft(build_1, repo, tmp_path / "draft") == 2
    assert "review invariant" in capsys.readouterr().err


def test_the_review_registers_load_from_the_package():
    registers = REAL_REVIEW_REGISTERS()
    assert len(registers) == 4
    assert all(registers)


def test_out_inside_the_repository_is_refused(build_1, repo, capsys):
    assert draft(build_1, repo, repo / "signoff") == 2
    assert "inside the repository" in capsys.readouterr().err


def _work_tree(path: Path) -> Path:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    path.mkdir(parents=True)
    git(path, "init", "--quiet")
    return path


@pytest.mark.parametrize("relative", ["signoff", "not/made/yet"])
def test_out_inside_another_git_work_tree_is_refused(
    build_1, repo, tmp_path, capsys, relative
):
    # Another checkout or a sign-off worktree: the private review file would
    # be one `git add` from a commit there.
    out = _work_tree(tmp_path / "other-checkout") / relative
    assert draft(build_1, repo, out) == 2
    assert "inside the git work tree" in capsys.readouterr().err
    assert not out.exists()
    status = TOOL.main(
        ["--verify", str(build_1), "--repo-root", str(repo), "--out", str(out)]
    )
    assert status == 2
    assert "inside the git work tree" in capsys.readouterr().err


def test_the_gitignore_covers_the_private_review_file():
    ignored = (REPOSITORY / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert {"PRIVATE_review.csv", "signoff-draft/"} <= set(ignored)


def test_a_build_that_passed_needs_nothing(tmp_path, capsys):
    registers = build_1_registers()
    registers["uk_target_fit"]["exclusions"] = {}
    registers["uk_local_area_support"]["exclusions"] = {
        "local_authority/E06999998": _entry("2026-10-01", "2026-12-01")
    }
    repo = make_repo(tmp_path / "repo", registers)
    build = write_run(
        tmp_path / "build-1",
        registers=registers,
        national={"synthetic.national_c@2025": 0.0213},
        local=LOCAL[1:],
    )
    out = tmp_path / "draft"
    assert draft(build, repo, out) == 0
    assert not (out / "signoff.patch").exists()
    body = (out / "PR_BODY.md").read_text()
    assert "Nothing to sign" in body
    # Build 1 itself is assembled, by the area entry's expiry.
    assert "**Assembly deadline: 2026-12-01**" in body


# ---------------------------------------------------------------------------
# The national line's approvals and the assembly deadline
# ---------------------------------------------------------------------------

UNSCOPED_KEY = "synthetic.national_c@2025"
BREACHING_KEY = "synthetic.national_a@2025"


@pytest.mark.parametrize(
    ("entry", "approved_on", "why", "review"),
    [
        # In force, and build 1 fits the row inside the bound: stale on the
        # local line.
        (
            _entry("2026-10-01", "2026-10-29", lines=None),
            APPROVED_ON,
            "fits the row inside the bound",
            False,
        ),
        # Approved for every line after the gate clock: premature there.
        (
            _entry("2026-10-23", "2026-11-20", lines=None),
            APPROVED_ON,
            "takes force 2026-10-23, after the gate clock",
            False,
        ),
        # In force for build 1, expired at the gate clock: kept all the same,
        # for the national line to renew or retire. The sign-off is approved
        # before the expiry, so the national graph test's review invariant
        # still holds.
        (
            _entry("2026-09-01", "2026-10-21", lines=None),
            "2026-10-20",
            "expired 2026-10-21; the national line must renew or retire it",
            True,
        ),
    ],
)
def test_an_unscoped_national_entry_build_2_would_fail_is_rescoped(
    tmp_path, entry, approved_on, why, review
):
    # The register is shared: the national line may still stand on it.
    registers = build_1_registers()
    registers["uk_target_fit"]["exclusions"][UNSCOPED_KEY] = entry
    repo = make_repo(tmp_path / "repo", registers)
    build = write_run(tmp_path / "build-1", registers=registers)
    out = tmp_path / "draft"
    clock = ("--approved-on", approved_on, "--gate-clock", GATE_CLOCK.isoformat())
    assert draft(build, repo, out, *clock) == 0

    receipt = json.loads((out / "signoff_draft.json").read_text())
    national = receipt["registers"]["uk_target_fit"]
    assert national["rescope"] == [UNSCOPED_KEY]
    assert national["remove"] == ["synthetic.national_e@2025"]
    assert national["review"] == ([UNSCOPED_KEY] if review else [])
    assert national["conflicts"] == []
    assert why in national["notes"][UNSCOPED_KEY]
    assert receipt["predicted_build_2"]["uk_target_fit"]["passes"] is True
    # Set aside on the local line, it no longer holds the gate clock back.
    assert receipt["gate_clock_range"][0] == approved_on
    body = (out / "PR_BODY.md").read_text()
    assert (
        "| National target-fit rows (`uk_target_fit`) | 2 | 0 | 0 | 1 | 1 | "
        f"{int(review)} |"
    ) in body
    assert 'now carry `"release_lines": ["national"]`' in body
    assert (
        f"National target-fit entries for review: `{UNSCOPED_KEY}`." in body
    ) is review

    apply_patch(repo, out / "signoff.patch")
    after = read_registers(repo)
    committed = after["uk_target_fit"]["exclusions"][UNSCOPED_KEY]
    assert committed == {**entry, "release_lines": ["national"]}
    record = coerce_reviewed_exclusions({UNSCOPED_KEY: committed}, label="after")
    assert record[UNSCOPED_KEY].applies_on("national")
    assert not record[UNSCOPED_KEY].applies_on("local")
    assert f'"{UNSCOPED_KEY}",' in (repo / TOOL.TERMINAL_GATES_TEST).read_text()
    (fragment,) = (repo / "changelog.d").glob("1301-uk-dense-signoff-*.changed.md")
    assert (
        "1 national target-fit entry that applied on every line is scoped to the "
        "national line, which still stands on it." in fragment.read_text()
    )

    build_2 = write_run(
        tmp_path / "build-2", registers=after, attempt_id=ATTEMPT_2, clock=GATE_CLOCK
    )
    status, result = verify(
        build_2, repo, tmp_path, "--draft-receipt", str(out / "signoff_draft.json")
    )
    assert status == 0, result["failures"]


def test_an_expired_kept_entry_that_breaks_the_review_invariant_is_unsignable(
    tmp_path,
):
    # Kept beside approvals made after it expired, the entry would fail the
    # national graph test's review invariant, so the national line must renew
    # or retire it first.
    registers = build_1_registers()
    registers["uk_target_fit"]["exclusions"][UNSCOPED_KEY] = _entry(
        "2026-09-01", "2026-10-21", lines=None
    )
    repo = make_repo(tmp_path / "repo", registers)
    build = write_run(tmp_path / "build-1", registers=registers)
    out = tmp_path / "draft"
    assert draft(build, repo, out) == 1
    receipt = json.loads((out / "signoff_draft.json").read_text())
    national = receipt["registers"]["uk_target_fit"]
    assert national["rescope"] == national["review"] == [UNSCOPED_KEY]
    reason = receipt["not_signable"]["uk_target_fit"]
    assert "review invariant" in reason
    assert (
        f"because {UNSCOPED_KEY} expired and is kept for the national line: "
        "renew or retire it there by hand, then draft again"
    ) in reason
    assert not (out / "signoff.patch").exists()
    assert "a hand fix and a new draft" in (out / "PR_BODY.md").read_text()


@pytest.mark.parametrize(
    ("entry", "review", "advice"),
    [
        (_entry("2026-09-01", "2026-10-21", lines=None), True, "sign the row by hand"),
        (
            _entry("2026-10-23", "2026-11-20", lines=None),
            False,
            "or choose a gate clock on or after 2026-10-23",
        ),
    ],
)
def test_a_breaching_row_never_replaces_its_entry_for_every_line(
    tmp_path, entry, review, advice
):
    # Build 1 breaches the bound on the row and its all-lines entry is out of
    # its window at the gate clock. The entry is kept for the national line,
    # and one key holds one entry, so the local line needs a hand sign-off.
    registers = build_1_registers()
    registers["uk_target_fit"]["exclusions"][BREACHING_KEY] = entry
    repo = make_repo(tmp_path / "repo", registers)
    build = write_run(tmp_path / "build-1", registers=registers)
    out = tmp_path / "draft"
    assert draft(build, repo, out) == 1
    receipt = json.loads((out / "signoff_draft.json").read_text())
    national = receipt["registers"]["uk_target_fit"]
    assert national["rescope"] == national["conflicts"] == [BREACHING_KEY]
    assert national["renew"] == []
    assert national["review"] == ([BREACHING_KEY] if review else [])
    note = national["notes"][BREACHING_KEY]
    assert "one key holds one entry, so no local-only entry can be drafted" in note
    assert advice in note
    assert "sign them by hand" in receipt["not_signable"]["uk_target_fit"]
    assert not (out / "signoff.patch").exists()


def test_a_gate_clock_on_its_approval_keeps_a_premature_all_lines_entry(tmp_path):
    registers = build_1_registers()
    registers["uk_target_fit"]["exclusions"][BREACHING_KEY] = _entry(
        "2026-10-23", "2026-11-20", lines=None
    )
    repo = make_repo(tmp_path / "repo", registers)
    build = write_run(tmp_path / "build-1", registers=registers)
    out = tmp_path / "draft"
    assert draft(build, repo, out, "--gate-clock", "2026-10-23") == 0
    national = json.loads((out / "signoff_draft.json").read_text())["registers"][
        "uk_target_fit"
    ]
    assert national["keep"] == [BREACHING_KEY]
    assert national["rescope"] == national["conflicts"] == []


def test_an_unscoped_local_cell_entry_is_still_removed(tmp_path):
    # The national line reads no local cell, so nothing is rescoped there.
    registers = build_1_registers()
    key = "census_households/E14999999/households"
    registers["uk_local_target_fit"]["exclusions"][key] = _entry(
        "2026-10-01", "2026-10-29", lines=None
    )
    repo = make_repo(tmp_path / "repo", registers)
    build = write_run(tmp_path / "build-1", registers=registers)
    out = tmp_path / "draft"
    assert draft(build, repo, out) == 0
    local = json.loads((out / "signoff_draft.json").read_text())["registers"][
        "uk_local_target_fit"
    ]
    assert local["remove"] == [key]
    assert local["rescope"] == []


@pytest.mark.parametrize(
    ("measures", "degenerate", "deadline", "set_by"),
    [
        (
            {"synthetic.measure_a@2025": _measure_exclusion("2026-11-05")},
            None,
            "2026-11-05",
            "measure_exclusion:synthetic.measure_a@2025",
        ),
        (
            MEASURE_EXCLUSIONS,
            {
                "household.synthetic_constant": _entry("2026-09-01", "2026-11-01"),
                # Not the local line's: it never caps the local release.
                "household.synthetic_national": _entry(
                    "2026-09-01", "2026-10-25", lines=("national",)
                ),
            },
            "2026-11-01",
            "degenerate:household.synthetic_constant",
        ),
    ],
)
def test_solve_scoped_expiries_cap_the_gate_clock_and_set_the_deadline(
    tmp_path, measures, degenerate, deadline, set_by
):
    registers = build_1_registers()
    repo = make_repo(
        tmp_path / "repo",
        registers,
        degenerate=None if degenerate is None else degenerate_register(degenerate),
    )
    build = write_run(
        tmp_path / "build-1", registers=registers, measure_exclusions=measures
    )
    out = tmp_path / "draft"
    assert draft(build, repo, out) == 0
    receipt = json.loads((out / "signoff_draft.json").read_text())
    assert receipt["gate_clock_range"] == [APPROVED_ON, deadline]
    assert receipt["assembly_deadline"] == {"date": deadline, "set_by": [set_by]}
    body = (out / "PR_BODY.md").read_text()
    assert f"**Assembly deadline: {deadline}**, when `{set_by}` expires" in body
    assert f"on or before {deadline}." in body


def test_a_gate_clock_past_a_solve_scoped_expiry_is_refused(repo, tmp_path, capsys):
    # In force for build 1's solve clock, expired at the planned gate clock:
    # renewing it would change the solve.
    measures = {"synthetic.measure_a@2025": _measure_exclusion("2026-10-21")}
    build = write_run(
        tmp_path / "build-1", registers=build_1_registers(), measure_exclusions=measures
    )
    out = tmp_path / "draft"
    assert draft(build, repo, out) == 2
    assert (
        "--gate-clock 2026-10-22 is after 2026-10-21, when "
        "measure_exclusion:synthetic.measure_a@2025 expires"
    ) in capsys.readouterr().err
    assert not out.exists()
    # A gate clock on the expiry itself is in time.
    assert draft(build, repo, out, "--approved-on", "2026-10-21") == 0
    receipt = json.loads((out / "signoff_draft.json").read_text())
    assert receipt["gate_clock_range"] == ["2026-10-21", "2026-10-21"]


def test_the_candidate_manifest_receipt_is_read_before_the_surface(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    name = "synthetic.measure_a@2025"
    for filename, expiry in (
        (TOOL.TARGET_SURFACE_FILENAME, "2027-01-31"),
        (TOOL.CANDIDATE_MANIFEST_FILENAME, "2026-12-31"),
    ):
        (run / filename).write_text(
            json.dumps({"measure_exclusions": {name: _measure_exclusion(expiry)}})
        )
        assert TOOL.receipted_measure_exclusions(run) == (
            {name: date.fromisoformat(expiry)},
            filename,
        )


def test_a_run_without_receipted_measure_exclusions_is_refused(
    build_1, repo, tmp_path, capsys
):
    (build_1 / TOOL.TARGET_SURFACE_FILENAME).unlink()
    assert draft(build_1, repo, tmp_path / "draft") == 2
    assert "the measure exclusions its solve ran under" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("indent", [1, 2])
def test_register_rendering_is_byte_stable(indent):
    payload = build_1_registers()["uk_local_area_support"]
    text = _dump(payload, indent)
    assert TOOL.render_register(text, "exclusions", payload["exclusions"]) == text
    edited = TOOL.render_register(text, "exclusions", {})
    assert (
        json.loads(edited)["bound_despite_support_floor"]
        == (payload["bound_despite_support_floor"])
    )
    assert edited.splitlines()[1].startswith(" " * indent + '"')


def test_a_non_canonical_register_is_refused():
    text = json.dumps({"schema_version": 2, "exclusions": {}}, indent=2)
    with pytest.raises(TOOL.InputError, match="canonical"):
        TOOL.render_register(text, "exclusions", {})


def test_pin_rewrite_touches_only_the_assertion_and_is_idempotent():
    loader = "uk_default_target_fit_reviewed_exclusions"
    keys = ["b@2025", "a@2025"]
    once, name = TOOL.rewrite_pin(PIN_SOURCE, loader, keys, local_only=True)
    assert name == "test_committed_target_fit_register_retains_only_live_deferrals"
    ast.parse(once)
    twice, _ = TOOL.rewrite_pin(once, loader, keys, local_only=True)
    assert twice == once
    before, after = PIN_SOURCE.splitlines(), once.splitlines()
    assert (
        after[: after.index("    assert set(register) == {")]
        == before[: before.index("    assert set(register) == {")]
    )
    assert '        "a@2025",\n        "b@2025",\n    }' in once
    emptied, _ = TOOL.rewrite_pin(once, loader, [], local_only=True)
    assert "    assert register == {}\n" in emptied
    assert "release_lines" not in emptied.split("def test_committed_local")[0]


def test_a_call_form_pin_gets_a_named_register():
    loader = "uk_default_local_target_fit_reviewed_exclusions"
    rewritten, _ = TOOL.rewrite_pin(PIN_SOURCE, loader, ["f/E1/m"], local_only=True)
    assert f"    register = {loader}()\n    assert set(register) == {{" in rewritten
    ast.parse(rewritten)


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


def _signed_off(build_1: Path, repo: Path, tmp_path: Path) -> dict:
    out = tmp_path / "draft"
    assert draft(build_1, repo, out) == 0
    apply_patch(repo, out / "signoff.patch")
    return read_registers(repo)


def verify(build_2: Path, repo: Path, tmp_path: Path, *extra: str) -> tuple[int, dict]:
    out = tmp_path / "verify"
    status = TOOL.main(
        [
            "--verify",
            str(build_2),
            "--repo-root",
            str(repo),
            "--out",
            str(out),
            *extra,
        ]
    )
    return status, json.loads((out / "signoff_verify.json").read_text())


def test_verify_passes_a_build_2_that_stood_on_every_entry(build_1, repo, tmp_path):
    registers = _signed_off(build_1, repo, tmp_path)
    build_2 = write_run(
        tmp_path / "build-2",
        registers=registers,
        attempt_id=ATTEMPT_2,
        clock=GATE_CLOCK,
    )
    status, result = verify(
        build_2,
        repo,
        tmp_path,
        "--build-1",
        str(build_1),
        "--draft-receipt",
        str(tmp_path / "draft" / "signoff_draft.json"),
    )
    assert status == 0, result["failures"]
    assert result["stood_on"] == {
        "uk_target_fit": 2,
        "uk_local_target_fit": 1,
        "uk_local_area_support": 1,
    }
    assert result["comparison"] == {
        "build_1_attempt_id": ATTEMPT_1,
        "solve_reused": True,
        "rows_differing": 0,
    }
    assert result["warnings"] == []


def _stale(registers):
    return {"national": {**NATIONAL, "synthetic.national_b@2025": 0.2}}


def _dormant(registers):
    registers["uk_local_target_fit"]["exclusions"]["census_households/E14999997/x"] = (
        _entry(APPROVED_ON, "2026-11-19")
    )
    return {}


def _expired(registers):
    return {"clock": date(2026, 11, 20)}


def _per_family(registers):
    return {"failed": {"uk_local_per_family_fit": {}}}


def _figure_in_reason(registers):
    entry = registers["uk_local_area_support"]["exclusions"][
        "local_authority/E06999998"
    ]
    entry["reason"] = "Synthetic reason stating rows 7 < 50."
    return {}


def _other_receipt(registers):
    entry = registers["uk_target_fit"]["exclusions"]["synthetic.national_a@2025"]
    entry["adjudication"] = "microcosm#1301 (another review)"
    return {}


@pytest.mark.parametrize(
    ("mutate", "commit_mutation", "message"),
    [
        (_stale, False, "stale_exclusions"),
        (_dormant, True, "did not stand on committed entries"),
        (_expired, False, "expired_exclusions"),
        (_per_family, False, "uk_local_per_family_fit"),
        (_other_receipt, False, "receipt for synthetic.national_a@2025 differs"),
    ],
)
def test_verify_fails_a_build_2_that_did_not_stand_on_the_sign_off(
    build_1, repo, tmp_path, mutate, commit_mutation, message
):
    registers = _signed_off(build_1, repo, tmp_path)
    build_registers = json.loads(json.dumps(registers))
    overrides = mutate(build_registers)
    if commit_mutation:
        make_repo(repo, build_registers)
    build_2 = write_run(
        tmp_path / "build-2",
        registers=build_registers,
        attempt_id=ATTEMPT_2,
        clock=overrides.get("clock", GATE_CLOCK),
        national=overrides.get("national"),
        failed=overrides.get("failed"),
    )
    status, result = verify(build_2, repo, tmp_path)
    assert status == 1
    assert any(message in line for line in result["failures"]), result["failures"]


def test_verify_warns_on_a_support_figure_in_the_signed_report(build_1, repo, tmp_path):
    """The signed local report gets the release's disclosure walk, which
    warns rather than fails, as the release contract does (microcosm#1301)."""
    registers = _signed_off(build_1, repo, tmp_path)
    build_registers = json.loads(json.dumps(registers))
    _figure_in_reason(build_registers)
    make_repo(repo, build_registers)
    build_2 = write_run(
        tmp_path / "build-2",
        registers=build_registers,
        attempt_id=ATTEMPT_2,
        clock=GATE_CLOCK,
    )
    status, result = verify(build_2, repo, tmp_path)
    assert status == 0, result["failures"]
    assert any(
        line.startswith("contract: ") and "measured support" in line
        for line in result["warnings"]
    ), result["warnings"]


def test_verify_surfaces_a_tampered_local_report(build_1, repo, tmp_path):
    registers = _signed_off(build_1, repo, tmp_path)
    build_2 = write_run(
        tmp_path / "build-2",
        registers=registers,
        attempt_id=ATTEMPT_2,
        clock=GATE_CLOCK,
    )
    path = build_2 / f"{STEM}.local_gates.json"
    report = json.loads(path.read_text())
    report["policy_sha256"] = "f" * 64
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    status, result = verify(build_2, repo, tmp_path)
    assert status == 1
    assert any(line.startswith("contract: ") for line in result["failures"])


def test_verify_reports_row_drift_and_no_solve_reuse(build_1, repo, tmp_path):
    registers = _signed_off(build_1, repo, tmp_path)
    build_2 = write_run(
        tmp_path / "build-2",
        registers=registers,
        attempt_id=ATTEMPT_2,
        clock=GATE_CLOCK,
        national={**NATIONAL, "synthetic.national_c@2025": 0.0214},
        artifacts={**SOLVE, "solution": "another-solution"},
    )
    status, result = verify(build_2, repo, tmp_path, "--build-1", str(build_1))
    assert status == 0, result["failures"]
    assert result["comparison"]["solve_reused"] is False
    assert result["comparison"]["rows_differing"] == 1
    assert "build 2 did not reuse build 1's solve" in result["warnings"]


@pytest.mark.parametrize(
    ("today", "outcome"),
    [
        (date(2026, 10, 27), None),
        (date(2026, 10, 28), "warning"),
        (date(2026, 10, 30), "warning"),
        (date(2026, 10, 31), "failure"),
    ],
)
def test_verify_warns_near_the_assembly_deadline_and_fails_past_it(
    repo, tmp_path, monkeypatch, today, outcome
):
    measures = {"synthetic.measure_a@2025": _measure_exclusion("2026-10-30")}
    build = write_run(
        tmp_path / "build-1", registers=build_1_registers(), measure_exclusions=measures
    )
    registers = _signed_off(build, repo, tmp_path)
    build_2 = write_run(
        tmp_path / "build-2",
        registers=registers,
        attempt_id=ATTEMPT_2,
        clock=GATE_CLOCK,
        measure_exclusions=measures,
    )
    monkeypatch.setattr(TOOL, "_today", lambda: today)
    status, result = verify(build_2, repo, tmp_path)
    assert result["assembly_deadline"] == {
        "date": "2026-10-30",
        "set_by": ["measure_exclusion:synthetic.measure_a@2025"],
    }
    warned = [line for line in result["warnings"] if "assembly deadline" in line]
    failed = [line for line in result["failures"] if "assembly deadline" in line]
    assert (status, bool(warned), bool(failed)) == {
        None: (0, False, False),
        "warning": (0, True, False),
        "failure": (1, False, True),
    }[outcome]


def test_verify_fails_a_rescope_committed_on_every_line(tmp_path):
    registers = build_1_registers()
    registers["uk_target_fit"]["exclusions"][UNSCOPED_KEY] = _entry(
        "2026-10-01", "2026-10-29", lines=None
    )
    repo = make_repo(tmp_path / "repo", registers)
    build = write_run(tmp_path / "build-1", registers=registers)
    signed = _signed_off(build, repo, tmp_path)
    # The sign-off committed with the entry left on every line.
    del signed["uk_target_fit"]["exclusions"][UNSCOPED_KEY]["release_lines"]
    make_repo(repo, signed)
    build_2 = write_run(
        tmp_path / "build-2", registers=signed, attempt_id=ATTEMPT_2, clock=GATE_CLOCK
    )
    status, result = verify(
        build_2,
        repo,
        tmp_path,
        "--draft-receipt",
        str(tmp_path / "draft" / "signoff_draft.json"),
    )
    assert status == 1
    assert (
        f"uk_target_fit: entries drafted for the national line only are not "
        f"committed so ['{UNSCOPED_KEY}']"
    ) in result["failures"]


# ---------------------------------------------------------------------------
# Lockstep with the repository
# ---------------------------------------------------------------------------


def test_registers_are_the_gate_parameters_this_checkout_declares():
    gates = {entry.id: entry for entry in load_country_spec("uk").gates.gates}
    wiring = REAL_REGISTER_WIRING()
    for spec in TOOL.REGISTERS:
        named = dict(gates[spec.gate_id].parameters).get(spec.parameter)
        assert named in (None, spec.resource), spec
        assert (wiring[spec.gate_id] is None) == (named == spec.resource)
        if named is not None:
            assert (REPOSITORY / spec.path).is_file(), spec
    assert wiring["uk_target_fit"] is None
    assert wiring["uk_local_area_support"] is None


def test_the_local_register_constant_matches_the_runtime_when_it_exists():
    from microcosm.build.uk_runtime import weighted_integrity

    runtime = getattr(
        weighted_integrity, "UK_LOCAL_TARGET_FIT_EXCLUSION_REGISTER_RESOURCE", None
    )
    assert runtime in (None, TOOL.LOCAL_TARGET_FIT_REGISTER.name)


def test_the_full_gate_report_filename_matches_the_driver():
    from microcosm.build.uk_runtime.full_build_cli import FULL_GATE_REPORT_FILENAME

    assert TOOL.FULL_GATE_REPORT_FILENAME == FULL_GATE_REPORT_FILENAME


def test_the_measure_exclusion_receipts_are_the_files_the_build_writes(tmp_path):
    from types import SimpleNamespace

    from microcosm.build.uk_runtime.full_build_cli import _materialize_evidence
    from microcosm.build.uk_runtime.rowwise_cli import MANIFEST_FILENAME

    assert TOOL.CANDIDATE_MANIFEST_FILENAME == MANIFEST_FILENAME
    # The build writes every node artifact as <node>.<artifact>.json.
    payload = b'{"measure_exclusions": {}}'
    manifest = SimpleNamespace(
        nodes={
            "uk.full.target_compilation": SimpleNamespace(
                opaque_artifacts={"surface": "surface-key"}
            )
        }
    )
    _materialize_evidence(
        manifest, SimpleNamespace(load_bytes=lambda key: payload), tmp_path
    )
    assert (tmp_path / TOOL.TARGET_SURFACE_FILENAME).read_bytes() == payload


def test_the_degenerate_register_is_the_runtime_register():
    from microcosm.build.uk_runtime.weighted_integrity import (
        UK_DEGENERATE_EXCLUSION_REGISTER_RESOURCE,
    )

    assert TOOL.DEGENERATE_REGISTER.name == UK_DEGENERATE_EXCLUSION_REGISTER_RESOURCE
    # The committed register loads through the tool, the local line's entries.
    assert TOOL.degenerate_expiries(REPOSITORY)


def test_the_committed_pins_are_found():
    for spec in TOOL.REGISTERS:
        if spec.loader is None or not (REPOSITORY / spec.path).is_file():
            continue
        # The pin is found by the committed loader's name.
        module = importlib.import_module(spec.loader_module)
        assert callable(getattr(module, spec.loader, None)), spec.loader
        sites = [
            site
            for relative in TOOL.PIN_SEARCH
            for site in TOOL.find_pins(
                (REPOSITORY / relative).read_text(encoding="utf-8"),
                spec.loader,
                spec.pin_block,
            )
        ]
        assert len(sites) == 1, (spec.gate_id, sites)


def test_the_committed_pins_rewrite_and_reparse():
    # Each committed pin takes a drafted key set in place, with or without
    # the local-line scope pin, and gives the committed one back.
    for spec in TOOL.REGISTERS:
        if spec.loader is None:
            continue
        payload = json.loads((REPOSITORY / spec.path).read_text(encoding="utf-8"))
        committed = sorted(payload[spec.block])
        for relative in TOOL.PIN_SEARCH:
            source = (REPOSITORY / relative).read_text(encoding="utf-8")
            if not TOOL.find_pins(source, spec.loader, spec.pin_block):
                continue
            for local_only in (False, True):
                drafted, _ = TOOL.rewrite_pin(
                    source,
                    spec.loader,
                    [*committed, "synthetic/E06999998/key"],
                    local_only=local_only,
                    block=spec.pin_block,
                )
                ast.parse(drafted)
                assert '"synthetic/E06999998/key",' in drafted
                added = set(drafted.splitlines()) - set(source.splitlines())
                assert all(len(line) <= TOOL.LINE_LENGTH for line in added)
                back, _ = TOOL.rewrite_pin(
                    drafted,
                    spec.loader,
                    committed,
                    local_only=False,
                    block=spec.pin_block,
                )
                assert back == source, (spec.gate_id, relative)


def test_the_generator_deferred_areas_are_read_from_its_source():
    from tools.generate_uk_local_target_references import (
        SUPPORT_FLOOR_DEFERRED_AREA_IDS,
    )

    assert TOOL.generator_deferred_areas(REPOSITORY) == {
        f"local_authority/{code}" for code in SUPPORT_FLOOR_DEFERRED_AREA_IDS
    }


def test_the_local_levels_are_the_full_gates_local_levels():
    from microcosm.build.uk_runtime.full_gates import _LOCAL_LEVELS

    assert TOOL.LOCAL_LEVELS == _LOCAL_LEVELS


def test_the_tool_never_drafts_into_solve_or_contract_pinned_registers():
    resources = {spec.resource for spec in TOOL.REGISTERS}
    assert "degenerate_reviewed_exclusions.json" not in resources
    assert "input_mass_reviewed_exclusions.json" not in resources
    assert all(spec.block == "exclusions" for spec in TOOL.REGISTERS)


def test_the_tool_drafts_only_gate_only_resources_when_they_are_declared():
    solve_scope = pytest.importorskip("microcosm.build.uk_runtime.solve_scope")
    declared = set(solve_scope.UK_GATE_ONLY_RESOURCES)
    assert {spec.resource for spec in TOOL.REGISTERS} <= declared
