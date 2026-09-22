"""Unit tests for the pure parts of tools/build_us_acs_local_release.py."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import shlex
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _load_tool_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_us_acs_local_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_acs_local_release",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_staging_builder_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_us_acs_multispine_base.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_acs_multispine_base",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_spine_composition_reports_per_spine_weight_and_size() -> None:
    module = _load_tool_module()
    households = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "household_spine": ["asec_puf", "asec_puf", "acs_2024_1yr"],
        }
    )
    persons = pd.DataFrame({"person_household_id": [1, 2, 2, 3, 3, 3]})
    weights = np.asarray([10.0, 30.0, 60.0])

    composition = module.spine_composition(households, persons, weights)

    asec = composition["asec_puf"]
    assert asec["households"] == 2
    assert asec["household_weight"] == pytest.approx(40.0)
    assert asec["household_weight_share"] == pytest.approx(0.4)
    # person weight: 10*1 + 30*2 = 70 -> 70/40 persons per household.
    assert asec["person_weight"] == pytest.approx(70.0)
    assert asec["implied_persons_per_household"] == pytest.approx(1.75)
    acs = composition["acs_2024_1yr"]
    assert acs["implied_persons_per_household"] == pytest.approx(3.0)
    overall = composition["_all"]
    assert overall["households"] == 3
    assert overall["effective_sample_size"] == pytest.approx(
        (10 + 30 + 60) ** 2 / (10**2 + 30**2 + 60**2)
    )


def test_finalize_reviewed_limitations_carries_staging_and_dedupes() -> None:
    module = _load_tool_module()
    staging_summary = {
        "base": {
            "donor_release": {
                "release_id": "populace-us-2024-buildo-sparse-rmloss100-x",
            }
        },
        "reviewed_limitations": [
            {"id": "acs_group_quarters_housing_universe", "status": "reviewed"},
            {
                "id": "cd_population_marginal_vintage_2020",
                "status": "stale_staging_copy",
            },
        ],
    }
    diagnostics = {
        "effective_sample_size": 4853.9,
        "ess_fraction": 0.003,
        "households": 1_600_000,
    }
    spine_qa = {
        "per_spine": {
            "acs_2024_1yr": {"ssi_incidence": 0.0259},
            "asec_puf": {"ssi_incidence": 0.0185},
        }
    }

    limitations = module.finalize_reviewed_limitations(
        staging_summary, diagnostics, spine_qa
    )

    by_id = {item["id"]: item for item in limitations}
    # Staging entries carried; finalize's own entry wins the id collision.
    assert "acs_group_quarters_housing_universe" in by_id
    assert by_id["cd_population_marginal_vintage_2020"]["status"] == (
        "reviewed_vintage"
    )
    # Lineage entries present, none blocking.
    for required in (
        "ssi_aged_band_collapse_inherited",
        "miscellaneous_income_loss_side_donor_defect",
        "tips_return_count_carrier_deficit_inherited",
        "low_effective_sample_size_lambda_zero",
        "donor_sparse_selection_training_set",
        "mixed_sub_puma_column_coverage",
    ):
        assert required in by_id, required
        assert by_id[required]["calibration_blocker"] is False
    ssi = by_id["ssi_aged_band_collapse_inherited"]
    assert ssi["measured_spine_ssi"] == {
        "acs_2024_1yr": 0.0259,
        "asec_puf": 0.0185,
    }
    donor = by_id["donor_sparse_selection_training_set"]
    assert "populace-us-2024-buildo-sparse-rmloss100-x" in donor["reason"]
    # Ids are unique after dedupe.
    assert len(by_id) == len(limitations)


def test_parse_args_enforces_stage_requirements(tmp_path: Path) -> None:
    module = _load_tool_module()
    base = [
        "--staging-h5",
        str(tmp_path / "staging.h5"),
        "--checkpoint-dir",
        str(tmp_path / "ckpt"),
    ]

    with pytest.raises(SystemExit):
        module._parse_args(["--stage", "materialize", *base])
    with pytest.raises(SystemExit):
        module._parse_args(["--stage", "calibrate", *base])
    with pytest.raises(SystemExit):
        module._parse_args(
            ["--stage", "package", *base, "--out-h5", str(tmp_path / "o.h5")]
        )

    args = module._parse_args(
        [
            "--stage",
            "all",
            *base,
            "--feed",
            str(tmp_path / "facts.jsonl"),
            "--out-h5",
            str(tmp_path / "out.h5"),
            "--out",
            str(tmp_path / "release"),
        ]
    )
    assert args.stages == ["materialize", "calibrate", "qa", "finalize", "package"]
    assert args.out_summary == tmp_path / "out.summary.json"
    assert args.gate_report == tmp_path / "ckpt" / "gate_summary.json"


def test_release_id_prefix_and_manifest_constants() -> None:
    module = _load_tool_module()
    assert module.RELEASE_ID_PREFIX == "populace-us-2024-buildo-acs-local"
    assert module.RELEASE_NAMESPACE == "buildo_acs_local"
    assert module.ARTIFACT_FILENAME == "populace_us_2024_acs_local.h5"
    assert module.HF_REPO_ID == "policyengine/populace-us"


def test_documented_staging_recipe_matches_legacy_and_release_parsers(
    tmp_path: Path,
) -> None:
    release = _load_tool_module()
    staging_builder = _load_staging_builder_module()
    recipe = shlex.split(release.LEGACY_STAGING_REFRESH_RECIPE)

    assert recipe[:3] == [
        "uv",
        "run",
        "tools/build_us_acs_multispine_base.py",
    ]
    staging_args = staging_builder._legacy._parse_args(recipe[3:])
    staging_summary = staging_args.summary or staging_args.out_h5.with_suffix(
        ".summary.json"
    )
    release_args = release._parse_args(
        [
            "--stage",
            "materialize",
            "--staging-h5",
            str(staging_args.out_h5),
            "--checkpoint-dir",
            str(tmp_path / "ckpt"),
            "--feed",
            str(tmp_path / "facts.jsonl"),
        ]
    )

    assert release._staging_summary_path(release_args) == staging_summary


def test_do_finalize_requires_calibration_diagnostics(tmp_path: Path) -> None:
    module = _load_tool_module()
    staging = tmp_path / "staging.h5"
    staging.touch()
    (tmp_path / "staging.summary.json").write_text(json.dumps({}))
    args = module._parse_args(
        [
            "--stage",
            "finalize",
            "--staging-h5",
            str(staging),
            "--checkpoint-dir",
            str(tmp_path / "ckpt"),
            "--out-h5",
            str(tmp_path / "out.h5"),
        ]
    )

    with pytest.raises(SystemExit, match="No calibration diagnostics"):
        module.do_finalize(args)


_UNSET = object()


def _package_args_before_evidence(
    module, tmp_path: Path, *, max_households=_UNSET, soi_mode="totals"
):
    """The package stage's inputs up to (not including) the qa/consumer evidence.

    ``max_households`` is what the staging summary records under
    ``orchestration``; the sentinel omits the block entirely. ``soi_mode`` is
    what ``materialize_rss.json`` records; the sentinel omits the file.
    """
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    if soi_mode is not _UNSET:
        (ckpt / "materialize_rss.json").write_text(json.dumps({"soi_mode": soi_mode}))
    staging = tmp_path / "staging.h5"
    staging.write_bytes(b"staging")
    summary: dict = {}
    if max_households is not _UNSET:
        summary["orchestration"] = {
            "max_households": max_households,
            "n_estimators": 32,
            "max_targets_per_fit": 8,
        }
    (tmp_path / "staging.summary.json").write_text(json.dumps(summary))
    out_h5 = tmp_path / "out.h5"
    out_h5.write_bytes(b"artifact")
    (ckpt / "calibration_diagnostics.json").write_text(json.dumps({"households": 1}))
    (ckpt / "gate_summary.json").write_text(json.dumps({"gates": {}}))
    (ckpt / "run_identity.json").write_text(
        json.dumps(
            {
                "staging_sha256": module._sha256(staging),
                "population_cells_dropped": [],
            }
        )
    )
    (tmp_path / "out.summary.json").write_text(json.dumps({"simulation_ready": True}))
    return module._parse_args(
        [
            "--stage",
            "package",
            "--staging-h5",
            str(staging),
            "--checkpoint-dir",
            str(ckpt),
            "--out-h5",
            str(out_h5),
            "--out",
            str(tmp_path / "release"),
            "--allow-dirty",
        ]
    )


@pytest.mark.parametrize(
    ("max_households", "message"),
    [
        (5000, r"capped at 5000 ACS household"),
        (0, r"capped at 0 ACS household"),
        (_UNSET, r"does not record orchestration\.max_households"),
    ],
    ids=["capped", "capped-at-zero", "cap-not-recorded"],
)
def test_package_refuses_a_capped_or_unattested_staging_run(
    tmp_path: Path, max_households, message
) -> None:
    """A capped smoke passes every finalize gate (the donor spine keeps every
    ladder cell populated), so the cap itself must refuse packaging, before
    any release directory exists."""

    module = _load_tool_module()
    args = _package_args_before_evidence(
        module, tmp_path, max_households=max_households
    )
    with pytest.raises(SystemExit, match=message):
        module.do_package(args)
    assert not (args.out / "package_result.json").exists()
    assert not (args.out / "releases").exists(), "a refused smoke leaves no release"


def test_package_checks_the_cap_before_reading_the_evidence(tmp_path: Path) -> None:
    """An uncapped summary reaches the evidence checks; the cap check is first."""

    module = _load_tool_module()
    args = _package_args_before_evidence(module, tmp_path, max_households=None)
    with pytest.raises(SystemExit, match="spine_qa.json is missing"):
        module.do_package(args)


def test_do_package_requires_qa_and_consumer_evidence(tmp_path: Path) -> None:
    """Absent evidence must refuse packaging, never read as vacuously green."""

    module = _load_tool_module()
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    staging = tmp_path / "staging.h5"
    staging.write_bytes(b"staging")
    (tmp_path / "staging.summary.json").write_text(
        json.dumps({"orchestration": {"max_households": None}})
    )
    out_h5 = tmp_path / "out.h5"
    out_h5.write_bytes(b"artifact")
    (ckpt / "calibration_diagnostics.json").write_text(json.dumps({"households": 1}))
    (ckpt / "gate_summary.json").write_text(json.dumps({"gates": {}}))
    (ckpt / "materialize_rss.json").write_text(json.dumps({"soi_mode": "totals"}))
    (ckpt / "run_identity.json").write_text(
        json.dumps(
            {
                "staging_sha256": module._sha256(staging),
                "population_cells_dropped": [],
            }
        )
    )
    args = module._parse_args(
        [
            "--stage",
            "package",
            "--staging-h5",
            str(staging),
            "--checkpoint-dir",
            str(ckpt),
            "--out-h5",
            str(out_h5),
            "--out",
            str(tmp_path / "release"),
            "--allow-dirty",
        ]
    )
    (tmp_path / "out.summary.json").write_text(json.dumps({"simulation_ready": True}))

    with pytest.raises(SystemExit, match="spine_qa.json is missing"):
        module.do_package(args)

    (ckpt / "spine_qa.json").write_text(
        json.dumps(
            {
                "plain_consumption": True,
                "artifact_sha256": module._sha256(out_h5),
                "per_spine": {},
            }
        )
    )
    with pytest.raises(SystemExit, match="consumer_export.json is missing"):
        module.do_package(args)


# ---------------------------------------------------------------------------
# SOI target surface: totals by default, full (soi_fiscal_distribution) opt-in
# ---------------------------------------------------------------------------


def _materialize_argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--stage",
        "materialize",
        "--staging-h5",
        str(tmp_path / "staging.h5"),
        "--checkpoint-dir",
        str(tmp_path / "ckpt"),
        "--feed",
        str(tmp_path / "facts.jsonl"),
        *extra,
    ]


def test_soi_mode_defaults_to_totals_and_full_is_an_explicit_opt_in(
    tmp_path: Path,
) -> None:
    """The 2026-09-22 ruling on #969: SOI totals are the ACS local default;
    ``full`` (adds every ``soi_fiscal_distribution`` spec) is reachable only
    by asking for it."""

    module = _load_tool_module()
    assert module.SOI_MODES == ("totals", "full")
    assert module.DEFAULT_SOI_MODE == module.SOI_MODE_TOTALS == "totals"
    assert module._parse_args(_materialize_argv(tmp_path)).soi_mode == "totals"
    assert (
        module._parse_args(_materialize_argv(tmp_path, "--soi-mode", "full")).soi_mode
        == "full"
    )
    signature = inspect.signature(module.state_admin_specs)
    assert signature.parameters["soi_mode"].default == "totals"
    with pytest.raises(SystemExit):
        module._parse_args(_materialize_argv(tmp_path, "--soi-mode", "ful"))


def _spec(role: str | None, *, state: bool = True) -> SimpleNamespace:
    metadata: dict[str, str] = {}
    if state:
        metadata["state_fips"] = "06"
    if role is not None:
        metadata["target_role"] = role
    return SimpleNamespace(metadata=metadata)


def test_soi_surface_predicate_drops_soi_fiscal_distribution_only_in_totals() -> None:
    module = _load_tool_module()
    totals = module.soi_surface_predicate("totals")
    full = module.soi_surface_predicate("full")

    band = _spec("soi_fiscal_distribution")
    assert not totals(band)
    assert full(band)
    for role in ("aca_ptc_returns", "aca_spending", None):
        assert totals(_spec(role)) and full(_spec(role)), role
    # Neither mode reaches past the state surface.
    for mode_predicate in (totals, full):
        assert not mode_predicate(_spec("aca_spending", state=False))
        assert not mode_predicate(_spec("soi_fiscal_distribution", state=False))


def test_unknown_soi_mode_is_refused_before_the_feed_is_read(tmp_path: Path) -> None:
    """A typo must never fall through to either surface (the old predicate
    treated every value other than ``full`` as totals)."""

    module = _load_tool_module()
    missing_feed = tmp_path / "never-read.jsonl"
    for call in (
        lambda: module.soi_surface_predicate("ful"),
        lambda: module.state_admin_specs(missing_feed, ["soi"], soi_mode="ful"),
        lambda: module.release_refresh_recipe("ful"),
    ):
        with pytest.raises(ValueError, match="soi_mode must be one of"):
            call()


@pytest.mark.parametrize("soi_mode", ["totals", "full"])
def test_release_refresh_recipe_reproduces_its_soi_mode(
    tmp_path: Path, soi_mode: str
) -> None:
    """The recipe names the mode, so re-running it cannot drift with the
    parser default."""

    module = _load_tool_module()
    recipe = shlex.split(module.release_refresh_recipe(soi_mode))
    assert recipe[:3] == ["uv", "run", "tools/build_us_acs_local_release.py"]
    assert recipe[recipe.index("--soi-mode") + 1] == soi_mode
    args = module._parse_args(recipe[3:])
    assert args.soi_mode == soi_mode
    assert args.stages == ["materialize", "calibrate", "qa", "finalize", "package"]


@pytest.mark.parametrize(
    ("recorded", "message"),
    [
        (_UNSET, r"records soi_mode=None"),
        ("bands", r"records soi_mode='bands'"),
    ],
    ids=["not-recorded", "unknown"],
)
def test_package_refuses_a_checkpoint_without_a_known_soi_mode(
    tmp_path: Path, recorded, message
) -> None:
    module = _load_tool_module()
    args = _package_args_before_evidence(
        module, tmp_path, max_households=None, soi_mode=recorded
    )
    with pytest.raises(SystemExit, match=message):
        module.do_package(args)
    assert not (args.out / "releases").exists(), "a refusal leaves no release"


def _write_complete_package_evidence(module, args) -> None:
    ckpt = args.checkpoint_dir
    artifact_sha = module._sha256(args.out_h5)
    (ckpt / "spine_qa.json").write_text(
        json.dumps(
            {
                "plain_consumption": True,
                "artifact_sha256": artifact_sha,
                "per_spine": {},
            }
        )
    )
    (ckpt / "consumer_export.json").write_text(json.dumps({"held_back_total": 0}))
    (ckpt / "consumer_reviewed_null_fills.json").write_text(json.dumps({"columns": {}}))


@pytest.mark.parametrize("recorded", ["totals", "full"])
def test_package_records_the_materialized_soi_mode_not_the_parser_default(
    tmp_path: Path, monkeypatch, recorded: str
) -> None:
    """The package invocation passes no ``--soi-mode`` (so the parser says
    ``totals``); the manifest and recipe must carry what materialize used."""

    module = _load_tool_module()
    monkeypatch.setattr(
        module,
        "_repo_code_identity",
        lambda allow_dirty: {"sha": "abc1234", "dirty": False, "branch": "test"},
    )
    args = _package_args_before_evidence(
        module, tmp_path, max_households=None, soi_mode=recorded
    )
    assert args.soi_mode == "totals"
    _write_complete_package_evidence(module, args)

    result = module.do_package(args)

    release_dir = Path(result["release_dir"])
    build_manifest = json.loads((release_dir / "build_manifest.json").read_text())
    release_manifest = json.loads((release_dir / "release_manifest.json").read_text())
    assert build_manifest["materialize"]["soi_mode"] == recorded
    for manifest in (build_manifest, release_manifest):
        recipe = shlex.split(manifest["refresh_recipe"]["release"])
        assert recipe[recipe.index("--soi-mode") + 1] == recorded


def test_pinned_feed_default_state_surface_carries_no_soi_fiscal_distribution() -> None:
    """On the real feed, the default call selects SOI totals and ``full``
    adds exactly the ``soi_fiscal_distribution`` specs, nothing else.

    The pinned consumer-facts feed is a 164 MB public aggregate export that
    no CI lane carries, so this runs only when ``MICROCOSM_US_CHRONICLE_FACTS``
    points at it (the convention of the #969 state-surface arm). Two registry
    compiles; about 3 GB peak RSS.
    """

    feed = os.environ.get("MICROCOSM_US_CHRONICLE_FACTS", "")
    if not feed or not Path(feed).exists():
        pytest.skip(
            "set MICROCOSM_US_CHRONICLE_FACTS to the pinned consumer-facts feed"
        )

    module = _load_tool_module()
    families = ["snap", "medicaid", "soi"]
    default_registry, _ = module.state_admin_specs(feed, families)
    full_registry, _ = module.state_admin_specs(feed, families, soi_mode="full")
    default_names = {spec.name for spec in default_registry.specs}
    full_by_name = {spec.name: spec for spec in full_registry.specs}

    assert default_names
    assert not [
        spec
        for spec in default_registry.specs
        if spec.metadata.get("target_role") == "soi_fiscal_distribution"
    ]
    assert default_names < set(full_by_name)
    added = [full_by_name[name] for name in set(full_by_name) - default_names]
    assert added
    assert {(spec.family, spec.metadata.get("target_role")) for spec in added} == {
        ("irs_soi", "soi_fiscal_distribution")
    }
