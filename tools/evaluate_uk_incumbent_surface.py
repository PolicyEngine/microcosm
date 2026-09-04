"""Score a UK candidate on the incumbent's target surface, every grain (#762 I9).

Reads the dense candidate H5 (calibrated weights), compiles the national
registries from the pinned Ledger artifact (the un-excluded registry, so the
signed-out rows are measured too), resolves the engine once over the frame
for the local metrics, builds the national constraint matrix over the
calibrated weights, and evaluates every row of the incumbent's vendored
national and local fixtures. The incumbent's own estimates on its local
surface come from the extractor's household-metric and wide-weight tables.

Outputs a JSON evidence file and a markdown summary; refuses nothing that
merely fits badly — the point is to see the ugly part.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from importlib import resources as importlib_resources
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.uk_runtime.frs_release import load_uk_frs_release
from microcosm.build.uk_runtime.incumbent_surface_evaluation import (
    classify_local_rows,
    classify_national_rows,
    evaluation_summary,
    load_incumbent_local_fixture,
    load_incumbent_national_fixture,
    load_uk_data_target_parity,
    match_national_rows,
    national_family_status,
    render_markdown,
)
from microcosm.build.uk_runtime.ledger_targets import (
    _uk_local_metric_target_ids,
    compile_uk_target_registry,
)
from microcosm.build.uk_runtime.local_targets import metric_names
from microcosm.build.uk_runtime.measure_simulation import (
    load_uk_calibration_measure_exclusions,
)
from microcosm.build.uk_runtime.rowwise_dataset import load_uk_rowwise_dataset
from microcosm.calibrate import TargetRegistry
from microcosm.calibrate.matrix import build_constraint_matrix


def _driver():
    spec = importlib.util.spec_from_file_location(
        "build_uk_rowwise_candidate",
        Path(__file__).resolve().with_name("build_uk_rowwise_candidate.py"),
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-h5", required=True, type=Path)
    p.add_argument("--candidate-manifest", required=True, type=Path)
    p.add_argument("--ledger-facts", required=True, type=Path)
    p.add_argument("--ledger-facts-sha256", required=True)
    p.add_argument("--ledger-manifest-sha256", required=True)
    p.add_argument("--incumbent-metrics-csv", type=Path)
    p.add_argument("--incumbent-weights-csv", type=Path)
    p.add_argument("--out-json", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    p.add_argument("--engine-blocks", type=int, default=1)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    driver = _driver()
    artifact = load_ledger_consumer_artifact(
        args.ledger_facts,
        expected_facts_sha256=args.ledger_facts_sha256,
        expected_manifest_sha256=args.ledger_manifest_sha256,
    )
    period = int(load_uk_frs_release().calibration_year)
    compilation = compile_uk_target_registry(artifact.facts, target_period=period)
    if compilation.unsupported:
        raise SystemExit(
            f"{len(compilation.unsupported)} national references failed to compile"
        )
    registry = compilation.registry  # un-excluded: signed-out rows are measured too
    exclusions_all = {e["name"]: e for e in load_uk_calibration_measure_exclusions()}
    # Every excluded row stays measurable except the counterfactual measures,
    # which have no live route (that is why they are excluded): the resolver
    # would refuse the whole pass on them.
    contract = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("uk_population_targets.json")
        .read_text(encoding="utf-8")
    )
    counterfactual_ids: set[str] = set()

    def _collect(node) -> None:
        if isinstance(node, dict):
            if (
                "target_id" in node
                and "input_substitution_counterfactual" in json.dumps(node)
            ):
                counterfactual_ids.add(str(node["target_id"]))
            for value in node.values():
                _collect(value)
        elif isinstance(node, list):
            for value in node:
                _collect(value)

    _collect(contract)
    unresolvable = {
        s.name
        for s in registry.specs
        if s.name in exclusions_all
        and str(s.metadata.get("contract_target_id")) in counterfactual_ids
    }
    resolver_registry = TargetRegistry(
        [s for s in registry.specs if s.name not in unresolvable], country="uk"
    )
    manifest = json.loads(args.candidate_manifest.read_text())
    bound = set()
    for name in json.loads(
        Path(str(manifest["outputs"]["calibration_diagnostics"]["path"])).read_text()
    )["targets"]:
        bound.add(str(name["name"]).rsplit("@", 1)[0])
    exclusions = exclusions_all

    print("loading the candidate frame ...", file=sys.stderr, flush=True)
    frame, _ = load_uk_rowwise_dataset(args.candidate_h5)
    weights = np.asarray(frame.resolve_weights("household").values, dtype=np.float64)

    print("resolving the engine over the frame ...", file=sys.stderr, flush=True)
    with tempfile.TemporaryDirectory(prefix="uk-incumbent-eval-") as scratch:
        prepared_frame, _restore, national_rows, local_metrics, resolution = (
            driver._resolve_candidate_engine_surface(
                frame,
                resolver_registry,
                period=period,
                scratch_dir=Path(scratch),
                band_edge_registry=registry,
                blocks=args.engine_blocks,
            )
        )
    print("building the national constraint matrix ...", file=sys.stderr, flush=True)
    # The resolver returns the adapter's prepared frame carrying the resolved
    # measure columns; the raw frame has none of them (403 skipped).
    problem = build_constraint_matrix(prepared_frame, national_rows.targets)
    estimates = np.asarray(problem.matrix @ weights, dtype=np.float64)
    skipped = {
        str(getattr(t, "name", t)): str(getattr(t, "reason", ""))
        for t in problem.skipped
    }
    est_by_name = {
        str(n).rsplit("@", 1)[0]: float(v)
        for n, v in zip(problem.names, estimates, strict=True)
    }
    our_specs = [
        {
            "name": s.name,
            "contract_target_id": s.metadata.get("contract_target_id"),
            "value": s.value,
        }
        for s in registry.specs
    ]
    nat_fixture = load_incumbent_national_fixture()
    parity = load_uk_data_target_parity()
    national = match_national_rows(nat_fixture["rows"], our_specs)
    national = classify_national_rows(
        national,
        bound_names=bound,
        exclusions=exclusions,
        source_status=national_family_status(parity),
    )
    national["candidate_estimate"] = [
        est_by_name.get(n) if isinstance(n, str) else None for n in national["our_name"]
    ]
    national.loc[national["our_name"].isin(unresolvable), "status"] = (
        "measure_excluded:unresolvable_counterfactual"
    )
    national["skip_reason"] = [
        skipped.get(n) if isinstance(n, str) else None for n in national["our_name"]
    ]

    print("evaluating the local surface ...", file=sys.stderr, flush=True)
    household = frame.table("household")
    area_cols = {"constituency": "constituency_code", "la": "local_authority_code"}
    local_est: dict[tuple[str, str, str], float] = {}
    for grain, metrics in local_metrics.items():
        codes = household[area_cols[grain]].astype(str).to_numpy()
        m = (
            metrics.reindex(household["household_id"].tolist())
            if not metrics.index.equals(pd.Index(household["household_id"]))
            else metrics
        )
        weighted = m.multiply(weights, axis=0)
        sums = weighted.groupby(codes).sum()
        area_type = "constituency" if grain == "constituency" else "local_authority"
        for area, row in sums.iterrows():
            for metric, value in row.items():
                local_est[(area_type, str(area), str(metric))] = float(value)
    loc_fixture = load_incumbent_local_fixture()
    target_ids = {metric: tid for tid, metric in _uk_local_metric_target_ids().items()}
    unmapped_concern = {
        "housing/council_tax_net": ("blocked_source", "local_council_tax_net"),
        "housing/scotland_private_rent_amount": (
            "reviewed_exclusion",
            "local_devolved_constituency_rent_anchors",
        ),
        "housing/scotland_private_renter_households": (
            "reviewed_exclusion",
            "local_devolved_constituency_rent_anchors",
        ),
        "housing/wales_private_rent_amount": (
            "reviewed_exclusion",
            "local_devolved_constituency_rent_anchors",
        ),
        "housing/wales_private_renter_households": (
            "reviewed_exclusion",
            "local_devolved_constituency_rent_anchors",
        ),
    }
    membership = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("local_target_reference_membership.json")
        .read_text(encoding="utf-8")
    )
    local = classify_local_rows(
        loc_fixture["rows"],
        metric_target_ids=target_ids,
        membership=membership,
        our_metric_names={
            "constituency": metric_names("constituency"),
            "la": metric_names("la"),
        },
        bound_names=bound,
        unmapped_concern=unmapped_concern,
    )
    local["candidate_estimate"] = [
        local_est.get((a, g, m)) if isinstance(m, str) else None
        for a, g, m in zip(
            local["area_type"], local["geography_id"], local["our_metric"], strict=True
        )
    ]
    if (
        args.incumbent_metrics_csv is not None
        and args.incumbent_weights_csv is not None
    ):
        print(
            "incumbent's own estimates on its local surface ...",
            file=sys.stderr,
            flush=True,
        )
        inc_metrics = pd.read_csv(args.incumbent_metrics_csv).set_index("household_id")
        inc_weights = pd.read_csv(args.incumbent_weights_csv).set_index("household_id")
        inc_weights = inc_weights.reindex(inc_metrics.index)
        inc_est: dict[tuple[str, str], float] = {}
        metric_cols = [
            c for c in inc_metrics.columns if c in set(local["our_metric"].dropna())
        ]
        metric_matrix = inc_metrics[metric_cols].to_numpy(dtype=np.float64)
        for area in inc_weights.columns:
            w = inc_weights[area].to_numpy(dtype=np.float64)
            if not np.isfinite(w).all():
                continue
            vals = w @ metric_matrix
            for metric, value in zip(metric_cols, vals, strict=True):
                inc_est[(str(area), metric)] = float(value)
        local["incumbent_estimate"] = [
            inc_est.get((g, m)) if isinstance(m, str) else None
            for g, m in zip(local["geography_id"], local["our_metric"], strict=True)
        ]
    summary = evaluation_summary(national, local)
    payload = {
        "schema_version": 1,
        "kind": "uk_incumbent_surface_evaluation",
        "candidate_h5": str(args.candidate_h5),
        "calibration_year": period,
        "incumbent_fixtures": {
            "national": nat_fixture.get("provenance"),
            "local": loc_fixture.get("provenance"),
        },
        "measure_resolution": dict(resolution),
        "summary": summary,
        "national_rows": national.replace({np.nan: None}).to_dict(orient="records"),
        "local_rows": local.replace({np.nan: None}).to_dict(orient="records"),
    }
    args.out_json.write_text(json.dumps(payload, indent=1, default=str))
    args.out_md.write_text(render_markdown(summary, national, local))
    print(
        json.dumps(
            {
                "national": summary["national"]["candidate"],
                "local": summary["local"]["candidate"],
                "local_incumbent": summary["local"].get("incumbent"),
            },
            indent=1,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
