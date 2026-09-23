"""Run the native US survey composition: financial -> PUF55 -> enrichment.

One process, one outer source verification epoch, one live owner chain. The
financial host (atomic prefix, base financial graph, property/tax/person-status
extensions and the child/household-role completion) runs first; the PUF55
extension and the survey enrichment (hours, housing, SPM, canonical state and
the original-arm PUF application) run over that same retained owner; the
development checkpoint writer then exports the enriched population and reads
it back. Nothing is reconstructed from receipts or files.

This is a development measurement, not a build or a release:

* the option profile is the reviewed development profile recorded in
  ``DEVELOPMENT_PROFILE`` (fraction, seeds, two-estimator models, candidate
  child scenario, packaged PUF donors, SPM ASEC scope policy). Other optional
  successors (immigration transfer, health completion, demographic/race inputs,
  full-original amount donors) stay off;
* the run is never release eligible, never calibrates and never publishes;
* survey poverty is not computed, targeted or used to select anything here.

The receipt this tool writes holds aggregates only: phase wall/CPU/RSS, entity
counts, node and cache-hit counts, SPM scope counts and refusal codes. Private
rows, IDs and source cells never enter it. Graph store, snapshots and the
development checkpoint stay under ``--output-root`` on the local machine.

Guards (all enforced in-process):

* admission: minimum available system memory, minimum free disk, and no other
  native graph run on the machine before anything is imported;
* CPU, wall and resident-memory ceilings, plus a system-memory floor that stops
  this run (rather than a co-tenant) when the machine runs short;
* no network, no subprocesses and no country-engine imports.

Example (1/1000 cold, the reviewed development profile)::

    PYTHONPATH=<spm-calculator PR45 checkout> .venv/bin/python -B \\
        tools/run_us_native_survey_composition.py \\
        --sources <staged run>/sources \\
        --geography-support <staged run>/geography/national-atomic-support.npz \\
        --puf-main <puf_2015.csv> --puf-demographic <demographics_2015.csv> \\
        --output-root <fresh dir> --receipt <receipt.json> --execute
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
import re
import resource
import signal
import sys
import threading
import time
import traceback
from fractions import Fraction
from pathlib import Path

PROTOCOL = "microcosm.us.native-survey-composition-run.v1"
GIB = 1024**3

# The reviewed development profile (current-native-composition PLAN, 2026-09-21).
# Declaration digests bind the reconstructed option objects to the reviewed
# documents; a changed option refuses before any source is read.
DEVELOPMENT_PROFILE = {
    "financial": {
        "fraction": [1, 1000],
        "seed": 20260908,
        "geography_seed": 20260908,
        "demographic_conditioning": True,
        "n_estimators": 2,
        "person_status": True,
        "household_roles": True,
        "rebase_property_taxes": True,
        "source_qualified_development_inputs": True,
        "population_retention": "all",
        "property_income": {
            "scales": [1.0, 1.0, 1.0, 1.0],
            "atol": 1e-10,
            "rtol": 1e-12,
            "n_estimators": 2,
            "completion_routing": True,
            "declaration_sha256": (
                "5b6429090202437caa9bc20a99ea30d42ec351b1ccf6a4c074d3f28c1e357dea"
            ),
        },
        "child_property": {
            "scenario_id": "us-native-completion45-v5-candidate",
            "scope": "candidate",
            "policy_id": "us-native-completion45-v5-minimum-one",
            "threshold": [1, 1, 1.0, 1.0],
            "required_patterns": [3],
            "transport": [1.0, [1.0, 1.0]],
            "stream": [
                "sha256-u53-v1",
                "us-native-completion45-v5-child-stream",
                0,
                7,
            ],
            "document_sha256": (
                "c62a63c6d5a15cf199ed5b9d938d1704be3b4647a14998fd7cfaa3a7b309ac6b"
            ),
        },
    },
    "geography": {
        "support_sha256": (
            "5edc0e77471ba31d550a1eed416d5b46ada0a35425718eb87cfabe4d66fe4960"
        ),
        "source_ids": [
            ["district", "census-2025-cd119-NationalCD119.txt"],
            ["population", "census-2020-dec-pl-api-P1_001N"],
            ["puma", "census-2020-Census-Tract-to-2020-PUMA"],
        ],
    },
    "puf": {"seed": 578, "n_estimators": 2, "zero_atol": 1e-8},
    "enrichment": {
        "groups": ["unemployment", "health_costs"],
        "n_estimators": 2,
        "spm_acs_profile": {
            "year": 2024,
            "source_vintage": "acs_2024_1yr",
            "admit_modeled": True,
            "admit_approved_inference": True,
        },
        "spm_asec_scope_policy": "ASEC_2025_INCOME_2024_SPM_POLICY",
        "spm_outside_role_placeholder": False,
        "canonical_state_input": True,
        "original_application_seed": 579,
        "immigration_transfer": None,
        "health_completion": False,
        "demographic_inputs": False,
        "race_hispanic_inputs": False,
        "full_original_amount_donors": False,
    },
}

THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "POPULACE_FIT_N_JOBS": "1",
    "POPULACE_FIT_PREDICT_WORKERS": "1",
    "JOBLIB_MULTIPROCESSING": "0",
}

# Command-line fragments that identify another native graph run on the machine.
NATIVE_RUN_MARKERS = (
    "run_us_native_survey_composition",
    "harness19",
    "launch_continuation.py",
    "launch_current.py",
    "run_atomic_survey_financial",
    "native-survey-composition",
)

_CODE = re.compile(r"^[A-Z][A-Z0-9_]{2,80}$")


class CompositionRefusedError(RuntimeError):
    """A named refusal of this runner (not of the maintained runtime)."""


def require(condition, code):
    if not condition:
        raise CompositionRefusedError("NATIVE_COMPOSITION_" + code)


def sanitized_error(error):
    """Exception type plus its message only when that message is a bare code.

    Maintained refusals carry codes such as ``PUF_PACKAGED_ROUTES``; arbitrary
    library messages may quote values, so they are withheld from the receipt.
    """
    message = str(error)
    codes = [token for token in re.split(r"[\s:;,()'\"]+", message) if token]
    if message and all(_CODE.match(token) for token in codes):
        shown = message
    else:
        found = [token for token in codes if _CODE.match(token)]
        shown = (
            f"message withheld ({len(message)} chars); codes: {found[:8]}"
            if found
            else f"message withheld ({len(message)} chars)"
        )
    chain = []
    cause = error.__cause__ or error.__context__
    while cause is not None and len(chain) < 4:
        text = str(cause)
        chain.append(
            {
                "type": type(cause).__name__,
                "message": text if _CODE.match(text or "-") else None,
            }
        )
        cause = cause.__cause__ or cause.__context__
    return {"type": type(error).__name__, "message": shown, "causes": chain}


# --------------------------------------------------------------------------
# Option reconstruction (pure, no source access)
# --------------------------------------------------------------------------


def completion_options(profile, host):
    """Rebuild the declared property/child options and check their digests."""
    child = host._completion_module().child
    empirical = child.empirical
    row = profile["property_income"]
    property_options = host._property_module().PropertyIncomeOptions(
        scales=tuple(row["scales"]),
        atol=row["atol"],
        rtol=row["rtol"],
        n_estimators=row["n_estimators"],
        completion_routing=row["completion_routing"],
    )
    require(
        hashlib.sha256(property_options.to_bytes()).hexdigest()
        == row["declaration_sha256"],
        "PROPERTY_DECLARATION_CHANGED",
    )
    row = profile["child_property"]
    require(row["scope"] == "candidate", "CHILD_SCOPE_MUST_REMAIN_CANDIDATE")
    threshold = empirical.SupportThreshold(*row["threshold"])
    child_options = child.ChildPropertyOptions(
        row["scenario_id"],
        row["scope"],
        empirical.SupportRequirements(
            row["policy_id"],
            row["scope"],
            threshold,
            threshold,
            threshold,
            tuple(row["required_patterns"]),
        ),
        empirical.JointTransport(row["transport"][0], tuple(row["transport"][1])),
        tuple(row["stream"]),
    )
    document = child_options.document()
    require(
        document["source_observation_claim"] is False, "CHILD_SOURCE_CLAIM_FORBIDDEN"
    )
    require(
        hashlib.sha256(host.codec.encode_json(document)).hexdigest()
        == row["document_sha256"],
        "CHILD_DECLARATION_CHANGED",
    )
    return property_options, child_options


def financial_arguments(
    profile, *, host, geography, sources, support, store, snapshots, resume
):
    require(resume in ("auto", "require"), "RESUME")
    row = profile["financial"]
    require(row["seed"] == row["geography_seed"], "SEEDS")
    properties, children = completion_options(row, host)
    return {
        "source_dir": Path(sources),
        "snapshot_root": Path(snapshots),
        "store_root": Path(store),
        "fraction": Fraction(*row["fraction"]),
        "seed": row["seed"],
        "geography_config": geography.AtomicSurveyReconstruction(
            support_path=str(Path(support).absolute()),
            support_sha256=profile["geography"]["support_sha256"],
            source_ids=tuple(
                tuple(item) for item in profile["geography"]["source_ids"]
            ),
            seed=row["geography_seed"],
        ),
        "demographic_conditioning": row["demographic_conditioning"],
        "n_estimators": row["n_estimators"],
        "property_income": properties,
        "child_property": children,
        "person_status": row["person_status"],
        "household_roles": row["household_roles"],
        "rebase_property_taxes": row["rebase_property_taxes"],
        "source_qualified_development_inputs": row[
            "source_qualified_development_inputs"
        ],
        "resume": resume,
        "return_values": True,
        "_population_retention": row["population_retention"],
    }


def donor_sources(puf, *, main, demographic):
    """Bind the packaged PUF pins to the supplied files, by bytes."""
    definition = puf.canonical.raw.packaged_definition()
    result = {}
    for pin, path in ((definition.main, main), (definition.demographic, demographic)):
        path = Path(path)
        require(path.is_file(), "PUF_FILE_MISSING")
        require(path.stat().st_size == pin.bytes, "PUF_FILE_SIZE")
        require(sha256_file(path) == pin.sha256, "PUF_FILE_SHA256")
        result[pin.source_name] = path
    return result


def puf_arguments(profile, *, donors, resume):
    require(resume in ("auto", "require"), "RESUME")
    row = profile["puf"]
    return {
        "donor_sources": donors,
        "seed": row["seed"],
        "n_estimators": row["n_estimators"],
        "zero_atol": row["zero_atol"],
        "fixture_definition": None,
        "resume": resume,
    }


def spm_policy(profile, source):
    row = profile["enrichment"]
    acs = row["spm_acs_profile"]
    require(
        row["spm_asec_scope_policy"] == "ASEC_2025_INCOME_2024_SPM_POLICY",
        "SPM_SCOPE_POLICY",
    )
    return {
        "acs_profile": source.acs.ACSAnalysisProfile(
            acs["year"],
            acs["source_vintage"],
            admit_modeled=acs["admit_modeled"],
            admit_approved_inference=acs["admit_approved_inference"],
        ),
        "asec_scope_policy": source.ASEC_2025_INCOME_2024_SPM_POLICY,
    }


def enrichment_arguments(profile, policy, *, resume):
    require(resume in ("auto", "require"), "RESUME")
    row = profile["enrichment"]
    return {
        "groups": tuple(row["groups"]),
        "n_estimators": row["n_estimators"],
        "resume": resume,
        "spm_acs_profile": policy["acs_profile"],
        "spm_asec_scope_policy": policy["asec_scope_policy"],
        "spm_outside_role_placeholder": row["spm_outside_role_placeholder"],
        "canonical_state_input": row["canonical_state_input"],
        "original_application_seed": row["original_application_seed"],
        "immigration_transfer": row["immigration_transfer"],
        "health_completion": row["health_completion"],
        "demographic_inputs": row["demographic_inputs"],
        "race_hispanic_inputs": row["race_hispanic_inputs"],
        "full_original_amount_donors": row["full_original_amount_donors"],
    }


# --------------------------------------------------------------------------
# Aggregate summaries and composition checks
# --------------------------------------------------------------------------


def sha256_file(path, chunk=8 * 1024**2):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def entity_counts(population):
    frame = population.frame
    return {entity: int(frame.n(entity)) for entity in frame.entities}


def graph_counts(run, *, inherited=()):
    order = list(run.compiled.order)
    parent = set(inherited)
    hits = [name for name in order if run.manifest.node(name).hit]
    receipts = run.manifest.receipts
    return {
        "nodes": len(order),
        "inherited_nodes": len(parent),
        "new_nodes": len(order) - len(parent & set(order)),
        "store_hits": len(hits),
        "new_node_hits": len([name for name in hits if name not in parent]),
        "node_loop_wall_seconds": round(
            sum(receipts[name].wall_time_s for name in order if name in receipts), 3
        ),
        "manifest_key": run.manifest.key,
    }


def node_wall_times(run, *, exclude=()):
    receipts = run.manifest.receipts
    skip = set(exclude)
    return sorted(
        (
            [name, round(receipts[name].wall_time_s, 3)]
            for name in run.compiled.order
            if name in receipts and name not in skip
        ),
        key=lambda row: -row[1],
    )


def assert_inherited(parent, child):
    """Every parent node is a hit in the child with unchanged key and artifacts."""
    for node in parent.compiled.order:
        before, after = parent.manifest.node(node), child.manifest.node(node)
        require(
            after.hit
            and before.key == after.key
            and before.kernel_impl_hash == after.kernel_impl_hash
            and before.artifacts == after.artifacts
            and before.opaque_artifacts == after.opaque_artifacts,
            "INHERITED_RECORD_CHANGED",
        )


def check_new_nodes(run, *, inherited, required):
    """Inherited nodes must hit; new nodes hit only on a required replay."""
    parent = set(inherited)
    require(parent.issubset(set(run.compiled.order)), "MISSING_INHERITED_NODE")
    for name in run.compiled.order:
        if name in parent:
            require(run.manifest.node(name).hit, "INHERITED_NODE_MISSED")
        elif required:
            require(run.manifest.node(name).hit, "REQUIRED_NODE_MISSED")


def spm_source_preflight(financial, policy):
    """Qualify SPM inputs on the financial owner before the PUF phase."""
    import pandas as pd

    from microcosm.build.spm_input_contract import OUTSIDE, UNRESOLVED
    from microcosm.build.us_runtime import current_survey_household_roles as roles
    from microcosm.build.us_runtime import current_survey_spm_projection as bridge
    from microcosm.build.us_runtime import current_survey_spm_source as source

    # The financial owner was verified as its run returned and is requalified
    # in the final phase; this preflight only borrows its retained preparation.
    preparation = financial.prefix.preparation
    qualified_roles = roles.qualify_current_survey_household_roles(preparation)
    reconciliation = roles.household_role_reconciliation(
        qualified_roles, financial.financial_population.frame
    )
    require(
        reconciliation["conflicting_cells"] == 0
        and reconciliation["incumbent_known_qualified_unbound"] == 0,
        "HOUSEHOLD_ROLE_RECONCILIATION",
    )
    qualified = source.qualify_current_survey_spm(preparation, **policy)
    projected = bridge.project_spm_inputs(
        qualified.source_frame,
        qualified.origins,
        qualified.roles,
        qualified.unit_status,
        financial.financial_population.frame,
        source_year=2024,
        year=2024,
        outside_role_placeholder=False,
    )
    bridge.validate_spm_projection(projected, qualified.validate)
    require(not qualified.unit_status.eq(UNRESOLVED).any(), "ACTUAL_SPM_UNRESOLVED")
    require(
        len(projected.unit_mapping) == 2 * len(qualified.unit_status),
        "SPM_CLONE_UNIT_MAPPING",
    )
    pd.testing.assert_series_equal(
        projected.nullable_source_roles, qualified.roles, check_exact=True
    )
    report = {
        "original_persons": len(qualified.roles),
        "original_units": len(qualified.unit_status),
        "source_missing_roles": int(qualified.roles.isna().sum()),
        "scope_counts": {
            str(key): int(value)
            for key, value in qualified.unit_status.value_counts().items()
        },
        "outside_units": int(qualified.unit_status.eq(OUTSIDE).sum()),
        "receiving_placeholder_persons": len(projected.placeholder_person_ids),
        "cloned_units": len(projected.unit_mapping),
        "household_role_reconciliation": {
            key: int(value)
            for key, value in reconciliation.items()
            if isinstance(value, (int,)) and not isinstance(value, bool)
        },
    }
    qualified.validate()
    return qualified, projected, report


def conserved_axes_weights(parent, child, compiled, added_nodes):
    """Enrichment keeps entity axes, weights and mass; only keep-all FILTERs add."""
    import pandas as pd

    from microcosm.build.us_runtime import survey_population_replay as replay
    from microcosm.graph import StructuralDelta

    left, right = parent.frame, child.frame
    require(
        left.entities == right.entities
        and left.weighted_entities == right.weighted_entities
        and left.links == right.links,
        "EXTENSION_ENTITY_CONTRACT",
    )
    for entity in left.entities:
        pd.testing.assert_index_equal(
            left.table(entity).index, right.table(entity).index, exact=True
        )
        pd.testing.assert_series_equal(
            left.table(entity)[entity + "_id"],
            right.table(entity)[entity + "_id"],
            check_exact=True,
        )
    replay._series(left.strata, right.strata)
    require(
        child.mass_ledger[: len(parent.mass_ledger)] == parent.mass_ledger
        and tuple(parent.design_weights) == tuple(child.design_weights),
        "EXTENSION_MASS_LEDGER_PREFIX",
    )
    lineage, cursor, seen = [], child.version, set()
    while cursor != parent.version:
        require(cursor not in seen and cursor in added_nodes, "EXTENSION_VERSION")
        seen.add(cursor)
        node = compiled.graph.node(cursor)
        require(
            node.structural is StructuralDelta.FILTER
            and node.mass == "conserve"
            and node.weights is None,
            "EXTENSION_NOT_KEEP_ALL_FILTER",
        )
        lineage.append(node)
        cursor = node.base
    added = child.mass_ledger[len(parent.mass_ledger) :]
    expected = tuple(reversed(lineage))
    require(len(added) == len(expected), "EXTENSION_MASS_LEDGER_COUNT")
    for record, node in zip(added, expected, strict=True):
        require(
            record.node_id == node.id
            and record.operation == node.structural.value
            and record.policy == node.mass
            and record.before_total == record.after_total
            and record.before_by_stratum == record.after_by_stratum
            and record.before_by_partition_stratum == record.after_by_partition_stratum,
            "EXTENSION_FILTER_MASS_CHANGED",
        )
    for entity in left.weighted_entities:
        a, b = left.weights_for(entity), right.weights_for(entity)
        require(
            a.kind is b.kind and replay._array_bytes_equal(a.values, b.values),
            "EXTENSION_WEIGHTS",
        )
    for entity in parent.design_weights:
        require(
            replay._array_bytes_equal(
                parent.design_weights[entity], child.design_weights[entity]
            ),
            "EXTENSION_DESIGN_WEIGHTS",
        )
    for entity in left.entities:
        for column in left.table(entity):
            if column in ("census_block_id", "census_block") or column.endswith(
                "_geoid"
            ):
                pd.testing.assert_series_equal(
                    left.table(entity)[column],
                    right.table(entity)[column],
                    check_exact=True,
                )
    return {"keep_all_filters_added": len(lineage)}


def check_spm(run, projected, *, required):
    """Enrichment carries the preflight's SPM columns after hours, in order."""
    import pandas as pd

    from microcosm.build.spm_input_contract import ROLE_INPUT, UNIVERSE_INPUT
    from microcosm.build.us_runtime import graph_current_survey_spm as spm
    from microcosm.build.us_runtime import graph_us_survey_enrichment as graph
    from microcosm.build.us_runtime.prior_year_income import (
        US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS,
    )

    frame = run.population.frame
    for (entity, column), expected in projected.columns.items():
        pd.testing.assert_series_equal(
            frame.table(entity).set_index(entity + "_id")[column],
            expected,
            check_exact=True,
        )
    require(
        frame.person[ROLE_INPUT].dtype == bool
        and not frame.person[ROLE_INPUT].isna().any()
        and not frame.spm_unit[UNIVERSE_INPUT].isna().any(),
        "SPM_COLUMNS_INCOMPLETE",
    )
    hours = graph.hours_graph
    names = (
        hours.SOURCE_NODE,
        hours.ASEC_NODE,
        hours.ACS_NODE,
        hours.ATTACH_NODE,
        spm.SOURCE_NODE,
        spm.PROJECT_NODE,
        spm.ATTACH_NODE,
    )
    order = list(run.compiled.order)
    require(
        all(name in order for name in names)
        and order.index(graph.housing_graph.ATTACH_NODE)
        < order.index(hours.SOURCE_NODE)
        < order.index(hours.ATTACH_NODE)
        < order.index(spm.SOURCE_NODE)
        < order.index(spm.PROJECT_NODE)
        < order.index(spm.ATTACH_NODE),
        "SPM_HOURS_ORDER",
    )
    if required:
        require(all(run.manifest.node(name).hit for name in names), "SPM_HOURS_HITS")
    present = {name for entity in frame.entities for name in frame.table(entity)}
    require(
        not set(US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS) & present, "PRIOR_WAGES_PRESENT"
    )
    return {
        "spm_columns_equal_to_preflight": True,
        "hours_spm_nodes": len(names),
        "prior_year_outputs_absent": True,
        "spm_universe_counts": {
            str(key): int(value)
            for key, value in frame.spm_unit[UNIVERSE_INPUT]
            .value_counts(dropna=False)
            .items()
        },
        "spm_role_true_persons": int(frame.person[ROLE_INPUT].sum()),
    }


def handoff_summary(checkpoint):
    """Aggregate view of the development checkpoint report (names and counts)."""
    report = checkpoint.report
    inventory = report.get("input_inventory") or []
    statuses = {}
    for row in inventory:
        status = str(row.get("status"))
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "owner_live_verified": bool(checkpoint.owner_live_verified),
        "release_eligible": report.get("release_eligible"),
        "input_inventory_rows": len(inventory),
        "input_inventory_status_counts": dict(sorted(statuses.items())),
        "missing_input_names": sorted(
            str(row.get("name")) for row in report.get("missing_inputs") or []
        ),
        "excluded_native_scope_names": sorted(
            str(row.get("name")) for row in report.get("excluded_native_scope") or []
        ),
        "source_model_flags": report.get("source_model_flags"),
        "entity_counts": {
            entity: int(checkpoint.frame.n(entity))
            for entity in checkpoint.frame.entities
        },
    }


class CheckLedger:
    """Runner-level composition checks: record a failure and keep going.

    Maintained-runtime refusals (inside the graph hosts, owners and writers)
    always stop the run. The runner's own cross-stage checks are recorded here
    instead, so a single expensive run reports every stage it can reach.
    """

    def __init__(self, record):
        self.record = record
        self.passed = []
        self.failed = []

    @contextlib.contextmanager
    def check(self, name):
        try:
            yield
        except Exception as error:  # noqa: BLE001 - recorded, run continues
            self.failed.append({"check": name, **sanitized_error(error)})
        else:
            self.passed.append(name)
        finally:
            self.record(
                "checks", {"passed": list(self.passed), "failed": list(self.failed)}
            )


def compose(
    *,
    profile,
    sources,
    support,
    puf_main,
    puf_demographic,
    output_root,
    resume,
    phase,
    record,
    child_operation_scope=True,
    write_checkpoint=True,
):
    """Run financial -> PUF55 -> enrichment -> checkpoint over one live owner.

    Returns the live runs and the runner check ledger. Every maintained owner
    is requalified inside the outer source epoch, the epoch closes with its
    unconditional full source validation, and detached pure seals run last.
    """
    from microcosm.build.us_runtime import current_survey_spm_source as source
    from microcosm.build.us_runtime import graph_atomic_survey_financial as host
    from microcosm.build.us_runtime import graph_survey_puf55 as puf
    from microcosm.build.us_runtime import graph_us_survey_enrichment as enrichment
    from microcosm.build.us_runtime import native_survey_handoff as handoff
    from microcosm.build.us_runtime import survey_atomic_geography as geography
    from microcosm.build.us_runtime import survey_population_preparation as owner
    from microcosm.build.us_runtime import survey_population_replay as replay

    output_root = Path(output_root)
    required = resume == "require"
    store = output_root / "graph-store"
    ledger = CheckLedger(record)
    with phase("arguments"):
        financial_kwargs = financial_arguments(
            profile,
            host=host,
            geography=geography,
            sources=sources,
            support=support,
            store=store,
            snapshots=output_root / "source-snapshots",
            resume=resume,
        )
        donors = donor_sources(puf, main=puf_main, demographic=puf_demographic)
        policy = spm_policy(profile, source)
        puf_kwargs = puf_arguments(profile, donors=donors, resume=resume)
        enrichment_kwargs = enrichment_arguments(profile, policy, resume=resume)

    qualified_spm = projected_spm = public_spm = None
    digests = {}
    epoch_context = owner.verification_epoch()
    epoch = epoch_context.__enter__()
    try:
        with phase("financial"):
            financial = host.run_atomic_survey_financial(**financial_kwargs)
        with phase("financial-summary"):
            row = {"entity_counts": entity_counts(financial.financial_population)}
            state = host._run_entry(financial)[2]
            with ledger.check("financial_profile"):
                completion = state.completion_boundary
                require(
                    completion is not None
                    and state.development_boundary is not None
                    and completion.household_roles is True,
                    "FINANCIAL_PROFILE",
                )
            with ledger.check("financial_graph_counts"):
                base = state.completion_boundary.base
                row["prefix_graph"] = graph_counts(base.prefix)
                row["base_graph"] = graph_counts(
                    base, inherited=base.prefix.compiled.order
                )
                row["completion_graph"] = graph_counts(
                    financial, inherited=base.compiled.order
                )
                row["slowest_nodes"] = node_wall_times(financial)[:12]
            with ledger.check("financial_prefix_counts"):
                row["prefix_entity_counts"] = {
                    name: entity_counts(getattr(financial.prefix, name))
                    for name in (
                        "observed_population",
                        "allocated_population",
                        "expanded_population",
                        "clone_population",
                    )
                }
            with ledger.check("financial_inheritance"):
                base = state.completion_boundary.base
                assert_inherited(base.prefix, base)
                assert_inherited(base, financial)
            with ledger.check("property_tax_gate"):
                tax = json.loads(state.tax_verification)
                row["property_tax_gate"] = {
                    name: tax.get(name)
                    for name in (
                        "complete",
                        "numeric_verified",
                        "rows",
                        "unknown_interest_persons",
                        "unknown_dividend_persons",
                    )
                }
                require(
                    tax.get("complete") is True and tax.get("numeric_verified") is True,
                    "PROPERTY_TAX_GATE_INCOMPLETE",
                )
            record("financial", row)
        with phase("spm-source-preflight"):
            with ledger.check("spm_source_preflight"):
                qualified_spm, projected_spm, spm_counts = spm_source_preflight(
                    financial, policy
                )
                record("spm_preflight", spm_counts)

        operation = (
            host.child_verification_operation(financial)
            if child_operation_scope
            else contextlib.nullcontext()
        )
        with operation:
            with phase("puf"):
                puf_run = puf.run_survey_puf55(financial, **puf_kwargs)
            with phase("puf-summary"):
                row = {"entity_counts": entity_counts(puf_run.population)}
                with ledger.check("puf_inheritance"):
                    require(puf_run.financial_run is financial, "PUF_PARENT")
                    assert_inherited(financial, puf_run)
                    check_new_nodes(
                        puf_run, inherited=financial.compiled.order, required=required
                    )
                with ledger.check("puf_graph_counts"):
                    row["graph"] = graph_counts(
                        puf_run, inherited=financial.compiled.order
                    )
                    row["slowest_new_nodes"] = node_wall_times(
                        puf_run, exclude=financial.compiled.order
                    )[:12]
                with ledger.check("puf_packaged_routes"):
                    boundary = puf._run_entry(puf_run)[2].boundary
                    row["routes"] = [route.profile.value for route in boundary.routes]
                    require(
                        boundary.donor_kernel._definition.route == "packaged",
                        "PUF_PACKAGED_ROUTES",
                    )
                record("puf", row)
            with phase("enrichment"):
                enriched = enrichment.run_us_survey_enrichment(
                    puf_run, **enrichment_kwargs
                )
            with phase("enrichment-summary"):
                row = {"entity_counts": entity_counts(enriched.population)}
                declared = set()
                with ledger.check("enrichment_inheritance"):
                    require(enriched.parent_run is puf_run, "ENRICHMENT_PARENT")
                    assert_inherited(puf_run, enriched)
                    check_new_nodes(
                        enriched, inherited=puf_run.compiled.order, required=required
                    )
                with ledger.check("enrichment_declared_roster"):
                    declared = {
                        node.id for node in enrichment._ISSUED[id(enriched)][1].nodes
                    }
                    row["declared_nodes"] = len(declared)
                    require(
                        set(enriched.compiled.order)
                        == set(puf_run.compiled.order) | declared,
                        "ENRICHMENT_DECLARED_ROSTER",
                    )
                with ledger.check("enrichment_graph_counts"):
                    row["graph"] = graph_counts(
                        enriched, inherited=puf_run.compiled.order
                    )
                    row["slowest_new_nodes"] = node_wall_times(
                        enriched, exclude=puf_run.compiled.order
                    )[:12]
                with ledger.check("enrichment_axes_weights_mass"):
                    row["axes_weights"] = conserved_axes_weights(
                        puf_run.population,
                        enriched.population,
                        enriched.compiled,
                        declared,
                    )
                if projected_spm is not None:
                    with ledger.check("enrichment_spm_columns"):
                        row["spm"] = check_spm(
                            enriched, projected_spm, required=required
                        )
                record("enrichment", row)
            with phase("public-spm"):
                public_spm = source.qualify_native_spm_inputs(enriched, **policy)
                if qualified_spm is not None:
                    with ledger.check("public_spm_matches_preflight"):
                        require(
                            public_spm._receiving_run is enriched
                            and public_spm.receipt == qualified_spm.receipt,
                            "SPM_LIVE_SOURCE_OWNER",
                        )
            if write_checkpoint:
                with phase("development-checkpoint"):
                    checkpoint = handoff.write_native_survey_development_checkpoint(
                        enriched, output_root / "development-input"
                    )
                    with ledger.check("checkpoint_readback"):
                        require(
                            checkpoint.owner_live_verified is True, "HANDOFF_NOT_LIVE"
                        )
                        replay.same_replayed_frame(
                            enriched.population.frame, checkpoint.frame
                        )
                    with ledger.check("checkpoint_spm_provenance"):
                        spm_receipt = checkpoint.report["owner_receipt"]["spm"]
                        require(
                            spm_receipt["enabled"] is True
                            and all(
                                spm_receipt[name]
                                for name in (
                                    "source_receipt_sha256",
                                    "source_sha256",
                                    "projection_sha256",
                                    "attachment_sha256",
                                )
                            ),
                            "HANDOFF_SPM_PROVENANCE",
                        )
                    with ledger.check("checkpoint_summary"):
                        record("development_checkpoint", handoff_summary(checkpoint))
            with phase("final-owner-requalification"):
                if qualified_spm is not None:
                    qualified_spm.validate()
                public_spm.validate()
                digests = {
                    "financial": financial.checked_view().digest,
                    "puf": hashlib.sha256(puf_run.checked_view().payload).hexdigest(),
                    "enrichment": enriched.checked_view().digest,
                }
        # The child verification operation closed above: it reconstructed each
        # participating child once more with reuse disabled.
    except BaseException:
        exception = sys.exc_info()
        with phase("failed-source-epoch-close"):
            suppressed = epoch_context.__exit__(*exception)
        if not suppressed:
            raise
    else:
        with phase("source-epoch-close"):
            epoch_context.__exit__(None, None, None)
    with phase("pure-seals"):
        for candidate in (qualified_spm, public_spm):
            if candidate is not None:
                source._check_retained_output(candidate, source._ISSUED.get(candidate))
        host._pure_run(financial, host._run_entry(financial))
        puf._pure_run(puf_run, puf._run_entry(puf_run))
        enrichment._pure_retained_run(enriched, enrichment._ISSUED.get(id(enriched)))
    record(
        "final",
        {
            "checked_view_digests": digests,
            "verification_epoch": {
                key: value
                for key, value in dict(epoch).items()
                if isinstance(value, (int, str, float, bool))
            },
            "runner_checks_passed": len(ledger.passed),
            "runner_checks_failed": len(ledger.failed),
            "release_eligible": False,
            "country_engine_used": False,
        },
    )
    return {
        "financial": financial,
        "puf": puf_run,
        "enrichment": enriched,
        "ledger": ledger,
    }


# --------------------------------------------------------------------------
# Guarded process
# --------------------------------------------------------------------------


class _ProcTaskInfo(ctypes.Structure):
    _fields_ = [
        ("pti_virtual_size", ctypes.c_uint64),
        ("pti_resident_size", ctypes.c_uint64),
        ("pti_total_user", ctypes.c_uint64),
        ("pti_total_system", ctypes.c_uint64),
        ("pti_threads_user", ctypes.c_uint64),
        ("pti_threads_system", ctypes.c_uint64),
        ("pti_policy", ctypes.c_int32),
        ("pti_faults", ctypes.c_int32),
        ("pti_pageins", ctypes.c_int32),
        ("pti_cow_faults", ctypes.c_int32),
        ("pti_messages_sent", ctypes.c_int32),
        ("pti_messages_received", ctypes.c_int32),
        ("pti_syscalls_mach", ctypes.c_int32),
        ("pti_syscalls_unix", ctypes.c_int32),
        ("pti_csw", ctypes.c_int32),
        ("pti_threadnum", ctypes.c_int32),
        ("pti_numrunning", ctypes.c_int32),
        ("pti_priority", ctypes.c_int32),
    ]


def current_rss():
    """Current resident bytes (macOS ``proc_pidinfo``; ``None`` elsewhere)."""
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    except OSError:
        return None
    info = _ProcTaskInfo()
    size = ctypes.sizeof(info)
    got = libproc.proc_pidinfo(
        os.getpid(), 4, ctypes.c_uint64(0), ctypes.byref(info), size
    )
    return int(info.pti_resident_size) if got == size else None


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return int(value if sys.platform == "darwin" else value * 1024)


def process_cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def other_native_runs(markers=NATIVE_RUN_MARKERS):
    """Other processes whose command line names a native graph run."""
    import psutil

    me = psutil.Process()
    mine = {me.pid, *(parent.pid for parent in me.parents())}
    found = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(process.info["cmdline"] or ())
        except (psutil.Error, TypeError):
            continue
        if process.info["pid"] in mine or "python" not in cmdline:
            continue
        hits = [marker for marker in markers if marker in cmdline]
        if hits:
            found.append({"pid": process.info["pid"], "markers": hits})
    return found


def admission(args):
    """Refuse before any import when the machine cannot host this run."""
    import psutil

    memory = psutil.virtual_memory()
    output_root = Path(args.output_root)
    probe = output_root
    while not probe.exists():
        probe = probe.parent
    disk = os.statvfs(probe)
    free_disk = disk.f_bavail * disk.f_frsize
    others = other_native_runs()
    result = {
        "available_memory_gib": round(memory.available / GIB, 2),
        "total_memory_gib": round(memory.total / GIB, 2),
        "free_disk_gib": round(free_disk / GIB, 2),
        "other_native_runs": others,
        "loadavg": list(os.getloadavg()),
        "refusals": [],
    }
    if memory.available < args.min_available_gib * GIB:
        result["refusals"].append("ADMISSION_AVAILABLE_MEMORY")
    if free_disk < args.min_free_disk_gib * GIB:
        result["refusals"].append("ADMISSION_FREE_DISK")
    if others and not args.allow_other_native_runs:
        result["refusals"].append("ADMISSION_OTHER_NATIVE_RUN")
    store = output_root / "graph-store"
    if args.resume == "auto" and store.exists() and not args.reuse_store:
        result["refusals"].append("ADMISSION_COLD_STORE_EXISTS")
    if args.resume == "require" and not store.is_dir():
        result["refusals"].append("ADMISSION_REQUIRED_STORE_MISSING")
    return result


class NoCountryEngine:
    """Meta-path finder refusing the country engine; the composition is engine-free."""

    blocked = 0

    def find_spec(self, fullname, path=None, target=None):
        if (
            fullname == "policyengine"
            or fullname.startswith("policyengine.")
            or fullname.startswith("policyengine_")
            or fullname.startswith("microcosm.build.uk_runtime")
        ):
            NoCountryEngine.blocked += 1
            raise ImportError("NATIVE_COMPOSITION_COUNTRY_ENGINE_IMPORT")
        return None


def install_process_guards():
    sys.meta_path.insert(0, NoCountryEngine())
    blocked = {"network": 0, "subprocess": 0, "events": []}

    def audit(event, args):
        if event.startswith("socket.") and event not in (
            "socket.gethostname",
            "socket.getservbyname",
        ):
            blocked["network"] += 1
            if len(blocked["events"]) < 20:
                blocked["events"].append(event)
            raise PermissionError("NATIVE_COMPOSITION_NETWORK_BLOCKED")
        if event in (
            "subprocess.Popen",
            "os.system",
            "os.posix_spawn",
            "os.exec",
            "os.fork",
            "os.forkpty",
            "pty.spawn",
        ):
            blocked["subprocess"] += 1
            if len(blocked["events"]) < 20:
                blocked["events"].append(event)
            raise PermissionError("NATIVE_COMPOSITION_SUBPROCESS_BLOCKED")

    sys.addaudithook(audit)
    return blocked


class Monitor:
    """Phase ledger, RSS series, stack-sample ledger and ceilings."""

    def __init__(self, args, receipt_path, base):
        self.args = args
        self.receipt_path = Path(receipt_path)
        self.base = base
        self.start = time.monotonic()
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.phases = []
        self.current = []
        self.records = {}
        self.rss_series = []
        self.ledger = {}
        self.low_memory_since = None
        self.status = "RUNNING"
        self.error = None
        self.system_available_min = None

    # -- recording ------------------------------------------------------
    @contextlib.contextmanager
    def phase(self, name):
        row = {
            "phase": name,
            "status": "RUNNING",
            "start_wall_s": round(time.monotonic() - self.start, 3),
            "start_cpu_s": round(process_cpu_seconds(), 3),
        }
        with self.lock:
            self.phases.append(row)
            self.current.append(name)
        try:
            yield row
        except BaseException as error:
            row["status"] = "FAILED"
            row["error"] = sanitized_error(error)
            raise
        else:
            row["status"] = "COMPLETED"
        finally:
            row["wall_s"] = round(
                time.monotonic() - self.start - row["start_wall_s"], 3
            )
            row["cpu_s"] = round(process_cpu_seconds() - row["start_cpu_s"], 3)
            row["rss_end_bytes"] = current_rss()
            row["peak_rss_bytes_so_far"] = peak_rss_bytes()
            with self.lock:
                self.current.remove(name)
            self.flush()

    def record(self, name, value):
        with self.lock:
            self.records[name] = value
        self.flush()

    # -- sampling -------------------------------------------------------
    def _sample_stacks(self, interval=0.25):
        previous_key, previous_cpu = (), 0.0
        main = threading.main_thread().ident
        while not self.done.wait(interval):
            cpu = time.process_time()
            frame = sys._current_frames().get(main)
            stack = []
            while frame is not None and len(stack) < 24:
                stack.append(
                    (Path(frame.f_code.co_filename).name, frame.f_code.co_name)
                )
                frame = frame.f_back
            del frame
            if previous_key:
                key = "|".join(f"{a}:{b}" for a, b in previous_key[:4])
                with self.lock:
                    row = self.ledger.setdefault(key, [0.0, 0])
                    row[0] += cpu - previous_cpu
                    row[1] += 1
            previous_key, previous_cpu = tuple(stack), cpu

    def _watch(self):
        import psutil

        last_flush = 0.0
        last_rss_sample = -1e9
        while not self.done.wait(2.0):
            now = time.monotonic() - self.start
            cpu = process_cpu_seconds()
            rss = current_rss()
            available = psutil.virtual_memory().available
            with self.lock:
                if (
                    self.system_available_min is None
                    or available < self.system_available_min
                ):
                    self.system_available_min = available
                if now - last_rss_sample >= self.args.rss_sample_seconds:
                    self.rss_series.append(
                        [round(now, 1), rss, round(cpu, 1), round(available / GIB, 2)]
                    )
                    last_rss_sample = now
            stop = None
            if peak_rss_bytes() > self.args.rss_ceiling_gib * GIB:
                stop = ("RSS_CEILING_REACHED", 137)
            elif cpu > self.args.cpu_ceiling_seconds:
                stop = ("CPU_CEILING_REACHED", 124)
            elif now > self.args.wall_ceiling_seconds:
                stop = ("WALL_CEILING_REACHED", 124)
            elif available < self.args.system_floor_gib * GIB:
                if self.low_memory_since is None:
                    self.low_memory_since = now
                elif now - self.low_memory_since >= self.args.system_floor_seconds:
                    stop = ("SYSTEM_MEMORY_FLOOR_REACHED", 138)
            else:
                self.low_memory_since = None
            if stop is not None:
                self.status = stop[0]
                self.error = {
                    "type": "Ceiling",
                    "message": stop[0],
                    "phases_open": list(self.current),
                }
                self.flush()
                os._exit(stop[1])
            if now - last_flush >= self.args.flush_seconds:
                self.flush()
                last_flush = now

    def start_threads(self):
        threading.Thread(target=self._sample_stacks, daemon=True, name="stacks").start()
        threading.Thread(target=self._watch, daemon=True, name="watch").start()

    # -- receipt --------------------------------------------------------
    def payload(self):
        with self.lock:
            ledger = sorted(self.ledger.items(), key=lambda kv: -kv[1][0])[:40]
            series = list(self.rss_series)
            rss_values = [row[1] for row in series if row[1] is not None]
            return {
                "protocol": PROTOCOL,
                "status": self.status,
                "error": self.error,
                "base": self.base,
                "process": {
                    "wall_seconds": round(time.monotonic() - self.start, 3),
                    "cpu_seconds": round(process_cpu_seconds(), 3),
                    "peak_rss_bytes": peak_rss_bytes(),
                    "peak_rss_gib": round(peak_rss_bytes() / GIB, 3),
                    "max_sampled_rss_bytes": max(rss_values) if rss_values else None,
                    "min_system_available_gib": None
                    if self.system_available_min is None
                    else round(self.system_available_min / GIB, 2),
                    "loadavg_now": list(os.getloadavg()),
                },
                "phases": [dict(row) for row in self.phases],
                "phases_open": list(self.current),
                "records": json.loads(json.dumps(self.records, default=str)),
                "sampled_cpu_ledger": [
                    {
                        "code_chain": key.split("|"),
                        "cpu_seconds": round(v[0], 2),
                        "samples": v[1],
                    }
                    for key, v in ledger
                ],
                "rss_series_columns": [
                    "seconds",
                    "resident_bytes",
                    "process_cpu_seconds",
                    "system_available_gib",
                ],
                "rss_series": series,
                "release_eligible": False,
            }

    def flush(self):
        payload = self.payload()
        tmp = self.receipt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, default=str) + "\n")
        os.replace(tmp, self.receipt_path)
        return payload


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sources", required=True, help="staged sources directory")
    parser.add_argument("--geography-support", required=True)
    parser.add_argument("--puf-main", required=True)
    parser.add_argument("--puf-demographic", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--receipt", required=True, help="aggregate receipt JSON")
    parser.add_argument("--resume", choices=("auto", "require"), default="auto")
    parser.add_argument(
        "--reuse-store",
        action="store_true",
        help="with --resume auto, continue over an existing store (not cold)",
    )
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--no-child-operation-scope", action="store_true")
    parser.add_argument("--min-available-gib", type=float, default=40.0)
    parser.add_argument("--min-free-disk-gib", type=float, default=20.0)
    parser.add_argument("--allow-other-native-runs", action="store_true")
    parser.add_argument("--rss-ceiling-gib", type=float, default=32.0)
    parser.add_argument("--cpu-ceiling-seconds", type=float, default=43200.0)
    parser.add_argument("--wall-ceiling-seconds", type=float, default=57600.0)
    parser.add_argument("--system-floor-gib", type=float, default=8.0)
    parser.add_argument("--system-floor-seconds", type=float, default=60.0)
    parser.add_argument("--rss-sample-seconds", type=float, default=30.0)
    parser.add_argument("--flush-seconds", type=float, default=30.0)
    parser.add_argument("--label", default="native-survey-composition")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run; without it only admission and option reconstruction run",
    )
    return parser.parse_args(argv)


def _git_head(tree):
    head = (tree / ".git").read_text().strip() if (tree / ".git").is_file() else None
    try:
        if head and head.startswith("gitdir:"):
            gitdir = Path(head.split(":", 1)[1].strip())
            ref = (gitdir / "HEAD").read_text().strip()
            if ref.startswith("ref:"):
                name = ref.split(":", 1)[1].strip()
                common = Path((gitdir / "commondir").read_text().strip())
                common = common if common.is_absolute() else gitdir / common
                for root in (gitdir, common):
                    if (root / name).is_file():
                        return (root / name).read_text().strip()
                packed = common / "packed-refs"
                if packed.is_file():
                    for line in packed.read_text().splitlines():
                        if line.endswith(" " + name):
                            return line.split()[0]
            return ref
    except OSError:
        return None
    return None


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    tree = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root).absolute()
    receipt = Path(args.receipt).absolute()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    gate = admission(args)
    base = {
        "label": args.label,
        "source_tree": str(tree),
        "source_head": _git_head(tree),
        "argv_flags": {
            key: value
            for key, value in vars(args).items()
            if key not in ("sources", "puf_main", "puf_demographic")
        },
        "inputs": {
            "sources": str(Path(args.sources).absolute()),
            "puf_main": str(Path(args.puf_main).absolute()),
            "puf_demographic": str(Path(args.puf_demographic).absolute()),
        },
        "profile": DEVELOPMENT_PROFILE,
        "admission": gate,
        "python": sys.version.split()[0],
        "thread_environment": THREAD_ENVIRONMENT,
        "scope": (
            "Development measurement of the native financial -> PUF55 -> "
            "enrichment composition. Not a build, calibration, certification or "
            "release; survey poverty is not computed or used."
        ),
    }
    if gate["refusals"]:
        payload = {"protocol": PROTOCOL, "status": "REFUSED_AT_ADMISSION", "base": base}
        receipt.write_text(json.dumps(payload, indent=1) + "\n")
        print(json.dumps({"status": payload["status"], "refusals": gate["refusals"]}))
        return 3

    os.environ.update(THREAD_ENVIRONMENT)
    os.environ["TZ"] = "UTC"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "tmp").mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(output_root / "tmp")
    hard = int(args.cpu_ceiling_seconds + 120)
    resource.setrlimit(resource.RLIMIT_CPU, (hard, hard))
    signal.alarm(int(args.wall_ceiling_seconds + 120))

    monitor = Monitor(args, receipt, base)
    blocked = install_process_guards()
    monitor.start_threads()
    status, code = "COMPLETED", 0
    try:
        with monitor.phase("imports"):
            import torch

            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            from microcosm.build.us_runtime import acs_housing_universe_source as acs

            stray = sorted(
                name
                for name, module in sys.modules.items()
                if name.split(".")[0] == "microcosm"
                and getattr(module, "__file__", None)
                and not str(Path(module.__file__).resolve()).startswith(
                    str(tree) + os.sep
                )
            )
            require(not stray, "MICROCOSM_MODULE_OUTSIDE_TREE")
            import spm_calculator.units as units

            assembler = sha256_file(units.__file__)
            base["spm_assembler"] = {
                "file": units.__file__,
                "sha256": assembler,
                "matches_runtime_pin": assembler == acs._SPM_ASSEMBLER_SHA256,
            }
            require(assembler == acs._SPM_ASSEMBLER_SHA256, "SPM_ASSEMBLER_PIN")
        if not args.execute:
            from microcosm.build.us_runtime import current_survey_spm_source as source
            from microcosm.build.us_runtime import graph_atomic_survey_financial as host
            from microcosm.build.us_runtime import graph_survey_puf55 as puf
            from microcosm.build.us_runtime import survey_atomic_geography as geo

            with monitor.phase("dry-arguments"):
                financial_arguments(
                    DEVELOPMENT_PROFILE,
                    host=host,
                    geography=geo,
                    sources=args.sources,
                    support=args.geography_support,
                    store=output_root / "graph-store",
                    snapshots=output_root / "source-snapshots",
                    resume=args.resume,
                )
                donor_sources(puf, main=args.puf_main, demographic=args.puf_demographic)
                enrichment_arguments(
                    DEVELOPMENT_PROFILE,
                    spm_policy(DEVELOPMENT_PROFILE, source),
                    resume=args.resume,
                )
            status = "DRY_RUN_ARGUMENTS_VERIFIED"
        else:
            result = compose(
                profile=DEVELOPMENT_PROFILE,
                sources=args.sources,
                support=args.geography_support,
                puf_main=args.puf_main,
                puf_demographic=args.puf_demographic,
                output_root=output_root,
                resume=args.resume,
                phase=monitor.phase,
                record=monitor.record,
                child_operation_scope=not args.no_child_operation_scope,
                write_checkpoint=not args.no_checkpoint,
            )
            if result["ledger"].failed:
                status = "COMPLETED_WITH_RUNNER_CHECK_FAILURES"
    except BaseException as error:  # noqa: BLE001 - recorded, then non-zero exit
        status, code = "REFUSED", 1
        monitor.error = sanitized_error(error)
        # Full traceback stays local, beside the store, never in the receipt.
        (output_root / "error-traceback.txt").write_text(traceback.format_exc())
    finally:
        monitor.status = status
        monitor.base["guards"] = {
            "country_engine_imports_blocked": NoCountryEngine.blocked,
            "network_events_blocked": blocked["network"],
            "subprocess_events_blocked": blocked["subprocess"],
            "blocked_event_names": list(blocked["events"]),
        }
        monitor.done.set()
        payload = monitor.flush()
    print(
        json.dumps(
            {
                "status": payload["status"],
                "error": payload["error"],
                "wall_seconds": payload["process"]["wall_seconds"],
                "cpu_seconds": payload["process"]["cpu_seconds"],
                "peak_rss_gib": payload["process"]["peak_rss_gib"],
            }
        )
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
