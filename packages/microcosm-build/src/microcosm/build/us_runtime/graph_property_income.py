"""Joint property-income draws and visible reconciliation on original persons.

This source-blind fragment consumes already qualified donor/recipient branches.
The country host owns source admission, original design-weight mapping, unknown
exclusions and later clone attachment. No source, tax treatment or release
authority follows from executing these numerical operations.
"""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import qrf
from microcosm.fit.graph_legacy_qrf import (
    legacy_qrf_apply_nodes,
    legacy_qrf_train_nodes,
)
from microcosm.fit.graph_legacy_train import LegacyQRFTrainKernel
from microcosm.fit.graph_signed_reconciliation import signed_reconciliation_node
from microcosm.frame import WeightKind
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    KernelBase,
    KernelResult,
    Node,
    Numeric,
    Owned,
    Slice,
    source_hash,
)
from microcosm.graph.keys import opaque_artifact_key

from .property_income_constants import (
    PROPERTY_COMPONENTS,
    PROPERTY_DRAW_COLUMNS,
    PROPERTY_REPORTED_TOTAL,
)

PROTOCOL = "microcosm.us.property-income-model.v1"
DRAW_SUMMARY_TYPE = ArtifactType("microcosm.us.property_income_joint_draws", 1)


def require(condition, reason):
    if not condition:
        raise ValueError("PROPERTY_INCOME_GRAPH_" + reason)


def _features(features):
    codec.names(features, "property predictors")
    require(
        PROPERTY_REPORTED_TOTAL in features
        and not set(features).intersection(
            (*PROPERTY_COMPONENTS, *PROPERTY_DRAW_COLUMNS)
        ),
        "FEATURE_ROSTER",
    )
    return features


def _draw_node(prefix, population, features, seed, phase):
    edges = []
    for i in range(len(PROPERTY_COMPONENTS)):
        producer = f"{prefix}.apply.{i:03d}"
        edges.extend(
            (
                ArtifactInput(f"raw_{i}", producer, "raw_draw", codec.RAW_TARGET_TYPE),
                ArtifactInput(
                    f"state_{i}", producer, "apply_state", codec.APPLY_STATE_TYPE
                ),
            )
        )
    edges.append(
        ArtifactInput(
            "training",
            f"{prefix}.fit.{len(PROPERTY_COMPONENTS) - 1:03d}",
            "training_state",
            codec.TRAINING_STATE_TYPE,
        )
    )
    return Node(
        prefix + ".draws",
        PropertyIncomeDrawColumnsKernel.ref,
        population=population,
        inputs=(Slice("person", _features(features)),),
        outputs=tuple(Owned("person", c, "float64") for c in PROPERTY_DRAW_COLUMNS),
        params={"prefix": prefix, "features": features, "seed": seed, "phase": phase},
        artifact_inputs=tuple(edges),
        artifact_outputs=(ArtifactOutput("summary", DRAW_SUMMARY_TYPE),),
        description="Materialize the complete joint draw on original recipient persons; retain model and ordered raw-draw history.",
    )


def property_income_nodes(
    prefix,
    *,
    donor_population,
    recipient_population,
    features,
    seed,
    n_estimators,
    scales,
    atol,
    rtol,
):
    """Declare four fits, four draws, draw placement and signed reconciliation.

    Donor and recipient populations must be separate original-person branches.
    The aggregate feature is a qualified ASEC reported total on the donor and
    adjusted ACS INTP on the recipient. Unknown anchors must be handled upstream.
    Scales and tolerances are required modeling parameters, with no defaults.
    """
    require(
        type(prefix) is str
        and bool(prefix)
        and type(seed) is int
        and seed >= 0
        and type(n_estimators) is int
        and n_estimators > 0
        and donor_population != recipient_population,
        "DECLARATION_PARAMETERS",
    )
    features = _features(features)
    phase = PROTOCOL + ":" + prefix
    fits = legacy_qrf_train_nodes(
        prefix + ".fit",
        population=donor_population,
        entity="person",
        predictors=features,
        targets=PROPERTY_COMPONENTS,
        seed=seed,
        n_estimators=n_estimators,
        zero_atol=0,
        phase=phase,
    )
    applies = legacy_qrf_apply_nodes(
        prefix + ".apply",
        population=recipient_population,
        fit_nodes=fits,
        seed=seed,
        phase=phase,
    )
    fits = tuple(replace(node, kernel=PropertyIncomeTrainKernel.ref) for node in fits)
    draws = _draw_node(prefix, recipient_population, features, seed, phase)
    reconcile = signed_reconciliation_node(
        prefix + ".reconcile",
        population=recipient_population,
        entity="person",
        anchor=PROPERTY_REPORTED_TOTAL,
        draws=PROPERTY_DRAW_COLUMNS,
        components=PROPERTY_COMPONENTS,
        nonnegative=(True, True, True, False),
        scales=scales,
        atol=atol,
        rtol=rtol,
        diagnostic_prefix=prefix + "_reconciliation",
    )
    return (*fits, *applies, draws, reconcile)


class PropertyIncomeTrainKernel(LegacyQRFTrainKernel):
    """The existing weighted QRF, restricted to the qualified property basis."""

    ref = "us.property_income.train@1"

    def implementation_hash(self):
        return codec.sha(
            codec.encode_json(
                {
                    "fragment": source_hash(sys.modules[__name__]),
                    "fit": LegacyQRFTrainKernel().implementation_hash(),
                    "components": PROPERTY_COMPONENTS,
                    "aggregate": PROPERTY_REPORTED_TOTAL,
                }
            )
        )

    def run(self, context):
        require(
            context.node.kernel == self.ref
            and context.params == context.node.params
            and tuple(context.params["targets"]) == PROPERTY_COMPONENTS,
            "TRAINING_DECLARATION",
        )
        _features(context.params["predictors"])
        require(
            set(context.tables) == {"person"}
            and context.weights["person"].kind is WeightKind.DESIGN,
            "ORIGINAL_DESIGN_WEIGHTS",
        )
        table = context.tables["person"]
        columns = (*context.params["predictors"], *PROPERTY_COMPONENTS)
        require(
            all(table[c].dtype == np.dtype("float64") for c in columns)
            and np.isfinite(table.loc[:, list(columns)].to_numpy()).all(),
            "KNOWN_DONOR_VALUES",
        )
        require(
            (table.loc[:, list(PROPERTY_COMPONENTS[:3])].to_numpy() >= 0).all()
            and np.array_equal(
                table.loc[:, list(PROPERTY_COMPONENTS)].sum(axis=1).to_numpy(),
                table[PROPERTY_REPORTED_TOTAL].to_numpy(),
            ),
            "RESOLVED_DONOR_COMPONENTS",
        )
        # Delegate the actual fit and its validation to the shared implementation.
        # The executable graph retains the stricter country kernel identity.
        return LegacyQRFTrainKernel().run(
            replace(
                context, node=replace(context.node, kernel=LegacyQRFTrainKernel.ref)
            ),
        )


class PropertyIncomeDrawColumnsKernel(KernelBase):
    ref = "us.property_income.draw_columns@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        dependencies=("numpy", "pandas"),
    )

    def implementation_hash(self):
        return source_hash(
            sys.modules[__name__],
            codec,
            qrf,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context):
        expected = _draw_node(
            population=context.node.population, **dict(context.params)
        )
        require(context.node.normative() == expected.normative(), "DRAW_DECLARATION")
        require(
            set(context.tables) == {"person"}
            and not context.sources
            and set(context.artifacts)
            == {edge.name for edge in expected.artifact_inputs},
            "DRAW_CONTEXT",
        )
        table = context.tables["person"]
        require(
            table.person_id.dtype == np.dtype("int64") and table.person_id.is_unique,
            "PERSON_IDENTITY",
        )
        require(
            all(
                table[c].dtype == np.dtype("float64")
                for c in context.params["features"]
            )
            and np.isfinite(
                table.loc[:, list(context.params["features"])].to_numpy()
            ).all(),
            "KNOWN_RECIPIENT_FEATURES",
        )
        edge_map = {e.name: e for e in expected.artifact_inputs}

        def payload(name, kind):
            value = codec.artifact(context, name, kind)
            require(
                value.key
                == opaque_artifact_key(value.producer_key, edge_map[name].artifact),
                "ARTIFACT_KEY",
            )
            return value.payload

        training, trained = codec.read_training(
            payload("training", codec.TRAINING_STATE_TYPE)
        )
        state = trained.to_dict()
        require(
            state["entity"] == "person"
            and tuple(state["predictors"]) == context.params["features"]
            and tuple(state["targets"]) == PROPERTY_COMPONENTS
            and tuple(state["completed_targets"]) == PROPERTY_COMPONENTS,
            "COMPLETE_TRAINING_HISTORY",
        )
        raw, columns = [], {}
        index = pd.Index(table.person_id.to_numpy(copy=True), name="person_id")
        for i, (target, column) in enumerate(
            zip(PROPERTY_COMPONENTS, PROPERTY_DRAW_COLUMNS, strict=True)
        ):
            raw_name, state_name = f"raw_{i}", f"state_{i}"
            require(
                context.artifacts[raw_name].producer_key
                == context.artifacts[state_name].producer_key,
                "DRAW_SIBLING_PRODUCER",
            )
            raw.append(payload(raw_name, codec.RAW_TARGET_TYPE))
            application, chain = codec.read_application(
                payload(state_name, codec.APPLY_STATE_TYPE)
            )
            require(
                chain.entity == "person"
                and tuple(chain.predictors) == context.params["features"]
                and tuple(chain.targets) == PROPERTY_COMPONENTS
                and tuple(chain.completed_targets) == PROPERTY_COMPONENTS[: i + 1]
                and chain.recipient_index == qrf._index_identity(table.index)
                and application["seed"] == context.params["seed"]
                and application["models"] == training["models"][: i + 1]
                and application["raw_targets"]
                == [
                    {"target": name, "sha256": codec.sha(value)}
                    for name, value in zip(
                        PROPERTY_COMPONENTS[: i + 1], raw, strict=True
                    )
                ],
                "ORDERED_JOINT_DRAW_HISTORY",
            )
            values = codec.read_raw_target(raw[-1], target=target, index=table.index)
            columns["person", column] = pd.Series(
                values.copy(), index=index, name=column
            )
        return KernelResult(
            columns=columns,
            artifacts={
                "summary": codec.encode_json(
                    {
                        "protocol": PROTOCOL,
                        "rows": len(table),
                        "components": PROPERTY_COMPONENTS,
                        "models": training["models"],
                        "raw_sha256": [codec.sha(value) for value in raw],
                        "allocation": "one_joint_draw_per_input_person",
                    }
                )
            },
            receipt={"protocol": PROTOCOL, "rows": len(table)},
        )
