"""Frozen helper verification functions, source 8f02ad5; no genuine I/O.

Extracted from runner.py and cached_runner.py. Only BOUND/ADDED names are supplied
by the integrated caller. Full frame and design checks retain original logic.
"""

import hashlib


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def compare_frames(left, right):
    import numpy as np
    import pandas as pd

    require(
        left.schema == right.schema and left.entities == right.entities, "FRAME_SCHEMA"
    )
    for entity in left.entities:
        pd.testing.assert_frame_equal(
            left.table(entity), right.table(entity), check_exact=True
        )
    pd.testing.assert_series_equal(left.strata, right.strata, check_exact=True)
    require(left.weighted_entities == right.weighted_entities, "WEIGHT_ENTITIES")
    for entity in left.weighted_entities:
        require(
            left.weights_for(entity).kind == right.weights_for(entity).kind,
            "WEIGHT_KIND",
        )
        require(
            np.array_equal(
                left.weights_for(entity).values.view("uint64"),
                right.weights_for(entity).values.view("uint64"),
            ),
            "WEIGHT_BYTES",
        )
    require(left.mass_log == right.mass_log, "FRAME_MASS_LOG")


def verify_clone(before, after):
    """Independent keyed audit; never call the direct clone as the oracle."""
    import numpy as np
    import pandas as pd

    from microcosm.build.us_runtime.support_provenance import (
        support_channel_column as channel,
    )
    from microcosm.build.us_runtime.support_provenance import (
        support_clone_index_column as role,
    )
    from microcosm.build.us_runtime.support_provenance import (
        support_source_id_column as source,
    )
    from microcosm.frame import WeightKind

    require(
        before.weighted_entities == after.weighted_entities == ("household",),
        "HOUSEHOLD_ONLY",
    )
    require(
        before.weights_for("household").kind
        is after.weights_for("household").kind
        is WeightKind.IMPORTANCE,
        "IMPORTANCE_KIND",
    )
    mappings = {}
    report = {}
    for entity in before.entities:
        b, a = before.table(entity), after.table(entity)
        idcol = before.schema.entity_id_column(entity)
        require(
            list(a.columns) == list(b.columns) and len(a) == 2 * len(b), "ENTITY_LAYOUT"
        )
        require(
            b[source(entity)].is_unique and b[role(entity)].eq(0).all(),
            "BASE_NATIVE_UNIQUE",
        )
        require(
            set(a[role(entity)]) == {0, 1}
            and not a.duplicated([source(entity), role(entity)]).any(),
            "PAIR_UNIQUE",
        )
        mappings[entity] = a.set_index(idcol)[[source(entity), role(entity)]]
        base = b.set_index(source(entity), drop=False).sort_index()
        for k in (0, 1):
            rows = (
                a.loc[a[role(entity)].eq(k)]
                .set_index(source(entity), drop=False)
                .sort_index()
            )
            require(rows.index.equals(base.index), "PAIR_COVERAGE")
            for col in b.columns:
                if col in (idcol, role(entity)) or (
                    entity == "person"
                    and col
                    in [
                        before.schema.membership_column(e)
                        for e in before.schema.group_entities
                    ]
                ):
                    continue
                pd.testing.assert_series_equal(base[col], rows[col], check_exact=True)
            if k == 0:
                pd.testing.assert_series_equal(
                    base[idcol], rows[idcol], check_exact=True
                )
        report[entity] = {
            "before_rows": len(b),
            "after_rows": len(a),
            "source_arms": {
                str(k): int(v) for k, v in b[channel(entity)].value_counts().items()
            },
            "keyed_columns_preserved": True,
        }
    person = after.table("person")
    bp = before.table("person").set_index(source("person"))
    for entity in before.schema.group_entities:
        membership = before.schema.membership_column(entity)
        group_sources = before.table(entity).set_index(
            before.schema.entity_id_column(entity)
        )[source(entity)]
        expected_source = (
            bp[membership]
            .map(group_sources)
            .reindex(person[source("person")])
            .to_numpy()
        )
        actual = mappings[entity].reindex(person[membership])
        require(
            np.array_equal(actual[source(entity)].to_numpy(), expected_source),
            "MEMBERSHIP_SOURCE",
        )
        require(
            np.array_equal(
                actual[role(entity)].to_numpy(), person[role("person")].to_numpy()
            ),
            "MEMBERSHIP_ROLE",
        )
    bs = pd.Series(
        before.strata.to_numpy(),
        index=before.table("person")[source("person")].to_numpy(),
    )
    require(
        np.array_equal(
            bs.reindex(person[source("person")]).to_numpy(), after.strata.to_numpy()
        ),
        "STRATA_LINEAGE",
    )
    b = before.table("household")
    a = after.table("household")
    weights = pd.DataFrame(
        {
            "source": a[source("household")],
            "role": a[role("household")],
            "weight": after.weights_for("household").values,
        }
    ).pivot(index="source", columns="role", values="weight")
    incoming = (
        pd.Series(before.weights_for("household").values, index=b[source("household")])
        .reindex(weights.index)
        .to_numpy()
    )
    require(
        np.array_equal(
            weights[0].to_numpy().view("uint64"), weights[1].to_numpy().view("uint64")
        ),
        "PAIR_SYMMETRY",
    )
    require(
        np.array_equal(
            (weights[0] + weights[1]).to_numpy().view("uint64"), incoming.view("uint64")
        ),
        "PAIR_SUM",
    )
    require(
        set(before.stratum_mass().index) == set(after.stratum_mass().index),
        "STRATUM_ROSTER",
    )
    require(
        np.allclose(before.stratum_mass(), after.stratum_mass(), rtol=1e-12, atol=0),
        "PERSON_MASS",
    )
    require(before.mass_log == after.mass_log, "CARRIED_FRAME_MASS_LOG")
    return {
        "entities": report,
        "pair_weight_bits_preserved": True,
        "all_memberships_and_strata_preserved": True,
        "person_mass_preserved": True,
        "prior_wage_stage": False,
        "release_eligible": False,
    }


def trace_anchors():
    from microcosm.graph.population import Population

    captured = {}

    def observe(frame, event, value):
        if (
            event == "return"
            and frame.f_code.co_filename.endswith("/microcosm/graph/population.py")
            and isinstance(value, Population)
        ):
            if value.version in (BOUND, ADDED[0]):
                captured[value.version] = {
                    e: w.copy() for e, w in value.design_weights.items()
                }

    return captured, observe


def verify_design(before, after, captured):
    import numpy as np

    from microcosm.build.us_runtime.support_provenance import support_source_id_column

    require(set(captured) == {BOUND, ADDED[0]}, "DESIGN_OBSERVATIONS_MISSING")
    source = captured[BOUND]
    cloned = captured[ADDED[0]]
    require(
        source.keys() == cloned.keys() and "household" in source,
        "DESIGN_ANCHOR_ENTITIES",
    )
    report = {}
    for entity, values in source.items():
        b = before.table(entity)
        a = after.table(entity)
        key = support_source_id_column(entity)
        lookup = {v: i for i, v in enumerate(b[key].to_numpy())}
        expected = values[[lookup[v] for v in a[key].to_numpy()]]
        require(
            np.array_equal(expected.view("uint64"), cloned[entity].view("uint64")),
            "DESIGN_LINEAGE_BITS",
        )
        total_before = float(values.sum())
        total_after = float(cloned[entity].sum())
        require(
            np.isclose(total_after, 2 * total_before, rtol=1e-12, atol=0),
            "DESIGN_TOTAL_DOUBLED",
        )
        report[entity] = {
            "before_rows": len(values),
            "after_rows": len(expected),
            "before_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            "after_sha256": hashlib.sha256(cloned[entity].tobytes()).hexdigest(),
            "source_anchor_copied_unhalved": True,
            "total_before": total_before,
            "total_after": total_after,
            "total_ratio": total_after / total_before,
            "future_calibration_caps_validated": False,
        }
    return report
