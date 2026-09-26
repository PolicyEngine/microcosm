"""Real tiny application plus full55 codec checks, never financial authority."""

import pytest

from microcosm.build.us_runtime import puf55_original_application as original


@pytest.mark.parametrize("seeds", [(31, 31), (True, 32), (31, False), (-1, 32)])
def test_original_application_seed_must_be_explicit_distinct_integer(seeds):
    with pytest.raises(ValueError, match="SEEDS"):
        original._seeds(*seeds)


@pytest.fixture(scope="module")
def real_chain(tmp_path_factory):
    """Reuse the core's actual small train/source/apply fixture, with a new draw seed."""
    import importlib.util
    import sys
    from dataclasses import replace
    from pathlib import Path

    from microcosm.graph import compile_graph, run_graph

    path = (
        Path(__file__).parents[2]
        / "microcosm-fit/tests/test_graph_legacy_apply_observed.py"
    )
    spec = importlib.util.spec_from_file_location("core_observed_component", path)
    fixture = importlib.util.module_from_spec(spec)
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, spec.name, fixture)
        spec.loader.exec_module(fixture)
        patch.setenv("POPULACE_FIT_N_JOBS", "1")
        patch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
        compiled, store, kernels, sources, fits, applies, *_ = fixture.setup(
            tmp_path_factory.mktemp("real-chain")
        )
        nodes = tuple(
            replace(n, params={**n.params, "seed": 73})
            if n.id.startswith("conditioned.")
            else n
            for n in compiled.graph.nodes
        )
        compiled = compile_graph(replace(compiled.graph, nodes=nodes))
        manifest = run_graph(compiled, sources=sources, store=store, kernels=kernels)
        yield fixture, compiled, store, kernels, sources, fits, applies, manifest


def _actual_parts(real_chain):
    from microcosm.graph import (
        ArtifactValue,
        Numeric,
        NumericScope,
        platform_fingerprint,
    )

    fixture, compiled, store, kernels, sources, fits, applies, manifest = real_chain
    by_id = {n.id: n for n in compiled.graph.nodes}

    def edge(node_id, name):
        record = manifest.node(node_id)
        output = next(o for o in by_id[node_id].artifact_outputs if o.name == name)
        key = record.opaque_artifacts[name]
        return ArtifactValue(
            store.load_bytes(key),
            output.type,
            key,
            record.key,
            NumericScope(Numeric.PLATFORM_BITWISE, platform=platform_fingerprint()),
        )

    matrix = edge("matrix", "matrix")
    steps = tuple(
        original.OriginalTargetArtifacts(
            edge(f.id, "model"),
            edge(f.id, "training_state"),
            edge(a.id, "raw_draw"),
            edge(a.id, "conditioning"),
            edge(a.id, "apply_state"),
        )
        for f, a in zip(fits, applies, strict=True)
    )
    fixed = tuple((name, edge("qualified", name)) for name in ("a", "b"))
    return dict(
        matrix=matrix,
        matrix_name="matrix",
        steps=steps,
        fixed_inputs=fixed,
        expected_fixed={name: (name, value.payload) for name, value in fixed},
        targets=("a", "b", "c"),
        predictors=("x", "indicator"),
        entity="household",
        clone_one_seed=31,
        original_application_seed=73,
    )


def test_real_models_different_draw_seed_fixed_prefix_and_required_replay(real_chain):
    import numpy as np

    from microcosm.graph import run_graph

    fixture, compiled, store, kernels, sources, _, applies, manifest = real_chain
    args = _actual_parts(real_chain)
    table, history = original._decode_chain(**args)
    np.testing.assert_array_equal(table.a, [3.0, 1.0, 2.0])
    np.testing.assert_array_equal(table.b.iloc[[0, 2]], [30.0, 10.0])
    np.testing.assert_array_equal(table.c.iloc[[0, 2]], table.b.iloc[[0, 2]] * 10)
    _verify_actual_draws_with_conditioned_prefix(args, table)
    assert history[0]["observed_rows"] == 3
    assert history[1]["observed_rows"] == 2
    assert history[2]["observed_rows"] == 0
    assert history[2]["draw_sha256"] == history[2]["conditioning_sha256"]
    assert all(
        row["fit_seed"] == 31 and row["application_seed"] == 73 for row in history
    )
    assert history[0]["draw_sha256"] != history[0]["conditioning_sha256"]
    replay = run_graph(
        compiled, sources=sources, store=store, kernels=kernels, resume="require"
    )
    assert all(replay.node(node).hit for node in compiled.order)
    for node in applies:
        for name in ("raw_draw", "conditioning", "apply_state"):
            assert fixture.blob(replay, store, node, name) == fixture.blob(
                manifest, store, node, name
            )


def _verify_actual_draws_with_conditioned_prefix(args, table):
    """Independently apply the actual fitted objects with the new draw stream."""
    import numpy as np
    import pandas as pd

    matrix = original.fixed_graph.parent.model_input.decode_recipient_matrix(
        args["matrix"].payload
    )
    prior = pd.DataFrame(index=matrix.features.index)
    state = None
    for target, step in zip(args["targets"], args["steps"], strict=True):
        fitted = original.qrf_target.LegacyQRFTargetArtifact.from_trusted_bytes(
            step.model.payload, expected_sha256=original.codec.sha(step.model.payload)
        )
        if state is None:
            _, draw_seed = np.random.SeedSequence(73).spawn(2)
            state = original.qrf.QRFChainState.from_dict(
                {
                    **fitted.training_state.to_dict(),
                    "recipient_index": None,
                    "draw_rng_state": np.random.default_rng(
                        draw_seed
                    ).bit_generator.state,
                }
            )
        result = original.qrf_target.apply_target(
            fitted, matrix.features, prior, state=state
        )
        assert (
            original.codec.encode_raw_target(
                result.raw_draw, target=target, index=matrix.features.index
            )
            == step.raw_draw.payload
        )
        packet = original.observed.decode_observed_matrix_apply_state(
            step.apply_state.payload
        )
        assert result.state.to_dict() == packet["application"]["state"]
        prior[target], state = table[target].to_numpy(), result.state


@pytest.mark.parametrize(
    "defect",
    (
        "raw",
        "merged",
        "model",
        "training",
        "prefix",
        "matrix",
        "observed",
        "producer",
        "scope",
        "legacy_state",
    ),
)
def test_actual_chain_refuses_mutated_envelopes_before_return(real_chain, defect):
    from dataclasses import replace

    from microcosm.graph import Numeric, NumericScope

    args = _actual_parts(real_chain)
    step = args["steps"][-1]
    if defect in ("raw", "merged", "model"):
        name = {"raw": "raw_draw", "merged": "conditioning", "model": "model"}[defect]
        step = replace(step, **{name: replace(getattr(step, name), payload=b"changed")})
    elif defect == "training":
        packet = original.codec.decode_json(step.training_state.payload)
        packet["state"]["model_config"]["seed"] = 73
        step = replace(
            step,
            training_state=replace(
                step.training_state, payload=original.codec.encode_json(packet)
            ),
        )
    elif defect == "prefix":
        packet = original.codec.decode_json(step.apply_state.payload)
        packet["application"]["prior_producer_keys"][0] = "e" * 64
        step = replace(
            step,
            apply_state=replace(
                step.apply_state, payload=original.codec.encode_json(packet)
            ),
        )
    elif defect == "matrix":
        args["matrix"] = replace(args["matrix"], producer_key="f" * 64)
    elif defect == "observed":
        name, value = args["fixed_inputs"][0]
        args["fixed_inputs"] = (
            (name, replace(value, payload=b"changed")),
            *args["fixed_inputs"][1:],
        )
    elif defect == "producer":
        step = replace(
            step, conditioning=replace(step.conditioning, producer_key="f" * 64)
        )
    elif defect == "scope":
        step = replace(
            step,
            raw_draw=replace(step.raw_draw, numerics=NumericScope(Numeric.BITWISE)),
        )
    else:
        packet = original.codec.decode_json(step.apply_state.payload)
        packet["schema_version"] = 1
        step = replace(
            step,
            apply_state=replace(
                step.apply_state, payload=original.codec.encode_json(packet)
            ),
        )
    args["steps"] = (*args["steps"][:-1], step)
    with pytest.raises(ValueError):
        original._decode_chain(**args)


def test_full55_declaration_reuses_existing_fits_with_distinct_arm_provenance():
    from test_us_puf55_observed_recipients import _invented_values

    qualified, _ = _invented_values(rules=original.values.DEVELOPMENT_RULES)
    routes = tuple(
        original.attachment.route_nodes(p, seed=31, n_estimators=2, zero_atol=0)
        for p in original.attachment.PROFILES
    )
    nodes = original.original_application_nodes(
        qualified,
        routes,
        population="receiving",
        clone_one_seed=31,
        original_application_seed=73,
    )
    assert len(nodes) == 110
    for position, route in enumerate(routes):
        group = nodes[55 * position : 55 * (position + 1)]
        assert tuple(n.params["target"] for n in group) == route.profile.targets
        for fit, apply in zip(route.fits, group, strict=True):
            assert apply.params["seed"] == 73 and fit.params["seed"] == 31
            assert (
                next(e for e in apply.artifact_inputs if e.name == "model").producer
                == fit.id
            )
            assert apply.id.startswith("survey_puf55.original.")
            assert (
                next(e for e in apply.artifact_inputs if e.name == "matrix").producer
                == original.fixed_graph.parent.recipient_node_ids(0)[1]
            )
            assert tuple(o.name for o in apply.artifact_outputs) == (
                "raw_draw",
                "conditioning",
                "apply_state",
            )
            for prior in (
                e for e in apply.artifact_inputs if e.name.startswith("prior_")
            ):
                assert prior.artifact == "conditioning"


@pytest.mark.parametrize("defect", ("arm", "fit_seed", "fit_target", "route_order"))
def test_full55_declaration_refuses_foreign_reused_routes(defect):
    from dataclasses import replace

    from test_us_puf55_observed_recipients import _invented_values

    qualified, _ = _invented_values(arm=1 if defect == "arm" else 0)
    routes = tuple(
        original.attachment.route_nodes(p, seed=31, n_estimators=2, zero_atol=0)
        for p in original.attachment.PROFILES
    )
    if defect in ("fit_seed", "fit_target"):
        first = routes[0].fits[0]
        first = replace(
            first,
            params={
                **first.params,
                "seed" if defect == "fit_seed" else "target": 999
                if defect == "fit_seed"
                else "alien",
            },
        )
        routes = (replace(routes[0], fits=(first, *routes[0].fits[1:])), *routes[1:])
    elif defect == "route_order":
        routes = routes[::-1]
    with pytest.raises(ValueError):
        original.original_application_nodes(
            qualified,
            routes,
            population="receiving",
            clone_one_seed=31,
            original_application_seed=73,
        )


def _codec_only_full55(real_chain):
    """Synthetic codec envelopes only, deliberately not 55 fitted models/authority."""
    import copy

    import numpy as np
    from test_us_puf55_observed_recipients import _invented_values

    from microcosm.graph import (
        ArtifactValue,
        Numeric,
        NumericScope,
        platform_fingerprint,
    )
    from microcosm.graph.keys import opaque_artifact_key

    qualified, _ = _invented_values(rules=original.values.DEVELOPMENT_RULES)
    template = _actual_parts(real_chain)["steps"][0]
    training_template = original.codec.decode_json(template.training_state.payload)
    state_template = original.codec.decode_json(template.apply_state.payload)
    routes = []

    def key(text):
        return original.codec.sha(text.encode())

    def edge(payload, type_, producer, name):
        return ArtifactValue(
            payload,
            type_,
            opaque_artifact_key(producer, name),
            producer,
            NumericScope(Numeric.PLATFORM_BITWISE, platform=platform_fingerprint()),
        )

    for profile, (_, matrix_bytes) in zip(
        original.attachment.PROFILES, qualified.recipients.matrices, strict=True
    ):
        matrix_key = key("synthetic-matrix-" + profile.value)
        fixed_key = key("synthetic-fixed")
        matrix = original.fixed_graph.parent.model_input.decode_recipient_matrix(
            matrix_bytes
        )
        names = original.fixed_graph.observed_artifact_names(qualified, profile.value)
        payloads = original.values.observed_target_artifacts(
            qualified,
            profile=profile.value,
            matrix_payload=matrix_bytes,
            matrix_producer_key=matrix_key,
        )
        fixed = {
            target: edge(
                payload,
                original.observed.OBSERVED_TARGET_TYPE,
                fixed_key,
                names[target],
            )
            for target, payload in payloads.items()
        }
        models, history, prior_keys, steps = [], [], [], []
        for i, target in enumerate(profile.targets):
            model = ("synthetic-not-a-model-" + profile.value + target).encode()
            model_key, apply_key = (
                key(profile.value + target + "fit"),
                key(profile.value + target + "apply"),
            )
            models.append(
                dict(
                    target=target,
                    sha256=original.codec.sha(model),
                    training_id=key(target + "training"),
                )
            )
            training = copy.deepcopy(training_template)
            training["state"].update(
                targets=list(profile.targets),
                predictors=list(profile.predictors),
                entity="tax_unit",
                completed_targets=list(profile.targets[: i + 1]),
            )
            training["models"] = copy.deepcopy(models)
            raw = np.arange(len(matrix.features), dtype="float64") + i + 0.25
            merged = raw.copy()
            supplied = fixed.get(target)
            if supplied is not None:
                metadata, values, known = original.observed._read_observed(
                    supplied.payload,
                    target=target,
                    index=matrix.features.index,
                    matrix_sha256=original.codec.sha(matrix_bytes),
                    matrix_producer_key=matrix_key,
                )
                merged[known] = values[known]
            else:
                metadata, known = {}, np.zeros(len(raw), dtype=bool)
            raw_bytes = original.codec.encode_raw_target(
                raw, target=target, index=matrix.features.index
            )
            merged_bytes = original.codec.encode_raw_target(
                merged, target=target, index=matrix.features.index
            )
            history.append(
                dict(
                    target=target,
                    draw_sha256=original.codec.sha(raw_bytes),
                    conditioning_sha256=original.codec.sha(merged_bytes),
                    model_producer_key=model_key,
                    observed_sha256=None
                    if supplied is None
                    else original.codec.sha(supplied.payload),
                    observed_producer_key=None if supplied is None else fixed_key,
                    source_sha256=None
                    if supplied is None
                    else metadata["source_sha256"],
                    observed_rows=int(known.sum()),
                )
            )
            packet = copy.deepcopy(state_template)
            packet.update(
                matrix_sha256=original.codec.sha(matrix_bytes),
                matrix_producer_key=matrix_key,
            )
            application = packet["application"]
            application["state"].update(training["state"])
            application["state"]["recipient_index"] = original.qrf._index_identity(
                matrix.features.index
            ).to_dict()
            application.update(
                seed=73,
                models=copy.deepcopy(models),
                raw_targets=copy.deepcopy(history),
                prior_producer_keys=list(prior_keys),
            )
            steps.append(
                original.OriginalTargetArtifacts(
                    edge(model, original.LEGACY_QRF_TARGET_TYPE, model_key, "model"),
                    edge(
                        original.codec.encode_json(training),
                        original.codec.TRAINING_STATE_TYPE,
                        model_key,
                        "training_state",
                    ),
                    edge(
                        raw_bytes, original.codec.RAW_TARGET_TYPE, apply_key, "raw_draw"
                    ),
                    edge(
                        merged_bytes,
                        original.observed.CONDITIONING_TARGET_TYPE,
                        apply_key,
                        "conditioning",
                    ),
                    edge(
                        original.codec.encode_json(packet),
                        original.observed.OBSERVED_MATRIX_APPLY_STATE_TYPE,
                        apply_key,
                        "apply_state",
                    ),
                )
            )
            prior_keys.append(apply_key)
        routes.append(
            original.OriginalRouteArtifacts(
                profile,
                edge(
                    matrix_bytes,
                    original.fixed_graph.parent.model_input.RECIPIENT_MATRIX_TYPE,
                    matrix_key,
                    original.fixed_graph.parent._NAMES[profile.value],
                ),
                tuple(steps),
                tuple(fixed.items()),
            )
        )
    return qualified, tuple(routes)


def test_full55_codec_merge_exact_roster_values_axis_and_descriptive_receipt(
    real_chain,
):
    import numpy as np

    qualified, routes = _codec_only_full55(real_chain)
    table, receipt = original.merge_puf55_original_conditioning(
        qualified, routes, clone_one_seed=31, original_application_seed=73
    )
    assert tuple(table) == original.attachment.PROFILES[0].targets
    assert table.index.tolist() == [10, 20, 30, 40, 50]
    for target in qualified.tax_unit_values:
        known = qualified.tax_unit_known.loc[table.index, target]
        np.testing.assert_array_equal(
            table.loc[known, target],
            qualified.tax_unit_values.loc[table.index, target].loc[known],
        )
    report = original.codec.decode_json(receipt)
    assert report["fit_seed"] == 31 and report["original_application_seed"] == 73
    assert len(report["routes"]) == 2 and all(
        len(r["history"]) == 55 for r in report["routes"]
    )
    assert report["raw_draws_retained"]
    assert not any(
        report[k]
        for k in (
            "finalization_performed",
            "mixed_person_knownness_resolved",
            "source_admission_issued",
            "release_eligible",
        )
    )
    table.iloc[0, 0] = -12345
    assert qualified.tax_unit_values.iloc[0, 0] != -12345


@pytest.mark.parametrize(
    "defect",
    [
        "missing_target",
        "target_order",
        "route_order",
        "matrix",
        "fixed_roster",
        "fixed_producer",
        "observed_count",
        "fit_seed",
        "apply_seed",
        "conditioning",
        "history",
    ],
)
def test_full55_codec_merge_refuses_mismatched_packets(real_chain, defect):
    from dataclasses import replace

    from microcosm.graph.keys import opaque_artifact_key

    qualified, routes = _codec_only_full55(real_chain)
    first = routes[0]
    step = first.steps[-1]
    if defect == "missing_target":
        first = replace(first, steps=first.steps[:-1])
    elif defect == "target_order":
        first = replace(
            first, steps=(*first.steps[:-2], first.steps[-1], first.steps[-2])
        )
    elif defect == "route_order":
        routes = routes[::-1]
        first = routes[0]
    elif defect == "matrix":
        first = replace(first, matrix=routes[1].matrix)
    elif defect == "fixed_roster":
        first = replace(first, fixed_inputs=first.fixed_inputs[:-1])
    elif defect == "fixed_producer":
        target, item = first.fixed_inputs[0]
        item = replace(
            item,
            producer_key="a" * 64,
            key=opaque_artifact_key(
                "a" * 64,
                original.fixed_graph.observed_artifact_names(
                    qualified, first.profile.value
                )[target],
            ),
        )
        first = replace(first, fixed_inputs=((target, item), *first.fixed_inputs[1:]))
    elif defect == "conditioning":
        step = replace(
            step,
            conditioning=replace(
                step.conditioning, payload=step.raw_draw.payload + b"!"
            ),
        )
    else:
        packet = original.codec.decode_json(step.apply_state.payload)
        if defect == "observed_count":
            packet["application"]["raw_targets"][-1]["observed_rows"] = 1
        elif defect == "fit_seed":
            packet["application"]["state"]["model_config"]["seed"] = 999
        elif defect == "apply_seed":
            packet["application"]["seed"] = 31
        else:
            packet["application"]["prior_producer_keys"][0] = "b" * 64
        step = replace(
            step,
            apply_state=replace(
                step.apply_state, payload=original.codec.encode_json(packet)
            ),
        )
    if defect in (
        "conditioning",
        "observed_count",
        "fit_seed",
        "apply_seed",
        "history",
    ):
        first = replace(first, steps=(*first.steps[:-1], step))
    with pytest.raises(ValueError):
        original.merge_puf55_original_conditioning(
            qualified,
            (first, *routes[1:]),
            clone_one_seed=31,
            original_application_seed=73,
        )


def test_full55_merger_refuses_mutation_of_prior_route_during_later_decode(
    real_chain, monkeypatch
):
    qualified, routes = _codec_only_full55(real_chain)
    decode = original._decode_chain
    previous = []

    def mutate(**kwargs):
        table, history = decode(**kwargs)
        if previous:
            previous[0].iloc[0, 0] += 1
        previous.append(table)
        return table, history

    monkeypatch.setattr(original, "_decode_chain", mutate)
    with pytest.raises(ValueError, match="FINAL_CONDITIONING_CHANGED"):
        original.merge_puf55_original_conditioning(
            qualified, routes, clone_one_seed=31, original_application_seed=73
        )


def test_original_application_module_keeps_protected_source_tripwires():
    import test_us_spine_blindness as rules

    name = "puf55_original_application.py"
    source = (rules._US_RUNTIME / name).read_text()
    assert name in rules._US_LAUNCH_GRAPH_RUNTIME_MODULES
    assert name not in rules._SOURCE_SPINE_PROVENANCE_OWNERS
    assert rules._source_spine_accesses(source)
    assert rules._non_owner_source_spine_accesses(name, source) == ()
    assert rules._non_owner_source_spine_accesses(
        name, 'def leak(frame): return frame["person_spine_source_id"]'
    )
