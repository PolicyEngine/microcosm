"""Actual invented original-source CREATE controls; no native data or fit.

The module fixture runs one cold and one required graph. Other cases borrow its
source paths, never its mutable outputs. Source-mutating cases own fresh files.
No financial fixture, source issuer imitation, engine or model pickle is used.
"""

import hashlib
import json
import sys
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import graph_puf55_canonical_donor as owner
from microcosm.graph import (
    ContentStore,
    Graph,
    KernelRegistry,
    Owned,
    Slice,
    compile_graph,
    run_graph,
)
from microcosm.graph.artifact_edges import descriptor
from microcosm.graph.kernel import KernelContext
from microcosm.graph.keys import opaque_artifact_key


def _original_sources(root, *, defect=None, full_finalization_support=False):
    """Encode complete invented deliveries with real pinned source definitions."""
    packaged = owner.raw.packaged_definition()
    rows = []
    for i in range(68):
        aggregate = i >= 64
        recid = 999996 + i - 64 if aggregate else 501000 + i
        row = {name: "0" for name in packaged.main.delivered_header}
        mars = 0 if aggregate else 1 + i % 4
        secondary = int(mars == 2)
        primary = int(not aggregate and i % 13 != 0)
        row.update(
            RECID=str(recid),
            FLPDYR="2014" if i % 7 == 0 else "2015",
            FLPDMO="12",
            MARS=str(mars),
            DSI="0",
            S006="0" if i == 0 else str(125 + i),
            XFPT=str(primary),
            XFST=str(secondary),
            XTOT=str(primary + secondary),
        )
        if aggregate:
            row["E19200"] = ""
        else:
            row.update(
                E00200=str(40000 + i),
                E00900=str(-300 if i % 5 == 0 else 2000 + i),
                E00600="120",
                E00650="30",
                E25940="700",
                E25980="500",
                E25920="100",
                E25960="50",
                E26110="20",
                E26170="200",
                E26190="300",
                E26160="10",
                E26180="20",
                E26100="5",
                E26270="1495",
                E00100=str(500 * i - 10000),
                E19200=str(1000 + i),
                E02400="50",
                E03230="100",
                E87530="300",
            )
            if full_finalization_support and int(row["S006"]) > 0:
                # Full-chain finalization needs positive weighted support for
                # its configured capital-gain distribution tail bound.
                row["E01100"] = str(100 + i)
        rows.append(row)
    if defect == "duplicate":
        rows[0]["RECID"] = rows[1]["RECID"]
    elif defect == "reordered":
        rows = [*reversed(rows[:64]), *rows[64:]]
    header = ",".join(packaged.main.delivered_header)
    if defect == "header":
        header = header.replace("RECID", "WRONG", 1)
    main = (
        header
        + "\n"
        + "\n".join(
            ",".join(row[name] for name in packaged.main.delivered_header)
            for row in rows
        )
        + "\n"
    ).encode()
    demo_row = dict(
        RECID="501000",
        AGEDP1="0",
        AGEDP2="0",
        AGEDP3="0",
        AGERANGE="2",
        EARNSPLIT="0",
        GENDER="1",
    )
    demo = (
        ",".join(packaged.demographic.delivered_header)
        + "\n"
        + ",".join(demo_row[name] for name in packaged.demographic.delivered_header)
        + "\n"
    ).encode()
    document = owner.raw.definition_document_json(packaged)
    document.update(route="test_fixture", authority="invented_fixture_nonauthority")
    for name, payload, count in (("main", main, 68), ("demographic", demo, 1)):
        document["sources"][name].update(
            bytes=len(payload),
            data_records=count,
            sha256=hashlib.sha256(payload).hexdigest(),
            git_blob_sha1=hashlib.sha1(
                b"blob " + str(len(payload)).encode() + b"\0" + payload
            ).hexdigest(),
        )
    definition = owner.raw.fixture_definition(document)
    root.mkdir()
    paths = {}
    for pin, payload in ((definition.main, main), (definition.demographic, demo)):
        path = root / (pin.name + ".csv")
        path.write_bytes(payload)
        paths[pin.source_name] = path
    return definition, paths, (main, demo)


@pytest.mark.parametrize(
    "case", ("default_unsupported", "supported", "positive_only_zero_weight")
)
def test_original_source_fixture_weighted_tail_support(tmp_path, case):
    target = "non_sch_d_capital_gains"
    support = owner.full.support
    configuration = support.puf_tax_detail_tail_bound_quantiles_identity()
    assert configuration == {target: 0.999}
    assert owner.source.DIRECT_MAPPINGS[target] == "E01100"
    assert all(target in profile.targets for profile in owner.numerical.PROFILES)

    baseline_definition, _, baseline_buffers = _original_sources(tmp_path / "default")
    definition, paths, buffers = _original_sources(
        tmp_path / "selected", full_finalization_support=case == "supported"
    )
    for pin, payload in zip(
        (definition.main, definition.demographic), buffers, strict=True
    ):
        assert paths[pin.source_name].read_bytes() == payload
        assert pin.bytes == len(payload)
        assert pin.sha256 == hashlib.sha256(payload).hexdigest()
    if case != "supported":
        # Explicit false keeps the default fixture's exact delivered bytes.
        assert buffers == baseline_buffers
        assert definition.canonical == baseline_definition.canonical
    assert buffers[1] == baseline_buffers[1]

    baseline = owner.source.decode_full_puf_source(
        *baseline_buffers, baseline_definition
    )
    decoded = owner.source.decode_full_puf_source(*buffers, definition)
    assert decoded.aggregate_tokens == baseline.aggregate_tokens
    assert decoded.status.keys() == baseline.status.keys()
    assert decoded.values.keys() == baseline.values.keys()
    for name in decoded.status:
        np.testing.assert_array_equal(decoded.status[name], baseline.status[name])
    for name in decoded.values:
        if name != "E01100":
            np.testing.assert_array_equal(decoded.values[name], baseline.values[name])
    np.testing.assert_array_equal(
        decoded.values["E01100"][~decoded.ordinary],
        baseline.values["E01100"][~baseline.ordinary],
    )

    observed, _, status = owner.source.observed_and_derived_return_columns(decoded)
    np.testing.assert_array_equal(status["RECID"], np.arange(501000, 501064))
    expected_weights = np.array([0, *range(126, 189)], dtype=np.float64) / 100
    weights = status["S006"].astype(np.float64) / 100
    np.testing.assert_array_equal(weights, expected_weights)
    values = observed[target].astype(np.float64, copy=True)
    expected_values = (
        np.array([0, *range(101, 164)], dtype=np.float64)
        if case == "supported"
        else np.zeros(64, dtype=np.float64)
    )
    np.testing.assert_array_equal(values, expected_values)
    np.testing.assert_array_equal(
        observed[target], decoded.values["E01100"][decoded.ordinary]
    )
    if case == "positive_only_zero_weight":
        # A detached negative numerical input; never alter source/donor bytes.
        values[0] = 100
        assert (values > 0).any()
        assert (weights[values > 0] == 0).all()
        np.testing.assert_array_equal(observed[target], np.zeros(64))
    if case == "supported":
        assert np.isfinite(values).all()
        assert ((values > 0) & (weights > 0)).sum() == 63
        cap = support._weighted_positive_donor_quantile(
            values, weights, configuration[target]
        )
        assert np.isfinite(cap) and 0 < cap <= values.max()
    else:
        with pytest.raises(
            ValueError, match=r"^PUF tail-bound donor has no positive donor support\.$"
        ):
            support._weighted_positive_donor_quantile(
                values, weights, configuration[target]
            )


def _context(kernel, paths):
    node = owner._node(kernel._definition, json.loads(kernel._state[2]))
    return KernelContext(
        node=node,
        tables={},
        weights={},
        strata=pd.Series([], dtype=object, name="stratum"),
        params=node.params,
        rng=np.random.default_rng(1),
        sources=dict(paths),
    )


@pytest.fixture(scope="module")
def canonical_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("puf55_canonical_source")
    definition, paths, buffers = _original_sources(root / "originals")
    kernel = owner.CanonicalPuf55DonorKernel(fixture_definition=definition)
    node = owner.canonical_puf55_donor_node(fixture_definition=definition)
    assert node == _context(kernel, paths).node
    compiled = compile_graph(Graph("us", kernel.source_refs, (node,)))
    store = ContentStore(root / "store", codecs=kernel.source_codecs)
    kernels = KernelRegistry()
    kernels.register(kernel)
    cold = run_graph(compiled, sources=paths, store=store, kernels=kernels)
    warm = run_graph(
        compiled, sources=paths, store=store, kernels=kernels, resume="require"
    )
    yield SimpleNamespace(
        definition=definition,
        paths=paths,
        buffers=buffers,
        kernel=kernel,
        node=node,
        compiled=compiled,
        cold=cold,
        warm=warm,
        store=store,
    )
    kernel._check_state(kernel._state)
    assert kernel._read_sources(tuple(paths.items()), kernel._state) == buffers


def _payloads(run, manifest):
    record = manifest.node(run.node.id)
    result = {}
    for output in run.node.artifact_outputs:
        assert record.typed_artifacts["outputs"][output.name] == descriptor(
            producer=run.node.id,
            artifact=output.name,
            type_=output.type,
            producer_key=record.key,
            capabilities=record.capabilities,
        )
        key = opaque_artifact_key(record.key, output.name)
        assert record.opaque_artifacts[output.name] == key
        result[output.name] = run.store.load_bytes(key)
    return result


def test_actual_create_and_required_replay_preserve_all_ordinary_sources(canonical_run):
    run = canonical_run
    assert tuple(run.compiled.order) == (run.node.id,)
    assert not run.cold.node(run.node.id).store_hit
    assert run.warm.node(run.node.id).store_hit
    assert run.cold.content_addressed == run.warm.content_addressed
    cold, warm = _payloads(run, run.cold), _payloads(run, run.warm)
    assert cold == warm
    original = owner.source.decode_full_puf_source(*run.buffers, run.definition)
    assert cold["full_return_source"] == owner.source.encode_full_puf_source(original)
    arrays, receipt = owner.envelope.decode_canonical_puf59(cold["canonical_donor"])
    np.testing.assert_array_equal(arrays["RECID"], np.arange(501000, 501064))
    assert receipt["rows"] == 64 and receipt["source_statistical_year"] == 2015
    assert arrays["weight"][0] == 0
    np.testing.assert_array_equal(
        arrays["weight"], original.status["S006"][original.ordinary] / 100
    )
    frame = run.cold.population(run.node.id)
    assert owner._frame_seal(frame) == owner._frame_seal(
        run.warm.population(run.node.id)
    )
    np.testing.assert_array_equal(frame.table("tax_unit").tax_unit_id, np.arange(1, 65))
    np.testing.assert_array_equal(frame.table("tax_unit").index, arrays["RECID"])
    report = json.loads(cold["donor_projection"])
    assert report["source_rows"] == 68 and report["ordinary_rows"] == 64
    assert report["zero_weight_rows"] == 1 and report["source_route"] == "test_fixture"
    for name in (
        "technical_person_rows_are_source_persons",
        "source_admission_issued",
        "population_admission_issued",
        "current_money_tax_unit_reconstruction",
        "release_eligible",
    ):
        assert report[name] is False


def test_both_predictor_slices_share_one_donor_cohort(canonical_run):
    run = canonical_run
    payload = _payloads(run, run.cold)["canonical_donor"]
    nine, _ = owner.projection.canonical_puf55_donor_from_artifact(
        payload, expected_artifact_sha256=owner._sha(payload)
    )
    eight, _ = owner.projection.canonical_puf55_donor_from_artifact(
        payload,
        expected_artifact_sha256=owner._sha(payload),
        profile=owner.full.PUF55_SURVEY_SS_NO_TOTAL,
    )
    pd.testing.assert_frame_equal(
        eight, nine.drop(columns=owner.full.SURVEY_SS_TOTAL_PREDICTOR)
    )
    frame = run.cold.population(run.node.id)
    for profile in (owner.full.PUF55_SURVEY_SS, owner.full.PUF55_SURVEY_SS_NO_TOTAL):
        declaration = Slice("tax_unit", profile.predictors)
        selected = frame.table(declaration.entity).loc[:, list(declaration.columns)]
        assert selected.shape == (64, len(profile.predictors))
        pd.testing.assert_frame_equal(selected, nine.loc[:, list(profile.predictors)])
        assert not any("prior" in name for name in selected)
    assert len(run.node.outputs) == 64
    assert len({ref.name for ref in run.kernel.source_refs}) == 2


@pytest.mark.parametrize("seed", [True, -1, 2**64, 1.5])
def test_seed_is_an_exact_uint64(canonical_run, seed):
    with pytest.raises(ValueError, match="PUF55_CANONICAL_SOURCE_SEED"):
        owner.canonical_puf55_donor_node(
            seed=seed, fixture_definition=canonical_run.definition
        )


@pytest.mark.parametrize("kind", ["packaged", "detached_pin"])
def test_caller_definition_cannot_issue_production_authority(canonical_run, kind):
    if kind == "packaged":
        definition = owner.raw.packaged_definition()
    else:
        original = canonical_run.definition
        definition = replace(
            original, main=replace(original.main, bytes=original.main.bytes + 1)
        )
    with pytest.raises(
        ValueError, match="PUF55_CANONICAL_SOURCE_(FIXTURE_AUTHORITY|DEFINITION_PINS)"
    ):
        owner.CanonicalPuf55DonorKernel(fixture_definition=definition)


@pytest.mark.parametrize("kind", ["params", "columns", "artifact", "sources"])
def test_exact_declaration_and_artifact_roster(canonical_run, kind):
    run = canonical_run
    context = _context(run.kernel, run.paths)
    if kind == "params":
        context = replace(context, params={**dict(context.params), "seed": 999})
    elif kind == "columns":
        context = replace(
            context,
            node=replace(
                context.node, outputs=(Owned("tax_unit", "wrong", "float64"),)
            ),
        )
    elif kind == "artifact":
        context = replace(
            context,
            node=replace(
                context.node, artifact_outputs=context.node.artifact_outputs[:-1]
            ),
        )
    else:
        context = replace(context, sources={})
    with pytest.raises(
        ValueError, match="PUF55_CANONICAL_SOURCE_DECLARATION_OR_CONTEXT"
    ):
        run.kernel.run(context)


def test_wrong_source_codec_cannot_replace_pinned_loader(canonical_run, monkeypatch):
    kernel = canonical_run.kernel
    codec = kernel._definition.main.codec
    with monkeypatch.context() as patch:
        patch.setitem(
            kernel.source_codecs._byte_loaders,
            codec,
            lambda *a, **k: canonical_run.buffers[0],
        )
        with pytest.raises(ValueError, match="PUF55_CANONICAL_SOURCE_CODEC_BINDING"):
            kernel.run(_context(kernel, canonical_run.paths))


def test_changed_source_bytes_refuse_before_construction(tmp_path):
    definition, paths, _ = _original_sources(tmp_path / "source_changed")
    kernel = owner.CanonicalPuf55DonorKernel(fixture_definition=definition)
    path = paths[definition.main.source_name]
    payload = path.read_bytes()
    path.write_bytes(payload.replace(b"40000", b"49999", 1))
    with pytest.raises(ValueError):
        kernel.run(_context(kernel, paths))


@pytest.mark.parametrize("defect", ["header", "duplicate"])
def test_pinned_bytes_still_require_actual_source_structure(tmp_path, defect):
    definition, paths, _ = _original_sources(tmp_path / defect, defect=defect)
    kernel = owner.CanonicalPuf55DonorKernel(fixture_definition=definition)
    with pytest.raises(ValueError):
        kernel.run(_context(kernel, paths))


def test_source_order_is_not_replaced_by_technical_identifiers(tmp_path):
    definition, paths, _ = _original_sources(tmp_path / "reordered", defect="reordered")
    kernel = owner.CanonicalPuf55DonorKernel(fixture_definition=definition)
    result = kernel.run(_context(kernel, paths))
    np.testing.assert_array_equal(
        result.frame.table("tax_unit").index, np.arange(501063, 500999, -1)
    )
    np.testing.assert_array_equal(
        result.frame.table("tax_unit").tax_unit_id, np.arange(1, 65)
    )
    assert result.frame.weights_for("tax_unit").values[-1] == 0


def test_implementation_identity_rechecks_interest_on_replay_path(canonical_run):
    kernel = canonical_run.kernel
    fired = False
    previous = sys.getprofile()
    target = owner.interest.puf_e19200_interest_components_asset_identity.__code__

    def profile(frame, event, arg):
        nonlocal fired
        if (
            event == "return"
            and frame.f_code is target
            and frame.f_back is not None
            and frame.f_back.f_code
            is owner.CanonicalPuf55DonorKernel.implementation_hash.__code__
        ):
            fired = True
            arg["asset_sha256"] = "0" * 64

    sys.setprofile(profile)
    try:
        with pytest.raises(
            ValueError, match="PUF55_CANONICAL_SOURCE_IMPLEMENTATION_RESOURCES"
        ):
            # run_graph invokes this even when run() will be a cache hit.
            kernel.implementation_hash()
    finally:
        sys.setprofile(previous)
    assert fired


def _at_last_resource_return(kernel, context, mutate):
    """A profiler exercises the intended last borrowed-I/O return boundary."""
    fired = False
    previous = sys.getprofile()
    target = owner.CanonicalPuf55DonorKernel._read_resources.__code__

    def profile(frame, event, arg):
        nonlocal fired
        if event == "return" and frame.f_code is target and frame.f_back is not None:
            caller = frame.f_back
            if (
                caller.f_code is owner.CanonicalPuf55DonorKernel.run.__code__
                and "result" in caller.f_locals
                and not fired
            ):
                fired = True
                mutate(caller.f_locals)

    sys.setprofile(profile)
    try:
        with pytest.raises(ValueError, match="PUF55_CANONICAL_SOURCE_"):
            kernel.run(context)
    finally:
        sys.setprofile(previous)
    assert fired


@pytest.mark.parametrize("kind", ["qbi", "growth", "bands"])
def test_late_recipe_or_band_mutation_refuses(canonical_run, monkeypatch, kind):
    run = canonical_run
    with monkeypatch.context() as patch:

        def mutate(_):
            if kind == "qbi":
                patch.setattr(owner.canonical.qbi, "PARAMETERS_SHA256", "0" * 64)
            elif kind == "growth":
                patch.setitem(
                    owner.canonical.growth._RECIPE, "sensitivity_cpi_factor", "9"
                )
            else:
                bands = owner.interest.US_PUF_E19200_AGI_BANDS
                patch.setattr(
                    owner.interest,
                    "US_PUF_E19200_AGI_BANDS",
                    (
                        replace(
                            bands[0],
                            investment_interest_amount=bands[
                                0
                            ].investment_interest_amount
                            + 1,
                        ),
                        *bands[1:],
                    ),
                )

        _at_last_resource_return(run.kernel, _context(run.kernel, run.paths), mutate)


@pytest.mark.parametrize(
    "kind", ["frame", "bytes", "mutable_bytes", "receipt", "paired_baseline"]
)
def test_late_detached_outputs_have_immutable_seals(canonical_run, kind):
    run = canonical_run

    def mutate(local):
        result = local["result"]
        if kind == "frame":
            result.frame.table("tax_unit").columns.name = "changed"
        elif kind == "bytes":
            result.artifacts["canonical_donor"] = b"wrong"
        elif kind == "mutable_bytes":
            result.artifacts["canonical_donor"] = bytearray(
                result.artifacts["canonical_donor"]
            )
        elif kind == "receipt":
            result.receipt["ordinary_rows"] = 1
        else:
            result.artifacts["canonical_donor"] = b"wrong"
            result.receipt["canonical_donor_sha256"] = owner._sha(b"wrong")

    _at_last_resource_return(run.kernel, _context(run.kernel, run.paths), mutate)


def test_late_source_path_substitution_refuses(canonical_run):
    run = canonical_run
    context = _context(run.kernel, run.paths)

    def mutate(_):
        context.sources[run.kernel._definition.main.source_name] = next(
            iter(run.paths.values())
        ).with_name("other.csv")

    _at_last_resource_return(run.kernel, context, mutate)


def test_late_function_replacement_refuses(canonical_run, monkeypatch):
    run = canonical_run
    with monkeypatch.context() as patch:

        def mutate(_):
            patch.setattr(
                owner.canonical.growth, "grow_puf_2015_to_2024", lambda *a, **k: None
            )

        _at_last_resource_return(run.kernel, _context(run.kernel, run.paths), mutate)


@pytest.mark.parametrize(
    "mutation",
    [
        "config",
        "defaults",
        "kwdefaults",
        "code",
        "closure",
        "alias",
        "dataclass",
        "array",
    ],
)
def test_live_marker_tracks_mutations_through_cyclic_class_closure(
    monkeypatch, mutation
):
    """A real class method closes over a mapping containing itself and aliases."""
    shared = [7]
    bindings = {"config": 11, "left": shared, "right": shared}

    @dataclass
    class Cyclic:
        value: int

        def method(self, value=3, *, factor=5):
            return bindings, value * factor

    def changed(self, value=3, *, factor=5):
        return bindings, value + factor

    method = Cyclic.method
    bindings.update(
        method=method,
        self=bindings,
        record=Cyclic(13),
        array=np.array([17.0], dtype=np.float64),
    )
    # _live walks real class methods, including dataclass-generated methods and
    # Python 3.14 class annotation closures, as it does in the production owners.
    module = SimpleNamespace(
        __name__=Cyclic.__module__, __file__=__file__, Cyclic=Cyclic
    )
    monkeypatch.setattr(owner, "_modules", lambda: (module,))
    baseline = owner._live()
    assert owner._live() == baseline
    if mutation == "config":
        bindings["config"] += 1
    elif mutation == "defaults":
        method.__defaults__ = (4,)
    elif mutation == "kwdefaults":
        method.__kwdefaults__["factor"] += 1
    elif mutation == "code":
        method.__code__ = changed.__code__
    elif mutation == "closure":
        replacement = dict(bindings, config=19)
        cell = method.__closure__[method.__code__.co_freevars.index("bindings")]
        cell.cell_contents = replacement
    elif mutation == "alias":
        # Equal leaf values in a new container must not hide changed aliasing.
        bindings["right"] = list(shared)
    elif mutation == "dataclass":
        bindings["record"].value += 1
    else:
        bindings["array"][0] += 1
    assert owner._live() != baseline


def test_live_marker_keeps_first_visit_depth_limit():
    cyclic = []
    current = cyclic
    for _ in range(16):
        child = []
        current.append(child)
        current = child
    current.append(cyclic)
    # The depth-17 back edge names its existing object; it does not expand it.
    assert owner._marker(cyclic) == owner._marker(cyclic)
    deep = 0
    for _ in range(17):
        deep = [deep]
    with pytest.raises(ValueError, match="PUF55_CANONICAL_SOURCE_LIVE_DEPTH"):
        owner._marker(deep)


def test_live_marker_distinguishes_empty_closure_cells():
    state = {}

    def cyclic():
        return state

    state["function"] = cyclic
    cell = cyclic.__closure__[0]
    populated = owner._marker(cyclic)
    del cell.cell_contents
    empty = owner._marker(cyclic)
    assert owner._marker(cyclic) == empty
    assert empty != populated
    cell.cell_contents = {}
    assert owner._marker(cyclic) != empty
    cell.cell_contents["function"] = cyclic
    assert owner._marker(cyclic) == populated


def test_live_marker_stable_when_normal_calls_retain_code_literals():
    def ordinary():
        def nested():
            return "nested ordinary result; retained outside its code object"

        return "ordinary result; retained outside its code object", nested

    code = ordinary.__code__
    baseline = owner._marker(ordinary)
    retained = [ordinary() for _ in range(16)]
    nested_results = [nested() for _, nested in retained]
    assert len(retained) == len(nested_results) == 16
    assert ordinary.__code__ is code
    assert owner._marker(ordinary) == baseline
    retained.clear()
    nested_results.clear()
    assert owner._marker(ordinary) == baseline


@pytest.mark.parametrize("growth_scheme", ("family_observed", "cpi_only"))
def test_live_marker_stable_after_retaining_actual_recipe(growth_scheme):
    # Exercise the exact function located by the startup diagnostic, using its
    # actual packaged recipe and band identity; no recipe or resource is mocked.
    recipe = owner._recipe
    code = recipe.__code__
    baseline = owner._marker(recipe)
    retained = [recipe(578, growth_scheme) for _ in range(8)]
    assert all(row == retained[0] for row in retained)
    assert all(row["growth_scheme"] == growth_scheme for row in retained)
    assert recipe.__code__ is code
    assert owner._marker(recipe) == baseline
    retained.clear()
    assert owner._marker(recipe) == baseline


@pytest.mark.parametrize(
    "mutation",
    ("equal_code_replacement", "different_code", "constant", "defaults", "kwdefaults"),
)
def test_live_marker_tracks_code_replacement_and_nested_mutable_defaults(mutation):
    def subject(values=None, *, options=None):
        return values, options, "original constant; independent of public bytecode"

    def replacement(values=None, *, options=None):
        return options, values, "different executable body"

    subject.__defaults__ = ([7],)
    subject.__kwdefaults__ = {"options": {"scale": 3}}
    baseline = owner._marker(subject)
    code = subject.__code__
    if mutation == "equal_code_replacement":
        subject.__code__ = code.replace()
        assert subject.__code__ is not code
        assert subject.__code__ == code
    elif mutation == "different_code":
        subject.__code__ = replacement.__code__
    elif mutation == "constant":
        subject.__code__ = code.replace(
            co_consts=tuple(
                "changed literal; bytecode remains identical"
                if type(value) is str
                else value
                for value in code.co_consts
            )
        )
        assert subject.__code__.co_code == code.co_code
        assert subject.__code__.co_consts != code.co_consts
    elif mutation == "defaults":
        subject.__defaults__[0].append(11)
    else:
        subject.__kwdefaults__["options"]["scale"] += 1
    assert owner._marker(subject) != baseline
