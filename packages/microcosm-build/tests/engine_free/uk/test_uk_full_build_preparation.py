"""Canonical CLI preparation authenticates real H5 checkpoint boundaries.

The checkpoint is the bound-spine stand-in (``uk_full_population_graph``'s
``spine_frame``): the dense line takes its clone count, sampling and
geography binding from the spine it binds (microcosm#931).
"""

import hashlib
import itertools
import json

import pytest

from microcosm.build.uk_runtime import full_build_cli as cli
from microcosm.build.uk_runtime import hmrc_uprating, spine_build
from microcosm.build.uk_runtime.atomic_area_support import SYSTEMS
from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    write_uk_national_frame,
)
from microcosm.graph import ContentStore, compile_graph, run_graph
from test_support.microcosm_build.uk_calibration_run import _bound_checkpoint
from test_support.microcosm_build.uk_full_build_cli import (
    geography_first_sidecar_fields,
    patch_support_register,
    synthetic_spine_geography,
)
from test_support.microcosm_build.uk_full_population_graph import spine_frame
from test_support.microcosm_build.uk_ladder_rowwise_clone import (
    toy_ladder as toy_ladder,
)


def _write_checkpoint(tmp_path, *, clones=1, sampling=None, posture="development"):
    """A bound checkpoint of the stand-in spine at ``clones`` geographic
    clones, with a geography-first sidecar recording its clone count, its
    root's ``sampling`` receipt (none at f100) and its release posture."""
    path = write_uk_national_frame(spine_frame(clones), tmp_path / "spine.h5")
    frame, _ = load_uk_national_frame(path)
    sidecar_path, gates_path, sidecar = _bound_checkpoint(tmp_path, frame)
    # A geography-first roster, which the dense line requires before it pins
    # any input (microcosm#931).
    sidecar.update(geography_first_sidecar_fields("frs_spine"))
    sidecar["geographic_clone_count"] = clones
    sidecar["sampling"] = sampling
    sidecar["release_posture"] = posture
    sidecar_path.write_text(json.dumps(sidecar))
    return path, sidecar_path, gates_path, sidecar


@pytest.fixture
def checkpoint_argv(tmp_path, toy_ladder, monkeypatch):
    """A dense request over a real bound checkpoint, as an argument list."""
    pytest.importorskip("tables")
    _, ladder_path = toy_ladder
    path, sidecar_path, gates_path, _ = _write_checkpoint(tmp_path)
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "facts.csv").write_text("fixture-only; unused during preparation")
    monkeypatch.setattr(spine_build, "_rules_engine", lambda: object())
    monkeypatch.setattr(
        spine_build, "_rules_engine_provenance", lambda: {"version": "fixture"}
    )
    # The engine stands in here, so its freeze check does too: the preparation
    # runs it before reading any input (microcosm#1095).
    monkeypatch.setattr(
        hmrc_uprating, "assert_installed_engine_is_pinned", lambda: "fixture"
    )
    argv = [
        "--release-role",
        "dense",
        "--input-h5",
        str(path),
        "--input-sidecar",
        str(sidecar_path),
        "--input-spine-gates",
        str(gates_path),
        "--ladder",
        str(ladder_path),
        "--ledger-facts",
        str(ledger),
        "--out",
        str(tmp_path / "out"),
        "--n-clones",
        "1",
        "--epochs",
        "8",
    ]
    return argv, sidecar_path, gates_path


@pytest.fixture
def checkpoint_request(checkpoint_argv):
    argv, sidecar_path, gates_path = checkpoint_argv
    return cli.parse_args(argv), sidecar_path, gates_path


def test_prepare_real_checkpoint_preserves_source_year_and_wires_preflight(
    checkpoint_request, tmp_path
):
    args, sidecar_path, _ = checkpoint_request
    sidecar = json.loads(sidecar_path.read_text())
    prepared = cli.prepare_full_build(args)
    assert prepared.full.config.source_year == 2023
    assert prepared.full.config.geography_levels is None
    # The spine's geography binding is the build's, its clone count beside
    # it, and no support is a source of the dense build's own (microcosm#931).
    assert prepared.bindings["geography"] == {
        **sidecar["geography"],
        "geographic_clone_count": 1,
    }
    assert prepared.bindings["spine"] == {
        "geographic_clone_count": 1,
        "sampling": {
            "fraction": 1.0,
            "seed": None,
            "sampled": False,
            "rung_token": "f100",
        },
        "release_posture": "development",
    }
    assert not any(name.endswith("_support") for name in prepared.sources)
    pool = prepared.full.graph.node("uk.full.pool")
    assert (pool.base, pool.artifact_inputs) == ("uk.full.spine_checkpoint", ())
    dense = prepared.full.graph.node("uk.full.dense")
    assert any(
        a.name == "preflight" and a.producer == "uk.full.gates.preflight"
        for a in dense.artifact_inputs
    )
    endpoint = cli._through(prepared.full.graph, "uk.full.spine_checkpoint")
    store = ContentStore(tmp_path / "store")
    first = run_graph(
        compile_graph(endpoint),
        sources=prepared.sources,
        store=store,
        kernels=prepared.kernels,
    )
    provenance_key = first.nodes["uk.full.spine_checkpoint"].opaque_artifacts[
        "spine_provenance"
    ]
    provenance = json.loads(store.load_bytes(provenance_key))
    assert provenance["stages"] == geography_first_sidecar_fields("frs_spine")["stages"]
    assert provenance["fit_weight_records"] == {"model": {"fit_weights_used": True}}
    # What the dense line takes from the spine it binds (microcosm#931).
    assert provenance["geographic_clone_count"] == 1
    assert provenance["geography"] == sidecar["geography"]
    assert provenance["sampling"] is None
    assert provenance["release_posture"] == "development"
    fresh = cli.prepare_full_build(args)
    replay = run_graph(
        compile_graph(cli._through(fresh.full.graph, "uk.full.spine_checkpoint")),
        sources=fresh.sources,
        store=store,
        kernels=fresh.kernels,
        resume="require",
    )
    assert replay.nodes["uk.full.spine_checkpoint"].hit
    assert not args.out.exists()


@pytest.mark.parametrize("damage", ["identity", "gate_bytes"])
def test_prepare_refuses_checkpoint_drift_before_registering_a_full_build(
    checkpoint_request, damage
):
    args, sidecar_path, gates_path = checkpoint_request
    if damage == "identity":
        value = json.loads(sidecar_path.read_text())
        value["uk_frame_content_identity"] = "f" * 64
        sidecar_path.write_text(json.dumps(value))
    else:
        gates_path.write_text(gates_path.read_text() + "\n")
    with pytest.raises(ValueError, match="identity mismatch|SHA-256 mismatch"):
        cli.prepare_full_build(args)
    assert not args.out.exists()


def _checkpoint_request(
    tmp_path, ladder_path, *extra, clones=1, sampling=None, posture="development"
):
    """A dense request over a fresh stand-in checkpoint at ``clones``, built
    in the release ``posture`` its sidecar records."""
    spine = tmp_path / "spine"
    spine.mkdir(parents=True)
    path, sidecar_path, gates_path, _ = _write_checkpoint(
        spine, clones=clones, sampling=sampling, posture=posture
    )
    ledger = tmp_path / "ledger"
    ledger.mkdir(exist_ok=True)
    return cli.parse_args(
        [
            "--release-role",
            "dense",
            "--input-h5",
            str(path),
            "--input-sidecar",
            str(sidecar_path),
            "--input-spine-gates",
            str(gates_path),
            "--ladder",
            str(ladder_path),
            "--ledger-facts",
            str(ledger),
            "--out",
            str(tmp_path / "out"),
            "--epochs",
            "8",
            *extra,
        ]
    )


@pytest.fixture
def stand_in_engine(monkeypatch):
    """The engine's provenance and its freeze stand in (microcosm#1095)."""
    monkeypatch.setattr(spine_build, "_rules_engine", lambda: object())
    monkeypatch.setattr(
        spine_build, "_rules_engine_provenance", lambda: {"version": "fixture"}
    )
    monkeypatch.setattr(
        hmrc_uprating, "assert_installed_engine_is_pinned", lambda: "fixture"
    )


@pytest.fixture
def spine_request(tmp_path, toy_ladder, stand_in_engine, monkeypatch):
    """Dense requests over an in-graph spine (``--spine-request``).

    The roster checks and the spine's own preparation stand in, since the
    packaged spec is not yet geography-first: the stand-in composes the
    declared spine at the request's ``--n-clones``, ``--sample-fraction`` and
    ``--sample-seed``, as the spine build does, with the geography block a
    synthetic request composes, over the toy conditioned supports under the
    synthetic register (microcosm#931), and records every request it parses
    in ``requests``. ``spine_request(spine_arguments, *extra)`` is a dense
    request whose spine request is ``spine_arguments`` on the synthetic
    fixture; ``extra`` are the dense request's own.
    """
    from types import SimpleNamespace

    from microcosm.build.uk_runtime import graph as uk_graph
    from microcosm.build.uk_runtime.country_adapter import (
        UK_SYNTHETIC_SUPPORT_REGISTER,
        uk_spine_geography_binding,
    )
    from test_support.microcosm_build.uk_conditioned_support_fixtures import (
        toy_conditioned_support_payloads,
    )

    monkeypatch.setattr(cli, "_require_geography_first_spine", lambda args: None)
    monkeypatch.setattr(cli, "require_geography_first_spine", lambda *a, **k: None)
    requests = []

    def composed(raw):
        requests.append(raw)
        graph = uk_graph.uk_spine_graph(
            source_mode="split",
            sample_fraction=raw.sample_fraction,
            sample_seed=raw.sample_seed,
            geographic_clones=raw.n_clones,
        )
        return SimpleNamespace(
            graph=graph,
            kernels=uk_graph.uk_registry(graph=graph),
            sources={},
            stages=(),
            stage_names=uk_graph.uk_spine_endpoint(graph).stage_names,
            engine_identity="fixture",
            frs_release=SimpleNamespace(time_period="2023"),
            geography=uk_spine_geography_binding(
                toy_conditioned_support_payloads(),
                register=UK_SYNTHETIC_SUPPORT_REGISTER,
            ),
        )

    monkeypatch.setattr(spine_build, "prepare_uk_spine_execution", composed)
    ledger = tmp_path / "ledger"
    ledger.mkdir(exist_ok=True)
    made = itertools.count()

    def dense(spine_arguments=(), *extra):
        path = tmp_path / f"request-{next(made)}.json"
        path.write_text(
            json.dumps(
                [
                    "--smoke",
                    "--synthetic-fixture-dir",
                    str(tmp_path),
                    *spine_arguments,
                ]
            )
        )
        return cli.parse_args(
            [
                "--release-role",
                "dense",
                "--spine-request",
                str(path),
                "--ladder",
                str(toy_ladder[1]),
                "--ledger-facts",
                str(ledger),
                "--out",
                str(tmp_path / "out"),
                "--epochs",
                "8",
                *extra,
            ]
        )

    dense.requests = requests
    return dense


#: The two routes to the spine the dense line binds: a bound checkpoint
#: (``--input-h5``) and an in-graph spine request (``--spine-request``).
ROUTES = ("input_h5", "spine_request")


def _dense_args(
    route, request, directory, *extra, clones=1, sampling=None, posture="development"
):
    """A dense request on ``route`` over a spine of ``clones`` geographic
    clones, sampled as ``sampling`` says (none at f100): a fresh stand-in
    checkpoint under ``directory`` built in the release ``posture``, or a
    spine request naming them (a synthetic smoke request, so never a release
    candidate's posture)."""
    if route == "input_h5":
        pytest.importorskip("tables")
        ladder = request.getfixturevalue("toy_ladder")[1]
        return _checkpoint_request(
            directory, ladder, *extra, clones=clones, sampling=sampling, posture=posture
        )
    spine = ["--n-clones", str(clones)]
    if sampling is not None:
        spine += [
            "--sample-fraction",
            str(sampling["fraction"]),
            "--sample-seed",
            str(sampling["seed"]),
        ]
    return request.getfixturevalue("spine_request")(spine, *extra)


def _attempt(monkeypatch, *, rung="f100"):
    """A dense attempt's evidence, as ``_dense_attempt`` starts it, with its
    build id minted at ``rung`` and the code pin standing in."""
    from microcosm.build.logbook_adoption import AttemptState

    monkeypatch.setattr(cli, "git_code_pin", lambda repository: "fixture-code-pin")
    return {
        "state": AttemptState(
            build_id=f"uk-local-candidate-{rung}-s42-fixture",
            identity_digest="unset",
            input_pins_digest="unset",
            phases_reached=["attempt_started"],
            gate_verdicts={},
        )
    }


def _digest(prepared, args) -> str:
    """The candidate identity digest of ``args`` as they stand."""
    from microcosm.build.uk_runtime.rowwise_cli import candidate_identity_digest

    return candidate_identity_digest(
        pins=prepared.pins, args=args, source_year=prepared.full.config.source_year
    )


@pytest.mark.parametrize("route", ROUTES)
def test_a_spine_gives_the_run_its_clone_count(route, request, tmp_path, monkeypatch):
    """K is the spine's (microcosm#931): with no --n-clones, a spine of seven
    clones, bound or requested, makes the run's K seven, which its recorded
    parameters, its identity digest (taken once K is bound), its
    configuration and its request bindings carry."""
    from microcosm.build.uk_runtime.rowwise_cli import rowwise_parameters

    args = _dense_args(route, request, tmp_path, clones=7)
    assert args.n_clones is None
    attempt = _attempt(monkeypatch)
    prepared = cli.prepare_full_build(args, attempt=attempt)
    assert args.n_clones == 7
    assert rowwise_parameters(args, source_year=2023)["n_clones"] == 7
    assert attempt["state"].identity_digest == _digest(prepared, args)
    assert prepared.full.config.n_clones == 7
    assert prepared.bindings["spine"]["geographic_clone_count"] == 7
    assert prepared.bindings["geography"]["geographic_clone_count"] == 7
    assert prepared.bindings["configuration"]["n_clones"] == 7


@pytest.mark.parametrize("route", ROUTES)
def test_an_asserted_clone_count_other_than_the_spines_is_refused(
    route, request, tmp_path
):
    """--n-clones only asserts the spine's K: one that differs is refused,
    naming both, and on a spine request pointing at the request's own
    --n-clones; one that agrees passes (microcosm#931)."""
    args = _dense_args(route, request, tmp_path, "--n-clones", "3", clones=7)
    with pytest.raises(ValueError) as refused:
        cli.prepare_full_build(args)
    assert (
        str(refused.value)
        == {
            "input_h5": (
                "--n-clones 3 differs from the bound spine's geographic clone count "
                "K = 7: the dense line takes K from its spine (microcosm#931)."
            ),
            "spine_request": (
                "--n-clones 3 differs from the spine request's --n-clones K = 7: "
                "the dense line takes K from its spine; set the clone count in the "
                "spine request (microcosm#931)."
            ),
        }[route]
    )
    agreeing = _dense_args(
        route, request, tmp_path / "again", "--n-clones", "7", clones=7
    )
    assert cli.prepare_full_build(agreeing).full.config.n_clones == 7


@pytest.mark.parametrize("route", ROUTES)
def test_engine_blocks_are_held_to_the_spines_clone_count(route, request, tmp_path):
    """With K from the spine, preparation holds the engine blocks to one or K
    once it binds the spine, naming K and not the operator's count
    (microcosm#931), and the measures node takes K beside them."""
    with pytest.raises(ValueError) as refused:
        cli.prepare_full_build(
            _dense_args(route, request, tmp_path, "--engine-blocks", "3")
        )
    assert str(refused.value) == (
        "--engine-blocks is neither one nor the spine's geographic clone count "
        "K = 1: the engine resolves the pool in one block or one block per clone "
        "(microcosm#931)."
    )
    blocks = _dense_args(
        route, request, tmp_path / "blocks", "--engine-blocks", "3", clones=3
    )
    prepared = cli.prepare_full_build(blocks)
    assert prepared.full.config.engine_blocks == 3
    measures = prepared.full.graph.node("uk.full.measures")
    assert (measures.params["engine_blocks"], measures.params["geographic_clones"]) == (
        3,
        3,
    )
    one = cli.prepare_full_build(
        _dense_args(route, request, tmp_path / "one", clones=3)
    ).full.graph.node("uk.full.measures")
    assert (one.params["engine_blocks"], one.params["geographic_clones"]) == (1, 3)


@pytest.mark.parametrize(
    "own",
    [(), ("--n-clones", "2"), ("--n-clones=2",)],
    ids=["no request K", "request K", "request K joined"],
)
def test_candidate_clone_counts_plan_the_spine_request_once_per_count(
    own, spine_request, capsys
):
    """--candidate-clone-counts plans the spine request once per count, each
    as the request's own --n-clones, in place of any the request names
    (microcosm#931); each plan's configuration carries its count."""
    args = spine_request(own, "--candidate-clone-counts", "1,3", "--dry-run")
    assert cli._dry_run(args) == 0
    inventories = json.loads(capsys.readouterr().out)
    assert [raw.n_clones for raw in spine_request.requests] == [1, 3]
    assert [item["n_clones"] for item in inventories] == [1, 3]
    assert [item["configuration"]["n_clones"] for item in inventories] == [1, 3]
    # Each candidate is planned on a copy of the run's arguments.
    assert args.n_clones is None


def test_a_sidecar_whose_clone_count_is_not_its_frames_is_refused(
    tmp_path, toy_ladder, stand_in_engine
):
    """The checkpoint authentication holds the sidecar's K to the frame's
    (microcosm#931), naming the two counts."""
    pytest.importorskip("tables")
    args = _checkpoint_request(tmp_path, toy_ladder[1], clones=3)
    sidecar = json.loads(args.input_sidecar.read_text())
    sidecar["geographic_clone_count"] = 2
    args.input_sidecar.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError) as refused:
        cli.prepare_full_build(args)
    assert str(refused.value) == (
        "input H5 build sidecar geographic_clone_count mismatch: sidecar 2, frame 3"
    )


@pytest.mark.parametrize("route", ROUTES)
def test_a_release_candidate_on_a_sampled_spine_is_refused(
    route, request, tmp_path, monkeypatch
):
    """The dense line samples nothing of its own: its rung and its seed are
    its spine's (microcosm#931), so a release candidate on a sampled spine is
    refused at preparation, where the spine's sampling is read (its sidecar's,
    or the request's own --sample-fraction), and a candidate build on it
    records the spine's rung and seed, its identity digest among them."""
    from microcosm.build.gate_battery import gate_signing_key_env
    from microcosm.build.uk_runtime.rowwise_cli import rowwise_parameters

    monkeypatch.setenv(
        gate_signing_key_env("uk"), "BQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQU="
    )
    # A seed other than the role's default (578), so a seed the run did not
    # take from its spine shows.
    sampling = {"fraction": 0.1, "seed": 7, "rung_token": "f010"}
    release = _dense_args(
        route, request, tmp_path, "--release-candidate", sampling=sampling
    )
    with pytest.raises(ValueError, match="Sampled builds cannot request release"):
        cli.prepare_full_build(release)
    assert (release.sample_fraction, release.n_clones) == (0.1, 1)
    if route == "spine_request":
        (raw,) = request.getfixturevalue("spine_request").requests
        assert raw.release_candidate is True and raw.sample_fraction == 0.1
    candidate = _dense_args(route, request, tmp_path / "candidate", sampling=sampling)
    attempt = _attempt(monkeypatch, rung="f010")
    candidate._build_id_rung = "f010"
    prepared = cli.prepare_full_build(candidate, attempt=attempt)
    assert prepared.full.config.effective_sample_fraction == 0.1
    assert (candidate.sample_fraction, candidate.sample_seed) == (0.1, 7)
    assert rowwise_parameters(candidate, source_year=2023)["sample_seed"] == 7
    assert attempt["state"].identity_digest == _digest(prepared, candidate)
    assert prepared.bindings["spine"]["sampling"] == {
        "fraction": 0.1,
        "seed": 7,
        "sampled": True,
        "rung_token": "f010",
    }


_RELEASE_CANDIDATE_SPINE = (
    "--release-candidate builds only on a release-candidate spine (microcosm#931): "
)
#: The binding failures a geography block can fail at its own K, in the
#: helper's order: the law, the identity, the digest and each system's pin.
_BINDING_FAILURES = [
    "its geography binding's assignment is not 'atomic'",
    "its geography binding's identity is not 'geography_household_key'",
    "its geography binding's definition_sha256 is not a SHA-256 hex digest",
    *(
        f"its geography binding's support_pins.{system} is not the registered "
        "conditioned support"
        for system in SYSTEMS
    ),
]


@pytest.fixture
def signing_key(monkeypatch):
    from microcosm.build.gate_battery import gate_signing_key_env

    monkeypatch.setenv(
        gate_signing_key_env("uk"), "BQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQU="
    )


def test_a_release_candidate_prepares_on_a_release_spine_at_the_doctrines_k(
    tmp_path, toy_ladder, stand_in_engine, signing_key, monkeypatch
):
    """A release candidate builds on a bound spine of the doctrine's 15
    geographic clones that the spine build made as a release candidate, whose
    geography binding is a release's: here the stand-in's own block, on the
    supports the register lists (microcosm#931)."""
    pytest.importorskip("tables")
    patch_support_register(monkeypatch)
    args = _checkpoint_request(
        tmp_path,
        toy_ladder[1],
        "--release-candidate",
        clones=15,
        posture="release_candidate",
    )
    prepared = cli.prepare_full_build(args)
    assert prepared.bindings["release_candidate"] is True
    assert prepared.bindings["spine"]["geographic_clone_count"] == 15
    assert prepared.bindings["spine"]["release_posture"] == "release_candidate"
    assert prepared.bindings["geography"] == {
        **synthetic_spine_geography(),
        "geographic_clone_count": 15,
    }
    assert args.n_clones == 15


def _legacy_assignment(sidecar_path) -> None:
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["geography"]["assignment"] = "legacy"
    sidecar_path.write_text(json.dumps(sidecar))


@pytest.mark.parametrize(
    ("clones", "posture", "registered", "edit", "failures"),
    [
        (
            7,
            "release_candidate",
            True,
            None,
            ["its geographic clone count is K = 7, not the doctrine's K = 15"],
        ),
        (15, "development", True, None, ["it was not built as a release candidate"]),
        (15, "release_candidate", True, _legacy_assignment, _BINDING_FAILURES[:1]),
        (15, "release_candidate", False, None, _BINDING_FAILURES[3:]),
        (
            1,
            "development",
            False,
            None,
            [
                "its geographic clone count is K = 1, not the doctrine's K = 15",
                "it was not built as a release candidate",
                *_BINDING_FAILURES[3:],
            ],
        ),
    ],
    ids=["K", "posture", "assignment", "register", "all"],
)
def test_a_release_candidate_on_a_spine_a_release_cannot_build_on_is_refused(
    tmp_path,
    toy_ladder,
    stand_in_engine,
    signing_key,
    monkeypatch,
    clones,
    posture,
    registered,
    edit,
    failures,
):
    """A bound spine at another K, in another posture, or whose geography
    binding is not a release's (another law, or pins off the conditioned
    register) is refused as a release candidate's, naming every failure and K,
    never a value (microcosm#931); the same spine builds a candidate."""
    pytest.importorskip("tables")
    if registered:
        patch_support_register(monkeypatch)
    args = _checkpoint_request(
        tmp_path, toy_ladder[1], "--release-candidate", clones=clones, posture=posture
    )
    if edit is not None:
        edit(args.input_sidecar)
    with pytest.raises(ValueError) as refused:
        cli.prepare_full_build(args)
    assert str(refused.value) == _RELEASE_CANDIDATE_SPINE + "; ".join(failures) + "."
    candidate = _checkpoint_request(
        tmp_path / "candidate", toy_ladder[1], clones=clones, posture=posture
    )
    if edit is not None:
        edit(candidate.input_sidecar)
    assert cli.prepare_full_build(candidate).bindings["release_candidate"] is False


@pytest.mark.parametrize("clones", [15, 7])
def test_a_release_candidate_on_a_spine_request_is_refused(
    spine_request, signing_key, monkeypatch, clones
):
    """A spine request's binding carries the geography block its preparation
    composes, through the sidecar writer's helper, and a synthetic request is
    a smoke build whose block is the toy supports' under the synthetic
    register: a release candidate on one is refused, naming every field its
    binding fails of a release's, its pins off the register and its register
    among them, before the full graph is composed (microcosm#931)."""
    patch_support_register(monkeypatch)

    def composed(*args, **kwargs):
        raise AssertionError("the full graph is composed after the refusal")

    monkeypatch.setattr(cli, "uk_full_graph", composed)
    args = spine_request(["--n-clones", str(clones)], "--release-candidate")
    with pytest.raises(ValueError) as refused:
        cli.prepare_full_build(args)
    failures = [
        *(
            []
            if clones == 15
            else [
                f"its geographic clone count is K = {clones}, not the doctrine's K = 15"
            ]
        ),
        "it was not built as a release candidate",
        *_BINDING_FAILURES[3:],
        "its geography binding's register is not 'sources.yaml'",
    ]
    assert str(refused.value) == _RELEASE_CANDIDATE_SPINE + "; ".join(failures) + "."
    (raw,) = spine_request.requests
    assert raw.release_candidate is True and raw.n_clones == clones


def test_a_spine_whose_sampling_is_not_a_rung_is_refused(
    tmp_path, toy_ladder, stand_in_engine
):
    """A bound spine's sidecar records a fraction off the scale ladder: the
    run would record a rung it has no token for, so preparation refuses it,
    naming the rungs and not the fraction (microcosm#931)."""
    pytest.importorskip("tables")
    args = _checkpoint_request(
        tmp_path, toy_ladder[1], sampling={"fraction": 0.37, "seed": 5}
    )
    with pytest.raises(ValueError) as refused:
        cli.prepare_full_build(args)
    assert str(refused.value) == (
        "The spine's sampling fraction is not a scale-ladder rung (one of f001, "
        "f010, f100): the dense line's rung is its spine's (microcosm#931)."
    )
    assert args.n_clones is None and args.sample_fraction == 1.0


def test_an_attempt_labelled_at_another_rung_than_its_spines_is_refused(
    tmp_path, toy_ladder, stand_in_engine
):
    """The dense attempt's build id is minted with its spine's rung, read
    ahead of preparation; an id whose rung is not the bound spine's is
    refused, naming both rung tokens, before the run records anything of the
    spine (microcosm#931)."""
    pytest.importorskip("tables")
    args = _checkpoint_request(
        tmp_path,
        toy_ladder[1],
        sampling={"fraction": 0.1, "seed": 7, "rung_token": "f010"},
    )
    args._build_id_rung = "f100"
    with pytest.raises(ValueError) as refused:
        cli.prepare_full_build(args)
    assert str(refused.value) == (
        "The attempt's build id carries rung f100, but the spine it binds is at "
        "rung f010: a dense build id takes its spine's rung (microcosm#931)."
    )
    assert (args.n_clones, args.sample_fraction) == (None, 1.0)


# ---------------------------------------------------------------------------
# microcosm#1301: a gate-only sign-off and a later gate clock leave the solve
# ---------------------------------------------------------------------------


@pytest.fixture
def stand_in_dependency_versions(monkeypatch):
    """Kernels hash their declared dependencies' versions; the engine-free
    environment lacks policyengine-uk, so absent ones read as a stand-in."""
    from importlib import metadata

    installed = metadata.version

    def version(distribution):
        try:
            return installed(distribution)
        except metadata.PackageNotFoundError:
            return "0+stand-in"

    monkeypatch.setattr(metadata, "version", version)


def _node_keys(prepared):
    """Every node key of the prepared graph, over fixed source identities."""
    from microcosm.graph.keys import node_key

    compiled = compile_graph(prepared.full.graph)
    sources = {
        source.name: hashlib.sha256(source.name.encode()).hexdigest()
        for source in compiled.graph.sources
    }
    keys: dict[str, str] = {}
    for node_id in compiled.order:
        kernel = prepared.kernels.get(compiled.graph.node(node_id).kernel)
        keys[node_id] = node_key(
            compiled,
            node_id,
            keys,
            kernel.implementation_hash(),
            sources,
            kernel_capabilities=kernel.capabilities,
        )
    return compiled, keys


def _descendants(compiled, node_id):
    found, pending = set(), [node_id]
    while pending:
        current = pending.pop()
        for successor, predecessors in compiled.predecessors.items():
            if current in predecessors and successor not in found:
                found.add(successor)
                pending.append(successor)
    return found


def _perturbed(monkeypatch, *resources):
    """Serve the packaged UK spec with ``resources`` edited (their hashes)."""
    from dataclasses import replace
    from types import MappingProxyType

    from microcosm.build import country_spec

    spec = country_spec.load_country_spec("uk")
    hashes = dict(spec.resource_hashes)
    for resource in resources:
        hashes[resource] = hashlib.sha256(hashes[resource].encode()).hexdigest()
    edited = replace(spec, resource_hashes=MappingProxyType(hashes))
    monkeypatch.setattr(country_spec, "_load_packaged_country_spec", lambda _: edited)


def test_a_gate_only_sign_off_rekeys_the_calibrated_gates_and_nothing_before(
    checkpoint_argv, monkeypatch, stand_in_dependency_versions
):
    """Every node the calibrated gates depend on keeps its key when every
    gate-only register is edited at once; the calibrated gates re-key."""
    from microcosm.build.uk_runtime.solve_scope import UK_GATE_ONLY_RESOURCES

    argv, _, _ = checkpoint_argv
    args = cli.parse_args([*argv, "--dataset-households", "2"])
    prepared = cli.prepare_full_build(args)
    compiled, before = _node_keys(prepared)
    gates = "uk.full.gates.calibrated"
    downstream = {gates, *_descendants(compiled, gates)}
    solve = set(compiled.order) - downstream
    assert {
        "uk.full.target_compilation",
        "uk.full.problem",
        "uk.full.gates.preflight",
        "uk.full.dense",
        "uk.full.size_refit",
        "uk.full.calibrated",
        "uk.full.holdout",
    } <= solve
    with monkeypatch.context() as patch:
        _perturbed(patch, *UK_GATE_ONLY_RESOURCES)
        _, signed = _node_keys(prepared)
    assert {node: signed[node] for node in solve} == {
        node: before[node] for node in solve
    }
    assert signed[gates] != before[gates]
    # Control: a solve-scoped register re-keys the solve from compilation on.
    with monkeypatch.context() as patch:
        _perturbed(patch, "calibration_measure_exclusions.json")
        _, edited = _node_keys(prepared)
    for node in ("uk.full.target_compilation", "uk.full.dense", "uk.full.holdout"):
        assert edited[node] != before[node], node


def test_a_later_gate_clock_rekeys_only_the_calibrated_gates_and_the_export(
    checkpoint_argv, stand_in_dependency_versions
):
    from datetime import date, timedelta

    argv, _, _ = checkpoint_argv
    solve = (date.today() - timedelta(days=2)).isoformat()
    gate = date.today().isoformat()
    first = cli.prepare_full_build(cli.parse_args([*argv, "--review-date", solve]))
    second = cli.prepare_full_build(
        cli.parse_args([*argv, "--review-date", solve, "--gate-review-date", gate])
    )
    assert (first.bindings["review_date"], first.bindings["gate_review_date"]) == (
        solve,
        solve,
    )
    assert (second.bindings["review_date"], second.bindings["gate_review_date"]) == (
        solve,
        gate,
    )
    _, before = _node_keys(first)
    _, after = _node_keys(second)
    assert before.keys() == after.keys()
    assert {node for node in before if before[node] != after[node]} == {
        "uk.full.gates.calibrated",
        "uk.full.export.prepare",
    }
    calibrated = second.full.graph.node("uk.full.gates.calibrated")
    assert calibrated.params["review_date"] == gate
    # The problem and the rotated holdout review the local binding
    # adjudications on the solve's date, as the target compilation reviews
    # the measure exclusions (microcosm#1278).
    for node in (
        "uk.full.gates.preflight",
        "uk.full.target_compilation",
        "uk.full.problem",
        "uk.full.holdout",
    ):
        assert second.full.graph.node(node).params["review_date"] == solve


def test_a_spine_requests_binding_carries_the_block_its_preparation_composes(
    monkeypatch,
):
    """A spine request binds the geography block its preparation composed,
    through the helper the spine build's sidecar writer calls, with K beside
    it (microcosm#931): on supports held to the register, a release-candidate
    request's binding passes the release check, as a bound spine's does. A
    request whose roster runs no area draw composes none, and its binding
    carries K alone."""
    from types import SimpleNamespace

    from microcosm.build.uk_runtime.country_adapter import (
        UK_SPINE_SUPPORT_REGISTER,
        uk_spine_geography_binding,
        uk_spine_geography_binding_failures,
    )
    from test_support.microcosm_build.uk_conditioned_support_fixtures import (
        toy_conditioned_support_payloads,
    )

    block = uk_spine_geography_binding(
        toy_conditioned_support_payloads(), register=UK_SPINE_SUPPORT_REGISTER
    )
    patch_support_register(monkeypatch, pins=block["support_pins"])
    raw = SimpleNamespace(
        n_clones=15,
        sample_fraction=1.0,
        sample_seed=0,
        smoke=False,
        release_candidate=True,
    )
    binding = cli._requested_spine_binding(raw, block)
    assert binding["geography"] == {**block, "geographic_clone_count": 15}
    assert binding["release_posture"] == "release_candidate"
    assert (
        uk_spine_geography_binding_failures(binding["geography"], clone_count=15) == ()
    )
    assert cli._require_release_candidate_spine(binding, clone_count=15) is None
    bare = cli._requested_spine_binding(raw, None)
    assert bare["geography"] == {"geographic_clone_count": 15}
    with pytest.raises(ValueError, match="it records no geography binding"):
        cli._require_release_candidate_spine(bare, clone_count=15)
