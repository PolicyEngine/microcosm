"""Informed L0 must preserve rare carriers without replacing the search."""

import numpy as np
import pytest
import torch
from scipy import sparse

from microcosm.calibrate.gates import HardConcrete
from microcosm.calibrate.initialization import contribution_initialization


def test_small_weight_large_measure_is_protected():
    init = contribution_initialization(
        sparse.csr_array([[1000.0, 0, 0], [0, 1, 1]]),
        np.array([0.01, 100.0, 100.0]),
        np.array([10.0, 200.0]),
    )
    assert init.protected[0]
    assert init.probabilities[0] > init.probabilities[1]


def test_protected_gates_stay_open_during_training_and_evaluation():
    gates = HardConcrete(
        3,
        initial_probabilities=np.array([0.2, 0.4, 0.8]),
        protected_mask=np.array([True, False, False]),
    )
    assert np.allclose(gates.get_active_prob().detach().numpy(), [1, 0.4, 0.8])
    with torch.no_grad():
        gates.qz_logits.fill_(-100)
    for training in (True, False):
        gates.train(training)
        assert gates()[0].item() == 1
    assert gates.get_active_prob()[0].item() == 1


@pytest.mark.parametrize(
    "probabilities", [[0, 0.5], [0.5, 1], [float("nan"), 0.5], [0.5]]
)
def test_invalid_initial_probabilities_refuse(probabilities):
    with pytest.raises(ValueError):
        HardConcrete(2, initial_probabilities=np.array(probabilities))


def test_mass_basis_budget_search_tracks_open_probability_mass():
    import numpy as np
    import pandas as pd

    from microcosm.calibrate import Target, TargetSet, calibrate
    from microcosm.calibrate.solve import (
        BUDGET_BASIS_NONZERO_COUNT,
        BUDGET_BASIS_OPEN_PROBABILITY_MASS,
    )
    from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

    n = 60
    rng = np.random.default_rng(3)
    household = pd.DataFrame(
        {"household_id": np.arange(1, n + 1), "x": rng.uniform(1, 5, n)}
    )
    person = pd.DataFrame(
        {"person_id": np.arange(1, n + 1), "person_household_id": np.arange(1, n + 1)}
    )
    frame = Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.ones(n), WeightKind.DESIGN)},
    )
    targets = TargetSet(
        [
            Target("count", "household", lambda f: np.ones(f.n("household")), 30.0),
            Target(
                "x",
                "household",
                lambda f: f.table("household")["x"].to_numpy(),
                30.0 * 3.0,
            ),
        ]
    )
    common = dict(epochs=200, learning_rate=0.05, seed=1, budget_iters=10)
    by_mass = calibrate(
        frame,
        targets,
        target_records=20,
        budget_basis=BUDGET_BASIS_OPEN_PROBABILITY_MASS,
        **common,
    )
    assert by_mass.options["budget_basis"] == BUDGET_BASIS_OPEN_PROBABILITY_MASS
    assert by_mass.gate_open_probabilities is not None
    mass = float(np.sum(by_mass.gate_open_probabilities))
    # The search's own tolerance is 5% of the budget (at least one record).
    assert abs(mass - 20) <= max(1, round(0.05 * 20)) + 1
    # The count basis on the same problem lands below the budget in mass.
    by_count = calibrate(
        frame,
        targets,
        target_records=20,
        budget_basis=BUDGET_BASIS_NONZERO_COUNT,
        **common,
    )
    assert by_count.options["budget_basis"] == BUDGET_BASIS_NONZERO_COUNT
    with pytest.raises(ValueError, match="budget_basis"):
        calibrate(frame, targets, target_records=20, budget_basis="bogus", **common)
    with pytest.raises(ValueError, match="target_records"):
        calibrate(
            frame, targets, budget_basis=BUDGET_BASIS_OPEN_PROBABILITY_MASS, **common
        )


def _stub_polarised_optimizer(thresholds, sharpness):
    """A stand-in optimizer whose gates open below a per-record log10 penalty."""
    import numpy as np

    def fake_optimize(matrix, targets, tlw, tls, cap, initial_weights, **kwargs):
        u = np.log10(float(kwargs["l0_lambda"]))
        logit = np.clip(sharpness * (thresholds - u), -700.0, 700.0)
        pi = 1.0 / (1.0 + np.exp(-logit))
        weights = np.asarray(initial_weights, dtype=np.float64) * pi
        trajectory = np.zeros(int(kwargs["epochs"]), dtype=np.float64)
        if kwargs.get("return_gate_open_probabilities"):
            return weights, trajectory, pi
        return weights, trajectory

    return fake_optimize


def test_feasibility_aware_search_stops_only_on_a_drawable_design(monkeypatch):
    """The S2 failure (microcosm#355, 2026-09-08) on a synthetic response.

    The first probe lands inside the +/-5% band but *below* the budget with
    near-binary gates, so an exact-count draw at 0.95 is infeasible. The plain
    mass basis stops there; the feasibility-aware search keeps bisecting
    toward a smaller penalty until a probe admits the draw.
    """
    import numpy as np
    import pandas as pd

    from microcosm.calibrate import (
        Target,
        TargetSet,
        calibrate,
        exact_k_design_feasibility,
    )
    from microcosm.calibrate import solve as solve_module
    from microcosm.calibrate.solve import BUDGET_BASIS_OPEN_PROBABILITY_MASS
    from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

    n, k = 1000, 500
    # Open thresholds uniform on [-7.12, 0.88] in log10(lambda): the bracket
    # mid-point probe (lambda 1e-3) opens ~485 gates, inside the band of 25.
    thresholds = -7.12 + 8.0 * (np.arange(n) + 0.5) / n
    monkeypatch.setattr(
        solve_module, "_optimize", _stub_polarised_optimizer(thresholds, 10.0)
    )
    household = pd.DataFrame({"household_id": np.arange(1, n + 1)})
    person = pd.DataFrame(
        {"person_id": np.arange(1, n + 1), "person_household_id": np.arange(1, n + 1)}
    )
    frame = Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.ones(n), WeightKind.DESIGN)},
    )
    targets = TargetSet(
        [Target("count", "household", lambda f: np.ones(f.n("household")), 500.0)]
    )
    common = dict(
        epochs=3,
        seed=1,
        budget_iters=10,
        target_records=k,
        budget_basis=BUDGET_BASIS_OPEN_PROBABILITY_MASS,
    )

    plain = calibrate(frame, targets, **common)
    plain_search = plain.options["budget_search"]
    assert plain_search["evaluations"] == 1
    assert plain_search["feasible_draw_pi_hi"] is None
    assert plain_search["selected_feasible"] is None
    assert abs(plain_search["selected_measure"] - k) <= plain_search["tolerance"]
    assert plain_search["selected_measure"] < k
    verdict = exact_k_design_feasibility(plain.gate_open_probabilities, k, 0.95)
    assert verdict["feasible"] is False and verdict["reason"] == "boundary_mass_short"

    events: list[dict] = []
    aware = calibrate(
        frame,
        targets,
        feasible_draw_pi_hi=0.95,
        progress_callback=events.append,
        **common,
    )
    search = aware.options["budget_search"]
    # One budget_probe event per probe, matching the receipt, then the stop.
    probe_events = [e for e in events if e.get("kind") == "budget_probe"]
    assert [e["l0_lambda"] for e in probe_events] == [
        p["l0_lambda"] for p in search["probes"]
    ]
    assert [e["verdict"] for e in probe_events] == [
        p["verdict"] for p in search["probes"]
    ]
    assert probe_events[0]["budget_iteration"] == 1
    assert probe_events[0]["budget_basis"] == BUDGET_BASIS_OPEN_PROBABILITY_MASS
    done = [e for e in events if e.get("kind") == "budget_search_done"]
    assert len(done) == 1 and done[0]["stopped_on"] == search["stopped_on"]
    assert done[0]["selected_l0_lambda"] == search["selected_l0_lambda"]
    # The stub optimizer emits no epoch events; only the search's own do.
    assert {e.get("kind") for e in events} == {"budget_probe", "budget_search_done"}
    assert aware.options["feasible_draw_pi_hi"] == 0.95
    assert search["stopped_on"] == "acceptable_within_tolerance"
    assert search["selected_feasible"] is True
    assert 1 < search["evaluations"] <= 10
    probes = search["probes"]
    assert probes[0]["verdict"] == "boundary_mass_short"
    assert probes[0]["l0_lambda"] == plain_search["selected_l0_lambda"]
    assert any(p["verdict"] == "certainties_exceed_k" for p in probes)
    assert probes[-1]["verdict"] == "feasible"
    assert aware.l0_lambda == probes[-1]["l0_lambda"] < probes[0]["l0_lambda"]
    assert abs(search["selected_measure"] - k) <= search["tolerance"]
    final = exact_k_design_feasibility(aware.gate_open_probabilities, k, 0.95)
    assert final["feasible"] is True

    with pytest.raises(ValueError, match="feasible_draw_pi_hi"):
        calibrate(frame, targets, epochs=3, target_records=k, feasible_draw_pi_hi=0.95)
    with pytest.raises(ValueError, match="feasible_draw_pi_hi"):
        calibrate(frame, targets, feasible_draw_pi_hi=1.5, **common)
    with pytest.raises(ValueError, match="feasible_draw_pi_hi"):
        calibrate(frame, targets, epochs=3, feasible_draw_pi_hi=0.95)


def test_feasibility_aware_search_returns_the_closest_run_when_nothing_is_drawable(
    monkeypatch,
):
    """No probe is both drawable and within tolerance: the search spends its
    budget and returns the best run by (drawable first, then distance), so
    the draw can refuse with a measurement instead of the search hiding it.
    """
    import numpy as np
    import pandas as pd

    from microcosm.calibrate import Target, TargetSet, calibrate
    from microcosm.calibrate import solve as solve_module
    from microcosm.calibrate.solve import BUDGET_BASIS_OPEN_PROBABILITY_MASS
    from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

    n, k = 200, 100
    # Steps so sharp that every probe the bisection visits opens either 90
    # gates (drawable: the tail's equal, tiny probabilities scale to the 10
    # places, but 90 is outside the band of 5) or 110 (more certainties than
    # k, never drawable).
    thresholds = np.concatenate(
        [np.full(90, 0.5), np.full(20, -3.3), np.full(90, -6.5)]
    )
    monkeypatch.setattr(
        solve_module, "_optimize", _stub_polarised_optimizer(thresholds, 400.0)
    )
    household = pd.DataFrame({"household_id": np.arange(1, n + 1)})
    person = pd.DataFrame(
        {"person_id": np.arange(1, n + 1), "person_household_id": np.arange(1, n + 1)}
    )
    frame = Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.ones(n), WeightKind.DESIGN)},
    )
    targets = TargetSet(
        [Target("count", "household", lambda f: np.ones(f.n("household")), 100.0)]
    )
    result = calibrate(
        frame,
        targets,
        epochs=3,
        seed=1,
        budget_iters=6,
        target_records=k,
        budget_basis=BUDGET_BASIS_OPEN_PROBABILITY_MASS,
        feasible_draw_pi_hi=0.95,
    )
    search = result.options["budget_search"]
    assert search["stopped_on"] == "budget_exhausted"
    assert search["evaluations"] == 6
    verdicts = {p["verdict"] for p in search["probes"]}
    assert verdicts == {"feasible", "certainties_exceed_k"}
    # A drawable probe outside the band beats an undrawable one at the same
    # distance; the returned run is that drawable 90-gate probe.
    assert search["selected_feasible"] is True
    assert search["selected_measure"] == 90
    assert abs(search["selected_measure"] - k) > search["tolerance"]
    assert result.gate_open_probabilities is not None


def test_warm_start_probes_the_given_penalty_first_and_brackets_a_miss(monkeypatch):
    """microcosm#1115: a known penalty is probed first; a hit ends the search
    after one probe, a miss brackets within half a decade and interpolates."""
    import numpy as np
    import pandas as pd

    from microcosm.calibrate import Target, TargetSet, calibrate
    from microcosm.calibrate import solve as solve_module
    from microcosm.calibrate.solve import (
        _L0_WARM_SPAN_DECADES,
        BUDGET_BASIS_OPEN_PROBABILITY_MASS,
    )
    from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

    n, k = 1000, 500
    thresholds = -7.12 + 8.0 * (np.arange(n) + 0.5) / n
    monkeypatch.setattr(
        solve_module, "_optimize", _stub_polarised_optimizer(thresholds, 10.0)
    )
    household = pd.DataFrame({"household_id": np.arange(1, n + 1)})
    person = pd.DataFrame(
        {"person_id": np.arange(1, n + 1), "person_household_id": np.arange(1, n + 1)}
    )
    frame = Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.ones(n), WeightKind.DESIGN)},
    )
    targets = TargetSet(
        [Target("count", "household", lambda f: np.ones(f.n("household")), 500.0)]
    )
    common = dict(
        epochs=3,
        seed=1,
        budget_iters=10,
        target_records=k,
        budget_basis=BUDGET_BASIS_OPEN_PROBABILITY_MASS,
        feasible_draw_pi_hi=0.95,
    )

    cold = calibrate(frame, targets, **common)
    cold_search = cold.options["budget_search"]
    # The cold path is untouched: mid-point first, no warm receipt.
    assert cold_search["probes"][0]["l0_lambda"] == 1e-3
    assert cold_search["initial_lambda"] is None
    assert cold_search["warm_edge_probe"] is None
    assert cold_search["stopped_on"] == "acceptable_within_tolerance"
    known = cold_search["selected_l0_lambda"]
    assert cold_search["evaluations"] > 2

    # A hit: the known penalty is probed first and the search stops there.
    warm = calibrate(frame, targets, l0_lambda=known, **common)
    search = warm.options["budget_search"]
    assert search["initial_lambda"] == known
    assert search["evaluations"] == 1
    assert search["probes"][0]["l0_lambda"] == known
    assert search["warm_edge_probe"] is None
    assert search["selected_feasible"] is True
    assert warm.l0_lambda == known
    np.testing.assert_array_equal(warm.weights, cold.weights)

    # A miss on the over-pruned side: the next probe sits half a decade below
    # the hint, the bracket closes on the measured ends, and the search lands
    # within the probes bisection would have needed from the hint alone.
    stale = known * 10**0.35
    missed = calibrate(frame, targets, l0_lambda=stale, **common)
    search = missed.options["budget_search"]
    probes = search["probes"]
    assert probes[0]["l0_lambda"] == pytest.approx(stale)
    assert probes[0]["verdict"] in ("boundary_mass_short", "boundary_short_of_draw")
    assert probes[1]["l0_lambda"] == pytest.approx(stale / 10**_L0_WARM_SPAN_DECADES)
    assert search["warm_edge_probe"]["l0_lambda"] == pytest.approx(
        probes[1]["l0_lambda"]
    )
    assert search["stopped_on"] == "acceptable_within_tolerance"
    assert search["selected_feasible"] is True
    assert search["evaluations"] <= 4
    assert abs(search["selected_measure"] - k) <= search["tolerance"]

    # A miss beyond the span hands the search the global bracket on that side
    # and it still settles within the iteration budget.
    far = known * 10**2.0
    recovered = calibrate(frame, targets, l0_lambda=far, **common)
    search = recovered.options["budget_search"]
    assert search["probes"][0]["l0_lambda"] == pytest.approx(far)
    assert search["warm_edge_probe"]["steer"] == "smaller_penalty"
    assert search["stopped_on"] == "acceptable_within_tolerance"
    assert search["selected_feasible"] is True


def _stub_budget_search(monkeypatch, thresholds, **kwargs):
    """Run the feasibility-aware mass-basis search on the stub optimizer.

    Returns the result and the progress events. ``thresholds`` sets, per
    record, the log10 penalty below which its gate opens (sharpness 10 per
    decade), so the open mass is a known, monotone function of the penalty.
    """
    import numpy as np
    import pandas as pd

    from microcosm.calibrate import Target, TargetSet, calibrate
    from microcosm.calibrate import solve as solve_module
    from microcosm.calibrate.solve import BUDGET_BASIS_OPEN_PROBABILITY_MASS
    from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

    n = len(thresholds)
    monkeypatch.setattr(
        solve_module, "_optimize", _stub_polarised_optimizer(thresholds, 10.0)
    )
    household = pd.DataFrame({"household_id": np.arange(1, n + 1)})
    person = pd.DataFrame(
        {"person_id": np.arange(1, n + 1), "person_household_id": np.arange(1, n + 1)}
    )
    frame = Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.ones(n), WeightKind.DESIGN)},
    )
    targets = TargetSet(
        [Target("count", "household", lambda f: np.ones(f.n("household")), n / 2)]
    )
    events: list[dict] = []
    options = dict(
        epochs=3,
        seed=1,
        budget_iters=10,
        target_records=n // 2,
        budget_basis=BUDGET_BASIS_OPEN_PROBABILITY_MASS,
        feasible_draw_pi_hi=0.95,
        progress_callback=events.append,
    )
    options.update(kwargs)
    return calibrate(frame, targets, **options), events


def _probed(search):
    return [probe["l0_lambda"] for probe in search["probes"]]


def _assert_no_repeated_probe(search):
    import math

    logs = [math.log10(lam) for lam in _probed(search)]
    for i, left in enumerate(logs):
        for right in logs[i + 1 :]:
            assert abs(left - right) > 1e-6, _probed(search)


_UK_BRACKET = (1e-10, 10.0)


def test_a_lower_bracket_settles_below_the_default_floor(monkeypatch):
    """microcosm#1301: a budget that needs a penalty near 10^-8.5 settles
    below 1e-7 when the caller's bracket reaches it, and the default bracket
    refuses the warm start rather than clamping it to 1e-7."""
    import numpy as np

    n = 1000
    # Open thresholds uniform on [-10.5, -6.5]: half the gates are open near
    # 10^-8.5, far below the default floor.
    thresholds = -10.5 + 4.0 * (np.arange(n) + 0.5) / n
    result, events = _stub_budget_search(
        monkeypatch, thresholds, l0_lambda=2.5e-8, budget_search_bracket=_UK_BRACKET
    )
    search = result.options["budget_search"]
    assert search["search_bracket"] == {"lower": 1e-10, "upper": 10.0}
    assert search["stopped_on"] == "acceptable_within_tolerance"
    assert search["selected_feasible"] is True
    assert result.l0_lambda < 1e-7
    assert search["probes"][0]["l0_lambda"] == 2.5e-8
    assert all(1e-10 <= lam <= 10.0 for lam in _probed(search))
    _assert_no_repeated_probe(search)
    done = [e for e in events if e.get("kind") == "budget_search_done"]
    assert done[0]["stopped_on"] == search["stopped_on"]
    assert done[0]["search_bracket"] == search["search_bracket"]

    with pytest.raises(ValueError, match="bracket"):
        _stub_budget_search(monkeypatch, thresholds, l0_lambda=2.5e-8)


def test_the_search_stops_on_the_floor_instead_of_reprobing_it(monkeypatch):
    """microcosm#1301: the 100,000-household search probed 1.9e-7, then 1e-7,
    then 1e-7 again on a bracket collapsed onto the floor. When even the
    floor opens too little mass the search now stops there by name."""
    import numpy as np

    n = 1000
    # Every gate opens only below 10^-10.3: no penalty in either bracket
    # opens enough mass.
    thresholds = -12.5 + 2.2 * (np.arange(n) + 0.5) / n

    replay, events = _stub_budget_search(monkeypatch, thresholds, l0_lambda=1.9e-7)
    search = replay.options["budget_search"]
    assert search["search_bracket"] == {"lower": 1e-7, "upper": 10.0}
    assert _probed(search) == pytest.approx([1.9e-7, 1e-7])
    assert search["evaluations"] == 2
    assert search["warm_edge_probe"]["steer"] == "smaller_penalty"
    assert search["stopped_on"] == "search_floor_reached"
    done = [e for e in events if e.get("kind") == "budget_search_done"]
    assert len(done) == 1 and done[0]["stopped_on"] == "search_floor_reached"
    _assert_no_repeated_probe(search)

    # A warm start on the floor that asks for a smaller penalty stops at
    # once; it is never sent half a decade upward.
    on_floor, _ = _stub_budget_search(monkeypatch, thresholds, l0_lambda=1e-7)
    search = on_floor.options["budget_search"]
    assert _probed(search) == [1e-7]
    assert search["warm_edge_probe"] is None
    assert search["stopped_on"] == "search_floor_reached"

    # The same within three probes on a caller's lower floor.
    lower, _ = _stub_budget_search(
        monkeypatch, thresholds, l0_lambda=1.5e-10, budget_search_bracket=_UK_BRACKET
    )
    search = lower.options["budget_search"]
    assert _probed(search) == pytest.approx([1.5e-10, 1e-10])
    assert search["stopped_on"] == "search_floor_reached"
    _assert_no_repeated_probe(search)


def test_the_search_stops_on_the_ceiling_by_name(monkeypatch):
    import numpy as np

    n = 1000
    # Every gate stays open up to 10^2.3: no penalty in the bracket closes
    # enough of them.
    thresholds = 2.3 + 2.2 * (np.arange(n) + 0.5) / n
    near, _ = _stub_budget_search(monkeypatch, thresholds, l0_lambda=5.0)
    search = near.options["budget_search"]
    assert _probed(search) == pytest.approx([5.0, 10.0])
    assert search["warm_edge_probe"]["steer"] == "larger_penalty"
    assert search["stopped_on"] == "search_ceiling_reached"
    _assert_no_repeated_probe(search)

    on_ceiling, _ = _stub_budget_search(monkeypatch, thresholds, l0_lambda=10.0)
    search = on_ceiling.options["budget_search"]
    assert _probed(search) == [10.0]
    assert search["stopped_on"] == "search_ceiling_reached"


def test_a_warm_search_probes_the_bound_beyond_its_edge(monkeypatch):
    """microcosm#1301: a warm start far from the floor whose edge still asks
    for a smaller penalty probes the floor next. An unreachable budget stops
    there by name in three probes; it no longer bisects toward the floor
    until all ten probes are spent without ever measuring it. The ceiling
    mirrors it."""
    import numpy as np

    n = 1000
    # Every gate opens only below 10^-10.3, beyond the UK floor of 1e-10.
    unreachable = -12.5 + 2.2 * (np.arange(n) + 0.5) / n
    result, events = _stub_budget_search(
        monkeypatch, unreachable, l0_lambda=2.5e-8, budget_search_bracket=_UK_BRACKET
    )
    search = result.options["budget_search"]
    assert search["evaluations"] <= 3
    assert _probed(search)[-1] == pytest.approx(1e-10)
    assert search["warm_edge_probe"]["steer"] == "smaller_penalty"
    assert search["stopped_on"] == "search_floor_reached"
    done = [e for e in events if e.get("kind") == "budget_search_done"]
    assert done[0]["stopped_on"] == "search_floor_reached"
    _assert_no_repeated_probe(search)

    # A budget between the floor and the edge: the floor steers back, and
    # the search interpolates between the two measured ends.
    below_edge = -11.5 + 4.0 * (np.arange(n) + 0.5) / n
    settled, _ = _stub_budget_search(
        monkeypatch, below_edge, l0_lambda=2.5e-8, budget_search_bracket=_UK_BRACKET
    )
    search = settled.options["budget_search"]
    assert _probed(search)[2] == pytest.approx(1e-10)
    assert search["stopped_on"] == "acceptable_within_tolerance"
    assert search["selected_feasible"] is True
    assert search["evaluations"] <= 6
    _assert_no_repeated_probe(search)

    # Every gate stays open up to 10^2.3, beyond the ceiling of 10.
    ceiling = 2.3 + 2.2 * (np.arange(n) + 0.5) / n
    result, _ = _stub_budget_search(
        monkeypatch, ceiling, l0_lambda=1e-3, budget_search_bracket=_UK_BRACKET
    )
    search = result.options["budget_search"]
    assert search["evaluations"] <= 3
    assert _probed(search)[-1] == pytest.approx(10.0)
    assert search["stopped_on"] == "search_ceiling_reached"
    _assert_no_repeated_probe(search)


def test_out_of_bracket_warm_starts_and_malformed_brackets_are_refused(monkeypatch):
    import numpy as np

    thresholds = -7.12 + 8.0 * (np.arange(1000) + 0.5) / 1000
    for warm in (2.5e-8, 20.0):
        with pytest.raises(ValueError, match="outside the L0 search bracket"):
            _stub_budget_search(monkeypatch, thresholds, l0_lambda=warm)
    with pytest.raises(ValueError, match="outside the L0 search bracket"):
        _stub_budget_search(
            monkeypatch, thresholds, l0_lambda=5e-11, budget_search_bracket=_UK_BRACKET
        )
    for bracket in (
        (1e-7,),
        (10.0, 1e-7),
        (0.0, 1.0),
        (float("nan"), 1.0),
        (1e-10, float("inf")),
        (True, 10.0),
        "1e-10,10",
    ):
        with pytest.raises(ValueError, match="budget_search_bracket"):
            _stub_budget_search(monkeypatch, thresholds, budget_search_bracket=bracket)
    with pytest.raises(ValueError, match="budget_search_bracket"):
        _stub_budget_search(
            monkeypatch,
            thresholds,
            target_records=None,
            budget_basis="nonzero_count",
            feasible_draw_pi_hi=None,
            budget_search_bracket=_UK_BRACKET,
        )
    # The default bracket is unchanged: the cold path still opens at 1e-3.
    cold, _ = _stub_budget_search(monkeypatch, thresholds)
    search = cold.options["budget_search"]
    assert search["search_bracket"] == {"lower": 1e-7, "upper": 10.0}
    assert search["probes"][0]["l0_lambda"] == 1e-3
