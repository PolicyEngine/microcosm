"""Finite hand-computed controls for the pooled paired empirical candidate."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.fit.joint_empirical import (
    JointEmpiricalModel,
    JointTransport,
    SupportRequirements,
    SupportThreshold,
    draw_joint_empirical,
    fit_joint_empirical,
)
from microcosm.graph import keyed_uniform


def policy(**changes):
    one = SupportThreshold(1, 1, 1.0, 1.0)
    return replace(
        SupportRequirements(
            "invented-four-households", "test", one, one, one, (0, 1, 2, 3)
        ),
        **changes,
    )


def inputs():
    frame = pd.DataFrame({"O": [0.0, 10.0, 0.0, 30.0], "D": [0.0, 0.0, 20.0, 40.0]})
    return frame, [(i,) for i in range(4)], [("h", i) for i in range(4)]


def fitted(weights=(1.0, 2.0, 3.0, 4.0), **kwargs):
    frame, keys, households = inputs()
    return fit_joint_empirical(
        frame,
        targets=("O", "D"),
        donor_keys=keys,
        household_keys=households,
        support=kwargs.pop("support", policy()),
        weights=weights,
        **kwargs,
    )


def test_unequal_weights_define_joint_patterns_and_paired_amounts():
    model = fitted()
    np.testing.assert_allclose(
        model.diagnostics["pattern_probabilities"], [0.1, 0.2, 0.3, 0.4]
    )
    result = draw_joint_empirical(
        model,
        pattern_uniforms=[0, 0.2, 0.4, 0.9],
        donor_uniforms=[0, 0, 0, 0],
        transport=JointTransport(1, (1, 1)),
    )
    np.testing.assert_array_equal(result.values, [[0, 0], [10, 0], [0, 20], [30, 40]])
    assert result.donor_keys == (None, (1,), (2,), (3,))
    assert result.values.flags.writeable is False


def test_zero_stress_is_imputation_and_keeps_qualified_model():
    model = fitted()
    result = draw_joint_empirical(
        model,
        pattern_uniforms=[0, 0.999],
        donor_uniforms=[0, 0.999],
        transport=JointTransport(0, (2, 3)),
    )
    np.testing.assert_array_equal(result.values, np.zeros((2, 2)))
    assert result.donor_keys == (None, None)
    assert result.model_sha256 == model.sha256


def test_positive_amount_factors_preserve_joint_pattern():
    result = draw_joint_empirical(
        fitted(),
        pattern_uniforms=[0.9],
        donor_uniforms=[0.8],
        transport=JointTransport(1, (2, 3)),
    )
    np.testing.assert_array_equal(result.values, [[60, 120]])


@pytest.mark.parametrize(
    "transport",
    [
        JointTransport(-1, (1, 1)),
        JointTransport(2, (1, 1)),
        JointTransport(1, (0, 1)),
        JointTransport(True, (1, 1)),
        JointTransport(float("nan"), (1, 1)),
    ],
)
def test_invalid_transport_refuses_without_clipping(transport):
    with pytest.raises((ValueError, TypeError)):
        draw_joint_empirical(
            fitted(), pattern_uniforms=[0.5], donor_uniforms=[0.5], transport=transport
        )


def test_permuted_donors_and_typed_large_ids_have_same_model_bytes():
    frame, keys, households = inputs()
    keys[0], keys[1] = (2**53 + 1,), (str(2**53 + 1),)
    common = dict(targets=("O", "D"), support=policy())
    first = fit_joint_empirical(
        frame,
        donor_keys=keys,
        household_keys=households,
        weights=[1, 2, 3, 4],
        **common,
    )
    order = [3, 1, 0, 2]
    second = fit_joint_empirical(
        frame.iloc[order],
        donor_keys=[keys[i] for i in order],
        household_keys=[households[i] for i in order],
        weights=[4, 2, 1, 3],
        **common,
    )
    assert first.to_bytes() == second.to_bytes()
    assert (
        JointEmpiricalModel.from_bytes(first.to_bytes()).to_bytes() == first.to_bytes()
    )


def test_shared_keyed_uniforms_are_invariant_to_batches_and_other_recipients():
    model = fitted()
    stream = ("sha256-u53-v1", "joint-test", 0, 91)

    def draw(keys):
        return draw_joint_empirical(
            model,
            pattern_uniforms=keyed_uniform(
                stream=stream, keys=[(*k, "pattern") for k in keys]
            ),
            donor_uniforms=keyed_uniform(
                stream=stream, keys=[(*k, "donor") for k in keys]
            ),
            transport=JointTransport(1, (1, 1)),
        ).values

    keys = [(2**53 + 1,), ("x",), (9,)]
    np.testing.assert_array_equal(
        draw(keys), np.vstack([draw(keys[:1]), draw(keys[1:])])
    )
    np.testing.assert_array_equal(draw(keys), draw([("other",), *keys])[1:])
    np.testing.assert_array_equal(draw(keys)[::-1], draw(keys[::-1]))


def test_weighted_pair_draw_uses_one_donor_for_both_targets():
    frame = pd.DataFrame({"O": [1.0, 9.0], "D": [100.0, 900.0]})
    one = SupportThreshold(1, 1, 1, 1)
    support = SupportRequirements("paired", "test", one, one, one, (3,))
    model = fit_joint_empirical(
        frame,
        targets=("O", "D"),
        donor_keys=[(1,), (2,)],
        household_keys=[("a",), ("b",)],
        support=support,
        weights=[1, 3],
    )
    result = draw_joint_empirical(
        model,
        pattern_uniforms=[0, 0.8, 0.2, 0.9],
        donor_uniforms=[0, 0.249, 0.25, 0.999],
        transport=JointTransport(1, (1, 1)),
    )
    np.testing.assert_array_equal(
        result.values, [[1, 100], [1, 100], [9, 900], [9, 900]]
    )


def test_exact_dyadic_pattern_boundaries_skip_zero_width_intervals():
    model = fitted(weights=(4, 1, 1, 2))
    result = draw_joint_empirical(
        model,
        pattern_uniforms=[0, 0.25, 0.5, 0.999],
        donor_uniforms=[0] * 4,
        transport=JointTransport(2, (1, 1)),
    )
    np.testing.assert_array_equal(result.patterns, [1, 2, 3, 3])


def test_effective_sizes_count_original_households_and_persons_separately():
    frame, keys, households = inputs()
    households[1] = households[0]
    model = fit_joint_empirical(
        frame,
        targets=("O", "D"),
        donor_keys=keys,
        household_keys=households,
        support=policy(),
        weights=[1, 2, 3, 4],
    )
    stats = model.diagnostics["overall"]
    assert stats["rows"] == 4 and stats["households"] == 3
    assert stats["person_ess"] == pytest.approx(100 / 30)
    assert stats["household_ess"] == pytest.approx(100 / 34)


def test_zero_mass_rows_preserve_raw_diagnostics_without_becoming_support():
    model = fitted(weights=(0, 2, 3, 4), support=policy(required_patterns=(1, 2, 3)))
    diagnostics = model.diagnostics
    assert diagnostics["raw_rows"] == 4 and diagnostics["zero_weight_rows"] == 1
    assert diagnostics["raw_zero_pair_rows"] == 1
    assert diagnostics["positive_weight_zero_pair_rows"] == 0
    assert diagnostics["absent_patterns"] == [0]


@pytest.mark.parametrize(
    "weights",
    [
        [0] * 4,
        [-1, 1, 1, 1],
        [float("inf"), 1, 1, 1],
        [float("nan"), 1, 1, 1],
        [1, 2],
        [True] * 4,
        "none",
    ],
)
def test_invalid_or_unweighted_fit_is_refused(weights):
    with pytest.raises((ValueError, TypeError)):
        fitted(weights=weights)


@pytest.mark.parametrize("scope", ["overall", "positive", "pattern"])
@pytest.mark.parametrize("name", ["rows", "households", "person_ess", "household_ess"])
def test_declared_support_minimum_is_enforced(scope, name):
    threshold = replace(getattr(policy(), scope), **{name: 10})
    with pytest.raises(ValueError, match="SUPPORT:" + scope):
        fitted(support=policy(**{scope: threshold}))


def test_absent_required_pattern_refuses_and_r_zero_cannot_bypass_model_support():
    frame, keys, homes = inputs()
    frame.loc[:, "O"] = 0
    frame.loc[:, "D"] = 0
    with pytest.raises(ValueError, match="SUPPORT:positive"):
        fit_joint_empirical(
            frame,
            targets=("O", "D"),
            donor_keys=keys,
            household_keys=homes,
            support=policy(),
            weights=[1, 2, 3, 4],
        )
    with pytest.raises(ValueError, match="SUPPORT:pattern0"):
        fitted(weights=[0, 1, 1, 1])


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_key",
        "boolean_key",
        "float_key",
        "negative",
        "nonfinite",
        "boolean_target",
    ],
)
def test_invalid_donor_identity_or_amount_refuses(change):
    frame, keys, homes = inputs()
    if change == "duplicate_key":
        keys[1] = keys[0]
    elif change == "boolean_key":
        keys[0] = (True,)
    elif change == "float_key":
        keys[0] = (1.0,)
    elif change == "negative":
        frame.loc[0, "O"] = -1
    elif change == "nonfinite":
        frame.loc[0, "O"] = float("nan")
    else:
        frame["O"] = True
    with pytest.raises((ValueError, TypeError)):
        fit_joint_empirical(
            frame,
            targets=("O", "D"),
            donor_keys=keys,
            household_keys=homes,
            support=policy(),
            weights=[1, 2, 3, 4],
        )


@pytest.mark.parametrize(
    "u,v",
    [
        ([1], [0.1]),
        ([-0.1], [0.1]),
        ([float("nan")], [0.1]),
        ([True], [0.1]),
        ([0.1], [float("inf")]),
        ([0.1], []),
        ([[0.1]], [0.1]),
    ],
)
def test_invalid_uniform_coordinates_refuse(u, v):
    with pytest.raises(ValueError):
        draw_joint_empirical(
            fitted(),
            pattern_uniforms=u,
            donor_uniforms=v,
            transport=JointTransport(1, (1, 1)),
        )


def test_lost_positive_probability_interval_refuses():
    frame = pd.DataFrame({"O": [1.0, 2.0], "D": [1.0, 2.0]})
    one = SupportThreshold(1, 1, 1, 1)
    support = SupportRequirements("lost-mass", "test", one, one, one, (3,))
    with pytest.raises(ValueError, match="LOST_POSITIVE_INTERVAL"):
        fit_joint_empirical(
            frame,
            targets=("O", "D"),
            donor_keys=[(1,), (2,)],
            household_keys=[(1,), (2,)],
            support=support,
            weights=[1, 1e-300],
        )


@pytest.mark.parametrize("factor", [1e308, 5e-324])
def test_amount_overflow_and_lost_positive_underflow_refuse(factor):
    frame = pd.DataFrame({"O": [1e-300 if factor < 1 else 100.0], "D": [100.0]})
    one = SupportThreshold(1, 1, 1, 1)
    model = fit_joint_empirical(
        frame,
        targets=("O", "D"),
        donor_keys=[(1,)],
        household_keys=[(1,)],
        support=SupportRequirements("amount-range", "test", one, one, one, (3,)),
        weights=[1],
    )
    with pytest.raises(ValueError, match="AMOUNT_TRANSPORT_RANGE"):
        draw_joint_empirical(
            model,
            pattern_uniforms=[0.5],
            donor_uniforms=[0.5],
            transport=JointTransport(1, (factor, 1)),
        )


def test_model_json_is_strict_data_and_summary_is_not_trusted():
    import json

    model = fitted()
    document = json.loads(model.to_bytes())
    document["claimed_verified"] = True
    with pytest.raises(ValueError, match="MODEL_DOCUMENT"):
        JointEmpiricalModel.from_bytes(json.dumps(document).encode())
    with pytest.raises(ValueError, match="DUPLICATE_JSON_KEY"):
        JointEmpiricalModel.from_bytes(b'{"protocol":1,"protocol":2}')
    document = json.loads(model.to_bytes())
    document["donors"][0]["key"][0] = ["int", "01"]
    with pytest.raises(ValueError, match="IDENTITY_INTEGER"):
        JointEmpiricalModel.from_bytes(json.dumps(document).encode())


def test_frame_front_door_resolves_original_household_design_weights():
    from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

    people, keys, _ = inputs()
    people["person_id"] = np.arange(1, 5, dtype=np.int64)
    people["person_household_id"] = [1, 1, 2, 2]
    households = pd.DataFrame({"household_id": [1, 2]})
    frame = Frame(
        {"person": people, "household": households},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.array([1.0, 3.0]), WeightKind.DESIGN)},
    )
    model = fit_joint_empirical(
        frame,
        targets=("O", "D"),
        donor_keys=keys,
        household_keys=[(1,), (1,), (2,), (2,)],
        support=policy(),
    )
    assert model.diagnostics["weight_kind"] == "design"
    np.testing.assert_allclose(
        model.diagnostics["pattern_probabilities"], [1 / 8, 1 / 8, 3 / 8, 3 / 8]
    )
    with pytest.raises(ValueError):
        fit_joint_empirical(
            frame,
            targets=("O", "D"),
            donor_keys=keys,
            household_keys=[(1,), (1,), (2,), (2,)],
            support=policy(),
            weights="calibrated",
        )


def test_dataframe_weight_default_is_not_silently_unweighted():
    frame, keys, households = inputs()
    with pytest.raises(ValueError, match="typed weights"):
        fit_joint_empirical(
            frame,
            targets=("O", "D"),
            donor_keys=keys,
            household_keys=households,
            support=policy(),
        )


@pytest.mark.parametrize("zero_weight", [2**-53, 2**-54, 2**-55])
def test_reference_retains_small_supported_zero_interval(zero_weight):
    frame = pd.DataFrame({"O": [0.0, 1.0], "D": [0.0, 1.0]})
    model = fit_joint_empirical(
        frame,
        targets=("O", "D"),
        donor_keys=[(0,), (1,)],
        household_keys=[(0,), (1,)],
        weights=[zero_weight, 1.0],
        support=policy(required_patterns=(0, 3)),
    )
    # Both inputs are exact powers of two. The cumulative endpoint rounds to
    # one, while its leading zero-pattern interval is representable.
    boundary = zero_weight
    result = draw_joint_empirical(
        model,
        pattern_uniforms=[
            0.0,
            np.nextafter(boundary, 0.0),
            boundary,
            np.nextafter(boundary, 1.0),
        ],
        donor_uniforms=[0.0] * 4,
        transport=JointTransport(1.0, (1.0, 1.0)),
    )
    np.testing.assert_array_equal(result.patterns, [0, 0, 3, 3])
    np.testing.assert_array_equal(result.values, [[0, 0], [0, 0], [1, 1], [1, 1]])


@pytest.mark.parametrize(
    "measure", ["rows", "households", "person_ess", "household_ess"]
)
def test_present_nonrequired_zero_pattern_must_meet_each_minimum(measure):
    frame = pd.DataFrame({"O": [0.0, 1.0, 2.0], "D": [0.0, 1.0, 2.0]})
    threshold = replace(policy().pattern, **{measure: 2})
    with pytest.raises(ValueError, match="SUPPORT:pattern0:" + measure):
        fit_joint_empirical(
            frame,
            targets=("O", "D"),
            donor_keys=[(0,), (1,), (2,)],
            household_keys=[(0,), (1,), (2,)],
            weights=[1.0, 1.0, 1.0],
            support=policy(pattern=threshold, required_patterns=(3,)),
        )


@pytest.mark.parametrize("factor", [0.5, 2**20, 2**-20])
def test_common_representable_weight_scaling_preserves_joint_law(factor):
    original = fitted()
    scaled = fitted(weights=[factor * x for x in [1.0, 2.0, 3.0, 4.0]])
    for name in ("pattern_probabilities", "absent_patterns"):
        assert original.diagnostics[name] == scaled.diagnostics[name]
    for name in ("person_ess", "household_ess", "rows", "households"):
        assert (
            original.diagnostics["overall"][name] == scaled.diagnostics["overall"][name]
        )
    arguments = dict(
        pattern_uniforms=[0.0, 0.15, 0.45, 0.95],
        donor_uniforms=[0.0, 0.2, 0.5, 0.9],
        transport=JointTransport(1, (1, 1)),
    )
    np.testing.assert_array_equal(
        draw_joint_empirical(original, **arguments).values,
        draw_joint_empirical(scaled, **arguments).values,
    )
    assert (
        original.to_bytes() != scaled.to_bytes()
    )  # Original weight provenance stays exact.


@pytest.mark.parametrize("destination", ["target", "weight_vector", "weight_column"])
@pytest.mark.parametrize("sign", [1, -1])
def test_wider_float_inputs_are_refused_before_narrowing(destination, sign):
    frame, keys, households = inputs()
    extended = np.dtype(np.longdouble).itemsize > 8
    raw = np.array([1, 2, 3, 4], dtype=np.longdouble)
    if extended:
        raw[0] = sign * np.longdouble("1e-4000")
    weights = [1.0, 2.0, 3.0, 4.0]
    if destination == "target":
        frame["O"] = raw
    elif destination == "weight_vector":
        weights = raw
    else:
        frame["weight"] = raw
        weights = "weight"
    arguments = dict(
        targets=("O", "D"),
        donor_keys=keys,
        household_keys=households,
        weights=weights,
        support=policy(required_patterns=(1, 3))
        if destination == "target"
        else policy(),
    )
    if extended:
        with pytest.raises(ValueError, match="DTYPE"):
            fit_joint_empirical(frame, **arguments)
    else:
        # On platforms where longdouble is binary64 this is supported input,
        # not a skipped case or a claim to exercise extended-exponent refusal.
        assert fit_joint_empirical(frame, **arguments).diagnostics["raw_rows"] == 4
