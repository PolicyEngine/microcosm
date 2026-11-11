"""is_apprentice on the UK spine: employees aged 21 to 24, realized to a per-head rate."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime import frs_person_draws
from microcosm.build.uk_runtime.frs_person_draws import (
    APPRENTICE_OUTPUT,
    APPRENTICE_RATE_KEY,
    UKFRSPersonDrawsStageTransform,
    apprentice_relief_schedule,
    derive_is_apprentice,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.receipt_counts import (
    UK_WITHHELD,
    distinct_count,
    own_source_households,
    support_band,
)
from test_support.microcosm_build.uk_clone_first_frames import (
    GEOGRAPHIC_RADIX,
    clone_first_frame,
)
from test_support.microcosm_build.uk_frs_stochastic import _Contract


def _apprentices(person, **options):
    """The draw on a raw frame, where each household is its own source."""

    return derive_is_apprentice(
        person, person_sources=own_source_households(person), **options
    )


_STATUSES = ("FT_EMPLOYED", "PT_EMPLOYED", "FT_SELF_EMPLOYED", "STUDENT")


def _people(n_per_cell: int = 2000) -> pd.DataFrame:
    """Every age from 18 to 30 crossed with four statuses, one household each."""

    ages = np.arange(18, 31)
    rows = [
        (age, status) for age in ages for status in _STATUSES for _ in range(n_per_cell)
    ]
    frame = pd.DataFrame(rows, columns=["age", "employment_status"])
    frame.insert(0, "person_id", np.arange(1, len(frame) + 1))
    frame.insert(1, "person_household_id", frame["person_id"])
    return frame


def _small(ages, statuses, *, households=None) -> pd.DataFrame:
    ids = np.arange(11, 11 + len(ages))
    return pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids if households is None else households,
            "age": ages,
            "employment_status": statuses,
        }
    )


def test_only_employees_aged_21_to_24_are_flagged() -> None:
    person = _people()

    flags = _apprentices(person, rate=0.06).flags

    age = person["age"].to_numpy()
    employee = person["employment_status"].isin(["FT_EMPLOYED", "PT_EMPLOYED"])
    eligible = (age >= 21) & (age <= 24) & employee.to_numpy()
    assert flags.dtype == bool
    assert not flags[~eligible].any()
    assert flags[eligible].any()


def test_flagged_weight_is_nearest_the_per_head_target() -> None:
    person = _people()
    rng = np.random.default_rng(0)
    weights = rng.uniform(100.0, 3000.0, len(person))
    rate = 0.06

    draw = _apprentices(person, rate=rate, person_weights=weights)

    age = person["age"].to_numpy()
    in_ages = (age >= 21) & (age <= 24)
    target = rate * weights[in_ages].sum()
    realized = weights[draw.flags].sum()
    # The walk stops at the cumulative weight nearest the target, so it misses
    # by at most half the last person's weight.
    assert abs(realized - target) <= weights.max() / 2
    receipt = draw.receipt
    assert receipt["rate_key"] == APPRENTICE_RATE_KEY
    assert receipt["population_mass"] == pytest.approx(weights[in_ages].sum())
    assert receipt["target"] == pytest.approx(target)
    assert receipt["realized_mass"] == pytest.approx(realized)
    assert receipt["realization_deviation"] == pytest.approx(realized - target)
    assert receipt["cap_bound"] is False
    assert receipt["eligible_support"] == "30+"
    assert receipt["flagged_support"] == "30+"


def test_people_are_flagged_in_the_order_of_their_identity_uniform() -> None:
    person = _small(
        [22, 22, 23, 24, 40, 23],
        [
            "FT_EMPLOYED",
            "STUDENT",
            "PT_EMPLOYED",
            "FT_EMPLOYED",
            "FT_EMPLOYED",
            "FT_EMPLOYED",
        ],
    )
    weights = np.array([1.0, 1.0, 1.0, 1.0, 5.0, 1.0])
    # Five people aged 21 to 24 at 0.4 a head: a target of two people.
    draw = _apprentices(person, rate=0.4, person_weights=weights)

    draws = stable_identity_uniforms(
        person["person_id"].to_numpy(), seed=0, salt=APPRENTICE_OUTPUT
    )
    eligible = np.array([True, False, True, True, False, True])
    first_two = np.argsort(np.where(eligible, draws, np.inf))[:2]
    expected = np.zeros(len(person), dtype=bool)
    expected[first_two] = True
    np.testing.assert_array_equal(draw.flags, expected)
    reordered = _apprentices(
        person.iloc[::-1].reset_index(drop=True),
        rate=0.4,
        person_weights=weights[::-1],
    )
    np.testing.assert_array_equal(reordered.flags, expected[::-1])


def test_nobody_is_flagged_when_the_target_is_nearer_zero() -> None:
    person = _small([22, 30], ["FT_EMPLOYED", "STUDENT"])

    # A target of 0.1 people sits nearer zero than the one employee's weight.
    draw = _apprentices(person, rate=0.1, person_weights=np.ones(2))

    assert not draw.flags.any()
    assert draw.receipt["realized_mass"] == 0.0
    assert draw.receipt["cap_bound"] is False


def test_a_short_eligible_weight_flags_every_eligible_employee() -> None:
    person = _small([21, 22, 23], ["FT_EMPLOYED", "UNEMPLOYED", "UNEMPLOYED"])

    draw = _apprentices(person, rate=0.9)

    np.testing.assert_array_equal(draw.flags, [True, False, False])
    assert draw.receipt["cap_bound"] is True


def test_no_eligible_employee_flags_nobody() -> None:
    person = _small([22, 30], ["STUDENT", "FT_EMPLOYED"])

    draw = _apprentices(person, rate=0.06)

    assert not draw.flags.any()
    assert draw.receipt["eligible_support"] == 0
    assert draw.receipt["cap_bound"] is True


def test_receipt_withholds_a_flagged_weight_on_fewer_than_three_households() -> None:
    person = _small([22] * 20, ["FT_EMPLOYED"] * 20)

    draw = _apprentices(person, rate=0.1)

    assert int(draw.flags.sum()) == 2
    assert draw.receipt["flagged_support"] == "<3"
    assert draw.receipt["realized_mass"] == "<3"
    assert draw.receipt["realization_deviation"] == "<3"
    assert draw.receipt["population_mass"] == 20.0


def test_receipt_withholds_a_population_on_fewer_than_three_households() -> None:
    # Six people aged 21 to 24 in two households, with a one-person target.
    person = _small(
        [22, 23, 21, 24, 22, 23],
        ["FT_EMPLOYED"] * 6,
        households=[1, 1, 1, 2, 2, 2],
    )

    draw = _apprentices(person, rate=1 / 6)

    receipt = draw.receipt
    assert receipt["population_support"] == "<3"
    for key in (
        "population_mass",
        "eligible_mass",
        "target",
        "realized_mass",
        "realization_deviation",
    ):
        assert receipt[key] == "<3", key


def test_receipt_counts_its_support_in_geographic_clone_zero() -> None:
    """Each geographic clone flags its own employees, so the receipt counts
    its support in clone 0's people, as one clone counts it, and withholds
    the figures clone 0 carries on fewer than three households; without the
    clone flags every person counts, as before (microcosm#931)."""

    from microcosm.build.uk_runtime.receipt_counts import support_band

    base = _small([22] * 20, ["FT_EMPLOYED"] * 20)
    clones = []
    for k in range(3):
        clone = base.copy()
        clone["person_id"] = base["person_id"] + k * 10**9
        clone["person_household_id"] = base["person_household_id"] + k * 10**9
        clones.append(clone)
    person = pd.concat(clones, ignore_index=True)
    sources = np.tile(base["person_household_id"].to_numpy(), 3)
    clone_zero = np.repeat([True, False, False], len(base))
    weights = np.full(len(person), 1 / 3)

    draw = derive_is_apprentice(
        person,
        rate=0.05,
        person_weights=weights,
        person_sources=sources,
        person_clone_zero=clone_zero,
    )
    in_clone_zero = len(set(sources[draw.flags & clone_zero]))
    over_clones = len(set(sources[draw.flags]))
    # The premise: clone 0 flags fewer than three households, every clone
    # together three or more.
    assert support_band(in_clone_zero) == "<3"
    assert support_band(over_clones) != "<3"
    assert draw.receipt["flagged_support"] == "<3"
    assert draw.receipt["realized_mass"] == "<3"
    assert draw.receipt["population_support"] == support_band(20)
    every = derive_is_apprentice(
        person, rate=0.05, person_weights=weights, person_sources=sources
    )
    assert every.receipt["flagged_support"] == support_band(over_clones)
    assert every.receipt["realized_mass"] != "<3"
    np.testing.assert_array_equal(every.flags, draw.flags)
    with pytest.raises(ValueError, match="one flag per person"):
        derive_is_apprentice(
            person,
            rate=0.05,
            person_sources=sources,
            person_clone_zero=clone_zero[:-1],
        )


def test_receipt_withholds_what_other_clones_flag_on_one_or_two_households() -> None:
    """The masses are taken over every clone, so a mass one or two source
    households carry in the other clones is withheld though clone 0 flags no
    one, and the printed support stays clone 0's (microcosm#931)."""

    base = _small([22] * 20, ["FT_EMPLOYED"] * 20).assign(employment_income=20_000.0)
    clones = []
    for k in range(3):
        clone = base.copy()
        clone["person_id"] = base["person_id"] + k * 10**9
        clone["person_household_id"] = base["person_household_id"] + k * 10**9
        if k == 0:
            # Clone 0 has no employee, so it flags no one.
            clone["employment_status"] = "STUDENT"
        clones.append(clone)
    person = pd.concat(clones, ignore_index=True)
    sources = np.tile(base["person_household_id"].to_numpy(), 3)
    clone_zero = np.repeat([True, False, False], len(base))
    weights = np.ones(len(person))

    draw = derive_is_apprentice(
        person,
        rate=1 / len(person),
        person_weights=weights,
        relief_schedule={
            "rate": 0.15,
            "secondary_threshold": 5_000.0,
            "apprentice_upper_secondary_threshold": 50_270.0,
        },
        person_sources=sources,
        person_clone_zero=clone_zero,
    )
    # The premise: clone 0 flags no one, the other clones one household.
    assert not draw.flags[clone_zero].any()
    assert len(set(sources[draw.flags])) == 1
    assert draw.receipt["flagged_support"] == 0
    assert draw.receipt["eligible_support"] == 0
    for key in ("realized_mass", "realization_deviation", "implied_relief"):
        assert draw.receipt[key] == "<3", key
    # Clone 0's population and the population over every clone are the same
    # twenty households, so the population's figures are given.
    assert draw.receipt["population_mass"] == float(len(person))


# --- the walk within each geographic clone (microcosm#931) -----------------------

#: Every rate per head from 0 to 1 in steps of 0.05.
_CLONE_RATES = tuple(step / 20 for step in range(21))
#: The receipt's fields that do not move with K: its supports, counted in
#: clone 0's people, and whether the draw is cap-bound, which compares the
#: eligible weight and the target over every clone, one clone's to rounding.
_K_INVARIANT_FIELDS = (
    "population_support",
    "eligible_support",
    "flagged_support",
    "cap_bound",
)


def _clone_base(households: int = 90) -> tuple[pd.DataFrame, np.ndarray]:
    """Invented people aged 18 to 30 of the four statuses in households of
    one to three, each with its household's weight."""

    rows = [
        (
            household * 1000 + member,
            household,
            18 + (7 * household + 3 * member) % 13,
            _STATUSES[(household + member) % 4],
        )
        for household in range(1, households + 1)
        for member in range(1, 2 + household % 3)
    ]
    person = pd.DataFrame(
        rows, columns=["person_id", "person_household_id", "age", "employment_status"]
    )
    household_weights = np.random.default_rng(931).uniform(100.0, 3000.0, households)
    return person, household_weights[person["person_household_id"].to_numpy() - 1]


def _as_clones(
    person: pd.DataFrame, weights: np.ndarray, clones: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """``clones`` geographic clones of ``person``, clone-major, as the
    clone-first spine makes them: clone k's ids plus k x the geographic
    radix, each at ``w / clones`` of its weight. Returns the people, their
    weights, their clone index and their true source households."""

    people = pd.concat(
        [
            person.assign(
                person_id=person["person_id"] + k * GEOGRAPHIC_RADIX,
                person_household_id=person["person_household_id"]
                + k * GEOGRAPHIC_RADIX,
            )
            for k in range(clones)
        ],
        ignore_index=True,
    )
    return (
        people,
        np.tile(weights / clones, clones),
        np.repeat(np.arange(clones, dtype=np.int64), len(person)),
        np.tile(person["person_household_id"].to_numpy(), clones),
    )


@pytest.mark.parametrize("clones", [2, 4])
def test_clone_zero_walks_as_one_clone_at_two_and_four_clones(clones: int) -> None:
    """Each clone walks its own employees to the rate times its own weight
    aged 21 to 24. Clone 0's people are one clone's at their weights over K,
    which a power of two divides exactly, so at every rate clone 0 takes the
    one-clone flags and every clone the flags its own people take alone. The
    supports, counted in clone 0's people, and whether the draw is cap-bound
    are one clone's; the masses are summed over the clones. One walk over
    every clone moves clone 0's flags at some rates (microcosm#931)."""

    base, weights = _clone_base()
    sources = base["person_household_id"].to_numpy()
    person, clone_weights, index, clone_sources = _as_clones(base, weights, clones)
    pooled_moves = 0
    for rate in _CLONE_RATES:
        one = derive_is_apprentice(
            base, rate=rate, person_weights=weights, person_sources=sources
        )
        many = derive_is_apprentice(
            person,
            rate=rate,
            person_weights=clone_weights,
            person_sources=clone_sources,
            person_clone_zero=index == 0,
            person_clone_index=index,
        )
        assert many.flags[index == 0].tolist() == one.flags.tolist(), rate
        for clone in range(clones):
            rows = index == clone
            alone = derive_is_apprentice(
                person.loc[rows].reset_index(drop=True),
                rate=rate,
                person_weights=clone_weights[rows],
                person_sources=clone_sources[rows],
            )
            assert many.flags[rows].tolist() == alone.flags.tolist(), (rate, clone)
        for key in _K_INVARIANT_FIELDS:
            assert many.receipt[key] == one.receipt[key], (rate, key)
        for key in ("population_mass", "eligible_mass", "target"):
            assert many.receipt[key] == pytest.approx(one.receipt[key], rel=1e-12)
        pooled = derive_is_apprentice(
            person,
            rate=rate,
            person_weights=clone_weights,
            person_sources=clone_sources,
            person_clone_zero=index == 0,
        )
        pooled_moves += pooled.flags[index == 0].tolist() != one.flags.tolist()
    assert pooled_moves


def test_an_exact_tie_nests_clone_zero_at_a_power_of_two_only() -> None:
    """Four employees of weight 1 at a rate of 0.625 put the one-clone walk's
    target, 2.5, midway between flagging two of them and three; the walk
    flags three. Dividing every weight by a power of two scales the tie
    exactly, so clone 0 keeps those flags at K = 2, 4, 8 and 16, which the
    engine tier's exact K = 4 class rests on. At K = 3, 5, 6, 7 and 15,
    w / K rounds and can break the tie by one person, the near-tie
    derive_is_apprentice names (microcosm#931)."""

    base = _small([22] * 4, ["FT_EMPLOYED"] * 4)
    weights = np.ones(len(base))
    sources = base["person_household_id"].to_numpy()
    one = derive_is_apprentice(
        base, rate=0.625, person_weights=weights, person_sources=sources
    )
    assert int(one.flags.sum()) == 3
    moved = {}
    for clones in (2, 3, 4, 5, 6, 7, 8, 15, 16):
        person, clone_weights, index, clone_sources = _as_clones(base, weights, clones)
        many = derive_is_apprentice(
            person,
            rate=0.625,
            person_weights=clone_weights,
            person_sources=clone_sources,
            person_clone_index=index,
        )
        moved[clones] = int((many.flags[index == 0] != one.flags).sum())
    assert all(moved[k] == 0 for k in (2, 4, 8, 16)), moved
    assert all(moved[k] <= 1 for k in (3, 5, 6, 7, 15)), moved
    assert any(moved[k] for k in (3, 5, 6, 7, 15)), moved


def test_the_supports_are_one_clones_at_three_clones() -> None:
    """At three clones, where dividing a weight by K rounds, each support
    band and whether the draw is cap-bound are one clone's at every rate:
    clone 0's walk compares one clone's prefixes to rounding, the supports
    are counted in clone 0's people in true source households, and the cap
    bound compares the eligible weight and the target over every clone,
    which are one clone's to rounding. Without person_clone_zero the
    supports are counted where the clone index is 0; the flagged sources
    counted over every clone would read another band at some rates."""

    base, weights = _clone_base()
    sources = base["person_household_id"].to_numpy()
    person, clone_weights, index, clone_sources = _as_clones(base, weights, 3)
    arguments = {"person_weights": clone_weights, "person_sources": clone_sources}
    bands_over_every_clone_differ = 0
    for rate in _CLONE_RATES:
        one = derive_is_apprentice(
            base, rate=rate, person_weights=weights, person_sources=sources
        )
        many = derive_is_apprentice(
            person, rate=rate, **arguments, person_clone_index=index
        )
        for key in _K_INVARIANT_FIELDS:
            assert many.receipt[key] == one.receipt[key], (rate, key)
        explicit = derive_is_apprentice(
            person,
            rate=rate,
            **arguments,
            person_clone_zero=index == 0,
            person_clone_index=index,
        )
        assert explicit.receipt == many.receipt
        over_every_clone = support_band(
            distinct_count(clone_sources, many.flags, clone_weights)
        )
        bands_over_every_clone_differ += (
            over_every_clone != one.receipt["flagged_support"]
        )
    assert bands_over_every_clone_differ


def _walk_before_clones(
    person: pd.DataFrame, *, rate: float, weights: np.ndarray
) -> tuple[np.ndarray, dict[str, object]]:
    """The flags and the receipt's figures as derive_is_apprentice took them
    before the target was met within each clone: one walk over every row."""

    age = pd.to_numeric(person["age"], errors="coerce").to_numpy(dtype=float)
    in_ages = (age >= 21) & (age < 25)
    eligible = in_ages & (
        person["employment_status"]
        .astype(str)
        .isin(("FT_EMPLOYED", "PT_EMPLOYED"))
        .to_numpy()
    )
    draws = stable_identity_uniforms(
        person["person_id"].to_numpy(), seed=0, salt=APPRENTICE_OUTPUT
    )
    population_mass = float(weights[in_ages].sum())
    eligible_mass = float(weights[eligible].sum())
    target = float(rate) * population_mass
    threshold = frs_person_draws._nearest_target_threshold(
        target, draws=draws[eligible], weights=weights[eligible]
    )
    flags = eligible & (draws < threshold)
    realized = float(weights[flags].sum())
    return flags, {
        "population_mass": population_mass,
        "eligible_mass": eligible_mass,
        "target": target,
        "cap_bound": bool(target > 0.0 and eligible_mass <= target),
        "realized_mass": realized,
        "realization_deviation": realized - target,
    }


def _bits(value: object) -> object:
    return np.float64(value).tobytes() if isinstance(value, float) else value


@pytest.mark.parametrize("seed", range(4))
def test_one_clone_walks_bit_for_bit_as_before(seed: int) -> None:
    """Without a clone index, and with one of zeros in int64, int8 or uint16,
    the flags and the receipt's masses, target, cap_bound, realized mass and
    deviation are, bit for bit and type for type, the walk's over every row
    as before the count was taken within each clone, on invented frames of
    every size up to eighty people, with zero weights among them, at random
    rates."""

    rng = np.random.default_rng(seed)
    compared: dict[str, int] = {}
    for _ in range(50):
        n = int(rng.integers(0, 81))
        ids = rng.choice(10**6, size=n, replace=False) + 1
        person = pd.DataFrame(
            {
                "person_id": ids,
                "person_household_id": ids,
                "age": rng.integers(18, 31, size=n),
                "employment_status": rng.choice(_STATUSES, size=n),
            }
        )
        weights = rng.uniform(0.0, 3000.0, size=n)
        weights[rng.random(n) < 0.1] = 0.0
        rate = float(rng.uniform())
        flags, before = _walk_before_clones(person, rate=rate, weights=weights)
        for clone_index in (
            None,
            np.zeros(n, dtype=np.int64),
            np.zeros(n, dtype=np.int8),
            np.zeros(n, dtype=np.uint16),
        ):
            draw = derive_is_apprentice(
                person,
                rate=rate,
                person_weights=weights,
                person_sources=ids,
                person_clone_index=clone_index,
            )
            assert draw.flags.dtype == flags.dtype
            assert draw.flags.tobytes() == flags.tobytes()
            for key, value in before.items():
                got = draw.receipt[key]
                if got == UK_WITHHELD:
                    continue
                compared[key] = compared.get(key, 0) + 1
                assert type(got) is type(value), key
                assert _bits(got) == _bits(value), key
    assert set(compared) == set(before) and min(compared.values()) > 50, compared


def test_the_stage_walks_each_geographic_clone(no_engine) -> None:
    """The stage hands the walk each person's household clone: on the
    clone-first stand-in at four clones, each clone flags 480 employees, its
    own target, clone 0 the one-clone frame's, and the receipt's supports are
    one clone's. One walk over every clone would flag others in clone 0."""

    stage = UKFRSPersonDrawsStageTransform(contract=_Contract(), stage=None)
    one_clone = stage(clone_first_frame(_two_household_frame(), 1)).table("person")
    one_receipt = stage.checkpoint_metadata()["evidence"]["apprentice_receipt"]
    frame = clone_first_frame(_two_household_frame(), 4)
    four_clones = stage(frame).table("person")
    four_receipt = stage.checkpoint_metadata()["evidence"]["apprentice_receipt"]
    clone = (four_clones["person_id"] // GEOGRAPHIC_RADIX).to_numpy()
    clone_zero = clone == 0
    flags = four_clones[APPRENTICE_OUTPUT].to_numpy()
    assert flags[clone_zero].tolist() == one_clone[APPRENTICE_OUTPUT].tolist()
    assert [int(flags[clone == k].sum()) for k in range(4)] == [480] * 4
    for key in _K_INVARIANT_FIELDS:
        assert four_receipt[key] == one_receipt[key], key
    pooled = derive_is_apprentice(
        frame.table("person"),
        rate=_Contract.rates[APPRENTICE_RATE_KEY],
        person_weights=frs_person_draws.uk_person_household_weights(frame),
        person_sources=own_source_households(frame.table("person")),
    )
    assert pooled.flags[clone_zero].tolist() != flags[clone_zero].tolist()


def test_refuses_a_clone_index_it_cannot_walk_within() -> None:
    person = _small([22] * 6, ["FT_EMPLOYED"] * 6)
    index = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    for options, error, message in (
        (
            {"person_clone_index": index[:-1]},
            ValueError,
            "person_clone_index must hold one clone per person",
        ),
        (
            {"person_clone_index": index.astype(float)},
            TypeError,
            "person_clone_index must hold an integer clone per person",
        ),
        (
            {"person_clone_index": index, "person_clone_zero": index == 1},
            ValueError,
            "person_clone_zero must flag exactly the people whose",
        ),
    ):
        with pytest.raises(error, match=message) as refused:
            _apprentices(person, rate=0.5, **options)
        assert not any(character.isdigit() for character in str(refused.value))


def test_refuses_a_frame_without_employment_status() -> None:
    person = pd.DataFrame({"person_id": [1], "person_household_id": [1], "age": [22]})

    with pytest.raises(ValueError, match="employment_status"):
        _apprentices(person, rate=0.06)


def test_refuses_a_rate_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _apprentices(_people(1), rate=1.5)


def _two_household_frame(**extra_person_columns):
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 4001),
            "person_benunit_id": np.arange(1, 4001),
            "person_household_id": np.repeat([1, 2], 2000),
            "age": np.tile([22, 23], 2000),
            "employment_status": np.repeat(["FT_EMPLOYED", "STUDENT"], 2000),
            **extra_person_columns,
        }
    )
    return uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": np.arange(1, 4001), "is_married": False}),
        household=pd.DataFrame(
            {
                "household_id": [1, 2],
                "region": ["LONDON", "WALES"],
                "household_weight": [1.0, 3.0],
            }
        ),
        time_period="2024",
    )


@pytest.fixture
def no_engine(monkeypatch):
    """Run the stage as the engine-free job does, whatever is installed."""

    monkeypatch.setattr(frs_person_draws, "_installed_uk_system", lambda: None)


def test_stage_realizes_the_target_at_the_frame_household_weights(no_engine) -> None:
    stage = UKFRSPersonDrawsStageTransform(contract=_Contract(), stage=None)

    stored = stage(_two_household_frame()).table("person")

    # Household 1 holds the employees and weighs 1, household 2 the students
    # and weighs 3: the target is 0.06 x (2000 + 6000) = 480 employees.
    flags = stored[APPRENTICE_OUTPUT].to_numpy()
    assert not flags[2000:].any()
    assert int(flags[:2000].sum()) == 480
    receipt = stage.checkpoint_metadata()["evidence"]
    assert receipt["stage"] == "frs_person_draws"
    apprentices = receipt["apprentice_receipt"]
    assert apprentices["build_year"] == 2024
    assert apprentices["rate"] == 0.06
    # Two households carry everyone, so the receipt withholds the weights.
    assert apprentices["population_support"] == "<3"
    assert apprentices["target"] == "<3"
    assert apprentices["flagged_support"] == "<3"


def test_stage_has_no_receipt_before_it_runs() -> None:
    stage = UKFRSPersonDrawsStageTransform(contract=_Contract(), stage=None)

    with pytest.raises(RuntimeError, match="completed stage run"):
        stage.checkpoint_metadata()


def test_stage_refuses_a_stored_flag(no_engine) -> None:
    frame = _two_household_frame(is_apprentice=False)

    with pytest.raises(ValueError, match="already stored"):
        UKFRSPersonDrawsStageTransform(contract=_Contract(), stage=None)(frame)


_SCHEDULE = {
    "period": 2024,
    "rate": 0.138,
    "secondary_threshold": 9_100.0,
    "apprentice_upper_secondary_threshold": 50_270.0,
}


def test_receipt_records_the_relief_the_flags_imply() -> None:
    pay = np.array([5_000.0, 20_000.0, 60_000.0] * 4)
    person = _small([22] * 12, ["FT_EMPLOYED"] * 12).assign(employment_income=pay)

    # Every employee is flagged: a rate of one a head is cap-bound.
    draw = _apprentices(person, rate=1.0, relief_schedule=_SCHEDULE)

    assert draw.flags.all()
    relieved = np.clip(np.minimum(pay, 50_270.0) - 9_100.0, 0.0, None)
    assert draw.receipt["implied_relief"] == pytest.approx(0.138 * relieved.sum())
    assert draw.receipt["relief_schedule"] == _SCHEDULE


def test_receipt_withholds_a_relief_on_fewer_than_three_households() -> None:
    person = _small([22] * 2, ["FT_EMPLOYED"] * 2).assign(employment_income=20_000.0)

    draw = _apprentices(person, rate=1.0, relief_schedule=_SCHEDULE)

    assert draw.receipt["implied_relief"] == "<3"


def test_receipt_records_no_relief_without_a_schedule() -> None:
    person = _small([22] * 12, ["FT_EMPLOYED"] * 12).assign(employment_income=1.0)

    assert "implied_relief" not in _apprentices(person, rate=0.5).receipt


def test_the_graph_node_reads_every_person_column_the_draw_needs() -> None:
    """A node receives its declared cells alone, so the declaration must name
    the pay the relief reads: the licensed spine at ab7c777e2 recorded no
    implied relief, and the gate blocked it, before the declaration followed
    (microcosm#1373)."""
    from microcosm.build.uk_runtime.graph import uk_spine_graph

    node = uk_spine_graph(source_mode="split").node("frs_person_draws")
    inputs = {
        (slice_.entity, column) for slice_ in node.inputs for column in slice_.columns
    }
    assert {
        ("person", "age"),
        ("person", "employment_status"),
        ("person", "employment_income"),
    } <= inputs


def _system(*, variables, apprentice_threshold=966.73):
    thresholds = SimpleNamespace()
    if apprentice_threshold is not None:
        thresholds.apprentice_upper_secondary_threshold = apprentice_threshold
    class_1 = SimpleNamespace(
        employer_scale=SimpleNamespace(
            components=[None, SimpleNamespace(rate=0.138, threshold=175.0)]
        ),
        thresholds=thresholds,
    )
    instants = []

    def parameters(instant):
        instants.append(instant)
        return SimpleNamespace(
            gov=SimpleNamespace(
                hmrc=SimpleNamespace(
                    national_insurance=SimpleNamespace(class_1=class_1)
                )
            )
        )

    return SimpleNamespace(
        variables=dict.fromkeys(variables), parameters=parameters
    ), instants


_SPLIT = ("qualifies_for_apprentice_minimum_wage", "is_in_first_year_of_apprenticeship")


def test_schedule_reads_the_engine_at_the_start_of_the_stage_year() -> None:
    system, instants = _system(variables=_SPLIT)

    schedule = apprentice_relief_schedule(system, "2024")

    assert instants == ["2024-01-01"]
    assert schedule == {
        "period": 2024,
        "rate": 0.138,
        "secondary_threshold": 175.0 * 52,
        "apprentice_upper_secondary_threshold": 966.73 * 52,
    }


def test_schedule_is_absent_without_an_engine() -> None:
    assert apprentice_relief_schedule(None, "2024") is None


def test_schedule_refuses_an_engine_without_the_minimum_wage_split() -> None:
    system, _ = _system(variables=())

    with pytest.raises(ValueError, match="2683"):
        apprentice_relief_schedule(system, "2024")


def test_schedule_refuses_an_engine_without_the_apprentice_threshold() -> None:
    system, _ = _system(variables=_SPLIT, apprentice_threshold=None)

    with pytest.raises(ValueError, match="apprentice_upper_secondary_threshold"):
        apprentice_relief_schedule(system, "2024")
