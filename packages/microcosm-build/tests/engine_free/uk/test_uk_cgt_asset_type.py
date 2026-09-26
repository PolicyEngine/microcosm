"""The CGT asset-type stage: residential flag, main asset type, receipts."""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime import cgt_asset_type
from microcosm.build.uk_runtime.cgt_asset_type import (
    CGT_ASSET_TYPE_COLUMN,
    CGT_ASSET_TYPE_DOMAIN,
    CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES,
    CGT_ASSET_TYPE_NONE,
    CGT_ASSET_TYPE_RESIDENTIAL,
    CGT_ASSET_TYPE_SUB_AEA,
    CGT_BADR_ELIGIBLE_TYPES,
    CGT_BADR_GAINS_COLUMN,
    CGT_RESIDENTIAL_GAINS_COLUMN,
    HMRC_CGT_ASSET_TYPE_RECORD_SETS,
    HMRC_CGT_ASSET_TYPE_RESOURCE,
    HMRC_CGT_TABLE4_BAND_LOWER_BOUNDS,
    UK_CGT_ASSET_TYPE_MASS_CONSERVATION_REASON,
    UK_CGT_ASSET_TYPE_STAGE_NAME,
    HMRCCGTAssetTypeFacts,
    HMRCCGTBADRBand,
    HMRCCGTTable7Type,
    UKCGTBADRParameters,
    assign_uk_cgt_asset_types,
    cgt_asset_type_operation_parameters,
    fit_type_weights,
    load_hmrc_cgt_asset_type_facts,
    solve_residential_logistic,
    uk_cgt_asset_type_stage_transform,
)
from microcosm.build.uk_runtime.cgt_imputation import (
    UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS,
    UKCGTPolicyParameters,
)
from microcosm.build.uk_runtime.content_identity import uk_frame_content_identity
from microcosm.build.uk_runtime.ledger_fact_vendoring import load_vendored_resource
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import Frame

PARAMETERS = UKCGTPolicyParameters(
    personal_allowance=12_570.0,
    personal_allowance_taper_threshold=100_000.0,
    personal_allowance_taper_rate=0.5,
    annual_exempt_amount=3_000.0,
    instant="2024-06-01",
    source="test",
)
BADR_PARAMETERS = UKCGTBADRParameters(
    lifetime_limit=1_000_000.0, rate=0.10, instant="2024-06-01", source="test"
)

#: Synthetic Table 7 rows shaped like the 2026 release (2023-24).
_SYNTHETIC_TABLE7 = (
    ("listed_shares", "financial", 1_077_000.0, 5.925e9),
    ("unlisted_shares", "financial", 290_000.0, 33.315e9),
    ("other_financial_assets", "financial", 1_097_000.0, 16.086e9),
    (
        "agricultural_commercial_industrial_land_buildings",
        "non_financial",
        13_000.0,
        2.037e9,
    ),
    (CGT_ASSET_TYPE_RESIDENTIAL, "non_financial", 167_000.0, 9.626e9),
    ("other_non_financial_assets", "non_financial", 20_000.0, 2.872e9),
)


def _synthetic_facts(
    gains: np.ndarray,
    weights: np.ndarray,
    *,
    badr_share: float = 0.1,
    table7=_SYNTHETIC_TABLE7,
    band_overrides: dict[int, tuple[float, float]] | None = None,
) -> HMRCCGTAssetTypeFacts:
    """Facts sized to a synthetic frame, independent of the vendored resource.

    Residential targets are 30% of the liable mass and gains; each Table 4.1
    band claims ``badr_share`` of the liable mass in its range at their mean
    gain (the open top band at the lifetime limit), so the non-residential
    pool needs a small slope; ``band_overrides`` replaces (count, gains).
    """

    liable = gains > PARAMETERS.annual_exempt_amount
    liable_mass = float(weights[liable].sum())
    liable_gains = float((weights * gains)[liable].sum())
    rows = tuple(
        HMRCCGTTable7Type(
            asset_type=name, category=category, disposals=d, proceeds=3 * g, gains=g
        )
        for name, category, d, g in table7
    )
    lowers = HMRC_CGT_TABLE4_BAND_LOWER_BOUNDS
    uppers = (*lowers[1:], None)
    bands = []
    for index, (lower, upper) in enumerate(zip(lowers, uppers, strict=True)):
        top = np.inf if upper is None else upper
        in_band = liable & (gains >= lower) & (gains < top)
        count = badr_share * float(weights[in_band].sum())
        mean = (
            BADR_PARAMETERS.lifetime_limit
            if upper is None
            else float((weights * gains)[in_band].sum())
            / max(float(weights[in_band].sum()), 1e-12)
        )
        count, band_gains = (band_overrides or {}).get(index, (count, count * mean))
        bands.append(
            HMRCCGTBADRBand(
                lower_bound=lower,
                upper_bound=upper,
                taxpayers=count,
                gains=band_gains,
                tax=None,
            )
        )
    return HMRCCGTAssetTypeFacts(
        table8a_taxpayers_total=0.3 * liable_mass,
        table8a_gains_total=0.3 * liable_gains,
        table8a_disposals_total=0.33 * liable_mass,
        table8a_tax_total=0.06 * liable_gains,
        table8b_individuals_taxpayers=1.0,
        table8b_individuals_gains=1.0,
        table8b_all_taxpayers=1.0,
        table8b_all_gains=1.0,
        table7_types=rows,
        table7_total_gains=sum(row.gains for row in rows),
        table7_total_disposals=sum(row.disposals for row in rows),
        table4_bands=tuple(bands),
        table4_individuals_taxpayers=sum(band.taxpayers for band in bands),
        table4_individuals_gains=sum(band.gains for band in bands),
        table4_individuals_tax=0.0,
        table4_trusts_gains=0.0,
        table4_trusts_tax=0.0,
        table4_all_taxpayers=sum(band.taxpayers for band in bands),
        table4_all_gains=sum(band.gains for band in bands),
        table4_all_tax=0.0,
        resource="synthetic.json",
        resource_sha256="synthetic",
        source_commit="synthetic",
    )


def _frame(gains, *, weights=None, time_period: str = "2024") -> Frame:
    rows = len(gains)
    person = pd.DataFrame(
        {
            "person_id": np.arange(rows, dtype="int64"),
            "person_household_id": np.arange(rows, dtype="int64"),
            "person_benunit_id": np.arange(rows, dtype="int64"),
            "capital_gains": np.asarray(gains, dtype=float),
            "age": np.full(rows, 45, dtype="int64"),
        }
    )
    for column in UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS:
        person[column] = 0.0
    household = pd.DataFrame(
        {
            "household_id": np.arange(rows, dtype="int64"),
            "household_weight": (
                np.full(rows, 60.0)
                if weights is None
                else np.asarray(weights, dtype=float)
            ),
            "region": np.full(rows, "LONDON", dtype=object),
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.arange(rows, dtype="int64")})
    return uk_national_frame(
        person=person, benunit=benunit, household=household, time_period=time_period
    )


def _synthetic_gains(rows: int = 20_000, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gains = np.where(rng.random(rows) < 0.8, np.exp(rng.normal(10.5, 1.8, rows)), 0.0)
    gains[:50] = -1_000.0
    return gains


class TestVendoredFacts:
    def test_committed_resource_types_and_restates_table_8a(self) -> None:
        facts = load_hmrc_cgt_asset_type_facts()

        assert facts.table8a_taxpayers_total == 205_000.0
        assert facts.table8a_gains_total == 12_914_000_000.0
        assert facts.table8b_individuals_taxpayers == 171_000.0
        assert facts.table8b_all_taxpayers == 173_000.0
        assert facts.residential_taxpayers_individuals_basis == pytest.approx(
            205_000.0 * 171_000.0 / 173_000.0
        )
        assert facts.residential_gains_individuals_basis == pytest.approx(
            12_914_000_000.0 * 10_337_000_000.0 / 10_904_000_000.0
        )
        assert {row.asset_type for row in facts.table7_types} == {
            CGT_ASSET_TYPE_RESIDENTIAL,
            *CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES,
        }
        shares = facts.non_residential_gains_shares()
        assert sum(shares.values()) == pytest.approx(1.0)
        assert shares["unlisted_shares"] > 0.5
        assert facts.table7_type("listed_shares").mean_gain_per_disposal < 10_000.0
        assert facts.table7_type("unlisted_shares").mean_gain_per_disposal > 100_000.0

    def test_every_declared_record_set_is_present(self) -> None:
        payload = load_vendored_resource(HMRC_CGT_ASSET_TYPE_RESOURCE)
        record_sets = {row["layout"]["record_set_id"] for row in payload["rows"]}

        assert record_sets == set(HMRC_CGT_ASSET_TYPE_RECORD_SETS)
        assert payload["consumers"] == ["uk_runtime.cgt_asset_type"]

    def test_refuses_a_resource_from_another_feed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = copy.deepcopy(load_vendored_resource(HMRC_CGT_ASSET_TYPE_RESOURCE))
        payload["source_fact_feed"]["facts_sha256"] = "0" * 64
        monkeypatch.setattr(
            cgt_asset_type, "load_vendored_resource", lambda _name: payload
        )

        with pytest.raises(ValueError, match="differs from the committed UK pin"):
            load_hmrc_cgt_asset_type_facts()

    def test_refuses_rows_from_another_workbook(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = copy.deepcopy(load_vendored_resource(HMRC_CGT_ASSET_TYPE_RESOURCE))
        for row in payload["rows"]:
            if row["layout"]["record_set_id"].endswith("table8a.ty2024"):
                row["source"]["source_sha256"] = "f" * 64
        monkeypatch.setattr(
            cgt_asset_type, "load_vendored_resource", lambda _name: payload
        )

        with pytest.raises(ValueError, match="other than the pinned"):
            load_hmrc_cgt_asset_type_facts()

    def test_committed_resource_types_table_4_1(self) -> None:
        facts = load_hmrc_cgt_asset_type_facts()

        bands = facts.table4_bands
        assert tuple(band.lower_bound for band in bands) == (
            HMRC_CGT_TABLE4_BAND_LOWER_BOUNDS
        )
        assert bands[-1].upper_bound is None
        assert all(
            band.upper_bound == following.lower_bound
            for band, following in zip(bands, bands[1:], strict=False)
        )
        assert [band.taxpayers for band in bands] == [
            5_000.0,
            8_000.0,
            6_000.0,
            8_000.0,
            11_000.0,
            8_000.0,
            8_000.0,
            7_000.0,
        ]
        assert bands[-1].gains == 6_787_000_000.0
        assert facts.table4_individuals_taxpayers == 61_000.0
        assert facts.table4_individuals_gains == 18_443_000_000.0
        assert facts.table4_individuals_tax == 1_821_000_000.0
        assert facts.table4_trusts_gains == 32_000_000.0
        assert facts.table4_all_gains == 18_475_000_000.0
        assert sum(band.gains for band in bands) == facts.table4_individuals_gains

    def test_refuses_table_4_rows_from_another_workbook(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = copy.deepcopy(load_vendored_resource(HMRC_CGT_ASSET_TYPE_RESOURCE))
        for row in payload["rows"]:
            if row["layout"]["record_set_id"].endswith("table4_1.ty2024.main"):
                row["source"]["source_sha256"] = "f" * 64
        monkeypatch.setattr(
            cgt_asset_type, "load_vendored_resource", lambda _name: payload
        )

        with pytest.raises(ValueError, match="Table 4 rows trace"):
            load_hmrc_cgt_asset_type_facts()

    def test_refuses_a_drifted_table_4_band_roster(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = copy.deepcopy(load_vendored_resource(HMRC_CGT_ASSET_TYPE_RESOURCE))
        payload["rows"] = [
            row
            for row in payload["rows"]
            if row["dimensions"].get("cgt_badr_ir_gain_band") != "gain_25000_to_49999"
        ]
        monkeypatch.setattr(
            cgt_asset_type, "load_vendored_resource", lambda _name: payload
        )

        with pytest.raises(ValueError, match="band roster drifted"):
            load_hmrc_cgt_asset_type_facts()

    def test_refuses_a_drifted_asset_type_roster(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = copy.deepcopy(load_vendored_resource(HMRC_CGT_ASSET_TYPE_RESOURCE))
        for row in payload["rows"]:
            if row["dimensions"].get("cgt_asset_type") == "listed_shares":
                row["dimensions"]["cgt_asset_type"] = "quoted_shares"
        monkeypatch.setattr(
            cgt_asset_type, "load_vendored_resource", lambda _name: payload
        )

        with pytest.raises(ValueError, match="asset-type roster drifted"):
            load_hmrc_cgt_asset_type_facts()


class TestResidentialSolve:
    def test_meets_both_targets_in_expectation(self) -> None:
        gains = np.exp(np.random.default_rng(3).normal(10.5, 1.8, 5_000))
        weights = np.full(gains.size, 60.0)
        count_target = 0.35 * weights.sum()
        gains_target = 0.12 * (weights * gains).sum()

        a, b, centre = solve_residential_logistic(
            gains, weights, count_target=count_target, gains_target=gains_target
        )
        probabilities = 1.0 / (1.0 + np.exp(-(a + b * (np.log(gains) - centre))))

        assert (weights * probabilities).sum() == pytest.approx(count_target, rel=1e-6)
        assert (weights * gains * probabilities).sum() == pytest.approx(
            gains_target, rel=1e-6
        )
        # Residential gainers are small relative to the whole: negative slope.
        assert b < 0

    def test_refuses_unattainable_targets(self) -> None:
        gains = np.exp(np.random.default_rng(3).normal(10.5, 1.0, 500))
        weights = np.full(gains.size, 10.0)

        with pytest.raises(ValueError, match="not below the liable taxpayer mass"):
            solve_residential_logistic(
                gains, weights, count_target=weights.sum(), gains_target=1.0
            )
        with pytest.raises(ValueError, match="not below the liable gains mass"):
            solve_residential_logistic(
                gains,
                weights,
                count_target=1.0,
                gains_target=(weights * gains).sum(),
            )
        # A mean far above what the largest gains can carry at this count.
        with pytest.raises(ValueError, match="outside the range"):
            solve_residential_logistic(
                gains,
                weights,
                count_target=0.5 * weights.sum(),
                gains_target=0.999 * (weights * gains).sum(),
            )


class TestTypeWeights:
    def test_reproduces_the_share_targets(self) -> None:
        facts = load_hmrc_cgt_asset_type_facts()
        gains = np.exp(np.random.default_rng(5).normal(11.0, 1.5, 4_000))
        weights = np.full(gains.size, 30.0)
        medians = {
            asset_type: facts.table7_type(asset_type).mean_gain_per_disposal
            for asset_type in CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES
        }
        targets = facts.non_residential_gains_shares()

        type_weights, probabilities, iterations = fit_type_weights(
            gains, weights, medians=medians, share_targets=targets
        )

        assert probabilities.shape == (gains.size, 5)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        achieved = ((weights * gains)[:, None] * probabilities).sum(axis=0) / (
            weights * gains
        ).sum()
        for index, asset_type in enumerate(CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES):
            assert achieved[index] == pytest.approx(targets[asset_type], abs=1e-5)
        assert type_weights.sum() == pytest.approx(1.0)
        assert iterations < 100


class TestTypeRestriction:
    def test_restricted_persons_draw_only_their_allowed_types(self) -> None:
        facts = load_hmrc_cgt_asset_type_facts()
        gains = np.exp(np.random.default_rng(7).normal(11.0, 1.5, 4_000))
        weights = np.full(gains.size, 30.0)
        medians = {
            asset_type: facts.table7_type(asset_type).mean_gain_per_disposal
            for asset_type in CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES
        }
        targets = facts.non_residential_gains_shares()
        allowed = np.ones((gains.size, 5), dtype=bool)
        restricted = np.zeros(gains.size, dtype=bool)
        restricted[::7] = True
        eligible = np.isin(
            np.asarray(CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES), CGT_BADR_ELIGIBLE_TYPES
        )
        allowed[restricted] = eligible

        _, probabilities, _ = fit_type_weights(
            gains, weights, medians=medians, share_targets=targets, allowed=allowed
        )

        assert (probabilities[restricted][:, ~eligible] == 0.0).all()
        achieved = ((weights * gains)[:, None] * probabilities).sum(axis=0) / (
            weights * gains
        ).sum()
        for index, asset_type in enumerate(CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES):
            assert achieved[index] == pytest.approx(targets[asset_type], abs=1e-5)

    def test_refuses_a_fit_that_does_not_converge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        facts = load_hmrc_cgt_asset_type_facts()
        gains = np.exp(np.random.default_rng(5).normal(11.0, 1.5, 1_000))
        medians = {
            asset_type: facts.table7_type(asset_type).mean_gain_per_disposal
            for asset_type in CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES
        }
        monkeypatch.setattr(cgt_asset_type, "_SHARE_FIT_ITERATIONS", 1)

        with pytest.raises(ValueError, match="did not reach"):
            fit_type_weights(
                gains,
                np.full(gains.size, 30.0),
                medians=medians,
                share_targets=facts.non_residential_gains_shares(),
            )

    def test_refuses_a_person_with_no_allowed_type(self) -> None:
        facts = load_hmrc_cgt_asset_type_facts()
        gains = np.array([10_000.0, 20_000.0])
        allowed = np.ones((2, 5), dtype=bool)
        allowed[1] = False
        medians = {
            asset_type: facts.table7_type(asset_type).mean_gain_per_disposal
            for asset_type in CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES
        }

        with pytest.raises(ValueError, match="at least one asset type"):
            fit_type_weights(
                gains,
                np.ones(2),
                medians=medians,
                share_targets=facts.non_residential_gains_shares(),
                allowed=allowed,
            )


class TestBADRClaims:
    def _run(self, gains=None, weights=None, **kwargs):
        gains = _synthetic_gains() if gains is None else gains
        weights = np.full(gains.size, 60.0) if weights is None else weights
        facts = kwargs.pop("facts", None) or _synthetic_facts(gains, weights)
        frame = _frame(gains, weights=weights)
        result, summary = assign_uk_cgt_asset_types(
            frame, facts, PARAMETERS, kwargs.pop("badr", BADR_PARAMETERS), **kwargs
        )
        return gains, result.table("person"), summary.evidence()

    def test_every_band_meets_its_targets_within_the_walk_bound(self) -> None:
        gains, person, evidence = self._run()
        qualifying = person[CGT_BADR_GAINS_COLUMN].to_numpy()
        types = person[CGT_ASSET_TYPE_COLUMN].to_numpy()
        residential = person[CGT_RESIDENTIAL_GAINS_COLUMN].to_numpy()
        claimant = qualifying > 0.0
        limit = BADR_PARAMETERS.lifetime_limit

        bands = evidence["badr"]["bands"]
        assert [band["lower_bound"] for band in bands] == list(
            HMRC_CGT_TABLE4_BAND_LOWER_BOUNDS
        )
        for band in bands:
            assert band["skipped"] is False
            assert band["expected_count"] == pytest.approx(
                band["count_target"], rel=1e-6
            )
            assert band["expected_gains"] == pytest.approx(
                band["gains_target"], rel=1e-6
            )
            assert abs(band["achieved_count"] - band["expected_count"]) <= band[
                "max_pool_weight"
            ] * (1 + 1e-9)
            if band["qualifying_amount"] == "lifetime_limit":
                assert band["achieved_gains"] == pytest.approx(
                    limit * band["achieved_count"]
                )
            else:
                bound = band["max_pool_weight"] * (
                    2 * band["pool_max_gain"] - band["pool_min_gain"]
                )
                assert abs(band["achieved_gains"] - band["expected_gains"]) <= bound
        # Below the limit the whole net gain qualifies; at the top, the limit.
        below = claimant & (gains < limit)
        assert (qualifying[below] == gains[below]).all()
        assert (qualifying[claimant & (gains >= limit)] == limit).all()
        assert (qualifying[claimant] <= gains[claimant]).all()
        assert (qualifying[~claimant] == 0.0).all()
        # The type follows the claim, and no claim is residential.
        assert set(types[claimant]) <= set(CGT_BADR_ELIGIBLE_TYPES)
        assert not (claimant & (residential > 0.0)).any()
        assert set(evidence["badr"]["invariants"].values()) == {0}
        assert evidence["seeds"]["badr_flag"] == 555
        assert evidence["asset_type"]["share_fit_converged"] is True
        assert evidence["badr"]["totals"]["achieved_rows"] == int(claimant.sum())

    def test_relief_rate_tax_takes_the_exempt_amount_last(self) -> None:
        tax = cgt_asset_type._relief_rate_tax(
            np.array([50_000.0, 30_000.0, 1_000_000.0, 2_000.0]),
            np.array([50_000.0, 40_000.0, 5_000_000.0, 2_500.0]),
            rate=0.1,
            annual_exempt_amount=3_000.0,
        )

        np.testing.assert_allclose(tax, [4_700.0, 3_000.0, 100_000.0, 0.0])

    def test_a_zero_band_is_skipped_and_an_empty_pool_is_refused(self) -> None:
        gains = _synthetic_gains()
        weights = np.full(gains.size, 60.0)
        skipped = _synthetic_facts(gains, weights, band_overrides={0: (0.0, 0.0)})
        _, _, evidence = self._run(gains, weights, facts=skipped)
        assert evidence["badr"]["bands"][0]["skipped"] is True

        # No net gain between 250,000 and 499,999, yet the band has a target.
        emptied = np.where((gains >= 250_000) & (gains < 500_000), 200_000.0, gains)
        facts = _synthetic_facts(emptied, weights, band_overrides={5: (1_000.0, 3.6e8)})
        with pytest.raises(ValueError, match="250,000 to 499,999"):
            self._run(emptied, weights, facts=facts)

    def test_an_unreachable_band_is_refused_by_name(self) -> None:
        gains = _synthetic_gains()
        weights = np.full(gains.size, 60.0)
        # A mean far above anything in the 10,000 to 24,999 range.
        facts = _synthetic_facts(
            gains, weights, band_overrides={1: (1_000.0, 1_000.0 * 24_990.0)}
        )

        with pytest.raises(ValueError, match="BADR band GBP 10,000 to 24,999"):
            self._run(gains, weights, facts=facts)

    def test_the_top_band_must_start_at_the_lifetime_limit(self) -> None:
        with pytest.raises(ValueError, match="must start at the limit"):
            self._run(
                badr=UKCGTBADRParameters(
                    lifetime_limit=2_000_000.0,
                    rate=0.10,
                    instant="2024-06-01",
                    source="test",
                )
            )

    def test_is_deterministic_and_seed_sensitive(self) -> None:
        _, first, _ = self._run()
        _, second, _ = self._run()
        _, other, _ = self._run(badr_seed=999)

        pd.testing.assert_frame_equal(first, second)
        assert not first[CGT_BADR_GAINS_COLUMN].equals(other[CGT_BADR_GAINS_COLUMN])
        # The residential draw comes first on its own seed and does not move.
        assert first[CGT_RESIDENTIAL_GAINS_COLUMN].equals(
            other[CGT_RESIDENTIAL_GAINS_COLUMN]
        )

    def test_refuses_claims_the_eligible_types_cannot_hold(self) -> None:
        tiny = tuple(
            (
                name,
                category,
                disposals,
                0.01e9 if name in CGT_BADR_ELIGIBLE_TYPES else g,
            )
            for name, category, disposals, g in _SYNTHETIC_TABLE7
        )
        gains = _synthetic_gains()
        weights = np.full(gains.size, 60.0)
        facts = _synthetic_facts(gains, weights, badr_share=0.5, table7=tiny)

        with pytest.raises(ValueError, match="above the Table 7 share"):
            self._run(gains, weights, facts=facts)


class TestAssignment:
    def test_flags_and_types_every_liable_gainer_and_nobody_else(self) -> None:
        facts = load_hmrc_cgt_asset_type_facts()
        gains = _synthetic_gains()
        frame = _frame(gains)

        result, summary = assign_uk_cgt_asset_types(
            frame, facts, PARAMETERS, BADR_PARAMETERS
        )

        person = result.table("person")
        types = person[CGT_ASSET_TYPE_COLUMN].to_numpy()
        residential = person[CGT_RESIDENTIAL_GAINS_COLUMN].to_numpy()
        liable = gains > PARAMETERS.annual_exempt_amount
        assert set(types) <= set(CGT_ASSET_TYPE_DOMAIN)
        assert (types[gains <= 0] == CGT_ASSET_TYPE_NONE).all()
        assert (types[(gains > 0) & ~liable] == CGT_ASSET_TYPE_SUB_AEA).all()
        assert not np.isin(
            types[liable], [CGT_ASSET_TYPE_NONE, CGT_ASSET_TYPE_SUB_AEA]
        ).any()
        flagged = types == CGT_ASSET_TYPE_RESIDENTIAL
        assert (residential[flagged] == gains[flagged]).all()
        assert (residential[~flagged] == 0.0).all()
        # Untouched columns and weights, one conservation receipt.
        assert (person["capital_gains"].to_numpy() == gains).all()
        assert result.mass_log[-1].reason == UK_CGT_ASSET_TYPE_MASS_CONSERVATION_REASON
        assert result.mass_log[-1].old_total == result.mass_log[-1].new_total

        evidence = summary.evidence()
        assert evidence["stage"] == UK_CGT_ASSET_TYPE_STAGE_NAME
        residential_receipt = evidence["residential"]
        assert residential_receipt["expected_count"] == pytest.approx(
            facts.residential_taxpayers_individuals_basis, rel=1e-6
        )
        assert residential_receipt["expected_gains"] == pytest.approx(
            facts.residential_gains_individuals_basis, rel=1e-6
        )
        # Systematic sampling lands within one person of the expected count.
        assert (
            abs(
                residential_receipt["achieved_count"]
                - residential_receipt["expected_count"]
            )
            <= 60.0
        )
        assert residential_receipt["achieved_gains"] == pytest.approx(
            residential_receipt["expected_gains"], rel=0.05
        )
        assert residential_receipt["logistic_slope"] < 0
        shares = evidence["asset_type"]["achieved_gains_share"]
        assert sum(shares.values()) == pytest.approx(1.0)
        assert shares["unlisted_shares"] > 0.4
        assert evidence["value_counts"][CGT_ASSET_TYPE_RESIDENTIAL] == int(
            flagged.sum()
        )
        assert len(evidence["composition_by_band"]) == 10
        assert evidence["facts"]["resource_sha256"] == facts.resource_sha256
        assert evidence["seeds"] == {
            "residential_flag": 553,
            "badr_flag": 555,
            "asset_type": 554,
        }
        badr = evidence["badr"]
        assert set(badr["invariants"].values()) == {0}
        assert badr["totals"]["gains_target"] == pytest.approx(
            facts.table4_individuals_gains, rel=1e-9
        )
        claimant = person[CGT_BADR_GAINS_COLUMN].to_numpy() > 0
        assert set(types[claimant]) <= set(CGT_BADR_ELIGIBLE_TYPES)

    def test_is_deterministic_and_seed_sensitive(self) -> None:
        facts = load_hmrc_cgt_asset_type_facts()
        frame = _frame(_synthetic_gains())

        first, _ = assign_uk_cgt_asset_types(frame, facts, PARAMETERS, BADR_PARAMETERS)
        second, _ = assign_uk_cgt_asset_types(frame, facts, PARAMETERS, BADR_PARAMETERS)
        other, _ = assign_uk_cgt_asset_types(
            frame, facts, PARAMETERS, BADR_PARAMETERS, asset_type_seed=999
        )

        assert uk_frame_content_identity(first) == uk_frame_content_identity(second)
        assert uk_frame_content_identity(first) != uk_frame_content_identity(other)

    def test_refuses_a_frame_that_cannot_hold_the_targets(self) -> None:
        facts = load_hmrc_cgt_asset_type_facts()
        frame = _frame([10_000.0, 20_000.0, 50_000.0], weights=[1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match="not below the liable taxpayer mass"):
            assign_uk_cgt_asset_types(frame, facts, PARAMETERS, BADR_PARAMETERS)

    def test_refuses_a_frame_already_classified(self) -> None:
        facts = load_hmrc_cgt_asset_type_facts()
        frame = _frame(_synthetic_gains())
        classified, _ = assign_uk_cgt_asset_types(
            frame, facts, PARAMETERS, BADR_PARAMETERS
        )

        with pytest.raises(ValueError, match="already carries"):
            assign_uk_cgt_asset_types(classified, facts, PARAMETERS, BADR_PARAMETERS)


class TestStageContract:
    def test_manifest_operations_restate_the_reviewed_parameters(self) -> None:
        spec = load_country_spec("uk")
        assert spec.sources is not None
        stage = spec.sources.stage_map()[UK_CGT_ASSET_TYPE_STAGE_NAME]
        expected = cgt_asset_type_operation_parameters()

        assert [operation.kind for operation in stage.operations] == list(expected)
        for operation in stage.operations:
            assert dict(operation.parameters) == expected[operation.kind]
        assert tuple(stage.outputs) == (
            CGT_ASSET_TYPE_COLUMN,
            CGT_RESIDENTIAL_GAINS_COLUMN,
            CGT_BADR_GAINS_COLUMN,
        )
        roles = {artifact["role"]: artifact for artifact in stage.artifacts}
        assert roles["cgt_asset_type_facts"]["resource"] == HMRC_CGT_ASSET_TYPE_RESOURCE
        assert roles["cgt_asset_type_facts"]["runtime_sha256_required"] is True

    def test_transform_refuses_a_drifted_declaration(self) -> None:
        from dataclasses import replace

        spec = load_country_spec("uk")
        assert spec.sources is not None
        stage = spec.sources.stage_map()[UK_CGT_ASSET_TYPE_STAGE_NAME]
        drifted = replace(stage, operations=tuple(reversed(stage.operations)))

        with pytest.raises(ValueError, match="operation order drifted"):
            uk_cgt_asset_type_stage_transform(drifted)

    def test_transform_runs_from_the_resource_and_the_seam(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = load_country_spec("uk")
        assert spec.sources is not None
        stage = spec.sources.stage_map()[UK_CGT_ASSET_TYPE_STAGE_NAME]
        gains = _synthetic_gains(6_000)
        weights = np.full(gains.size, 300.0)
        frame = _frame(gains, weights=weights)
        # The frame holds a few thousand gainers, so the Table 4.1 bands are
        # sized to it; the vendored Table 8 and Table 7 rows drive the rest.
        vendored = load_hmrc_cgt_asset_type_facts()
        sized = _synthetic_facts(gains, weights)
        facts = dataclasses.replace(
            vendored,
            **{
                name: getattr(sized, name)
                for name in vendored.__dataclass_fields__
                if name.startswith("table4_")
            },
        )
        resolved: list[str] = []

        def load_facts():
            resolved.append("facts")
            return facts

        def load_parameters(period):
            assert period == "2024"
            resolved.append("parameters")
            return PARAMETERS

        monkeypatch.setattr(
            cgt_asset_type, "load_hmrc_cgt_asset_type_facts", load_facts
        )
        monkeypatch.setattr(cgt_asset_type, "uk_cgt_policy_parameters", load_parameters)

        def load_badr_parameters(period):
            assert period == "2024"
            resolved.append("badr_parameters")
            return BADR_PARAMETERS

        monkeypatch.setattr(
            cgt_asset_type, "uk_cgt_badr_parameters", load_badr_parameters
        )
        from_resource = uk_cgt_asset_type_stage_transform(stage)
        result_a = from_resource(frame)
        assert resolved == ["facts", "parameters", "badr_parameters"]

        seam = uk_cgt_asset_type_stage_transform(
            stage, facts=facts, parameters=PARAMETERS, badr_parameters=BADR_PARAMETERS
        )
        result_b = seam(frame)

        assert uk_frame_content_identity(result_a) == uk_frame_content_identity(
            result_b
        )
        assert from_resource.checkpoint_metadata() == seam.checkpoint_metadata()
        assert from_resource.checkpoint_metadata()["evidence"]["stage"] == (
            UK_CGT_ASSET_TYPE_STAGE_NAME
        )


def test_synthetic_facts_type_check_without_the_feed() -> None:
    """A synthetic facts object drives the stage without the vendored rows."""
    from microcosm.build.uk_runtime.cgt_asset_type import HMRCCGTTable7Type

    rows = tuple(
        HMRCCGTTable7Type(
            asset_type=name, category="financial", disposals=d, proceeds=3 * g, gains=g
        )
        for name, d, g in (
            ("listed_shares", 1_000_000.0, 5.0e9),
            ("unlisted_shares", 300_000.0, 33.0e9),
            ("other_financial_assets", 1_000_000.0, 16.0e9),
            ("agricultural_commercial_industrial_land_buildings", 13_000.0, 2.0e9),
            (CGT_ASSET_TYPE_RESIDENTIAL, 170_000.0, 10.0e9),
            ("other_non_financial_assets", 20_000.0, 3.0e9),
        )
    )
    gains = _synthetic_gains(3_000)
    table4 = _synthetic_facts(gains, np.full(3_000, 5.0))
    facts = HMRCCGTAssetTypeFacts(
        table8a_taxpayers_total=2_000.0,
        table8a_gains_total=60_000_000.0,
        table8a_disposals_total=2_200.0,
        table8a_tax_total=12_000_000.0,
        table8b_individuals_taxpayers=95.0,
        table8b_individuals_gains=95.0,
        table8b_all_taxpayers=100.0,
        table8b_all_gains=100.0,
        table7_types=rows,
        table7_total_gains=sum(row.gains for row in rows),
        table7_total_disposals=sum(row.disposals for row in rows),
        table4_bands=table4.table4_bands,
        table4_individuals_taxpayers=table4.table4_individuals_taxpayers,
        table4_individuals_gains=table4.table4_individuals_gains,
        table4_individuals_tax=0.0,
        table4_trusts_gains=0.0,
        table4_trusts_tax=0.0,
        table4_all_taxpayers=table4.table4_all_taxpayers,
        table4_all_gains=table4.table4_all_gains,
        table4_all_tax=0.0,
        resource="synthetic.json",
        resource_sha256="synthetic",
        source_commit="synthetic",
    )
    frame = _frame(gains, weights=np.full(3_000, 5.0))

    _, summary = assign_uk_cgt_asset_types(frame, facts, PARAMETERS, BADR_PARAMETERS)

    assert summary.evidence()["residential"]["count_target_individuals_basis"] == (
        pytest.approx(1_900.0)
    )
