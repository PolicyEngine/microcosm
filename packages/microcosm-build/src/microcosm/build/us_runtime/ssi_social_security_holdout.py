"""Held-out SSI/Social Security receipt overlap, never a calibration target.

The US release reports this diagnostic beside its post-export reform validation.
SSA's December 2024 recipient snapshot is a comparator to positive annual model
benefits, rather than a fitted count or a release gate (microcosm#1178).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from microcosm.build.us_runtime.ssi_take_up import US_SSI_TAKE_UP_AGE_TARGETS


@dataclass(frozen=True)
class SSISocialSecurityBenchmark:
    ssi_recipients: int
    # Table 9's four mutually exclusive "With Social Security" income rows:
    # no other income; earned only; unearned only; earned and unearned.
    social_security_by_income: tuple[int, int, int, int]

    @property
    def concurrent_recipients(self) -> int:
        return sum(self.social_security_by_income)

    @property
    def share(self) -> float:
        return self.concurrent_recipients / self.ssi_recipients


#: SSA SSI Annual Statistical Report, 2024, Table 9, "Number", December 2024.
#: Fixed held-out observations, never passed to the calibration/take-up registry.
#: Summing the four "With Social Security" rows gives 27.3489% (18–64) and
#: 57.1327% (65+); retain the actual counts rather than rounded 27% and 57%.
SSA_SSI_SOCIAL_SECURITY_2024_TABLE_9 = MappingProxyType(
    {
        "18_64": SSISocialSecurityBenchmark(
            3_951_866, (984_554, 42_344, 51_677, 2_215)
        ),
        "65_plus": SSISocialSecurityBenchmark(
            2_469_103, (1_270_739, 14_429, 124_121, 1_375)
        ),
    }
)
SSA_SSI_SOCIAL_SECURITY_SOURCE_URL = (
    "https://www.ssa.gov/policy/docs/statcomps/ssi_asr/2024/sect02.html#table9"
)


def ssi_social_security_holdout_unavailable(
    error: Exception, *, period: int, release_id: str | None = None
) -> dict[str, Any]:
    """An observational receipt when plan discovery or scoring is unavailable."""
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "unavailable",
        "period": int(period),
        "target_role": "validation",
        "enforced": False,
        "used_for_calibration": False,
        "source_url": SSA_SSI_SOCIAL_SECURITY_SOURCE_URL,
        "error": f"{type(error).__name__}: {error}",
    }
    if release_id is not None:
        payload["release_id"] = release_id
    return payload


def ssi_social_security_holdout_payload(
    ages: np.ndarray,
    ssi: np.ndarray,
    social_security: np.ndarray,
    weights: np.ndarray,
    *,
    period: int,
    release_id: str | None = None,
) -> dict[str, Any]:
    """Share of positive-SSI recipients with positive Social Security by age.

    Each person carries their household's calibrated weight, as supplied by
    the written-H5 scorer's person results. Empty weighted recipient bands
    report null shares and differences, not zero or a failed gate.
    """
    ages, ssi, social_security, weights = (
        np.asarray(values, dtype=np.float64)
        for values in (ages, ssi, social_security, weights)
    )
    if ages.ndim != 1 or any(
        values.shape != ages.shape for values in (ssi, social_security, weights)
    ):
        raise ValueError("SSI overlap needs aligned one-dimensional person arrays.")
    if any(
        not np.isfinite(values).all()
        for values in (ages, ssi, social_security, weights)
    ) or np.any(weights < 0):
        raise ValueError("SSI overlap needs finite values and nonnegative weights.")

    rows = []
    for band in US_SSI_TAKE_UP_AGE_TARGETS:
        benchmark = SSA_SSI_SOCIAL_SECURITY_2024_TABLE_9.get(band.key)
        if benchmark is None:
            continue
        recipients = band.contains(ages) & (ssi > 0)
        denominator = float(weights[recipients].sum())
        numerator = float(weights[recipients & (social_security > 0)].sum())
        share = numerator / denominator if denominator else None
        rows.append(
            {
                "age_band": band.key,
                "label": band.label,
                "minimum_age": band.minimum_age,
                "maximum_age": band.maximum_age,
                "weighted_ssi_recipients": denominator,
                "weighted_concurrent_recipients": numerator,
                "share": share,
                "benchmark_ssi_recipients": benchmark.ssi_recipients,
                "benchmark_concurrent_recipients": benchmark.concurrent_recipients,
                "benchmark_share": benchmark.share,
                "difference_percentage_points": (
                    100 * (share - benchmark.share) if share is not None else None
                ),
            }
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "available",
        "period": int(period),
        "target_role": "validation",
        "enforced": False,
        "used_for_calibration": False,
        "measure": "share of SSI recipients receiving Social Security",
        "weight_basis": "person's calibrated household_weight",
        "denominator": "persons with ssi > 0 in the age band",
        "numerator": "denominator persons with social_security > 0",
        "benchmark": {
            "source": "SSA SSI Annual Statistical Report, 2024",
            "source_url": SSA_SSI_SOCIAL_SECURITY_SOURCE_URL,
            "year": 2024,
            "table": 9,
            "period": "December 2024",
            "population": "recipients of federally administered SSI payments",
            "derivation": (
                "sum the four With Social Security Number rows / All recipients"
            ),
        },
        "comparability": (
            "SSA counts recipients in December 2024, including federally "
            "administered state supplementation; the model counts positive "
            f"annual ssi and social_security benefits in {int(period)}. "
            "This is a held-out diagnostic, not an exact population or period "
            "match, a calibration target, or a release gate."
        ),
        "age_bands": rows,
    }
    if release_id is not None:
        payload["release_id"] = release_id
    return payload


def ssi_social_security_holdout_from_sim(
    simulation: Any, *, period: int, release_id: str | None = None
) -> dict[str, Any]:
    """Measure aligned person results from the household-batched H5 scorer."""
    age = simulation.calculate("age", period)
    ssi = simulation.calculate("ssi", period)
    social_security = simulation.calculate("social_security", period)
    return ssi_social_security_holdout_payload(
        np.asarray(age),
        np.asarray(ssi),
        np.asarray(social_security),
        np.asarray(age.weights),
        period=period,
        release_id=release_id,
    )


def write_ssi_social_security_holdout(
    payload: dict[str, Any], path: Path | str
) -> Path:
    path = Path(path)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return path
