"""Twin-conditional uniforms for the SPI stage-1 draw (microcosm#840).

The stage-1 forest of ``hmrc_spi_income_spine`` draws each SPI-channel
recipient's income leaves from the SPI tape conditioned on the stage-1
predictors. Which quantile of that conditional a recipient received used to be
a model random number, so the recipient's own observed income, its FRS twin's,
carried nothing into the draw: a lone parent on a low wage drew the pay of a
random taxpayer of her age, gender and region. This module supplies the
quantiles instead. A reference forest fitted on the FRS base rows with the same
predictors and chain order locates each twin's observed leaves on the FRS
conditional (:meth:`microcosm.fit.FittedRegimeGatedQRF.conditional_uniforms`,
a Rosenblatt transform: the sign uniform is the twin's position in the gate's
cumulative class probabilities, the quantile its position on the sign's
magnitude forest, each leaf chained on the twin's observed earlier leaves), and
the stage-1 forest draws at those uniforms. The recipient then receives the SPI
income of someone in the same position among SPI people like it as the twin
holds among FRS people like it: the FRS relation between composition and
income rank is kept, the SPI levels and tail are imported.

Leaves without an FRS analogue, or whose analogue is degenerate on the FRS
rows, draw at uniforms keyed on the recipient's identity
(:func:`microcosm.build.stochastic_assignment.stable_identity_uniforms`, the
key the band-donor resample uses), as do the placements inside a class or a
flat stretch of a conditional; no model random stream is consumed, so a
recipient's draw depends on its twin and its identity alone, whatever the row
order and whoever else is drawn.

The reference forest is cross-fitted by predictor cell. A forest's conditional
at a row it was trained on holds that row's own value as an atom, and under
the weighted bootstrap a row's copies fill its own leaves: on the 2024-25 spine
a twin's own value took a median 0.23 (pay) to 0.74 (dividends) of its
conditional's mass, so its located quantile was mostly the random placement
inside that atom and the draw kept under half of the twin's rank within a
cell. The rows are therefore split into folds by a stable hash of their
predictor cell (gender, region, whole-year age and the four bridge
predictors), each fold's recipients are located by a forest fitted on the
other folds, and because a copy shares its twin's cell, the forest that
locates it never saw the twin. Keying the folds on the cell rather than the
row keeps every row of a cell on one forest, so the located quantiles stay
monotone in the twin's value within the cell.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.frs_hmrc_source import (
    FRS_HMRC_INCPBEN_COLUMN,
    FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN,
    FRS_HMRC_PAY_COLUMN,
    FRS_HMRC_SRP_REGULAR_CODE5_COLUMN,
    FRS_HMRC_UBISJA_COLUMN,
)
from microcosm.build.uk_runtime.spi_support import (
    SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN,
    SPI_HMRC_EMPLOYMENT_EXPENSES_COLUMN,
    SPI_HMRC_INCAPACITY_BENEFIT_INCOME_COLUMN,
    SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN,
    SPI_HMRC_OTHER_INCOME_COLUMN,
    SPI_HMRC_OTHER_SOCIAL_SECURITY_INCOME_COLUMN,
    SPI_HMRC_PAY_COLUMN,
    SPI_HMRC_STATE_PENSION_INCOME_COLUMN,
    SPI_HMRC_TAXABLE_TERMINATION_PAY_COLUMN,
    SPI_HMRC_UNEMPLOYMENT_BENEFIT_INCOME_COLUMN,
    SPI_INCOME_QRF_OUTPUT_COLUMNS,
    SPI_PROPERTY_FINANCE_COSTS_COLUMN,
)

#: The reference forest's fit-weight record name.
SPI_TRANSPORT_FIT_NAME = "uk_frs_reference_conditional"
#: The reference forest's seed is the stage seed plus this offset: stage 1 is
#: the seed, stage 2 the seed + 1 and the band-donor resample the seed + 2.
SPI_TRANSPORT_SEED_OFFSET = 3
#: The manifest's name for this draw.
SPI_TRANSPORT_DRAW = "twin_conditional_quantiles"
#: Identity-keyed uniforms are salted ``<salt>:<target>:<sign|quantile>``.
SPI_TRANSPORT_SALT = "spi_stage1"
#: The identity a recipient's uniforms are keyed on: the FRS person it copies.
SPI_TRANSPORT_KEY_COLUMN = "person_source_id"
#: Taxable savings interest on the twin: the FRS gross interest less the
#: tax-free part, the concept the tape's INCBBS leaf holds.
TAXABLE_SAVINGS_INTEREST_ANALOGUE = "taxable_savings_interest"
#: Per-group uniformity is reported only over at least this many rows.
SPI_TRANSPORT_GROUP_MINIMUM_ROWS = 10
#: The reference forest is cross-fitted in this many folds keyed on the
#: predictor cell; a frame with fewer distinct cells uses one fold per cell.
SPI_TRANSPORT_FOLDS = 5
#: The fold draw's salt; the fold of a cell is ``floor(uniform * folds)``.
SPI_TRANSPORT_FOLD_SALT = "spi_stage1:reference_fold"
#: The columns whose values name a predictor cell, age at the whole year.
SPI_TRANSPORT_CELL_COLUMNS = (
    "gender",
    "region",
    "age",
    "main_source",
    "has_pay",
    "has_trade",
    "at_or_over_spa",
)

#: The FRS twin's analogue of every stage-1 output (``None``: no analogue, the
#: leaf draws at identity-keyed uniforms). The analogue is the column on the
#: FRS rows that holds the same concept the tape leaf holds, or a retained
#: FRS leaf crosswalked to it by ``frs_hmrc_spine_leaves``.
SPI_STAGE1_ANALOGUE_COLUMNS: Mapping[str, str | None] = {
    "self_employment_income": "self_employment_income",
    "savings_interest_income": TAXABLE_SAVINGS_INTEREST_ANALOGUE,
    "dividend_income": "dividend_income",
    "private_pension_income": "private_pension_income",
    "property_income": "property_income",
    "other_investment_income": None,
    "gift_aid": None,
    "charitable_investment_gifts": None,
    SPI_HMRC_PAY_COLUMN: FRS_HMRC_PAY_COLUMN,
    SPI_HMRC_EMPLOYMENT_BENEFITS_COLUMN: None,
    SPI_HMRC_EMPLOYMENT_EXPENSES_COLUMN: None,
    SPI_HMRC_INCAPACITY_BENEFIT_INCOME_COLUMN: FRS_HMRC_INCPBEN_COLUMN,
    SPI_HMRC_OTHER_SOCIAL_SECURITY_INCOME_COLUMN: (
        FRS_HMRC_OSSBEN_IDENTIFIABLE_SUBSET_COLUMN
    ),
    SPI_HMRC_TAXABLE_TERMINATION_PAY_COLUMN: None,
    SPI_HMRC_UNEMPLOYMENT_BENEFIT_INCOME_COLUMN: FRS_HMRC_UBISJA_COLUMN,
    SPI_HMRC_MISCELLANEOUS_EMPLOYMENT_INCOME_COLUMN: None,
    SPI_HMRC_OTHER_INCOME_COLUMN: "miscellaneous_income",
    SPI_HMRC_STATE_PENSION_INCOME_COLUMN: FRS_HMRC_SRP_REGULAR_CODE5_COLUMN,
    # Restricted finance costs reach the FRS rows only at property_components,
    # after this stage, so the twin has no analogue to locate (microcosm#1106).
    SPI_PROPERTY_FINANCE_COSTS_COLUMN: None,
}


@dataclass(frozen=True)
class SPITransportResult:
    """The stage-1 uniforms per target and the receipt that explains them."""

    quantiles: dict[str, np.ndarray]
    sign_uniforms: dict[str, np.ndarray]
    receipt: dict[str, Any]


def assert_analogue_map_covers(outputs: Sequence[str]) -> None:
    """Refuse a stage-1 output surface the analogue map does not name.

    A new stage-1 leaf must say whether the twin carries its concept; a leaf
    drawn at identity uniforms by default would silently lose the twin's
    standing on it.
    """

    declared = set(SPI_STAGE1_ANALOGUE_COLUMNS)
    expected = set(outputs)
    if declared != expected:
        raise ValueError(
            "SPI stage-1 analogue map does not cover the stage-1 outputs: "
            f"unmapped {sorted(expected - declared)}, "
            f"stale {sorted(declared - expected)}."
        )


def analogue_targets(
    outputs: Sequence[str] = SPI_INCOME_QRF_OUTPUT_COLUMNS,
) -> tuple[str, ...]:
    """The stage-1 outputs with an FRS analogue, in stage-1 chain order."""

    return tuple(
        target for target in outputs if SPI_STAGE1_ANALOGUE_COLUMNS.get(target)
    )


def analogue_source_columns() -> tuple[str, ...]:
    """The FRS person columns the analogues read."""

    columns = {
        column
        for column in SPI_STAGE1_ANALOGUE_COLUMNS.values()
        if column and column != TAXABLE_SAVINGS_INTEREST_ANALOGUE
    }
    columns |= {"savings_interest_income", "tax_free_savings_income"}
    return tuple(sorted(columns))


def observed_analogues(
    person: pd.DataFrame,
    outputs: Sequence[str] = SPI_INCOME_QRF_OUTPUT_COLUMNS,
) -> pd.DataFrame:
    """The twin's observed values of every analogue target, named by target.

    Rows are FRS persons (the twins, or the SPI copies before the draw, which
    still carry the twin's values). Taxable savings interest is derived as the
    gross FRS interest less the tax-free part, floored at zero.
    """

    missing = sorted(set(analogue_source_columns()) - set(person.columns))
    if missing:
        raise ValueError(f"SPI transport analogues are missing column(s): {missing}.")
    taxable_interest = np.maximum(
        pd.to_numeric(person["savings_interest_income"], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        - pd.to_numeric(person["tax_free_savings_income"], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan
        ),
        0.0,
    )
    observed = pd.DataFrame(index=person.index)
    for target in analogue_targets(outputs):
        column = SPI_STAGE1_ANALOGUE_COLUMNS[target]
        if column == TAXABLE_SAVINGS_INTEREST_ANALOGUE:
            observed[target] = taxable_interest
        else:
            observed[target] = pd.to_numeric(person[column], errors="coerce").to_numpy(
                dtype=np.float64, na_value=np.nan
            )
    values = observed.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad = [
            str(column)
            for position, column in enumerate(observed.columns)
            if not np.isfinite(values[:, position]).all()
        ]
        raise ValueError(f"SPI transport analogues are not finite: {bad}.")
    return observed


def cell_fold_ids(
    predictors: pd.DataFrame, *, folds: int, seed: int, used: int | None = None
) -> tuple[np.ndarray, int]:
    """Assign every row to a fold by a stable hash of its predictor cell.

    ``predictors`` holds the raw (unencoded) stage-1 predictors; a cell is the
    tuple of :data:`SPI_TRANSPORT_CELL_COLUMNS` with age at the whole year.
    The fold is ``floor(uniform * used)`` of the cell's identity-keyed
    uniform, so two frames whose rows share a cell share its fold. ``used``
    is ``folds`` capped at the number of distinct cells in ``predictors``, or
    the caller's value (the training frame's) for a second frame.
    """

    if not isinstance(folds, int) or folds <= 0:
        raise ValueError("folds must be a positive integer.")
    missing = sorted(set(SPI_TRANSPORT_CELL_COLUMNS) - set(predictors.columns))
    if missing:
        raise ValueError(f"fold cells need predictor column(s) {missing}.")
    cells = predictors[list(SPI_TRANSPORT_CELL_COLUMNS)].copy()
    cells["age"] = np.floor(
        pd.to_numeric(cells["age"], errors="coerce").to_numpy(dtype=np.float64)
    )
    keys = cells.astype(str).agg("|".join, axis=1).to_numpy()
    if used is None:
        used = min(folds, len(set(keys.tolist())))
    if not isinstance(used, int) or used <= 0:
        raise ValueError("used must be a positive integer.")
    uniforms = stable_identity_uniforms(keys, seed=seed, salt=SPI_TRANSPORT_FOLD_SALT)
    return np.minimum((uniforms * used).astype(np.int64), used - 1), used


def cross_fit_transport(
    fit_reference: Any,
    *,
    training_predictors: pd.DataFrame,
    encoded_training: pd.DataFrame,
    training_observed: pd.DataFrame,
    training_folds: np.ndarray,
    training_identities: np.ndarray,
    used: int,
    recipient_predictors: pd.DataFrame,
    encoded_recipients: pd.DataFrame,
    observed: pd.DataFrame,
    identities: np.ndarray,
    groups: np.ndarray | None,
    seed: int,
    folds: int,
    allow_in_sample: bool = False,
    outputs: Sequence[str] = SPI_INCOME_QRF_OUTPUT_COLUMNS,
) -> SPITransportResult:
    """Locate every recipient on a reference forest that never saw its cell.

    ``fit_reference(encoded, observed)`` fits one reference forest on the given
    training rows and returns it (its ``targets`` are :func:`analogue_targets`).
    ``training_folds`` assigns the training rows to folds
    (:func:`cell_fold_ids`); recipients take the fold of their own cell, so the
    forest that locates a copy was fitted without its twin. Each fold's
    recipients are located by :func:`transport_uniforms`; the receipt is
    assembled over every recipient and names the folds.
    """

    assert_analogue_map_covers(outputs)
    n = len(encoded_recipients)
    identities = np.asarray(identities)
    if identities.shape != (n,):
        raise ValueError("SPI transport identities must have one entry per recipient.")
    if len(observed) != n or not observed.index.equals(encoded_recipients.index):
        raise ValueError("SPI transport observed analogues must align to recipients.")
    if len(training_folds) != len(encoded_training):
        raise ValueError("training folds must align to the training rows.")
    training_identities = np.asarray(training_identities)
    if training_identities.shape != (len(encoded_training),):
        raise ValueError("training identities must align to the training rows.")
    if not isinstance(used, int) or used <= 0:
        raise ValueError("used must be a positive integer.")
    if len(training_folds) and int(training_folds.max()) >= used:
        raise ValueError("training folds must lie below the fold count used.")
    # The training rows' folds and the recipients' come from the same draw
    # over the same cells at the same fold count, so a recipient's fold is its
    # twin's fold; the twins the training rows hold are checked, so a cell
    # value that moved between twin and copy cannot put a twin's forest on
    # its copy.
    recipient_folds, _ = cell_fold_ids(
        recipient_predictors, folds=folds, seed=seed, used=used
    )
    twin_folds = pd.Series(training_folds, index=training_identities)
    if not twin_folds.index.is_unique:
        raise ValueError("training identities must be unique (one row per twin).")
    twins = twin_folds.reindex(identities)
    known = twins.notna().to_numpy()
    if known.any() and not np.array_equal(
        twins.to_numpy()[known].astype(np.int64), recipient_folds[known]
    ):
        raise ValueError(
            "a recipient's fold differs from its twin's: the fold-key cell values "
            "moved between the twin and the copy, so the hold-out would not hold."
        )
    # With one fold (a frame of one predictor cell) there is nothing to hold
    # out: only a sampled rung or a synthetic fixture may locate in-sample,
    # and the receipt says so.
    cross_fitted = used > 1
    if not cross_fitted and not allow_in_sample:
        raise ValueError(
            "the SPI reference forest cannot be cross-fitted: the training rows "
            "fill one fold. In-sample location is allowed only on a sampled rung "
            "or a synthetic fixture (allow_in_sample)."
        )
    quantiles = {target: np.empty(n, dtype=np.float64) for target in outputs}
    sign_uniforms = {target: np.empty(n, dtype=np.float64) for target in outputs}
    fold_receipts: dict[str, dict[str, Any]] = {}
    in_sample_folds: list[int] = []
    regimes_by_fold: dict[str, list[str | None]] = {target: [] for target in outputs}
    classes_by_fold: dict[str, list[tuple[int, ...]]] = {
        target: [] for target in outputs
    }
    degenerate_rows = dict.fromkeys(outputs, 0)
    for fold in range(used):
        score = np.flatnonzero(recipient_folds == fold)
        if score.size == 0:
            continue
        train = (
            np.flatnonzero(training_folds != fold)
            if cross_fitted
            else np.arange(len(encoded_training))
        )
        if train.size == 0:
            # A frame too small for the hash to fill every fold's complement
            # (the synthetic fixtures): this fold's recipients are located
            # in-sample, and the receipt names the fold.
            if not allow_in_sample:
                raise ValueError(
                    f"SPI reference fold {fold} leaves no training rows to fit "
                    "on; in-sample location is allowed only on a sampled rung "
                    "or a synthetic fixture (allow_in_sample)."
                )
            train = np.arange(len(encoded_training))
            in_sample_folds.append(int(fold))
        reference = fit_reference(
            encoded_training.iloc[train], training_observed.iloc[train]
        )
        part = transport_uniforms(
            reference,
            encoded_recipients=encoded_recipients.iloc[score],
            observed=observed.iloc[score],
            identities=identities[score],
            groups=None,
            seed=seed,
            outputs=outputs,
        )
        for target in outputs:
            quantiles[target][score] = part.quantiles[target]
            sign_uniforms[target][score] = part.sign_uniforms[target]
            regimes_by_fold[target].append(
                part.receipt["targets"][target]["reference_regime"]
            )
            fold_classes = part.receipt["targets"][target].get("reference_classes")
            if fold_classes is not None:
                classes_by_fold[target].append(tuple(fold_classes))
            if target in part.receipt["degenerate_targets"]:
                degenerate_rows[target] += int(score.size)
        fold_receipts[f"fold_{fold}"] = {
            "training_rows": int(train.size),
            "scored_recipients": int(score.size),
            "in_sample": fold in in_sample_folds,
        }
    shared_identities = int(pd.Series(identities).duplicated().sum())
    targets = analogue_targets(outputs)
    target_receipts: dict[str, dict[str, Any]] = {}
    for target in outputs:
        analogue = SPI_STAGE1_ANALOGUE_COLUMNS[target]
        if analogue is None:
            source, positive = "identity", None
        elif degenerate_rows[target] == n:
            source, positive = "identity_degenerate_analogue", None
        elif degenerate_rows[target]:
            source = "twin_conditional_with_degenerate_folds"
            positive = observed[target].to_numpy(dtype=np.float64) > 0.0
        else:
            source = "twin_conditional"
            positive = observed[target].to_numpy(dtype=np.float64) > 0.0
        regimes = [regime for regime in regimes_by_fold[target] if regime is not None]
        distinct_classes = sorted(set(classes_by_fold[target]))
        if len(distinct_classes) > 1:
            raise ValueError(
                f"SPI reference folds disagree on the sign classes of {target!r}: "
                f"{distinct_classes}."
            )
        target_receipts[target] = {
            "analogue": analogue,
            "source": source,
            "reference_regime": sorted(set(regimes)) if regimes else None,
            "reference_classes": (
                list(distinct_classes[0]) if distinct_classes else None
            ),
            "identity_rows_from_degenerate_folds": int(degenerate_rows[target]),
            "kolmogorov": _uniformity(
                sign_uniforms[target], quantiles[target], positive
            ),
            "by_group": _uniformity_by_group(
                sign_uniforms[target], quantiles[target], positive, groups
            ),
        }
    receipt = {
        "draw": SPI_TRANSPORT_DRAW,
        "fit_name": SPI_TRANSPORT_FIT_NAME,
        "seed_offset": SPI_TRANSPORT_SEED_OFFSET,
        "identity_key": SPI_TRANSPORT_KEY_COLUMN,
        "salt": f"{SPI_TRANSPORT_SALT}:<target>:<sign|quantile>",
        "seed": int(seed),
        "recipients": int(n),
        "recipients_sharing_a_twin": shared_identities,
        "folds": int(used),
        "folds_declared": int(folds),
        "cross_fitted": bool(cross_fitted) and not in_sample_folds,
        "in_sample_folds": in_sample_folds,
        "fold_key": "|".join(SPI_TRANSPORT_CELL_COLUMNS) + " (age at the whole year)",
        "fold_salt": SPI_TRANSPORT_FOLD_SALT,
        "fold_receipts": fold_receipts,
        "analogue_targets": list(targets),
        "identity_targets": [
            target for target in outputs if SPI_STAGE1_ANALOGUE_COLUMNS[target] is None
        ],
        "degenerate_targets": [
            target
            for target in outputs
            if degenerate_rows[target] == n and SPI_STAGE1_ANALOGUE_COLUMNS[target]
        ],
        "group_minimum_rows": SPI_TRANSPORT_GROUP_MINIMUM_ROWS,
        "targets": target_receipts,
    }
    return SPITransportResult(
        quantiles=quantiles, sign_uniforms=sign_uniforms, receipt=receipt
    )


def identity_uniforms(
    identities: np.ndarray, *, target: str, part: str, seed: int
) -> np.ndarray:
    """The identity-keyed uniform of every recipient for one target and part."""

    if part not in ("sign", "quantile"):
        raise ValueError("part must be 'sign' or 'quantile'.")
    return stable_identity_uniforms(
        identities, seed=seed, salt=f"{SPI_TRANSPORT_SALT}:{target}:{part}"
    )


def transport_uniforms(
    reference: Any,
    *,
    encoded_recipients: pd.DataFrame,
    observed: pd.DataFrame,
    identities: np.ndarray,
    groups: np.ndarray | None,
    seed: int,
    outputs: Sequence[str] = SPI_INCOME_QRF_OUTPUT_COLUMNS,
) -> SPITransportResult:
    """Locate every drawn recipient's twin on the reference forest.

    ``reference`` is the fitted reference forest (predictors: the encoded
    stage-1 predictors; targets: :func:`analogue_targets`). ``observed`` holds
    the twin's analogue values named by target (:func:`observed_analogues`),
    row-aligned with ``encoded_recipients``. ``identities`` keys the placement
    and identity uniforms; ``groups`` labels each recipient for the per-group
    uniformity receipt (the ``main_source`` bridge level).
    """

    assert_analogue_map_covers(outputs)
    n = len(encoded_recipients)
    identities = np.asarray(identities)
    if identities.shape != (n,):
        raise ValueError("SPI transport identities must have one entry per recipient.")
    if len(observed) != n or not observed.index.equals(encoded_recipients.index):
        raise ValueError("SPI transport observed analogues must align to recipients.")
    # Two copies of one FRS person (the band-donor stage stacks further copies
    # of base households) share a twin, so they share their uniforms and
    # their draws, as they share their band-pool draw; the receipt counts them.
    shared_identities = int(pd.Series(identities).duplicated().sum())
    sign_placement = {
        target: identity_uniforms(identities, target=target, part="sign", seed=seed)
        for target in outputs
    }
    quantile_placement = {
        target: identity_uniforms(identities, target=target, part="quantile", seed=seed)
        for target in outputs
    }
    targets = analogue_targets(outputs)
    if tuple(reference.targets) != targets:
        raise ValueError(
            "SPI reference forest targets drifted from the analogue targets: "
            f"{list(reference.targets)} != {list(targets)}."
        )
    located = reference.conditional_uniforms(
        pd.concat([encoded_recipients, observed[list(targets)]], axis=1),
        sign_placement={target: sign_placement[target] for target in targets},
        quantile_placement={target: quantile_placement[target] for target in targets},
    )
    regimes = dict(reference.regimes())
    # The sign classes the reference's gate orders; None for a degenerate
    # analogue, whose uniforms are placements rather than located positions.
    classes = {
        target: (
            None
            if target in located.degenerate
            else [int(value) for value in reference.sign_classes(target)]
        )
        for target in targets
    }
    quantiles: dict[str, np.ndarray] = {}
    sign_uniforms: dict[str, np.ndarray] = {}
    target_receipts: dict[str, dict[str, Any]] = {}
    for target in outputs:
        analogue = SPI_STAGE1_ANALOGUE_COLUMNS[target]
        if analogue is None:
            quantiles[target] = quantile_placement[target]
            sign_uniforms[target] = sign_placement[target]
            source = "identity"
            positive = None
        else:
            quantiles[target] = np.asarray(located.quantiles[target], dtype=np.float64)
            sign_uniforms[target] = np.asarray(
                located.sign_uniforms[target], dtype=np.float64
            )
            source = (
                "identity_degenerate_analogue"
                if target in located.degenerate
                else "twin_conditional"
            )
            positive = observed[target].to_numpy(dtype=np.float64) > 0.0
        target_receipts[target] = {
            "analogue": analogue,
            "source": source,
            "reference_regime": regimes.get(target),
            "reference_classes": classes.get(target),
            "kolmogorov": _uniformity(
                sign_uniforms[target], quantiles[target], positive
            ),
            "by_group": _uniformity_by_group(
                sign_uniforms[target], quantiles[target], positive, groups
            ),
        }
    receipt = {
        "draw": SPI_TRANSPORT_DRAW,
        "fit_name": SPI_TRANSPORT_FIT_NAME,
        "seed_offset": SPI_TRANSPORT_SEED_OFFSET,
        "identity_key": SPI_TRANSPORT_KEY_COLUMN,
        "salt": f"{SPI_TRANSPORT_SALT}:<target>:<sign|quantile>",
        "seed": int(seed),
        "recipients": int(n),
        "recipients_sharing_a_twin": shared_identities,
        "analogue_targets": list(targets),
        "identity_targets": [
            target for target in outputs if SPI_STAGE1_ANALOGUE_COLUMNS[target] is None
        ],
        "degenerate_targets": list(located.degenerate),
        "ungated_targets": list(located.ungated),
        "group_minimum_rows": SPI_TRANSPORT_GROUP_MINIMUM_ROWS,
        "targets": target_receipts,
    }
    return SPITransportResult(
        quantiles=quantiles, sign_uniforms=sign_uniforms, receipt=receipt
    )


def _kolmogorov(values: np.ndarray) -> float | None:
    """The Kolmogorov distance of ``values`` from U[0, 1)."""

    values = np.sort(np.asarray(values, dtype=np.float64))
    n = len(values)
    if n == 0:
        return None
    steps = np.arange(1, n + 1, dtype=np.float64) / n
    return float(
        max(np.abs(steps - values).max(), np.abs(values - (steps - 1.0 / n)).max())
    )


def _uniformity(
    sign: np.ndarray, quantile: np.ndarray, positive: np.ndarray | None
) -> dict[str, Any]:
    """Uniformity of the sign uniforms, and of the quantiles over positive rows."""

    rows = np.ones(len(sign), dtype=bool) if positive is None else positive
    positive_rows = int(rows.sum())
    return {
        "rows": int(len(sign)),
        "sign": _kolmogorov(sign),
        # A count under the group floor is not reported (CD171 5.2.1), as the
        # by-group receipt withholds a small group.
        "positive_rows": (
            positive_rows if positive_rows >= SPI_TRANSPORT_GROUP_MINIMUM_ROWS else None
        ),
        "quantile": _kolmogorov(quantile[rows]) if rows.any() else None,
    }


def _uniformity_by_group(
    sign: np.ndarray,
    quantile: np.ndarray,
    positive: np.ndarray | None,
    groups: np.ndarray | None,
) -> dict[str, Any] | None:
    if groups is None:
        return None
    groups = np.asarray(groups)
    out: dict[str, Any] = {}
    for label in sorted({str(value) for value in groups.tolist()}):
        mask = groups.astype(str) == label
        if int(mask.sum()) < SPI_TRANSPORT_GROUP_MINIMUM_ROWS:
            out[label] = None
            continue
        out[label] = _uniformity(
            sign[mask], quantile[mask], None if positive is None else positive[mask]
        )
    return out


__all__ = [
    "SPI_STAGE1_ANALOGUE_COLUMNS",
    "SPI_TRANSPORT_CELL_COLUMNS",
    "SPI_TRANSPORT_FOLDS",
    "SPI_TRANSPORT_FOLD_SALT",
    "cell_fold_ids",
    "cross_fit_transport",
    "SPI_TRANSPORT_DRAW",
    "SPI_TRANSPORT_FIT_NAME",
    "SPI_TRANSPORT_GROUP_MINIMUM_ROWS",
    "SPI_TRANSPORT_KEY_COLUMN",
    "SPI_TRANSPORT_SALT",
    "SPI_TRANSPORT_SEED_OFFSET",
    "SPITransportResult",
    "TAXABLE_SAVINGS_INTEREST_ANALOGUE",
    "analogue_source_columns",
    "analogue_targets",
    "assert_analogue_map_covers",
    "identity_uniforms",
    "observed_analogues",
    "transport_uniforms",
]
