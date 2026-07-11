"""SCF-imputed financial-asset inputs and household net worth.

Without this stage the published dataset stores none of the liquid-asset
input columns SSI's resource test reads, so each silently takes the
engine's 0 default: ``bank_account_assets``, ``stock_assets``, and
``bond_assets`` are absent, ``ssi_countable_resources`` (which sums exactly
these three per ``gov.ssa.ssi.eligibility.resources.countable``) computes 0
for every record, ``meets_ssi_resource_test`` is universally true, and every
SSI resource-limit reform silently scores $0. That is the failure of
populace issues #356/#368 (the dropped-column counterpart of #278's zeroed
input bases).

The three person columns are the SSI countable-resource leaves in PolicyEngine-US
(``policyengine_us/parameters/gov/ssa/ssi/eligibility/resources/
countable.yaml``: ``bank_account_assets`` / ``stock_assets`` /
``bond_assets``). They are imputed from the Federal Reserve Survey of
Consumer Finances 2022 public summary extract (``rscfp2022.dta``), using
the regime-gated weighted QRF (``populace.fit.QRF``) — the same imputer the
PUF support stage uses. Each output is a summation of SCF summary-extract
balance-sheet components — the SCF half of the retired enhanced-CPS
construction (cited against the retired pipeline @ 42ed5d45:
``utils/asset_imputation.py`` ``SCF_FINANCIAL_ASSET_TARGETS`` and
``datasets/scf/scf.py``; the component tuples below match it exactly; nothing
is invented). The retired pipeline additionally draws these three leaves from
a SIPP donor and blends the two sources per household
(``FINANCIAL_ASSET_SOURCE_SCF_PROBABILITY = 0.5``), where SIPP supplies the
realistic low liquid-asset mass at the bottom of the distribution. This stage
ships the SCF draw only; the consequence is quantified in the baseline note
below, and the SIPP blend is the populace #356 follow-up:

- ``bank_account_assets`` ← SCF ``liq`` (checking, savings, money-market,
  call accounts).
- ``stock_assets`` ← SCF ``stocks`` + ``nmmf`` (directly-held stock plus
  stock/non-money-market mutual funds — the one code-level summation the
  canonical pipeline applies).
- ``bond_assets`` ← SCF ``bond`` (savings bonds plus other bonds).

Predictors (the eight the ``scf_wealth`` manifest stage declares, matching
the retired SCF QRF's predictor set): age, is_female, cps_race, is_married,
own_children_in_household, employment_income, interest_dividend_income,
social_security_pension_income. SCF-side predictors are the head/household
values in the summary extract (``age``, ``hhsex``, ``racecl5``, ``married``,
``kids``, ``wageinc``, ``intdivinc``, ``ssretinc``); the donor weight is
``wgt``. SCF missing-data sentinels (-1, -7, -8, -9) are replaced with 0 at
the source boundary, per the manifest's ``replace_sentinels`` operation.

Grain (the manifest's ``head_carry`` operation, and the canonical
SCF-source rule): the SCF summary extract is household-grain, so the QRF is
fit on household records (head demographics + household income). It draws
one liquid-wealth vector per recipient household — conditioned on the
reference person's own characteristics — which is carried onto that
reference person; every other household member takes $0. This mirrors the
retired pipeline's reference-person carry
(``combine_sipp_and_scf_financial_assets``: the reference person carries the
household total). Imputing per person instead — drawing a full household-grain
asset value for every member from a donor that is 98.6% nonzero — would put
liquid wealth on children and non-earners and inflate the per-person incidence
far past the incumbent's.

Head-carry is the retired grain and a necessary floor, but on its own it does
not reproduce the dense-native baseline. Measured by the populace #368
acceptance probe (seed 0, policyengine-us 1.764.6): imputing onto the Build H
dense frame makes ``ssi_countable_resources`` nonzero for 40.3% of people (the
dense-native reference is 42.5%) and the $10k/$20k reform scores +$9.7B at 2026
— comfortably above the $1B reform-coverage floor, so the reform binds and the
gate turns green. The delta runs above the dense-native +$1.6B / +$16.1B
reference (restored baseline 4.74M recipients vs 8.05M) for two documented
reasons, both #356 follow-ups, neither a bug in this stage:

1. The probe overlays assets onto a frame whose weights were already calibrated
   with *no* assets, so the SSI caseload was absorbed by the weight solve;
   #356 requires the asset restore to ship with an SSI-target refit. Post-hoc
   asset overlays on populace are documented to land at baseline 3.0-4.0M and a
   $10k/$20k delta of +$9-26B (#356) — this probe's +$9.7B sits in that band.
   The stage itself runs in-build *before* calibration (the correct location),
   so a full certified rebuild re-solves the weights with assets present.
2. It ships the SCF draw *without* the retired SIPP blend above, assigning more
   liquid wealth to the SSI-marginal population than the SIPP+SCF reference.

The nonzero *incidence* already matches the dense-native reference (40.3% vs
42.5%); the gap is amount and calibration, not the grain.

The stage also restores the household ``net_worth`` input from the SCF
``networth`` summary-extract field. At archived commit ``42ed5d45``,
``utils/asset_imputation.py`` lines 15-19 and 182-184 make that field the direct
``scf_net_worth`` QRF anchor; ``calibration/source_impute.py`` lines 1146-1164
fits it, and lines 1324-1338 reconcile construction-only balance-sheet
components to the anchor before exporting their signed sum as ``net_worth``.
Persisting the source-backed anchor is therefore the policy-facing result of
the retired reconciliation without manufacturing or exporting its internal
``scf_*`` component columns. ``net_worth`` is signed: indebted households may
have negative values, so it must never be clipped at zero.

Healing behavior: a frame already carrying all three person columns and the
household net-worth column *with signal* passes through untouched (idempotent).
A constant ``bank_account_assets`` or ``net_worth`` column — indistinguishable
from an engine broadcast default — is re-imputed rather than trusted.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.source_manifest import SourceStageSpec, load_source_manifest
from populace.build.us_runtime.eligibility_inputs import _own_children_in_household
from populace.frame import Frame
from populace.frame.units import US_SCHEMA

__all__ = [
    "SCF_2022_SUMMARY_EXTRACT_MEMBER",
    "SCF_2022_SUMMARY_EXTRACT_MEMBER_SHA256",
    "SCF_2022_SUMMARY_EXTRACT_URL",
    "SCF_2022_SUMMARY_EXTRACT_ZIP_SHA256",
    "SCF_FINANCIAL_ASSET_TARGET_COMPONENTS",
    "SCF_NET_WORTH_TARGET_COMPONENTS",
    "SCF_WEALTH_PREDICTORS",
    "US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS",
    "US_SCF_NET_WORTH_OUTPUT_COLUMNS",
    "US_SCF_WEALTH_NONCONSTANT_HOUSEHOLD_COLUMNS",
    "US_SCF_WEALTH_NONCONSTANT_PERSON_COLUMNS",
    "US_SCF_WEALTH_STAGE_NAME",
    "fetch_scf_2022_summary_extract",
    "impute_us_scf_financial_assets",
    "impute_us_scf_net_worth",
    "load_scf_2022_financial_asset_donor",
    "us_scf_wealth_signal_gate",
    "us_scf_wealth_stage_spec",
    "us_scf_wealth_summary",
    "with_us_scf_wealth_inputs",
]

US_SCF_WEALTH_STAGE_NAME = "scf_wealth"

#: The Federal Reserve SCF 2022 public summary extract (SAS/Stata ``.dta``
#: inside a zip). The stage is fit on this fixed-vintage public file; the
#: source is declared in ``source_stages.json`` and ``US_DONORS`` with the
#: same citation.
SCF_2022_SUMMARY_EXTRACT_URL = (
    "https://www.federalreserve.gov/econres/files/scfp2022s.zip"
)
#: The Stata member inside the zip.
SCF_2022_SUMMARY_EXTRACT_MEMBER = "rscfp2022.dta"
#: SHA-256 of the downloaded zip and its extracted Stata member. The extract is
#: a fixed public artifact, so the build pins both digests and refuses any
#: payload that does not match — a silently reissued vintage cannot slip in
#: (the ``build_us_block_ladder_artifact._sha256`` provenance convention).
#: Verified against the Federal Reserve download on 2026-07-09.
SCF_2022_SUMMARY_EXTRACT_ZIP_SHA256 = (
    "3bb4d890ae2463ff6039ec7692e375f544dd98a55a37ca2cb2340354b9cc9d80"
)
SCF_2022_SUMMARY_EXTRACT_MEMBER_SHA256 = (
    "6b8dd2d935a76ed225ddebc80fb2db22a467f0c80d9a1acaa67b4584aa4bafd1"
)
#: Federal Reserve econres rejects the default urllib user-agent (HTTP 403), so
#: the fetch presents a browser user-agent.
_SCF_HTTP_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

#: The PolicyEngine-facing person input columns this stage produces — the
#: three SSI countable-resource leaves (a real subset of the ``scf_wealth``
#: manifest stage's declared outputs).
US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS: tuple[str, ...] = (
    "bank_account_assets",
    "stock_assets",
    "bond_assets",
)

#: The signed household-grain PolicyEngine input restored from SCF networth.
US_SCF_NET_WORTH_OUTPUT_COLUMNS: tuple[str, ...] = ("net_worth",)

#: Release gates require these person columns to carry signal (≥2 values).
US_SCF_WEALTH_NONCONSTANT_PERSON_COLUMNS: tuple[str, ...] = (
    US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS
)
US_SCF_WEALTH_NONCONSTANT_HOUSEHOLD_COLUMNS: tuple[str, ...] = (
    US_SCF_NET_WORTH_OUTPUT_COLUMNS
)

#: Each output column and the SCF summary-extract components it sums
#: (retired pipeline @ 42ed5d45, ``utils/asset_imputation.py``).
SCF_FINANCIAL_ASSET_TARGET_COMPONENTS: dict[str, tuple[str, ...]] = {
    "bank_account_assets": ("liq",),
    "stock_assets": ("stocks", "nmmf"),
    "bond_assets": ("bond",),
}
SCF_NET_WORTH_TARGET_COMPONENTS: dict[str, tuple[str, ...]] = {
    "net_worth": ("networth",),
}

#: The predictor columns the imputation conditions on, in the manifest's
#: declared order.
SCF_WEALTH_PREDICTORS: tuple[str, ...] = (
    "age",
    "is_female",
    "cps_race",
    "is_married",
    "own_children_in_household",
    "employment_income",
    "interest_dividend_income",
    "social_security_pension_income",
)

#: SCF summary-extract missing-data sentinels (manifest ``replace_sentinels``).
_SCF_SENTINELS: tuple[int, ...] = (-1, -7, -8, -9)

#: SCF ``racecl5`` -> ``cps_race`` code (retired ``datasets/scf/scf.py``:
#: 1 White, 2 Black, 3 Hispanic, 4 Asian, 5 -> 7 Other).
_SCF_RACECL5_TO_CPS_RACE: dict[int, int] = {1: 1, 2: 2, 3: 3, 4: 4, 5: 7}

_DEFAULT_N_ESTIMATORS = 100

#: The donor weight column name handed to the QRF DataFrame fit (kept
#: distinct from every predictor/target name so it is never read as a
#: feature).
_DONOR_WEIGHT_COLUMN = "scf_weight"

#: Raw SCF summary-extract columns used to construct the shared eight-feature
#: donor surface.  The auto-loan family reads the full SCF for its targets but
#: deliberately reuses this summary-extract predictor construction so its QRF
#: conditions on the same concepts as the retired pipeline and the SSI-assets
#: stage.
_SCF_SUMMARY_PREDICTOR_SOURCE_COLUMNS: tuple[str, ...] = (
    "wgt",
    "age",
    "hhsex",
    "racecl5",
    "married",
    "kids",
    "wageinc",
    "intdivinc",
    "ssretinc",
)

_PERSON_WEIGHT_COLUMN = "person_weight"
_HOUSEHOLD_ID_COLUMN = "person_household_id"

#: Weighted person-level nonzero-share plausibility bands. Centred on the
#: pinned incumbent eCPS parity reference (bank 0.54, stock 0.16, bond 0.03)
#: with generous width — the gate exists to catch an all-zero or constant
#: surface, not to pin a point estimate.
_BANK_NONZERO_SHARE_BAND = (0.25, 0.85)
_STOCK_NONZERO_SHARE_BAND = (0.03, 0.40)
_BOND_NONZERO_SHARE_BAND = (0.001, 0.12)
#: Household net worth is nearly always nonzero and legitimately signed. The
#: pinned eCPS nonzero share is 0.997241; the SHA-pinned SCF donor is 0.996678
#: nonzero, 0.921249 positive, and 0.075429 negative by survey weight.
_NET_WORTH_NONZERO_SHARE_BAND = (0.90, 1.0)
_NET_WORTH_POSITIVE_SHARE_BAND = (0.75, 1.0)
_NET_WORTH_NEGATIVE_SHARE_BAND = (0.001, 0.25)


def us_scf_wealth_stage_spec() -> SourceStageSpec:
    """Load the packaged ``scf_wealth`` source-stage manifest entry.

    The policy-facing produced columns must be a subset of the stage's declared
    outputs, so the manifest stays the single source declaration (survey,
    citation, sentinel policy) this runtime implements.
    """

    manifest = load_source_manifest(
        files("populace.build.us").joinpath("source_stages.json")
    )
    stage_map = manifest.stage_map()
    if US_SCF_WEALTH_STAGE_NAME not in stage_map:
        raise ValueError(
            f"US source manifest declares no {US_SCF_WEALTH_STAGE_NAME!r} stage."
        )
    spec = stage_map[US_SCF_WEALTH_STAGE_NAME]
    declared = set(spec.outputs)
    required_outputs = (
        *US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS,
        *US_SCF_NET_WORTH_OUTPUT_COLUMNS,
    )
    missing = [column for column in required_outputs if column not in declared]
    if missing:
        raise ValueError(
            f"{US_SCF_WEALTH_STAGE_NAME!r} manifest stage does not declare "
            f"output(s) {missing}; the runtime and manifest have drifted."
        )
    return spec


def _replace_sentinels(values: pd.Series) -> pd.Series:
    """Replace SCF missing-data sentinels with 0 (manifest boundary rule)."""

    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.mask(numeric.isin(_SCF_SENTINELS), 0.0).fillna(0.0)


def _scf_summary_predictor_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the common SCF predictor/weight table from summary-extract rows.

    The function keeps row order and row count unchanged.  Callers that join a
    second SCF artifact (the auto-loan port) can therefore align the returned
    predictors through the summary extract's ``(y1, yy1)`` keys before applying
    the positive-weight filter.
    """

    missing = sorted(set(_SCF_SUMMARY_PREDICTOR_SOURCE_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(
            f"SCF 2022 summary extract missing required column(s): {missing}."
        )

    donor = pd.DataFrame(index=raw.index)
    donor["age"] = _replace_sentinels(raw["age"]).to_numpy(dtype=np.float64)
    donor["is_female"] = (pd.to_numeric(raw["hhsex"], errors="coerce") == 2).to_numpy(
        dtype=np.float64
    )
    donor["cps_race"] = (
        pd.to_numeric(raw["racecl5"], errors="coerce")
        .map(_SCF_RACECL5_TO_CPS_RACE)
        .fillna(7)
        .to_numpy(dtype=np.float64)
    )
    donor["is_married"] = (
        pd.to_numeric(raw["married"], errors="coerce") == 1
    ).to_numpy(dtype=np.float64)
    donor["own_children_in_household"] = np.maximum(
        _replace_sentinels(raw["kids"]).to_numpy(dtype=np.float64), 0.0
    )
    donor["employment_income"] = _replace_sentinels(raw["wageinc"]).to_numpy(
        dtype=np.float64
    )
    donor["interest_dividend_income"] = _replace_sentinels(raw["intdivinc"]).to_numpy(
        dtype=np.float64
    )
    donor["social_security_pension_income"] = _replace_sentinels(
        raw["ssretinc"]
    ).to_numpy(dtype=np.float64)
    weight = pd.to_numeric(raw["wgt"], errors="coerce").fillna(0.0)
    donor[_DONOR_WEIGHT_COLUMN] = np.maximum(weight.to_numpy(dtype=np.float64), 0.0)
    return donor.loc[:, [*SCF_WEALTH_PREDICTORS, _DONOR_WEIGHT_COLUMN]]


def _sha256_hexdigest(payload: bytes) -> str:
    """SHA-256 hex digest of a byte payload."""

    import hashlib

    return hashlib.sha256(payload).hexdigest()


def fetch_scf_2022_summary_extract(
    cache_dir: str | Path | None = None,
    *,
    expected_member_sha256: str | None = SCF_2022_SUMMARY_EXTRACT_MEMBER_SHA256,
    expected_zip_sha256: str | None = SCF_2022_SUMMARY_EXTRACT_ZIP_SHA256,
) -> Path:
    """Download, verify, and cache the SCF 2022 public summary extract.

    The extract is a fixed public artifact (a single Stata file in a zip), so
    the build pins its SHA-256 and refuses any payload that does not match: a
    reproducible, sha-verified cache. A cached member whose digest matches is
    reused without a network call; a cached member whose digest has drifted is
    re-fetched. The provisioning helper the release builder calls; unit tests
    inject a donor table directly and never hit the network.

    Args:
        cache_dir: Directory to cache the extracted ``.dta`` in. Defaults to
            ``~/.cache/populace/scf``.
        expected_member_sha256: Pinned SHA-256 of ``rscfp2022.dta``. Verified
            on both the cache-hit and the fresh-download paths. ``None`` skips
            member verification (only the no-network cache-return path uses it).
        expected_zip_sha256: Pinned SHA-256 of the downloaded zip. Verified on
            the fresh-download path. ``None`` skips zip verification.

    Returns:
        The path to the extracted, sha-verified ``rscfp2022.dta``.

    Raises:
        ValueError: If a freshly downloaded payload does not match its pin.
    """

    import io
    import urllib.request
    import zipfile

    root = (
        Path(cache_dir)
        if cache_dir is not None
        else Path.home() / ".cache" / "populace" / "scf"
    )
    root.mkdir(parents=True, exist_ok=True)
    target = root / SCF_2022_SUMMARY_EXTRACT_MEMBER
    if target.exists() and target.stat().st_size > 0:
        if expected_member_sha256 is None:
            return target
        if _sha256_hexdigest(target.read_bytes()) == expected_member_sha256:
            return target
        # A cached member that no longer matches the pin is stale; re-fetch.

    request = urllib.request.Request(
        SCF_2022_SUMMARY_EXTRACT_URL,
        headers={"User-Agent": _SCF_HTTP_USER_AGENT},
    )
    with urllib.request.urlopen(request) as response:  # noqa: S310
        payload = response.read()
    if expected_zip_sha256 is not None:
        digest = _sha256_hexdigest(payload)
        if digest != expected_zip_sha256:
            raise ValueError(
                "SCF 2022 summary-extract zip failed sha-256 verification: "
                f"expected {expected_zip_sha256}, got {digest}."
            )
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        with archive.open(SCF_2022_SUMMARY_EXTRACT_MEMBER) as member:
            member_bytes = member.read()
    if expected_member_sha256 is not None:
        digest = _sha256_hexdigest(member_bytes)
        if digest != expected_member_sha256:
            raise ValueError(
                f"SCF 2022 summary-extract member "
                f"{SCF_2022_SUMMARY_EXTRACT_MEMBER!r} failed sha-256 "
                f"verification: expected {expected_member_sha256}, got {digest}."
            )
    target.write_bytes(member_bytes)
    return target


def load_scf_2022_financial_asset_donor(path: str | Path) -> pd.DataFrame:
    """Read the SCF 2022 summary extract into a financial-asset donor table.

    Args:
        path: Path to the Federal Reserve SCF 2022 public summary extract
            (``rscfp2022.dta``, from
            ``https://www.federalreserve.gov/econres/files/scfp2022s.zip``).

    Returns:
        A household-grain donor DataFrame carrying the three liquid-asset
        targets, signed ``net_worth``, the eight predictor columns, and the
        donor weight column (``scf_weight``).

    Raises:
        ValueError: If the extract is missing a required source column.
    """

    raw = pd.read_stata(path, convert_categoricals=False)
    required = {
        "liq",
        "stocks",
        "nmmf",
        "bond",
        "networth",
        *_SCF_SUMMARY_PREDICTOR_SOURCE_COLUMNS,
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(
            f"SCF 2022 summary extract missing required column(s): {missing}."
        )

    donor = pd.DataFrame(index=raw.index)
    # Targets: component sums, sentinel-cleaned, floored at zero.
    for output, components in SCF_FINANCIAL_ASSET_TARGET_COMPONENTS.items():
        total = np.zeros(len(raw), dtype=np.float64)
        for component in components:
            total = total + _replace_sentinels(raw[component]).to_numpy(
                dtype=np.float64
            )
        donor[output] = np.maximum(total, 0.0)
    # SCF networth is already the complete signed balance-sheet aggregate.
    # Preserve negative values: they are indebted-household source signal, not
    # missing-value sentinels.
    donor["net_worth"] = pd.to_numeric(raw["networth"], errors="coerce").fillna(0.0)
    predictors = _scf_summary_predictor_table(raw)
    for column in (*SCF_WEALTH_PREDICTORS, _DONOR_WEIGHT_COLUMN):
        donor[column] = predictors[column].to_numpy(dtype=np.float64)
    return donor.loc[donor[_DONOR_WEIGHT_COLUMN] > 0].reset_index(drop=True)


def _recipient_cps_race(person: pd.DataFrame) -> np.ndarray:
    """Map raw CPS race/Hispanic-origin onto the SCF ``cps_race`` coding."""

    race = pd.to_numeric(person["PRDTRACE"], errors="coerce").fillna(0)
    hispanic = pd.to_numeric(person["PRDTHSP"], errors="coerce").fillna(0)
    result = np.full(len(person), 7.0, dtype=np.float64)  # Other
    result[(race == 1).to_numpy()] = 1.0  # White
    result[(race == 2).to_numpy()] = 2.0  # Black
    result[(race == 4).to_numpy()] = 4.0  # Asian
    result[(hispanic > 0).to_numpy()] = 3.0  # Hispanic overrides race
    return result


def _sum_present(person: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """Sum the numeric columns that are present, treating absence as 0."""

    total = np.zeros(len(person), dtype=np.float64)
    for column in columns:
        if column in person.columns:
            total = total + pd.to_numeric(person[column], errors="coerce").fillna(
                0.0
            ).to_numpy(dtype=np.float64)
    return total


#: Person income components summed into each SCF income predictor, matching
#: the SCF summary extract's household income aggregates on the recipient's
#: own (person-grain) income.
_RECIPIENT_INTEREST_DIVIDEND_COLUMNS = (
    "taxable_interest_income",
    "tax_exempt_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
)
_RECIPIENT_SS_PENSION_COLUMNS = (
    "social_security_retirement",
    "social_security_disability",
    "social_security_survivors",
    "social_security_dependents",
    "taxable_private_pension_income",
    "tax_exempt_private_pension_income",
)

#: Raw/derived recipient person columns the predictor construction reads.
US_SCF_WEALTH_REQUIRED_SOURCE_COLUMNS: tuple[str, ...] = (
    "age",
    "is_female",
    "PRDTRACE",
    "PRDTHSP",
    "A_MARITL",
    "PEPAR1",
    "PEPAR2",
    "PH_SEQ",
    "A_LINENO",
    "employment_income_before_lsr",
    _HOUSEHOLD_ID_COLUMN,
)


def _household_head_mask(person: pd.DataFrame) -> np.ndarray:
    """One reference person (household head) per household.

    The head is the lowest-line-number person within each household (the CPS
    householder is line 1); ties/absent line numbers fall back to the first
    row. This mirrors the retired pipeline's reference-person rule (the first
    person of each household).
    """

    household = pd.to_numeric(person[_HOUSEHOLD_ID_COLUMN], errors="coerce").to_numpy()
    lineno = pd.to_numeric(person["A_LINENO"], errors="coerce").fillna(9_999).to_numpy()
    position = np.arange(len(person))
    # Sort by household, then line number, then original position; the first
    # row of each household group in that order is its head.
    order = np.lexsort((position, lineno, household))
    ordered_household = household[order]
    is_first_in_group = np.empty(len(person), dtype=bool)
    is_first_in_group[0] = True
    is_first_in_group[1:] = ordered_household[1:] != ordered_household[:-1]
    mask = np.zeros(len(person), dtype=bool)
    mask[order[is_first_in_group]] = True
    return mask


def _household_head_predictor_table(person: pd.DataFrame) -> pd.DataFrame:
    """Build the person-grain predictor table the QRF draws from.

    One row per person: the person's own demographics and their own
    (person-level) income components. The retired pipeline predicted the SCF
    model onto the CPS *person* frame using each person's own characteristics
    and then kept only the reference person's draw (the head-carry applied to
    the output, not the predictors). Conditioning on the applicant's own
    income — not a household total — is what keeps a low-income head who
    happens to live with a higher earner from being assigned that earner's
    asset profile and wrongly failing the SSI resource test.
    """

    missing = [
        column
        for column in US_SCF_WEALTH_REQUIRED_SOURCE_COLUMNS
        if column not in person.columns
    ]
    if missing:
        raise ValueError(
            f"US SCF-wealth imputation requires recipient person column(s): {missing}."
        )

    features = pd.DataFrame(index=person.index)
    features["age"] = pd.to_numeric(person["age"], errors="coerce").fillna(0.0)
    features["is_female"] = (
        person["is_female"].astype(float)
        if person["is_female"].dtype != object
        else pd.to_numeric(person["is_female"], errors="coerce").fillna(0.0)
    )
    features["cps_race"] = _recipient_cps_race(person)
    features["is_married"] = (
        pd.to_numeric(person["A_MARITL"], errors="coerce").isin([1, 2, 3])
    ).to_numpy(dtype=np.float64)
    features["own_children_in_household"] = _own_children_in_household(person)
    features["employment_income"] = np.maximum(
        pd.to_numeric(person["employment_income_before_lsr"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64),
        0.0,
    )
    features["interest_dividend_income"] = _sum_present(
        person, _RECIPIENT_INTEREST_DIVIDEND_COLUMNS
    )
    features["social_security_pension_income"] = _sum_present(
        person, _RECIPIENT_SS_PENSION_COLUMNS
    )
    return features.loc[:, list(SCF_WEALTH_PREDICTORS)]


def _draw_us_scf_targets(
    person: pd.DataFrame,
    donor: pd.DataFrame,
    *,
    targets: tuple[str, ...],
    seed: int,
    n_estimators: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Draw SCF household targets for one reference person per household."""

    from populace.fit import QRF

    donor_missing = [
        column
        for column in (*SCF_WEALTH_PREDICTORS, *targets, _DONOR_WEIGHT_COLUMN)
        if column not in donor.columns
    ]
    if donor_missing:
        raise ValueError(f"SCF donor table missing column(s): {donor_missing}.")

    fit_frame = donor.loc[
        :,
        [*SCF_WEALTH_PREDICTORS, *targets, _DONOR_WEIGHT_COLUMN],
    ].copy()
    for column in (*SCF_WEALTH_PREDICTORS, *targets):
        fit_frame[column] = pd.to_numeric(fit_frame[column], errors="coerce")
        if not np.isfinite(fit_frame[column].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"SCF donor column {column!r} contains nonfinite values.")

    fitted = QRF(n_estimators=int(n_estimators), seed=int(seed)).fit(
        fit_frame,
        predictors=list(SCF_WEALTH_PREDICTORS),
        targets=list(targets),
        weights=_DONOR_WEIGHT_COLUMN,
    )
    head_mask = _household_head_mask(person)
    head_features = _household_head_predictor_table(person).loc[head_mask]
    return fitted.predict(head_features), head_mask


def impute_us_scf_financial_assets(
    person: pd.DataFrame,
    donor: pd.DataFrame,
    *,
    seed: int,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
) -> pd.DataFrame:
    """Impute the three SSI financial-asset columns onto a person table.

    Fits the regime-gated weighted QRF on the SCF donor (weighted by the
    donor's survey weight), draws one liquid-wealth vector per household from
    the household head's characteristics, and carries it onto the head; every
    other household member takes $0 (the head-carry grain the module
    docstring documents).

    Args:
        person: The recipient person table (must carry the source columns in
            ``US_SCF_WEALTH_REQUIRED_SOURCE_COLUMNS``).
        donor: An SCF financial-asset donor table (as from
            :func:`load_scf_2022_financial_asset_donor`).
        seed: Imputation seed (drives the QRF fit and the draw).
        n_estimators: Forest size per target.

    Returns:
        A DataFrame indexed like ``person`` with the three output columns,
        each non-negative float64 and zero off the household heads.
    """

    drawn, head_mask = _draw_us_scf_targets(
        person,
        donor,
        targets=US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS,
        seed=seed,
        n_estimators=n_estimators,
    )
    result = pd.DataFrame(index=person.index)
    head_positions = np.flatnonzero(head_mask)
    for column in US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS:
        values = np.zeros(len(person), dtype=np.float64)
        values[head_positions] = np.maximum(
            np.asarray(drawn[column], dtype=np.float64), 0.0
        )
        result[column] = values
    return result


def impute_us_scf_net_worth(
    person: pd.DataFrame,
    household: pd.DataFrame,
    donor: pd.DataFrame,
    *,
    seed: int,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
) -> pd.Series:
    """Impute signed SCF net worth and align it to household-table order.

    The retired source imputer treats SCF ``networth`` as a direct QRF anchor,
    predicts at person grain, and keeps one reference-person value for each
    household. Its internal component reconciliation is constructed to equal
    this anchor. This port persists that policy-facing household result without
    exposing the construction-only ``scf_*`` components.
    """

    if "household_id" not in household:
        raise ValueError("US SCF net-worth imputation requires household_id.")
    drawn, head_mask = _draw_us_scf_targets(
        person,
        donor,
        targets=US_SCF_NET_WORTH_OUTPUT_COLUMNS,
        seed=seed,
        n_estimators=n_estimators,
    )
    reference_household_ids = person.loc[
        head_mask, _HOUSEHOLD_ID_COLUMN
    ].to_numpy()
    if pd.Series(reference_household_ids).duplicated().any():
        raise ValueError(
            "US SCF net-worth imputation produced duplicate reference households."
        )
    values = pd.to_numeric(drawn["net_worth"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("US SCF net-worth predictions contain nonfinite values.")
    by_household = pd.Series(values, index=reference_household_ids)
    household_ids = household["household_id"].to_numpy()
    aligned = by_household.reindex(household_ids)
    if aligned.isna().any():
        missing = household_ids[aligned.isna().to_numpy()][:5].tolist()
        raise ValueError(
            "US SCF net-worth imputation does not cover household id(s) "
            f"{missing}."
        )
    return pd.Series(
        aligned.to_numpy(dtype=np.float64),
        index=household.index,
        name="net_worth",
    )


def _bank_assets_carry_signal(person: pd.DataFrame) -> bool:
    """Whether the persisted bank-assets column is trustworthy as-is.

    A constant column (one observed value) is indistinguishable from the
    engine's broadcast 0 default and must be re-imputed, not passed through.
    """

    values = pd.to_numeric(person["bank_account_assets"], errors="coerce").dropna()
    return values.nunique() > 1


def _net_worth_carries_signal(household: pd.DataFrame) -> bool:
    """Whether persisted household net worth is nondefault and nonconstant."""

    if "net_worth" not in household:
        return False
    values = pd.to_numeric(household["net_worth"], errors="coerce").dropna()
    return values.nunique() > 1 and bool((values != 0).any())


def with_us_scf_wealth_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    scf_donor: pd.DataFrame,
) -> Frame:
    """Impute SSI financial assets and signed household net worth.

    A frame already carrying all three person outputs and household net worth
    with nonconstant signal passes through untouched (idempotent). Missing or
    default-valued surfaces are independently healed from the SCF donor, so an
    existing liquid-asset draw is not replaced merely because net worth is new.

    Args:
        frame: A US-schema frame whose person table carries the recipient
            source columns.
        seed: Build-wide imputation seed.
        time_period: The dataset's time period (recorded; the derivation is
            time-invariant given the donor vintage).
        scf_donor: The SCF financial-asset donor table (from
            :func:`load_scf_2022_financial_asset_donor`).

    Returns:
        A new frame whose person table carries all three asset columns and
        whose household table carries signed ``net_worth``.

    Raises:
        ValueError: If the frame is not US-schema.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("US SCF-wealth inputs require the US schema.")
    # Load (and validate) the manifest stage for provenance even when the
    # imputation is skipped, so a manifest/runtime drift fails loudly.
    us_scf_wealth_stage_spec()
    person = frame.table("person")
    household = frame.table("household")
    have_all_assets = all(
        column in person.columns for column in US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS
    )
    assets_carry_signal = have_all_assets and _bank_assets_carry_signal(person)
    net_worth_carries_signal = _net_worth_carries_signal(household)
    if assets_carry_signal and net_worth_carries_signal:
        return frame

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    if not assets_carry_signal:
        imputed_assets = impute_us_scf_financial_assets(
            person, scf_donor, seed=int(seed)
        )
        for column in US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS:
            tables["person"][column] = imputed_assets[column].to_numpy(
                dtype=np.float64
            )
    if not net_worth_carries_signal:
        tables["household"]["net_worth"] = impute_us_scf_net_worth(
            person,
            household,
            scf_donor,
            seed=int(seed),
        ).to_numpy(dtype=np.float64)
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def us_scf_wealth_summary(frame: Frame) -> dict[str, object]:
    """Weighted financial-asset and net-worth diagnostics."""

    person = frame.table("person")
    household = frame.table("household")
    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    total_weight = float(weights.sum())
    household_weights = np.asarray(
        frame.resolve_weights("household").values, dtype=np.float64
    )
    total_household_weight = float(household_weights.sum())

    def _nonzero_share(column: str) -> float:
        values = pd.to_numeric(person[column], errors="coerce").fillna(0.0).to_numpy()
        positive = values > 0
        return (
            float(weights[positive].sum()) / total_weight if total_weight > 0 else 0.0
        )

    def _mean_positive(column: str) -> float:
        values = pd.to_numeric(person[column], errors="coerce").fillna(0.0).to_numpy()
        positive = values > 0
        positive_weight = float(weights[positive].sum())
        return (
            float(np.average(values[positive], weights=weights[positive]))
            if positive_weight > 0
            else 0.0
        )

    unique_counts = {
        column: int(pd.to_numeric(person[column], errors="coerce").dropna().nunique())
        for column in US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS
        if column in person.columns
    }
    net_worth = (
        pd.to_numeric(household["net_worth"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if "net_worth" in household
        else np.asarray([], dtype=np.float64)
    )

    def _net_worth_share(mask: np.ndarray) -> float:
        return (
            float(household_weights[mask].sum()) / total_household_weight
            if total_household_weight > 0 and len(mask) == len(household_weights)
            else 0.0
        )

    return {
        "bank_account_assets_nonzero_share": _nonzero_share("bank_account_assets")
        if "bank_account_assets" in person.columns
        else 0.0,
        "stock_assets_nonzero_share": _nonzero_share("stock_assets")
        if "stock_assets" in person.columns
        else 0.0,
        "bond_assets_nonzero_share": _nonzero_share("bond_assets")
        if "bond_assets" in person.columns
        else 0.0,
        "bank_account_assets_mean_positive": _mean_positive("bank_account_assets")
        if "bank_account_assets" in person.columns
        else 0.0,
        "bank_nonzero_share_band": list(_BANK_NONZERO_SHARE_BAND),
        "stock_nonzero_share_band": list(_STOCK_NONZERO_SHARE_BAND),
        "bond_nonzero_share_band": list(_BOND_NONZERO_SHARE_BAND),
        "unique_counts": unique_counts,
        "net_worth_unique_count": int(
            pd.to_numeric(household["net_worth"], errors="coerce").dropna().nunique()
        )
        if "net_worth" in household
        else 0,
        "net_worth_nonfinite": int(np.count_nonzero(~np.isfinite(net_worth))),
        "net_worth_nonzero_share": _net_worth_share(
            np.isfinite(net_worth) & (net_worth != 0)
        ),
        "net_worth_positive_share": _net_worth_share(net_worth > 0),
        "net_worth_negative_share": _net_worth_share(net_worth < 0),
        "net_worth_weighted_total": float(
            np.sum(np.nan_to_num(net_worth) * household_weights)
        )
        if len(net_worth) == len(household_weights)
        else 0.0,
        "net_worth_nonzero_share_band": list(_NET_WORTH_NONZERO_SHARE_BAND),
        "net_worth_positive_share_band": list(_NET_WORTH_POSITIVE_SHARE_BAND),
        "net_worth_negative_share_band": list(_NET_WORTH_NEGATIVE_SHARE_BAND),
    }


def us_scf_wealth_signal_gate(frame: Frame) -> GateResult:
    """Require financial assets and net worth to carry plausible distributions.

    Fails when a column is missing or constant, or when a weighted
    nonzero-share leaves its plausibility band — each of which reproduces
    the everyone-has-$0-resources failure of populace #356 (SSI resource-test
    reforms silently score $0).
    """

    person = frame.table("person")
    missing = [
        column
        for column in US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS
        if column not in person.columns
    ]
    missing_household = [
        column
        for column in US_SCF_NET_WORTH_OUTPUT_COLUMNS
        if column not in frame.table("household")
    ]
    if missing or missing_household:
        return GateResult(
            name="scf_wealth_signal",
            passed=False,
            failures=(
                f"person columns missing: {missing}; household columns missing: "
                f"{missing_household}."
            ,),
            details={
                "missing_person": missing,
                "missing_household": missing_household,
            },
        )

    summary = us_scf_wealth_summary(frame)
    failures: list[str] = []
    for column, count in summary["unique_counts"].items():
        if count < 2:
            failures.append(
                f"{column}: constant column (one observed value) — the "
                "financial-asset surface carries no signal."
            )
    for share_key, band, label in (
        ("bank_account_assets_nonzero_share", _BANK_NONZERO_SHARE_BAND, "bank"),
        ("stock_assets_nonzero_share", _STOCK_NONZERO_SHARE_BAND, "stock"),
        ("bond_assets_nonzero_share", _BOND_NONZERO_SHARE_BAND, "bond"),
    ):
        share = float(summary[share_key])
        low, high = band
        if not (low <= share <= high):
            failures.append(
                f"{label} nonzero share {share:.3f} outside plausibility band "
                f"[{low}, {high}]."
            )
    if int(summary["net_worth_nonfinite"]):
        failures.append(
            f"net_worth: {summary['net_worth_nonfinite']} nonfinite value(s)."
        )
    if int(summary["net_worth_unique_count"]) < 2:
        failures.append("net_worth: constant column carries no signal.")
    for share_key, band, label in (
        (
            "net_worth_nonzero_share",
            _NET_WORTH_NONZERO_SHARE_BAND,
            "nonzero",
        ),
        (
            "net_worth_positive_share",
            _NET_WORTH_POSITIVE_SHARE_BAND,
            "positive",
        ),
        (
            "net_worth_negative_share",
            _NET_WORTH_NEGATIVE_SHARE_BAND,
            "negative",
        ),
    ):
        share = float(summary[share_key])
        low, high = band
        if not low <= share <= high:
            failures.append(
                f"net_worth {label} share {share:.3f} outside plausibility "
                f"band [{low}, {high}]."
            )
    return GateResult(
        name="scf_wealth_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
