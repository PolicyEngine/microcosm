"""SIPP/SCF-blended financial assets and SCF-anchored household net worth.

Without this stage the published dataset stores none of the liquid-asset
input columns SSI's resource test reads, so each silently takes the
engine's 0 default: ``bank_account_assets``, ``stock_assets``, and
``bond_assets`` are absent, ``ssi_countable_resources`` (which sums exactly
these three per ``gov.ssa.ssi.eligibility.resources.countable``) computes 0
for every record, ``meets_ssi_resource_test`` is universally true, and every
SSI resource-limit reform silently scores $0. That is the failure of
populace issues #356/#368 (the dropped-column counterpart of #278's zeroed
input bases).

The three person columns are the SSI countable-resource leaves in
PolicyEngine-US (``bank_account_assets`` / ``stock_assets`` / ``bond_assets``).
Populace issue #374 restores the retired enhanced-CPS source blend: each
recipient household makes one seeded 50/50 source draw, then receives its full
three-leaf vector from either the Federal Reserve Survey of Consumer Finances
2022 or Census SIPP 2023. Leaves are never mixed within a household. SIPP
restores the realistic low liquid-asset mass absent from the SCF-only stage.

The SCF side uses the public summary extract (``rscfp2022.dta``) and the
regime-gated weighted QRF (``populace.fit.QRF``). Its targets exactly match
archived commit ``42ed5d45`` (``utils/asset_imputation.py``):

- ``bank_account_assets`` ← SCF ``liq`` (checking, savings, money-market,
  call accounts).
- ``stock_assets`` ← SCF ``stocks`` + ``nmmf`` (directly-held stock plus
  stock/non-money-market mutual funds — the one code-level summation the
  canonical pipeline applies).
- ``bond_assets`` ← SCF ``bond`` (savings bonds plus other bonds).

The SCF QRF's eight predictors are age, sex, race, marriage, own children,
employment income, interest/dividend income, and Social Security/pension
income. SCF-side values are ``age``, ``hhsex``, ``racecl5``, ``married``,
``kids``, ``wageinc``, ``intdivinc``, and ``ssretinc``; ``wgt`` is the survey
weight. Missing-data sentinels (-1, -7, -8, -9) become 0 at the source boundary.

The SIPP side is the SHA-pinned ``pu2023.csv`` artifact and December
(``MONTHCODE == 12``) person records. It maps the three leaves directly from
``TVAL_BANK`` / ``TVAL_STMF`` / ``TVAL_BOND``. Its predictor mapping matches
the *final* asset QRF at ``42ed5d45``: employment, interest, dividend, rental,
Social Security, retirement, non-SSI income, age, sex, marriage, under-18
count, under-6 count, and household size. That is the reviewed 13-field mapping,
not the later ten-field shorthand quoted in #374. The pinned 2023 bytes plus
the final 13-field transform are intentional: the archived commit had already
changed its default file to ``pu2024.csv``, while #374 explicitly names the
earlier 2023 donor. See :mod:`populace.build.us_runtime.sipp_financial_assets`
for the exact annualization, target-quality masks, and target-balanced cap.

Grain is one three-leaf draw per household, carried by the CPS reference person;
every other member receives $0 from either source. This is the explicit #374
contract and fixes a literal archive asymmetry: archived
``combine_sipp_and_scf_financial_assets`` zeroed non-reference SCF values but
left person-level SIPP predictions in place. Populace applies the requested
reference-person carry to both sources. The selector consumes the build seed
and period; the archive instead used an unseeded stable hash of period and
household id. Using the stage seed is likewise an explicit #374 requirement.

Pre-blend measured baseline (retained for comparison): the populace #368
acceptance probe (seed 0, PolicyEngine-US 1.764.6) overlaid the former SCF-only
stage onto the Build H dense frame. ``ssi_countable_resources`` became nonzero
for 40.3% of people versus 42.5% in the dense-native reference, and the 2026
$10k/$20k reform scored +$9.7B versus the dense-native +$1.6B. Those figures
predate this #374 SIPP blend. Two effects explained the gap:

1. The probe overlays assets onto a frame whose weights were already calibrated
   with *no* assets, so the SSI caseload was absorbed by the weight solve;
   #356 requires the asset restore to ship with an SSI-target refit. Post-hoc
   asset overlays on populace are documented to land at baseline 3.0-4.0M and a
   $10k/$20k delta of +$9-26B (#356) — this probe's +$9.7B sits in that band.
   The stage itself runs in-build *before* calibration (the correct location),
   so a full certified rebuild re-solves the weights with assets present.
2. The probe used the former SCF-only draw, assigning more liquid wealth to the
   SSI-marginal population than the SIPP+SCF construction now shipped here.

The nonzero *incidence* already matches the dense-native reference (40.3% vs
42.5%); the gap is amount and calibration, not the grain.

Household ``net_worth`` stays exactly as before: a signed QRF draw from SCF
``networth``. At ``42ed5d45`` the blended leaves entered an internal component
reconciliation, but they were protected while SCF-only components were adjusted
until the signed sum equaled the direct ``scf_net_worth`` anchor. Populace does
not manufacture those construction-only components, so retaining the direct
SCF anchor preserves the archive's observable result. Indebted households may
have negative net worth; it is never clipped.

Healing behavior: a frame already carrying all three person columns, the blend
audit, and household net worth with signal passes through untouched. Supplying
a SIPP donor to an old SCF-only frame (no audit) redraws the three leaves; a
constant asset or net-worth surface is likewise healed.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.source_manifest import SourceStageSpec, load_source_manifest
from populace.build.us_runtime.eligibility_inputs import _own_children_in_household
from populace.build.us_runtime.sipp_financial_assets import (
    SIPP_2023_FINANCIAL_ASSET_DONOR_REPOSITORY_ID_PARTS,
    SIPP_2023_FINANCIAL_ASSET_DONOR_REPOSITORY_TYPE,
    SIPP_2023_FINANCIAL_ASSET_DONOR_REVISION,
    SIPP_2023_FINANCIAL_ASSET_DONOR_SHA256,
    SIPP_2023_FINANCIAL_ASSET_DONOR_SIZE_BYTES,
    SIPP_FINANCIAL_ASSET_MODEL_PREDICTORS,
    SIPP_FINANCIAL_ASSET_SOURCE_COLUMNS,
    SIPP_FINANCIAL_ASSET_TARGET_ALLOCATION_COLUMNS,
    SIPP_FINANCIAL_ASSET_TARGET_SOURCE_COLUMNS,
    impute_us_sipp_financial_assets,
)
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
    "FINANCIAL_ASSET_BLEND_AUDIT_KEY",
    "FINANCIAL_ASSET_SOURCE_SCF_PROBABILITY",
    "US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS",
    "US_SCF_NET_WORTH_OUTPUT_COLUMNS",
    "US_SCF_WEALTH_NONCONSTANT_HOUSEHOLD_COLUMNS",
    "US_SCF_WEALTH_NONCONSTANT_PERSON_COLUMNS",
    "US_SCF_WEALTH_STAGE_NAME",
    "financial_asset_source_is_scf",
    "fetch_scf_2022_summary_extract",
    "impute_us_sipp_scf_financial_assets",
    "impute_us_scf_financial_assets",
    "impute_us_scf_net_worth",
    "load_scf_2022_financial_asset_donor",
    "us_scf_wealth_signal_gate",
    "us_scf_wealth_stage_spec",
    "us_scf_wealth_summary",
    "with_us_scf_wealth_inputs",
]

US_SCF_WEALTH_STAGE_NAME = "scf_wealth"

#: One household chooses one source for its complete three-leaf vector.
FINANCIAL_ASSET_SOURCE_SCF_PROBABILITY = 0.5
#: DataFrame ``attrs`` key carrying scalar diagnostics for the seeded blend.
FINANCIAL_ASSET_BLEND_AUDIT_KEY = "sipp_scf_financial_asset_blend"

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
_BANK_LOW_TAIL_THRESHOLD = 2_000.0
_SCF_SOURCE_SHARE_BAND = (0.4, 0.6)

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
    sipp_artifacts = [
        artifact
        for artifact in spec.artifacts
        if artifact.get("member") == "pu2023.csv"
    ]
    if len(sipp_artifacts) != 1:
        raise ValueError(
            f"{US_SCF_WEALTH_STAGE_NAME!r} manifest must declare exactly one "
            "pu2023.csv artifact."
        )
    sipp_artifact = sipp_artifacts[0]
    expected_pin = {
        "revision": SIPP_2023_FINANCIAL_ASSET_DONOR_REVISION,
        "sha256": SIPP_2023_FINANCIAL_ASSET_DONOR_SHA256,
        "size_bytes": SIPP_2023_FINANCIAL_ASSET_DONOR_SIZE_BYTES,
    }
    pin_drift = {
        key: (sipp_artifact.get(key), expected)
        for key, expected in expected_pin.items()
        if sipp_artifact.get(key) != expected
    }
    if pin_drift:
        raise ValueError(
            f"{US_SCF_WEALTH_STAGE_NAME!r} SIPP artifact pin drift: {pin_drift}."
        )
    repository = sipp_artifact.get("repository")
    if not isinstance(repository, Mapping):
        raise ValueError(
            f"{US_SCF_WEALTH_STAGE_NAME!r} SIPP artifact repository is missing."
        )
    name_parts = repository.get("name_parts", ())
    if not isinstance(name_parts, list) or not all(
        isinstance(part, str) for part in name_parts
    ):
        raise ValueError(
            f"{US_SCF_WEALTH_STAGE_NAME!r} SIPP repository name parts are invalid."
        )
    declared_repo_id = f"{repository.get('owner', '')}/{''.join(name_parts)}"
    expected_repo_id = "".join(SIPP_2023_FINANCIAL_ASSET_DONOR_REPOSITORY_ID_PARTS)
    if (
        repository.get("provider") != "huggingface_hub"
        or declared_repo_id != expected_repo_id
        or repository.get("repo_type")
        != SIPP_2023_FINANCIAL_ASSET_DONOR_REPOSITORY_TYPE
    ):
        raise ValueError(
            f"{US_SCF_WEALTH_STAGE_NAME!r} SIPP repository declaration has drifted."
        )
    sipp_reads = [
        operation
        for operation in spec.operations
        if operation.kind == "read_table"
        and operation.parameters.get("table") == "sipp_2023_person"
    ]
    if len(sipp_reads) != 1:
        raise ValueError(
            f"{US_SCF_WEALTH_STAGE_NAME!r} manifest must declare exactly one "
            "SIPP person-table read."
        )
    sipp_read = sipp_reads[0].parameters
    declared_source_columns = tuple(sipp_read.get("source_columns", ()))
    if len(declared_source_columns) != len(set(declared_source_columns)) or set(
        declared_source_columns
    ) != set(SIPP_FINANCIAL_ASSET_SOURCE_COLUMNS):
        raise ValueError(
            f"{US_SCF_WEALTH_STAGE_NAME!r} SIPP source columns have drifted."
        )
    declared_allocation_columns = {
        target: tuple(columns)
        for target, columns in sipp_read.get(
            "target_allocation_status_columns", {}
        ).items()
    }
    if (
        sipp_read.get("delimiter") != "|"
        or sipp_read.get("month_column") != "MONTHCODE"
        or sipp_read.get("month") != 12
        or sipp_read.get("weight") != "WPFINWGT"
        or sipp_read.get("targets") != SIPP_FINANCIAL_ASSET_TARGET_SOURCE_COLUMNS
        or sipp_read.get("allocation_status_observed_values") != [0, 1, 9]
        or declared_allocation_columns != SIPP_FINANCIAL_ASSET_TARGET_ALLOCATION_COLUMNS
    ):
        raise ValueError(
            f"{US_SCF_WEALTH_STAGE_NAME!r} SIPP read/mask declaration has drifted."
        )
    sipp_fits = [
        operation
        for operation in spec.operations
        if operation.kind == "fit_weighted_qrf"
        and operation.parameters.get("donor") == "sipp_2023_person"
    ]
    if len(sipp_fits) != 1:
        raise ValueError(
            f"{US_SCF_WEALTH_STAGE_NAME!r} manifest must declare exactly one "
            "SIPP financial-asset QRF."
        )
    sipp_fit = sipp_fits[0].parameters
    if tuple(sipp_fit.get("predictors", ())) != SIPP_FINANCIAL_ASSET_MODEL_PREDICTORS:
        raise ValueError(
            f"{US_SCF_WEALTH_STAGE_NAME!r} SIPP predictor mapping has drifted."
        )
    if sipp_fit.get("targets") != SIPP_FINANCIAL_ASSET_TARGET_SOURCE_COLUMNS:
        raise ValueError(
            f"{US_SCF_WEALTH_STAGE_NAME!r} SIPP target mapping has drifted."
        )
    blends = [
        operation
        for operation in spec.operations
        if operation.kind == "derive"
        and operation.parameters.get("method") == "seeded_household_source_block_blend"
    ]
    if len(blends) != 1 or blends[0].parameters.get("scf_probability") != (
        FINANCIAL_ASSET_SOURCE_SCF_PROBABILITY
    ):
        raise ValueError(f"{US_SCF_WEALTH_STAGE_NAME!r} 50/50 block blend has drifted.")
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


def financial_asset_source_is_scf(
    household_ids: pd.Series | np.ndarray,
    *,
    seed: int,
    time_period: int,
    scf_probability: float = FINANCIAL_ASSET_SOURCE_SCF_PROBABILITY,
) -> np.ndarray:
    """Draw one reproducible financial-asset source per household.

    Household ids are sorted before drawing, so reordering people does not
    change a household's source.  The build seed and period feed a dedicated
    #374 seed stream; the returned mask is therefore independent of either
    source model's QRF draw stream.
    """

    probability = float(scf_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("scf_probability must lie in [0, 1].")
    numeric_ids = pd.to_numeric(pd.Series(household_ids), errors="coerce")
    ids = numeric_ids.to_numpy()
    if numeric_ids.isna().any() or not np.isfinite(ids).all():
        raise ValueError("Financial-asset source selection requires household ids.")
    unique_ids = np.unique(ids)
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(time_period), 374])
    )
    source_by_household = pd.Series(
        rng.random(len(unique_ids)) < probability,
        index=unique_ids,
    )
    selected = source_by_household.reindex(ids)
    if selected.isna().any():  # pragma: no cover - guarded by the same id vector.
        raise ValueError("Financial-asset source selection failed to map a household.")
    return selected.to_numpy(dtype=bool)


def impute_us_sipp_scf_financial_assets(
    person: pd.DataFrame,
    scf_donor: pd.DataFrame,
    sipp_donor: pd.DataFrame,
    *,
    seed: int,
    time_period: int,
    n_estimators: int = _DEFAULT_N_ESTIMATORS,
) -> pd.DataFrame:
    """Blend complete SIPP or SCF leaf vectors at recipient-household grain.

    Both source models first produce their own reference-person-carried
    vectors.  A single seeded household mask then chooses all three columns
    from one of those vectors.  Scalar diagnostics in
    :data:`FINANCIAL_ASSET_BLEND_AUDIT_KEY` let the release gate compare the
    shipped blend with its parallel SCF-only draw without persisting a
    policy-facing source-indicator column.
    """

    scf = impute_us_scf_financial_assets(
        person,
        scf_donor,
        seed=int(seed),
        n_estimators=int(n_estimators),
    )
    sipp = impute_us_sipp_financial_assets(
        person,
        sipp_donor,
        seed=int(seed),
        n_estimators=int(n_estimators),
    )
    head_mask = _household_head_mask(person)
    head_positions = np.flatnonzero(head_mask)
    head_household_ids = person.loc[head_mask, _HOUSEHOLD_ID_COLUMN]
    head_source_is_scf = financial_asset_source_is_scf(
        head_household_ids,
        seed=int(seed),
        time_period=int(time_period),
    )
    source_is_scf = np.zeros(len(person), dtype=bool)
    source_is_scf[head_positions] = head_source_is_scf

    result = pd.DataFrame(index=person.index)
    for column in US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS:
        result[column] = np.where(
            source_is_scf,
            scf[column].to_numpy(dtype=np.float64),
            sipp[column].to_numpy(dtype=np.float64),
        )

    scf_head_bank = scf.loc[head_mask, "bank_account_assets"].to_numpy(dtype=np.float64)
    blended_head_bank = result.loc[head_mask, "bank_account_assets"].to_numpy(
        dtype=np.float64
    )
    selected_scf_bank = blended_head_bank[head_source_is_scf]
    selected_sipp_bank = blended_head_bank[~head_source_is_scf]

    def _median_or_nan(values: np.ndarray) -> float:
        return float(np.median(values)) if len(values) else float("nan")

    household_count = int(len(head_positions))
    scf_count = int(np.count_nonzero(head_source_is_scf))
    sipp_count = household_count - scf_count
    result.attrs[FINANCIAL_ASSET_BLEND_AUDIT_KEY] = {
        "schema_version": 1,
        "seed": int(seed),
        "time_period": int(time_period),
        "scf_probability": FINANCIAL_ASSET_SOURCE_SCF_PROBABILITY,
        "household_count": household_count,
        "scf_household_count": scf_count,
        "sipp_household_count": sipp_count,
        "scf_household_share": (
            float(scf_count / household_count) if household_count else float("nan")
        ),
        "bank_low_tail_threshold": _BANK_LOW_TAIL_THRESHOLD,
        "scf_only_bank_low_tail_share": (
            float(np.mean(scf_head_bank <= _BANK_LOW_TAIL_THRESHOLD))
            if household_count
            else float("nan")
        ),
        "blended_bank_low_tail_share": (
            float(np.mean(blended_head_bank <= _BANK_LOW_TAIL_THRESHOLD))
            if household_count
            else float("nan")
        ),
        "scf_selected_bank_median": _median_or_nan(selected_scf_bank),
        "sipp_selected_bank_median": _median_or_nan(selected_sipp_bank),
    }
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
    reference_household_ids = person.loc[head_mask, _HOUSEHOLD_ID_COLUMN].to_numpy()
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
            f"US SCF net-worth imputation does not cover household id(s) {missing}."
        )
    return pd.Series(
        aligned.to_numpy(dtype=np.float64),
        index=household.index,
        name="net_worth",
    )


def _financial_assets_carry_signal(person: pd.DataFrame) -> bool:
    """Whether the persisted financial-asset surface is trustworthy as-is.

    Nonfinite values mark a corrupted surface and force re-imputation. The
    engine-default check is JOINT across the three leaves: a surface where
    every leaf is constant (the all-zero engine default) must be re-imputed,
    but a single legitimately constant leaf must not force a redraw — bond
    holdings are ~97% zero in the donor, so a small or chunked frame can
    draw a constant bond column from a perfectly healthy imputation. A
    per-leaf nonconstancy test made pass-through platform-dependent (the
    #510 CI failure) and would silently redraw small production chunks;
    flattened cross-leaf uniqueness would accept three DISTINCT constant
    leaves, so the check requires at least one genuinely nonconstant leaf.
    """

    any_nonconstant = False
    for column in US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS:
        values = pd.to_numeric(person[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if len(values) != len(person) or not np.isfinite(values).all():
            return False
        if np.unique(values).size >= 2:
            any_nonconstant = True
    return any_nonconstant


def _net_worth_carries_signal(household: pd.DataFrame) -> bool:
    """Whether persisted household net worth is finite and nonconstant."""

    if "net_worth" not in household:
        return False
    values = pd.to_numeric(household["net_worth"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    return (
        len(values) == len(household)
        and bool(np.isfinite(values).all())
        and np.unique(values).size > 1
        and bool((values != 0).any())
    )


def _blend_audit_matches(
    person: pd.DataFrame,
    *,
    seed: int,
    time_period: int,
) -> bool:
    audit = person.attrs.get(FINANCIAL_ASSET_BLEND_AUDIT_KEY)
    return (
        isinstance(audit, Mapping)
        and audit.get("schema_version") == 1
        and audit.get("seed") == int(seed)
        and audit.get("time_period") == int(time_period)
        and audit.get("scf_probability") == FINANCIAL_ASSET_SOURCE_SCF_PROBABILITY
    )


def with_us_scf_wealth_inputs(
    frame: Frame,
    *,
    seed: int,
    time_period: int,
    scf_donor: pd.DataFrame,
    sipp_donor: pd.DataFrame | None = None,
) -> Frame:
    """Impute blended financial assets and SCF-anchored household net worth.

    A frame already carrying all three person outputs and household net worth
    with nonconstant signal passes through untouched (idempotent). When a SIPP
    donor is supplied, the frame must also carry the matching blend audit;
    this makes a former SCF-only checkpoint heal into the #374 blend. Missing
    or default-valued surfaces are independently healed, so an existing asset
    draw is not replaced merely because net worth is new.

    Args:
        frame: A US-schema frame whose person table carries the recipient
            source columns.
        seed: Build-wide imputation seed.
        time_period: The dataset's time period, used with ``seed`` for the
            household source selector.
        scf_donor: The SCF financial-asset donor table (from
            :func:`load_scf_2022_financial_asset_donor`).
        sipp_donor: Optional SIPP financial-asset donor table. Supplying it
            enables the seeded 50/50 SIPP/SCF blend; omitting it preserves the
            historical SCF-only behavior for existing callers.

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
    blend_requested = sipp_donor is not None
    blend_carries_signal = not blend_requested or _blend_audit_matches(
        person,
        seed=int(seed),
        time_period=int(time_period),
    )
    assets_carry_signal = (
        have_all_assets
        and _financial_assets_carry_signal(person)
        and blend_carries_signal
    )
    net_worth_carries_signal = _net_worth_carries_signal(household)
    if assets_carry_signal and net_worth_carries_signal:
        return frame

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    if not assets_carry_signal:
        if sipp_donor is None:
            imputed_assets = impute_us_scf_financial_assets(
                person,
                scf_donor,
                seed=int(seed),
            )
        else:
            imputed_assets = impute_us_sipp_scf_financial_assets(
                person,
                scf_donor,
                sipp_donor,
                seed=int(seed),
                time_period=int(time_period),
            )
        for column in US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS:
            tables["person"][column] = imputed_assets[column].to_numpy(dtype=np.float64)
        if FINANCIAL_ASSET_BLEND_AUDIT_KEY in imputed_assets.attrs:
            tables["person"].attrs[FINANCIAL_ASSET_BLEND_AUDIT_KEY] = dict(
                imputed_assets.attrs[FINANCIAL_ASSET_BLEND_AUDIT_KEY]
            )
        else:
            tables["person"].attrs.pop(FINANCIAL_ASSET_BLEND_AUDIT_KEY, None)
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
    blend_audit = person.attrs.get(FINANCIAL_ASSET_BLEND_AUDIT_KEY)
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
    finite_net_worth = np.isfinite(net_worth)

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
            finite_net_worth & (net_worth != 0)
        ),
        "net_worth_positive_share": _net_worth_share(
            finite_net_worth & (net_worth > 0)
        ),
        "net_worth_negative_share": _net_worth_share(
            finite_net_worth & (net_worth < 0)
        ),
        "net_worth_weighted_total": float(
            np.sum(np.where(finite_net_worth, net_worth, 0.0) * household_weights)
        )
        if len(net_worth) == len(household_weights)
        else 0.0,
        "net_worth_nonzero_share_band": list(_NET_WORTH_NONZERO_SHARE_BAND),
        "net_worth_positive_share_band": list(_NET_WORTH_POSITIVE_SHARE_BAND),
        "net_worth_negative_share_band": list(_NET_WORTH_NEGATIVE_SHARE_BAND),
        "financial_asset_blend": (
            dict(blend_audit) if isinstance(blend_audit, Mapping) else None
        ),
    }


def us_scf_wealth_signal_gate(
    frame: Frame,
    *,
    require_sipp_blend: bool = False,
) -> GateResult:
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
                f"{missing_household}.",
            ),
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
    if require_sipp_blend:
        audit = summary["financial_asset_blend"]
        if not isinstance(audit, Mapping):
            failures.append(
                "SIPP/SCF blend audit is missing; assets may still be SCF-only."
            )
        else:
            try:
                head_mask = _household_head_mask(person)
                expected_source_is_scf = financial_asset_source_is_scf(
                    person.loc[head_mask, _HOUSEHOLD_ID_COLUMN],
                    seed=int(audit["seed"]),
                    time_period=int(audit["time_period"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                failures.append(f"SIPP/SCF blend audit cannot be verified: {error}")
                expected_source_is_scf = np.asarray([], dtype=bool)
                head_mask = np.zeros(len(person), dtype=bool)
            household_count = int(len(expected_source_is_scf))
            scf_count = int(np.count_nonzero(expected_source_is_scf))
            sipp_count = household_count - scf_count
            if household_count <= 0 or scf_count <= 0 or sipp_count <= 0:
                failures.append(
                    "SIPP/SCF blend did not select both sources: "
                    f"households={household_count}, SCF={scf_count}, "
                    f"SIPP={sipp_count}."
                )
            recorded_counts = (
                int(audit.get("household_count", 0)),
                int(audit.get("scf_household_count", 0)),
                int(audit.get("sipp_household_count", 0)),
            )
            actual_counts = (household_count, scf_count, sipp_count)
            if recorded_counts != actual_counts:
                failures.append(
                    "SIPP/SCF blend audit source counts do not match the seeded "
                    f"household selector: recorded={recorded_counts}, "
                    f"actual={actual_counts}."
                )
            scf_share = (
                float(scf_count / household_count) if household_count else float("nan")
            )
            low, high = _SCF_SOURCE_SHARE_BAND
            if not np.isfinite(scf_share) or not low <= scf_share <= high:
                failures.append(
                    f"SCF household source share {scf_share:.3f} outside seeded "
                    f"blend band [{low}, {high}]."
                )
            head_bank = person.loc[head_mask, "bank_account_assets"].to_numpy(
                dtype=np.float64
            )
            if scf_count > 0 and sipp_count > 0:
                actual_scf_median = float(np.median(head_bank[expected_source_is_scf]))
                actual_sipp_median = float(
                    np.median(head_bank[~expected_source_is_scf])
                )
                recorded_scf_median = float(
                    audit.get("scf_selected_bank_median", float("nan"))
                )
                recorded_sipp_median = float(
                    audit.get("sipp_selected_bank_median", float("nan"))
                )
                if not np.isclose(recorded_scf_median, actual_scf_median):
                    failures.append(
                        "SIPP/SCF blend audit SCF bank median does not match "
                        f"shipped SCF-selected heads: recorded="
                        f"{recorded_scf_median:,.2f}, actual={actual_scf_median:,.2f}."
                    )
                if not np.isclose(recorded_sipp_median, actual_sipp_median):
                    failures.append(
                        "SIPP/SCF blend audit SIPP bank median does not match "
                        f"shipped SIPP-selected heads: recorded="
                        f"{recorded_sipp_median:,.2f}, "
                        f"actual={actual_sipp_median:,.2f}."
                    )
                if actual_sipp_median >= actual_scf_median:
                    failures.append(
                        "SIPP-selected household bank median is not below the "
                        f"SCF-selected median: SIPP=${actual_sipp_median:,.2f}, "
                        f"SCF=${actual_scf_median:,.2f}."
                    )
            scf_only_low_tail = float(
                audit.get("scf_only_bank_low_tail_share", float("nan"))
            )
            blended_low_tail = float(
                audit.get("blended_bank_low_tail_share", float("nan"))
            )
            actual_blended_low_tail = (
                float(np.mean(head_bank <= _BANK_LOW_TAIL_THRESHOLD))
                if household_count
                else float("nan")
            )
            if not np.isclose(
                blended_low_tail,
                actual_blended_low_tail,
                equal_nan=False,
            ):
                failures.append(
                    "SIPP/SCF blend audit low-tail share does not match the "
                    f"shipped heads: recorded={blended_low_tail:.3f}, "
                    f"actual={actual_blended_low_tail:.3f}."
                )
            if (
                not np.isfinite(scf_only_low_tail)
                or not np.isfinite(actual_blended_low_tail)
                or actual_blended_low_tail <= scf_only_low_tail
            ):
                failures.append(
                    "Blended bank-asset low-tail share did not increase over "
                    f"SCF-only at ${_BANK_LOW_TAIL_THRESHOLD:,.0f}: "
                    f"blend={actual_blended_low_tail:.3f}, "
                    f"SCF-only={scf_only_low_tail:.3f}."
                )
    return GateResult(
        name="scf_wealth_signal",
        passed=not failures,
        failures=tuple(failures),
        details=summary,
    )
