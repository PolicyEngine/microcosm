# populace-us build provenance

Verbatim snapshots of the scripts that produced the published artifact
(build `populace-us-2024-9f1260b-20260611`, HF revision `4a8e7d39eb9e`),
plus its release manifest. The chain (`run_chain.sh`) runs: full build
(`build_us_candidate.py`, with the primary-source imputation stages in
`primary_source_impute.py`) -> surface extraction
(`extract_target_surface.py`) -> calibration + artifact
(`build_dataset.py`) -> simulation-dependent enrichment
(`enrich_artifact.py`) -> acceptance gates (`check_parity.py`, using
`populace.build` gates).

Every donor is a primary source (CPS ASEC, IRS PUF, Fed SCF 2022, Census
SIPP, CPS-ORG, MEPS-IC parameters, Census ACS 2022); the enhanced CPS is
the scoring benchmark only. These are audit copies — the living code is
in the build worktree until the populace.build.us port lands, after which
the port is canonical and these snapshots freeze.
