# Prepare age-15 hours donors and recipients

`tools/acs_hours_donor_preparation.py` selects numeric age-15 donor and recipient records for a later hours model. It does not impute hours, interpret missing earnings as zero, change weights, patch an H5, or call a country model. It imports NumPy and the standard library only.

The donor role is the complement of the accepted full ACS recovery set, restricted to numeric clone index zero. This follows the published Build P ASEC/PUF lineage convention. It is **not a fresh join to raw ASEC files**. ACS rows also have clone zero, so clone index alone cannot identify donors. The helper verifies the full ACS sidecar's row positions, current IDs, source keys, household coverage and SPORDER presence before taking its complement.

The caller must explicitly choose `pooled_2022_2024` or `source_2024_only`. Both select source age 15, retaining the original source year. Pooling survey years is a transport choice, not a claim that all those observations came from the 2024 survey. Recipients are all ACS records with source `AGEP == 15`, including any supplied observed hours. Preparation does not declare every selected recipient eligible for imputation or overwrite observed hours.

## Numeric inputs

The input person projection must contain every parent person exactly once **in original parent row order**. The household projection must contain every parent household exactly once; its order is unrestricted. An independently reviewed numeric extractor owns that projection and its declaration of the parent identity. This helper does not reopen H5 to authenticate the extraction.

| Input | Required fields | Suggested dtype |
| --- | --- | --- |
| Person identities and age | `person_id`, `person_household_id`, `person_source_id`, `source_year`, `source_household_id`, `source_row_id`, `person_support_clone_index`, `A_AGE` | `int64` |
| Person raw evidence | `HRSWK`, `WKSWORK`, `A_HRS1`, `SPORDER`, `AGEP`, `WAGP`, `SEMP` | `float64`, preserving NaN |
| Household identities | `household_id`, `household_support_clone_index` | `int64` |

Use one-dimensional structured `.npy` arrays. Extra scalar numeric person fields are preserved in each selected output row. Object, string, Boolean, complex, nested and array-valued fields are refused. Numeric IDs must be finite integers within the exact binary64 integer range. Do not put `parent_person_row`, `WKHP`, `WKL` or `FWKHP` in the person projection: these output fields come from input position or the separately accepted source recovery.

The helper takes the full `acs-source-hours.npy` and `RECOVERY.json` produced by the numeric source-recovery tool. It refuses a pilot receipt, mismatched full-parent counts, duplicate IDs or keys, missing or mixed household membership, disagreement between person and household clone indices, and ACS source-age disagreement. Numeric source identities can collide between ACS and the donor pool, and source-row IDs can repeat across donor years or clone roles; uniqueness is checked within the appropriate source/year/clone namespace.

The projection receipt contains just three SHA256 declarations:

```json
{
  "parent_sha256": "<exact parent H5 digest>",
  "person_sha256": "<person.npy digest>",
  "household_sha256": "<household.npy digest>"
}
```

Supply the independently accepted SHA256 of that receipt and the accepted full recovery receipt. The helper verifies these receipt bytes, all three numeric file hashes, the ACS sidecar size, and agreement of parent identity. It rechecks numeric file hashes after preparation. The result explicitly labels projection provenance as the trusted upstream extractor's declaration.

## Source values and missingness

ASEC raw domains are `HRSWK` 0–99, `WKSWORK` 0–52, and `A_HRS1` −1–99. `A_HRS1 = -1` remains a valid NIU code. `A_AGE` permits 0–80 and 85, with 80 and 85 representing the published older-age groups. These codes are specified in the Census ASEC public-use dictionaries for [2022](https://www2.census.gov/programs-surveys/cps/datasets/2022/march/asec2022_ddl_pub_full.pdf), [2023](https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asec2023_ddl_pub_full.pdf), and [2024](https://www2.census.gov/programs-surveys/cps/datasets/2024/march/asec2024_ddl_pub_full.pdf), person-record demographic, edited labor-force and work-experience sections.

Annual hours and weeks must agree on zero status. Reference-week hours may differ from annual usual hours; there is no equality requirement between `A_HRS1` and `HRSWK`. The helper refuses missing or out-of-domain donor raw hours rather than silently cleaning them. It preserves all three source values without applying the existing engine-input nonnegative/clipping transformation.

Recovered ACS `WKHP`, `WKL` and `FWKHP` are copied exactly, including NaNs. `WAGP`, `SEMP`, every optional numeric raw feature and their NaNs/NIU codes also pass through unchanged. No earnings-derived worker indicator or source-unknown-to-zero conversion is created. The receipt retains the accepted ACS reconstruction basis, unknown original staging revision and PUMA-comparison qualification.

## Run and outputs

```bash
uv run --no-sync python tools/acs_hours_donor_preparation.py \
  --person-path /absolute/person.npy \
  --household-path /absolute/household.npy \
  --acs-path /absolute/acs-source-hours.npy \
  --recovery-path /absolute/RECOVERY.json \
  --projection-path /absolute/projection.json \
  --expected-recovery-sha256 "$RECOVERY_SHA256" \
  --expected-projection-sha256 "$PROJECTION_SHA256" \
  --donor-year-policy source_2024_only \
  --output-dir /absolute/new-output-directory
```

The new output directory contains `asec-age15-donors.npy`, `acs-age15-recipients.npy`, and `PREPARATION.json`. Existing output directories are refused. Each numeric sidecar retains the original person projection columns and adds the explicit `parent_person_row`; recipients additionally carry the three recovered ACS hours fields. Order follows the parent. The JSON gives source/receipt/tool and sidecar digests, counts by donor year, selected source-hour ranges and recipient missingness counts. Read sidecars with **`np.load(..., allow_pickle=False)`**.

This implementation is covered by invented fixtures for colliding source identities, baseline/PUF/ACS separation, year policy, raw ages, annual/reference-week distinctions, NIU/earnings preservation, invalid domains, duplicate keys, malformed membership, incomplete ACS evidence, file binding and overwrite refusal. It requires full numeric input projections even when its age-15 output is small.

The next model decision is the donor-year policy and a small set of predictors measured comparably in both surveys. That decision must specify how source-unknown recipient values are represented. This preparation step makes no choice of estimator, model features, weights or draw method.
