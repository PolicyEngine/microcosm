# US ACS donor receipt qualification

A local, non-publishing procedure that adds three reported-receipt input
columns — `person.receives_wic`, `spm_unit.receives_snap` and
`spm_unit.receives_tanf` — to an exact, pinned Build P donor so that the donor
satisfies current main's hard ACS transfer-consumption gate. Every other cell,
weight, index and entity table is preserved byte-for-byte.

The tool is [`tools/build_us_acs_donor_receipt_qualification.py`](../tools/build_us_acs_donor_receipt_qualification.py).
It certifies nothing: it produces a local H5 and a receipt, never a release, a
staged bundle, a calibration or a latest pointer. See
[the PR-CI / certification boundary](../CLAUDE.md) for why that distinction is
load-bearing.

## Why the donor needs qualifying

`declared_acs_transfer_target_families()` declares `receives_wic` on `person`
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:426`)
and `receives_snap` / `receives_tanf` on `spm_unit`
(`acs_transfer.py:452`) as `model_required_boolean` transfer targets.
`acs_transfer_donor_requirements` (`acs_transfer.py:844-870`) folds every
declared target into the donor requirement set, and
`_require_dense_donor_coverage`
(`tools/_legacy/build_us_acs_multispine_base.py:349-420`) refuses a donor that
cannot supply them.

The certified July Build P donor predates those declarations. It carries the
raw ASEC answers the three columns are derived from — `WICYN`, `SPM_SNAPSUB`,
`PAW_VAL` — but not the derived columns themselves, because the pre-clone stage
that produces them (`_pre_clone_enrichment_stage` in
`tools/build_us_puf_support_base.py:1999-2013`) post-dates the donor.

## What the three producers read and emit

All three live in
`packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py` and are
called exactly as production calls them; this tool reimplements none of their
rules.

| Producer | Reads | Emits |
| --- | --- | --- |
| `reported_wic_receipt_carrier(person)` (`cps_carried.py:352-365`) | `WICYN`, through `_integer_source` (missing or non-numeric coerces to 0) | person-grain `numpy` bool array, `WICYN == 1` |
| `reported_snap_receipt_by_spm_unit(person)` (`cps_carried.py:328-349`) | `SPM_SNAPSUB` and `person_spm_unit_id`; raises if either is absent | Series indexed by `person_spm_unit_id`, true where the member maximum is positive |
| `reported_tanf_enrollment_by_spm_unit(person, source)` (`cps_carried.py:266-325`) | `person_spm_unit_id`, `PAW_VAL`, and `PAW_TYP` — from the frame when present, otherwise joined transiently from the pinned sidecar | Series indexed by `person_spm_unit_id`, true where any member has `PAW_VAL > 0` **and** `PAW_TYP ∈ {1, 3}` |
| `_fill_spm_unit_reported_enrollment_inputs(person, spm_unit, *, public_assistance_type_source)` (`cps_carried.py:486-510`) | only `spm_unit.columns` and `spm_unit["spm_unit_id"]` | writes `receives_tanf` and `receives_snap` in place, `reindex(spm_unit_id).fillna(False)` |

Because `_fill_spm_unit_reported_enrollment_inputs` reads nothing from the SPM
table but its column list and its `spm_unit_id` (`cps_carried.py:493-510`), the
tool hands it a single-column projection of the donor's SPM table. The function
cannot distinguish that projection from the full table — the tool has already
refused if either output column exists on any entity — and the projection keeps
the donor's own SPM table unmutated. The function writes into what it is given.

Three semantics the producers' own docstrings establish, and that this
qualification therefore inherits:

- `PAW_VAL`, `SPM_SNAPSUB` and `WICYN` are **annual** reported facts broadcast
  to all twelve modeled months. The annual source cannot reveal entry or exit
  timing.
- `PAW_VAL > 0` misses TANF units that received zero dollars, including
  sanctioned cases; and `PAW_TYP` 2 (other cash welfare) is deliberately **not**
  TANF. In the 2023 ASEC, 271 of 682 PAW-positive SPM units report no TANF type,
  so gating on `PAW_VAL` alone would mark general assistance as TANF
  ([microcosm#591](https://github.com/PolicyEngine/microcosm/issues/591)).
- `receives_wic` is stored on the adult-female `WICYN` reporter as the
  **carrier** of her SPM unit's receipt fact. It does not identify the
  beneficiary. Every new or regrouped consumer of it needs explicit
  re-adjudication under microcosm#591.

## How `PAW_TYP` is restored for a pooled three-income-year donor

The donor pools three ASEC income years. `PAW_TYP` is absent from the frozen
`census_cps_*.h5` inputs it descends from, so
`microcosm.build.us_runtime.public_assistance_type_source` restores it from the
official Census public-use archives, sharing
`education_assistance_source.ASEC_EDUCATION_ASSISTANCE_ARCHIVES`' per-income-year
pins.

### Which archive serves which `source_year`

`source_year` is the **income** year; the archive published the following March
carries that income year's person universe.

| `source_year` | Survey year | Pinned member | Pinned rows |
| --- | --- | --- | --- |
| 2022 | 2023 | `pppub23.csv` | 146,133 |
| 2023 | 2024 | `pppub24.csv` | 144,265 |
| 2024 | 2025 | `pppub25.csv` | 142,125 |

`load_asec_public_assistance_type_sources`
(`public_assistance_type_source.py:232-353`) verifies each path by exact byte
length and SHA-256 — or, for a zip, zip size and SHA-256 plus the member's
`file_size` and CRC32 — requires the six source columns plus `A_FNLWGT`,
requires the pinned row count, normalises `PERIDNUM` to exactly 22 digits,
requires `PERIDNUM` uniqueness, requires every `PAW_TYP ∈ {0,1,2,3}` and every
`PAW_VAL` finite and non-negative, and then re-measures each year's full
`PAW_TYP` composition against `ASEC_PUBLIC_ASSISTANCE_TYPE_AUDIT_PINS`
(income year 2022: counts `(145377, 428, 306, 22)`, 756 PAW-positive rows, 450
TANF-typed). Any drift refuses the load.

### How rows join

`fill_asec_public_assistance_type_source`
(`public_assistance_type_source.py:356-471`) joins per `source_year` on exact
22-digit `PERIDNUM`, then re-checks redundant Census identity for the identity
columns the frame actually carries (`:430-440`):

| Sidecar column | Frame column compared |
| --- | --- |
| `PH_SEQ` | `source_household_id` when present, otherwise `PH_SEQ` |
| `P_SEQ` | `P_SEQ` |
| `A_LINENO` | `A_LINENO` |

The preference for `source_household_id` is not a nicety on a pooled donor — it
is the only correct comparison. `asec_pool._remap_source_year`
(`asec_pool.py:283, 314-315`) copies the raw ASEC `PH_SEQ` into
`source_household_id` and then **overwrites the frame's `PH_SEQ` column** with a
dense, cross-year-offset pool household id, which it also assigns to
`household_id`. On the Build P donor this is directly observable in aggregate:
the frame's `PH_SEQ` and `source_household_id` disagree on all 166,321 rows,
`PH_SEQ` has no cross-year collisions (47,277 distinct values against
15,361 + 15,724 + 16,192 per year), and `source_household_id` does collide
across years (38,652 distinct values) exactly as a year-local Census sequence
must. Against the pinned 2022 sidecar, the sidecar's `PH_SEQ` matches the
frame's `source_household_id` on 54,464 of 54,464 rows and the frame's own
`PH_SEQ` on none. `source_household_id` on this donor **is** the ASEC `PH_SEQ`.

`PERIDNUM` is not unique on this pooled, cloned donor — 78,547 of its 166,321
person rows share a `PERIDNUM` with another row — and it does not need to be.
The join requires uniqueness only of the *sidecar*, and reindexes the year's
unique sidecar rows onto the frame's keys, so repeated keys receive the same
restored value.

### A row with no match

The join refuses outright (`public_assistance_type_source.py:421-427`). Any
frame `PERIDNUM` absent from that income year's sidecar fails the whole restore,
whether or not its `PAW_VAL` is positive. There is no imputation and no
fallback: a frame whose vintage does not match its pinned archive fails loudly,
by design, and `ASEC_EDUCATION_ASSISTANCE_ARCHIVES`' docstring records that
pinning the wrong vintage drops coverage to about 33% (the CPS rotation-group
overlap) and so fails this check.

Because that refusal message, and several others in this area, embed raw
`PERIDNUM` values and row indices, the tool never propagates a producer's
message. It re-raises the producer's name and exception class only.

## What the staging gate requires

`_require_dense_donor_coverage`
(`tools/_legacy/build_us_acs_multispine_base.py:349-420`), in order:

1. **Resolve the donor channel first.** `resolve_acs_donor_channel`
   (`acs_transfer.py:3247-3290`) maps the default `ACS_DONOR_CHANNEL_AUTO` to
   the `puf_tax_detail` support role and returns `Frame.select`ed sub-frame.
   Every check below runs on that sub-frame, not on the whole donor.
2. **The required entity must exist** on the selected frame.
3. **The column must be on that entity.** A column found on a different entity
   is reported with its actual owner and still fails.
4. **`pd.notna(values).any()`** — at least one observed value.
5. **`default_valued_columns_gate`** (`packages/microcosm-build/src/microcosm/build/gates.py:1185`)
   — not every observed value may equal the engine default. For these three
   booleans the engine default is `False`, so each needs **at least one true
   value inside the selected channel**.

That first step is the one a pooled-donor receipt can most easily mis-report: a
column that is well populated across the whole donor but constant `False` inside
the `puf_tax_detail` selection passes an aggregate eyeball and still fails the
gate. The receipt therefore reports the selected-channel true count separately,
and the tool fails closed on it.

## Why this is a qualification, not a new imputation

Each of the three columns is a deterministic, total function of values the donor
already stores — `WICYN`, `SPM_SNAPSUB`, `PAW_VAL`, `person_spm_unit_id`,
`spm_unit_id` — plus one immutable, pinned, hash-verified Census source field,
`PAW_TYP`, joined by exact `PERIDNUM` identity with redundant household, person
and line-number checks.

No model is fit. No draw is taken. No seed is consumed. No weight, index, entity
table or other cell changes. Nothing is predicted for an unmatched row: the join
refuses instead. The operation recovers survey answers the donor's own
respondents gave and that the donor already carries in raw form, and it is exactly
the derivation the maintained pre-clone stage performs today for every newly
built base.

What it is **not**: it does not certify the donor, does not calibrate, does not
publish, does not stage, and does not make the donor a release. Its receipt is
build evidence only.

## Refusals

The tool refuses, before writing anything, when:

1. the parent's SHA-256 is not one of the two exact pinned parents: the
   reviewed Build P parent (the same digest as
   `microcosm.data.source_enrichment.PARENT_DATASET_SHA256`), or the published
   national default `populace-us-2024-spm-20260915`, which is that population
   with the native `is_spm_independent_minor_role` column. Measured on
   21 September 2026, counts only: under the locked engine's adult rule the
   Build P file has 28 of 59,900 SPM units with no classified adult and the
   role-carrying file has none, so the second parent is the better ACS donor.
   The receipt records which lineage was qualified;
2. the output directory already exists or is a symlink;
3. any of `receives_wic`, `receives_snap`, `receives_tanf` already exists on any
   entity table;
4. `PAW_TYP` already exists on `person` — it would silently bypass the pinned
   archive path the receipt claims;
5. any required raw column is absent. `SPM_SNAPSUB`, `PAW_VAL` and `WICYN` are
   checked by the tool **before** the producers run, because `_source`
   (`cps_carried.py:378-385`) coerces an absent column to zeros and
   `_fill_spm_unit_reported_enrollment_inputs:504-506` substitutes
   `SPM_SNAPSUB = 0.0`, either of which would emit a silently all-`False`
   column;
6. any producer refuses — archive hash, CRC, row-count or audit drift, a
   non-unique or non-22-digit `PERIDNUM`, an unmatched `PERIDNUM`, an identity
   mismatch, or the missing-`PAW_TYP` refusal — re-raised with the producer's
   name and exception class only;
7. any added column has no true value, either across the whole donor or inside
   the channel the staging gate would select;
8. any preservation check fails, at the HDF object level or after reloading the
   child through the maintained loader;
9. the producer source files change while the tool runs.

## Receipt

The receipt (`donor_receipt_qualification.json`, written beside the child H5)
records aggregate facts only — never a row value, identifier or array:

- parent and child SHA-256, and the child's filename;
- producer module paths, their SHA-256, the git commit and dirty flag, and
  runtime library versions;
- the archive pins used per income year and the loader's measured
  `source_audit`;
- per column: true and false counts by `source_year`, by support role when the
  donor carries support-role metadata, and the true count inside the
  gate-selected channel;
- the list of columns added with entity and dtype, and the HDF preservation
  report.

## Running it

```bash
env -u UV_FROZEN uv run --no-sync python tools/build_us_acs_donor_receipt_qualification.py \
  --parent-h5 /path/to/populace_us_2024_buildp.h5 \
  --output-dir /path/to/donor-qualified
```

`--source-cache` defaults to `~/.cache/microcosm/cps/asec_education`, the same
directory `fetch_asec_education_assistance_source` caches into. The tool never
downloads: it requires the pinned members to be present locally and verifies
them by digest.
