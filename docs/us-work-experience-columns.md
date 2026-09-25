# Industry and worked-last-year on every US person record (#719)

The county-file person schema carries a working indicator, industry and
occupation on every person record. Occupation already shipped
(`detailed_occupation_recode`, CPS ASEC `POCCU2`). This change adds three
PolicyEngine-US person inputs (defined in policyengine-us#9311, released in
1.820.2, locked at 2.2.1):

| column | meaning | CPS ASEC source | ACS 2024 source |
|---|---|---|---|
| `detailed_industry_recode` | industry of the longest job last year, 23 detailed groups | `WEIND` | `INDP` via `ACS_INDP_TO_WEIND` |
| `major_industry_recode` | the same job, 15 major groups | `WEMIND` | `WEIND_TO_WEMIND` of the above |
| `worked_last_year` | worked at any time last year | `WKSWORK > 0` | `WKWN > 0` |

Codes (`WEIND`): 0 = not in universe (under 15; ACS: under 16), 1–21 =
civilian industry groups, 22 = Armed Forces, 23 = age 15+ (ACS: 16+) and did
not work last year. `WEMIND` nests them: 0 = not in universe, 1–13 civilian
major groups, 14 = Armed Forces, 15 = did not work.

## Why these CPS columns

`WEIND`/`WEMIND` describe the longest job last year, the same job `POCCU2`
describes, and the same reference year as the income the file carries. The
survey-week recodes `A_DTIND`/`A_MJIND` describe the current job instead.
`worked_last_year` uses `WKSWORK > 0` rather than `WORKYN = 1`: in all three
pinned members `WEIND` carries a worker code exactly where `WKSWORK > 0` (zero
exceptions), while `WORKYN = 1` misses 642 / 572 / 594 allocation rows per
year that have weeks worked and an industry.

## Where each spine gets them

- **ASEC rows (and their PUF-support clones).** No `census_cps` H5 vintage
  carries `WEIND`/`WEMIND`, not even 2024. `asec_census_person_columns`
  (#720's restoration) appends both to every vintage from its pinned Census
  person member by exact `PERIDNUM` join before pooling. They sit outside the
  2026-08-23 offline-fix receipt (`ASEC_CENSUS_PERSON_COLUMNS_BEYOND_OFFLINE_FIX`).
  `WKSWORK` is already in every H5.
- **ACS rows.** The release predictor join (`acs_release_predictors`,
  crosswalk version 2) materializes `WEIND`, `WEMIND` and `WKSWORK` for ACS
  people from the pinned ACS 2024 person archive, the same boundary that maps
  `OCCP` to `POCCU2`.
- **Every row.** The release-time `org_wages` carry
  (`derive_us_org_occupation_inputs`) turns them into the three columns next
  to `detailed_occupation_recode`.

## The ACS crosswalk

ACS 2024 `INDP` uses the 2022 Census industry code list, as does the 2025 ASEC
member (`pppub25.csv`), where `INDUSTRY` determines `WEIND` exactly (259
codes). The 2023/2024 members use the 2017 list; the 225 codes the lists share
map to the same group in both. `ACS_INDP_TO_WEIND` takes every civilian code's
group from `pppub25.csv` and reviews the eight ACS-only codes:

- 9670, 9680, 9690, 9770, 9780, 9790, 9870 (Army, Air Force, Navy, Marines,
  Coast Guard, branch not specified, Reserves or National Guard) → 22. CPS
  collapses them into 9890, which the members map to 22.
- 9920 ("unemployed, no work experience in the last 5 years or never
  worked") → 23, as `OCCP` 9920 maps to `POCCU2` 53.

Blank `INDP` follows the `OCCP` rule: 0 below age 16 (age 15 keeps the
out-of-universe sentinel rather than inventing CPS code 23), 23 from age 16,
where only ESR 6 may be blank. `WKSWORK` is `WKWN` (weeks worked in the past 12
months) or 0.

Reference periods differ, as they already do for occupation: an observed ACS
`INDP` describes the most recent job in the past five years, and `WKWN` counts
the 12 months before interview; the CPS columns describe the prior calendar
year. So on ACS rows an industry can sit on a person with
`worked_last_year = False` (a job held one to five years ago).

## Checked identities

| identity | ASEC rows | ACS rows | where checked |
|---|---|---|---|
| `WEMIND = WEIND_TO_WEMIND[WEIND]` | yes | yes | restoration (per vintage), ACS join (by construction), `org_wages` carry and gate |
| `WKSWORK > 0` ⇒ `WEIND` in 1–22 | yes | yes | restoration, ACS join, `org_wages` carry and gate |
| `WEIND` in 1–22 ⇒ `WKSWORK > 0` | yes | no (5-year `INDP`) | restoration only, where rows are known to be ASEC |

## Real-data evidence (2026-09-24)

ASEC members `pppub23/24/25.csv` (pins in `spm_role_source.ASEC_SPM_ROLE_SOURCES`):

- `INDUSTRY` → `WEIND` and → `WEMIND` are functions in every vintage (263 /
  263 / 259 codes); `WEIND` → `WEMIND` is the same function in all three.
- `WEIND` in 1–22 iff `INDUSTRY > 0` iff `WKSWORK > 0`: zero violations in
  each vintage. `WEIND = 0` exactly for ages 0–14; `WEIND = 23` only from 15.
- Weighted (`A_FNLWGT`) shares: nonzero `WEIND` 0.8217 / 0.8220 / 0.8244;
  worked last year 0.5200 / 0.5232 / 0.5228.

ACS 2024 one-year person archive (SHA-256 `afdc6d90…`, the release pin), all
3,422,888 persons run through `_crosswalk_people`:

- No refusal fires; all 265 observed `INDP` codes are mapped.
- `INDP` and `OCCP` are blank on exactly the same rows; a blank at 16+ occurs
  only with ESR 6.
- `WKWN > 0` iff `WKL = 1` (zero disagreements); every person with weeks
  worked has a worker `WEIND` code.
- 264,806 persons carry an industry without weeks worked, exactly the count
  with `WKL = 2` (last worked one to five years ago).
- Weighted (`PWGTP`) shares: nonzero `WEIND` 0.8123; worked last year 0.5385.

The `org_wages` gate bands (`detailed_industry_recode` and
`major_industry_recode` nonzero 0.70–0.92, `worked_last_year` 0.40–0.62)
bracket both spines.

## What this does not do

It changes code and manifests only. Released files carry the columns once the
pooled base and multispine pool are rebuilt with the restoration and a new
release runs (#922 sequencing step 1); until then the release input-coverage
gate is red on these three columns by design.
