# MEPS-IC employer-premium cells

`meps_ic_esi_premium_cells.json` holds the MEPS-IC averages the
`meps_esi_premiums` stage assigns to ESI policyholders
(`microcosm.build.us_runtime.esi_premiums`,
[PolicyEngine/microcosm#454](https://github.com/PolicyEngine/microcosm/issues/454);
method in `docs/us-esi-employer-premiums.md`).

AHRQ publishes the State tables only as PDF. Each value is copied as
published; `null` marks a cell AHRQ suppressed (`--`). The fallback for a
suppressed cell is stage logic, not data.

| Key | Source | Contents |
|---|---|---|
| `private_state_2025` | MEPS-IC 2025 Series II, Tables II.C.1/2, II.D.1/2, II.E.1/2 | Average total premium and average employee contribution per enrolled employee at private-sector establishments, single / family / employee-plus-one, by State, for all firms, under 50 and 50 or more employees |
| `private_national_2024` | MEPS-IC 2024 Series II, United States rows | The same six national averages a year earlier; they age the government cells |
| `public_division_2024` | MEPS-IC 2024 Series III, Tables III.C.1/2, III.D.1/2, III.E.1/2 | The same averages for State and local government jobs, by census division, for all governments and State governments. 2024 is the latest year AHRQ has published for this series |
| `no_contribution_share_2025` | MEPS-IC 2025 Series II, Tables II.C.4.a, II.D.4.a, II.E.4.a, United States row | Percent of enrollees whose coverage required no employee contribution, by tier, for all firms, under 50 and 50 or more employees. National only: most State cells of these tables are flagged unreliable or suppressed |
| `private_enrollment_national` | MEPS-IC 2024 and 2025 Series II, Tables II.B.1, II.B.2, II.B.2.b, II.C.4, II.D.4, II.E.4, United States row | Private-sector employees, the percent in establishments that offer health insurance, the percent enrolled there, and each tier's share of enrollees: the inputs of the active-employee cross-check |
| `state_census_division` | the division headings of Table II.C.1 | State FIPS to census division |

`sources` pins each PDF by URL, byte length and SHA-256.

## How it is built

```
uv run python tools/build_us_meps_ic_esi_cells.py --pdf-dir /tmp/meps-ic
```

The tool downloads the three pinned PDFs (about 25 MB) into `--pdf-dir`,
refuses any whose length or digest differs from its pin, extracts text with
`pdftotext -layout` (poppler) and reads one row per State, division or
firm-size label. It refuses a label that matches no line or more than one, a
row with the wrong number of values, a table whose title names another survey
year, and a United States row that differs from the AHRQ national figure it
must reproduce (for example single $9,025 and family $26,281 in 2025; the
2024 rows of MEPS-IC Research Findings #54). A national row whose kept column
is suppressed or flagged unreliable is refused too. `--check` regenerates in
memory and fails if any published value, title or pin differs; it ignores the
`pdftotext` version and the `text_line` positions, which depend on the
installed poppler.

The stage pins this file's own SHA-256 (`esi_premiums._CELLS_SHA256`) and
refuses to load any other bytes, so a regeneration must be re-pinned there
and in `us/source_stages.json`.
