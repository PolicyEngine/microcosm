# Largest single record of each QRF output (#1268): receipts

Pull request: PolicyEngine/microcosm#1268. It adds
`microcosm.build.gates.single_record_concentration` and the
`surface.single_record_concentration` block of `qrf_tail_concentration.json`.
This folder holds the scripts and logs behind the numbers that pull request
quotes.

## What the scripts measure

For every numeric QRF-imputed output of one certified US release:

- the largest single record's share of the column's weighted absolute mass
  (`|value| × weight`), the top-1 share;
- that record's weight;
- the top-100 share, the number the QRF tail gate grades on the columns it
  checks;
- and the number of carriers, the records with positive mass.

`route_a_top1.py` takes these once at the release's calibrated household
weights and once at the initial weights in its calibration sidecar.

## Release and commit

| Item | Value |
|---|---|
| Release | `populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z` (Route A) |
| `populace_us_2024.h5`, sha256 | `417d23aed4d044de2e657a7b5d856136975d25739dbdb4ddd72866f841de1559` |
| Recorded `qrf_tail_concentration.json`, sha256 | `8311943453d524a5b1a286cd440e40ea56a5909da0aa9971d065da9cc72cdc58` |
| Commit that ran the measurement | `a6644d62b2f7b6d4e2afbd91212e8f25dd82ac1d` |

On 2026-11-11 the Hugging Face dataset repository `policyengine/populace-us`
was public and ungated. Its revision
`07db2a23ca6598abb837c154a085655eeeefca73` holds this release:
`populace_us_2024.h5` (the sha256 above) and
`populace_us_2024_calibration.npz` (sha256
`d1403f75c549f030dcc13c2df3c2e4f7365b67105caa1bb0440e9ab8683f2d55`) at the top
level, and `release_manifest.json` and `qrf_tail_concentration.json` under
`releases/<release id>/`.

`route_a_top1.py` loads the release tool and the post-export probe from a
checkout. It reads the H5's QRF columns with the probe's loader and runs the
release tool's `_qrf_tail_concentration_gate` on them. At `a6644d62b` the
source manifest lists 106 QRF outputs, and the H5 carries 68 of them as
numeric columns.

Before it reports anything, the script checks its own inputs and stops on any
difference:

- the H5 and the recorded `qrf_tail_concentration.json` hash to the sha256s
  in the release manifest;
- at calibrated weights, the gate's verdict, its details after a JSON round
  trip and the four surface lists (checked, dense, absent and non-numeric
  columns) equal the recorded file's, and the probe's `compare_qrf_tail`
  finds no difference across the 33 columns with a top-100 share;
- the sidecar's `household_weight` equals the H5's household weights element
  for element, so `initial_household_weight` aligns by row.

`route_a_top1.log` records the outcome of each check.

## Files

| File | What it holds |
|---|---|
| `route_a_top1.py` | The Route A measurement. |
| `route_a_top1.md` | Its table: all 68 measured columns, sorted by top-1 share at final weights. |
| `route_a_top1.log` | Its printed summary: hashes, commit, the checks above and the columns above 5% and 10%. |
| `calibration_weight_stats.py`, `.log` | Household weights before and after calibration. |
| `shared_largest_records.py` | Which columns above 5% share a largest record or a household. |
| `minimize.py`, `.log` | Inputs whose gate share passes 1 under every legal arrangement of the partition. |
| `monotone_k.py`, `.log` | Whether a larger `top_k` can lower the top-k share. |

Two outputs stay out of the repository because they name records:
`route_a_top1.json`, which `route_a_top1.py` writes beside the table and which
gives each column's largest record by `record_id` and `household_id`, and the
printed output of `shared_largest_records.py`.

This folder's copies differ from the scripts that ran in form only. `ruff
format` reflowed them, three lines gained a `# noqa` comment, and
`shared_largest_records.py` lost an unused `numpy` import. With that import
set aside, each copy parses to the same syntax tree as the script that ran.

## How to rerun

`minimize.py` and `monotone_k.py` need only numpy:

```bash
python minimize.py > minimize.log
python monotone_k.py . > monotone_k.log   # the argument: the folder that holds minimize.py
```

A rerun on 2026-11-11 (numpy 2.4.6, macOS arm64) printed both committed logs
byte for byte. Two lines of `minimize.log` describe the machine that ran it:
the line that prints numpy's own partition, and the last line, which prints
the numpy version and the platform. Every other line comes from sums the
script arranges itself.

The other three scripts need the release's files. Download the four files
named above at that revision and lay them out as `route_a_top1.py` reads them:

```text
populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z/
  artifacts/populace_us_2024.h5
  artifacts/populace_us_2024_calibration.npz
  releases/populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z/release_manifest.json
  releases/populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z/qrf_tail_concentration.json
```

Then run:

```bash
python route_a_top1.py <checkout of a6644d62b> <the folder above> <output folder>
python calibration_weight_stats.py <populace_us_2024.h5> <populace_us_2024_calibration.npz>
python shared_largest_records.py <output folder>/route_a_top1.json <populace_us_2024.h5>
```

`route_a_top1.py` needs a Python environment with the checkout's packages
installed; the other two need h5py and numpy. On 2026-11-11, reruns on that
revision's files gave the same results: `calibration_weight_stats.py` printed
the committed log byte for byte, and `shared_largest_records.py` printed the
shared records and households that
[Largest records that columns share](#largest-records-that-columns-share)
reports.

Run `route_a_top1.py` against a checkout of `a6644d62b`. The script stops
unless the absent QRF outputs equal the recorded ones, and the source manifest
has since gained eight QRF outputs that this H5 does not carry
(`detailed_industry_recode`, `federal_parent_plus_loan_balance`,
`federal_student_loan_balance`, `federal_student_loan_graduate_share`,
`is_income_based_repayment_new_borrower`, `major_industry_recode`,
`passive_partnership_s_corp_income` and `worked_last_year`).

## Results

### Top-1 shares in the release

`route_a_top1.md` lists all 68 columns. Four carry no mass (`s_corp_income`
and the three second-home mortgage columns), so 64 have a share.

| Weights | Columns above 5% | Columns above 10% |
|---|---:|---:|
| Final (calibrated) | 17 | 7 |
| Initial | 13 | 8 |

The table's "largest record" cell prints the record's signed value, and the
mass beside it is `|value| × weight`: `farm_rent_income` reads
`$-288,692 × 3,896.9 = $1.12B`.

### Calibration weights

From `calibration_weight_stats.log`, over the release's 352,932 households:

| Statistic | Initial | Final |
|---|---:|---:|
| Total weight | 138,098,416.6 | 138,098,416.6 |
| Median | 378.04 | 117.78 |
| 99th percentile | 1,021.20 | 3,730.77 |
| Minimum | 0.1028 | 0.0207 |
| Maximum | 2,371.62 | 9,740.07 |

The final-to-initial ratio runs from 0.014248 to 5.0000 across households,
and no household starts at weight 0.

### Largest records that columns share

`shared_largest_records.py` maps each largest record to its household for the
17 columns above 5% at final weights:

- One person's record is the largest record of both W-2 wage columns,
  `sstb_w2_wages_from_qualified_business` and
  `w2_wages_from_qualified_business`.
- One household holds three columns' largest records. One of its members
  holds the largest record of `sstb_self_employment_income_before_lsr` and of
  `sstb_unadjusted_basis_qualified_property`, and a tax unit in that household
  holds the largest record of `unrecaptured_section_1250_gain`.
- The other 12 columns have their largest records in 12 different households.

So 14 households hold the largest records of these 17 columns. The script
prints each record's household and not its tax unit. A separate lookup in the
H5's person table found that the person in the second finding belongs to the
tax unit named there.

### The gate's share can pass 1

`tail_concentration_gate` sums its tail in the order `np.partition` leaves it
and its total in record order. `minimize.py` sums the tail in every legal
arrangement of a partition:

- The nine-record input
  `[2, 3, 2, 1, 1e16, 7, 1e15, 1, 0.5]` at `top_k = 8` gives a share of
  0.9999999999999998, 1.0 or 1.0000000000000002, depending on the tail's
  order.
- With `top_k` of 1 or 2, no input of three to five records from the script's
  ten-value grid gives a share above 1.0 in any legal arrangement.
- With four records and `top_k = 3`, 81 grid inputs give a share above 1 in
  every legal arrangement. One of them is `[2**53 + 2, 3, 1, 0.5]`: the
  record-order total is `2**53 + 4`, and the two legal tails give
  1.0000000000000002 and 1.0000000000000004, while the exact share rounds to
  1.0.

`single_record_concentration` therefore caps `top_k_share` at 1.

### A larger top-k can lower the share

`monotone_k.py` searches every four- and five-record input from an eight-value
grid, at every `top_k` below the record count. It compares shares capped at 1,
as `single_record_concentration` reports them:

- For 2,651 pairs of an input and a `top_k`, some legal arrangement at `top_k`
  gives a larger share than some legal arrangement at `top_k + 1`. For example
  `[0.5, 0.5, 0.5, 0.5, 2**53 + 2]` can give 1.0 at `top_k = 3` and
  0.9999999999999998 at `top_k = 4`.
- For no pair does every arrangement at `top_k` give more than every
  arrangement at `top_k + 1`.

So a larger `top_k` can lower `top_k_share` under some arrangement of the
partition, and on this grid never under all of them. The function's test
module pins one such input, `[0.5, 1, 2**53, 0.5, 0.5]`, with the arrangement
fixed.

### Mutation run

At commit `3ccae300a`, a run applied fourteen mutants to
`single_record_concentration`, one at a time, and the function's test module
(`test_single_record_concentration.py`) failed on every one. The mutants:

1. the smallest weight among tied records, in place of the largest;
2. no cap on `top_k_share`;
3. a cap one rounding step too loose;
4. `carriers < top_k` for `carriers <= top_k`;
5. no shortcut when every carrier sits in the top `top_k`;
6. a tail one record short;
7. the tail summed in sorted order;
8. no check that the total is finite;
9. zero-mass records counted as carriers;
10. weights left out of the finite filter;
11. signed mass in place of absolute mass;
12. the median mass in place of the largest;
13. the top-1 mass over the tail in place of the total;
14. overflow left to warn.

That run's script patched a checkout in place, so this folder does not hold
it. The pull request's description reports a later mutation pass, and this
folder holds no log of either run.
