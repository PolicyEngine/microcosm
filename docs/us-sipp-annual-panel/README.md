# SIPP nominal-2022 annual panel pilot

This is an invented-tested source helper for root review. It is not an accepted
income harmonization, model input, representative estimate, or release artifact.
The saved v1/v2 reports remain unchanged. Their quoted evidence was reverified;
the author's failed external job did not confer final acceptance.

The checkout starts at `c1b8324183819cf91511c002bf724a2676524e87`, fetched from
the original repository's previously fetched main. Live upstream fetch failed
with DNS unavailable. A current-main recheck is still needed outside this sandbox.
Only `sipp_december_source.py` and `sipp_measurements.py` were copied, byte for
byte, from accepted source head `61f2f4998c08778c247a6ce050d482e71e153293`.
The latter supplies the all-member December mean weight; no housing/rent gate is
used. The frozen source master hash is declaration-only, never a payload read.

The executable contract is `sipp_annual_panel_contract.json`, beside the helper.
Its 67 fields carry authenticated dictionary pages, grains, associated flags and
public document hashes. `public-verification.json` records 91/91 v1 and 85/85 v2
quotation checks, public PDF/text hashes, and preservation hashes of the old
reports/verifier files. `authenticate_sipp_annual_panel_public.py` can regenerate
this metadata using the existing public authority package; it does not run either
report generator. Quote checks authenticate text placement, not all deductions.

Selection is fixed before execution: the first32 canonical, exactly 14-digit,
lexically sorted SSUIDs with at least one December row having THHLDSTATUS1..4,
RIN_UNIV1 and a usable native ERESIDENCEID. Weight, age and monetary values never
select units. Pass1 projects identifiers/status only and streams a SHA256; pass2
streams the same pinned file and keeps **all** rows for those SSUIDs. Both hashes
and sizes must match the declaration. Physical monetary bytes necessarily pass
through the raw hash/tokenizer; they are not projected or interpreted in pass1.
No December-membership, December-only or age15 filter is applied to pass2.
Duplicate December keys refuse in pass1; selected person-month duplicates refuse
in pass2. Person-grain fields, including THHLDSTATUS and annual lump/rollover
amounts and flags, must repeat lexically exactly (blanks included).

The guide's printed37 tables3-6/3-7 establish that RREL includes self (code99),
including a one-person example. Every row must name its own person exactly once
with that code; a missing self is never inserted. Collected sets are checked
against each live roster vector and its union, plus RHNUMPER/RHNUMPERWT2.
An occupied slot16 is **unknown suppressed tail**, even if reported counts agree.
Unsaturated agreement permits only a published-file diagnostic; it never proves
physical household or suppressed-slot completeness. Monthly sums include children
receiving SSI and use their current residence, not a December-fixed roster.
Only THTOTINC is compared, with chosen dollar tolerance `income terms + 1`.
The tolerance is not a Census guarantee; discrepancies remain diagnostics.

Annual amounts sum twelve admissible months of December-age15+ contributors,
regardless of earlier residence. Missing rows, pre15 non-SSI months, frame gaps,
unknown ages/statuses/amounts and uncertain December membership disclose incomplete
coverage. A full-selected sum is null if any household is incomplete or a live
December member has no usable residence. Complete-subset sums have explicit
counts. A newborn imposes no twelve-month requirement on adult contributors;
unknown newborn SSI receipt may still prevent that month's identity check.
Under15 rows remain in the panel, not in the adult annual sum. The weight mean
uses all live December members, including children and zero weights. Unknown
weights suppress only the corresponding weighted term and are counted.

All original status codes0..9 are counted by associated field and native grain.
The optional partition 0/1/2-6/7/8/9 has **six** groups. Imputed finite values are
usable; code9 leaves component quality unknown. Blank values, missing flags and
flag0 are not monetary zeros. Numeric values must satisfy documented domains;
unknown lexical tokens refuse. The genuine public NIU lexical convention remains
unbound; encountering an unsupported convention requires review, not coercion.
Signed TPTOTINC/THTOTINC amounts are retained. TLMPAMT/TROLLAMT are counted once
per person in their conditional universes; rollover above gross is diagnosed,
not clipped or subtracted. There is no annual Type2 income read or sum: output
counts respondent-month reference occurrences, not unique Type2 persons.

ACS child-SSI scope remains unknown: token absence does not establish matching.
T2 gross-lump-sum comparisons are deferred; any future discrepancy would depend on
nonnegative suppressed components and recode/rounding assumptions, and cannot
unconditionally falsify gross treatment. All six suppressed income components
are named in output. Cross-survey harmonization, child-SSI annual sums,
component reconstructions, optional sensitivities, T1/T2, price conversion,
pooled comparisons, fitting and policy evaluation are outside this pilot.
Reported HTOTVAL and HINCP recipient totals are unchanged.

## Prepared command for root review — not executed on genuine data

Copy `root-binding.UNBOUND.json` to a new review-controlled binding and supply
root-approved source-only source_kind, absolute source path, exact SHA256 and byte
size. Keep survey2023/income2022, contract hash and first32 selection fixed.
The supplied template refuses before any source stat/open and creates no output.
No genuine path or payload pin is supplied by this preparation.

```sh
/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-launch-current-main-20260905/.venv/bin/python \
  -B -X pycache_prefix=/private/tmp/sipp-annual-root-review-20260906-absent \
  -I tools/run_sipp_annual_panel.py \
  --binding docs/us-sipp-annual-panel/root-binding.UNBOUND.json \
  --output-dir /Users/maxghenis/architecture-reviews/microcosm-20260904/us-housing-annual-panel-root-first32
```

Run from this checkout. Fixed ceilings: 900 seconds wall, 840 seconds worker CPU,
1GiB worker RSS, 16GiB input, 10million source rows, 1million December IDs,
100,000 selected panel rows, and 16MiB per output file. RSS uses the worker's
OS peak-RSS counter in a 50ms watchdog begun before imports and checked before
success. This is a sampled termination threshold, not a kernel allocation cap;
transient overshoot is possible. Wall timeout kills the worker process group.
Every Python child receives explicit `-B -X pycache_prefix=<fresh absolute path>
-I`; environment-only flags and existing prefixes refuse. Larger pilots have no
CLI option and require code/contract review.

Output directory creation is exclusive. Success writes exactly
`aggregate-diagnostics.json`, `worker.log`, and `execution-receipt.json`.
Only aggregate diagnostics and code/resource provenance leave the worker.
Partial/failed directories remain as evidence; a retry must use a new directory.
The execution receipt must say complete; mere presence of aggregate JSON is
insufficient, since a final resource refusal may occur after writing it.
No output file or existing directory is reused. Run the fixture generator with a
new `--output-dir` to make 40 invented SSUIDs/480 rows plus an invented binding;
then pass that binding to the same runner. The pilot selects32/384 rows.

Tests: `tools/verify_sipp_annual_panel.py --output-dir <new-directory>` under the
same Python flags. `--require-git-bytes` requires every loaded local code digest
to match the committed Git blob. It refuses Microcosm imports from other checkouts.
The focused tests need no engines, data files, external plugins or inherited
application fixtures. CI also inventories the flat `test_us_sipp_annual_panel.py`.
