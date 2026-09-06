# The first PUF monetary source projection

`microcosm.build.us_runtime.puf_monetary_source` is the next bounded increment
after the raw status slice
([`docs/us-puf-raw-source-first-slice.md`](us-puf-raw-source-first-slice.md)).
It does two things the status slice deliberately did not: it authenticates the
**whole** delivered layout against the retained IRS booklets, and it decodes a
small, explicitly enumerated set of **monetary** columns at source-return
grain.

It still does not grow, restate, combine or age anything. Every projected
column carries `target_year_use: refused_pending_independent_growth_decision`.

## What it produces

`microcosm.us.puf_2015_monetary_source_projection` v1: one opaque,
self-describing payload at source-return grain, 207,696 rows in delivered
main-file order.

| Group | Columns | Typed as |
| --- | --- | --- |
| identity | `RECID` | `int64` |
| status | `disclosure_aggregate` | `uint8`, derived from `RECID` |
| status | `demographic_status` | `int8`, **carried from the status artifact** |
| amounts | the twelve enumerated fields | `int64` whole dollars |
| amounts | the same twelve, lexically | 13-byte NUL-padded ASCII |

262 bytes a row, so 207,696 rows is 54,416,352 bytes — under the 64 MiB body
bound, with room for one more column of the same shape and not many more.

## The twelve columns, and why those twelve

The brief was wages, interest, dividends, business and farm income, pensions
and capital gains, *as the documentation supports*. In delivered order:

| Field | Publisher position and label | Concept group |
| --- | --- | --- |
| `E00200` | 1, Salaries and wages | wages |
| `E00300` | 2, Interest received | interest |
| `E00400` | 3, Tax-exempt interest income | interest |
| `E00600` | 4, Dividends included in AGI | dividends |
| `E00650` | 5, Qualified Dividends | dividends |
| `E00900` | 8, Business or profession (Schedule C) net profit/loss (+/-) | business income |
| `E01000` | 9, Net capital gain or loss (+/-) | capital gains |
| `E01500` | 13, Total pensions and annuities received | pensions |
| `E01700` | 14, Pensions and annuities included in AGI | pensions |
| `E02100` | 16, Schedule F net profit/loss (+/-) | farm income |
| `P22250` | 101, Short-term gains less losses net of carryover | capital gains |
| `P23250` | 104, Long-term gains less losses net of carryover | capital gains |

Every label above is the publisher's own text, retained verbatim in the
packaged document together with the exact character span and span digest of the
record-layout row it came from.

## Nothing is inferred from a prefix

The publisher's "Amount Fields and Large Codes" list ends with a **Misc Codes**
block holding `RECID`, `S006`, `S008`, `S009`, `WSAMP` and `TXRT`. None is
money; `TXRT` carries an implied decimal point (28.0 percent is spelled `280`)
and `S006` carries two. The same list holds `T27800` and `S27860`. So neither
an `E`/`P` prefix nor membership of the amount list decides anything: each
column's admission is a declaration in the packaged document, and the six Misc
Codes are marked `misc_code_not_an_amount` explicitly rather than filtered by a
name rule.

## Per-column declarations

**Units.** Whole US dollars, no implied decimals, a twelve-digit publisher
field width. The authority is the disclosure section's rounding grid: amounts
above $100,000 to four significant digits, $10,000–$100,000 to the nearest
$100, $5–$10,000 to the nearest $10, and *nonzero amounts under $5 set to $2
with the sign retained*.

**Missingness.** A delivered zero is a numeric value in the edited public
file. The rounding rule does not establish that the original return reported
zero or rule out other disclosure edits. Neither booklet declares a
missing-value sentinel or a blank convention for an amount field, so an empty
token is a **refusal** — preserved and reviewed — never an invented zero.
Every projected column remains `not_admitted_pending_basis_review` for fitting;
a source label alone does not settle its period or donor/recipient basis.
In particular, the nearest-$10 rule does not specify the $5 tie. Absence,
nonapplicability, reported zero and a publisher edit cannot be distinguished
from a delivered zero token alone.

**Sign.** The parser accepts a signed token for every projected column and
records what it observed. Three of the twelve print the `(+/-)` marker on their
layout row (`E00900`, `E01000`, `E02100`). `P22250` and `P23250` do not, but
the publisher's own detailed definitions on PDF page 24 describe netted
quantities ("all short-term capital gains less losses ... except for the
short-term capital loss carryover"), so their negatives are documented by
definition text instead. The remaining seven are recorded as
`not_documented_unresolved`: the marker's absence is preserved as a missing
marker, never enforced as a nonnegative domain and never resolved by taking an
absolute value. A negative wage parses, is retained, and is counted — so a
later review can settle the domain from evidence rather than from a guess.
`-0` is refused outright.

**Return grain.** One delivered main record. Not a person, not a household, and
— by the publisher's own request — not necessarily one taxpayer's return. The
file supports national estimation only.

**Source-period status.** `unresolved_raw_FLPDYR`, everywhere. The cover says
Tax Year 2015; the code definitions print FLPDYR 2011–2014; the disclosure
section removed returns filed for years earlier than Tax Year 2012; the
reviewed header audit observed 2012–2015. Until that is adjudicated, no growth
recipe may touch these columns.

**Source pin identity.** The document restates no pin as its own authority. It
binds the reviewed raw source definition by digest and each delivered file by
SHA-256, git blob SHA-1, byte count and canonical header digest, and the
artifact header repeats all of them plus the status artifact's digest.

**Edit status.** Four universal disclosure edits are cited for every column
(rounding, rebalancing, aggregation, "may or may not contain data from just one
tax return"). Wages carry two more: multivariate blurring wherever at least two
of wages, state and local income taxes and real estate taxes are nonzero, and
up to 10 percent uniform random noise on nonitemized returns sampled at 0.07
percent. A wage value is an edited amount.

## What stays unresolved, and is kept anyway

The booklets do not state that qualified dividends sit inside dividends in AGI,
that pensions in AGI sit inside pensions received, or that `E01000` composes
out of `P22250` and `P23250`. Those questions are recorded per column as named
`unresolved` entries, and `derived_combinations` is `refused` for every column:
no residual, no difference, no reconciliation is computed anywhere in this
slice.

A column whose *primary semantics* the booklets do not settle would be decoded
and retained under `fit_admission: unresolved_retained` and refused as a fit
input — retained and marked, never invented, and never a reason to halt the
rest of the projection. None of the twelve is in that state today; each has a
verbatim publisher label at a numbered layout position. The path is exercised
on an invented fixture declaration so the mechanism is real rather than
aspirational.

## Three refusals the increment is defined by

**`E22250` is not aliased onto `P22250`.** It appears in neither booklet and in
no delivered header. A projection declaration naming it is refused; a delivered
header carrying it is refused. Normalising one onto the other would be a guess.

**No uncapped self-employment earnings are derived.** `E30400` is the combined
primary *and secondary* net earnings from self-employment **subject to Social
Security tax** — the lesser of net Schedule SE earnings and the maximum amount
subject to the tax — and `E30500` is the secondary taxpayer's own capped
amount. Their difference is a capped primary residual, and no gross-up recovers
what the cap removed. The refusal is recorded with both definition texts. Note
that the shorter layout row for `E30400` says "primary taxpayer": the label and
the detailed definition disagree and the definition governs.

**Tax outcomes stay source-only.** Nothing outside the twelve is projected, and
the specific traps are named rather than left to omission: `E06500` (total
income tax, which the archived SOI measure table also binds to *income tax
before credits*), `E05800`, `E08800`, `E10300`, `E25920` (partnership total
passive loss specifically, not a generic gross loss), `E01100`, `E22320`,
`E22370`, `E02000`, `E26270`, `E27200`, `E01400`, `E02400` and `E00100`.

## Every delivered record is projected

All 207,696, including the 88,021 with no demographic row and the four
publisher aggregates. Nothing is filtered. Every derived fact — row count,
zero, negative and positive census, minimum, maximum and sum — is reported for
three **disjoint** row classes that must partition the rows exactly once:

- `disclosure_aggregate` — the four `RECID`s 999996–999999, which stand for
  1,124 excluded returns each side of an AGI cut. An aggregate is an aggregate
  first, even when a demographic row is keyed to it, so it can never be
  averaged into a per-return statistic.
- `demographic_matched` — non-aggregate rows the demographic file matched.
- `demographic_unmatched` — non-aggregate rows it did not. Absence of a
  demographic row says nothing about an amount and zeroes nothing.

## Bound to the reviewed status artifact

The producer node takes `microcosm.us.puf_2015_raw_return_status` as an
artifact input. It refuses unless its own `RECID` sequence and aggregate flags
are element-wise identical to that artifact's, and it **carries**
`demographic_status` from it rather than re-joining the demographic file. The
two artifacts therefore cannot disagree about which returns matched, and the
join semantics stay owned by the one node that was reviewed for them.

## Raw producer identity

This module imports `puf_raw_source` for its definition, codec and artifact
decoder. The root integration correction moves the raw definition from the
country package to `us_runtime`, with identical definition bytes and unchanged
decoding rules. The loader's package name therefore changes, creating a new
`USPufRawSourceKernel.implementation_hash` and node namespace. Previously
accepted artifacts remain associated with their original implementation.
`test_the_raw_source_module_matches_the_reviewed_packaging_identity` pins the
corrected loader bytes. The monetary reader still has its own CSV helpers;
sharing those helpers would be a separate implementation change.

## Where the document lives, and why

`packages/microcosm-build/src/microcosm/build/us_runtime/puf_2015_monetary_source_projection.json`,
beside `acs_2024_1yr_sources.json` — **not** in `microcosm/build/us`. That
directory is the US country package: its `country_package.json` declares every
file it ships and `spec_engine.load_bundle` refuses the whole package when an
undeclared file appears beside the manifest. A runtime source adapter's
resource does not belong in that closure.
`test_the_packaged_document_is_a_runtime_resource_not_a_country_spec` asserts
the placement rather than leaving it to habit.

## The generator

`tools/build_us_puf_source_layout.py` produces the document from the three
content-pinned booklet text extractions and the packaged raw source definition.
It reads no PUF microdata, imports no country model and computes no policy. For
each of the 222 + 7 delivered columns it requires exactly one hit for the
publisher's own `position. NAME` / `position NAME` pattern on the record-layout
pages, retains the whole matched line, and refuses if the matched offsets are
not monotonically increasing in delivered order — which is how the claim "the
delivered order is the printed order" is established rather than assumed. It
then matches every column again in the February 2023 edition and lists the
thirteen positions whose field code that PDF's text extraction splits with one
interior space (`P 04470`, `RE CID`, …).

```bash
python tools/build_us_puf_source_layout.py --documents-dir <booklet-texts> --write
python tools/build_us_puf_source_layout.py --documents-dir <booklet-texts> --verify
```

`--verify` needs the pinned booklet texts present locally; the packaged
document records the generator's own SHA-256, and a test compares it against
the tool on disk whenever the repo layout is available. The interpreter
version is execution provenance, written outside the document identity when
`--receipt <local-json>` is supplied. Each invocation rehashes all three
booklet texts, and the supplied directory is not part of the document.

The document records an unresolved record-count wording conflict: the
technical description says 207,696 CSV records including the field-name row,
while the accepted delivery audit measured 207,696 data records plus a header.
The decoder follows the authenticated delivered count and does not rewrite
the publisher's wording to reconcile it.

## Metadata at the consumer boundary

Artifact schema version 2 carries per-column grain, units, period status,
fitting refusal, unresolved codes, sign evidence and zero semantics in its
facts. The decoded object exposes the same metadata as an immutable mapping,
and both graph receipts retain it. Readback reconstructs these declarations
from the authenticated projection document and refuses altered metadata.
No fitting consumer is implemented; metadata preserves the required decision
boundary but does not prevent callers from copying raw arrays into arbitrary
code. A future admitted fitting consumer must explicitly check it.

The bounded genuine runner must authenticate the status artifact separately,
execute the typed producer kernel, and pass its measured digest as
`expected_status_artifact_sha256` when reading the monetary artifact back.

## Running it

```bash
uv sync --all-packages --locked          # no us/uk extras are needed
uv run pytest packages/microcosm-build/tests/test_us_puf_monetary_source.py
uv run ruff check .
uv run python tools/ci_test_groups.py --verify
```

The test file is `test_us_p…`, so it lands in the `us-p` engine group and the
`rest` fast group and never under `[defaulted]`. It needs no country engine.

## What this slice did not do

No PUF was read. No `.h5`, `.dta` or survey CSV was opened. No build, fit,
model calculation, factor generation or network download was run. Every fixture
is invented and written under `tmp_path` from literals in the test module.

## Remaining spine work

None of this is done, and none of it is implied by anything above.

1. **A first genuine run.** Everything here is verified on invented full-width
   fixtures. A genuine parse belongs in a separate reviewed private runner,
   after root and peer code review, and must name its host population, measure
   peak RSS, and preserve any amount-grammar refusal as a refusal receipt
   rather than widening the grammar. The twelve-digit bound, the whole-dollar
   grammar and the "no missing sentinel" policy are all propositions a first
   genuine parse can legitimately refuse.
2. **Donor-to-recipient contract.** Construct the ASEC+ACS survey multispine
   first, prepare its common predictors and gap fills, then attach
   PUF-enriched clones of that combined pool. Survey records supply household
   and person structure. PUF remains a return-grain tax-detail donor; it does
   not supply new observed households. The fit must define compatible
   return/tax-unit predictors and explicit allocation of transferred amounts
   to existing people. Both survey origins must reach the same fit/apply
   surface. One filer per return in a fixture establishes none of this.
3. **Period adjudication.** Until the FLPDYR conflict is settled by source
   evidence or an explicit recorded decision, no uniform-source-year
   `GrowthRecipe` can honestly cover these rows.
4. **Target-year growth.** A reviewed factor table still has to be generated
   and its digest added to `REVIEWED_FACTOR_TABLE_DIGESTS`, and the per-field
   choice between price restatement and per-return aging still has to be made
   (see [`docs/us-puf-growth-first-slice.md`](us-puf-growth-first-slice.md),
   obligations 2, 3 and 5). Only then may a rule name one of these columns.
5. **Distinct weights.** `S006` is not in this artifact. Its role is a donor
   return weight, never a household population weight. Clone attachment must
   preserve the survey pool's total starting household mass, and clone rows
   must not count as additional independent source households in district
   support diagnostics.
6. **Fit and apply.** No donor/recipient basis check, no QRF, no matrix, no
   calibration. The monetary metadata vocabulary exists in
   `puf_growth.puf_growth_monetary_metadata`; wiring these columns into it is a
   separate reviewed step.
7. **Aggregate-record disaggregation.** The four aggregates stay tagged and
   whole. Turning them into returns is a separate reviewed decision, and the
   `AGIR1` cutpoints printed on PDF 13 disagree with the disclosure paragraph
   on PDF 4.
8. **The other 161 amount fields.** They are declared, cited and source-only.
   Selecting any of them is another increment with the same per-column burden.

## Integration corrections

The root correction moves the raw definition into `us_runtime`, resolving its
undeclared membership in the US specification package and keeping runtime
metadata out of the country-spec executable-key scan. The archived provenance
strings in `asec_housing_status.py` and the negative import checks in
`test_us_puf_raw_source.py` receive explicit entries in the existing historical
reference allowlist. Neither imports or executes the retired data package.
The country package loader and these launch-contract checks remain enforced.
