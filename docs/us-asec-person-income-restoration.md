# ASEC person-income source restoration

The frozen income-2022 and income-2023 raw HDFs omit PTOTVAL. Pooling them with
the income-2024 HDF creates an incomplete column. The original Census person
members contain the observations. This attachment restores those source values
without replacing historical checkpoints or imputing income.

The closed local chain is original v4 (P), household-observation attachment (H),
and person-income attachment (T). The producer in
`microcosm.build.us_runtime.asec_person_income_source` accepts local P/H paths,
explicit paths to each of the three registered person CSV members, and a new
output directory. Public APIs accept no source hash or pin overrides. T appends
one int64 person column, `asec_PTOTVAL`, preserving legacy PTOTVAL and every
other parent column, axis, weight, stratum and history item.

The exact join uses `(source_year, PERIDNUM)`, with original household ID, person
line number and age cross-checks. Every cohort must have full, unique coverage.
Every nonmissing incumbent amount must agree, and the complete 2024 incumbent
must be present. CSV fields are parsed as exact integer tokens, with no fillna.
Invalid source amounts, keys, vintages, bytes and conflicting observations refuse.
Source values and person identifiers do not appear in error messages or receipts.

`restore_asec_person_income_source` produces the new checkpoint and a value-free
receipt. `verify_asec_person_income_source` independently reconstructs the full
attachment from authenticated parents and official member snapshots, then checks
the exact canonical checkpoint bytes against a private candidate snapshot. It
serializes the independently reconstructed frame and receipt into a private file,
requires the candidate's byte length and SHA256 to match that file, and never
opens candidate HDF content. The candidate's observed digest alone grants no
authority. This prevents candidate links, external storage, virtual datasets,
attributes or declared shapes from controlling a read before authentication.

T accepts only the deterministic canonical checkpoint encoding produced by the
reviewed writer. An equivalent frame in a different HDF encoding is refused.
The explicit encoding contract is part of the receipt and implementation hash.
T identity is environment-locked: produce and verify it with the reviewed locked
dependencies, preserve those exact bytes, and produce a new immutable attachment
after an intentional environment or implementation change. Historical source
kinds and generic checkpoint readers retain their existing contracts.

The producer reconstructs once to write T and again through the verifier before
publication. Verification also writes one private canonical checkpoint, adding
temporary disk I/O approximately equal to the attachment size. These independent
replays remain mandatory. Measure their time and peak memory on the source-only
rung before choosing a larger model population.

The output directory must be new. Exclusive directory creation and hardlink
publication never replace an existing destination. A failure during final
publication can leave a partially populated newly created directory; inspect it
and use a new destination on retry. The producer does not remove an existing or
raced destination to recover from that failure.

`load_authenticated_restored_current_money_source` issues the new kind only after
that comparison. It binds all parents, the new attachment digest, source reader
and schema implementation, and exact logical-field mapping. The old two-file
loader remains available and continues to refuse money readiness when its source
amounts are incomplete.

The new source and money header use schema 2. Their exact field override maps
logical PTOTVAL to `asec_PTOTVAL` and sets zero provenance to
`authenticated_census_csv_encoded_zero_v1`. Zero-origin bytes are:

| Code | Meaning |
|---|---|
| 0 | No encoded zero slot: nonzero, missing, or a distinct declared NIU code |
| 1 | Frozen fillna zero origin unresolved |
| 2 | Zero explicitly encoded in the authenticated Census CSV |

Only restored PTOTVAL may use code 2. The other 32 fields retain their previous
provenance. Legacy and synthetic sources retain schema 1 and code 1. Code 2
attests the source encoding, not a respondent answer: the dictionaries define
PTOTVAL's question universe as persons aged 15+, and under-15 source-encoded
zeros must not be described as elicited income responses. Restatement preserves
origin bytes, and field validation and exact content replay reject origin swaps.
No monetary domain, factor, completeness rule or field roster is relaxed.

Tax-unit construction consumes the ready target-current total. Money and tax
results must be regenerated for T's identity. Housing status must also be loaded
against that same authenticated source; identical raw household observations do
not permit reusing the old source identity. Cross-source tax/housing composition
continues to refuse.

The implementation tests use invented sources and private fixture pins. They do
not certify an actual population or authorize publication. The genuine producer
rung follows independent review of frozen code, then a source-only full-file
attachment and completeness audit before any larger build.
