# ASEC reported-income accounting

The prepared ASEC graph adds six person-level source observations: wages, net
self-employment income, SSI, cash public assistance, total money income and an
unallocated residual. They use each person's income year and express amounts in
2024 annual average C-CPI-U dollars. These are reported source amounts. They do
not replace simulated SSI or TANF and do not complete the eight ACS income
categories or establish a calibrated national or district comparison.

The fixed arithmetic order is `W=WSAL_VAL`, `E=SEMP_VAL+FRSE_VAL`, `S=SSI_VAL`,
`P=PAW_VAL_2024_price`, `T=PTOTVAL`, and `U=T-(((W+E)+S)+P)`. Each result is a
float64 person column named `asec_reported_{wage_income,
net_self_employment_income,ssi,cash_public_assistance,total_income,
income_unallocated}_2024_price`. E and U may be negative. U is an accounting
residual, not ACS other income or proof that every source component reconciles.
Capital gains are not added to these calculations. No weighting or household
aggregation occurs here. All selected household members remain, including
children, group-quarters residents and households with unknown tenure.

## Separate public-assistance source evidence

`asec_income_observations.py` authenticates `PAW_VAL` and `PAW_YN` from the same
three pinned Census PERSON CSV members already in the ten-file preparation
roster. Income years 2022, 2023 and 2024 correspond to surveys 2023, 2024 and 2025.
The join uses the full 22-digit PERIDNUM within income year and cross-checks
native household, line number and age against the authenticated T1 parent.
Every source person must join exactly once. The parent PAW_VAL column must exist
with a numeric, non-boolean dtype; every known incumbent must equal the source.
Zero known coverage is allowed and counted, including in 2024. A missing column
is refused. PAW_TYP stays under the existing TANF source handling and is outside
this new artifact.

The [2023](https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asec2023_ddl_pub_full.pdf),
[2024](https://www2.census.gov/programs-surveys/cps/datasets/2024/march/asec2024_ddl_pub_full.pdf)
and [2025](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf)
dictionaries give PAW_VAL a 0–99,999 domain and PAW_YN a 0/1/2 domain. Positive
amounts require receipt code 1. Codes 0 and 2 require amount zero. Below age 15,
both must be zero. Code 1 with zero amount is retained; code 0 at age 15 or above
is retained and counted. The five derived classes distinguish positive amounts,
zeros below receipt age, reported-receipt zeros, reported nonreceipt, and NIU
within the age universe. These describe publisher encoding, not respondent
intent. The pinned JSON resource records the three dictionary references and
the seven monetary-basis fields separately for each income year.

Only the validated money spec supplies price factors. The new PAW adapter uses
the same divide-then-multiply-nonzero operation as the sealed restater. It copies
2024 amounts and positive-zero bytes without arithmetic. The existing 33-field
schema, T1 and S verifier closures, original missing values, tax partitions,
housing evidence and money buffers are not changed.

## Two fixed binary artifacts

The **full PAW source artifact** is
`ArtifactType("microcosm.us.asec_income_observations", 1)`, kind
`microcosm.us.asec_income_observations.v1`, filename
`asec_income_observations.bin`. Its magic is `MCAINCOB\x01`, followed by a
little-endian uint32 header length, canonical ASCII JSON, the body, and a
32-byte SHA-256 of all preceding bytes. The header is at most 65,536 bytes and
the full artifact allows 1–600,000 persons. Its eight column-major buffers are
person_id, income_year, source_household_id, A_LINENO, A_AGE, PAW_YN, PAW_VAL and
PAW_VAL_2024_price. The first seven are little-endian int64; the last is
little-endian float64. Column k starts at `k*rows*8`; the body is `rows*64`
bytes. Source reconstruction owns immutable byte buffers and issues the
authenticated object. Replay independently rebuilds expected bytes and checks
the candidate's exact size and hash before any candidate parsing. It uses S's
bounded descriptor snapshot helper; an inherited `STUDENT_*` size/change reason
still denotes a sanitized refusal from that helper.

The **derived accounting artifact** is
`ArtifactType("microcosm.us.asec_reported_income", 3)`, kind
`microcosm.us.asec_reported_income.v3`. It uses magic `MCAINACC\x03`, the same
uint32/header/body/SHA-256 framing, a 65,536-byte header limit, and 1–600,000
selected persons. Its column-major body has 91 bytes per person:

1. person_id, income_year and A_AGE, each little-endian int64;
2. the six result columns in W/E/S/P/T/U order, each little-endian float64;
3. status, validity and zero_origin uint8 buffers for each of WSAL_VAL,
   SEMP_VAL, FRSE_VAL, SSI_VAL and PTOTVAL, in that field and axis order;
4. PAW_VAL status, validity, zero_origin and source_encoding_class uint8 buffers.

Thus columns 0–8 start at `k*rows*8`; evidence column j starts at
`72*rows+j*rows`. Every header column entry has exactly name, dtype, offset,
bytes and sha256. The closed header contains schema_version, artifact_kind,
rows, columns, body_sha256, inputs, source, arithmetic_order, monetary_basis,
old_evidence_axes, residual_meaning and release_eligible. Inputs contain the
prepared-receipt, selection, selected-money, income-payload and frame-context
hashes, the source and selection producer keys, and the person-position hash.
Source contains the original source identity, T1, scope and money header/content
identities. No caller-provided digest can issue full source authority. The
accounting reader reconstructs from its bound inputs and compares exact bytes;
it does not parse a candidate into trusted results.

Old W/E/S/T status, validity and origin buffers are copied exactly as they are.
In particular, unresolved frozen-fill origins stay unresolved; restored PTOTVAL
keeps its own authenticated Census encoding origin. PAW adds its independently
authenticated encoded-zero origin and derived encoding class. The new source
authority never upgrades older fields.

## Graph generation and declared reads

Prepared source kind, receipt, codec, stage identity and all five kernel refs
move to generation 3. CREATE emits frame_context, current_money,
prepared_receipt, housing_universe and income_observations. Its Frame inventory
contains none of the six reported-income columns. The preparation receipt binds
the new artifact's header, content and payload hashes, rows, columns, kind and
scope. The preparation implementation identity moves to schema 3 and binds the
new loader and its implementation/resource closure.

The new `asec_prepared.reported_income` node follows whole-household selection,
parallel to the existing engine branch. It reads the explicit
`Slice("person", ("source_year",))`, selected current-money bytes, full PAW
evidence, selection, the prepared receipt and frame context. Constructed
person IDs and source positions must match the selected money, source cohort
and SELECT's position digest. Receipt and full PAW must come from the same
CREATE execution; selected money, selection and context must come from the same
SELECT execution. Its exact output tuple is reported_income v3 followed by
frame_context v1; it owns the six new float64 columns and refuses incumbents.
No raw PAW Frame column is a fallback input.

The selection schema/type remains v2, and selected-money and HU transport
remain v1. Their formats and meanings are unchanged; the new prepared receipt
changes their parent binding. HU's receipt-aware reader explicitly accepts the
v3 preparation generation. Historical v2 artifacts remain preserved, but the
v3 runtime invalidates their cache keys. A new cold run must precede a v3 warm
replay. No new source file or source pin is introduced. The four real engine
outputs and their defaults remain separate from reported observations.

## Effective CSV parser profile

The source receipt records the field-size limit and the dialect of the default
`csv.reader` that actually reads member headers. Re-registering the name `excel`
does not change that default reader, so an unused named registration does not
change the recorded parser, source identity or warm result. The live field-size
limit does affect parsing: source validation and the prepared-stage cache key
both bind it. Preparation never changes either process setting.
