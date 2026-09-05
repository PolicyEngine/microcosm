# Pure ASEC current-money contract

`microcosm.build.us_runtime.asec_current_money` implements the bounded
`us_asec_current_money_ccpiu_2024_v1` declaration, decoder and restater. It runs
on explicitly supplied person/household projections. The original pure API uses
synthetic evidence; the separate [authenticated source adapter](us-asec-current-money-source.md)
now issues verified scope from the reviewed checkpoint pair. The pure module does not
load source rows, construct a Frame, execute microunit or a country model, fit,
calibrate, or alter the nominal recipe. Current wages are included. No prior
wage output, model, export or delivery interface is introduced.

The three JSON resources are a closed, pinned v1 projection of the reviewed
33-field Census authority (registry SHA256
`bdeb4d2fa1c37c6064530d560a62b12c515118608845b30d01b06c2eeeda8c6e`). They preserve
all 99 vintage citations, dictionary hashes, ranges and four zero-semantics
enums. Operational paths/timestamps and the unused workbook binary are excluded.
The consumer resource is a declaration of the corrected route; live nominal
reader alternatives remain unchanged. Compilation rejects any amended v1 bytes,
including unknown keys, fields, duplicate writers, raw alternatives or prior
wage surfaces. A change requires a reviewed resource/recipe revision, not a
runtime override or automatic pin refresh.

## Calls and authority

1. `load_current_money_resources()` reads the packaged resources and verifies
   installed microunit **0.1.0**, including the actual bytes of
   `microunit/tax_unit_construction.py` at SHA256
   `79007065113f9600601433dc2633f4392c675e28b87a7383ab8a1f7c6279d47e`.
   It uses distribution metadata and reads bytes without importing or executing
   microunit. An explicit `microunit_search_path` can select a separately pinned
   environment. Installed paths never enter identity.
2. Pure callers can instead inject `CurrentMoneyResources` bytes, or use
   `packaged_current_money_resources` with explicit inspected-code claims for
   fixtures. **An injected claim does not verify an installed distribution.**
3. `compile_asec_current_money_spec(resources, SyntheticAsecSource(...))` binds
   the resources, inspected reader claim, actual three-module source closure,
   Python/NumPy/pandas versions and platform, and explicit invented source,
   cohort, ED_VAL archive/member, attachment and selection pins.
4. `classify_asec_money(AsecMoneyViews(person, household, scope), spec, scope)`
   yields a temporary `DecodedAsecMoney` with normalized nominal arrays.
   `restate_asec_current_money(decoded, spec)` yields target-only
   `RestatedAsecMoney`. `require_complete_current_money(restated, spec)` returns
   only **`SyntheticReadyCurrentMoney`**. Every artifact header says
   **`source_authentication=synthetic_unverified`**. `production=True` refuses.
5. `encode_current_money(ready)` / `decode_current_money(payload, bindings)`
   persist/replay that synthetic artifact with its exact expected spec, source,
   scope, raw-input and representation bindings. Decoding requires the expected
   `MoneyBindings`; a caller-written hash does not authenticate original data.

There is no public authenticated source or verified readiness constructor.
Only the separate checkpoint loader can issue authenticated evidence. A different string label cannot promote a synthetic
source. State constructors are reserved to the validated transition functions,
so ordinary callers cannot accidentally relabel normalized nominal arrays as
restated/ready. These checks prevent accidental misuse in trusted Python; they
are not a security boundary against reflection or access to private internals.

## Values, position and numerical contract

Inputs support NumPy signed/unsigned integers, float64, and the enumerated
pandas nullable integer/Float64 equivalents. Float32/Float32, booleans,
object/string, complex and unlisted extensions refuse. Received amounts must
be finite integral dollars within the field-specific domain. Actual nulls stay
invalid and prevent readiness. Missing columns never default to zero.

Zero classification uses `zero_semantics`, not the list of NIU codes. CSP zero
means none-or-NIU; CHSP zero is declared NIU. ANN -1 is declared NIU normalized
to positive zero dollars, while ANN zero is an ordinary coded zero. Every
received zero retains unresolved original-zero provenance: the frozen loader's
`fillna(0)` gives no authority to infer original blanks. ANN -1 is not a
received zero, so it has NIU status without a zero-origin claim. All encoded
numeric zeros, including invalid slots, use positive-zero bytes.

SEMP -1/-9999 and valid rental/farm/total-income losses remain signed amounts.
Wages and capital gains refuse negatives. SPM_ENGVAL accepts only the
uncontested 0..10,000 interval; 10,001..99,999 refuses as disputed, and values
outside both published statements refuse as outside the codebook range. No
new account routing or universe exclusion is applied.

For income years 2022/2023/2024 (survey years 2023/2024/2025), the published
C-CPI-U cells are 163.6/170.0/174.4. Each cell is parsed directly to float64,
then target/source is divided, then each normalized nonzero amount is
multiplied once. Target-2024 values copy normalized bytes directly; zeros never
enter arithmetic. Factor expressions and little-endian IEEE-754 bits are in
the immutable spec. This is a conservative `platform_bitwise` scope, with no
cross-platform byte-equivalence claim. CPI belongs to this source restater;
later corrected leaf formulas must consume these target views once.

`AsecMoneyScope` carries ordered unique person/household IDs, native source
keys, income years and actual household/SPM membership. Keys must be unique
within year. `AsecMoneyViews.scope` records the positional association of its
projected tables; it must match the separately expected scope. The scope is
explicit caller evidence for synthetic sources. The separate verified loader
establishes that mapping from actual checkpoint bytes. Signed-int64 pandas Index/RangeIndex labels, including
duplicate/nondefault labels, are preserved separately in the fingerprint;
other index representations refuse before values are decoded. Projection and
permutation must move the explicit coordinate vectors with their rows.

HTOTVAL has exactly one address: household `asec_HTOTVAL`, retaining household
grain and a distinct target-current observation. Person aliases cannot satisfy
it. SPM amounts remain repeated on person rows with their native SPM grain;
years and received amount/status must agree across actual members. Household
observation years must agree with every member year. No first-member or maximum
operation hides disagreement. Arithmetic never reads weights or writes tables,
indices, raw fields, metadata, or source objects.

## Artifact and integration limits

Amounts and status/validity/zero-origin vectors are immutable bytes, with
read-only NumPy views. The restated payload has one target amount copy; it
contains no nominal copy and cannot recover raw observations by division.
Expected binding retains the explicit immutable spec. The framing is magic,
uint32 header length, canonical JSON header, ordered field buffers, SHA256 over
all preceding bytes. Decoding checks exact length/checksum, bindings, row/dtype
schema, closed status values and invalid-slot/zero encoding. No pickle is used.

Resources are limited to 512 KiB each, headers to 64 KiB, scope to 1,000,000
persons/400,000 households, and encoded payload to the resulting 356.5 MB upper
bound (32 person fields and one household field, 11 bytes per row per field,
plus framing). These are source bounds, not a claim of measured full-size
memory use or compatibility with a graph byte-codec limit. The pure slice is
verified on tiny invented fixtures only.

The separately reviewed source adapter supplies byte authentication and real
microunit tax-only reconstruction. Graph adapters, corrected current-leaf readers, clone lineage,
other donor price bases and end-to-end candidate validation remain separate
reviewable work. This module conveys no release or production readiness.
