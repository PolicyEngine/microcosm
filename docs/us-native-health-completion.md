# Optional native health coverage completion

`run_us_survey_enrichment(..., health_completion=True)` enables a development
model for seven coverage categories that ACS does not separately observe. The
default is `False`; the existing literal health projection remains unchanged.
This option does not qualify a release or establish complete input coverage.

The model uses the full authenticated original ASEC donor roster and original
household DESIGN weights, independently of which survey households were selected
for receiving support. Training excludes people with an unknown target or required
predictor. The predictors are observed age, sex and observation-time state. A
missing required ACS predictor causes refusal; there is no filled predictor or
assigned-block-state fallback.

The seven targets come from the existing coverage field contract:

| Coverage category | ASEC item |
| --- | --- |
| Marketplace | `NOW_MRK` |
| Non-marketplace direct purchase | `NOW_NONM` |
| Medicaid | `NOW_CAID` |
| Other means-tested coverage | `NOW_OTHMT` |
| TRICARE | `NOW_MIL` |
| CHAMPVA | `NOW_CHAMPVA` |
| VA | `NOW_VACARE` |

Valid allocated source answers retain their values and allocation status. Blank
or unrecognized codes remain unknown. In particular, broader ACS coverage items
and ASEC `NOW_MCAID` do not become aliases for these narrower categories.

The graph exposes the full donor projection, target encoding and donor selection,
one original-ACS predictor matrix, seven fitted QRF target artifacts and seven
ordered apply steps. Each ACS original receives one set of draws, then its two
existing clones receive the same values. The attachment preserves every incoming
column, structural ID, membership, weight and source observation.

The original source-recode nodes retain their nullable values. On the receiving
population, `__known` and `__source_status` continue to describe source knowledge,
while a separate `__imputed` flag identifies modeled cells. A modeled true/false
value does not make an unobserved ACS answer source-known. ASEC unknown responses
and unknown ESI/IHS observations are outside this seven-category completion.

ASEC coverage is observed at the 2025 interview and ACS at the 2024 interview.
Transporting the relationship between those years is an explicit development
assumption. Scientific qualification is pending: held-out performance, donor
exclusion and allocation patterns, rare coverage categories, broader ACS category
consistency and sensitivity to the interview-year difference still need review.
The chained model does not guarantee that every predicted seven-category tuple
appeared together in a donor record.

Source authority stays with the live retained preparation and enrichment owner.
The private projection, graph artifacts and descriptive receipts cannot recreate
that authority. The host checks exact fitted-donor and draw/matrix identities,
reconstructs complete outputs on replay and requalifies sources at its final I/O
boundary. The full combined country-owner lifecycle and actual-data fit require
their own separately admitted verification; small invented graph tests do not
substitute for those checks.
