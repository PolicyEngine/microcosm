# Selected ACS immigration source projection

`borrow_current_acs_immigration_projection(preparation)` in
`microcosm.build.us_runtime.current_acs_immigration_source_projection` returns
selected original ACS source inputs from a live authenticated survey preparation.
It internally invokes the existing immigration literal reader and verifies the
retained ACS native owner. Call `validate()` after the consumer's final relevant
I/O; exclusive access to mutable source and output tables remains required.

The result contains three narrow tables and a descriptive receipt:

- `raw` preserves the six original string fields SERIALNO, SPORDER, CIT, POBP,
  YOEP and AGEP, indexed by the selected receiving person identifier.
- `person` binds typed CIT/POBP/YOEP/AGEP, observed SEX and `is_female`, original
  household state, receiving and native identifiers, source namespace, source
  year 2024 and observation year 2024. Nullable YOEP, NIU, public-code precision
  and lower/upper bounds retain missingness and censoring without a midpoint.
- `households` binds receiving/native household identifiers, original SERIALNO,
  original state token ST and typed state FIPS.

There is no Frame, weight vector, status assignment, fit, stock calibration,
graph attachment, model input admission or release approval. In particular,
selected ACS people are not a complete-country stock denominator. The existing
shared literal reader also verifies ASEC source files, but this result projects
only ACS fields and consumes no prior-income values. It is opt-in and changes no
build default.

## Source meanings

The [2024 one-year PUMS dictionary](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.txt)
defines the domains. The [2024 PUMS code list](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/code_lists/ACSPUMS2024CodeLists.xls)
defines detailed countries grouped under public POBP codes. The person projection
retains the ACS code namespace; it does not claim that the 222-code domain equals
ASEC PENATVTY.

| Input | Admitted evidence |
|---|---|
| CIT | Codes 1–5: US-born, specified US territories, US parents abroad, naturalized, noncitizen. No unknown code is filled. |
| POBP | Published 2024 public categories, including state births and grouped residuals. 451 means Sudan only. 464 groups Tunisia, Western Sahara and South Sudan; exact South Sudan origin is unavailable. |
| YOEP | Blank is NIU only for CIT 1. CIT 2–5 require a published code. 1938 means 1938 or earlier; 1939 means 1939–1944; 1945–2024 are individual calendar years. |
| AGEP | Published 0–99 code, directly equal to the native observed age. Source top-coding is retained, without an exact-age claim. |
| SEX | Native source code 1 or 2 and exact agreement with the native boolean. No unallocated-response claim. |
| State | Original household ST through exact original membership; 50 states and DC. This view does not admit Puerto Rico Community Survey semantics. |

The [2024 subject definitions, Year of Entry, printed page 146](https://www2.census.gov/programs-surveys/acs/tech_docs/subject_definitions/2024_ACSSubjectDefinitions.pdf)
include CIT 2 and 3 in the entry-question universe although Census calls them
native-born. Entry refers to the latest occasion of coming to live in the US,
with documented reporting ambiguity for repeat entrants. It is not proof of
first entry or continuous residence. Birthplace is not proof of nationality or
legal immigration status. Other public POBP categories can also be grouped or
unspecified: `published_category` deliberately makes no exact-country claim.

## Identity and verification

Each row joins the exact selected source namespace/native person key to original
SERIALNO/SPORDER, original household membership, observed age and sex, and the
same household's state. IDs never pass through a float conversion. The pure
projection supports arbitrary integer coordinates; authentic issuance retains
the existing upstream preparation's coordinate contract.

The live preparation and native issuer remain the source authorities. The view
retains independent immutable seals for the qualified literals and output tables;
altering a literal owner's callback closure does not move those seals. Copies,
detached replacement tables, a copied receipt and a different preparation cannot
stand in for the retained view. Final pure identity and table checks run after
source and implementation-file I/O. These checks detect changes at validation
boundaries; they do not make concurrent pandas mutation atomic.

Invented archive tests exercise the actual source catalogue, native issuance and
literal capture path, including household and group-quarters observations. Pure
domain/coordinate controls and mutation, copied-owner, late-I/O and editable-
closure controls complement them. This evidence does not qualify real sources,
immigration rules or a release artifact.
