# ASEC current-money graph bridge

`us_runtime.graph_asec_prepared` supplies a bounded four-node graph over the
authenticated ASEC source. It prepares current money, selects whole households,
derives corrected CPS leaves, and evaluates four complete PolicyEngine-US output
closures. It provides local engineering evidence. It does not establish annual
student classification, population representativeness, tax or benefit scoring,
calibration, or release eligibility.

| Node | Operation | Result |
| --- | --- | --- |
| `asec_prepared.create` | Authenticate T1/money, compose observed student controls S, reconstruct actual tax units, attach housing status, promote graph dtypes | Full population, typed frame context, encoded money body, preparation receipt |
| `asec_prepared.select_households` | Seeded exact-count selection by source-year stratum | Selected population, context, selection receipt, separately typed selected money |
| `asec_prepared.cps_carried_current` | Read corrected amounts and declared routing codes | 24 person leaves including `age`, one SPM childcare leaf, updated context |
| `asec_prepared.engine_current_money` | Admit exact installed formula closures and invoke the actual PE adapter | Evaluation artifact; no formula-owned population columns |

## Original source and separate student controls

The `us-asec-prepared-current-money-v1` source codec accepts one directory with
exactly the ten names in `asec_prepared_source.PREPARED_SOURCE_FILES`: raw parent
P, household attachment H, restored T1, its canonical restoration receipt, three
housing cohort HDFs, and three official Census PERSON CSV members. The roster
accepts files directly, with no archive extraction or filename discovery.
Directory content keys bind relative names and bytes for caching. Existing
closed source pins and canonical replay establish authority.

`prepare_asec_current_money_population` performs this composition through the
reviewed public APIs:

```python
source = load_authenticated_restored_current_money_source(
    parent, household_attachment, t1, member_paths=members
)
ready = source.ready()
controls = load_authenticated_student_controls(source, ready, member_paths=members)
with_students = attach_student_controls(source, ready, controls)
tax = reconstruct_current_money_tax_units(with_students, ready)
housing = load_authenticated_housing_status(source, cohort_paths=cohorts)
attached = attach_housing_status(tax, housing)
```

S is reconstructed from the same official CSV members already in the directory.
It needs no additional source file and does not bypass canonical T1 replay.
The preparation receipt separately binds the original T1 file digest and source
identity, S receipt/content/attachment digests, full money header/content/spec
identities, tax and housing receipts, restoration receipt, and pre/post-promotion
Frame signatures. The prepared implementation identity includes S's own closure.
T1's original source and monetary verification closures remain unchanged.
The restoration sidecar's observation and frame hashes, per-cohort joins and
incumbent comparison counts must match facts derived from that authenticated
source. A canonical receipt with false summary claims refuses before student
or tax preparation, even when its checkpoint bytes are authentic.

Income-2022/2023 missing `A_ENRLW` and `A_FTPT` observations stay missing in their
original columns. S supplies only the authenticated `asec_A_ENRLW` and
`asec_A_FTPT` aliases. Actual tax construction projects those aliases into a
transient input view; housing authenticates the original T1 object. All known
incumbent controls, including complete income 2024, must agree exactly with S.
No fill, cast of missing observations, or alternate source authority is accepted.

These controls describe the preceding survey week in income-cohort-plus-one.
They retain the existing microunit proxy interpretation. Receipts explicitly set
`annual_five_month_student_status_validated` to false. The separate reviewed S
implementation is described in [the student controls contract](us-asec-student-controls.md).
S was composed locally after native and Fable approval of its frozen revision
`553084def66b5abd315e0e25bae9cd5e986d4867`; that is not root integration or approval
for genuine source production or a graph population run.

## Typed artifacts and exact selection

Full `ReadyCurrentMoney` exists only inside reviewed source preparation. Its
encoded body travels as `BoundCurrentMoneyBody`; household selection creates
`SelectedCurrentMoney`. Neither type is a `RestatedAsecMoney` subclass or can
originate full-source readiness. The selected codec binds the original body
header/content, preparation receipt, selection receipt, and ordered person and
household identities.

All 33 money fields retain their exact selected amount, status, validity and
zero-origin bytes. Factor/source evidence remains bound through the parent
money header. The graph preserves complete household membership, relationship
structure, source-year strata and original selected design weights. Selection
does not normalize mass. The separately approved engineering rung is 15 whole
households, five per cohort, fraction `0.0001`, seed `578`; it remains a separate
genuine-run decision.

Graph dtype promotion records every transition and checks exact value/null
preservation. Unsupported types, nonempty source metadata or mass history, and
unexpected weight authority refuse. The graph reuses the existing closed typed
frame context without widening its accepted metadata.

## Corrected leaves and model boundary

The corrected leaf node declares only eight routing columns from the raw person
table: age, four distribution codes, other-income reason, and two Social Security
reasons. Raw source dollar columns are never declared amount inputs. Existing
nominal owned leaves refuse at graph construction. Amounts come exclusively
from the selected current-money artifact.

The leaf contract records existing split constants, Social Security reason and
age-62 routing, OI_OFF 20/12 routing, IRA code 4, ANN NIU contribution zero,
signed losses, and exact member-consistent SPM childcare grain. It introduces no
new price factors, prior-year wages, housing amounts or retirement splits.

| Admitted output | Complete produced input leaves |
| --- | --- |
| `employment_income` | `employment_income_before_lsr` |
| `dividend_income` | `qualified_dividend_income`, `non_qualified_dividend_income` |
| `social_security` | `social_security_retirement`, `social_security_disability`, `social_security_survivors`, `social_security_dependents` |
| `capital_gains` | `long_term_capital_gains_before_response`, `short_term_capital_gains` |

Admission checks the full installed formula closure digest, formula/edge counts,
entity and PolicyEngine-US version for every root. The empty defaults allowlist
is pinned in `asec_current_money_engine_defaults_v1.json`. Total self-employment
income remains blocked by the absent SSTB input; total interest by tax-exempt
interest; total pensions by public-pension leaves. Constituent leaves retain
their declared meanings without implying those totals are complete.

The model contract uses period 2024, baseline parameters and the installed
PolicyEngine-US 1.819.0 / Core 3.31.0 runtime. These are pinned current execution
identities, with no claim to be the latest versions. Fresh runtime inventories
bind every package Python file and every JSON/YAML parameter file. Runtime
admission also checks actual package import locations against those inventories;
the engine kernel includes this check in its implementation identity before
cache lookup. Output arrays and closure
evidence stay in an evaluation artifact and never become formula-owned input
columns. There is no prior-wage or tax/benefit scoring claim.

## Cache identity and verification

`us_asec_prepared_graph` declares the four nodes against an explicit prepared
column inventory; `us_asec_prepared_registry` supplies their actual kernels.
The registered source codec and CREATE share preparation code. The executor
validates the codec and keys source bytes; CREATE performs preparation once,
including mandatory authenticated source reconstruction. It does not invoke
the codec to prepare the same population a second time.

The `asec_prepared_v1` implementation scope binds graph/source/runtime modules,
reviewed packaged resources, installed dependencies and actual loader identity.
Typed artifacts and producer keys bind body, selection, context and evaluation.
Required-cache lookup refuses changed source bytes, implementation, missing or
corrupt artifacts. Cache-only fresh-process execution must make zero kernel
calls.

Verification uses only tiny invented files with private test-scope pin
substitutions, the real source/tax/housing APIs, real graph executor and real PE
adapter. Direct, cold fresh-store and cache-only fresh-process paths compare
population/checkpoint, context, selection, selected-money, leaf and evaluation
bytes. Negative tests cover altered sources, body/context/selection bindings,
incumbent nominal leaves, missing routing and model/input drift. The author
report records exact frozen heads, loaded module hashes, actual commands,
preserved development failures and checks not run. This contract does not
self-certify any release.

The standalone current-money consumers resource remains an immutable source
authority snapshot because its bytes belong to the existing T1 verification
closure. The separately pinned `asec_current_money_graph_consumers_v1.json`
resource describes this slice's superseding wiring status. Only microunit and
this corrected graph slice are wired; other nominal consumer paths remain
pending. Existing source
addresses are preserved.
