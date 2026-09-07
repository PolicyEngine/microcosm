# US pre-clone prior-year graph

`microcosm.build.us_runtime.graph_prior_year` exposes the existing measured
adjacent-year source join as two graph nodes. Both declare the actual outer
phase `prepare_multispine_source_inputs_for_clone`:

- `early_preparation.prior_year.boundary` validates the assembled pre-clone
  person evidence through the registered source owner, then retains every row.
- `early_preparation.prior_year` reads available raw and incumbent person cells,
  invokes `prepare_us_prior_year_person` on the actual selected source table,
  and merges its three outputs by person ID.

The production order is the authored first three source operations: CPS-carried
inputs, measured hours, then prior-year inputs. Pass the hours boundary as
`base` and `early_preparation.hours.gate` as `context_producer` to
`us_prior_year_source_nodes`. That typed artifact dependency ensures the join
stays unreached if the mandatory hours gate fails, including required cache
replay. The hours derivation's pending context has a different type and cannot
satisfy the accepted-context input. Later stages consume this node's updated
context to retain the transitive gate dependency.

The registered `observe_multispine_prior_year_person` owns the existing CPS
selection, role validation, ephemeral role removal, all-null output omission,
and output-ID merge. It passes canonically resolved person `Weights`, selected
in the original person order, to the actual helper. Selected group counts use
actual memberships. Full counts and untouched entity descriptors come from the
authenticated US frame context; no incomplete Frame carries full-population
metadata. The graph retains the original person index and requires the context
v1 contract's default unnamed group indices and single explicit weight source.
The legacy direct API retains its existing wider input contract.

This slice is source-only. Positive clone indices refuse at the boundary. It
performs the actual unique `(source_year, PERIDNUM)` join, eligibility-flag and
sentinel handling, fallback, and null/presence rules, with no QRF construction.
The later support-population path still requires actual person/tax-unit
predictors and a separately declared fit/apply integration. `source_year` remains
the income year and money remains nominal; this adapter does not uprate money.

All three pre-clone outputs are owned, including transient
`employment_income_last_year`. Its original formula-owned removal belongs to
the later post-clone operation. Native non-CPS cells survive the declared merge.
Existing numeric output dtypes are retained when the real merge supports them;
boolean incumbents must already use the canonical nullable `boolean` graph
representation. The adapter refuses incompatible declarations instead of casting
observed input cells.

Each node's source receipt describes its single real operator, with local
`order_index=0`. The composed fixture checks those receipts in authored order
against the corresponding direct single-operator calls, and separately checks
the complete three-operator direct receipt against its pre-edit byte digest.
These ordered source receipts and the preserved legacy `mass_log` in the typed
context are distinct from the executor's structural accounting records.

Implementation identity binds the adapter, actual source helper and shared
Frame/graph runtime modules, the source-stage resource, dependency versions, and
the live registered output/formula-owned projections. The conservative helper
module hash also binds its unused support implementation; it does not imply a
support model is fitted by this source node. No source-provenance exemption or
frozen graph interface changes are introduced.
