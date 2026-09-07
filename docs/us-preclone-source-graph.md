# Complete nominal US pre-clone source graph

`microcosm.build.us_runtime.graph_preclone.us_preclone_source_nodes` composes
the six existing source stages over an assembled US pool: CPS carried inputs,
accepted measured hours, historical prior-year parity, relationships, actual
fresh-source housing and eligibility. The factory refuses a registered operator
order that differs from this six-stage recipe.

The module only declares and connects existing kernels. Housing runs the real
authenticated donor source, fit, recipient apply, measured household/SPM leaves,
five-leaf ID merge and acceptance gate. Eligibility runs its real person
preparation kernel afterward. The relationship and eligibility stages use the
existing accepted context interfaces; the composition contains no housing
stand-in or substitute calculation.

Call the factory with the initial column declarations, full-pool population and
its accepted context producer, plus the existing authenticated rent donor
descriptor:

```python
nodes = us_preclone_source_nodes(
    columns,
    base=assembled_population,
    context_producer=accepted_pool_context,
    donor_descriptor=rent_descriptor,
)
register_us_preclone_source_kernels(registry, source_codecs=source_codecs)
```

The caller declares the population source and existing rent source in the graph
as usual. The registration helper registers each stage and the rent kernels once.
`n_estimators` configures the existing rent fit and defaults to its original
setting; the tests use two trees and a generated donor fixture.

The next stage consumes `PRECLONE_FINAL_POPULATION` and the typed context from
`PRECLONE_ACCEPTED_CONTEXT`. Prior-year preparation depends on the accepted
hours gate, housing depends on the real relationship context, and eligibility
depends on the accepted housing gate. Housing's output population is its merged
full pool. Only outputs owned on that completed population enter eligibility's
column inventory; donor and source-projection columns cannot appear as invented
full-pool inputs. Withheld hours or housing acceptance leaves later stages
unreached on cold, automatic and required replay.

This recipe preserves the original nominal money convention and frozen
historical prior-year behavior. It adds no wage-history interface, prior-wage
science, money decoder, price adjustment, country-model calculation or new
scientific threshold. Corrected current-money decoding and the separate
prior-self-employment successor are later reviewed integration scopes.

## Verification and original baseline

The complete synthetic fixture starts from the eight-person mixed CPS/ACS pool
used by the housing oracle. An independent run authenticated every file and
imported Microcosm module against original commit
`f01964c7c71e1ae575ac4c16766dbdaed1a342d5`, then executed one original six-operator
call with a generated 60-row rent donor and an actual two-tree fit. Its final
frame matched the original housing golden's eligibility frame exactly. The
complete ordered receipt is committed as
`packages/microcosm-build/tests/golden/us_preclone_source_receipt_f019.json`;
its original SHA-256 is
`cf36fccec4106cbfd86d8361e1a469af3297ed0f7a3cbd9114ffb931e51631ea`.

The new graph fixture compares every entity, cell, null, dtype, index, membership,
weight kind and vector, strata, metadata and legacy mass record with both the
original golden and the actual direct chain. It checks the exact context bytes
and reconstructs the six ordered single-operator receipts against the original
complete receipt. It runs cold, automatic and required replay, failed real gates,
and changed recipient ages and person order. Recipient changes must actually
reach and recompute later stages while retaining the unchanged donor source
and fitted forest.

These are engineering parity checks on invented fixtures. They do not certify
a real population, calibration, housing representativeness or a release. The
fresh-source housing scope and its existing incumbent and duplicate-index
refusals still apply. Full-dataset memory profiling and the independently
reviewed small real-data rung remain required before scaling a build.
