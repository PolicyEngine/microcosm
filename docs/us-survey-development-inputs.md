# Source-qualified survey development inputs

The native financial host accepts
`source_qualified_development_inputs=True` to publish four current ASEC income
mappings before PUF application. The default is `False`.

The option adds `survey_development_inputs.source_projection` and
`survey_development_inputs.attach`. The first node publishes the authenticated
routing basis, source literals, knownness and named development assumptions.
The second maps each qualified original person to its complete pair of survey
clones. Both nodes bind the declared parent inputs and genuine producer keys.
Existing canonical columns cause refusal.

| Person output | Maintained development mapping |
| --- | --- |
| `taxable_private_pension_income` | Apply the maintained pension fraction to qualified pension and annuity amounts when both are known. |
| `taxable_ira_distributions` | Preserve the qualified regular-IRA route amount when the distribution and routing are known. |
| `rental_income` | Use the qualified net-property amount as the existing rental proxy. |
| `farm_operations_income` | Preserve the qualified farm amount, including a known negative amount. |

The implementation reuses
[`_development_person_values`](../packages/microcosm-build/src/microcosm/build/us_runtime/puf55_survey_observed.py)
and the current ASEC income-routing qualifier. Graph parameters and typed
artifacts record the existing rule names and values. They retain the raw totals
and unresolved source taxability or component statuses. These development
mappings do not establish observed taxable amounts or release-science acceptance.

American Community Survey (ACS) rows remain unknown. Ambiguous, off-route and
otherwise unqualified ASEC values remain unknown as well. A not-in-universe code
does not establish a zero. The attachment preserves identifiers, other columns,
typed weights, strata and population accounting.

The new attachment follows the selected base financial/property/tax output.
When child completion is enabled, the completion receiving version carries the
four additions from the genuine base. Household roles, child property and the
final tax gate remain the later writers. The output selector returns that final
tax gate for completed runs. Compact retention keeps the development attachment
in the base and retains the existing completion owner.

The tests execute genuine invented source preparation and routing, then real
source and attachment kernels with cold execution, required replay and
independent artifact/Population reconstruction. They also exercise the actual
child, household-role and tax declaration factories through the compiler and
check the compact roster. These component tests do not execute the complete
financial/property/PUF host. That continuation requires separate verification
before dataset-release claims.
