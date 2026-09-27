# Native SPM amount source

`qualify_current_spm_amounts(preparation, spm_inputs)` projects current ASEC
`SPM_ENGVAL` and `SPM_CHILDCAREXPNS` through the existing authenticated money
owner and qualified original SPM membership. It returns descriptive person and
unit tables with evidence. It issues no source capability and does not attach
canonical consumer inputs. Callers must retain both genuine owners and
requalify after relevant I/O.

The projection joins exact current-year native keys and IDs, checks original
household/person coordinates, and preserves complete literal SPM membership.
Amounts repeated on person records represent one unit amount; they are never
summed. Replicated amount bits, validity, status and zero origin must agree.
An absent money member leaves the whole unit unknown. Historical restated
amounts cannot fill current-year observations. ACS units remain unobserved.

Missing money uses the maintained canonical positive-zero storage encoding,
with missing status and invalidity retained; that storage value is not an
observed zero. Frozen-storage zeros also remain unknown unless their original
evidence authenticates a Census-encoded zero. Unresolved annual membership
does not qualify a nonzero amount merely because it is present.

The existing current-money domain and resource pins remain authoritative for
admission. In particular, this adapter does not resolve the conflicting energy
range in the ASEC dictionary or bypass the original owner's refusal. Childcare
is labeled reported uncapped expenditure. It is not claimed to be an observed
amount before subsidy.

The pure `project_spm_amounts` function accepts supplied descriptive evidence
for inspection and testing. Its output and `spm_amount_values_seal` hash confer
no authority. The genuine borrower checks both retained owners, exact active
implementation dependencies and source bytes before and after projection.

This layer performs no fitting, clone transport, graph attachment, calibration
or consumer export. Choosing unit rather than person donor weighting, completing
ACS amounts, and resolving consumer naming remain separate modeling work. Passing
invented fixtures establishes these contracts, not actual-data quality or release
readiness.
