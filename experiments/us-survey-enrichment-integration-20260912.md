# Survey amount and health integration

The release branch now includes the source-qualified health fragment from
`7cb1384be82ae4bd559577c5ebe142bf708e8570` and the fixed post-PUF enrichment host
from `99c35b0865e6f6cfe86ede15a10da91360a5e66b`. Their local integration commits
are `f92418a859e576ab7fd40919d3e9d75f41bd1ee4` and
`091261c0e979d0ee191ddbaf08a069b2554d7165`. Both cherry-picks were conflict-free.

The preceding main merge incorporated shared graph PR #913, main revision
`a9cc63e737fd4619a304117dc2ec7dcd86d97901`, without changing the release branch's
existing source tree. The checked PUF API, eighth financial output, source-string
hashing optimization, ACS anchor qualification and signed reconciliation helper
are preserved. None of those building blocks is interpreted as native acceptance
of this integrated candidate.

## Accepted combined stage

The host retains the original checked PUF run and original survey source owners.
It adds four amounts and nine coverage fields to `survey_puf55.receiving`, with
health explicitly dependent on the amount attachment. Source reporting-universe,
knownness and allocation information remain distinct from numeric values.
The [source/model specification](../docs/us-current-survey-amount-successor.md)
records UC classification, PHIP_VAL rather than PHIP_VAL2, the conditional
health-cost chain, deliberate ACS clone-sharing and narrower coverage gaps.

On the accepted prior implementation, the 18-person invented fixture executed
26 new nodes over a validated 245-node prefix. Required replay hit all 271 nodes;
complete Frame readback preserved parent columns, owners, geography, design
anchors and mass ledger. The diagnostic profile contained 74 of 161 input names,
leaving 87 missing. UC had ten known and eight unresolved values; all 18 had the
three medical-cost amounts and ESI, while narrower Medicaid coverage had eight
known and ten unresolved values. These are invented counts, not population
estimates, native coverage or evidence of statistical quality.

Acceptance combines the first five passing tests from the main v3 attempt and
the corrected final parent-revocation test from a focused v4 attempt. The v3
result remains failed: its last test named a nonexistent income column and
stopped before mutation. The corrected test uses the actual
`employment_income_before_lsr` column and proves permanent parent and child
revocation, including after restoring the original value. It passed without
repeating the already-passed required replay and Frame readback. There is no
claim of one all-green full v3 run. Independent source and evidence review
approved this six-control coverage at unchanged production bytes.

| Preserved local record | SHA256 |
| --- | --- |
| Combined v3 receipt | `19495b1cf3e6aad2f60771c8dce557ccbdc6a78b98c1a1ad453229276ac7e4a4` |
| Combined v3 aggregate acceptance | `07ac957851b19546a5ea1d2123975b9fa616dc8fe94a75e171c93e2379a094da` |
| Focused v4 parent-revocation receipt | `42bbeeae05b01a737d3c50cf3b5569be37cab43d71378fd8e78872965697170c` |

The v3 acceptance took 1,388.313 seconds with 917,979,136 bytes peak RSS; the
focused v4 control took 624.872 seconds with 885,669,888 bytes peak RSS. Both
retained 857 Python source pins and Torch thread settings of 1/1. These wrappers
are not the strict native pilot audit guard. The source-reviewed execution path
used invented fixtures; receipt flags alone do not prove native admission or
independent network/resource restrictions. Failed attempts remain preserved.

## Checks after integration

At integrated source `091261c0e979d0ee191ddbaf08a069b2554d7165`, 80 tests passed
with zero failures, errors or skips in 27.149 seconds wall time; peak RSS was
503,201,792 bytes. The selection includes all 32 amount controls, all 31 health
source/fragment controls, the actual invented UC source-qualification test and
the cheap financial/source controls, including the conserving interest
attachment. The health fragment executes cold and required replay. The expensive
financial/PUF/combined fixture was not selected again.

All 477 Python files under the shard source directories retained their hashes;
Torch threads remained 1/1. CPU and wall limits were applied. This is bounded
invented integration evidence, not the native admission guard.

| Integrated record | SHA256 |
| --- | --- |
| Receipt | `b2a764144afe952bba7a0f274cbf4f71ad6f0e46174743610127df7d42e30ba9` |
| JUnit result | `201fff9ac69b9259a9a7c11a291904dac97f082a92364521b622fc80d6805074` |

Actual declaration functions, invoked with invented descriptive values, confirm
three financial model targets, eight financial output fields and ten financial
extension nodes. Each PUF route retains 55 fits and 55 applications. The amount
fragment has 14 nodes and health has 12; together they compile with the explicit
parent and attachment dependencies. The complete host therefore still declares
271 nodes. The added tax-exempt-interest output changes the existing attachment
and its source identity, not the target chain or node count. No issued parent,
native source or full graph execution is established by this declaration probe.

Native PUF execution, original-channel tax-detail completion, remaining inputs,
full-candidate coverage, model evaluation, calibration and release verification
remain separate work. The ACS anchor qualifier and signed projection helper are
adopted source/numeric building blocks; the current enrichment host does not yet
invoke them to reconcile property or retirement components.
