# Final report: pkg3 post-transfer receipt validation failure #2

## Outcome

Fixed at executable commit `a932974f`.

The supplied weeks-unemployed failure was a receipt-generation bug, not an
invalid carrier model and not a count-target exception. Candidate capacity and
prefix selection reduced the same ordered weights through different float64
paths, so the prefix exceeded its declared candidate mass by one ULP. The
required cross-target audit then exposed the same class of bug one level up:
the first repair composed maximum capacity from independently rounded
partition endpoints, putting both child-support maxima 42 ULP above the whole
recipient mass.

The complete fix changes both generating relationships:

- one immutable `_PrefixSchedule` now supplies each candidate endpoint and all
  prefix-selection evidence (`post_transfer_calibration.py:282-287,471-515,844-872,891-928`);
- maximum capacity is generated from the row union
  `fixed_positive | allowed_positive | zero_candidates`, zero-masked onto the
  recipient-weight vector with identical length, order, and reduction topology
  to `recipient_total` (`post_transfer_calibration.py:823-885`).

The exact validator inequalities are unchanged: maximum must not exceed its
recipient superset, and a reported prefix must not exceed its candidate set
(`post_transfer_calibration.py:1457-1475,1499-1529`). No tolerance, threshold,
band, gate, comparator, seed, fold, target, or carrier constraint changed.

Both SHA-pinned no-build harnesses now validate. All five package test roots,
repository-wide Ruff, touched-file formatting, and whitespace checks are
green. This lane ran no host build and made no push. The serial host owner owns
the next 1% rerun against `a932974f`.

## No-build checkpoint reproduction

The read-only stage root was:

```text
/Users/maxghenis/PolicyEngine/_buildo-runtime/out/battery-verify/pkg3/pool.checkpoints/stacked/8f5077d6a1d5440b241f22fe4d20ad1d889924a27d094cb669e1035f9306546b
```

The current no-build checks are:

```text
uv run python tools/reproduce_us_post_transfer_weeks_checkpoint.py \
  --checkpoint-stage-root <stage-root-above> --expect valid
uv run python tools/audit_us_post_transfer_child_support_checkpoints.py \
  --checkpoint-stage-root <stage-root-above> --expect valid
```

### Weeks-unemployed failure

`tools/reproduce_us_post_transfer_weeks_checkpoint.py` validates the assembled
Frame plus unemployment-compensation and weeks target file, identity, and raw-
draw hashes. It reconstructs only the native clone-0 vectors and calls the live
kernel and strict validator; it performs no fit, DAG execution, artifact write,
or build (`reproduce_us_post_transfer_weeks_checkpoint.py:1-6,27-58,76-143,146-254`).

Pinned artifacts:

| Artifact | File SHA-256 | Identity SHA-256 | Raw-draw SHA-256 |
| --- | --- | --- | --- |
| `assembled.checkpoint.h5` | `5ce1815fc44dc43c7c24ccf27526852b8f1bddbdfe371255410a22f9b56ac015` | whole file pinned | n/a |
| `019__unemployment_compensation.h5` | `dc6637936ed4bd0322d38eaa3a4920fd137565f314387db3b3fdc7dfd6bc3086` | `708722093ca610426175998d50bbb6663585b07ffef912899f17adc90520f51f` | `e32d1559668e10b24abad8e1d639e4dbade964a712925bfe8f56d3136b839840` |
| `000__weeks_unemployed.h5` | `898397733aa3e5d8ec7d6679cb16a0504e826e25d23ca2c788f4397e0e061a43` | `d0d554ba05045e39a07f0f9515c83bbf754f067df12b8247f4bf3866162c4bdd` | `0214c8dcbc118676336069b906a07ee6145f2178542b6c5b4fb5899ad62d09f3` |

The production owner selects ASEC clone-0 reference rows, ACS clone-0
recipient rows, transferred nonnull mutable cells, and positive-UC mutable rows
for both the weeks allowed and addition masks
(`stacked_spine.py:8960-8971,8995-9032`). The pinned replay contains:

- 38,604 native person rows;
- 4,311 reference and 34,293 recipient/mutable rows;
- 134 positive reference rows;
- 24 initial recipient positives, all disallowed; and
- 32 positive-UC addition candidates
  (`reproduce_us_post_transfer_weeks_checkpoint.py:146-193`).

At reproduction commit `4cc41652`, the harness with `--expect invalid` exits
zero only for the exact supplied failure. Candidate capacity is
`85,676.23791782455`; the ID-ordered upper prefix is
`85,676.23791782456`; the excess is `1.4551915228366852e-11`. The sole false
relationship is:

```text
upper_prefix_mass <= addition_candidate_mass
```

It raises exactly:

```text
ValueError: Frame post-transfer calibration person/source_operator_weeks_unemployed/weeks_unemployed: match-reference carrier capacity relationships are invalid.
```

The harness pins that predicate, both floats, and the complete error rather
than accepting any aggregate validation failure
(`reproduce_us_post_transfer_weeks_checkpoint.py:210-231,257-286`).

Against `a932974f`, `--expect valid` reports candidate and upper prefix both
`85,676.23791782456`, zero prefix/candidate delta, no failed relationships,
and a valid receipt. The attainable-union maximum under the recipient
reduction topology is `85,676.23791782453`.

### Child-support cross-target reproduction

`tools/audit_us_post_transfer_child_support_checkpoints.py` reconstructs the
same native clone-0 support and half weights for both child-support targets.
It pins each target's whole-file, identity, and raw-draw hashes, then requires
both receipts—not merely one—to match the requested exact red or green state
(`audit_us_post_transfer_child_support_checkpoints.py:1-80,91-207,210-302`).

| Artifact | File SHA-256 | Identity SHA-256 | Raw-draw SHA-256 |
| --- | --- | --- | --- |
| `000__child_support_expense.h5` | `d119075e19fb767f3d8d24c7c0149d0df1ed963774a4b93d96974a72b3ac9bfe` | `41e3a6e3877fda23107b27bcd85aa6dd95e0f341d1e4b079defa6847f90b4cab` | `8b2845aff0aa0695d98ae30828523bf6bca9c5d4ed5d2d91d2d1a636bb917600` |
| `001__child_support_received.h5` | `66120896d5793f3d737f9ffac2058e2196992e357f8d869f4b31b259d041b3aa` | `41e3a6e3877fda23107b27bcd85aa6dd95e0f341d1e4b079defa6847f90b4cab` | `ea7f2eebb430b654acc639ef6ee6ed482207ffd74d54ba3a47cb55056813a381` |

Against the incomplete candidate-only repair `d7b12bab`, both receipts fail
only:

```text
maximum_attainable_mass <= recipient_total
```

For each target, recipient total is `79,926,522.10879111`; the partition-
composed maximum is `79,926,522.10879174`; the excess is
`6.258487701416016e-07`, or 42 ULP. The underlying endpoints are:

| Target | Allowed-positive mass | Addition-candidate mass | Old composed maximum |
| --- | ---: | ---: | ---: |
| `child_support_expense` | `71,696.09739141785` | `79,854,826.01140033` | `79,926,522.10879174` |
| `child_support_received` | `180,209.75664861224` | `79,746,312.35214312` | `79,926,522.10879174` |

Against `a932974f`, both strict receipts validate. Their attainable-union
maximum equals recipient total exactly, `79,926,522.10879111`, while the
independently rounded diagnostic partition sum remains
`79,926,522.10879174`. This demonstrates that the generating set relationship,
not the validator, was repaired.

## Root cause and semantic decision

Float64 addition is order-sensitive. The original code used a masked
`ndarray.sum` for candidate capacity and a separately ordered `np.cumsum` for
prefix selection. The first repair correctly unified those two values, but it
then added independently rounded fixed, existing-positive, and zero-candidate
endpoints to describe a different claim: the maximum mass of their row union.

Both strict invariants are semantically correct:

1. a prefix cannot exceed the candidate set from which it was selected; and
2. an attainable subset cannot exceed its recipient superset.

The final maximum implementation retains the recipient vector's length and
order and replaces unattainable entries with zero. With finite nonnegative
weights, each attainable leaf is less than or equal to its corresponding
recipient leaf, and the identical reduction topology preserves that ordering
through every floating-point addition (`post_transfer_calibration.py:775-783,823-885`).
It does not use `min`, `nextafter`, a tolerance, or a post-hoc clamp.

The regressions lock both numerical mechanisms for every late
`match_reference` declaration:

- the exact 32 production weeks weights distinguish masked sum from ordered
  prefix by one ULP;
- a four-weight case makes independently rounded capacity partitions exceed
  their whole set by one ULP; and
- a constrained proper-subset case makes a compressed subset sum
  `0x1.433526fbe1946p+48`, or `0.0625`, greater than its superset
  `0x1.433526fbe1945p+48`; the same-topology union validates exactly
  (`test_us_post_transfer_calibration.py:544-753`).

Weeks remains a valid positive-carrier calibration target. Its source accepts
only integer `-1` or `0..52` and maps `-1` to zero; its QRF path rounds, clips,
positive-UC-gates, and revalidates `0..52`; its carrier event is `weeks > 0`
(`weeks_unemployed.py:791-800,911-983,1218-1222`). Post-transfer amount mapping
uses only positive reference-donor support (`post_transfer_calibration.py:577-626,690-705`).
Count-valued support therefore does not invalidate weighted carrier capacity.

The ACS runtime's explicit discrete-numeric set contains only two mortgage-year
targets; other numeric targets use the ordinary numeric encoding
(`acs_transfer.py:129-138,3035-3117`). QRF's at-most-32-value “near-discrete”
branch is a leaf-storage optimization, not a carrier semantic type
(`microcosm-fit/qrf.py:388-401,482-503`). Annual child-support and disability
amounts also entered that optimization in the host log, which independently
rules out treating it as a weeks-specific count exception.

## Seven-target late-transfer audit

The immutable registry declares seven late targets. Six use
`match_reference`; disability benefits uses `preserve_recipient` and never
emits capacity or prefix evidence
(`post_transfer_calibration.py:208-258,840-932,1319-1335`).

| Late target | Evidence and verdict |
| --- | --- |
| `child_support_expense` | Covered. Nonnegative annual `CHSP_VAL`, not a count (`child_support.py:166-201,369-383`). Its pinned checkpoint fails the old whole-capacity relationship and validates the final union mechanism. |
| `child_support_received` | Covered. Nonnegative annual `CSP_VAL`, not a count (`child_support.py:166-201,369-383`). Its pinned checkpoint has the same red/green proof. |
| `disability_benefits` | Inapplicable to this capacity bug. It is a nonnegative annual two-slot amount excluding workers' compensation (`disability_benefits.py:184-220,382-395`) and uses `preserve_recipient`; its inspected checkpoint keeps before/after carrier mass at `42,658.57948297383` with `capacity=None` and `selection=None`, as required by the preserve-mode receipt branch (`post_transfer_calibration.py:1319-1335`). |
| `weeks_unemployed` | Covered. Sole semantic count target, integer `0..52`, with carrier additions constrained to positive-UC mutable rows (`weeks_unemployed.py:791-800,911-983,1218-1222`; `stacked_spine.py:8995-9008`). Exact pinned red/green replay proves reducer order caused the failure. |
| `workers_compensation` | Covered. Nonnegative annual `WC_VAL`, not a count (`workers_compensation.py:143-184,337-355`). It uses the default mutable carrier/addition masks (`post_transfer_calibration.py:786-812`) and the shared six-spec regressions. |
| `spm_unit_energy_subsidy` | Covered. Nonnegative measured `SPM_ENGVAL`, checked within unit and reduced to SPM-unit float64 (`energy_subsidy.py:157-233,537-557`). Its entity grain changes the weights, not the set/reduction mechanism; the shared regressions cover its declaration. |
| `pre_subsidy_care_expenses` | Covered. Nonnegative monetary care expense. ACS reconciliation restricts carriers to qualifying people and at most one per tax unit; the late owner admits one stable zero candidate per empty unit (`acs_transfer.py:660-739,1277-1299`; `stacked_spine.py:8728-8746,8977-8986`). The proper-subset six-spec regression covers this constrained structure. |

Current zero-based late-DAG positions are child support 24, disability 25,
weeks 30, workers' compensation 31, energy subsidy 32, and adult care 34.
Registry scheduling and stacked execution are deterministic, and each group
calibrates before returning (`us_late_producer_registry.py:1338-1396,2013-2019`;
`stacked_spine.py:10054-10095,10927-10931`). The failed host run produced child,
disability, and weeks checkpoints but stopped before workers, energy, and adult
care. Verdicts for those later targets are therefore source/mask proofs plus
shared-kernel regressions, not claims of nonexistent checkpoint replay.

## Verification

On the exact tree committed as `a932974f`, these commands ran serially under
the owner-provided memory guard and exited zero:

```text
uv run pytest packages/microcosm-fit/tests -q
uv run pytest packages/microcosm-calibrate/tests -q
uv run pytest packages/microcosm-data/tests -q
uv run pytest packages/microcosm-frame/tests -q
uv run pytest packages/microcosm-build/tests -q
uv run ruff check .
uv run ruff format --check <four touched Python files>
git diff --check
```

The focused post-transfer file passed all 47 cases. The complete build root
also covered stacked-spine, late-DAG, multispine pool, H5, pool-tool, terminal
receipt, and owner-mask paths. Only established skips and warnings appeared.

Both current checkpoint commands exited zero with `--expect valid`. Detached
temporary worktrees proved the hardened red side: weeks against `4cc41652` and
both child receipts against `d7b12bab`. The temporary worktrees were removed.

No host build ran from this lane. The read-only host log and checkpoint tree
were not modified. The owner-provided untracked `.codex-memory-guard.py` and
`_BUILD-FAILURE-1PCT.txt` remain unchanged. Nothing was pushed.

## Commit lineage and handoff

- `b533bc61` — open and commit the progress journal;
- `4cc41652` — add the SHA-pinned weeks reproduction and red regression;
- `d7b12bab` — bind candidate capacity to its ordered selection schedule;
- `47742720` — record the cross-target child-support escalation;
- `a932974f` — generate maximum capacity from the attainable row union, add
  the child red/green harness and exact proper-subset regressions, complete the
  seven-target audit, and record the green suite; and
- the following documentation-only commit — close `PROGRESS.md` and publish
  this report without changing the tested executable tree.

The next authorized action is the serial host owner's 1% rerun at executable
commit `a932974f`. Certification, publication, and release-chain mutation stay
outside this lane.
