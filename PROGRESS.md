# Income parser-profile review fixes — progress

State: implementing bounded fixes on a dependent worktree of the frozen
first slice `8b18fc18aceadcf45578c58b6bfc3fe563cfb58e` (branch
`us-income-parser-review-fixes-20260905`). The frozen author checkout at
`_worktrees/microcosm-us-asec-reported-income-first-slice-20260905` is
untouched and was verified clean before and after every step.

## Done
- Fresh `git fetch`; `origin/main` at `c1b83241`, frozen head 114 ahead / 0
  behind, explicit dependent child branch created from the frozen commit.
- Read `us-asec-reported-income-author.md`, the freeze binding and
  `us-income-first-slice-code-fable51-review.md`. No `income-native` Zeno
  report file present in R at start; his actionable P2 was taken from the
  lane prompt.
- P2: bound the live CSV parser limit/profile into the new source identity
  (`parser_profile()` in `asec_income_observations._implementation`), the
  prepared receipt (`income_observations.parser_profile`), the graph bind
  crosscheck, and the `asec_prepared_v3` stage manifest `runtime_options`.
  No process global is set by implementation code.
- Fable low items 1-5 addressed narrowly (person-entity assertion,
  CREATE-time price basis crosscheck, `match=` on every mutation refusal,
  hand-computed P/U row, bool and nullable-integer incumbent cases).
- Red->green cases added for the raw PAW API and for the full four-node
  pure graph replay.

## Next
- Run the pure source/graph suites and the already-authorized invented
  engine fixture suite with the author runner guards.
- Freeze the commit, verify the message from Git, write the durable report
  and receipt into R.
