# populace-build

The build end of the populace stack: typed stage plans over
`populace.frame.Frame` with **declarative donor graphs** (every imputation
names its donor survey and fails loudly — no silent fallbacks), and the
**dataset acceptance gates** every release must pass before publishing:

- **parity** — every variable layer the incumbent populates, the candidate
  populates (no all-zero gaps on engine-known inputs);
- **support** — every imputed value lies inside its donor's realized range;
- **aggregate-vs-admin** — weighted aggregates land within declared tolerance
  of administrative anchors from the
  [target registry](../populace-calibrate/), with signs checked (this gate
  catches the class of failure where calibration silently drives net
  short-term capital gains to −$3.9T);
- **per-family fit** — the calibration's within-10% share, reported per
  source family so one family cannot hide inside the global average;
- **rotated holdout** — deterministic target folds so *every* target is held
  out exactly once across rotations, instead of one lucky split.

All gate losses use the calibrator's relative-error loss
`mean(((est − target)/(target + 1))²)` — scorers consume the same functions,
so there is no calibrator-vs-scorer objective mismatch.

The `us` extra adds the engine + survey loaders for the US stage plan.

## US plan status

`populace.build.us` declares the US build: stage order, donor graph with
citations (`US_DONORS`), and the manifest-ready `BuildConfig`. The stage
*implementations* are injected (`us_plan(implementations)`) and the plan
refuses to assemble with any stage missing — no stubs, no fallbacks. The
proven implementations currently live with the active build worktree; they
are being ported here as their canonical home immediately after the in-flight
v3 release ships (porting mid-release would either fork the implementations
or destabilize the release — the two failure modes this repo exists to
prevent). The port lands with the worktree copies deleted in the same commit.
