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
- **export surface** — every replacement artifact can prove that its
  exported variables match a reference surface, with only documented
  structural extras or reviewed exclusions (for UK, this is the eFRS
  compatibility check);
- **target surface** — the calibration target set covers the reference
  target surface and may only be wider, not narrower (for UK, Populace must
  calibrate to at least the eFRS target surface);
- **per-family fit** — the calibration's within-10% share, reported per
  source family so one family cannot hide inside the global average;
- **rotated holdout** — deterministic target folds so *every* target is held
  out exactly once across rotations, instead of one lucky split.

All gate losses use the calibrator's bounded relative-error loss
`mean(((est − target)/(target + 1))²)` — scorers consume the same functions,
so there is no calibrator-vs-scorer objective mismatch.

The `us` extra adds the engine and survey loaders for the US stage plan.

## US plan status

`populace.build.us` declares the US build: stage order, donor graph with
citations (`US_DONORS`), and the manifest-ready `BuildConfig`. The stage
*implementations* are injected (`us_plan(implementations)`) and the plan
refuses to assemble with any stage missing — no stubs, no fallbacks. The
canonical implementations live in `populace.build.us.sources`; release-specific
benchmark comparison harnesses live outside this repo.
