# native-scale-transport — measurement harnesses and receipts

The three harnesses this lane ran, committed as they ran, plus the receipts they
wrote. Nothing here is a build, a certification or a release artifact; every
measurement JSON carries `"release_eligible": false` and a scope line saying so.

| file | what it ran |
|---|---|
| `harness19_base.py` | the baseline: the 19-node financial graph at 1/1000 on the branch point `5ff889814`, uncapped by node, with a current-RSS trace and the runner/node-loop split |
| `harness19_after_with_required_replay.py` | the same at this branch's head, followed by a `resume="require"` replay of the same graph against the store the cold run wrote, comparing manifest key, node keys, artifact identities, receipts and every store object's bytes |
| `harness19_tenth.py` | the 1/10 run: 158,737 source households, above the 96,860 the single bounded encode admitted |

All three descend from `~/PolicyEngine/_worktrees/microcosm-verify-once/.measure/harness19_verify_once.py`,
which is itself the v5 pilot probe parameterised; what is measured — the call,
the seed, the geography support, the sampler and its interval — is unchanged
from the accepted v4 cold pilot's own arguments except where a file's docstring
says otherwise.

Each harness needs `MEASURE_BASE` (the source tree), `MEASURE_RUN` (the staged
run inputs), `MEASURE_PROBE` (the output directory), `MEASURE_LABEL` and
`MEASURE_HEAD`, and is started through `setsid_exec.py` so it leads its own
session and the pid the shell prints is the pid to record.
