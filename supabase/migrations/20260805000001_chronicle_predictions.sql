-- Seed the Chronicle prediction history harvested for populace#628.
--
-- Keep the source records as JSON so heterogeneous predicted and actual
-- values retain their original JSON types.  The timestamp fields are cast to
-- Chronicle's timestamptz columns when the recordset is expanded.

WITH prediction_seed AS (
    SELECT *
    FROM jsonb_to_recordset(
        $chronicle_predictions$
        [
          {"id":"p001","ts":"2026-08-03T07:30-04:00","claim":"run 6 pool build completes in 6-8h from dispatch","type":"interval_hours","predicted":[6,8],"resolved":"2026-08-03T17:24-04:00","outcome":"MISS","actual":"8h timeout kill mid-transfer; transfer phase alone >7.5h","note":"inherited the run-5 era estimate without re-deriving from stage structure"},
          {"id":"p002","ts":"2026-08-03T09:45-04:00","claim":"run 6 agreement verdict by 16:00-21:30 ET Sun[Mon]","type":"time_window","resolved":"2026-08-03T17:24-04:00","outcome":"MISS","actual":"attempt 4 timeout-killed 17:23; no verdict that day"},
          {"id":"p003","ts":"2026-08-03T22:31-04:00","claim":"IF attempt 5b holds ~7h unbroken, transferred banks ~05:30 and verdict ~08-09","type":"conditional_time","resolved":"2026-08-04T02:55-04:00","outcome":"CONDITION_FAILED","actual":"5b preempted at 4h45m; third consecutive ~4.5-5h window","note":"the conditional was honest; the implicit P(window holds) was too high after two 4.7h windows"},
          {"id":"p004","ts":"2026-08-03T23:00-04:00","claim":"run-5-attempt-4-style windows: preemption in the 4-8h band","type":"binary","p":0.7,"resolved":"2026-08-03T22:05-04:00","outcome":"HIT","actual":"5a died 4.7h; 5b 4.75h; pattern held"},
          {"id":"p005","ts":"2026-08-04T09:45-04:00","claim":"transferred boundary lands 13:00-14:00 ET Tue","type":"time_window","resolved":"2026-08-04T13:27-04:00","outcome":"HIT","actual":"transferred+simulated both banked by 13:27"},
          {"id":"p006","ts":"2026-08-04T09:45-04:00","claim":"agreement verdict 16:00-17:30 ET Tue","type":"time_window","resolved":"2026-08-04T19:06-04:00","outcome":"MISS","actual":"19:06 (dtype bug + fix round intervened); miss by 1.6-3.1h"},
          {"id":"p007","ts":"2026-08-04T13:40-04:00","claim":"verdict pulled in to ~14:30-15:30 ET","type":"time_window","resolved":"2026-08-04T13:50-04:00","outcome":"MISS","actual":"run 7b died 15 min later on the PERIDNUM dtype bug","note":"issued minutes before an unknown-unknown fired; classic tail-risk under-weighting in a never-exercised code path"},
          {"id":"p008","ts":"2026-08-04T14:30-04:00","claim":"IF fix permits banked resume, verdict this evening","type":"conditional_time","resolved":"2026-08-04T19:06-04:00","outcome":"HIT","actual":"resume worked; verdict 19:06"},
          {"id":"p009","ts":"2026-08-04T16:00-04:00","claim":"#614 fix permits resume without ledger bump (no 9h rebuild)","type":"binary","p":0.6,"resolved":"2026-08-04T17:45-04:00","outcome":"HIT","actual":"load-time normalization; stored bytes untouched"},
          {"id":"p010","ts":"2026-08-02T23:45-04:00","claim":"total pool cost ~$70-80","type":"interval_usd","predicted":[70,80],"resolved":"2026-08-04T19:06-04:00","outcome":"MISS","actual":"~$105-120 through run 7","note":"flagged the breach in real time, but the original interval was too tight for a pipeline with unknown phase costs"},
          {"id":"p011","ts":"2026-08-03T12:00-04:00","claim":"peak RSS ~241GB means 330GiB container sizing is right (256 would OOM)","type":"binary","p":0.9,"resolved":"2026-08-04T13:27-04:00","outcome":"HIT","actual":"run 7 full build completed within 330GiB; profile peak 241GB"},
          {"id":"p012","ts":"2026-08-04T19:40-04:00","claim":"agreement gate would fail on first full-pool run (implicit: I gave no probability — treated pass as default)","type":"postmortem","outcome":"UNGRADED_BUT_INSTRUCTIVE","note":"never stated P(gate pass) before the verdict; the gate failing was treated as surprise when the #403 audit history made failure plausibly >40%. Lesson: state P(pass) before every gate."},
          {"id":"p013","ts":"2026-08-04T19:40-04:00","claim":"publish Thu-Fri (post-gate-fail revision)","type":"date_window","predicted":["2026-08-06","2026-08-07"],"outcome":"OPEN"},
          {"id":"p014","ts":"2026-08-05T09:45-04:00","claim":"publish Friday 8/7","type":"date","predicted":"2026-08-07","outcome":"OPEN"},
          {"id":"p015","ts":"2026-08-05T09:45-04:00","claim":"#616 merges by midday Wed (round 3 clean)","type":"time_window","predicted":"2026-08-05T13:00-04:00","p":0.55,"outcome":"OPEN","note":"two HOLD rounds so far; base rate for round-3 APPROVE on this PR family is good (607: r3 clean; 608: r3 clean) but this reviewer keeps finding real layers"},
          {"id":"p016","ts":"2026-08-05T09:45-04:00","claim":"10% pilot build's by-origin battery verdict lands this evening (Wed)","type":"time_window","predicted":"2026-08-05T22:00-04:00","p":0.5,"outcome":"OPEN","note":"gated on #616 merge + adoption lane + first stacked Modal run; three serial unknowns"},
          {"id":"p017","ts":"2026-08-05T09:45-04:00","claim":"pilot battery PASSES on first stacked build","type":"binary","p":0.45,"outcome":"OPEN","note":"stated BEFORE the run per p012's lesson; first contact between new architecture and real data; the completeness gate + declared metrics are new surface"}
        ]
        $chronicle_predictions$::jsonb
    ) AS source(
        id text,
        ts timestamptz,
        claim text,
        type text,
        predicted jsonb,
        p numeric,
        resolved timestamptz,
        outcome text,
        actual jsonb,
        note text
    )
)
INSERT INTO chronicle.predictions (
    id,
    ts,
    claim,
    type,
    predicted,
    p,
    resolved,
    outcome,
    actual,
    note
)
SELECT
    id,
    ts,
    claim,
    type,
    predicted,
    p,
    resolved,
    outcome,
    actual,
    note
FROM prediction_seed
ON CONFLICT (id) DO NOTHING;
