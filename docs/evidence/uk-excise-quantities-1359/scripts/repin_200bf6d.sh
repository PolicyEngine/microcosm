#!/usr/bin/env bash
# microcosm#1359 B1: re-pin the UK Chronicle feed to 200bf6d55 (chronicle main after
# PolicyEngine/chronicle#384) in repos/populace-1359, in docs/uk-chronicle-feed-repin.md's
# order: the pin, then the vendored resources (first, on a fresh pin), the national
# references and membership, the local census, the local references and membership, the
# compile-parity receipts one surface at a time, and the coverage manifest.
#
# Memory (24 GiB machine): the generators and parity hold every parsed fact, so the national
# steps read a national cut (constituency, MSOA and county rows dropped, as on abf0921) and the
# local steps a local cut (constituency and local-authority rows, the #1306 approach). Each heavy
# step waits for a quiet machine. Writes only inside the tree and data/ukds/acceptance.
set -euo pipefail
T=/Users/mariajuaristi/Desktop/PolicyEngine/repos/populace-1359
D=/Users/mariajuaristi/Desktop/PolicyEngine/data/ukds/acceptance
A=$D/chronicle-uk-artifact-200bf6d
NAT=$D/chronicle-uk-artifact-200bf6d-national
LOC=$D/chronicle-uk-artifact-200bf6d-local-levels
UK=$T/packages/microcosm-build/src/microcosm/build/uk
PY=$T/.venv/bin/python
COMMIT=200bf6d559cf85bc9c2c5ee4df8c50ba0f5d09d1
NATIONAL_LABEL=chronicle-uk-200bf6d-consumer-facts
LOCAL_LABEL=chronicle-uk-artifact-200bf6d
FACTS=$A/consumer_facts.jsonl
LOG=$D/1359-excise/b1_repin.log
FROM=${1:-pin}

step() { echo "== $1 $(date -u +%FT%TZ)" | tee -a "$LOG"; }
run() { /usr/bin/time -l nice -n 10 "$@" >> "$LOG" 2>&1; }
quiet() {
  # Wait up to 30 minutes until no other process holds more than 3 GB.
  local deadline=$(( $(date +%s) + 1800 ))
  while true; do
    local big
    big=$( (ps -axo rss=,pid=,command= | awk -v me=$$ '$1 > 3000000 && $2 != me' | grep -v -E "awk|ps -axo" || true) | wc -l | tr -d ' ')
    if [[ $big -eq 0 ]]; then echo "   quiet at $(date -u +%FT%TZ)" >> "$LOG"; return; fi
    if [[ $(date +%s) -ge $deadline ]]; then
      echo "   waited 30 min; $big other process(es) above 3 GB; proceeding at low priority" >> "$LOG"
      ps -axo rss=,etime=,command= | awk '$1 > 3000000' | cut -c1-160 >> "$LOG"
      return
    fi
    sleep 60
  done
}
reached=0
at() { [[ $reached -eq 1 || $FROM == "$1" ]] && reached=1; [[ $reached -eq 1 ]]; }

cd "$T"
FACTS_SHA=$(shasum -a 256 "$FACTS" | cut -d' ' -f1)
MANIFEST_SHA=$(shasum -a 256 "$A/manifest.json" | cut -d' ' -f1)
step "b1 from $FROM in $T at $(git rev-parse --short HEAD); facts $FACTS_SHA manifest $MANIFEST_SHA"

if at pin; then
  step "0. .codex-work mirrors, 1. the pin and the validation register's restatement"
  mkdir -p .codex-work
  ln -sfn "$FACTS" .codex-work/consumer_facts_uk.jsonl
  ln -sfn "$A/manifest.json" .codex-work/consumer_facts_uk_manifest.json
  ln -sfn "$A" .codex-work/uk-artifact
  FACTS_SHA="$FACTS_SHA" MANIFEST_SHA="$MANIFEST_SHA" COMMIT="$COMMIT" A="$A" UK="$UK" "$PY" - >> "$LOG" 2>&1 <<'EOF'
import json, os
from pathlib import Path
uk, art = Path(os.environ["UK"]), Path(os.environ["A"])
manifest = json.loads((art / "manifest.json").read_text())
feed_path = uk / "chronicle_feed.json"
feed = json.loads(feed_path.read_text())
assert manifest["consumer_fact_schema_sha256"] == feed["consumer_fact_schema_sha256"], "schema moved"
assert manifest["schema_version"] == feed["artifact_schema_version"]
assert manifest["consumer_fact_schema_versions"] == feed["consumer_fact_schema_versions"]
assert manifest["facts_sha256"] == os.environ["FACTS_SHA"]
feed.update(
    source_commit=os.environ["COMMIT"],
    fact_row_count=int(manifest["fact_row_count"]),
    facts_sha256=os.environ["FACTS_SHA"],
    manifest_sha256=os.environ["MANIFEST_SHA"],
)
feed_path.write_text(json.dumps(feed, indent=2) + "\n")
levels_path = uk / "local_validation_levels.json"
levels = json.loads(levels_path.read_text())
levels["source_feed"].update(
    sha256=os.environ["FACTS_SHA"],
    manifest_sha256=os.environ["MANIFEST_SHA"],
    source_commit=os.environ["COMMIT"],
)
levels_path.write_text(json.dumps(levels, indent=2, ensure_ascii=False) + "\n")
print("pin:", feed["source_commit"], feed["fact_row_count"])
EOF
fi

if at vendor; then
  step "2. vendored resources from the full artifact"
  quiet
  run "$PY" tools/vendor_uk_ledger_facts.py --ledger-facts .codex-work/uk-artifact \
    --ledger-facts-sha256 "$FACTS_SHA" --ledger-manifest-sha256 "$MANIFEST_SHA"
fi

if at cuts; then
  step "3. national and local cuts (streamed)"
  mkdir -p "$NAT" "$LOC"
  FACTS="$FACTS" FACTS_SHA="$FACTS_SHA" NAT="$NAT" LOC="$LOC" python3 - >> "$LOG" 2>&1 <<'EOF'
import collections, json, os
src = os.environ["FACTS"]
nat_drop = {"constituency", "msoa", "county"}
loc_keep = {"constituency", "local_authority"}
# microcosm#1262 binds Scotland's private rents at BRMA grain (statistical_scope
# rows, S33...), so the local cut keeps every statistical_scope row but the small
# areas (LSOAs, data zones, output areas) no local reference reads.
small_areas = ("E01", "S00", "S01", "W01", "N08")
levels = collections.Counter()
kept = collections.Counter()
with open(src) as rows, open(os.path.join(os.environ["NAT"], "consumer_facts.jsonl"), "w") as nat, \
        open(os.path.join(os.environ["LOC"], "consumer_facts.jsonl"), "w") as loc:
    for line in rows:
        if not line.strip():
            continue
        geography = json.loads(line)["geography"]
        level = geography["level"]
        levels[level] += 1
        if level not in nat_drop:
            nat.write(line); kept["national"] += 1
        if level in loc_keep or (
            level == "statistical_scope" and not str(geography["id"]).startswith(small_areas)
        ):
            loc.write(line); kept["local"] += 1
for name, path, rule in (("national", os.environ["NAT"], {"dropped_levels": sorted(nat_drop)}),
                         ("local", os.environ["LOC"], {"kept_levels": sorted(loc_keep),
                                                       "kept_statistical_scope_except_prefixes": list(small_areas)})):
    receipt = {"source": src, "source_sha256": os.environ["FACTS_SHA"], **rule,
               "rows_by_geography_level": dict(sorted(levels.items())), "kept_rows": kept[name]}
    with open(os.path.join(path, "cut_receipt.json"), "w") as out:
        json.dump(receipt, out, indent=1)
print(json.dumps({"levels": dict(sorted(levels.items())), "kept": dict(kept)}))
EOF
fi

if at national; then
  step "4. national references and membership (national cut)"
  quiet
  run "$PY" tools/generate_uk_target_references.py --contract "$UK/uk_population_targets.json" \
    --ledger-facts "$NAT/consumer_facts.jsonl" --source-fact-feed "$NATIONAL_LABEL" \
    --output "$UK/target_references.json" --membership-report "$UK/target_reference_membership.json"
fi

if at census; then
  step "5. local census"
  run "$PY" tools/census_uk_local_targets.py
fi

if at local; then
  step "6. local references and membership (local cut)"
  quiet
  run "$PY" tools/generate_uk_local_target_references.py --ledger-facts "$LOC/consumer_facts.jsonl" \
    --source-fact-feed "$LOCAL_LABEL"
fi

if at parity_national; then
  step "7a. compile parity, national surface (national cut)"
  quiet
  run "$PY" tools/build_uk_ledger_compile_parity_signed_differences.py \
    --ledger-facts "$NAT/consumer_facts.jsonl" --surface national
fi

if at parity_local; then
  step "7b. compile parity, local surface (local cut)"
  quiet
  run "$PY" tools/build_uk_ledger_compile_parity_signed_differences.py \
    --ledger-facts "$LOC/consumer_facts.jsonl" --surface local
fi

if at coverage; then
  step "8. release-input coverage manifest"
  run "$PY" tools/build_uk_release_input_coverage_manifest.py
fi

step "done"
git status --short >> "$LOG"
git diff --stat >> "$LOG"
