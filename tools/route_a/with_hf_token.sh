#!/bin/bash
# Only the exec'd release process receives the runtime credential.
set +x
set -euo pipefail

[ "$#" -ge 2 ] || { echo "usage: with_hf_token.sh <agent-secret> <command> [args...]" >&2; exit 64; }
secret_helper=$1
shift
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN HUGGING_FACE_TOKEN_MAX
HF_TOKEN=$("$secret_helper" get HUGGING_FACE_TOKEN_MAX 2>/dev/null) || {
  echo "FAIL: HF credential lookup failed" >&2
  exit 1
}
[ -n "$HF_TOKEN" ] || { echo "FAIL: empty HF credential" >&2; exit 1; }
export HF_TOKEN
exec "$@"
