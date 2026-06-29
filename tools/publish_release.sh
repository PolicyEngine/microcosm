#!/usr/bin/env bash
#
# Publish a Populace release and fire its Slack alert, in one tracked path.
#
# Thin wrapper around `populace-publish-release` so that every ship goes
# through the same command and the release alert is guaranteed to be wired.
# The alert itself lives in the publish CLI (populace.data.slack.notify_release)
# and is a no-op unless the channel webhook env var is set — this wrapper just
# loads those from an optional gitignored env file and warns if they're missing.
#
# Usage:
#   tools/publish_release.sh <release_dir> [--repo-id …] [--artifact-root …] …
# All arguments are passed straight through to `populace-publish-release`.
#
# One-time setup on the build machine (see README "Release alerts"):
#   cp tools/release.env.example tools/release.env   # then fill in the URLs
# or just export SLACK_WEBHOOK_POPULACE_US / _UK in your shell.

set -euo pipefail

ENV_FILE="${POPULACE_RELEASE_ENV:-$(dirname "$0")/release.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

if [[ -z "${SLACK_WEBHOOK_POPULACE_US:-}" && -z "${SLACK_WEBHOOK_POPULACE_UK:-}" ]]; then
  echo "warning: no SLACK_WEBHOOK_POPULACE_US/_UK set — this release will" >&2
  echo "         publish WITHOUT a Slack alert. Set them in your shell or in" >&2
  echo "         $ENV_FILE (see README 'Release alerts')." >&2
fi

exec populace-publish-release "$@"
