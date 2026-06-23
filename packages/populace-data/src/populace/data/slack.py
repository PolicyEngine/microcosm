"""Best-effort Slack notification when a Populace release is published.

Sent inline from :func:`populace.data.publish_cli.main` right after
``latest.json`` is uploaded, so the alert is real-time (no polling) and fires
wherever the publish runs. The webhook URL comes from a per-country env var, so
this is a no-op unless it is configured, and a failed post is logged rather than
raised — a Slack hiccup must never fail a release.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from typing import Any

CHANNEL_ENV: dict[str, str] = {
    "us": "SLACK_WEBHOOK_POPULACE_US",
    "uk": "SLACK_WEBHOOK_POPULACE_UK",
}

DASHBOARD_URL = "https://calibration-diagnostics.vercel.app/populace"


def country_for_repo(repo_id: str) -> str:
    return "uk" if "uk" in repo_id.lower() else "us"


def _default_post(url: str, payload: dict[str, Any]) -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        response.read()


def notify_release(
    repo_id: str,
    release_id: str,
    updated_at: str | None = None,
    *,
    webhook: str | None = None,
    post: Callable[[str, dict[str, Any]], None] | None = None,
) -> bool:
    """Announce a published release to the country's Slack channel.

    Returns True if a message was sent. Returns False (no-op) when the webhook
    env var is unset. Never raises — a failed post is logged, not fatal.
    """
    country = country_for_repo(repo_id)
    url = webhook or os.environ.get(CHANNEL_ENV[country])
    if not url:
        return False

    label = "🇬🇧 UK" if country == "uk" else "🇺🇸 US"
    context = " · ".join(
        part
        for part in (
            repo_id,
            f"published {updated_at}" if updated_at else "",
            f"<{DASHBOARD_URL}|calibration diagnostics>",
        )
        if part
    )
    payload = {
        "text": f"New Populace {country.upper()} release: {release_id}",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f":rocket: *New Populace {label} release*\n`{release_id}`",
                },
            },
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": context}],
            },
        ],
    }
    try:
        (post or _default_post)(url, payload)
        return True
    except Exception as exc:
        print(f"::warning:: Populace Slack notification failed: {exc}")
        return False
