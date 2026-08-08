from microcosm.data.slack import country_for_repo, notify_release


def test_country_inferred_from_repo_id() -> None:
    assert country_for_repo("policyengine/populace-us") == "us"
    assert country_for_repo("policyengine/populace-uk-private") == "uk"


def test_notify_is_noop_without_webhook(monkeypatch) -> None:
    monkeypatch.delenv("SLACK_WEBHOOK_POPULACE_US", raising=False)
    sent: list = []
    result = notify_release(
        "policyengine/populace-us",
        "populace-us-2024-abc-20260620T000000Z",
        post=lambda url, payload: sent.append((url, payload)),
    )
    assert result is False
    assert sent == []


def test_notify_posts_to_country_webhook(monkeypatch) -> None:
    monkeypatch.setenv("SLACK_WEBHOOK_POPULACE_UK", "https://hooks.slack.test/uk")
    sent: list = []
    result = notify_release(
        "policyengine/populace-uk-private",
        "populace-uk-2023-def-20260620T000000Z",
        "2026-06-20T00:00:00+00:00",
        post=lambda url, payload: sent.append((url, payload)),
    )
    assert result is True
    assert len(sent) == 1
    url, payload = sent[0]
    assert url == "https://hooks.slack.test/uk"
    assert "populace-uk-2023-def" in payload["text"]
    assert "UK" in payload["blocks"][0]["text"]["text"]


def test_notify_warns_when_unset_and_requested(monkeypatch, capsys) -> None:
    monkeypatch.delenv("SLACK_WEBHOOK_POPULACE_US", raising=False)
    result = notify_release(
        "policyengine/populace-us",
        "populace-us-2024-abc-20260620T000000Z",
        warn_if_unset=True,
    )
    assert result is False
    captured = capsys.readouterr()
    assert "SLACK_WEBHOOK_POPULACE_US is not set" in captured.out + captured.err


def test_notify_silent_when_unset_and_not_requested(monkeypatch, capsys) -> None:
    monkeypatch.delenv("SLACK_WEBHOOK_POPULACE_US", raising=False)
    result = notify_release(
        "policyengine/populace-us",
        "populace-us-2024-abc-20260620T000000Z",
    )
    assert result is False
    captured = capsys.readouterr()
    assert "SLACK_WEBHOOK" not in captured.out + captured.err


def test_notify_never_raises_on_post_failure() -> None:
    def boom(url, payload):
        raise RuntimeError("slack down")

    result = notify_release(
        "policyengine/populace-us",
        "populace-us-2024-abc-20260620T000000Z",
        webhook="https://hooks.slack.test/us",
        post=boom,
    )
    assert result is False
