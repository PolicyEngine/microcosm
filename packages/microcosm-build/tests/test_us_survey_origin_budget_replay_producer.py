"""Cheap loaded-code producer refusals; no financial/source fixture request."""

import pytest

from microcosm.build.us_runtime import survey_origin_budget as owner


@pytest.mark.parametrize(
    "name", ("same_replayed_population", "same_replayed_frame", "_series")
)
def test_budget_producer_refuses_replaced_replay_dependency(monkeypatch, name):
    # The real producer first accepts its own source/module baseline. Replacing
    # a function with behavior-preserving delegation must still invalidate that
    # baseline; this checks producer identity, without inventing a budget issuer.
    owner._producer()
    replay = owner.geography.replay
    original = getattr(replay, name)

    def changed(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(replay, name, changed)
    assert getattr(replay, name) is not original
    with pytest.raises(owner.SurveyOriginBudgetError, match="^PRODUCER_CHANGED$"):
        owner._producer()
