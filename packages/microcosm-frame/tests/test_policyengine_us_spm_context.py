"""SPM configuration reaches metadata without importing a country engine."""

from types import SimpleNamespace

import pytest

from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine


@pytest.mark.parametrize("selection", [None, {}, {"geography_kind": "national"}])
def test_metadata_system_uses_captured_spm_selection_once(monkeypatch, selection):
    calls = []
    expected = None if selection is None else dict(selection)
    system = SimpleNamespace(
        variables={
            "source_input": SimpleNamespace(
                entity=SimpleNamespace(key="person"),
                value_type=float,
                definition_period="year",
            )
        }
    )

    def constructor(**kwargs):
        calls.append(kwargs)
        return system

    engine = PolicyEngineUSEngine(spm=selection)
    if selection is not None:
        selection["geography_kind"] = "changed_by_caller"
    monkeypatch.setattr(
        engine,
        "_import_policyengine_us",
        lambda: SimpleNamespace(CountryTaxBenefitSystem=constructor),
    )
    assert engine.variable_metadata("source_input").entity == "person"
    assert engine.variable_metadata("source_input").dtype == "float"
    assert len(calls) == 1
    assert calls[0] == ({} if expected is None else {"spm": expected})
    if expected is not None:
        assert calls[0]["spm"] is not engine._spm
        calls[0]["spm"]["geography_kind"] = "changed_by_constructor"
        assert engine._spm == expected


def test_failed_system_constructor_cannot_mutate_next_spm_attempt(monkeypatch):
    calls = []
    system = object()

    def constructor(*, spm):
        calls.append(dict(spm))
        spm["geography_kind"] = "changed_by_constructor"
        if len(calls) == 1:
            raise ValueError("invented constructor failure")
        return system

    engine = PolicyEngineUSEngine(spm={"geography_kind": "national"})
    monkeypatch.setattr(
        engine,
        "_import_policyengine_us",
        lambda: SimpleNamespace(CountryTaxBenefitSystem=constructor),
    )
    with pytest.raises(ValueError, match="invented constructor failure"):
        engine._tax_benefit_system()
    assert engine._tax_benefit_system() is system
    assert calls == [{"geography_kind": "national"}] * 2
    assert engine._spm == {"geography_kind": "national"}
