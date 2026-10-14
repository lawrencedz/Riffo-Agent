"""Safe offline strategy checks for Riffo v1. No API calls or actual matches."""

from types import SimpleNamespace

from agent.agent import FastRedAlert, StrategicProvider


class FakeProvider:
    model = "test-model"

    def __init__(self):
        self.messages = None

    def complete_structured(self, messages, schema_name, schema, *, timeout=None):
        self.messages = messages
        return {"ok": True}


def raw(*, units=None, buildings=None, production=None, available=None, attack_now=None, economy=None):
    return {
        "observation": {
            "units": units or [],
            "buildings": buildings or [],
            "production": production or [],
            "available_production": available or [],
            "economy": economy or {},
        },
        "legal_actions": {"attack_now": attack_now},
    }


def commander():
    return FastRedAlert(FakeProvider(), log=lambda *_: None)


def test_deploy_before_build():
    agent = commander()
    choice = agent._heuristic(raw(units=[{"type": "mcv", "id": 44}], available=["powr"]))
    assert choice[0] == {"cmd": "deploy", "units": [44]}


def test_place_ready_structure():
    agent = commander()
    choice = agent._heuristic(raw(production=[{"item": "powr", "ready": True}]))
    assert choice[0] == {"cmd": "place", "item": "powr"}


def test_build_base_power_first():
    agent = commander()
    choice = agent._heuristic(raw(available=["powr", "tent", "proc"]))
    assert choice[0] == {"cmd": "build", "item": "powr"}


def test_does_not_queue_duplicate_building():
    agent = commander()
    choice = agent._heuristic(raw(available=["powr", "tent"], production=[{"item": "powr", "ready": False}]))
    assert choice[0] == {"cmd": "build", "item": "tent"}


def test_uses_server_provided_attack_order():
    agent = commander()
    units = [{"type": "e1", "id": i} for i in range(5)]
    order = {"cmd": "attack_move", "units": [1, 2, 3, 4], "to": [50, 50]}
    choice = agent._heuristic(raw(units=units, attack_now=order))
    assert choice[0] == order


def test_strategy_prompt_preserves_schema_and_timeout():
    backend = FakeProvider()
    wrapped = StrategicProvider(backend)
    schema = {"type": "object"}
    got = wrapped.complete_structured(
        [{"role": "system", "content": "Original"}, {"role": "user", "content": '{"game_type": "pokemon_vgc_doubles_draft"}'}],
        "choose_action", schema, timeout=2.0
    )
    assert got == {"ok": True}
    assert "VGC DOUBLES" in backend.messages[0]["content"]
    assert backend.messages[0]["content"].startswith("Original")


def test_werewolf_instructions():
    backend = FakeProvider()
    wrapped = StrategicProvider(backend)
    wrapped.complete_structured(
        [{"role": "system", "content": "Original"}, {"role": "user", "content": '{"game_type": "werewolf"}'}],
        "choose_action", {}, timeout=None
    )
    assert "STRATEGY FOR WEREWOLF" in backend.messages[0]["content"]


def test_realtime_provider_caps_timeout():
    from agent.agent import ShortTimeoutProvider

    class Recorder(FakeProvider):
        def complete_structured(self, messages, schema_name, schema, *, timeout=None):
            self.timeout = timeout
            return super().complete_structured(messages, schema_name, schema, timeout=timeout)

    backend = Recorder()
    wrapped = ShortTimeoutProvider(backend)
    wrapped.complete_structured([], "red_alert_orders", {})
    assert backend.timeout == 8.0
    wrapped.complete_structured([], "red_alert_orders", {}, timeout=3.0)
    assert backend.timeout == 3.0


def test_heuristics_take_priority_over_model_calls():
    # Even if an LLM planning interval is due, do not block immediate base building.
    from agent.agent import FastRedAlert

    now = [100.0]
    backend = FakeProvider()
    agent = FastRedAlert(backend, clock=lambda: now[0], log=lambda *_: None)
    now[0] = 200.0
    state = SimpleNamespace(raw=raw(units=[{"type": "mcv", "id": 1}]))
    answer = agent.choose_action(state, SimpleNamespace(game_config=None, game_type="red_alert"))
    assert answer is not None
    assert agent._llm.calls == 0
    assert backend.messages is None


def test_redalert_stays_below_limited_model_budget():
    agent = commander()
    assert agent.MAX_LLM_CALLS <= 12
    assert agent.LLM_INTERVAL >= 45
