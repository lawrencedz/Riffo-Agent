"""Offline policy tests: no Groq key or UCLA server needed."""
from types import SimpleNamespace
from agent.agent import PokemonCoach, GroqCircuitBreaker, RiffoAgent, FastRedAlert
from examples.llm.providers import ProviderError


def A(action_id, payload=None, label=None):
    return SimpleNamespace(action_id=action_id, input={"action":payload} if payload else {}, label=label)

def S(actions, phase="moving", observation=None):
    return SimpleNamespace(legal_actions=actions, phase=phase, raw={"observation": observation or {}})

def C(game_type="pokemon_vgc_doubles_draft"):
    return SimpleNamespace(game_type=game_type)

def test_draft_prefers_strong_pokemon():
    coach=PokemonCoach(log=lambda *_:None)
    pool=[{"card_id":"x", "species":"Magikarp"},{"card_id":"y", "species":"Incineroar"}]
    act=coach.choose_action(S([A("draft_pick:x"), A("draft_pick:y")],"draft",{"available_cards":pool}),C())
    assert act.action.action_id == "draft_pick:y"

def test_team_preview_chooses_four_distinct_species():
    coach=PokemonCoach(log=lambda *_:None)
    roster=["Magikarp", "Tornadus", "Incineroar", "Rillaboom", "Flutter Mane", "Meowth"]
    template={"type":"select_lineup","roster":roster}
    output=coach.choose_action(S([A("select_lineup", template)]),C()).action
    assert output["type"]=="select_lineup"
    assert len(output["bring"])==4 and len(set(output["bring"]))==4
    assert "Tornadus" in output["leads"]
    assert set(output["leads"]) <= set(output["bring"])

def test_doubles_prefers_tailwind_and_enemy_target():
    coach=PokemonCoach(log=lambda *_:None)
    slots=[
        {"slot":0,"options":[{"type":"move","move_id":"tailwind","targets":[]},{"type":"move","move_id":"tackle","targets":[1,-2]}]},
        {"slot":1,"options":[{"type":"move","move_id":"surgingstrikes","targets":[1,2,-1]}]},
    ]
    result=coach.choose_action(S([A("doubles_turn",{"type":"doubles_turn","slots":slots})]),C()).action
    assert result["slot_0"]["move_id"]=="tailwind"
    assert result["slot_1"]["target"] in (1,2)
    assert result["slot_1"]["target"]>0

def test_tailwind_is_not_spammed():
    coach=PokemonCoach(log=lambda *_:None)
    slots=[{"slot":i,"options":[{"type":"move","move_id":"tailwind","targets":[]},{"type":"move","move_id":"surgingstrikes","targets":[1]}]} for i in (0,1)]
    state=S([A("doubles_turn",{"type":"doubles_turn","slots":slots})])
    coach.choose_action(state,C())
    second=coach.choose_action(state,C()).action
    assert all(second[f"slot_{i}"]["move_id"]!="tailwind" for i in (0,1))

def test_forced_switch_uses_distinct_pokemon():
    coach=PokemonCoach(log=lambda *_:None)
    slots=[{"slot":i,"force_switch":True,"options":[{"type":"switch","species":"Incineroar"},{"type":"switch","species":"Tornadus"}]} for i in (0,1)]
    ans=coach.choose_action(S([A("doubles_turn",{"type":"doubles_turn","slots":slots})]),C()).action
    assert ans["slot_0"]["species"] != ans["slot_1"]["species"]

def test_pokemon_never_calls_model(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only-no-network')
    agent=RiffoAgent()
    agent._provider.complete_structured=lambda *a,**k: (_ for _ in ()).throw(AssertionError("Unexpected Groq call"))
    ans=agent.choose_action(S([A("draft_pick:a", label="Tornadus")],"draft"),C())
    assert ans.action.action_id=="draft_pick:a"

def test_circuit_opens_on_429_and_prevents_retries():
    class Reject:
        model='test'
        n=0
        def complete_structured(self,*args,**kwargs):
            self.n+=1
            raise ProviderError("OpenAI request failed (HTTP 429)")
    provider=Reject()
    cb=GroqCircuitBreaker(provider,clock=lambda:100)
    try: cb.complete_structured([],"message",{})
    except ProviderError: pass
    assert cb.blocked
    try: cb.complete_structured([],"message",{})
    except ProviderError: pass
    assert provider.n == 1

def test_circuit_stays_open_for_413():
    class Reject:
        model='test'
        def complete_structured(self,*args,**kwargs):
            raise ProviderError("OpenAI request failed (HTTP 413)")
    cb=GroqCircuitBreaker(Reject())
    try: cb.complete_structured([],"message",{})
    except ProviderError: pass
    assert cb.oversize

def test_redalert_llm_is_capped():
    assert FastRedAlert.MAX_LLM_CALLS <= 4
    assert FastRedAlert.LLM_INTERVAL >= 120

def test_targeting_favors_super_effective_enemy():
    coach=PokemonCoach(log=lambda *_:None)
    move={"type":"move","move_id":"thunderbolt","targets":[1,2],
          "target_options":[{"target":1,"side":"opponent","species":"Groudon"},
                            {"target":2,"side":"opponent","species":"Pelipper"}]}
    slots=[{"slot":0,"options":[move]},
           {"slot":1,"options":[{"type":"move","move_id":"tackle","targets":[1]}]}]
    ans=coach.choose_action(S([A("doubles_turn",{"type":"doubles_turn","slots":slots})]),C()).action
    assert ans["slot_0"]["target"] == 2

def test_two_slots_do_not_both_set_tailwind():
    coach=PokemonCoach(log=lambda *_:None)
    slots=[{"slot":i,"options":[{"type":"move","move_id":"tailwind","targets":[]},
                                {"type":"move","move_id":"surgingstrikes","targets":[1]}]} for i in (0,1)]
    ans=coach.choose_action(S([A("doubles_turn",{"type":"doubles_turn","slots":slots})]),C()).action
    assert sum(ans[f"slot_{i}"].get("move_id")=="tailwind" for i in (0,1)) <= 1

