"""Safety regression tests against simulated UCLA interfaces (no live API)."""
from altruagent import SendMessage, WithReasoning, TERMINATE_MESSAGING
from agent.agent import RiffoAgent, WerewolfCoach
from test_riffo_v5 import FakeModel, S, A, C


class HallucinatingModel(FakeModel):
    """A model that would fabricate damaging chat if unrestricted."""
    def complete_structured(self, messages, schema_name, schema, *, timeout=None):
        self.calls.append(schema_name)
        if schema_name == 'riffo_rank':
            return {'index': 0, 'justification': 'Use verifiable sources.'}
        if schema_name == 'riffo_chat':
            return {'message': 'I saw Player5 kill someone. Vote Player0 and Player1!', 'send': True}
        raise AssertionError(schema_name)


def test_chat_uses_safe_candidates_not_freeform_hallucinations():
    model = HallucinatingModel()
    a = RiffoAgent(provider=model, log=lambda *_: None)
    s = S([A(i) for i in (1,2,3,4,5,6)], observation='You are Player0. Your role: villager.', seat=0)
    # The local evidence request is the first message; the AI may speak second.
    m1 = a.choose_message(s, C('werewolf', 0))
    m2 = a.choose_message(s, C('werewolf', 0))
    assert isinstance(m1, SendMessage)
    assert m2 is TERMINATE_MESSAGING
    assert 'I saw Player5' not in m1.content
    assert 'Vote Player0' not in m1.content
    assert 'riffo_chat' not in model.calls
    assert 'riffo_rank' in model.calls
    assert a.choose_message(s, C('werewolf', 0)) is TERMINATE_MESSAGING


def test_seer_cleared_player_no_self_vote_public_message():
    a = RiffoAgent(provider=HallucinatingModel(), log=lambda *_: None)
    s = S([A(i) for i in (0,2,3,4,5,6)], observation='You are Player1. Your role: seer. You investigated Player0 and found villager.', seat=1)
    messages = [a.choose_message(s, C('werewolf', 1)), a.choose_message(s, C('werewolf', 1))]
    assert all(isinstance(m, SendMessage) for m in messages)
    assert all('Vote Player1' not in m.content and 'vote Player0' not in m.content for m in messages)
    assert a.werewolf.local.seer_results[0] == 'villager'


def test_verified_wolf_seer_announces_factual_target():
    a = RiffoAgent(provider=HallucinatingModel(), log=lambda *_: None)
    s = S([A(i) for i in (0,2,3,4,5,6)], observation='You are Player1. Your role: seer. You investigated Player4 and found wolf.', seat=1)
    m = a.choose_message(s, C('werewolf', 1))
    assert isinstance(m, SendMessage) and 'Player4' in m.content and 'wolf' in m.content


def test_contradictory_message_not_counted_as_accusation_of_cleared_player():
    p = WerewolfCoach(log=lambda *_: None)
    p.messages = [(4, 'Player0 is a confirmed villager. Please vote Player1.')]
    assert p._public_suspicion(0, {}) < 0
    assert p._public_suspicion(1, {}) > 0


def test_self_vote_not_selected_even_if_legal():
    a = RiffoAgent(provider=HallucinatingModel(), log=lambda *_: None)
    s = S([A(i) for i in range(7)], observation='You are Player1. Your role: villager.', seat=1)
    decision = a.choose_action(s, C('werewolf', 1))
    assert isinstance(decision, WithReasoning)
    assert decision.action.action_id != '1'


def test_wolf_mate_not_targeted_at_night():
    a = RiffoAgent(provider=HallucinatingModel(), log=lambda *_: None)
    s = S([A(i) for i in (0,2,3,4,5,6)], observation='You are Player1. Your role: wolf. Your wolf teammate is Player4.',
          game_state={'phase': 'night_wolf', 'day': 1, 'alive': [f'Player{i}' for i in range(7)], 'dead': []}, seat=1)
    decision = a.choose_action(s, C('werewolf', 1))
    assert decision.action.action_id != '4'
