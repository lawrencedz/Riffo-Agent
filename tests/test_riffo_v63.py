"""New VGC turn-memory checks for Riffo v6.3 (against offline SDK contracts)."""
from types import SimpleNamespace as NS
from agent.agent import RiffoAgent
from altruagent import WithReasoning
from examples.llm.providers import ProviderError


class Model:
    model = 'gpt-4.1-mini'
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []
    def complete_structured(self, messages, schema_name, schema, *, timeout=None):
        self.calls.append((messages, schema_name, timeout))
        result = self.answers.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def state(active=('Incineroar', 'Rillaboom')):
    options = [
        {'slot': 0, 'active': {'species': active[0]}, 'options': [
            {'type': 'move', 'move_id': 'Flare Blitz', 'targets': [1]},
            {'type': 'move', 'move_id': 'Protect', 'targets': []},
            {'type': 'move', 'move_id': 'Fake Out', 'targets': [1]},
        ]},
        {'slot': 1, 'active': {'species': active[1]}, 'options': [
            {'type': 'move', 'move_id': 'Fake Out', 'targets': [2]},
            {'type': 'move', 'move_id': 'Wood Hammer', 'targets': [2]},
            {'type': 'move', 'move_id': 'Protect', 'targets': []},
        ]}
    ]
    return NS(legal_actions=[NS(action_id='doubles_turn', label='',
                                input={'action': {'type': 'doubles_turn', 'slots': options}})],
              raw={'observation': {'battle': 'observed matchup'}},
              new_messages=[], phase='moving', state_version=1)


def ctx():
    return NS(game_type='pokemon_vgc_doubles_draft', seat_position=0, game_config={})


def ans(slot0=0, slot1=0):
    target0 = 1 if slot0 in (0, 2) else None
    target1 = 2 if slot1 in (0, 1) else None
    return {'slot_0': {'option': slot0, 'target': target0},
            'slot_1': {'option': slot1, 'target': target1},
            'reasoning_summary': 'consider the board'}


def chosen(out):
    return out.action if isinstance(out, WithReasoning) else out


def test_fakout_wasted_after_first_turn_is_model_corrected():
    # First turn Rillaboom uses Fake Out; next observed turn it has stayed in.
    model = Model([ans(0, 0), ans(0, 0), ans(0, 1)])
    a = RiffoAgent(provider=model, log=lambda *args: None)
    a._opening_review_done = True
    first = chosen(a.choose_action(state(), ctx()))
    second = chosen(a.choose_action(state(), ctx()))
    assert first['slot_1']['move_id'] == 'Fake Out'
    assert second['slot_1']['move_id'] == 'Wood Hammer'
    assert len(model.calls) == 3
    assert 'Fake Out will fail' in model.calls[2][0][1]['content']
    assert 'battle_mechanics' in model.calls[1][0][1]['content']


def test_consecutive_protect_warning_triggers_revision():
    model = Model([ans(1, 1), ans(1, 1), ans(0, 1)])
    a = RiffoAgent(provider=model, log=lambda *args: None)
    a._opening_review_done = True
    first = chosen(a.choose_action(state(), ctx()))
    second = chosen(a.choose_action(state(), ctx()))
    assert first['slot_0']['move_id'] == 'Protect'
    assert second['slot_0']['move_id'] == 'Flare Blitz'
    assert 'consecutive Protect' in model.calls[2][0][1]['content']


def test_switch_resets_first_turn_fakeout_eligibility():
    model = Model([ans(0, 1), ans(2, 1)])
    a = RiffoAgent(provider=model, log=lambda *args: None)
    a._opening_review_done = True
    a.choose_action(state(), ctx())
    response = chosen(a.choose_action(state(active=('Iron Hands', 'Rillaboom')), ctx()))
    assert response['slot_0']['move_id'] == 'Fake Out'
    assert len(model.calls) == 2, 'A newly observed active Pokemon should not trigger correction'


def test_model_failure_during_correction_keeps_valid_original():
    model = Model([ans(0, 1), ans(0, 0), ProviderError('offline')])
    a = RiffoAgent(provider=model, log=lambda *args: None)
    a._opening_review_done = True
    a.choose_action(state(), ctx())
    response = chosen(a.choose_action(state(), ctx()))
    assert response['slot_1']['move_id'] == 'Fake Out'  # legal but strategically poor
    assert len(model.calls) == 3


def test_existing_werewolf_and_rts_paths_unchanged():
    a = RiffoAgent(provider=Model([]), log=lambda *args: None)
    assert a.wolf_teammates == set()
    assert a._realtime_players.get('red_alert') is not None
