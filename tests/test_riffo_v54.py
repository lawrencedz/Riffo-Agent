"""Riffo v5.4 focused offline regression checks; no network calls or secrets."""
from types import SimpleNamespace as NS
from agent.agent import (
    RiffoAgent, TacticalPokemon, TacticalPokemonCoach,
    _MOVE_TYPES, _MOVE_POWER_ESTIMATE, _type_multiplier,
)

class FakeModel:
    model='gpt-4.1-mini'
    def __init__(self): self.calls=[]
    def complete_structured(self, messages, schema_name, schema, *, timeout=None):
        self.calls.append(schema_name)
        return {'index':0,'justification':'Best balanced risk-reward choice.'}

def lineup_state(*, opponents=()):
    roster=['Incineroar','Amoonguss','Rillaboom','Cresselia','Kingambit','Iron Bundle']
    action=NS(action_id='select_lineup',label='',input={'action':{'type':'select_lineup','roster':roster}})
    return NS(legal_actions=[action],raw={'observation':{'opponent_roster':list(opponents)}},phase='team_preview')

def test_pollen_puff_has_real_enemy_damage_metadata():
    assert _MOVE_TYPES['pollenpuff']=='bug'
    assert _MOVE_POWER_ESTIMATE['pollenpuff']==90
    assert _type_multiplier('bug','Chi-Yu')==1  # Bug vs Dark/Fire cancels out
    assert _type_multiplier('bug','Gholdengo')==.25

def test_dont_spore_gholdengo():
    c=TacticalPokemonCoach(log=lambda *_:None)
    c.current_target_species={1:'Gholdengo', 2:'Chi-Yu'}
    opt={'type':'move','move_id':'spore','targets':[1,2]}
    assert c._score_move(opt,0,2,None)>c._score_move(opt,0,1,None)+100

def test_draco_meteor_repeated_special_attack_penalized():
    c=TacticalPokemonCoach(log=lambda *_:None)
    c.active_species={0:'Archaludon'}
    opt={'type':'move','move_id':'dracometeor','targets':[1]}
    first=c._score_move(opt,0,1,None)
    c.last_moves[0]='dracometeor'
    second=c._score_move(opt,0,1,None)
    assert first-second>=60

def test_counter_preview_prioritizes_better_four_against_ryan():
    p=TacticalPokemon(FakeModel(),log=lambda *_:None);p._my_seat=0
    state=lineup_state(opponents=['Chi-Yu','Iron Hands','Gholdengo'])
    options=p._alternatives(state)
    assert options and len(options)<=9
    best=options[0]['action']
    assert 'Iron Bundle' in best['bring']
    assert 'Incineroar' in best['bring']
    assert 'Cresselia' not in best['bring']
    assert len({tuple(sorted(c['action']['bring'])) for c in options})>1


def test_no_opponent_info_still_legal_preview():
    p=TacticalPokemon(FakeModel(),log=lambda *_:None);p._my_seat=0
    st=lineup_state()
    result=p._alternatives(st)
    assert result
    assert all(len(c['action']['bring'])==4 and len(c['action']['leads'])==2 for c in result)


def test_live_entry_point_uses_tactical_coach_and_model():
    fake=FakeModel()
    agent=RiffoAgent(provider=fake,log=lambda *_:None)
    assert isinstance(agent.pokemon,TacticalPokemon)
    st=lineup_state(opponents=['Chi-Yu','Gholdengo','Iron Hands'])
    answer=agent.choose_action(st,NS(game_type='pokemon_vgc_doubles_draft',seat_position=0))
    assert answer.action['type']=='select_lineup'
    assert 'Iron Bundle' in answer.action['bring']
    assert fake.calls==['riffo_rank']
