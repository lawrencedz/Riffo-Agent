"""New matchup and Red Alert regressions, no network or secrets required."""
from agent.agent import MatchupPokemonCoach, MatchupHybridPokemon, FastRedAlert, _type_multiplier
from types import SimpleNamespace as NS
from altruagent import WithReasoning


class FakeModel:
    model='gpt-4o-mini'
    def __init__(self): self.calls=0
    def complete_structured(self, messages, schema_name, schema, *, timeout=None):
        self.calls += 1
        return {'index':0, 'justification':'best grounded alternative'}


def A(id,template=None): return NS(action_id=id,input={'action':template} if template else {},label=None)
def S(actions,observation=None,phase='moving'):
    return NS(legal_actions=actions,raw={'observation': observation or {}},phase=phase)


def test_full_type_effectiveness_handles_immunity_and_resistance():
    assert _type_multiplier('electric','Groudon') == 0
    assert _type_multiplier('electric','Pelipper') == 4
    assert _type_multiplier('grass','Heatran') == .25
    assert _type_multiplier('dragon','Flutter Mane') == 0
    assert _type_multiplier('water','Groudon') == 2


def test_pokemon_prefers_pelipper_over_immune_groudon():
    coach=MatchupPokemonCoach(log=lambda *args: None)
    coach.active_species={0:'Miraidon'}
    option={'type':'move','move_id':'thunderbolt', 'targets':[1,2],
            'target_options':[{'target':1,'side':'opponent','species':'Groudon'},
                              {'target':2,'side':'opponent','species':'Pelipper'}]}
    assert coach._score_move(option,0,2,None) > coach._score_move(option,0,1,None)+100


def test_doubles_candidates_target_effective_opponent_and_use_ai():
    model=FakeModel(); hybrid=MatchupHybridPokemon(model,log=lambda *_:None)
    template={'type':'doubles_turn', 'slots':[
      {'slot':0,'active':{'species':'Miraidon'},'options':[
           {'type':'move','move_id':'thunderbolt','targets':[1,2],
            'target_options':[{'target':1,'side':'opponent','species':'Groudon'},
                              {'target':2,'side':'opponent','species':'Pelipper'}]}]},
      {'slot':1,'active':{'species':'Incineroar'},'options':[
            {'type':'move','move_id':'fakeout','targets':[1,2]}]}]}
    # One initial select from candidates; the first turn may use local policy.
    ans=hybrid.choose_action(S([A('doubles_turn',template)]), NS(game_type='pokemon_vgc_doubles_draft'))
    assert isinstance(ans,WithReasoning)
    assert ans.action['slot_0']['target']==2


def test_redalert_cannot_place_infantry_as_building():
    commander=FastRedAlert(FakeModel(), log=lambda *args:None,clock=lambda:1000)
    raw={'observation':{'units':[], 'buildings':[{'type':'proc','id':7}],
      'production':[{'item':'e1','ready':True}],
      'available_production':['e1'], 'economy':{'cash':3500}}, 'legal_actions':{}}
    decision=commander._heuristic(raw)
    assert decision is not None
    assert decision[0]['cmd']=='train'


def test_redalert_attack_cooldown_key_independent_of_unit_composition():
    one={'cmd':'attack_move','units':[1,2],'to':[50,52]}
    two={'cmd':'attack_move','units':[3,4,5],'to':[49,51]}
    assert FastRedAlert._order_key(one)==FastRedAlert._order_key(two)
