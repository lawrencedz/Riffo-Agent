"""Riffo v6.1: only tactical model review and Showdown-reference grounding."""
from types import SimpleNamespace as NS
from agent.agent import RiffoAgent, PokemonReference, _type_multiplier
from altruagent import WithReasoning
from examples.llm.providers import ProviderError


def action(i, **kwargs):return NS(action_id=str(i), label=str(i),input=kwargs)
def ctx():return NS(game_type='pokemon_vgc_doubles_draft',seat_position=0,game_config={})
def state(t,obs=None,phase='moving'):
    return NS(legal_actions=[action(t,action=obs['template'])],raw={'observation':obs.get('observation',{})},phase=phase,
              state_version=1,new_messages=[])
def selected(ans):return ans.action if isinstance(ans, WithReasoning) else ans

class Model:
    model='gpt-4.1-mini'
    def __init__(self,answers):self.answers=list(answers);self.calls=[]
    def complete_structured(self,messages,schema_name,schema,*,timeout=None):
        self.calls.append((messages,schema_name,timeout))
        if not self.answers:raise ProviderError('out of responses')
        ans=self.answers.pop(0)
        if isinstance(ans,Exception):raise ans
        return ans


def battle_state():
    return state('doubles_turn', {'observation': {'opponents':[{'species':'Dragonite'}]},
        'template':{'type':'doubles_turn','slots':[
         {'slot':0,'options':[{'type':'move','move_id':'Wood Hammer','targets':[1]},{'type':'move','move_id':'Protect','targets':[]}]},
         {'slot':1,'options':[{'type':'move','move_id':'Ice Beam','targets':[1]},{'type':'pass'}]},
        ]}}, phase='moving')

def test_type_chart_complete_and_dragonite_quarter_grass():
    assert len(__import__('agent.agent',fromlist=['_TYPE_CHART'])._TYPE_CHART)==18
    assert _type_multiplier('Grass',['Dragon','Flying'])==.25
    assert _type_multiplier('Ice',['Dragon','Flying'])==4
    assert _type_multiplier('Ground',['Flying'])==0

def test_reference_metadata_and_move_multiplier_without_static_species_list():
    r=PokemonReference();r.species={'dragonite':{'types':['Dragon','Flying'],'base_speed':80}}
    r.moves={'woodhammer':{'type':'Grass','power':120}}
    prompt={'slots':[{'options':[{'type':'move','move_id':'Wood Hammer','targets':[{'target':1,'is':'OPPONENT Dragonite'}]}]}]}
    b=r.brief({},prompt)
    assert b['species']['dragonite']['types']==['Dragon','Flying']
    assert b['visible_move_matchups_BASE_TYPES_ONLY'][0]['type_multiplier']==0.25


def test_opening_turn_independent_review_can_choose_protect():
    original={'slot_0':{'option':0,'target':1},'slot_1':{'option':0,'target':1},'reasoning_summary':'Go offensive'}
    revised={'slot_0':{'option':1,'target':None},'slot_1':{'option':0,'target':1},'reasoning_summary':'Protect and attack'}
    model=Model([original,revised]);a=RiffoAgent(provider=model,log=lambda *_:None)
    out=selected(a.choose_action(battle_state(),ctx()))
    assert out['slot_0']['move_id']=='Protect'
    assert len(model.calls)==2
    assert 'independent VGC tactical reviewer' in model.calls[1][0][0]['content']


def test_second_turn_does_not_pay_for_repeated_review():
    original={'slot_0':{'option':0,'target':1},'slot_1':{'option':0,'target':1},'reasoning_summary':'Attack'}
    model=Model([original,original,original]);a=RiffoAgent(provider=model,log=lambda *_:None)
    a.choose_action(battle_state(),ctx());a.choose_action(battle_state(),ctx())
    assert len(model.calls)==3


def test_review_provider_error_keeps_original_legal_action():
    original={'slot_0':{'option':0,'target':1},'slot_1':{'option':0,'target':1},'reasoning_summary':'Attack'}
    model=Model([original,ProviderError('temporary model failure')]);a=RiffoAgent(provider=model,log=lambda *_:None)
    out=selected(a.choose_action(battle_state(),ctx()))
    assert out['slot_0']['move_id']=='Wood Hammer' and out['slot_1']['target']==1


def test_preview_second_look_lead_can_be_revised():
    template={'type':'select_lineup','roster':['Milotic','Thundurus','Rillaboom','Baxcalibur','Grimmsnarl','Whimsicott']}
    original={'bring':['Milotic','Thundurus','Rillaboom','Baxcalibur'],'leads':['Milotic','Thundurus'],'reasoning_summary':'Attack'}
    revised={'bring':['Milotic','Thundurus','Rillaboom','Baxcalibur'],'leads':['Rillaboom','Milotic'],'reasoning_summary':'Safer lead'}
    model=Model([original,revised]);a=RiffoAgent(provider=model,log=lambda *_:None)
    s=state('select_lineup',{'template':template,'observation':{'opponent_roster':['Dragonite','Iron Hands']}},phase='team_preview')
    result=selected(a.choose_action(s,ctx()))
    assert result['leads']==['Rillaboom','Milotic'] and len(model.calls)==2
