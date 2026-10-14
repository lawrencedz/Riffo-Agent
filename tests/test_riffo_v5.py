"""Offline contract and policy tests. No network or credentials required."""
from types import SimpleNamespace as NS
from altruagent import WithReasoning,SendMessage,TERMINATE_MESSAGING,WAIT
from examples.llm.providers import ProviderError
from agent.agent import RiffoAgent, HybridPokemon,HybridWerewolf,HybridRedAlert,AIUtilitySelector

class FakeModel:
 model='gpt-4o-mini'
 def __init__(self,fail=False, red_orders=None):
  self.calls=[];self.fail=fail;self.red_orders=red_orders
 def complete_structured(self,messages,schema_name,schema,*,timeout=None):
  self.calls.append(schema_name)
  if self.fail:raise ProviderError('OpenAI request failed (HTTP 429)')
  if schema_name=='riffo_rank':
   # choose 0; ensure output is a legal candidate
   return {'index':0,'justification':'Based on expected utility.'}
  if schema_name=='riffo_chat':return {'message':'Can everyone explain their voting evidence?', 'send':True}
  if schema_name=='red_alert_orders':return {'orders':self.red_orders or []}
  raise AssertionError(schema_name)


def C(game,seat=0,config=None):return NS(game_type=game,seat_position=seat,game_config=config)
def A(id,template=None,label=None):return NS(action_id=str(id),input={'action':template} if template else {},label=label)

def S(actions, *, phase='moving',observation=None,game_state=None,seat=0, messages=None):
 players=[{'position':i,'name':f'Player{i}','agent_id':f'a{i}'} for i in range(7)]
 raw={'observation':observation or {},'your_position':seat,'players':players,
      'game_state':game_state or {'phase':'day_vote','day':1,'alive':[f'Player{i}' for i in range(7)],'dead':[]}}
 return NS(legal_actions=actions,phase=phase,raw=raw,observation=observation,
           state_version=5,new_messages=[NS(index=i,sender=s,content=txt) for i,(s,txt) in enumerate(messages or [])])

def test_ai_selector_chooses_valid():
 m=FakeModel()
 selector=AIUtilitySelector(m,max_calls=2,log=lambda *_:None)
 options=[{'action':'A','utility':11,'description':'A'}, {'action':'B','utility':9,'description':'B'}]
 answer,used=selector.rank('werewolf',options,{'evidence':'mixed'})
 assert answer['action']=='A' and used
 assert m.calls==['riffo_rank']

def test_selector_returns_local_after_429():
 m=FakeModel(fail=True);selector=AIUtilitySelector(m,max_calls=3,log=lambda *_:None)
 options=[{'action':'A','utility':1,'description':'A'},{'action':'B','utility':9,'description':'B'}]
 answer,used=selector.rank('pokemon',options,{})
 assert answer['action']=='B' and not used

def test_pokemon_draft_ai_and_validity():
 m=FakeModel();agent=RiffoAgent(provider=m,log=lambda *_:None)
 actions=[A('draft_pick:a',label='Magikarp'),A('draft_pick:b',label='Incineroar')]
 st=S(actions,phase='draft',observation={'available_cards':[{'card_id':'a','species':'Magikarp'},{'card_id':'b','species':'Incineroar'}]})
 out=agent.choose_action(st,C('pokemon_vgc_doubles_draft'))
 assert isinstance(out,WithReasoning) and out.action.action_id=='draft_pick:b'
 assert 'riffo_rank' in m.calls

def test_pokemon_lineup_legal():
 m=FakeModel();agent=RiffoAgent(provider=m,log=lambda *_:None)
 roster=['Magikarp','Incineroar','Rillaboom','Tornadus','Flutter Mane','Meowth']
 action=A('select_lineup',{'type':'select_lineup','roster':roster})
 out=agent.choose_action(S([action]),C('pokemon_vgc_doubles_draft'))
 assert out.action['type']=='select_lineup'
 assert len(out.action['bring'])==4 and len(set(out.action['bring']))==4
 assert set(out.action['leads'])<=set(out.action['bring'])
 assert 'riffo_rank' in m.calls

def test_pokemon_doubles_valid():
 m=FakeModel();agent=RiffoAgent(provider=m,log=lambda *_:None)
 slots=[{'slot':0,'options':[{'type':'move','move_id':'tailwind','targets':[]},{'type':'move','move_id':'tackle','targets':[1,-1]}]},
        {'slot':1,'options':[{'type':'move','move_id':'surgingstrikes','targets':[1,2,-2]}]}]
 out=agent.choose_action(S([A('doubles_turn',{'type':'doubles_turn','slots':slots})]),C('pokemon_vgc_doubles_draft'))
 assert out.action['type']=='doubles_turn'
 assert out.action['slot_1'].get('target',0)>0

def test_werewolf_vote_uses_ai():
 m=FakeModel();agent=RiffoAgent(provider=m,log=lambda *_:None)
 st=S([A(i) for i in (1,2,3,4,5,6,7)],observation='You are Player0. Your role: villager.')
 ans=agent.choose_action(st,C('werewolf'))
 assert isinstance(ans,WithReasoning) and ans.action.action_id!='7'
 assert 'riffo_rank' in m.calls

def test_werewolf_seer_respects_verified_wolf():
 m=FakeModel();agent=RiffoAgent(provider=m,log=lambda *_:None)
 st=S([A(i) for i in (0,1,2,3,4,5,6,7) if i!=2],observation='You are Player2. Your role: seer. You investigated Player4 and found wolf.',seat=2)
 ans=agent.choose_action(st,C('werewolf',2))
 assert ans.action.action_id=='4'

def test_werewolf_wolf_does_not_vote_mate():
 m=FakeModel();agent=RiffoAgent(provider=m,log=lambda *_:None)
 st=S([A(i) for i in (2,4,5,7)],observation='You are Player1. Your role: wolf. Your wolf teammate is Player4.',seat=1)
 ans=agent.choose_action(st,C('werewolf',1))
 assert ans.action.action_id in {'2','5'}

def test_werewolf_message_ai_and_no_spam():
 m=FakeModel();agent=RiffoAgent(provider=m,log=lambda *_:None)
 st=S([A(1)],observation='You are Player0. Your role: villager.')
 msg=agent.choose_message(st,C('werewolf'))
 assert isinstance(msg,SendMessage) and 'evidence' in msg.content
 again=agent.choose_message(st,C('werewolf'))
 assert again is TERMINATE_MESSAGING
 assert m.calls.count('riffo_chat')==1

def test_werewolf_falls_back_without_api():
 m=FakeModel(fail=True);agent=RiffoAgent(provider=m,log=lambda *_:None)
 st=S([A(i) for i in (1,2,3,7)],observation='You are Player0. Your role: villager.')
 ans=agent.choose_action(st,C('werewolf'))
 assert isinstance(ans,WithReasoning) and ans.action.action_id in {'1','2','3'}

def test_red_alert_deploy_without_ai():
 m=FakeModel();agent=RiffoAgent(provider=m,log=lambda *_:None)
 st=S([],observation={'units':[{'id':9,'type':'mcv'}], 'buildings':[], 'production':[], 'available_production':[]})
 ans=agent.choose_action(st,C('red_alert'))
 assert isinstance(ans,WithReasoning) and ans.action['orders'][0]=={'cmd':'deploy','units':[9]}
 assert 'red_alert_orders' not in m.calls

def test_red_alert_chooses_ai_when_util_positive():
 m=FakeModel(red_orders=[{'cmd':'attack_move','units':[1,2,3,4,5,6], 'to':[40,50]}])
 agent=RiffoAgent(provider=m,log=lambda *_:None)
 st=S([],observation={'units':[{'id':i,'type':'e1'} for i in range(1,7)], 'buildings':[{'id':9,'type':'proc'}], 'production':[], 'available_production':[]})
 agent.red_alert._last_llm-=100
 ans=agent.choose_action(st,C('red_alert',config={}))
 assert isinstance(ans,WithReasoning) and ans.action['orders'][0]['cmd']=='attack_move'
 assert m.calls.count('red_alert_orders')==1

def test_no_red_alert_invalid_spam_when_no_actions():
 m=FakeModel();agent=RiffoAgent(provider=m,log=lambda *_:None)
 st=S([],observation={'units':[],'buildings':[],'production':[],'available_production':[]})
 assert agent.choose_action(st,C('red_alert')) is WAIT

def test_utility_scores_army_and_power():
 m=FakeModel();agent=RiffoAgent(provider=m,log=lambda *_:None)
 raw={'observation':{'economy':{'power_provided':0,'power_drained':30, 'cash':2500},'buildings':[],'units':[]}}
 scorer=agent.red_alert._order_utility
 assert scorer({'cmd':'build','item':'powr'},raw)>scorer({'cmd':'build','item':'barr'},raw)
 assert scorer({'cmd':'deploy','units':[1]},raw)>scorer({'cmd':'build','item':'powr'},raw)

def test_create_agent_factory_with_fake_provider(monkeypatch):
 import agent.agent as mod
 monkeypatch.setattr(mod,'provider_from_env',lambda:FakeModel())
 assert isinstance(mod.create_agent(),RiffoAgent)


def test_model_can_override_close_local_utility():
    class Model(FakeModel):
        def complete_structured(self, messages, schema_name, schema, *, timeout=None):
            self.calls.append(schema_name)
            return {'index':1,'justification':'Future positioning offsets slight local difference.'}
    m=Model();selector=AIUtilitySelector(m,max_calls=4,log=lambda *_:None)
    options=[{'action':'conservative','utility':15,'description':'Hold'},
             {'action':'adaptive','utility':12,'description':'Position for next round'}]
    chosen,used=selector.rank('pokemon',options,{'enemy_strategy':'unknown'})
    assert chosen['action']=='adaptive' and used


def test_werewolf_discussion_claim_not_treated_as_private_seer_fact():
    m=FakeModel();agent=RiffoAgent(provider=m,log=lambda *_:None)
    st=S([A(i) for i in (1,2,3,4,5,6,7)],
         observation='You are Player0. Your role: seer.',
         messages=[(3,'I investigated Player4 and found a wolf!')])
    answer=agent.choose_action(st,C('werewolf'))
    assert isinstance(answer,WithReasoning)
    assert agent.werewolf.local.seer_results=={}


def test_red_alert_model_can_be_rejected_by_utility_score():
    m=FakeModel(red_orders=[{'cmd':'attack_move','units':[2], 'to':[2,5]}])
    agent=RiffoAgent(provider=m,log=lambda *_:None)
    st=S([],observation={'units':[{'id':2,'type':'e1'}],
        'buildings':[{'type':'fact'}], 'production':[],
        'available_production':['powr','tent'],
        'economy':{'power_provided':0,'power_drained':50,'cash':2500}})
    agent.red_alert._last_llm-=100
    answer=agent.choose_action(st,C('red_alert',config={}))
    assert isinstance(answer,WithReasoning)
    assert answer.action['orders'][0]['cmd']=='build'
    assert answer.action['orders'][0]['item']=='powr'
