"""Retained Pokemon and Werewolf regression tests from v6.0 (without obsolete RTS assertions)."""
"""Focused Riffo 6 offline tests. Run on the real starter or via offline SDK stubs."""
from types import SimpleNamespace as NS
from agent.agent import RiffoAgent, _role, _seat
from altruagent import WAIT, WithReasoning, SendMessage, TERMINATE_MESSAGING
from examples.llm.providers import ProviderError


def A(i, **kwargs):return NS(action_id=str(i), label=str(i),input=kwargs)
def C(kind,seat=0):return NS(game_type=kind,seat_position=seat,game_config={})
def S(kind,action_ids,*,obs=None,phase='moving',game=None,seat=0,version=1):
 return NS(legal_actions=[A(i) for i in action_ids],raw={'observation':obs or {},'game_state':game or {},'your_position':seat,
                 'players':[{'position':i,'name':f'Player{i}'} for i in range(7)]},
           observation=obs or {},phase=phase,new_messages=[],state_version=version)

class Model:
 model='gpt-4.1-mini'
 def __init__(self,answers=(),failure=False):self.answers=list(answers);self.calls=[];self.failure=failure
 def complete_structured(self,messages,schema_name,schema,*,timeout=None):
  self.calls.append((schema_name,messages,timeout))
  if self.failure:raise ProviderError('model down')
  if self.answers:return self.answers.pop(0)
  return {'action_id':'2','reasoning_summary':'Using public evidence'}

def turn(a):return a.action if isinstance(a,WithReasoning) else a

def test_helpers_parse_private_role_and_seat():
 assert _role('You are Player5. Your role: werewolf.')=='wolf'
 assert _role({'your_role':'seer'})=='seer'
 assert _seat('lawrencedz') is None and _seat('Player4')==4

def test_model_choose_pokemon_draft_not_fixed_ranking():
 p=Model([{'action_id':'draft_pick:z','reasoning_summary':'Better team synergy'}])
 a=RiffoAgent(provider=p,log=lambda *_:None)
 s=S('pokemon_vgc_doubles_draft',[],obs={'available_cards':[{'card_id':'x','species':'Incineroar'},
                                              {'card_id':'z','species':'Rotom-Wash'}]},phase='draft')
 s.legal_actions=[A('draft_pick:x'),A('draft_pick:z')]
 assert turn(a.choose_action(s,C('pokemon_vgc_doubles_draft'))).action_id=='draft_pick:z'
 assert len(p.calls)==1 and 'Incineroar' in p.calls[0][1][1]['content']

def test_model_team_preview_validated():
 ans={'bring':['Whimsicott','Basculegion','Incineroar','Flutter Mane'],
      'leads':['Whimsicott','Basculegion'],'reasoning_summary':'Answer opposing Tailwind'}
 p=Model([ans]);a=RiffoAgent(provider=p,log=lambda *_:None)
 s=S('pokemon_vgc_doubles_draft',[],obs={'opponent_roster':['Dragonite','Ursaluna']},phase='team_preview')
 s.legal_actions=[A('select_lineup',action={'type':'select_lineup','roster':['Incineroar','Flutter Mane','Basculegion','Whimsicott','Cresselia','Baxcalibur']})]
 action=turn(a.choose_action(s,C('pokemon_vgc_doubles_draft')))
 assert action['type']=='select_lineup' and action['leads']==['Whimsicott','Basculegion']

def test_model_doubles_full_legal_options_no_static_move_table():
 p=Model([{'slot_0':{'option':1,'target':None},'slot_1':{'option':0,'target':2},'reasoning_summary':'Protect and attack'}]);a=RiffoAgent(provider=p,log=lambda *_:None)
 s=S('pokemon_vgc_doubles_draft',[],obs={'battle':{'weather':'rain'}},phase='moving')
 s.legal_actions=[A('doubles_turn',action={'type':'doubles_turn','slots':[
  {'slot':0,'options':[{'type':'move','move_id':'Glaive Rush','targets':[1,2]},{'type':'move','move_id':'Protect','targets':[]}]},
  {'slot':1,'options':[{'type':'move','move_id':'Thunderbolt','targets':[1,2]},{'type':'pass'}]},
 ]})]
 out=turn(a.choose_action(s,C('pokemon_vgc_doubles_draft')))
 assert out['slot_0']['move_id']=='Protect' and out['slot_1']['target']==2


def test_wolf_teammate_never_lynched_even_if_model_requests_it():
 p=Model([{'action_id':'4'},{'action_id':'4'}]); a=RiffoAgent(provider=p,log=lambda *_:None)
 s=S('werewolf',[0,1,2,3,4,6,7],obs={'your_role':'wolf','wolf_team':[{'position':4},{'position':5}]},
     game={'phase':'day_vote','day':2},seat=5)
 action=turn(a.choose_action(s,C('werewolf',5)))
 assert action.action_id!='4'
 assert a.wolf_teammates=={4}


def test_night_one_wolf_teammate_deduced_from_authorized_actions():
 p=Model([{'action_id':'3'}]);a=RiffoAgent(provider=p,log=lambda *_:None)
 s=S('werewolf',[0,1,2,3,6],obs='You are Player5. Your role: werewolf.',game={'phase':'night_wolf','day':1,'dead':[]},seat=5)
 a.choose_action(s,C('werewolf',5))
 assert a.wolf_teammates=={4}


def test_werewolf_genuine_message_passes_through():
 p=Model([{'decision':'send','message':'On Day 1, Player4 voted against Player2, who was revealed as Seer. Why?',
           'recipient':None}]);a=RiffoAgent(provider=p,log=lambda *_:None)
 s=S('werewolf',[],obs={'your_role':'villager'},phase='messaging',game={'phase':'day_vote','day':2,
 'dead':[{'position':2,'role':'seer'}],'vote_history':[{'votes':{'Player4':'Player2'}}]})
 msg=a.choose_message(s,C('werewolf'))
 assert isinstance(msg,SendMessage) and 'Player4' in msg.content
 assert 'vote_history' in p.calls[0][1][1]['content']


def test_werewolf_fabricated_seer_claim_suppressed():
 p=Model([{'decision':'send','message':'I investigated Player4 and they are definitely a wolf','recipient':None}]); a=RiffoAgent(provider=p,log=lambda *_:None)
 s=S('werewolf',[],obs={'your_role':'villager'},phase='messaging')
 assert a.choose_message(s,C('werewolf')) is TERMINATE_MESSAGING


def test_llm_failure_uses_legal_pokemon_fallback():
 p=Model(failure=True);a=RiffoAgent(provider=p,log=lambda *_:None)
 s=S('pokemon_vgc_doubles_draft',[],phase='draft')
 s.legal_actions=[A('draft_pick:a'),A('draft_pick:b')]
 out=turn(a.choose_action(s,C('pokemon_vgc_doubles_draft')))
 assert out.action_id=='draft_pick:a'
