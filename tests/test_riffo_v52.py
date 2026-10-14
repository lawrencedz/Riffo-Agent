"""Offline policy tests; no real API calls or live Red Alert simulator."""
from types import SimpleNamespace as NS
from altruagent import WAIT,WithReasoning
from agent.agent import EconomyFirstRedAlert,CounterDraftPokemon,_opponent_species,_counter_score

class FakeModel:
    model='gpt-4o-mini'
    def __init__(self): self.calls=[]
    def complete_structured(self,messages,schema_name,schema,*,timeout=None):
        self.calls.append(schema_name)
        if schema_name=='riffo_rank':return {'index':0,'justification':'counter strategy'}
        if schema_name=='red_alert_orders':return {'orders':[]}
        raise AssertionError(schema_name)


def sample(*, cash=5000, buildings=(), units=(), production=(), available=(), attack=None):
    return {'observation':{'economy':{'cash':cash,'power_provided':180,'power_drained':80},
       'units':list(units),'buildings':[{'id':30+i,'type':b} for i,b in enumerate(buildings)],
       'production':list(production),'available_production':list(available)},
       'legal_actions':{'attack_now':attack} if attack else {}}

def unit(i, typ='e1', idle=False):return {'id':i,'type':typ,'idle':idle}

def test_initial_deploy_and_first_power():
    p=EconomyFirstRedAlert(FakeModel(),log=lambda *_:None,clock=lambda:1000)
    a=p.choose_action(NS(raw=sample(units=[unit(1,'mcv')],available=['powr'])),NS(game_type='red_alert',game_config={}))
    assert a.action['orders']==[{'cmd':'deploy','units':[1]}]
    b=p._build_plan(sample(buildings=['fact'],available=['powr']))[0]
    assert b[0]=={'cmd':'build','item':'powr'}

def test_refinery_gets_priority_before_infantry_spending():
    p=EconomyFirstRedAlert(FakeModel(),log=lambda *_:None,clock=lambda:1000)
    plan,_=p._build_plan(sample(buildings=['fact','powr','barr'],available=['proc','e1'],cash=2500))
    assert any(o['cmd']=='build' and o['item']=='proc' for o in plan)
    assert not any(o['cmd']=='train' for o in plan), plan

def test_batched_infantry_instead_of_pairs():
    p=EconomyFirstRedAlert(FakeModel(),log=lambda *_:None,clock=lambda:1000)
    plan,_=p._build_plan(sample(buildings=['fact','powr','barr','proc'],cash=3500,available=['e1','weap']))
    trained=[o for o in plan if o['cmd']=='train']
    assert trained and trained[0]['count']>=4,plan

def test_understrength_attack_is_held():
    p=EconomyFirstRedAlert(FakeModel(),log=lambda *_:None,clock=lambda:1000)
    for n in (1,4,8):
        soldiers=[unit(i) for i in range(n)]
        raw=sample(buildings=['fact','powr','proc','barr','weap'],units=soldiers,
                   available=['e1'], attack={'cmd':'attack_move','units':list(range(n)),'to':[60,50]})
        orders,_=p._build_plan(raw)
        assert not any(o['cmd'] in ('attack','attack_move') for o in orders)

def test_massed_attack_late_game():
    p=EconomyFirstRedAlert(FakeModel(),log=lambda *_:None,clock=lambda:1000)
    soldiers=[unit(i) for i in range(18)]
    raw=sample(buildings=['fact','powr','proc','barr','weap'],units=soldiers,
               available=['e1'],attack={'cmd':'attack_move','units':list(range(18)),'to':[60,50]})
    orders,_=p._build_plan(raw)
    assert any(o['cmd']=='attack_move' and len(o['units'])==18 for o in orders)

def test_ready_structure_placed_before_anything_else():
    p=EconomyFirstRedAlert(FakeModel(),log=lambda *_:None,clock=lambda:1000)
    orders,_=p._build_plan(sample(buildings=['fact','powr'],production=[{'item':'proc','ready':True}],available=['e1']))
    assert orders==[{'cmd':'place','item':'proc'}]

def test_idle_harvester_not_attacking():
    p=EconomyFirstRedAlert(FakeModel(),log=lambda *_:None,clock=lambda:1000)
    raw=sample(buildings=['fact','powr','proc','barr','weap'],units=[unit(88,'harv',True)],available=['e1'])
    orders,_=p._build_plan(raw)
    assert any(o['cmd']=='harvest' and o['units']==[88] for o in orders)

def test_extract_opponent_only_not_draft_pool():
    obs={'available_cards':[{'species':'Groudon'}],
         'opponent_roster':[{'species':'Pelipper'},{'species':'Kyogre'}]}
    assert _opponent_species(obs)==['Pelipper','Kyogre']
    assert _counter_score('Rillaboom',['Kyogre']) > _counter_score('Heatran',['Kyogre'])

def test_counter_draft_creates_safe_actions_and_model_is_asked():
    model=FakeModel();p=CounterDraftPokemon(model,log=lambda *_:None)
    obs={'available_cards':[{'card_id':'a','species':'Heatran'}, {'card_id':'b','species':'Rillaboom'}],
         'opponent_roster':['Kyogre']}
    state=NS(legal_actions=[NS(action_id='draft_pick:a',input={},label='Heatran'),
                            NS(action_id='draft_pick:b',input={},label='Rillaboom')],
             raw={'observation':obs},phase='draft')
    candidates=p._alternatives(state)
    assert len(candidates)==2
    assert candidates[0]['action'].action_id=='draft_pick:b'
    answer=p.choose_action(state,NS(game_type='pokemon_vgc_doubles_draft'))
    assert isinstance(answer,WithReasoning)
    assert answer.action.action_id=='draft_pick:b'
    assert 'riffo_rank' in model.calls


def test_extract_public_two_seat_rosters_without_self_leak():
    obs={'rosters':{'0':[{'species':'Groudon'}],'1':[{'species':'Kyogre'}]}}
    assert _opponent_species(obs,0)==['Kyogre']
    assert _opponent_species(obs,1)==['Groudon']
    assert _opponent_species(obs)==[]


def test_defend_base_even_with_small_army():
    p=EconomyFirstRedAlert(FakeModel(),log=lambda *_:None,clock=lambda:1000)
    raw=sample(buildings=['fact','powr','proc','barr'],units=[unit(i) for i in range(3)],available=['e1'])
    raw['observation']['base_center']=[50,50]
    raw['observation']['visible_enemies']=[{'id':77,'type':'e1','x':51,'y':54}]
    orders,_=p._build_plan(raw)
    assert any(o['cmd']=='attack' and o['target']==77 and len(o['units'])==3 for o in orders)


def test_attack_waits_if_visible_enemy_army_is_larger():
    p=EconomyFirstRedAlert(FakeModel(),log=lambda *_:None,clock=lambda:1000)
    raw=sample(buildings=['fact','powr','proc','barr','weap'],units=[unit(i) for i in range(15)],available=['e1'],attack={'cmd':'attack_move','units':list(range(15)),'to':[60,50]})
    raw['observation']['visible_enemies']=[{'id':100+i,'type':'e1'} for i in range(20)]
    orders,_=p._build_plan(raw)
    assert not any(o['cmd']=='attack_move' for o in orders)
