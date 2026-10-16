"""AI-first Red Alert contract tests: async command handling, legal guards, trends."""
import time
from types import SimpleNamespace as S
from agent.agent import RiffoAgent
from agent.redalert_brain import AdaptiveRedAlert
from altruagent import WAIT, WithReasoning


class Provider:
    model = 'gpt-4.1-mini'
    def __init__(self, answer=None, error=None):
        self.answer = answer or {'orders': [], 'reasoning_summary': 'react to opponent'}
        self.error = error
        self.calls = []
    def complete_structured(self, messages, name, schema, *, timeout=None):
        self.calls.append((messages, name, timeout))
        if self.error:
            raise self.error
        return self.answer


def rt(player, tick=0, *, cash=5000, units=None, buildings=None, production=None, available=None, enemies=None, enemy_buildings=None):
    units = units if units is not None else []
    buildings = buildings if buildings is not None else []
    production = production if production is not None else []
    return S(raw={'observation': {
        'units': units, 'buildings':buildings, 'production':production,
        'visible_enemies':enemies or [],'visible_enemy_buildings':enemy_buildings or [],
        'available_production': available or ['powr','tent','proc','weap','e1','e3','harv'],
        'base_center':[10,10],
        'economy':{'cash':cash,'power_provided':100,'power_drained':50},
        'tick':tick,
    },'time':{'game_seconds':tick},'limits':{},'legal_actions':{},'map':{'bounds':[0,0,100,100]}})


def ctx():
    return S(game_type='red_alert',game_config={'rules': {'buildings': {'proc': {'cost': 2000}, 'weap': {'cost': 2000}, 'powr': {'cost': 300}}, 'units': {'e1': {'cost': 100}, 'harv': {'cost': 1400}}}})


def take(result):
    return result.action['orders'] if isinstance(result, WithReasoning) else []


def complete(ai, answer, clock, state):
    ai.generation += 1
    ai.replies.put((ai.generation, clock[0], answer, None))
    ai.worker_active = True
    return ai._consume(state.raw, ctx())


def test_integration_agent_routes_red_alert():
    ai=RiffoAgent(provider=Provider(), log=lambda _:None)
    assert ai._realtime_players['red_alert'] is AdaptiveRedAlert


def test_deploy_mcv_without_model():
    p=AdaptiveRedAlert(Provider(),log=lambda _:None)
    a=p._quick(rt(p,units=[{'id':55,'type':'mcv'}]).raw,rt(p,units=[{'id':55,'type':'mcv'}]).raw['observation'])
    assert take(a)==[{'cmd':'deploy','units':[55]}]


def test_place_ready_without_model():
    p=AdaptiveRedAlert(Provider(),log=lambda _:None)
    s=rt(p,production=[{'item':'weap','ready':True}]);assert take(p._quick(s.raw,s.raw['observation']))==[{'cmd':'place','item':'weap'}]


def test_worker_runs_in_background():
    class Slow(Provider):
        def complete_structured(self,*a,**kw):
            time.sleep(.14)
            return {'orders':[], 'reasoning_summary':'scout'}
    p=AdaptiveRedAlert(Slow(),log=lambda _:None)
    now=time.monotonic()
    assert p.choose_action(rt(p),ctx()) is WAIT
    assert time.monotonic()-now<.13  # model cannot block the tick
    time.sleep(.16)
    p.choose_action(rt(p),ctx())
    assert p.last_reason=='scout'


def test_actual_ai_build_order_validated():
    t=[100.]
    p=AdaptiveRedAlert(Provider(),log=lambda _:None,clock=lambda:t[0]);s=rt(p, cash=2500,buildings=[{'type':'fact','id':4},{'type':'proc','id':5}]);
    answer={'orders':[{'cmd':'build','item':'weap'}],'reasoning_summary':'opponent is teching, start tanks'}
    a=complete(p,answer,t,s)
    assert take(a)==[{'cmd':'build','item':'weap'}]


def test_early_infantry_burn_prevented_while_refinery_queued():
    p=AdaptiveRedAlert(Provider(),log=lambda _:None)
    s=rt(p,cash=3792,buildings=[{'type':'fact','id':1},{'type':'tent','id':2}],production=[{'item':'proc','ready':False}]);
    orders,bad=p._safe_orders([{'cmd':'train','item':'e1','count':3}],s.raw,ctx())
    assert orders == [] and bad==1


def test_ai_can_choose_emergency_troops_when_enemies_in_base():
    p=AdaptiveRedAlert(Provider(),log=lambda _:None)
    s=rt(p,cash=900,buildings=[{'type':'fact','id':1},{'type':'tent','id':2}], enemies=[{'id':99,'type':'e1','x':11,'y':12}]);
    orders,bad=p._safe_orders([{'cmd':'train','item':'e1','count':2}],s.raw,ctx())
    assert orders==[{'cmd':'train','item':'e1','count':2}]


def test_ai_must_use_real_ids_and_visible_targets():
    p=AdaptiveRedAlert(Provider(),log=lambda _:None)
    s=rt(p,units=[{'type':'e1','id':42}], enemies=[{'id':9,'type':'e1','x':11,'y':12}]);
    orders,bad=p._safe_orders([{'cmd':'attack','units':[88],'target':9},
                               {'cmd':'attack','units':[42],'target':99},
                               {'cmd':'attack','units':[42],'target':9}],s.raw,ctx())
    assert orders==[{'cmd':'attack','units':[42],'target':9}] and bad==2


def test_no_build_queue_spam():
    p=AdaptiveRedAlert(Provider(),log=lambda _:None)
    s=rt(p,production=[{'item':'proc','ready':False}]);
    orders,bad=p._safe_orders([{'cmd':'build','item':'proc'}],s.raw,ctx())
    assert orders==[] and bad==1


def test_old_orders_are_revalidated_against_current_units():
    p=AdaptiveRedAlert(Provider(),log=lambda _:None); t=[50.];p.clock=lambda:t[0]
    s=rt(p,units=[])
    a=complete(p,{'orders':[{'cmd':'attack_move','units':[88],'to':[50,50]}]},t,s)
    assert a is WAIT


def test_repeated_identical_order_blocked_for_cooldown():
    t=[10.];p=AdaptiveRedAlert(Provider(),log=lambda _:None,clock=lambda:t[0]);s=rt(p)
    a=p._send([{'cmd':'build','item':'powr'}], 'go');assert take(a)
    orders,bad=p._safe_orders([{'cmd':'build','item':'powr'}],s.raw,ctx())
    assert orders==[] and bad==1


def test_stall_restarts_harvester_even_when_idle_false():
    t=[100.];p=AdaptiveRedAlert(Provider(),log=lambda _:None,clock=lambda:t[0]);p.last_cash_increase=50
    s=rt(p,units=[{'type':'harv','id':17,'idle':False,'x':30,'y':33}],cash=100)
    a=p._quick(s.raw,s.raw['observation']);assert take(a)==[{'cmd':'harvest','units':[17]}]


def test_opponent_sightings_and_trends_reach_model():
    t=[50.];p=AdaptiveRedAlert(Provider(),log=lambda _:None,clock=lambda:t[0])
    s=rt(p,units=[{'type':'e1','id':2}],enemies=[{'type':'2tnk','id':44,'x':50,'y':50}],enemy_buildings=[{'type':'weap','id':65}]);p._observe(s.raw,s.raw['observation'])
    summary=p._snapshot(s.raw,s.raw['observation'],ctx())
    assert summary['observed_opponent_not_omniscient']['enemy_units_visible_now']['2tnk']==1
    assert 'weap' in summary['observed_opponent_not_omniscient']['enemy_building_types_seen_at_least_once']


def test_error_keeps_autonomy_and_sets_retry():
    t=[100.];p=AdaptiveRedAlert(Provider(),log=lambda _:None,clock=lambda:t[0]);p.worker_active=True;p.generation=3
    p.replies.put((3,100.,None,'ProviderError: HTTP 429'))
    assert p._consume(rt(p).raw,ctx()) is WAIT
    assert p.errors==1 and p.next_call>=105


def test_reply_stale_discarded():
    t=[100.];p=AdaptiveRedAlert(Provider(),log=lambda _:None,clock=lambda:t[0]);p.worker_active=True;p.generation=2
    p.replies.put((2,40.,{'orders':[{'cmd':'build','item':'weap'}]},None))
    assert p._consume(rt(p).raw,ctx()) is WAIT


def test_reply_send_trains_after_infrastructure_complete():
    t=[100.];p=AdaptiveRedAlert(Provider(),log=lambda _:None,clock=lambda:t[0]);s=rt(p,cash=1500,buildings=[{'type':'proc','id':1},{'type':'weap','id':2}]);
    ans={'orders':[{'cmd':'train','item':'e1','count':3}],'reasoning_summary':'Counter a fast infantry rush'}
    assert take(complete(p,ans,t,s))==[{'cmd':'train','item':'e1','count':3}]


def test_map_out_of_bounds_rejected():
    p=AdaptiveRedAlert(Provider(),log=lambda _:None)
    s=rt(p,units=[{'id':12,'type':'e1'}]);
    orders,bad=p._safe_orders([{'cmd':'attack_move','units':[12],'to':[999,999]}],s.raw,ctx())
    assert orders==[] and bad==1


def test_unknown_price_training_rejected_until_core_complete():
    p=AdaptiveRedAlert(Provider(),log=lambda _:None)
    s=rt(p,cash=4500,available=['newunit','proc','weap'])
    orders,bad=p._safe_orders([{'cmd':'train','item':'newunit','count':2}],s.raw,ctx())
    assert orders==[] and bad==1


def test_build_can_be_attempted_below_full_price_server_decides():
    p=AdaptiveRedAlert(Provider(),log=lambda _:None)
    s=rt(p,cash=1300,buildings=[{'type':'proc','id':5}])
    proposed=[{'cmd':'build','item':'weap'}]
    orders,bad=p._safe_orders(proposed,s.raw,ctx())
    assert orders==proposed and bad==0


def test_outage_recovery_makes_one_essential_build_not_fixed_macro():
    t=[100.];p=AdaptiveRedAlert(Provider(), log=lambda _:None, clock=lambda:t[0]);p.errors=2
    s=rt(p, cash=5000, buildings=[{'type':'fact','id':3},{'type':'powr','id':4},{'type':'tent','id':5}])
    out=p._outage_fallback(s.raw,s.raw['observation'],ctx())
    assert take(out)==[{'cmd':'build','item':'proc'}]

def test_outage_recovery_never_runs_when_model_available():
    p=AdaptiveRedAlert(Provider(), log=lambda _:None)
    s=rt(p)
    assert p._outage_fallback(s.raw,s.raw['observation'],ctx()) is WAIT
