"""Targeted Red Alert macro tests. Use actual SDK when available."""
import time
from agent.agent import FastRedAlert
from test_riffo_v60 import Model, C, rstate, cmds


def test_factory_before_second_refinery_in_safe_opening():
    p = FastRedAlert(Model(), log=lambda *_: None)
    s = rstate(buildings=['fact','powr','barr','proc'], cash=2100,
               available=['proc','weap','e1'])
    orders = cmds(p.choose_action(s, C('red_alert')))
    assert {'cmd':'build','item':'weap'} in orders, orders
    assert not any(o.get('item')=='proc' for o in orders), orders
    assert not any(o['cmd']=='train' for o in orders), 'Do not use second-refinery savings on excess infantry'


def test_wait_for_factory_capital_instead_of_burning_cash():
    p = FastRedAlert(Model(), log=lambda *_: None)
    s = rstate(buildings=['fact','powr','barr','proc'], cash=1800,
               available=['proc','weap','e1'])
    orders = cmds(p.choose_action(s, C('red_alert')))
    assert not any(o.get('item')=='proc' for o in orders), orders
    assert not any(o['cmd']=='train' for o in orders), orders


def test_factory_after_two_refineries():
    p = FastRedAlert(Model(), log=lambda *_: None)
    s = rstate(buildings=['fact','powr','barr','proc','proc'], cash=2500,
               available=['proc','weap','e1'])
    orders = cmds(p.choose_action(s, C('red_alert')))
    assert {'cmd':'build','item':'weap'} in orders, orders


def test_visible_early_rush_can_override_refinery_and_train_defenders():
    p = FastRedAlert(Model(), log=lambda *_: None)
    s = rstate(buildings=['fact','powr','barr','proc'], cash=2500,
               units=[{'type':'e1','id':4}], available=['proc','weap','e1'],
               enemies=[{'type':'e1','id':900,'cell':[51,50]}])
    orders = cmds(p.choose_action(s, C('red_alert')))
    assert any(o['cmd']=='attack' and o['target']==900 for o in orders), orders
    assert any(o['cmd']=='train' and o['item']=='e1' for o in orders), orders


def test_strategy_model_delay_does_not_delay_immediate_orders():
    class SlowModel(Model):
        def complete_structured(self,*args,**kwargs):
            time.sleep(0.4)
            return {'stance':'balanced','priority':'economy','reason':'Expand'}
    now=[0.0]
    p=FastRedAlert(SlowModel(),log=lambda *_:None,clock=lambda:now[0]);now[0]=50
    s=rstate(buildings=['fact','powr','barr'],cash=4500,available=['proc','e1'])
    start=time.monotonic()
    orders=cmds(p.choose_action(s,C('red_alert')))
    assert time.monotonic()-start<0.25, 'A network call must not block RTS actions'
    assert {'cmd':'build','item':'proc'} in orders, orders
    assert p.ai_calls==1


def test_server_rejections_are_visible():
    messages=[]
    p=FastRedAlert(Model(),log=messages.append)
    s=rstate(buildings=['fact','powr','barr'],cash=3000,available=['proc'])
    s.raw['last_orders']={'batch':17,'pending':False,'results':[{'index':0,'cmd':'build','status':'rejected','reason':'insufficient_funds'}]}
    s.raw['recent_order_problems']=[{'cmd':'build','status':'rejected','reason':'insufficient_funds'}]
    p.choose_action(s,C('red_alert'))
    assert any('insufficient_funds' in line for line in messages),messages
    first_n=len(messages)
    p.choose_action(s,C('red_alert'))
    assert sum('recent order problems' in x for x in messages)==1
