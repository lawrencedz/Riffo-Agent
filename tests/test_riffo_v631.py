"""Reproductions of the RTS $100 economy stall and opener."""
from agent.agent import FastRedAlert
from test_riffo_v60 import Model, C, rstate, cmds


def test_factory_is_built_before_second_refinery():
    agent = FastRedAlert(Model(), log=lambda *_: None)
    state = rstate(buildings=['fact','powr','tent','proc'], cash=2500,
                   available=['proc','weap','e1'])
    orders = cmds(agent.choose_action(state, C('red_alert')))
    assert {'cmd':'build','item':'weap'} in orders, orders
    assert not any(o.get('item')=='proc' for o in orders), orders


def test_factory_savings_not_spent_on_infantry():
    agent = FastRedAlert(Model(), log=lambda *_: None)
    state = rstate(buildings=['fact','powr','tent','proc'], cash=1850,
                   available=['proc','weap','e1'])
    orders = cmds(agent.choose_action(state, C('red_alert')))
    assert not any(o.get('cmd')=='train' for o in orders), orders


def test_second_refinery_follows_factory():
    agent = FastRedAlert(Model(), log=lambda *_: None)
    state = rstate(buildings=['fact','powr','tent','proc','weap'], cash=2500,
                   power=150,drain=80, available=['proc','e1'])
    orders = cmds(agent.choose_action(state, C('red_alert')))
    assert {'cmd':'build','item':'proc'} in orders, orders


def test_power_headroom_before_second_refinery():
    agent = FastRedAlert(Model(), log=lambda *_: None)
    state = rstate(buildings=['fact','powr','tent','proc','weap'], cash=2500,
                   power=100,drain=80, available=['proc','powr','e1'])
    orders = cmds(agent.choose_action(state, C('red_alert')))
    assert {'cmd':'build','item':'powr'} in orders, orders
    assert not any(o.get('item')=='proc' for o in orders), orders


def test_idle_harvester_receives_harvest_order():
    agent = FastRedAlert(Model(), log=lambda *_: None)
    state = rstate(buildings=['fact','powr','tent','proc','weap'], cash=100,
                   units=[{'id':18,'type':'harv','idle':True}], available=[])
    orders=cmds(agent.choose_action(state,C('red_alert')))
    assert any(o['cmd']=='harvest' and o['units']==[18] for o in orders),orders
