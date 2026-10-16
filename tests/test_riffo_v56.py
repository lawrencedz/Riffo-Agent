"""Riffo v5.6 Red Alert regression checks; no UCLA or model access required."""
from types import SimpleNamespace as NS
from altruagent import WAIT, WithReasoning
from agent.agent import RiffoAgent, RushResilientRedAlert


class Model:
    model = 'gpt-4.1-mini'
    def __init__(self): self.calls = 0
    def complete_structured(self, *args, **kwargs):
        self.calls += 1
        return {'orders': []}


def unit(i, kind='e1', **kwargs):
    return {'id': i, 'type': kind, **kwargs}


def state(*, cash=5000, buildings=(), units=(), production=(), available=(),
          power=180, drain=80, enemies=(), base=(50,50), attack=None):
    return {'observation':{
        'economy':{'cash':cash,'power_provided':power,'power_drained':drain},
        'buildings':[{'type':item,'id':100+i, 'cell':[50, 50]} for i, item in enumerate(buildings)],
        'units':list(units), 'production':list(production),
        'available_production':list(available), 'base_center':list(base),
        'visible_enemies':list(enemies)},
        'legal_actions':{'attack_now':attack} if attack else {}}


def commander(model=None, clock=lambda:1000.0):
    return RushResilientRedAlert(model or Model(), log=lambda *_:None,clock=clock)


def orders(p, **kwargs):
    return p._build_plan(state(**kwargs))[0]


def contains(plan, cmd, item=None):
    return [o for o in plan if o['cmd']==cmd and (item is None or o.get('item')==item)]


def test_agent_uses_v56_rts_while_preserving_other_games():
    agent=RiffoAgent(provider=Model(),log=lambda *_:None)
    assert isinstance(agent.red_alert, RushResilientRedAlert)
    assert 'TacticalPokemon' in type(agent.pokemon).__name__
    assert 'EvidenceHybridWerewolf' in type(agent.werewolf).__name__


def test_power_rush_build_then_barracks_then_refinery():
    assert contains(orders(commander(),buildings=['fact'],available=['powr']), 'build','powr')
    assert contains(orders(commander(),buildings=['fact','powr'], available=['barr','proc']), 'build','barr')
    plan=orders(commander(),buildings=['fact','powr','barr'], available=['powr','proc','e1'],
                power=180,drain=172)
    assert contains(plan,'build','proc'),plan
    assert not contains(plan,'build','powr'),plan


def test_critical_first_refinery_not_blocked_by_very_mild_power_shortfall():
    plan=orders(commander(),buildings=['fact','powr','barr'], available=['powr','proc','e1'],
                power=180,drain=191)
    assert contains(plan,'build','proc'),plan
    assert not contains(plan,'build','powr'),plan


def test_genuine_blackout_gets_power():
    plan=orders(commander(),buildings=['fact','powr','barr'], available=['powr','proc','e1'],
                power=100,drain=160)
    assert contains(plan,'build','powr'),plan


def test_barracks_trains_early_defenders_simultaneously_with_refinery():
    plan=orders(commander(),buildings=['fact','powr','barr'], available=['proc','e1'], cash=5000)
    assert contains(plan,'build','proc'),plan
    trainees=contains(plan,'train','e1')
    assert trainees and trainees[0]['count']>=4,plan
    assert sum(commander()._item_cost(o['item'])*o.get('count',1)
               for o in plan if o['cmd'] in ('train','build'))<=5000


def test_under_rush_defends_even_before_refinery_online():
    p=commander()
    plan=orders(p,buildings=['fact','powr','barr'],units=[unit(1),unit(2)],
                enemies=[unit(700,'e1',cell=[54,50])],available=['proc','e1'],cash=2400)
    assert any(o['cmd']=='attack' and o['target']==700 and set(o['units'])=={1,2} for o in plan),plan
    assert contains(plan,'build','proc'),plan
    assert contains(plan,'train','e1'),plan


def test_refinery_before_factory_and_second_refinery_not_blocked_by_two_harvesters():
    p=commander()
    plan=orders(p, buildings=['fact','powr','barr','proc'], available=['weap','e1'],cash=4200)
    assert contains(plan,'build','weap'),plan
    plan=orders(commander(),buildings=['fact','powr','barr','proc','weap'],
                units=[unit(5,'harv'),unit(6,'harv')],available=['proc','powr','e1'],
                cash=4800)
    assert contains(plan,'build','proc'),plan
    assert not contains(plan,'build','powr'),plan


def test_queued_refinery_does_not_trigger_redundant_second_refinery_or_factory():
    p=commander()
    plan=orders(p,buildings=['fact','powr','barr'],
                production=[{'queue':'building','item':'proc','ready':False}],
                available=['weap','proc','e1'],cash=4100)
    assert not contains(plan,'build','weap'),plan
    assert not contains(plan,'build','proc'),plan


def test_completed_refinery_placed_instead_of_power():
    p=commander()
    plan=orders(p, buildings=['fact','powr','barr'],
                production=[{'queue':'building','item':'proc','ready':True}],
                available=['powr','e1'],power=180,drain=176)
    assert plan==[{'cmd':'place','item':'proc'}],plan


def test_group_defense_uses_authoritative_ids_and_valid_unit_count():
    p=commander()
    plan=orders(p,buildings=['fact','powr','barr'],units=[unit(i) for i in range(9)],
                enemies=[unit(99,'e1',cell=[51,51])],cash=800,available=['e1'])
    defense=[o for o in plan if o['cmd']=='attack']
    assert defense and defense[0]['units']==list(range(9)) and defense[0]['target']==99


def test_no_premature_cross_map_attack():
    plan=orders(commander(),buildings=['fact','powr','barr','proc','weap'],
                units=[unit(i) for i in range(5)],available=['e1'],
                attack={'cmd':'attack_move','units':list(range(5)),'to':[60,50]})
    assert not contains(plan,'attack_move'),plan


def test_can_attack_with_massed_army_after_establishing_economy():
    plan=orders(commander(),buildings=['fact','powr','barr','proc','weap'],
                units=[unit(i) for i in range(18)],available=['e1'],
                attack={'cmd':'attack_move','units':list(range(18)),'to':[60,50]})
    assert contains(plan,'attack_move'),plan


def test_never_spends_more_than_available_cash_even_with_early_defense():
    p=commander()
    raw=state(buildings=['fact','powr','barr'],available=['proc','e1'],cash=2200,
              enemies=[unit(1000,cell=[50,50])])
    plan,_=p._build_plan(raw)
    spend=sum(p._item_cost(o['item'])*o.get('count',1) for o in plan if o['cmd'] in ('build','train'))
    assert spend<=2200,plan


def test_offline_choose_action_uses_multi_order_batch_without_model():
    fake=Model();p=commander(fake)
    raw=state(buildings=['fact','powr','barr'],available=['proc','e1'],cash=5000)
    act=p.choose_action(NS(raw=raw),NS(game_type='red_alert',game_config={}))
    assert isinstance(act,WithReasoning)
    assert act.action['type']=='orders' and len(act.action['orders'])>=2
    assert fake.calls==0


def test_authoritative_config_cost_used_when_present():
    p=commander()
    p._game_config={'rules':{'buildings':{'proc':{'cost':2600}},'units':{'e1':{'cost':175}}}}
    assert p._item_cost('proc')==2600 and p._item_cost('e1')==175
    plan=orders(p,buildings=['fact','powr','barr'],available=['proc','e1'],cash=2000)
    assert not contains(plan,'build','proc'),plan
    assert not contains(plan,'train','e1'),plan
