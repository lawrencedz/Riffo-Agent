"""New v5.5 evidence and pack-safety tests, all offline, no credentials."""
from types import SimpleNamespace as NS
from altruagent import WithReasoning, SendMessage, TERMINATE_MESSAGING
from agent.agent import RiffoAgent, EvidenceWerewolfCoach, EvidenceHybridWerewolf, TacticalPokemon
from examples.llm.providers import ProviderError

class FakeModel:
    model='gpt-4.1-mini'
    def __init__(self, fail=False):
        self.fail=fail
        self.calls=[]
    def complete_structured(self,messages,schema_name,schema,*,timeout=None):
        self.calls.append(schema_name)
        if self.fail: raise ProviderError('Temporary model failure')
        return {'index':0, 'justification':'Use the verified public evidence.'}

def A(id):
    return NS(action_id=str(id), input={}, label='')

def C(game,seat=0):
    return NS(game_type=game, seat_position=seat, game_config={})

def S(actions, *, observation=None, game_state=None, seat=0):
    players=[{'position':i,'name':f'Player{i}','agent_id':f'a{i}'} for i in range(7)]
    raw={'observation':observation or {},'your_position':seat,'players':players,
         'game_state':game_state or {'phase':'day_vote','day':1,
                        'alive':[f'Player{i}' for i in range(7)], 'dead':[]}}
    return NS(legal_actions=actions,phase='messaging',raw=raw,observation=observation,
              state_version=5,new_messages=[])



def seat(answer):
    choice = answer.action if isinstance(answer, WithReasoning) else answer
    return int(choice.action_id)


def test_structured_private_pack_protected_from_day_lynch():
    a=RiffoAgent(provider=FakeModel(),log=lambda *_:None)
    s=S([A(i) for i in (0,1,2,3,4,6,7)],
        observation={'your_role':'wolf','wolf_team':[{'position':4},{'position':5}]},seat=5)
    assert seat(a.choose_action(s,C('werewolf',5)))!=4
    assert a.werewolf.local.wolf_mates=={4}


def test_private_plaintext_other_wolf_detected_day2():
    a=RiffoAgent(provider=FakeModel(),log=lambda *_:None)
    s=S([A(i) for i in (0,1,3,4,6,7)],
        observation='You are Player5. Your role: werewolf. Other wolf: Player4.',seat=5,
        game_state={'phase':'day_vote','day':2,'alive':['Player0','Player1','Player4','Player5','Player6'],'dead':[]})
    assert seat(a.choose_action(s,C('werewolf',5)))!=4
    assert 4 in a.werewolf.local.wolf_mates


def test_night_legal_targets_identify_living_partner():
    a=RiffoAgent(provider=FakeModel(),log=lambda *_:None)
    s=S([A(i) for i in (0,1,2,3,6)],
        observation='You are Player5. Your role: werewolf.',seat=5,
        game_state={'phase':'night_wolf','day':2,'alive':[f'Player{i}' for i in range(7)],'dead':[]})
    ans=a.choose_action(s,C('werewolf',5))
    assert 4 in a.werewolf.local.wolf_mates
    assert seat(ans) in {0,1,2,3,6}


def test_ambiguous_names_cannot_be_misattributed():
    model=FakeModel()
    a=RiffoAgent(provider=model,log=lambda *_:None)
    s=S([A(i) for i in (0,1,2,3,4,6,7)], observation='You are Player5. Your role: wolf.',seat=5)
    s.raw['players']=[{'position':i,'name':'lawrencedz' if i>1 else 'alime588'} for i in range(7)]
    p=a.werewolf.local
    p._context(s,C('werewolf',5))
    assert p._seat('lawrencedz') is None
    assert p._seat('alime588') is None
    assert p._seat('Player4')==4


def test_revealed_seer_vote_becomes_targeted_evidence_message():
    model=FakeModel();a=RiffoAgent(provider=model,log=lambda *_:None)
    history=[{'day':1,'votes':[{'voter':'Player3','target':'Player2'},
                                {'voter':'Player4','target':'Player2'}]}]
    gs={'phase':'day_vote','day':2,'alive':['Player0','Player1','Player3','Player4','Player5','Player6'],
        'dead':[{'position':2,'role':'seer','cause':'vote'}], 'vote_history':history}
    s=S([A(i) for i in (0,1,3,4,5,6,7)],observation='You are Player6. Your role: villager.',seat=6,game_state=gs)
    msg=a.choose_message(s,C('werewolf',6))
    assert isinstance(msg,SendMessage)
    assert 'Day 1' in msg.content and 'Player2' in msg.content and 'Seer' in msg.content
    assert 'Player3' in msg.content or 'Player4' in msg.content
    assert 'riffo_rank' in model.calls


def test_voters_against_revealed_seer_have_higher_suspicion():
    p=EvidenceWerewolfCoach(log=lambda *_:None)
    gs={'vote_history':[{'day':1,'votes':{'Player4':'Player2','Player6':'Player3'}}]}
    assert p._vote_history_suspicion(4,gs,{2:'seer'})>p._vote_history_suspicion(6,gs,{2:'seer'})


def test_model_failure_uses_evidence_based_message_without_fabrication():
    a=RiffoAgent(provider=FakeModel(fail=True),log=lambda *_:None)
    gs={'phase':'day_vote','day':2,'alive':[f'Player{i}' for i in (0,1,3,4,5,6)],
        'dead':[{'position':2,'role':'seer'}],
        'vote_history':[{'day':1,'votes':{'Player4':'Player2'}}]}
    s=S([A(1)],observation='You are Player6. Your role: villager.',seat=6,game_state=gs)
    m=a.choose_message(s,C('werewolf',6))
    assert isinstance(m,SendMessage) and 'Player4' in m.content and 'Player2' in m.content
    assert 'observed' not in m.content


def test_wolf_private_coordination_then_public_safe_question():
    a=RiffoAgent(provider=FakeModel(),log=lambda *_:None)
    obs={'your_role':'werewolf','other_wolves':[4]}
    s=S([A(i) for i in (0,1,2,3,4,6,7)],observation=obs,seat=5)
    first=a.choose_message(s,C('werewolf',5))
    second=a.choose_message(s,C('werewolf',5))
    third=a.choose_message(s,C('werewolf',5))
    assert isinstance(first,SendMessage) and first.recipients==[4]
    assert isinstance(second,SendMessage) and second.recipients==[]
    assert third is TERMINATE_MESSAGING
    assert 'vote on Player4' not in second.content


def test_no_private_role_leak_in_villager_message():
    a=RiffoAgent(provider=FakeModel(),log=lambda *_:None)
    s=S([A(i) for i in (0,1,2,3,4,6,7)],
        observation={'your_role':'villager','public_transcript':'Player4 is a wolf'},seat=5)
    msg=a.choose_message(s,C('werewolf',5))
    assert isinstance(msg,SendMessage)
    assert 'investigated' not in msg.content


def test_pokemon_v54_preserved():
    a=RiffoAgent(provider=FakeModel(),log=lambda *_:None)
    assert isinstance(a.pokemon,TacticalPokemon)
    assert isinstance(a.werewolf,EvidenceHybridWerewolf)


def test_wolf_does_not_publicly_accuse_packmate_after_seer_lynch():
    a=RiffoAgent(provider=FakeModel(),log=lambda *_:None)
    gs={'phase':'day_vote','day':2,
        'alive':['Player0','Player1','Player3','Player4','Player5','Player6'],
        'dead':[{'position':2,'role':'seer'}],
        'vote_history':[{'day':1,'votes':{'Player4':'Player2','Player3':'Player2'}}]}
    s=S([A(i) for i in (0,1,3,4,6,7)],
        observation={'your_role':'wolf','other_wolves':[4]},seat=5,game_state=gs)
    private=a.choose_message(s,C('werewolf',5))
    public=a.choose_message(s,C('werewolf',5))
    assert isinstance(private,SendMessage) and private.recipients==[4]
    assert isinstance(public,SendMessage)
    assert 'Player4,' not in public.content and 'Player4 voted' not in public.content
    assert 'Player3' in public.content
