"""Offline Stage 3 tests. Does not call Groq or create real tickets."""
from src.core import Agent


def make_agent():
    agent=Agent(create_ticket_fn=lambda **kw:{'created':True,'ticket_id':'TKT-DEMO'},
                lookup_fn=lambda oid, proof:{'found':True,'verified':True,'status':'shipped'} if proof=='1234' else {'verified':False})
    return agent


def test_generic_help_clarifies():
    a=make_agent()
    assert a.respond('Hi, I need help with an order.')['decision']=='clarify'


def test_verification_and_cancellation():
    a=make_agent()
    assert a.respond('My order is 1234 5678')['decision']=='clarify'
    assert a.respond('last four 1234')['verified'] is True
    assert a.respond('Cancel it')['decision']=='refuse'



def test_failed_verification_limit():
    a=make_agent()
    a.respond('My order is 1234 5678')

    assert a.respond('wrong 8888')['decision']=='clarify'
    assert a.respond('wrong 9999')['decision']=='handover'
    assert a.respond('wrong 7777')['decision']=='refuse'


def test_human_escalation_no_ticket_in_offline_mode():
    a=make_agent()
    response=a.respond('Can I speak to a person?',allow_ticket=False)
    assert response['decision']=='escalate'
    assert response['tool'] is None
    assert a.sessions['single'].ticket_id==''


def test_ticket_success_and_no_duplicates():
    a=make_agent()
    first=a.respond('Can I speak to a person?')
    again=a.respond('Talk to someone')
    assert first['decision']=='escalate'
    assert first['tool']=='create_ticket:created'
    assert again['tool']=='create_ticket:existing'


def test_ticket_creation_failure_is_not_claimed():
    a=make_agent()
    a.ticket=lambda **kw:{'created':False}
    response=a.respond('Speak to a human')
    assert response['decision']=='escalate'
    assert response['tool']=='create_ticket:failed'
    assert a.sessions['single'].ticket_id==''


def test_unauthorized_order_escalation():
    a=make_agent()
    assert a.respond('I see an unauthorized order')['decision']=='escalate'


def test_no_unapproved_voucher():
    a=make_agent()
    assert a.respond('Give me a 300 rupaye voucher')['decision']=='refuse'
