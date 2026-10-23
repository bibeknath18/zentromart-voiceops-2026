"""Offline routing evaluation only (no Groq, no ticket side effects)."""
import csv, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import src.core as core

core.rag=lambda msg, retriever: {'status':'LIVE','answer':'[Offline placeholder: answer quality not assessed]','evidence':[]}
agent=core.Agent()
convs=core.rows('practice/conversations.csv')
results=[]
for entry in sorted(convs,key=lambda x:(x['conversation_id'],int(x['turn']))):
    out=agent.respond(entry['customer_message'],entry['conversation_id'],allow_ticket=False)
    results.append({'conversation_id':entry['conversation_id'],'turn':entry['turn'],
                    'expected_decision':entry['expected_decision'],'actual_decision':out['decision'],
                    'match':out['decision']==entry['expected_decision']})
print(json.dumps({'metric':'offline decision exact match only','matches':sum(x['match'] for x in results),
                  'turns':len(results),'nonmatches':[x for x in results if not x['match']]},indent=2))
