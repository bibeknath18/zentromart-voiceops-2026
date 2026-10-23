"""Official practice input coverage; do not conflate extractive previews with true RAG LLM evaluation."""
import csv,json,sys,time,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core import DATA,ROOT,Retriever,rag,baseline,Agent,rows

def eval_all():
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    questions=rows('practice/questions.csv');retriever=Retriever();bres=[];rres=[]
    for q in questions:
        question=q['question'];gold=[a.strip() for a in q.get('gold_evidence','').split(';') if a.strip() and '.csv' not in a]
        b=baseline(question)
        time.sleep(3)
        r=rag(question,retriever)
        time.sleep(3)
        evidence=[e['id'] for e in r.get('evidence',[])]
        # Retrieval evidence recall proxy excludes structural csv references; IDs genuinely derived from practice schema.
        hits=[g for g in gold if g in evidence]
        bres.append({
    'id':q['id'],
    'status':b['status'],
    'answer':b.get('answer'),
    'error':b.get('error'),
    'reference_answer':q.get('reference_answer')
})
        rres.append({
    'id':q['id'],
    'status':r['status'],
    'answer':r.get('answer'),
    'error':r.get('error'),
    'evidence':evidence,
    'gold_kb_sections':gold,
    'gold_hits':hits,
    'gold_recall_at_5':len(hits)/len(gold) if gold else None
})
    (out/'baseline_raw.json').write_text(json.dumps(bres,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'rag_raw.json').write_text(json.dumps(rres,ensure_ascii=False,indent=2),encoding='utf-8')
    valid=[r for r in rres if r['gold_recall_at_5'] is not None]
    retrieval={'evaluated_count':len(valid),'mean_gold_section_recall_at_5':sum(r['gold_recall_at_5'] for r in valid)/len(valid) if valid else None,'note':'Exploratory KB section retrieval metric, not answer quality or official grade; some IDs might be non-KB or require different evidence handling'}
    (out/'retrieval_metrics.json').write_text(json.dumps(retrieval,indent=2))
    model_available=any(r['status']=='LIVE' for r in bres)
    (out/'baseline_results.json').write_text(json.dumps({'total':len(bres),'live_responses':sum(r['status']=='LIVE' for r in bres),'accuracy':None,'note':'No scoring without a live generated response and independent adjudication'},indent=2))
    (out/'rag_results.json').write_text(json.dumps({'total':len(rres),'live_responses':sum(r['status']=='LIVE' for r in rres),'accuracy':None,'retrieval':retrieval,'note':'Extractive preview is NOT genuine generated RAG answer evaluation'},indent=2))
    agent=Agent();convs=rows('practice/conversations.csv');outputs=[]
    for r in sorted(convs,key=lambda x:(x['conversation_id'],int(x['turn']))):
        result=agent.respond(r['customer_message'],r['conversation_id'],allow_ticket=False)
        outputs.append({'conversation_id':r['conversation_id'],'turn':r['turn'],'expected_decision':r['expected_decision'],'actual_decision':result['decision'],'reply':result['answer'],'evidence':[e['id'] for e in result.get('evidence',[])],'verification_state':result['phase'],'tool':result['tool']})
    (out/'conversation_raw.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2),encoding='utf-8')
    stats={'conversations':len(set(r['conversation_id'] for r in convs)),'turns':len(outputs),'decision_exact_matches':sum(x['expected_decision']==x['actual_decision'] for x in outputs),'note':'Deterministic offline decisions with tickets disabled to avoid effects; not a live LLM multi-turn test.'}
    (out/'conversation_metrics.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps({'baseline':len(bres),'rag':len(rres),'retrieval':retrieval,'multi_turn':stats},indent=2))
if __name__=='__main__':eval_all()
