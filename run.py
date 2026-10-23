import argparse,json,sys,re
from pathlib import Path
from src.core import Agent,rag,baseline
from src.voice import GnaniVoice,VoiceError,validate_audio

def main():
    p=argparse.ArgumentParser(description='ZentroMart support / Gnani voice batch')
    p.add_argument('--input-dir');p.add_argument('--output-dir');p.add_argument('--question');p.add_argument('--baseline',action='store_true');p.add_argument('--session-id',default='cli')
    a=p.parse_args()
    if a.question:
        print(json.dumps(baseline(a.question) if a.baseline else Agent().respond(a.question,a.session_id),ensure_ascii=False,indent=2));return 0
    if not a.input_dir or not a.output_dir:p.error('Provide --question OR both --input-dir and --output-dir')
    inp=Path(a.input_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    if not inp.is_dir():print('Input directory missing',file=sys.stderr);return 2
    agent=Agent();voice=None;results=[]
    for f in sorted(inp.iterdir()):
        if not f.is_file():continue
        result={'input':f.name}
        try:
            validate_audio(f)
            if voice is None:voice=GnaniVoice()
            result.update(voice.process(f,out/(f.stem+'_reply.wav'),agent,'file:'+f.name))
        except Exception as exc:result.update(status='failed',error=str(exc))
        (out/(f.stem+'_result.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');results.append(result)
    (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'files':len(results),'success':sum(r.get('status')=='success' for r in results),'failed':sum(r.get('status')=='failed' for r in results)},indent=2))
    return 1 if any(r.get('status')!='success' for r in results) else 0
if __name__=='__main__':sys.exit(main())
