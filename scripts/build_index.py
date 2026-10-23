import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core import Retriever,ROOT
r=Retriever()
p=ROOT/'reports'/'index_manifest.json'
p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps({'type':'hybrid-word-character-tfidf','sections':len(r.chunks),'files':len({c['filename'] for c in r.chunks}),'chunk_strategy':'heading-aware','index':'rebuilt in memory on load; no external embeddings','section_ids':[x['id'] for x in r.chunks]},indent=2),encoding='utf-8')
print('Indexed',len(r.chunks),'sections')
