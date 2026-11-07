"""Operational ERP lookup. Evaluation builds a different registry from training rows only."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from .preprocess import preprocess
from .material_schema import silver_family,code_scheme

class Registry:
    def __init__(self,entries,provenance=None):
        self.entries=entries;self.provenance=provenance or {};self.exact=defaultdict(list);self.spec=defaultdict(list)
        for e in entries:
            self.exact[tuple(e['normalized_input'])].append(e)
            if e['normalized_input'][1]:self.spec[e['normalized_input'][1]].append(e)
    @classmethod
    def build(cls,rows,provenance=None):
        entries=[];seen=set()
        for r in rows:
            key=(r['품목명'],r['규격명'],r['품목코드'])
            if key in seen:continue
            seen.add(key);p=preprocess(r['품목명'],r['규격명']);family,ev=silver_family(p)
            entries.append({'id':hashlib.sha256(json.dumps(key,ensure_ascii=False).encode()).hexdigest()[:20],
                'registered_name':r['품목명'],'registered_spec':r['규격명'],'code':r['품목코드'],'code_length':len(r['품목코드']),
                'normalized_input':list(p.key),'component_family':family or 'UNKNOWN','scheme':code_scheme(r['품목코드']),
                'source_file':r.get('_source_file'),'source_row':r.get('_source_row'),'family_label_source':'unambiguous name alias' if family else 'unknown',
                'division_observed':r.get('사업부','')})
        return cls(entries,provenance)
    def save(self,path):
        Path(path).write_text(json.dumps({'entries':self.entries,'provenance':self.provenance},ensure_ascii=False),encoding='utf-8')
    @classmethod
    def load(cls,path):
        data=json.loads(Path(path).read_text(encoding='utf-8'));return cls(data['entries'],data['provenance'])
    def search(self,p,family):
        exact=self.exact.get(p.key,[])
        if exact:matches=exact;source='registry_exact'
        elif p.key[1] and family!='UNKNOWN':
            matches=[e for e in self.spec.get(p.key[1],[]) if e['component_family']==family];source='registry_candidate'
        else:matches=[];source='registry_candidate'
        results=[]
        codes={e['code'] for e in matches}
        for e in matches:
            different=p.raw['품목명']!=e['registered_name']
            results.append({**e,'code_source':source,'status':'confirmed' if source=='registry_exact' and len(codes)==1 else 'review',
                'auto_confirmed':source=='registry_exact' and len(codes)==1,
                'input_name':p.raw['품목명'],'input_spec':p.raw['규격명'],'name_differs':different,
                'normalized_name_matches':p.key[0]==e['normalized_input'][0],
                'match_basis':['normalized name and specification exact'] if source=='registry_exact' else ['normalized specification exact','component family agrees'],
                'evidence':[{'source_column':'규격명','matched_text':p.raw['규격명'],'normalized_value':p.key[1],'transformation':'NFKC+uppercase+dash+whitespace'}],
                'reason':'Registered name differs; user review required' if source=='registry_candidate' else 'Existing ERP master record; no generated segments'})
        return results
