import json,re
from pathlib import Path
from erp_model.shared.schema import SLOTS

class CodeRules:
    def __init__(self,path='config/code_rules.json'):
        self.document=json.loads(Path(path).read_text(encoding='utf-8'))
        self.rules={r['code_scheme_id']:r for r in self.document['rules'] if r.get('verified') and r.get('source')}
        for r in self.rules.values():
            re.compile(r['pattern'])
            for slot,v in r['slots'].items():
                if slot not in SLOTS:raise ValueError('Unknown slot '+slot)
                if not v.get('not_applicable') and not 0<=v['start']<v['end']<=r['length']:raise ValueError('Invalid verified slice')
    def parse(self,code):
        original=str(code);text=original.strip().upper()
        matches=[r for r in self.rules.values() if len(text)==r['length'] and re.fullmatch(r['pattern'],text)]
        if len(matches)!=1:
            return {'code_scheme_id':None,'labels':{},'issues':{'schema':'No unique verified code rule'},'status':'unsupported_schema'}
        rule=matches[0];labels={};issues={}
        for slot,v in rule['slots'].items():
            if v.get('not_applicable'):continue
            value=text[v['start']:v['end']]
            if 'allowed' in v and value not in v['allowed']:issues[slot]='Recorded field is outside verified allowed labels'
            elif v.get('not_only_hyphens') and not value.strip('-'):issues[slot]='No part number recorded'
            else:labels[slot]=value
        return {'code_scheme_id':rule['code_scheme_id'],'labels':labels,'issues':issues,
                'status':'confirmed' if not issues else 'review','source':rule['source']}
    def family_schemes(self,family):return [key for key,r in self.rules.items() if family in r.get('component_families',[])]
    def assemble(self,scheme,values):
        rule=self.rules.get(scheme)
        if not rule:return None
        chars=[None]*rule['length']
        for slot,v in rule['slots'].items():
            if v.get('not_applicable'):continue
            value=values.get(slot)
            if not isinstance(value,str) or len(value)!=v['end']-v['start']:return None
            if 'allowed' in v and value not in v['allowed']:return None
            if v.get('not_only_hyphens') and not value.strip('-'):return None
            for i,ch in enumerate(value,v['start']):
                if chars[i] not in [None,ch]:return None
                chars[i]=ch
        if any(c is None for c in chars):return None
        code=''.join(chars)
        return code if re.fullmatch(rule['pattern'],code) else None
