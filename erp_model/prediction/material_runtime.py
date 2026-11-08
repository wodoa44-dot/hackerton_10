"""All-material orchestration: independent schema-aware handlers and operational lookup."""
import copy,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import joblib
from scipy.sparse import hstack
from erp_model.shared.schema import SLOTS
from erp_model.encoders.preprocess import preprocess
from erp_model.encoders.legacy_candidates import expanded_preprocess
from erp_model.encoders.rules.material_schema import DIODE_SCHEME,code_scheme
from erp_model.models.registry import Registry
from erp_model.prediction.verified_predictor import VerifiedSlotPredictor
from erp_model.encoders.rules.code_rules import CodeRules

CONFIRMED_SCHEMES={DIODE_SCHEME:{'slots':SLOTS,'length':13,'source':'User-provided general diode code table'}}

def slot_applicability(scheme_id,slot):
    rule=CONFIRMED_SCHEMES.get(scheme_id)
    if not rule:return 'unsupported_schema'
    return 'applicable' if slot in rule['slots'] else 'not_applicable'

class MaterialPredictor:
    def __init__(self,directory='artifacts/material_v2',slot=None,registry=None,verified_directory=None):
        self.directory=Path(directory);self.config=json.loads((self.directory/'config.json').read_text(encoding='utf-8'))
        self.family_model=joblib.load(self.directory/'family.joblib')
        self.registry=registry if registry is not None else Registry.load(self.directory/'operational_master.json')
        self.slots=[slot] if slot else SLOTS
        self.verified_directory=Path(verified_directory) if verified_directory else self.directory.parent/'verified_v3'
        self.rules=CodeRules(self.verified_directory/'code_rules.json')
        self.base=Path(self.config['base_models'])
        if not self.base.exists():self.base=self.directory.parent/'run1'
    def prepare(self,name,spec='',scheme=None):
        p=(expanded_preprocess if self.config['expanded_candidates'] else preprocess)(name,spec)
        fam=self.family_model.identify(p);matches=self.registry.search(p,fam['value'])
        if fam['value']=='UNKNOWN' and matches:
            families={m['component_family'] for m in matches}-{'UNKNOWN'}
            if len(families)==1:
                fam={'value':next(iter(families)),'status':'review','source':'registry_name_alias','evidence':matches[0]['evidence'],'candidates':[{'value':next(iter(families)),'score':None}]}
        schemes=sorted({m['scheme']['code_scheme_id'] for m in matches})
        if scheme:
            chosen=scheme;selection='explicit_user_scheme'
        elif len(schemes)==1:chosen=schemes[0];selection='registered_code_structure'
        elif schemes:chosen='MULTIPLE_SCHEMES';selection='registry_selection_required'
        elif len(self.rules.family_schemes(fam['value']))==1:chosen=self.rules.family_schemes(fam['value'])[0];selection='proposed_verified_schema_for_component_family'
        else:chosen='UNRESOLVED';selection='no_verified_code_rules'
        if chosen in self.rules.rules and fam['value']!='UNKNOWN' and fam['value'] not in self.rules.rules[chosen].get('component_families',[]):

            chosen='UNRESOLVED';selection='explicit_scheme_conflicts_with_component_family'
        context={'slot':'context','raw_input':p.raw,'normalized_input':p.normalized,'component_family':fam['value'],'family':fam,
            'code_scheme_id':chosen,'code_scheme_candidates':schemes,'scheme_selection':selection,
            'registry_candidates':matches,'registry_provenance':self.registry.provenance,
            'required_information':[] if chosen in self.rules.rules else [
                '확인된 ERP·고객사 코드체계 문서 및 각 구간의 의미',
                '발급 주체·고객사 식별자(해당 규칙에서 필요한 경우)',
                '마스터에서 할당하는 품번·일련번호 및 발급 권한(해당하는 경우)'],
            'generation_supported':chosen in self.rules.rules,
            'slot_layout':self.rules.rules.get(chosen,{}).get('slots',{}),
            'expected_code_length':self.rules.rules.get(chosen,{}).get('length'),
            'scheme_note':'Only prefix and actual length observed; no segment meanings inferred' if chosen not in self.rules.rules else 'Confirmed general diode 13-character rule',
            'physical_evidence':[e for e in p.evidence if e['slot']=='physical']}
        p.material_context=context
        return p,context
    def core(self,slot=None):
        return VerifiedSlotPredictor(self.verified_directory,slot)
    def result(self,slot,p,context,core=None):
        try:
            scheme=context['code_scheme_id'];family=context['family']
            rule=self.rules.rules.get(scheme)
            applies='unsupported_schema' if rule is None else 'not_applicable' if rule['slots'].get(slot,{}).get('not_applicable') or slot not in rule['slots'] else 'applicable'
            if applies!='applicable':
                return {'slot':slot,'value':None,'predicted_value':None,'status':applies,
                    'top_candidates':[],'candidates':[],'evidence':family['evidence'],'code_scheme_id':scheme,'component_family':context['component_family'],
                    'reason':('Confirmed schema has no such slot' if applies=='not_applicable' else
                              '현재 코드체계의 구간 의미가 확인되지 않아 다이오드 전용 값을 적용하지 않습니다.'),
                    'required_information':context['required_information']}
            result=core.predict_slot(slot,p)
            result.update(top_candidates=result.get('candidates',[]),code_scheme_id=scheme,component_family=context['component_family'])
            return result
        except Exception as e:
            return {'slot':slot,'value':None,'predicted_value':None,'status':'error','top_candidates':[],'candidates':[],
                    'evidence':[],'reason':type(e).__name__+': '+str(e),'code_scheme_id':context['code_scheme_id']}
    def assemble(self,results,context):
        matches=context['registry_candidates'];common={'slot':'assembly','component_family':context['component_family'],
            'code_scheme_id':context['code_scheme_id'],'registry_candidates':matches,'auto_confirmed':False,
            'value':None,'display_code':None,'predicted_code':None,'code_length':None,'code_source':None,
            'status':'review','required_information':context['required_information']}
        if matches:
            if len(matches)==1:
                candidate=matches[0]
                common.update(display_code=candidate['code'],code_length=candidate['code_length'],code_source=candidate['code_source'],
                    status=candidate['status'],auto_confirmed=candidate['auto_confirmed'],value=candidate['code'] if candidate['auto_confirmed'] else None,
                    code_scheme_id=candidate['scheme']['code_scheme_id'],reason=candidate['reason'])
            else:common['reason']='등록 후보가 여러 개입니다. 원문과 코드를 비교하고 하나를 선택하세요.'
            return common
        if context['code_scheme_id'] in self.rules.rules:
            by={r['slot']:r for r in results}
            predicted={s:r.get('predicted_value') for s,r in by.items() if r['status'] not in ['error','unsupported_schema','not_applicable']}
            code=self.rules.assemble(context['code_scheme_id'],predicted)
            required=[s for s,v in self.rules.rules[context['code_scheme_id']]['slots'].items() if not v.get('not_applicable')]
            auto=code is not None and all(by.get(s,{}).get('status')=='confirmed' for s in required)
            common.update(display_code=code,predicted_code=code,code_length=len(code) if code else None,
                code_source='model_assembled' if code else None,status='confirmed' if auto else 'review',value=code if auto else None,
                auto_confirmed=auto,predicted_values=predicted,
                reason='Verified code-field predictions assembled; automatic confirmation is separate' if code else '실제 코드값 후보가 없는 칸이 있거나 확인된 코드 구조를 만족하지 못했습니다.')
            return common
        common.update(status='unsupported_schema',reason='등록 후보가 없고 신규 코드를 생성할 확인된 규칙이 없습니다. 필요한 추가 정보를 확인하세요.')
        return common
    def events(self,name,spec='',scheme=None):
        p,context=self.prepare(name,spec,scheme)
        applicable=context['code_scheme_id'] in self.rules.rules
        core=None
        if applicable:
            core=self.core(self.slots[0] if len(self.slots)==1 else None)
            p._text_cache=(id(core.shared['text'][0]),hstack([v.transform([p.normalized[c]]) for v,c in zip(core.shared['text'],['품목명','규격명'])],format='csr'))
        if len(self.slots)==1:
            result=self.result(self.slots[0],p,context,core);result['context']=context;yield result;return
        yield context
        results=[]
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures=[pool.submit(self.result,s,p,context,core) for s in SLOTS]
            for future in as_completed(futures):
                result=future.result();results.append(result);yield result
        yield self.assemble(results,context)
