"""Candidate expansion justified by training-only misses. No label access at inference."""
import re
from erp_model.encoders.preprocess import preprocess,generate_candidates

def expanded_preprocess(name,spec=''):
    p=preprocess(name,spec);by={c['value']:c for c in p.candidates}
    variants=[]
    for col,text in p.normalized.items():
        for pattern,replace,method in [
            (r'\bIN(?=\d{3,})','1N','ocr_I_to_1_in_diode_series'),
            (r'(?<=[A-Z0-9])(?P<a>\d)\.(?P<b>\d)',r'\g<a>R\g<b>','decimal_R_notation'),
        ]:
            if re.search(pattern,text):
                modified=dict(p.normalized);modified[col]=re.sub(pattern,replace,text)
                variants.append((modified,col,method))
    for modified,col,method in variants:
        for c in generate_candidates(modified):
            if c['value'] in by:continue
            c['stage']=4;c['features']['alias_generated']=1
            c['features']['exact']=0;c['features']['trim']=1
            c['origins']=[{'source_column':col,'matched_text':p.raw[col],'normalized_value':c['value'],'transformation':method+'+candidate_generation'}]
            by[c['value']]=c
    p.candidates=list(by.values())
    return p
