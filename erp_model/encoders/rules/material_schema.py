"""Evidence-backed family names and code scheme inventory; no guessed segment semantics."""
import re
from erp_model.encoders.preprocess import normalize,raw_span
from erp_model.shared.schema import labels

DIODE_SCHEME='DIODE_GENERAL_13'
FAMILY_RULES={
 'THERMISTOR':r'\b(?:NTC|PTC|POLYSWITCH)\b|\bTHERMISTOR\b|서미스터|써미스터',
 'RESISTOR':r'\bRES\b|\bRESISTER\b|\bRES[ -]?(?:M-OXIDE|ARRAY|SMD ARRAY|CEMENT|W-FUSIBLE|WIRE WOUND)\b|\bRESISTOR\b|\bRES[ -]?(?:CHIP|CARBON|METAL|NETWORK)\b|\bR[ -]?(?:CARBON|METAL|CHIP)\b|저항',
 'DIODE':r'\bDIODE\b|다이오드',
 'LED':r'\bLED\b|발광다이오드',
 'CAPACITOR':r'\bCAP[ -]?TANTAL\b|\bC[ -]?(?:CHIP|AL|TANTAL|MYLAR)\b|\bCAPACITOR\b|\bCAP[ -]?(?:CHIP|CERAMIC|ELECTRO|FILM)\b|\bC[ -]?(?:CERAMIC|ELECTRO|FILM)\b|콘덴서|커패시터',
 'TRANSISTOR':r'\bTR(?:[ -]CHIP)?\b|\bTRANSISTOR\b|\bMOSFET\b|\bFET\b|트랜지스터',
 'IC':r'\bIC\b|\bINTEGRATED CIRCUIT\b|집적회로',
 'CONNECTOR':r'\bCONNECTOR\b|\bCONN\b|커넥터|콘넥터',
 'PCB':r'\bPCB\b|인쇄회로기판',
 'HARNESS':r'\bHARNESS\b|\bWIRE AS\b|하네스|하니스',
 'RELAY':r'\bRELAY\b|릴레이',
 'VARISTOR':r'\bVARISTOR\b|바리스터',
 'FUSE':r'\bFUSE\b|퓨즈|휴즈',
 'INDUCTOR':r'\bINDUCTOR\b|\bCOIL\b|인덕터|코일',
 'TRANSFORMER':r'\bTRANSFORMER\b|변압기|트랜스포머',
 'SWITCH':r'\bSWITCH\b|스위치',
 'SENSOR':r'\bSENSOR\b|센서',
 'SCREW':r'\bSCREW\b|스크류|나사',
 'LABEL':r'\bLABEL\b|\bSTICKER\b|라벨|스티커',
 'PACKAGING':r'\bCARTON\b|\bPACKING BOX\b|포장박스|포장상자',
}
def family_evidence(prepared, name_only=False):
    hits=[]
    for family,pattern in FAMILY_RULES.items():
        for col,text in prepared.normalized.items():
            if name_only and col!='품목명':continue
            for m in re.finditer(pattern,text):
                hits.append({'source_column':col,'matched_text':raw_span(prepared.raw[col],m.start(),m.end()),'normalized_value':family,'transformation':'versioned_component_name_alias_v1','family':family})
    families={h['family'] for h in hits}
    if 'LED' in families:families.discard('DIODE')
    if 'THERMISTOR' in families:families.discard('RESISTOR');families.discard('SENSOR')
    # Assembly descriptions mention connectors/PCBs in the spec. The name is authoritative for silver labels.
    return sorted(families),hits

def silver_family(prepared):
    families,evidence=family_evidence(prepared,name_only=True)
    return (families[0] if len(families)==1 else None),evidence

def code_scheme(code):
    original=str(code);upper=original.upper().strip();lab,valid,reasons=labels(upper)
    if valid:
        return {'code_scheme_id':DIODE_SCHEME,'length':len(original),'interpretation':'confirmed_rule','generation_supported':True,
                'rule_source':'User supplied general diode 1+1+7+2+1+1 code table','company_rule':None,'applicability':list(lab)}
    prefix_match=re.match(r'(?:\d{2}[A-Z]{2}|[A-Z]{1,4}|\d{3,4})',upper)
    prefix=prefix_match.group() if prefix_match else 'MIXED'
    return {'code_scheme_id':f'OBSERVED_{prefix}_L{len(original)}','length':len(original),'prefix':prefix,
            'interpretation':'opaque_observed_structure','generation_supported':False,
            'rule_source':'Observed code prefix/length only; segment meanings and issuer not verified',
            'company_rule':None,'review_reasons':reasons or ['No confirmed generation or segment mapping rules'],
            'required_information':['ERP/customer code scheme documentation','Issuer/customer identifier if required','Master-assigned sequence or part identifier if required']}
