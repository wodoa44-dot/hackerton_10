"""Stateless, evidence-preserving shared preprocessing. Never sees ERP labels."""
import re
import unicodedata
from dataclasses import dataclass

VERSION = '1.0'
DASHES = str.maketrans({c:'-' for c in '‐‑‒–—―−﹘﹣－'})
def normalize(s):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(s or '')).translate(DASHES).replace('µ','U').replace('μ','U').upper()).strip()

def raw_span(raw, start, end):
    # Character map through stateless normalization, preserving raw evidence.
    chars=[]; positions=[]
    for i,c in enumerate(raw):
        for ch in unicodedata.normalize('NFKC',c).translate(DASHES).replace('µ','U').replace('μ','U').upper():
            if ch.isspace():
                if not chars or chars[-1]==' ': continue
                chars.append(' ');positions.append(i)
            else:chars.append(ch);positions.append(i)
    if chars and chars[-1]==' ':chars.pop();positions.pop()
    if start>=len(positions) or end<=start:return ''
    return raw[positions[start]:positions[min(end-1,len(positions)-1)]+1]

PATTERNS = {
 'classification': {'diode':r'DIODE|다이오드', 'led':r'\bLED\b', '1n':r'\b1N\d', 'bzx':r'\bBZX', 'sod':r'\bSOD[ -]?\d'},
 'trait': {'bridge':r'BRIDGE|브리지|브릿지|\b(?:GBU|KBL|KBP|KBU|DB|MB)\d', 'tvs':r'\bTVS\b|\b(?:SMBJ|SMCJ|P6KE)',
           'zener':r'ZENER|제너|\b(?:BZX|BZT|1N52|2EZ)', 'schottky':r'SCH?OTTKY|쇼트키|\b(?:BAT|SS|SB|SK)\d',
           'ultrafast':r'ULTRA[ -]?FAST|\b(?:UF|SF|HER)\d', 'rectifier':r'RECTIFIER|\b1N400\d', 'switching':r'\b(?:1N4148|BAV|BAS)'},
 'case': {x:pat for x,pat in [('DO35',r'\bDO[ -]?35\b'),('DO41',r'\bDO[ -]?41\b'),('DO201',r'\bDO[ -]?201(?:AD)?\b'),
          ('SOT23',r'\bSOT[ -]?23\b'),('SOD123',r'\bSOD[ -]?123\b'),('SOD323',r'\bSOD[ -]?323\b'),
          ('MELF',r'\b(?:MINI[ -]?)?MELF\b'),('SMA',r'\bSMA\b'),('SMB',r'\bSMB\b'),('SMC',r'\bSMC\b'),('TO220',r'\bTO[ -]?220\b')]},
 'packing': {'tape':r'\b(?:TAPING|TAPPING|TAPE)\b','tp':r'(?<![A-Z0-9])TP(?![A-Z0-9])','reel':r'\bREEL\b','bulk':r'\bBULK\b','forming':r'\bFORMING\b'},
 'vendor': {'D':r'\bDEASAN\b|대산','J':r'\bJGD\b','T':r'\bTSC\b|TAIWAN\s+SEMI(?:CONDUCTOR)?',
            'V':r'\bVISHAY\b','Y':r'\bSAMYANG\b|삼양',
            'PANJIT':r'\bPANJIT\b','DACHANG':r'\bDACHANG\b|대창','ON':r'\bON\s+SEMI(?:CONDUCTOR)?\b|\bONSEMI\b',
            'ROHM':r'\bROHM\b','DIODES':r'\bDIODES\s+INC\b','SUNGHO':r'성호반도체','ST':r'\bSTMICROELECTRONICS\b'}
}
UNITS = re.compile(r'(?<![A-Z0-9])([0-9]+(?:\.[0-9]+)?)\s*(KV|MV|V|MA|UA|A|MW|KW|W|MM|CM|UM|OHM|UF|NF|PF|HZ|KHZ|MHZ)(?![A-Z])')
TOKEN = re.compile(r'[A-Z0-9]+(?:[-./][A-Z0-9]+)*-?')
@dataclass
class Prepared:
    raw: dict
    normalized: dict
    evidence: list
    structured: dict
    candidates: list
    key: tuple

def candidate_features(text, token, value, start, method, column, stage):
    clean=value.rstrip('-')
    return {'value':value,'source_column':column,'matched_text':token,'normalized_value':value,'transformation':method,'stage':stage,
            'features': {'length':len(clean),'alpha':sum(c.isalpha() for c in clean),'digits':sum(c.isdigit() for c in clean),
              'position':start/max(1,len(text)), 'source_name':int(column=='품목명'), 'exact':int(method=='original'),
              'join':int('join' in method), 'trim':int(method not in ['original','space_join']),
              'internal':int('internal' in method), 'hyphen':int('-' in clean),
              'looks_unit':int(bool(re.fullmatch(r'[0-9.]+(?:MA|A|MV|V|W|MM|CM|UF|PF|NF|OHM|HZ)',clean))),
              'looks_case':int(bool(re.fullmatch(r'(?:DO|SOD|SOT|TO|SC)[-]?[0-9A-Z]+',clean)))}}

def generate_candidates(normalized, width=7, allow_numeric=False):
    found={}
    def add(text, token, val, start, method, col, stage):
        val=val.strip(' ./')
        if not (2<=len(val)<=width and re.fullmatch(r'[A-Z0-9-]+',val) and (allow_numeric or any(c.isalpha() for c in val)) and any(c.isdigit() for c in val)): return
        val=val.ljust(width,'-')
        item=candidate_features(text,token,val,start,method,col,stage)
        if val not in found: found[val]=dict(item, origins=[{k:item[k] for k in ['source_column','matched_text','normalized_value','transformation']}])
        else:
            found[val]['stage']=min(stage,found[val]['stage'])
            origin={k:item[k] for k in ['source_column','matched_text','normalized_value','transformation']}
            if origin not in found[val]['origins']: found[val]['origins'].append(origin)
            for k in ['exact','join','internal']: found[val]['features'][k]=max(found[val]['features'][k],item['features'][k])
    for col,text in normalized.items():
        tokens=list(TOKEN.finditer(text)); sources=[(m.group(),m.start(),'original') for m in tokens]
        for a,b in zip(tokens,tokens[1:]):
            if text[a.end():b.start()].isspace() and ((a.group().isalpha() and b.group().isdigit()) or (a.group().isdigit() and b.group().isalpha())):
                sources.append((a.group()+b.group(),a.start(),'space_join'))
        for token,start,method in sources:
            variants=[(token,method)]
            for val,tag in [(token.replace('-',''),'hyphen_removed'),(re.sub(r'[-./]','',token),'punctuation_removed')]:
                if val!=token: variants.append((val,method+'+'+tag))
            for val,tag in variants:
                stage=1 if tag=='original' else 2
                add(text,token,val,start,tag,col,stage)
                for length in range(2,width+1):
                    if len(val)>length:
                        add(text,token,val[:length],start,tag+'+prefix',col,2)
                        add(text,token,val[-length:],start,tag+'+suffix',col,2)
                # Keep numeric/suffix tail, shorten the leading alphabetic segment.
                match=re.fullmatch(r'([A-Z]+)([0-9].*)',val)
                if match and len(val)>width:
                    letters,tail=match.groups()
                    if len(tail)<width: add(text,token,letters[:width-len(tail)]+tail,start,tag+'+shorten_leading_alpha',col,2)
                for left in (1,2):
                    for right in range(1,5):
                        inner=val[left:-right]
                        if 4<=len(inner)<=width: add(text,token,inner,start,tag+'+internal',col,3)
    return list(found.values())

def preprocess(name, spec=''):
    raw={'품목명':str(name or ''),'규격명':str(spec or '')}; norm={k:normalize(v) for k,v in raw.items()}
    evidence=[]; structured={s:{} for s in PATTERNS}; structured['part_number']={}
    for slot,pats in PATTERNS.items():
        for key,pattern in pats.items():
            hits=[]
            for col,text in norm.items():
                for m in re.finditer(pattern,text):
                    matched=raw_span(raw[col], m.start(), m.end())
                    hit={'slot':slot,'source_column':col,'matched_text':matched,'normalized_value':key,'transformation':'NFKC+uppercase+dash+alias_regex','normalized_match':m.group()}
                    evidence.append(hit);hits.append(hit)
            structured[slot][key]=len(hits)
    for col,text in norm.items():
        for m in UNITS.finditer(text):
            unit=m.group(2); number=float(m.group(1)); factor={'KV':1000,'MV':.001,'MA':.001,'UA':.000001,'MW':.001,'KW':1000,'CM':10,'UM':.001}.get(unit,1)
            base={'KV':'V','MV':'V','MA':'A','UA':'A','MW':'W','KW':'W','CM':'MM','UM':'MM'}.get(unit,unit)
            evidence.append({'slot':'physical','source_column':col,'matched_text':raw_span(raw[col],m.start(),m.end()),'normalized_value':str(number*factor)+base,'transformation':'regex+SI conversion'})
            for slot in ['trait','case']:
                structured[slot]['physical_'+base]=max(structured[slot].get('physical_'+base,0),number*factor)
        for m in re.finditer(r'\b\d+(?:\.\d+)?\s*[*X×]\s*\d+(?:\.\d+)?(?:\s*[*X×]\s*\d+(?:\.\d+)?)?',text):
            evidence.append({'slot':'physical','source_column':col,'matched_text':raw_span(raw[col],m.start(),m.end()),'normalized_value':m.group().replace(' ','').replace('×','X').replace('*','X'),'transformation':'dimension_regex'})
            structured['case']['dimension_present']=1
    candidates=generate_candidates(norm)
    for c in candidates:
        for origin in c['origins']:
            col=origin['source_column']; token=origin['matched_text']
            match=re.search(re.escape(token),norm[col])
            if match is None and 'space_join' in origin['transformation']:
                match=re.search(r'\s*'.join(re.escape(ch) for ch in token),norm[col])
            if match:origin['matched_text']=raw_span(raw[col],match.start(),match.end())
    return Prepared(raw,norm,evidence,structured,candidates,(norm['품목명'],norm['규격명']))

# The same pool is created from input text at training and inference time.
# Widths are not selected from a gold code or another head's prediction.
def identifier_candidates(normalized):
    pool = []
    for width in (4, 5, 6, 7, 8, 10):
        for candidate in generate_candidates(normalized, width=width, allow_numeric=True):
            clean = candidate["value"].rstrip("-")
            physical = re.fullmatch(r"[0-9.]+(?:MA|A|MV|V|KV|W|MM|CM|UF|PF|NF|OHM|HZ|K|M|R|PIN|P|L|T|C|%)", clean)
            package = re.fullmatch(r"(?:DO|SOD|SOT|TO|SC)[-]?[0-9A-Z]+", clean)
            if physical or package or re.fullmatch(r"(?:X[0-9]R|C0G|NP0|NPO|Y5[PVUR]|Z5U)", clean):
                continue
            candidate["features"]["field_width"] = width
            pool.append(candidate)
    return pool
