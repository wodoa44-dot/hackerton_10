import re
SLOTS = ['classification', 'trait', 'part_number', 'case', 'packing', 'vendor']
CASES = '-- AD BD BS C4 D1 D3 D4 DB GJ KL KP M3 M4 MD MR P6 R1 S1 S2 S3 S5 SA SB SC T1 T2 T3 TA TB WB'.split()
ALLOWED = {'classification': ['D','OTHER'], 'trait': list('BFPRSUWZ'), 'case': CASES,
           'packing': list('-BMRT'), 'vendor': list('-DJTVY')}
PATTERN = re.compile(r'D[BFPRSUWZ][A-Z0-9-]{11}')
def labels(code):
    code = str(code).upper().strip()
    if PATTERN.fullmatch(code):
        d = dict(zip(SLOTS, ['D',code[1],code[2:9],code[9:11],code[11],code[12]]))
        reasons = [s+': invalid '+d[s] for s in ['case','packing','vendor'] if d[s] not in ALLOWED[s]]
        if not d['part_number'].strip('-'): reasons.append('empty part_number')
        return d, not reasons, reasons
    if code.startswith('DL') or not code.startswith('D'):
        return {'classification':'OTHER'}, False, []
    return {}, False, ['classification label requires review']
