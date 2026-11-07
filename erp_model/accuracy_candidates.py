"""Input-only identifier candidates with atomic tokenization and explicit transformations."""
import re
from .preprocess import candidate_features

WIDTHS = (4, 5, 6, 7, 8, 10)
ATOM = re.compile(r"[A-Z0-9]+(?:[-.][A-Z0-9]+)*")
PHYSICAL = re.compile(r"\d+(?:\.\d+)?(?:KV|MV|V|VAC|VDC|MA|UA|A|MW|KW|W|MM|CM|UF|NF|PF|MF|MFD|OHM|KOHM|HZ|KHZ|MHZ|PIN|POLE|P|L|C|R|K|M)(?:[BCDFGJKMZ])?")
PACKAGE = re.compile(r"(?:DO|SOD|SOT|TO|SC|SOIC|SOP|DIP)[-]?\d+[A-Z]*")
SIZES = {"0603","1005","1608","2012","3216","3225","3528","4520","4532","5025","5750","6032","6432","7343"}
COLORS = {"BLACK":"1","BLK":"1","RED":"2","YELLOW":"3","YEL":"3","BLUE":"4","BLU":"4","GREEN":"5","GRN":"5","WHITE":"6","WHT":"6","VIOLET":"9","NATURAL":"0"}


def candidates(normalized):
    found = {}
    text_all = " ".join(normalized.values())
    color_codes = {v for k,v in COLORS.items() if re.search(r"(?<![A-Z])"+k+r"(?![A-Z])", text_all)}

    def add(column, text, raw, start, value, width, method, left=0, right=0, joined=0):
        if not value or len(value)>width or not re.fullmatch(r"[A-Z0-9-]+",value) or not any(c.isdigit() for c in value):
            return
        value = value.ljust(width,"-")
        key = value
        item = candidate_features(text,raw,value,start,method,column,1 if method=="original" else 2)
        clean = value.rstrip("-")
        physical = bool(PHYSICAL.fullmatch(raw))
        item["features"].update({
            "field_width":width,"raw_length":len(raw),"clean_length":len(clean),
            "left_removed":left,"right_removed":right,"joined":joined,
            "source_spec":int(column=="규격명"),"raw_physical":int(physical),
            "metric_size":int(raw in SIZES),"numeric_token":int(raw.isdigit()),
            "short_token":int(len(raw)<4),"raw_package":int(bool(PACKAGE.fullmatch(raw))),
            "packing_token":int(bool(re.fullmatch(r"[FSBR]\d{1,2}T?",raw))),
            "full_token":int(left==right==0),"zero_padding":int(method=="left_zero_pad"),
            "hyphen_removed":int("compact" in method),"window":int("window" in method),
            "color_append":int("color" in method),"physical_composition":int("current" in method),
        })
        if key not in found or (left+right < found[key]["features"]["left_removed"]+found[key]["features"]["right_removed"]):
            item["origins"]=[{k:item[k] for k in ["source_column","matched_text","normalized_value","transformation"]}]
            found[key]=item

    for column,text in normalized.items():
        for match in ATOM.finditer(text):
            raw=match.group()
            if PHYSICAL.fullmatch(raw) or PACKAGE.fullmatch(raw) or re.fullmatch(r"(?:X\dR|C0G|NP0|NPO|Y5[PVUR]|Z5U)",raw):
                continue
            variants=[(raw,"original"),(re.sub(r"[-.]","",raw),"compact")]
            if re.match(r"IN\d{3,}",raw):variants.append(("1N"+raw[2:],"ocr_I_to_1"))
            for value,method in variants:
                for width in WIDTHS:
                    add(column,text,raw,match.start(),value,width,method)
                    if len(value)>width:
                        for left in range(min(6,len(value)-width)+1):
                            add(column,text,raw,match.start(),value[left:left+width],width,method+"_window",left,len(value)-left-width)
                    # Retain a whole shorter body after removing a manufacturer prefix/suffix.
                    for left in range(0,min(5,len(value)-3)):
                        for right in range(0,min(4,len(value)-left-2)):
                            body=value[left:len(value)-right if right else None]
                            if (left or right) and len(body)<=width and len(body)>=4:
                                add(column,text,raw,match.start(),body,width,method+"_trim",left,right)
                    compact=value.replace("-","")
                    if compact.isdigit() and len(compact)>=5 and len(compact)<width:
                        add(column,text,raw,match.start(),compact.zfill(width),width,"left_zero_pad")
                    if color_codes and column=="규격명" and len(compact)>=5:
                        for color in color_codes:
                            add(column,text,raw,match.start(),compact+color,width,method+"_color")
        if re.search(r"FUSE|퓨즈|휴즈",normalized["품목명"]):
            current=re.search(r"(?:^|[\s,;])(\d+(?:\.\d+)?)\s*(MA|A)(?![A-Z])",text)
            if current:
                amps=float(current[1])/(1000 if current[2]=="MA" else 1)
                cents=round(amps*100)
                code=str(cents//100)+"A"+str(cents%100).zfill(2)
                series=re.findall(r"(?<![A-Z0-9])(?:0?(\d{3})\d|([A-Z]{2,3}-?\d[A-Z]?))",text)
                stems={a or b.replace("-","") for a,b in series}
                stems.update(re.findall(r"\b(MXP|MXEP|MXF|SR5H?|TE5|SS5)\b",text))
                for stem in stems:
                    for width in (7,8,10):
                        for value in [stem+code,code+stem]:
                            add(column,text,current[0],current.start(),value,width,"current_series",joined=1)
    # Retain the old input-only coverage while adding cleaner atomic candidates.
    from .preprocess import identifier_candidates
    for item in identifier_candidates(normalized):
        if item["value"] in found:
            continue
        raw=item["matched_text"]; clean=item["value"].rstrip("-")
        item["features"].update({"raw_length":len(raw),"clean_length":len(clean),
            "raw_physical":int(bool(PHYSICAL.fullmatch(raw))),
            "metric_size":int(raw in SIZES),"raw_package":int(bool(PACKAGE.fullmatch(raw))),
            "short_token":int(len(raw)<4),"legacy_candidate":1,
            "compound_units":int(bool(re.search(r"\d[WVA]/|\dV.*\dW",raw))),
            "packing_token":int(bool(re.fullmatch(r"[FSBR]\d{1,2}T?",raw)))})
        found[item["value"]]=item
    return list(found.values())
