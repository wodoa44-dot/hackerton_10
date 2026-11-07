"""Code equivalence and independently readable numeric input constraints."""
import re,math
from .numeric_inputs import resistance
from .schema import SLOTS

def _detail(book,code):
    parsed=book.parse(code)
    return {"parts":[part for slot in SLOTS for part in book.parts(parsed["category_id"],slot,parsed["values"][slot]) or []]+book.layout(parsed["category_id"])[1]}

def decode_res(value):
    if "R" in value:return float(value.replace("R",".") or 0)
    if "M" in value:return float(value.replace("M","."))*1e-3
    if value.isdigit():return int(value[:-1])*10**int(value[-1])
    return None


def semantic_key(book,code):
    parsed=book.parse(code)
    if not parsed:return None
    detail=_detail(book,code);out=[]
    for part in detail["parts"]:
        value=part["value"]
        if part["key"]=="res":
            try:value=round(decode_res(value),10)
            except (ValueError,TypeError):return None
        out.append((part["key"],value))
    return parsed["category_id"],tuple(out)


def numeric_checks(book,code,p,capacity_mode=None, resistance_reader=resistance):
    """Only independently readable input numbers; missing evidence is unscored."""
    parsed=book.parse(code)
    if not parsed:return []
    category=parsed["category_id"];text=" ".join(p.normalized.values());detail=_detail(book,code)
    fields={f["k"]:f for f in book.categories[category]["fields"]}
    values={part["key"]:part["value"] for part in detail["parts"]};checks=[]
    def record(key,predicted,expected):
        if predicted is not None and expected is not None:
            checks.append({"field":key,"predicted":predicted,"input":expected,
                           "consistent":math.isclose(predicted,expected,rel_tol=1e-5,abs_tol=1e-9)})
    if "res" in values:
        try:record("res",decode_res(values["res"]),resistance_reader(text))
        except (ValueError,TypeError):pass
    if "cap" in values:
        found=re.findall(r"(?<![A-Z0-9.])(\d+(?:\.\d+)?)\s*(PF|NF|UF)(?![A-Z])",text)
        magnitudes={float(v)*{"PF":1,"NF":1e3,"UF":1e6}[u] for v,u in found}
        # CE/CM source codes have mixed unit conventions: do not guess their truth units.
        if len(magnitudes)==1 and (category not in {"CE","CM"} or capacity_mode in {"pf","uf"}):
            value=values["cap"]
            try:
                decoded=float(value.replace("R",".")) if "R" in value else int(value[:-1])*10**int(value[-1])
                if category in {"CE","CM"} and capacity_mode=="uf":decoded*=1e6
                record("cap",decoded,next(iter(magnitudes)))
            except (ValueError,TypeError):pass
    if "volt" in values:
        explicit={float(v) for v in re.findall(r"(?<![A-Z0-9.])(\d+(?:\.\d+)?)\s*V(?:AC|DC)?(?![A-Z])",text)}
        label=next((o[1] for o in fields["volt"].get("o",[]) if o[0]==values["volt"]),"")
        m=re.search(r"(\d+(?:\.\d+)?)\s*V",label.upper())
        if len(explicit)==1 and m:record("volt",float(m[1]),next(iter(explicit)))
    return checks
