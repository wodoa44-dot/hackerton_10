"""Original resistance reader shared by numeric auditing and runtime overrides."""
import re

def resistance(text):
    m=re.search(r"(?<![A-Z0-9.])(\d+(?:\.\d+)?)\s*(K|M|R|OHM)(?=[BDFGJK](?![A-Z])|(?:OHM)?(?![A-Z0-9]))",text)
    if m:return float(m[1])*{"K":1e3,"M":1e6,"R":1,"OHM":1}[m[2]]
    m=re.search(r"(?:^|[ ,])([BDFGJK])(?:급)?\s*[/ ]*\s*(\d+(?:\.\d+)?)(?![A-Z0-9.])",text)
    if m:return float(m[2])
    m=re.search(r"(?:^|[ ,])(\d+(?:\.\d+)?)\s+([BDFGJK])(?:$|[ ,])",text)
    return float(m[1]) if m else None
