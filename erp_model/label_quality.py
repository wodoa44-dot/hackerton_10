"""Input/label consistency checks shared by training and its audit."""
from collections import defaultdict

def explicit_resistance(text):
    """Read an unambiguous explicit resistance, excluding part IDs and power fractions."""
    import math
    import re
    from .preprocess import normalize
    text = normalize(text).replace("Ω", "Ω")
    found = []
    for match in re.finditer(r"(?<![A-Z0-9./-])([0-9]*)([RKM])([0-9]+)(?![A-Z0-9.])", text):
        left, unit, right = match.groups()
        if left or unit == "R":
            found.append(float((left or "0") + "." + right) * {"R": 1, "K": 1000, "M": 1000000}[unit])
    for match in re.finditer(r"(?<![A-Z0-9./-])([0-9]+(?:\.[0-9]+)?)\s*(MOHMS?|KOHMS?|OHMS?|MΩ|KΩ|Ω|MR|KR|R|K|M)(?![A-Z0-9.])", text):
        number, unit = match.groups()
        factor = .001 if unit == "MR" else (1000000 if unit.startswith("M") else 1000 if unit.startswith("K") else 1)
        found.append(float(number) * factor)
    if not found:
        return None
    return found[0] if all(math.isclose(value, found[0], rel_tol=1e-9, abs_tol=1e-12) for value in found) else None


def conflicting_indices(indices, keys, values):
    by_key = defaultdict(set)
    for i in indices:
        by_key[keys[i]].add(values[i])
    bad = {key for key, labels in by_key.items() if len(labels) > 1}
    return {i for i in indices if keys[i] in bad}, len(bad)


def unambiguous_indices(keys, values):
    """Retain single-target groups; preserve multi-target source rows for review."""
    indices = list(range(len(values)))
    conflicts, _ = conflicting_indices(indices, keys, values)
    return [i for i in indices if i not in conflicts]
