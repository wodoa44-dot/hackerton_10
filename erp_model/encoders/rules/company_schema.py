from erp_model.shared.paths import PROJECT_ROOT
"""Validated reversible layouts from the manual, with explicitly identified ERP extensions."""
import json
import re
from pathlib import Path

from erp_model.shared.schema import SLOTS

BOOK_PATH = PROJECT_ROOT / "config" / "company_rules.json"


def actual_prefix(code):
    text = str(code).strip().upper()
    match = re.match(r"[0-9]{2}|[A-Z]|[0-9]", text)
    return match.group() if match else text[:1]


def field_slot(category, key):
    if key == "pack":
        return "packing"
    if key in {"vend", "ver", "origin"}:
        return "vendor"
    if key in {"pn", "sub", "res", "cap", "model", "seq"}:
        return "part_number"
    if key == "name" and category["id"] in {"J", "JACC"}:
        return "part_number"
    if key in {"shape", "case", "dia", "d", "len", "pitch", "type", "pin", "tip", "head", "slot"}:
        return "case"
    return "trait"


class CompanySchemas:
    def __init__(self, path=BOOK_PATH):
        self.document = json.loads(Path(path).read_text(encoding="utf-8"))
        self.categories = {r["id"]: r for r in self.document["categories"] if r["id"] != "FREE"}
        self.source = self.document["source"]
        manual_path = Path(path).with_name("manual_source.json")
        self.manual = json.loads(manual_path.read_text(encoding="utf-8")) if manual_path.exists() else {}

    def structural_category(self, code):
        text = str(code).strip().upper()
        if not re.fullmatch(r"[A-Z0-9-]+", text):
            return None
        if re.fullmatch(r"J[A-Z0-9]{2}[A-Z0-9]{6}S[0-9]{3}", text):
            return "JACC"
        matches = [c for c in self.categories.values()
                   if c["id"] != "JACC" and text.startswith(c["prefix"])
                   and len(text) == len(c["prefix"]) + sum(f["n"] for f in c["fields"])]
        return max(matches, key=lambda c: len(c["prefix"]))["id"] if matches else None

    def field_error(self, category_id, field, value):
        key, kind = field["k"], field["t"]
        if len(value) != field["n"] or not re.fullmatch(r"[A-Z0-9-]+", value):
            return "length_or_characters"
        if kind == "fix":
            return None if value == field["def"] else "fixed_separator"
        if kind == "sel":
            options = {option[0] for option in field.get("o", [])}
            if field.get("allow_missing"):
                options.add("-" * field["n"])
            return None if value in options else "not_in_code_table"
        if kind == "cap":
            return None if re.fullmatch(r"(?:[0-9]{3}|[0-9]R[0-9]|R[0-9]{2})", value) else "invalid_capacitance"
        if kind in {"res3", "res4"}:
            return None if re.fullmatch(r"(?:[0-9]+|[0-9]*[RM][0-9]*)", value) and any(c.isdigit() for c in value) else "invalid_resistance"
        if kind == "serial":
            return None if value.isdigit() else "invalid_serial"
        if category_id == "H" and key in {"dia", "len"}:
            return None if value.isdigit() else "invalid_dimension"
        if key in {"pn", "sub", "name", "model", "shape"} and not value.strip("-"):
            return "missing_identifier"
        return None

    def validate(self, category_id, code):
        if category_id not in self.categories:
            return [{"field": "_category", "reason": "unknown_category"}]
        category = self.categories[category_id]
        length = len(category["prefix"]) + sum(f["n"] for f in category["fields"])
        if not isinstance(code, str) or len(code) != length or not code.startswith(category["prefix"]):
            return [{"field": "_code", "reason": "prefix_or_length"}]
        issues, pos = [], len(category["prefix"])
        for field in category["fields"]:
            value = code[pos:pos + field["n"]]
            error = self.field_error(category_id, field, value)
            if error:
                issues.append({"field": field["k"], "label": field["l"], "value": value,
                               "slot": field_slot(category, field["k"]), "reason": error})
            pos += field["n"]
        return issues

    def inspect(self, code):
        text = str(code).strip().upper()
        category_id = self.structural_category(text)
        if category_id is None:
            return {"status": "outside_manual", "category_id": None, "issues": [],
                    "actual_prefix": actual_prefix(text), "length": len(text)}
        issues = self.validate(category_id, text)
        return {"status": "layout_conflict" if issues else "matched",
                "category_id": category_id, "issues": issues,
                "actual_prefix": actual_prefix(text), "length": len(text),
                "manual_pages": self.manual.get("page_map", {}).get(category_id, []),
                "schema_origin": self.categories[category_id].get("schema_origin", "csv_extension" if category_id in {"S", "JACC"} else "manual")}

    def find(self, code):
        audit = self.inspect(code)
        return audit["category_id"] if audit["status"] == "matched" else None

    def layout(self, category_id):
        category = self.categories[category_id]
        prefix = category["prefix"]
        major_length = len(actual_prefix(prefix))
        layout = {slot: [] for slot in SLOTS}
        layout["classification"].append({"start": 0, "end": major_length, "key": "_major", "label": "분류"})
        if len(prefix) > major_length:
            layout["trait"].append({"start": major_length, "end": len(prefix), "key": "_subtype", "label": "중분류"})
        fixed, position = [], len(prefix)
        for field in category["fields"]:
            span = {"start": position, "end": position + field["n"], "key": field["k"], "label": field["l"]}
            if field["t"] == "fix":
                fixed.append({**span, "value": field["def"]})
            else:
                layout[field_slot(category, field["k"])].append(span)
            position += field["n"]
        return layout, fixed, position

    def parse(self, code):
        text = str(code).strip().upper()
        category_id = self.find(text)
        if category_id is None:
            return None
        layout, _, _ = self.layout(category_id)
        values = {slot: "".join(text[s["start"]:s["end"]] for s in spans) for slot, spans in layout.items()}
        if self.assemble(category_id, values) != text:
            return None
        return {"category_id": category_id, "values": values,
                "manual_pages": self.manual.get("page_map", {}).get(category_id, [])}

    def parts(self, category_id, slot, value):
        spans = self.layout(category_id)[0][slot]
        if not isinstance(value, str) or len(value) != sum(s["end"] - s["start"] for s in spans):
            return None
        result, offset = [], 0
        for item in spans:
            size = item["end"] - item["start"]
            result.append({**item, "value": value[offset:offset + size]})
            offset += size
        return result

    def valid_slot(self, category_id, slot, value):
        parts = self.parts(category_id, slot, value)
        if parts is None:
            return False
        category = self.categories[category_id]
        fields = {f["k"]: f for f in category["fields"]}
        for part in parts:
            if part["key"] == "_major" and part["value"] != actual_prefix(category["prefix"]):
                return False
            if part["key"] == "_subtype" and part["value"] != category["prefix"][len(actual_prefix(category["prefix"])):]:
                return False
            if part["key"] in fields and self.field_error(category_id, fields[part["key"]], part["value"]):
                return False
        return True

    def assemble(self, category_id, values):
        layout, fixed, length = self.layout(category_id)
        chars = [None] * length
        for slot, spans in layout.items():
            if not spans:
                continue
            if not self.valid_slot(category_id, slot, values.get(slot)):
                return None
            for item in self.parts(category_id, slot, values[slot]):
                chars[item["start"]:item["end"]] = item["value"]
        for item in fixed:
            chars[item["start"]:item["end"]] = item["value"]
        if any(v is None for v in chars):
            return None
        code = "".join(chars)
        return code if not self.validate(category_id, code) else None

    def allowed_categories(self, family):
        groups = {
            "DIODE": ["D"], "LED": ["DL"],
            "RESISTOR": [key for key in self.categories if key.startswith("R")],
            "CAPACITOR": [key for key in self.categories if key.startswith("C")],
            "IC": ["S"], "TRANSISTOR": ["S"], "THERMISTOR": ["17"],
            "SWITCH": ["18"], "CONNECTOR": ["19"], "FUSE": ["25"],
            "VARISTOR": ["2590", "2591"], "SCREW": ["H"], "LABEL": ["98"], "PACKAGING": ["98"],
        }
        return groups.get(family)
