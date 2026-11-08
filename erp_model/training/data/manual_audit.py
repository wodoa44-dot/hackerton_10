"""Compare source CSV codes with manual layouts without changing original records."""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from erp_model.encoders.rules.company_schema import CompanySchemas
from erp_model.encoders.preprocess import normalize

CAP_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(PF|NF|UF|MFD|MF)(?![A-Z])")


def capacitances(spec):
    text = normalize(spec).replace("Μ", "U")
    factors = {"PF": 1, "NF": 1000, "UF": 1e6, "MFD": 1e6, "MF": 1e6}
    return [float(m[1]) * factors[m[2]] for m in CAP_PATTERN.finditer(text)]


def decode_capacity(value):
    if re.fullmatch(r"\d{3}", value):
        return int(value[:2]) * 10 ** int(value[2])
    if re.fullmatch(r"(?:\dR\d|R\d{2})", value):
        return float(value.replace("R", "."))
    return None


def capacity_modes(row, parsed):
    if not parsed or parsed["category_id"] not in {"CE", "CM"}:
        return []
    value = decode_capacity(parsed["values"]["part_number"])
    inputs = capacitances(row["규격명"])
    if value is None or not inputs:
        return []
    close = lambda a, b: abs(a - b) <= max(1e-8, abs(b) * .005)
    return [mode for mode, scale in [("pf", 1), ("uf", 1e6)]
            if any(close(value * scale, physical) for physical in inputs)]


def capacity_policy(rows, book):
    counts = defaultdict(Counter)
    for row in rows:
        parsed = book.parse(row["품목코드"])
        for mode in capacity_modes(row, parsed):
            counts[parsed["category_id"]][mode] += 1
    return {category: {"mode": max(counts[category], key=counts[category].get) if counts[category] else "uf",
                       "evidence_counts": dict(counts[category]),
                       "scope": "retained_training_only",
                       "manual_mode": "uf"}
            for category in ["CE", "CM"]}


def audit(train_path, evaluation_path, out_dir):
    book = CompanySchemas()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    status_counts = Counter()
    by_file, by_category, conflicts = defaultdict(Counter), defaultdict(Counter), Counter()
    rows, samples, numeric = [], defaultdict(list), defaultdict(Counter)
    source_hashes = {}
    for path in [train_path, evaluation_path]:
        source_hashes[str(path)] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            for index, row in enumerate(csv.DictReader(stream), 2):
                result = book.inspect(row["품목코드"])
                status = result["status"]
                category = result["category_id"] or ""
                status_counts[status] += 1
                by_file[str(path)][status] += 1
                by_category[category or "outside_manual"][status] += 1
                for issue in result["issues"]:
                    conflicts[category + "." + issue["field"] + ":" + issue["reason"]] += 1
                parsed = book.parse(row["품목코드"])
                modes = capacity_modes(row, parsed)
                if parsed and category in {"CE", "CM"} and capacitances(row["규격명"]):
                    numeric[category]["+".join(modes) if modes else "value_conflict"] += 1
                reason = "; ".join(i["field"] + "=" + i["value"] + " (" + i["reason"] + ")" for i in result["issues"])
                item = {"source_file": str(path), "source_row": index, **row,
                        "manual_status": status, "manual_category": category,
                        "manual_pages": ",".join(map(str, result.get("manual_pages", []))),
                        "comparison_notes": reason, "capacity_mode": "+".join(modes),
                        **{slot: parsed["values"][slot] if parsed else "" for slot in
                           ["classification", "trait", "part_number", "case", "packing", "vendor"]}}
                rows.append(item)
                if len(samples[status]) < 4:
                    samples[status].append({key: item[key] for key in ["품목코드", "품목명", "규격명", "comparison_notes"]})
    report = {"manual_source": book.manual, "total_rows": len(rows),
              "status_counts": dict(status_counts),
              "by_file": {k: dict(v) for k, v in by_file.items()},
              "by_category": {k: dict(v) for k, v in by_category.items()},
              "field_conflicts": dict(conflicts), "capacity_encoding": {k: dict(v) for k, v in numeric.items()},
              "source_hashes": source_hashes, "samples": dict(samples),
              "interpretation": "layout_conflict means not eligible for manual-based slot labels; the original code may belong to a different valid customer scheme"}
    (out / "manual_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "manual_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = ["# 매뉴얼과 CSV 대조", "", f"원본 {len(rows):,}행을 보존한 채 코드표와 자리 배치를 대조했습니다.", "",
            "| 구분 | 행 수 |", "|---|---:|"]
    text.extend(f"| {key} | {value:,} |" for key, value in status_counts.items())
    text.extend(["", "## 적용 원칙", "",
        "matched는 코드표의 모든 필드와 고정 구분자를 통과하고 분해 후 원래 코드로 재조립되는 행입니다.",
        "layout_conflict는 매뉴얼 규칙과 맞지 않는 행입니다. 고객사/기존 관리 코드일 수 있으므로 원본을 고치거나 상세 학습 라벨로 사용하지 않습니다.",
        "outside_manual은 매뉴얼에 확인된 자리 규칙이 없는 형식입니다. 등록 코드 조회와 코드 분류에 사용할 수 있습니다.",
        "학습 선택에는 원본 training 중 평가 입력 중복을 제외한 데이터만 사용합니다. 전체 CSV 대조 결과를 검증 성능으로 사용하지 않습니다.",
        "", "## 문서 내부 불일치", ""])
    for item in book.manual.get("errata", []):
        text.append(f"- {item['page']}쪽: {item['conflict']}. {item['resolution']}")
    text.extend(["", "## 전해콘덴서 용량 표기", "", json.dumps(report["capacity_encoding"], ensure_ascii=False),
                 "매뉴얼은 uF 기준입니다. 실사용 코드에는 pF와 uF 기준이 함께 존재하여, retained training에서 근거 건수를 집계한 기본 정책과 함께 출력합니다."])
    (out / "MANUAL_COMPARISON.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    report = audit("dataset/ERP_원재료_training.csv", "dataset/ERP_원재료_inference.csv", "artifacts/generation_v5")
    print(json.dumps({k: report[k] for k in ["total_rows", "status_counts", "capacity_encoding"]}, ensure_ascii=False))
