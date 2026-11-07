"""Six independent outputs, validated manual assembly, and separate registered-code lookup."""
import atexit
import hashlib
import json
import re
import select
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.preprocessing import normalize as normalize_matrix

from .company_schema import CompanySchemas, actual_prefix
from .material_schema import silver_family
from .preprocess import preprocess, identifier_candidates
from .registry import Registry
from .schema import SLOTS


class CompanyBridge:
    def __init__(self, node, rules_path=None):
        self.node = (shutil.which(node) if node else None) or shutil.which("node")
        self.rules_path = rules_path
        self.process = None
        self.lock = threading.Lock()
        atexit.register(self.close)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None

    def query(self, payload):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                if not self.node:
                    raise RuntimeError("Company code generator requires Node.js")
                args = [self.node, str(Path(__file__).with_name("company_bridge.cjs"))]
                if self.rules_path:
                    args.append(str(self.rules_path))
                self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1)
            self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            if not select.select([self.process.stdout], [], [], 15)[0]:
                self.close()
                raise RuntimeError("Company generator response timeout")
            line = self.process.stdout.readline()
            if not line:
                self.close()
                raise RuntimeError("Company generator stopped before returning a result")
            response = json.loads(line)
            if response.get("error"):
                raise RuntimeError(response["error"])
            return response["proposals"]


class GenerationState:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        self.shared = joblib.load(self.directory / "preprocessor.joblib")["text"]
        reference = joblib.load(self.directory / "reference.joblib")
        self.reference_rows, self.reference_matrix = reference["rows"], reference["matrix"]
        self.values = json.loads((self.directory / "scheme_values.json").read_text(encoding="utf-8"))
        frozen_rules = self.directory / "company_rules.json"
        self.book = CompanySchemas(frozen_rules) if frozen_rules.exists() else CompanySchemas()
        if frozen_rules.exists() and self.manifest.get("rules_sha256"):
            if hashlib.sha256(frozen_rules.read_bytes()).hexdigest() != self.manifest["rules_sha256"]:
                raise ValueError("Model and code-rule snapshot differ")
        self.registry = Registry.load(self.directory / "operational_master.json")
        self.models, self.model_lock = {}, threading.Lock()
        self._encoder_bundle, self._encoder_lock = None, threading.Lock()
        self.bridge = CompanyBridge(self.manifest.get("node_executable"), frozen_rules if frozen_rules.exists() else None)

    def model(self, slot):
        with self.model_lock:
            if slot not in self.models:
                model = joblib.load(self.directory / (slot + ".joblib"))
                model.feature.text = self.shared
                if getattr(model.feature, "encoder_dimensions", 0):
                    if not self.manifest.get("encoder_augmentation"):
                        raise ValueError("Augmented tree artifact has no encoder manifest")
                    # Batch CSV/evaluation callers can invoke the raw bundle
                    # before prepare(); feature construction must work there too.
                    model.feature._encoder_provider = self.ensure_embeddings
                self.models[slot] = model
            return self.models[slot]

    def ensure_embeddings(self, prepared, batch_size=64):
        """Attach input-only vectors once; TF-IDF retrieval remains unchanged."""
        config = self.manifest.get("encoder_augmentation")
        if not config:
            return
        directory = (self.directory / config["directory"]).resolve()
        signature = str(directory)
        with self._encoder_lock:
            missing = [p for p in prepared if
                       getattr(p, "_shared_char_embedding_signature", None) != signature]
            if not missing:
                return
            if self._encoder_bundle is None:
                from .encoder_runtime import EncoderBundle
                self._encoder_bundle = EncoderBundle(directory, device=config.get("runtime_device", "cpu"))
            vectors = np.asarray(self._encoder_bundle.encode_vectors(missing, batch_size=batch_size), dtype=np.float32)
            expected = int(config["embedding_dimension"])
            if vectors.shape != (len(missing), expected) or not np.isfinite(vectors).all():
                raise ValueError("Shared character encoder returned invalid embeddings")
            for row, vector in zip(missing, vectors):
                row._shared_char_embedding = vector
                row._shared_char_embedding_signature = signature

    def text_features(self, prepared):
        cached = getattr(prepared, "_text_cache", None)
        if cached is None or cached[0] != id(self.shared[0]):
            matrix = hstack([v.transform([prepared.normalized[c]]) for v, c in
                             zip(self.shared, ["품목명", "규격명"])], format="csr")
            prepared._text_cache = (id(self.shared[0]), matrix)
        return prepared._text_cache[1]

    def neighbors(self, prepared, count=24):
        scores = (normalize_matrix(self.text_features(prepared)) @ self.reference_matrix.T).toarray().ravel()
        indices = np.argsort(-scores, kind="stable")[:count]
        return [dict(self.reference_rows[i], similarity=float(scores[i])) for i in indices if scores[i] > 0]


@lru_cache(maxsize=2)
def load_state(directory):
    return GenerationState(directory)


def candidates_for(slot, prepared, state):
    cfg = state.manifest["slots"][slot]
    if cfg.get("model_status") != "trained":
        return []
    model = state.model(slot)
    if slot == "part_number":
        ranking = getattr(prepared, "_generation_ranking", None)
        if ranking is None:
            ranking = model.rankings([prepared])[0]["ensemble"]
        order = np.argsort(ranking, kind="stable")
        return [{"value": prepared.candidates[i]["value"], "score": float(ranking[i]),
                 "score_type": "mean_rank_lower_is_better",
                 "input_origin": {k: prepared.candidates[i][k] for k in
                                  ["source_column", "matched_text", "transformation"]}} for i in order]
    probabilities = getattr(prepared, "_generation_probabilities", {}).get(slot)
    if probabilities is None:
        probabilities = model.probabilities([prepared], cfg["classes"])[cfg["model"]][0]
    order = np.argsort(-probabilities, kind="stable")
    return [{"value": cfg["classes"][i], "score": float(probabilities[i]),
             "score_type": "probability_uncalibrated"} for i in order]


class GenerationPredictor:
    def __init__(self, directory="artifacts/generation_v5", slot=None, registry=None):
        self.state = load_state(str(Path(directory).resolve()))
        self.slots = [slot] if slot else list(SLOTS)
        self.registry = registry if registry is not None else self.state.registry

    def prepare(self, name, spec, scheme=None, prepared=None):
        p = prepared if prepared is not None else preprocess(name, spec)
        signature = (id(self.state), id(self.registry), name, spec, scheme)
        if getattr(p, "_generation_signature", None) == signature:
            return p, p.generation_context
        self.state.ensure_embeddings([p])
        self.state.text_features(p)
        if str(self.state.manifest.get("version", "")).startswith("5") and not getattr(p, "_identifier_pool_ready", False):
            p.candidates = identifier_candidates(p.normalized)
            p._identifier_pool_ready = True
        family, evidence = silver_family(p)
        family = family or "UNKNOWN"
        matches = [m for m in self.registry.search(p, family)
                   if re.match(r"[A-Z0-9]", m["code"].upper())]
        near = self.state.neighbors(p)
        book = self.state.book
        plan = None
        codes = {m["code"].upper() for m in matches}
        if len(codes) == 1:
            match = matches[0]
            parsed = book.parse(match["code"])
            plan = {"code": match["code"].upper(), "category_id": parsed["category_id"] if parsed else None,
                    "source": match["code_source"], "auto": match["code_source"] == "registry_exact",
                    "sources": {}, "segments": [], "issues": [], "registered": match}
        elif matches:
            prefixes = {actual_prefix(m["code"]) for m in matches}
            plan = {"code": None, "category_id": None, "source": "registry_ambiguous", "auto": False,
                    "sources": {}, "segments": [], "issues": ["등록 코드가 여러 개입니다. 원문을 확인하고 후보를 선택하세요."],
                    "observed_prefix": next(iter(prefixes)) if len(prefixes) == 1 else None}
        else:
            category = "D" if scheme == "DIODE_GENERAL_13" else scheme
            if category and category.startswith("COMPANY_"):
                category = category[len("COMPANY_"):]
            if category and category not in book.categories:
                category = None
            allowed = [category] if category else book.allowed_categories(family)
            if allowed is None:
                text = " ".join(p.normalized.values())
                allowed = [c["id"] for c in book.categories.values()
                           if c.get("itemName") and c["itemName"].upper() in text]
            proposals, preparation_issues = [], []
            if allowed:
                valid_neighbors = [r for r in near if book.parse(r["품목코드"])]
                neighbors = [{"row": [r["품목코드"], r["품목명"], r["규격명"]], "score": r["similarity"]}
                             for r in valid_neighbors]
                try:
                    proposals = self.state.bridge.query({"text": " ".join(p.normalized.values()),
                        "neighbors": neighbors, "allowed": allowed, "category": category,
                        "capacity_policy": self.state.manifest.get("capacity_policy", {})})
                    proposals.sort(key=lambda item: (item["valid"], item["score"]), reverse=True)
                except Exception as error:
                    preparation_issues.append("공통 규칙 해석 오류: " + type(error).__name__)
            if proposals:
                proposal = proposals[0]
                if proposal["id"] == "D":
                    codes_by_trait = {"bridge": "B", "tvs": "P", "zener": "Z", "schottky": "S",
                                      "ultrafast": "U", "switching": "W", "rectifier": "R"}
                    explicit = {v for k, v in codes_by_trait.items() if p.structured["trait"].get(k, 0)}
                    if len(explicit) == 1:
                        proposal["code"] = proposal["code"][:1] + next(iter(explicit)) + proposal["code"][2:]
                        proposal["sources"]["char"] = "rule"
                valid = proposal["valid"] and not book.validate(proposal["id"], proposal["code"])
                plan = {"code": proposal["code"] if valid else None, "category_id": proposal["id"],
                        "source": "model_assembled", "auto": False, "sources": proposal["sources"],
                        "segments": proposal["segments"], "issues": proposal["issues"] + preparation_issues,
                        "proposal_code": proposal["code"], "capacity_mode": proposal.get("capacity_mode"),
                        "default_fields": proposal.get("defaults", [])}
            elif allowed and len(allowed) == 1:
                plan = {"code": None, "category_id": allowed[0], "source": "model_assembled", "auto": False,
                        "sources": {}, "segments": [], "issues": preparation_issues or ["입력에서 필요한 코드 구간을 확인해야 합니다."]}
            if (plan is None or plan["code"] is None) and not preparation_issues:
                compatible = [r for r in near if family == "UNKNOWN" or silver_family(preprocess(r["품목명"], r["규격명"]))[0] == family]
                if compatible and compatible[0]["similarity"] >= .55 and (plan is None or not plan.get("category_id")):
                    row = compatible[0]
                    parsed = book.parse(row["품목코드"])
                    plan = {"code": row["품목코드"].upper(), "category_id": parsed["category_id"] if parsed else None,
                            "source": "reference_candidate", "auto": False, "sources": {}, "segments": [],
                            "issues": ["유사한 등록 품목의 코드 후보입니다. 원문과 입력 차이를 확인하세요."], "reference": row}
        if plan is None:
            plan = {"code": None, "category_id": None, "source": None, "auto": False,
                    "sources": {}, "segments": [], "issues": ["입력에서 품번 또는 적용 가능한 코드 규칙을 확인해야 합니다."]}
        category_id = plan["category_id"]
        if category_id:
            layout, fixed, length = book.layout(category_id)
            slot_layout = {slot: {"length": sum(s["end"] - s["start"] for s in spans), "spans": spans}
                           for slot, spans in layout.items()}
            scheme_id = "COMPANY_" + category_id
            scheme_title = book.categories[category_id]["title"]
        else:
            slot_layout, fixed = {}, []
            length = len(plan["code"]) if plan["code"] else None
            prefix = actual_prefix(plan["code"]) if plan["code"] else (plan.get("observed_prefix") or "")
            scheme_id = "ERP_" + prefix + ("_L" + str(length) if length else "")
            scheme_title = "등록 품목코드"
        validation = book.inspect(plan["code"]) if plan["code"] else None
        context = {"slot": "context", "raw_input": p.raw, "normalized_input": p.normalized,
            "component_family": family, "family": {"value": family, "evidence": evidence},
            "code_scheme_id": scheme_id, "code_scheme_title": scheme_title,
            "code_scheme_candidates": ["COMPANY_" + k for k in book.allowed_categories(family) or []],
            "registry_candidates": matches, "registry_provenance": self.registry.provenance,
            "generation_supported": bool(category_id), "expected_code_length": length, "slot_layout": slot_layout,
            "fixed_segments": fixed, "schema_validation": validation,
            "manual_pages": book.manual.get("page_map", {}).get(category_id, []),
            "schema_origin": book.categories[category_id].get("schema_origin", "csv_extension" if category_id in {"S", "JACC"} else "manual") if category_id else "registered_only",
            "scheme_note": book.source if category_id else "등록된 코드 형식을 보존합니다. 구간 의미는 매뉴얼에서 확인되지 않았습니다.",
            "capacity_mode": plan.get("capacity_mode"), "capacity_policy": self.state.manifest.get("capacity_policy", {}).get(category_id),
            "required_information": plan["issues"]}
        if category_id:
            for field in book.categories[category_id]["fields"]:
                if field["k"] != "vend":
                    continue
                for option in field.get("o", []):
                    aliases = option[2].split(",") if len(option) > 2 else [option[1].split("(")[0]]
                    for alias in aliases:
                        alias = alias.strip().upper()
                        if len(alias) < 3 or alias == "미지정":
                            continue
                        for column, text in p.normalized.items():
                            if re.search(r"(?<![A-Z0-9])" + re.escape(alias) + r"(?![A-Z0-9])", text):
                                p.evidence.append({"slot": "vendor", "source_column": column, "matched_text": alias,
                                    "normalized_value": option[0] if option[0] in {"D", "J", "T", "V", "Y"} else alias,
                                    "erp_code": option[0], "transformation": "manual_vendor_alias"})
        p.generation_plan, p.generation_context, p._generation_signature = plan, context, signature
        return p, context

    def predict_slot(self, slot, p):
        context, plan, book = p.generation_context, p.generation_plan, self.state.book
        category_id = plan["category_id"]
        result = {"slot": slot, "value": None, "predicted_value": None, "status": "review",
            "candidates": [], "top_candidates": [], "evidence": [v for v in p.evidence if v["slot"] in {slot, "physical"}],
            "code_scheme_id": context["code_scheme_id"], "component_family": context["component_family"],
            "manual_pages": context["manual_pages"], "reason": "", "fields": []}
        try:
            if slot != "classification" and category_id is None:
                result.update(status="unsupported_schema", reason="이 등록 코드의 구간 의미가 매뉴얼에 확인되지 않아 해당 칸의 해석을 보류합니다.")
                return result
            if slot != "classification" and not book.layout(category_id)[0][slot]:
                result.update(status="not_applicable", reason="이 품목의 코드 규칙에는 해당 구간이 없습니다.")
                return result
            candidates = candidates_for(slot, p, self.state)
            registered = plan["source"] in {"registry_exact", "registry_candidate", "reference_candidate"}
            if slot == "classification":
                value = (actual_prefix(plan["code"]) if plan["code"] else plan.get("observed_prefix") or
                         (actual_prefix(book.categories[category_id]["prefix"]) if category_id else
                          candidates[0]["value"] if candidates else None))
                result["predicted_value"] = value
                selected = {"value": value, "score": None,
                            "score_type": "registered_prefix" if registered else "manual_schema_prefix"}
                if not registered and not category_id:
                    selected = next((c for c in candidates if c["value"] == value), selected)
                result["candidates"] = [selected] + [c for c in candidates if c["value"] != value][:7] if value else candidates[:8]
                result["reason"] = "등록 코드에서 읽은 실제 분류값입니다." if registered else "품목명·규격명과 코드 규칙으로 선택한 분류값입니다."
            elif category_id is None:
                result.update(status="unsupported_schema", reason="이 등록 코드의 구간 의미가 매뉴얼에 확인되지 않아 해당 칸의 해석을 보류합니다.")
                return result
            else:
                spans = book.layout(category_id)[0][slot]
                if not spans:
                    result.update(status="not_applicable", reason="이 품목의 코드 규칙에는 해당 구간이 없습니다.")
                    return result
                candidates = [c for c in candidates if book.valid_slot(category_id, slot, c["value"])]
                if slot == "part_number":
                    direct = [c for c in candidates if c.get("input_origin", {}).get("transformation") in
                              {"original", "space_join", "original+hyphen_removed", "original+punctuation_removed"}]
                    if direct:
                        candidates = direct
                code = plan.get("code") or plan.get("proposal_code") or ""
                value = "".join(code[s["start"]:s["end"]] for s in spans) if len(code) == context["expected_code_length"] else None
                if value is not None and not book.valid_slot(category_id, slot, value):
                    value = None
                sources = plan["sources"]
                invalid_explicit = {s["k"] for s in plan.get("segments", [])
                                    if (not s.get("ok", True) or "자동 축약" in s.get("msg", ""))
                                    and sources.get(s["k"]) in {"rule", "manual_transform"}}
                if any(span["key"] in invalid_explicit for span in spans):
                    result.update(candidates=candidates[:8], top_candidates=candidates[:8], requires_input_review=True,
                                  reason="입력값을 매뉴얼의 구간 길이와 코드표로 표현할 수 없습니다. 해당 값을 확인하세요.")
                    return result
                field_sources = {}
                if not registered:
                    model_value = candidates[0]["value"] if candidates else None
                    model_parts = book.parts(category_id, slot, model_value) if model_value else None
                    if model_parts:
                        pieces = []
                        for item in model_parts:
                            key = item["key"]
                            explicit = sources.get(key) in {"rule", "manual_transform"} or key == "_subtype"
                            original = code[item["start"]:item["end"]]
                            pieces.append(original if explicit and len(original) == item["end"] - item["start"] else item["value"])
                            field_sources[key] = "input_rule" if explicit else "independent_model"
                        proposed = "".join(pieces)
                        if book.valid_slot(category_id, slot, proposed):
                            value = proposed
                if value is not None:
                    result["predicted_value"] = value
                    result["fields"] = book.parts(category_id, slot, value) or []
                    for field in result["fields"]:
                        field["source"] = "registered_code" if registered else field_sources.get(
                            field["key"], "input_rule" if sources.get(field["key"]) in {"rule", "manual_transform"} or field["key"] == "_subtype" else "rule_default")
                    selected = {"value": value, "score": None, "score_type": "registered_field" if registered else "manual_rule_and_independent_model"}
                    if not registered:
                        selected = next((c for c in candidates if c["value"] == value), selected)
                    candidates = [selected] + [c for c in candidates if c["value"] != value]
                result["candidates"] = candidates[:8]
                result["reason"] = "등록 코드에서 매뉴얼 규칙에 따라 읽은 값입니다." if registered else "입력 단서와 해당 칸의 독립 모델을 결합한 결과입니다."
                if slot == "vendor" and not registered:
                    vendors = {v["normalized_value"] for v in p.evidence if v["slot"] == "vendor"}
                    if len(vendors) > 1:
                        result["reason"] = "conflicting_manufacturer_evidence"
                    elif not vendors:
                        result["reason"] = "no_explicit_manufacturer_evidence"
            if plan["auto"] and result["predicted_value"] is not None:
                result.update(value=result["predicted_value"], status="confirmed", reason="등록된 품목명과 규격명이 정규화 후 일치합니다.")
            result["top_candidates"] = result["candidates"]
            return result
        except Exception as error:
            result.update(status="error", reason=type(error).__name__ + ": " + str(error))
            return result

    def assemble(self, results, p):
        plan, context = p.generation_plan, p.generation_context
        by_slot = {r["slot"]: r for r in results}
        values = {slot: r.get("predicted_value") for slot, r in by_slot.items()}
        code = plan["code"] if plan["source"] in {"registry_exact", "registry_candidate", "reference_candidate"} else None
        if plan["source"] == "model_assembled" and plan["category_id"]:
            if self.state.book.categories[plan["category_id"]].get("generation_enabled", True):
                code = self.state.book.assemble(plan["category_id"], values)
        issues = list(plan["issues"])
        if not code and plan["category_id"]:
            for slot, spans in self.state.book.layout(plan["category_id"])[0].items():
                if spans and values.get(slot) is None:
                    issues.append(slot + ": 입력 근거 또는 유효한 예측값이 필요합니다.")
        source = plan["source"] if code else None
        return {"slot": "assembly", "value": code if plan["auto"] else None,
            "display_code": code, "predicted_code": code, "predicted_values": values,
            "status": "confirmed" if plan["auto"] and code else "review",
            "auto_confirmed": bool(plan["auto"] and code), "code_length": len(code) if code else None,
            "code_source": source, "code_scheme_id": context["code_scheme_id"],
            "component_family": context["component_family"], "registry_candidates": context["registry_candidates"],
            "manual_pages": context["manual_pages"], "schema_origin": context["schema_origin"],
            "capacity_mode": context["capacity_mode"], "required_information": issues,
            "validation_passed": bool(code and (source != "model_assembled" or not self.state.book.validate(plan["category_id"], code))),
            "reason": ("매뉴얼 코드표를 검증한 여섯 칸 값을 품목별 위치에 조립했습니다." if source == "model_assembled" else
                       "등록된 코드 후보입니다. 원문과 입력 정보를 함께 확인하세요." if code else " · ".join(issues))}

    def events(self, name, spec="", scheme=None, prepared=None):
        p, context = self.prepare(name, spec, scheme, prepared)
        if len(self.slots) == 1:
            result = self.predict_slot(self.slots[0], p)
            result["context"] = context
            yield result
            return
        yield context
        results = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            jobs = [pool.submit(self.predict_slot, slot, p) for slot in SLOTS]
            for job in as_completed(jobs):
                result = job.result()
                results.append(result)
                yield result
        yield self.assemble(results, p)
