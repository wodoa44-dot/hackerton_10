"""Audit displayed-result accuracy, scope, majority labels, and placeholders."""
from collections import Counter
from .schema import SLOTS
from .company_schema import actual_prefix
from .train import metrics


def displayed_slot(assembly,head_result,slot):
    # The web selects the first full-code hypothesis. Unknown decoded fields stay unknown.
    if assembly.get("spectrum"):
        return assembly["spectrum"][0]["values"].get(slot) or ""
    return head_result.get("predicted_value") or ""


def label_for(book,code,slot):
    if slot=="classification":return actual_prefix(code)
    parsed=book.parse(code)
    return parsed["values"][slot] if parsed else None


def summarize_slot(truth,prediction,training_counts,total):
    pairs=list(zip(truth,prediction));n=len(pairs)
    designated=[(t,p) for t,p in pairs if t.strip("-")]
    majority=training_counts.most_common(1)[0][0] if training_counts else None
    return {
        "eligible_n":n,"total_rows":total,"coverage":n/total if total else None,
        "excluded_or_not_applicable_n":total-n,
        "correct":sum(t==p for t,p in pairs),
        "placeholder_n":n-len(designated),
        "designated_n":len(designated),
        "designated_correct":sum(t==p for t,p in designated),
        "designated_accuracy":sum(t==p for t,p in designated)/len(designated) if designated else None,
        "training_majority_label":majority,
        "training_majority_baseline":sum(t==majority for t,p in pairs)/n if n and majority is not None else None,
        "majority_basis":"retained training reference labels; no evaluation-label selection",
        "interpretation":"Final displayed values include rules, models, and retrieval. This is not a standalone learned-model score."
    }


def audit_slots(book,registry,truths,preds,total):
    counters={s:Counter() for s in SLOTS}
    for entry in registry.entries:
        for slot in SLOTS:
            value=label_for(book,entry["code"],slot)
            if value:counters[slot][value]+=1
    return {s:summarize_slot(truths[s],preds[s],counters[s],total) for s in SLOTS}
