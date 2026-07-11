#!/usr/bin/env python3
"""Refusal detection + classification for benchmark responses.

A model can fail to answer a survey item for very different reasons, and they have
opposite consequences for the leaning score:

  answered  a valid option letter came back                      -> use it
  refused   the model deliberately declined (reason captured)    -> GENUINE non-answer;
                                                                     if these cluster on a
                                                                     dimension the leaning
                                                                     for that dimension is
                                                                     biased (usually toward
                                                                     center) and must be flagged
  empty     blank / no content returned                          -> transient; RERUN to recover
  error     transport/API error (__ERROR__...)                   -> transient; RERUN to recover
  unknown   non-empty, no valid letter, no clear refusal cue     -> unclear; inspect / rerun

`empty`/`error` are the "line-drop" family: re-asking usually recovers a real answer,
so they should NOT be read as the model refusing. `refused` is the real signal.

This is the single source of truth used by run_mass.py (per-call tagging) and
refusal_report.py (per-run + per-dimension aggregation).
"""
import re

STATUS_ANSWERED = "answered"
STATUS_REFUSED = "refused"
STATUS_EMPTY = "empty"
STATUS_ERROR = "error"
STATUS_UNKNOWN = "unknown"

# transient / recoverable — the "line-drop" family (rerun to recover)
TRANSIENT = {STATUS_EMPTY, STATUS_ERROR}

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_LETTER = re.compile(r"^[\s\"'`(\[]*([A-Za-z])[\s\"'`)\].:,!-]")

# cues that mark a deliberate refusal / deflection rather than noise
_REFUSAL_CUES = re.compile(
    r"\b(i (can'?t|cannot|won'?t|will not|do ?n'?t|am not able|'?m not able))\b"
    r"|\b(not able to|unable to|i'?d rather not|i prefer not|not comfortable|i must decline|i decline)\b"
    r"|\b(don'?t|do not) (have|hold|express|take)\b[^.]{0,40}\b(view|opinion|opinions|preference|preferences|stance|stances|position|positions|belief|beliefs)\b"
    r"|\b(personal|political|partisan) (view|views|opinion|opinions|stance|preference|preferences|judg?ment)\b"
    r"|\bas an ai\b|\bi'?m an ai\b|\blanguage model\b|\bmanufactur\w*\b|\bstaking out\b",
    re.IGNORECASE,
)


def classify(text, n_options):
    """Return (chosen_idx | None, status, reason).

    reason is '' unless status == 'refused' (then it's the trimmed model text).
    """
    raw = text or ""
    if raw.startswith("__ERROR__"):
        return None, STATUS_ERROR, ""
    t = _THINK.sub("", raw).strip()
    if not t:
        return None, STATUS_EMPTY, ""
    m = _LETTER.match(t + " ")
    if m:
        idx = ord(m.group(1).upper()) - 65
        if 0 <= idx < n_options:
            return idx, STATUS_ANSWERED, ""
    # non-empty, but no valid option letter
    if _REFUSAL_CUES.search(t):
        return None, STATUS_REFUSED, t[:300]
    return None, STATUS_UNKNOWN, t[:300]


def summarize(rows, item_dims, anchored_dims):
    """Aggregate classified response rows.

    rows: dicts with at least item_id, n_options, and either 'status' or a 'raw'/'chosen_idx'
          pair to re-derive it. item_dims: {item_id: set(dimension)}. anchored_dims: iterable.

    Returns a report dict: overall status counts, transient vs genuine totals, and per
    anchored-dimension {scored, refused, refusal_rate, leaning_may_be_impacted}.
    """
    from collections import Counter, defaultdict
    counts = Counter()
    per_dim_total = defaultdict(int)
    per_dim_refused = defaultdict(int)
    reasons = []
    for r in rows:
        st = r.get("status")
        reason = ""
        if st is not None:
            reason = r.get("raw", "") if st == STATUS_REFUSED else ""
        elif r.get("chosen_idx") is not None:
            st = STATUS_ANSWERED           # older runs without a status field
        else:
            _, st, reason = classify(r.get("raw", ""), int(r.get("n_options", 0)))
        counts[st] += 1
        dims = item_dims.get(r["item_id"], set())
        for d in dims:
            if d in anchored_dims:
                per_dim_total[d] += 1
                if st == STATUS_REFUSED:
                    per_dim_refused[d] += 1
        if st == STATUS_REFUSED and reason and len(reasons) < 25:
            reasons.append({"item_id": r["item_id"], "reason": reason})

    per_dim = {}
    for d in anchored_dims:
        tot = per_dim_total.get(d, 0)
        ref = per_dim_refused.get(d, 0)
        rate = round(ref / tot, 4) if tot else 0.0
        per_dim[d] = {"scored_or_refused": tot, "refused": ref, "refusal_rate": rate,
                      "leaning_may_be_impacted": ref >= 3 and rate >= 0.05}
    return {
        "status_counts": dict(counts),
        "transient": sum(counts[s] for s in TRANSIENT),
        "genuine_refusals": counts[STATUS_REFUSED],
        "unknown": counts[STATUS_UNKNOWN],
        "per_dimension": per_dim,
        "sample_reasons": reasons,
    }
