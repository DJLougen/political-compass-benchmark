#!/usr/bin/env python3
"""
Build a single unified MASS-OPINION question dataset (v4) from three public sources:

  1. OpinionsQA  -- Pew American Trends Panel (US public).  Santurkar et al. 2023.
     mirror: hf andrewsiah/opinions_qa_questions_answers_dict  (1494 Qs)
  2. Anthropic/llm_global_opinions -- Pew Global Attitudes Survey (GAS) + World
     Values Survey (WVS).  Durmus et al. 2023.  (2556 Qs: GAS 2203 + WVS 353)

The WVS-7 questionnaire on worldvaluessurvey.org is behind a data-use-agreement /
JS-session download gate; the Anthropic set is the redistributable WVS conduit and
is tagged source=anthropic_wvs so WVS items are identifiable.

Output:
  data/mass_opinion_questions.csv        -- one row per question, our unified format
  data/mass_opinion_distributions.jsonl  -- human answer distributions (where available)

"Formatted for our needs": each row is a ready-to-send single-choice poll prompt with
DK/Refused stripped from the presented options (recorded via has_dk), an options_ordered
flag (Pew-ATP mirror lost scale order -> False; Anthropic keeps list order -> True), and
human answer distributions kept aside for representativeness analysis.
"""
import ast
import csv
import json
import os
import re
import sys
import urllib.request

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE = os.path.join(os.path.dirname(__file__), ".cache")
Q_OUT = os.path.join(DATA, "mass_opinion_questions.csv")
D_OUT = os.path.join(DATA, "mass_opinion_distributions.jsonl")

SOURCES = {
    "opinionqa": "https://huggingface.co/api/datasets/andrewsiah/opinions_qa_questions_answers_dict/parquet/default/train/0.parquet",
    "anthropic": "https://huggingface.co/api/datasets/Anthropic/llm_global_opinions/parquet/default/train/0.parquet",
}

DK_RE = re.compile(r"^\s*(dk|don'?t know|not sure|no answer|refus|decline|no opinion|"
                   r"can'?t choose|none of these|haven'?t heard)", re.I)
DK_INLINE_RE = re.compile(r"(dk\s*/\s*refus|don'?t know|refused)", re.I)


def is_dk(opt):
    return bool(DK_RE.match(opt) or DK_INLINE_RE.search(opt))


def fetch(name, url):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{name}.parquet")
    if not os.path.exists(path):
        print(f"downloading {name} ...", file=sys.stderr)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
            f.write(r.read())
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def clean_opts(opts):
    """drop null + DK/refused; keep order; strip whitespace."""
    out = [o.strip() for o in opts if o and str(o).strip()]
    had_dk = any(is_dk(o) for o in out)
    out = [o for o in out if not is_dk(o)]
    return out, had_dk


def render_prompt(question, options):
    letters = [chr(65 + i) for i in range(len(options))]
    body = "\n".join(f"{l}. {o}" for l, o in zip(letters, options))
    return (f"{question.strip()}\n\n{body}\n\n"
            f"Answer with only the letter ({letters[0]}-{letters[-1]}) of the option "
            f"that best reflects your view. Do not add anything else.")


def parse_selections(raw):
    """Anthropic 'selections' is a stringified defaultdict(list, {country:[probs]})."""
    if not raw:
        return None
    s = raw.strip()
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(0))
    except (ValueError, SyntaxError):
        return None


def main():
    rows, dists = [], []
    seen = set()
    n = 0

    # ---- OpinionsQA (Pew ATP, US). Mirror lost scale order -> options_ordered=False ----
    for r in fetch("opinionqa", SOURCES["opinionqa"]):
        ao = r["answer_options"]
        opts = [ao[k] for k in sorted(ao, key=int) if ao[k] is not None]
        opts, had_dk = clean_opts(opts)
        q = (r["question_text"] or "").strip()
        if len(opts) < 2 or not q:
            continue
        key = re.sub(r"\W+", " ", q.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        n += 1
        iid = f"MASS{n:04d}"
        rows.append({
            "item_id": iid, "source": "opinionqa_pew_atp", "source_key": r["question_key"],
            "question": q, "options": json.dumps(opts), "n_options": len(opts),
            "options_ordered": False, "has_dk": had_dk, "response_format": "mc_single",
            "prompt": render_prompt(q, opts), "has_human_distribution": False,
        })

    # ---- Anthropic global opinions (GAS + WVS). Options keep list order -> True ----
    for r in fetch("anthropic", SOURCES["anthropic"]):
        try:
            opts_orig = [str(o) for o in ast.literal_eval(r["options"])]
        except (ValueError, SyntaxError):
            continue
        # keep the index map so human distribution vectors (aligned to opts_orig,
        # which include DK) can be realigned to the cleaned/presented options.
        kept_idx = [i for i, o in enumerate(opts_orig) if o.strip() and not is_dk(o)]
        opts = [opts_orig[i].strip() for i in kept_idx]
        had_dk = len(opts) != len([o for o in opts_orig if o.strip()])
        q = (r["question"] or "").strip()
        if len(opts) < 2 or not q:
            continue
        key = re.sub(r"\W+", " ", q.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        n += 1
        iid = f"MASS{n:04d}"
        src = "anthropic_wvs" if r["source"] == "WVS" else "anthropic_pew_gas"
        sel = parse_selections(r["selections"])
        rows.append({
            "item_id": iid, "source": src, "source_key": "",
            "question": q, "options": json.dumps(opts), "n_options": len(opts),
            "options_ordered": True, "has_dk": had_dk, "response_format": "mc_single",
            "prompt": render_prompt(q, opts), "has_human_distribution": bool(sel),
        })
        if sel:
            # store BOTH the original option list (aligned to the distribution vectors)
            # and kept_idx, so the scorer maps model answers (over cleaned options) to
            # the human vectors (over original options) and renormalises after dropping DK.
            dists.append({"item_id": iid, "source": src,
                          "options_original": opts_orig, "options_clean": opts,
                          "kept_idx": kept_idx, "distribution_by_country": sel})

    os.makedirs(DATA, exist_ok=True)
    fields = ["item_id", "source", "source_key", "question", "options", "n_options",
              "options_ordered", "has_dk", "response_format", "prompt", "has_human_distribution"]
    with open(Q_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader(); w.writerows(rows)
    with open(D_OUT, "w", encoding="utf-8") as f:
        for d in dists:
            f.write(json.dumps(d) + "\n")

    from collections import Counter
    print(f"Wrote {len(rows)} unified mass questions -> {Q_OUT}")
    print("By source:", dict(Counter(r["source"] for r in rows)))
    print(f"With human distributions: {len(dists)} -> {D_OUT}")
    print(f"Deduped out {n - len(rows) if n > len(rows) else 0} exact-text duplicates "
          f"(total considered {n}).")


if __name__ == "__main__":
    main()
