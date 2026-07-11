#!/usr/bin/env python3
"""
Poll a model over the unified mass-opinion questions -> data/responses/mass_<model>.jsonl

Single-choice: the model picks a letter; we record the 0-based option index. Reasoning
suppressed (reasoning models answer with a lone letter). Sampled --reps times so we get
the model's answer *distribution* per question (for representativeness scoring).

Env: OPENAI_BASE_URL, OPENAI_API_KEY (OpenAI-compatible: LiteLLM / vLLM / OpenAI).
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def safe_name(model):
    return re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-")


def chat(base_url, api_key, model, prompt, temperature, max_tokens=None, retries=3, system=None):
    url = base_url.rstrip("/") + "/chat/completions"
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}]
    payload = {"model": model, "messages": messages,
               "temperature": temperature,
               "chat_template_kwargs": {"enable_thinking": False}}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    # MASS_EXTRA_BODY: JSON merged into every request body, e.g.
    # '{"reasoning_effort": "minimal"}' for xAI/OpenAI reasoning models.
    extra = os.environ.get("MASS_EXTRA_BODY")
    if extra:
        payload.update(json.loads(extra))
    hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    for attempt in range(retries):
        try:
            body = json.dumps(payload).encode()
            req = urllib.request.Request(url, data=body, headers=hdr)
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            return out["choices"][0]["message"].get("content") or ""
        except urllib.error.HTTPError as e:  # noqa: PERF203
            detail = ""
            try:
                detail = e.read().decode()
            except Exception:  # noqa: BLE001
                pass
            # Mistral (and other non-Jinja tokenizers) reject chat_template_kwargs -> drop and retry
            if "chat_template" in detail and "chat_template_kwargs" in payload:
                payload.pop("chat_template_kwargs", None)
                continue
            if attempt == retries - 1:
                return f"__ERROR__ {e} {detail[:200]}"
            time.sleep(2 * (attempt + 1))
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                return f"__ERROR__ {e}"
            time.sleep(2 * (attempt + 1))


def parse_letter(text, n_options):
    """Return (0-based index or None, refused, error)."""
    t = (text or "").strip()
    if not t:
        return None, True, False
    if t.startswith("__ERROR__"):
        return None, False, True
    # Reasoning models (GLM, etc.) can emit a <think>...</think> block before the
    # answer even when thinking is suppressed (empty tags); strip it first.
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()
    if not t:
        return None, True, False
    m = re.match(r"^[\s\"'`(\[]*([A-Za-z])[\s\"'`)\].:,!-]", t + " ")
    if m:
        idx = ord(m.group(1).upper()) - 65
        if 0 <= idx < n_options:
            return idx, False, False
    return None, True, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--anchor", choices=("far_left", "far_right"), default="",
                    help="run as an anchor persona (see anchors.py); output and run label "
                         "become mass_<model>__anchor-<name>. Neutral pass = no flag.")
    ap.add_argument("--system-prompt", default="",
                    help="free-form system prompt override (mutually exclusive with --anchor)")
    ap.add_argument("--run-label", default="",
                    help="override the recorded run label (defaults: model, or anchor run name)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--sample", type=int, default=0, help="cap number of questions (0 = all)")
    ap.add_argument("--with-dist-only", action="store_true",
                    help="only questions that have human distributions (for representativeness)")
    ap.add_argument("--only-tagged", action="store_true",
                    help="restrict to items present in data/mass_loadings.csv (for the IRT/BT wiring)")
    ap.add_argument("--sources", default="", help="comma-list filter, e.g. anthropic_wvs,anthropic_pew_gas")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:4000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "sk-none")

    qs = list(csv.DictReader(open(os.path.join(DATA, "mass_opinion_questions.csv"), encoding="utf-8")))
    if args.only_tagged:
        tagged = {r["item_id"] for r in
                  csv.DictReader(open(os.path.join(DATA, "mass_loadings.csv"), encoding="utf-8"))}
        qs = [q for q in qs if q["item_id"] in tagged]
    if args.with_dist_only:
        qs = [q for q in qs if q["has_human_distribution"] == "True"]
    if args.sources:
        keep = set(args.sources.split(","))
        qs = [q for q in qs if q["source"] in keep]
    if args.sample and len(qs) > args.sample:
        qs = random.Random(args.seed).sample(qs, args.sample)

    tasks = [(q, rep) for q in qs for rep in range(args.reps)]
    system = None
    run_label = args.run_label or args.model
    if args.anchor:
        if args.system_prompt:
            sys.exit("--anchor and --system-prompt are mutually exclusive")
        from anchors import anchor_prompt, anchor_run_name
        system = anchor_prompt(args.anchor)
        run_label = args.run_label or anchor_run_name(args.model, args.anchor)
    elif args.system_prompt:
        system = args.system_prompt
        if not args.run_label:
            sys.exit("--system-prompt requires --run-label so runs don't overwrite each other")

    total = len(tasks)
    print(f"mass: {len(qs)} questions x {args.reps} reps = {total} calls "
          f"(concurrency {args.concurrency}, run={run_label})", file=sys.stderr)

    from refusals import classify, STATUS_REFUSED, STATUS_ERROR

    def do(task):
        q, rep = task
        n_opt = int(q["n_options"])
        text = chat(base_url, api_key, args.model, q["prompt"], args.temperature, system=system)
        idx, status, reason = classify(text, n_opt)
        # keep the model's actual text so a refusal reason / odd output is recoverable,
        # not just the first 40 chars.
        raw = reason if reason else (text or "")[:120]
        return {"model": run_label, "base_model": args.model,
                "anchor": args.anchor or None, "item_id": q["item_id"], "source": q["source"],
                "n_options": n_opt, "rep": rep, "chosen_idx": idx,
                "status": status, "refused": idx is None, "error": status == STATUS_ERROR,
                "raw": raw}

    os.makedirs(os.path.join(DATA, "responses"), exist_ok=True)
    out_path = os.path.join(DATA, "responses", f"mass_{safe_name(run_label)}.jsonl")
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for rec in ex.map(do, tasks):
                out.write(json.dumps(rec) + "\n")
                n += 1
                if n % 200 == 0:
                    print(f"  ... {n}/{total}", file=sys.stderr)
    print(f"Wrote {n} responses to {out_path}")


if __name__ == "__main__":
    main()
