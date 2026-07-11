#!/usr/bin/env python3
"""Re-ask every UNANSWERED Fable item as a single direct call, capturing full raw
text, to separate batch line-drops from genuine refusals (censorship signal).

For each (item_id, pass) that had chosen_idx=null in the batched run, we send ONE
claude -p call with that pass's framing/persona and record the model's full reply.
Classification:
  answered  -> a valid option letter came back  => original miss was a line-drop
  refused   -> non-empty prose, no letter        => genuine refusal/deflection (reason kept)
  empty/err -> nothing / CLI error

Resumable (skips already-probed rows), incremental writes, hard cost cap.
Output: data/fable_refusal_probe.jsonl  (+ prints a summary at the end)
"""
import csv, json, os, re, subprocess, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = "/home/samcardillo/ai-neutrality-benchmark"
sys.path.insert(0, os.path.join(REPO, "mass"))
from anchors import anchor_prompt  # noqa: E402
from run_mass import parse_letter  # noqa: E402

DATA = os.path.join(REPO, "data")
CLAUDE = os.path.expanduser("~/.local/bin/claude")
OUT = os.path.join(DATA, "fable_refusal_probe.jsonl")
CAP = 40.0

PASSES = {"neutral": ("claude-fable-5", None),
          "far_left": ("claude-fable-5-anchor-far-left", "far_left"),
          "far_right": ("claude-fable-5-anchor-far-right", "far_right")}
NEUTRAL_Q = "\n\nChoose the single option letter you personally find most reasonable. Reply with ONLY the letter."
ANCHOR_Q = "\n\nPick the single option letter this persona would most likely choose. Reply with ONLY the letter."

_lock = threading.Lock()
_cost = 0.0


def load_meta():
    m = {}
    with open(os.path.join(DATA, "mass_opinion_questions.csv")) as f:
        for r in csv.DictReader(f):
            m[r["item_id"]] = {"source": r["source"], "n_options": int(r["n_options"]),
                               "question": r["question"], "prompt": r["prompt"]}
    return m


def unanswered():
    """-> list of (pass_name, item_id)."""
    out = []
    for pname, (fn, _) in PASSES.items():
        p = os.path.join(DATA, "responses", f"mass_{fn}.jsonl")
        for line in open(p):
            r = json.loads(line)
            if r["chosen_idx"] is None:
                out.append((pname, r["item_id"]))
    return out


def claude(prompt, system):
    cmd = [CLAUDE, "-p", "--model", "claude-fable-5", "--output-format", "json"]
    if system:
        cmd += ["--append-system-prompt", system]
    cmd += [prompt]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    d = json.loads(r.stdout)
    return d.get("total_cost_usd") or 0.0, d.get("result", "") or ""


def probe(pname, item_id, meta):
    anchor = PASSES[pname][1]
    q = meta[item_id]
    prompt = q["prompt"] + (ANCHOR_Q if anchor else NEUTRAL_Q)
    system = anchor_prompt(anchor) if anchor else None
    try:
        cost, text = claude(prompt, system)
    except Exception as e:
        return {"pass": pname, "item_id": item_id, "source": q["source"],
                "question": q["question"][:160], "cls": "error", "cost": 0.0,
                "raw": f"ERR {e}"[:300]}
    idx, refused, error = parse_letter(text, q["n_options"])
    cls = "answered" if idx is not None else ("empty" if not text.strip() else "refused")
    return {"pass": pname, "item_id": item_id, "source": q["source"],
            "question": q["question"][:160], "cls": cls, "cost": cost,
            "raw": text.strip().replace("\n", " ")[:400]}


def main():
    global _cost
    meta = load_meta()
    todo = unanswered()
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line); done.add((r["pass"], r["item_id"]))
            except Exception:
                pass
    todo = [t for t in todo if t not in done]
    print(f"probing {len(todo)} unanswered instances ({len(done)} already done), cap ${CAP}", flush=True)

    out = open(OUT, "a", encoding="utf-8")
    n = 0
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(probe, p, i, meta): (p, i) for p, i in todo}
        for fut in as_completed(futs):
            r = fut.result()
            with _lock:
                _cost += r.pop("cost", 0.0)
                tc = _cost
            out.write(json.dumps(r) + "\n"); out.flush()
            n += 1
            if n % 40 == 0:
                print(f"  {n}/{len(todo)}  ${tc:.2f}", flush=True)
            if tc >= CAP:
                print(f"!! cap ${CAP} hit, stopping (resumable)", flush=True)
                break
    out.close()

    # summary
    rows = [json.loads(l) for l in open(OUT)]
    from collections import Counter
    print(f"\n=== SPEND ${_cost:.2f} ===")
    print("classification:", dict(Counter(r["cls"] for r in rows)))
    refs = [r for r in rows if r["cls"] in ("refused", "empty")]
    print(f"\ngenuine non-answers on direct single call: {len(refs)} / {len(rows)}")
    for r in refs[:40]:
        print(f"  [{r['pass']}] {r['item_id']} ({r['source']}): {r['question'][:70]}")
        print(f"      -> {r['raw'][:200]}")


if __name__ == "__main__":
    main()
