#!/usr/bin/env python3
"""Per-model refusal report: how many non-answers were transient (rerun to recover)
vs genuine refusals, which dimensions they hit, and whether the leaning is impacted.

Reads a model's neutral + anchor response files, classifies every row (mass/refusals.py),
writes data/stances/refusals_<model>.json, and prints operator guidance:
  - transient (empty/error) -> re-run to recover (batched line-drops, timeouts)
  - genuine refusals        -> reasons captured; flags anchored dimensions whose
                               leaning may be biased by selective refusal.
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, HERE)
from refusals import summarize, TRANSIENT  # noqa: E402
from run_mass import safe_name  # noqa: E402
from anchors import anchor_run_name, ANCHORED_DIMS  # noqa: E402

ANCHORED = list(ANCHORED_DIMS)


def item_dim_map(loadings):
    """{item_id: set(anchored dimension it loads on)}."""
    dims = {}
    with open(loadings, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s = dims.setdefault(row["item_id"], set())
            for d in ANCHORED:
                try:
                    if abs(float(row.get(f"load_{d}", 0) or 0)) > 0:
                        s.add(d)
                except ValueError:
                    pass
    return dims


def read_rows(run_label):
    p = os.path.join(DATA, "responses", f"mass_{safe_name(run_label)}.jsonl")
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--loadings", default=os.path.join(DATA, "source_option_loadings.csv"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    idmap = item_dim_map(args.loadings)
    passes = {"neutral": args.model,
              "far_left": anchor_run_name(args.model, "far_left"),
              "far_right": anchor_run_name(args.model, "far_right")}

    report = {"model": args.model, "passes": {}}
    impacted = set()
    for pname, rl in passes.items():
        rows = read_rows(rl)
        if rows is None:
            continue
        s = summarize(rows, idmap, ANCHORED)
        s["total"] = len(rows)
        report["passes"][pname] = s
        if pname == "neutral":  # the reported position derives from the neutral run
            impacted = {d for d, v in s["per_dimension"].items() if v["leaning_may_be_impacted"]}
    report["leaning_impacted_dims"] = sorted(impacted)

    out = args.out or os.path.join(DATA, "stances", f"refusals_{safe_name(args.model)}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ---- operator guidance ----
    print(f"\n=== refusal report: {args.model} ===", file=sys.stderr)
    for pname, s in report["passes"].items():
        c = s["status_counts"]
        print(f"[{pname}] {s['total']} responses | "
              f"answered {c.get('answered', 0)}, refused {s['genuine_refusals']}, "
              f"transient(empty/error) {s['transient']}, unknown {s['unknown']}", file=sys.stderr)
    tot_transient = sum(s["transient"] for s in report["passes"].values())
    tot_refused = sum(s["genuine_refusals"] for s in report["passes"].values())
    if tot_transient:
        print(f"\n-> {tot_transient} transient non-answers (empty/error). These are usually "
              f"line-drops/timeouts — RE-RUN to recover them (the pipeline resumes; batched "
              f"runners re-ask). They do NOT indicate the model refusing.", file=sys.stderr)
    if tot_refused:
        print(f"\n-> {tot_refused} GENUINE refusals (model declined; reasons saved in {out}).",
              file=sys.stderr)
        neu = report["passes"].get("neutral", {})
        for r in neu.get("sample_reasons", [])[:3]:
            print(f"     e.g. {r['item_id']}: {r['reason'][:120]}", file=sys.stderr)
    if impacted:
        print(f"\n!! LEANING MAY BE IMPACTED on: {', '.join(sorted(impacted))}. "
              f"Selective refusal on these dimensions can bias the position (typically toward "
              f"center). Treat those positions with caution.", file=sys.stderr)
    elif tot_refused:
        print("\n   Genuine refusals are too few per dimension to materially bias any anchored "
              "position (none exceeded the impact threshold).", file=sys.stderr)
    print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
