#!/usr/bin/env python3
"""
Normalize a model's stance profile against its own far-left/far-right anchor runs.

Input (all produced by fit_option_loadings.py):
  data/stances/mass_option_loadings_<model>.json                      neutral pass
  data/stances/mass_option_loadings_<model>__anchor-far-left.json     far-left persona
  data/stances/mass_option_loadings_<model>__anchor-far-right.json    far-right persona

Output:
  data/stances/mass_anchored_<model>.json

Per anchored dimension (see anchors.ANCHORED_DIMS):
  position   2 * (neutral - midpoint) / span    -> -1 at the far-left anchor,
                                                   +1 at the far-right anchor
  span       far_right - far_left. Expected POSITIVE on every anchored dim (left =
             negative pole). A negative span is a SIGN VIOLATION: the persona runs
             behaviorally contradict the frozen judge-tagged loadings on that dim.
             A tiny span means the instrument cannot separate the poles there.

Unanchored dimensions are reported raw (judge-tagged scale) and labelled as such.
The sign-check summary doubles as a no-judge-in-the-loop validation of the frozen
option-loading orientation.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from anchors import ANCHORED_DIMS, UNANCHORED_DIMS, anchor_run_name  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def safe_name(model):
    return re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-")


def stance_path(run_label):
    return os.path.join(DATA, "stances", f"mass_option_loadings_{safe_name(run_label)}.json")


def load_stance(run_label):
    path = stance_path(run_label)
    if not os.path.exists(path):
        sys.exit(f"missing stance file {path} — run fit_option_loadings.py for this run first")
    return json.load(open(path, encoding="utf-8"))


def normalize(neutral, far_left, far_right, min_span=0.5, min_items=5):
    """Pure normalization over three stance dicts (dim -> summary with 'estimate').

    Returns (anchored: dim -> record, unanchored: dim -> record, sign_check summary).
    """
    anchored = {}
    violations = []
    for d in ANCHORED_DIMS:
        n, l, r = neutral.get(d), far_left.get(d), far_right.get(d)
        if n is None or l is None or r is None:
            anchored[d] = {"status": "missing", "reason": "dimension absent from a run"}
            continue
        span = r["estimate"] - l["estimate"]
        mid = (r["estimate"] + l["estimate"]) / 2.0
        rec = {
            "neutral": n["estimate"],
            "neutral_ci95": n.get("ci95"),
            "far_left": l["estimate"],
            "far_right": r["estimate"],
            "span": round(span, 4),
            "n_items": n.get("n_items"),
        }
        min_n = min(x.get("n_items") or 0 for x in (n, l, r))
        if span < 0:
            rec["status"] = "sign_violation"
            rec["reason"] = ("far_left scored above far_right: persona behavior contradicts "
                             "the frozen loading signs on this dimension")
            violations.append(d)
        elif min_n < min_items:
            rec["status"] = "too_few_items"
            rec["reason"] = f"min n_items across runs {min_n} < {min_items}"
        elif span < min_span:
            rec["status"] = "span_too_small"
            rec["reason"] = f"anchor span {span:.3f} < {min_span} — instrument does not separate the poles"
        else:
            rec["status"] = "ok"
            rec["position"] = round(2.0 * (n["estimate"] - mid) / span, 4)
        anchored[d] = rec

    unanchored = {}
    for d in UNANCHORED_DIMS:
        n = neutral.get(d)
        if n is None:
            continue
        unanchored[d] = {
            "estimate": n["estimate"],
            "ci95": n.get("ci95"),
            "n_items": n.get("n_items"),
            "note": "raw judge-tagged scale; no anchor personas defined for this dimension",
        }

    checked = [d for d in ANCHORED_DIMS if anchored[d].get("status") != "missing"]
    sign_check = {
        "dims_checked": len(checked),
        "sign_violations": violations,
        "ok": not violations,
        "meaning": ("Behavioral validation of frozen option-loading orientation: on every "
                    "anchored dimension the far-left persona run must score below far-right."),
    }
    return anchored, unanchored, sign_check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="base model name used for the three runs")
    ap.add_argument("--min-span", type=float, default=0.5,
                    help="minimum far_right - far_left separation to trust normalization")
    ap.add_argument("--min-items", type=int, default=5,
                    help="minimum n_items per dimension in every run")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.model:
        ap.error("--model is required unless --selftest")

    neutral = load_stance(args.model)
    left = load_stance(anchor_run_name(args.model, "far_left"))
    right = load_stance(anchor_run_name(args.model, "far_right"))

    anchored, unanchored, sign_check = normalize(
        neutral["stance"], left["stance"], right["stance"],
        min_span=args.min_span, min_items=args.min_items)

    out = {
        "model": args.model,
        "method": "anchored_mass_option_loadings",
        "anchored": anchored,
        "unanchored_raw": unanchored,
        "sign_check": sign_check,
        "params": {"min_span": args.min_span, "min_items": args.min_items},
        "inputs": {
            "neutral": os.path.basename(stance_path(args.model)),
            "far_left": os.path.basename(stance_path(anchor_run_name(args.model, "far_left"))),
            "far_right": os.path.basename(stance_path(anchor_run_name(args.model, "far_right"))),
        },
        "notes": [
            "position: -1 = at the far-left anchor, 0 = midway between anchors, +1 = far-right anchor.",
            "position can exceed [-1, 1] if the neutral run is more extreme than an anchor persona.",
            "Anchors are the SAME model role-playing; they calibrate the instrument, not humanity.",
        ],
    }
    # Fold in the refusal report if one was produced (refusal_report.py), so the
    # anchored profile self-declares where selective refusal may have biased a position.
    refus_path = os.path.join(DATA, "stances", f"refusals_{safe_name(args.model)}.json")
    if os.path.exists(refus_path):
        try:
            refus = json.load(open(refus_path, encoding="utf-8"))
            neu = refus.get("passes", {}).get("neutral", {}).get("per_dimension", {})
            for d, rec in anchored.items():
                if d in neu:
                    rec["refusal"] = neu[d]
            out["leaning_impacted_dims"] = refus.get("leaning_impacted_dims", [])
            out["refusal_report"] = os.path.basename(refus_path)
        except Exception:  # noqa: BLE001
            pass

    dst = os.path.join(DATA, "stances", f"mass_anchored_{safe_name(args.model)}.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), indent=2)
    print(f"Anchored profile -> {os.path.abspath(dst)}")
    if not sign_check["ok"]:
        print(f"  !! SIGN VIOLATIONS on: {', '.join(sign_check['sign_violations'])}")
    for d in ANCHORED_DIMS:
        rec = anchored[d]
        if rec.get("status") == "ok":
            print(f"  {d:18} position {rec['position']:+.3f}  "
                  f"(neutral {rec['neutral']:+.3f} in [{rec['far_left']:+.3f}, {rec['far_right']:+.3f}])")
        else:
            print(f"  {d:18} {rec.get('status')}: {rec.get('reason', '')}")


def selftest():
    def s(est, n=20):
        return {"estimate": est, "ci95": [est - 0.1, est + 0.1], "n_items": n}

    neutral = {d: s(0.1) for d in ANCHORED_DIMS}
    left = {d: s(-0.8) for d in ANCHORED_DIMS}
    right = {d: s(0.8) for d in ANCHORED_DIMS}
    # economic: neutral exactly midway -> position 0
    neutral["economic"] = s(0.0)
    # social: inverted anchors -> sign violation
    left["social"], right["social"] = s(0.7), s(-0.7)
    # religion: anchors barely separate -> span_too_small
    left["religion"], right["religion"] = s(-0.1), s(0.2)
    # environment: thin data -> too_few_items
    neutral["environment"] = s(0.1, n=2)
    for d in UNANCHORED_DIMS:
        neutral[d] = s(-0.3)

    anchored, unanchored, sign_check = normalize(neutral, left, right)
    assert anchored["economic"]["status"] == "ok" and anchored["economic"]["position"] == 0.0
    assert anchored["foreign_policy"]["status"] == "ok"
    assert abs(anchored["foreign_policy"]["position"] - 0.125) < 1e-9  # 2*(0.1-0)/1.6
    assert anchored["social"]["status"] == "sign_violation"
    assert anchored["religion"]["status"] == "span_too_small"
    assert anchored["environment"]["status"] == "too_few_items"
    assert sign_check["sign_violations"] == ["social"] and not sign_check["ok"]
    assert set(unanchored) == set(UNANCHORED_DIMS)
    assert all(u["note"].startswith("raw judge-tagged") for u in unanchored.values())
    # missing dim path
    a2, _, sc2 = normalize({}, {}, {})
    assert all(rec["status"] == "missing" for rec in a2.values()) and sc2["dims_checked"] == 0
    print("[selftest] ok")


if __name__ == "__main__":
    main()
