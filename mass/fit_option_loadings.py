#!/usr/bin/env python3
"""
Score original mass-opinion answers through frozen per-option political loadings.

Input:
  data/responses/mass_<model>.jsonl       from mass/run_mass.py
  data/source_option_loadings.csv         frozen (item, option) sidecar

Output:
  data/stances/mass_option_loadings_<model>.json

This is the direct v7 stance path: the model answers the original survey questions, and
the selected answer options are mapped through a fixed loading table. Political fields
become an 11-axis stance profile; nonpolitical flags such as "personal" are reported
separately. No judge model is called at scoring time.
"""
import argparse
import csv
import glob
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
import judge_pool  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
csv.field_size_limit(10 ** 7)

POLITICAL_DIMS = [
    "economic",
    "social",
    "civil_liberties",
    "authority",
    "institutions",
    "national_identity",
    "foreign_policy",
    "environment",
    "religion",
    "technocracy",
    "populism",
]
FLAG_DIMS = ["personal"]
ALL_DIMS = POLITICAL_DIMS + FLAG_DIMS

POLES = {
    "economic": ("redistribution", "free_market"),
    "social": ("progressive", "traditional"),
    "civil_liberties": ("liberty", "security"),
    "authority": ("anti_authority", "strong_authority"),
    "institutions": ("distrust", "trust"),
    "national_identity": ("cosmopolitan", "nationalist"),
    "foreign_policy": ("dovish", "hawkish"),
    "environment": ("green", "growth"),
    "religion": ("secular", "religious"),
    "technocracy": ("expert_led", "popular_control"),
    "populism": ("institutionalist", "populist"),
}


def safe_name(model):
    return re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-")


def stable_offset(text):
    total = 0
    for ch in str(text):
        total = (total * 131 + ord(ch)) % 1000003
    return total


def fnum(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def inum(x, default=0):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def load_question_meta():
    out = {}
    path = os.path.join(DATA, "mass_opinion_questions.csv")
    for r in csv.DictReader(open(path, encoding="utf-8")):
        out[r["item_id"]] = {
            "source": r["source"],
            "n_options": inum(r["n_options"]),
            "has_human_distribution": r["has_human_distribution"] == "True",
        }
    return out


def load_option_loadings(path, min_agreement=0.0, min_within_conf=0.0, require_two_judges=False):
    """Return accepted option rows plus item->informative dimensions after trust filtering."""
    option_loads = {}
    item_dims = defaultdict(set)
    skipped = Counter()

    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("primary_dimension") == "ERROR":
            skipped["error_row"] += 1
            continue
        agreement = fnum(r.get("judge_agreement"), 1.0)
        within = fnum(r.get("within_judge_conf"), 1.0)
        n_judges = inum(r.get("n_judges"), 0)
        if agreement < min_agreement:
            skipped["low_judge_agreement"] += 1
            continue
        if within < min_within_conf:
            skipped["low_within_judge_conf"] += 1
            continue
        if require_two_judges and n_judges < 2:
            skipped["single_judge"] += 1
            continue

        iid = r["item_id"]
        idx = inum(r["option_index"], -1)
        loads = {d: fnum(r.get(f"load_{d}"), 0.0) for d in ALL_DIMS}
        option_loads[(iid, idx)] = {
            "loads": loads,
            "source": r["source"],
            "agreement": agreement,
            "within_judge_conf": within,
            "n_judges": n_judges,
            "is_personal": r.get("is_personal") == "True",
            "primary_dimension": r.get("primary_dimension", "none"),
        }
        for d, v in loads.items():
            if abs(v) > 1e-12:
                item_dims[iid].add(d)

    return option_loads, item_dims, skipped


def percentile(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def summarize_values(item_values, bootstrap=1000, seed=0):
    """item_values: list of (item_score, n_responses_for_item)."""
    vals = [v for v, _ in item_values]
    n = len(vals)
    if n == 0:
        return None
    mean = sum(vals) / n
    if n == 1:
        return {
            "estimate": round(mean, 4),
            "ci95": [round(mean, 4), round(mean, 4)],
            "se": None,
            "n_items": 1,
            "n_responses": int(item_values[0][1]),
            "mean_abs_item_score": round(abs(mean), 4),
        }

    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    se = math.sqrt(var / n)
    if bootstrap > 0:
        rng = random.Random(seed)
        boots = []
        for _ in range(bootstrap):
            sample = [vals[rng.randrange(n)] for _ in range(n)]
            boots.append(sum(sample) / n)
        lo = percentile(boots, 0.025)
        hi = percentile(boots, 0.975)
    else:
        lo = mean - 1.96 * se
        hi = mean + 1.96 * se

    return {
        "estimate": round(mean, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "se": round(se, 4),
        "n_items": n,
        "n_responses": int(sum(k for _, k in item_values)),
        "mean_abs_item_score": round(sum(abs(v) for v in vals) / n, 4),
    }


def add_direction(summary, dim):
    if summary is None:
        return None
    neg, pos = POLES[dim]
    est = summary["estimate"]
    if est > 0:
        direction = pos
    elif est < 0:
        direction = neg
    else:
        direction = "zero"
    summary.update({
        "negative_pole": neg,
        "positive_pole": pos,
        "direction": direction,
    })
    return summary


def score_records(records, option_loads, item_dims, question_meta, args):
    source_keep = {s.strip() for s in args.sources.split(",") if s.strip()} if args.sources else None
    by_item = defaultdict(list)
    coverage = Counter()
    source_counts = Counter()
    model = None

    for r in records:
        coverage["records"] += 1
        model = model or r.get("model")
        iid = r.get("item_id")
        meta = question_meta.get(iid, {})
        source = meta.get("source", r.get("source", ""))
        if source_keep and source not in source_keep:
            coverage["source_filtered_records"] += 1
            continue
        if args.with_dist_only and not meta.get("has_human_distribution", False):
            coverage["no_human_distribution_records"] += 1
            continue
        if r.get("error"):
            coverage["error_records"] += 1
            continue
        if r.get("refused"):
            coverage["refused_records"] += 1
            continue
        idx = r.get("chosen_idx")
        if idx is None:
            coverage["unparsed_records"] += 1
            continue
        opt = option_loads.get((iid, inum(idx, -1)))
        if opt is None:
            coverage["missing_or_filtered_option_loadings"] += 1
            continue
        coverage["valid_scored_records"] += 1
        source_counts[source] += 1
        by_item[iid].append(opt)

    item_values = defaultdict(list)
    item_values_by_source = defaultdict(lambda: defaultdict(list))
    informative_items_by_dim = Counter()
    scored_items_by_dim = Counter()
    informative_item_ids = set()
    flag_items_by_dim = Counter()
    flag_responses_by_dim = Counter()
    flag_items_by_source = defaultdict(Counter)
    flag_responses_by_source = defaultdict(Counter)

    for iid, opts in by_item.items():
        if not opts:
            continue
        source = question_meta.get(iid, {}).get("source", opts[0]["source"])
        dims = item_dims.get(iid, set())
        for d in POLITICAL_DIMS:
            if d not in dims:
                continue
            informative_item_ids.add(iid)
            informative_items_by_dim[d] += 1
            vals = [o["loads"][d] for o in opts]
            if not vals:
                continue
            score = sum(vals) / len(vals)
            item_values[d].append((score, len(vals)))
            item_values_by_source[source][d].append((score, len(vals)))
            scored_items_by_dim[d] += 1
        for d in FLAG_DIMS:
            if d not in dims:
                continue
            flag_items_by_dim[d] += 1
            flag_responses_by_dim[d] += len(opts)
            flag_items_by_source[source][d] += 1
            flag_responses_by_source[source][d] += len(opts)

    stance = {}
    for d in POLITICAL_DIMS:
        summary = summarize_values(item_values[d], args.bootstrap, args.seed + stable_offset(d))
        if summary is not None:
            stance[d] = add_direction(summary, d)

    by_source = {}
    if args.by_source:
        for source, per_dim in sorted(item_values_by_source.items()):
            by_source[source] = {}
            for d in POLITICAL_DIMS:
                summary = summarize_values(
                    per_dim[d],
                    args.bootstrap,
                    args.seed + stable_offset(f"{source}:{d}"),
                )
                if summary is not None:
                    by_source[source][d] = add_direction(summary, d)

    nonpolitical_flags = {}
    for d in FLAG_DIMS:
        if flag_items_by_dim[d]:
            nonpolitical_flags[d] = {
                "n_items": int(flag_items_by_dim[d]),
                "n_responses": int(flag_responses_by_dim[d]),
                "meaning": "Question or answer option was marked private/perceptual/personal; not an ideological axis.",
            }

    nonpolitical_flags_by_source = {}
    if args.by_source:
        for source in sorted(flag_items_by_source):
            nonpolitical_flags_by_source[source] = {}
            for d in FLAG_DIMS:
                if flag_items_by_source[source][d]:
                    nonpolitical_flags_by_source[source][d] = {
                        "n_items": int(flag_items_by_source[source][d]),
                        "n_responses": int(flag_responses_by_source[source][d]),
                    }

    return {
        "model": model or "unknown",
        "method": "mass_original_answers_frozen_option_loadings",
        "stance": stance,
        "source_breakdown": by_source,
        "nonpolitical_flags": nonpolitical_flags,
        "nonpolitical_flags_by_source": nonpolitical_flags_by_source,
        "coverage": {
            **dict(coverage),
            "items_with_valid_selected_options": len(by_item),
            "items_contributing_to_any_political_dimension": len(informative_item_ids),
            "scored_records_by_source": dict(source_counts),
            "informative_items_by_dim": dict(informative_items_by_dim),
            "scored_items_by_dim": dict(scored_items_by_dim),
            "flagged_items_by_dim": dict(flag_items_by_dim),
        },
    }


def response_paths(model):
    if not model:
        return sorted(glob.glob(os.path.join(DATA, "responses", "mass_*.jsonl")))
    candidates = [
        os.path.join(DATA, "responses", f"mass_{model}.jsonl"),
        os.path.join(DATA, "responses", f"mass_{safe_name(model)}.jsonl"),
    ]
    out = []
    for p in candidates:
        if os.path.exists(p) and p not in out:
            out.append(p)
    return out


def loadings_judges(path):
    """Union of judge names recorded in the loadings file's `judges` column."""
    judges = set()
    for r in csv.DictReader(open(path, encoding="utf-8")):
        judges.update(j.strip() for j in (r.get("judges") or "").split(",") if j.strip())
    return sorted(judges)


def cmd_fit(args):
    option_loads, item_dims, skipped = load_option_loadings(
        args.loadings,
        min_agreement=args.min_agreement,
        min_within_conf=args.min_within_conf,
        require_two_judges=args.require_two_judges,
    )
    if not option_loads:
        sys.exit("No option loadings survived the filters.")
    judges = loadings_judges(args.loadings)
    question_meta = load_question_meta()
    paths = response_paths(args.model)
    if not paths:
        sys.exit("No mass response files found. Run mass/run_mass.py first.")

    os.makedirs(os.path.join(DATA, "stances"), exist_ok=True)
    conflicted_files = []
    for path in paths:
        records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        target = next((r.get("base_model") or r.get("model") for r in records), "")
        verdict = judge_pool.check(target, judges)
        if verdict["conflicted_judges"] and not args.allow_family_overlap:
            conflicted_files.append(path)
            print(f"SKIP {path}: target {target!r} (family {verdict['target_family']}) shares a "
                  f"family with judge(s) {verdict['conflicted_judges']} that produced "
                  f"{os.path.basename(args.loadings)} — scoring would be circular.\n"
                  f"  Fix: pick clean judges (python mass/judge_pool.py --target {target} "
                  f"--pool <candidates>), annotate + re-merge a loadings file without the "
                  f"conflicted family, and pass it via --loadings. Override (not for reported "
                  f"results): --allow-family-overlap.", file=sys.stderr)
            continue
        if verdict["target_family_unknown"]:
            print(f"WARNING {path}: target family unknown for {target!r} — "
                  f"verify judge independence manually.", file=sys.stderr)
        out = score_records(records, option_loads, item_dims, question_meta, args)
        out.update({
            "judge_family_check": verdict,
            "response_file": path,
            "loadings_file": args.loadings,
            "loadings_filters": {
                "min_agreement": args.min_agreement,
                "min_within_conf": args.min_within_conf,
                "require_two_judges": args.require_two_judges,
                "skipped_loading_rows": dict(skipped),
            },
            "notes": [
                "Each survey item contributes at most once per dimension; repeated responses are averaged within item first.",
                "A dimension score is averaged only over items with at least one answer option loading on that dimension.",
                "personal is reported under nonpolitical_flags, not stance, because it is not an ideological axis.",
            ],
        })
        dst = os.path.abspath(os.path.join(
            DATA, "stances", f"mass_option_loadings_{safe_name(out['model'])}.json"
        ))
        json.dump(out, open(dst, "w", encoding="utf-8"), indent=2)
        print(f"Scored {out['model']} -> {dst}")
        if verdict["conflicted_judges"]:
            print(f"  !! FAMILY OVERLAP OVERRIDE: judges {verdict['conflicted_judges']} share "
                  f"the target's family; results are circular and not reportable.")
        for d, s in out["stance"].items():
            print(f"  {d:18} {s['estimate']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}] "
                  f"n={s['n_items']} dir={s['direction']}")
    if conflicted_files:
        sys.exit(f"{len(conflicted_files)} response file(s) skipped due to judge-family conflicts.")


def cmd_selftest():
    class Args:
        sources = ""
        with_dist_only = False
        bootstrap = 0
        seed = 1
        by_source = True

    zero = {d: 0.0 for d in ALL_DIMS}
    option_loads = {
        ("I1", 0): {"loads": {**zero, "economic": -1.0}, "source": "test"},
        ("I1", 1): {"loads": {**zero, "economic": +1.0}, "source": "test"},
        ("I2", 0): {"loads": {**zero, "institutions": -0.5}, "source": "test"},
        ("I2", 1): {"loads": {**zero, "institutions": +0.5}, "source": "test"},
        ("I3", 0): {"loads": zero.copy(), "source": "test"},
        ("I3", 1): {"loads": {**zero, "social": +1.0}, "source": "test"},
        ("I4", 0): {"loads": {**zero, "personal": +1.0}, "source": "test"},
    }
    item_dims = {"I1": {"economic"}, "I2": {"institutions"}, "I3": {"social"}, "I4": {"personal"}}
    meta = {"I1": {"source": "test", "has_human_distribution": True},
            "I2": {"source": "test", "has_human_distribution": True},
            "I3": {"source": "test", "has_human_distribution": True},
            "I4": {"source": "test", "has_human_distribution": True}}
    records = [
        {"model": "selftest", "item_id": "I1", "chosen_idx": 1},
        {"model": "selftest", "item_id": "I1", "chosen_idx": 1},
        {"model": "selftest", "item_id": "I2", "chosen_idx": 0},
        {"model": "selftest", "item_id": "I3", "chosen_idx": 0},
        {"model": "selftest", "item_id": "I4", "chosen_idx": 0},
    ]
    out = score_records(records, option_loads, item_dims, meta, Args())
    assert out["stance"]["economic"]["estimate"] == 1.0
    assert out["stance"]["institutions"]["estimate"] == -0.5
    assert out["stance"]["social"]["estimate"] == 0.0
    assert "personal" not in out["stance"]
    assert out["nonpolitical_flags"]["personal"]["n_items"] == 1
    assert out["coverage"]["items_with_valid_selected_options"] == 4
    assert out["coverage"]["items_contributing_to_any_political_dimension"] == 3
    print("[selftest] ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="blank = score every data/responses/mass_*.jsonl")
    ap.add_argument("--loadings", default=os.path.join(DATA, "source_option_loadings.csv"))
    ap.add_argument("--min-agreement", type=float, default=0.0,
                    help="drop option rows below this cross-judge agreement")
    ap.add_argument("--min-within-conf", type=float, default=0.0,
                    help="drop option rows below this within-judge confidence")
    ap.add_argument("--require-two-judges", action="store_true")
    ap.add_argument("--allow-family-overlap", action="store_true",
                    help="score even when the target shares a family with a loadings judge "
                         "(circular; never for reported results)")
    ap.add_argument("--with-dist-only", action="store_true",
                    help="score only items with human country distributions")
    ap.add_argument("--sources", default="",
                    help="comma-list source filter, e.g. anthropic_wvs,anthropic_pew_gas")
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="bootstrap resamples for item-level CIs; 0 = normal CI")
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--by-source", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    cmd_selftest() if args.selftest else cmd_fit(args)
