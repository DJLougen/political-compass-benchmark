#!/usr/bin/env python3
"""
Judge pool + family-conflict validation.

Rule (mass/AGENT.md): judge != scored model. Loadings tagged by judge J must never
score a target model from J's family/derivative — that is circular (a family grades
its own homework). This module is the single registry of model families and the
check every scoring entry point runs.

Matching is keyword-based on name tokens (split on non-alphanumerics), so it
catches derivatives that keep the base name ("dolphin-llama3-8b" -> meta,
"qwen3-6-27b-instruct-ft" -> qwen). Unknown names are reported as such — an
unknown target is a WARNING, not a pass.

CLI:
  python mass/judge_pool.py --target <model> --judges spark-latest,claude-fable-5
  python mass/judge_pool.py --target <model> --pool gemini-3-5-flash,claude-fable-5,qwen3-6-27b --need 2
  python mass/judge_pool.py --selftest
"""
import argparse
import re
import sys

# family -> keywords. Tokens of a model name are matched by prefix for keywords of
# length >= 4 and exactly for shorter ones (avoids "hy" firing inside "physics").
FAMILY_KEYWORDS = {
    "qwen": ["qwen", "qwq", "qvq"],
    "anthropic": ["claude", "haiku", "sonnet", "opus", "fable"],
    "google": ["gemini", "gemma", "bard", "palm"],
    "openai": ["gpt", "chatgpt", "openai", "davinci", "o1", "o3", "o4"],
    "meta": ["llama"],
    "mistral": ["mistral", "mixtral", "magistral", "ministral", "codestral"],
    "deepseek": ["deepseek"],
    "tencent": ["hunyuan", "hy3"],
    "xai": ["grok"],
    "zhipu": ["glm", "chatglm"],
    "moonshot": ["kimi", "moonshot"],
    "cohere": ["command", "aya"],
    "microsoft": ["phi"],
    "amazon": ["nova", "titan"],
    "ibm": ["granite"],
    "allenai": ["olmo", "tulu"],
    "nvidia": ["nemotron", "nvidia"],
}

# Canonical judge pool (preference order). Seven distinct families; gemma-31b is the
# same-provider fallback for gemini and is auto-skipped when gemini is picked.
DEFAULT_POOL = [
    "deepseek-v4-flash",
    "gemini-3-5-flash",
    "gemma-31b",
    "qwen3-6-27b",
    "NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
    "gpt-oss-120b",
    "glm-5.2",
    "kimi-2.6",
]

# Repo-specific judge ids that don't carry their family in the name.
# spark-latest = local Qwen deployment (mass/README v5); fable = Claude; hy3 = Hunyuan.
ALIASES = {
    "spark": "qwen",
    "spark-latest": "qwen",
    "fable": "anthropic",
    "hy3": "tencent",
}


def _tokens(name):
    return [t for t in re.split(r"[^a-z0-9]+", (name or "").lower()) if t]


def family_of(name):
    """Best-effort family for a model/judge name; 'unknown' when nothing matches."""
    low = (name or "").strip().lower()
    if low in ALIASES:
        return ALIASES[low]
    toks = _tokens(low)
    for tok in toks:
        if tok in ALIASES:
            return ALIASES[tok]
    for fam, keys in FAMILY_KEYWORDS.items():
        for key in keys:
            for tok in toks:
                if (len(key) >= 4 and tok.startswith(key)) or tok == key:
                    return fam
    return "unknown"


def conflicted_judges(target_model, judges):
    """Judges sharing a family with the target. Unknown families never match."""
    tfam = family_of(target_model)
    if tfam == "unknown":
        return []
    return [j for j in judges if family_of(j) == tfam]


def check(target_model, judges):
    """Full verdict dict for scoring `target_model` with loadings tagged by `judges`."""
    tfam = family_of(target_model)
    conflicts = conflicted_judges(target_model, judges)
    return {
        "target_model": target_model,
        "target_family": tfam,
        "judges": {j: family_of(j) for j in judges},
        "conflicted_judges": conflicts,
        "target_family_unknown": tfam == "unknown",
        "ok": not conflicts and tfam != "unknown",
    }


def pick_judges(target_model, pool, need=2):
    """Pick `need` judges from `pool` (ordered by preference): no judge from the
    target's family, and at most one judge per family. Raises if impossible."""
    tfam = family_of(target_model)
    picked, used_fams = [], set()
    for cand in pool:
        fam = family_of(cand)
        if fam == "unknown":
            continue
        if fam == tfam or fam in used_fams:
            continue
        picked.append(cand)
        used_fams.add(fam)
        if len(picked) == need:
            return picked
    raise ValueError(
        f"cannot pick {need} clean judges for target {target_model!r} (family {tfam!r}) "
        f"from pool {list(pool)!r}; got only {picked!r}. Extend the pool with another family.")


def selftest():
    assert family_of("qwen3-6-27b") == "qwen"
    assert family_of("Qwen2.5-72B-Instruct-FT") == "qwen"
    assert family_of("spark-latest") == "qwen"
    assert family_of("claude-fable-5") == "anthropic"
    assert family_of("gemini-3-5-flash") == "google"
    assert family_of("gemma4-31b") == "google"
    assert family_of("gpt-oss-120b") == "openai"
    assert family_of("hy3") == "tencent"
    assert family_of("dolphin-llama3-8b") == "meta"
    assert family_of("physics-tutor-9000") == "unknown"  # 'hy' must not fire mid-token
    assert family_of("o3-mini") == "openai"

    v5_judges = ["spark-latest", "claude-fable-5"]
    assert conflicted_judges("qwen3-235b", v5_judges) == ["spark-latest"]
    assert conflicted_judges("claude-opus-5", v5_judges) == ["claude-fable-5"]
    assert conflicted_judges("gemini-3-5-flash", v5_judges) == []
    assert conflicted_judges("mystery-model", v5_judges) == []  # unknown target: no false conflicts

    v = check("qwen3-235b", v5_judges)
    assert not v["ok"] and v["conflicted_judges"] == ["spark-latest"]
    v = check("mystery-model", v5_judges)
    assert not v["ok"] and v["target_family_unknown"]  # unknown is a warning, not a pass
    assert check("grok-4", v5_judges)["ok"]

    pool = ["gemini-3-5-flash", "claude-fable-5", "qwen3-6-27b", "gpt-oss-120b"]
    assert pick_judges("qwen3-235b", pool, 2) == ["gemini-3-5-flash", "claude-fable-5"]
    assert pick_judges("claude-opus-5", pool, 2) == ["gemini-3-5-flash", "qwen3-6-27b"]
    assert pick_judges("gemini-3-pro", pool, 3) == ["claude-fable-5", "qwen3-6-27b", "gpt-oss-120b"]
    try:
        pick_judges("gemini-3-pro", ["gemini-3-5-flash", "gemma4-31b"], 1)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    # canonical pool: every member resolves, families are as designed
    fams = [family_of(j) for j in DEFAULT_POOL]
    assert fams == ["deepseek", "google", "google", "qwen", "nvidia", "openai", "zhipu", "moonshot"]
    assert family_of("NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4") == "nvidia"
    assert family_of("deepseek-v4-flash") == "deepseek"
    assert family_of("glm-5.2") == "zhipu"
    assert family_of("kimi-2.6") == "moonshot"
    assert family_of("kimi2.6") == "moonshot"  # no separator variant
    # anthropic target: 3 clean judges, one per family, gemma skipped as gemini dup
    assert pick_judges("claude-opus-5", DEFAULT_POOL, 3) == [
        "deepseek-v4-flash", "gemini-3-5-flash", "qwen3-6-27b"]
    # qwen target: qwen judge excluded, still 3 clean families available
    assert pick_judges("qwen3-235b", DEFAULT_POOL, 3) == [
        "deepseek-v4-flash", "gemini-3-5-flash", "NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"]
    # google target falls back past both google entries
    assert pick_judges("gemini-3-pro", DEFAULT_POOL, 3) == [
        "deepseek-v4-flash", "qwen3-6-27b", "NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"]
    print("[selftest] ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="")
    ap.add_argument("--judges", default="", help="comma-list of judge names to validate against")
    ap.add_argument("--pool", default="",
                    help="comma-list judge pool in preference order; 'default' = DEFAULT_POOL")
    ap.add_argument("--need", type=int, default=3,
                    help="judges to pick from --pool (v6 anchoring standard: >=3 families)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.target:
        sys.exit("--target required (or --selftest)")
    if args.pool:
        pool = (DEFAULT_POOL if args.pool.strip().lower() == "default"
                else [p.strip() for p in args.pool.split(",") if p.strip()])
        picked = pick_judges(args.target, pool, args.need)
        print(f"target {args.target} (family {family_of(args.target)}) -> judges: {', '.join(picked)}")
        return
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    v = check(args.target, judges)
    print(f"target:  {v['target_model']}  (family: {v['target_family']})")
    for j, fam in v["judges"].items():
        mark = "CONFLICT" if j in v["conflicted_judges"] else "ok"
        print(f"judge:   {j}  (family: {fam})  [{mark}]")
    if v["target_family_unknown"]:
        print("WARNING: target family unknown — verify manually before trusting this check.")
    print("verdict:", "OK" if v["ok"] else "NOT OK")
    sys.exit(0 if v["ok"] else 1)


if __name__ == "__main__":
    main()
