#!/usr/bin/env python3
"""
Frozen political-loading annotation for the ORIGINAL survey questions.

For every question in data/mass_opinion_questions.csv and EVERY answer option, assign a
sparse loading vector over a 12-dimension ideology space. The loading is the ideological
stance that *choosing that option* signals (not the topic of the question). Output is a
frozen sidecar CSV at the (item, option) grain — the raw dataset CSV is never modified.

Method: a rubric-grounded LLM judge annotates a whole question at once (sees all options,
returns one sparse vector per option). N stochastic passes per judge are aggregated by
snapping the per-(option,dim) mean to {-1,-0.5,0,+0.5,+1}; pass agreement is recorded as a
confidence. Multi-judge is supported (--judges a,b) and aggregated the same way, so a second
independent judge can be folded in later without re-annotating.

  python mass/annotate_option_loadings.py --judges spark-latest --passes 3 [--sample N]
  -> data/source_option_loadings.csv

Env: OPENAI_BASE_URL, OPENAI_API_KEY.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from run_mass import chat, safe_name  # noqa: E402

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "/home/samcardillo/.local/bin/claude")


def judge_call(base_url, api_key, judge, prompt, temperature, max_tokens=700):
    """Dispatch one annotation call. 'cli:<model>' routes through the Claude Code CLI
    headless (a cross-family judge, e.g. cli:claude-fable-5); anything else is an HTTP
    call to the local OpenAI-compatible stack (LiteLLM -> vLLM/llama.cpp)."""
    if judge.startswith("cli:"):
        model = judge[4:]
        for attempt in range(4):
            try:
                r = subprocess.run([CLAUDE_BIN, "-p", prompt, "--model", model],
                                   capture_output=True, text=True, timeout=180)
                if r.returncode == 0 and (r.stdout or "").strip():
                    return r.stdout
                last = f"__ERROR__ rc={r.returncode} {(r.stderr or '')[:120]}"
            except Exception as e:  # noqa: BLE001
                last = f"__ERROR__ {e}"
            time.sleep(3 * (attempt + 1))  # backoff on transient rate-limit / balance errors
        return last
    return chat(base_url, api_key, judge, prompt, temperature, max_tokens=max_tokens)

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
csv.field_size_limit(10 ** 7)

DIMS = ["economic", "social", "civil_liberties", "authority", "institutions",
        "national_identity", "foreign_policy", "environment", "religion",
        "technocracy", "populism", "personal"]
ALLOWED = [-1.0, -0.5, 0.0, 0.5, 1.0]

# Two named poles per dimension. The model only names WHICH pole a choice leans (concrete
# words, never a signed number) and we own the sign, so it can never invert an axis whose
# name suggests the wrong direction (civil_liberties / environment / technocracy footguns).
# neg_label -> negative sign, pos_label -> positive sign.
POLES = {
    "economic":          ("redistribution", "free_market"),
    "social":            ("progressive",    "traditional"),
    "civil_liberties":   ("liberty",        "security"),
    "authority":         ("anti_authority", "strong_authority"),
    "institutions":      ("distrust",       "trust"),
    "national_identity": ("cosmopolitan",   "nationalist"),
    "foreign_policy":    ("dovish",         "hawkish"),
    "environment":       ("green",          "growth"),
    "religion":          ("secular",        "religious"),
    "technocracy":       ("expert_led",     "popular_control"),
    "populism":          ("institutionalist", "populist"),
}
POLE_SIGN = {}
for _d, (_neg, _pos) in POLES.items():
    POLE_SIGN[(_d, _neg)] = -1.0
    POLE_SIGN[(_d, _pos)] = +1.0

RUBRIC = """You are a political-science expert annotating survey ANSWER OPTIONS for ideological content.

For each answer option, decide what ideological stance a respondent signals by CHOOSING that
option. Tag only the dimensions the choice clearly signals (most options touch 1-3; some touch
none). For each, name WHICH POLE the choice leans and how strongly (strength 1 = clear/strong,
0.5 = mild/partial). Never output numbers for the axis direction — just the pole word below.

Score the stance the choice REVEALS, not the topic of the question. A pole is defined by its
described content, NOT by whether it is "left" or "right": e.g. gun control, hate-speech limits
and public-health mandates are the SECURITY pole of civil_liberties (a restriction) even though
they are left-coded; defending free speech or opposing surveillance is the LIBERTY pole even if
right-coded. Judge by the concrete descriptions:

DIMENSIONS  (dim: pole "<word>" = ...  |  pole "<word>" = ...):
economic:          "redistribution" = public provision, regulation, worker protection, egalitarian outcomes | "free_market" = free markets, low taxes, privatization, deregulation, property rights
social:            "progressive" = permissive, anti-discrimination, gender/sexual liberalism | "traditional" = socially conservative, conventional family/gender/sexual norms
civil_liberties:   "liberty" = individual rights, due process, privacy, free expression, limits on policing/surveillance | "security" = order over liberty, policing power, surveillance, restriction for safety
authority:         "anti_authority" = anti-authoritarian, decentralized, skeptical of coercive state power | "strong_authority" = law-and-order, strong state, obedience, hierarchy, coercive enforcement
institutions:      "distrust" = anti-establishment, anti-elite, low confidence in courts/media/parliament/science | "trust" = confidence in courts, media, parliament, science, elections, international bodies
national_identity: "cosmopolitan" = multicultural, pro-immigration, global identity | "nationalist" = sovereigntist, cultural protection, immigration restriction
foreign_policy:    "dovish" = diplomacy, non-intervention, international cooperation | "hawkish" = military force, unilateralism, strategic dominance
environment:       "green" = precaution, climate action, environmental regulation, sustainability | "growth" = industry/energy development prioritized over environmental constraint
religion:          "secular" = secularism, church-state separation, low religious authority in politics | "religious" = religiosity, religious authority/tradition in public life
technocracy:       "expert_led" = expert governance, scientific administration, evidence-based policy | "popular_control" = common-sense popular control over experts, skepticism of expert rule
populism:          "institutionalist" = pluralist, procedural democracy, checks and balances | "populist" = people-vs-elite framing, anti-establishment, majoritarian anti-elite politics

is_personal FLAG: set true when the whole item is about private life, personal wellbeing, a
private preference, or a factual/perceptual belief with no clean political stance (then give
every option an empty tag list []). This is a flag, not an ideology axis.

Rules:
- Sparse: tag a dimension only when the choice clearly signals a stance on it.
- Neutral / "don't know" / middle / "not sure" options usually get [].
- Opposing options on the same axis should name OPPOSITE poles.
- Options are NOT assumed ordered; judge each on its own content.
- Use ONLY the pole words listed above, spelled exactly.

Respond with STRICT JSON only, no prose, no markdown fences:
{"is_personal": <bool>, "options": [{"i": <index>, "tags": [{"dim": "<dim>", "pole": "<word>", "strength": 1 or 0.5}, ...]}, ...]}
One object per option, in index order. No tags = "tags": []."""


def build_prompt(question, options):
    lines = [f"  [{i}] {o}" for i, o in enumerate(options)]
    return (f"{RUBRIC}\n\nQUESTION:\n{question}\n\nANSWER OPTIONS:\n" + "\n".join(lines) +
            f"\n\nReturn JSON for exactly {len(options)} options (indices 0..{len(options)-1}).")


def snap(x):
    return min(ALLOWED, key=lambda a: abs(a - x))


def parse_one_obj(obj, n_options):
    """One annotation object {is_personal, options:[{i,tags:[{dim,pole,strength}]}]}
    -> (is_personal, [ {dim: value} per option ]). Maps (dim,pole)->signed value."""
    is_personal = bool(obj.get("is_personal", False))
    per = [dict() for _ in range(n_options)]
    for o in obj.get("options", []):
        try:
            i = int(o["i"])
        except Exception:  # noqa: BLE001
            continue
        if not (0 <= i < n_options):
            continue
        clean = {}
        for tag in (o.get("tags") or []):
            d = tag.get("dim")
            pole = (tag.get("pole") or "").strip().lower()
            if (d, pole) not in POLE_SIGN:
                continue
            try:
                strength = float(tag.get("strength", 1))
            except Exception:  # noqa: BLE001
                strength = 1.0
            strength = 0.5 if strength <= 0.5 else 1.0
            clean[d] = POLE_SIGN[(d, pole)] * strength
        per[i] = clean
    return is_personal, per


def parse_annotation(text, n_options):
    """Single-question response -> (is_personal, per_option_loads) or None on failure."""
    t = (text or "").strip()
    if t.startswith("__ERROR__") or not t:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None
    return parse_one_obj(obj, n_options)


def build_batch_prompt(batch):
    """batch: list of (q_index, question, options). One prompt, many questions."""
    blocks = []
    for qi, question, options in batch:
        opt_lines = "\n".join(f"    [{i}] {o}" for i, o in enumerate(options))
        blocks.append(f"QUESTION {qi} ({len(options)} options):\n{question}\nOPTIONS:\n{opt_lines}")
    body = "\n\n".join(blocks)
    return (RUBRIC + "\n\nAnnotate EACH question below independently. Respond with STRICT JSON "
            "only, no prose:\n"
            '{"annotations": [{"q": <question number>, "is_personal": <bool>, '
            '"options": [{"i": <index>, "tags": [{"dim": "<dim>", "pole": "<word>", '
            '"strength": 1 or 0.5}, ...]}, ...]}, ...]}\n'
            "Include one object per question, each with one object per its options.\n\n" + body)


def parse_batch(text, batch):
    """-> {q_index: (is_personal, per_option_loads)} for questions that parsed cleanly."""
    t = (text or "").strip()
    if t.startswith("__ERROR__") or not t:
        return {}
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return {}
    nopt = {qi: len(options) for qi, _, options in batch}
    out = {}
    for a in obj.get("annotations", []):
        try:
            qi = int(a["q"])
        except Exception:  # noqa: BLE001
            continue
        if qi not in nopt:
            continue
        out[qi] = parse_one_obj(a, nopt[qi])
    return out


def annotate_one(base_url, api_key, judge, temperature, question, options, passes):
    """Run `passes` stochastic annotations; aggregate. Returns (is_personal, per_option_loads, conf)."""
    prompt = build_prompt(question, options)
    n = len(options)
    # collect per (option,dim) list of values, and personal votes
    vals = [defaultdict(list) for _ in range(n)]
    personal_votes = 0
    ok = 0
    # scale token budget with option count so many-option items don't truncate (~180 tok/option)
    mt = max(700, 300 + 180 * n)
    for p in range(passes):
        raw = judge_call(base_url, api_key, judge, prompt, temperature if passes > 1 else 0.0,
                         max_tokens=mt)
        parsed = parse_annotation(raw, n)
        if parsed is None:
            continue
        ok += 1
        is_personal, per = parsed
        personal_votes += 1 if is_personal else 0
        for i in range(n):
            reported = per[i]
            for d in DIMS:
                if d == "personal":
                    continue
                vals[i][d].append(reported.get(d, 0.0))
    if ok == 0:
        return None
    is_personal = personal_votes > ok / 2.0
    per_final, confs = [], []
    for i in range(n):
        final = {}
        agree_terms = []
        touched = [d for d in DIMS if d != "personal" and any(abs(x) > 1e-9 for x in vals[i][d])]
        for d in touched:
            series = vals[i][d] + [0.0] * (ok - len(vals[i][d]))
            sv = snap(sum(series) / ok)
            if abs(sv) > 1e-9:
                final[d] = sv
            agree_terms.append(sum(1 for x in series if snap(x) == sv) / ok)
        conf = round(sum(agree_terms) / len(agree_terms), 3) if agree_terms else 1.0
        per_final.append(final)
        confs.append(conf)
    return is_personal, per_final, confs, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="spark-latest", help="comma-list of judge models")
    ap.add_argument("--passes", type=int, default=3, help="stochastic passes per judge")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--batch", type=int, default=0, help="questions per call (0=one-per-call). "
                    "Amortizes CLI system-prompt overhead; single judge, single pass.")
    ap.add_argument("--sample", type=int, default=0, help="cap #questions (0 = all)")
    ap.add_argument("--sources", default="", help="comma-list source filter")
    ap.add_argument("--out", default=os.path.join(DATA, "source_option_loadings.csv"))
    ap.add_argument("--resume", default="", help="existing CSV: carry over its good items, "
                    "only re-annotate ERROR/missing ones (repairs a partial run)")
    args = ap.parse_args()
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:4000/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "sk-none")
    judges = [j for j in args.judges.split(",") if j]

    rows = list(csv.DictReader(open(os.path.join(DATA, "mass_opinion_questions.csv"), encoding="utf-8")))
    if args.sources:
        keep = set(args.sources.split(","))
        rows = [r for r in rows if r["source"] in keep]
    if args.sample and len(rows) > args.sample:
        rows = rows[:args.sample]

    # --resume: carry over items already annotated cleanly; only redo ERROR/missing ones.
    carried, good_ids = {}, set()
    if args.resume and os.path.exists(args.resume):
        prev = defaultdict(list)
        for pr in csv.DictReader(open(args.resume, encoding="utf-8")):
            prev[pr["item_id"]].append(pr)
        for iid, prs in prev.items():
            if prs and not any(p["primary_dimension"] == "ERROR" for p in prs):
                carried[iid] = prs
                good_ids.add(iid)
    todo = [r for r in rows if r["item_id"] not in good_ids]
    print(f"annotating {len(todo)} questions ({len(good_ids)} carried over) x {len(judges)} "
          f"judge(s) x {args.passes} passes (concurrency {args.concurrency})", file=sys.stderr)

    def do(r):
        options = json.loads(r["options"])
        # aggregate across judges: collect each judge's per-(option,dim) snapped value
        j_personal, j_perload, j_conf, j_ok = [], [], [], []
        for judge in judges:
            res = annotate_one(base_url, api_key, judge, args.temperature,
                               r["question"], options, args.passes)
            if res is None:
                continue
            isp, per, confs, ok = res
            j_personal.append(isp); j_perload.append(per); j_conf.append(confs); j_ok.append(ok)
        if not j_perload:
            return r, options, None
        n = len(options)
        is_personal = sum(j_personal) > len(j_personal) / 2.0
        per_final, confs = [], []
        for i in range(n):
            acc = defaultdict(list)
            for per in j_perload:
                for d in DIMS:
                    if d == "personal":
                        continue
                    acc[d].append(per[i].get(d, 0.0))
            final = {}
            for d, series in acc.items():
                sv = snap(sum(series) / len(series))
                if abs(sv) > 1e-9:
                    final[d] = sv
            per_final.append(final)
            confs.append(round(sum(c[i] for c in j_conf) / len(j_conf), 3))
        return r, options, (is_personal, per_final, confs, sum(j_ok))

    header = (["item_id", "source", "source_key", "option_index", "option_text",
               "n_options", "options_ordered", "is_personal", "primary_dimension",
               "primary_load"] + [f"load_{d}" for d in DIMS] +
              ["confidence", "judges", "passes_ok"])

    def rows_for(r, options, res):
        """Build the option-rows (list of value-lists) for one question."""
        if res is None:
            return [[r["item_id"], r["source"], r["source_key"], i, opt,
                     r["n_options"], r["options_ordered"], "", "ERROR", 0.0] +
                    [0.0] * len(DIMS) + [0.0, ",".join(judges), 0]
                    for i, opt in enumerate(options)]
        is_personal, per_final, confs, ok = res
        out = []
        for i, opt in enumerate(options):
            loadvec = [per_final[i].get(d, 0.0) for d in DIMS]
            if is_personal:
                loadvec = [0.0] * len(DIMS)
                loadvec[DIMS.index("personal")] = 1.0
            nonzero = {d: v for d, v in zip(DIMS, loadvec) if abs(v) > 1e-9 and d != "personal"}
            if is_personal:
                pdim, pload = "personal", 1.0
            elif nonzero:
                pdim = max(nonzero, key=lambda d: abs(nonzero[d]))
                pload = nonzero[pdim]
            else:
                pdim, pload = "none", 0.0
            out.append([r["item_id"], r["source"], r["source_key"], i, opt,
                        r["n_options"], r["options_ordered"],
                        is_personal, pdim, pload] + loadvec + [confs[i], ",".join(judges), ok])
        return out

    # annotate only the to-do questions, keyed by item_id
    computed = {}
    if args.batch > 0:
        judge = judges[0]   # batching is single-judge, single-pass
        chunks = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]

        def do_chunk(chunk):
            batch = [(k, r["question"], json.loads(r["options"])) for k, r in enumerate(chunk)]
            raw = judge_call(base_url, api_key, judge, build_batch_prompt(batch), 0.0, max_tokens=6000)
            parsed = parse_batch(raw, batch)
            results = []
            for k, r in enumerate(chunk):
                options = json.loads(r["options"])
                if k in parsed:
                    isp, per = parsed[k]
                    results.append((r, options, (isp, per, [1.0] * len(options), 1)))
                else:
                    results.append((r, options, None))   # failed -> per-question retry
            return results

        failed = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            done = 0
            for results in ex.map(do_chunk, chunks):
                done += 1
                if done % 20 == 0:
                    print(f"  ... batch {done}/{len(chunks)}", file=sys.stderr)
                for r, options, res in results:
                    if res is None:
                        failed.append(r)
                    else:
                        computed[r["item_id"]] = rows_for(r, options, res)
        if failed:
            print(f"  retrying {len(failed)} unparsed questions individually", file=sys.stderr)
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                for r, options, res in ex.map(do, failed):
                    computed[r["item_id"]] = rows_for(r, options, res)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            done = 0
            for r, options, res in ex.map(do, todo):
                done += 1
                if done % 100 == 0:
                    print(f"  ... {done}/{len(todo)}", file=sys.stderr)
                computed[r["item_id"]] = rows_for(r, options, res)

    # write in the original question order: carried rows verbatim, else freshly computed
    written = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            iid = r["item_id"]
            if iid in carried:
                for pr in carried[iid]:
                    w.writerow([pr[h] for h in header]); written += 1
            else:
                for row in computed.get(iid, []):
                    w.writerow(row); written += 1
    print(f"Wrote {written} option rows -> {args.out} "
          f"({len(good_ids)} carried, {len(todo)} re-annotated)")


if __name__ == "__main__":
    main()
