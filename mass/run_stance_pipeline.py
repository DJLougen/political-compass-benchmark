#!/usr/bin/env python3
"""
One-command anchored stance pipeline (v8). Runs, scores and anchor-normalizes a
target model over the original mass-opinion survey bank.

  export OPENAI_BASE_URL=http://localhost:4000/v1 OPENAI_API_KEY=sk-...
  python mass/run_stance_pipeline.py --model <m> --reps 5 --concurrency 32

Steps (each skipped automatically when its output already exists; --force reruns):
  0. judge-family gate     target must not share a family with any loadings judge
  1. neutral pass          run_mass.py --model <m>
  2. far-left anchor       run_mass.py --model <m> --anchor far_left
  3. far-right anchor      run_mass.py --model <m> --anchor far_right
  4. score all three       fit_option_loadings.py (frozen judge-tagged loadings)
  5. anchor-normalize      apply_anchors.py -> data/stances/mass_anchored_<m>.json

Use --dry-run to print the exact commands without executing anything.
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
import judge_pool  # noqa: E402
from anchors import anchor_run_name  # noqa: E402
from run_mass import safe_name  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def responses_path(run_label):
    return os.path.join(DATA, "responses", f"mass_{safe_name(run_label)}.jsonl")


def stance_path(run_label):
    return os.path.join(DATA, "stances", f"mass_option_loadings_{safe_name(run_label)}.json")


def run(cmd, dry):
    print("+", " ".join(cmd), flush=True)
    if dry:
        return
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"step failed (exit {r.returncode}): {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--sample", type=int, default=0, help="cap questions per run (0 = all 3,987)")
    ap.add_argument("--loadings", default=os.path.join(DATA, "source_option_loadings.csv"))
    ap.add_argument("--min-agreement", type=float, default=0.75,
                    help="judge-agreement trust filter on loadings (README-recommended 0.75)")
    ap.add_argument("--require-two-judges", action="store_true", default=True)
    ap.add_argument("--min-span", type=float, default=0.5)
    ap.add_argument("--skip-anchors", action="store_true",
                    help="neutral pass + raw scoring only (no anchor normalization)")
    ap.add_argument("--force", action="store_true", help="rerun steps whose outputs exist")
    ap.add_argument("--allow-family-overlap", action="store_true",
                    help="bypass the judge-family gate (circular; never for reported results)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    py = sys.executable

    # 0. judge-family gate — fail before burning any API calls
    from fit_option_loadings import loadings_judges
    judges = loadings_judges(args.loadings)
    verdict = judge_pool.check(args.model, judges)
    print(f"judge-family gate: target={args.model} (family {verdict['target_family']}) "
          f"judges={verdict['judges']}")
    if verdict["conflicted_judges"] and not args.allow_family_overlap:
        sys.exit(
            f"ABORT: judges {verdict['conflicted_judges']} share the target's family — scoring "
            f"would be circular.\nFix: pick clean judges "
            f"(python mass/judge_pool.py --target {args.model} --pool <candidates>), annotate + "
            f"re-merge loadings without that family, pass via --loadings.")
    if verdict["target_family_unknown"]:
        print("WARNING: target family unknown — verify judge independence manually.")

    # 1-3. response runs
    runs = [("neutral", None, args.model)]
    if not args.skip_anchors:
        runs += [("far_left anchor", "far_left", anchor_run_name(args.model, "far_left")),
                 ("far_right anchor", "far_right", anchor_run_name(args.model, "far_right"))]
    for title, anchor, run_label in runs:
        out = responses_path(run_label)
        if os.path.exists(out) and not args.force:
            print(f"skip {title}: {out} exists (use --force to rerun)")
            continue
        cmd = [py, os.path.join(HERE, "run_mass.py"), "--model", args.model,
               "--reps", str(args.reps), "--temperature", str(args.temperature),
               "--concurrency", str(args.concurrency)]
        if args.sample:
            cmd += ["--sample", str(args.sample)]
        if anchor:
            cmd += ["--anchor", anchor]
        run(cmd, args.dry_run)

    # 4. score every run against the frozen loadings
    for _, _, run_label in runs:
        if os.path.exists(stance_path(run_label)) and not args.force:
            print(f"skip scoring: {stance_path(run_label)} exists")
            continue
        cmd = [py, os.path.join(HERE, "fit_option_loadings.py"),
               "--model", run_label, "--loadings", args.loadings,
               "--min-agreement", str(args.min_agreement), "--by-source"]
        if args.require_two_judges:
            cmd += ["--require-two-judges"]
        if args.allow_family_overlap:
            cmd += ["--allow-family-overlap"]
        run(cmd, args.dry_run)

    # 5. refusal report (classifies non-answers: transient/rerun vs genuine refusal;
    #    flags anchored dimensions whose leaning may be biased by selective refusal)
    run([py, os.path.join(HERE, "refusal_report.py"), "--model", args.model,
         "--loadings", args.loadings], args.dry_run)

    # 6. anchor normalization (folds the refusal flags into the anchored profile)
    if args.skip_anchors:
        print("done (anchors skipped): raw stance at", stance_path(args.model))
        return
    run([py, os.path.join(HERE, "apply_anchors.py"), "--model", args.model,
         "--min-span", str(args.min_span)], args.dry_run)
    if not args.dry_run:
        print("done: anchored profile at",
              os.path.abspath(os.path.join(DATA, "stances", f"mass_anchored_{safe_name(args.model)}.json")))


if __name__ == "__main__":
    main()
