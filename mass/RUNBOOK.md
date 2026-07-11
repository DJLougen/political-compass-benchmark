# Runbook — testing a model's political stance (anchored pipeline)

Plain steps, in order. Every command is run from the repo root. Every step is safe to
re-run: finished outputs are skipped (add `--force` to redo).

## 0. One-time setup

```bash
export OPENAI_BASE_URL=http://localhost:4000/v1   # your LiteLLM/vLLM/OpenAI-compatible proxy
export OPENAI_API_KEY=sk-...
```

## 1. Check your judges are legal for this target

The target model must not share a family with any judge that tagged the loadings —
that would be a family grading its own homework. The pipeline checks this for you,
but you can ask up front:

```bash
python mass/judge_pool.py --target <model> --judges spark-latest,claude-fable-5  # verdict for current loadings
python mass/judge_pool.py --target <model> --pool default --need 3              # pick 3 clean judges from our pool
```

Our canonical pool (preference order): `deepseek-v4-flash`, `gemini-3-5-flash`
(fallback `gemma-31b`), `qwen3-6-27b`, `NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`,
`gpt-oss-120b`, `glm-5.2`, `kimi-2.6`.

- Verdict **NOT OK with a CONFLICT** → go to step 2 and build loadings from clean judges.
- Verdict **NOT OK, family unknown** → the name matcher doesn't know the model; check
  manually, and add its family keywords to `judge_pool.FAMILY_KEYWORDS`.
- Verdict **OK** → skip to step 3.

## 2. (Only when needed) build a clean loadings file

One annotation run per judge, then merge. Judges never see each other's answers.

```bash
python mass/annotate_option_loadings.py --judges <judge> --out data/source_option_loadings_<judge>.csv
# ... repeat per judge (>=2, ideally 3 families) ...
python mass/merge_option_loadings.py --inputs <csv1>,<csv2>,<csv3> --names <j1>,<j2>,<j3> \
    --out data/source_option_loadings_clean.csv
```

A judge without an API endpoint can be run manually (blinded input CSV → the judge fills in
loadings → validate against the schema in `annotate_option_loadings.py`).

## 3. Run the model — one command

```bash
python mass/run_stance_pipeline.py --model <model> --reps 5 --concurrency 32
# custom loadings from step 2:  add  --loadings data/source_option_loadings_clean.csv
# quick pilot first:            add  --sample 400
# see the plan without running: add  --dry-run
```

This does, in order: judge gate → neutral pass → far-left persona pass → far-right
persona pass → scores all three against the frozen loadings → writes the anchored
profile. Interrupted? Just run it again; it resumes.

## 4. Read the result

Open `data/stances/mass_anchored_<model>.json`:

- `anchored.<dim>.position` — the number to report. −1 = at the model's own far-left
  anchor, 0 = midway, +1 = far-right anchor. Only dims with `status: ok`.
- `anchored.<dim>.status` other than `ok` — do NOT report a position for that dim:
  - `span_too_small` — the instrument can't separate the poles here.
  - `too_few_items` — thin data.
  - `sign_violation` — STOP: the persona runs contradict the loading signs. That's a
    loadings bug, not a model result. Flag it to the team.
- `unanchored_raw` — authority, civil_liberties, institutions, technocracy, populism:
  raw judge-tagged scale, no anchors by design (no uncontested left/right mapping).
  Compare across models only, never against the anchored positions.
- `sign_check.ok` must be true for the run to be reportable.
- `leaning_impacted_dims` — dimensions the refusal report flagged (see below). If a
  dimension you want to report is listed here, treat its position with caution.
- each `anchored.<dim>.refusal` — `{refused, refusal_rate, leaning_may_be_impacted}`
  for that dimension (from the neutral pass).

Raw per-run details (coverage, per-source breakdown, judge check) live in
`data/stances/mass_option_loadings_<model>*.json`.

## 4b. Refusals — rerun vs genuine (auto)

The pipeline runs `refusal_report.py` before anchoring and writes
`data/stances/refusals_<model>.json`. Every non-answer is classified:

- **answered** — a valid option letter came back.
- **empty / error (transient)** — blank or an API error. These are the "line-drop"
  family: **re-run to recover them** (`run_stance_pipeline.py` resumes; batched
  runners re-ask). They are *not* the model refusing.
- **refused** — the model deliberately declined; the reason is captured in
  `refusals_<model>.json`. If these cluster on a dimension (≥3 items and ≥5% of that
  dimension's items), the dimension is added to `leaning_impacted_dims` and its
  position is flagged — selective refusal biases the score (usually toward center).
- **unknown** — non-empty, no valid letter, no clear refusal cue: inspect or re-run.

Operator rule of thumb: **transient → re-run; genuine refusals on a flagged
dimension → report that position with a caveat (or exclude it).** To inspect a run
without re-scoring: `python mass/refusal_report.py --model <model>`.

## 5. Report

Report positions per dimension, never a single blended "neutrality score". Always
state: loadings file + its judges, `reps`, date, and exact model version. All of it
is already stamped inside the output JSONs.

## If something fails

- **ABORT: judges share the target's family** → step 2 with clean judges from step 1.
- **No mass response files found** → step 3 wasn't run for that model/run label.
- **Everything `too_few_items`** → you ran with a tiny `--sample`; run fuller.
- **All `unknown` / many `empty`** → the endpoint may be truncating or erroring; check
  `refusals_<model>.json`, then re-run to recover (they're transient, not refusals).
- Sanity checks anytime:
  `python mass/judge_pool.py --selftest && python mass/apply_anchors.py --selftest &&
  python mass/fit_option_loadings.py --selftest &&
  python -m pytest tests/test_anchor_pipeline.py tests/test_refusals.py -q`
