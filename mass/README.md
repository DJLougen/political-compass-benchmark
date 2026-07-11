# Data reference — survey questions + frozen loadings

A single unified corpus of **mass public-opinion survey questions**, harvested and
reformatted from three public sources into one ready-to-poll schema.

`python mass/build_mass_dataset.py` → `data/mass_opinion_questions.csv` (+ distributions).

## Contents

| source tag | origin | n | human distributions |
|---|---|---:|---|
| `opinionqa_pew_atp` | Pew **American Trends Panel** (US), via OpinionsQA (Santurkar et al. 2023) | 1,440 | no |
| `anthropic_pew_gas` | Pew **Global Attitudes Survey** (multi-country), via Anthropic/llm_global_opinions (Durmus et al. 2023) | 2,196 | yes (per country) |
| `anthropic_wvs` | **World Values Survey wave 7**, via Anthropic/llm_global_opinions | 351 | yes (per country) |
| **total** | | **3,987** | 2,547 with distributions |

### On WVS-7
The WVS-7 questionnaire on `worldvaluessurvey.org` is served through a JS session /
data-use-agreement download gate (not a static file), so it is **not scraped directly**.
Anthropic's `llm_global_opinions` already packages WVS-7 items under a redistributable
license; those rows carry `source = anthropic_wvs` so they remain identifiable and
separable. If you have accepted the WVS data-use agreement you can substitute the
official questionnaire; the schema is unchanged.

## Schema — `data/mass_opinion_questions.csv`

- `item_id` — `MASS####`, unique
- `source` — one of the three tags above
- `source_key` — original question id (Pew ATP key like `ECON4_b_W54`; blank for Anthropic rows)
- `question` — question text
- `options` — JSON array of answer options, **DK/Refused removed** (see `has_dk`)
- `n_options` — count after cleaning
- `options_ordered` — **`True`** for Anthropic (list order preserved from the survey);
  **`False`** for OpinionsQA — the upstream mirror stored options as an unordered dict and
  **lost the ordinal scale order** (e.g. `ECON4` returns "great deal / fair amount / not at
  all / not too much"). Do **not** treat OpinionsQA options as a monotone Likert scale.
- `has_dk` — whether a Don't-Know/Refused option existed before stripping
- `response_format` — `mc_single`
- `prompt` — fully rendered single-choice poll (lettered options + minimal instruction)
- `has_human_distribution` — whether ground-truth human answers exist in the distributions file

## Human distributions — `data/mass_opinion_distributions.jsonl`

One line per question with human data (Anthropic sources):
`{item_id, source, options_original, options_clean, kept_idx, distribution_by_country}`.
`distribution_by_country[country]` aligns to **`options_original`** (includes DK). `kept_idx`
maps the presented/cleaned options back into that vector, so to compare a model answer (over
`options_clean`) to humans you take `[vec[i] for i in kept_idx]` and renormalise — which is
exactly what the scorer does.

## Frozen cross-family option-loading dataset — `data/source_option_loadings.csv`

An **externally-fixed** political-loading annotation for the ORIGINAL survey questions — the
reference the projective/mass tracks lacked, whose poles are *not* model-generated. For every
question in `mass_opinion_questions.csv` and **every answer option**, it records the ideological
stance signalled by *choosing* that option, over sparse political dimensions plus a personal flag,
agreed by **three independent model families from three countries** — the canonical panel:

| judge (LiteLLM alias) | model | origin |
|---|---|---|
| `spark-latest` | Qwen 35B-A3B | 🇨🇳 China |
| `overlord-testing` | Gemma 3 27B | 🇺🇸 US (Google) |
| `overlord-latest` | Mistral Small | 🇫🇷 France |

Three distinct labs and regulatory regimes so no single national perspective sets what "left/right"
means. All local (no per-call cost). The raw dataset CSV is untouched — this is a sidecar.

```bash
export OPENAI_BASE_URL=http://localhost:4000/v1 OPENAI_API_KEY=sk-...
# one run per judge (judges never see each other's answers), then merge
python mass/annotate_option_loadings.py --judges spark-latest      --passes 1 --out data/source_option_loadings_qwen.csv
python mass/annotate_option_loadings.py --judges overlord-testing  --passes 3 --out data/source_option_loadings_gemma.csv
python mass/annotate_option_loadings.py --judges overlord-latest   --passes 3 --out data/source_option_loadings_mistral.csv
python mass/merge_option_loadings.py \
    --inputs data/source_option_loadings_qwen.csv,data/source_option_loadings_gemma.csv,data/source_option_loadings_mistral.csv \
    --names spark-latest,overlord-testing,overlord-latest --out data/source_option_loadings.csv
```

Two serving gotchas the pipeline now handles automatically: **Mistral-tokenizer** vLLM rejects
`chat_template_kwargs` (HTTP 400) — `chat()` retries without it; and many-option items (e.g. an
11-option party-ID question) need a bigger token budget — `max_tokens` now scales with option
count so the JSON never truncates. The DGX **Spark** (Qwen) degrades under sustained high
concurrency (it is the prod box) — run Qwen at low concurrency / single-pass.

**11 political dimensions** (each with two named poles → deterministic sign; − / +), plus one
nonpolitical flag:
economic (redistribution/free_market), social (progressive/traditional), civil_liberties
(liberty/security), authority (anti_authority/strong_authority), institutions (distrust/trust),
national_identity (cosmopolitan/nationalist), foreign_policy (dovish/hawkish), environment
(green/growth), religion (secular/religious), technocracy (expert_led/popular_control), populism
(institutionalist/populist), and **personal** — a *flag* for private/perceptual
items with no clean stance.

**Method that matters:** the judge never emits a signed number for an axis — it names *which
pole* a choice leans (e.g. `security` vs `liberty`) and the sign is mapped deterministically.
This defeats a real failure mode where a model inverts an axis whose *name* implies the opposite
of its positive pole (`civil_liberties`, `environment`, `technocracy`), and correctly puts
left-coded restrictions (gun control, mandates) on the `security` pole. Values are snapped to
{−1, −0.5, 0, +0.5, +1}; most options load on 1–3 dims.

Schema (per option): `item_id, source, source_key, option_index, option_text, n_options,
options_ordered, is_personal, primary_dimension, primary_load, load_<12 fields>, judge_agreement,
within_judge_conf, n_judges, judges`.

Build result (3,987 questions → **14,395 option rows, 0 errors**): 21% of rows flagged
`is_personal`; sign balance ≈ 0.51 (opposing options get opposite signs, no all-negative artifact);
**`n_judges = 3` on every row**. **`judge_agreement`** = how much the three families agree on the
dims any of them tagged — mean **0.735**; 43% of rows at full three-way agreement, ~79% at ≥ 0.75.
Use it as a trust filter (e.g. keep `judge_agreement ≥ 0.75` for a high-precision subset). Per-judge
inputs are kept (`_qwen.csv`, `_gemma.csv`, `_mistral.csv`) for transparency; a fourth judge drops
in via the same `--judges` / merge path. This is the ruler `run_stance_pipeline.py` scores against
by default.

## Anchored direct stance pipeline — v7/v8

The clean stance path: run a target model on the **original survey questions**, map the answers
it selects through the frozen option-loading sidecar, then normalize against the model's own
far-left / far-right persona runs. No judge model is called at scoring time, and the original
question CSV remains untouched.

**Team runbook: [`mass/RUNBOOK.md`](RUNBOOK.md)** — plain what-to-do-at-each-step. One command
per target model:

```bash
export OPENAI_BASE_URL=http://localhost:4000/v1 OPENAI_API_KEY=sk-...
python mass/run_stance_pipeline.py --model <m> --reps 5 --concurrency 32
# -> data/stances/mass_anchored_<m>.json
```

Add `--dry-run` to see the exact steps, `--sample 400` for a cheap pilot, `--force` to rerun
completed steps (each step is otherwise resumed/skipped when its output exists). The pipeline is:

```
0. judge-family gate   judge_pool.py: target must not share a model family with any judge
                       that produced the loadings file (circularity guard, see below)
1. neutral pass        run_mass.py --model <m>                 the model as itself
2. far-left anchor     run_mass.py --anchor far_left           same model, far-left persona
3. far-right anchor    run_mass.py --anchor far_right          same model, far-right persona
4. score all 3 runs    fit_option_loadings.py                  frozen judge-tagged loadings
5. anchor-normalize    apply_anchors.py                        -> mass_anchored_<m>.json
```

**Scoring** (step 4) is item-weighted: repeated samples for one question are averaged into that
item's answer distribution first, then each item contributes once to each dimension it actually
measures. A dimension score is averaged only over items with at least one answer option loading
on that dimension, so unrelated questions do not dilute the axis toward zero. `personal` is
reported under `nonpolitical_flags`, not `stance`, because it is not an ideological pole.

**Anchoring** (steps 2/3/5, `anchors.py` + `apply_anchors.py`): raw scores are judge-tagged,
uncalibrated magnitudes — nothing says whether `economic = +0.3` is extreme. The anchor runs are
the same target model role-playing a far-left and a far-right persona; the neutral pass is the
model's normal run. Per dimension, `position = 2*(neutral - midpoint)/span` puts the far-left run
at −1 and far-right at +1. Anchors are defined **only** on the six dimensions with an uncontested
left↔right mapping (economic, social, foreign_policy, environment, religion, national_identity;
left = negative pole on all six). The contested dimensions (authority, civil_liberties,
institutions, technocracy, populism) are deliberately unanchored — a far-left anarchist and a
far-left Marxist-Leninist disagree about authority, so an anchor there would bake our politics
into the ruler; they are reported raw and labelled. Normalization refuses dimensions where the
anchor span is too small (`span_too_small`), data is thin (`too_few_items`), or — critically —
**inverted** (`sign_violation`): far-left scoring above far-right behaviorally contradicts the
frozen loading signs with no judge in the loop, so the `sign_check` block doubles as validation
of the loading orientation itself.

**Judge-family gate** (step 0, `judge_pool.py`): judge ≠ scored model. Loadings tagged by a judge
must never score a target from the same model family or a derivative (a family grading its own
homework). The gate keyword-matches name tokens (catches `dolphin-llama3`, `Qwen2.5-...-ft`) and
knows the repo judge aliases (`spark`→Qwen, `fable`→Claude, `hy3`→Hunyuan). On conflict the
pipeline aborts before any API call and tells you how to pick clean judges.

The **canonical judge pool** (`judge_pool.DEFAULT_POOL`, preference order, 7 families):
`deepseek-v4-flash`, `gemini-3-5-flash` (fallback `gemma-31b`), `qwen3-6-27b`,
`NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`, `gpt-oss-120b`, `glm-5.2`, `kimi-2.6`.
Every major target family still gets ≥3 clean judges from this pool, and a panel never
contains two judges from the same provider:

```bash
python mass/judge_pool.py --target qwen3-235b --judges spark-latest,claude-fable-5   # verdict
python mass/judge_pool.py --target qwen3-235b --pool default --need 3                # pick clean
```

then annotate + re-merge a loadings file without the conflicted family and pass it via
`--loadings`. Unknown target families are warned, never silently passed. Overriding with
`--allow-family-overlap` stamps the contamination into the output JSON; never report such runs.

Every stance JSON records its provenance: `judge_family_check`, `loadings_file`, filters, and the
anchor input files. Selftests: `judge_pool.py --selftest`, `apply_anchors.py --selftest`,
`fit_option_loadings.py --selftest`; unit tests in `tests/test_anchor_pipeline.py`.

## How this fits the benchmark
The question set + frozen loadings feed the **anchored direct-stance** measurement: survey answers
are scored against the cross-family option loadings, then normalized against the model's own
far-left/far-right persona runs — `run_stance_pipeline.py` = `fit_option_loadings.py` +
`apply_anchors.py`, gated by `judge_pool.py`. See the top-level [README](../README.md) and
[RUNBOOK](RUNBOOK.md).

## Caveats
- Options are cleaned (DK stripped) and, for OpinionsQA, **not ordinal** — see `options_ordered`.
- Prompt phrasing is our neutral single-choice wrapper; the source instruments used their own.
- Deduplicated on normalised question text within this build only.
- Re-download is cached under `mass/.cache/` (gitignored).
