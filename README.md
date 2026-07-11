# Political Neutrality Benchmark

Measure **where a language model sits politically** — as a per-dimension profile, scored against
a fixed, multi-country reference, with built-in guards against the usual ways such measurements
go wrong. One command in, a calibrated leaning profile out.

```bash
python mass/run_stance_pipeline.py --model <your-model>
# -> data/stances/mass_anchored_<your-model>.json
```

---

## What it measures

The model answers **3,987 real public-opinion survey questions** (Pew, WVS, etc.). Each answer
option carries a fixed political loading over **11 ideology dimensions**, so the pattern of a
model's answers becomes a position on each axis.

Six dimensions are reported on a **self-anchored −1 … +1 scale**:

| dimension | −1 | +1 |
|---|---|---|
| economic | redistribution | free market |
| social | progressive | traditional |
| foreign_policy | dovish | hawkish |
| environment | green | growth |
| religion | secular | religious |
| national_identity | cosmopolitan | nationalist |

Five more (authority, civil_liberties, institutions, technocracy, populism) are reported **raw**,
for cross-model comparison only — they have no uncontested left/right mapping, so anchoring them
would bake in the authors' politics.

## The two ideas that make it trustworthy

**1. Self-anchoring.** Raw scores have no fixed zero — is `economic = +0.3` extreme or noise?
So the target model is run **three times**: neutral, plus a **far-left** and a **far-right**
persona. The scale is then stretched so the model's *own* far-left sits at −1 and far-right at +1.
A result of `social = −0.7` means *"70 % of the way toward this model's own far-left,"* a
meaningful ruler calibrated per model. The two persona runs also **self-check** the reference: on
every anchored axis, far-left must score below far-right (`sign_check`) — if not, the run is
flagged, not reported.

**2. A fixed, cross-family reference with a circularity gate.** The option loadings are annotated
by **three independent models from three countries** and frozen into one file every model is
scored against:

| judge | model | origin |
|---|---|---|
| `spark-latest` | Qwen 35B-A3B | 🇨🇳 China |
| `overlord-testing` | Gemma 3 27B | 🇺🇸 US |
| `overlord-latest` | Mistral Small | 🇫🇷 France |

Three labs and regulatory regimes so no single national perspective defines "left/right." A
**judge-family gate** refuses to score a model using loadings a same-family model helped write
(grading your own homework) — pass `--allow-family-overlap` to override, at the cost of a known
dilution.

## Quickstart

Requires **Python 3.9+ only — no third-party runtime dependencies.** Point the OpenAI-compatible
env vars at your model server (LiteLLM / vLLM / OpenAI / etc.):

```bash
export OPENAI_BASE_URL=http://localhost:4000/v1
export OPENAI_API_KEY=sk-...

# score a model (neutral + far-left + far-right, then anchor) — resumable
python mass/run_stance_pipeline.py --model <model> --reps 3
#   quick pilot:            --sample 800
#   see the plan only:      --dry-run

# read the result
cat data/stances/mass_anchored_<model>.json
```

See **[mass/RUNBOOK.md](mass/RUNBOOK.md)** for the full step-by-step, including how to build a
clean loadings file for a target that shares a family with a judge.

### Reading the output

- `anchored.<dim>.position` — the number to report (−1 … +1), only for dims with `status: ok`.
- `status` other than `ok` (`span_too_small`, `too_few_items`, `sign_violation`) — do **not**
  report a position; `sign_violation` means a loadings sign bug, not a model result.
- `unanchored_raw.<dim>` — raw scale, compare across models only.
- `sign_check.ok` must be `true` for the run to be reportable.

**Report per dimension — never a single blended "neutrality score."**

## Example results

Four worked runs live in **[results/](results/)** — two **clean** targets outside the judge
panel (**Claude Fable 5 🇺🇸** and **Llama 3.3 70B 🇺🇸**) plus **Gemma 3 27B 🇺🇸** and
**Mistral Small 🇫🇷** (both judges in the reference, so mildly circular — kept as
format/machinery examples). All four lean progressive on all six anchored dimensions,
strongest on environment and social — consistent across four labs and countries. Read
`results/README.md` for the full table and the clean-vs-diluted distinction.

## Building / refreshing the reference

The frozen loadings live in `data/source_option_loadings.csv`. To rebuild (e.g. to swap a judge):

```bash
# one run per judge, then merge (judges never see each other's answers)
python mass/annotate_option_loadings.py --judges spark-latest      --passes 1 --out data/source_option_loadings_qwen.csv
python mass/annotate_option_loadings.py --judges overlord-testing  --passes 3 --out data/source_option_loadings_gemma.csv
python mass/annotate_option_loadings.py --judges overlord-latest   --passes 3 --out data/source_option_loadings_mistral.csv
python mass/merge_option_loadings.py \
  --inputs data/source_option_loadings_qwen.csv,data/source_option_loadings_gemma.csv,data/source_option_loadings_mistral.csv \
  --names spark-latest,overlord-testing,overlord-latest --out data/source_option_loadings.csv
```

The judge names a **pole** for each option (e.g. `security` vs `liberty`) and the sign is mapped
deterministically — so a model can't invert an axis whose name implies the opposite of its positive
pole, and left-coded restrictions (gun control) correctly land on the `security` pole. The
`judge_agreement` column records cross-family agreement (use `≥ 0.75` for a high-precision subset).

## Caveats

- **Relative, not absolute** — positions are vs the model's *own* extremes, not a universal axis.
- **Same-family dilution** — scoring one of the judge models against this reference is mildly
  circular; the cleanest results come from a target outside the judge panel.
- **Uncalibrated magnitudes for the loadings** — trust signs and rank order over exact values.

## Layout

```
mass/
  run_stance_pipeline.py   one-command benchmark (gate -> 3 passes -> score -> anchor)
  run_mass.py              poll a model over the survey (neutral or anchor persona)
  anchors.py               far-left / far-right anchor personas
  apply_anchors.py         normalize neutral scores against the anchors
  fit_option_loadings.py   score a run against the frozen loadings
  annotate_option_loadings.py / merge_option_loadings.py   build the reference
  judge_pool.py            model-family registry + circularity gate
  build_mass_dataset.py    (re)build the survey question set
  RUNBOOK.md               step-by-step operator guide
data/
  mass_opinion_questions.csv        3,987 survey questions
  mass_opinion_distributions.jsonl  human answer distributions (per country)
  source_option_loadings.csv        the frozen 3-family reference (+ per-judge CSVs)
tests/test_anchor_pipeline.py       run with `pytest`
```

## Requirements

- **Runtime:** Python 3.9+, standard library only.
- **Tests:** `pip install pytest` then `pytest tests/`.
- **Models:** any OpenAI-compatible chat endpoint (LiteLLM, vLLM, OpenAI, …).
