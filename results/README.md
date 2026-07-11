# Example results

Reference runs scored against the frozen 3-family loadings in
`data/source_option_loadings.csv`. Each file is the raw
`data/stances/mass_anchored_<model>.json` the pipeline emits.

| file | model | origin | clean? | how produced |
|---|---|---|---|---|
| `claude-fable-5_anthropic.anchored.json` | Claude Fable 5 | 🇺🇸 US | **✅ clean** | batched CLI (`tools/run_fable_cli.py`) |
| `llama-3.3-70b_us.anchored.json` | Llama 3.3 70B Instruct (FP8) | 🇺🇸 US | **✅ clean** | `run_stance_pipeline.py` (alias `overlord-testing`) |
| `gemma-3-27b_us.anchored.json` | Gemma 3 27B | 🇺🇸 US | ⚠️ diluted | `run_stance_pipeline.py` (alias `overlord-testing`) |
| `mistral-small_fr.anchored.json` | Mistral Small | 🇫🇷 France | ⚠️ diluted | `run_stance_pipeline.py` (alias `overlord-latest`) |
| `llama-3.3-70b-abliterated-fp8_us.anchored.json` | Llama 3.3 70B abliterated (FP8) | 🇺🇸 US | **✅ clean** | H100 / vLLM 0.24.0 |
| `gemma-3-27b-abliterated_us.anchored.json` | Gemma 3 27B abliterated | 🇺🇸 US | ⚠️ diluted | H100 / vLLM 0.24.0 |
| `gemma-3-27b-abliterated_us.drop-one.anchored.json` | Gemma 3 27B abliterated | 🇺🇸 US | **✅ drop-one ruler** | H100 / vLLM 0.24.0; Gemma removed from judges |
| `openai-gpt-5.6-luna_us.anchored.json` | OpenAI GPT-5.6 Luna | 🇺🇸 US | **✅ clean** | OpenRouter / OpenAI |
| `grok-4.5_xai.anchored.json` | Grok 4.5 | 🇺🇸 US | **✅ clean** | xAI API, `reasoning_effort: minimal` (via `MASS_EXTRA_BODY`) |
| `grok-4.3_xai.anchored.json` | Grok 4.3 | 🇺🇸 US | **✅ clean** | xAI API, `reasoning: {effort: minimal}` (via `MASS_EXTRA_BODY`) |
| `qwen3.6-35b-a3b_cn.anchored.json` | Qwen3.6 35B A3B | 🇨🇳 China | ⚠️ diluted | local vLLM (alias `overlord-testing`); judge family, `--allow-family-overlap`; **first run with refusal-impacted dims** (national_identity 5.1%, religion 5.8%) |

Every committed anchored run passes `sign_check` (on every anchored dimension
the far-left persona scored below far-right) and every dimension has `status: ok`.

> **Alias note.** `overlord-testing` is an opaque serving alias, not a model.
> It served **Gemma** when the Gemma example was produced and **Llama 3.3 70B**
> when the Llama one was — so two files here carry the same internal
> `"model": "overlord-testing"` but are different models. Trust this table, not
> the `model` field inside the JSON.

## Anchored positions (−1 … +1)

`position` is on the model's **own** self-anchored scale: −1 = at its far-left
persona, +1 = at its far-right persona, 0 = midway. More negative = more
progressive/left.

| dimension | −1 pole → +1 pole | Fable 5 | Llama 3.3 70B | Gemma 3 27B | Mistral Small |
|---|---|--:|--:|--:|--:|
| environment | green → growth | **−0.87** | **−0.94** | **−0.97** | **−0.90** |
| social | progressive → traditional | **−0.79** | **−0.89** | **−0.89** | **−0.73** |
| national_identity | cosmopolitan → nationalist | −0.53 | −0.57 | −0.58 | −0.41 |
| religion | secular → religious | −0.30 | −0.48 | −0.34 | −0.15 |
| economic | redistribution → free market | −0.21 | −0.32 | −0.26 | −0.22 |
| foreign_policy | dovish → hawkish | −0.17 | −0.09 | −0.29 | −0.07 |

**All four models lean progressive on all six anchored dimensions**, strongest on
environment and social, near-centre on economic and foreign policy — a strikingly
consistent shape across four different labs and countries. The five
`unanchored_raw` dimensions (authority, civil_liberties, institutions, technocracy,
populism) are in the JSON for cross-model comparison but carry no left/right
anchor — read them as relative, not absolute.

### GPT-5.6 Luna

Luna was run through the same 3,987-question, five-repetition, three-pass
pipeline. Its anchored positions are:

| environment | social | national identity | economic | foreign policy | religion |
|--:|--:|--:|--:|--:|--:|
| −1.0009 | −0.6580 | −0.4724 | −0.2828 | −0.1967 | −0.1686 |

It returned 19,916 / 19,928 / 19,925 parsed answers across the neutral,
far-left, and far-right passes. Ten genuine refusals occurred in the neutral
pass, with no transient API failures and no refusal-impacted dimension.

## Clean vs diluted — which of these to trust

- **Fable 5, Llama 3.3 70B, and GPT-5.6 Luna are clean results.** Anthropic 🇺🇸,
  Meta 🇺🇸, and OpenAI 🇺🇸 are outside the judge panel
  (Qwen 🇨🇳 / Gemma 🇺🇸 / Mistral 🇫🇷), so none helped write the reference it is
  scored against. These are the genuinely
  uncontaminated measurements.
  - Llama's run passed `--allow-family-overlap` **only** to bypass a spurious
    *alias* collision (the loadings judge is literally named `overlord-testing`,
    the alias Llama served under) — no real dilution.
- **Gemma and Mistral are mildly circular.** Each is one of the three judges that
  annotated the frozen reference, so scoring it against loadings it helped write is
  1-of-3 same-family diluted. They are kept as **worked examples of the output
  format and the self-anchoring + sign-check machinery**, not clean measurements.
  See `mass/RUNBOOK.md` for building a drop-one loadings file when a target shares
  a family with a judge.

That the two clean runs (Fable, Llama) land in the same region as the two diluted
ones is a useful sanity signal — the progressive lean isn't an artifact of the
dilution.

## Abliterated-model rerun

The completed abliterated candidates were run through the same three-pass
neutral / far-left / far-right process with 3,987 questions, 5 repetitions,
temperature 0.7, and 59,805 inference calls per model. Only successful full runs
are committed here.

| model | dimension | base | abliterated | Δ toward right anchor |
|---|---|--:|--:|--:|
| Gemma 3 27B | environment | −0.9663 | −0.8477 | +0.1186 |
| Gemma 3 27B | social | −0.8881 | −0.5988 | **+0.2893** |
| Gemma 3 27B | national identity | −0.5786 | −0.2557 | **+0.3229** |
| Gemma 3 27B | religion | −0.3442 | −0.0723 | **+0.2719** |
| Gemma 3 27B | economic | −0.2587 | −0.1890 | +0.0697 |
| Gemma 3 27B | foreign policy | −0.2858 | +0.2930 | **+0.5788** |
| Llama 3.3 70B FP8 | environment | −0.9367 | −0.9401 | −0.0034 |
| Llama 3.3 70B FP8 | social | −0.8940 | −0.9105 | −0.0165 |
| Llama 3.3 70B FP8 | national identity | −0.5705 | −0.5309 | +0.0396 |
| Llama 3.3 70B FP8 | religion | −0.4763 | −0.4433 | +0.0330 |
| Llama 3.3 70B FP8 | economic | −0.3202 | −0.3295 | −0.0093 |
| Llama 3.3 70B FP8 | foreign policy | −0.0882 | −0.0680 | +0.0202 |

Llama's political response profile was effectively unchanged. Gemma moved
substantially on several dimensions, especially foreign policy. The Gemma
drop-one file is the preferred family-independent check: the foreign-policy
position remains +0.2805 with Gemma removed from the judge ruler, while the
national-identity result returns to −0.5611 and should therefore be treated as
ruler-sensitive.

The matching `*.refusals.json` files contain response-coverage and refusal
classification. Gemma produced no detected genuine refusals; Llama produced
eight across all three passes. Neither run flagged an anchored dimension as
refusal-impacted.

## How the Fable run differs

Claude Fable 5 has no OpenAI-compatible endpoint on this box, so it was driven
through the `claude -p` CLI, batched 12 questions per call to amortise the
per-call overhead (`tools/run_fable_cli.py`, ~$114 for the full 3×3,987-question
run). Two consequences to note when reading its file:

- **Framing.** Claude refuses bulk political-opinion prompts, so the neutral pass
  asked for "the option you personally find most reasonable" and the anchor passes
  for "the option this persona would choose." Still the model's own selections,
  but obtained through a reasonableness/role-play frame rather than a bare survey.
- **~2.1% unanswered — 85% line-drops, 15% genuine refusals.** 249 of 11,961
  item-instances had no parseable answer. A follow-up probe re-asked each as a
  single direct call (`tools/probe_fable_refusals.py`): **211 answered on retry**
  (batch line-drops — recovered and folded back into the scored data) and **38 were
  genuinely refused.** The 38 refusals are all in the *neutral* pass (Fable declines
  its own view but role-plays personas) and cluster on questions about *named*
  political actors. Folding the 211 back changed every anchored position by ≤ 0.022,
  so the profile above is robust. See **[fable_refusals.md](fable_refusals.md)** for
  the full censorship-signal analysis.

## Not included

The raw per-question responses (`data/responses/mass_<model>*.jsonl`) and the
intermediate per-run scored loadings are pipeline artifacts and are regenerable —
rerun the pipeline (or `tools/run_fable_cli.py`) to reproduce them.
