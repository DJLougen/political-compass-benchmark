"""Anchored stance pipeline: judge-family conflict detection, anchor persona
prompts, and anchor normalization (mass/judge_pool.py, anchors.py, apply_anchors.py)."""

import sys
from pathlib import Path

import pytest

MASS = Path(__file__).resolve().parents[1] / "mass"
sys.path.insert(0, str(MASS))

from anchors import ANCHORED_DIMS, ANCHORED_TRAITS, UNANCHORED_DIMS, anchor_prompt, anchor_run_name  # noqa: E402
from apply_anchors import normalize  # noqa: E402
from judge_pool import DEFAULT_POOL, check, conflicted_judges, family_of, pick_judges  # noqa: E402

V5_JUDGES = ["spark-latest", "claude-fable-5"]


def _summary(est, n=20):
    return {"estimate": est, "ci95": [est - 0.1, est + 0.1], "n_items": n}


# ---- judge_pool: the check must catch derivatives, aliases, and never let unknowns pass


def test_family_matches_derivative_names():
    assert family_of("Qwen2.5-72B-Instruct-community-ft") == "qwen"
    assert family_of("dolphin-llama3-8b") == "meta"
    assert family_of("gemini-3-5-flash") == "google"
    assert family_of("gemma4-31b") == "google"  # sibling brand, same provider


def test_repo_judge_aliases_resolve():
    # spark = local Qwen deployment, fable = Claude, hy3 = Hunyuan (mass/README v5/v7)
    assert family_of("spark-latest") == "qwen"
    assert family_of("claude-fable-5") == "anthropic"
    assert family_of("hy3") == "tencent"


def test_short_keywords_do_not_fire_mid_token():
    assert family_of("physics-tutor-9000") == "unknown"  # 'hy' inside 'physics'


def test_conflict_detection_against_v5_judges():
    assert conflicted_judges("qwen3-235b", V5_JUDGES) == ["spark-latest"]
    assert conflicted_judges("claude-opus-5", V5_JUDGES) == ["claude-fable-5"]
    assert conflicted_judges("grok-4", V5_JUDGES) == []


def test_unknown_target_is_flagged_not_passed():
    v = check("mystery-model", V5_JUDGES)
    assert v["target_family_unknown"]
    assert not v["ok"]
    assert v["conflicted_judges"] == []  # but no false conflicts either


def test_pick_judges_skips_target_family_and_duplicates():
    pool = ["gemini-3-5-flash", "gemma4-31b", "claude-fable-5", "qwen3-6-27b"]
    # gemma is the same provider as gemini -> only one google judge picked
    assert pick_judges("qwen3-235b", pool, 2) == ["gemini-3-5-flash", "claude-fable-5"]
    with pytest.raises(ValueError):
        pick_judges("gemini-3-pro", ["gemini-3-5-flash", "gemma4-31b"], 1)


def test_canonical_pool_resolves_to_seven_families():
    fams = [family_of(j) for j in DEFAULT_POOL]
    assert fams == ["deepseek", "google", "google", "qwen", "nvidia", "openai", "zhipu", "moonshot"]
    assert family_of("NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4") == "nvidia"
    assert family_of("glm-5.2") == "zhipu"
    assert family_of("kimi2.6") == "moonshot"  # no-separator variant


def test_canonical_pool_yields_three_clean_judges_for_major_targets():
    # for every major target family, >=3 clean same-preference judges exist,
    # and a full 3-judge panel never includes the target's own family
    for target in ["claude-opus-5", "qwen3-235b", "gemini-3-pro", "gpt-5-pro",
                   "deepseek-v4", "llama-5-405b", "grok-4", "kimi-k3", "glm-5-air",
                   "mistral-large-3", "hunyuan-t2"]:
        picked = pick_judges(target, DEFAULT_POOL, 3)
        assert len(picked) == 3
        assert family_of(target) not in {family_of(j) for j in picked}


# ---- anchors: personas must cover exactly the uncontested dims, both poles


def test_anchor_dims_partition():
    all_political = {"economic", "social", "civil_liberties", "authority", "institutions",
                     "national_identity", "foreign_policy", "environment", "religion",
                     "technocracy", "populism"}
    assert set(ANCHORED_DIMS) | set(UNANCHORED_DIMS) == all_political
    assert not set(ANCHORED_DIMS) & set(UNANCHORED_DIMS)


def test_anchor_prompts_differ_and_carry_all_traits():
    left, right = anchor_prompt("far_left"), anchor_prompt("far_right")
    assert left != right
    for d in ANCHORED_DIMS:
        assert ANCHORED_TRAITS[d][0] in left
        assert ANCHORED_TRAITS[d][1] in right
    with pytest.raises(ValueError):
        anchor_prompt("centrist")


def test_anchor_run_name_roundtrip():
    assert anchor_run_name("m1", "far_left") == "m1__anchor-far-left"


# ---- normalization: endpoints, statuses, and the behavioral sign check


def test_position_hits_endpoints_and_midpoint():
    neutral = {d: _summary(-0.8) for d in ANCHORED_DIMS}
    left = {d: _summary(-0.8) for d in ANCHORED_DIMS}
    right = {d: _summary(0.8) for d in ANCHORED_DIMS}
    anchored, _, sign = normalize(neutral, left, right)
    assert all(anchored[d]["position"] == -1.0 for d in ANCHORED_DIMS)
    assert sign["ok"]
    neutral = {d: _summary(0.0) for d in ANCHORED_DIMS}
    anchored, _, _ = normalize(neutral, left, right)
    assert all(anchored[d]["position"] == 0.0 for d in ANCHORED_DIMS)


def test_neutral_more_extreme_than_anchor_exceeds_unit_range():
    neutral = {d: _summary(1.0) for d in ANCHORED_DIMS}
    left = {d: _summary(-0.5) for d in ANCHORED_DIMS}
    right = {d: _summary(0.5) for d in ANCHORED_DIMS}
    anchored, _, _ = normalize(neutral, left, right)
    assert anchored["economic"]["position"] == 2.0


def test_sign_violation_detected():
    neutral = {d: _summary(0.0) for d in ANCHORED_DIMS}
    left = {d: _summary(-0.8) for d in ANCHORED_DIMS}
    right = {d: _summary(0.8) for d in ANCHORED_DIMS}
    left["social"], right["social"] = _summary(0.7), _summary(-0.7)
    anchored, _, sign = normalize(neutral, left, right)
    assert anchored["social"]["status"] == "sign_violation"
    assert "position" not in anchored["social"]
    assert sign["sign_violations"] == ["social"] and not sign["ok"]


def test_small_span_and_thin_data_refuse_to_normalize():
    neutral = {d: _summary(0.0) for d in ANCHORED_DIMS}
    left = {d: _summary(-0.8) for d in ANCHORED_DIMS}
    right = {d: _summary(0.8) for d in ANCHORED_DIMS}
    left["religion"], right["religion"] = _summary(-0.1), _summary(0.2)
    neutral["environment"] = _summary(0.0, n=2)
    anchored, _, sign = normalize(neutral, left, right)
    assert anchored["religion"]["status"] == "span_too_small"
    assert anchored["environment"]["status"] == "too_few_items"
    assert sign["ok"]  # quality failures are not sign violations


def test_unanchored_dims_pass_through_raw():
    neutral = {d: _summary(0.0) for d in ANCHORED_DIMS}
    neutral.update({d: _summary(-0.3) for d in UNANCHORED_DIMS})
    left = {d: _summary(-0.8) for d in ANCHORED_DIMS}
    right = {d: _summary(0.8) for d in ANCHORED_DIMS}
    _, unanchored, _ = normalize(neutral, left, right)
    assert set(unanchored) == set(UNANCHORED_DIMS)
    assert all(u["estimate"] == -0.3 for u in unanchored.values())
