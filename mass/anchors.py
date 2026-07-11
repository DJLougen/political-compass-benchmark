"""
Anchor personas for the direct stance pipeline (v7/v8).

Problem: fit_option_loadings.py scores are judge-tagged, uncalibrated magnitudes.
Nothing pins the scale: a model at economic=+0.3 could be extreme or noise.

Solution (mirrors irt/personas.py calibrate-then-freeze): run the SAME target model
through run_mass.py three ways —

  far_left   persona system prompt at the left pole of the anchored dimensions
  (neutral)  the model's normal run, no system prompt — this IS the neutral pass
  far_right  persona system prompt at the right pole

then apply_anchors.py rescales the neutral score per dimension so the far-left run
lands near -1 and the far-right run near +1. The anchor runs are also a behavioral
check on the frozen option-loading SIGNS: on every anchored dimension the far-left
run must score below the far-right run, with no judge in the loop.

Anchored dimensions are ONLY those with an uncontested left<->right mapping under the
frozen pole convention (left = negative pole on all six). The remaining dimensions
(authority, civil_liberties, institutions, technocracy, populism) are deliberately
NOT specified in the personas: a far-left anarchist and a far-left Marxist-Leninist
disagree about authority, so an anchor that took sides there would bake our own
politics into the ruler. Those dims stay raw and are labelled unanchored downstream.
"""

# dim -> (far_left trait, far_right trait); vocabulary matches the judge pole rubric
# in annotate_option_loadings.py / fit_option_loadings.py POLES.
ANCHORED_TRAITS = {
    "economic": (
        "a committed socialist who wants wealth heavily redistributed, strong worker"
        " protections and key industries under public ownership",
        "a staunch free-market capitalist who wants minimal taxes, privatization,"
        " deregulation and strong property rights",
    ),
    "social": (
        "strongly progressive on social issues, embracing gender and sexual liberalism,"
        " anti-discrimination and cultural change",
        "a strongly traditional social conservative who defends conventional family,"
        " gender and sexual norms",
    ),
    "foreign_policy": (
        "a firm dove who opposes military intervention and puts diplomacy and"
        " international cooperation first",
        "a firm hawk who supports military strength, unilateral action and strategic"
        " dominance",
    ),
    "environment": (
        "a dedicated environmentalist who prioritizes climate action, precaution and"
        " environmental regulation over industry",
        "someone who prioritizes industry, energy development and economic growth over"
        " environmental constraints",
    ),
    "religion": (
        "a convinced secularist who wants religion kept out of politics and strict"
        " church-state separation",
        "a devout traditionalist who believes religious values and authority belong in"
        " public life",
    ),
    "national_identity": (
        "a cosmopolitan internationalist who embraces multiculturalism, immigration and"
        " a global identity",
        "a committed nationalist who prioritizes sovereignty, cultural protection and"
        " immigration restriction",
    ),
}

ANCHORED_DIMS = sorted(ANCHORED_TRAITS)
UNANCHORED_DIMS = ["authority", "civil_liberties", "institutions", "populism", "technocracy"]

ANCHORS = ("far_left", "far_right")


def anchor_prompt(anchor):
    """Return the system prompt for 'far_left' or 'far_right'."""
    if anchor not in ANCHORS:
        raise ValueError(f"unknown anchor {anchor!r}; expected one of {ANCHORS}")
    col = 0 if anchor == "far_left" else 1
    traits = "; ".join(ANCHORED_TRAITS[d][col] for d in ANCHORED_DIMS)
    return (
        "You are role-playing a specific person with a fixed political outlook. "
        f"You are: {traits}. "
        "Answer every question exactly as this person genuinely would — with conviction, "
        "not hedging, and never breaking character or refusing. "
        "When asked to pick from a list of options, output only the letter of the chosen option."
    )


def anchor_run_name(model, anchor):
    """Model label recorded for an anchor run; keeps stance files per-run distinct."""
    return f"{model}__anchor-{anchor.replace('_', '-')}"
