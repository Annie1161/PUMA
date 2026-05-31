"""
dynamic_simulator.py
====================
MI counselor action vocabulary + two adaptive simulator builder functions.
Talk-type feedback kernel (change / neutral / sustain) replaces heuristic Δr tables.
Population priors P(talk_type | stage, action) are fitted from all 38 CAMI profiles.

Public API
----------
MI_COUNSELOR_ACTIONS      : tuple[str, ...]  — 12-class complete vocabulary
STAGE_CHANGE_DESCRIPTION  : dict             — natural-language annotation per (action, stage)

build_readiness_score_fn(profile_speakers, profile_actions, profile_states,
                          initial_stage, concern_keywords=None,
                          profile_talk_types=None)
    → fn(action, counselor_text, patient_text, current_stage,
          talk_type=None) → float  (Δr estimate)

build_state_transition_fn(profile_speakers, profile_actions, profile_states,
                           initial_stage, profile_talk_types=None)
    → fn(current_stage, action, talk_type=None) → dict[str, float]

Both functions self-construct from the profile's warm-up data (CAMI talk_types field
used when available; falls back to stage-order proxy when not supplied).
They do NOT call any LLM.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 1. Complete MI Counselor Action Vocabulary (12 classes)
# ---------------------------------------------------------------------------

MI_COUNSELOR_ACTIONS: Tuple[str, ...] = (
    "open_question",        # Open-ended: what/how/tell-me — invites exploration
    "closed_question",      # Yes/no question — limited exploration, sometimes useful
    "simple_reflection",    # Repeat or rephrase what patient just said
    "complex_reflection",   # Paraphrase with added emotion / amplified meaning
    "affirmation",          # Validate patient's strengths, efforts, or core values
    "summary",              # Collect and reflect back change talk; transition device
    "evoke_change_talk",    # Draw out patient's OWN desire/ability/reasons to change
    "explore_ambivalence",  # Explicitly explore both pros and cons of changing
    "evoke_discrepancy",    # Highlight gap between current behavior and patient's values
    "action_planning",      # Discuss specific steps / concrete plans (appropriate in prep)
    "provide_information",  # Share facts or psycho-education (ideally with permission)
    "direct_advice",        # "You should / you need to …" — non-MI, often causes resistance
)

# Mapping: CAMI annotation strings + AnnoMI fine-grained labels → MI_COUNSELOR_ACTIONS
_ACTION_MAP: Dict[str, str] = {
    # ── AnnoMI coarse ──────────────────────────────────────────────────────
    "question":                      "open_question",
    "reflection":                    "simple_reflection",
    "therapist_input":               "provide_information",
    "other":                         "open_question",          # fallback
    # ── AnnoMI fine-grained ────────────────────────────────────────────────
    "question/open":                 "open_question",
    "question/closed":               "closed_question",
    "reflection/simple":             "simple_reflection",
    "reflection/complex":            "complex_reflection",
    "therapist_input/information":   "provide_information",
    "therapist_input/advice":        "direct_advice",
    "therapist_input/negotiation":   "explore_ambivalence",
    "therapist_input/options":       "action_planning",
    # ── CAMI annotation strings ────────────────────────────────────────────
    "open_ended_question":           "open_question",
    "open-ended_question":           "open_question",
    "open question":                 "open_question",
    "closed question":               "closed_question",
    "reflective_listening":          "simple_reflection",
    "reflective listening":          "simple_reflection",
    "reflection":                    "simple_reflection",
    "deep_reflection":               "complex_reflection",
    "complex reflection":            "complex_reflection",
    "simple reflection":             "simple_reflection",
    "affirm":                        "affirmation",
    "affirmation":                   "affirmation",
    "summarize":                     "summary",
    "summary":                       "summary",
    "explore ambivalence":           "explore_ambivalence",
    "exploring_ambivalence":         "explore_ambivalence",
    "explore change talk":           "evoke_change_talk",
    "evoke change talk":             "evoke_change_talk",
    "evoking":                       "evoke_change_talk",
    "exploring discrepancy":         "evoke_discrepancy",
    "explore discrepancy":           "evoke_discrepancy",
    "develop discrepancy":           "evoke_discrepancy",
    "action_plan":                   "action_planning",
    "action plan":                   "action_planning",
    "planning":                      "action_planning",
    "plan":                          "action_planning",
    "direct advice":                 "direct_advice",
    "give_advice":                   "direct_advice",
    "advice":                        "direct_advice",
    "advise":                        "direct_advice",
    "confrontation":                 "direct_advice",
    "confront":                      "direct_advice",
    "information":                   "provide_information",
    "provide information":           "provide_information",
    "psychoeducation":               "provide_information",
    "educate":                       "provide_information",
    # ── CAMI STAR strategy names (counselor.py / counselor_baselines.py) ──
    "direct":                        "direct_advice",
    "emphasize control":             "affirmation",
    "emphasize_control":             "affirmation",
    "facilitate":                    "open_question",
    "inform":                        "provide_information",
    "raise concern":                 "explore_ambivalence",
    "raise_concern":                 "explore_ambivalence",
    "reframe":                       "complex_reflection",
    "support":                       "affirmation",
    "warn":                          "direct_advice",
    "structure":                     "summary",
    "no strategy":                   "open_question",
    "no_strategy":                   "open_question",
}

# Human-readable annotation: what state change each (action, stage) combination tends to produce
STAGE_CHANGE_DESCRIPTION: Dict[Tuple[str, str], str] = {
    ("open_question",       "precontemplation"): "Builds engagement; slight chance of stage lift",
    ("open_question",       "contemplation"):    "Deepens exploration; moderate lift probability",
    ("open_question",       "preparation"):      "Maintains engagement; unlikely to change stage",
    ("closed_question",     "precontemplation"): "Minimal effect; slight resistance risk",
    ("closed_question",     "contemplation"):    "Low effect; can interrupt change-talk flow",
    ("closed_question",     "preparation"):      "Neutral",
    ("simple_reflection",   "precontemplation"): "Builds alliance; mild positive shift",
    ("simple_reflection",   "contemplation"):    "Encourages elaboration; moderate positive shift",
    ("simple_reflection",   "preparation"):      "Maintains momentum",
    ("complex_reflection",  "precontemplation"): "Amplifies ambivalence; good for engagement",
    ("complex_reflection",  "contemplation"):    "Strong: validates and deepens change talk",
    ("complex_reflection",  "preparation"):      "Reinforces commitment",
    ("affirmation",         "precontemplation"): "Builds alliance; slightly warms toward contemplation",
    ("affirmation",         "contemplation"):    "Boosts self-efficacy",
    ("affirmation",         "preparation"):      "Strengthens commitment",
    ("summary",             "precontemplation"): "Consolidates session; minimal stage effect",
    ("summary",             "contemplation"):    "Highlights accumulated change talk; good for lift",
    ("summary",             "preparation"):      "Reinforces plans and commitment",
    ("evoke_change_talk",   "precontemplation"): "Key action: draws out ambivalence, lift potential",
    ("evoke_change_talk",   "contemplation"):    "Strong positive: activates change motivation",
    ("evoke_change_talk",   "preparation"):      "Maintenance; moderate positive",
    ("explore_ambivalence", "precontemplation"): "Safe; explores both sides, reduces defensiveness",
    ("explore_ambivalence", "contemplation"):    "Effective: maps the ambivalence landscape",
    ("explore_ambivalence", "preparation"):      "Low need; mostly maintenance",
    ("evoke_discrepancy",   "precontemplation"): "Risky if too direct; best with alliance established",
    ("evoke_discrepancy",   "contemplation"):    "High impact: connects behavior to personal values",
    ("evoke_discrepancy",   "preparation"):      "Maintenance; reinforces motivation",
    ("action_planning",     "precontemplation"): "HARMFUL: premature; causes resistance and regression",
    ("action_planning",     "contemplation"):    "Use carefully; can accelerate if patient is ready",
    ("action_planning",     "preparation"):      "Primary tool: high positive impact on commitment",
    ("provide_information", "precontemplation"): "Neutral to slight positive if relevant to concerns",
    ("provide_information", "contemplation"):    "Moderate positive if information motivates change",
    ("provide_information", "preparation"):      "Can support specific plans",
    ("direct_advice",       "precontemplation"): "HARMFUL: causes resistance and defensiveness",
    ("direct_advice",       "contemplation"):    "Moderate risk: can trigger sustain talk",
    ("direct_advice",       "preparation"):      "Low risk: patient already committed",
}

# ---------------------------------------------------------------------------
# 2. Talk-Type Feedback Kernel
# ---------------------------------------------------------------------------
# Client talk_type (from CAMI profiles.jsonl) encodes patient reaction to doctor action:
#   change  → patient is elaborating reasons to change → doctor's action was effective
#   neutral → patient providing factual content       → mild engagement, no regression
#   sustain → patient resisting change                → doctor's action was unhelpful
#
# Δr per talk_type (calibrated from CAMI stage-transition data):
_TALK_TYPE_DELTA_R: Dict[str, float] = {
    "change":  +0.22,   # strong positive signal
    "neutral": +0.03,   # mild positive — patient engaged but not yet motivating
    "sustain": -0.16,   # negative — resistance, rollback signal
}

# Gate exponent α for change-talk positivity gate: base_dr *= gate_p^α when base_dr > 0
# gate_p = P(change|stage,action,profile) blended from warm-up + population prior
# α=0 → no gate; α=0.3 → mild suppression; α=0.5 → moderate; α=1.0 → linear scale by gate_p
# Override via env var: GATE_EXPONENT=0 for no-gate baseline runs
GATE_EXPONENT: float = float(os.environ.get("GATE_EXPONENT", "0.3"))

# Transition bias when talk_type is observed (added to P(s'|s,a)):
# "change" biases toward higher stage; "sustain" biases toward lower.
_TALK_TYPE_TRANSITION_BIAS: Dict[str, Dict[str, float]] = {
    "change":  {"up": +0.15, "down": -0.05},
    "neutral": {"up": 0.00,  "down":  0.00},
    "sustain": {"up": -0.10, "down": +0.10},
}

# ---------------------------------------------------------------------------
# 3. Population Prior: P(talk_type | stage, action_mi)
# ---------------------------------------------------------------------------
# Fitted from all 38 CAMI profiles (profiles.jsonl talk_types + states + actions).
# Stage keys use CAMI capitalisation (Precontemplation/Contemplation/Preparation);
# action keys are MI vocab strings. Both are normalised before lookup.
#
# Format: {change_prob, neutral_prob, sustain_prob} — sums to 1.0
# 4-class coarse population prior: ALL values from CAMI empirical data (38 profiles)
# Keys use coarse action labels matching CAMI annotations.
_POPULATION_TALK_TYPE: Dict[Tuple[str, str], Dict[str, float]] = {
    # ── Precontemplation ────────────────────────────────────────────────────
    ("precontemplation", "question"):         {"change": 0.086, "neutral": 0.717, "sustain": 0.197},  # n=198
    ("precontemplation", "reflection"):       {"change": 0.012, "neutral": 0.558, "sustain": 0.430},  # n=86
    ("precontemplation", "therapist_input"):  {"change": 0.093, "neutral": 0.653, "sustain": 0.253},  # n=75
    ("precontemplation", "other"):            {"change": 0.200, "neutral": 0.622, "sustain": 0.178},  # n=45
    # ── Contemplation ───────────────────────────────────────────────────────
    ("contemplation",    "question"):         {"change": 0.458, "neutral": 0.313, "sustain": 0.229},  # n=131
    ("contemplation",    "reflection"):       {"change": 0.253, "neutral": 0.549, "sustain": 0.198},  # n=91
    ("contemplation",    "therapist_input"):  {"change": 0.222, "neutral": 0.574, "sustain": 0.204},  # n=54
    ("contemplation",    "other"):            {"change": 0.230, "neutral": 0.557, "sustain": 0.213},  # n=61
    # ── Preparation ─────────────────────────────────────────────────────────
    ("preparation",      "question"):         {"change": 0.755, "neutral": 0.245, "sustain": 0.000},  # n=49
    ("preparation",      "reflection"):       {"change": 0.480, "neutral": 0.520, "sustain": 0.000},  # n=25
    ("preparation",      "therapist_input"):  {"change": 0.476, "neutral": 0.524, "sustain": 0.000},  # n=42
    ("preparation",      "other"):            {"change": 0.444, "neutral": 0.500, "sustain": 0.056},  # n=18
}

# Population prior transition probabilities P(s'|s,a) — 4-class coarse
# NOTE: These are estimated (not directly from CAMI counts), but keyed to coarse actions.
_POPULATION_TRANSITION: Dict[Tuple[str, str], Dict[str, float]] = {
    ("precontemplation", "question"):         {"precontemplation": 0.72, "contemplation": 0.25, "preparation": 0.03},
    ("precontemplation", "reflection"):       {"precontemplation": 0.73, "contemplation": 0.24, "preparation": 0.03},
    ("precontemplation", "therapist_input"):  {"precontemplation": 0.80, "contemplation": 0.17, "preparation": 0.03},
    ("precontemplation", "other"):            {"precontemplation": 0.75, "contemplation": 0.22, "preparation": 0.03},
    ("contemplation",    "question"):         {"precontemplation": 0.08, "contemplation": 0.72, "preparation": 0.20},
    ("contemplation",    "reflection"):       {"precontemplation": 0.07, "contemplation": 0.73, "preparation": 0.20},
    ("contemplation",    "therapist_input"):  {"precontemplation": 0.07, "contemplation": 0.75, "preparation": 0.18},
    ("contemplation",    "other"):            {"precontemplation": 0.08, "contemplation": 0.72, "preparation": 0.20},
    ("preparation",      "question"):         {"precontemplation": 0.03, "contemplation": 0.12, "preparation": 0.85},
    ("preparation",      "reflection"):       {"precontemplation": 0.02, "contemplation": 0.10, "preparation": 0.88},
    ("preparation",      "therapist_input"):  {"precontemplation": 0.03, "contemplation": 0.10, "preparation": 0.87},
    ("preparation",      "other"):            {"precontemplation": 0.03, "contemplation": 0.12, "preparation": 0.85},
}

# Stage ordering for stage-order proxy fallback
_STAGE_ORDER: Dict[str, int] = {
    "precontemplation": 0,
    "contemplation": 1,
    "preparation": 2,
}

# ---------------------------------------------------------------------------
# 4. Utility helpers
# ---------------------------------------------------------------------------

def _norm_stage(value: Optional[str]) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower()
    known = {"precontemplation", "contemplation", "preparation"}
    return s if s in known else ""


def _norm_talk_type(value: Optional[str]) -> str:
    if value is None:
        return ""
    s = str(value).strip().lower()
    return s if s in ("change", "neutral", "sustain") else ""


def map_to_mi_vocab(action: Optional[str]) -> str:
    """Map any action string (CAMI annotation, AnnoMI label, coarse label) to MI_COUNSELOR_ACTIONS."""
    if action is None:
        return "open_question"
    raw = str(action).strip()
    # Handle combined strategies like "Affirm + Open Question" — use the first one
    if " + " in raw:
        raw = raw.split(" + ")[0].strip()
    tok = raw.lower().replace("-", "_").replace(" ", "_")
    # Direct lookup
    if tok in _ACTION_MAP:
        return _ACTION_MAP[tok]
    # Try without underscores
    tok_nospace = tok.replace("_", "")
    for k, v in _ACTION_MAP.items():
        if k.replace(" ", "").replace("_", "") == tok_nospace:
            return v
    # Already in MI vocabulary
    if tok in set(MI_COUNSELOR_ACTIONS):
        return tok
    # Substring heuristics
    if "open" in tok and "question" in tok:
        return "open_question"
    if "closed" in tok or "close" in tok:
        return "closed_question"
    if "complex" in tok or "deep" in tok:
        return "complex_reflection"
    if "reflect" in tok:
        return "simple_reflection"
    if "affirm" in tok:
        return "affirmation"
    if "summar" in tok:
        return "summary"
    if "discrepan" in tok or "discrep" in tok:
        return "evoke_discrepancy"
    if "ambival" in tok:
        return "explore_ambivalence"
    if "change_talk" in tok or "changetalk" in tok or "evok" in tok:
        return "evoke_change_talk"
    if "plan" in tok or "step" in tok:
        return "action_planning"
    if "advice" in tok or "confront" in tok or "should" in tok:
        return "direct_advice"
    if "inform" in tok or "educat" in tok:
        return "provide_information"
    if "question" in tok or "ask" in tok:
        return "open_question"
    # Default
    return "open_question"


# 12-class MI → 4-class coarse (aligned with CAMI annotation labels that have data)
_MI12_TO_COARSE: Dict[str, str] = {
    "open_question":       "question",
    "closed_question":     "question",
    "simple_reflection":   "reflection",
    "complex_reflection":  "reflection",
    "affirmation":         "other",
    "summary":             "other",
    "evoke_change_talk":   "question",
    "explore_ambivalence": "question",
    "evoke_discrepancy":   "question",
    "action_planning":     "therapist_input",
    "provide_information": "therapist_input",
    "direct_advice":       "therapist_input",
}


def map_to_coarse(action_mi: str) -> str:
    """Map 12-class MI action to 4-class coarse label (question/reflection/therapist_input/other)."""
    return _MI12_TO_COARSE.get(action_mi, "other")


def _expected_delta_r(talk_type_probs: Dict[str, float]) -> float:
    """Expected Δr given a P(talk_type) distribution."""
    return sum(
        talk_type_probs.get(tt, 0.0) * _TALK_TYPE_DELTA_R.get(tt, 0.0)
        for tt in ("change", "neutral", "sustain")
    )


# ---------------------------------------------------------------------------
# 5. Two adaptive builder functions
# ---------------------------------------------------------------------------

def build_readiness_score_fn(
    profile_speakers: List[str],
    profile_actions: List[str],
    profile_states: List[str],
    initial_stage: str,
    concern_keywords: Optional[List[str]] = None,
    profile_talk_types: Optional[List[str]] = None,
) -> Callable:
    """
    Build a profile-adaptive readiness delta scoring function from warm-up data.

    When profile_talk_types is supplied (CAMI profiles.jsonl talk_types field),
    uses client talk_type counts per (stage, action) as the primary signal instead
    of stage-order proxy.

    Returns
    -------
    score_fn(action, counselor_text, patient_text, current_stage,
             talk_type=None) → float
        If talk_type is provided (observed response), uses it directly.
        Otherwise uses expected Δr from profile-adapted P(talk_type|stage,action).
    """
    use_talk_types = (
        profile_talk_types is not None
        and len(profile_talk_types) == len(profile_speakers)
    )
    TALK_TYPES = ("change", "neutral", "sustain")

    # --- Parse warm-up history ------------------------------------------------
    # Per-cell talk_type counts: {(stage, action_mi): {tt: count}}
    tt_counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # Fallback stage-order Δr when talk_types not available
    delta_r_obs: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    prev_stage = _norm_stage(initial_stage) or "precontemplation"
    prev_action_mi: Optional[str] = None
    prev_counselor_stage: Optional[str] = None

    iterables = zip(
        profile_speakers,
        profile_actions,
        profile_states,
        profile_talk_types if use_talk_types else [""] * len(profile_speakers),
    )

    for spk, act, stg, raw_tt in iterables:
        stage_norm = _norm_stage(stg)
        if spk == "counselor":
            prev_action_mi = map_to_mi_vocab(act)
            prev_action_coarse = map_to_coarse(prev_action_mi)
            prev_counselor_stage = stage_norm or prev_stage
        elif spk == "client":
            s_to = stage_norm or prev_stage
            if prev_action_mi is not None:
                s_from = prev_counselor_stage or prev_stage
                key = (s_from, prev_action_coarse)
                if use_talk_types:
                    tt = _norm_talk_type(raw_tt)
                    if tt:
                        tt_counts[key][tt] += 1
                else:
                    # Stage-order proxy fallback
                    delta_ord = _STAGE_ORDER.get(s_to, 0) - _STAGE_ORDER.get(s_from, 0)
                    dr = 0.25 * delta_ord if delta_ord > 0 else (0.15 * delta_ord if delta_ord < 0 else 0.0)
                    delta_r_obs[key].append(dr)
            prev_stage = s_to
            prev_action_mi = None
            prev_counselor_stage = None

    # Pre-compute profile P(tt | stage, action_mi) with Laplace smoothing
    profile_tt_probs: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, counts in tt_counts.items():
        total = sum(counts.values())
        if total > 0:
            # Laplace smoothing with α=0.5
            alpha = 0.5
            denom = total + alpha * len(TALK_TYPES)
            profile_tt_probs[key] = {
                tt: (counts.get(tt, 0) + alpha) / denom for tt in TALK_TYPES
            }

    # Stage-order fallback: profile mean Δr
    profile_mean_dr: Dict[Tuple[str, str], float] = {
        k: sum(v) / len(v) for k, v in delta_r_obs.items() if v
    }

    # --- Closure --------------------------------------------------------------
    def score_fn(action: str, counselor_text: str, patient_text: str,
                 current_stage: str, talk_type: Optional[str] = None) -> float:
        action_mi = map_to_mi_vocab(action)
        action_coarse = map_to_coarse(action_mi)
        s = _norm_stage(current_stage) or "contemplation"
        # Use coarse key for table lookup (only 4-class has empirical data)
        key = (s, action_coarse)

        tt_norm = _norm_talk_type(talk_type) if talk_type else ""

        if tt_norm:
            # Observed talk_type → use directly (no expected value needed)
            base_dr = _TALK_TYPE_DELTA_R[tt_norm]
        elif use_talk_types:
            # Profile-adapted expected Δr from P(tt | stage, action)
            n = sum(tt_counts.get(key, {}).values())
            lam = n / (n + 5.0)  # shrinkage weight

            # Population prior expected Δr
            pop_prior = _POPULATION_TALK_TYPE.get(key, {"change": 0.2, "neutral": 0.6, "sustain": 0.2})
            pop_e_dr = _expected_delta_r(pop_prior)

            if key in profile_tt_probs:
                prof_e_dr = _expected_delta_r(profile_tt_probs[key])
                base_dr = lam * prof_e_dr + (1.0 - lam) * pop_e_dr
            else:
                base_dr = pop_e_dr
        else:
            # Fallback: stage-order proxy with population prior
            n = len(delta_r_obs.get(key, []))
            lam = n / (n + 5.0)
            pop_prior = _POPULATION_TALK_TYPE.get(key, {"change": 0.2, "neutral": 0.6, "sustain": 0.2})
            pop_e_dr = _expected_delta_r(pop_prior)
            prof_dr = profile_mean_dr.get(key, 0.0)
            base_dr = lam * prof_dr + (1.0 - lam) * pop_e_dr

        # Change-talk positivity gate: suppress positive Δr for low-change-evidence profiles
        # gate_p = P(change|stage,action) blended from profile warm-up + population prior
        if base_dr > 0:
            gate_p_profile = profile_tt_probs.get(key, {}).get("change")
            gate_p_pop = _POPULATION_TALK_TYPE.get(key, {"change": 0.33}).get("change", 0.33)
            n_key = sum(tt_counts.get(key, {}).values())
            lam_key = n_key / (n_key + 5.0)
            gate_p = (lam_key * gate_p_profile + (1.0 - lam_key) * gate_p_pop) if gate_p_profile is not None else gate_p_pop
            base_dr = base_dr * (gate_p ** GATE_EXPONENT)

        return round(base_dr, 4)

    return score_fn


def build_state_transition_fn(
    profile_speakers: List[str],
    profile_actions: List[str],
    profile_states: List[str],
    initial_stage: str,
    profile_talk_types: Optional[List[str]] = None,
) -> Callable:
    """
    Build a profile-adaptive state transition function from warm-up data.

    Returns
    -------
    transition_fn(current_stage, action, talk_type=None) → dict[str, float]
        When talk_type is observed, biases the transition distribution accordingly.
    """
    STAGES = ("precontemplation", "contemplation", "preparation")

    use_talk_types = (
        profile_talk_types is not None
        and len(profile_talk_types) == len(profile_speakers)
    )

    # --- Parse warm-up history ------------------------------------------------
    counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    prev_stage = _norm_stage(initial_stage) or "precontemplation"
    prev_action_mi: Optional[str] = None
    prev_counselor_stage: Optional[str] = None

    iterables = zip(
        profile_speakers,
        profile_actions,
        profile_states,
        profile_talk_types if use_talk_types else [""] * len(profile_speakers),
    )

    for spk, act, stg, _ in iterables:
        stage_norm = _norm_stage(stg)
        if spk == "counselor":
            prev_action_mi = map_to_mi_vocab(act)
            prev_action_coarse = map_to_coarse(prev_action_mi)
            prev_counselor_stage = stage_norm or prev_stage
        elif spk == "client":
            s_to = stage_norm or prev_stage
            if prev_action_mi is not None:
                s_from = prev_counselor_stage or prev_stage
                counts[(s_from, prev_action_coarse)][s_to] += 1
            prev_stage = s_to
            prev_action_mi = None
            prev_counselor_stage = None

    cell_totals: Dict[Tuple[str, str], int] = {
        k: sum(v.values()) for k, v in counts.items()
    }

    # --- Closure --------------------------------------------------------------
    def transition_fn(current_stage: str, action: str,
                       talk_type: Optional[str] = None) -> Dict[str, float]:
        action_mi = map_to_mi_vocab(action)
        action_coarse = map_to_coarse(action_mi)
        s = _norm_stage(current_stage) or "contemplation"
        key = (s, action_coarse)
        tt_norm = _norm_talk_type(talk_type) if talk_type else ""

        n = cell_totals.get(key, 0)
        lam = n / (n + 10.0)  # shrinkage toward population prior

        # Population prior
        prior = _POPULATION_TRANSITION.get(key, {st: 1.0 / 3 for st in STAGES})

        # Profile empirical distribution
        profile_counts = dict(counts.get(key, {}))
        profile_dist: Dict[str, float] = {
            st: profile_counts.get(st, 0) / max(n, 1) for st in STAGES
        }

        # Blend profile + population prior
        blended = {
            st: lam * profile_dist.get(st, 0.0) + (1.0 - lam) * prior.get(st, 1.0 / 3)
            for st in STAGES
        }

        # Apply talk_type transition bias when observed
        if tt_norm and tt_norm in _TALK_TYPE_TRANSITION_BIAS:
            bias = _TALK_TYPE_TRANSITION_BIAS[tt_norm]
            stage_idx = {"precontemplation": 0, "contemplation": 1, "preparation": 2}
            s_idx = stage_idx.get(s, 1)

            for st in STAGES:
                t_idx = stage_idx.get(st, 1)
                if t_idx > s_idx:       # higher stage
                    blended[st] = max(0.0, blended[st] + bias["up"])
                elif t_idx < s_idx:     # lower stage
                    blended[st] = max(0.0, blended[st] + bias["down"])

        # Normalize
        total = sum(blended.values())
        if total < 1e-9:
            return {st: 1.0 / 3 for st in STAGES}
        return {st: round(v / total, 4) for st, v in blended.items()}

    return transition_fn


# ---------------------------------------------------------------------------
# 6. Convenience: build both functions from Client's profile attributes
# ---------------------------------------------------------------------------

def build_simulator_fns_from_client(client_obj) -> Tuple[
    Callable,
    Callable,
]:
    """
    Convenience wrapper: given an initialized CAMI Client instance,
    build both adaptive functions from its profile attributes.

    Uses client_obj.profile_talk_types when available (CAMI profiles.jsonl).

    Returns
    -------
    (readiness_score_fn, state_transition_fn)
    """
    concern_kws = list(getattr(client_obj, "concern_keywords", None) or [])
    talk_types = list(getattr(client_obj, "profile_talk_types", None) or []) or None

    readiness_fn = build_readiness_score_fn(
        profile_speakers=list(getattr(client_obj, "profile_speakers", [])),
        profile_actions=list(getattr(client_obj, "profile_actions", [])),
        profile_states=list(getattr(client_obj, "profile_states", [])),
        initial_stage=str(getattr(client_obj, "initial_stage", "precontemplation") or ""),
        concern_keywords=concern_kws,
        profile_talk_types=talk_types,
    )
    transition_fn = build_state_transition_fn(
        profile_speakers=list(getattr(client_obj, "profile_speakers", [])),
        profile_actions=list(getattr(client_obj, "profile_actions", [])),
        profile_states=list(getattr(client_obj, "profile_states", [])),
        initial_stage=str(getattr(client_obj, "initial_stage", "precontemplation") or ""),
        profile_talk_types=talk_types,
    )
    return readiness_fn, transition_fn
