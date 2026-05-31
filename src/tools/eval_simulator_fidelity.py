"""
Simulator Fidelity Evaluation — Replicating "Consistent Client Simulation"
(Yang et al., ACL 2025) evaluation methodology.

Two-phase pipeline:
  Phase 1  (simulation model, default Qwen3-8B):
    Replay real counselor utterances through each simulator; collect
    generated client responses + action/state predictions.

  Phase 2  (judge model, default Qwen3-32B, replacing GPT-4 / human experts):
    Run LLM-judge evaluation on the generated conversations following the
    CAMI paper metric suite exactly.

Metric suite (from Yang et al., ACL 2025 Tables 1–5):
  Table 1 – Profile Consistency
    Personas%   Motivation%   Beliefs%   Plans%   Receptivity-ρ
  Table 2 – Dynamic Behaviour
    Avg.Rec   MR@20   Avg.MS   Act.KL
  Table 3 – Turn-Level (ROUGE / BERTScore)
    ROUGE-1   ROUGE-2   ROUGE-L   BERTScore
  Table 4/5 – Consistency & Human-likeness  (human → Qwen judge)
    Consistency-Score   HumanLike-Score

Simulators compared (12 total):
  1.  Random               – uniform random template response
  2.  Majority             – most-frequent talk_type per stage
  3.  Pop. Prior           – P(talk_type|stage) sampled
  4.  CAMI profile_rules   – CAMI rule-based simulator
  5.  CAMI dynamic         – CAMI pre-computed closure simulator
  6.  Profile-based        – LLM + client profile only (Yosef 2024; Wang 2024b)
  7.  Pro+Act-based        – LLM + profile + action descriptions (Zhang 2024)
  8.  Yang et al. (ACL25)  – full ConsistentMIClientSimulator pipeline (state+topic+engagement+action)
  9.  Ours (SSV4)          – MISC-action-driven P(talk_type|stage,action)
 10.  Ours (SSV5)          – correct causal chain + CAMI action vocab
 11.  Ours (SSV6)          – semantic triggers + LLM-generated responses
 12.  Ours (SSV7)          – profile-grounded LLM + Dirichlet action dist + semantic triggers
"""

import argparse
import json
import math
import os
import random
import re
import sys
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEP_DIR = os.path.join(SCRIPT_DIR, "..", "FEP-Agent")
sys.path.insert(0, FEP_DIR)
sys.path.insert(0, SCRIPT_DIR)

from simple_simulator import _build_population_table, STAGES, MISC_ACTIONS
from simple_simulator_v4 import SimpleSimulatorV4
from simple_simulator_v5 import SimpleSimulatorV5, _sample_patient_action, _build_patient_response
from simple_simulator_v6 import SimpleSimulatorV6
from simple_simulator_v7 import (
    SimpleSimulatorV7, build_population_action_dist, load_all_profiles,
)
from misc_annotator import CLASSIFY_SYSTEM_PROMPT, _parse_label

TT_LABELS    = ["change", "neutral", "sustain"]
STATE_LABELS = ["precontemplation", "contemplation", "preparation"]


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def load_profiles(path: str, test_num: int = 0) -> List[Dict]:
    profiles = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            profiles.append(json.loads(line))
            if 0 < test_num <= len(profiles):
                break
    return profiles


def classify_misc(counselor_utt: str, context: str, state: str,
                  tokenizer, lm) -> str:
    import torch
    user_prompt = (
        f"Dialogue context:\n{context[-500:]}\n\n"
        f"Client state: {state}\n\n"
        f"Counselor utterance to classify:\n\"{counselor_utt}\"\n\n"
        f"Return JSON: {{\"label\": \"<MISC category>\"}}"
    )
    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(lm.device)
    with torch.no_grad():
        out = lm.generate(**inputs, max_new_tokens=64, do_sample=False)
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip()
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    raw = re.sub(r'</?think>', '', raw).strip()
    return _parse_label(raw)


def _norm_stage(s: Optional[str]) -> str:
    if not s:
        return "precontemplation"
    return s.strip().lower()


def _talk_type_from_action(action: str) -> str:
    a = action.lower().strip()
    if a in {"acknowledge", "accept", "plan", "engage"}:
        return "change"
    elif a in {"deny", "downplay", "blame", "reject"}:
        return "sustain"
    return "neutral"


# ---------------------------------------------------------------------------
# Standard metrics (ROUGE, BERTScore, Action KL, Motivation)
# ---------------------------------------------------------------------------

def compute_action_kl(sim_actions: List[str], gold_actions: List[str]) -> float:
    """KL(gold || sim) over action distributions. Lower = better."""
    all_actions = set(sim_actions) | set(gold_actions)
    n_sim, n_gold = len(sim_actions), len(gold_actions)
    if n_sim == 0 or n_gold == 0:
        return float("inf")
    sim_counts  = Counter(sim_actions)
    gold_counts = Counter(gold_actions)
    eps = 1e-8
    kl = sum(p * math.log(p / (sim_counts.get(a, 0) / n_sim + eps))
             for a in all_actions
             if (p := gold_counts.get(a, 0) / n_gold) > 0)
    return round(kl, 4)


def _extract_first_transition_turns(state_trajectories: List[List[str]]) -> List[Optional[int]]:
    """Extract the turn index of the first state transition for each profile.
    Returns None if the profile never leaves precontemplation."""
    result = []
    for traj in state_trajectories:
        found = None
        for t, s in enumerate(traj):
            if s in ("contemplation", "preparation"):
                found = t
                break
        result.append(found)
    return result


def _extract_transition_turns_by_type(state_trajectories: List[List[str]]) -> Dict[str, List[Optional[int]]]:
    """Extract per-profile turn indices for PC→C and C→P transitions separately.
    Returns {"pc_to_c": [...], "c_to_p": [...]} with None for profiles that
    never reach the target transition."""
    pc_to_c = []
    c_to_p  = []
    for traj in state_trajectories:
        pc_c_turn = None
        c_p_turn  = None
        prev = None
        for t, s in enumerate(traj):
            if prev == "precontemplation" and s == "contemplation":
                if pc_c_turn is None:
                    pc_c_turn = t
            elif prev == "contemplation" and s == "preparation":
                if c_p_turn is None:
                    c_p_turn = t
            # Also handle direct PC→P (count as both)
            elif prev == "precontemplation" and s == "preparation":
                if pc_c_turn is None:
                    pc_c_turn = t
            prev = s
        pc_to_c.append(pc_c_turn)
        c_to_p.append(c_p_turn)
    return {"pc_to_c": pc_to_c, "c_to_p": c_to_p}


def compute_motivation_metrics(state_trajectories: List[List[str]],
                               max_turn: int = 20) -> Dict:
    first_transitions = _extract_first_transition_turns(state_trajectories)
    steps_list = [t for t in first_transitions if t is not None]
    reached = sum(1 for t in steps_list if t < max_turn)
    n = len(state_trajectories)
    avg_steps = sum(steps_list) / len(steps_list) if steps_list else None

    # Also extract per-type transitions
    by_type = _extract_transition_turns_by_type(state_trajectories)

    return {
        "motivation_rate_at_%d" % max_turn: round(reached / n, 4) if n else 0.0,
        "avg_motivation_steps": round(avg_steps, 2) if avg_steps is not None else None,
        "n_reached": len(steps_list),
        "n_total":   n,
        "per_profile_first_transition": first_transitions,
        "per_profile_pc_to_c": by_type["pc_to_c"],
        "per_profile_c_to_p":  by_type["c_to_p"],
    }


def _spearman_ttc(sim_list: List[Optional[int]],
                  real_list: List[Optional[int]],
                  max_turn: int = 50) -> Dict:
    """Compute Spearman TTC for a single transition type."""
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return {"rho": None, "p": None, "n": 0}
    pairs = []
    for s, r in zip(sim_list, real_list):
        sv = s if s is not None else max_turn
        rv = r if r is not None else max_turn
        pairs.append((sv, rv))
    if len(pairs) < 3:
        return {"rho": None, "p": None, "n": 0}
    sim_vals, real_vals = zip(*pairs)
    if len(set(sim_vals)) <= 1 or len(set(real_vals)) <= 1:
        return {"rho": 0.0, "p": 1.0, "n": len(pairs)}
    rho, p = spearmanr(sim_vals, real_vals)
    return {"rho": round(rho, 3), "p": round(p, 4), "n": len(pairs)}


def compute_ttc(sim_transitions: List[Optional[int]],
                real_transitions: List[Optional[int]]) -> Dict:
    """Transition Timing Correlation: Spearman between simulated and real
    first-transition turns, per profile. Measures whether the simulator
    differentiates slow-transitioning vs fast-transitioning profiles."""
    result = _spearman_ttc(sim_transitions, real_transitions)
    return {"ttc_rho": result["rho"], "ttc_p": result["p"], "n": result["n"]}


def compute_ttc_by_type(sim_motivation: Dict, gt_motivation: Dict) -> Dict:
    """Compute TTC separately for PC→C and C→P transitions."""
    result = {}
    for key in ("pc_to_c", "c_to_p"):
        sim_key = "per_profile_" + key
        sim_list  = sim_motivation.get(sim_key, [])
        real_list = gt_motivation.get(sim_key, [])
        if sim_list and real_list:
            ttc = _spearman_ttc(sim_list, real_list)
            result[key] = ttc
        else:
            result[key] = {"rho": None, "p": None, "n": 0}
    return result


def compute_rouge(predictions: List[str], references: List[str]) -> Dict:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return {"rouge1": None, "rouge2": None, "rougeL": None}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            continue
        s = scorer.score(ref, pred)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    n = len(r1)
    return {
        "rouge1": round(sum(r1) / n * 100, 2) if n else None,
        "rouge2": round(sum(r2) / n * 100, 2) if n else None,
        "rougeL": round(sum(rl) / n * 100, 2) if n else None,
        "n": n,
    }


def compute_bertscore(predictions: List[str], references: List[str]) -> Dict:
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        return {"bertscore_f1": None}
    valid = [(p, r) for p, r in zip(predictions, references) if p and r]
    if not valid:
        return {"bertscore_f1": None}
    preds, refs = zip(*valid)
    P, R, F1 = bert_score_fn(list(preds), list(refs), lang="en", verbose=False)
    return {"bertscore_f1": round(F1.mean().item() * 100, 2), "n": len(valid)}


# ---------------------------------------------------------------------------
# Ground truth extraction & population tables
# ---------------------------------------------------------------------------

def extract_ground_truth(profiles: List[Dict]) -> Dict:
    gt = {"action_dist": Counter(), "tt_dist": Counter(),
          "state_trajectories": [], "per_turn": []}
    for pi, p in enumerate(profiles):
        speakers   = p.get("speakers", [])
        utterances = p.get("utterances", [])
        states     = p.get("states", [])
        talk_types = p.get("talk_types", [])
        actions    = p.get("actions", [])
        n = min(len(speakers), len(utterances), len(states),
                len(talk_types), len(actions))
        traj = []
        for i in range(n):
            if speakers[i] == "client":
                gt["action_dist"][actions[i]] += 1
                gt["tt_dist"][talk_types[i].lower()] += 1
                traj.append(_norm_stage(states[i]))
                gt["per_turn"].append({
                    "profile_pos": pi, "turn_idx": i,
                    "gold_action": actions[i],
                    "gold_tt": talk_types[i].lower(),
                    "gold_state": _norm_stage(states[i]),
                    "gold_response": utterances[i],
                })
        gt["state_trajectories"].append(traj)
    return gt


def build_population_tt_by_stage(profiles: List[Dict]) -> Dict[str, Dict[str, float]]:
    counts = defaultdict(Counter)
    for p in profiles:
        speakers   = p.get("speakers", [])
        states     = p.get("states", [])
        talk_types = p.get("talk_types", [])
        n = min(len(speakers), len(states), len(talk_types))
        for i in range(n):
            if speakers[i] == "client":
                st = _norm_stage(states[i])
                tt = talk_types[i].lower()
                if st in STATE_LABELS and tt in TT_LABELS:
                    counts[st][tt] += 1
    table = {}
    for st, ct in counts.items():
        total = sum(ct.values())
        table[st] = {tt: ct.get(tt, 0) / total for tt in TT_LABELS}
    return table


# ---------------------------------------------------------------------------
# CAMI-paper LLM judge prompts
# (Yang et al., ACL 2025 "Consistent Client Simulation" §4 annotation pipeline)
# Qwen3-32B replaces GPT-4 (Tables 1–3) and human experts (Tables 4–5)
# ---------------------------------------------------------------------------

_EXTRACT_COMPONENTS_PROMPT = """\
You are analyzing a motivational interviewing (MI) counseling session to extract the client's profile information.

Counselor goal: {goal}
Client's problematic behavior: {behavior}

Counseling session:
{conversation}

Extract what the CLIENT explicitly mentioned:
- personas: background facts / personal information the client shared
- motivation: reasons the client expressed that might motivate behavior change
- beliefs: obstacles, doubts, or reasons the client raised against changing
- plans: specific change plans or steps the client proposed or agreed to

Output ONLY a JSON object (no other text):
{{"personas": ["..."], "motivation": ["..."], "beliefs": ["..."], "plans": ["..."]}}
If a category is empty, use []."""

_ENTAILMENT_PROMPT = """\
Does the DERIVED TEXT contain information that is entailed by (consistent with or supported by) the ORIGINAL TEXT?

ORIGINAL: {original}

DERIVED: {derived}

Answer only "Yes" or "No"."""

_RECEPTIVITY_PROMPT = """\
Analyze the following MI counseling session and rate the client's receptivity to behavior change on a 1–5 scale:
1 – Very resistant: refuses to engage, denies any problem
2 – Mostly resistant: rarely shows openness
3 – Moderate: mixed signals, sometimes receptive
4 – Mostly receptive: generally open to change
5 – Very receptive: actively engages, highly motivated

Session:
{conversation}

Receptivity score (1–5, respond with only a digit):"""

_CONSISTENCY_PROMPT = """\
Rate how consistently this simulated client reflects the given client profile throughout the counseling session.

Client profile:
{profile}

Counseling session:
{conversation}

1 – Very inconsistent (contradicts or ignores the profile)
2 – Mostly inconsistent
3 – Moderately consistent
4 – Mostly consistent
5 – Very consistent (accurately reflects all profile aspects)

Consistency score (1–5, respond with only a digit):"""

_HUMANLIKE_PROMPT = """\
Rate how natural and human-like the CLIENT's responses sound in this counseling session.

{conversation}

1 – Very artificial (formulaic, clearly generated)
2 – Mostly artificial
3 – Moderately natural
4 – Mostly natural
5 – Very natural (indistinguishable from a real human client)

Human-likeness score (1–5, respond with only a digit):"""


# ---------------------------------------------------------------------------
# Judge helper functions
# ---------------------------------------------------------------------------

def _judge_generate(messages: List[Dict], tokenizer, judge_lm,
                    max_new_tokens: int = 256) -> str:
    """Single-turn generation with the judge LLM."""
    import torch
    # Ensure /no_think system message to suppress Qwen3 <think> blocks
    if not any(m.get("role") == "system" for m in messages):
        messages = [{"role": "system", "content": "/no_think"}] + messages
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(judge_lm.device)
    with torch.no_grad():
        out = judge_lm.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, repetition_penalty=1.05)
    resp = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    # Still strip any residual <think> blocks as safety net
    resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
    resp = re.sub(r'</?think>', '', resp).strip()
    return resp


def _extract_components(conv_text: str, goal: str, behavior: str,
                        tokenizer, judge_lm) -> Dict[str, List[str]]:
    prompt = _EXTRACT_COMPONENTS_PROMPT.format(
        goal=goal, behavior=behavior, conversation=conv_text)
    raw = _judge_generate([{"role": "user", "content": prompt}],
                          tokenizer, judge_lm, max_new_tokens=512)
    try:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            return {k: obj.get(k, []) for k in ("personas", "motivation", "beliefs", "plans")}
    except (json.JSONDecodeError, AttributeError):
        pass
    return {"personas": [], "motivation": [], "beliefs": [], "plans": []}


def _check_entailment(original_item: str, derived_items: List[str],
                      tokenizer, judge_lm) -> bool:
    """Does any derived item entail this original item?"""
    if not derived_items:
        return False
    derived_text = "; ".join(derived_items)
    prompt = _ENTAILMENT_PROMPT.format(original=original_item, derived=derived_text)
    resp = _judge_generate([{"role": "user", "content": prompt}],
                           tokenizer, judge_lm, max_new_tokens=8)
    return "yes" in resp.lower()


def compute_profile_consistency(generated_convs: List[Dict], profiles: List[Dict],
                                tokenizer, judge_lm, logger) -> Dict:
    """
    Profile consistency following CAMI annotation pipeline (§4):
    Extract profile components from generated conv → entailment check vs. original.
    Returns Personas%, Motivation%, Beliefs%, Plans% (all ×100, higher=better).
    """
    scores = {"personas": [], "motivation": [], "beliefs": [], "plans": []}

    for pi, (conv_data, profile) in enumerate(zip(generated_convs, profiles)):
        goal     = profile.get("topic", "")
        behavior = profile.get("Behavior", "")
        derived  = _extract_components(conv_data["text"], goal, behavior,
                                       tokenizer, judge_lm)

        # Personas
        orig_p = profile.get("Personas", [])
        if orig_p:
            n_ok = sum(_check_entailment(p, derived["personas"], tokenizer, judge_lm)
                       for p in orig_p)
            scores["personas"].append(n_ok / len(orig_p))

        # Motivation (last element is the key motivation)
        orig_m = profile.get("Motivation", [])
        if orig_m:
            mot = orig_m[-1] if isinstance(orig_m, list) else str(orig_m)
            scores["motivation"].append(
                1.0 if _check_entailment(mot, derived["motivation"], tokenizer, judge_lm)
                else 0.0)

        # Beliefs
        orig_b = profile.get("Beliefs", [])
        if orig_b:
            n_ok = sum(_check_entailment(b, derived["beliefs"], tokenizer, judge_lm)
                       for b in orig_b)
            scores["beliefs"].append(n_ok / len(orig_b))

        # Plans
        orig_pl = profile.get("Acceptable Plans", [])
        if orig_pl:
            n_ok = sum(_check_entailment(pl, derived["plans"], tokenizer, judge_lm)
                       for pl in orig_pl)
            scores["plans"].append(n_ok / len(orig_pl))

        if (pi + 1) % 5 == 0 or pi + 1 == len(profiles):
            logger.info("    Profile consistency: %d/%d done", pi + 1, len(profiles))

    def _avg(lst):
        return round(sum(lst) / len(lst) * 100, 2) if lst else None

    return {k: _avg(v) for k, v in scores.items()}


def compute_receptivity_metrics(generated_convs: List[Dict], profiles: List[Dict],
                                tokenizer, judge_lm, logger) -> Dict:
    """
    Simulate receptivity scoring (judge LLM replaces GPT-4 in CAMI paper).
    Returns: avg_receptivity (mean ± std), rho (Spearman vs. real).
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        logger.warning("scipy not available; skipping receptivity correlation")
        return {"avg_receptivity": None, "receptivity_rho": None}

    sim_recs, real_recs = [], []
    for pi, (conv_data, profile) in enumerate(zip(generated_convs, profiles)):
        suggestibilities = profile.get("suggestibilities", [])
        if not suggestibilities:
            continue
        real_rec = sum(suggestibilities) / len(suggestibilities)

        prompt = _RECEPTIVITY_PROMPT.format(conversation=conv_data["text"])
        resp   = _judge_generate([{"role": "user", "content": prompt}],
                                 tokenizer, judge_lm, max_new_tokens=8)
        m = re.search(r'[1-5]', resp)
        sim_rec = int(m.group()) if m else 3

        sim_recs.append(sim_rec)
        real_recs.append(real_rec)

        if (pi + 1) % 5 == 0 or pi + 1 == len(profiles):
            logger.info("    Receptivity scoring: %d/%d done", pi + 1, len(profiles))

    avg_rec = round(sum(sim_recs) / len(sim_recs), 2) if sim_recs else None
    if len(sim_recs) >= 3:
        rho, _ = spearmanr(sim_recs, real_recs)
        rho = round(rho, 3)
    else:
        rho = None
    return {"avg_receptivity": avg_rec, "receptivity_rho": rho}


def score_consistency_humanlike(generated_convs: List[Dict], profiles: List[Dict],
                                tokenizer, judge_lm, logger) -> Dict:
    """
    LLM judge replaces human expert evaluation (Tables 4–5 in CAMI paper).
    Scores consistency (1–5) and human-likeness (1–5) per conversation.
    """
    cons_scores, hl_scores = [], []
    for pi, (conv_data, profile) in enumerate(zip(generated_convs, profiles)):
        # Build compact profile text
        behavior = profile.get("Behavior", "")
        goal     = profile.get("topic", "")
        p_lines  = [f"Behavior: {behavior}", f"Goal: {goal}"]
        personas = profile.get("Personas", [])
        if personas:
            p_lines.append("Personas: " + "; ".join(personas[:3]))
        motivation = profile.get("Motivation", [])
        if motivation:
            mot = motivation[-1] if isinstance(motivation, list) else str(motivation)
            p_lines.append(f"Motivation: {mot}")
        beliefs = profile.get("Beliefs", [])
        if beliefs:
            p_lines.append("Beliefs: " + "; ".join(beliefs[:3]))
        plans = profile.get("Acceptable Plans", [])
        if plans:
            p_lines.append("Plans: " + "; ".join(plans[:2]))
        profile_text = "\n".join(p_lines)

        # Consistency
        prompt_c = _CONSISTENCY_PROMPT.format(
            profile=profile_text, conversation=conv_data["text"])
        resp_c = _judge_generate([{"role": "user", "content": prompt_c}],
                                 tokenizer, judge_lm, max_new_tokens=8)
        mc = re.search(r'[1-5]', resp_c)
        cons_scores.append(int(mc.group()) if mc else 3)

        # Human-likeness
        prompt_h = _HUMANLIKE_PROMPT.format(conversation=conv_data["text"])
        resp_h = _judge_generate([{"role": "user", "content": prompt_h}],
                                 tokenizer, judge_lm, max_new_tokens=8)
        mh = re.search(r'[1-5]', resp_h)
        hl_scores.append(int(mh.group()) if mh else 3)

        if (pi + 1) % 5 == 0 or pi + 1 == len(profiles):
            logger.info("    Consistency/HL scoring: %d/%d done", pi + 1, len(profiles))

    _avg = lambda lst: round(sum(lst) / len(lst), 2) if lst else None
    return {"consistency": _avg(cons_scores), "human_likeness": _avg(hl_scores)}


# ---------------------------------------------------------------------------
# Generic replay framework
# ---------------------------------------------------------------------------

def replay_profiles(profiles: List[Dict], step_fn, logger, name: str = "sim"):
    """
    Replay real counselor utterances through a simulator.

    step_fn(profile, pi, counselor_utt, context_lines, turn_idx)
        → (pred_tt, pred_state, pred_response, pred_action)

    Returns:
        metrics  – standard metrics dict
        gen_convs – list[dict] with reconstructed conversation texts per profile
    """
    all_resp_preds, all_resp_golds = [], []
    all_action_preds, all_action_golds = [], []
    state_trajectories = []
    gen_convs = []          # NEW – one entry per profile

    for pi, profile in enumerate(profiles):
        speakers   = profile.get("speakers", [])
        utterances = profile.get("utterances", [])
        states     = profile.get("states", [])
        talk_types = profile.get("talk_types", [])
        actions    = profile.get("actions", [])
        n = min(len(speakers), len(utterances), len(states),
                len(talk_types), len(actions))

        seed_n = min(2, n)
        context_lines = []
        conv_turns    = []      # for profile consistency evaluation
        for i in range(seed_n):
            role = "Client" if speakers[i] == "client" else "Counselor"
            context_lines.append(f"{role}: {utterances[i]}")
            conv_turns.append(f"{role}: {utterances[i]}")

        state_traj = []
        for i in range(seed_n, n):
            if speakers[i] == "counselor" and i + 1 < n and speakers[i + 1] == "client":
                pred_tt, pred_state, pred_resp, pred_action = step_fn(
                    profile, pi, utterances[i], context_lines, i)

                all_resp_preds.append(pred_resp or "")
                all_resp_golds.append(utterances[i + 1])
                if pred_action:
                    all_action_preds.append(pred_action)
                all_action_golds.append(actions[i + 1])
                state_traj.append(pred_state)

                # Build simulated conversation for judge evaluation
                conv_turns.append(f"Counselor: {utterances[i]}")
                conv_turns.append(f"Client: {pred_resp or utterances[i + 1]}")

            role = "Client" if speakers[i] == "client" else "Counselor"
            context_lines.append(f"{role}: {utterances[i]}")

        state_trajectories.append(state_traj)
        gen_convs.append({
            "profile_idx": pi,
            "text":        "\n".join(conv_turns),
        })

        if (pi + 1) % 10 == 0 or pi == len(profiles) - 1:
            logger.info("  [%s] %d/%d profiles done", name, pi + 1, len(profiles))

    action_kl  = compute_action_kl(all_action_preds, all_action_golds) \
                 if all_action_preds else None
    motivation = compute_motivation_metrics(state_trajectories)
    rouge      = compute_rouge(all_resp_preds, all_resp_golds)
    bertscore  = compute_bertscore(all_resp_preds, all_resp_golds)

    metrics = {
        "action_kl":  action_kl,
        "motivation": motivation,
        "rouge":      rouge,
        "bertscore":  bertscore,
        "state_trajectories": state_trajectories,
    }
    return metrics, gen_convs


# ---------------------------------------------------------------------------
# Simulator step functions
# ---------------------------------------------------------------------------

def make_random_step(pop_tt, seed=42):
    rng = random.Random(seed)
    templates = {
        "change":  ["I guess maybe I should think about changing.",
                    "You might have a point there."],
        "neutral": ["I see what you mean.", "That's interesting."],
        "sustain": ["I don't think that's a problem.", "I'm fine the way I am."],
    }
    def step_fn(profile, pi, counselor_utt, context_lines, turn_idx):
        tt   = rng.choice(TT_LABELS)
        resp = rng.choice(templates[tt])
        return tt, "precontemplation", resp, None
    return step_fn


def make_majority_step(pop_tt):
    majority = {st: max(dist, key=dist.get) for st, dist in pop_tt.items()}
    templates = {
        "change":  "I've been thinking about it, and maybe I should try to change.",
        "neutral": "I see what you mean.",
        "sustain": "I don't think I need to change anything right now.",
    }
    def step_fn(profile, pi, counselor_utt, context_lines, turn_idx):
        states    = profile.get("states", [])
        cur_state = _norm_stage(states[turn_idx]) if turn_idx < len(states) else "precontemplation"
        tt        = majority.get(cur_state, "neutral")
        return tt, cur_state, templates[tt], None
    return step_fn


def make_population_prior_step(pop_tt, seed=42):
    rng = random.Random(seed)
    templates = {
        "change":  ["Maybe you have a point.", "I guess I could try."],
        "neutral": ["I hear what you're saying.", "That's something to think about."],
        "sustain": ["I really don't see the problem.", "It's just how things are."],
    }
    def step_fn(profile, pi, counselor_utt, context_lines, turn_idx):
        states    = profile.get("states", [])
        cur_state = _norm_stage(states[turn_idx]) if turn_idx < len(states) else "precontemplation"
        dist      = pop_tt.get(cur_state, {"neutral": 1.0})
        labels    = list(dist.keys())
        weights   = [dist[l] for l in labels]
        tt        = rng.choices(labels, weights=weights, k=1)[0]
        resp      = rng.choice(templates[tt])
        return tt, cur_state, resp, None
    return step_fn


def make_cami_step(profiles, simulator_mode, logger):
    """CAMI client simulator (profile_rules or dynamic mode)."""
    from baseline.mi_coaching.CAMI.agents.client import Client as CAMIClient

    clients = {}
    for pi, profile in enumerate(profiles):
        speakers   = profile.get("speakers", [])
        utterances = profile.get("utterances", [])
        states     = profile.get("states", [])
        talk_types = profile.get("talk_types", [])
        n = min(len(speakers), len(utterances), len(states), len(talk_types))
        client = CAMIClient(
            goal=profile["topic"], behavior=profile["Behavior"],
            reference="\n".join(
                f"{'Client' if s == 'client' else 'Counselor'}: {u}"
                for s, u in zip(speakers, utterances)),
            personas=profile.get("Personas", []),
            initial_stage=states[0] if states else "Precontemplation",
            final_stage=states[-1] if states else "Preparation",
            motivation=profile.get("Motivation", []),
            beliefs=profile.get("Beliefs", []),
            plans=profile.get("Acceptable Plans", []),
            speakers=speakers, utterances=utterances,
            actions=profile.get("actions", []),
            states=states, talk_types=talk_types,
            simulator_mode=simulator_mode,
            receptivity=3.0, model="Qwen/Qwen3-8B",
            wikipedia_dir="", retriever_path="",
        )
        seed_n = min(2, n)
        ctx = [f"{'Client' if speakers[i] == 'client' else 'Counselor'}: {utterances[i]}"
               for i in range(seed_n)]
        client.set_context(ctx)
        clients[pi] = client

    def step_fn(profile, pi, counselor_utt, context_lines, turn_idx):
        client = clients[pi]
        client.receive(f"Counselor: {counselor_utt}")
        sim_action = client.select_action()
        sim_tt     = _talk_type_from_action(sim_action)
        sim_state  = _norm_stage(client.state)
        try:
            resp = client.reply()
            if isinstance(resp, str):
                resp = re.sub(r'^Client:\s*', '', resp).strip()
        except Exception:
            resp = ""
        client.update_state()
        return sim_tt, sim_state, resp, sim_action
    return step_fn


# ---------------------------------------------------------------------------
# Profile-based and Pro+Act-based baselines
# (Yang et al., ACL 2025 §3 — these methods have no original published code;
#  implemented faithfully from paper descriptions)
# Profile-based: Yosef et al. 2024; Wang et al. 2024b
# Pro+Act-based: Zhang et al. 2024
# ---------------------------------------------------------------------------

_CAMI_ACTION2PROMPT = {
    "Deny":        "directly refuse to admit your behavior is problematic or needs change.",
    "Downplay":    "downplay the importance or impact of your behavior.",
    "Blame":       "blame external factors or others to justify your behavior.",
    "Inform":      "share details about your background, experiences, or emotions.",
    "Engage":      "interact with counselor consistently based on your state.",
    "Hesitate":    "show uncertainty, indicating ambivalence about change.",
    "Doubt":       "express skepticism about the practicality or success of proposed changes.",
    "Acknowledge": "acknowledge the need for change.",
    "Accept":      "agree to adopt the suggested action plan.",
    "Reject":      "decline the proposed plan, deeming it unsuitable.",
    "Plan":        "propose or detail steps for a change plan.",
    "Terminate":   "express a desire to end the session.",
}

_ACTION2STATE = {
    "Deny": "precontemplation", "Downplay": "precontemplation", "Blame": "precontemplation",
    "Hesitate": "contemplation", "Doubt": "contemplation", "Acknowledge": "contemplation",
    "Accept": "preparation", "Reject": "preparation", "Plan": "preparation",
    "Inform": None, "Engage": None, "Terminate": None,
}


def _build_profile_system_prompt(profile: Dict) -> str:
    behavior  = profile.get("Behavior", "problematic behavior")
    goal      = profile.get("topic",    "behavior change")
    personas  = profile.get("Personas", [])
    motivation = profile.get("Motivation", [])
    beliefs   = profile.get("Beliefs", [])
    plans     = profile.get("Acceptable Plans", [])
    states    = profile.get("states", [])
    init_state = states[0] if states else "Precontemplation"
    p_str  = "\n".join(f"- {x}" for x in personas)  or "  (none)"
    m_str  = (motivation[-1] if isinstance(motivation, list) and motivation
              else str(motivation))
    b_str  = "\n".join(f"- {x}" for x in beliefs)   or "  (none)"
    pl_str = "\n".join(f"- {x}" for x in plans)     or "  (none)"
    return (
        f"You are a client in an MI counseling session.\n"
        f"Problematic behavior: {behavior}\n"
        f"Counselor's goal: {goal}\n\n"
        f"Your persona:\n{p_str}\n\n"
        f"Current mental state: {init_state}\n"
        f"  Precontemplation = don't think behavior is problematic.\n"
        f"  Contemplation    = feel it may be problematic but remain ambivalent.\n"
        f"  Preparation      = ready to plan specific steps toward change.\n\n"
        f"Your motivation for change: {m_str}\n"
        f"Your beliefs obstructing change:\n{b_str}\n"
        f"Your preferred change plans:\n{pl_str}\n"
    )


def _hf_generate(messages: List[Dict], tokenizer, lm,
                 max_new_tokens: int = 128) -> str:
    import torch
    text   = tokenizer.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(lm.device)
    with torch.no_grad():
        out = lm.generate(**inputs, max_new_tokens=max_new_tokens,
                          do_sample=False, repetition_penalty=1.05)
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
    resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
    resp = re.sub(r'</?think>', '', resp).strip()
    return resp


def make_profile_based_step(profiles: List[Dict], tokenizer, lm):
    """Profile-based baseline (Yosef et al., 2024; Wang et al., 2024b)."""
    sys_prompts = {}
    for pi, p in enumerate(profiles):
        sys_prompts[pi] = _build_profile_system_prompt(p) + (
            "\nRespond naturally as this client (1–3 sentences). "
            "Do NOT start with 'Client:'.")

    def step_fn(profile, pi, counselor_utt, context_lines, turn_idx):
        messages = [{"role": "system", "content": sys_prompts[pi]}]
        for line in context_lines[-6:]:
            if line.startswith("Counselor: "):
                messages.append({"role": "user",      "content": line[11:]})
            elif line.startswith("Client: "):
                messages.append({"role": "assistant", "content": line[8:]})
        messages.append({"role": "user", "content": counselor_utt})
        resp = _hf_generate(messages, tokenizer, lm)
        if resp.startswith("Client: "):
            resp = resp[8:]
        states    = profile.get("states", [])
        cur_state = _norm_stage(states[turn_idx]) if turn_idx < len(states) else "precontemplation"
        rl = resp.lower()
        if any(w in rl for w in ["i'll", "i will", "i should try", "you're right",
                                   "i agree", "i want to change", "i'm ready"]):
            pred_tt = "change"
        elif any(w in rl for w in ["i don't think", "not a problem", "i'm fine",
                                    "no need to", "doesn't bother"]):
            pred_tt = "sustain"
        else:
            pred_tt = "neutral"
        return pred_tt, cur_state, resp.strip(), None
    return step_fn


def make_pro_act_based_step(profiles: List[Dict], tokenizer, lm):
    """Pro+Act-based baseline (Zhang et al., 2024)."""
    actions_block = "\n".join(f"  {k}: {v}" for k, v in _CAMI_ACTION2PROMPT.items())
    sys_prompts = {}
    for pi, p in enumerate(profiles):
        sys_prompts[pi] = _build_profile_system_prompt(p) + (
            f"\nAvailable client actions:\n{actions_block}\n\n"
            "For each counselor utterance choose the most appropriate action then generate "
            "a concise response (1–3 sentences).\n"
            "Output ONLY JSON: {\"action\": \"<ActionName>\", \"response\": \"<text>\"}")

    def step_fn(profile, pi, counselor_utt, context_lines, turn_idx):
        messages = [{"role": "system", "content": sys_prompts[pi]}]
        for line in context_lines[-6:]:
            if line.startswith("Counselor: "):
                messages.append({"role": "user",      "content": line[11:]})
            elif line.startswith("Client: "):
                messages.append({"role": "assistant", "content": line[8:]})
        messages.append({"role": "user", "content": counselor_utt})
        raw = _hf_generate(messages, tokenizer, lm, max_new_tokens=192)

        pred_action = None
        resp = raw.strip()
        try:
            m = re.search(r'\{[^{}]*"action"\s*:.*?\}', resp, re.DOTALL)
            if m:
                obj = json.loads(m.group())
                pred_action = obj.get("action", "").strip()
                resp = obj.get("response", resp).strip()
            else:
                am = re.search(r'"action"\s*:\s*"([^"]+)"', resp)
                rm = re.search(r'"response"\s*:\s*"([^"]+)"', resp)
                if am: pred_action = am.group(1).strip()
                if rm: resp = rm.group(1).strip()
        except (json.JSONDecodeError, AttributeError):
            pass

        if pred_action:
            for k in _CAMI_ACTION2PROMPT:
                if k.lower() == pred_action.lower():
                    pred_action = k
                    break
            else:
                pred_action = None

        if resp.startswith("Client: "):
            resp = resp[8:]

        if pred_action:
            pred_tt    = _talk_type_from_action(pred_action)
            pred_state = _ACTION2STATE.get(pred_action) or "precontemplation"
        else:
            rl = resp.lower()
            pred_tt = ("change" if any(w in rl for w in ["i'll", "i agree", "you're right"])
                       else "sustain" if any(w in rl for w in ["not a problem", "i'm fine"])
                       else "neutral")
            states    = profile.get("states", [])
            pred_state = _norm_stage(states[turn_idx]) if turn_idx < len(states) else "precontemplation"

        return pred_tt, pred_state, resp.strip(), pred_action
    return step_fn


def make_ssv4_step(profiles, pop_table, tokenizer, lm):
    """Our SimpleSimulatorV4."""
    simulators = {pi: SimpleSimulatorV4.from_profile(p, pop_table)
                  for pi, p in enumerate(profiles)}
    rngs = {pi: random.Random(42 + pi) for pi in range(len(profiles))}

    def step_fn(profile, pi, counselor_utt, context_lines, turn_idx):
        sim, rng  = simulators[pi], rngs[pi]
        misc_action = classify_misc(counselor_utt, "\n".join(context_lines[-4:]),
                                    sim.stage, tokenizer, lm)
        pred_tt, new_stage, done, info = sim.step(misc_action, counselor_utt, rng)
        return pred_tt, sim.stage, info.get("patient_response", ""), None
    return step_fn


def make_ssv5_step(profiles, pop_table, tokenizer, lm):
    """Our SimpleSimulatorV5 (correct causal chain + CAMI action vocab)."""
    simulators = {pi: SimpleSimulatorV5.from_profile(p, pop_table)
                  for pi, p in enumerate(profiles)}
    rngs = {pi: random.Random(42 + pi) for pi in range(len(profiles))}

    def step_fn(profile, pi, counselor_utt, context_lines, turn_idx):
        sim, rng  = simulators[pi], rngs[pi]
        misc_action = classify_misc(counselor_utt, "\n".join(context_lines[-4:]),
                                    sim.stage, tokenizer, lm)
        patient_action, new_stage, done, info = sim.step(misc_action, counselor_utt, rng)
        return _talk_type_from_action(patient_action), sim.stage, \
               info.get("patient_response", ""), patient_action
    return step_fn


def make_ssv6_step(profiles, pop_table, tokenizer, lm):
    """Our SimpleSimulatorV6 (semantic triggers + LLM responses)."""
    simulators = {pi: SimpleSimulatorV6.from_profile(p, pop_table, tokenizer, lm)
                  for pi, p in enumerate(profiles)}
    rngs = {pi: random.Random(42 + pi) for pi in range(len(profiles))}

    def step_fn(profile, pi, counselor_utt, context_lines, turn_idx):
        sim, rng  = simulators[pi], rngs[pi]
        misc_action = classify_misc(counselor_utt, "\n".join(context_lines[-4:]),
                                    sim.stage, tokenizer, lm)
        patient_action, new_stage, done, info = sim.step(misc_action, counselor_utt, rng)
        return _talk_type_from_action(patient_action), sim.stage, \
               info.get("patient_response", ""), patient_action
    return step_fn


def make_ssv7_step(profiles, pop_table, pop_action_dist, tokenizer, lm):
    """Our SimpleSimulatorV7 (semantic triggers + Dirichlet action dist + LLM responses)."""
    simulators = {
        pi: SimpleSimulatorV7.from_profile(p, pop_table, pop_action_dist, tokenizer, lm)
        for pi, p in enumerate(profiles)
    }
    rngs = {pi: random.Random(42 + pi) for pi in range(len(profiles))}

    def step_fn(profile, pi, counselor_utt, context_lines, turn_idx):
        sim, rng = simulators[pi], rngs[pi]
        misc_action = classify_misc(counselor_utt, "\n".join(context_lines[-4:]),
                                    sim.stage, tokenizer, lm)
        patient_action, new_stage, done, info = sim.step(misc_action, counselor_utt, rng)
        return _talk_type_from_action(patient_action), sim.stage, \
               info.get("patient_response", ""), patient_action
    return step_fn


# ---------------------------------------------------------------------------
# Yang et al. (ACL 2025) ConsistentMIClientSimulator integration
# ---------------------------------------------------------------------------

def _patch_yang_openai_with_hf(tokenizer, lm):
    """Monkey-patch the Yang et al. client module to use our local HF model
    instead of OpenAI API calls.  Returns the patched client module."""
    import importlib
    import torch

    YANG_DIR = os.path.join(SCRIPT_DIR, "..", "baseline", "ConsistentMIClientSimulator")

    # Set dummy OpenAI key so their module-level OpenAI() constructor doesn't fail
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "sk-dummy-not-used"
    if not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = "http://localhost:0"

    # Import the client module, avoiding namespace conflicts with any existing
    # "agents" package by temporarily manipulating sys.path and sys.modules
    saved_agents = sys.modules.pop("agents", None)
    saved_agents_client = sys.modules.pop("agents.client", None)
    saved_agents_env = sys.modules.pop("agents.env", None)
    sys.path.insert(0, YANG_DIR)
    try:
        yang_client_mod = importlib.import_module("agents.client")
    finally:
        sys.path.remove(YANG_DIR)
        # Rename the imported modules to avoid future conflicts
        yang_agents = sys.modules.pop("agents", None)
        yang_agents_client = sys.modules.pop("agents.client", None)
        sys.modules.pop("agents.env", None)
        # Store under unique names
        if yang_agents:
            sys.modules["yang_agents"] = yang_agents
        if yang_agents_client:
            sys.modules["yang_agents.client"] = yang_agents_client
        # Restore original modules
        if saved_agents is not None:
            sys.modules["agents"] = saved_agents
        if saved_agents_client is not None:
            sys.modules["agents.client"] = saved_agents_client
        if saved_agents_env is not None:
            sys.modules["agents.env"] = saved_agents_env

    def _hf_chat(messages, temperature=0.2, top_p=0.1, max_tokens=256,
                 json_mode=False):
        """Route an OpenAI-style chat call through our HF model."""
        # Deep copy to avoid mutating caller's list
        messages = [dict(m) for m in messages]
        # Add /no_think for Qwen3 models
        has_sys = False
        for m in messages:
            if m["role"] == "system":
                has_sys = True
                if "/no_think" not in m["content"]:
                    m["content"] = m["content"] + "\n/no_think"
                break
        if not has_sys:
            messages = [{"role": "system", "content": "/no_think"}] + messages

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=4096).to(lm.device)
        with torch.no_grad():
            out = lm.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=(temperature > 0.05),
                temperature=max(temperature, 0.01),
                top_p=top_p,
                repetition_penalty=1.05)
        resp = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True).strip()
        # Strip Qwen3 <think> blocks
        resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
        resp = re.sub(r'</?think>', '', resp).strip()
        return resp

    # Patch the three API functions used by their client
    def patched_get_precise_response(messages, model=None, temperature=0.2, top_p=0.1):
        return _hf_chat(messages, temperature=temperature, top_p=top_p, max_tokens=256)

    def patched_get_chatbot_response(messages, model=None, temperature=0.7, top_p=0.8, max_tokens=100):
        return _hf_chat(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    def patched_get_json_response(messages, model=None, temperature=0.2, top_p=0.1):
        return _hf_chat(messages, temperature=temperature, top_p=top_p,
                        max_tokens=256, json_mode=True)

    yang_client_mod.get_precise_response = patched_get_precise_response
    yang_client_mod.get_chatbot_response = patched_get_chatbot_response
    yang_client_mod.get_json_response    = patched_get_json_response

    # Also patch the env module
    try:
        yang_env_mod = importlib.import_module("agents.env")
        yang_env_mod.get_precise_response = patched_get_precise_response
    except Exception:
        pass

    return yang_client_mod


def make_yang_step(profiles, tokenizer, lm, logger):
    """Yang et al. (ACL 2025) ConsistentMIClientSimulator.
    Uses their full pipeline: state transition, topic retrieval (BGE-ReRanker),
    engagement tracking, action selection, information selection, response gen."""

    yang_client_mod = _patch_yang_openai_with_hf(tokenizer, lm)
    YangClient = yang_client_mod.Client

    YANG_DIR = os.path.join(SCRIPT_DIR, "..", "baseline", "ConsistentMIClientSimulator")
    wikipedia_dir = os.path.join(YANG_DIR, "wikipedias")
    retriever_path = os.path.join(YANG_DIR, "retriever")

    # Check for local retriever; fall back to HuggingFace model name
    if not os.path.isdir(retriever_path):
        retriever_path = "BAAI/bge-reranker-v2-m3"
    logger.info("Yang et al.: retriever=%s, wiki=%s", retriever_path, wikipedia_dir)

    # Pre-load shared retriever (568M params) once, then inject into all Clients
    import torch as _torch
    from transformers import AutoTokenizer as _AT, AutoModelForSequenceClassification as _AMSC
    _retr_device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    logger.info("  Loading shared BGE-ReRanker on %s...", _retr_device)
    _shared_retr_tok = _AT.from_pretrained(retriever_path)
    _shared_retr_model = _AMSC.from_pretrained(retriever_path).to(_retr_device)
    _shared_retr_model.eval()

    # Monkey-patch Client.__init__ to skip per-instance retriever loading
    _orig_init = YangClient.__init__

    def _patched_init(self, *args, **kwargs):
        # Temporarily replace the model loading lines by patching torch/transformers
        _AT_orig = yang_client_mod.AutoTokenizer
        _AMSC_orig = yang_client_mod.AutoModelForSequenceClassification

        class _FakeRetrTokenizer:
            @staticmethod
            def from_pretrained(*a, **kw):
                return _shared_retr_tok

        class _FakeRetrModel:
            @staticmethod
            def from_pretrained(*a, **kw):
                # Return a wrapper that has a no-op .to() and .eval()
                class _Wrapper:
                    def to(self, *a, **kw): return _shared_retr_model
                    def eval(self): return _shared_retr_model
                return _Wrapper()

        yang_client_mod.AutoTokenizer = _FakeRetrTokenizer
        yang_client_mod.AutoModelForSequenceClassification = _FakeRetrModel
        try:
            _orig_init(self, *args, **kwargs)
        finally:
            yang_client_mod.AutoTokenizer = _AT_orig
            yang_client_mod.AutoModelForSequenceClassification = _AMSC_orig
        # Ensure all clients share the same retriever instance
        self.retriever = _shared_retr_model
        self.retriever_tokenizer = _shared_retr_tok

    YangClient.__init__ = _patched_init

    # Build clients per profile
    clients = {}
    for pi, profile in enumerate(profiles):
        speakers   = profile.get("speakers", [])
        utterances = profile.get("utterances", [])
        states     = profile.get("states", [])
        n = min(len(speakers), len(utterances), len(states))

        reference = ""
        for s, u in zip(speakers[:50], utterances[:50]):
            role = "Client" if s == "client" else "Counselor"
            reference += f"{role}: {u}\n"

        suggestibilities = profile.get("suggestibilities", [3])
        receptivity = sum(suggestibilities) / len(suggestibilities)

        client = YangClient(
            goal=profile["topic"],
            behavior=profile.get("Behavior", ""),
            reference=reference,
            personas=list(profile.get("Personas", [])),
            initial_stage=states[0] if states else "Precontemplation",
            final_stage=states[-1] if states else "Preparation",
            motivation=list(profile.get("Motivation", ["unknown"])),
            beliefs=list(profile.get("Beliefs", [])),
            plans=list(profile.get("Acceptable Plans", [])),
            receptivity=receptivity,
            model="local",  # unused after patching
            wikipedia_dir=wikipedia_dir,
            retriever_path=retriever_path,
        )
        # Seed initial context (same 2-turn seed as other simulators)
        seed_n = min(2, n)
        ctx = [f"{'Client' if speakers[i] == 'client' else 'Counselor'}: {utterances[i]}"
               for i in range(seed_n)]
        client.context = ctx
        clients[pi] = client

    # Restore original __init__
    YangClient.__init__ = _orig_init

    _yang_state_map = {
        "Precontemplation": "precontemplation",
        "Contemplation": "contemplation",
        "Preparation": "preparation",
        "Motivation": "contemplation",  # Motivation is transitional → contemplation
    }

    # Build reverse map: action prompt substring → action name
    _action_prompt_keywords = {
        "Deny":        "directly refuse to admit",
        "Downplay":    "downplay the importance",
        "Blame":       "blame external factors",
        "Inform":      "share details about",
        "Engage":      "interact with counselor",
        "Hesitate":    "show uncertainty",
        "Doubt":       "skepticism about the practicality",
        "Acknowledge": "acknowledge the need for change",
        "Accept":      "agree to adopt",
        "Reject":      "decline the proposed plan",
        "Plan":        "propose or detail steps",
        "Terminate":   "express a desire to end",
    }

    def _parse_yang_action(raw_reply):
        """Extract the action name from Yang et al. output_instruction."""
        # Check each action's prompt fingerprint in the output
        for action, keyword in _action_prompt_keywords.items():
            if keyword in raw_reply:
                return action
        return None

    def step_fn(profile, pi, counselor_utt, context_lines, turn_idx):
        client = clients[pi]
        client.receive(f"Counselor: {counselor_utt}")

        try:
            raw_reply = client.reply()
        except Exception as e:
            logger.warning("Yang client reply failed (profile %d): %s", pi, e)
            raw_reply = "Client: I see."

        pred_action = _parse_yang_action(raw_reply)

        # Extract just the client response text
        resp = raw_reply
        # Strip instruction blocks [...] at the beginning
        resp = re.sub(r'\[.*?\]', '', resp).strip()
        resp = re.sub(r'^Client:\s*', '', resp).strip()
        if not resp:
            resp = "I see."

        pred_state = _yang_state_map.get(client.state, "precontemplation")
        pred_tt = _talk_type_from_action(pred_action) if pred_action else "neutral"

        return pred_tt, pred_state, resp, pred_action

    return step_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles_path",     type=str,
                        default="../baseline/mi_coaching/CAMI/annotations/profiles.jsonl")
    parser.add_argument("--profiles_misc_path", type=str, default="")
    parser.add_argument("--model",       type=str, default="Qwen/Qwen3-8B",
                        help="Model for simulation (Profile-based/Pro+Act/SSV4-6/MISC)")
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen3-32B",
                        help="Model for LLM-judge evaluation (replaces GPT-4/human experts)")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load judge model in 4-bit (required if GPU VRAM < 64GB)")
    parser.add_argument("--test_num",    type=int, default=0)
    parser.add_argument("--skip_judge",  action="store_true",
                        help="Skip LLM-judge phase (profile consistency, receptivity, scores)")
    parser.add_argument("--output_json", type=str,
                        default="logs/results/simulator_fidelity.json")
    args = parser.parse_args()

    logger = get_logger("eval_sim_fidelity")
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    profiles = load_profiles(args.profiles_path, args.test_num)
    logger.info("Loaded %d profiles", len(profiles))

    misc_path = args.profiles_misc_path or args.profiles_path
    pop_table = _build_population_table(misc_path)
    pop_tt    = build_population_tt_by_stage(profiles)
    all_profiles_for_pop = load_all_profiles(args.profiles_path)
    pop_action_dist = build_population_action_dist(all_profiles_for_pop)
    gt        = extract_ground_truth(profiles)
    logger.info("Ground truth: %d client turns, action_dist=%s",
                len(gt["per_turn"]), dict(gt["action_dist"].most_common(5)))

    gt_motivation = compute_motivation_metrics(gt["state_trajectories"])
    logger.info("GT motivation: MR@20=%.2f, avg_steps=%s",
                gt_motivation.get("motivation_rate_at_20", 0),
                gt_motivation.get("avg_motivation_steps"))

    # ── Phase 1: Load simulation model ────────────────────────────────────
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    logger.info("Loading simulation model %s ...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    lm = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    lm.eval()

    results    = {}
    sim_order  = []
    all_gen_convs = {}   # {sim_key: [conv_dict per profile]}

    def _run(key, label, step_fn, sim_idx, total):
        logger.info("=== [%2d/%d] %s ===", sim_idx, total, label)
        metrics, gen_convs = replay_profiles(profiles, step_fn, logger, key)
        metrics["simulator"] = label
        results[key]        = metrics
        all_gen_convs[key]  = gen_convs
        sim_order.append(key)

    N_SIM = 12
    _run("random",       "Random",            make_random_step(pop_tt),               1, N_SIM)
    _run("majority",     "Majority",           make_majority_step(pop_tt),             2, N_SIM)
    _run("pop_prior",    "Pop. Prior",         make_population_prior_step(pop_tt),     3, N_SIM)
    _run("cami_pr",      "CAMI profile_rules", make_cami_step(profiles, "profile_rules", logger), 4, N_SIM)
    _run("cami_dyn",     "CAMI dynamic",       make_cami_step(profiles, "dynamic",    logger),  5, N_SIM)
    _run("profile_based","Profile-based",      make_profile_based_step(profiles, tokenizer, lm), 6, N_SIM)
    _run("pro_act_based","Pro+Act-based",      make_pro_act_based_step(profiles, tokenizer, lm), 7, N_SIM)
    _run("yang_et_al",   "Yang et al. (ACL25)",make_yang_step(profiles, tokenizer, lm, logger), 8, N_SIM)
    _run("ssv4",         "Ours (SSV4)",        make_ssv4_step(profiles, pop_table, tokenizer, lm), 9, N_SIM)
    _run("ssv5",         "Ours (SSV5)",        make_ssv5_step(profiles, pop_table, tokenizer, lm), 10, N_SIM)
    _run("ssv6",         "Ours (SSV6)",        make_ssv6_step(profiles, pop_table, tokenizer, lm), 11, N_SIM)
    _run("ssv7",         "Ours (SSV7)",        make_ssv7_step(profiles, pop_table, pop_action_dist, tokenizer, lm), 12, N_SIM)

    # ── Compute TTC for all simulators ──────────────────────────────────
    gt_transitions = gt_motivation.get("per_profile_first_transition", [])
    for key in sim_order:
        sim_mot = results[key].get("motivation", {})
        sim_transitions = sim_mot.get("per_profile_first_transition", [])
        ttc = compute_ttc(sim_transitions, gt_transitions)
        results[key]["ttc"] = ttc
        # Compute per-type TTC (PC→C and C→P separately)
        ttc_by_type = compute_ttc_by_type(sim_mot, gt_motivation)
        results[key]["ttc_pc_to_c"] = ttc_by_type.get("pc_to_c", {})
        results[key]["ttc_c_to_p"]  = ttc_by_type.get("c_to_p", {})
        # Remove state_trajectories from results to keep JSON small
        results[key].pop("state_trajectories", None)
        # Remove per-profile lists from motivation to keep JSON clean
        if "motivation" in results[key]:
            results[key]["motivation"].pop("per_profile_first_transition", None)
            results[key]["motivation"].pop("per_profile_pc_to_c", None)
            results[key]["motivation"].pop("per_profile_c_to_p", None)

    # Also compute ROUGE / BERTScore for ground-truth reference
    gt_rouge = compute_rouge(
        [t["gold_response"] for t in gt["per_turn"]],
        [t["gold_response"] for t in gt["per_turn"]])  # upper bound: same text

    # ── Phase 2: LLM judge evaluation ─────────────────────────────────────
    if not args.skip_judge:
        # Unload simulation model to free VRAM before loading judge model
        same_model = (args.judge_model == args.model)
        if not same_model:
            logger.info("Unloading simulation model...")
            del lm
            torch.cuda.empty_cache()
            logger.info("Loading judge model %s (4bit=%s) ...",
                        args.judge_model, args.load_in_4bit)
            if args.load_in_4bit:
                from transformers import BitsAndBytesConfig
                bnb_cfg = BitsAndBytesConfig(load_in_4bit=True,
                                             bnb_4bit_compute_dtype=torch.bfloat16)
                judge_lm = AutoModelForCausalLM.from_pretrained(
                    args.judge_model, trust_remote_code=True,
                    quantization_config=bnb_cfg, device_map="auto")
            else:
                judge_lm = AutoModelForCausalLM.from_pretrained(
                    args.judge_model, trust_remote_code=True,
                    torch_dtype=torch.bfloat16, device_map="auto")
            judge_tokenizer = AutoTokenizer.from_pretrained(
                args.judge_model, trust_remote_code=True)
            judge_lm.eval()
        else:
            judge_lm        = lm
            judge_tokenizer = tokenizer

        for key in sim_order:
            logger.info("--- Judge phase: %s ---", key)
            gen_convs = all_gen_convs[key]

            logger.info("  Computing profile consistency...")
            prof_cons = compute_profile_consistency(
                gen_convs, profiles, judge_tokenizer, judge_lm, logger)

            logger.info("  Computing receptivity...")
            recep = compute_receptivity_metrics(
                gen_convs, profiles, judge_tokenizer, judge_lm, logger)

            logger.info("  Scoring consistency / human-likeness...")
            ch = score_consistency_humanlike(
                gen_convs, profiles, judge_tokenizer, judge_lm, logger)

            results[key].update({
                "profile_consistency": prof_cons,
                "receptivity":         recep,
                "consistency_score":   ch["consistency"],
                "humanlike_score":     ch["human_likeness"],
            })
    else:
        logger.info("Skipping LLM-judge phase (--skip_judge)")

    # ── Save ──────────────────────────────────────────────────────────────
    # Clean up non-serializable data before saving
    gt_motivation_clean = {k: v for k, v in gt_motivation.items()
                           if k != "per_profile_first_transition"}
    payload = {
        "config": {
            "profiles_path": args.profiles_path,
            "model":         args.model,
            "judge_model":   args.judge_model if not args.skip_judge else "skipped",
            "n_profiles":    len(profiles),
        },
        "ground_truth": {
            "action_dist":     dict(gt["action_dist"]),
            "tt_dist":         dict(gt["tt_dist"]),
            "motivation":      gt_motivation_clean,
        },
        "results": results,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ── Print CAMI-format comparison table ────────────────────────────────
    names = {
        "random":        "Random",
        "majority":      "Majority",
        "pop_prior":     "Pop. Prior",
        "cami_pr":       "CAMI profile_rules",
        "cami_dyn":      "CAMI dynamic",
        "profile_based": "Profile-based",
        "pro_act_based": "Pro+Act-based",
        "yang_et_al":    "Yang et al. (ACL25)",
        "ssv4":          "Ours (SSV4)",
        "ssv5":          "Ours (SSV5)",
        "ssv6":          "Ours (SSV6)",
        "ssv7":          "Ours (SSV7)",
    }

    W = 155

    # ── Table 1: Profile Consistency ──────────────────────────────────────
    print("\n" + "=" * W)
    print("TABLE 1 — Profile Consistency  (judge: %s)" %
          (args.judge_model if not args.skip_judge else "skipped"))
    print("=" * W)
    print("%-22s | %9s %11s %9s %8s | %6s" % (
        "Simulator", "Personas%↑", "Motivation%↑", "Beliefs%↑", "Plans%↑", "Recep-ρ↑"))
    print("-" * W)
    for key in sim_order:
        r  = results[key]
        pc = r.get("profile_consistency") or {}
        rc = r.get("receptivity")         or {}
        print("%-22s | %9s %11s %9s %8s | %6s" % (
            names.get(key, key),
            _fmt(pc.get("personas"),   ".1f"),
            _fmt(pc.get("motivation"), ".1f"),
            _fmt(pc.get("beliefs"),    ".1f"),
            _fmt(pc.get("plans"),      ".1f"),
            _fmt(rc.get("receptivity_rho"), ".3f"),
        ))
    print("=" * W)

    # ── Table 2: Dynamic Behaviour ────────────────────────────────────────
    print("\n" + "=" * W)
    print("TABLE 2 — Dynamic Behaviour")
    print("=" * W)
    print("%-22s | %8s | %6s  %7s | %7s" % (
        "Simulator", "Avg.Rec↑", "MR@20↓*", "Avg.MS↑*", "Act.KL↓"))
    print("-" * W)
    gt_mr = gt_motivation.get("motivation_rate_at_20", 0) * 100
    gt_ms = gt_motivation.get("avg_motivation_steps")
    print("%-22s | %8s | %5.1f%%  %7s | %7s" % (
        "Ground Truth", "—",
        gt_mr, "%.1f" % gt_ms if gt_ms else "—", "0.0000"))
    for key in sim_order:
        r   = results[key]
        rc  = r.get("receptivity") or {}
        mot = r.get("motivation",  {})
        akl = r.get("action_kl")
        mr  = mot.get("motivation_rate_at_20", 0) * 100
        ms  = mot.get("avg_motivation_steps")
        print("%-22s | %8s | %5.1f%%  %7s | %7s" % (
            names.get(key, key),
            _fmt(rc.get("avg_receptivity"), ".2f"),
            mr,
            "%.1f" % ms if ms is not None else "—",
            "%.4f" % akl if akl is not None else "—",
        ))
    print("=" * W)
    print("* MR@20 lower = harder to motivate (closer to real); "
          "Avg.MS higher = more steps needed (closer to real 9.8)")

    # ── Table 3: Turn-Level ───────────────────────────────────────────────
    print("\n" + "=" * W)
    print("TABLE 3 — Turn-Level (ROUGE / BERTScore)")
    print("=" * W)
    print("%-22s | %7s %7s %7s | %10s" % (
        "Simulator", "ROUGE-1↑", "ROUGE-2↑", "ROUGE-L↑", "BERTScore↑"))
    print("-" * W)
    for key in sim_order:
        r   = results[key]
        rou = r.get("rouge",     {})
        bs  = r.get("bertscore", {})
        print("%-22s | %7s %7s %7s | %10s" % (
            names.get(key, key),
            _fmt(rou.get("rouge1"), ".1f"),
            _fmt(rou.get("rouge2"), ".1f"),
            _fmt(rou.get("rougeL"), ".1f"),
            _fmt(bs.get("bertscore_f1"), ".1f"),
        ))
    print("=" * W)

    # ── Table 4/5: Consistency, Human-likeness & TTC ────────────────────
    print("\n" + "=" * W)
    print("TABLE 4/5 — Consistency, Realism & Transition Timing  "
          "(Qwen judge + TTC)")
    print("=" * W)
    print("%-22s | %14s | %14s | %7s %7s | %9s %9s" % (
        "Simulator", "Consistency↑", "Realism↑", "TTC-ρ↑", "TTC-p",
        "PC→C-ρ↑", "C→P-ρ↑"))
    print("-" * W)
    for key in sim_order:
        r = results[key]
        ttc = r.get("ttc", {})
        ttc_pc = r.get("ttc_pc_to_c", {})
        ttc_cp = r.get("ttc_c_to_p", {})
        print("%-22s | %14s | %14s | %7s %7s | %9s %9s" % (
            names.get(key, key),
            _fmt(r.get("consistency_score"), ".2f"),
            _fmt(r.get("humanlike_score"),   ".2f"),
            _fmt(ttc.get("ttc_rho"), ".3f"),
            _fmt(ttc.get("ttc_p"),   ".4f"),
            _fmt(ttc_pc.get("rho"), ".3f"),
            _fmt(ttc_cp.get("rho"), ".3f"),
        ))
    print("=" * W)
    print("TTC-ρ = Spearman(sim vs real first-transition turn); "
          "PC→C-ρ = precontemplation→contemplation only; "
          "C→P-ρ = contemplation→preparation only")
    logger.info("Saved to %s", args.output_json)


def _fmt(val, fmt_spec):
    if val is None:
        return "—"
    return ("%s" % format(val, fmt_spec.lstrip("%")))


if __name__ == "__main__":
    main()
