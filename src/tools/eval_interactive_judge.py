"""
LLM-as-Judge evaluation for interactive counselor–patient conversations.

Following CAMI (Yang et al., ACL 2025) expert evaluation methodology:
  - Consistency: 4 dimensions (Persona, Beliefs, Motivation, Plan), each 1–5
  - Realism: overall 1–5
  - MITI Global Scores (counselor-side, MITI 4.2.1):
    Cultivating Change Talk, Softening Sustain Talk, Partnership, Empathy, each 1–5

Input: pre-generated conversation histories from eval_simple_sim.py
       (simple_sim_{mode}_n{N}.json files)

Judge: Qwen3-32B (4-bit), replacing human MI experts.

Usage:
    python eval_interactive_judge.py \
        --results_dir logs/results \
        --profiles_path data/cami/profiles.jsonl \
        --test_num 5 \
        --judge_model Qwen/Qwen3-32B \
        --load_in_4bit \
        --output_json logs/results/interactive_judge.json
"""

import argparse
import json
import math
import os
import re
import sys
import logging
from typing import Dict, List, Optional, Tuple
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Conversation reconstruction
# ---------------------------------------------------------------------------

def reconstruct_conversation(history: List[Dict]) -> str:
    """Rebuild conversation text from eval_simple_sim.py history."""
    lines = []
    for t in history:
        counselor_utt = t.get("counselor_utt", "")
        # Strip "Counselor: " prefix if present
        if counselor_utt.startswith("Counselor: "):
            counselor_utt = counselor_utt[11:]
        patient_resp = t.get("patient_response", "")
        lines.append(f"Counselor: {counselor_utt}")
        lines.append(f"Client: {patient_resp}")
    return "\n".join(lines)


def reconstruct_real_conversation(profile: Dict, max_turns: int = 40) -> str:
    """Rebuild real conversation from profile data."""
    speakers = profile.get("speakers", [])
    utterances = profile.get("utterances", [])
    n = min(len(speakers), len(utterances), max_turns)
    lines = []
    for i in range(n):
        role = "Counselor" if speakers[i] == "counselor" else "Client"
        lines.append(f"{role}: {utterances[i]}")
    return "\n".join(lines)


def build_profile_text(profile: Dict) -> str:
    """Build compact profile description for judge prompt."""
    lines = []
    lines.append(f"Problematic behavior: {profile.get('Behavior', '')}")
    lines.append(f"Counseling goal: {profile.get('topic', '')}")

    personas = profile.get("Personas", [])
    if personas:
        lines.append("Persona:")
        for p in personas:
            lines.append(f"  - {p}")

    motivation = profile.get("Motivation", [])
    if motivation:
        mot = motivation[-1] if isinstance(motivation, list) else str(motivation)
        lines.append(f"Motivation for change: {mot}")

    beliefs = profile.get("Beliefs", [])
    if beliefs:
        lines.append("Beliefs obstructing change:")
        for b in beliefs:
            lines.append(f"  - {b}")

    plans = profile.get("Acceptable Plans", [])
    if plans:
        lines.append("Acceptable change plans:")
        for p in plans:
            lines.append(f"  - {p}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Judge prompts (following CAMI §4 expert evaluation)
# ---------------------------------------------------------------------------

_CONSISTENCY_PROMPT = """\
You are an expert in motivational interviewing (MI) counseling. You are evaluating whether a simulated client's behavior during a counseling session is consistent with the given client profile.

## Client Profile
{profile}

## Counseling Session
{conversation}

## Task
Rate the consistency of the CLIENT's behavior with the given profile on each dimension (1-5 scale):

1. **Persona Consistency**: Does the client express background facts matching the profile's persona?
   1 = contradicts or ignores persona, 5 = accurately reflects all persona aspects

2. **Beliefs Consistency**: Does the client express the beliefs/obstacles described in the profile?
   1 = never mentions beliefs, 5 = clearly expresses all beliefs

3. **Motivation Consistency**: Does the client express the motivation for change described in the profile?
   1 = never mentions motivation, 5 = clearly expresses motivation

4. **Plan Consistency**: Does the client mention or respond to the change plans in the profile?
   1 = never mentions plans, 5 = actively discusses plans

Output ONLY a JSON object (no other text):
{{"persona": <1-5>, "beliefs": <1-5>, "motivation": <1-5>, "plan": <1-5>}}"""

_REALISM_PROMPT = """\
You are an expert in motivational interviewing (MI) counseling with extensive experience working with real clients. You are evaluating whether a simulated client's behavior in a counseling session is realistic compared to how real human clients behave.

## Counseling Session
{conversation}

## Task
Rate how realistic the CLIENT's responses are compared to real human clients in MI counseling sessions.

Consider:
- Do the client's responses sound natural and human-like?
- Does the client show appropriate emotional reactions?
- Does the client's resistance/openness evolve naturally over the conversation?
- Are there signs of artificial or formulaic behavior (e.g., being too agreeable, switching positions too quickly, using therapy jargon)?

A common flaw in simulated clients is being excessively compromising — agreeing too readily to change without showing realistic resistance or ambivalence.

Rate on a 1-5 scale:
1 = Very unrealistic (clearly artificial, formulaic responses)
2 = Mostly unrealistic
3 = Moderately realistic
4 = Mostly realistic
5 = Very realistic (indistinguishable from a real human client)

Respond with ONLY a single digit (1-5)."""


# ---------------------------------------------------------------------------
# MITI Global Scores prompt (counselor-side, following MITI 4.2.1 manual)
# ---------------------------------------------------------------------------

_MITI_GLOBAL_PROMPT = """\
You are a certified MITI (Motivational Interviewing Treatment Integrity) coder with extensive training in evaluating counselor competence. You are rating the COUNSELOR's performance in this motivational interviewing session using the four MITI global scores.

## Counseling Session
{conversation}

## Task
Rate the COUNSELOR on each of the following four dimensions using a 1–5 Likert scale.

1. **Cultivating Change Talk**: To what extent does the counselor actively encourage the client's own language in favor of change and confidence in making that change? Look for open questions that evoke change talk, affirmations of change-related statements, reflections that highlight the client's change language, and summaries that collect change talk.
   1 = No attention to or active avoidance of client change talk
   2 = Sporadic attention with no consistent effort
   3 = Moderate and intentional effort to cultivate change talk
   4 = Consistent and skillful cultivation with good timing
   5 = Outstanding; nearly every counselor response strategically deepens change talk

2. **Softening Sustain Talk**: To what extent does the counselor avoid focusing on or amplifying the client's arguments against change or for the status quo? A high score means the counselor does NOT dwell on sustain talk, does NOT argue, and gently redirects when sustain talk appears.
   1 = Actively focuses on sustain talk, argues, confronts, or amplifies resistance
   2 = Frequently attends to sustain talk with little effort to redirect
   3 = Some attention to sustain talk, but makes moderate effort to redirect
   4 = Rarely attends to sustain talk; redirects smoothly when it appears
   5 = Excellent; consistently de-emphasizes sustain talk and smoothly redirects

3. **Partnership**: To what extent does the counselor convey that expertise and wisdom about change reside within the client? A high score reflects shared power, mutual exploration, and avoidance of unsolicited advice, lecturing, or taking the expert role.
   1 = Actively takes expert role; lectures, directs, or prescribes
   2 = Frequently assumes expertise with minimal client input
   3 = Some collaborative moments but also some directing
   4 = Predominantly collaborative; client's autonomy is generally honored
   5 = Outstanding partnership; the counselor consistently defers to the client's expertise about their own life

4. **Empathy**: To what extent does the counselor understand or make an effort to grasp the client's perspective and experience? Look for accurate reflections of emotion, validation of feelings, and genuine curiosity about the client's inner world.
   1 = No effort to understand the client's perspective; dismissive or judgmental
   2 = Sporadic and shallow attempts at understanding
   3 = Moderate understanding; some accurate reflections but misses key emotions
   4 = Good understanding with mostly accurate reflections of meaning and affect
   5 = Deep, accurate empathy; the counselor consistently captures nuanced feelings and unspoken meaning

Output ONLY a JSON object (no other text):
{{"cultivating_change_talk": <1-5>, "softening_sustain_talk": <1-5>, "partnership": <1-5>, "empathy": <1-5>}}"""


# ---------------------------------------------------------------------------
# Judge generation
# ---------------------------------------------------------------------------

def _judge_generate(messages: List[Dict], tokenizer, judge_lm,
                    max_new_tokens: int = 256) -> str:
    import torch
    # Add /no_think to suppress Qwen3 thinking blocks
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
    resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
    resp = re.sub(r'</?think>', '', resp).strip()
    return resp


def judge_consistency(conv_text: str, profile: Dict,
                      tokenizer, judge_lm) -> Dict[str, int]:
    """Score 4 consistency dimensions (1-5 each)."""
    profile_text = build_profile_text(profile)
    prompt = _CONSISTENCY_PROMPT.format(
        profile=profile_text, conversation=conv_text)
    raw = _judge_generate([{"role": "user", "content": prompt}],
                          tokenizer, judge_lm, max_new_tokens=64)
    # Parse JSON
    try:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            return {
                "persona": int(obj.get("persona", 3)),
                "beliefs": int(obj.get("beliefs", 3)),
                "motivation": int(obj.get("motivation", 3)),
                "plan": int(obj.get("plan", 3)),
            }
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # Fallback: try to find individual scores
    scores = {}
    for dim in ["persona", "beliefs", "motivation", "plan"]:
        m = re.search(rf'"{dim}"\s*:\s*([1-5])', raw)
        scores[dim] = int(m.group(1)) if m else 3
    return scores


def judge_realism(conv_text: str, tokenizer, judge_lm) -> int:
    """Score realism (1-5)."""
    prompt = _REALISM_PROMPT.format(conversation=conv_text)
    raw = _judge_generate([{"role": "user", "content": prompt}],
                          tokenizer, judge_lm, max_new_tokens=8)
    m = re.search(r'[1-5]', raw)
    return int(m.group()) if m else 3


_MITI_DIMS = ["cultivating_change_talk", "softening_sustain_talk", "partnership", "empathy"]


def judge_miti_global(conv_text: str, tokenizer, judge_lm) -> Dict[str, int]:
    """Score 4 MITI global dimensions (1-5 each) for the COUNSELOR."""
    prompt = _MITI_GLOBAL_PROMPT.format(conversation=conv_text)
    raw = _judge_generate([{"role": "user", "content": prompt}],
                          tokenizer, judge_lm, max_new_tokens=64)
    # Parse JSON
    try:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            return {dim: int(obj.get(dim, 3)) for dim in _MITI_DIMS}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # Fallback: try to find individual scores
    scores = {}
    for dim in _MITI_DIMS:
        m = re.search(rf'"{dim}"\s*:\s*([1-5])', raw)
        scores[dim] = int(m.group(1)) if m else 3
    return scores


# ---------------------------------------------------------------------------
# TTC computation
# ---------------------------------------------------------------------------

def compute_ttc(sim_results: List[Dict], real_transitions: List[Optional[int]]) -> Dict:
    """TTC from interactive conversations."""
    sim_transitions = []
    for r in sim_results:
        hist = r.get("history", [])
        first_trans = None
        for t_idx, t in enumerate(hist):
            if t.get("stage_before") == "precontemplation" and \
               t.get("stage_after") != "precontemplation":
                first_trans = t_idx
                break
        sim_transitions.append(first_trans)

    max_turn = 50
    pairs = []
    for s, r in zip(sim_transitions, real_transitions):
        sv = s if s is not None else max_turn
        rv = r if r is not None else max_turn
        pairs.append((sv, rv))

    if len(pairs) < 3:
        return {"ttc_rho": None, "ttc_p": None, "n": len(pairs)}
    sim_vals, real_vals = zip(*pairs)
    if len(set(sim_vals)) <= 1 or len(set(real_vals)) <= 1:
        return {"ttc_rho": 0.0, "ttc_p": 1.0, "n": len(pairs)}
    rho, p = spearmanr(sim_vals, real_vals)
    return {"ttc_rho": round(rho, 3), "ttc_p": round(p, 4), "n": len(pairs)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,
                        default="logs/results")
    parser.add_argument("--profiles_path", type=str,
                        default="data/cami/profiles.jsonl")
    parser.add_argument("--results_suffix", type=str, default="n29",
                        help="Suffix of result files: simple_sim_{mode}_{suffix}.json")
    parser.add_argument("--test_num", type=int, default=0,
                        help="Number of profiles to evaluate (0=all)")
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--output_json", type=str,
                        default="logs/results/interactive_judge.json")
    args = parser.parse_args()

    logger = get_logger("interactive_judge")

    # Load profiles
    profiles_by_idx = {}
    with open(args.profiles_path) as f:
        for line in f:
            p = json.loads(line.strip())
            profiles_by_idx[p.get("idx", len(profiles_by_idx))] = p
    logger.info("Loaded %d profiles", len(profiles_by_idx))

    # Compute real transition turns
    real_transitions_by_idx = {}
    for idx, p in profiles_by_idx.items():
        speakers = p.get("speakers", [])
        states = p.get("states", [])
        n = min(len(speakers), len(states))
        first_trans = None
        client_turn = 0
        for i in range(n):
            if speakers[i] == "client":
                if states[i] != "Precontemplation" and first_trans is None:
                    first_trans = client_turn
                client_turn += 1
        real_transitions_by_idx[idx] = first_trans

    # Load results for each mode
    modes = ["puma"]
    mode_data = {}  # mode -> list of per-profile dicts
    for mode in modes:
        fname = f"simple_sim_{mode}_{args.results_suffix}.json"
        fpath = os.path.join(args.results_dir, fname)
        if not os.path.exists(fpath):
            logger.warning("Missing: %s", fpath)
            continue
        with open(fpath) as f:
            data = json.load(f)
        for r in data.get("results", []):
            if r.get("mode") == mode:
                mode_data[mode] = r.get("per_profile", [])
                break
        logger.info("Loaded %s: %d profiles", mode, len(mode_data.get(mode, [])))

    # Add "real" as a special mode
    modes_with_real = list(mode_data.keys()) + ["real"]

    # Determine which profile indices to evaluate
    if mode_data:
        first_mode = list(mode_data.values())[0]
        all_profile_idxs = [pp["profile_idx"] for pp in first_mode]
    else:
        all_profile_idxs = sorted(profiles_by_idx.keys())

    if args.test_num > 0:
        all_profile_idxs = all_profile_idxs[:args.test_num]
    logger.info("Evaluating %d profiles", len(all_profile_idxs))

    # ── Load judge model ──
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    logger.info("Loading judge model %s (4bit=%s) ...", args.judge_model, args.load_in_4bit)
    judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)
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
    judge_lm.eval()
    logger.info("Judge model loaded.")

    # ── Evaluate each mode ──
    all_results = {}

    for mode in modes_with_real:
        logger.info("=== Evaluating: %s ===", mode)
        consistency_scores = {"persona": [], "beliefs": [], "motivation": [], "plan": []}
        realism_scores = []
        miti_scores = {dim: [] for dim in _MITI_DIMS}
        per_profile_results = []
        sim_results_for_ttc = []

        for pi_idx, profile_idx in enumerate(all_profile_idxs):
            profile = profiles_by_idx.get(profile_idx)
            if profile is None:
                continue

            # Build conversation text
            if mode == "real":
                conv_text = reconstruct_real_conversation(profile)
            else:
                pp_list = mode_data.get(mode, [])
                pp = None
                for p in pp_list:
                    if p.get("profile_idx") == profile_idx:
                        pp = p
                        break
                if pp is None:
                    continue
                conv_text = reconstruct_conversation(pp.get("history", []))
                sim_results_for_ttc.append(pp)

            # Judge: Consistency (4 dimensions)
            cons = judge_consistency(conv_text, profile, judge_tokenizer, judge_lm)
            for dim in ["persona", "beliefs", "motivation", "plan"]:
                consistency_scores[dim].append(cons[dim])

            # Judge: Realism
            realism = judge_realism(conv_text, judge_tokenizer, judge_lm)
            realism_scores.append(realism)

            # Judge: MITI Global Scores (counselor-side)
            miti = judge_miti_global(conv_text, judge_tokenizer, judge_lm)
            for dim in _MITI_DIMS:
                miti_scores[dim].append(miti[dim])

            per_profile_results.append({
                "profile_idx": profile_idx,
                "consistency": cons,
                "realism": realism,
                "miti_global": miti,
            })

            if (pi_idx + 1) % 5 == 0 or pi_idx + 1 == len(all_profile_idxs):
                logger.info("  [%s] %d/%d profiles done", mode, pi_idx + 1, len(all_profile_idxs))

        # Aggregate
        def _avg(lst):
            return round(sum(lst) / len(lst), 2) if lst else None

        agg = {
            "persona": _avg(consistency_scores["persona"]),
            "beliefs": _avg(consistency_scores["beliefs"]),
            "motivation": _avg(consistency_scores["motivation"]),
            "plan": _avg(consistency_scores["plan"]),
            "consistency_avg": _avg([
                sum(v) / 4 for v in zip(
                    consistency_scores["persona"],
                    consistency_scores["beliefs"],
                    consistency_scores["motivation"],
                    consistency_scores["plan"],
                )
            ]) if consistency_scores["persona"] else None,
            "realism": _avg(realism_scores),
            # MITI Global Scores (counselor quality)
            "cultivating_change_talk": _avg(miti_scores["cultivating_change_talk"]),
            "softening_sustain_talk": _avg(miti_scores["softening_sustain_talk"]),
            "partnership": _avg(miti_scores["partnership"]),
            "empathy": _avg(miti_scores["empathy"]),
            "miti_global_avg": _avg([
                sum(v) / 4 for v in zip(*(miti_scores[d] for d in _MITI_DIMS))
            ]) if miti_scores["cultivating_change_talk"] else None,
            "n_profiles": len(per_profile_results),
        }

        # TTC (only for modes with state machines, not "real")
        if mode != "real" and sim_results_for_ttc:
            real_trans = [real_transitions_by_idx.get(pid) for pid in all_profile_idxs
                          if profiles_by_idx.get(pid) is not None]
            ttc = compute_ttc(sim_results_for_ttc, real_trans)
            agg["ttc"] = ttc

        all_results[mode] = {
            "aggregate": agg,
            "per_profile": per_profile_results,
        }

    # ── Save ──
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    payload = {
        "config": {
            "judge_model": args.judge_model,
            "results_suffix": args.results_suffix,
            "test_num": args.test_num,
            "n_profiles": len(all_profile_idxs),
        },
        "results": {k: v["aggregate"] for k, v in all_results.items()},
        "per_profile": {k: v["per_profile"] for k, v in all_results.items()},
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ── Print results table ──
    W = 130
    print("\n" + "=" * W)
    print("Interactive Conversation Judge Evaluation  (judge: %s)" % args.judge_model)
    print("Following CAMI (Yang et al., ACL 2025) Table 36 + MITI 4.2.1 Global Scores")
    print("=" * W)

    # Client-side metrics
    print("\n--- Client-Side (Simulator Quality) ---")
    print("%-18s | %8s %8s %8s %8s | %14s | %7s | %7s %7s" % (
        "Method", "Persona↑", "Beliefs↑", "Motiv.↑", "Plan↑",
        "Consistency↑", "Realism↑", "TTC-ρ↑", "TTC-p"))
    print("-" * W)
    for mode in modes_with_real:
        if mode not in all_results:
            continue
        a = all_results[mode]["aggregate"]
        ttc = a.get("ttc", {})
        ttc_rho = ttc.get("ttc_rho")
        ttc_p = ttc.get("ttc_p")
        print("%-18s | %8s %8s %8s %8s | %14s | %7s | %7s %7s" % (
            mode if mode != "real" else "Real (human)",
            _fmt(a.get("persona")),
            _fmt(a.get("beliefs")),
            _fmt(a.get("motivation")),
            _fmt(a.get("plan")),
            _fmt(a.get("consistency_avg")),
            _fmt(a.get("realism")),
            _fmt(ttc_rho, ".3f") if ttc_rho is not None else "—",
            _fmt(ttc_p, ".4f") if ttc_p is not None else "—",
        ))

    # Counselor-side MITI Global Scores
    print("\n--- Counselor-Side (MITI Global Scores) ---")
    print("%-18s | %8s %8s %8s %8s | %10s" % (
        "Method", "CultCT↑", "SoftST↑", "Partn.↑", "Empath↑", "MITI-Avg↑"))
    print("-" * W)
    for mode in modes_with_real:
        if mode not in all_results:
            continue
        a = all_results[mode]["aggregate"]
        print("%-18s | %8s %8s %8s %8s | %10s" % (
            mode if mode != "real" else "Real (human)",
            _fmt(a.get("cultivating_change_talk")),
            _fmt(a.get("softening_sustain_talk")),
            _fmt(a.get("partnership")),
            _fmt(a.get("empathy")),
            _fmt(a.get("miti_global_avg")),
        ))
    print("=" * W)
    logger.info("Saved to %s", args.output_json)


def _fmt(val, spec=".2f"):
    if val is None:
        return "—"
    return format(val, spec)


if __name__ == "__main__":
    main()
