"""
Patient Simulator Evaluation — FEP counselor + various patient simulators.

Following CAMI (Yang et al., ACL 2025) Table 36 expert evaluation:
  Phase 1: FEP counselor interacts with each patient simulator → generate conversations
  Phase 2: Qwen3-32B judge scores Consistency (Persona/Beliefs/Motivation/Plan) + Realism

Patient simulators compared:
  1. CAMI profile_rules
  2. CAMI dynamic
  3. Profile-based (LLM + profile only)
  4. Pro+Act-based (LLM + profile + action descriptions)
  5. Ours (SSV5)
  6. Ours (SSV6)
  7. Ours (SSV7)
  8. Real (ground truth, upper bound)

Usage:
    python eval_patient_simulator.py \
        --profiles_path ../baseline/mi_coaching/CAMI/annotations/profiles.jsonl \
        --model Qwen/Qwen3-8B \
        --judge_model Qwen/Qwen3-32B \
        --load_in_4bit \
        --test_num 5 \
        --output_json logs/results/patient_sim_eval.json
"""

import argparse
import json
import math
import os
import random
import re
import sys
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEP_DIR = os.path.join(SCRIPT_DIR, "..", "FEP-Agent")
sys.path.insert(0, FEP_DIR)
sys.path.insert(0, SCRIPT_DIR)

from simple_simulator import _build_population_table, STAGES
from simple_simulator_v5 import SimpleSimulatorV5
from simple_simulator_v6 import SimpleSimulatorV6
from simple_simulator_v7 import (
    SimpleSimulatorV7, build_population_action_dist, load_all_profiles,
)
from misc_annotator import CLASSIFY_SYSTEM_PROMPT, _parse_label


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# MISC Classifier (reused from eval_simple_sim.py)
# ---------------------------------------------------------------------------

class MISCClassifier:
    def __init__(self, tokenizer, lm):
        self.tokenizer = tokenizer
        self.lm = lm

    def classify(self, counselor_utt: str, context: str = "", state: str = "") -> str:
        import torch
        user_prompt = (
            f"Dialogue context:\n{context[-500:]}\n\n"
            f"Client state: {state}\n\n"
            f"Counselor utterance to classify:\n\"{counselor_utt}\"\n\n"
            f"Return JSON: {{\"label\": \"<MISC category>\"}}"
        )
        messages = [
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.lm.device)
        with torch.no_grad():
            out = self.lm.generate(**inputs, max_new_tokens=64, do_sample=False)
        raw = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip()
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        raw = re.sub(r'</?think>', '', raw).strip()
        return _parse_label(raw)


# ---------------------------------------------------------------------------
# Patient simulator factories — each returns (simulator_obj, step_fn)
#   step_fn(misc_action, counselor_utt, rng) → (patient_response, stage, done)
# ---------------------------------------------------------------------------

def _build_profile_system_prompt(profile: Dict) -> str:
    behavior = profile.get("Behavior", "problematic behavior")
    goal = profile.get("topic", "behavior change")
    personas = profile.get("Personas", [])
    motivation = profile.get("Motivation", [])
    beliefs = profile.get("Beliefs", [])
    plans = profile.get("Acceptable Plans", [])
    states = profile.get("states", [])
    init_state = states[0] if states else "Precontemplation"
    p_str = "\n".join(f"- {x}" for x in personas) or "  (none)"
    m_str = (motivation[-1] if isinstance(motivation, list) and motivation
             else str(motivation))
    b_str = "\n".join(f"- {x}" for x in beliefs) or "  (none)"
    pl_str = "\n".join(f"- {x}" for x in plans) or "  (none)"
    return (
        f"You are a client in an MI counseling session.\n"
        f"Problematic behavior: {behavior}\n"
        f"Counselor's goal: {goal}\n\n"
        f"Your persona:\n{p_str}\n\n"
        f"Current mental state: {init_state}\n"
        f"Your motivation for change: {m_str}\n"
        f"Your beliefs obstructing change:\n{b_str}\n"
        f"Your preferred change plans:\n{pl_str}\n"
    )


def _hf_generate(messages, tokenizer, lm, max_new_tokens=128):
    import torch
    text = tokenizer.apply_chat_template(messages, tokenize=False,
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


_CAMI_ACTION2PROMPT = {
    "Deny": "directly refuse to admit your behavior is problematic.",
    "Downplay": "downplay the importance of your behavior.",
    "Blame": "blame external factors to justify your behavior.",
    "Inform": "share details about your background or emotions.",
    "Engage": "interact with counselor consistently.",
    "Hesitate": "show uncertainty about change.",
    "Doubt": "express skepticism about proposed changes.",
    "Acknowledge": "acknowledge the need for change.",
    "Accept": "agree to adopt the suggested plan.",
    "Reject": "decline the proposed plan.",
    "Plan": "propose or detail steps for a change plan.",
    "Terminate": "express a desire to end the session.",
}


class ProfileBasedPatient:
    """Profile-based baseline: LLM + profile only, no action/state control."""

    def __init__(self, profile, tokenizer, lm):
        self.profile = profile
        self.tokenizer = tokenizer
        self.lm = lm
        self.sys_prompt = _build_profile_system_prompt(profile) + (
            "\nRespond naturally as this client (1-3 sentences). "
            "Do NOT start with 'Client:'.\n/no_think")
        self.conversation = []

    def step(self, misc_action, counselor_utt, rng):
        messages = [{"role": "system", "content": self.sys_prompt}]
        for line in self.conversation[-6:]:
            if line.startswith("Counselor: "):
                messages.append({"role": "user", "content": line[11:]})
            elif line.startswith("Client: "):
                messages.append({"role": "assistant", "content": line[8:]})
        messages.append({"role": "user", "content": counselor_utt})
        resp = _hf_generate(messages, self.tokenizer, self.lm)
        if resp.startswith("Client: "):
            resp = resp[8:]
        self.conversation.append(f"Counselor: {counselor_utt}")
        self.conversation.append(f"Client: {resp}")
        return resp, "precontemplation", False  # no state tracking


class ProActBasedPatient:
    """Pro+Act-based baseline: LLM + profile + action descriptions."""

    def __init__(self, profile, tokenizer, lm):
        self.profile = profile
        self.tokenizer = tokenizer
        self.lm = lm
        actions_block = "\n".join(f"  {k}: {v}" for k, v in _CAMI_ACTION2PROMPT.items())
        self.sys_prompt = _build_profile_system_prompt(profile) + (
            f"\nAvailable client actions:\n{actions_block}\n\n"
            "Choose the most appropriate action then generate a response (1-3 sentences).\n"
            "Output ONLY JSON: {\"action\": \"<ActionName>\", \"response\": \"<text>\"}\n"
            "/no_think")
        self.conversation = []

    def step(self, misc_action, counselor_utt, rng):
        messages = [{"role": "system", "content": self.sys_prompt}]
        for line in self.conversation[-6:]:
            if line.startswith("Counselor: "):
                messages.append({"role": "user", "content": line[11:]})
            elif line.startswith("Client: "):
                messages.append({"role": "assistant", "content": line[8:]})
        messages.append({"role": "user", "content": counselor_utt})
        raw = _hf_generate(messages, self.tokenizer, self.lm, max_new_tokens=192)
        resp = raw.strip()
        try:
            m = re.search(r'\{[^{}]*"action"\s*:.*?\}', resp, re.DOTALL)
            if m:
                obj = json.loads(m.group())
                resp = obj.get("response", resp).strip()
        except (json.JSONDecodeError, AttributeError):
            pass
        if resp.startswith("Client: "):
            resp = resp[8:]
        self.conversation.append(f"Counselor: {counselor_utt}")
        self.conversation.append(f"Client: {resp}")
        return resp, "precontemplation", False


class CAMIPatient:
    """CAMI client simulator wrapper."""

    def __init__(self, profile, simulator_mode):
        from baseline.mi_coaching.CAMI.agents.client import Client as CAMIClient
        speakers = profile.get("speakers", [])
        utterances = profile.get("utterances", [])
        states = profile.get("states", [])
        talk_types = profile.get("talk_types", [])
        self.client = CAMIClient(
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
        seed_n = min(2, len(speakers))
        ctx = [f"{'Client' if speakers[i] == 'client' else 'Counselor'}: {utterances[i]}"
               for i in range(seed_n)]
        self.client.set_context(ctx)

    def step(self, misc_action, counselor_utt, rng):
        self.client.receive(f"Counselor: {counselor_utt}")
        action = self.client.select_action()
        stage = (self.client.state or "Precontemplation").lower()
        try:
            resp = self.client.reply()
            if isinstance(resp, str):
                resp = re.sub(r'^Client:\s*', '', resp).strip()
        except Exception:
            resp = ""
        self.client.update_state()
        done = stage == "preparation"
        return resp, stage, done


# ---------------------------------------------------------------------------
# Judge prompts & functions (same as eval_interactive_judge.py)
# ---------------------------------------------------------------------------

_CONSISTENCY_PROMPT = """\
You are an expert in motivational interviewing (MI) counseling. You are evaluating whether a simulated client's behavior during a counseling session is consistent with the given client profile.

## Client Profile
{profile}

## Counseling Session
{conversation}

## Task
Rate the consistency of the CLIENT's behavior with the given profile on each dimension (1-5 scale).

IMPORTANT: A real client expresses profile information NATURALLY — through relevant context, not by listing facts. High consistency means the client's words and behavior ALIGN with the profile, not that the client explicitly states every detail. Conversely, if the client unnaturally dumps profile information (e.g., listing all beliefs unprompted, or mentioning plans before the counselor brings them up), that is a sign of artificial behavior and should NOT receive a 5.

1. **Persona Consistency**: Does the client's behavior naturally reflect the persona described in the profile?
   1 = contradicts the persona, 3 = partially consistent, 5 = behavior naturally aligns with all persona aspects

2. **Beliefs Consistency**: Does the client naturally express beliefs consistent with the profile when relevant?
   1 = contradicts or ignores beliefs, 3 = some beliefs emerge naturally, 5 = beliefs emerge naturally in appropriate contexts

3. **Motivation Consistency**: Does the client's expressed motivation align with the profile?
   1 = contradicts motivation, 3 = partially aligned, 5 = motivation naturally aligns with the profile

4. **Plan Consistency**: When change plans are discussed, does the client respond consistently with the profile's acceptable plans?
   1 = rejects plans that should be acceptable, 3 = partially consistent, 5 = naturally consistent with acceptable plans

Output ONLY a JSON object:
{{"persona": <1-5>, "beliefs": <1-5>, "motivation": <1-5>, "plan": <1-5>}}"""

_REALISM_PROMPT = """\
You are an expert in motivational interviewing (MI) counseling. Evaluate whether the CLIENT's behavior is realistic compared to real human clients.

Consider:
- Natural, human-like responses with appropriate hesitations
- Realistic emotional reactions and resistance patterns
- Natural evolution of openness over the conversation
- A common flaw is being excessively compromising — agreeing too readily without realistic resistance

## Counseling Session
{conversation}

Rate realism on a 1-5 scale:
1 = Very unrealistic (formulaic, artificial)
2 = Mostly unrealistic
3 = Moderately realistic
4 = Mostly realistic
5 = Very realistic (indistinguishable from real human client)

Respond with ONLY a single digit (1-5)."""


def _judge_generate(messages, tokenizer, judge_lm, max_new_tokens=256):
    import torch
    if not any(m.get("role") == "system" for m in messages):
        messages = [{"role": "system", "content": "/no_think"}] + messages
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(judge_lm.device)
    with torch.no_grad():
        out = judge_lm.generate(**inputs, max_new_tokens=max_new_tokens,
                                do_sample=False, repetition_penalty=1.05)
    resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()
    resp = re.sub(r'<think>.*?</think>', '', resp, flags=re.DOTALL).strip()
    resp = re.sub(r'</?think>', '', resp).strip()
    return resp


def judge_consistency(conv_text, profile, tokenizer, judge_lm):
    profile_text = _build_profile_text(profile)
    prompt = _CONSISTENCY_PROMPT.format(profile=profile_text, conversation=conv_text)
    raw = _judge_generate([{"role": "user", "content": prompt}],
                          tokenizer, judge_lm, max_new_tokens=64)
    try:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            obj = json.loads(m.group())
            return {k: int(obj.get(k, 3)) for k in ("persona", "beliefs", "motivation", "plan")}
    except (json.JSONDecodeError, ValueError):
        pass
    return {"persona": 3, "beliefs": 3, "motivation": 3, "plan": 3}


def judge_realism(conv_text, tokenizer, judge_lm):
    prompt = _REALISM_PROMPT.format(conversation=conv_text)
    raw = _judge_generate([{"role": "user", "content": prompt}],
                          tokenizer, judge_lm, max_new_tokens=8)
    m = re.search(r'[1-5]', raw)
    return int(m.group()) if m else 3


def _build_profile_text(profile):
    lines = [f"Problematic behavior: {profile.get('Behavior', '')}",
             f"Counseling goal: {profile.get('topic', '')}"]
    for key, label in [("Personas", "Persona"), ("Beliefs", "Beliefs"),
                       ("Motivation", "Motivation"), ("Acceptable Plans", "Plans")]:
        items = profile.get(key, [])
        if items:
            if key == "Motivation":
                lines.append(f"{label}: {items[-1] if isinstance(items, list) else items}")
            else:
                lines.append(f"{label}:")
                for item in items:
                    lines.append(f"  - {item}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TTC
# ---------------------------------------------------------------------------

def compute_ttc(sim_first_transitions, real_first_transitions):
    from scipy.stats import spearmanr
    max_turn = 50
    pairs = [(s if s is not None else max_turn, r if r is not None else max_turn)
             for s, r in zip(sim_first_transitions, real_first_transitions)]
    if len(pairs) < 3:
        return {"ttc_rho": None, "ttc_p": None}
    sv, rv = zip(*pairs)
    if len(set(sv)) <= 1 or len(set(rv)) <= 1:
        return {"ttc_rho": 0.0, "ttc_p": 1.0}
    rho, p = spearmanr(sv, rv)
    return {"ttc_rho": round(rho, 3), "ttc_p": round(p, 4)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles_path", type=str,
                        default="../baseline/mi_coaching/CAMI/annotations/profiles.jsonl")
    parser.add_argument("--profiles_misc_path", type=str, default="")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen3-32B")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--test_num", type=int, default=5)
    parser.add_argument("--max_turns", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_judge", action="store_true")
    parser.add_argument("--output_json", type=str,
                        default="logs/results/patient_sim_eval.json")
    args = parser.parse_args()

    logger = get_logger("patient_sim_eval")
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    # Load profiles
    profiles = []
    with open(args.profiles_path) as f:
        for line in f:
            profiles.append(json.loads(line.strip()))
            if 0 < args.test_num <= len(profiles):
                break
    logger.info("Loaded %d profiles", len(profiles))

    # Population tables
    misc_path = args.profiles_misc_path or args.profiles_path
    pop_table = _build_population_table(misc_path)
    all_profiles_for_pop = load_all_profiles(args.profiles_path)
    pop_action_dist = build_population_action_dist(all_profiles_for_pop)

    # Real first-transition turns (for TTC)
    real_first_transitions = []
    for p in profiles:
        speakers, states = p.get("speakers", []), p.get("states", [])
        n = min(len(speakers), len(states))
        ft = None
        client_turn = 0
        for i in range(n):
            if speakers[i] == "client":
                if states[i] != "Precontemplation" and ft is None:
                    ft = client_turn
                client_turn += 1
        real_first_transitions.append(ft)

    # Patch args for doctor_factory
    _defaults = dict(
        client="qwen_hf", summary_model="default", persona_model="default",
        generation_model="default", planner_model="default",
        max_input_length=4096, max_output_length=1024,
        persona_json_retry_max=2, cami_preparation_calibration=True,
        state_label_space="cami_stage", persona_state_space="cami_stage",
        state_predict_from_probs=True, observation_focused_state=True,
        client_simulator_mode="dynamic", init_utterances=2,
        usr_name="Client", agent_name="Counselor", log_step=5,
        relevance_memory_number=1, context_memory_number=12, dist_thres=1.5,
        memory_cache=None, ori_mem_query=False, memory_session_turn_gap=0,
        max_agent_personas=0, max_user_personas=0,
        therapist_action_granularity="mi12",
        persona_single_step=False, persona_history_max=6,
        persona_short_term_window_turns=4, persona_long_term_summary_turns=12,
        persona_therapist_action="",
        disable_planner=False, fep_efe_action=True,
        sampling=False, sampling_step=10,
        sampling_path="logs/sampling", sampling_file_name="train.json",
        gpus="0", annomi_jsonl_path="", dataset="cami",
        data_path=os.path.join(FEP_DIR, "..", "baseline", "mi_coaching", "CAMI", "annotations"),
        data_name="profiles.jsonl", generation_out=False,
        next_state_eval_mode="auto",
        annomi_init_turns=8, cami_warmup_ratio=0.5, cami_min_eval_turns=3,
        obs_state_head_window_turns=0,
        openai_base_url=None, api_key="empty",
        semantic_overlap=False, semantic_model_name="all-MiniLM-L6-v2",
        retriever_path="", wikipedia_dir="", profile_path="",
        fep_lambda_e=0.4, fep_lambda_p=0.6,
        baseline_mode="fep", miti_behavior_scores=False,
        results_file="", log_file="", sample_ids="",
    )
    for k, v in _defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    args.test_num_main = args.test_num

    from interactive_cami_text_eval import doctor_factory, build_seed_history

    # ── Phase 1: Load simulation model ──
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    logger.info("Loading simulation model %s ...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    lm = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    lm.eval()
    classifier = MISCClassifier(tokenizer, lm)

    # ── Define patient simulators ──
    SIM_KEYS = [
        "cami_pr", "cami_dyn",
        "profile_based", "pro_act_based",
        "ssv5", "ssv6", "ssv7",
        "real",
    ]
    SIM_NAMES = {
        "cami_pr": "CAMI profile_rules",
        "cami_dyn": "CAMI dynamic",
        "profile_based": "Profile-based",
        "pro_act_based": "Pro+Act-based",
        "ssv5": "Ours (SSV5)",
        "ssv6": "Ours (SSV6)",
        "ssv7": "Ours (SSV7)",
        "real": "Real (human)",
    }

    all_results = {}
    all_conversations = {}  # sim_key -> [conv_text per profile]

    for sim_key in SIM_KEYS:
        if sim_key == "real":
            # Real conversations — just collect texts for judge evaluation
            conv_texts = []
            for pi, profile in enumerate(profiles):
                speakers = profile.get("speakers", [])
                utterances = profile.get("utterances", [])
                n = min(len(speakers), len(utterances))
                lines = []
                for i in range(n):
                    role = "Counselor" if speakers[i] == "counselor" else "Client"
                    lines.append(f"{role}: {utterances[i]}")
                conv_texts.append("\n".join(lines))
            all_conversations["real"] = conv_texts
            all_results["real"] = {"sim_first_transitions": real_first_transitions}
            logger.info("=== Real: %d profiles ===", len(profiles))
            continue

        logger.info("=== [%s] %s ===", sim_key, SIM_NAMES[sim_key])
        conv_texts = []
        sim_first_transitions = []
        sim_summaries = []

        for pi, profile in enumerate(profiles):
            rng = random.Random(args.seed + pi * 101)
            seed_history = build_seed_history(profile, 2)

            # Create patient simulator
            if sim_key == "cami_pr":
                patient = CAMIPatient(profile, "profile_rules")
            elif sim_key == "cami_dyn":
                patient = CAMIPatient(profile, "dynamic")
            elif sim_key == "profile_based":
                patient = ProfileBasedPatient(profile, tokenizer, lm)
            elif sim_key == "pro_act_based":
                patient = ProActBasedPatient(profile, tokenizer, lm)
            elif sim_key == "ssv5":
                patient = SimpleSimulatorV5.from_profile(profile, pop_table)
            elif sim_key == "ssv6":
                patient = SimpleSimulatorV6.from_profile(profile, pop_table, tokenizer, lm)
            elif sim_key == "ssv7":
                patient = SimpleSimulatorV7.from_profile(
                    profile, pop_table, pop_action_dist, tokenizer, lm)

            # Create FEP doctor (fixed)
            # Use unique sample_id per (sim_key, profile) to avoid ChromaDB collision
            unique_id = hash(f"{sim_key}_{pi}") % 100000
            doctor = doctor_factory("fep", args, logger, profile, unique_id)

            # Initialize
            patient_utt = seed_history[-1].replace("Client: ", "") if seed_history else "Hello."
            conversation_context = list(seed_history)
            conv_lines = list(seed_history)

            first_transition = None

            for turn in range(args.max_turns):
                # FEP doctor generates response
                doctor_utt = doctor.reply(patient_utt)

                # Classify into MISC
                context_str = "\n".join(conversation_context[-4:])
                stage = getattr(patient, 'stage', 'precontemplation')
                if hasattr(patient, 'client'):
                    stage = getattr(patient.client, 'state', 'Precontemplation').lower()
                misc_action = classifier.classify(doctor_utt, context_str, stage)

                # Patient simulator step
                if sim_key in ("ssv5", "ssv6", "ssv7"):
                    patient_action, new_stage, done, info = patient.step(
                        misc_action, doctor_utt, rng)
                    patient_utt = info.get("patient_response", "")
                    cur_stage = patient.stage
                elif sim_key in ("cami_pr", "cami_dyn"):
                    patient_utt, cur_stage, done = patient.step(
                        misc_action, doctor_utt, rng)
                else:
                    patient_utt, cur_stage, done = patient.step(
                        misc_action, doctor_utt, rng)

                # Track first transition
                if first_transition is None and cur_stage != "precontemplation":
                    first_transition = turn

                conv_lines.append(f"Counselor: {doctor_utt}")
                conv_lines.append(f"Client: {patient_utt}")
                conversation_context.append(f"Counselor: {doctor_utt}")
                conversation_context.append(f"Client: {patient_utt}")

                doctor.receive(f"Client: {patient_utt}")

                if done:
                    break

            conv_texts.append("\n".join(conv_lines))
            sim_first_transitions.append(first_transition)

            final_stage = getattr(patient, 'stage', cur_stage)
            logger.info("  [%s] profile %d: stage=%s turns=%d trans_turn=%s",
                        sim_key, profile.get("idx", pi), final_stage,
                        turn + 1, first_transition)

            del doctor
            import gc; gc.collect()
            torch.cuda.empty_cache()

        all_conversations[sim_key] = conv_texts
        all_results[sim_key] = {
            "sim_first_transitions": sim_first_transitions,
        }

        # TTC
        ttc = compute_ttc(sim_first_transitions, real_first_transitions)
        all_results[sim_key]["ttc"] = ttc
        logger.info("  [%s] TTC: ρ=%s p=%s", sim_key,
                    ttc.get("ttc_rho"), ttc.get("ttc_p"))

    # ── Phase 2: Judge evaluation ──
    if not args.skip_judge:
        same_model = (args.judge_model == args.model)
        if not same_model:
            logger.info("Unloading sim model, loading judge %s ...", args.judge_model)
            del lm
            torch.cuda.empty_cache()
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
            judge_lm = lm
            judge_tokenizer = tokenizer

        for sim_key in SIM_KEYS:
            logger.info("--- Judge: %s ---", sim_key)
            conv_texts = all_conversations[sim_key]
            cons_scores = {"persona": [], "beliefs": [], "motivation": [], "plan": []}
            realism_scores = []

            for pi, (conv_text, profile) in enumerate(zip(conv_texts, profiles)):
                cons = judge_consistency(conv_text, profile, judge_tokenizer, judge_lm)
                for dim in cons:
                    cons_scores[dim].append(cons[dim])
                realism = judge_realism(conv_text, judge_tokenizer, judge_lm)
                realism_scores.append(realism)

                if (pi + 1) % 5 == 0 or pi + 1 == len(profiles):
                    logger.info("  [%s] judge %d/%d done", sim_key, pi + 1, len(profiles))

            _avg = lambda lst: round(sum(lst) / len(lst), 2) if lst else None
            all_results[sim_key].update({
                "persona": _avg(cons_scores["persona"]),
                "beliefs": _avg(cons_scores["beliefs"]),
                "motivation": _avg(cons_scores["motivation"]),
                "plan": _avg(cons_scores["plan"]),
                "consistency_avg": _avg([sum(x)/4 for x in zip(
                    cons_scores["persona"], cons_scores["beliefs"],
                    cons_scores["motivation"], cons_scores["plan"])]),
                "realism": _avg(realism_scores),
            })

    # ── Save ──
    # Remove non-serializable fields
    for k in all_results:
        all_results[k].pop("sim_first_transitions", None)

    payload = {
        "config": {
            "model": args.model,
            "judge_model": args.judge_model if not args.skip_judge else "skipped",
            "test_num": args.test_num,
            "max_turns": args.max_turns,
            "n_profiles": len(profiles),
        },
        "results": all_results,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ── Print table ──
    W = 105
    print("\n" + "=" * W)
    print("Patient Simulator Evaluation — FEP counselor + various patient simulators")
    print("Judge: %s | Profiles: %d" % (
        args.judge_model if not args.skip_judge else "skipped", len(profiles)))
    print("=" * W)
    print("%-22s | %7s %7s %7s %7s | %11s | %7s | %7s" % (
        "Patient Simulator", "Pers.↑", "Bel.↑", "Mot.↑", "Plan↑",
        "Consist.↑", "Realism↑", "TTC-ρ↑"))
    print("-" * W)
    for sim_key in SIM_KEYS:
        r = all_results.get(sim_key, {})
        ttc = r.get("ttc", {})
        print("%-22s | %7s %7s %7s %7s | %11s | %7s | %7s" % (
            SIM_NAMES.get(sim_key, sim_key),
            _fmt(r.get("persona")),
            _fmt(r.get("beliefs")),
            _fmt(r.get("motivation")),
            _fmt(r.get("plan")),
            _fmt(r.get("consistency_avg")),
            _fmt(r.get("realism")),
            _fmt(ttc.get("ttc_rho"), ".3f") if ttc.get("ttc_rho") is not None else "—",
        ))
    print("=" * W)
    logger.info("Saved to %s", args.output_json)


def _fmt(val, spec=".2f"):
    if val is None:
        return "—"
    return format(val, spec)


if __name__ == "__main__":
    main()
