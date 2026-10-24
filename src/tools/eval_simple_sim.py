"""
Evaluate the PUMA counselor on SimpleSimulator.

Uses the PUMA counselor from src/agent/counselor.py.
Environment is SimpleSimulator (profile-grounded, LLM-driven patient simulation).
Counselor utterances are classified into MISC 17-class to determine action.

Usage:
    python eval_simple_sim.py \
        --profiles_misc_path data/cami/profiles_misc.jsonl \
        --model Qwen/Qwen3-8B \
        --test_num 5 \
        --max_turns 20 \
        --output_json logs/results/simple_sim_sub5.json
"""

import argparse
import json
import os
import random
import re
import sys
import logging
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.join(SCRIPT_DIR, "..", "agent")
sys.path.insert(0, AGENT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from simulator_common import _build_population_table, MISC_ACTIONS, STAGES
from simple_simulator import SimpleSimulator, build_population_action_dist, load_all_profiles
from misc_annotator import CLASSIFY_SYSTEM_PROMPT, _parse_label


# ---------------------------------------------------------------------------
# MISC Classifier
# ---------------------------------------------------------------------------

class MISCClassifier:
    """Classify counselor utterance into MISC 17-class."""

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
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.lm.device)
        with torch.no_grad():
            out = self.lm.generate(**inputs, max_new_tokens=64, do_sample=False)
        raw = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        # Strip Qwen3 <think> blocks
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        raw = re.sub(r'</?think>', '', raw).strip()
        return _parse_label(raw)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def load_profiles(path: str, test_num: int) -> List[Dict]:
    profiles = []
    seen = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            idx = p.get("idx", len(profiles))
            if idx not in seen:
                seen.add(idx)
                profiles.append(p)
            if test_num > 0 and len(profiles) >= test_num:
                break
    return profiles


def run_eval(args):
    import torch
    import torch.backends.cudnn as cudnn
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Ensure CUDA determinism
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    logger = get_logger("eval_simple_sim")

    # Load model
    logger.info("Loading model %s ...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    lm = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    lm.eval()

    classifier = MISCClassifier(tokenizer, lm)

    # Populate shared counselor options — must match all attrs
    # used by PUMACounselor, Generator, Personas, and Planner.
    _defaults = dict(
        max_input_length=4096, max_output_length=1024,
        persona_json_retry_max=2, cami_preparation_calibration=True,
        state_label_space="cami_stage", persona_state_space="cami_stage",
        state_predict_from_probs=True, observation_focused_state=True,
        init_utterances=2,
        usr_name="Client", agent_name="Counselor", log_step=5,
        relevance_memory_number=1, context_memory_number=12, dist_thres=1.5,
        memory_cache=None, ori_mem_query=False, memory_session_turn_gap=0,
        max_agent_personas=0, max_user_personas=0,
        therapist_action_granularity="misc17",
        persona_single_step=False, persona_history_max=6,
        persona_short_term_window_turns=4, persona_long_term_summary_turns=12,
        persona_therapist_action="",
        disable_planner=False, fep_efe_action=True,
        prior_fusion_mode="none", fep_soft_fusion_beta=0.35,
        planner_no_wm_input=False,
        efe_recency_lambda=0.4,
        sampling=False, sampling_step=10,
        sampling_path="logs/sampling", sampling_file_name="train.json",
        gpus="0",
        dataset="cami",
        data_path=os.path.dirname(args.profiles_misc_path),
        data_name="profiles.jsonl",
        generation_out=False,
        next_state_eval_mode="auto",
        annomi_init_turns=8, cami_warmup_ratio=0.5, cami_min_eval_turns=3,
        obs_state_head_window_turns=0,
        fep_lambda_e=0.4, fep_lambda_p=0.6,
        miti_behavior_scores=False,
        results_file="", log_file="",
        sample_ids="",
    )
    for k, v in _defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    args.test_num_main = args.test_num

    from counselor import PUMACounselor, build_seed_history

    # Load profiles & build population table
    profiles = load_profiles(args.profiles_misc_path, args.test_num)
    all_profiles = load_all_profiles(args.profiles_misc_path)
    pop_table = _build_population_table(args.profiles_misc_path)
    pop_action_dist = build_population_action_dist(all_profiles)
    logger.info("Loaded %d profiles (%d total), %d population cells",
                len(profiles), len(all_profiles), len(pop_table))

    modes = ["puma"]
    logger.info("Modes: %s", modes)

    all_results = []

    for mode_idx, mode in enumerate(modes):
        logger.info("=== Mode: %s ===", mode)
        mode_results = []

        for pi, profile in enumerate(profiles):
            run_seed = args.seed + pi * 101 + mode_idx * 10007
            rng = random.Random(run_seed)
            random.seed(run_seed)
            np.random.seed(run_seed % (2**31))
            torch.manual_seed(run_seed)
            torch.cuda.manual_seed_all(run_seed)
            sim = SimpleSimulator.from_profile(
                profile, pop_table, pop_action_dist, tokenizer, lm,
            )
            seed_history = build_seed_history(profile, getattr(args, "init_utterances", 2))

            # Create doctor
            doctor = PUMACounselor(args, logger, pi, seed_history, model=lm, tokenizer=tokenizer)

            # Initialize conversation
            patient_utt = seed_history[-1].replace("Client: ", "") if seed_history else "Hello."
            conversation_context = list(seed_history)

            pc_thresh = getattr(sim, 'threshold_pc_to_c', None) or getattr(sim, 'trigger_coverage_for_pc_to_c', 0.3)
            logger.info("  [%s] profile %s (PC→C=%.2f, C→P=%.2f): ",
                         mode, profile.get("idx", pi),
                         pc_thresh, sim.threshold_c_to_p)

            action_counts = Counter()

            for turn in range(args.max_turns):
                # Doctor generates response
                doctor_utt = doctor.reply(patient_utt)

                # Classify utterance into MISC 17-class
                context_str = "\n".join(conversation_context[-4:])
                misc_action = classifier.classify(doctor_utt, context_str, sim.stage)
                action_counts[misc_action] += 1

                # Simulator step
                patient_action, new_stage, done, info = sim.step(misc_action, doctor_utt, rng)

                # Patient response from LLM
                patient_utt = info["patient_response"]
                conversation_context.append(f"Counselor: {doctor_utt}")
                conversation_context.append(f"Client: {patient_utt}")

                # Feed patient response back to doctor
                doctor.receive(f"Client: {patient_utt}")

                if done:
                    break

            summary = sim.summary()
            summary["action_distribution"] = dict(action_counts)
            summary["explicit_actions"] = None  # could track if needed
            mode_results.append(summary)

            logger.info("    lift=%d stage=%s turns=%d actions=%s",
                         summary["stage_lift"], summary["final_stage"],
                         summary["total_turns"],
                         " ".join(f"{a}={c}" for a, c in action_counts.most_common(3)))

            # Release per-profile state; the shared model stays loaded
            del doctor
            import gc; gc.collect()
            torch.cuda.empty_cache()

        # Aggregate
        lifts = [r["stage_lift"] for r in mode_results]
        preps = [int(r["reached_preparation"]) for r in mode_results]
        turns_list = [r["total_turns"] for r in mode_results]
        coverages = [r.get("trigger_coverage", 0) for r in mode_results]

        # Aggregate action distribution
        total_action_counts = Counter()
        for r in mode_results:
            total_action_counts.update(r.get("action_distribution", {}))
        total_actions = sum(total_action_counts.values())

        agg = {
            "mode": mode,
            "n_profiles": len(mode_results),
            "avg_stage_lift": sum(lifts) / len(lifts) if lifts else 0,
            "preparation_rate": sum(preps) / len(preps) if preps else 0,
            "avg_turns": sum(turns_list) / len(turns_list) if turns_list else 0,
            "avg_trigger_coverage": sum(coverages) / len(coverages) if coverages else 0,
            "action_distribution": {
                k: round(v / total_actions, 3) for k, v in total_action_counts.most_common()
            } if total_actions > 0 else {},
        }
        logger.info("  [%s] SUMMARY: lift=%.3f prep=%.1f%% turns=%.1f coverage=%.1f%%",
                     mode, agg["avg_stage_lift"], agg["preparation_rate"] * 100,
                     agg["avg_turns"], agg["avg_trigger_coverage"] * 100)

        all_results.append({
            "mode": mode,
            "aggregate": agg,
            "per_profile": mode_results,
        })

    # Save
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    payload = {
        "config": {
            "model": args.model,
            "test_num": args.test_num,
            "max_turns": args.max_turns,
            "seed": args.seed,
            "modes": modes,
            "w_change": 1.0,
            "w_neutral": 0.15,
            "w_sustain": -1.0,
        },
        "results": all_results,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Saved to %s", args.output_json)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Mode':<20s} {'Lift':>6} {'Prep%':>6} {'Turns':>6} {'TrigCov':>7} | Top Actions")
    print("-" * 90)
    for r in all_results:
        a = r["aggregate"]
        top3 = " ".join(f"{k}={v:.0%}" for k, v in list(a["action_distribution"].items())[:3])
        print(f"{a['mode']:<20s} {a['avg_stage_lift']:>6.2f} {a['preparation_rate']:>5.0%} {a['avg_turns']:>6.1f} {a['avg_trigger_coverage']:>6.0%} | {top3}")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles_misc_path", type=str,
                        default="data/cami/profiles_misc.jsonl")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--test_num", type=int, default=5)
    parser.add_argument("--max_turns", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default="logs/results/simple_sim.json")
    # FEP ablation flags (override defaults in _defaults dict)
    parser.add_argument("--disable_planner", action="store_true", default=False,
                        help="Disable TransitionPlanner in FEP doctor.")
    parser.add_argument("--observation_focused_state", type=str, default="True",
                        help="Enable observation-focused state head (True/False).")
    parser.add_argument("--fep_efe_action", type=str, default="True",
                        help="Enable EFE action selection (True/False).")
    parser.add_argument("--cami_preparation_calibration", type=str, default="True",
                        help="Enable CAMI preparation calibration (True/False).")
    parser.add_argument("--fep_lambda_e", type=float, default=0.4,
                        help="Epistemic weight in EFE action selection.")
    parser.add_argument("--fep_lambda_p", type=float, default=0.6,
                        help="Pragmatic weight in EFE action selection.")
    parser.add_argument("--fixed_state", type=str, default=None,
                        help="Skip prior inference; use fixed state (e.g. 'contemplation'). "
                             "World model and EFE planning still run with this fixed state.")
    parser.add_argument("--prior_fusion_mode", type=str, default="none",
                        choices=["none", "soft_fusion"],
                        help="Dynamic FEPCounselor: enable soft-fusion of obs head with planner prior. "
                             "'none' (default) = obs head argmax only (original behavior). "
                             "'soft_fusion' = blend lp_label with planner prior (mirrors static MSC.py).")
    parser.add_argument("--fep_soft_fusion_beta", type=float, default=0.35,
                        help="Global weight of planner prior in soft_fusion mode.")
    parser.add_argument("--planner_no_wm_input", type=str, default="False",
                        help="Ablation: feed empty world-model strings to planner + EFE selector "
                             "(True/False). Tests whether WM context informs planning.")
    parser.add_argument("--efe_recency_lambda", type=float, default=0.4,
                        help="Coefficient of recency-aware exploration penalty in EFE: "
                             "g_explore += lambda * k where k = action count in last 8 turns "
                             "(capped at 3). Default 0.4. Set 0 to disable.")
    args = parser.parse_args()
    # Convert string booleans
    for attr in ("observation_focused_state", "fep_efe_action", "cami_preparation_calibration",
                 "planner_no_wm_input"):
        v = getattr(args, attr, "True")
        setattr(args, attr, str(v).strip().lower() in ("true", "1", "yes"))
    run_eval(args)
