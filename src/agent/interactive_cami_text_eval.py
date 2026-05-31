from __future__ import annotations

import path_setup  # noqa: F401

import argparse
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tools import build_llm_client
from tools.defaults import DEFAULT_LLM_CLIENT, DEFAULT_QWEN_HF_MODEL
from tools.therapist_action import normalize_therapist_action

from Module.EventMemory import EventMemory
from Module.Generator import Generator
from Module.Personas import Personas
from Module.Planner import TransitionPlanner
from baseline.mi_coaching.CAMI.agents.client import Client as CAMIClient
from baseline.mi_coaching.CAMI.agents.counselor import CAMI as CAMICounselor
from baseline.mi_coaching.CAMI.agents.counselor_baselines import (
    BaseCounselor,
    DIIRCounselor,
    CoSCounselor,
    CAMITECounselor,
)
from baseline.mi_coaching.CAMI.agents.env import heuristic_moderator

# ---------------------------------------------------------------------------
# Per-profile personalized weights loader
# ---------------------------------------------------------------------------

def _load_personalized_weights(data_path: str) -> List[Dict]:
    """Load per-profile personalized readiness weights computed from CAMI data.
    data_path can be either the directory containing profiles.jsonl or the jsonl file itself.
    """
    # If data_path is a file, use its directory; if it's already a dir, use as-is
    if os.path.isfile(data_path):
        search_dir = os.path.dirname(data_path)
    else:
        search_dir = data_path
    weights_path = os.path.join(search_dir, "profile_personalized_weights.json")
    if not os.path.isfile(weights_path):
        return []
    try:
        with open(weights_path) as f:
            data = json.load(f)
        return data.get("profiles", [])
    except Exception:
        return []


STAGE_TO_SCORE = {
    "precontemplation": 0,
    "contemplation": 1,
    "preparation": 2,
}
THERAPIST_ACTIONS_COARSE: Tuple[str, ...] = ("question", "reflection", "therapist_input", "other")


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("yes", "true", "t", "1", "y"):
        return True
    if s in ("no", "false", "f", "0", "n", ""):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def _token_set(text: Optional[str]) -> set[str]:
    if not text:
        return set()
    return {
        tok
        for tok in "".join(ch.lower() if ch.isalnum() else " " for ch in str(text)).split()
        if tok
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(len(a & b) / len(a | b))


def _canonical_action_label(action: Optional[str]) -> str:
    v = normalize_therapist_action(
        action,
        granularity="coarse",
        vocab=THERAPIST_ACTIONS_COARSE,
        fallback_other=True,
    )
    return v or "other"


def _observation_rule_sentence(rule: Dict[str, Any]) -> str:
    state = normalize_stage(rule.get("latent_state")) or "unknown_state"
    reply_shape = str(rule.get("reply_shape", "") or "").strip()
    if not reply_shape:
        reply_shape = "no explicit observation pattern"
    return f"When the client is in {state}, their observable reply pattern is: {reply_shape}."


def _transition_rule_sentence(rule: Dict[str, Any]) -> str:
    s_from = normalize_stage(rule.get("s_from")) or "unknown_state"
    s_to = normalize_stage(rule.get("s_to")) or "unknown_state"
    action = _canonical_action_label(rule.get("action_a"))
    return f"If the client is in {s_from} and the counselor uses {action}, the next state tends to be {s_to}."


class SemanticRuleEncoder:
    """
    Optional sentence-level semantic encoder for rule-overlap evaluation.
    Falls back gracefully when the model cannot be loaded.
    """

    def __init__(self, model_name: str, logger=None):
        self.model_name = str(model_name or "").strip()
        self.logger = logger
        self.available = False
        self._model = None
        self._cache: Dict[str, np.ndarray] = {}

        if not self.model_name:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_name)
            self.available = True
        except Exception as exc:  # pragma: no cover
            if self.logger:
                self.logger.warning(
                    "[semantic_overlap] failed to load model=%s; disable semantic overlap. err=%s",
                    self.model_name,
                    exc,
                )

    def _encode(self, text: str) -> Optional[np.ndarray]:
        if not self.available or self._model is None:
            return None
        key = str(text or "")
        if key in self._cache:
            return self._cache[key]
        vec = self._model.encode(key, normalize_embeddings=True)
        arr = np.asarray(vec, dtype=np.float32)
        self._cache[key] = arr
        return arr

    def cosine(self, text_a: str, text_b: str) -> Optional[float]:
        va = self._encode(text_a)
        vb = self._encode(text_b)
        if va is None or vb is None:
            return None
        return float(np.dot(va, vb))


def observation_model_overlap(sim_rules: List[Dict[str, Any]], sys_rules: List[Dict[str, Any]]) -> Optional[float]:
    if not sim_rules:
        return None
    scores = []
    for sim in sim_rules:
        state = normalize_stage(sim.get("latent_state"))
        sim_tokens = _token_set(sim.get("reply_shape"))
        candidates = [
            _jaccard(sim_tokens, _token_set(sys.get("reply_shape")))
            for sys in sys_rules
            if normalize_stage(sys.get("latent_state")) == state
        ]
        scores.append(max(candidates) if candidates else 0.0)
    return float(sum(scores) / len(scores)) if scores else None


def transition_model_overlap(sim_rules: List[Dict[str, Any]], sys_rules: List[Dict[str, Any]]) -> Optional[float]:
    if not sim_rules:
        return None
    sim_edges = {
        (
            normalize_stage(r.get("s_from")),
            _canonical_action_label(r.get("action_a")),
            normalize_stage(r.get("s_to")),
        )
        for r in sim_rules
    }
    sys_edges = {
        (
            normalize_stage(r.get("s_from")),
            _canonical_action_label(r.get("action_a")),
            normalize_stage(r.get("s_to")),
        )
        for r in sys_rules
    }
    sim_edges.discard((None, "", None))
    sys_edges.discard((None, "", None))
    if not sim_edges:
        return None
    return float(len(sim_edges & sys_edges) / len(sim_edges | sys_edges)) if sys_edges else 0.0


def observation_model_overlap_normalized(
    sim_rules: List[Dict[str, Any]], sys_rules: List[Dict[str, Any]]
) -> Optional[float]:
    """Normalized observation overlap: compare reply_shape with stopword removal and stemming."""
    _STOPWORDS = frozenset(
        "a an the is are was were be been being have has had do does did will would "
        "shall should may might can could of in on at to for with by from and or but "
        "not no nor so yet as if it its they them their he she his her this that these "
        "those i me my we our you your".split()
    )

    def _norm_tokens(text: Optional[str]) -> set[str]:
        if not text:
            return set()
        tokens = _token_set(text)
        return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}

    if not sim_rules:
        return None
    scores = []
    for sim in sim_rules:
        state = normalize_stage(sim.get("latent_state"))
        sim_tokens = _norm_tokens(sim.get("reply_shape"))
        candidates = [
            _jaccard(sim_tokens, _norm_tokens(sys.get("reply_shape")))
            for sys in sys_rules
            if normalize_stage(sys.get("latent_state")) == state
        ]
        scores.append(max(candidates) if candidates else 0.0)
    return float(sum(scores) / len(scores)) if scores else None


def transition_structural_consistency(
    sim_rules: List[Dict[str, Any]], sys_rules: List[Dict[str, Any]]
) -> Optional[float]:
    """Structural consistency: (s_from, s_to) edge overlap ignoring action label."""
    if not sim_rules:
        return None
    sim_edges = {
        (normalize_stage(r.get("s_from")), normalize_stage(r.get("s_to")))
        for r in sim_rules
    }
    sys_edges = {
        (normalize_stage(r.get("s_from")), normalize_stage(r.get("s_to")))
        for r in sys_rules
    }
    sim_edges.discard((None, None))
    sys_edges.discard((None, None))
    if not sim_edges:
        return None
    return float(len(sim_edges & sys_edges) / len(sim_edges | sys_edges)) if sys_edges else 0.0


def observation_model_semantic_alignment(
    sim_rules: List[Dict[str, Any]],
    sys_rules: List[Dict[str, Any]],
    encoder: Optional[SemanticRuleEncoder],
) -> Tuple[Optional[float], List[Dict[str, Any]]]:
    if not sim_rules:
        return None, []
    if encoder is None or not encoder.available:
        return None, []

    alignments: List[Dict[str, Any]] = []
    scores: List[float] = []
    for sim in sim_rules:
        state = normalize_stage(sim.get("latent_state"))
        sim_sent = _observation_rule_sentence(sim)
        candidates = [sys for sys in sys_rules if normalize_stage(sys.get("latent_state")) == state]
        if not candidates:
            alignments.append(
                {
                    "sim_rule_id": sim.get("rule_id"),
                    "sim_sentence": sim_sent,
                    "best_sys_rule_id": None,
                    "best_sys_sentence": None,
                    "semantic_similarity": 0.0,
                }
            )
            scores.append(0.0)
            continue

        best_score = -1.0
        best_sys = None
        best_sys_sent = ""
        for sys_rule in candidates:
            sys_sent = _observation_rule_sentence(sys_rule)
            sim_score = encoder.cosine(sim_sent, sys_sent)
            s = float(sim_score) if sim_score is not None else 0.0
            if s > best_score:
                best_score = s
                best_sys = sys_rule
                best_sys_sent = sys_sent
        scores.append(best_score)
        alignments.append(
            {
                "sim_rule_id": sim.get("rule_id"),
                "sim_sentence": sim_sent,
                "best_sys_rule_id": (best_sys or {}).get("rule_id"),
                "best_sys_sentence": best_sys_sent,
                "semantic_similarity": best_score,
            }
        )
    avg = float(sum(scores) / len(scores)) if scores else None
    return avg, alignments


def transition_model_semantic_alignment(
    sim_rules: List[Dict[str, Any]],
    sys_rules: List[Dict[str, Any]],
    encoder: Optional[SemanticRuleEncoder],
) -> Tuple[Optional[float], List[Dict[str, Any]]]:
    if not sim_rules:
        return None, []
    if encoder is None or not encoder.available:
        return None, []

    alignments: List[Dict[str, Any]] = []
    scores: List[float] = []
    for sim in sim_rules:
        s_from = normalize_stage(sim.get("s_from"))
        sim_sent = _transition_rule_sentence(sim)
        candidates = [sys for sys in sys_rules if normalize_stage(sys.get("s_from")) == s_from]
        if not candidates:
            candidates = list(sys_rules)
        if not candidates:
            alignments.append(
                {
                    "sim_rule_id": sim.get("rule_id"),
                    "sim_sentence": sim_sent,
                    "best_sys_rule_id": None,
                    "best_sys_sentence": None,
                    "semantic_similarity": 0.0,
                }
            )
            scores.append(0.0)
            continue

        best_score = -1.0
        best_sys = None
        best_sys_sent = ""
        for sys_rule in candidates:
            sys_sent = _transition_rule_sentence(sys_rule)
            sim_score = encoder.cosine(sim_sent, sys_sent)
            s = float(sim_score) if sim_score is not None else 0.0
            if s > best_score:
                best_score = s
                best_sys = sys_rule
                best_sys_sent = sys_sent
        scores.append(best_score)
        alignments.append(
            {
                "sim_rule_id": sim.get("rule_id"),
                "sim_sentence": sim_sent,
                "best_sys_rule_id": (best_sys or {}).get("rule_id"),
                "best_sys_sentence": best_sys_sent,
                "semantic_similarity": best_score,
            }
        )
    avg = float(sum(scores) / len(scores)) if scores else None
    return avg, alignments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive doctor evaluation in the CAMI client environment (text-only first)."
    )
    parser.add_argument(
        "--profile_path",
        type=str,
        default="../baseline/mi_coaching/CAMI/annotations/profiles.jsonl",
        help="CAMI profiles.jsonl path.",
    )
    parser.add_argument(
        "--wikipedia_dir",
        type=str,
        default="../baseline/mi_coaching/CAMI/wikipedias",
        help="CAMI client simulator Wikipedia directory.",
    )
    parser.add_argument(
        "--retriever_path",
        type=str,
        default="BAAI/bge-reranker-v2-m3",
        help="Retriever used by the native CAMI client simulator.",
    )
    parser.add_argument(
        "--doctor_modes",
        type=str,
        default="fep,cami_text",
        help=(
            "Comma-separated doctor modes to run. Supported: "
            "fep, cami_text, base_counselor, diir, cos, cami_te, cami_full."
        ),
    )
    parser.add_argument(
        "--annomi_jsonl_path",
        type=str,
        default="",
        help="Path to AnnoMI transcripts JSONL — used by DIIR for demonstration retrieval.",
    )
    parser.add_argument("--test_num", type=int, default=5, help="How many profiles to evaluate.")
    parser.add_argument("--max_turns", type=int, default=12, help="Maximum counselor turns per run.")
    parser.add_argument(
        "--client_simulator_mode",
        choices=["profile_rules", "legacy", "dynamic"],
        default="profile_rules",
        help="Which CAMI patient simulator implementation to use. dynamic = profile-adaptive closures from dynamic_simulator.py.",
    )
    parser.add_argument(
        "--init_utterances",
        type=int,
        default=2,
        help="How many real profile utterances to use as warm-start context. Must end on a client turn.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument(
        "--output_json",
        type=str,
        default="logs/results/interactive_cami_text_eval.json",
        help="Where to save aggregated results.",
    )
    parser.add_argument(
        "--semantic_overlap",
        type=_str2bool,
        default=True,
        help="Whether to compute rule-level semantic overlap based on natural-language rule descriptions.",
    )
    parser.add_argument(
        "--semantic_model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence embedding model for semantic rule overlap.",
    )

    # Shared LLM args
    parser.add_argument(
        "--client",
        choices=["chatgpt", "chatglm", "qwen_hf", "qwen_local"],
        default=DEFAULT_LLM_CLIENT,
    )
    parser.add_argument("--model", type=str, default=DEFAULT_QWEN_HF_MODEL)
    parser.add_argument("--openai_base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", "empty"))
    parser.add_argument("--max_input_length", type=int, default=8192)
    parser.add_argument("--max_output_length", type=int, default=512)
    parser.add_argument("--gpus", type=str, default="0")

    # FEP module args
    parser.add_argument("--summary_model", type=str, default="default")
    parser.add_argument("--persona_model", type=str, default="default")
    parser.add_argument("--generation_model", type=str, default="default")
    parser.add_argument("--planner_model", type=str, default="default")
    parser.add_argument("--usr_name", type=str, default="Client")
    parser.add_argument("--agent_name", type=str, default="Counselor")
    parser.add_argument("--relevance_memory_number", type=int, default=1)
    parser.add_argument("--context_memory_number", type=int, default=12)
    parser.add_argument("--dist_thres", type=float, default=1.5)
    parser.add_argument("--memory_cache", default=None)
    parser.add_argument("--ori_mem_query", action="store_true", default=False)
    parser.add_argument("--memory_session_turn_gap", type=int, default=0)
    parser.add_argument("--max_agent_personas", type=int, default=0)
    parser.add_argument("--max_user_personas", type=int, default=0)
    parser.add_argument("--therapist_action_granularity", choices=["coarse", "fine", "mi12"], default="coarse")
    parser.add_argument("--persona_single_step", action="store_true", default=False)
    parser.add_argument("--persona_history_max", type=int, default=6)
    parser.add_argument("--persona_short_term_window_turns", type=int, default=4)
    parser.add_argument("--persona_long_term_summary_turns", type=int, default=12)
    parser.add_argument("--persona_therapist_action", type=str, default="")
    parser.add_argument("--persona_json_retry_max", type=int, default=2)
    parser.add_argument("--state_label_space", choices=["mi_talk", "cami_stage"], default="cami_stage")
    parser.add_argument("--persona_state_space", choices=["mi_talk", "cami_stage"], default="cami_stage")
    parser.add_argument("--disable_planner", action="store_true", default=False)
    parser.add_argument("--cami_preparation_calibration", type=_str2bool, default=True)
    parser.add_argument("--state_predict_from_probs", type=_str2bool, default=True,
                        help="Use q_turn_state_probs argmax for state prediction (True) or text extraction (False/legacy).")
    parser.add_argument("--observation_focused_state", type=_str2bool, default=False,
                        help="Add focused observation LLM call for state classification with Bayesian transition prior.")
    parser.add_argument("--obs_state_head_window_turns", type=int, default=0,
                        help="Raw dialogue window for obs state head. 0 = all available history.")
    parser.add_argument("--sampling", action="store_true", default=False)
    parser.add_argument("--sampling_step", type=int, default=10)
    parser.add_argument("--sampling_path", type=str, default="logs/sampling")
    parser.add_argument("--sampling_file_name", type=str, default="train.json")
    return parser


def get_logger(filename: Optional[str] = None):
    logger = logging.getLogger("interactive_cami_text_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    )
    logger.addHandler(stream_handler)

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
        logger.addHandler(handler)
    return logger


def load_profiles(path: str, test_num: int) -> List[Dict]:
    profiles: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            profiles.append(json.loads(line))
            if test_num > 0 and len(profiles) >= test_num:
                break
    return profiles


def normalize_stage(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower()
    return s if s in STAGE_TO_SCORE else None


def _extract_cami_stage_from_text_dyn(text: Optional[str]) -> Optional[str]:
    """Lightweight planner-output → cami_stage extractor for dynamic FEPCounselor.

    Mirrors MSC._extract_cami_stage_from_text but avoids cross-module import to keep
    interactive_cami_text_eval self-contained.
    """
    if not text:
        return None
    # JSON-style: "client_talk_type": "preparation" or "client_stage": "..."
    m = re.search(r'"(?:client_talk_type|client_stage)"\s*:\s*"([a-zA-Z_]+)"', text)
    if m:
        return normalize_stage(m.group(1))
    # Bullet-style: "- client_talk_type: preparation"
    m = re.search(r"-\s*(?:client_talk_type|client_stage)\s*:\s*([a-zA-Z_]+)", text)
    if m:
        return normalize_stage(m.group(1))
    # Bare key=value
    m = re.search(r"(?:client_talk_type|client_stage)\s*[:=]\s*([a-zA-Z_]+)", text)
    if m:
        return normalize_stage(m.group(1))
    return None


def strip_prefixed_response(text: str, prefix: str) -> str:
    if not text:
        return ""
    s = text.strip().replace("\n", " ")
    if prefix in s:
        s = s.split(prefix, 1)[-1].strip()
    return s.strip().strip('"').strip("'")


def ensure_prefix(text: str, prefix: str) -> str:
    s = strip_prefixed_response(text, prefix)
    return f"{prefix} {s}".strip()


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TurnResult:
    doctor_utterance: str
    patient_utterance: str
    state_before: Optional[str]
    state_after: Optional[str]
    predicted_current_state: Optional[str]
    therapist_action: Optional[str] = None


def build_seed_history(profile: Dict, init_utterances: int) -> List[str]:
    speakers = profile.get("speakers", [])
    utterances = profile.get("utterances", [])
    if not speakers or not utterances:
        return [
            "Counselor: Hello. How are you?",
            "Client: I am good. What about you?",
        ]

    n = min(init_utterances, len(speakers), len(utterances))
    if n <= 0:
        n = min(2, len(speakers), len(utterances))

    if n % 2 != 0:
        n -= 1
    if n <= 0:
        n = min(2, len(speakers), len(utterances))

    history: List[str] = []
    for speaker, utterance in zip(speakers[:n], utterances[:n]):
        role = "Client" if speaker == "client" else "Counselor"
        history.append(f"{role}: {utterance}")

    if not history or not history[-1].startswith("Client:"):
        return [
            "Counselor: Hello. How are you?",
            "Client: I am good. What about you?",
        ]
    return history


class CAMIEnvAdapter:
    def __init__(self, args, profile: Dict, personalized_weights: Optional[Dict] = None):
        self.args = args
        self.profile = profile
        self.turn_count = 0
        self.seed_history = build_seed_history(profile, getattr(args, "init_utterances", 2))
        self.client = CAMIClient(
            goal=profile["topic"],
            behavior=profile["Behavior"],
            reference=self._build_reference(profile),
            personas=profile.get("Personas", []),
            initial_stage=profile.get("states", ["Precontemplation"])[0],
            final_stage=profile.get("states", ["Preparation"])[-1],
            motivation=profile.get("Motivation", []),
            beliefs=profile.get("Beliefs", []),
            plans=profile.get("Acceptable Plans", []),
            speakers=profile.get("speakers", []),
            utterances=profile.get("utterances", []),
            actions=profile.get("actions", []),
            states=profile.get("states", []),
            talk_types=profile.get("talk_types", []),
            simulator_mode=args.client_simulator_mode,
            receptivity=self._mean(profile.get("suggestibilities", []), default=3.0),
            model=args.model,
            wikipedia_dir=args.wikipedia_dir,
            retriever_path=args.retriever_path,
            personalized_weights=personalized_weights or {},
        )
        self.client.set_context(self.seed_history)
        self.conversation = list(self.seed_history)

    @staticmethod
    def _mean(values: List[float], default: float) -> float:
        return float(sum(values) / len(values)) if values else default

    @staticmethod
    def _build_reference(profile: Dict) -> str:
        parts = []
        for speaker, utterance in zip(profile.get("speakers", []), profile.get("utterances", [])):
            role = "Client" if speaker == "client" else "Counselor"
            parts.append(f"{role}: {utterance}")
        return "\n".join(parts)

    def current_patient_utterance(self) -> str:
        return strip_prefixed_response(self.conversation[-1], "Client:")

    def snapshot(self) -> Dict[str, object]:
        return {
            "state": normalize_stage(getattr(self.client, "state", None)),
            "engagement": getattr(self.client, "engagement", None),
            "error_topic_count": getattr(self.client, "error_topic_count", None),
            "turn_count": self.turn_count,
        }

    def step(self, doctor_utterance: str, explicit_therapist_action: str | None = None):
        doctor_line = ensure_prefix(doctor_utterance, "Counselor:")
        self.client.receive(doctor_line)
        self.conversation.append(doctor_line)
        self.turn_count += 1

        raw = self.client.reply(explicit_therapist_action=explicit_therapist_action)
        patient_line = self.client.context[-1]
        self.conversation.append(patient_line)

        done = False
        apply_heuristic_stop = True
        if (
            getattr(self.args, "client_simulator_mode", "legacy") in ("profile_rules", "dynamic")
            and self.turn_count < 4
        ):
            apply_heuristic_stop = False
        if (
            apply_heuristic_stop
            and len(self.conversation) >= 4
            and heuristic_moderator(self.conversation)
        ):
            done = True
        if self.turn_count >= self.args.max_turns:
            done = True
        if "You are motivated because" in raw:
            done = True
        if "suggest further discussion be deferred to a later time" in raw:
            done = True

        return strip_prefixed_response(patient_line, "Client:"), self.snapshot(), done


class TextOnlyCAMICounselor:
    def __init__(self, goal: str, behavior: str, model: str, seed_history: List[str], logger=None):
        self.core = CAMICounselor(goal=goal, behavior=behavior, model=model)
        self.goal = goal
        self.behavior = behavior
        self.logger = logger
        self.root_topics = ["Health", "Interpersonal Relationships", "Economy", "Law", "Education"]
        self._profile_topic_prior = self._build_profile_topic_prior(goal, behavior)
        self._last_doctor_response = None
        self.core.conversation = list(seed_history)
        self.core.messages = [self.core.messages[0]]
        for line in seed_history:
            role = "assistant" if line.startswith("Counselor:") else "user"
            self.core.messages.append({"role": role, "content": line})
        self.last_current_state = normalize_stage(self.core.infer_state())
        self._last_action: Optional[str] = None

    def _build_profile_topic_prior(self, goal: str, behavior: str) -> Dict[str, int]:
        text = f"{goal} {behavior}".lower()
        priors = {topic: 0 for topic in self.root_topics}
        if any(tok in text for tok in ("smok", "weed", "drug", "drink", "alcohol", "medicine", "medication", "activity")):
            priors["Health"] += 3
        if any(tok in text for tok in ("wife", "husband", "family", "parent", "child", "relationship")):
            priors["Interpersonal Relationships"] += 2
        if any(tok in text for tok in ("job", "work", "boss", "money", "salary", "bills")):
            priors["Economy"] += 2
        if any(tok in text for tok in ("arrest", "legal", "law", "court", "custody")):
            priors["Law"] += 2
        if any(tok in text for tok in ("school", "study", "class", "exam", "student")):
            priors["Education"] += 2
        priors["Health"] += 1
        return priors

    def _topic_keyword_scores(self, text: str) -> Dict[str, float]:
        text_l = (text or "").lower()
        keyword_map = {
            "Health": [
                "health",
                "sleep",
                "stress",
                "relax",
                "groggy",
                "anxiety",
                "depress",
                "body",
                "energy",
                "drink",
                "drinking",
                "smok",
                "weed",
                "drug",
                "substance",
            ],
            "Interpersonal Relationships": [
                "wife",
                "husband",
                "family",
                "friend",
                "relationship",
                "son",
                "daughter",
                "child",
                "kids",
                "parents",
                "people",
                "tension",
                "together",
                "social",
            ],
            "Economy": [
                "job",
                "work",
                "boss",
                "bills",
                "money",
                "house",
                "rent",
                "pay",
                "career",
                "salary",
            ],
            "Law": [
                "police",
                "arrest",
                "jail",
                "court",
                "lawyer",
                "legal",
                "custody",
                "ticket",
                "probation",
                "ua",
            ],
            "Education": [
                "school",
                "class",
                "exam",
                "study",
                "college",
                "student",
                "scholarship",
            ],
        }
        scores = {topic: float(self._profile_topic_prior.get(topic, 0)) for topic in self.root_topics}
        for topic, keywords in keyword_map.items():
            scores[topic] += sum(text_l.count(keyword) for keyword in keywords)
        return scores

    def _rank_root_topics(self) -> List[str]:
        recent_lines = self.core.conversation[-6:]
        recent_text = "\n".join(recent_lines)
        scores = self._topic_keyword_scores(recent_text)
        ranked = sorted(self.root_topics, key=lambda topic: (-scores.get(topic, 0.0), self.root_topics.index(topic)))
        return ranked or ["Health"]

    def _select_topic(self, retry: bool = False) -> tuple[str, str, str]:
        ranked = self._rank_root_topics()
        current = self.core.topic_stack[-1] if self.core.topic_stack else None
        if retry and len(ranked) > 1:
            candidates = [topic for topic in ranked if topic != current]
            topic = candidates[0] if candidates else ranked[0]
        else:
            topic = ranked[0]
        action = "Stay" if current == topic else "Switch"
        explanation = f"Profile-root-topic + recent-context rerank selected {topic} (retry={retry})."
        self.core.topic_stack = [topic]
        self.core.initialized = True
        self.core.explored_topics.append(topic)
        return explanation, action, topic

    def _normalize_counselor_response(self, raw: str) -> str:
        m = re.search(r"(Counselor:\s*.*)", (raw or "").replace("\n", " "))
        response = m.group(1) if m else ensure_prefix((raw or "").replace("*", "").replace("#", ""), "Counselor:")
        return ensure_prefix(response.replace("*", "").replace("#", ""), "Counselor:")

    def _is_repetitive(self, response: str) -> bool:
        if not self._last_doctor_response:
            return False
        prev = strip_prefixed_response(self._last_doctor_response, "Counselor:")
        curr = strip_prefixed_response(response, "Counselor:")
        if not prev or not curr:
            return False
        if curr == prev:
            return True
        return SequenceMatcher(None, prev, curr).ratio() >= 0.92

    def _generate_with_topic(self, state: str, topic: str) -> str:
        strategy_analysis, selected_strategies = self.core.select_strategy(state)
        last_utterance = self.core.messages[-1]["content"]
        response, final_strategy = self.core.generate(last_utterance, topic, state, selected_strategies)
        if final_strategy == "Combined Strategies":
            final_strategy = " + ".join(selected_strategies)
            if len(selected_strategies) >= 2:
                strategy_description = (
                    self.core.strategy2description[selected_strategies[0]]
                    + "\n- "
                    + self.core.strategy2description[selected_strategies[1]]
                )
            elif len(selected_strategies) == 1:
                strategy_description = self.core.strategy2description[selected_strategies[0]]
            else:
                strategy_description = self.core.strategy2description["No Strategy"]
        else:
            strategy_description = self.core.strategy2description[final_strategy]
        # Store the MI strategy for simulator feedback (not the topic action)
        self._last_mi_strategy = final_strategy
        response = self._normalize_counselor_response(response)
        response = self.core.refine(
            self.core.conversation[-5:],
            response,
            strategy_description,
            self.core.topic2description[topic],
        )
        response = self._normalize_counselor_response(response)
        if self.logger:
            self.logger.info(
                "[cami_text] state=%s topic=%s strategies=%s final_strategy=%s",
                state,
                topic,
                selected_strategies,
                final_strategy,
            )
        return response

    def receive(self, response: str):
        self.core.receive(ensure_prefix(response, "Client:"))

    def infer_current_state(self) -> Optional[str]:
        return self.last_current_state

    def last_therapist_action(self) -> Optional[str]:
        return getattr(self, "_last_mi_strategy", None) or self._last_action

    def system_world_model_snapshot(self) -> Optional[Dict[str, object]]:
        return None

    def reply(self) -> str:
        state = self.core.infer_state()
        _, action, topic = self._select_topic(retry=False)
        response = self._generate_with_topic(state, topic)
        if self._is_repetitive(response):
            if self.logger:
                self.logger.info("[cami_text] repetitive reply detected; retrying once with alternate reranked topic.")
            self.core.explored_topics.pop()
            _, action, topic = self._select_topic(retry=True)
            response = self._generate_with_topic(state, topic)
        self.core.messages.append({"role": "assistant", "content": response})
        self.core.conversation.append(response)
        self._last_doctor_response = response
        self._last_action = action
        self.last_current_state = normalize_stage(self.core.infer_state())
        return response


class FEPCounselor:
    def __init__(self, args, logger, sample_id: int, seed_history: List[str]):
        self.args = args
        self.logger = logger
        if args.client in ("chatglm", "qwen_hf"):
            lora_map = {
                "EventSummary": args.summary_model,
                "PersonaExtraction": args.persona_model,
                "ResponseGenerator": args.generation_model,
                "Planner": args.planner_model,
            }
            self.client = build_llm_client(args, logger, lora_map=lora_map)
        else:
            self.client = build_llm_client(args, logger, lora_map=None)

        self.memory_bank = EventMemory(
            self.client,
            sample_id=f"interactive_{sample_id}",
            logger=logger,
            args=args,
            memory_cache=args.memory_cache,
        )
        self.personas = Personas(self.client, logger=logger, args=args)
        self.generator = Generator(self.client, [], sample_id, logger=logger, args=args)
        self.planner = TransitionPlanner(self.client, logger=logger, args=args)
        self.prev_doctor_utt = strip_prefixed_response(seed_history[-2], "Counselor:")
        self.turn_id = 0
        self.last_current_state = None
        self._recent_actions: List[str] = []  # for EFE diversity constraint
        for idx, line in enumerate(seed_history):
            self.memory_bank.short_term_memory.append({"idx": idx, "turn": 0, "dialog": line})

        for idx in range(1, max(1, len(seed_history) - 1)):
            if not seed_history[idx].startswith("Client:"):
                continue
            if idx + 1 == len(seed_history):
                break
            if not seed_history[idx - 1].startswith("Counselor:"):
                continue
            self.personas.traits_update(
                strip_prefixed_response(seed_history[idx], "Client:"),
                strip_prefixed_response(seed_history[idx - 1], "Counselor:"),
                therapist_action=None,
            )

    def _merged_memories(self, items: List[Dict]) -> str:
        if not items:
            return "No relevant Memories."
        return "\n".join(
            f"{item.get('summary') or item.get('dialog') or ''}".strip()
            for item in items
            if (item.get("summary") or item.get("dialog"))
        ) or "No relevant Memories."

    def infer_current_state(self) -> Optional[str]:
        return self.last_current_state

    def last_therapist_action(self) -> Optional[str]:
        return self.personas.last_therapist_action()

    def system_world_model_snapshot(self) -> Dict[str, object]:
        return self.personas.snapshot()

    def receive(self, response: str):
        # FEP updates its user model inside reply(), so receive is a no-op placeholder.
        del response

    # ------------------------------------------------------------------
    # EFE-based action selection
    # ------------------------------------------------------------------
    # Stage ordering for pragmatic value: higher index = more desirable goal state.
    _STAGE_RANK = {
        "precontemplation": 0.0,
        "contemplation": 0.5,
        "preparation": 1.0,
        # fall-through values for mi_talk space
        "change": 1.0,
        "sustain": 0.0,
        "neutral": 0.3,
        "follow_up": 0.5,
    }

    @staticmethod
    def _parse_planner_confidence(text: str) -> float:
        """Extract confidence from planner text: '(model confidence: 0.72)'."""
        m = re.search(r"model confidence:\s*([0-9]+(?:\.[0-9]+)?)", text or "")
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return 0.5

    @staticmethod
    def _parse_planner_next_stage(text: str) -> Optional[str]:
        """Extract client_talk_type from planner roll-forward text."""
        # Pattern: "- client_talk_type: contemplation"
        m = re.search(r"-\s*client_talk_type\s*:\s*(\w+)", text or "")
        if m:
            return m.group(1).strip().lower()
        return None

    def _select_action_efe(self) -> Tuple[str, str]:
        """
        Select therapist action by minimising Expected Free Energy:

            G(a) = E_q(o_{t+1}|a)[ H(q(s_{t+1}|o_{t+1},a)) ]            (exploration)
                 + E_q(s_{t+1},o_{t+1}|a)[ -log p_pref(o_{t+1}) ]       (exploitation)

        Concretely, in a single LLM call we ask the model to roll the user
        world model forward under each candidate action a in A and report:
          - hat_o(a):     most likely client response under q(o_{t+1}|a)
          - p_state(a):   3-class softmax over s_{t+1} ∈ {precontemplation,
                          contemplation, preparation} given (a, hat_o(a)).
                          We use H[p_state] as a Monte-Carlo estimate of the
                          expected posterior entropy (one-sample expectation).
          - p_pref(a):    probability that hat_o(a) is a "preferred" observation
                          under p_pref. Preference is concentrated on
                          {change-talk × preparation-stage}, i.e. the patient
                          shows commitment / explicit next step (MI goal).
          - g_explore(a) = H[p_state(a)]                  (nats, ≥0)
          - g_exploit(a) = -log p_pref(a)                  (nats, ≥0)
          - G(a)         = g_explore + g_exploit
        We then pick a* = argmin_a G(a). No hard diversity rule — repetition
        is naturally penalised because once the posterior over s_{t+1} given a
        familiar (a, hat_o) collapses, H is small but so is the information
        gained on average; the exploration term reflects the *predicted*
        posterior entropy, which stays high only for actions whose outcomes
        are still uncertain under the learned world model.

        Returns (chosen_action, "").
        """
        actions = [a for a in (self.personas.therapist_actions or ()) if a and a != "other"]
        if not actions:
            return "other", ""

        current_stage = (self.last_current_state or "precontemplation").lower()
        # F-dyn ablation: when --planner_no_wm_input, blank the WM context fed to
        # EFE action selector so it must choose action from current_stage + utterance only.
        _no_wm_efe = bool(getattr(self.args, "planner_no_wm_input", False))
        obs_str = "" if _no_wm_efe else self.personas._format_observation_model()
        tr_str = "" if _no_wm_efe else self.personas._format_transition_model()

        # v3: pass a longer recency window so the LLM can reason about
        # information saturation across more than the last 3 turns.
        recent_full = self._recent_actions[-8:] if hasattr(self, '_recent_actions') and self._recent_actions else []
        recent = recent_full[-3:]
        recent_line = ", ".join(recent) if recent else "(none)"
        from collections import Counter as _Counter
        _recent_counts = _Counter(recent_full)
        if _recent_counts:
            recent_counts_line = ", ".join(
                f"{a}={c}" for a, c in sorted(_recent_counts.items(), key=lambda kv: -kv[1])
            )
        else:
            recent_counts_line = "(none used yet)"
        actions_list = "\n".join(f"- {a}" for a in actions)

        sys_prompt = (
            "You are the action-selection module of an active-inference MI counselor. "
            "For each candidate therapist action a, you internally simulate the patient's "
            "world model one step ahead under the learned q(o_{t+1}|a), then report a "
            "posterior over the next stage s_{t+1} and a preference probability for the "
            "predicted observation. Output a single CSV table — nothing else."
        )

        prompt = f"""# Task
For every candidate therapist action a in A, internally imagine the most
likely next patient utterance hat_o(a) under q(o_{{t+1}}|a), then report:

  p_state(a) = q(s_{{t+1}} | hat_o(a), a)   — a 3-class posterior (sums to 1)
                                              over {{precontemplation,
                                                     contemplation,
                                                     preparation}}.
                                              Use values that reflect how
                                              SHARP your belief becomes once
                                              the predicted observation is
                                              taken into account.

  p_pref(a)  = p_pref(hat_o(a))             — probability in (0, 1) that
                                              hat_o(a) is a PREFERRED
                                              observation.

I will compute Expected Free Energy

    G(a) = H[p_state(a)]  +  -ln p_pref(a)

(both terms in NATS) and pick a* = argmin_a G(a). The first term is the
exploration value (expected posterior entropy after observing hat_o); the
second is the exploitation value (negative log preference of the predicted
observation).

# Preference distribution p_pref

p_pref(o) is a **stationary** preference distribution over observations,
specifying what kind of patient response the counselor wants to see in the
end. It does NOT depend on history — it is the goal posterior.
A preferred observation is one that BOTH
  (i) contains change talk (desire / ability / reason / need / commitment /
      taking step), AND
  (ii) is consistent with the patient being in the **preparation** stage
       (a concrete next step, plan, quota, time horizon, help-seeking).
Sustain talk, defensiveness, or pure ambivalence -> low p_pref (≈0.05–0.15).
Vague intent without a step -> partial (≈0.3). Concrete change step -> high
(≈0.8–0.95). Stay strictly in (0, 1) — never 0 or 1.

# Honest p_state estimation — IMPORTANT

The exploration term H[q(s_{{t+1}} | hat_o, a)] measures how much your
posterior over the next stage TIGHTENS once you imagine observing hat_o.
For this to be meaningful, p_state(a) must reflect the GENUINE posterior
update, not a wishful sharpening:

  - If action a has been used REPEATEDLY in recent turns (counts shown
    below), the patient has already responded to that intervention several
    times. Your predicted hat_o(a) is close to a paraphrase of past
    responses — it carries little NEW evidence about s_{{t+1}}. So
    q(s_{{t+1}} | hat_o, a) ≈ q(s_{{t+1}} | a) ≈ the current-stage prior
    (one-hot smoothed by ≈0.1).

      => p_state for a repeated action should NOT be artificially sharp
         just because hat_o is easy to predict. A predictable hat_o gives
         no update — its posterior should be roughly as spread as the
         prior. Honest p_state for a repeated action: nearly uniform
         within the realistic next-stage set, with a slight tilt toward
         the current stage.

  - For a FRESH action (not in the recent window), the actual patient
    response could surprise you. The posterior depends on what hat_o turns
    out to be: if hat_o would clearly indicate a stage transition, p_state
    can be sharper; otherwise it remains spread.

This is the only adjustment you need to make. Do not modify p_pref based
on history — p_pref is stationary.

# Current context

Current patient state s_t: {current_stage}

Recent therapist actions (most recent last): {recent_line}

Recent action usage counts (last 8 turns): {recent_counts_line}

Patient observation model (what you have learned about this patient):
{obs_str[:1200] if obs_str else "(no observations yet — world model is uninformative; expect high entropy)"}

Transition model (action -> next-state effects you have learned):
{tr_str[:1200] if tr_str else "(no transition data yet — expect high entropy)"}

# Candidate actions A
{actions_list}

# Output format — STRICT

Return ONE CSV table with header, one row per candidate action, IN THE
SAME ORDER as listed above. Use `|` (pipe) as the column separator. Do NOT
emit code fences, JSON, prose, or any text before or after the table.

Schema:

action | p_pc | p_c | p_p | p_pref

Where:
  - action: the action name copied verbatim from the list above.
  - p_pc, p_c, p_p: components of p_state for {{precontemplation,
    contemplation, preparation}}. Each in [0, 1]; the three should sum
    to ~1 (I will renormalise).
  - p_pref: probability in (0, 1) that hat_o(a) is preferred.

Example (illustrative — not the answer):

action | p_pc | p_c | p_p | p_pref
Open Question | 0.20 | 0.55 | 0.25 | 0.40
Affirm | 0.15 | 0.45 | 0.40 | 0.55

Now produce the table for this turn."""

        # ---- helpers --------------------------------------------------------
        def _entropy(p: Dict[str, float]) -> float:
            keys = ("precontemplation", "contemplation", "preparation")
            vals = []
            for k in keys:
                v = float(p.get(k, 0.0)) if isinstance(p, dict) else 0.0
                if not np.isfinite(v) or v < 0.0:
                    v = 0.0
                vals.append(v)
            s = sum(vals)
            if s <= 0:
                return float(np.log(3.0))  # uniform fallback
            vals = [v / s for v in vals]
            h = 0.0
            for v in vals:
                if v > 0:
                    h -= v * float(np.log(v))
            return h

        def _clip_pref(x) -> float:
            try:
                v = float(x)
            except (TypeError, ValueError):
                v = 0.5
            if not np.isfinite(v):
                v = 0.5
            return min(max(v, 1e-3), 1.0 - 1e-3)

        def _match_action(name: str) -> Optional[str]:
            n = (name or "").strip().strip('"').strip("'").lower()
            for a in actions:
                if a.lower() == n:
                    return a
            for a in actions:
                if a.lower() in n or n in a.lower():
                    return a
            return None

        chosen: Optional[str] = None
        scores_log: List[Tuple[str, float, float, float]] = []

        try:
            # Pipe-separated CSV is short and robust; ~25 chars per row x 17 rows
            # well under 1500 tokens, leaving slack for any preamble Qwen3 may emit.
            raw = self.client.employ(sys_prompt, prompt, "Planner", max_tokens=1500)
            text = raw.strip()
            # Strip code fences if any
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

            best: Optional[Tuple[float, str]] = None
            seen_actions: set = set()
            for line in text.splitlines():
                line = line.strip()
                if not line or "|" not in line:
                    continue
                cols = [c.strip() for c in line.split("|")]
                if len(cols) < 5:
                    continue
                # Skip header row (case-insensitive)
                if cols[0].lower() == "action":
                    continue
                a_match = _match_action(cols[0])
                if not a_match or a_match in seen_actions:
                    continue
                try:
                    p_pc = float(cols[1]); p_c = float(cols[2]); p_p = float(cols[3])
                    p_pref = _clip_pref(cols[4])
                except (ValueError, TypeError):
                    continue
                # v4: p_pref stays as the strict stationary preference (no
                # history-dependent reweighting). Recency correction goes
                # to the EXPLORATION term instead, which is theoretically
                # the right place: the LLM tends to hallucinate confident
                # p_state for repeated actions, but a repeated action
                # offers ~0 information gain — its true expected posterior
                # should be close to the prior. We compensate by adding a
                # linear penalty lambda*k (k = uses in last 8 turns,
                # capped at 3) to H[p_state]. lambda=0.4 puts the max
                # boost (k=3) at 1.2 nats, comparable to typical
                # -log p_pref scale (0.5–2 nats).
                k = min(_recent_counts.get(a_match, 0), 3)
                lambda_recency = float(getattr(self.args, "efe_recency_lambda", 0.4))
                p_state = {
                    "precontemplation": p_pc,
                    "contemplation": p_c,
                    "preparation": p_p,
                }
                g_explore_raw = _entropy(p_state)
                g_explore = g_explore_raw + lambda_recency * k
                g_exploit = -float(np.log(p_pref))
                g_total = g_explore + g_exploit
                scores_log.append((a_match, g_explore, g_exploit, g_total))
                seen_actions.add(a_match)
                if best is None or g_total < best[0]:
                    best = (g_total, a_match)

            if best is None:
                raise ValueError("no parseable rows in CSV response")
            chosen = best[1]

        except Exception as e:
            if self.logger:
                self.logger.warning("[FEP-EFE] strict-EFE scoring failed: %s — falling back to least-used action", e)
            from collections import Counter
            usage = Counter(self._recent_actions) if hasattr(self, '_recent_actions') else Counter()
            chosen = min(actions, key=lambda a: usage.get(a, 0))

        # Update recent actions
        if not hasattr(self, '_recent_actions'):
            self._recent_actions = []
        self._recent_actions.append(chosen)

        if self.logger:
            if scores_log:
                top = sorted(scores_log, key=lambda r: r[3])[:3]
                top_str = "; ".join(
                    f"{a}: G={g:.3f} (H={h:.3f}, -lnP={p:.3f})" for (a, h, p, g) in top
                )
                self.logger.info("[FEP-EFE] argmin G -> %s | top3: %s", chosen, top_str)
            else:
                self.logger.info("[FEP-EFE] selected action=%s (no scores parsed)", chosen)
        return chosen, ""

    def reply(self, patient_utterance: str) -> str:
        # --fixed_state: skip prior inference entirely, use a fixed state label.
        # World model (p filter) and EFE planning still run with this fixed state.
        fixed_state = getattr(self.args, "fixed_state", None)
        if fixed_state:
            # No q-filter LLM call — just set state to the fixed label
            self.last_current_state = normalize_stage(fixed_state)
            # Still run p-filter to update world model (observation/transition rules)
            # but skip the q-filter's state inference
            self.personas.traits_update(
                patient_utterance,
                self.prev_doctor_utt,
                therapist_action=None,
            )
            # Override whatever state the q-filter inferred
            self.personas._latent_mi["client_talk_type"] = fixed_state
            if self.logger:
                self.logger.debug("[FEP] fixed_state=%s (no prior inference)", fixed_state)
        else:
            current_state, _, _ = self.personas.traits_update(
                patient_utterance,
                self.prev_doctor_utt,
                therapist_action=None,
            )
            # Use q_turn_state_probs argmax when --state_predict_from_probs=True
            use_probs = getattr(self.args, "state_predict_from_probs", True)
            probs = self.personas.q_turn_state_probs or {}
            if use_probs and probs:
                best_label = max(probs, key=probs.get)
                lp_label = normalize_stage(best_label)
                if self.logger:
                    self.logger.debug(
                        "[FEP] infer_state probs_argmax=%s probs=%s",
                        lp_label, probs,
                    )
            else:
                try:
                    state_json = json.loads(current_state)
                    lp_label = normalize_stage(state_json.get("client_talk_type"))
                except json.JSONDecodeError:
                    lp_label = None

            # Soft-fusion with planner prior set on PREVIOUS turn (mirrors MSC.py:1148-1224).
            # Active iff prior_fusion_mode != 'none' AND a prior exists from the previous turn.
            _fusion_mode = getattr(self.args, "prior_fusion_mode", "none")
            planner_prior = getattr(self.personas, "_planner_prior", None)
            final_label = lp_label
            if planner_prior is not None and lp_label and _fusion_mode == "soft_fusion":
                _state_labels = ["precontemplation", "contemplation", "preparation"]
                _utt = patient_utterance or ""
                _n = len(_utt.split())
                _hedges = {"maybe", "i guess", "i don't know", "kind of", "sort of",
                           "mm", "hmm", "yeah", "okay", "right", "uh", "um",
                           "not sure", "i mean", "well"}
                _has_hedge = any(h in _utt.lower() for h in _hedges)
                if _n < 6:
                    _alpha = 0.50
                elif _n < 12 or _has_hedge:
                    _alpha = 0.65
                elif _n < 25:
                    _alpha = 0.75
                else:
                    _alpha = 0.85
                _beta = float(getattr(self.args, "fep_soft_fusion_beta", 0.35))
                _lp_probs = {k: (_alpha if k == lp_label else (1 - _alpha) / (len(_state_labels) - 1))
                             for k in _state_labels}
                _fused = {k: (1 - _beta) * _lp_probs[k] + _beta * planner_prior.get(k, 0.15)
                          for k in _state_labels}
                final_label = max(_fused, key=_fused.get)
                if self.logger and final_label != lp_label:
                    self.logger.info(
                        "[FEP] soft_fusion lp=%s prior=%s alpha=%.2f beta=%.2f -> %s",
                        lp_label, max(planner_prior, key=planner_prior.get),
                        _alpha, _beta, final_label,
                    )
            self.last_current_state = final_label
            # Feed fused label back into Personas so p filter uses it
            if final_label:
                self.personas._latent_mi["client_talk_type"] = final_label

        # --- EFE-based action selection (opt-in via --fep_efe_action) --------
        use_efe = getattr(self.args, "fep_efe_action", False)
        chosen_action: Optional[str] = None
        if use_efe and not getattr(self.args, "disable_planner", False):
            chosen_action, _rollforward = self._select_action_efe()

        # --- Run planner to produce next-state prior for NEXT turn's soft_fusion.
        # Mirrors MSC.py:798-817. Only runs when planner is active and fusion is enabled.
        _fusion_mode_post = getattr(self.args, "prior_fusion_mode", "none")
        if (
            _fusion_mode_post != "none"
            and not getattr(self.args, "disable_planner", False)
        ):
            try:
                _no_wm = bool(getattr(self.args, "planner_no_wm_input", False))
                _post_state = self.planner.predict_post_action_state(
                    self.personas._format_current_state(),
                    "" if _no_wm else self.personas._format_observation_model(),
                    "" if _no_wm else self.personas._format_transition_model(),
                    action=chosen_action,
                    therapist_utterance="",  # filled in next turn after generator runs
                    therapist_actions=self.personas.therapist_actions,
                )
                _next_pred = _extract_cami_stage_from_text_dyn(_post_state)
                self.personas.set_planner_prior(_next_pred)
                if self.logger:
                    self.logger.debug(
                        "[FEP] planner next-state prior=%s (no_wm_input=%s)",
                        _next_pred, _no_wm,
                    )
            except Exception as _e:
                if self.logger:
                    self.logger.warning("[FEP] planner predict_post_action_state failed: %s", _e)

        current_turn = self.turn_id + 1
        context_memories = self.memory_bank.context_retrieve(
            patient_utterance,
            n_results=self.args.context_memory_number,
            current_turn=current_turn,
            datatype="text",
        )
        related_memories = self.memory_bank.relevance_retrieve(
            patient_utterance,
            n_results=self.args.relevance_memory_number,
            dist_thres=self.args.dist_thres,
            current_turn=current_turn,
            datatype="text",
        )
        merged_context = "\n".join(
            f"[TURN {m['idx']}] : {m['dialog']}." for m in context_memories
        )
        merged_memories = self._merged_memories(related_memories)

        response = self.generator.response_build(
            inquiry=patient_utterance,
            context=merged_context,
            memories=merged_memories,
            user_traits=self.personas.format_full_user_model(),
            agent_traits=self.personas._format_agent_traits(),
            therapist_action=chosen_action,
        )
        response = ensure_prefix(strip_prefixed_response(response, "RESPONSE:"), "Counselor:")

        self.memory_bank.short_term_memory.append(
            {
                "idx": len(self.memory_bank.short_term_memory),
                "turn": current_turn,
                "dialog": response,
            }
        )
        self.prev_doctor_utt = strip_prefixed_response(response, "Counselor:")
        self.turn_id += 1
        return response


class _BaselineCounselorWrapper:
    """
    Thin adapter wrapping BaseCounselor / DIIRCounselor / CoSCounselor / CAMITECounselor
    / CAMICounselor so they satisfy the dynamic-eval doctor interface:
        reply() -> str
        receive(utterance: str)
        infer_current_state() -> Optional[str]
        last_therapist_action() -> Optional[str]
        system_world_model_snapshot() -> None
    """

    def __init__(self, core, seed_history: List[str]):
        self.core = core
        self._last_action: Optional[str] = None
        # Replay seed turns so the counselor has conversation context.
        for i, line in enumerate(seed_history):
            if line.startswith("Client:") or line.startswith(f"{getattr(core, 'usr_name', 'Client')}:"):
                self.core.receive(line)
            # Counselor lines are usually skipped — they are the seed doctor turns
            # which we just note without replying.

    @staticmethod
    def _extract_strategy_from_metadata(raw: str) -> Optional[str]:
        """Extract the Final/Retrieved Strategy from baseline counselor metadata prefix."""
        # DIIR: [DIIR || ... || Retrieved Strategy: Open Question || ...]
        m = re.search(r"Retrieved Strategy:\s*([^|\]]+)", raw)
        if m:
            return m.group(1).strip()
        # CoS / CAMI-TE / CAMI-FULL: [... || Final Strategy: Reflect || ...]
        m = re.search(r"Final Strategy:\s*([^|\]]+)", raw)
        if m:
            return m.group(1).strip()
        return None

    def receive(self, response: str):
        self.core.receive(response)

    def reply(self) -> str:
        raw = self.core.reply()
        # Extract strategy before stripping metadata
        self._last_action = self._extract_strategy_from_metadata(raw)
        # Normalise to "Counselor: ..." prefix
        return ensure_prefix(strip_prefixed_response(raw, "Counselor:"), "Counselor:")

    def infer_current_state(self) -> Optional[str]:
        return None  # baseline counselors don't track MI stage

    def last_therapist_action(self) -> Optional[str]:
        return self._last_action

    def system_world_model_snapshot(self) -> None:
        return None


def doctor_factory(mode: str, args, logger, profile: Dict, sample_id: int):
    seed_history = build_seed_history(profile, getattr(args, "init_utterances", 2))
    if mode == "fep":
        return FEPCounselor(args, logger, sample_id, seed_history)
    if mode == "cami_text":
        return TextOnlyCAMICounselor(
            goal=profile["topic"],
            behavior=profile["Behavior"],
            model=args.model,
            seed_history=seed_history,
            logger=logger,
        )
    # ── Counselor agent baselines ─────────────────────────────────────────────
    if mode == "base_counselor":
        core = BaseCounselor(
            goal=profile["topic"],
            behavior=profile["Behavior"],
            model=args.model,
        )
        return _BaselineCounselorWrapper(core, seed_history)
    if mode == "diir":
        annomi_path = getattr(args, "annomi_jsonl_path", "") or ""
        core = DIIRCounselor(
            goal=profile["topic"],
            behavior=profile["Behavior"],
            model=args.model,
            annomi_jsonl_path=annomi_path,
        )
        return _BaselineCounselorWrapper(core, seed_history)
    if mode == "cos":
        core = CoSCounselor(
            goal=profile["topic"],
            behavior=profile["Behavior"],
            model=args.model,
        )
        return _BaselineCounselorWrapper(core, seed_history)
    if mode in ("cami_te", "cami_full"):
        # cami_full uses the original CAMI counselor (with topic exploration);
        # cami_te uses CAMITECounselor (no topic exploration).
        if mode == "cami_full":
            core = CAMICounselor(
                goal=profile["topic"],
                behavior=profile["Behavior"],
                model=args.model,
            )
        else:
            core = CAMITECounselor(
                goal=profile["topic"],
                behavior=profile["Behavior"],
                model=args.model,
            )
        return _BaselineCounselorWrapper(core, seed_history)
    raise ValueError(f"Unsupported doctor mode: {mode}")


def run_single_profile(
    mode: str,
    args,
    logger,
    profile: Dict,
    sample_id: int,
    semantic_encoder: Optional[SemanticRuleEncoder] = None,
    personalized_weights: Optional[Dict] = None,
) -> Dict[str, object]:
    env = CAMIEnvAdapter(args, profile, personalized_weights=personalized_weights)
    doctor = doctor_factory(mode, args, logger, profile, sample_id)
    patient_utt = env.current_patient_utterance()
    initial_state = env.snapshot()["state"]
    simulator_world_model = env.client.simulator_world_model_snapshot()

    turns: List[TurnResult] = []
    done = False
    while not done:
        state_before = env.snapshot()["state"]
        if mode == "fep":
            doctor_utt = doctor.reply(patient_utt)
        else:
            doctor_utt = doctor.reply()
        predicted_current_state = doctor.infer_current_state()
        turn_action = getattr(doctor, "last_therapist_action", lambda: None)()
        patient_utt, after_snapshot, done = env.step(doctor_utt, explicit_therapist_action=turn_action)
        doctor.receive(f"Client: {patient_utt}")

        turns.append(
            TurnResult(
                doctor_utterance=doctor_utt,
                patient_utterance=patient_utt,
                state_before=state_before,
                state_after=after_snapshot["state"],
                predicted_current_state=predicted_current_state,
                therapist_action=turn_action,
            )
        )

    final_state = env.snapshot()["state"]
    simulator_world_model = env.client.simulator_world_model_snapshot()
    # Action distribution tracking
    action_counts: Dict[str, int] = {}
    for t in turns:
        a = t.therapist_action or "unknown"
        action_counts[a] = action_counts.get(a, 0) + 1
    n_turns_total = max(1, len(turns))
    action_distribution = {k: v / n_turns_total for k, v in action_counts.items()}
    therapist_input_ratio = action_distribution.get("therapist_input", 0.0)
    action_collapse = therapist_input_ratio >= 0.65

    current_state_matches = [
        float(t.predicted_current_state == t.state_before)
        for t in turns
        if t.predicted_current_state is not None and t.state_before is not None
    ]
    state_progress = [
        float(STAGE_TO_SCORE[t.state_after] >= STAGE_TO_SCORE[t.state_before])
        for t in turns
        if t.state_before in STAGE_TO_SCORE and t.state_after in STAGE_TO_SCORE
    ]
    system_world_model = doctor.system_world_model_snapshot()
    overlap_metrics = {
        "observation_model_overlap": None,
        "observation_model_overlap_normalized": None,
        "transition_model_overlap": None,
        "transition_structural_consistency": None,
        "observation_model_semantic_overlap": None,
        "transition_model_semantic_overlap": None,
    }
    semantic_alignment = {
        "observation": [],
        "transition": [],
    }
    if system_world_model:
        sim_obs_rules = simulator_world_model.get("p_o_given_s", {}).get("rules", [])
        sys_obs_rules = system_world_model.get("observation_model_block", [])
        sim_tr_rules = simulator_world_model.get("p_s_next_given_s_a", {}).get("rules", [])
        sys_tr_rules = system_world_model.get("transition_model_block", [])

        overlap_metrics["observation_model_overlap"] = observation_model_overlap(
            sim_obs_rules,
            sys_obs_rules,
        )
        overlap_metrics["observation_model_overlap_normalized"] = observation_model_overlap_normalized(
            sim_obs_rules,
            sys_obs_rules,
        )
        overlap_metrics["transition_model_overlap"] = transition_model_overlap(
            sim_tr_rules,
            sys_tr_rules,
        )
        overlap_metrics["transition_structural_consistency"] = transition_structural_consistency(
            sim_tr_rules,
            sys_tr_rules,
        )
        if semantic_encoder is not None and semantic_encoder.available:
            obs_sem, obs_align = observation_model_semantic_alignment(
                sim_obs_rules,
                sys_obs_rules,
                semantic_encoder,
            )
            tr_sem, tr_align = transition_model_semantic_alignment(
                sim_tr_rules,
                sys_tr_rules,
                semantic_encoder,
            )
            overlap_metrics["observation_model_semantic_overlap"] = obs_sem
            overlap_metrics["transition_model_semantic_overlap"] = tr_sem
            semantic_alignment["observation"] = obs_align
            semantic_alignment["transition"] = tr_align

    return {
        "mode": mode,
        "sample_id": sample_id,
        "profile_idx": profile.get("idx"),
        "goal": profile.get("topic"),
        "behavior": profile.get("Behavior"),
        "initial_state": initial_state,
        "final_state": final_state,
        "reached_preparation": final_state == "preparation",
        "stage_lift": (
            STAGE_TO_SCORE[final_state] - STAGE_TO_SCORE[initial_state]
            if final_state in STAGE_TO_SCORE and initial_state in STAGE_TO_SCORE
            else None
        ),
        "turns": [
            {
                "doctor_utterance": t.doctor_utterance,
                "patient_utterance": t.patient_utterance,
                "state_before": t.state_before,
                "state_after": t.state_after,
                "predicted_current_state": t.predicted_current_state,
                "therapist_action": t.therapist_action,
            }
            for t in turns
        ],
        "action_distribution": action_counts,
        "action_collapse": action_collapse,
        "therapist_input_ratio": therapist_input_ratio,
        "metrics": {
            "num_turns": len(turns),
            "avg_current_state_acc": (
                float(sum(current_state_matches) / len(current_state_matches))
                if current_state_matches
                else None
            ),
            "avg_non_regressive_transition": (
                float(sum(state_progress) / len(state_progress)) if state_progress else None
            ),
            "observation_model_overlap": overlap_metrics["observation_model_overlap"],
            "observation_model_overlap_normalized": overlap_metrics["observation_model_overlap_normalized"],
            "transition_model_overlap": overlap_metrics["transition_model_overlap"],
            "transition_structural_consistency": overlap_metrics["transition_structural_consistency"],
            "observation_model_semantic_overlap": overlap_metrics["observation_model_semantic_overlap"],
            "transition_model_semantic_overlap": overlap_metrics["transition_model_semantic_overlap"],
        },
        "simulator_world_model": simulator_world_model,
        "system_world_model": system_world_model,
        "semantic_rule_alignment": semantic_alignment,
    }


def aggregate_results(results: List[Dict[str, object]]) -> Dict[str, object]:
    by_mode: Dict[str, List[Dict[str, object]]] = {}
    for item in results:
        by_mode.setdefault(item["mode"], []).append(item)

    summary = {}
    for mode, rows in by_mode.items():
        lifts = [r["stage_lift"] for r in rows if r["stage_lift"] is not None]
        reach = [float(bool(r["reached_preparation"])) for r in rows]
        turns = [r["metrics"]["num_turns"] for r in rows if r["metrics"]["num_turns"] is not None]
        current_acc = [
            r["metrics"]["avg_current_state_acc"]
            for r in rows
            if r["metrics"]["avg_current_state_acc"] is not None
        ]
        transition = [
            r["metrics"]["avg_non_regressive_transition"]
            for r in rows
            if r["metrics"]["avg_non_regressive_transition"] is not None
        ]
        obs_overlap = [
            r["metrics"]["observation_model_overlap"]
            for r in rows
            if r["metrics"]["observation_model_overlap"] is not None
        ]
        obs_overlap_norm = [
            r["metrics"]["observation_model_overlap_normalized"]
            for r in rows
            if r["metrics"].get("observation_model_overlap_normalized") is not None
        ]
        trans_overlap = [
            r["metrics"]["transition_model_overlap"]
            for r in rows
            if r["metrics"]["transition_model_overlap"] is not None
        ]
        trans_struct = [
            r["metrics"]["transition_structural_consistency"]
            for r in rows
            if r["metrics"].get("transition_structural_consistency") is not None
        ]
        obs_sem_overlap = [
            r["metrics"]["observation_model_semantic_overlap"]
            for r in rows
            if r["metrics"]["observation_model_semantic_overlap"] is not None
        ]
        trans_sem_overlap = [
            r["metrics"]["transition_model_semantic_overlap"]
            for r in rows
            if r["metrics"]["transition_model_semantic_overlap"] is not None
        ]
        summary[mode] = {
            "n_runs": len(rows),
            "avg_stage_lift": float(sum(lifts) / len(lifts)) if lifts else None,
            "preparation_rate": float(sum(reach) / len(reach)) if reach else None,
            "avg_num_turns": float(sum(turns) / len(turns)) if turns else None,
            "avg_current_state_acc": float(sum(current_acc) / len(current_acc)) if current_acc else None,
            "avg_non_regressive_transition": (
                float(sum(transition) / len(transition)) if transition else None
            ),
            "avg_observation_model_overlap": (
                float(sum(obs_overlap) / len(obs_overlap)) if obs_overlap else None
            ),
            "avg_observation_model_overlap_normalized": (
                float(sum(obs_overlap_norm) / len(obs_overlap_norm)) if obs_overlap_norm else None
            ),
            "avg_transition_model_overlap": (
                float(sum(trans_overlap) / len(trans_overlap)) if trans_overlap else None
            ),
            "avg_transition_structural_consistency": (
                float(sum(trans_struct) / len(trans_struct)) if trans_struct else None
            ),
            "avg_observation_model_semantic_overlap": (
                float(sum(obs_sem_overlap) / len(obs_sem_overlap)) if obs_sem_overlap else None
            ),
            "avg_transition_model_semantic_overlap": (
                float(sum(trans_sem_overlap) / len(trans_sem_overlap)) if trans_sem_overlap else None
            ),
            "action_collapse_count": sum(1 for r in rows if r.get("action_collapse")),
            "avg_therapist_input_ratio": (
                float(sum(r.get("therapist_input_ratio", 0.0) for r in rows) / len(rows))
                if rows else None
            ),
        }
        # Profile coverage check
        profile_idxs = [r.get("profile_idx") for r in rows]
        unique_idxs = set(profile_idxs)
        summary[mode]["profile_coverage"] = {
            "total": len(profile_idxs),
            "unique": len(unique_idxs),
            "duplicates": len(profile_idxs) - len(unique_idxs),
        }
    return summary


def main():
    args = build_parser().parse_args()
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    logger = get_logger(args.output_json.replace(".json", ".log"))
    semantic_encoder = None
    if getattr(args, "semantic_overlap", True):
        semantic_encoder = SemanticRuleEncoder(args.semantic_model_name, logger=logger)
        if semantic_encoder.available:
            logger.info(
                "[semantic_overlap] enabled model=%s",
                semantic_encoder.model_name,
            )
        else:
            logger.info("[semantic_overlap] disabled (model unavailable).")

    profiles = load_profiles(args.profile_path, args.test_num)
    modes = [m.strip() for m in args.doctor_modes.split(",") if m.strip()]
    logger.info("Loaded %d profiles, evaluating modes=%s", len(profiles), modes)

    # Load per-profile personalized weights (optional; falls back to global if not found)
    personalized_weights_list = _load_personalized_weights(args.profile_path)
    logger.info("Loaded personalized weights for %d profiles", len(personalized_weights_list))

    results: List[Dict[str, object]] = []
    for sample_id, profile in enumerate(profiles):
        # Get personalized weights for this sample (by position in jsonl)
        p_weights = None
        if personalized_weights_list and sample_id < len(personalized_weights_list):
            p_weights = personalized_weights_list[sample_id].get("scales", {})

        for mode_idx, mode in enumerate(modes):
            run_seed = args.seed + sample_id * 101 + mode_idx
            set_all_seeds(run_seed)
            logger.info(
                "Running sample_id=%d profile_idx=%s mode=%s seed=%d personalized_weights=%s",
                sample_id,
                profile.get("idx"),
                mode,
                run_seed,
                "yes" if p_weights else "no",
            )
            results.append(
                run_single_profile(
                    mode,
                    args,
                    logger,
                    profile,
                    sample_id,
                    semantic_encoder=semantic_encoder,
                    personalized_weights=p_weights,
                )
            )

    payload = {
        "run": {
            "profile_path": args.profile_path,
            "wikipedia_dir": args.wikipedia_dir,
            "retriever_path": args.retriever_path,
            "doctor_modes": modes,
            "test_num": args.test_num,
            "max_turns": args.max_turns,
            "seed": args.seed,
            "client": args.client,
            "model": args.model,
            "client_simulator_mode": args.client_simulator_mode,
            "semantic_overlap": bool(getattr(args, "semantic_overlap", True)),
            "semantic_model_name": args.semantic_model_name,
            "semantic_model_loaded": bool(semantic_encoder.available) if semantic_encoder else False,
            "setting": "text-only first: doctors receive dialogue text only; CAMI client remains the fixed environment.",
        },
        "summary": aggregate_results(results),
        "results": results,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Saved interactive evaluation results to %s", args.output_json)


if __name__ == "__main__":
    main()
