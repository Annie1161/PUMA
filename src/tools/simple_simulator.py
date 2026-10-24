"""
SimpleSimulator — Profile-grounded, LLM-driven patient simulator.

Design:
  1. Triggers: use raw profile sentences (Beliefs/Motivation/Plans) directly.
     Personas are context only.
  2. Action selection: Dirichlet-smoothed per-profile distribution as reference,
     LLM chooses final action given context.
  3. Response generation: fully LLM-generated from (action + context + profile).

Causal chain:
  Doctor action → E[Δr] update readiness → readiness determines state
  → LLM selects patient action (guided by per-profile distribution)
  → LLM generates response (grounded in profile + conversation)
"""

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from simulator_common import STAGES, compute_e_dr_table


def load_all_profiles(path: str) -> List[Dict]:
    """Load all unique profiles from jsonl (for building population stats)."""
    profiles = []
    seen = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                p = json.loads(line)
                idx = p.get("idx", len(profiles))
                if idx not in seen:
                    seen.add(idx)
                    profiles.append(p)
    return profiles


# ---------------------------------------------------------------------------
# Trigger: raw profile sentences
# ---------------------------------------------------------------------------

TRIGGER_SOURCES = {
    "belief":     {"boost": 0.2},
    "motivation": {"boost": 0.4},
    "plan":       {"boost": 0.5},
}

# Patient actions per state (for prompt reference only)
PATIENT_ACTIONS_BY_STATE = {
    "precontemplation": ["Inform", "Engage", "Downplay", "Blame", "Deny"],
    "contemplation":    ["Acknowledge", "Engage", "Inform", "Hesitate", "Doubt"],
    "preparation":      ["Plan", "Engage", "Accept", "Inform", "Reject"],
}


@dataclass
class ProfileTrigger:
    """A trigger derived directly from a profile sentence."""
    source: str     # "belief", "motivation", "plan"
    text: str       # original sentence from profile
    boost: float    # discovery bonus for readiness


def extract_triggers(profile: Dict) -> List[ProfileTrigger]:
    """Extract triggers from Beliefs, Motivation (long sentences only), Plans."""
    triggers = []

    for belief in profile.get("Beliefs", []):
        if len(belief) > 10:
            triggers.append(ProfileTrigger(
                source="belief", text=belief, boost=TRIGGER_SOURCES["belief"]["boost"],
            ))

    for mot in profile.get("Motivation", []):
        if len(mot) > 20:  # skip short labels like "Health", "Economy"
            triggers.append(ProfileTrigger(
                source="motivation", text=mot, boost=TRIGGER_SOURCES["motivation"]["boost"],
            ))

    for plan in profile.get("Acceptable Plans", []):
        if len(plan) > 10:
            triggers.append(ProfileTrigger(
                source="plan", text=plan, boost=TRIGGER_SOURCES["plan"]["boost"],
            ))

    return triggers


# ---------------------------------------------------------------------------
# Semantic trigger matcher — raw sentences
# ---------------------------------------------------------------------------

# Global shared sentence-transformer model (avoid OOM from multiple loads)
_SHARED_ST_MODEL = None

def _get_st_model(model_name: str = "all-MiniLM-L6-v2"):
    global _SHARED_ST_MODEL
    if _SHARED_ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SHARED_ST_MODEL = SentenceTransformer(model_name)
    return _SHARED_ST_MODEL


class SemanticTriggerMatcher:
    """Match counselor utterances to triggers via embedding cosine similarity."""

    def __init__(self, triggers: List[ProfileTrigger], model_name: str = "all-MiniLM-L6-v2"):
        self.model = _get_st_model(model_name)
        self.triggers = triggers
        self.trigger_texts = [t.text for t in triggers]

        if self.trigger_texts:
            self.trigger_embeddings = self.model.encode(
                self.trigger_texts, normalize_embeddings=True,
            )
        else:
            self.trigger_embeddings = np.array([])

    def match(self, counselor_utt: str, threshold: float = 0.45) -> List[Tuple[int, float]]:
        """Return list of (trigger_index, similarity_score) above threshold."""
        if not self.trigger_texts or not counselor_utt.strip():
            return []
        utt_emb = self.model.encode([counselor_utt], normalize_embeddings=True)
        sims = (utt_emb @ self.trigger_embeddings.T)[0]
        matched = [(i, float(s)) for i, s in enumerate(sims) if s >= threshold]
        matched.sort(key=lambda x: -x[1])
        return matched


# ---------------------------------------------------------------------------
# Per-profile Dirichlet-smoothed action distribution
# ---------------------------------------------------------------------------

def compute_edr_threshold(
    speakers: List[str],
    states: List[str],
    misc_actions: List[str],
    e_dr_table: Dict[Tuple[str, str], float],
    from_stage: str,
    to_stage: str,
    avg_content_gate: float = 0.3,
) -> Optional[float]:
    """Replay history with E[Δr] × avg_content_gate to compute threshold.

    Unlike the old talk-type replay, this uses the same scale as the simulator's
    actual readiness updates, preventing threshold/update mismatch.
    """
    n = min(len(speakers), len(states), len(misc_actions))
    readiness = 0.0
    in_stage = False

    for j in range(n):
        st = states[j].lower() if states[j] else ""

        if st == from_stage:
            in_stage = True

        # Detect transition
        if in_stage and j > 0:
            prev_st = states[j - 1].lower() if states[j - 1] else ""
            if prev_st == from_stage and st == to_stage:
                return readiness

        # Accumulate: counselor turn followed by client turn
        if (in_stage and st == from_stage and speakers[j] == "counselor"
                and j + 1 < n and speakers[j + 1] == "client"):
            action = misc_actions[j]
            base = e_dr_table.get((from_stage, action), 0.0)
            readiness += base * avg_content_gate

    return None  # No transition found


def build_population_action_dist(profiles: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Build population-level P(action | stage) from all profiles."""
    counts = defaultdict(Counter)
    for p in profiles:
        speakers = p.get("speakers", [])
        states = p.get("states", [])
        actions = p.get("actions", [])
        n = min(len(speakers), len(states), len(actions))
        for i in range(n):
            if speakers[i] == "client":
                st = states[i].lower()
                act = actions[i]
                if st in STAGES:
                    counts[st][act] += 1

    dist = {}
    for stage, ct in counts.items():
        total = sum(ct.values())
        if total > 0:
            dist[stage] = {a: c / total for a, c in ct.items()}
    return dist


def build_profile_action_dist(
    profile: Dict,
    pop_dist: Dict[str, Dict[str, float]],
    alpha: float = 5.0,
) -> Dict[str, Dict[str, float]]:
    """Dirichlet-smoothed per-profile P(action | stage)."""
    speakers = profile.get("speakers", [])
    states = profile.get("states", [])
    actions = profile.get("actions", [])
    n = min(len(speakers), len(states), len(actions))

    profile_counts = defaultdict(Counter)
    for i in range(n):
        if speakers[i] == "client":
            st = states[i].lower()
            act = actions[i]
            if st in STAGES:
                profile_counts[st][act] += 1

    dist = {}
    for stage in STAGES:
        p_pop = pop_dist.get(stage, {})
        p_obs = profile_counts.get(stage, Counter())
        n_obs = sum(p_obs.values())

        all_actions = sorted(set(list(p_pop.keys()) + list(p_obs.keys())))
        posterior = {}
        for act in all_actions:
            numerator = p_obs.get(act, 0) + alpha * p_pop.get(act, 0)
            denominator = n_obs + alpha
            posterior[act] = numerator / denominator

        # Normalize
        total = sum(posterior.values())
        if total > 0:
            posterior = {a: v / total for a, v in posterior.items()}
        dist[stage] = posterior

    return dist


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _strip_think(text: str) -> str:
    """Remove Qwen3 <think> blocks."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'</?think>', '', text).strip()
    return text


def _llm_generate(tokenizer, lm, messages: List[Dict], max_tokens: int = 80) -> str:
    """Generate text from chat messages (greedy decoding)."""
    import torch
    from transformers import GenerationConfig
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(lm.device)
    gen_config = GenerationConfig(
        max_new_tokens=max_tokens,
        do_sample=False,
    )
    with torch.no_grad():
        out = lm.generate(**inputs, generation_config=gen_config)
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return _strip_think(raw)


def select_patient_action(
    stage: str,
    action_dist: Dict[str, float],
    counselor_utt: str,
    conversation: List[str],
    profile: Dict,
    matched_trigger_texts: List[str],
    tokenizer, lm,
) -> str:
    """LLM selects patient action given distribution + context."""
    # Build distribution string
    dist_lines = []
    for act, prob in sorted(action_dist.items(), key=lambda x: -x[1]):
        dist_lines.append(f"  {act}: {prob:.0%}")
    dist_str = "\n".join(dist_lines)

    behavior = profile.get("Behavior", "the behavior")
    topic = profile.get("topic", "this issue")
    personas = profile.get("Personas", [])[:4]
    personas_str = "; ".join(personas) if personas else "(none)"

    trigger_str = ""
    if matched_trigger_texts:
        trigger_str = f"\nThe counselor just touched on: {'; '.join(matched_trigger_texts[:2])}"

    context = "\n".join(conversation[-4:]) if conversation else "(start)"

    sys_prompt = f"""You are simulating a real patient's behavioral response in a counseling session about {topic} ({behavior}).
The patient's background: {personas_str}
/no_think"""

    user_prompt = f"""Recent conversation:
{context}

The counselor just said: "{counselor_utt}"{trigger_str}

The patient is in the {stage} stage. The probability distribution of this patient's typical actions is:
{dist_str}

IMPORTANT: Do NOT always pick the most probable action. Real patients vary their responses naturally. Consider what the counselor just said and pick the action that fits THIS specific moment. For example:
- If the counselor asks a factual question → Inform (share information)
- If the counselor reflects back feelings → Acknowledge or Hesitate
- If the counselor challenges → Downplay, Deny, or Doubt
- If the counselor discusses plans → Plan or Reject
Use the probability distribution as a rough guide for this patient's personality, but let the conversation context determine the specific choice.

Which action does the patient take right now? Reply with ONLY the action name."""

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    raw = _llm_generate(tokenizer, lm, messages, max_tokens=20)

    # Parse: find the best matching action name
    raw_lower = raw.lower().strip().strip('"').strip("'")
    for act in action_dist:
        if act.lower() in raw_lower:
            return act

    # Fallback: pick highest-probability action
    return max(action_dist, key=action_dist.get)


def generate_patient_response(
    patient_action: str,
    counselor_utt: str,
    conversation: List[str],
    profile: Dict,
    matched_trigger_texts: List[str],
    tokenizer, lm,
) -> str:
    """Generate patient response using LLM."""
    behavior = profile.get("Behavior", "the behavior")
    topic = profile.get("topic", "this issue")
    personas = profile.get("Personas", [])[:5]
    beliefs = profile.get("Beliefs", [])[:4]

    personas_str = "\n".join(f"- {p}" for p in personas) if personas else "(none)"
    beliefs_str = "\n".join(f"- {b}" for b in beliefs) if beliefs else "(none)"

    trigger_context = ""
    if matched_trigger_texts:
        trigger_context = "\nRelevant to this moment:\n" + "\n".join(
            f"- {t}" for t in matched_trigger_texts[:2]
        )

    context = "\n".join(conversation[-6:]) if conversation else "(start)"

    # Action-specific instruction
    action_instructions = {
        "Inform": "Share factual information about your situation related to what was asked.",
        "Engage": "Show you're listening and respond to what the counselor said.",
        "Downplay": "Minimize the issue. Show you don't think it's a big deal.",
        "Blame": "Deflect responsibility to external factors.",
        "Deny": "Refuse to acknowledge there's a problem.",
        "Acknowledge": "Show you're starting to see the counselor's point, but remain tentative.",
        "Hesitate": "Express uncertainty about change. You want to but aren't sure.",
        "Doubt": "Question whether change is possible or worth it.",
        "Plan": "Discuss concrete steps you might take toward change.",
        "Accept": "Agree to try something. Show readiness to move forward.",
        "Reject": "Decline what's being suggested.",
    }
    instruction = action_instructions.get(patient_action, "Respond naturally.")

    sys_prompt = f"""You are a patient in a counseling session about {topic} ({behavior}).
Your background:
{personas_str}
Your beliefs:
{beliefs_str}
{trigger_context}
Rules: Reply in first person, 1-3 sentences, natural conversational tone with hesitations and fillers like real speech. Directly respond to what the counselor just said. Do NOT mention therapy techniques or that you're in counseling.
/no_think"""

    user_prompt = f"""Conversation so far:
{context}

The counselor just said: "{counselor_utt}"

Your action: {patient_action} — {instruction}

Respond as the patient (1-3 sentences):"""

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    raw = _llm_generate(tokenizer, lm, messages, max_tokens=80)

    # Clean up
    response = raw.split("\n")[0].strip()
    for prefix in ("Patient:", "Client:", "patient:", "client:"):
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
    if not response or len(response) < 3:
        response = "I'm not sure what to say about that."
    return response


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

@dataclass
class SimpleSimulator:
    """Profile-grounded, LLM-driven patient simulator."""

    profile_idx: int
    profile: Dict
    topic: str
    behavior: str
    initial_stage: str

    # Thresholds
    trigger_coverage_for_pc_to_c: float = 0.3
    threshold_c_to_p: float = 2.0

    # Triggers + semantic matcher
    triggers: List[ProfileTrigger] = field(default_factory=list)
    discovered_triggers: Set[int] = field(default_factory=set)
    trigger_hit_counts: Dict[int, int] = field(default_factory=dict)
    trigger_matcher: Optional[SemanticTriggerMatcher] = None

    # Per-profile Dirichlet-smoothed action distributions
    action_dist: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # E[Δr] table
    e_dr_table: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # LLM references
    tokenizer: object = None
    lm: object = None

    # Runtime state
    stage: str = "precontemplation"
    readiness: float = 0.0
    turn_count: int = 0
    conversation: List[str] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)

    # Constants
    _CONTENT_GATE_OFF = 0.1
    _CONTENT_GATE_ON = 1.0
    _TRIGGER_REPEAT_DECAY = 0.5
    _SEMANTIC_THRESHOLD = 0.45

    def step(
        self,
        action: str,
        counselor_utt: str = "",
        rng: Optional[random.Random] = None,
    ) -> Tuple[str, str, bool, Dict]:
        rng = rng or random.Random()
        self.conversation.append(f"Counselor: {counselor_utt}")

        # === DOCTOR SIDE ===

        # 1. Semantic trigger matching (raw profile sentences)
        matched_raw = (
            self.trigger_matcher.match(counselor_utt, self._SEMANTIC_THRESHOLD)
            if self.trigger_matcher else []
        )
        matched_indices = [idx for idx, score in matched_raw]
        match_scores = {idx: score for idx, score in matched_raw}
        new_discoveries = [i for i in matched_indices if i not in self.discovered_triggers]
        is_trigger_hit = len(matched_indices) > 0

        # Update counts
        if is_trigger_hit:
            for i in matched_indices:
                self.trigger_hit_counts[i] = self.trigger_hit_counts.get(i, 0) + 1
        for i in new_discoveries:
            self.discovered_triggers.add(i)

        # 2. Readiness: E[Δr] × content_gate
        base_e_dr = self.e_dr_table.get((self.stage, action), 0.0)

        if is_trigger_hit:
            min_hits = min(self.trigger_hit_counts.get(i, 1) for i in matched_indices)
            repeat_decay = self._TRIGGER_REPEAT_DECAY ** max(0, min_hits - 1)
            best_score = max(match_scores.values())
            content_gate = self._CONTENT_GATE_OFF + (
                self._CONTENT_GATE_ON - self._CONTENT_GATE_OFF
            ) * repeat_decay * best_score
        else:
            content_gate = self._CONTENT_GATE_OFF

        delta = base_e_dr * content_gate
        self.readiness += delta

        # Discovery bonus
        discovery_bonus = sum(self.triggers[i].boost for i in new_discoveries)
        self.readiness += discovery_bonus

        # 3. State transition
        prev_stage = self.stage
        trigger_coverage = len(self.discovered_triggers) / max(len(self.triggers), 1)

        if self.stage == "precontemplation":
            if trigger_coverage >= self.trigger_coverage_for_pc_to_c:
                self.stage = "contemplation"
                self.readiness = 0.0
        elif self.stage == "contemplation":
            if self.readiness >= self.threshold_c_to_p:
                self.stage = "preparation"

        self.turn_count += 1
        trigger_coverage = len(self.discovered_triggers) / max(len(self.triggers), 1)

        # === PATIENT SIDE ===

        # Matched trigger texts for LLM context
        matched_trigger_texts = [self.triggers[i].text for i in matched_indices]

        # 4. LLM selects patient action (guided by per-profile distribution)
        stage_dist = self.action_dist.get(self.stage, {})
        patient_action = select_patient_action(
            self.stage, stage_dist, counselor_utt,
            self.conversation, self.profile, matched_trigger_texts,
            self.tokenizer, self.lm,
        )

        # 5. LLM generates patient response
        patient_response = generate_patient_response(
            patient_action, counselor_utt,
            self.conversation, self.profile, matched_trigger_texts,
            self.tokenizer, self.lm,
        )
        self.conversation.append(f"Client: {patient_response}")

        done = self.stage == "preparation"

        info = {
            "trigger_hit": is_trigger_hit,
            "new_discovery": len(new_discoveries) > 0,
            "matched_triggers": [
                (self.triggers[i].text[:60], match_scores.get(i, 0))
                for i in matched_indices
            ],
            "discovery_bonus": discovery_bonus,
            "n_discovered": len(self.discovered_triggers),
            "n_total_triggers": len(self.triggers),
            "trigger_coverage": trigger_coverage,
            "patient_action": patient_action,
            "patient_response": patient_response,
        }

        self.history.append({
            "turn": self.turn_count,
            "doctor_action": action,
            "counselor_utt": counselor_utt[:200],
            "base_e_dr": base_e_dr,
            "content_gate": content_gate,
            "delta": delta,
            "discovery_bonus": discovery_bonus,
            "readiness": self.readiness,
            "stage_before": prev_stage,
            "stage_after": self.stage,
            "trigger_hit": is_trigger_hit,
            "trigger_coverage": trigger_coverage,
            "matched_triggers": [
                {"text": self.triggers[i].text[:80], "score": round(match_scores.get(i, 0), 3)}
                for i in matched_indices
            ],
            "patient_action": patient_action,
            "patient_response": patient_response[:300],
        })

        return patient_action, self.stage, done, info

    def summary(self) -> Dict:
        stage_lift = STAGES.index(self.stage) - STAGES.index(self.initial_stage)
        return {
            "profile_idx": self.profile_idx,
            "topic": self.topic,
            "initial_stage": self.initial_stage,
            "final_stage": self.stage,
            "stage_lift": stage_lift,
            "reached_preparation": self.stage == "preparation",
            "total_turns": self.turn_count,
            "trigger_coverage": len(self.discovered_triggers) / max(len(self.triggers), 1),
            "n_discovered": len(self.discovered_triggers),
            "n_total_triggers": len(self.triggers),
            "threshold_c_to_p": self.threshold_c_to_p,
            "trigger_coverage_for_pc_to_c": self.trigger_coverage_for_pc_to_c,
            "triggers": [
                {"source": t.source, "text": t.text[:80], "discovered": i in self.discovered_triggers}
                for i, t in enumerate(self.triggers)
            ],
            "action_dist": {
                stage: {a: round(p, 3) for a, p in dist.items()}
                for stage, dist in self.action_dist.items()
            },
            "history": self.history,
        }

    @classmethod
    def from_profile(
        cls,
        profile: Dict,
        population_tt_probs: Dict[Tuple[str, str], Dict[str, float]],
        pop_action_dist: Dict[str, Dict[str, float]],
        tokenizer,
        lm,
        trigger_coverage_threshold: float = 0.3,
        semantic_model_name: str = "all-MiniLM-L6-v2",
        alpha: float = 5.0,
    ) -> "SimpleSimulator":
        speakers = profile.get("speakers", [])
        states = profile.get("states", [])
        talk_types = profile.get("talk_types", [])
        misc_actions = profile.get("misc_actions", [])

        initial_stage = (states[0].lower() if states else "precontemplation")

        e_dr_table = compute_e_dr_table(population_tt_probs)

        # E[Δr]-based threshold (same scale as simulator's readiness updates)
        threshold_cp = compute_edr_threshold(
            speakers, states, misc_actions, e_dr_table,
            "contemplation", "preparation", avg_content_gate=0.3,
        )
        if threshold_cp is None:
            threshold_cp = 0.5  # conservative default
        triggers = extract_triggers(profile)
        trigger_matcher = SemanticTriggerMatcher(triggers, semantic_model_name)
        action_dist = build_profile_action_dist(profile, pop_action_dist, alpha)

        return cls(
            profile_idx=profile.get("idx", 0),
            profile=profile,
            topic=profile.get("topic", ""),
            behavior=profile.get("Behavior", ""),
            initial_stage=initial_stage,
            trigger_coverage_for_pc_to_c=trigger_coverage_threshold,
            threshold_c_to_p=threshold_cp,
            triggers=triggers,
            trigger_matcher=trigger_matcher,
            e_dr_table=e_dr_table,
            action_dist=action_dist,
            tokenizer=tokenizer,
            lm=lm,
            stage=initial_stage,
            readiness=0.0,
        )
