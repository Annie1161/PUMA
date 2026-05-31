"""
Simplified MI Patient Simulator — MISC 17-class, Talk-Type Driven.

Design:
    1. Each turn: counselor picks action → simulator samples talk_type → readiness updates
    2. Scoring: change=+1, neutral=+0.3, sustain=-1 (fixed across all patients)
    3. Thresholds: per-profile, derived from CAMI warm-up history replay
    4. P(talk_type | stage, action): from MISC-annotated CAMI data (profiles_misc.jsonl)

Usage:
    simulator = SimpleSimulator.from_profile(profile_dict)
    for turn in range(max_turns):
        action = counselor.select_action()  # one of MISC_ACTIONS
        talk_type, new_state, done = simulator.step(action)
"""

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAGES = ("precontemplation", "contemplation", "preparation")

# Scoring weights (fixed, universal)
W_CHANGE = 1.0
W_NEUTRAL = 0.3
W_SUSTAIN = -1.0

# MISC 17-class action vocabulary
MISC_ACTIONS = (
    "Open Question",
    "Closed Question",
    "Simple Reflection",
    "Complex Reflection",
    "Affirm",
    "Reframe",
    "Support",
    "Emphasize Control",
    "Advise with Permission",
    "Facilitate",
    "Give Information",
    "Structure",
    "Raise Concern",
    "Confront",
    "Direct",
    "Warn",
    "Advise without Permission",
)

# Population-level P(talk_type | stage, action) from MISC-annotated CAMI data
# Only includes cells with n >= 2
_POPULATION_TT: Dict[Tuple[str, str], Dict[str, float]] = {}


def _build_population_table(profiles_misc_path: str) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Build P(talk_type | stage, action) from annotated profiles."""
    profiles = []
    with open(profiles_misc_path) as f:
        for line in f:
            line = line.strip()
            if line:
                profiles.append(json.loads(line))

    counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

    for p in profiles:
        speakers = p.get("speakers", [])
        states = p.get("states", [])
        misc_actions = p.get("misc_actions", [])
        talk_types = p.get("talk_types", [])
        n = min(len(speakers), len(states), len(misc_actions), len(talk_types))

        for j in range(n - 1):
            if speakers[j] == "counselor" and speakers[j + 1] == "client":
                st = states[j].lower() if states[j] else ""
                act = misc_actions[j]
                tt = talk_types[j + 1]
                if st in STAGES and tt in ("change", "neutral", "sustain"):
                    counts[(st, act)][tt] += 1

    table = {}
    for key, ct in counts.items():
        total = sum(ct.values())
        if total >= 2:
            table[key] = {
                "change": ct.get("change", 0) / total,
                "neutral": ct.get("neutral", 0) / total,
                "sustain": ct.get("sustain", 0) / total,
            }
    return table


def _compute_profile_threshold(
    speakers: List[str],
    states: List[str],
    talk_types: List[str],
    from_stage: str,
    to_stage: str,
) -> Optional[float]:
    """Replay history with fixed scores to compute readiness at transition."""
    n = min(len(speakers), len(states), len(talk_types))
    readiness = 0.0
    in_stage = False

    for j in range(n):
        st = states[j].lower() if states[j] else ""

        # Detect stage entry
        if st == from_stage:
            in_stage = True

        # Detect transition
        if in_stage and j > 0:
            prev_st = states[j - 1].lower() if states[j - 1] else ""
            if prev_st == from_stage and st == to_stage:
                return readiness

        # Accumulate
        if in_stage and st == from_stage and speakers[j] == "client":
            tt = talk_types[j]
            if tt == "change":
                readiness += W_CHANGE
            elif tt == "neutral":
                readiness += W_NEUTRAL
            elif tt == "sustain":
                readiness += W_SUSTAIN

    return None  # No transition found


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

@dataclass
class SimpleSimulator:
    """Simple MI patient simulator driven by talk_type sampling."""

    profile_idx: int
    topic: str
    behavior: str
    initial_stage: str

    # Per-profile thresholds
    threshold_pc_to_c: float = 0.0
    threshold_c_to_p: float = 2.0

    # Per-profile P(talk_type | stage, action) from warm-up (overrides population)
    profile_tt_probs: Dict[Tuple[str, str], Dict[str, float]] = field(default_factory=dict)

    # Population fallback
    population_tt_probs: Dict[Tuple[str, str], Dict[str, float]] = field(default_factory=dict)

    # Runtime state
    stage: str = "precontemplation"
    readiness: float = 0.0
    turn_count: int = 0
    history: List[Dict] = field(default_factory=list)

    def step(self, action: str, rng: Optional[random.Random] = None) -> Tuple[str, str, bool]:
        """
        Process one counselor action.

        Returns:
            talk_type: sampled patient talk_type ("change", "neutral", "sustain")
            stage: current stage after update
            done: whether patient reached preparation
        """
        rng = rng or random.Random()

        # Get P(talk_type | stage, action)
        key = (self.stage, action)
        probs = self.profile_tt_probs.get(key) or self.population_tt_probs.get(key)
        if probs is None:
            # Ultimate fallback: neutral-dominated
            probs = {"change": 0.1, "neutral": 0.7, "sustain": 0.2}

        # Sample talk_type
        r = rng.random()
        if r < probs["change"]:
            talk_type = "change"
        elif r < probs["change"] + probs["neutral"]:
            talk_type = "neutral"
        else:
            talk_type = "sustain"

        # Update readiness
        delta = {"change": W_CHANGE, "neutral": W_NEUTRAL, "sustain": W_SUSTAIN}[talk_type]
        self.readiness += delta
        self.turn_count += 1

        # Check transitions
        prev_stage = self.stage
        if self.stage == "precontemplation" and self.readiness >= self.threshold_pc_to_c:
            self.stage = "contemplation"
            # Reset readiness for next stage
            self.readiness = 0.0
        elif self.stage == "contemplation" and self.readiness >= self.threshold_c_to_p:
            self.stage = "preparation"

        done = self.stage == "preparation"

        self.history.append({
            "turn": self.turn_count,
            "action": action,
            "talk_type": talk_type,
            "delta": delta,
            "readiness": self.readiness,
            "stage_before": prev_stage,
            "stage_after": self.stage,
        })

        return talk_type, self.stage, done

    def summary(self) -> Dict:
        """Return summary of the simulation run."""
        stage_lift = STAGES.index(self.stage) - STAGES.index(self.initial_stage)
        return {
            "profile_idx": self.profile_idx,
            "topic": self.topic,
            "initial_stage": self.initial_stage,
            "final_stage": self.stage,
            "stage_lift": stage_lift,
            "reached_preparation": self.stage == "preparation",
            "total_turns": self.turn_count,
            "threshold_pc_to_c": self.threshold_pc_to_c,
            "threshold_c_to_p": self.threshold_c_to_p,
            "history": self.history,
        }

    @classmethod
    def from_profile(
        cls,
        profile: Dict,
        population_tt_probs: Dict[Tuple[str, str], Dict[str, float]],
    ) -> "SimpleSimulator":
        """Build a simulator from a CAMI profile with MISC annotations."""
        speakers = profile.get("speakers", [])
        states = profile.get("states", [])
        talk_types = profile.get("talk_types", [])
        misc_actions = profile.get("misc_actions", [])

        initial_stage = (states[0].lower() if states else "precontemplation")

        # Compute per-profile thresholds from warm-up replay
        threshold_pc = _compute_profile_threshold(
            speakers, states, talk_types, "precontemplation", "contemplation"
        )
        threshold_cp = _compute_profile_threshold(
            speakers, states, talk_types, "contemplation", "preparation"
        )

        # Population median fallbacks
        if threshold_pc is None:
            threshold_pc = 0.0
        if threshold_cp is None:
            threshold_cp = 2.0

        # Build per-profile P(talk_type | stage, action) from warm-up
        profile_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        n = min(len(speakers), len(states), len(misc_actions), len(talk_types))
        for j in range(n - 1):
            if speakers[j] == "counselor" and speakers[j + 1] == "client":
                st = states[j].lower() if states[j] else ""
                act = misc_actions[j]
                tt = talk_types[j + 1]
                if st in STAGES and tt in ("change", "neutral", "sustain"):
                    profile_counts[(st, act)][tt] += 1

        profile_tt_probs = {}
        for key, ct in profile_counts.items():
            total = sum(ct.values())
            if total >= 1:  # Even 1 observation overrides population for this profile
                profile_tt_probs[key] = {
                    "change": ct.get("change", 0) / total,
                    "neutral": ct.get("neutral", 0) / total,
                    "sustain": ct.get("sustain", 0) / total,
                }

        return cls(
            profile_idx=profile.get("idx", 0),
            topic=profile.get("topic", ""),
            behavior=profile.get("Behavior", ""),
            initial_stage=initial_stage,
            threshold_pc_to_c=threshold_pc,
            threshold_c_to_p=threshold_cp,
            profile_tt_probs=profile_tt_probs,
            population_tt_probs=population_tt_probs,
            stage=initial_stage,
            readiness=0.0,
        )


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    misc_path = sys.argv[1] if len(sys.argv) > 1 else \
        "../baseline/mi_coaching/CAMI/annotations/profiles_misc.jsonl"

    # Build population table
    pop_table = _build_population_table(misc_path)
    print(f"Population table: {len(pop_table)} cells")

    # Load profiles
    profiles = []
    with open(misc_path) as f:
        for line in f:
            line = line.strip()
            if line:
                profiles.append(json.loads(line))

    # Deduplicate
    seen = set()
    unique = []
    for p in profiles:
        idx = p.get("idx", id(p))
        if idx not in seen:
            seen.add(idx)
            unique.append(p)

    # Run simulation with random action baseline
    rng = random.Random(42)
    good_actions = ["Open Question", "Facilitate", "Give Information"]
    bad_actions = ["Complex Reflection", "Reframe", "Confront"]

    for strategy_name, action_pool in [("Good MI", good_actions), ("Bad MI", bad_actions), ("Random", list(MISC_ACTIONS))]:
        lifts = []
        preps = []
        for p in unique[:10]:  # Test on first 10 profiles
            sim = SimpleSimulator.from_profile(p, pop_table)
            for _ in range(20):  # Max 20 turns
                action = rng.choice(action_pool)
                _, stage, done = sim.step(action, rng)
                if done:
                    break
            s = sim.summary()
            lifts.append(s["stage_lift"])
            preps.append(int(s["reached_preparation"]))

        avg_lift = sum(lifts) / len(lifts)
        prep_rate = sum(preps) / len(preps)
        print(f"  {strategy_name:10s}: avg_lift={avg_lift:.2f}  prep_rate={prep_rate:.0%}  (n={len(lifts)})")
