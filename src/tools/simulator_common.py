"""Shared vocabulary and population statistics for the current MI simulator."""

import json
from collections import Counter, defaultdict
from typing import Dict, Tuple

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


def compute_e_dr_table(
    population_tt_probs: Dict[Tuple[str, str], Dict[str, float]],
) -> Dict[Tuple[str, str], float]:
    """Compute E[Δr] = P(change) + 0.3 P(neutral) - P(sustain)."""
    weights = {"change": W_CHANGE, "neutral": W_NEUTRAL, "sustain": W_SUSTAIN}
    return {
        key: sum(probs.get(talk_type, 0.0) * weight
                 for talk_type, weight in weights.items())
        for key, probs in population_tt_probs.items()
    }
