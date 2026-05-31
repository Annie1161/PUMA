"""
Convert KMI dialogues to CAMI-compatible simulator profiles.

Uses LLM (Qwen3-8B via vLLM or HF) to:
1. Assign client stage labels (Precontemplation/Contemplation/Preparation) per turn
2. Assign client talk_types (change/neutral/sustain) per turn
3. Extract Personas, Beliefs, Acceptable Plans, Motivation
4. Estimate suggestibility

Input:  dataset/kmi/transcripts.jsonl (annomi_jsonl format)
Output: dataset/kmi/profiles.jsonl (CAMI profile format for simulator)
"""

import json
import os
import sys
import argparse
import re
from typing import List, Dict, Optional

# KMI therapist label -> CAMI coarse action mapping
KMI_ACTION_MAP = {
    "open_question": "question",
    "closed_question": "question",
    "complex_reflection": "reflection",
    "simple_reflection": "reflection",
    "affirm": "therapist_input",
    "give_information": "therapist_input",
    "advise_with_permission": "therapist_input",
    "other": "other",
}

PROFILE_EXTRACTION_PROMPT = """You are analyzing a motivational interviewing (MI) dialogue transcript.

The dialogue is between a Therapist and a Client about: {category}

Your task:
1. For each CLIENT utterance, assign:
   - **stage**: One of "Precontemplation" (no intent to change), "Contemplation" (considering change), "Preparation" (ready/planning to change)
   - **talk_type**: One of "neutral" (factual, no direction), "change" (expressing desire/ability/need/commitment to change), "sustain" (defending status quo, resisting)

2. Extract structured profile information:
   - **topic**: The main issue being discussed (e.g., "managing emotions", "building confidence")
   - **Behavior**: Short label for the problematic pattern (e.g., "emotional dysregulation", "lack of motivation")
   - **Personas**: 3-5 bullet points describing the client's situation, characteristics, and context
   - **Beliefs**: 2-3 core beliefs the client holds about their situation
   - **Acceptable Plans**: 1-2 concrete actions the client seems willing to consider
   - **Motivation**: What drives the client toward change (list 2-3 domains + a summary sentence)
   - **suggestibility**: A number 1.0-5.0 indicating how receptive the client is (1=very resistant, 5=very open)

Here is the dialogue:

{dialogue}

Respond in this exact JSON format:
{{
  "topic": "...",
  "Behavior": "...",
  "Personas": ["...", "...", "..."],
  "Beliefs": ["...", "..."],
  "Acceptable Plans": ["..."],
  "Motivation": ["domain1", "domain2", "You are motivated because ..."],
  "suggestibility": 3.5,
  "client_annotations": [
    {{"utterance_idx": 0, "stage": "Precontemplation", "talk_type": "neutral"}},
    {{"utterance_idx": 1, "stage": "Contemplation", "talk_type": "change"}},
    ...
  ]
}}

IMPORTANT:
- client_annotations should have one entry per CLIENT utterance (not therapist)
- Stages should generally progress forward (Pre -> Cont -> Prep) but can stay the same
- Most KMI dialogues show progression from Precontemplation to Contemplation/Preparation
- Be conservative: only mark "Preparation" if the client explicitly states plans or commitment
"""


def format_dialogue_for_prompt(utterances: List[Dict]) -> str:
    """Format KMI utterances into readable dialogue text."""
    lines = []
    for i, u in enumerate(utterances):
        speaker = "Therapist" if u["speaker"] == "therapist" else "Client"
        lines.append(f"[{i}] {speaker}: {u['utterance_text']}")
    return "\n".join(lines)


def build_profile_from_kmi(
    kmi_record: Dict,
    llm_response: Dict,
) -> Optional[Dict]:
    """Convert a KMI transcript + LLM annotations into a CAMI-style profile."""
    utterances = kmi_record["utterances"]
    category = kmi_record.get("category", "general counseling")

    # Build speakers, utterances, actions, states, talk_types lists
    speakers = []
    utts = []
    actions = []
    states = []
    talk_types = []

    # Index client annotations by their position
    client_annotations = llm_response.get("client_annotations", [])
    client_idx_map = {a["utterance_idx"]: a for a in client_annotations}

    client_count = 0
    for i, u in enumerate(utterances):
        if u["speaker"] == "therapist":
            speakers.append("counselor")
            utts.append(u["utterance_text"])
            action_raw = u.get("main_therapist_behaviour", "other")
            actions.append(KMI_ACTION_MAP.get(action_raw, "other"))
            # For therapist turns, use the most recent client's stage
            if states:
                states.append(states[-1])
            else:
                states.append("Precontemplation")
            talk_types.append("neutral")
        else:
            speakers.append("client")
            utts.append(u["utterance_text"])
            actions.append("other")

            # Get annotation for this client utterance
            ann = client_idx_map.get(client_count, {})
            stage = ann.get("stage", "Contemplation")
            tt = ann.get("talk_type", "neutral")

            # Validate
            if stage not in ("Precontemplation", "Contemplation", "Preparation"):
                stage = "Contemplation"
            if tt not in ("change", "neutral", "sustain"):
                tt = "neutral"

            states.append(stage)
            talk_types.append(tt)
            client_count += 1

    if len(speakers) < 4:
        return None

    profile = {
        "idx": kmi_record.get("transcript_id", "unknown"),
        "topic": llm_response.get("topic", category),
        "Behavior": llm_response.get("Behavior", "behavioral pattern"),
        "speakers": speakers,
        "utterances": utts,
        "actions": actions,
        "states": states,
        "talk_types": talk_types,
        "Personas": llm_response.get("Personas", [f"Client seeking help with {category}"]),
        "Beliefs": llm_response.get("Beliefs", ["Uncertain about change"]),
        "Acceptable Plans": llm_response.get("Acceptable Plans", ["Consider small steps"]),
        "Motivation": llm_response.get("Motivation", [category, "You are motivated because of personal growth"]),
        "suggestibilities": [llm_response.get("suggestibility", 3.5)],
    }

    return profile


def convert_without_llm(kmi_record: Dict) -> Dict:
    """
    Heuristic conversion without LLM — assign stages based on dialogue position.
    Good enough for a quick small-sample test.
    """
    utterances = kmi_record["utterances"]
    category = kmi_record.get("category", "general counseling")

    speakers = []
    utts = []
    actions = []
    states = []
    talk_types = []

    # Count total client turns for stage assignment
    n_client = sum(1 for u in utterances if u["speaker"] == "client")
    client_count = 0

    for u in utterances:
        if u["speaker"] == "therapist":
            speakers.append("counselor")
            utts.append(u["utterance_text"])
            action_raw = u.get("main_therapist_behaviour", "other")
            actions.append(KMI_ACTION_MAP.get(action_raw, "other"))
            if states:
                states.append(states[-1])
            else:
                states.append("Precontemplation")
            talk_types.append("neutral")
        else:
            speakers.append("client")
            utts.append(u["utterance_text"])
            actions.append("other")

            # Heuristic stage assignment by position
            progress = client_count / max(n_client - 1, 1)
            if progress < 0.35:
                stage = "Precontemplation"
                tt = "neutral"
            elif progress < 0.7:
                stage = "Contemplation"
                # Check for change language heuristics
                text_lower = u["utterance_text"].lower()
                if any(w in text_lower for w in ["want to", "i'd like", "hope", "wish", "try", "change", "better"]):
                    tt = "change"
                else:
                    tt = "neutral"
            else:
                stage = "Preparation"
                text_lower = u["utterance_text"].lower()
                if any(w in text_lower for w in ["will", "going to", "plan", "start", "commit", "ready"]):
                    tt = "change"
                elif any(w in text_lower for w in ["but", "hard", "difficult", "can't", "not sure"]):
                    tt = "sustain"
                else:
                    tt = "change"

            states.append(stage)
            talk_types.append(tt)
            client_count += 1

    # Extract topic from category
    topic_map = {
        "Mental Health": "improving mental well-being",
        "Interpersonal Relationships": "improving interpersonal relationships",
        "Ego·Personality": "building self-confidence and identity",
        "Career·Employment": "career development and job satisfaction",
        "Academic·Examination": "academic performance and study habits",
        "Addiction·Obsession": "reducing addictive behaviors",
        "Family": "improving family relationships",
    }

    behavior_map = {
        "Mental Health": "mental health struggles",
        "Interpersonal Relationships": "relationship difficulties",
        "Ego·Personality": "low self-esteem",
        "Career·Employment": "career dissatisfaction",
        "Academic·Examination": "academic pressure",
        "Addiction·Obsession": "addictive behavior",
        "Family": "family conflict",
    }

    topic = topic_map.get(category, "personal growth")
    behavior = behavior_map.get(category, "behavioral pattern")

    # Simple persona extraction from first few client utterances
    client_utts = [u["utterance_text"] for u in utterances if u["speaker"] == "client"]
    personas = []
    if client_utts:
        personas.append(f"You are struggling with {behavior}")
        if len(client_utts) > 1:
            personas.append(f"You mentioned: \"{client_utts[0][:80]}\"")
        if len(client_utts) > 3:
            personas.append(f"You are considering change: \"{client_utts[-1][:80]}\"")
    if not personas:
        personas = [f"Client dealing with {category}"]

    profile = {
        "idx": kmi_record.get("transcript_id", "unknown"),
        "topic": topic,
        "Behavior": behavior,
        "speakers": speakers,
        "utterances": utts,
        "actions": actions,
        "states": states,
        "talk_types": talk_types,
        "Personas": personas,
        "Beliefs": [
            f"You believe your {behavior} is hard to overcome",
            "You think change might be possible with the right support",
        ],
        "Acceptable Plans": [
            "You would consider taking small steps toward improvement",
        ],
        "Motivation": [
            category,
            "Personal growth",
            f"You are motivated because you want to overcome {behavior}",
        ],
        "suggestibilities": [3.5],
    }

    return profile


def main():
    parser = argparse.ArgumentParser(description="Convert KMI dialogues to CAMI simulator profiles")
    parser.add_argument("--input", type=str, default="dataset/kmi/transcripts.jsonl",
                        help="Input KMI JSONL file")
    parser.add_argument("--output", type=str, default="dataset/kmi/profiles.jsonl",
                        help="Output profiles JSONL file")
    parser.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic",
                        help="Conversion mode: heuristic (fast, no LLM) or llm (uses Qwen for annotation)")
    parser.add_argument("--num", type=int, default=0, help="Number of profiles to convert (0=all)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model for LLM mode")
    parser.add_argument("--openai_base_url", type=str, default=None, help="vLLM endpoint URL")
    args = parser.parse_args()

    # Load KMI data
    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if args.num > 0:
        records = records[:args.num]

    print(f"Converting {len(records)} KMI dialogues to profiles (mode={args.mode})")

    profiles = []
    for i, rec in enumerate(records):
        if args.mode == "heuristic":
            profile = convert_without_llm(rec)
        else:
            # LLM mode - would require setting up the LLM client
            # For now, fall back to heuristic
            print(f"  LLM mode not yet implemented, using heuristic for record {i}")
            profile = convert_without_llm(rec)

        if profile:
            profiles.append(profile)

        if (i + 1) % 100 == 0:
            print(f"  Converted {i + 1}/{len(records)}")

    # Write output
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in profiles:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(profiles)} profiles to {args.output}")

    # Also write a small-5 version
    small_path = args.output.replace(".jsonl", "_small5.jsonl")
    with open(small_path, "w", encoding="utf-8") as f:
        for p in profiles[:5]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Also wrote 5-sample file to {small_path}")


if __name__ == "__main__":
    main()
