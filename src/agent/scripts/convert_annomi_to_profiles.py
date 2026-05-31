"""
Convert AnnoMI transcripts to CAMI-compatible profiles_misc.jsonl for dynamic evaluation.

AnnoMI has gold annotations for talk_type and therapist_action, so those fields are
directly used. States (PC/C/P) are derived from talk_type sequences. Persona fields
(Beliefs, Motivation, Plans, Personas) are extracted via LLM from client utterances.

Usage:
    python convert_annomi_to_profiles.py \
        --input ../../artifacts/annomi/annomi_transcripts.jsonl \
        --output ../../artifacts/annomi/annomi_profiles_misc.jsonl \
        --model Qwen/Qwen3-8B \
        --num 0
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# MISC action mapping (AnnoMI fine-grained → MISC 17-class)
# ---------------------------------------------------------------------------

ANNOMI_TO_MISC = {
    # fine-grained qualified
    "question/open":                "Open Question",
    "question/closed":              "Closed Question",
    "reflection/simple":            "Simple Reflection",
    "reflection/complex":           "Complex Reflection",
    "therapist_input/information":  "Give Information",
    "therapist_input/advice":       "Advise with Permission",
    "therapist_input/negotiation":  "Emphasize Control",
    "therapist_input/options":      "Emphasize Control",
    # coarse fallback
    "question":         "Open Question",
    "reflection":       "Simple Reflection",
    "therapist_input":  "Give Information",
    "other":            "Facilitate",
}

# CAMI-style coarse patient actions per talk_type
PATIENT_ACTION_MAP = {
    "change":  "Acknowledge",
    "sustain": "Downplay",
    "neutral": "Inform",
}

# ---------------------------------------------------------------------------
# State derivation from talk_type sequence
# ---------------------------------------------------------------------------

def derive_states(talk_types: List[str], speakers: List[str]) -> List[str]:
    """
    Derive readiness stages from gold talk_type sequence.

    Logic:
      - Start at precontemplation
      - Transition PC→C: after seeing ≥2 cumulative change utterances
      - Transition C→P: after cumulative change utterances are ≥3× sustain

    Therapist turns inherit the current stage.
    """
    states = []
    stage = "Precontemplation"
    change_count = 0
    sustain_count = 0

    for tt, spk in zip(talk_types, speakers):
        if spk == "client":
            if tt == "change":
                change_count += 1
            elif tt == "sustain":
                sustain_count += 1

            if stage == "Precontemplation" and change_count >= 2:
                stage = "Contemplation"
            elif stage == "Contemplation" and change_count >= 3 * max(sustain_count, 1):
                stage = "Preparation"

        states.append(stage)

    return states


# ---------------------------------------------------------------------------
# LLM-based persona extraction
# ---------------------------------------------------------------------------

PERSONA_EXTRACTION_PROMPT = """You are analyzing a motivational interviewing (MI) counseling transcript.
Given the client's utterances below, extract structured persona information.

Client utterances:
{client_utterances}

Topic of the session: {topic}

Return a JSON object with these fields:
- "Personas": list of 2-4 short sentences describing the client's background and situation (e.g., "You smoke weed to relax after a long day.")
- "Beliefs": list of 2-4 beliefs the client holds about their behavior or situation (e.g., "You believe weed helps you unwind.")
- "Motivation": list of 2-4 motivation domains (short phrases like "Health", "Family", or longer sentences if the client expressed specific motivation)
- "Acceptable Plans": list of 1-3 concrete action steps the client would consider (e.g., "You consider cutting down from daily to weekends only.")

All items should be in second person ("You ..."). Be specific to this client's actual words.
Return ONLY the JSON object, no other text."""


def extract_persona_llm(transcript: Dict, tokenizer, lm) -> Dict:
    """Extract Personas/Beliefs/Motivation/Plans using LLM."""
    client_utts = []
    for u in transcript["utterances"]:
        if u["speaker"] == "client":
            client_utts.append(u["utterance_text"])

    client_text = "\n".join(f"- {u}" for u in client_utts)
    topic = transcript.get("topic", "behavioral change")

    prompt = PERSONA_EXTRACTION_PROMPT.format(
        client_utterances=client_text[:3000],
        topic=topic,
    )

    messages = [
        {"role": "system", "content": "You are a clinical psychology expert."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(lm.device)

    with torch.no_grad():
        out = lm.generate(**inputs, max_new_tokens=512, do_sample=False)
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # Strip Qwen3 <think> blocks
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    raw = re.sub(r'</?think>', '', raw).strip()

    # Parse JSON
    try:
        # Find JSON object in response
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            data = {}
    except json.JSONDecodeError:
        data = {}

    return {
        "Personas": data.get("Personas", [f"You are seeking help with {topic}."]),
        "Beliefs": data.get("Beliefs", [f"You are uncertain about changing your {topic} behavior."]),
        "Motivation": data.get("Motivation", [topic.capitalize()]),
        "Acceptable Plans": data.get("Acceptable Plans", ["You would consider taking small steps."]),
    }


# ---------------------------------------------------------------------------
# Convert one transcript
# ---------------------------------------------------------------------------

def convert_transcript(transcript: Dict, tokenizer, lm) -> Optional[Dict]:
    """Convert one AnnoMI transcript to CAMI profiles_misc format."""
    utts = transcript.get("utterances", [])
    if len(utts) < 4:
        return None

    tid = transcript.get("transcript_id", "unknown")
    topic = transcript.get("topic", "behavioral change")

    # Build parallel arrays
    speakers = []
    utterances = []
    talk_types = []
    actions_coarse = []
    misc_actions = []

    for u in utts:
        spk = u["speaker"]
        speakers.append("counselor" if spk == "therapist" else "client")
        utterances.append(u.get("utterance_text", ""))

        # Talk type (gold for client, inherit for therapist)
        tt = u.get("client_talk_type")
        if spk == "therapist":
            tt = tt or "neutral"  # therapist turns
        talk_types.append(tt or "neutral")

        # Actions
        if spk == "therapist":
            fine = u.get("therapist_action_fine_qualified") or u.get("therapist_action_coarse") or "other"
            fine = fine.lower()
            coarse = u.get("therapist_action_coarse") or "other"
            actions_coarse.append(coarse.lower())
            misc_actions.append(ANNOMI_TO_MISC.get(fine, ANNOMI_TO_MISC.get(coarse.lower(), "Facilitate")))
        else:
            # Client action from talk_type
            actions_coarse.append(PATIENT_ACTION_MAP.get(tt or "neutral", "Inform"))
            misc_actions.append(PATIENT_ACTION_MAP.get(tt or "neutral", "Inform"))

    # Derive states from talk_type sequence
    states = derive_states(talk_types, speakers)

    # Extract persona fields via LLM
    persona = extract_persona_llm(transcript, tokenizer, lm)

    profile = {
        "idx": f"annomi_{tid}",
        "topic": topic,
        "Behavior": topic,
        "speakers": speakers,
        "utterances": utterances,
        "actions": actions_coarse,
        "misc_actions": misc_actions,
        "talk_types": talk_types,
        "states": states,
        "suggestibilities": [3.5],
        "quality": transcript.get("mi_quality", "unknown"),
        **persona,
    }
    return profile


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert AnnoMI transcripts to profiles_misc format")
    parser.add_argument("--input", type=str, required=True, help="Path to annomi_transcripts.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Output profiles_misc.jsonl path")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="HuggingFace model for persona extraction")
    parser.add_argument("--num", type=int, default=0, help="Number of transcripts to convert (0=all)")
    parser.add_argument("--min_pairs", type=int, default=5, help="Skip transcripts with fewer client-therapist pairs")
    args = parser.parse_args()

    # Load model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    lm = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    lm.eval()
    print("Model loaded.")

    # Load transcripts
    transcripts = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                transcripts.append(json.loads(line))
    if args.num > 0:
        transcripts = transcripts[:args.num]
    print(f"Loaded {len(transcripts)} transcripts.")

    # Convert
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    profiles = []
    with open(args.output, "w") as fout:
        for i, rec in enumerate(transcripts):
            # Check minimum pairs
            client_turns = sum(1 for u in rec["utterances"] if u["speaker"] == "client")
            if client_turns < args.min_pairs:
                print(f"  [{i+1}/{len(transcripts)}] Skip {rec.get('transcript_id')}: only {client_turns} client turns")
                continue

            profile = convert_transcript(rec, tokenizer, lm)
            if profile:
                fout.write(json.dumps(profile, ensure_ascii=False) + "\n")
                profiles.append(profile)
                print(f"  [{i+1}/{len(transcripts)}] Converted {profile['idx']} ({len(profile['utterances'])} utts, "
                      f"topic={profile['topic'][:40]})")
            else:
                print(f"  [{i+1}/{len(transcripts)}] Skip {rec.get('transcript_id')}: too short")

    print(f"\nDone: {len(profiles)} profiles written to {args.output}")


if __name__ == "__main__":
    main()
