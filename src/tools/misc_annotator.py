"""
MISC (Motivational Interviewing Skill Code) Annotator.

Re-annotates CAMI counselor utterances with the full 17-class MISC taxonomy
(Miller et al., 2002) using a local LLM (Qwen).

Input:  profiles.jsonl (CAMI 38 profiles)
Output: profiles_misc.jsonl (same format + misc_actions field)

Usage:
    python misc_annotator.py \
        --input_path data/cami/profiles.jsonl \
        --output_path data/cami/profiles_misc.jsonl \
        --model Qwen/Qwen3-8B
"""

import argparse
import json
import os
import re
import sys
from typing import List, Dict, Optional

# ---------------------------------------------------------------------------
# MISC 17-class taxonomy (Miller et al., 2002; aligned with CAMI strategies)
# ---------------------------------------------------------------------------

MISC_LABELS = [
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
]

MISC_LABEL_SET = {l.lower().replace(" ", "_") for l in MISC_LABELS}

MISC_DESCRIPTIONS = {
    "Open Question": "Open-ended question that invites elaboration, leaves latitude for response. E.g. 'What brings you here today?', 'How do you feel about your drinking?'",
    "Closed Question": "Yes/no question or one that elicits a short specific answer. E.g. 'Did you drink last week?', 'How many cigarettes a day?'",
    "Simple Reflection": "Restatement or shallow paraphrase of client content — adds nothing new. E.g. 'So you don't want to quit.'",
    "Complex Reflection": "Reflection that adds meaning, makes an inference, emphasizes emotion, or summarizes multiple themes beyond surface content. E.g. 'It sounds like part of you wants to change but another part isn't sure.'",
    "Affirm": "Says something positive or complimentary to the client, acknowledging strengths or efforts. E.g. 'You've shown real courage by coming here.'",
    "Reframe": "Suggests a different meaning for an experience expressed by the client, placing it in a new light. E.g. 'Maybe this setback is actually showing you what matters most.'",
    "Support": "Generally supportive, understanding comments not codable as Affirm or Reflect. E.g. 'That must have been really difficult for you.'",
    "Emphasize Control": "Directly acknowledges or emphasizes the client's freedom of choice, autonomy, ability to decide. E.g. 'It's entirely up to you.'",
    "Advise with Permission": "Gives advice, suggestion, or solution WITH prior permission from the client. E.g. 'Would it be okay if I shared some information?' followed by advice.",
    "Facilitate": "Simple utterances that function as acknowledgments encouraging the client to keep sharing. E.g. 'Mm-hmm', 'Tell me more', 'Go on.'",
    "Give Information": "Provides factual information, explains something, or gives feedback without directing. E.g. 'Alcohol affects the liver by...'",
    "Structure": "Comments about what will happen in the session, transitions between topics, or session logistics. E.g. 'Let's start by talking about...'",
    "Raise Concern": "Points out a possible problem with client's plan, done collaboratively. E.g. 'I wonder if that might be challenging because...'",
    "Confront": "Directly disagrees, argues, corrects, shames, blames, criticizes, judges, or questions the client's honesty. E.g. 'That's not true and you know it.'",
    "Direct": "Gives an order, command, or direction using imperative language. E.g. 'You need to stop drinking right now.'",
    "Warn": "Provides a warning or threat implying negative consequences. E.g. 'If you keep smoking, you will get cancer.'",
    "Advise without Permission": "Gives unsolicited advice without asking permission first. E.g. 'You should join AA immediately.'",
}

CLASSIFY_SYSTEM_PROMPT = """\
You are an expert Motivational Interviewing (MI) behavior coder trained on the MI Skill Code (MISC, Miller et al., 2002).

Your task: classify a single counselor utterance into EXACTLY ONE of the following 17 MISC behavior categories.

Categories:
""" + "\n".join(
    f"- {label}: {MISC_DESCRIPTIONS[label]}" for label in MISC_LABELS
) + """

Important rules:
- Choose the SINGLE most dominant behavior in the utterance.
- If the utterance contains a question AND a reflection, prioritize whichever is the main intent.
- "Advise with Permission" requires evidence that the counselor asked permission first. If no permission was sought, use "Advise without Permission" or "Give Information".
- "Facilitate" is for very short acknowledgments only (e.g., "Mm-hmm", "I see", "Go on").
- Return ONLY valid JSON: {"label": "<one of the 17 categories>"}

/no_think"""


def _parse_label(raw: str) -> str:
    """Extract label from LLM response."""
    if not raw:
        return "Facilitate"
    # Try JSON parse
    m = re.search(r'"label"\s*:\s*"([^"]+)"', raw)
    if m:
        candidate = m.group(1).strip()
        # Exact match
        if candidate in MISC_LABELS:
            return candidate
        # Case-insensitive match
        for label in MISC_LABELS:
            if candidate.lower() == label.lower():
                return label
    # Fallback: find any label in text
    text = raw.lower()
    for label in MISC_LABELS:
        if label.lower() in text:
            return label
    return "Facilitate"


def annotate_profiles(
    input_path: str,
    output_path: str,
    model: str = "Qwen/Qwen3-8B",
    batch_size: int = 1,
):
    """Annotate all counselor utterances in profiles.jsonl with MISC labels."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading model {model}...")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    lm = AutoModelForCausalLM.from_pretrained(
        model, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    lm.eval()

    # Load profiles
    profiles = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                profiles.append(json.loads(line))

    print(f"Loaded {len(profiles)} profiles from {input_path}")

    # Collect all counselor utterances with context
    tasks = []  # (profile_idx, turn_idx, prompt)
    for pi, p in enumerate(profiles):
        speakers = p.get("speakers", [])
        utterances = p.get("utterances", [])
        states = p.get("states", [])
        n = min(len(speakers), len(utterances))

        for j in range(n):
            if speakers[j] == "counselor":
                # Build context: last 2 turns before this utterance
                context_lines = []
                for k in range(max(0, j - 2), j):
                    role = "Client" if speakers[k] == "client" else "Counselor"
                    context_lines.append(f"{role}: {utterances[k]}")
                context_str = "\n".join(context_lines) if context_lines else "(start of conversation)"

                state = states[j] if j < len(states) else "unknown"
                counselor_utt = utterances[j]

                user_prompt = (
                    f"Dialogue context:\n{context_str}\n\n"
                    f"Client state: {state}\n\n"
                    f"Counselor utterance to classify:\n\"{counselor_utt}\"\n\n"
                    f"Return JSON: {{\"label\": \"<MISC category>\"}}"
                )

                tasks.append((pi, j, user_prompt))

    print(f"Total counselor utterances to annotate: {len(tasks)}")

    # Run inference one by one (896 utterances, fast enough)
    results = {}  # (profile_idx, turn_idx) -> label
    label_counts = {}

    for idx, (pi, j, user_prompt) in enumerate(tasks):
        messages = [
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(lm.device)
        with torch.no_grad():
            out = lm.generate(**inputs, max_new_tokens=64, temperature=1.0, top_p=1.0, do_sample=False)
        raw_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        label = _parse_label(raw_text)
        results[(pi, j)] = label
        label_counts[label] = label_counts.get(label, 0) + 1

        if (idx + 1) % 100 == 0:
            print(f"  Annotated {idx + 1}/{len(tasks)}...")

    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:30s}  {count:4d}  ({count/len(tasks)*100:.1f}%)")

    # Write output profiles with misc_actions field
    for pi, p in enumerate(profiles):
        speakers = p.get("speakers", [])
        n = len(speakers)
        misc_actions = []
        for j in range(n):
            if speakers[j] == "counselor":
                misc_actions.append(results.get((pi, j), "Facilitate"))
            else:
                # Keep original client action
                orig_actions = p.get("actions", [])
                if j < len(orig_actions):
                    misc_actions.append(orig_actions[j])
                else:
                    misc_actions.append("")
        p["misc_actions"] = misc_actions

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in profiles:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nSaved annotated profiles to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MISC annotator for CAMI profiles")
    parser.add_argument("--input_path", type=str,
                        default="data/cami/profiles.jsonl")
    parser.add_argument("--output_path", type=str,
                        default="data/cami/profiles_misc.jsonl")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    args = parser.parse_args()
    annotate_profiles(args.input_path, args.output_path, args.model)
