"""PUMA counselor for dialogue rollouts with the local patient simulator."""

from __future__ import annotations

import path_setup  # noqa: F401

import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from Module.EventMemory import EventMemory
from Module.Generator import Generator
from Module.Personas import Personas
from Module.Planner import TransitionPlanner

class LocalModel:
    """Minimal local inference shared by the PUMA modules."""

    def __init__(self, args, *, model=None, tokenizer=None):
        import torch

        if (model is None) != (tokenizer is None):
            raise ValueError("Provide both model and tokenizer, or neither.")
        self.args = args
        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            gpu = str(getattr(args, "gpus", "0"))
            if torch.cuda.is_available():
                device_map = "auto" if "," in gpu else f"cuda:{gpu}"
                dtype = torch.float16
            else:
                device_map, dtype = "cpu", torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                args.model, trust_remote_code=True,
                torch_dtype=dtype, device_map=device_map,
            )
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        self.model, self.tokenizer = model, tokenizer
        self.model.eval()

    def employ(self, system, user, name="default", *,
               temperature=None, top_p=None, max_tokens=None, **_):
        import torch

        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        if getattr(self.tokenizer, "chat_template", None):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
            except TypeError:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
        else:
            prompt = system + "\n" + user
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True,
            max_length=self.args.max_input_length,
        ).to(self.model.device)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        options = {
            "max_new_tokens": int(max_tokens if max_tokens is not None else self.args.max_output_length),
            "pad_token_id": pad_id, "do_sample": False,
        }
        if temperature is not None and float(temperature) > 0:
            options.update(do_sample=True, temperature=float(temperature))
            if top_p is not None:
                options["top_p"] = float(top_p)
        with torch.inference_mode():
            output = self.model.generate(**inputs, **options)
        text = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text


STAGE_TO_SCORE = {"precontemplation": 0, "contemplation": 1, "preparation": 2}


def normalize_stage(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower()
    return s if s in STAGE_TO_SCORE else None



def _extract_cami_stage_from_text_dyn(text: Optional[str]) -> Optional[str]:
    """Lightweight planner-output → cami_stage extractor for dynamic PUMACounselor.

    Mirrors MSC._extract_cami_stage_from_text but avoids cross-module import to keep
    the counselor self-contained.
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



class PUMACounselor:
    def __init__(self, args, logger, sample_id: int, seed_history: List[str], *, model=None, tokenizer=None):
        self.args = args
        self.logger = logger
        self.client = LocalModel(args, model=model, tokenizer=tokenizer)

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
            self._update_personas(
                strip_prefixed_response(seed_history[idx], "Client:"),
                strip_prefixed_response(seed_history[idx - 1], "Counselor:"),
            )

    def _update_personas(self, patient_utterance: str, therapist_utterance: str) -> None:
        fixed_state = getattr(self.args, "fixed_state", None)
        if fixed_state:
            current_state, _, _ = self.personas.traits_update(
                patient_utterance, therapist_utterance,
                state_override=fixed_state,
            )
        else:
            if getattr(self.args, "observation_focused_state", False):
                focused_probs = self.personas._observation_focused_state_call(patient_utterance)
                if focused_probs:
                    self.personas._latent_mi["client_talk_type"] = max(focused_probs, key=focused_probs.get)
            if (getattr(self.args, "persona_single_step", False)
                    and getattr(self.args, "prior_fusion_mode", "none") != "soft_fusion"):
                current_state, _, _ = self.personas.traits_update(patient_utterance, therapist_utterance)
            else:
                current_state, _, _ = self.personas.traits_update(
                    patient_utterance, therapist_utterance,
                    state_resolver=lambda label, probs: self._resolve_state_before_rules(
                        label, probs, patient_utterance or "",
                    ),
                )
        probs = self.personas.q_turn_state_probs
        if probs:
            self.last_current_state = normalize_stage(max(probs, key=probs.get))
        else:
            self.last_current_state = normalize_stage(json.loads(current_state).get("client_talk_type"))

    def _resolve_state_before_rules(
        self, inferred_label: str, probabilities: Dict[str, float], utterance: str,
    ) -> Dict[str, float]:
        """Resolve the current state before the persona module updates its rules."""
        if getattr(self.args, "state_predict_from_probs", True) and probabilities:
            observed_label = normalize_stage(max(probabilities, key=probabilities.get))
        else:
            observed_label = normalize_stage(inferred_label)
        if observed_label is None:
            return probabilities

        prior = getattr(self.personas, "_planner_prior", None)
        if getattr(self.args, "prior_fusion_mode", "none") != "soft_fusion" or not prior:
            if max(probabilities, key=probabilities.get) == observed_label:
                return probabilities
            return {label: (0.9 if label == observed_label else 0.05) for label in STAGE_TO_SCORE}

        word_count = len(utterance.split())
        hedges = {"maybe", "i guess", "i don't know", "kind of", "sort of",
                  "mm", "hmm", "yeah", "okay", "right", "uh", "um",
                  "not sure", "i mean", "well"}
        has_hedge = any(hedge in utterance.lower() for hedge in hedges)
        if word_count < 6:
            alpha = 0.50
        elif word_count < 12 or has_hedge:
            alpha = 0.65
        elif word_count < 25:
            alpha = 0.75
        else:
            alpha = 0.85
        beta = float(getattr(self.args, "fep_soft_fusion_beta", 0.35))
        observation_probs = {label: (alpha if label == observed_label else (1 - alpha) / 2)
                             for label in STAGE_TO_SCORE}
        fused = {label: (1 - beta) * observation_probs[label] + beta * prior.get(label, 0.15)
                 for label in STAGE_TO_SCORE}
        final_label = max(fused, key=fused.get)
        if self.logger and final_label != observed_label:
            self.logger.info(
                "[PUMA] state fusion observation=%s prior=%s alpha=%.2f beta=%.2f -> %s",
                observed_label, max(prior, key=prior.get), alpha, beta, final_label,
            )
        return fused

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

        # Pass a longer recency window so the LLM can reason about
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
                # p_pref stays as the strict stationary preference (no
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
        self._update_personas(patient_utterance, self.prev_doctor_utt)

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
